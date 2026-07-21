"""Topolens — Streamlit inference app.

Cinematic Instrument design system: dark film-grade palette, Fraunces display
type, staggered CSS reveal, letterbox image frames.  All ML logic is delegated
to models/ and render/ — nothing is reimplemented here.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import importlib.metadata
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path plumbing — app/ is one level below the project root.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402 — after sys.path fix
from models.cnn_model import CustomCNNRegressor  # noqa: E402
from models.dataset import build_image_transform, compute_normalization_stats  # noqa: E402
from models.gradcam import GradCAM, overlay_cam  # noqa: E402
from models.gnn_baseline import load_graph_count_gcn, predict_graph_counts as predict_gcn_graph_counts  # noqa: E402
from models.graph_statistic_baseline import build_density_table, predict_graph_counts as predict_graph_stat_counts  # noqa: E402
from render.render_graphs import render_graph  # noqa: E402
from topolens_utils import compute_density, density_bucket, load_yaml_config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "cnn_best.pt"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
INPUT_SIZE: Tuple[int, int] = (224, 224)
PROCESSED_LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "labels_processed.csv"

# Pick a real example thumbnail from data/images/ for the empty-state preview.
def _find_example_image() -> Optional[Path]:
    """Return the first PNG found in data/images/, or None."""
    if IMAGES_DIR.exists():
        pngs = sorted(IMAGES_DIR.glob("*.png"))
        if pngs:
            return pngs[0]
    return None

# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_model_and_gradcam() -> Tuple[CustomCNNRegressor, Tuple, GradCAM]:
    """Load CNN checkpoint, compute normalization stats, build GradCAM."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        normalize_stats = checkpoint.get("normalize_stats", None)
    else:
        state_dict = checkpoint
        normalize_stats = None

    model = CustomCNNRegressor()
    model.load_state_dict(state_dict)
    model.eval()

    if normalize_stats is None:
        normalize_stats = compute_normalization_stats(
            str(TRAIN_CSV), str(IMAGES_DIR), INPUT_SIZE
        )

    gcam = GradCAM(model)
    return model, normalize_stats, gcam


@st.cache_resource(show_spinner="Loading graph baselines…")
def load_graph_comparison_assets() -> Tuple[object, float]:
    """Load the GCN checkpoint and heuristic fallback density once per session."""
    gnn_model = load_graph_count_gcn(PROJECT_ROOT / "models" / "checkpoints" / "gnn_best.pt")
    _, fallback_density = build_density_table(PROJECT_ROOT / "data" / "splits" / "train.csv")
    return gnn_model, fallback_density


@st.cache_data(show_spinner=False)
def load_warning_reference_data() -> dict:
    """Load the training ceiling plus failure-case MAEs needed for live warnings."""
    data: dict = {}

    try:
        train_df = pd.read_csv(TRAIN_CSV)
        data["max_train_n"] = int(train_df["num_nodes"].max())
    except Exception:
        data["max_train_n"] = None

    summary_df = safe_read_csv(RESULTS_DIR / "failure_case_summary.csv")
    if summary_df is not None and {"split", "category", "mean_vertex_mae", "mean_edge_mae"}.issubset(summary_df.columns):
        held_out_summary = summary_df[summary_df["split"] == "held_out"]

        size_row = held_out_summary[held_out_summary["category"] == "out_of_distribution_size"]
        if not size_row.empty:
            row = size_row.iloc[0]
            data["size_warning"] = {
                "label": "out_of_distribution_size",
                "vertex_mae": float(row["mean_vertex_mae"]),
                "edge_mae": float(row["mean_edge_mae"]),
            }

        dense_row = held_out_summary[held_out_summary["category"] == "high_density_clutter"]
        if not dense_row.empty:
            row = dense_row.iloc[0]
            data["dense_warning"] = {
                "label": "high_density_clutter",
                "vertex_mae": float(row["mean_vertex_mae"]),
                "edge_mae": float(row["mean_edge_mae"]),
                "source": "summary",
            }

    if "dense_warning" not in data:
        categories_df = safe_read_csv(RESULTS_DIR / "failure_case_categories.csv")
        if categories_df is not None and {"split", "density_bucket", "abs_vertex_error", "abs_edge_error"}.issubset(categories_df.columns):
            dense_rows = categories_df[(categories_df["split"] == "held_out") & (categories_df["density_bucket"] == "dense")]
            if not dense_rows.empty:
                data["dense_warning"] = {
                    "label": "dense_bucket_fallback",
                    "vertex_mae": float(dense_rows["abs_vertex_error"].mean()),
                    "edge_mae": float(dense_rows["abs_edge_error"].mean()),
                    "source": "fallback",
                }

    return data


@st.cache_data(show_spinner=False)
def load_limitations_data() -> dict:
    """Read real numbers from CSVs for the Model Limitations expander."""
    data: dict = {}

    fail_path = RESULTS_DIR / "failure_case_summary.csv"
    if fail_path.exists():
        df_fail = pd.read_csv(fail_path)
        ood = df_fail[
            (df_fail["split"] == "held_out")
            & (df_fail["category"] == "out_of_distribution_size")
        ]
        normal = df_fail[
            (df_fail["split"] == "held_out")
            & (df_fail["category"] == "in_distribution_normal")
        ]
        if not ood.empty:
            data["ood_vertex_mae"] = round(float(ood.iloc[0]["mean_vertex_mae"]), 2)
            data["ood_edge_mae"] = round(float(ood.iloc[0]["mean_edge_mae"]), 2)
            data["ood_n"] = int(ood.iloc[0]["n"])
        if not normal.empty:
            data["normal_vertex_mae"] = round(float(normal.iloc[0]["mean_vertex_mae"]), 2)
            data["normal_edge_mae"] = round(float(normal.iloc[0]["mean_edge_mae"]), 2)
            data["normal_n"] = int(normal.iloc[0]["n"])

    probe_path = RESULTS_DIR / "probe_summary.csv"
    if probe_path.exists():
        df_probe = pd.read_csv(probe_path)
        overall = df_probe[df_probe["breakdown"] == "overall"]
        orig_row = overall[overall["variant"] == "original"]
        const_row = overall[overall["variant"] == "constant_node_size"]
        if not orig_row.empty and not const_row.empty:
            data["orig_vertex_mae"] = round(float(orig_row.iloc[0]["vertex_mae"]), 2)
            data["const_vertex_mae"] = round(float(const_row.iloc[0]["vertex_mae"]), 2)

    return data


def load_ground_truth_for_file(filename: str) -> Optional[Tuple[Optional[object], int, int]]:
    """Tries to find ground-truth graph structure and counts for a file name."""
    import pandas as pd
    stem = Path(filename).stem
    csv_path = PROCESSED_LABELS_CSV
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        row = df[df["graph_id"] == stem]
        if not row.empty:
            true_v = int(row.iloc[0]["num_nodes"])
            true_e = int(row.iloc[0]["num_edges"])
            graph_path = PROJECT_ROOT / row.iloc[0]["graph_path"]
            if graph_path.exists():
                import networkx as nx
                G = nx.read_graphml(graph_path)
                return G, true_v, true_e
            return None, true_v, true_e
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_inference(
    pil_img: Image.Image,
    model: CustomCNNRegressor,
    normalize_stats: Tuple,
) -> Tuple[int, int, torch.Tensor]:
    transform = build_image_transform(normalize_stats[0], normalize_stats[1], INPUT_SIZE)
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0)  # [1, 3, H, W]

    with torch.no_grad():
        log_pred = model(tensor)  # [1, 2]

    raw = torch.expm1(log_pred).round().clamp_min(0)
    pred_v = int(raw[0, 0].item())
    pred_e = int(raw[0, 1].item())
    return pred_v, pred_e, tensor


def compute_gradcam_overlays(
    pil_img: Image.Image,
    input_tensor: torch.Tensor,
    gcam: GradCAM,
) -> Tuple[Image.Image, Image.Image]:
    cam_v = gcam.generate(input_tensor, target_index=0)
    cam_e = gcam.generate(input_tensor, target_index=1)
    overlay_v = overlay_cam(pil_img.copy(), cam_v)
    overlay_e = overlay_cam(pil_img.copy(), cam_e)
    return overlay_v, overlay_e


def parse_graph_file(uploaded_file) -> "nx.Graph":  # type: ignore[name-defined]
    import networkx as nx

    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix == ".graphml":
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            G = nx.read_graphml(tmp_path)
        finally:
            os.unlink(tmp_path)
        return G

    if suffix in (".csv", ".txt"):
        sep = "," if suffix == ".csv" else r"\s+"
        lines = raw_bytes.decode("utf-8", errors="replace")
        import io as _io
        df_edges = pd.read_csv(_io.StringIO(lines), sep=sep, header=None, engine="python")
        G = nx.Graph()
        for row in df_edges.itertuples(index=False):
            G.add_edge(str(row[0]), str(row[1]))
        return G

    raise ValueError(f"Unsupported file type: {suffix!r}")


def render_graph_to_pil(G: "nx.Graph") -> Image.Image:  # type: ignore[name-defined]
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = render_graph(G, graph_id="uploaded_graph", out_dir=tmp_dir)
        img_path = result["image_path"]
        img = Image.open(img_path).convert("RGB")
        img.load()
    return img

# ---------------------------------------------------------------------------
# CSS / design system injection
# ---------------------------------------------------------------------------
EDITORIAL_CSS = """
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=Plus+Jakarta+Sans:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base typography & containers ──────────────────────────────────────── */
.stApp {
    background-color: #F5F5F5 !important;
}

.stMarkdown p:not(.hero-title):not(.hero-subtitle), .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #000000 !important;
}

/* ── Expander text & background visibility (Neo-Brutalist) ────────────────── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 3px solid #000000 !important;
    border-radius: 4px !important;
    box-shadow: 4px 4px 0px #000000 !important;
}

/* ── Hero title ────────────────────────────────────────────────────────── */
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 4.5rem !important;
    line-height: 0.95 !important;
    color: #000000 !important;
    letter-spacing: -0.04em !important;
    margin: 0 !important;
    padding: 0 !important;
    text-transform: uppercase !important;
    font-style: italic !important;
    text-align: left !important;
    display: block !important;
}

/* ── Hero subtitle ─────────────────────────────────────────────────────── */
.hero-subtitle {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1.05rem !important;
    color: #000000 !important;
    opacity: 0.85 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    text-align: left !important;
    display: block !important;
}

/* ── Thick solid wrap behind hero title ─────────────────────────────────── */
.hero-rule-wrap {
    border-bottom: 4px solid #000000;
    padding: 2.2rem 0;
    margin-bottom: 2.2rem;
    margin-top: 0.6rem;
    text-align: left;
}

/* ── Section label (Syne black caps) ───────────────────────────────────── */
.section-label {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 0.80rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #000000;
    margin-bottom: 0.35rem;
    margin-top: 1.5rem;
}

/* ── Hero stat numbers ─────────────────────────────────────────────────── */
.hero-stat-number {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    font-size: 5rem;
    line-height: 1.0;
    color: #000000;
    letter-spacing: -0.02em;
    display: block;
}

/* ── Letterbox image frame (Neo-Brutalist offset) ──────────────────────── */
.letterbox-frame {
    border: 3px solid #000000;
    padding: 10px;
    box-shadow: 6px 6px 0px #000000;
    background: #FFFFFF;
    display: inline-block;
    max-width: 100%;
}

/* ── Thin solid divider ─────────────────────────────────────────────────── */
.topo-divider {
    border: none;
    border-top: 3px solid #000000;
    margin: 1.8rem 0;
}

/* ── Monospace readout ─────────────────────────────────────────────────── */
.mono-readout {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem;
    color: #000000;
}

/* ── Staggered fade-in reveal ─────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.reveal-1 { animation: fadeInUp 0.50s ease both; animation-delay: 0.05s; }
.reveal-2 { animation: fadeInUp 0.50s ease both; animation-delay: 0.16s; }
.reveal-3 { animation: fadeInUp 0.50s ease both; animation-delay: 0.28s; }
.reveal-4 { animation: fadeInUp 0.50s ease both; animation-delay: 0.40s; }
.reveal-5 { animation: fadeInUp 0.50s ease both; animation-delay: 0.52s; }

/* ── Sidebar tweaks (Solid Cinematic Dark style) ────────────────────────── */
[data-testid="stSidebar"] {
    background: #1E1F26 !important;
    border-right: 3px solid #000000 !important;
}
[data-testid="stSidebar"] .section-label {
    margin-top: 1rem;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Suppress default Streamlit metric card chrome ────────────────────── */
[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Neo-Brutalist Grid Tables ────────────────────────────────────────── */
table {
    background-color: #FFFFFF !important;
    border-collapse: collapse !important;
    width: 100% !important;
    margin-bottom: 1rem !important;
    border: 3px solid #000000 !important;
    box-shadow: 4px 4px 0px #000000 !important;
}
th {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.75rem !important;
    padding: 12px !important;
    border: 1px solid #000000 !important;
}
th * {
    color: #FFFFFF !important;
}
td {
    border: 1px solid #000000 !important;
    padding: 10px !important;
    color: #000000 !important;
    font-size: 0.85rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Custom Neo-Brutalist Card layout */
.neobrutalist-card {
    background: #FFFFFF !important;
    border: 3px solid #000000 !important;
    padding: 18px !important;
    box-shadow: 4px 4px 0px #000000 !important;
    border-radius: 4px !important;
    margin-bottom: 1rem !important;
}
</style>
"""

def inject_css() -> None:
    """Inject the Editorial design-system CSS once per run."""
    st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


def render_hero_title() -> None:
    st.markdown(
        '<div class="hero-rule-wrap reveal-1">'
        '<p class="hero-title">Topolens</p>'
        '<p class="hero-subtitle">CNN-powered graph topology estimator &mdash; predicts vertex &amp; edge counts directly from rendered 2D graph images.</p>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_letterbox(img: Image.Image, caption: str = "", max_width: int = 420) -> None:
    """Render a PIL image inside a letterbox frame, capped at max_width px."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    st.markdown(
        f'<div class="letterbox-frame" style="max-width:{max_width}px;">',
        unsafe_allow_html=True,
    )
    st.image(buf, width=max_width)
    if caption:
        st.markdown(f'<span class="mono-readout">{caption}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_hero_stat(label: str, value: int, reveal_class: str) -> None:
    """Render a single hero-stat block: tracked label above, giant italic number."""
    st.markdown(
        f'<div class="{reveal_class}">'
        f'<p class="section-label">{label}</p>'
        f'<span class="hero-stat-number">{value:,}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_limitations_expander(lim: dict) -> None:
    """Collapsed expander with factual limitation bullets drawn from CSV data."""
    with st.expander("Model limitations", expanded=False):
        ood_v = lim.get("ood_vertex_mae", "—")
        ood_e = lim.get("ood_edge_mae", "—")
        ood_n = lim.get("ood_n", "—")
        norm_v = lim.get("normal_vertex_mae", "—")
        norm_e = lim.get("normal_edge_mae", "—")
        norm_n = lim.get("normal_n", "—")
        orig_v = lim.get("orig_vertex_mae", "—")
        const_v = lim.get("const_vertex_mae", "—")

        st.markdown(
            f"""
<span style="font-family:'Source Sans 3',sans-serif; font-size:0.88rem; color:#403B35;">

- **Out-of-distribution size ceiling (n > 100).** The training set contains
  no graphs above 100 nodes. On the held-out real-world split (n = {ood_n}
  graphs exceeding this ceiling), held-out vertex MAE is **{ood_v}** and edge
  MAE is **{ood_e}**, compared with vertex MAE {norm_v} / edge MAE {norm_e}
  for the {norm_n} in-distribution graphs in the same split.
  Do not use predictions on large graphs without awareness of this gap.

- **Node-size shortcut confound.** The renderer scales node radius as
  `max(15, 4000/n)`, so larger nodes signal fewer vertices. A shortcut probe
  (constant node size) raises vertex MAE from **{orig_v}** to
  **{const_v}** overall — strong evidence the model exploits visual node
  size rather than topology alone to count vertices.

- **Synthetic training distribution.** The model was trained entirely on
  NetworkX-generated graphs (Erdős–Rényi, Barabási–Albert, Watts–Strogatz,
  random trees, dense). Graphs with markedly different visual structure —
  hierarchical, planar, or domain-specific layouts — may fall outside the
  learned distribution.

- **Single layout algorithm.** All training images use Graphviz `sfdp`
  (spring-electrical). An alternative-layout probe (Kamada-Kawai) shows
  minimal vertex MAE change but measurable edge MAE variance, indicating
  partial layout sensitivity for edge estimation.

</span>
""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Home view components
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset_composition_table() -> pd.DataFrame:
    """Read labels_processed.csv and build a clean cross-tab of generator vs size-tier."""
    if not PROCESSED_LABELS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PROCESSED_LABELS_CSV)
        tiers_order = ["tiny", "small", "medium", "large", "xlarge"]
        ct = pd.crosstab(df["generator"], df["tier"])
        existing_cols = [c for c in tiers_order if c in ct.columns]
        ct = ct[existing_cols]
        ct["Total"] = ct.sum(axis=1)
        
        # Add column summary row
        total_row = ct.sum(axis=0)
        total_row.name = "Total"
        ct = pd.concat([ct, pd.DataFrame([total_row])])
        ct.index.name = "Generator / Dataset"
        return ct
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_headline_metrics() -> dict:
    """Extract overall baseline comparison MAE values for display on Home."""
    summary_path = RESULTS_DIR / "summary_comparison.csv"
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path)
        overall = df[(df["breakdown"] == "overall") & (df["metric"] == "mae")]
        metrics = {}
        for _, row in overall.iterrows():
            split = row["split"]
            target = row["target"]
            key_prefix = f"{split}_{target}"
            metrics[f"{key_prefix}_cnn"] = row["cnn"]
            metrics[f"{key_prefix}_gcn"] = row["gcn"]
            metrics[f"{key_prefix}_heuristic"] = row["graph_statistic"]
        return metrics
    except Exception:
        return {}


def safe_read_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file safely and return None on any failure."""
    try:
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)
    except Exception:
        return None


def format_number(value: object, decimals: int = 2) -> str:
    """Format a numeric value for display, or return a fallback string."""
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def render_data_unavailable(message: str = "data unavailable") -> None:
    """Render a compact fallback card for partially missing research data."""
    st.markdown(
        f"""
        <div class="neobrutalist-card" style="background:#FFF3CD;">
            <p style="font-size:0.9rem; font-weight:700; margin:0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_warning_banner(
    warning_data: dict,
    num_nodes: int,
    density_value: float,
    is_graph_upload: bool,
    size_triggered: bool,
    dense_triggered: bool,
) -> None:
    """Render the live Predict warning banner when a size or density trigger fires."""
    if not size_triggered and not dense_triggered:
        return

    max_train_n = warning_data.get("max_train_n")
    size_warning = warning_data.get("size_warning")
    dense_warning = warning_data.get("dense_warning")

    lines: list[str] = []
    if size_triggered and size_warning and max_train_n is not None:
        lines.append(
            f"This graph has {num_nodes} nodes, above the training ceiling of {max_train_n}. "
            f"Historical held-out accuracy for graphs in this range: vertex MAE {size_warning['vertex_mae']:.2f}, "
            f"edge MAE {size_warning['edge_mae']:.2f}. Treat this prediction with reduced confidence."
        )

    if dense_triggered and dense_warning:
        density_prefix = (
            "This graph's density, based on its actual graph topology, falls in the dense bucket."
            if is_graph_upload
            else "This graph's estimated density, based on this model's own predicted counts, falls in the dense bucket."
        )
        lines.append(
            f"{density_prefix} Historical held-out accuracy for dense graphs: vertex MAE {dense_warning['vertex_mae']:.2f}, "
            f"edge MAE {dense_warning['edge_mae']:.2f}. Treat this prediction with reduced confidence."
        )

    if not lines:
        return

    st.markdown(
        f"""
        <div class="neobrutalist-card" style="background:#FFF3CD; border-color:#000000;">
            <p style="font-size:0.9rem; font-weight:700; margin:0 0 0.55rem 0;">Reduced-confidence warning</p>
            <p style="font-size:0.88rem; margin:0; line-height:1.55;">{"<br><br>".join(lines)}</p>
            <p style="font-size:0.82rem; margin:0.65rem 0 0 0;">
                See Research Insights &rarr; Failure-Case Taxonomy for the full analysis of this failure mode.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_insights_section() -> None:
    """Render the Research Insights section with independently safe subsections."""
    st.markdown('<p class="section-label reveal-2">Research Insights</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="neobrutalist-card">
            <p style="font-size:0.95rem; margin:0;">
                This section connects the model's outputs to probe results, layout sensitivity checks, and failure-case patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 1) Grad-CAM
    # ------------------------------------------------------------------
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">1. Grad-CAM - What Does the Model Look At?</p>', unsafe_allow_html=True)

    gradcam_by_generator = PROJECT_ROOT / "report" / "figures" / "gradcam_grid_by_generator.png"
    gradcam_failure_cases = PROJECT_ROOT / "report" / "figures" / "gradcam_grid_failure_cases.png"
    if gradcam_by_generator.exists() and gradcam_failure_cases.exists():
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            render_image_in_frame(gradcam_by_generator, "Grad-CAM by generator", 380)
        with col_cam2:
            render_image_in_frame(gradcam_failure_cases, "Grad-CAM failure cases", 380)
        st.markdown(
            """
            <div class="neobrutalist-card">
                <p style="font-size:0.9rem; margin:0;">
                    Grad-CAM highlights the image regions that the CNN prediction is most sensitive to, so brighter areas show which visual cues the model is using when it estimates graph size.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_data_unavailable()

    # ------------------------------------------------------------------
    # 2) Shortcut-learning probe
    # ------------------------------------------------------------------
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">2. Shortcut-Learning Probe (Node-Size Confound)</p>', unsafe_allow_html=True)

    probe_df = safe_read_csv(RESULTS_DIR / "probe_summary.csv")
    probe_fig = PROJECT_ROOT / "report" / "figures" / "probe_variant_mae_comparison.png"
    if probe_df is not None and {"breakdown", "variant", "vertex_mae"}.issubset(probe_df.columns):
        probe_overall = probe_df[probe_df["breakdown"] == "overall"]
        orig_row = probe_overall[probe_overall["variant"] == "original"]
        const_row = probe_overall[probe_overall["variant"] == "constant_node_size"]
        if not orig_row.empty and not const_row.empty:
            orig_vertex_mae = float(orig_row.iloc[0]["vertex_mae"])
            const_vertex_mae = float(const_row.iloc[0]["vertex_mae"])
            delta_vertex_mae = const_vertex_mae - orig_vertex_mae
            delta_pct = (delta_vertex_mae / orig_vertex_mae * 100.0) if orig_vertex_mae else float("nan")

            col_probe1, col_probe2 = st.columns([1.2, 1])
            with col_probe1:
                if probe_fig.exists():
                    render_image_in_frame(probe_fig, "Probe MAE comparison", 380)
                else:
                    render_data_unavailable()
            with col_probe2:
                st.table(
                    pd.DataFrame(
                        {
                            "variant": ["original", "constant_node_size", "delta"],
                            "vertex_mae": [
                                format_number(orig_vertex_mae, 3),
                                format_number(const_vertex_mae, 3),
                                f"{delta_vertex_mae:+.3f}",
                            ],
                        }
                    )
                )
                st.markdown(
                    f"""
                    <div class="neobrutalist-card">
                        <p style="font-size:0.9rem; margin:0;">
                            Vertex MAE changes by <strong>{delta_vertex_mae:+.3f}</strong> when node size is forced constant, which is <strong>{delta_pct:+.1f}%</strong> relative to the original render. That jump means the model was relying on node-size cues, not topology alone, to count vertices.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            render_data_unavailable()
    else:
        render_data_unavailable()

    # ------------------------------------------------------------------
    # 3) Layout sensitivity probe
    # ------------------------------------------------------------------
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">3. Layout-Sensitivity Probe</p>', unsafe_allow_html=True)

    if probe_df is not None and {"breakdown", "variant", "vertex_mae"}.issubset(probe_df.columns):
        probe_overall = probe_df[probe_df["breakdown"] == "overall"]
        orig_row = probe_overall[probe_overall["variant"] == "original"]
        alt_row = probe_overall[probe_overall["variant"] == "alt_layout"]
        if not orig_row.empty and not alt_row.empty:
            orig_vertex_mae = float(orig_row.iloc[0]["vertex_mae"])
            alt_vertex_mae = float(alt_row.iloc[0]["vertex_mae"])
            delta_vertex_mae = alt_vertex_mae - orig_vertex_mae
            delta_pct = (delta_vertex_mae / orig_vertex_mae * 100.0) if orig_vertex_mae else float("nan")

            col_layout1, col_layout2 = st.columns([1.2, 1])
            with col_layout1:
                if probe_fig.exists():
                    render_image_in_frame(probe_fig, "Probe MAE comparison", 380)
                else:
                    render_data_unavailable()
            with col_layout2:
                st.table(
                    pd.DataFrame(
                        {
                            "variant": ["original", "alt_layout", "delta"],
                            "vertex_mae": [
                                format_number(orig_vertex_mae, 3),
                                format_number(alt_vertex_mae, 3),
                                f"{delta_vertex_mae:+.3f}",
                            ],
                        }
                    )
                )
                st.markdown(
                    f"""
                    <div class="neobrutalist-card">
                        <p style="font-size:0.9rem; margin:0;">
                            Vertex MAE changes by <strong>{delta_vertex_mae:+.3f}</strong> under the alternate layout, a <strong>{delta_pct:+.1f}%</strong> shift from the original layout. A non-zero shift means the model is partially sensitive to how the graph is arranged on the page, not just to the graph structure itself.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            render_data_unavailable()
    else:
        render_data_unavailable()

    # ------------------------------------------------------------------
    # 4) Ink-coverage correlation and failure-case taxonomy
    # ------------------------------------------------------------------
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">4. Ink-Coverage Correlation & Failure-Case Taxonomy</p>', unsafe_allow_html=True)

    corr_df = safe_read_csv(RESULTS_DIR / "ink_coverage_correlations.csv")
    fail_df = safe_read_csv(RESULTS_DIR / "failure_case_summary.csv")
    corr_ok = corr_df is not None and {"split", "metric_x", "metric_y", "pearson_r", "p_value", "n"}.issubset(corr_df.columns)
    fail_ok = fail_df is not None and {"split", "category", "mean_vertex_mae", "mean_edge_mae", "n"}.issubset(fail_df.columns)

    ink_fraction_label = "Ink fraction"
    component_area_label = "Component area"

    if corr_ok and fail_ok:
        held_out_corr = corr_df[corr_df["split"] == "held_out"].copy()
        held_out_corr = held_out_corr[["metric_x", "metric_y", "pearson_r", "p_value", "n"]]

        ood_row = fail_df[(fail_df["split"] == "held_out") & (fail_df["category"] == "out_of_distribution_size")]
        normal_row = fail_df[(fail_df["split"] == "held_out") & (fail_df["category"] == "in_distribution_normal")]

        if not ood_row.empty and not normal_row.empty:
            ood_vertex_mae = float(ood_row.iloc[0]["mean_vertex_mae"])
            normal_vertex_mae = float(normal_row.iloc[0]["mean_vertex_mae"])
            ood_edge_mae = float(ood_row.iloc[0]["mean_edge_mae"])
            normal_edge_mae = float(normal_row.iloc[0]["mean_edge_mae"])
            vertex_delta = ood_vertex_mae - normal_vertex_mae
            edge_delta = ood_edge_mae - normal_edge_mae
            vertex_ratio = (ood_vertex_mae / normal_vertex_mae) if normal_vertex_mae else float("nan")
            edge_ratio = (ood_edge_mae / normal_edge_mae) if normal_edge_mae else float("nan")

            col_ink1, col_ink2 = st.columns([1.1, 1])
            with col_ink1:
                if held_out_corr.empty:
                    render_data_unavailable()
                else:
                    st.table(held_out_corr)
                    st.markdown(
                        f"""
                        <div class="neobrutalist-card">
                            <p style="font-size:0.9rem; margin:0;">
                                {ink_fraction_label} is the share of the image covered by visible graph ink, while {component_area_label} is the average area of one connected visible blob. On the held-out split, the correlations stay modest: ink_fraction tracks node count weakly, and component_area is also only weakly related to node count.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with col_ink2:
                st.table(
                    pd.DataFrame(
                        {
                            "category": ["out_of_distribution_size", "in_distribution_normal", "contrast"],
                            "mean_vertex_mae": [
                                format_number(ood_vertex_mae, 3),
                                format_number(normal_vertex_mae, 3),
                                f"{vertex_delta:+.3f}",
                            ],
                            "mean_edge_mae": [
                                format_number(ood_edge_mae, 3),
                                format_number(normal_edge_mae, 3),
                                f"{edge_delta:+.3f}",
                            ],
                        }
                    )
                )
                st.markdown(
                    f"""
                    <div class="neobrutalist-card">
                        <p style="font-size:0.9rem; margin:0;">
                            The held-out out-of-distribution-size graphs are harder by <strong>{vertex_delta:+.3f}</strong> vertex MAE and <strong>{edge_delta:+.3f}</strong> edge MAE versus in-distribution-normal graphs, which is about <strong>{vertex_ratio:.1f}x</strong> and <strong>{edge_ratio:.1f}x</strong> worse respectively.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            fig_col1, fig_col2, fig_col3 = st.columns(3)
            with fig_col1:
                render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "ink_coverage_vs_node_count.png", "Ink coverage vs node count", 300)
            with fig_col2:
                render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "component_size_vs_node_count.png", "Component size vs node count", 300)
            with fig_col3:
                render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "worst_case_image_grid.png", "Worst-case image grid", 300)

            fig_col4, fig_col5 = st.columns(2)
            with fig_col4:
                render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "error_vs_num_nodes.png", "Error vs number of nodes", 380)
            with fig_col5:
                render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "error_vs_density.png", "Error vs density", 380)
        else:
            render_data_unavailable()
    else:
        render_data_unavailable()


def get_reproducibility_info() -> Tuple[str, str, str, str]:
    """Retrieve library versions and short git commit hash safely."""
    torch_ver = "Unknown"
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        pass

    torchvision_ver = "Unknown"
    try:
        import torchvision
        torchvision_ver = torchvision.__version__
    except Exception:
        pass

    nx_ver = "Unknown"
    try:
        import networkx as nx
        nx_ver = nx.__version__
    except Exception:
        try:
            nx_ver = importlib.metadata.version("networkx")
        except Exception:
            pass

    git_hash = ""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=True
        )
        git_hash = res.stdout.strip()
    except Exception:
        pass

    return torch_ver, torchvision_ver, nx_ver, git_hash


def render_home_section(lim_data: dict) -> None:
    st.markdown('<p class="section-label reveal-2">Project Overview</p>', unsafe_allow_html=True)
    
    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.markdown(
            """
            This research investigates whether visual, CNN-based regressions trained on 2D graph visualizations 
            can compete with graph-native representation methods (like Graph Convolutional Networks) for structural property estimation.
            
            By rendering graph topologies using spring-electrical layout algorithms, we test whether a neural network can learn to reconstruct graph properties (e.g. node and edge counts) purely from structural visual cues, bypassing raw matrix calculations.
            """
        )
    with col_intro2:
        st.markdown(
            """
            <div class="neobrutalist-card">
                <span style="font-weight: 800; font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;">Section Guide</span>
                <p style="font-size: 0.85rem; margin-top: 8px;">
                    <strong>Home</strong> &mdash; Project introduction, composition stats, and footer.<br>
                    <strong>Predict</strong> &mdash; Perform dynamic visual topology regression on graph files/renders.<br>
                    <strong>Results</strong> &mdash; Metrics comparison dashboard, breakdowns, and error scatters.<br>
                    <strong>Research Insights</strong> &mdash; Grad-CAM attention, shortcut-learning probe, layout sensitivity, and failure-case analysis.<br>
                    <strong>Models</strong> &mdash; Live parameter inspection, architectures, and loss logs.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Dataset Composition Summary</p>', unsafe_allow_html=True)
    
    ct_table = load_dataset_composition_table()
    if not ct_table.empty:
        st.table(ct_table)
        st.markdown(
            "<span class='mono-readout'>Source: data/processed/labels_processed.csv. Shows counts across size tiers (tiny: 5-10, small: 10-25, medium: 25-50, large: 50-100, xlarge: >100 nodes).</span>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Dataset labels file missing or failed to parse.")

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Rendering Pipeline Summary</p>', unsafe_allow_html=True)
    st.markdown(
        r"""
        Graphs are laid out using Graphviz's **sfdp** (spring-electrical) projection algorithm and rendered at a resolution of **224×224 pixels**. 
        To prevent visual occlusion on larger graphs, the node rendering radius scales dynamically according to the formula: 
        $$r = \max(15, \frac{4000}{n})$$ 
        where $n$ represents the vertex count. As a consequence, larger node sizes visually correlate with lower vertex counts.
        """
    )
    
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Headline Performance Comparison (Overall MAE)</p>', unsafe_allow_html=True)
    
    metrics = load_headline_metrics()
    if metrics:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div class="neobrutalist-card">
                    <span style="font-weight: 800; font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;">In-Distribution Test Split</span>
                    <p style="font-size: 0.9rem; margin-top: 8px;">
                        <strong>Vertices MAE:</strong><br>
                        &bull; CNN: <b>{metrics.get("test_num_vertices_cnn", 0.0):.3f}</b><br>
                        &bull; GCN: <b>{metrics.get("test_num_vertices_gcn", 0.0):.3f}</b><br>
                        &bull; Heuristic: <b>{metrics.get("test_num_vertices_heuristic", 0.0):.3f}</b> (trivial baseline)<br>
                        <strong>Edges MAE:</strong><br>
                        &bull; CNN: <b>{metrics.get("test_num_edges_cnn", 0.0):.3f}</b><br>
                        &bull; GCN: <b>{metrics.get("test_num_edges_gcn", 0.0):.3f}</b><br>
                        &bull; Heuristic: <b>{metrics.get("test_num_edges_heuristic", 0.0):.3f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="neobrutalist-card">
                    <span style="font-weight: 800; font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;">Held-Out Real-World Split</span>
                    <p style="font-size: 0.9rem; margin-top: 8px;">
                        <strong>Vertices MAE:</strong><br>
                        &bull; CNN: <b>{metrics.get("held_out_num_vertices_cnn", 0.0):.3f}</b><br>
                        &bull; GCN: <b>{metrics.get("held_out_num_vertices_gcn", 0.0):.3f}</b><br>
                        &bull; Heuristic: <b>{metrics.get("held_out_num_vertices_heuristic", 0.0):.3f}</b> (trivial baseline)<br>
                        <strong>Edges MAE:</strong><br>
                        &bull; CNN: <b>{metrics.get("held_out_num_edges_cnn", 0.0):.3f}</b><br>
                        &bull; GCN: <b>{metrics.get("held_out_num_edges_gcn", 0.0):.3f}</b><br>
                        &bull; Heuristic: <b>{metrics.get("held_out_num_edges_heuristic", 0.0):.3f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Overall comparison metrics file missing or failed to parse.")

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    render_limitations_expander(lim_data)
    
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    
    # Footer
    torch_v, torchvision_v, nx_v, git_hash = get_reproducibility_info()
    footer_text = f"Topolens &mdash; Seed: {config.RANDOM_SEED} | CNN Checkpoint: models/checkpoints/cnn_best.pt"
    if git_hash:
        footer_text += f" | Commit: {git_hash}"
    footer_text += f" | torch: {torch_v} | torchvision: {torchvision_v} | networkx: {nx_v}"
    
    st.markdown(
        f'<p class="mono-readout" style="text-align: center; color: #7F7F7F; padding-top: 1rem;">{footer_text}</p>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Results view components
# ---------------------------------------------------------------------------
def render_results_section() -> None:
    st.markdown('<p class="section-label reveal-2">Baseline Model Comparisons</p>', unsafe_allow_html=True)
    
    summary_path = RESULTS_DIR / "summary_comparison.csv"
    if not summary_path.exists():
        st.error("Missing summary_comparison.csv")
        return
        
    df_comp = pd.read_csv(summary_path)
    
    # Baseline comparison overall table
    df_overall = df_comp[df_comp["breakdown"] == "overall"][
        ["split", "target", "metric", "cnn", "gcn", "graph_statistic"]
    ].rename(columns={
        "split": "Split",
        "target": "Target",
        "metric": "Metric",
        "cnn": "CNN MAE/RMSE",
        "gcn": "GCN MAE/RMSE",
        "graph_statistic": "Heuristic MAE/RMSE"
    })
    
    for col in ["CNN MAE/RMSE", "GCN MAE/RMSE", "Heuristic MAE/RMSE"]:
        df_overall[col] = df_overall[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "—")
        
    st.table(df_overall)
    st.markdown(
        """
        <div style="background-color: #FFF3CD; border: 2px solid #000000; box-shadow: 2px 2px 0px #000000; padding: 10px; border-radius: 4px; font-size: 0.85rem;">
            <strong>Notice: Heuristic Vertex MAE Triviality.</strong> The Heuristic baseline model scores exactly <code>0.000</code> error on vertex counts. 
            This is <strong>structurally trivial</strong> because the baseline heuristic has access to the true graph node-list dimension directly, rather than performing any visual prediction or pattern representation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Detailed Metric Breakdowns</p>', unsafe_allow_html=True)
    
    tab_gen, tab_tier, tab_density = st.tabs(["By Generator Type", "By Size Tier", "By Density Bucket"])
    
    with tab_gen:
        df_gen = df_comp[df_comp["breakdown"] == "generator"][
            ["split", "group_value", "target", "metric", "cnn", "gcn", "graph_statistic"]
        ].rename(columns={
            "split": "Split",
            "group_value": "Generator/Dataset",
            "target": "Target",
            "metric": "Metric",
            "cnn": "CNN",
            "gcn": "GCN",
            "graph_statistic": "Heuristic"
        })
        st.dataframe(df_gen, width="stretch")
        
    with tab_tier:
        df_tier = df_comp[df_comp["breakdown"] == "tier"][
            ["split", "group_value", "target", "metric", "cnn", "gcn", "graph_statistic"]
        ].rename(columns={
            "split": "Split",
            "group_value": "Size Tier",
            "target": "Target",
            "metric": "Metric",
            "cnn": "CNN",
            "gcn": "GCN",
            "graph_statistic": "Heuristic"
        })
        st.dataframe(df_tier, width="stretch")
        
    with tab_density:
        df_density = df_comp[df_comp["breakdown"] == "density_bucket"][
            ["split", "group_value", "target", "metric", "cnn", "gcn", "graph_statistic"]
        ].rename(columns={
            "split": "Split",
            "group_value": "Density Bucket",
            "target": "Target",
            "metric": "Metric",
            "cnn": "CNN",
            "gcn": "GCN",
            "graph_statistic": "Heuristic"
        })
        st.dataframe(df_density, width="stretch")

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">CNN Prediction vs. Actual Scatter Plots</p>', unsafe_allow_html=True)
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.markdown('<span class="section-label" style="font-size:0.75rem;">In-Distribution Test Split</span>', unsafe_allow_html=True)
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "cnn_pred_num_vertices_scatter_test.png", "CNN Pred vs Actual Vertices (Test Split)", 380)
        st.write("")
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "cnn_pred_num_edges_scatter_test.png", "CNN Pred vs Actual Edges (Test Split)", 380)
    with col_sc2:
        st.markdown('<span class="section-label" style="font-size:0.75rem;">Held-Out Real-World Split</span>', unsafe_allow_html=True)
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "cnn_pred_num_vertices_scatter_held_out.png", "CNN Pred vs Actual Vertices (Held-Out Split)", 380)
        st.write("")
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "cnn_pred_num_edges_scatter_held_out.png", "CNN Pred vs Actual Edges (Held-Out Split)", 380)

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Dataset Distributions & Rendering Grid</p>', unsafe_allow_html=True)
    
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "graph_counts_by_tier.png", "Dataset Counts by Size Tier", 380)
        st.write("")
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "node_edge_distributions_by_generator.png", "Node/Edge Distributions by Generator Type", 380)
    with col_dist2:
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "node_edge_distributions_overall.png", "Overall Node and Edge Distributions", 380)
        st.write("")
        render_image_in_frame(PROJECT_ROOT / "report" / "figures" / "sample_render_grid.png", "Sample Render Grid (Graphviz sfdp)", 380)


def render_image_in_frame(img_path: Path, caption: str = "", max_width: int = 420) -> None:
    """Helper to display images in the letterbox frames."""
    if img_path.exists():
        img = Image.open(img_path)
        render_letterbox(img, caption=caption, max_width=max_width)
    else:
        st.warning(f"Figure not found: {img_path.name}")

# ---------------------------------------------------------------------------
# Models view components
# ---------------------------------------------------------------------------
def render_models_section(model: CustomCNNRegressor) -> None:
    st.markdown('<p class="section-label reveal-2">Custom CNN Architecture</p>', unsafe_allow_html=True)
    
    col_cnn1, col_cnn2 = st.columns([3, 2])
    with col_cnn1:
        st.markdown(
            """
            The primary model is a custom lightweight Convolutional Neural Network regressor consisting of four sequentially scaling feature blocks.
            It targets vertex and edge count estimation simultaneously using a joint Mean Squared Error loss.
            
            - **Target Encoding:** Targets are encoded as $\\log(1 + y)$ during training to compress wide dynamic scales, decoded at runtime via $e^x - 1$.
            - **Input dimensions:** Normalized 224&times;224 px RGB graph render layouts.
            """
        )
        
        # Load CNN checkpoint metadata if available
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            epoch = checkpoint.get("epoch", "Unknown")
            best_val = checkpoint.get("best_val_loss", "Unknown")
            best_val_str = f"{best_val:.6f}" if isinstance(best_val, float) else str(best_val)
            st.markdown(
                f"""
                <div class="neobrutalist-card">
                    <span style="font-weight: 800; font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;">CNN Checkpoint Details</span>
                    <p style="font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">
                        &bull; Path: <code>models/checkpoints/cnn_best.pt</code><br>
                        &bull; Convergence Epoch: <b>{epoch}</b><br>
                        &bull; Best Validation Loss (MSE): <b>{best_val_str}</b><br>
                        &bull; Parameters: <b>{model.num_parameters():,}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as exc:
            st.warning(f"Could not load checkpoint details: {exc}")
            
    with col_cnn2:
        st.markdown("##### PyTorch Layer Representation")
        st.code(str(model), language="python")

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">GCN Baseline (GraphCountGCN)</p>', unsafe_allow_html=True)
    
    col_gcn1, col_gcn2 = st.columns([3, 2])
    with col_gcn1:
        st.markdown(
            """
            The Graph Convolutional Network (GCN) baseline takes raw graph topology inputs (normalized degrees) and performs message passing.
            
            **The Out-of-Distribution Mean-Pooling Design Decision:**
            Switching from sum-pooling (`global_add_pool`) to mean-pooling (`global_mean_pool`) and normalizing degree features by each graph's max degree was a critical design choice. Sum-pooling scales embedding magnitudes directly with size; when evaluated on larger real-world graphs (e.g. PROTEINS up to 620 nodes) that fall outside the synthetic training ceiling of 100 nodes, sum-pooled outputs pushed the linear regression head into massive extrapolation, causing exponential decoding predictions to explode. Mean-pooling guarantees size-stable structural embeddings.
            """
        )
        
        gcn_info = {}
        try:
            from models.gnn_baseline import GraphCountGCN
            gcn_model = GraphCountGCN()
            gcn_info["structure"] = str(gcn_model)
            gcn_info["params"] = f"{gcn_model.num_parameters():,}"
        except Exception as e:
            gcn_info["structure"] = f"Unavailable: {e}"
            gcn_info["params"] = "N/A"
            
        try:
            gnn_checkpoint = torch.load(PROJECT_ROOT / "models" / "checkpoints" / "gnn_best.pt", map_location="cpu", weights_only=False)
            gcn_info["epoch"] = gnn_checkpoint.get("epoch", "Unknown")
            best_val = gnn_checkpoint.get("best_val_loss", "Unknown")
            gcn_info["best_val_loss"] = f"{best_val:.6f}" if isinstance(best_val, float) else str(best_val)
        except Exception:
            gcn_info["epoch"] = "Unknown"
            gcn_info["best_val_loss"] = "Unknown"
            
        st.markdown(
            f"""
            <div class="neobrutalist-card">
                <span style="font-weight: 800; font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;">GCN Checkpoint Details</span>
                <p style="font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">
                    &bull; Path: <code>models/checkpoints/gnn_best.pt</code><br>
                    &bull; Convergence Epoch: <b>{gcn_info.get("epoch")}</b><br>
                    &bull; Best Validation Loss (MSE): <b>{gcn_info.get("best_val_loss")}</b><br>
                    &bull; Parameters: <b>{gcn_info.get("params")}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_gcn2:
        st.markdown("##### GCN Module Structure")
        st.code(gcn_info.get("structure", "N/A"), language="python")

    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Graph-Statistic Heuristic Baseline</p>', unsafe_allow_html=True)
    st.markdown(
        r"""
        The graph-statistic heuristic baseline acts as a non-learned benchmark:
        - **Vertices:** Predicted exactly from the input graph's node list. Since it is allowed to observe the true count directly, it has a structurally trivial MAE of 0.0.
        - **Edges:** Estimated using the average density computed exclusively on the synthetic training split:
          $$\hat{e} = \text{density}_{\text{generator}} \times \frac{n(n - 1)}{2}$$
        - **Generalization limits:** Because it assumes a fixed generator-level density factor, the edge heuristic scales quadratically ($O(n^2)$) and systematically overestimates edges on larger sparse structures (e.g. tree graphs whose actual density drops as $2/n$).
        """
    )
    
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Training Setup & Curves</p>', unsafe_allow_html=True)
    
    cfg = load_yaml_config()
    st.markdown(
        f"""
        <div class="neobrutalist-card">
            <span style="font-weight: 800; font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;">Hyperparameters & config.yaml Config</span>
            <p style="font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">
                &bull; Random Seed: <b>{cfg.get("project", {}).get("seed", "Unknown")}</b><br>
                &bull; CNN Batch Size: <b>{cfg.get("model", {}).get("cnn", {}).get("batch_size")}</b> | Optimizer: <b>{cfg.get("model", {}).get("cnn", {}).get("optimizer")}</b> | Loss: <b>{cfg.get("model", {}).get("cnn", {}).get("loss")}</b><br>
                &bull; CNN Epochs: <b>{cfg.get("model", {}).get("cnn", {}).get("epochs")}</b> | Learning Rate: <b>{cfg.get("model", {}).get("cnn", {}).get("learning_rate")}</b><br>
                &bull; GCN Layers: <b>{cfg.get("model", {}).get("baseline", {}).get("layers")}</b> | Hidden Dimension: <b>{cfg.get("model", {}).get("baseline", {}).get("hidden_dim")}</b> | Epochs: <b>{cfg.get("model", {}).get("baseline", {}).get("epochs")}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_curve1, col_curve2 = st.columns(2)
    
    cnn_log_path = RESULTS_DIR / "cnn_training_log.csv"
    if cnn_log_path.exists():
        df_cnn_log = pd.read_csv(cnn_log_path)
        df_cnn_plot = df_cnn_log[["epoch", "train_loss", "val_loss"]].rename(
            columns={"train_loss": "Train Loss", "val_loss": "Val Loss"}
        ).set_index("epoch")
        with col_curve1:
            st.markdown("##### CNN Training & Validation Loss")
            st.line_chart(df_cnn_plot)
            
    gnn_log_path = RESULTS_DIR / "gnn_training_log.csv"
    if gnn_log_path.exists():
        df_gnn_log = pd.read_csv(gnn_log_path)
        df_gnn_plot = df_gnn_log[["epoch", "train_loss", "val_loss"]].rename(
            columns={"train_loss": "Train Loss", "val_loss": "Val Loss"}
        ).set_index("epoch")
        with col_curve2:
            st.markdown("##### GCN Training & Validation Loss")
            st.line_chart(df_gnn_plot)


# To save images during live app demo
def save_novel_upload(
    pil_img: Image.Image,
    original_filename: str,
    source_type: str,
    pred_v: float,
    pred_e: float,
    true_v: Optional[int] = None,
    true_e: Optional[int] = None,
    pred_v_gcn: Optional[float] = None,
    pred_e_gcn: Optional[float] = None,
    pred_v_stat: Optional[float] = None,
    pred_e_stat: Optional[float] = None,
) -> None:
    """Persist a live-app upload's rendered image + predictions to disk.

    The saved image keeps the uploaded file's own name (so a rendered graph
    is easy to trace back to the .graphml/.csv/etc. that produced it) rather
    than a content-hash filename. Deduplicates by content hash in the
    manifest so Streamlit reruns don't create repeat rows; if two different
    uploads share the same filename, a numeric suffix is appended so neither
    image is overwritten. Best-effort: logging failures never break the UI.
    """
    upload_dir = PROJECT_ROOT / "render" / "novel_uploads"  # match your renamed folder
    upload_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = upload_dir / "manifest.csv"

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    content_hash = hashlib.sha256(img_bytes).hexdigest()[:16]

    manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    if not manifest_df.empty and content_hash in manifest_df["content_hash"].values:
        return  # already logged this exact image

    stem = Path(original_filename).stem
    saved_name = f"{stem}.png"
    existing_names = set(manifest_df["saved_filename"]) if not manifest_df.empty else set()
    suffix = 1
    while saved_name in existing_names:
        saved_name = f"{stem}_{suffix}.png"
        suffix += 1

    with open(upload_dir / saved_name, "wb") as f:
        f.write(img_bytes)

    new_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": content_hash,
        "saved_filename": saved_name,
        "original_filename": original_filename,
        "source_type": source_type,
        "pred_vertices_cnn": round(float(pred_v), 2),
        "pred_edges_cnn": round(float(pred_e), 2),
        "pred_vertices_gcn": round(float(pred_v_gcn), 2) if pred_v_gcn is not None else "",
        "pred_edges_gcn": round(float(pred_e_gcn), 2) if pred_e_gcn is not None else "",
        "pred_vertices_baseline": round(float(pred_v_stat), 2) if pred_v_stat is not None else "",
        "pred_edges_baseline": round(float(pred_e_stat), 2) if pred_e_stat is not None else "",
        "true_vertices": true_v if true_v is not None else "",
        "true_edges": true_e if true_e is not None else "",
    }
    manifest_df = pd.concat([manifest_df, pd.DataFrame([new_row])], ignore_index=True)
    manifest_df.to_csv(manifest_path, index=False)


# ---------------------------------------------------------------------------
# Predict view components (Refactored)
# ---------------------------------------------------------------------------
def render_predict_section(
    model: CustomCNNRegressor,
    normalize_stats: Tuple,
    gcam: GradCAM,
    lim_data: dict,
) -> None:
    # Two-column layout for Upload (left) and Image Display (right)
    col_left, col_right = st.columns([1, 1.3])
    
    uploaded = None
    active_file = None
    
    with col_left:
        st.markdown('<p class="section-label" style="margin-top: 0;">Upload</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="Upload graph images or files",
            type=["png", "jpg", "jpeg", "graphml", "csv", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            if len(uploaded) > 1:
                st.markdown(
                    '<p style="font-family:\'Plus Jakarta Sans\', sans-serif; font-size:0.95rem; font-weight:600; margin-top:1rem; margin-bottom:0.5rem;">Select file to analyze</p>',
                    unsafe_allow_html=True,
                )
                active_file = st.selectbox(
                    "Select file to analyze",
                    options=uploaded,
                    format_func=lambda f: f.name,
                    label_visibility="collapsed",
                )
            else:
                active_file = uploaded[0]
                
    # Now, process active file / example image and render in col_right
    pil_img: Optional[Image.Image] = None
    G: Optional[object] = None
    true_v_csv: Optional[int] = None
    true_e_csv: Optional[int] = None
    
    if not uploaded:
        with col_right:
            st.markdown('<p class="section-label" style="margin-top: 0;">Example Preview</p>', unsafe_allow_html=True)
            example = _find_example_image()
            if example:
                ex_img = Image.open(example).convert("RGB")
                render_letterbox(ex_img, caption=f"data/images/{example.name}", max_width=480) # made larger and prominent!
            else:
                st.info("No example image found in data/images/.")
        st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
        render_limitations_expander(lim_data)
        return
        
    # If uploaded is present, process it
    suffix = Path(active_file.name).suffix.lower()
    is_graph_upload = suffix in (".graphml", ".csv", ".txt")
    
    # Try to find ground truth from dataset labels CSV for ANY type of file
    true_data = load_ground_truth_for_file(active_file.name)
    if true_data is not None:
        G_loaded, true_v_csv, true_e_csv = true_data
        if G_loaded is not None:
            G = G_loaded
            
    if suffix in (".png", ".jpg", ".jpeg"):
        try:
            pil_img = Image.open(active_file).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read image: {exc}")
            return
            
    elif suffix in (".graphml", ".csv", ".txt"):
        with st.spinner("Parsing and rendering graph…"):
            try:
                G = parse_graph_file(active_file)
                pil_img = render_graph_to_pil(G)
            except Exception as exc:
                st.error(f"Failed to process graph file: {exc}")
                return
    else:
        st.error(f"Unsupported file type: {suffix}")
        return
        
    with col_right:
        st.markdown('<p class="section-label" style="margin-top: 0;">Input graph image</p>', unsafe_allow_html=True)
        render_letterbox(pil_img, max_width=480) # made larger and prominent!
        
    # ── Inference ────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        try:
            pred_v, pred_e, input_tensor = run_inference(pil_img, model, normalize_stats)
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return
            
    # ── Hero stat display ────────────────────────────────────────────
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label reveal-2">Predicted structure</p>',
        unsafe_allow_html=True,
    )
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        render_hero_stat("Vertices", pred_v, "reveal-3")
    with stat_col2:
        render_hero_stat("Edges", pred_e, "reveal-4")
    
    gcn_v = gcn_e = stat_v = stat_e = None

    if G is not None:
        fallback_density = None
        try:
            gnn_model, fallback_density = load_graph_comparison_assets()
            gcn_v, gcn_e = predict_gcn_graph_counts(gnn_model, G)
        except Exception:
            pass

        try:
            if fallback_density is None:
                try:
                    _, fallback_density = build_density_table(PROJECT_ROOT / "data" / "splits" / "train.csv")
                except Exception:
                    fallback_density = None
            if fallback_density is not None:
                stat_v, stat_e = predict_graph_stat_counts(G, fallback_density)
        except Exception:
            pass

    if is_graph_upload and G is not None:
        st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label">Multi-model comparison</p>',
            unsafe_allow_html=True,
        )

        comparison_rows = [
            {"Model": "CNN", "Predicted Vertices": pred_v, "Predicted Edges": pred_e},
            {"Model": "GCN", "Predicted Vertices": gcn_v if gcn_v is not None else "data unavailable", "Predicted Edges": gcn_e if gcn_e is not None else "data unavailable"},
            {"Model": "Graph-statistic baseline", "Predicted Vertices": stat_v if stat_v is not None else "data unavailable", "Predicted Edges": stat_e if stat_e is not None else "data unavailable"},
        ]
        comparison_df = pd.DataFrame(comparison_rows)
        st.table(comparison_df)
    elif not is_graph_upload:
        st.markdown(
            "<div class='neobrutalist-card'><p style='font-size:0.9rem; margin:0;'>Only CNN inference is available for image uploads — GCN and the graph-statistic baseline require the underlying graph topology, which isn't recoverable from a rendered image alone.</p></div>",
            unsafe_allow_html=True,
        )
    
    try:
        save_novel_upload(
            pil_img=pil_img,
            original_filename=active_file.name,
            source_type="graph_upload" if is_graph_upload else "image_upload",
            pred_v=pred_v,
            pred_e=pred_e,
            true_v=true_v_csv,
            true_e=true_e_csv,
            pred_v_gcn=gcn_v,
            pred_e_gcn=gcn_e,
            pred_v_stat=stat_v,
            pred_e_stat=stat_e,
        )
    except Exception:
        pass

    warning_data = load_warning_reference_data()
    max_train_n = warning_data.get("max_train_n")

    if is_graph_upload and G is not None:
        warning_num_nodes = int(G.number_of_nodes())
        warning_density = compute_density(int(G.number_of_nodes()), int(G.number_of_edges()))
    else:
        warning_num_nodes = int(pred_v)
        warning_density = compute_density(int(pred_v), int(pred_e))

    size_triggered = bool(max_train_n is not None and warning_num_nodes > int(max_train_n))
    dense_triggered = density_bucket(warning_density) == "dense"

    if size_triggered or dense_triggered:
        st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
        render_prediction_warning_banner(
            warning_data,
            warning_num_nodes,
            warning_density,
            is_graph_upload=is_graph_upload,
            size_triggered=size_triggered,
            dense_triggered=dense_triggered,
        )
        
    # ── Grad-CAM ─────────────────────────────────────────────────────
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label">Model attention (Grad-CAM)</p>',
        unsafe_allow_html=True,
    )
    show_cam = st.checkbox("Show model attention overlays", value=False)
    
    if show_cam:
        with st.spinner("Computing Grad-CAM…"):
            try:
                overlay_v, overlay_e = compute_gradcam_overlays(
                    pil_img, input_tensor, gcam
                )
            except Exception as exc:
                st.error(f"Grad-CAM computation failed: {exc}")
                overlay_v, overlay_e = None, None
                
        if overlay_v is not None:
            st.markdown(
                '<p class="section-label reveal-5">Vertex-target attention &nbsp;|&nbsp; Edge-target attention</p>',
                unsafe_allow_html=True,
            )
            cam_col1, cam_col2 = st.columns(2)
            with cam_col1:
                render_letterbox(overlay_v, caption="Vertex count target", max_width=420)
            with cam_col2:
                render_letterbox(overlay_e, caption="Edge count target", max_width=420)
                
    # ── Ground Truth & Topological Comparison (if ground truth available) ─────
    has_ground_truth = False
    true_v = 0
    true_e = 0
    if G is not None:
        true_v = G.number_of_nodes()
        true_e = G.number_of_edges()
        has_ground_truth = True
    elif true_v_csv is not None:
        true_v = true_v_csv
        true_e = true_e_csv
        has_ground_truth = True
        
    if has_ground_truth:
        st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label">Empirical Validation (Ground Truth vs Prediction)</p>',
            unsafe_allow_html=True,
        )
        
        err_v_cnn = abs(true_v - pred_v)
        err_e_cnn = abs(true_e - pred_e)
        acc_v_cnn = f"{max(0, 100 - (err_v_cnn / true_v * 100)):.1f}%" if true_v > 0 else "N/A"
        acc_e_cnn = f"{max(0, 100 - (err_e_cnn / true_e * 100)):.1f}%" if true_e > 0 else "N/A"

        if gcn_v is not None and gcn_e is not None:
            err_v_gcn = abs(true_v - gcn_v)
            err_e_gcn = abs(true_e - gcn_e)
            acc_v_gcn = f"{max(0, 100 - (err_v_gcn / true_v * 100)):.1f}%" if true_v > 0 else "N/A"
            acc_e_gcn = f"{max(0, 100 - (err_e_gcn / true_e * 100)):.1f}%" if true_e > 0 else "N/A"
        else:
            acc_v_gcn = "N/A"
            acc_e_gcn = "N/A"

        if stat_v is not None and stat_e is not None:
            err_v_stat = abs(true_v - stat_v)
            err_e_stat = abs(true_e - stat_e)
            acc_v_stat = f"{max(0, 100 - (err_v_stat / true_v * 100)):.1f}%" if true_v > 0 else "N/A"
            acc_e_stat = f"{max(0, 100 - (err_e_stat / true_e * 100)):.1f}%" if true_e > 0 else "N/A"
        else:
            acc_v_stat = "N/A"
            acc_e_stat = "N/A"

        df_compare = pd.DataFrame({
            "Property": ["Vertices (Nodes)", "Edges (Connections)"],
            "Ground Truth": [true_v, true_e],
            "CNN Prediction": [pred_v, pred_e],
            "GCN Prediction": [
                gcn_v if (gcn_v is not None) else "N/A",
                gcn_e if (gcn_e is not None) else "N/A"
            ],
            "Baseline Prediction": [
                stat_v if (stat_v is not None) else "N/A",
                stat_e if (stat_e is not None) else "N/A"
            ],
            "Accuracy (CNN)": [acc_v_cnn, acc_e_cnn],
            "Accuracy (GCN)": [acc_v_gcn, acc_e_gcn],
            "Accuracy (Baseline)": [acc_v_stat, acc_e_stat],
        })
        
        st.table(df_compare)
        
        # Additional topological stats
        if G is not None:
            st.markdown(
                '<p class="section-label">Topological Properties</p>',
                unsafe_allow_html=True,
            )
            
            if true_v > 1:
                density = (2 * true_e) / (true_v * (true_v - 1))
            else:
                density = 0.0
                
            import networkx as nx
            try:
                num_components = nx.number_connected_components(G)
            except Exception:
                num_components = 1
                
            if true_v < 300:
                try:
                    avg_clustering = nx.average_clustering(G)
                    clustering_str = f"{avg_clustering:.4f}"
                except Exception:
                    clustering_str = "N/A"
            else:
                clustering_str = "Skipped (graph too large)"
                
            prop_col1, prop_col2, prop_col3 = st.columns(3)
            with prop_col1:
                st.markdown(
                    f'<div style="text-align: center; padding: 10px; background-color: #FFFFFF; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 4px;">'
                    f'<span style="font-size: 0.65rem; color: #C08A00; text-transform: uppercase; font-weight: 800; letter-spacing: 0.1em;">Graph Density</span>'
                    f'<p style="font-size: 1.8rem; font-family: \'Syne\', sans-serif; color: #000000; margin: 5px 0 0 0; font-weight: 800;">{density:.4f}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with prop_col2:
                st.markdown(
                    f'<div style="text-align: center; padding: 10px; background-color: #FFFFFF; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 4px;">'
                    f'<span style="font-size: 0.65rem; color: #9C47FF; text-transform: uppercase; font-weight: 800; letter-spacing: 0.1em;">Connected Components</span>'
                    f'<p style="font-size: 1.8rem; font-family: \'Syne\', sans-serif; color: #000000; margin: 5px 0 0 0; font-weight: 800;">{num_components}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with prop_col3:
                st.markdown(
                    f'<div style="text-align: center; padding: 10px; background-color: #FFFFFF; border: 2px solid #000000; box-shadow: 3px 3px 0px #000000; border-radius: 4px;">'
                    f'<span style="font-size: 0.65rem; color: #00B5FF; text-transform: uppercase; font-weight: 800; letter-spacing: 0.1em;">Avg Clustering Coeff</span>'
                    f'<p style="font-size: 1.8rem; font-family: \'Syne\', sans-serif; color: #000000; margin: 5px 0 0 0; font-weight: 800;">{clustering_str}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
    st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
    render_limitations_expander(lim_data)

# ---------------------------------------------------------------------------
# Sidebar layout dispatcher
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render sidebar elements: Navigation selection."""
    with st.sidebar:
        st.markdown(
            '<p class="section-label">Navigation</p>', unsafe_allow_html=True
        )
        
        section = st.radio(
            label="Navigation Sections",
            options=["Home", "Predict", "Results", "Research Insights", "Models"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-label">About</p>', unsafe_allow_html=True
        )
        st.markdown(
            """
<span style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.80rem; color:#A0A0AB; line-height:1.6;">
Topolens estimates vertex and edge counts from a rendered graph image
using a custom four-block CNN trained on 2,500 synthetic graphs
(NetworkX generators, Graphviz <code>sfdp</code> layout, 224 &times; 224 px PNG).
Checkpoint: <code>models/checkpoints/cnn_best.pt</code>.
</span>
""",
            unsafe_allow_html=True,
        )

    return section

# ---------------------------------------------------------------------------
# Main application entry
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Topolens — Graph Structure Estimator",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # Load model and limitations data (globally cached)
    try:
        model, normalize_stats, gcam = load_model_and_gradcam()
        lim_data = load_limitations_data()
    except Exception as exc:
        st.error(f"Failed to load model checkpoint or config parameters: {exc}")
        return

    # Sidebar Navigation
    section = render_sidebar()

    # Main column — constrain width for readability
    col_main, col_spacer = st.columns([2, 1])

    with col_main:
        render_hero_title()
        
        if section == "Home":
            render_home_section(lim_data)
        elif section == "Predict":
            render_predict_section(model, normalize_stats, gcam, lim_data)
        elif section == "Results":
            render_results_section()
        elif section == "Research Insights":
            render_research_insights_section()
        elif section == "Models":
            render_models_section(model)


if __name__ == "__main__":
    main()
