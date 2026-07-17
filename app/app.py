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
from pathlib import Path
from typing import Optional, Tuple

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
from render.render_graphs import render_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "cnn_best.pt"
TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
INPUT_SIZE: Tuple[int, int] = (224, 224)

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
    """Load CNN checkpoint, compute normalization stats, build GradCAM.

    The checkpoint file is a wrapped dict saved by train_cnn.py with keys:
    ``state_dict``, ``normalize_stats``, ``epoch``, ``best_val_loss``, etc.
    We unpack accordingly and reuse the stored normalize_stats so inference
    uses exactly the same mean/std as training.

    Returns
    -------
    model : CustomCNNRegressor
        Loaded model in eval mode.
    normalize_stats : tuple
        (mean, std) tuples used by build_image_transform.
    gcam : GradCAM
        GradCAM instance with hooks already registered.
    """
    # weights_only=False is required because the checkpoint contains Python
    # dicts with metadata beyond plain tensors.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # Unpack wrapped checkpoint dict (saved by train_cnn.py)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        normalize_stats = checkpoint.get("normalize_stats", None)
    else:
        # Fallback: assume raw state_dict
        state_dict = checkpoint
        normalize_stats = None

    model = CustomCNNRegressor()
    model.load_state_dict(state_dict)
    model.eval()

    # If normalize_stats not in checkpoint, compute from training split
    if normalize_stats is None:
        normalize_stats = compute_normalization_stats(
            str(TRAIN_CSV), str(IMAGES_DIR), INPUT_SIZE
        )

    gcam = GradCAM(model)
    return model, normalize_stats, gcam


@st.cache_data(show_spinner=False)
def load_limitations_data() -> dict:
    """Read real numbers from CSVs for the Model Limitations expander.

    Returns a dict with the key metrics so the expander bullets are always
    in sync with the on-disk results files.
    """
    data: dict = {}

    # failure_case_summary.csv
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

    # probe_summary.csv — overall row, constant_node_size vs original
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
    """Tries to find ground-truth graph structure and counts for a file name.

    Looks up the stem in labels_processed.csv.
    """
    import pandas as pd
    stem = Path(filename).stem
    csv_path = PROJECT_ROOT / "data" / "processed" / "labels_processed.csv"
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
    """Pre-process image, run model forward pass.

    Returns
    -------
    pred_vertices, pred_edges : int
        Rounded counts in raw (non-log) space.
    input_tensor : torch.Tensor
        The pre-processed tensor (kept for GradCAM).
    """
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
    """Compute vertex-target and edge-target Grad-CAM overlays."""
    cam_v = gcam.generate(input_tensor, target_index=0)
    cam_e = gcam.generate(input_tensor, target_index=1)
    overlay_v = overlay_cam(pil_img.copy(), cam_v)
    overlay_e = overlay_cam(pil_img.copy(), cam_e)
    return overlay_v, overlay_e


def parse_graph_file(uploaded_file) -> "nx.Graph":  # type: ignore[name-defined]
    """Parse an uploaded .graphml or edge-list (.csv / .txt) file."""
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
        # Treat as whitespace/comma-separated edge list: src dst [weight]
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
    """Render a NetworkX graph server-side and return a PIL Image."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = render_graph(G, graph_id="uploaded_graph", out_dir=tmp_dir)
        img_path = result["image_path"]
        img = Image.open(img_path).convert("RGB")
        # Copy pixels into memory before the tmpdir is cleaned up
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
</style>

"""


def inject_css() -> None:
    """Inject the Editorial design-system CSS once per run."""
    st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

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
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> object:
    """Render sidebar inputs. Returns list of uploaded files, or None."""
    with st.sidebar:
        st.markdown(
            '<p class="section-label">Upload</p>', unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            label="Upload graph images or files",
            type=["png", "jpg", "jpeg", "graphml", "csv", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
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
(NetworkX generators, Graphviz <code>sfdp</code> layout, 224 × 224 px PNG).
Checkpoint: <code>models/checkpoints/cnn_best.pt</code>.
</span>
""",
            unsafe_allow_html=True,
        )

    return uploaded


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    st.set_page_config(
        page_title="Topolens — Graph Structure Estimator",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    # Sidebar
    uploaded = render_sidebar()

    # Main column — constrain width for readability
    col_main, col_spacer = st.columns([2, 1])

    with col_main:
        render_hero_title()

        # Load model (cached)
        try:
            model, normalize_stats, gcam = load_model_and_gradcam()
            lim_data = load_limitations_data()
        except Exception as exc:
            st.error(f"Failed to load model checkpoint: {exc}")
            return

        # ── Empty state ─────────────────────────────────────────────────
        if not uploaded:  # Handles None or empty list
            st.markdown(
                '<p style="font-family:\'Plus Jakarta Sans\', sans-serif; font-size:1.05rem; font-weight:500;" class="reveal-2">No input yet — upload one or more '
                "graph images or graph files in the sidebar.</p>",
                unsafe_allow_html=True,
            )
            example = _find_example_image()
            if example:
                ex_img = Image.open(example).convert("RGB")
                st.markdown(
                    '<p style="font-family:\'Plus Jakarta Sans\', sans-serif; font-size:0.95rem; font-weight:600; margin-bottom:0.5rem;" class="reveal-3">Example input (from training split)</p>',
                    unsafe_allow_html=True,
                )
                render_letterbox(ex_img, caption=f"data/images/{example.name}", max_width=380)

            render_limitations_expander(lim_data)
            return

        # ── Select active file if multiple uploaded ──────────────────────
        if len(uploaded) > 1:
            st.markdown(
                '<p style="font-family:\'Plus Jakarta Sans\', sans-serif; font-size:0.95rem; font-weight:600; margin-bottom:0.5rem;" class="reveal-2">Select file to analyze</p>',
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

        # ── Process active upload ─────────────────────────────────────────
        pil_img: Optional[Image.Image] = None
        G: Optional[object] = None
        true_v_csv: Optional[int] = None
        true_e_csv: Optional[int] = None
        suffix = Path(active_file.name).suffix.lower()

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

        # ── Image preview ────────────────────────────────────────────────
        st.markdown(
            '<p class="section-label reveal-2">Input graph image</p>',
            unsafe_allow_html=True,
        )
        render_letterbox(pil_img, max_width=420)

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
                    render_letterbox(overlay_v, caption="Vertex count target", max_width=340)
                with cam_col2:
                    render_letterbox(overlay_e, caption="Edge count target", max_width=340)

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
            
            err_v = abs(true_v - pred_v)
            err_e = abs(true_e - pred_e)
            
            # Build DataFrame for display
            df_compare = pd.DataFrame({
                "Property": ["Vertices (Nodes)", "Edges (Connections)"],
                "Ground Truth": [true_v, true_e],
                "CNN Prediction": [pred_v, pred_e],
                "Absolute Error": [err_v, err_e],
                "Accuracy": [
                    f"{max(0, 100 - (err_v / true_v * 100)):.1f}%" if true_v > 0 else "N/A",
                    f"{max(0, 100 - (err_e / true_e * 100)):.1f}%" if true_e > 0 else "N/A"
                ]
            })
            
            st.table(df_compare)
            
            # Additional topological stats (if G structure is loaded/parsed)
            if G is not None:
                st.markdown(
                    '<p class="section-label">Topological Properties</p>',
                    unsafe_allow_html=True,
                )
                
                # Density
                if true_v > 1:
                    density = (2 * true_e) / (true_v * (true_v - 1))
                else:
                    density = 0.0
                    
                # Connected components
                import networkx as nx
                try:
                    num_components = nx.number_connected_components(G)
                except Exception:
                    num_components = 1
                    
                # Clustering coefficient
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

        # ── Limitations ──────────────────────────────────────────────────
        st.markdown('<hr class="topo-divider">', unsafe_allow_html=True)
        render_limitations_expander(lim_data)


if __name__ == "__main__":
    main()
