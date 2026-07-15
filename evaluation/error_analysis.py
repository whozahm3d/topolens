"""CNN error analysis for Topolens."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from models.cnn_model import CustomCNNRegressor
from models.dataset import TopolensImageDataset
from topolens_utils import ensure_dir, load_yaml_config


COLOR_MAP = {"sparse": "#2563eb", "medium": "#f59e0b", "dense": "#dc2626"}


def load_checkpoint():
    checkpoint_path = Path(config.DATA_DIR).parent / "models" / "checkpoints" / "cnn_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CNN checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = CustomCNNRegressor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["normalize_stats"]


def predict_split(split_name: str, model, normalize_stats) -> pd.DataFrame:
    split_csv = Path(config.DATA_DIR) / "splits" / f"{split_name}.csv"
    df = pd.read_csv(split_csv).reset_index(drop=True)
    dataset = TopolensImageDataset(split_csv, config.IMAGES_DIR, normalize_stats=normalize_stats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(load_yaml_config()["model"]["cnn"]["batch_size"]), shuffle=False, num_workers=0)

    preds = []
    with torch.no_grad():
        for images, _ in loader:
            preds.append(model(images))
    stacked = torch.nan_to_num(torch.cat(preds, dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    raw_preds = torch.expm1(stacked)
    raw_preds = torch.nan_to_num(raw_preds, nan=0.0, posinf=1_000_000.0, neginf=0.0)
    raw_preds = raw_preds.clamp(min=0.0, max=1_000_000.0).round().to(torch.int64).numpy()

    out = df.copy()
    out["pred_num_vertices"] = raw_preds[:, 0]
    out["pred_num_edges"] = raw_preds[:, 1]
    out["abs_edge_error"] = (out["pred_num_edges"] - out["num_edges"]).abs()
    out["abs_vertex_error"] = (out["pred_num_vertices"] - out["num_nodes"]).abs()
    out["density"] = out["density"] if "density" in out.columns else out.apply(lambda row: 0.0 if row["num_nodes"] < 2 else (2 * row["num_edges"]) / (row["num_nodes"] * (row["num_nodes"] - 1)), axis=1)
    return out


def plot_faceted_scatter(df: pd.DataFrame, split_name: str, target_col: str, actual_col: str, title_suffix: str) -> None:
    generators = sorted(df["generator"].unique())
    ncols = 2
    nrows = int(np.ceil(len(generators) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)

    for idx, generator in enumerate(generators):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["generator"] == generator]
        for bucket, bucket_df in sub.groupby("density_bucket"):
            ax.scatter(
                bucket_df[actual_col],
                bucket_df[target_col],
                s=12,
                alpha=0.65,
                color=COLOR_MAP.get(bucket, "#6b7280"),
                label=bucket,
            )
        bounds = [0, max(sub[actual_col].max(), sub[target_col].max()) if len(sub) else 1]
        ax.plot(bounds, bounds, linestyle="--", color="#111827", linewidth=1)
        ax.set_title(generator)
        ax.set_xlabel(f"Actual {actual_col}")
        ax.set_ylabel(f"Predicted {target_col}")
        ax.grid(alpha=0.2)

    for idx in range(len(generators), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"CNN predicted vs actual {title_suffix} ({split_name})", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(config.DATA_DIR).parent / "report" / "figures" / f"cnn_{target_col}_scatter_{split_name}.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_worst_error_table(test_df: pd.DataFrame, held_out_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([test_df.assign(split="test"), held_out_df.assign(split="held_out")], ignore_index=True)
    combined["abs_edge_error"] = (combined["pred_num_edges"] - combined["num_edges"]).abs()
    worst = combined.sort_values("abs_edge_error", ascending=False).head(10).copy()
    worst["density"] = worst["density"].round(4)
    return worst[
        [
            "split",
            "graph_id",
            "generator",
            "tier",
            "density",
            "num_nodes",
            "pred_num_vertices",
            "num_edges",
            "pred_num_edges",
            "abs_edge_error",
        ]
    ]


def main() -> None:
    model, normalize_stats = load_checkpoint()
    test_df = predict_split("test", model, normalize_stats)
    held_out_df = predict_split("held_out", model, normalize_stats)

    for split_name, df in (("test", test_df), ("held_out", held_out_df)):
        plot_faceted_scatter(df, split_name, "pred_num_vertices", "num_nodes", "vertex count")
        plot_faceted_scatter(df, split_name, "pred_num_edges", "num_edges", "edge count")

    worst = build_worst_error_table(test_df, held_out_df)
    out_path = Path(config.DATA_DIR).parent / "evaluation" / "results" / "cnn_worst_edge_errors.csv"
    ensure_dir(out_path.parent)
    worst.to_csv(out_path, index=False)
    print(worst.to_string(index=False))
    print(f"Saved worst-error table to {out_path}")


if __name__ == "__main__":
    main()
