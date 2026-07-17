"""Orchestration script for failure-case taxonomy and worst-error grid rendering."""

from __future__ import annotations
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
import topolens_utils
from evaluation.error_analysis import load_checkpoint, predict_split
from topolens_utils import ensure_dir

CAT_COLORS = {
    "out_of_distribution_size": "#8b5cf6",  # purple
    "high_density_clutter": "#ef4444",      # red
    "in_distribution_normal": "#10b981"     # green
}

SPLIT_MARKERS = {
    "test": "o",
    "held_out": "s"
}


def categorize_row(row, max_train_n: int) -> str:
    """Categorize graph prediction results based on taxonomy rules."""
    if row["num_nodes"] > max_train_n:
        return "out_of_distribution_size"
    
    density_val = row["density"]
    if topolens_utils.density_bucket(density_val) == "dense":
        return "high_density_clutter"
        
    return "in_distribution_normal"


def main() -> None:
    print("Starting failure-case taxonomy analysis...")
    
    # 1. Load model and predict splits
    model, normalize_stats = load_checkpoint()
    test_df = predict_split("test", model, normalize_stats)
    held_out_df = predict_split("held_out", model, normalize_stats)
    
    # 2. Compute max_train_n from training dataset
    train_csv = Path(config.DATA_DIR) / "splits" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing training split at {train_csv}")
    
    train_df = pd.read_csv(train_csv)
    max_train_n = int(train_df["num_nodes"].max())
    print(f"Empirically derived training distribution max N: {max_train_n}")
    
    # 3. Categorize graphs in test and held-out splits
    test_df["category"] = test_df.apply(lambda r: categorize_row(r, max_train_n), axis=1)
    held_out_df["category"] = held_out_df.apply(lambda r: categorize_row(r, max_train_n), axis=1)
    
    test_df["split"] = "test"
    held_out_df["split"] = "held_out"
    
    combined_df = pd.concat([test_df, held_out_df], ignore_index=True)
    
    # Save categories file
    results_dir = Path(config.RESULTS_DIR)
    ensure_dir(results_dir)
    combined_df.to_csv(results_dir / "failure_case_categories.csv", index=False)
    print(f"Saved failure case categories to {results_dir / 'failure_case_categories.csv'}")
    
    # 4. Generate failure case summary statistics
    summary_records = []
    for (split_name, category), sub_df in combined_df.groupby(["split", "category"]):
        summary_records.append({
            "split": split_name,
            "category": category,
            "mean_vertex_mae": float(sub_df["abs_vertex_error"].mean()),
            "median_vertex_mae": float(sub_df["abs_vertex_error"].median()),
            "mean_edge_mae": float(sub_df["abs_edge_error"].mean()),
            "median_edge_mae": float(sub_df["abs_edge_error"].median()),
            "n": int(len(sub_df))
        })
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(results_dir / "failure_case_summary.csv", index=False)
    print(f"Saved failure case summary to {results_dir / 'failure_case_summary.csv'}")
    
    # 5. Generate Figures
    figures_dir = Path(config.REPORT_FIGURES_DIR)
    ensure_dir(figures_dir)
    
    # Figure 1: error vs num_nodes
    fig, ax = plt.subplots(figsize=(10, 6))
    for (split_name, category), sub_df in combined_df.groupby(["split", "category"]):
        ax.scatter(
            sub_df["num_nodes"],
            sub_df["abs_edge_error"],
            s=25,
            alpha=0.65,
            color=CAT_COLORS[category],
            marker=SPLIT_MARKERS[split_name],
            label=f"{split_name.capitalize()} - {category.replace('_', ' ')}"
        )
    ax.set_title("Absolute Edge Error vs Number of Nodes by Taxonomy Category", fontsize=12)
    ax.set_xlabel("Number of Nodes (True)")
    ax.set_ylabel("Absolute Edge Error")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
    plt.tight_layout()
    fig.savefig(figures_dir / "error_vs_num_nodes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {figures_dir / 'error_vs_num_nodes.png'}")
    
    # Figure 2: error vs density
    fig, ax = plt.subplots(figsize=(10, 6))
    for (split_name, category), sub_df in combined_df.groupby(["split", "category"]):
        ax.scatter(
            sub_df["density"],
            sub_df["abs_edge_error"],
            s=25,
            alpha=0.65,
            color=CAT_COLORS[category],
            marker=SPLIT_MARKERS[split_name],
            label=f"{split_name.capitalize()} - {category.replace('_', ' ')}"
        )
    ax.set_title("Absolute Edge Error vs Density by Taxonomy Category", fontsize=12)
    ax.set_xlabel("Graph Density (True)")
    ax.set_ylabel("Absolute Edge Error")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb")
    plt.tight_layout()
    fig.savefig(figures_dir / "error_vs_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {figures_dir / 'error_vs_density.png'}")
    
    # Figure 3: Worst-case 15 failure grid
    print("Rendering 15 worst-case images to grid...")
    worst_15 = combined_df.sort_values(by="abs_edge_error", ascending=False).head(15)
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(worst_15.iterrows()):
        ax = axes[idx]
        graph_id = row["graph_id"]
        
        img_path = Path(row["image_path"])
        if not img_path.exists():
            img_path = Path(config.DATA_DIR).parent / row["image_path"]
            
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        
        actual_v = row["num_nodes"]
        actual_e = row["num_edges"]
        pred_v = row["pred_num_vertices"]
        pred_e = row["pred_num_edges"]
        edge_err = row["abs_edge_error"]
        cat = row["category"]
        split = row["split"]
        
        ax.set_title(f"Rank {idx+1}: {graph_id}", fontsize=10, fontweight="bold")
        label_text = (
            f"Split: {split}\n"
            f"V: {pred_v} (Act: {actual_v})\n"
            f"E: {pred_e} (Act: {actual_e})\n"
            f"Edge Error: {edge_err}\n"
            f"Category: {cat.replace('_', ' ')}"
        )
        # Position label text cleanly below each subplot
        ax.text(0.5, -0.05, label_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec="#d1d5db", alpha=0.95))
        
    plt.tight_layout()
    fig.savefig(figures_dir / "worst_case_image_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid figure: {figures_dir / 'worst_case_image_grid.png'}")


if __name__ == "__main__":
    main()
