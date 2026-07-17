"""Orchestration script for ink-coverage / dot-size correlational analysis."""

from __future__ import annotations
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from evaluation.error_analysis import load_checkpoint, predict_split
from evaluation.image_statistics import compute_ink_fraction, compute_component_stats
from topolens_utils import ensure_dir

COLOR_MAP = {"sparse": "#2563eb", "medium": "#f59e0b", "dense": "#dc2626"}


def add_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    ink_fractions = []
    num_comps = []
    mean_areas = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing image statistics"):
        img_path = Path(row["image_path"])
        # Ensure path is resolved relative to repo root if needed
        if not img_path.exists():
            img_path = Path(config.DATA_DIR).parent / row["image_path"]
            
        ink_f = compute_ink_fraction(img_path)
        n_c, m_a = compute_component_stats(img_path)
        
        ink_fractions.append(ink_f)
        num_comps.append(n_c)
        mean_areas.append(m_a)
        
    df["ink_fraction"] = ink_fractions
    df["num_components"] = num_comps
    df["mean_component_area"] = mean_areas
    return df


def compute_correlations(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    pairs = [
        ("ink_fraction", "num_nodes"),
        ("ink_fraction", "num_edges"),
        ("ink_fraction", "pred_num_vertices"),
        ("ink_fraction", "pred_num_edges"),
        ("mean_component_area", "num_nodes"),
        ("mean_component_area", "pred_num_vertices"),
        ("num_components", "num_nodes"),
    ]
    
    records = []
    for x_col, y_col in pairs:
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        r_val, p_val = pearsonr(x, y)
        records.append({
            "split": split_name,
            "metric_x": x_col,
            "metric_y": y_col,
            "pearson_r": float(r_val),
            "p_value": float(p_val),
            "n": int(len(df))
        })
    return pd.DataFrame(records)


def main() -> None:
    print("Starting ink-coverage and component size analysis...")
    
    # 1. Load model and predict splits
    model, normalize_stats = load_checkpoint()
    test_df = predict_split("test", model, normalize_stats)
    held_out_df = predict_split("held_out", model, normalize_stats)
    
    # 2. Compute statistics for all images in both splits
    print("Processing test split images...")
    test_df = add_image_stats(test_df)
    print("Processing held_out split images...")
    held_out_df = add_image_stats(held_out_df)
    
    # 3. Compute Pearson correlations
    test_corr = compute_correlations(test_df, "test")
    held_out_corr = compute_correlations(held_out_df, "held_out")
    
    correlations_df = pd.concat([test_corr, held_out_corr], ignore_index=True)
    results_dir = Path(config.RESULTS_DIR)
    ensure_dir(results_dir)
    correlations_df.to_csv(results_dir / "ink_coverage_correlations.csv", index=False)
    print(f"Saved correlations table to {results_dir / 'ink_coverage_correlations.csv'}")
    
    # 4. Generate Figures
    figures_dir = Path(config.REPORT_FIGURES_DIR)
    ensure_dir(figures_dir)
    
    # Figure 1: ink_fraction vs num_nodes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (split_name, df) in enumerate([("test", test_df), ("held_out", held_out_df)]):
        ax = axes[idx]
        for bucket, sub_df in df.groupby("density_bucket"):
            ax.scatter(
                sub_df["num_nodes"],
                sub_df["ink_fraction"],
                s=15,
                alpha=0.7,
                color=COLOR_MAP.get(bucket, "#6b7280"),
                label=bucket
            )
        ax.set_title(f"{split_name.capitalize()} Split")
        ax.set_xlabel("Number of Nodes (True)")
        ax.set_ylabel("Ink Fraction")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Ink Fraction vs Node Count by Density Bucket", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / "ink_coverage_vs_node_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {figures_dir / 'ink_coverage_vs_node_count.png'}")
    
    # Figure 2: mean_component_area vs num_nodes (log-log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (split_name, df) in enumerate([("test", test_df), ("held_out", held_out_df)]):
        ax = axes[idx]
        # Filter out invalid values to avoid log(0) issues safely
        valid_df = df[df["mean_component_area"] > 0]
        for bucket, sub_df in valid_df.groupby("density_bucket"):
            ax.scatter(
                sub_df["num_nodes"],
                sub_df["mean_component_area"],
                s=15,
                alpha=0.7,
                color=COLOR_MAP.get(bucket, "#6b7280"),
                label=bucket
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{split_name.capitalize()} Split")
        ax.set_xlabel("Number of Nodes (True) [Log Scale]")
        ax.set_ylabel("Mean Component Area (Pixels) [Log Scale]")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.suptitle("Mean Connected Component Size vs Node Count by Density Bucket", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / "component_size_vs_node_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {figures_dir / 'component_size_vs_node_count.png'}")


if __name__ == "__main__":
    main()
