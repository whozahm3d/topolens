"""Run inference on probe variants to analyze size shortcuts and layout sensitivity."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from evaluation.evaluate import load_cnn_model, safe_counts
from models.dataset import build_image_transform
from topolens_utils import ensure_dir

def predict_single_image(img_path: Path, transform, model) -> tuple[int, int]:
    with Image.open(img_path) as img:
        rgb_img = img.convert("RGB")
        input_tensor = transform(rgb_img).unsqueeze(0)
    
    with torch.no_grad():
        preds = model(input_tensor)
        raw_counts = safe_counts(preds.cpu().numpy())[0]
        
    return int(raw_counts[0]), int(raw_counts[1])

def main():
    print("Starting shortcut probe inference...")
    
    manifest_path = Path(config.DATA_DIR) / "processed" / "probe_manifest.csv"
    if not manifest_path.exists():
        print("Probe manifest not found. Run render_probe_variants.py first.")
        return
        
    manifest_df = pd.read_csv(manifest_path)
    
    checkpoint_path = Path(config.CHECKPOINTS_DIR) / "cnn_best.pt"
    model, normalize_stats = load_cnn_model(checkpoint_path)
    img_transform = build_image_transform(normalize_stats[0], normalize_stats[1])
    
    results = []
    
    unique_graphs = manifest_df.drop_duplicates(subset=["graph_id"])
    
    for _, row in tqdm(unique_graphs.iterrows(), total=len(unique_graphs), desc="Probing"):
        graph_id = row["graph_id"]
        actual_v = row["num_nodes"]
        actual_e = row["num_edges"]
        
        variants_for_graph = manifest_df[manifest_df["graph_id"] == graph_id]
        
        # 1. Original
        orig_img_path = Path(config.DATA_DIR) / "images" / f"{graph_id}.png"
        pred_v, pred_e = predict_single_image(orig_img_path, img_transform, model)
        # Note: Original layouts are networkx_spring if generated on this machine previously
        # We assume "original" here.
        results.append({
            "graph_id": graph_id,
            "generator": row["generator"],
            "tier": row["tier"],
            "num_nodes": actual_v,
            "num_edges": actual_e,
            "variant": "original",
            "pred_num_vertices": pred_v,
            "pred_num_edges": pred_e,
            "abs_vertex_error": abs(pred_v - actual_v),
            "abs_edge_error": abs(pred_e - actual_e),
            "layout_algorithm_used": "original" 
        })
        
        # 2. Probe variants
        for _, v_row in variants_for_graph.iterrows():
            img_path = Path(v_row["image_path"])
            pred_v, pred_e = predict_single_image(img_path, img_transform, model)
            results.append({
                "graph_id": graph_id,
                "generator": row["generator"],
                "tier": row["tier"],
                "num_nodes": actual_v,
                "num_edges": actual_e,
                "variant": v_row["variant"],
                "pred_num_vertices": pred_v,
                "pred_num_edges": pred_e,
                "abs_vertex_error": abs(pred_v - actual_v),
                "abs_edge_error": abs(pred_e - actual_e),
                "layout_algorithm_used": v_row["layout_algorithm_used"]
            })
            
    results_df = pd.DataFrame(results)
    ensure_dir(Path(config.RESULTS_DIR))
    
    # Save predictions
    results_df.to_csv(Path(config.RESULTS_DIR) / "probe_predictions.csv", index=False)
    
    # Summary
    summary_records = []
    
    def add_summary_block(breakdown: str, group_value: str, sub_df: pd.DataFrame):
        for variant in ["original", "constant_node_size", "alt_layout"]:
            v_df = sub_df[sub_df["variant"] == variant]
            if not v_df.empty:
                mean_v_mae = v_df["abs_vertex_error"].mean()
                mean_e_mae = v_df["abs_edge_error"].mean()
                summary_records.append({
                    "breakdown": breakdown,
                    "group_value": group_value,
                    "variant": variant,
                    "vertex_mae": mean_v_mae,
                    "edge_mae": mean_e_mae,
                    "n": len(v_df)
                })
                
    add_summary_block("overall", "all", results_df)
    for tier, sub_df in results_df.groupby("tier"):
        add_summary_block("tier", str(tier), sub_df)
        
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(Path(config.RESULTS_DIR) / "probe_summary.csv", index=False)
    
    # Figure 1: grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    overall = summary_df[summary_df["breakdown"] == "overall"]
    
    variants = ["original", "constant_node_size", "alt_layout"]
    v_maes = [overall[overall["variant"] == v]["vertex_mae"].values[0] for v in variants]
    e_maes = [overall[overall["variant"] == v]["edge_mae"].values[0] for v in variants]
    
    x = np.arange(len(variants))
    width = 0.6
    
    axes[0].bar(x, v_maes, width, color=['#4F46E5', '#10B981', '#F59E0B'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants)
    axes[0].set_title("Vertex MAE")
    
    axes[1].bar(x, e_maes, width, color=['#4F46E5', '#10B981', '#F59E0B'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants)
    axes[1].set_title("Edge MAE")
    
    plt.tight_layout()
    ensure_dir(Path(config.REPORT_FIGURES_DIR))
    fig.savefig(Path(config.REPORT_FIGURES_DIR) / "probe_variant_mae_comparison.png", dpi=150)
    plt.close(fig)
    
    # Figure 2: Examples grid
    # Pick 5 examples from different tiers
    example_ids = []
    tiers_available = list(config.NODE_TIERS.keys())
    # Try to get 1 from each tier, plus one more
    for tier in tiers_available:
        tier_df = unique_graphs[unique_graphs["tier"] == tier]
        if not tier_df.empty:
            example_ids.append(tier_df.iloc[0]["graph_id"])
    
    if len(example_ids) < 5 and len(unique_graphs) >= 5:
        remaining = unique_graphs[~unique_graphs["graph_id"].isin(example_ids)]["graph_id"].tolist()
        example_ids.extend(remaining[:5 - len(example_ids)])
        
    n_examples = len(example_ids)
    if n_examples > 0:
        fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))
        if n_examples == 1:
            axes = np.expand_dims(axes, 0)
            
        for i, graph_id in enumerate(example_ids):
            row_data = unique_graphs[unique_graphs["graph_id"] == graph_id].iloc[0]
            actual_v = row_data["num_nodes"]
            actual_e = row_data["num_edges"]
            
            for j, variant in enumerate(variants):
                res_row = results_df[(results_df["graph_id"] == graph_id) & (results_df["variant"] == variant)].iloc[0]
                pred_v = res_row["pred_num_vertices"]
                pred_e = res_row["pred_num_edges"]
                
                if variant == "original":
                    img_path = Path(config.DATA_DIR) / "images" / f"{graph_id}.png"
                else:
                    img_path = Path(config.DATA_DIR) / "images_probe" / variant / f"{graph_id}.png"
                    
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                
                if i == 0:
                    axes[i, j].set_title(variant)
                
                label = f"V: {pred_v} (Act: {actual_v})\nE: {pred_e} (Act: {actual_e})"
                axes[i, j].text(0.5, -0.1, label, transform=axes[i, j].transAxes, 
                                ha="center", va="top", fontsize=10)
                                
        plt.tight_layout()
        fig.savefig(Path(config.REPORT_FIGURES_DIR) / "probe_variant_examples_grid.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    print("Probe analysis complete.")

if __name__ == "__main__":
    main()
