"""Orchestration script for Grad-CAM interpretation."""

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

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from evaluation.evaluate import load_cnn_model, safe_counts
from models.dataset import build_tensor_only_transform, build_image_transform
from models.gradcam import GradCAM, overlay_cam
from topolens_utils import ensure_dir

def main():
    print("Starting Grad-CAM interpretation...")
    ensure_dir(Path(config.REPORT_FIGURES_DIR) / "gradcam")
    
    # 1. Load the frozen CNN + normalize_stats
    checkpoint_path = Path(config.CHECKPOINTS_DIR) / "cnn_best.pt"
    model, normalize_stats = load_cnn_model(checkpoint_path)
    gradcam = GradCAM(model)
    
    # Setup transforms
    img_transform = build_image_transform(normalize_stats[0], normalize_stats[1])
    
    # 2. Systematic sample (20 "typical case" graphs)
    test_df = pd.read_csv(Path(config.DATA_DIR) / "splits" / "test.csv")
    rng = np.random.default_rng(config.RANDOM_SEED)
    
    systematic_samples = []
    for gen in config.GENERATORS:
        for tier in config.NODE_TIERS.keys():
            subset = test_df[(test_df["generator"] == gen) & (test_df["tier"] == tier)].copy()
            if not subset.empty:
                subset = subset.sort_values(by="graph_id")
                # shuffle indices deterministically
                indices = np.arange(len(subset))
                rng.shuffle(indices)
                sampled_row = subset.iloc[indices[0]]
                systematic_samples.append(sampled_row)
                
    systematic_df = pd.DataFrame(systematic_samples)
    
    # 3. Failure-case sample (10 worst-error graphs)
    worst_csv = Path(config.DATA_DIR).parent / "evaluation" / "results" / "cnn_worst_edge_errors.csv"
    if worst_csv.exists():
        failure_df = pd.read_csv(worst_csv)
    else:
        print(f"Warning: {worst_csv} not found, skipping failure cases.")
        failure_df = pd.DataFrame()
        
    all_samples = pd.concat([systematic_df, failure_df]).drop_duplicates(subset=["graph_id"])
    print(f"Generating CAMs for {len(all_samples)} graphs...")
    
    # For annotating prediction errors
    preds_dict = {}
    
    # 4. Generate CAMs
    for _, row in all_samples.iterrows():
        graph_id = row["graph_id"]
        img_path = Path(config.DATA_DIR) / "images" / f"{graph_id}.png"
        
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            input_tensor = img_transform(rgb_img).unsqueeze(0)
            
        # Get predictions
        with torch.no_grad():
            preds = model(input_tensor)
            raw_counts = safe_counts(preds.cpu().numpy())[0]
            preds_dict[graph_id] = raw_counts
            
        # Vertex CAM
        cam_v = gradcam.generate(input_tensor, target_index=0)
        overlay_v = overlay_cam(rgb_img, cam_v)
        overlay_v.save(Path(config.REPORT_FIGURES_DIR) / "gradcam" / f"{graph_id}_vertex.png")
        
        # Edge CAM
        cam_e = gradcam.generate(input_tensor, target_index=1)
        overlay_e = overlay_cam(rgb_img, cam_e)
        overlay_e.save(Path(config.REPORT_FIGURES_DIR) / "gradcam" / f"{graph_id}_edge.png")
        
    # 5. Composite grid figures
    print("Assembling composite grids...")
    
    # Grid 1: by generator
    # one representative example per generator (5 rows)
    fig, axes = plt.subplots(len(config.GENERATORS), 2, figsize=(8, 4 * len(config.GENERATORS)))
    for idx, gen in enumerate(config.GENERATORS):
        # find the first sampled graph for this generator
        gen_rows = systematic_df[systematic_df["generator"] == gen]
        if not gen_rows.empty:
            graph_id = gen_rows.iloc[0]["graph_id"]
            v_img = Image.open(Path(config.REPORT_FIGURES_DIR) / "gradcam" / f"{graph_id}_vertex.png")
            e_img = Image.open(Path(config.REPORT_FIGURES_DIR) / "gradcam" / f"{graph_id}_edge.png")
            
            axes[idx, 0].imshow(v_img)
            axes[idx, 0].axis("off")
            axes[idx, 0].set_title(f"{gen}\n{graph_id} (Vertex CAM)")
            
            axes[idx, 1].imshow(e_img)
            axes[idx, 1].axis("off")
            axes[idx, 1].set_title(f"{gen}\n{graph_id} (Edge CAM)")
            
    plt.tight_layout()
    fig.savefig(Path(config.REPORT_FIGURES_DIR) / "gradcam_grid_by_generator.png", dpi=150)
    plt.close(fig)
    
    # Grid 2: failure cases
    if not failure_df.empty:
        n_failures = len(failure_df)
        fig, axes = plt.subplots(n_failures, 2, figsize=(8, 4 * n_failures))
        
        # Ensure axes is 2D even if n_failures == 1
        if n_failures == 1:
            axes = np.expand_dims(axes, 0)
            
        for idx, row in failure_df.iterrows():
            graph_id = row["graph_id"]
            v_img = Image.open(Path(config.REPORT_FIGURES_DIR) / "gradcam" / f"{graph_id}_vertex.png")
            e_img = Image.open(Path(config.REPORT_FIGURES_DIR) / "gradcam" / f"{graph_id}_edge.png")
            
            pred_v, pred_e = preds_dict.get(graph_id, (0, 0))
            
            # Use real ground truth from the df if available, else from row dict
            actual_e = row.get("num_edges", "?")
            
            axes[idx, 0].imshow(v_img)
            axes[idx, 0].axis("off")
            axes[idx, 0].set_title(f"{graph_id} (Vertex CAM)")
            
            axes[idx, 1].imshow(e_img)
            axes[idx, 1].axis("off")
            axes[idx, 1].set_title(f"Edge CAM\nPred: {pred_e}, Act: {actual_e}")
            
        plt.tight_layout()
        fig.savefig(Path(config.REPORT_FIGURES_DIR) / "gradcam_grid_failure_cases.png", dpi=150)
        plt.close(fig)

    print("Grad-CAM generation complete.")

if __name__ == "__main__":
    main()
