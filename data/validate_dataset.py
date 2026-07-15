# data/validate_dataset.py
import os
import sys
import json
import pandas as pd
import networkx as nx
# pyrefly: ignore [missing-import]
from PIL import Image

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def validate_dataset():
    print("Starting dataset validation...")
    
    labels_csv = getattr(config, "PROCESSED_LABELS_CSV", config.LABELS_CSV)
    if not os.path.exists(labels_csv):
        labels_csv = config.LABELS_CSV
    if not os.path.exists(labels_csv):
        print(f"Error: {labels_csv} does not exist. Run rendering script first.")
        sys.exit(1)

    try:
        df = pd.read_csv(labels_csv)
    except Exception as e:
        print(f"Error reading labels.csv: {e}")
        sys.exit(1)
        
    errors = []
    
    # 1. Check duplicate graph_ids
    duplicate_ids = df[df.duplicated("graph_id")]["graph_id"].tolist()
    if duplicate_ids:
        errors.append(f"Duplicate graph_ids found: {duplicate_ids}")
        
    print(f"Verifying {len(df)} rows in labels.csv...")
    for idx, row in df.iterrows():
        graph_id = row["graph_id"]
        graph_path = row["graph_path"]
        image_path = row["image_path"]
        expected_nodes = row["num_nodes"]
        expected_edges = row["num_edges"]
        
        # 2. Check image existence and size
        if not os.path.exists(image_path):
            errors.append(f"[{graph_id}] Image file not found: {image_path}")
            continue
            
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width != config.IMAGE_SIZE or height != config.IMAGE_SIZE:
                    errors.append(f"[{graph_id}] Image size mismatch. Expected {config.IMAGE_SIZE}x{config.IMAGE_SIZE}, got {width}x{height}")
        except Exception as e:
            errors.append(f"[{graph_id}] Failed to open image: {e}")
            
        # 3. Check graphml reloading and nodes/edges consistency
        if not os.path.exists(graph_path):
            errors.append(f"[{graph_id}] GraphML file not found: {graph_path}")
            continue
            
        try:
            G = nx.read_graphml(graph_path)
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            if num_nodes != expected_nodes:
                errors.append(f"[{graph_id}] Node count mismatch. CSV: {expected_nodes}, GraphML: {num_nodes}")
            if num_edges != expected_edges:
                errors.append(f"[{graph_id}] Edge count mismatch. CSV: {expected_edges}, GraphML: {num_edges}")
        except Exception as e:
            errors.append(f"[{graph_id}] Failed to read GraphML: {e}")
            
    # Report errors and exit if validation failed
    if errors:
        print("\n--- VALIDATION FAILED ---")
        for err in errors[:50]:  # Cap print to first 50 errors
            print(f"  * {err}")
        if len(errors) > 50:
            print(f"  * ... and {len(errors) - 50} more errors.")
        sys.exit(1)
        
    print("\n--- VALIDATION PASSED ---")
    print("All image files exist, have correct resolution, and match GraphML structure.")
    
    # 4. Print Summary Statistics
    print("\n================ DATASET SUMMARY ================")
    print(f"Total Graphs: {len(df)}")
    
    # Counts by source
    print("\nCounts by Source:")
    source_counts = df["source"].value_counts()
    for src, count in source_counts.items():
        print(f"  - {src}: {count}")
        
    # Counts per (generator, tier)
    print("\nCounts per (generator, tier):")
    # For synthetic graphs
    syn_df = df[df["source"] == "synthetic"]
    cell_counts = syn_df.groupby(["generator", "tier"]).size().unstack(fill_value=0)
    print(cell_counts)
    
    # For real graphs
    real_df = df[df["source"] == "real"]
    if not real_df.empty:
        real_cell_counts = real_df.groupby(["generator", "tier"]).size().unstack(fill_value=0)
        print("\nReal Dataset Tiers:")
        print(real_cell_counts)
        
    # Stats (min, max, mean) of num_nodes and num_edges overall and per generator
    print("\nOverall structural statistics:")
    print(df[["num_nodes", "num_edges"]].describe().loc[["min", "max", "mean"]])
    
    print("\nStructural statistics per generator/dataset:")
    for gen_name, sub_df in df.groupby("generator"):
        print(f"\nGenerator: {gen_name}")
        print(sub_df[["num_nodes", "num_edges"]].describe().loc[["min", "max", "mean"]])
        
    print("=================================================")


if __name__ == "__main__":
    validate_dataset()
