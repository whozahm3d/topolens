"""Generates controlled probe variants (constant node size and alternative layouts)."""

import os
import sys
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from render.render_graphs import render_graph
from topolens_utils import ensure_dir, load_yaml_config

def main():
    print("Starting probe variants rendering...")
    
    cfg = load_yaml_config()
    
    # 1. Load data/splits/test.csv only
    test_csv = Path(config.DATA_DIR) / "splits" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}")
    
    df = pd.read_csv(test_csv)
    
    # 2. Stratified sample (min(2, available) rows for each generator x tier)
    rng = np.random.default_rng(config.RANDOM_SEED)
    sampled_rows = []
    
    for gen in config.GENERATORS:
        for tier in config.NODE_TIERS.keys():
            subset = df[(df["generator"] == gen) & (df["tier"] == tier)]
            if not subset.empty:
                n_samples = min(2, len(subset))
                # Sort first for deterministic shuffling base
                subset = subset.sort_values(by="graph_id")
                indices = np.arange(len(subset))
                rng.shuffle(indices)
                for idx in indices[:n_samples]:
                    sampled_rows.append(subset.iloc[idx])
                    
    probe_df_source = pd.DataFrame(sampled_rows)
    print(f"Sampled {len(probe_df_source)} graphs for probing.")
    
    # Setup directories
    probe_dir = Path(config.DATA_DIR) / "images_probe"
    constant_size_dir = probe_dir / "constant_node_size"
    alt_layout_dir = probe_dir / "alt_layout"
    
    ensure_dir(constant_size_dir)
    ensure_dir(alt_layout_dir)
    
    manifest_records = []
    
    target_node_size = cfg["render"]["node_size"]
    
    # 3. Produce variants
    for _, row in tqdm(probe_df_source.iterrows(), total=len(probe_df_source), desc="Rendering probe variants"):
        graph_id = row["graph_id"]
        graph_path = row["graph_path"]
        layout_seed = int(row["layout_seed"])
        
        G = nx.read_graphml(graph_path)
        
        # Original image path
        original_image_path = (Path(config.DATA_DIR) / "images" / f"{graph_id}.png").as_posix()
        
        # a. "constant_node_size" variant
        res_constant = render_graph(
            G=G,
            graph_id=graph_id,
            out_dir=str(constant_size_dir),
            seed=layout_seed,
            node_size_fn=lambda n: target_node_size,
            layout_override=None
        )
        manifest_records.append({
            "graph_id": graph_id,
            "generator": row["generator"],
            "tier": row["tier"],
            "num_nodes": row["num_nodes"],
            "num_edges": row["num_edges"],
            "density": row["density"],
            "density_bucket": row["density_bucket"],
            "variant": "constant_node_size",
            "image_path": res_constant["image_path"],
            "layout_algorithm_used": res_constant["layout_algorithm"],
            "original_image_path": original_image_path
        })
        
        # b. "alt_layout" variant
        res_alt = render_graph(
            G=G,
            graph_id=graph_id,
            out_dir=str(alt_layout_dir),
            seed=layout_seed,
            node_size_fn=None,
            layout_override="kamada_kawai"
        )
        manifest_records.append({
            "graph_id": graph_id,
            "generator": row["generator"],
            "tier": row["tier"],
            "num_nodes": row["num_nodes"],
            "num_edges": row["num_edges"],
            "density": row["density"],
            "density_bucket": row["density_bucket"],
            "variant": "alt_layout",
            "image_path": res_alt["image_path"],
            "layout_algorithm_used": res_alt["layout_algorithm"],
            "original_image_path": original_image_path
        })
        
    # 4. Write manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = Path(config.DATA_DIR) / "processed" / "probe_manifest.csv"
    ensure_dir(manifest_path.parent)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Probe manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
