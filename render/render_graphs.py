# render/render_graphs.py
import os
import sys
import json
import hashlib
import pandas as pd
import networkx as nx
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from tqdm import tqdm

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from topolens_utils import compute_density, density_bucket, ensure_dir

# Add Anaconda Graphviz Library path on Windows to avoid .bat wrapping issues in pydot
if sys.platform == "win32":
    graphviz_bin = r"E:\Anaconda\envs\py-dev\Library\bin"
    if os.path.exists(graphviz_bin) and graphviz_bin not in os.environ["PATH"]:
        os.environ["PATH"] = graphviz_bin + os.pathsep + os.environ["PATH"]


def seed_from_id(graph_id: str) -> int:
    """Derives a deterministic 32-bit integer seed from a string ID."""
    h = hashlib.sha256(graph_id.encode())
    return int(h.hexdigest(), 16) % (2**32)


def render_graph(
    G: nx.Graph, graph_id: str, out_dir: str, image_size=config.IMAGE_SIZE, dpi=config.IMAGE_DPI, seed=None,
    node_size_fn=None, layout_override=None
) -> dict:
    """Computes a layout and renders G to PNG.
    Style requirements:
      - White background
      - No axes/labels
      - Uniform node color (modern slate/indigo style)
      - No arrows
      - node_size = max(15, 4000 / n)
      - Thin fixed-width edges (width=0.6)
      - figsize=(image_size/100, image_size/100), dpi=100
    
    Returns:
      dict: {graph_id, layout_algorithm, layout_seed, image_path}
    """
    n = G.number_of_nodes()
    
    # Deterministic layout seed
    layout_seed = seed if seed is not None else seed_from_id(graph_id)
    
    # 1. Compute Layout with Fallbacks
    pos = None
    layout_algorithm = None
    
    if layout_override == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(G)
            layout_algorithm = "kamada_kawai"
        except Exception:
            pos = nx.spring_layout(G, seed=layout_seed + 1)
            layout_algorithm = "spring_fallback_from_kamada_kawai"
    else:
        # Try pygraphviz sfdp
        try:
            # pyrefly: ignore [missing-import]
            import pygraphviz
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='sfdp')
            layout_algorithm = "graphviz_sfdp"
        except Exception:
            # Try pydot sfdp
            try:
                # pyrefly: ignore [missing-import]
                import pydot
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='sfdp')
                layout_algorithm = "graphviz_sfdp"
            except Exception:
                # Fall back to NetworkX spring layout
                pos = nx.spring_layout(G, seed=layout_seed)
                layout_algorithm = "networkx_spring"
            
    # 2. Render and Draw
    fig, ax = plt.subplots(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    
    # Ensure white background and turn off axes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    # Adjust subplots to occupy full figure area
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if n > 0 and pos:
        # Adjust margins to prevent node clipping
        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
        y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Scaling node size: max(15, 4000 / n)
        # Note: This makes node density in the image correlate with n, which the CNN
        # could exploit as a shortcut instead of reading topology. Flagged for Phase 3 analysis.
        if node_size_fn is not None:
            node_size = node_size_fn(n)
        else:
            node_size = max(15, 4000 / n)
        
        # Modern, clean aesthetics: deep Indigo nodes (#4F46E5) and subtle Slate edges (#94A3B8)
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            node_size=node_size,
            node_color='#4F46E5',
            edgecolors='none'
        )
        
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            width=0.6,
            edge_color='#94A3B8',
            arrows=False
        )
        
    image_name = f"{graph_id}.png"
    image_path = os.path.join(out_dir, image_name)
    
    # Save image with exact pixel resolution
    plt.savefig(image_path, format='png', facecolor='white', dpi=dpi)
    plt.close(fig)
    
    # Return metadata
    return {
        "graph_id": graph_id,
        "layout_algorithm": layout_algorithm,
        "layout_seed": layout_seed,
        "image_path": image_path.replace("\\", "/")
    }


def find_graph_files():
    """Finds all GraphML files in raw synthetic and real-world directories."""
    graph_files = []
    
    # Synthetic
    if os.path.exists(config.RAW_SYNTHETIC_DIR):
        for f in os.listdir(config.RAW_SYNTHETIC_DIR):
            if f.endswith(".graphml"):
                graph_files.append(os.path.join(config.RAW_SYNTHETIC_DIR, f))
                
    # Real
    if os.path.exists(config.RAW_REAL_DIR):
        for root, _, files in os.walk(config.RAW_REAL_DIR):
            for f in files:
                if f.endswith(".graphml"):
                    graph_files.append(os.path.join(root, f))
                    
    return sorted(graph_files)


def curate_and_export_gephi(metadata_list):
    """Curates ~20-25 diverse graphs (covering different generators, tiers, and sources)
    and saves them to data/gephi_demo/.
    """
    print("Curating graphs for Gephi demo...")
    os.makedirs(config.GEPHI_DEMO_DIR, exist_ok=True)
    
    df = pd.DataFrame(metadata_list)
    curated_ids = []
    
    # 1. From synthetic cells (5 generators x 4 tiers) = 20 cells
    # We will pick 1 graph from each cell
    for generator in config.GENERATORS:
        for tier in config.NODE_TIERS.keys():
            cell_df = df[(df["generator"] == generator) & (df["tier"] == tier)]
            if not cell_df.empty:
                # Select the first one in the list
                curated_ids.append(cell_df.iloc[0]["graph_id"])
                
    # 2. Add some real-world graphs (e.g. 2 from MUTAG and 2 from PROTEINS)
    for dataset_name in config.TU_DATASETS:
        real_df = df[(df["source"] == "real") & (df["generator"] == dataset_name)]
        if not real_df.empty:
            # Pick a small and a large one if possible
            real_df_sorted = real_df.sort_values(by="num_nodes")
            curated_ids.append(real_df_sorted.iloc[0]["graph_id"])
            if len(real_df_sorted) > 1:
                curated_ids.append(real_df_sorted.iloc[-1]["graph_id"])
                
    # Copy selected files to gephi_demo directory
    copied_count = 0
    for graph_id in curated_ids:
        row = df[df["graph_id"] == graph_id].iloc[0]
        src_path = row["graph_path"]
        dst_path = os.path.join(config.GEPHI_DEMO_DIR, f"{graph_id}.graphml")
        
        try:
            # Load and save to make sure it's a plain, clean GraphML file
            G = nx.read_graphml(src_path)
            nx.write_graphml(G, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {graph_id} to Gephi demo: {e}")
            
    print(f"Exported {copied_count} curated graphs to {config.GEPHI_DEMO_DIR}")


def main():
    print("Starting rendering pipeline...")
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    processed_dir = os.path.dirname(config.PROCESSED_LABELS_CSV)
    if processed_dir:
        ensure_dir(processed_dir)
    
    graph_files = find_graph_files()
    if not graph_files:
        print("No GraphML files found. Please run generation and loading scripts first.")
        sys.exit(1)
        
    print(f"Found {len(graph_files)} GraphML files to render.")
    
    metadata_list = []
    pbar = tqdm(total=len(graph_files), desc="Rendering graphs")
    
    for graph_file in graph_files:
        try:
            G = nx.read_graphml(graph_file)
            
            # Extract metadata from G.graph or fallback to filesystem attributes
            graph_id = G.graph.get("graph_id", os.path.splitext(os.path.basename(graph_file))[0])
            source = G.graph.get("source", "synthetic" if "synthetic" in graph_file else "real")
            generator = G.graph.get("generator", "unknown")
            tier = G.graph.get("tier", "unknown")
            gen_params = G.graph.get("gen_params", "{}")
            
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            # Generate rendering seed
            seed = seed_from_id(graph_id)
            
            # Render graph
            render_result = render_graph(G, graph_id, config.IMAGES_DIR, seed=seed)
            
            # Log render choice (stdout warning level, can see fallback behavior)
            if render_result["layout_algorithm"] == "networkx_spring":
                # Only log spring layouts since they represent a fallback
                pass
                
            # Store full record matching the schema
            metadata_list.append({
                "graph_id": graph_id,
                "source": source,
                "generator": generator,
                "tier": tier,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "gen_params": gen_params,
                "layout_algorithm": render_result["layout_algorithm"],
                "layout_seed": render_result["layout_seed"],
                "graph_path": graph_file.replace("\\", "/"),
                "image_path": render_result["image_path"]
            })
            
        except Exception as e:
            print(f"Failed to render {graph_file}: {e}")
            
        pbar.update(1)
        
    pbar.close()
    
    df = pd.DataFrame(metadata_list)
    # Save only the processed labels artifact for downstream training/evaluation.
    processed_df = df.copy()
    processed_df["density"] = processed_df.apply(
        lambda row: compute_density(int(row["num_nodes"]), int(row["num_edges"])),
        axis=1,
    )
    processed_df["density_bucket"] = processed_df["density"].map(density_bucket)
    processed_df.to_csv(config.PROCESSED_LABELS_CSV, index=False)
    print(f"Successfully saved processed labels to {config.PROCESSED_LABELS_CSV}")

    # Remove the legacy labels.csv if it exists so there is a single canonical labels file.
    if os.path.exists(config.LABELS_CSV):
        os.remove(config.LABELS_CSV)
    
    # Print layout statistics
    layout_stats = df["layout_algorithm"].value_counts()
    print("Layout algorithm distribution:")
    for algo, count in layout_stats.items():
        print(f"  - {algo}: {count}")
        
    # Curate and export Gephi sample graphs
    curate_and_export_gephi(metadata_list)


if __name__ == "__main__":
    main()
