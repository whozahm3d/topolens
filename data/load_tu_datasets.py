# data/load_tu_datasets.py
import os
import sys
import json
import networkx as nx
from tqdm import tqdm
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch_geometric.datasets import TUDataset
# pyrefly: ignore [missing-import]
from torch_geometric.utils import to_networkx

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def get_node_tier(num_nodes: int) -> str:
    """Classifies the graph node count into a configuration tier.
    Anything above 100 nodes is classified as 'xlarge'.
    """
    if num_nodes < 5:
        return "tiny"
    elif 5 <= num_nodes < 10:
        return "tiny"
    elif 10 <= num_nodes < 25:
        return "small"
    elif 25 <= num_nodes < 50:
        return "medium"
    elif 50 <= num_nodes <= 100:
        return "large"
    else:
        return "xlarge"


def process_and_save_dataset(dataset_name: str):
    """Downloads a TUDataset, converts each graph to a simple undirected NetworkX graph,
    strips features, assigns metadata, and saves as GraphML.
    """
    print(f"Loading {dataset_name} dataset...")
    # TUDataset handles downloading and caching automatically
    dataset = TUDataset(root=os.path.join(config.DATA_DIR, "raw"), name=dataset_name)
    
    out_dir = os.path.join(config.RAW_REAL_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    
    pbar = tqdm(total=len(dataset), desc=f"Converting {dataset_name}")
    for idx, data in enumerate(dataset):
        # Convert PyG Data to NetworkX (to_undirected=True yields undirected representation)
        # Note: to_networkx converts PyG Data, returning a MultiDiGraph or DiGraph
        G_pyg = to_networkx(data, to_undirected=True)
        
        # Strip features and make it a simple undirected Graph
        G = nx.Graph()
        G.add_nodes_from(G_pyg.nodes())
        G.add_edges_from(G_pyg.edges())
        
        # Remove any self-loops just in case
        G.remove_edges_from(nx.selfloop_edges(G))
        
        num_nodes = G.number_of_nodes()
        tier = get_node_tier(num_nodes)
        
        graph_id = f"real_{dataset_name}_{idx + 1:04d}"
        
        # Set standardized metadata attributes
        G.graph["graph_id"] = graph_id
        G.graph["source"] = "real"
        G.graph["generator"] = dataset_name
        G.graph["tier"] = tier
        G.graph["gen_params"] = json.dumps({})  # No gen_params for real dataset
        
        # Save as GraphML
        file_path = os.path.join(out_dir, f"{graph_id}.graphml")
        nx.write_graphml(G, file_path)
        
        pbar.update(1)
        
    pbar.close()
    print(f"Successfully processed {len(dataset)} graphs for {dataset_name}")


def main():
    os.makedirs(config.RAW_REAL_DIR, exist_ok=True)
    for dataset_name in config.TU_DATASETS:
        process_and_save_dataset(dataset_name)


if __name__ == "__main__":
    main()
