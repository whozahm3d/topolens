# data/generate_synthetic.py
import os
import sys
import random
import hashlib
import json
import networkx as nx
from tqdm import tqdm

# Add project root to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def seed_from_id(graph_id: str) -> int:
    """Derives a deterministic 32-bit integer seed from a string ID using SHA-256."""
    h = hashlib.sha256(graph_id.encode())
    return int(h.hexdigest(), 16) % (2**32)


def generate_single_graph(generator: str, tier: str, index: int) -> nx.Graph:
    """Generates a single synthetic graph based on generator family, node tier, and index.
    Uses a deterministic seed derived from the unique graph_id."""
    graph_id = f"syn_{generator}_{tier}_{index:04d}"
    graph_seed = seed_from_id(graph_id)
    
    # Initialize a local random number generator for parameter sampling
    rng = random.Random(graph_seed)
    
    # Get node range for the tier
    tier_min, tier_max = config.NODE_TIERS[tier]
    
    # Sample n uniformly from the tier range
    n = rng.randint(tier_min, tier_max)
    
    params = {"n": n}
    
    # Generate graph based on family
    if generator == "erdos_renyi":
        p = rng.uniform(0.05, 0.3)
        params["p"] = p
        G = nx.erdos_renyi_graph(n, p, seed=graph_seed)
        
    elif generator == "barabasi_albert":
        # m must be in [1, n-1]
        m = rng.randint(1, max(1, min(5, n - 1)))
        params["m"] = m
        G = nx.barabasi_albert_graph(n, m, seed=graph_seed)
        
    elif generator == "watts_strogatz":
        p = rng.uniform(0.05, 0.3)
        # k must be an even integer in [2, n-1]
        max_k = min(6, n - 1)
        even_ks = [i for i in range(2, max_k + 1) if i % 2 == 0]
        k = rng.choice(even_ks) if even_ks else 2
        params["k"] = k
        params["p"] = p
        G = nx.watts_strogatz_graph(n, k, p, seed=graph_seed)
        
    elif generator == "random_tree":
        G = nx.random_labeled_tree(n, seed=graph_seed)
        
    elif generator == "dense":
        p = rng.uniform(0.4, 0.7)
        params["p"] = p
        G = nx.erdos_renyi_graph(n, p, seed=graph_seed)
        
    else:
        raise ValueError(f"Unknown generator: {generator}")
        
    # Enforce simple undirected graphs: no self-loops, no multi-edges
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Attach metadata to graph object
    G.graph["graph_id"] = graph_id
    G.graph["source"] = "synthetic"
    G.graph["generator"] = generator
    G.graph["tier"] = tier
    G.graph["gen_params"] = json.dumps(params)
    
    return G


def main():
    print("Starting synthetic graph generation...")
    os.makedirs(config.RAW_SYNTHETIC_DIR, exist_ok=True)
    
    total_graphs = len(config.GENERATORS) * len(config.NODE_TIERS) * config.GRAPHS_PER_CELL
    pbar = tqdm(total=total_graphs, desc="Generating synthetic graphs")
    
    for generator in config.GENERATORS:
        for tier in config.NODE_TIERS.keys():
            for i in range(1, config.GRAPHS_PER_CELL + 1):
                G = generate_single_graph(generator, tier, i)
                graph_id = G.graph["graph_id"]
                file_path = os.path.join(config.RAW_SYNTHETIC_DIR, f"{graph_id}.graphml")
                
                # Save as GraphML
                nx.write_graphml(G, file_path)
                pbar.update(1)
                
    pbar.close()
    print(f"Successfully generated {total_graphs} synthetic graphs in {config.RAW_SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()
