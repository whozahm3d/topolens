"""
Generates a synthetic graph that deliberately exceeds both reduced-confidence
warning thresholds in app/app.py (render_prediction_warning_banner):
  1. num_nodes > max_train_n (100, from data/splits/train.csv)
  2. density_bucket(density) == "dense" (density >= 0.40, from topolens_utils.py)

Used to verify the dual-trigger warning path live, since no natural example
in the current dataset combines both conditions simultaneously.

Output: data/images_probe/dense_test_150n.graphml
"""

import networkx as nx
from pathlib import Path

OUTPUT_DIR = Path("data/images_probe")
OUTPUT_FILE = OUTPUT_DIR / "dense_test_150n.graphml"

MAX_TRAIN_N = 100      # must exceed this (data/splits/train.csv ceiling)
DENSE_THRESHOLD = 0.40  # must meet/exceed this (topolens_utils.density_bucket)


def main() -> None:
    G = nx.erdos_renyi_graph(n=150, p=0.5, seed=42)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)  # 2m / (n*(n-1))

    print(f"nodes={n}, edges={m}, density={density:.4f}")

    assert n > MAX_TRAIN_N, f"node count {n} must exceed training ceiling {MAX_TRAIN_N}"
    assert density >= DENSE_THRESHOLD, f"density {density:.4f} must fall in dense bucket (>= {DENSE_THRESHOLD})"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()