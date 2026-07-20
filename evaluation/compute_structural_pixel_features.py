"""
Compute per-graph structural properties and pixel statistics — run locally in the topolens repo root.

Computes two things per graph, merged onto failure_case_categories.csv:
  1. ink_fraction   — fraction of non-background pixels in the rendered image
                       (needed for the pixel-only baseline: ink_fraction -> edge count)
  2. diameter, avg_clustering — structural properties from the graphml
                       (needed for the diameter/clustering vs. error correlations)

Output: evaluation/results/structural_pixel_features.csv
Upload that file back to continue the pixel-baseline and structural-correlation analyses.
"""

import pandas as pd
import numpy as np
import networkx as nx
from PIL import Image

INPUT = "evaluation/results/failure_case_categories.csv"
OUTPUT = "evaluation/results/structural_pixel_features.csv"

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} graphs from {INPUT}")

# ---------- 1. Ink fraction from rendered images ----------
def compute_ink_fraction(image_path, bg_threshold=200):
    """Fraction of pixels darker than bg_threshold (i.e. not background)."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    return float((arr < bg_threshold).sum()) / arr.size

ink_fractions = []
for i, p in enumerate(df["image_path"]):
    try:
        ink_fractions.append(compute_ink_fraction(p))
    except Exception as e:
        ink_fractions.append(np.nan)
        print(f"  [ink] failed on row {i} ({p}): {e}")
df["ink_fraction"] = ink_fractions
print(f"Ink fraction computed for {df['ink_fraction'].notna().sum()}/{len(df)} graphs")

# ---------- 2. Structural properties from graphml ----------
diameters = []
clusterings = []
is_connected_flags = []

for i, p in enumerate(df["graph_path"]):
    try:
        G = nx.read_graphml(p).to_undirected()
        connected = nx.is_connected(G)
        is_connected_flags.append(connected)
        if connected:
            diameters.append(nx.diameter(G))
        else:
            # use the largest connected component so a single disconnected
            # graph doesn't just fail outright
            largest_cc = max(nx.connected_components(G), key=len)
            diameters.append(nx.diameter(G.subgraph(largest_cc)))
        clusterings.append(nx.average_clustering(G))
    except Exception as e:
        diameters.append(np.nan)
        clusterings.append(np.nan)
        is_connected_flags.append(np.nan)
        print(f"  [struct] failed on row {i} ({p}): {e}")

df["diameter"] = diameters
df["avg_clustering"] = clusterings
df["is_connected"] = is_connected_flags
print(f"Diameter computed for {df['diameter'].notna().sum()}/{len(df)} graphs")
print(f"  ({(df['is_connected']==False).sum()} disconnected graphs used largest-CC diameter)")

# ---------- Save ----------
df.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
print(df[["graph_id", "ink_fraction", "diameter", "avg_clustering", "is_connected"]].describe(include="all"))