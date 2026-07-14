
# 2026-07-14 — Phase 1: dataset generation + rendering

- Generated 2500 synthetic graphs (5 generators × 4 tiers × 125 each)
- Loaded MUTAG (188 graphs) and PROTEINS (1113 graphs) via PyTorch Geometric
- Rendered 3801/3801 images successfully. Layout: 3801 networkx_spring, 0 graphviz_sfdp
  (Graphviz binaries not found on PATH on this machine — pydot fallback triggered for
  every image, not just a subset. Dataset currently uses a single deterministic layout
  algorithm rather than a mix. Flag for Limitations: no layout-diversity coverage yet.)
- validate_dataset.py: PASS — all images exist, correct resolution, match GraphML structure
- Structural stats: num_nodes range 4–620, num_edges range 0–3291
  - Synthetic tiers capped at 100 nodes as designed
  - PROTEINS contributes 67 "xlarge" graphs (>100 nodes, up to 620) — well outside the
    synthetic training distribution. Useful generalization test case, but also a candidate
    for exclusion/separate handling in Phase 2 if it skews the loss.
  - random_tree confirmed exactly n-1 edges across all 500 tree graphs, as designed
  - erdos_renyi min edges = 0 (a few small/sparse draws landed on a totally disconnected
    graph) — expected given the sampling range, not a bug
- data/gephi_demo/: 24 curated GraphML files exported for manual layout.
