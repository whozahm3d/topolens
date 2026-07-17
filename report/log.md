
# 2026-07-13 — Project Setup & Planning

- Installed Python and Git, initialized GitHub repository (github.com/whozahm3d/topolens)
- Finalized project scope and dataset strategy: primary dataset is synthetic graphs
  via NetworkX generators (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, random
  trees, dense/near-complete graphs); secondary dataset is TU Dortmund benchmark
  (MUTAG/PROTEINS) for real-world validation
- Created full project folder structure (data, render, models, evaluation,
  notebooks, app, report) and pushed to GitHub
- Wrote and committed config.yaml (centralized seed, dataset, rendering, and
  model parameters), README.md, and requirements.txt
- Set up Python environment (Anaconda base env), resolved package
  installation/version issues, and verified all core libraries import correctly:
  PyTorch 2.11.0 (CPU), torch-geometric 2.8.0, NetworkX, Streamlit, Gradio

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

# 2026-07-15 - Phase 2: Graphviz + training/eval scaffolding

- Installed Graphviz into `E:\Anaconda\envs\py-dev` so `sfdp` is available on Windows again.
- Verified `sfdp -V` works from the conda environment and confirmed the render pipeline can reach `graphviz_sfdp` instead of the old spring-layout fallback.
- The full 3,801-image rerender is now unblocked, but it still needs a completed rerun because the Graphviz pass is CPU-bound and slow on this machine.
- Updated `render/render_graphs.py` to emit only `data/processed/labels_processed.csv` and remove the legacy `data/labels.csv` output path so the render stage has one canonical labels artifact.
- Simplified `render/render_graphs.py` to write only the processed labels artifact and retire the legacy `data/labels.csv` output path.
- Added Phase 2 split, dataset, CNN, GNN, baseline, evaluation, and error-analysis scripts under `data/`, `models/`, and `evaluation/`.
- Added a Phase 2 notebook workbook in `notebooks/` so the pipeline can be run and tested interactively like Phase 1.
- Expanded the Phase 2 notebook with inline charts, rendered image previews, model prediction tables, held-out breakdowns, and layout verification output so it doubles as a results dashboard.

# 2026-07-16 — Phase 2: full training on Colab GPU + GCN pooling bug fix

- Ran the full 30-epoch training for both models on Colab (T4 GPU), replacing the
  2-epoch smoke-test checkpoints that all prior evaluation numbers were based on.
  CPU projection had estimated ~5 hours for the CNN alone; the GPU run for both
  models combined took ~6-7 minutes.
- CNN converged cleanly: train_loss 1.32 → 0.12, best val_loss 0.023. Test overall
  MAE dropped from 35.76 (smoke-test, effectively predicting near-zero for every
  graph) to 2.5-3.5 for vertices and 22.8-33 for edges.
- GCN converged during training (val_loss down to ~0.92-1.0) but the first full
  evaluation surfaced a real bug, not just undertraining: held-out MAE exploded
  into the thousands (PROTEINS edges MAE 4,570; sparse-bucket edges MAE 7,452),
  while the CNN and the GCN's own in-distribution test-split numbers stayed sane.
  Root cause: node features were raw, unnormalized degree values, pooled into a
  graph embedding via `global_add_pool` (sum). Held-out real graphs (PROTEINS, up
  to several hundred nodes) sit well outside the synthetic training distribution's
  size range, so the summed embedding pushed the regression head into
  extrapolation, and since predictions are decoded via expm1 (log1p-trained
  targets), even a moderate log-space error exploded multiplicatively once
  converted back to real counts.
- Fix: switched `global_add_pool` to `global_mean_pool` in `GraphCountGCN.forward`,
  and normalized the per-node degree feature by each graph's max degree in
  `graphml_to_pyg_data` (`models/gnn_baseline.py`). Both `evaluate.py` and
  `error_analysis.py` import these directly, so no other files needed changes.
- First attempt at applying this fix was lost when the Colab runtime was
  discarded before the commit went through; the notebook was rerun from a fresh
  clone (images, splits, and processed labels came in correctly from the
  already-committed repo — re-rendering was started once as a precaution but
  cancelled after ~10 minutes once confirmed unnecessary, since Graphviz output
  is deterministic and already matched what was in git). The fix was reapplied
  before this training run, and confirmed in the file before committing.
- Retrained the GCN only (CNN checkpoint unaffected). Held-out sparse-bucket
  edges MAE dropped from 7,452 to 61.9; PROTEINS edges MAE dropped from 4,570 to
  32.5, vertices MAE from 3,134 to 9.96.
- Known tradeoff: mean-pooling removes the implicit "sum acts like a node-count
  signal" that sum-pooling provided, so the GCN is now stable but consistently
  less accurate than the CNN across every breakdown (e.g. dense-bucket edges MAE:
  CNN ~1.6 vs GCN ~76). Acceptable for a baseline comparison model. Possible
  follow-up: concatenate `num_nodes` as an explicit scalar feature before the
  head, giving the model a direct size signal instead of inferring it from
  pooled degree statistics.
- `cnn_worst_edge_errors.csv` no longer shows the CNN predicting 0 for every
  large graph (previous symptom of the smoke-test-only checkpoint) — predictions
  are now in the correct order of magnitude even on the largest test graphs.
- Committed and pushed from Colab directly (GitHub PAT stored via Colab
  Secrets, not hardcoded in the notebook). Commit 288a4c5.

# 2026-07-16 — Phase 2: graph-statistic baseline fix + retrain

- Found that `graph_statistic_baseline.py` and `evaluate.py`'s
  `predict_graph_statistic()` were reading `num_nodes`/`num_edges` directly
  from ground truth instead of estimating them — a data-leakage bug, not a
  real baseline. `graph_statistic_metrics.csv` showing exactly 0.0 MAE/RMSE
  across every split and breakdown is what surfaced it.
- Replaced with a non-leaking heuristic: node count is still returned
  exactly (structurally trivial to observe, not a meaningful prediction
  target), but edge count is now estimated as
  `mean_density(generator) * n*(n-1)/2`, where `mean_density` is computed
  once from the train split only, per generator, with a global fallback for
  MUTAG/PROTEINS (never present in train).
- Edited `models/graph_statistic_baseline.py` (full rewrite) and three
  spots in `evaluation/evaluate.py` (new import, replaced
  `predict_graph_statistic`, added a `build_density_table` call in
  `main()`).
- Result: no longer an oracle, but the heuristic performs poorly on sparse
  and large graphs. Density isn't constant across generators — a random
  tree's true density is `2/n` and shrinks as `n` grows, so applying an
  averaged density to a quadratic term systematically overestimates edges
  outside the calibration range. Held-out xlarge-tier edges MAE: 4,927.7 vs
  CNN 203.6 and GCN 265.6. Kept as-is rather than engineered further — it's
  a legitimate, reportable finding (naive non-learned heuristics don't
  generalize across size/sparsity regimes) rather than a bug to fix away.
- Full notebook rerun on Colab: `RUN_FULL_TRAINING=True` retrained both CNN
  and GCN from scratch (not strictly required for the baseline fix, done
  anyway). Numbers shifted slightly from the previous run due to
  GPU/cuDNN non-determinism (e.g. test overall vertices MAE 4.5 → 3.2;
  held-out xlarge vertices MAE 85.3 → 96.6) — same regime throughout, no
  regressions.
- Pushed in two commits: `ecdd8cd` (results CSVs + regenerated figures from
  the retrain) and `e6cca5d` (notebook re-saved with the new inline
  outputs). Checkpoints (`cnn_best.pt`, `gnn_best.pt`) pulled manually from
  Colab into `models/checkpoints/` afterward (gitignored, not pushed via
  git).

# 2026-07-17 — Phase 3 Batch 1: render refactor, Grad-CAM, shortcut/layout probes

- Refactored `render_graphs.py` to support `node_size_fn` and `layout_override` (with kamada_kawai fallback).
- Regression check: 3/3 re-rendered graphs byte-identical to `data/images/` under default args (graphviz_sfdp layout confirmed). Root cause of initial failure: `.venv` pydot could not launch `sfdp.BAT` via subprocess on Windows; fixed by prepending `E:\Anaconda\envs\py-dev\Library\bin` (containing `sfdp.exe`) to PATH in `render_graphs.py`.
- Implemented `GradCAM` for the custom CNN in `models/gradcam.py` (target layer `features[3]`). Bug fixed: `cm.get_cmap` removed in newer Matplotlib; replaced with `matplotlib.colormaps['jet']`.
- Ran `evaluation/interpretation.py`: generated 60 CAM overlays (30 graphs × vertex + edge) in `report/figures/gradcam/`, plus composite grids `gradcam_grid_by_generator.png` (1.07 MB) and `gradcam_grid_failure_cases.png` (4.1 MB).
- Ran `render/render_probe_variants.py`: sampled 40 graphs from test split (20 cells × 2), rendered 40 constant-size variants to `data/images_probe/constant_node_size/` and 40 alt-layout (Kamada-Kawai) variants to `data/images_probe/alt_layout/`. Manifest written to `data/processed/probe_manifest.csv` (80 rows).
- Ran `evaluation/shortcut_probe.py`: 120 predictions (40 graphs × 3 variants). Results in `evaluation/results/probe_predictions.csv` and `evaluation/results/probe_summary.csv`.
- **Headline numbers (overall MAE, probe_summary.csv):**
  | variant | vertex_mae | edge_mae |
  |---|---|---|
  | original | 3.43 | 54.18 |
  | constant_node_size | 10.48 | 39.80 |
  | alt_layout | 3.08 | 35.53 |
  - Vertex MAE nearly triples under constant node size (3.43 → 10.48): strong evidence the CNN uses node size as a shortcut for vertex counting.
  - Edge MAE improves under both probe conditions — consistent with the model having learned node-size/layout cues that partially hurt edge estimation on the original renders.
  - Layout change (alt_layout) has minimal vertex MAE impact (3.43 → 3.08), suggesting the model is not strongly overfit to sfdp layout topology.
- Verification: `data/images/` (3802 files), `data/splits/` (5 files), and `data/processed/labels_processed.csv` (844,253 bytes) all show pre-run mtimes — none modified.
