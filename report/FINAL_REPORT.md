# Topolens: Estimating Graph Structural Properties from Rendered Images

This report documents the design, training, and evaluation of **Topolens**, a
CNN-based system that estimates vertex and edge counts directly from rendered
2D graph images without access to raw graph topology. A four-block
`CustomCNNRegressor` is trained on 2,500 synthetic graphs (five NetworkX
generators, four size tiers, Graphviz `sfdp` layout, 224 × 224 px PNG) and
evaluated against a GCN baseline (`GraphCountGCN`) and a density-heuristic
statistical baseline. Quantitative evaluation covers per-split MAE/RMSE
breakdowns; three interpretability analyses (Grad-CAM attention maps,
node-size shortcut probe, Kamada-Kawai layout-sensitivity probe) and an
ink-coverage correlational study characterize how the model achieves its
predictions and where it fails. The Streamlit inference app (`app/app.py`)
exposes both image and graph-file upload, live Grad-CAM overlays, and an
empirical validation panel that compares CNN predictions against ground-truth
counts from the training dataset.

---

## 1. Problem Framing

_Populate from `progress_log.md` Day 1 "Why" section — the CNN-vs-GNN research
question: whether a convolutional model operating on rendered graph images can
estimate structural properties (vertex count, edge count) competitively with a
GNN operating on raw topology, and what visual shortcuts or failure modes that
image-based approach produces._

---

## 2. Methodology

### 2.1 Dataset

_Populate from `config.yaml` (`dataset:` block), `progress_log.md` Phase 1
entries, and `data/processed/labels_processed.csv` schema. Cover synthetic
generators (5 types × 4 tiers × 125 graphs = 2,500) and real-world held-out
sets (MUTAG/PROTEINS via TUDataset). Note the 67 PROTEINS graphs exceeding
the 100-node training ceiling._

### 2.2 Rendering Pipeline

_Populate from `render/render_graphs.py`, `config.yaml` (`render:` block), and
Phase 1/2 `report/log.md` entries. Cover Graphviz `sfdp` layout, image
resolution (224 × 224 px), the node-size scaling formula
`max(15, 4000/n)`, and the probe variants generated in Batch 1._

### 2.3 Model Architecture

_Populate from `models/cnn_model.py` (`CustomCNNRegressor`), `models/gnn_baseline.py`
(`GraphCountGCN`), and `models/graph_statistic_baseline.py`. Cover the four-block
CNN, log1p target encoding, and the mean-pool GCN fix documented in `report/log.md`
(2026-07-16)._

### 2.4 Training Setup

_Populate from `config.yaml` (`model:` block), `models/train_cnn.py`, and
`report/log.md` training entries (2026-07-16). Cover 30 epochs, Adam optimizer,
MSE loss, GPU run on Colab T4, checkpoint path `models/checkpoints/cnn_best.pt`._

---

## 3. Baseline Comparison

_Populate from `evaluation/results/summary_comparison.csv`. Present CNN vs. GNN
vs. graph-statistic heuristic across test and held-out splits._

> **Note:** The graph-statistic baseline's 0.0 vertex MAE is structurally trivial —
> it returns the true node count directly rather than estimating it. Flag explicitly
> rather than presenting as a meaningful baseline result. The edge heuristic
> (`mean_density × n(n−1)/2`) is the only non-leaking component.

---

## 4. Quantitative Results

_Populate from `evaluation/results/cnn_metrics.csv`, `summary_comparison.csv`,
and the Phase 2 scatter figures (`report/figures/cnn_pred_num_vertices_scatter_*.png`,
`report/figures/cnn_pred_num_edges_scatter_*.png`). Cover per-split MAE/RMSE
breakdowns by generator type and density bucket._

---

## 5. Model Interpretation & Failure Analysis

### 5.1 Grad-CAM: What Does the Model Look At?

_Populate from `report/figures/gradcam_grid_by_generator.png` and
`report/figures/gradcam_grid_failure_cases.png`. Describe spatial attention
patterns across generator types and failure cases. Individual CAM overlays
are in `report/figures/gradcam/`._

### 5.2 Shortcut-Learning Probe (Node-Size Confound)

_Populate from `evaluation/results/probe_summary.csv` — compare `original` vs.
`constant_node_size` variant rows. The overall vertex MAE change under
constant node size is the primary quantitative evidence for or against
node-size shortcut exploitation. See also `report/figures/probe_variant_mae_comparison.png`
and `report/figures/probe_variant_examples_grid.png`._

### 5.3 Layout-Sensitivity Probe

_Populate from `evaluation/results/probe_summary.csv` — compare `original` vs.
`alt_layout` (Kamada-Kawai) variant rows, broken down by tier. Assess whether
the model is overfit to `sfdp` layout topology._

### 5.4 Ink-Coverage Correlational Analysis

_Populate from `evaluation/results/ink_coverage_correlations.csv`. Report
Pearson r values for ink fraction vs. node/edge counts, and mean component
area vs. predicted vertex count, across test and held-out splits. See also
`report/figures/ink_coverage_vs_node_count.png` and
`report/figures/component_size_vs_node_count.png`._

### 5.5 Failure-Case Taxonomy

_Populate from `evaluation/results/failure_case_summary.csv` and
`evaluation/results/failure_case_categories.csv`. The contrast between
`out_of_distribution_size` (n > 100 training ceiling) and `in_distribution_normal`
on the held-out split is the headline finding for this section. See also
`report/figures/error_vs_num_nodes.png`, `report/figures/error_vs_density.png`,
and `report/figures/worst_case_image_grid.png`._

---

## 6. Discussion

_Written by hand — synthesizes Sections 5.1–5.5 into a direct answer to the
Section 1 research question. Address whether the CNN's image-based approach
is competitive with the GNN baseline, under what conditions, and what the
interpretation results reveal about how the model achieves its predictions._

---

## 7. Limitations

_Written by hand. Must cover:_

- _Layout-algorithm sensitivity (sfdp vs. Kamada-Kawai; probe evidence)_
- _Dataset size (2,500 synthetic graphs; no held-out synthetic graphs above 100 nodes)_
- _Synthetic-to-real distribution gap (PROTEINS/MUTAG generalization)_
- _Image resolution and node-size tradeoffs (224 × 224 px, node radius formula)_
- _The empirically confirmed node-size shortcut (Section 5.2 probe finding)_
- _The out-of-distribution size ceiling (n > 100; Section 5.5 MAE values)_

---

## 8. Future Work

_Placeholder only. Potential directions include: multi-resolution inputs,
layout-augmented training, GNN augmented with image features, explicit
node-count input conditioning, and extension to directed or weighted graphs._

---

## Appendix: Figure & Table Index

### Figures (`report/figures/`)

| File | Description |
|---|---|
| `gradcam_grid_by_generator.png` | Grad-CAM overlay grid, one column per generator type |
| `gradcam_grid_failure_cases.png` | Grad-CAM overlays for worst-error cases |
| `probe_variant_mae_comparison.png` | Bar chart: MAE by probe variant |
| `probe_variant_examples_grid.png` | Side-by-side renders: original / constant-size / alt-layout |
| `ink_coverage_vs_node_count.png` | Scatter: ink fraction vs. node count |
| `component_size_vs_node_count.png` | Scatter: mean component area vs. node count |
| `error_vs_num_nodes.png` | Scatter: prediction error vs. graph size |
| `error_vs_density.png` | Scatter: prediction error vs. graph density |
| `worst_case_image_grid.png` | Grid of highest-MAE test images |
| `cnn_pred_num_vertices_scatter_held_out.png` | Predicted vs. true vertex count (held-out) |
| `cnn_pred_num_vertices_scatter_test.png` | Predicted vs. true vertex count (test) |
| `cnn_pred_num_edges_scatter_held_out.png` | Predicted vs. true edge count (held-out) |
| `cnn_pred_num_edges_scatter_test.png` | Predicted vs. true edge count (test) |
| `node_edge_distributions_by_generator.png` | Distribution plots by generator |
| `node_edge_distributions_overall.png` | Overall node/edge count distributions |
| `graph_counts_by_tier.png` | Graph counts per tier |
| `sample_render_grid.png` | Sample rendered images grid |

### Tables (`evaluation/results/`)

| File | Description |
|---|---|
| `cnn_metrics.csv` | CNN MAE/RMSE by split and breakdown |
| `gnn_metrics.csv` | GNN MAE/RMSE by split and breakdown |
| `graph_statistic_metrics.csv` | Graph-statistic baseline metrics |
| `summary_comparison.csv` | Side-by-side model comparison |
| `probe_summary.csv` | Probe MAE by variant and tier |
| `probe_predictions.csv` | Per-graph probe predictions (raw) |
| `ink_coverage_correlations.csv` | Pearson r for pixel statistics vs. counts |
| `failure_case_summary.csv` | MAE by failure category and split |
| `failure_case_categories.csv` | Per-graph failure category assignments |
| `baseline_predictions.csv` | Baseline per-graph predictions |
| `cnn_worst_edge_errors.csv` | Worst-error graphs for CNN edge prediction |
| `cnn_training_log.csv` | Epoch-level CNN training log |
| `gnn_training_log.csv` | Epoch-level GNN training log |

### Application (`app/`)

| File | Description |
|---|---|
| `app.py` | Streamlit inference app — **Neo-Brutalist design system** (Syne display + Plus Jakarta Sans body); single multi-type file uploader (`png`, `jpg`, `jpeg`, `graphml`, `csv`, `txt`); multi-file select; CNN inference + hero stat display; Grad-CAM toggle (vertex-target + edge-target overlays); empirical validation panel (Ground Truth vs Prediction table + topological-property cards) active for both graph files and dataset images matched via `labels_processed.csv`; model limitations expander with live CSV values. |
| `.streamlit/config.toml` | Theme: `backgroundColor = "#F5F5F5"`, `primaryColor = "#000000"`, `secondaryBackgroundColor = "#E5E5E5"`, `textColor = "#000000"`. |
