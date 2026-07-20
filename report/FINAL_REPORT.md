# Topolens: Estimating Graph Structural Properties from Rendered Images

_One-paragraph abstract — placeholder only, written by hand once Sections
1–7 are complete._

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

> **Note:** When discussing this section, you can add the following sentence
to your report to bring in the statistical significance results from the
additional notebook:

**Statistical significance (Wilcoxon signed-rank, n=40 paired graphs).**
Populated from `evaluation/results/topolens_part4_free_analyses_results.csv`.
The vertex-error increase under `constant_node_size` is significant
(median 1.0 → 11.0, p=3.6×10⁻⁶, 33/40 graphs worse), confirming the shortcut
with a formal test rather than a summary MAE delta alone. The edge-error
comparison requires care: mean edge error appears to _improve_
(54.18 → 39.80), but this is an artifact of two extreme-outlier dense-large
graphs; the median (3.5 → 20.0, 35/40 graphs worse) and the Wilcoxon test
itself (p=0.0003, significant in the worsening direction) both confirm the
shortcut also degrades edge prediction for the large majority of graphs.
Report median and win/loss counts alongside — or instead of — the mean for
this comparison.

### 5.3 Layout-Sensitivity Probe

_Populate from `evaluation/results/probe_summary.csv` — compare `original` vs.
`alt_layout` (Kamada-Kawai) variant rows, broken down by tier. Assess whether
the model is overfit to `sfdp` layout topology._

**Statistical significance (Wilcoxon signed-rank, n=40 paired graphs).** No
significant layout effect was found on either vertex (p=0.499) or edge
(p=0.898) error under the `alt_layout` (Kamada-Kawai) variant. Median deltas
are 0 (vertex) and ~1.0 (edge), with near-even win/loss splits (17/16 and
18/16). The −18.65 mean edge-error delta that a naive summary would report
is driven almost entirely by a single outlier graph. **This is a clean null
result: the model is not meaningfully overfit to the `sfdp` layout
algorithm**, which should be stated plainly in Section 7 (Limitations) rather
than assumed as a risk.

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

### 5.6 Live-App Validation: Spot Check & Novel-Image Sanity Check

_Populated from `evaluation/results/topolens_spotcheck_results.csv` (Part 1) and
`evaluation/results/topolens_novelimage_results.csv` (Part 2). Unlike Sections
5.1–5.5, which analyze the offline held-out/test split, this section validates
the deployed app's live inference path — including the warning banner and
multi-model comparison — against inputs chosen and uploaded after training._

**Part 1 — 23-graph live-app spot check.** Deliberately selected `.graphml`
graphs spanning every generator, density bucket, and the 100-node training
ceiling, uploaded through the Predict page to exercise CNN + GCN +
graph-statistic baseline together. Clean in-distribution graphs (11 of 23, no
warning fired) produced CNN vertex MAE ≈5.6 and edge MAE ≈17.7, consistent
with the offline held-out figures in Section 5.5. Two findings refine the
offline analysis:

- **Density and size interact.** Dense-flagged graphs that are also large
  (one 99-node dense graph, edge error 1,084) failed far worse than
  dense-but-small/medium graphs (edge errors 1–32), despite the warning
  banner citing the same dense-bucket MAE (1.12) for both. The dense-bucket
  warning's historical MAE appears to understate risk specifically for
  graphs that are dense _and_ large.
- **GCN degraded more than the CNN on out-of-distribution size**, contrary
  to the expectation that a graph-native model generalizes better on size
  than an image-based one (mean vertex error 237 vs. CNN 179; mean edge
  error 428 vs. CNN 336; n=6 OOD-size graphs). This bears directly on the
  Section 1 research question and is addressed in Section 6.

**Part 2 — 7-image novel-image sanity check (no ground truth).** Tested the
app against images never seen during training: a scanned academic figure, a
digitally hand-drawn directed graph, a cartoon-icon diagram, two clean-vector
diagrams in unfamiliar styles, one image close in style to the training
renders, and one full uncropped notebook photo (tested standalone,
containing two diagrams plus prose). Predicted vertex/edge counts scaled
with _visual unfamiliarity relative to the training render style_, not with
hand-drawn-ness per se: the image closest in style to training renders had
the smallest relative overcount (~1.6x); the two furthest (rectangle/text
nodes, dense numeric-edge-label network) had the largest, up to a predicted
474 vertices on one case (~4.7x the training ceiling).

**Part 3 — revised combinatorial-plausibility claim.** Section 5.5 previously stated
(via the Part 3 free-item check) that zero predictions across the 1,676
held-out/test set ever have predicted edges exceed the maximum possible
edges for the predicted vertex count. This holds for all 1,676 offline
predictions and 29 of the 30 live-app predictions collected in Parts 1–2. The
one exception is the uncropped notebook photo above: predicted V=5, E=19,
where the combinatorial maximum for 5 vertices is 10. This is best explained
as an input-distribution failure — the only image tested that combined
multiple diagrams and prose text in one frame — rather than a breakdown of
the property under normal single-graph input. The claim should be stated as
qualified, not absolute, going forward.

**Part 4 (free items) — warning-banner validation.** Using all 1,676
held-out+test predictions in `failure_case_categories.csv`, binned by
`num_nodes` to hold size roughly constant, the dense-bucket warning's
validity depends entirely on which target it's warning about:

- For **vertex count**, the warning is backwards — dense graphs have lower
  error than sparse graphs at every size bin tested (e.g. 75–100 nodes:
  dense median error 5.0 vs. sparse 21.0). Dense is the safest bucket for
  vertex prediction, not a risk signal.
- For **edge count**, the warning is directionally correct but
  size-miscalibrated. Dense and sparse edge error are statistically tied at
  small sizes, then diverge sharply with size, reaching a 3.3x gap by
  75–100 nodes (dense median 224.0 vs. sparse median 67.5) — full-population
  confirmation of the dense+large failure mode identified in the Part 1 spot
  check (Section 5.6 above).

- _The dense-bucket warning conflates two opposite risk profiles (Section
  5.6, Part 4): safe for vertex prediction at all sizes, safe for edge
  prediction only at small sizes, high-risk for edge prediction at large
  sizes. Recommend a size-aware warning split rather than the current flat
  MAE citation._

The current single "dense bucket" warning (flat MAE citation of 0.43/1.12)
therefore conflates two graphs with opposite risk profiles. A size-aware
warning — distinguishing dense+small (reliable) from dense+large
(high-risk) — would be both more accurate and more actionable; noted as a
concrete recommendation in Section 7.

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
- _Dense-bucket warning miscalibration for large dense graphs (Section 5.6:
  dense-but-large graphs fail far worse than the cited dense-bucket MAE
  suggests — the warning's historical accuracy figure is likely dominated by
  small dense graphs)_
- _GCN baseline's out-of-distribution-size degradation (Section 5.6: on the
  live spot check, the GCN baseline degraded more than the CNN on
  OOD-size graphs, n=6 — small sample, worth naming as a limitation of the
  baseline comparison's generalizability rather than a settled result)_
- _Combinatorial-plausibility claim is empirically qualified, not absolute
  (Section 5.6: one exception found out of 30 live-app predictions, on an
  input — multiple diagrams plus text in one frame — outside anything seen
  in training)_

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
| `topolens_spotcheck_results.csv` | Live-app spot check: 23 graphs, CNN/GCN/baseline predictions, warning-fired status (Section 5.6, Part 1) |
| `topolens_novelimage_results.csv` | Live-app novel-image sanity check: 7 images, no ground truth, combinatorial-plausibility check (Section 5.6, Part 2) |

### Application (`app/`)

| File | Description |
|---|---|
| `app.py` | Streamlit inference app — **Neo-Brutalist design system** (Syne display + Plus Jakarta Sans body); single multi-type file uploader (`png`, `jpg`, `jpeg`, `graphml`, `csv`, `txt`); multi-file select; CNN inference + hero stat display; Grad-CAM toggle (vertex-target + edge-target overlays); empirical validation panel (Ground Truth vs Prediction table + topological-property cards) active for both graph files and dataset images matched via `labels_processed.csv`; model limitations expander with live CSV values. |
| `.streamlit/config.toml` | Theme: `backgroundColor = "#F5F5F5"`, `primaryColor = "#000000"`, `secondaryBackgroundColor = "#E5E5E5"`, `textColor = "#000000"`. |
