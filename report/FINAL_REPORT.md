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

**Pixel-only baseline (Section 5.6, Part 4, item 3).** A simple linear
regression of `ink_fraction → num_edges` explains 82% of edge-count variance
on synthetic test data (r=0.905) but is essentially uninformative on
real-world held-out data (r=0.059) — see Section 5.6 for the full breakdown
and its implication for the CNN's real-world generalization.

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

**Part 4 (free items) — statistical and correlational follow-ups.** Four
analyses requiring no new data collection, only further treatment of
existing evaluation CSVs. Full numeric results in
`evaluation/results/topolens_part4_free_analyses_results.csv` (items 1–2) and
`evaluation/results/topolens_pixel_structural_correlations.csv` (items 3–4).

_1. Wilcoxon signed-rank test on probe deltas (n=40, paired per graph)._ The
shortcut probe's vertex-error increase under `constant_node_size` is
significant (median 1.0 → 11.0, p=3.6×10⁻⁶, 33/40 graphs worse) — this
upgrades the Section 5.2 finding from a summary MAE delta to a formal
significance result. The edge-error comparison requires care: the _mean_
appears to improve (54.18 → 39.80), but this is an artifact of two extreme
outlier graphs (both dense+large, the same failure mode as the Part 1
spot-check's worst case); the median (3.5 → 20.0, 35/40 graphs worse) and
the Wilcoxon test itself (p=0.0003, significant in the _worsening_
direction) both confirm the shortcut degrades edge prediction for the large
majority of graphs — report median/win-loss counts, not the mean, for this
comparison. Separately, no significant layout effect was found (Section 5.3)
under `alt_layout`/Kamada-Kawai on either vertex (p=0.499) or edge (p=0.898)
error — a clean null result contradicting what raw mean MAE alone would
have suggested (again driven by a single outlier graph).

_2. Warning-banner (dense-bucket) validation, holding size constant via
bins, across all 1,676 held-out+test predictions._ The dense-bucket warning's
validity depends entirely on which target it's warning about. For **vertex
count**, the warning is backwards — dense graphs have lower error than
sparse graphs at every size bin tested (e.g. 75–100 nodes: dense median
error 5.0 vs. sparse 21.0); dense is the safest bucket for vertex
prediction, not a risk signal. For **edge count**, the warning is
directionally correct but size-miscalibrated: dense and sparse edge error
are statistically tied at small sizes, then diverge sharply, reaching a
3.3x gap by 75–100 nodes (dense median 224.0 vs. sparse median 67.5) — this
is the full-population statistical confirmation of the Part 1 spot-check's
"dense+large is catastrophic" finding. The current single "dense bucket"
warning (flat MAE citation of 0.43/1.12) conflates two graphs with opposite
risk profiles and should ideally be split into a size-aware version.

_3. Pixel-only baseline: linear regression of `ink_fraction → num_edges`._
On the test split (375 graphs, 100% synthetic), ink coverage alone explains
82% of edge-count variance (r=0.905, R²=0.819), though the CNN is still
~3.3x more accurate in absolute MAE (25.39 vs. 83.73 for the linear fit). On
the held-out split (1,301 graphs, 100% real MUTAG/PROTEINS), ink coverage is
essentially uninformative (r=0.059, R²=0.004) — `ink_fraction` barely varies
(0.092–0.192) despite true edge counts spanning 5–1,049. **This means the
CNN's real-world generalization cannot be explained by an ink-counting
shortcut, since that shortcut does not meaningfully exist in real-graph
renders** — a reassuring result for the Section 1 research question. Within
held-out, the correlation is highly heterogeneous by category:
`high_density_clutter` alone shows r=0.947, while `in_distribution_normal`
shows a slight negative correlation (r=−0.206); the aggregate near-zero
figure masks this.

_4. Structural correlations beyond size/density: diameter and clustering
coefficient vs. error._ `avg_clustering` is highly collinear with `density`
(r=0.836 test, r=0.596 held-out) and mostly redescribes it rather than
adding independent signal. `diameter` is more independent of `num_nodes`
(r=0.336 test, r=0.719 held-out), and its partial correlation with error
after controlling for `num_nodes` remains meaningful: +0.478 (test, vertex
error) and −0.287 (held-out, vertex error). Notably, **the sign flips
between synthetic and real data** — on synthetic graphs, a
larger-than-expected diameter for a given size predicts _worse_ vertex
accuracy; on real graphs, the opposite. Reported as an exploratory,
non-obvious finding rather than a settled mechanism; likely reflects a
structural difference between how the synthetic generators and real
molecule/protein graphs produce diameter at matched size.

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
- _The empirically confirmed node-size shortcut (Section 5.2 probe finding,
  now statistically confirmed via Wilcoxon signed-rank test, Section 5.6
  Part 4 — report median/win-loss counts for edge error, not the mean,
  which is outlier-distorted)_
- _The out-of-distribution size ceiling (n > 100; Section 5.5 MAE values)_
- _The dense-bucket warning conflates two opposite risk profiles (Section
  5.6, Part 4): safe (in fact the safest bucket) for vertex prediction at
  every size tested, safe for edge prediction only at small sizes, and
  high-risk for edge prediction at large sizes (3.3x worse than sparse by
  75–100 nodes). Recommend a size-aware warning split rather than the
  current flat MAE citation (0.43/1.12)._
- _GCN baseline's out-of-distribution-size degradation (Section 5.6: on the
  live spot check, the GCN baseline degraded more than the CNN on
  OOD-size graphs, n=6 — small sample, worth naming as a limitation of the
  baseline comparison's generalizability rather than a settled result)_
- _Combinatorial-plausibility claim is empirically qualified, not absolute
  (Section 5.6, Part 3: one exception found out of 30 live-app predictions,
  on an input — multiple diagrams plus text in one frame — outside anything
  seen in training)_
- _The pixel-only baseline result (Section 5.6, Part 4) should be read as
  reassuring rather than as evidence of a shortcut: ink-coverage explains
  most edge-count variance on synthetic data but almost none on real
  held-out data, meaning the CNN's real-world accuracy is not attributable
  to pixel-counting — however, synthetic-only evaluation may still overstate
  how much the CNN has learned beyond low-level pixel statistics, and
  synthetic results should be read with that caveat._

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
| `topolens_part4_free_analyses_results.csv` | Wilcoxon signed-rank test results (shortcut + layout probes) and warning-banner validation by size bin (Section 5.6, Part 4, items 1–2) |
| `structural_pixel_features.csv` | Per-graph ink fraction, diameter, and clustering coefficient merged onto `failure_case_categories.csv`, generated by `compute_structural_pixel_features.py` (source data for Part 4, items 3–4) |
| `topolens_pixel_structural_correlations.csv` | Pixel-only baseline regression stats and structural (diameter/clustering) correlation + collinearity + partial-correlation results (Section 5.6, Part 4, items 3–4) |

### Application (`app/`)

| File | Description |
|---|---|
| `app.py` | Streamlit inference app — **Neo-Brutalist design system** (Syne display + Plus Jakarta Sans body); single multi-type file uploader (`png`, `jpg`, `jpeg`, `graphml`, `csv`, `txt`); multi-file select; CNN inference + hero stat display; Grad-CAM toggle (vertex-target + edge-target overlays); empirical validation panel (Ground Truth vs Prediction table + topological-property cards) active for both graph files and dataset images matched via `labels_processed.csv`; model limitations expander with live CSV values. |
| `.streamlit/config.toml` | Theme: `backgroundColor = "#F5F5F5"`, `primaryColor = "#000000"`, `secondaryBackgroundColor = "#E5E5E5"`, `textColor = "#000000"`. |
