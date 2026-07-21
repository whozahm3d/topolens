# Topolens: Estimating Graph Structural Properties from Rendered Images

**Abstract.** Topolens investigates whether a convolutional neural network
operating purely on rendered images of graphs can estimate structural
properties — vertex count and edge count — competitively with a graph-native
model operating on raw topology. We render 2,500 synthetic graphs (5
generator families × 4 size tiers) via Graphviz `sfdp` and hold out 1,301 real
graphs from the MUTAG and PROTEINS benchmarks as a generalization test never
seen in training. A custom 4-block CNN is compared against a 3-layer GCN
baseline and a density-based graph-statistic heuristic. The CNN outperforms
the GCN baseline on every metric across both splits, in some cases by 4–6x,
answering the core research question in the affirmative for this setup — but
interpretation analysis (Grad-CAM, shortcut probes, pixel-only regression, and
a live-app validation round) shows this accuracy is partly attributable to an
identified node-size rendering confound rather than pure topological
understanding, and the comparison's fairness is qualified by the GCN
baseline's minimal (single-scalar, degree-only) node features. Both findings
are discussed as substantive, not incidental, results.

---

## 1. Problem Framing

Graph Neural Networks (GNNs) are the default tool for learning on graph
data because they operate directly on topology — nodes, edges, and message
passing between them. Topolens asks a deliberately contrarian question:
if a graph is instead *rendered as an image* and handed to an ordinary CNN,
how much of its structure can still be recovered, and by what means?

This is not a claim that CNNs should replace GNNs. It is a controlled
comparison meant to expose what a purely visual model can and cannot infer
about structure, and — just as importantly — *how* it gets there. Two
sub-questions follow directly from this framing and organize the rest of
the report:

1. **Competitiveness.** Can a CNN trained on rendered graph images match or
   beat a GNN trained on the same underlying graphs, for the task of
   predicting vertex and edge counts, on both in-distribution synthetic data
   and out-of-distribution real-world data?
2. **Mechanism.** If the CNN performs well, is it doing something that
   resembles structural reasoning, or is it exploiting rendering artifacts —
   node size, layout geometry, ink density — that happen to correlate with
   the labels but would not survive a change in rendering convention?

Section 3–4 answer (1) quantitatively. Section 5 answers (2) through Grad-CAM,
targeted shortcut probes, and correlational analysis. Section 6 synthesizes
both into a single answer to the framing question above.

---

## 2. Methodology

### 2.1 Dataset

**Synthetic (primary, used for training/validation/testing).** 2,500 graphs
generated via NetworkX, stratified across 5 generator families —
Erdős–Rényi, Barabási–Albert, Watts–Strogatz, random trees, and dense
Erdős–Rényi variants — crossed with 4 node-count tiers:

| Tier | Node range |
|---|---|
| tiny | 5–10 |
| small | 10–25 |
| medium | 25–50 |
| large | 50–100 |

125 graphs per (generator × tier) cell (5 × 4 × 125 = 2,500). Each graph's
render/layout seed is derived deterministically from its `graph_id` via
SHA-256, so regenerating the dataset from scratch is reproducible bit-for-bit
(verified directly — see `test_render.py`). Split 70/15/15 (train 1,750 /
val 375 / test 375), stratified by generator type, seed 42.

**Real-world (secondary, held-out generalization test).** MUTAG (188 graphs)
and PROTEINS (1,113 graphs) loaded via `TUDataset`/PyTorch Geometric and
stripped to plain topology (node/edge structure only — no chemical/biological
node or edge attributes are used, keeping the task identical to the synthetic
case). Combined held-out set: 1,301 graphs, **never used in training or
model selection**. Notably, 67 of the PROTEINS graphs exceed the 100-node
training ceiling (up to 620 nodes) — this is a deliberate, not incidental,
out-of-distribution generalization test and is treated as such throughout
Section 5.

### 2.2 Rendering Pipeline

Each graph is rendered to a 224×224 px PNG at 100 DPI via a three-stage
fallback chain: `pygraphviz`'s `sfdp` layout → `pydot`'s `sfdp` layout (if
`pygraphviz` is unavailable) → NetworkX's deterministic spring layout (if
neither Graphviz binding is available). In practice, **all 3,801 rendered
images** (2,500 synthetic + 1,301 real) used `graphviz_sfdp` — confirmed via
the `layout_algorithm` column in `data/processed/labels_processed.csv`, and
corroborated by every image file's modification time falling inside a single
~4.5-hour render batch, ruling out a mixed-layout dataset. This had been a
genuine open question after an early Graphviz-binary-missing incident (Day
2) forced a temporary spring-layout fallback; the full re-render on Day 3
resolved it, and the above confirms no stale spring-layout images survived
into the final dataset.

Rendering style: white background, uniform node color, no axes/labels/arrows,
thin fixed-width edges (width 0.6), and node radius scaled as
`max(15, 4000 / n)` — i.e. nodes shrink as graphs grow. This formula was
flagged as a potential shortcut-learning confound at design time (a model
could learn to infer node count from apparent node size rather than counting
node "blobs"); Section 5.2 and 5.6 confirm this concern was justified. A
small set of ~26 additional images was rendered manually in Gephi for
qualitative visual comparison only — not used in training or evaluation.

### 2.3 Model Architecture

**CNN (`CustomCNNRegressor`).** Four convolutional blocks (3×3 conv → batch
norm → ReLU → 2×2 max-pool), channel progression 3→32→64→128→256, followed
by global adaptive average pooling and a fully-connected head
(256→128→ReLU→Dropout(0.2)→2). Both targets (vertex count, edge count) are
predicted jointly via `log1p` transformation and inverted at inference time
via `expm1` + rounding + clamping to non-negative integers — this stabilizes
training across the wide dynamic range of counts (5 to 1,049 edges).

**GCN baseline (`GraphCountGCN`).** Three `GCNConv` layers (1→64→64→64) with
`BatchNorm1d` after each, `global_mean_pool`, and the same FC-head shape and
`log1p` target convention as the CNN, for a fair loss-scale comparison. The
single input node feature is **degree normalized by the graph's maximum
degree** — no richer structural encodings (clustering coefficient, spectral
features, positional encodings) are provided. This is a real design choice
with consequences discussed in Section 6. Mean pooling was substituted for
an earlier sum-pooling implementation after a Day 4 bug: sum-pooling
entangled the pooled representation with node count in a way that degraded
catastrophically on differently-sized held-out graphs (thousands-scale MAE
before the fix).

**Graph-statistic baseline.** Not a learned model. Vertex count is returned
directly as the true node count — a trivial oracle, flagged explicitly as
non-comparable and included only so the edge-count heuristic has a place to
live. Edge count is estimated as `mean_density_per_generator × n(n−1)/2`,
where per-generator mean density is fit exclusively on the training split.

### 2.4 Training Setup

Both models trained for 30 epochs, Adam optimizer, MSE loss on `log1p`
targets, batch size 32, learning rate 0.001 (`config.yaml`), fixed seed 42
for data splitting and weight initialization. Training moved from a local
CPU-only run (projected ~5 hours for the CNN alone) to Google Colab (T4 GPU)
on Day 4. Best-validation-loss checkpoints are saved to
`models/checkpoints/cnn_best.pt` and `gnn_best.pt` (not last-epoch weights).
One full retrain was later run to isolate GPU/cuDNN non-determinism between
runs — this shifted the test-split overall vertex MAE from 4.5 to 3.2
between two otherwise-identical runs, noted as a reproducibility caveat in
Section 7.

---

## 3. Baseline Comparison

> **Note:** The graph-statistic baseline's 0.00 vertex MAE is structurally
> trivial — it returns the true node count directly rather than estimating
> it. It is reported for completeness, not as a meaningful competing result.
> Only its edge heuristic (`mean_density × n(n−1)/2`) is a genuine, non-leaking
> prediction.

**Test split (in-distribution synthetic, n=375):**

| Target | Metric | CNN | GCN | Graph-statistic |
|---|---|---|---|---|
| Vertices | MAE | **3.20** | 19.05 | 0.00* |
| Vertices | RMSE | **6.08** | 26.61 | 0.00* |
| Edges | MAE | **25.39** | 148.98 | 120.53 |
| Edges | RMSE | **81.75** | 391.67 | 233.51 |

**Held-out split (real MUTAG + PROTEINS, n=1,301):**

| Target | Metric | CNN | GCN | Graph-statistic |
|---|---|---|---|---|
| Vertices | MAE | **10.62** | 21.15 | 0.00* |
| Vertices | RMSE | **31.30** | 44.09 | 0.00* |
| Edges | MAE | **28.03** | 56.84 | 363.65 |
| Edges | RMSE | **67.72** | 92.82 | 2,155.29 |

The CNN wins on every single metric across both splits — by roughly 4–6x on
the synthetic test split, and by 2x on the real held-out split. It also beats
the density-based graph-statistic heuristic on edge count in the aggregate,
despite that heuristic having direct access to true density statistics fit
on the training data. This is the headline quantitative result motivating
Section 5's interpretation work: a result this decisive demands scrutiny of
*why*, not just acceptance of the number.

---

## 4. Quantitative Results

**By generator type (test split, MAE):**

| Generator | CNN Vert. | GCN Vert. | CNN Edges | GCN Edges | Graph-stat Edges |
|---|---|---|---|---|---|
| barabasi_albert | 2.83 | 18.68 | 13.80 | 81.53 | 184.23 |
| dense | 1.75 | 21.65 | 75.63 | 497.71 | 76.67 |
| erdos_renyi | 2.47 | 15.49 | 22.48 | 96.92 | 51.83 |
| random_tree | 5.69 | 17.71 | 4.77 | 17.60 | 87.81 |
| watts_strogatz | 3.27 | 21.71 | 10.27 | 51.15 | 202.09 |

The CNN's vertex MAE is remarkably stable across generator families (1.7–5.7)
while the GCN's is uniformly worse and similarly flat (15.5–21.7) — the GCN's
weakness is not concentrated in any one topology family. Edge prediction is
harder for both models on `dense` graphs, where absolute edge counts are
largest and the density heuristic — with direct access to per-generator
density — is (expectedly) the strongest of the three.

**By source dataset (held-out split, MAE):**

| Dataset | CNN Vert. | GCN Vert. | CNN Edges | GCN Edges | Graph-stat Edges |
|---|---|---|---|---|---|
| MUTAG | 1.07 | 5.45 | 1.85 | 17.03 | 24.60 |
| PROTEINS | 12.23 | 23.80 | 32.45 | 63.57 | 420.93 |

PROTEINS is markedly harder for every model than MUTAG — expected, since it
contains the 67 graphs exceeding the training's 100-node ceiling, and its
size range is far broader overall (MUTAG graphs are small and structurally
homogeneous; PROTEINS spans 4–620 nodes).

**By density bucket (held-out split, MAE):**

| Bucket | CNN Vert. | GCN Vert. | CNN Edges | GCN Edges | Graph-stat Edges |
|---|---|---|---|---|---|
| dense | 0.43 | 13.94 | 1.13 | 72.10 | 6.51 |
| medium | 2.03 | 7.57 | 5.91 | 44.96 | 12.63 |
| sparse | 18.75 | 32.13 | 49.11 | 61.90 | 683.54 |

This is a counter-intuitive but consistent finding, confirmed again at full
scale in Section 5.6: **dense graphs are the easiest bucket for the CNN, not
the hardest**, and sparse graphs are by far the worst (18.75 vertex MAE, 49.11
edge MAE — an order of magnitude above the dense bucket). This directly
foreshadows Section 5.5's failure taxonomy, where out-of-distribution *size*
turns out to be the dominant failure mode, not visual clutter or density as
originally hypothesized at design time. Per-graph scatter plots
(`report/figures/cnn_pred_num_vertices_scatter_*.png`,
`cnn_pred_num_edges_scatter_*.png`) and error-vs-size/density plots
(`error_vs_num_nodes.png`, `error_vs_density.png`) show this pattern visually:
error grows with node count far more sharply than with density.

---

## 5. Model Interpretation & Failure Analysis

### 5.1 Grad-CAM: What Does the Model Look At?

Grad-CAM is computed at `features[3]`, the CNN's final (256-channel)
convolutional block, with two independent backward passes per image — one
targeting the vertex-count output, one targeting the edge-count output
(`models/gradcam.py`). Because this layer sits after four 2×2 max-pool
stages, the resulting heatmap has a native resolution of only 14×14 for a
224×224 input; overlays should be read as coarse regions of interest, not
precise per-node attribution.

**By generator type** (`gradcam_grid_by_generator.png`). Activation is
consistently concentrated on node clusters rather than empty background
across all five generator families — the model is not attending to
irrelevant image regions. The spatial *spread* of activation tracks visual
density rather than topology type directly: sparse, tree-like graphs
(`random_tree`, low-`p` `erdos_renyi`) show activation tightly localized
around a small number of node hubs, while `dense` and `watts_strogatz`
graphs show activation spread broadly across the whole node-cluster region.
Vertex-target and edge-target maps overlap substantially but are not
identical: the vertex-target map is more tightly localized on individual
node blobs, while the edge-target map is more diffuse, extending into the
inter-node space where edges are drawn — consistent with edge count
requiring information about the connective tissue between nodes rather than
node identity alone.

**Failure cases** (`gradcam_grid_failure_cases.png`). For the worst-error
graphs, activation frequently concentrates on a single densely-packed
sub-region rather than spreading across the graph's full spatial extent.
This is consistent with two findings elsewhere in this report rather than
an independent discovery: it's compatible with the out-of-distribution-size
failure mode (Section 5.5), where a model trained on smaller, less
spatially extensive graphs may not have learned to attend across a large
image's full area, and with the node-size shortcut (Section 5.2), where
attention gravitating toward node-blob regions is exactly the mechanism a
model exploiting apparent node size would be expected to show. Individual
per-image overlays are in `report/figures/gradcam/` for closer inspection
of specific cases.

### 5.2 Shortcut-Learning Probe (Node-Size Confound)

The renderer scales node radius as `max(15, 4000/n)` (Section 2.2), flagged
at design time as a possible shortcut: a model could infer vertex count from
node *size* rather than from counting distinct nodes. To test this directly,
the same 40 probe graphs (10 per tier) were re-rendered with node radius
fixed at a **constant** value regardless of `n`, holding topology, layout
algorithm, edge style, and image resolution identical to the canonical
render — isolating node-size as the only variable that changes.

| Tier | Original vertex MAE | Constant-node-size vertex MAE | Original edge MAE | Constant-node-size edge MAE |
|---|---|---|---|---|
| tiny | 0.3 | 1.4 | 1.6 | 3.5 |
| small | 1.0 | 8.5 | 3.3 | 21.2 |
| medium | 3.5 | 15.3 | 16.3 | 51.7 |
| large | 8.9 | 16.7 | 195.5 | 82.8 |
| **overall (n=40)** | **3.425** | **10.475** | **54.175** | **39.8** |

Overall vertex MAE roughly triples under constant node size (3.43 → 10.48),
the single largest effect size of any probe in this report, and the effect
holds in every tier without exception. Its *relative* magnitude, however, is
not driven by absolute graph size the way the OOD-size failure mode is
(Section 5.5): the small tier shows the largest multiplicative jump (1.0 →
8.5, 8.5×) while the large tier shows the smallest (8.9 → 16.7, 1.9×) despite
having the largest absolute error under both conditions. This is consistent
with node size being a comparatively strong, easily-exploited cue in
low-to-mid-range graphs, where individual node blobs are still visually
distinguishable, and a comparatively weaker one in large/dense graphs, where
node blobs already overlap and the model likely leans on other pixel-level
cues such as ink coverage (Section 5.4) instead.

Overall edge MAE appears to *decrease* under constant node size (54.18 →
39.80), which read in isolation would suggest the shortcut actively helps
edge estimation. Section 5.6 (Part 4, item 1) shows this reading is an
artifact: two dense/large outlier graphs dominate the mean, while the median
(3.5 → 20.0 with node size held constant) and a Wilcoxon signed-rank test
(p ≈ 0.0003) both confirm edge accuracy degrades for the large majority of
probe graphs (35 of 40) under the same intervention — the same statistical
correction applied to the vertex-MAE tripling (Wilcoxon p ≈ 3.6×10⁻⁶).
Taken together, this probe is the single strongest piece of evidence in the
report that part of the CNN's headline accuracy (Section 3) comes from a
rendering convention rather than pure topological inference. It is the
finding that most directly qualifies the Section 3 result and motivates
both the Discussion (Section 6) and the Limitations (Section 7).

### 5.3 Layout-Sensitivity Probe

The same 40 probe graphs (10 per tier) used in Section 5.2 were re-rendered
with NetworkX's Kamada-Kawai layout in place of Graphviz `sfdp` — same node
size, same edge style, only the node *positions* change — to test whether
the model has overfit to `sfdp`'s specific spatial conventions.

| Tier | Original vertex MAE | Alt-layout vertex MAE | Original edge MAE | Alt-layout edge MAE |
|---|---|---|---|---|
| tiny | 0.3 | 0.4 | 1.6 | 1.6 |
| small | 1.0 | 1.0 | 3.3 | 4.9 |
| medium | 3.5 | 3.8 | 16.3 | 13.2 |
| large | 8.9 | 7.1 | 195.5 | 122.4 |
| **overall (n=40)** | **3.425** | **3.075** | **54.175** | **35.525** |

Vertex accuracy is essentially flat across the layout change at every tier.
The overall edge MAE appears to *improve* under `alt_layout` (54.18 →
35.53), but per-graph inspection shows this is the same kind of mean
distortion flagged in Section 5.6: the median edge error is actually
slightly *worse* under Kamada-Kawai (3.5 → 4.5), and only 22 of 40 graphs
improve — the aggregate mean is dominated by one graph,
`syn_dense_large_0069` (the same graph already flagged as the worst Day-7
spot-check and shortcut-probe failure), whose edge error dropped by 512
(1,084 → 572) under the new layout, alongside two other large/dense graphs
that improved substantially (−90, −74); several small/medium graphs got
moderately worse (+17, +14, +12) in the same swap. This is formalized as a
null result in the Wilcoxon signed-rank test already reported in Section
5.6, Part 4, item 1 (p=0.499 vertex, p=0.898 edge) — the layout swap
produces no statistically significant change in either direction, in
contrast to the node-size swap (Section 5.2), which does.

Taken together with Section 5.2, this is a useful contrast: the model *is*
measurably reliant on the node-size rendering convention, but is *not*
measurably reliant on `sfdp`'s specific node-placement geometry — at least
against this one alternative layout algorithm. This should not be read as
general layout robustness (Section 7 notes the test covers only one
alternative, Kamada-Kawai, out of many layout families with meaningfully
different visual conventions), but within that scope, the model's accuracy
does not appear contingent on `sfdp` specifically.

### 5.4 Ink-Coverage Correlational Analysis

Two low-level pixel statistics — `ink_fraction` (share of non-background
pixels) and `mean_component_area` (average size of connected dark-pixel
blobs) — were correlated against true and predicted counts, to test whether
the CNN's outputs track these simple image statistics rather than genuine
structural features.

| Pair | Test (synthetic) r | Held-out (real) r |
|---|---|---|
| ink_fraction vs. num_nodes | 0.538 | 0.070 |
| ink_fraction vs. num_edges | 0.824 | 0.093 |
| ink_fraction vs. pred_num_vertices | 0.618 | 0.172 |
| ink_fraction vs. pred_num_edges | 0.845 | 0.465 |
| mean_component_area vs. num_nodes | 0.552 | −0.083 |
| mean_component_area vs. pred_num_vertices | 0.627 | 0.001 |
| num_components vs. num_nodes | −0.240 | 0.308 |

(All test-split correlations n=375, p<10⁻²⁸; held-out n=1301, p-values range
from <10⁻⁶⁹ down to 0.97 for the near-zero rows — see
`evaluation/results/ink_coverage_correlations.csv` for exact values.)

On the synthetic test split, both statistics behave as expected of a
shortcut: `ink_fraction` correlates almost identically with true edge count
(r=0.824) and the CNN's *predicted* edge count (r=0.845), and
`mean_component_area` correlates about as strongly with predicted vertex
count (r=0.627) as with true vertex count (r=0.552). The CNN's synthetic-data
outputs track these pixel statistics about as well as they track the ground
truth. `num_components` shows a modest negative correlation with node count
(r=−0.240) — larger synthetic graphs tend to have *fewer* separate
components, consistent with how the generator families are constructed
(Barabási–Albert graphs are connected by construction; Erdős–Rényi graphs at
these tiers sit above the giant-component threshold).

On the held-out (real) split, these relationships largely collapse.
`ink_fraction` vs. true edge count falls to r=0.093 — essentially no
relationship — while `ink_fraction` vs. *predicted* edge count remains
moderate at r=0.465. That gap (0.093 true vs. 0.465 predicted) is itself
informative: the CNN's real-data predictions still partially echo the
pixel-coverage cue even where that cue no longer tracks the actual target,
which is independent evidence (alongside the pixel-only baseline below) that
some of the model's real-world error traces back to a synthetic-data
shortcut that doesn't transfer. `mean_component_area` vs. predicted vertex
count drops to essentially zero (r=0.001), and `num_components` vs. node
count *flips sign* entirely (test r=−0.240 → held-out r=+0.308) — MUTAG's
small, dense molecular graphs and PROTEINS' larger, sparser contact maps
simply do not follow the synthetic generators' size-vs-connectivity pattern,
confirming the synthetic and real distributions differ structurally and not
only in scale.

**Pixel-only baseline (Section 5.6, Part 4, item 3).** A directly-fit linear
regression of `ink_fraction → num_edges` explains 82% of edge-count variance
on synthetic test data (r=0.905, R²=0.819) but is essentially uninformative
on real-world held-out data (r=0.059, R²=0.004) — the same qualitative
pattern as the raw correlations above, and the clearer piece of evidence that
the CNN's real-world accuracy is not simple pixel-counting (see Section 5.6
for the full breakdown). Note that this regression's r (0.905) and this
section's raw Pearson r for the same nominal pair (0.824) do not match
exactly: the two are computed by different scripts using different
background thresholds (`image_statistics.py` uses `white_threshold=250`;
`compute_structural_pixel_features.py` uses `bg_threshold=200`), a
discrepancy identified but never reconciled to one canonical `ink_fraction`
definition (see `progress_log.md`, 20 Jul entry, and Section 7). Both figures
support the same conclusion at different resolutions of the same measurement
convention, but the exact magnitude should not be treated as more precise
than that.

### 5.5 Failure-Case Taxonomy

Every test and held-out graph was assigned to exactly one of three
categories, checked in priority order: **out_of_distribution_size** (node
count exceeds the training set's empirical maximum, derived from
`data/splits/train.csv` rather than hard-coded — 100 nodes in practice) takes
precedence over density; among in-training-size graphs,
**high_density_clutter** (`density ≥ 0.40`, the same `dense` bucket used in
Section 4) is separated from **in_distribution_normal** (density below 0.40,
covering both the `sparse` and `medium` buckets).

| Split | Category | Mean vertex MAE | Median vertex MAE | Mean edge MAE | Median edge MAE | n |
|---|---|---|---|---|---|---|
| held_out | high_density_clutter | 0.43 | 0.0 | 1.13 | 1.0 | 144 |
| held_out | in_distribution_normal | 6.67 | 3.0 | 20.79 | 10.0 | 1,090 |
| held_out | out_of_distribution_size | 96.64 | 71.0 | 203.60 | 155.0 | 67 |
| test | high_density_clutter | 1.27 | 1.0 | 51.82 | 4.0 | 110 |
| test | in_distribution_normal | 4.00 | 2.0 | 14.42 | 3.0 | 265 |

(The synthetic test set is capped at 100 nodes by construction, so no
`out_of_distribution_size` graphs exist in that split — the category is only
populated by held-out real-world graphs, where MUTAG/PROTEINS include an
"xlarge" tier up to 620 nodes.)

On the held-out split, the ranking is unambiguous: `out_of_distribution_size`
is both the smallest category by count (67 of 1,301 graphs, ~5%) and by a
wide margin the worst by error — mean vertex MAE 96.64 and edge MAE 203.60,
roughly 14× and 10× `in_distribution_normal`'s 6.67 and 20.79 respectively.
`high_density_clutter` is not merely tolerable but the single *best*-performing
category (mean vertex MAE 0.43, edge MAE 1.13) — confirming, at full-population
scale, the counter-intuitive density-bucket finding from Section 4: dense,
visually cluttered graphs are not a failure mode for this model, they are its
strongest regime.

On the test split, the same ordering largely holds: `high_density_clutter`
again outperforms `in_distribution_normal` on vertex count (1.27 vs. 4.00),
though this reverses for edge count (51.82 vs. 14.42) — consistent with
Section 4's generator-level finding that dense graphs are harder specifically
for edge counting, given their much larger absolute edge counts, even while
remaining easy for vertex counting. The test-split `high_density_clutter` row
also shows a large mean/median gap for edge MAE (mean 51.82 vs. median 4.0,
n=110) — the same kind of outlier-driven mean distortion flagged throughout
Section 5.6, plausibly involving the same dense/large outlier graph
(`syn_dense_large_0069`) already identified in Sections 5.3 and 5.6. The
median is the more representative summary of this category's typical
performance.

Taken together, these results reorder the failure modes anticipated at
design time: Section 2.2 flagged visual clutter and node overlap as the
primary expected risk, but the taxonomy shows **size generalization, not
density, is the dominant and by far most severe failure mode**. A graph need
only exceed the size the model was trained on to fail badly; it does not
need to be visually cluttered to fail, and being visually cluttered does not,
on its own, predict failure. This finding directly motivates the Section 6
discussion and the Section 8 future-work item on extending the training size
ceiling.

### 5.6 Live-App Validation: Spot Check & Novel-Image Sanity Check

*Populated from `evaluation/results/topolens_spotcheck_results.csv` (Part 1) and
`evaluation/results/topolens_novelimage_results.csv` (Part 2). Unlike Sections
5.1–5.5, which analyze the offline held-out/test split, this section validates
the deployed app's live inference path — including the warning banner and
multi-model comparison — against inputs chosen and uploaded after training.*

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
  graphs that are dense *and* large.
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
with *visual unfamiliarity relative to the training render style*, not with
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
`evaluation/results/topolens_free_analyses_results.csv` (items 1–2) and
`evaluation/results/topolens_pixel_structural_correlations.csv` (items 3–4).

*1. Wilcoxon signed-rank test on probe deltas (n=40, paired per graph).* The
shortcut probe's vertex-error increase under `constant_node_size` is
significant (median 1.0 → 11.0, p=3.6×10⁻⁶, 33/40 graphs worse) — this
upgrades the Section 5.2 finding from a summary MAE delta to a formal
significance result. The edge-error comparison requires care: the *mean*
appears to improve (54.18 → 39.80), but this is an artifact of two extreme
outlier graphs (both dense+large, the same failure mode as the Part 1
spot-check's worst case); the median (3.5 → 20.0, 35/40 graphs worse) and
the Wilcoxon test itself (p=0.0003, significant in the *worsening*
direction) both confirm the shortcut degrades edge prediction for the large
majority of graphs — report median/win-loss counts, not the mean, for this
comparison. Separately, no significant layout effect was found (Section 5.3)
under `alt_layout`/Kamada-Kawai on either vertex (p=0.499) or edge (p=0.898)
error — a clean null result contradicting what raw mean MAE alone would
have suggested (again driven by a single outlier graph).

*2. Warning-banner (dense-bucket) validation, holding size constant via
bins, across all 1,676 held-out+test predictions.* The dense-bucket warning's
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

*3. Pixel-only baseline: linear regression of `ink_fraction → num_edges`.*
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

*4. Structural correlations beyond size/density: diameter and clustering
coefficient vs. error.* `avg_clustering` is highly collinear with `density`
(r=0.836 test, r=0.596 held-out) and mostly redescribes it rather than
adding independent signal. `diameter` is more independent of `num_nodes`
(r=0.336 test, r=0.719 held-out), and its partial correlation with error
after controlling for `num_nodes` remains meaningful: +0.478 (test, vertex
error) and −0.287 (held-out, vertex error). Notably, **the sign flips
between synthetic and real data** — on synthetic graphs, a
larger-than-expected diameter for a given size predicts *worse* vertex
accuracy; on real graphs, the opposite. Reported as an exploratory,
non-obvious finding rather than a settled mechanism; likely reflects a
structural difference between how the synthetic generators and real
molecule/protein graphs produce diameter at matched size.

---

## 6. Discussion

The Section 1 research question — can an image-based CNN compete with a
graph-native GNN on structural regression — is answered **yes, decisively,
for this specific comparison**: the CNN beats the GCN baseline on every
metric, on both in-distribution synthetic data and out-of-distribution real
data, by margins as large as 4–6x. But the honest answer needs two immediate
qualifications, both of which the interpretation work in Section 5 was
specifically designed to surface.

**First, part of the CNN's win is a confound, not pure structural
understanding.** The rendering pipeline scales node radius inversely with
node count (`max(15, 4000/n)`) by design, and Section 5.2/5.6 confirm — with
a formal Wilcoxon test (p=3.6×10⁻⁶) — that holding node size artificially
constant significantly degrades vertex accuracy for 33 of 40 probed graphs.
This means the CNN is not purely "counting nodes" the way a human would; it
is partly reading node count off of apparent node size, a signal that exists
only because of a specific, otherwise-arbitrary rendering choice. This is a
genuinely useful research result in its own right: it demonstrates that an
image-based structural estimator's accuracy can be substantially inflated by
rendering conventions that are invisible in the reported metric and would
not survive a change of renderer, plotting library, or diagram style.
Reassuringly, the pixel-only ink-coverage regression (Section 5.6, Part 4
item 3) shows this is not simply "counting pixels" either — ink coverage
explains most edge-count variance on synthetic data but almost none on real
data, so whatever the CNN has learned generalizes beyond crude pixel
statistics even if it does not fully escape the node-size channel.

**Second, the GCN baseline used here is a weak opponent by design, which
qualifies how much weight the headline comparison should carry.** Its only
node feature is a single normalized-degree scalar — no clustering
coefficient, no spectral or positional encoding, no multi-hop aggregated
statistics. A rendered image, by contrast, implicitly encodes a great deal of
structure a bare 1-D degree feature cannot: node position, local clutter,
overlap, and layout-derived proxies for centrality and community structure
that Graphviz's `sfdp` layout algorithm computes for free. The CNN is not
just "seeing structure a GNN can't" in any fundamental sense — it is
benefiting from a richer effective input representation than this
particular GNN was given. A GNN with comparably rich node features (degree,
clustering coefficient, k-hop neighborhood statistics, or learned positional
encodings) would very plausibly close some or all of this gap. This reframes
the finding from "CNNs beat GNNs at structural inference" to the narrower
and more defensible: *for a fixed, minimal feature budget, funneling a graph
through Graphviz's layout and a CNN recovers more usable structural signal
than handing a GNN only raw degree* — itself a non-obvious and useful result,
but not a general claim about the two paradigms.

The single most striking secondary finding is that **the GCN degraded more
than the CNN under out-of-distribution graph size** (Section 5.6, Part 1;
mean vertex error 237 vs. CNN's 179, n=6). This runs counter to the intuitive
expectation that message-passing models, being size-agnostic by
architecture, should generalize better to unseen graph sizes than a CNN
trained on fixed-resolution images where larger graphs are visually denser
and more cluttered. The sample is small (n=6) and this should not be treated
as a settled result, but it is consistent with a plausible mechanism: mean
pooling over a fixed-size embedding may lose more information as the number
of aggregated nodes grows far beyond the training distribution than a CNN's
convolutional features — which are computed over a fixed-resolution image
regardless of node count — lose as an image gets visually busier. This is
flagged as a direction for future work (Section 8) rather than a conclusion.

Finally, the density-bucket results (Sections 4 and 5.6) overturned a design-
time assumption: dense graphs, expected to be the hardest case due to visual
clutter and node overlap, are in fact the *easiest* bucket for vertex
prediction and only become harder than sparse graphs for edge prediction,
and only once size is also large. Out-of-distribution *size*, not density or
clutter, is the dominant failure mode (Section 5.5) — a genuinely
counter-intuitive finding that changes what future model or rendering
improvements should prioritize.

---

## 7. Limitations

**Rendering and layout sensitivity.** All reported results are specific to
Graphviz's `sfdp` layout algorithm, which produced 100% of the training,
test, and held-out images. The layout-sensitivity probe (Section 5.3) found
no significant effect from switching to Kamada-Kawai on the small probed
sample (p=0.499 vertex, p=0.898 edge), which is a reassuring but narrow
result — it does not establish robustness to layout algorithms with
substantially different visual conventions (e.g. hierarchical/Sugiyama-style
layouts, circular layouts, or force-directed layouts with different edge
bundling), none of which were tested.

**Dataset scale and synthetic-to-real distribution gap.** The synthetic
training set is 2,500 graphs capped at 100 nodes; no synthetic graphs above
this ceiling were generated or evaluated, so the model's behavior above 100
nodes is characterized only through the real-world PROTEINS subset (67
graphs, up to 620 nodes) and the 7-image live novel-image check, both of
which show substantial and increasing overcounting well beyond the ceiling
(up to ~4.7x on the most unfamiliar test image). This is a real
generalization gap, not merely an untested edge case.

**Image resolution, node-size tradeoff, and the confirmed shortcut.** Images
are rendered at a fixed 224×224 px with node radius scaling as
`max(15, 4000/n)`. This choice was flagged as a potential shortcut at design
time and is now empirically confirmed as a real, statistically significant
contributor to the CNN's vertex-count accuracy (Section 5.2, formalized in
Section 5.6 Part 4 via Wilcoxon signed-rank test: median vertex error
increase 1.0 → 11.0 under constant node size, p=3.6×10⁻⁶, 33/40 graphs
worse). Reported accuracy figures should be read as partly attributable to
this rendering convention and would likely be lower under a fixed-node-size
rendering scheme. The edge-count comparison shows the same direction of
effect via median and Wilcoxon test (p=0.0003) despite a misleading mean-based
signal driven by two outlier graphs — report medians and win/loss counts for
this specific comparison, not means.

**Out-of-distribution size ceiling.** The 100-node training ceiling is the
single dominant failure mode identified in this project (Section 5.5),
outweighing density or visual clutter, both in the offline failure taxonomy
and in the live novel-image check.

**Dense-bucket warning miscalibration.** The deployed app's current warning
banner treats "dense" as a single flat-risk bucket (MAE citation 0.43/1.12).
Section 5.6 (Part 4, item 2) shows this is safe — in fact the safest bucket
— for vertex prediction at every size tested; safe for edge prediction only
at small sizes; and high-risk for edge prediction at large sizes (3.3x worse
than sparse by 75–100 nodes). A size-aware warning split is recommended over
the current flat citation.

**Baseline-comparison fairness.** The GCN baseline's only node feature is
normalized degree — no richer structural encodings were provided (Section
6). The headline CNN-vs-GCN result should be read as specific to this
minimal-feature GCN, not as a general claim that image-based inference
outperforms graph-native inference. Separately, the GCN's out-of-distribution
-size degradation exceeding the CNN's (Section 5.6, n=6) is a small-sample
result and should be treated as a direction for further investigation, not a
settled finding about GNN generalization.

**Combinatorial-plausibility claim is qualified, not absolute.** One
exception was found in 30 live-app predictions (Section 5.6, Part 3): an
input combining multiple diagrams and prose text in a single frame produced
a predicted edge count exceeding the combinatorial maximum for the predicted
vertex count. This appears to be an input-distribution failure specific to
non-single-graph images rather than a breakdown of the property under normal
use, but the claim of "never violated" should not be repeated without this
caveat.

**Pixel-only baseline interpretation.** The ink-coverage regression (Section
5.6, Part 4, item 3) should be read as reassuring rather than incriminating:
it explains most edge-count variance on synthetic data but almost none on
real held-out data, meaning the CNN's real-world accuracy is not attributable
to simple pixel-counting. However, this comparison is itself only possible on
synthetic data where the shortcut is strong; synthetic-only evaluation may
still overstate how much the CNN has learned beyond low-level pixel
statistics, and results on synthetic data specifically should be read with
that caveat in mind.

**Ink-fraction measurement inconsistency.** Two different scripts compute
`ink_fraction` with two different background thresholds —
`evaluation/image_statistics.py` (`white_threshold=250`, used for Section
5.4's correlation table) and `evaluation/compute_structural_pixel_features.py`
(`bg_threshold=200`, used for the Section 5.6 pixel-only-baseline regression)
— and were never reconciled to one canonical definition. Both give the same
qualitative conclusion (ink coverage is a strong edge-count proxy on
synthetic data, a weak one on real data), but the exact correlation
magnitudes reported in Sections 5.4 and 5.6 are not directly comparable to
each other and should not be cited interchangeably.

**Training reproducibility.** Two otherwise-identical full training runs
(differing only in random GPU/cuDNN non-determinism on Colab) produced test
overall vertex MAE of 4.5 and 3.2 respectively. Reported figures are from
the later run; the gap between runs indicates the headline metrics carry
some run-to-run variance not captured by the fixed random seed alone.

---

## 8. Future Work

- **Layout-augmented training**, mixing multiple layout algorithms (Kamada-
  Kawai, hierarchical, circular) per graph during training, to directly
  test and reduce dependence on any single layout's visual conventions,
  extending the narrow Section 5.3 result to a training-time intervention.
- **A fixed-node-size rendering variant trained end-to-end**, to measure how
  much of the CNN's accuracy survives once the confirmed node-size shortcut
  (Section 5.2/5.6/7) is removed at the source rather than only probed
  post-hoc.
- **A richer GNN baseline** — clustering coefficient, degree-distribution
  moments, k-hop aggregated statistics, or learned positional encodings as
  node features — to make the Section 6 CNN-vs-GNN comparison fair on
  feature budget, not just architecture.
- **A larger, more diverse real-world out-of-distribution test set** beyond
  MUTAG/PROTEINS, to confirm or refute the small-sample (n=6) finding that
  the GCN baseline degrades more than the CNN under OOD graph size.
- **Explicit node-count conditioning or auxiliary tasks** (e.g. predicting
  node size distribution as an auxiliary output) to make the model's
  reliance on the node-size shortcut visible and controllable rather than
  implicit.
- **A size-aware warning banner** in the deployed app, replacing the current
  flat dense-bucket MAE citation with the size-binned risk profile
  established in Section 5.6, Part 4, item 2.
- **Extension to directed, weighted, or attributed graphs**, which this
  project deliberately excluded to isolate pure topology (vertex/edge count)
  as the target property.

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
| `topolens_free_analyses_results.csv` | Wilcoxon signed-rank test results (shortcut + layout probes) and warning-banner validation by size bin (Section 5.6, Part 4, items 1–2) |
| `structural_pixel_features.csv` | Per-graph ink fraction, diameter, and clustering coefficient merged onto `failure_case_categories.csv`, generated by `compute_structural_pixel_features.py` (source data for Part 4, items 3–4) |
| `topolens_pixel_structural_correlations.csv` | Pixel-only baseline regression stats and structural (diameter/clustering) correlation + collinearity + partial-correlation results (Section 5.6, Part 4, items 3–4) |

### Application (`app/`)

| File | Description |
|---|---|
| `app.py` | Streamlit inference app — **Cinematic Instrument design system** (dark film-grade palette, Fraunces display type); single multi-type file uploader (`png`, `jpg`, `jpeg`, `graphml`, `csv`, `txt`); multi-file select; CNN inference + hero stat display; Grad-CAM toggle (vertex-target + edge-target overlays); empirical validation panel (Ground Truth vs Prediction table + topological-property cards) active for both graph files and dataset images matched via `labels_processed.csv`; model limitations expander with live CSV values; live uploads and predictions logged to `render/novel_uploads/`. |
| `.streamlit/config.toml` | Theme: `backgroundColor = "#F5F5F5"`, `primaryColor = "#000000"`, `secondaryBackgroundColor = "#E5E5E5"`, `textColor = "#000000"`. |
