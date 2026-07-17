# Topolens — Progress Log

This is the narrative record of the project: what was done each day, how it
was done, and — critically — *why* each decision was made. It's meant to
read as the story of the project, useful for status updates, the final
report's methodology section, and for anyone (including future-me) trying
to understand the reasoning behind a choice months later.

For granular technical detail — exact bugs hit, exact numbers, what broke
and how it was fixed — see `report/log.md`. This document stays at the
reasoning level; that one stays at the debugging level.

---

## Day 1 — 13 Jul 2026 — Project Setup & Planning

### Goal

Go from "I've been assigned this project" to a fully scoped plan with a
working environment and a committed skeleton, so that Day 2 could start
writing pipeline code immediately instead of still making design decisions.

### What was done

- Installed Python and Git, initialized the GitHub repository
  (`github.com/whozahm3d/topolens`).
- Finalized the project scope and dataset strategy: a **primary synthetic
  dataset** generated via NetworkX (Erdős–Rényi, Barabási–Albert,
  Watts–Strogatz, random trees, dense/near-complete graphs), and a
  **secondary real-world validation set** from the TU Dortmund benchmark
  collection (MUTAG / PROTEINS).
- Created the full project folder structure (`data/`, `render/`,
  `models/`, `evaluation/`, `notebooks/`, `app/`, `report/`) and pushed it
  to GitHub.
- Wrote and committed `config.yaml` (centralized seed, dataset, rendering,
  and model parameters), `README.md`, and `requirements.txt`.
- Set up the Python environment (Anaconda base env), resolved package
  installation/version conflicts, and verified all core libraries import
  correctly: PyTorch 2.11.0 (CPU), torch-geometric 2.8.0, NetworkX,
  Streamlit, Gradio.

### How

Plan-first: the scope, dataset strategy, and folder layout were all decided
*before* any pipeline code was written, so that Day 2 onward could be pure
implementation against a fixed plan rather than design-while-coding.
`config.yaml` was written as the intended single source of truth for every
tunable parameter (seed, node ranges, generator params, image size, model
architecture, hyperparameters) specifically so that no script would
hardcode a value that later needs to change in five places at once.

### Why

**Why a CNN on rendered images at all, instead of a GNN?** This is the
core research question the whole project is built around, and it was
locked in on Day 1 because it shapes every downstream decision. GNNs
consume graph structure directly — adjacency, node/edge features — via
message passing, which is the "native" way to reason about a graph. A CNN
looking at a rendered image gets none of that; it only sees pixels
produced by a layout algorithm's spatial arrangement of dots and lines.
The interesting question is exactly *how much* structural information
survives that projection from graph → 2D layout → pixels, and whether a
generic vision model can recover it. That's not a weaker or "lesser"
approach to apologize for — it's a deliberate contrast to the graph-native
literature, and it has a real-world analogue: plenty of situations only
ever produce a graph as an image (a diagram, a scanned figure, a
visualization with no underlying data file), and the question of whether
useful structural counts can still be recovered from that image alone is
genuinely useful to answer.

**Why a dual dataset (synthetic + real) instead of just one?**

- *Synthetic (NetworkX generators)* gives exact, unambiguous ground truth
  for every graph (no labeling noise) and lets the dataset be built with
  deliberately balanced coverage across structural regimes — sparse vs.
  dense, tree-like vs. clustered vs. random, tiny vs. large — which a
  regression model needs to generalize rather than overfit to one
  distribution. The supervisor explicitly greenlit a self-generated
  dataset as long as the ground truth is accurate, which synthetic
  generation trivially satisfies.
- *Real-world (MUTAG/PROTEINS)* exists to test whether patterns learned on
  synthetic, randomly-generated graphs transfer to organic graphs with
  their own structural regularities (molecular valence constraints for
  MUTAG, protein contact-map structure for PROTEINS) that generic random
  graph models don't reproduce. It's configured as a held-out
  generalization test — never used in training — specifically so it stays
  a clean measure of external validity rather than becoming part of the
  training distribution.

**Why centralize everything in `config.yaml` upfront?** Because the
project is being framed as research-style (methodology, reproducibility,
a Limitations section), not a one-off script. Fixed seeds and
non-hardcoded parameters are what make results reproducible and what let
later phases (model architecture, hyperparameters) change without
touching the data-generation code.

### Issues

None blocking — package/version resolution took some trial and error but
didn't stall the timeline.

### Next

Move from planning into Phase 1 implementation: generate the synthetic
graphs, load the real datasets, build the rendering pipeline.

---

## Day 2 — 14 Jul 2026 — Dataset Generation & Rendering (Phase 1)

### Goal

Produce a complete, validated, labeled image dataset — the input every
later phase depends on — and confirm it's correct before building
anything on top of it.

### What was done

- Generated the full synthetic dataset: 2,500 graphs across 5 generator
  families (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, random trees,
  dense) × 4 node-count tiers, 125 graphs per cell.
- Loaded and converted MUTAG (188 graphs) and PROTEINS (1,113 graphs) via
  PyTorch Geometric, stripped down to plain topology (no node/edge
  features — Topolens only cares about structure).
- Rendered all 3,801 graphs to 224×224 PNGs via the NetworkX/Graphviz/
  matplotlib pipeline, with a Graphviz sfdp → pydot → NetworkX
  spring-layout fallback chain.
- Curated and exported ~24 diverse graphs (spanning generators, tiers, and
  both real datasets) to `data/gephi_demo/` for manual, presentation-
  quality rendering in Gephi.
- Ran `validate_dataset.py` — confirmed every image exists, is the correct
  resolution, and matches its GraphML node/edge counts. **PASS.**
- Fixed bugs in and ran the dataset sanity-check notebook
  (`01_dataset_sanity_check.ipynb`): node/edge distribution histograms,
  per-generator breakdowns, tier balance chart, and a 12-image sample
  render grid — saved to `report/figures/`.
- Pushed the full dataset and code to GitHub in batched commits (Antigravity
  IDE hit its weekly quota mid-environment-setup, so the pipeline was run
  and debugged manually via terminal instead of through the agentic IDE).

### How

Each graph — synthetic or real — gets a deterministic seed derived from a
SHA-256 hash of its own `graph_id`, so any single graph or the entire
dataset can be regenerated byte-for-byte identically later. Rendering is
fully scripted (`render_graphs.py`) rather than done by hand, so it can run
unattended across thousands of graphs and stay reproducible. A validation
pass (`validate_dataset.py`) runs as a separate, cheap step *before* any
training happens, specifically to catch integrity bugs — missing images,
wrong resolution, GraphML/CSV mismatches — while they're still cheap to
diagnose, rather than discovering them as a mysterious training-time
failure.

### Why

**Why these five generator families specifically?** They're not arbitrary
— together they cover the major structural regimes graphs can take: ER for
uniformly random connectivity, BA for scale-free/preferential-attachment
structure (hub-and-spoke), WS for small-world clustering, random trees for
the sparsest possible connected structure (exactly n−1 edges, no cycles),
and dense ER for the near-complete end of the spectrum. A CNN that only
ever saw one of these would tell us nothing about whether visual inference
generalizes across topology types — the whole point is breadth.

**Why a hybrid rendering approach (scripted NetworkX/Graphviz for the bulk,
Gephi for a curated subset) instead of just one tool?** Bulk generation of
3,800+ images needs to be scriptable, deterministic, and unattended —
Gephi is an interactive desktop tool, not something you bulk-automate
3,801 times. But Gephi produces visually polished, presentation-grade
layouts that are worth having for a handful of report figures. So the
split plays to each tool's strength: NetworkX/Graphviz/matplotlib for
everything that needs to be reproducible at scale, Gephi for the ~24
samples that need to look good in a write-up. This also happens to align
with what the supervisor was implicitly pointing at when they referenced
existing graph-to-image tooling — this is that ecosystem (Gephi/Graphviz/
NetworkX), used deliberately rather than reinvented.

**Why the Graphviz → pydot → spring-layout fallback chain?** Graphviz's
`sfdp` algorithm generally produces cleaner, less-cluttered layouts than
NetworkX's default spring layout, especially as graphs get larger — fewer
overlapping nodes, fewer crossing edges. It was the intended primary
layout. The fallback chain exists purely for robustness in case Graphviz
isn't installed on a given machine. That's exactly what happened here —
Graphviz binaries weren't on PATH, so all 3,801 images fell back to
NetworkX spring layout. This is flagged as an open item (see below), not
silently accepted.

**Why the `node_size = max(15, 4000/n)` scaling in the render script?**
Without inverse scaling by node count, a 600-node graph would render as an
illegible smear of overlapping circles, and a 5-node graph would render as
near-invisible dots. Scaling node size down as `n` grows keeps every image
legible across the full 4–620 node range. The tradeoff — flagged directly
in the render code — is that this makes visual node *density* a near-direct
function of `n`, which means the CNN could in principle learn to estimate
vertex count largely from "how much of the image is covered by dots"
rather than from reading actual structural cues like branching or
clustering. That's a shortcut-learning risk worth explicitly testing for
in the Phase 3 analysis (e.g., checking whether edge-count accuracy holds
up on graphs with similar `n` but very different densities).

**Why a fifth "xlarge" tier that isn't in the original 4-tier config?**
The synthetic generator is capped at 100 nodes by design (`config.py`'s
`NODE_TIERS`), but PROTEINS contains real graphs up to 620 nodes. Rather
than silently lumping those into "large" or excluding them, they're
tagged as a distinct "xlarge" tier — used only by the real-data loader —
so Phase 2/3 analysis can explicitly see and reason about out-of-
distribution performance instead of it being invisible in the tier
breakdown.

**Why validate before training, and why a dedicated sanity-check notebook?**
A validation pass is cheap (seconds) compared to a wasted training run
(hours), so it runs as its own gate. The notebook exists for a different
reason: visual/statistical confirmation that the dataset actually looks
right — balanced tiers, sane distributions, legible renders — and to
produce the exact figures (`report/figures/`) that the final report's
methodology section will need. Both are about catching problems, and
generating evidence, before the expensive part of the project starts.

### Issues

- **No layout diversity yet.** All 3,801 images used the NetworkX
  spring-layout fallback rather than the intended Graphviz sfdp, because
  Graphviz wasn't available on this machine. Options going into Phase 2:
  fix the Graphviz install and re-render for layout diversity, or treat
  single-layout as a controlled variable and write it up as an explicit
  Limitation.
- **PROTEINS distribution skew.** 67 "xlarge" graphs (>100 nodes, up to
  620) sit well outside the synthetic training range. Needs a decision:
  exclude from training, or keep in as a deliberate generalization test
  (config already marks TU-Dortmund data as held-out/never-trained-on,
  which leans toward the latter).
- **`config.py` vs `config.yaml`.** Two config files exist with
  overlapping but non-identical settings. All Phase 1 scripts import
  `config.py` (the simpler one); `config.yaml` — despite being described
  as the single source of truth — currently isn't read by any code, and
  it's the only place model/training hyperparameters live. Needs
  resolving before Phase 2 training code is written.

### Next

Phase 2: CNN regression model + GNN/graph-statistic baseline, trained on
the synthetic split and evaluated (including held-out) against MUTAG/
PROTEINS.

---

## Day 3 — 15 Jul 2026 — Phase 2: CNN Training + GNN Baseline

### Goal

Unblock the layout-diversity gap flagged at the end of Day 2, and get
Phase 2's full pipeline (splits, CNN, GNN baseline, graph-statistic
baseline, evaluation) scaffolded and runnable before any GPU training
happens.

### What was done

- Installed Graphviz into the conda environment and confirmed `sfdp` is
  reachable, unblocking the render pipeline from the NetworkX spring-layout
  fallback.
- Deferred the actual full re-render (CPU-bound, slow on this machine) —
  scaffolding Phase 2 code first didn't depend on it.
- Simplified `render_graphs.py` to emit a single canonical labels artifact.
- Added the full Phase 2 script set (splits, dataset loader, CNN model,
  GNN baseline, graph-statistic baseline, evaluate, error-analysis) and a
  Phase 2 notebook doubling as an interactive results dashboard.

### How

Scaffolded CNN and GNN code side by side rather than building the CNN
pipeline first and bolting the GNN on afterward, since the core research
question is a head-to-head comparison — asymmetric effort between the two
would undermine that from the start.

### Why

Fixed the Graphviz gap directly rather than accepting single-layout
spring rendering as a permanent Limitation, since the root cause was a
local environment gap, not a design constraint — worth the fix instead of
writing around it.

### Issues

Full dataset re-render still pending — CPU-bound and not yet run at end
of day. Carried into Day 4 as a dependency check (confirmed unnecessary
once Colab GPU training started, since Graphviz output is deterministic
and the committed dataset already matched).

### Next

Move training onto Colab GPU: smoke-test first, then full training run.

---

## Day 4 — 16 Jul 2026 — Phase 2: GPU Training, GCN Bug Fix, Baseline Correction

### Goal

Get real, trustworthy checkpoints and evaluation numbers across all three
models (CNN, GCN, graph-statistic) — replacing 2-epoch smoke-test
placeholders with full training, and catching any bugs hiding behind
untrustworthy early metrics.

### What was done

- Ran full 30-epoch training for both models on Colab (T4 GPU).
- Found and fixed a GCN pooling bug (`global_add_pool` → `global_mean_pool`,
  plus per-graph degree normalization) that was causing held-out MAE to
  explode into the thousands on out-of-distribution real graphs.
- Found and fixed a second, separate bug: the graph-statistic "baseline"
  was reading ground truth directly rather than estimating anything,
  giving it a meaningless 0.0 MAE everywhere. Replaced it with a
  train-derived density heuristic.
- Reran full training and evaluation after both fixes, pulled fresh
  checkpoints locally, and committed everything.

### How

Both bugs were caught the same way: by noticing a metric that was
physically implausible — either too perfect (0.0 MAE baseline) or too
broken (thousands-scale MAE on the GCN) to be real — and tracing backward
from there, rather than assuming the numbers were correct and moving on.

### Why

Chose `global_mean_pool` over clipping or capping the GCN's outputs, since
the root cause was a genuine architectural mismatch (sum-pooling
implicitly encodes graph size, and that assumption breaks hard once you're
off-distribution), not a numerical-stability issue that clipping would
have papered over. Similarly, chose to fix the baseline's leakage rather
than quietly drop it from the report — a fair, if weak, baseline is worth
more to the research narrative than a meaningless oracle number.

### Issues

The density-based replacement baseline is itself weak on sparse/large
graphs, since density isn't constant across generators or graph sizes.
Decided to report this as-is (a real, defensible finding) rather than
spend remaining time engineering a better heuristic (e.g. a linear-in-`n`
fit per generator) — the deadline doesn't leave slack for further baseline
work, and the current result already supports the core narrative.

### Next

Phase 3: model interpretation / failure-case analysis, and drafting the
final report (Methodology, Results, Discussion, Limitations).

---

## Day 5 — 17 Jul 2026 — Phase 3: Interpretation, Analysis & App

### Goal

Turn the trained checkpoints into insight: understand *what* the CNN
actually learned, characterize exactly *where* it fails and why, build the
Streamlit demo app the project requires as a deliverable, and lay the
skeleton of the final report — all without touching the training artifacts
or rerunning a single training step.

### What was done

**Batch 1 — Grad-CAM & shortcut/layout probes**

- Refactored `render_graphs.py` to accept `node_size_fn` and
  `layout_override` arguments (with Kamada-Kawai fallback), enabling
  controlled probe variants to be rendered without changing the canonical
  dataset.
- Implemented `GradCAM` for the custom CNN (`models/gradcam.py`), targeting
  `features[3]` — the last spatial convolutional layer before global pooling,
  where the network's spatial attention is most interpretable.
- Ran `evaluation/interpretation.py`: produced 60 CAM overlays (30 test
  graphs × vertex + edge targets) and two composite grids saved to
  `report/figures/gradcam/`.
- Rendered two probe variant sets (40 constant-node-size images, 40
  Kamada-Kawai alt-layout images) via `render/render_probe_variants.py` and
  ran `evaluation/shortcut_probe.py` to get predictions on all three
  variants per graph.
- Headline shortcut-probe result: vertex MAE nearly **tripled** when node
  rendering size was held constant (3.43 → 10.48), confirming the CNN uses
  apparent node size as a strong proxy for vertex counting. Edge MAE
  *improved* under both probe conditions, suggesting the same size cue that
  helps vertex estimation slightly hurts edge estimation on original renders.
  Layout change (sfdp → Kamada-Kawai) had minimal vertex MAE impact (3.43 →
  3.08), ruling out heavy overfit to sfdp's specific spatial layout.

**Batch 2 — Ink-coverage analysis & failure taxonomy**

- Implemented `evaluation/image_statistics.py` (ink fraction, connected-
  component label stats) and `evaluation/ink_coverage_analysis.py`, running
  over 375 test + 1,301 held-out images. Saved correlation tables and
  scatter plots to `report/figures/`.
- Implemented `evaluation/failure_taxonomy.py`: categorized every test and
  held-out prediction into `out_of_distribution_size`, `high_density_clutter`,
  or `in_distribution_normal` buckets, then computed per-category MAE.
- Headline ink-coverage finding: ink fraction correlates strongly with
  predicted edge count (Pearson *r* = 0.845 on test), providing a pixel-
  level explanation for why the CNN performs well in-distribution — the total
  line coverage in the image is a reliable proxy for edge count when graph
  size is controlled. That correlation collapses on held-out graphs (*r* =
  0.465 for edges) as graph sizes leave the training distribution.
- Headline failure-taxonomy finding: the one failure category that dominates
  is out-of-distribution size (`num_nodes > 100`, PROTEINS graphs): held-out
  OOD vertex MAE 96.6 vs. in-distribution 6.7. High-density graphs are
  *not* a failure mode — the CNN handles them better than average.
- Created the Phase 3 results notebook
  (`notebooks/03_model_interpretation_and_vulnerability_probes.ipynb`),
  covering Batch 1 and Batch 2 outputs inline.

**Batch 3 — Streamlit app, report skeleton & UI polish**

- Created `app/app.py` with a full Warm Editorial design system (Playfair
  Display / Source Sans 3 typography, amber primary color, cream background,
  staggered `fadeInUp` animations). Two input modes: PNG/JPG image upload
  and `.graphml`/`.csv`/`.txt` graph file upload; Grad-CAM overlay toggle
  for both targets.
- The app's "Model limitations" expander reads real numbers at startup from
  the evaluation CSVs rather than hardcoding them, so the displayed metrics
  stay in sync with any future evaluation run automatically.
- Fixed two bugs found during smoke testing: a checkpoint-loading mismatch
  (the checkpoint is a wrapped dict, not a raw `state_dict`) and a Streamlit
  1.59 deprecation warning in the image display call.
- Iterated the design to a Neo-Brutalist system (Syne + Plus Jakarta Sans,
  flat black borders, offset box-shadows), multi-file upload with a
  selectbox switcher, and ground-truth validation for dataset images (looks
  up `labels_processed.csv` by filename stem and renders a prediction vs.
  truth table when the image is from the known dataset).
- Created `report/FINAL_REPORT.md`: header-only skeleton matching the exact
  Task I submission structure, with a literal file-index appendix pointing at
  every figure and result CSV produced across all three phases.

### How

Phase 3 was run as three sequenced batches precisely so that each batch had
a clearly defined verification step before the next one started: confirm
no training artifact was modified, confirm all prior result files are
byte-unchanged, then proceed. This mattered because interpretation tools
(Grad-CAM, probe scripts) load the checkpoint read-only — any accidental
side-effect on `data/images/` or `data/splits/` would silently invalidate
earlier phase results, so the explicit checks weren't ceremony, they were
the safety net. The Streamlit app deliberately imports all ML logic from the
existing module tree rather than reimplementing anything inline, so the app
is a thin UI layer over already-tested code rather than a second, untested
copy of the inference pipeline.

### Why

**Why Grad-CAM on `features[3]` specifically?** `features[3]` is the last
spatial block before the global average pooling that collapses the feature
map to a vector. After that point, spatial information is gone and CAM
can't be computed in the traditional sense. `features[3]` is therefore the
last layer where "where in the image did this prediction come from" is still
a meaningful question — which is exactly what the interpretation step needed
to answer.

**Why run probe variants instead of just examining Grad-CAM?** Grad-CAM
shows *where* the network looks; it doesn't directly answer *what visual
feature it is responding to*. The shortcut probes answer that by controlled
intervention: hold the graph topology constant and vary one rendering
parameter (node size, layout algorithm) at a time, then measure whether
predictions change. If they do, that rendering parameter is a confound the
model has learned to rely on. CAM and probes are complementary — neither
alone is sufficient.

**Why report the density-based baseline's failure as a finding rather than
engineering a better heuristic?** The baseline's systematic failure on
sparse/large graphs isn't a problem to be solved — it's a data point in the
argument for learned approaches. A density heuristic that's calibrated on
one size regime and applied to another breaks because graph density is not
scale-invariant; the CNN and GCN, trained end-to-end, implicitly learn
that. Replacing the baseline with a more sophisticated formula would
produce a more competitive number at the cost of obscuring exactly the
point the comparison is meant to make.

**Why a Neo-Brutalist redesign of the Streamlit app?** The initial Warm
Editorial design was aesthetically coherent but visually soft in a way
that doesn't read well as a technical demo. Neo-Brutalism (flat offset
shadows, heavy borders, high-contrast type hierarchy) communicates
structure and precision — qualities that are appropriate for a tool whose
output is numerical predictions — and it differentiates the app from the
default Streamlit aesthetic in a way that signals deliberate design intent
rather than the framework's out-of-the-box defaults.

### Issues

- `report/FINAL_REPORT.md` is a skeleton only — all analytical sections
  (`Methodology`, `Results`, `Discussion`, `Limitations`) are headers with
  no content yet. Writing the report is the remaining work.
- The Streamlit app's Grad-CAM overlay currently requires the CNN checkpoint
  to be present at `models/checkpoints/cnn_best.pt` locally; it will not
  load on a fresh clone without the checkpoint (which is gitignored). This
  is a known deployment gap — documented in the app's sidebar but not
  resolved.

### Next

Write the final report: fill in the `FINAL_REPORT.md` skeleton with the
Methodology, Results, Discussion, and Limitations sections, drawing on the
data already in `evaluation/results/` and `report/figures/`.
