
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

# 2026-07-17 — Phase 3 Batch 2: ink-coverage analysis, failure taxonomy, notebook

- Added `scipy>=1.10` to `requirements.txt` (connected-component labeling).
- Implemented `evaluation/image_statistics.py`: `compute_ink_fraction()` and `compute_component_stats()` (scipy.ndimage.label, 4-connectivity).
- Implemented `evaluation/ink_coverage_analysis.py`: ran on 375 test + 1301 held_out images; saved `evaluation/results/ink_coverage_correlations.csv`, `report/figures/ink_coverage_vs_node_count.png`, `report/figures/component_size_vs_node_count.png`.
- Implemented `evaluation/failure_taxonomy.py`: derived `max_train_n = 100` from `data/splits/train.csv`; saved `failure_case_categories.csv`, `failure_case_summary.csv`, `error_vs_num_nodes.png`, `error_vs_density.png`, `worst_case_image_grid.png`.
- Created `notebooks/03_phase3_interpretation.ipynb`: covers Batch 1 + Batch 2 results inline. Bug fixed: notebook `os.getcwd()` returns the notebooks/ subdirectory, not the project root — added `os.chdir(project_root)` to the setup cell so all config-relative paths resolve correctly.
- **Headline numbers — ink_coverage_correlations.csv (Pearson r):**

  | split | metric_x | metric_y | r |
  |---|---|---|---|
  | test | ink_fraction | num_edges | **0.824** |
  | test | ink_fraction | pred_num_edges | **0.845** |
  | test | mean_component_area | pred_num_vertices | **0.627** |
  | test | num_components | num_nodes | -0.240 |
  | held_out | ink_fraction | pred_num_edges | **0.465** |
  | held_out | ink_fraction | num_nodes | 0.070 |
  | held_out | num_components | num_nodes | 0.308 |

- **Headline numbers — failure_case_summary.csv:**

  | split | category | mean_vertex_mae | mean_edge_mae | n |
  |---|---|---|---|---|
  | held_out | out_of_distribution_size | **96.64** | **203.60** | 67 |
  | held_out | in_distribution_normal | 6.67 | 20.79 | 1090 |
  | held_out | high_density_clutter | 0.43 | 1.13 | 144 |
  | test | in_distribution_normal | 4.00 | 14.42 | 265 |
  | test | high_density_clutter | 1.27 | 51.82 | 110 |

- Verification: `data/images/`, `data/splits/`, `data/processed/labels_processed.csv`, and all Batch 1 outputs (probe_predictions.csv, probe_summary.csv, gradcam/ images) confirmed byte-unchanged after all Batch 2 scripts ran. No retraining occurred.

# 2026-07-17 — Correction: failure_taxonomy threshold + log trim

- Verified `failure_taxonomy.py` was already using `topolens_utils.density_bucket(density) == "dense"` (≥0.40); re-ran script to confirm outputs match that threshold. `failure_case_summary.csv` numbers unchanged.
- Trimmed interpretive narrative from Phase 3 Batch 2 entry (ink-coverage and failure-taxonomy bullet blocks removed; tables and raw numbers retained).

# 2026-07-17 — Phase 3 Batch 3: Streamlit app, report skeleton

- Created `app/.streamlit/config.toml`: light theme, `primaryColor = "#C4793A"` (amber),
  `backgroundColor = "#F4EFE6"` (cream), `secondaryBackgroundColor = "#EAE4D8"`.
- Created `app/app.py`: Warm Editorial design system ("The Psychology Behind Type Choices" theme). Imports:
  `models/cnn_model.py::CustomCNNRegressor`, `models/dataset.py::build_image_transform`
  - `compute_normalization_stats`, `models/gradcam.py::GradCAM` + `overlay_cam`,
  `render/render_graphs.py::render_graph`. No ML logic reimplemented.
  - `@st.cache_resource` loads checkpoint + normalize stats + GradCAM instance.
  - Two input modes: (1) PNG/JPG image upload → preprocess → predict;
    (2) .graphml/.csv/.txt upload → networkx parse → server-side render → predict.
  - Hero stat display: Playfair Display italic numbers in amber, no metric card chrome.
  - Grad-CAM checkbox: computes both vertex-target and edge-target overlays via
    `GradCAM.generate()` / `overlay_cam()`, displayed in letterbox frames.
  - Model limitations expander reads real numbers at startup from
    `evaluation/results/failure_case_summary.csv` (OOD size MAE) and
    `evaluation/results/probe_summary.csv` (node-size shortcut delta) —
    not hardcoded from prompt.
  - CSS injected via `st.markdown`: Google Fonts (Playfair Display, Source Sans 3,
    JetBrains Mono), clean layout, warm drop-shadows, double-rule header border,
    staggered `fadeInUp` keyframes (5 delay tiers).
- Created `report/FINAL_REPORT.md`: header-only skeleton, exact Task I structure,
  zero analytical content. Appendix section is a literal file index of
  `report/figures/*.png` and `evaluation/results/*.csv`.
- Smoke test performed (`streamlit run app/app.py`, localhost:8501). Two bugs
  found and fixed:
  1. **Checkpoint loading bug**: `train_cnn.py` saves a wrapped dict with keys
     `state_dict`, `normalize_stats`, `epoch`, `best_val_loss`, `architecture`,
     `input_size`, `config` — not a raw `state_dict`. Fixed `load_model_and_gradcam()`
     to unpack `checkpoint["state_dict"]` and reuse `checkpoint["normalize_stats"]`
     (exact training stats, no recompute needed). Changed `weights_only=True` →
     `weights_only=False` as required for a dict containing non-tensor Python objects.
  2. **Cosmetic / deprecation**: removed `page_icon="🔭"` (set to `None`); replaced
     `st.image(buf, use_container_width=False, width=max_width)` with
     `st.image(buf, width=max_width)` to silence Streamlit 1.59 deprecation warnings.
- Post-fix: app loads checkpoint cleanly, predictions display correctly for both
  input modes, Grad-CAM toggle functional, limitations expander shows real CSV
  values. Import-time check also re-confirmed: `app.app: import OK`.
- **UI/UX redesign — multi-file input, Neo-Brutalist design, ground-truth validation**
  (iterated within same session, no separate batch boundary):
  - **Multi-file input**: Replaced two separate upload toggles with a single
    `st.file_uploader` accepting `["png", "jpg", "jpeg", "graphml", "csv", "txt"]`
    simultaneously (`accept_multiple_files=True`). Active file selected via
    `st.selectbox` when more than one file is uploaded. File bytes read via
    `.getvalue()` (not `.read()`) to survive Streamlit reruns.
  - **Ground-truth validation for image uploads**: Added `load_ground_truth_for_file()`
    helper. Filename stem looked up against `data/processed/labels_processed.csv`
    (column `graph_id`); if matched, `num_nodes`/`num_edges` and corresponding GraphML
    are retrieved. Validation table (Ground Truth vs CNN Prediction, Absolute Error,
    Accuracy %) and topological-properties cards now render for dataset images, not
    just graph file uploads.
  - **Neo-Brutalist design system** (full `EDITORIAL_CSS` rewrite): Syne 800 +
    Plus Jakarta Sans body + JetBrains Mono; `#F5F5F5` background; `3px solid #000`
    borders with `4px 4px 0px #000` flat offset shadows on expanders and tables;
    sidebar `#1E1F26` charcoal; per-card accent colors (density mustard `#C08A00`,
    components violet `#9C47FF`, clustering sky `#00B5FF`). `config.toml` updated
    to `primaryColor = "#000000"`, `backgroundColor = "#F5F5F5"`.
  - **Icon/widget bug fixes**: sidebar collapse arrow and file-uploader button both
    rendered incorrectly because `[data-testid="stSidebar"] span` was overriding
    the Material Icons ligature font. Fixed by removing `span` from the sidebar
    CSS selector. Section-label paragraphs switched from all-caps Syne to Plus
    Jakarta Sans body styling.
- Verified (by listing): `data/images/`, `data/splits/`,
  `data/processed/labels_processed.csv`, and all Batch 1/2/3 output files are
  untouched. No retraining occurred.

# 2026-07-18 — Correction: scope of "untouched" claims; progress_log.md Day 5 addition

- The "untouched" and "confirmed byte-unchanged" claims in the Batch 1, Batch 2,
  and Batch 3 entries above refer specifically to: `data/images/`, `data/splits/`,
  `data/processed/labels_processed.csv`, and the output files from earlier batches
  (probe CSVs, gradcam images, ink-coverage and failure-taxonomy results). Those
  claims do not cover `progress_log.md`.

- `progress_log.md` has a "Day 5 — 17 Jul 2026" section covering Phase 3 Batch 1,
  2, and 3. This section was present in the file as of 2026-07-18. Its authorship
  is unconfirmed: it was not within the explicitly scoped deliverables for any of
  the three batches, but it is consistent with agentic tooling having written it
  during the Batch 3 session. No claim is made either way.

- This note is appended to make the log an accurate record: the Batch 1/2/3
  entries' "untouched" language was never false for the files it named, but it
  was silent on `progress_log.md`, and that silence is now explicit.

# 2026-07-18 — Batch 4a: app navigation shell, Home/Results/Models sections

- Re-structured `app/app.py` to add a multi-section navigation shell (Home, Predict, Results, Research Insights, Models).
- Relocated the Predict view into a dedicated function; shifted the file uploader and active file selection to a side-by-side layout (left: upload, right: enlarged preview image display).
- Implemented the Home landing page, including live dataset composition crosstab (generator vs size-tier) from `labels_processed.csv`, headline MAE metrics from `summary_comparison.csv`, rendering pipeline notes, and reproducibility info.
- Implemented the Results view dashboard showing baseline metric tables, breakdown tabs, scatter plots, and dataset distributions. Added copy highlighting Heuristic Vertex MAE structural triviality.
- Implemented the Models view displaying dynamic module layers and parameter counts for CNN and GCN models, convergence stats, hyperparameters, note on mean-pooling bug fix, and loss log line charts.
- Added placeholders for Research Insights view.
- Verified all sections locally via user-driven Streamlit run smoke tests; cleared syntax escape warning messages and container width deprecation warnings. No models were retrained and protected dataset/image files remain unchanged.

# 2026-07-18 — Batch 4b: Research Insights section, live multi-model comparison, context-aware warnings, cross-links

- Implemented `render_research_insights_section()` in `app/app.py`: four subsections
  (Grad-CAM, shortcut-learning probe, layout-sensitivity probe, ink-coverage/failure-
  taxonomy), each reading `probe_summary.csv` / `ink_coverage_correlations.csv` /
  `failure_case_summary.csv` live via `safe_read_csv()`, with independent
  `render_data_unavailable()` fallback per subsection on missing/malformed data.
- Implemented live multi-model comparison in `render_predict_section()`: for
  `.graphml`/`.csv`/`.txt` uploads, runs CNN + GCN + graph-statistic baseline side
  by side; image-only uploads show CNN prediction plus an explanatory line on why
  GCN/baseline aren't available. Added `predict_graph_counts()` single-graph
  inference helpers to `models/gnn_baseline.py` and
  `models/graph_statistic_baseline.py` (previously batch-only); no inference logic
  reimplemented in `app.py`.
- Implemented `render_prediction_warning_banner()`: fires on `num_nodes >
  max_train_n` (derived from `data/splits/train.csv`) and/or
  `density_bucket() == "dense"`, citing real MAE from `failure_case_summary.csv`
  (`high_density_clutter` category present directly, no fallback needed). Image-only
  uploads derive density from the CNN's own predicted counts, labeled explicitly as
  predicted, not ground truth. Both triggers cited together when both fire.
- Added cross-link text ("See Research Insights → Failure-Case Taxonomy...")
  inline with the warning banner (Task Q).
- Verified live via `real_PROTEINS_0006.graphml` (336 nodes / 816 edges, exceeds
  `max_train_n=100`, density 0.0145 — sparse bucket): CNN pred 73/81, GCN pred
  29/74, graph-statistic baseline pred 336/15448; size-warning banner correctly
  cited vertex MAE 96.64 / edge MAE 203.60 alone (density condition not met).
  Research Insights subsections confirmed rendering live-computed values matching
  source CSVs exactly (e.g. shortcut-learning delta +7.050 / +205.8%).
- Dual-trigger path (size + density warnings firing together) verified live via
  synthetic Erdős–Rényi graph `data/images_probe/dense_test_150n.graphml`
  (generated by `evaluation/generate_dual_trigger_testcase.py`, seed=42, p=0.5):
  150 nodes, 5591 edges, ground-truth density 0.5003 — falls in dense bucket per
  `density_bucket()` (threshold ≥0.40). Banner correctly cited both size-based MAE
  (vertex 96.64, edge 203.60, for nodes above `max_train_n=100`) and density-based
  MAE (vertex 0.43, edge 1.12, for dense-bucket graphs) together, with cross-link
  to Research Insights → Failure-Case Taxonomy rendered below both citations. CNN
  predictions (96 vertices / 1,971 edges) far understated true structure
  (150 / 5,591), consistent with the warning's stated purpose of flagging
  out-of-distribution inputs.
- No retraining occurred. `data/images/`, `data/splits/`,
  `data/processed/labels_processed.csv`, `evaluation/results/*.csv`, and
  `report/figures/*.png` confirmed unchanged.

# 2026-07-19 — Live-app spot check (Part 1): 23-graph test round

- Ran the 23-graph live-app spot check per `topolens_testing_and_research_extension_plan.md`
  Part 1: each graph uploaded via the Predict page as a `.graphml` file (not image),
  exercising the full Batch 4b path (CNN + GCN + graph-statistic baseline, warning
  banner, ground-truth lookup) in one pass.
- Data quality issue found and corrected: test #3 (`syn_watts_strogatz_small_0031.graphml`)
  initially returned ground truth V=72/E=272 instead of the expected V=13/E=39 — wrong
  file had been uploaded. Retested with the correct file; ground truth now matches the
  plan (V=13, E=39, density 0.5000), dense-bucket warning fired as expected. All other
  22/23 rows' app-reported ground truth matched the plan's stated true V/E exactly on
  first pass.
- Full results (CNN/GCN/baseline predictions, absolute errors, warning-fired status)
  compiled into `evaluation/results/spotcheck_results.csv`.
- Headline findings:
  - Clean in-distribution graphs (11 of 23, no warning fired): CNN vertex MAE ≈5.6,
    edge MAE ≈17.7 — consistent with held-out `failure_case_summary.csv` in-distribution
    figures (6.67 / 20.79).
  - Density + size interaction: dense-flagged graphs that are also large (99-node dense
    graph, edge error 1084) fail far worse than dense-but-small/medium graphs (edge
    errors 1–32), despite both citing the same dense-bucket MAE (1.12) in the warning.
    Candidate refinement for Failure-Case Taxonomy / Limitations.
  - 99-vs-101-node boundary pair (#20/#21): vertex error close (23 vs 30), edge error
    diverges more (56 vs 119) — partially supports "soft degradation not a cliff" for
    vertex count, less so for edge count.
  - GCN degraded more than CNN on out-of-distribution-size graphs in this sample (mean
    vertex error 237 vs CNN 179; mean edge error 428 vs CNN 336, n=6) — counter to the
    expectation that the graph-native baseline generalizes better on size than the
    image-based CNN. Worth addressing directly in the report, not assumed away.
  - Far-OOD predictions (504/620-node graphs) stayed combinatorially plausible for CNN
    and GCN, consistent with the Part 3 finding across the full 1,676-prediction set.
    Graph-statistic baseline showed the opposite failure mode: exact vertices, edges
    overestimated ~30–40x at this scale.

# 2026-07-19 — Novel-image sanity check (Part 2): 7 images, no ground truth

- Uploaded 7 non-dataset images via the Predict page (image-upload path, CNN-only
  inference) per Part 2 of the testing plan: a genuine scanned academic figure, a
  clean-vector unfamiliar-layout diagram, a digitally hand-drawn directed graph
  (rectangle nodes), a cartoon-icon social network, a weighted network with numeric
  edge labels, a near-training-style circular-node render, and — tested first,
  standalone — a photographed notebook page containing two hand-drawn diagrams plus
  prose.
- Full results (predicted V/E, estimated density, warning-fired status,
  combinatorial-plausibility check) compiled into
  `evaluation/results/topolens_novelimage_results.csv`.
- Finding: predicted vertex/edge counts scale with visual unfamiliarity relative to
  training-render style, not with hand-drawn-ness per se. The two images closest in
  visual grammar to training renders (clean vector diagrams) had the smallest
  relative overcounts (~1.6–2x); the two furthest (rectangle+text nodes, dense
  numeric-edge-label network) had the largest, up to predicted V=474 on one case
  (~4.7x the n=100 training ceiling).
- Combinatorial-plausibility violation found: the standalone full-page photo (two
  diagrams + handwritten prose, never cropped to isolate a single graph) predicted
  V=5, E=19 — 19 exceeds the maximum 10 possible edges for 5 vertices. This is the
  only violation found across all live-app testing to date (29/30 combined Part 1 +
  Part 2 predictions stayed combinatorially plausible). Grad-CAM for this case showed
  attention spread across both diagrams and bleeding onto the prose text, consistent
  with the image sitting outside anything in the training distribution (always a
  single clean graph, no accompanying text or second diagram). Not fully isolated
  from hand-drawn style itself — the two other real hand-drawn/scanned images in this
  batch, both cropped to a single diagram, stayed plausible, which is reasonable but
  indirect evidence that clutter, not hand-drawn style, was the cause.
- Data quality note: OCR-based extraction of the Streamlit metric numbers from
  screenshots was unreliable for several values (large bold display font resists
  automated digit segmentation at screenshot resolution); final numbers for 4 of 7
  images were confirmed directly by the user rather than extracted.

# 2026-07-19 — Part 3: revised combinatorial-plausibility claim

- The Phase 3 finding "zero predictions ever have `pred_num_edges` exceed max
  possible edges for predicted vertices" (originally confirmed against
  `failure_case_categories.csv`, 1,676 held-out + test predictions) needs updating
  in light of the Part 2 finding above.
- Revised claim: the property holds across all 1,676 offline held-out/test
  predictions and across 29 of 30 live-app predictions collected in Parts 1–2 (23
  `.graphml` uploads + 7 novel images). The single exception
  (`handwritten_graph_1.jpeg`, full uncropped notebook page) is best explained as an
  input-distribution failure (multi-diagram + text in one frame), not a breakdown of
  the property under normal single-graph input.
- Action: update the corresponding claim in `FINAL_REPORT.md` (added to Section 5.6)
  from an unqualified "never" to the qualified form above. No new computation
  required — this is a data-driven correction, not new analysis.

# 2026-07-20 — Part 4 (free items): Wilcoxon probe significance + warning-banner validation

- Ran the two "free" Part 4 analyses that required no new data collection, only
  statistical treatment of existing CSVs (`probe_predictions.csv`,
  `failure_case_categories.csv`). Full results compiled into
  `evaluation/results/topolens_part4_free_analyses_results.csv`.

**Wilcoxon signed-rank test on probe deltas (n=40, paired per graph):**

- Shortcut probe (`constant_node_size` vs. `original`): vertex error worsened
  significantly (median 1.0 → 11.0, p=3.6e-06, 33/40 graphs worse) — confirms
  the node-size shortcut with a real significance test, not just a summary MAE
  delta. Edge error required care: the mean *appears* to improve
  (54.18 → 39.80) but this is an artifact of two extreme outlier graphs
  (`syn_dense_large_0069`, `syn_dense_large_0055` — both dense+large, the same
  failure mode flagged in the Part 1 spot check). The median tells the real
  story (3.5 → 20.0, 35/40 graphs worse, p=0.0003 significant in the
  worsening direction). **Report median/win-loss counts for this comparison,
  not the mean** — the mean is directionally backwards from what actually
  happens to most graphs.
- Layout probe (`alt_layout`/Kamada-Kawai vs. `original`): no significant
  effect on either vertex (p=0.499) or edge (p=0.898) error. Median delta is
  0 for vertices, ~1.0 for edges; win/loss splits are near coin-flips
  (17/16 and 18/16). The apparent -18.65 mean edge delta is driven almost
  entirely by one outlier (`syn_dense_large_0069` again, delta -512). Clean
  null result — the model is not meaningfully sensitive to sfdp vs.
  Kamada-Kawai layout, which is good news for the Limitations section and
  contradicts what raw mean MAE alone would have suggested.

**Warning-banner (dense bucket) validation, holding size constant via bins,
using all 1,676 held-out+test predictions in `failure_case_categories.csv`:**

- Naive flagged-vs-unflagged comparison (mixing dense-bucket with
  OOD-size) shows flagged graphs have higher mean error, appearing to
  validate the banner — but this is confounded, and even at that level the
  median is actually *lower* for flagged graphs (1.0 vs 3.0 vertex error),
  which should have been a red flag on its own.
- Deconfounded by binning on `num_nodes` (0–15, 15–30, 30–50, 50–75,
  75–100) and comparing dense vs. sparse within each bin:
  - **Vertex prediction: the dense-bucket warning is backwards.** Dense
    graphs have lower vertex error than sparse graphs at every single size
    bin (e.g. 75–100 nodes: dense median 5.0 vs. sparse median 21.0). Dense
    is the *safest* bucket for vertex count, not a risk flag.
  - **Edge prediction: the warning is directionally right but badly
    size-miscalibrated.** Dense and sparse edge error are tied at the
    smallest size (0–15 nodes, median ~1.0 both), then diverge steadily as
    size grows, crossing over around 15–30 nodes and reaching a 3.3x gap by
    75–100 nodes (dense median 224.0 vs. sparse median 67.5). This is the
    full 1,676-graph statistical confirmation of the Part 1 spot-check
    finding that dense+large graphs are the real failure mode, not density
    alone.
- Action: the current dense-bucket warning text (flat MAE citation of
  0.43/1.12) conflates two graphs with opposite risk profiles and should
  ideally be split into a size-aware version (dense+small: reliable;
  dense+large: high risk) — flagged as a concrete, evidence-backed
  recommendation for `FINAL_REPORT.md` Limitations / Future Work.

**Not run:** pixel-only ink-fraction baseline (Part 4 free item 3) —
requires per-graph raw `ink_fraction` values, which don't exist in any
currently exported CSV (`ink_coverage_correlations.csv` only has aggregated
Pearson r). Needs the raw per-graph pass regenerated before this can be run.

# 2026-07-20 — Part 4 (free items, completed): pixel-only baseline + structural correlations

- Ran the two remaining "free" Part 4 items using
  `evaluation/results/structural_pixel_features.csv` (per-graph ink fraction +
  diameter + clustering coefficient, computed locally via
  `compute_structural_pixel_features.py`). Full results in
  `evaluation/results/topolens_pixel_structural_correlations.csv`.

**Pixel-only baseline (`ink_fraction → num_edges`, simple linear regression):**

- Test split (375 graphs, 100% synthetic): r=0.905, R²=0.819, linear-fit
  MAE=83.73 vs. CNN's actual MAE=25.39. Ink coverage alone explains 82% of
  edge-count variance on synthetic data — a substantial chunk of the CNN's
  synthetic-data accuracy could in principle come from pixel counting alone,
  though the CNN is still ~3.3x more accurate in absolute MAE terms.
- Held-out split (1,301 graphs, 100% real MUTAG/PROTEINS): r=0.059,
  R²=0.004, linear-fit MAE=47.25 vs. CNN's actual MAE=28.03. Ink coverage is
  essentially uninformative on real graphs — `ink_fraction` barely varies
  (0.092–0.192) despite true edge counts spanning 5–1,049. **This means the
  CNN's real-world generalization cannot be explained by an ink-counting
  shortcut, since that shortcut doesn't meaningfully exist in real-graph
  renders.** Genuinely reassuring result for the core CNN-vs-GNN research
  question.
- Held-out breakdown by category is informative: `high_density_clutter`
  alone shows r=0.947 (ink coverage does track edges for real dense
  graphs specifically), while `in_distribution_normal` shows a slight
  negative correlation (r=-0.206) and `out_of_distribution_size` shows
  r=0.368. The aggregate near-zero correlation masks real heterogeneity
  across these subgroups.

**Structural correlations (diameter, clustering vs. error):**

- Raw correlations are all significant (p<0.05, most p<0.0001 given large
  n), but `avg_clustering` is highly collinear with `density` (r=0.836
  test, 0.596 held-out) — it's mostly redescribing density, not adding
  independent signal, and shouldn't be presented as a separate finding.
- `diameter` is more independent of `num_nodes` (r=0.336 test, r=0.719
  held-out) and its partial correlation with error, controlling for
  `num_nodes`, is still meaningful: +0.478 (test, vertex error) and -0.287
  (held-out, vertex error). **The sign flips between synthetic and real
  data** — on synthetic graphs, a larger-than-expected diameter for a given
  size predicts worse vertex accuracy; on real graphs, the opposite. Worth
  reporting as an exploratory, non-obvious finding rather than a settled
  mechanism — likely reflects a structural difference between synthetic
  generators and real molecule/protein graph topology at matched size.

- All 4 "free" Part 4 items are now complete (Wilcoxon significance test,
  warning-banner validation, pixel-only baseline, structural correlations).
  Decision on "cheap"/"real effort" tiers still pending.

## Day 10 — 21 Jul 2026 — External Repo Audit: Six Flagged Issues Reviewed

### Context / Goal

An external review (not self-generated) flagged six potential problems in
the repo: hardcoded paths, an incomplete final report, a config-duality
risk, a git/.gitignore conflict, layout-consistency doubt, and dependency
version gaps. Goal: verify each claim against the actual repo state (not
just take the review at face value) and fix whatever was real.

### What was done

- **Problem 1 (hardcoded paths) — fixed.** `render/render_graphs.py`'s
  hardcoded `E:\Anaconda\envs\py-dev\Library\bin` replaced with a
  `CONDA_PREFIX`/`sys.prefix` lookup plus an optional `TOPOLENS_GRAPHVIZ_BIN`
  override. `find_sfdp.py` rewritten as a portable diagnostic (`shutil.which`
  first, then active-env prefix). `diagnose_graphviz.py` — reviewed, was
  already portable (uses `sys.executable`/`shutil.which`), no fix needed;
  since it and `find_sfdp.py`'s one-off purpose was already served, both
  were deleted, along with `test_pydot.py` (ad-hoc smoke test) and
  `temp_render/` (its output).
- **Problem 2 (incomplete report) — fixed.** `FINAL_REPORT.md` Sections
  1–4, 6, 7, and 8 written from `progress_log.md`, `report/log.md`,
  `summary_comparison.csv`, and the model source files. Section 5 (already
  complete) and the Appendix carried over unchanged, with one correction —
  the Appendix's app description still said "Neo-Brutalist"/Syne, updated
  to match the app's actual current "Cinematic Instrument"/Fraunces design.
- **Problem 3 (config duality) — documentation fix (Option A), not a full
  refactor.** Verified 8 of 18 scripts import both `config.py` and
  `config.yaml`; the only genuine *conflict* (not just duplication) was
  `config.yaml`'s `dataset.synthetic.node_range: [5, 150]` vs. `config.py`'s
  `NODE_TIERS` (capped at 100) — confirmed nothing actually reads that yaml
  field, so correcting it to `[5, 100]` was zero-risk. Added comments to
  both files clarifying which file is authoritative for which values,
  rather than doing a full single-source-of-truth refactor this late in
  the timeline.
- **Problem 4 (checkpoints vs. .gitignore) — not an actual issue.**
  `git ls-files models/checkpoints/` confirmed only `.gitkeep` is tracked;
  `cnn_best.pt`/`gnn_best.pt` were never committed. `.gitignore` was already
  working correctly.
- **Problem 5 (layout consistency) — not an actual issue, confirmed and
  logged.** `labels_processed.csv`'s `layout_algorithm` column: 3,801/3,801
  rows are `graphviz_sfdp`. Cross-checked via image file mtimes, which all
  fall inside a single ~4.5-hour render batch — no evidence of a mixed
  spring/sfdp dataset surviving from the Day 2 fallback incident.
- **Problem 6 (dependency gaps) — fixed.** `requirements.txt` updated:
  `pygraphviz` kept commented out (it needs a system-level Graphviz install
  `pip` can't provide) but now documents why and confirms the pipeline
  works without it via the pydot fallback; `torch`, `torch-geometric`, and
  `streamlit` pinned to exact versions confirmed against the live
  environment (`torch==2.13.0`, `torch-geometric==2.8.0`,
  `streamlit==1.59.2` — note `torch` had drifted to 2.13.0 from the 2.11.0
  mentioned in earlier logs, independent of this project). One claim in the
  original review was found to be incorrect and not acted on: `evaluate.py`'s
  `try/except` around `torch_geometric` does not fail silently — it raises
  `ImportError` immediately, which is already correct fail-fast behavior.
- **Added a new feature along the way:** live-app uploads (images and
  graph files) are now saved to `render/novel_uploads/` with a
  `manifest.csv` logging CNN/GCN/graph-statistic predictions and any known
  ground truth, deduplicated by content hash, with the saved image named
  after the uploaded file (not a hash) for traceability. Formerly-empty
  `render/gephi_samples/` was repurposed/renamed for this rather than kept
  as dead weight.

### How

Every claim was checked against the actual repo/data before being acted on,
rather than trusted at face value — several of the six turned out to be
wrong or already resolved (Problems 4 and 5), and one sub-claim within
Problem 6 (the "silent failure" characterization) was also wrong. This
caught real issues (the `node_range` conflict, the hardcoded paths) without
introducing unnecessary churn on things that weren't actually broken.

### Why

**Why Option A (doc fix) over a full config.py/config.yaml refactor for
Problem 3?** The project is 10 days into a 1-week deadline and essentially
feature-complete; a full single-source-of-truth rewrite touching 18 files
is real surface area for regression this late, for a conflict that (once
verified) turned out to be dead/unused documentation rather than a live
bug. The correction plus clarifying comments removes the actual risk
(someone trusting the wrong `node_range` value) at effectively zero cost.

**Why verify Problems 4 and 5 with actual commands instead of just fixing
them as described?** Both turned out to be non-issues — `git ls-files`
showed the checkpoints were never tracked, and the mtime/label check on
the real data showed no mixed-layout contamination. Fixing things that
aren't broken wastes the remaining time and risks introducing new bugs
into a working pipeline.

### Issues

None blocking. `torch` version drift (2.11.0 → 2.13.0) happened
independent of this project and wasn't something any of the six flagged
problems anticipated — worth keeping an eye on if the environment gets
upgraded again before submission.

### Next

All six flagged issues are resolved or confirmed closed. Remaining open
item from Day 9: Sections 5.1 and 5.3 of `FINAL_REPORT.md` (Grad-CAM
description, layout-sensitivity probe write-up) are still template
placeholders — everything else in the report is now complete.

# 2026-07-21 — Codebase health check + five targeted fixes

Full read-through of every file in the repo (26 files across `app/`,
`data/`, `evaluation/`, `models/`, `render/`, root). Purpose: confirm
no silent regressions had been introduced across the Day 5–10 patch
sessions before final submission.

- **`requirements.txt` invalid torch pin**: Corrected `torch==2.13.0` to `torch==2.5.1` (the actual stable build used for GPU training), updated `torchvision>=0.20` and `torch-geometric==2.6.0`, and clarified in comments that training ran on Colab T4 GPU (CUDA) while `map_location="cpu"` handles local CPU execution.
- **Unsafe PyTorch checkpoint loads in `evaluate.py`**: Added `weights_only=False` to `load_cnn_model()` and `load_gnn_model()` in `evaluation/evaluate.py` to support checkpoints with nested Python metadata (`normalize_stats` tuple, `config` dict).
- **Synchronized UI copy in `app.py`**: Removed stale "(coming soon)" label from Research Insights sidebar description; updated to describe actual submodules (Grad-CAM attention, shortcut-learning probe, layout sensitivity, failure-case taxonomy).
- **Added Graphviz spring fallback warning in `render_graphs.py`**: Replaced dead `pass` block with `[WARN]` print statement notifying when `graphviz_sfdp` layout is missing and falls back to `networkx_spring`.
- **System Verification**: Tested `app/app.py` via Streamlit server execution; verified clean startup on port 8501.

