# Topolens

Estimating graph structural properties (vertex count, edge count) from
rendered images using a CNN — an intentional contrast to graph-native
methods like GNNs, exploring whether visual inference is competitive.

## Overview

Topolens converts graphs into rendered images and trains a CNN to
regress vertex and edge counts purely from pixels, without access to
the underlying graph structure. Results are compared against a
graph-native baseline (GCN or graph-statistic heuristic) to evaluate
whether the image-based approach is competitive.

## Project Status

Internship project — 1 week timeline. See `progress_log.md` for
day-by-day progress.

## Pipeline

1. **Dataset generation** (`data/`, `render/`) — synthetic graphs via
   NetworkX generators (Erdős–Rényi, Barabási–Albert, Watts–Strogatz,
   random trees, dense graphs), rendered to images with a consistent
   layout algorithm.
2. **Model training** (`models/`) — CNN regression model + baseline
   (GCN or graph-statistic heuristic).
3. **Evaluation** (`evaluation/`) — MAE/RMSE comparison, CNN vs baseline.
4. **Error analysis** (`evaluation/`) — breakdown by graph density and
   generator type to explain failure patterns.
5. **Report** (`report/`) — full research-style write-up including
   methodology, results, discussion, and limitations.
6. **Stretch goal** (`app/`) — Streamlit app for interactive inference,
   built only after steps 1-5 are complete.

## Setup

\`\`\`bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
\`\`\`

## Project Structure

See `config.yaml` for all dataset, rendering, and model parameters —
this is the single source of truth; scripts should read from it rather
than hardcoding values.

\`\`\`
topolens/
├── data/           # raw, processed, splits, rendered images
├── render/         # image generation scripts
├── models/         # CNN + baseline model code, checkpoints
├── evaluation/      # metrics, error analysis, results
├── notebooks/       # exploration and experiments
├── app/             # stretch goal: Streamlit inference app
└── report/          # final write-up + figures
\`\`\`

## Reproducibility

All randomness (dataset generation, splits, model initialization) is
controlled by a single seed (`config.yaml → project.seed`).

## License

MIT
