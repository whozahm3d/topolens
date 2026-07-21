<div align="center">

# 🔭 Topolens

**Visual Topology Regression via Deep Convolutional Networks**

*An empirical investigation into whether 2D visual representations of graphs contain sufficient structural signals to recover graph-level topological invariants (vertex & edge counts) without graph-native message passing.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch__Geometric-2.6.0-3C2179.svg?logo=pytorch&logoColor=white)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59.2-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Pipeline-Verified-success.svg)](#-reproducibility--experimental-protocol)

---

[Key Findings](#-key-experimental-findings) •
[System Architecture](#-system-architecture) •
[Quickstart](#-quickstart) •
[Interactive Web App](#-interactive-web-application) •
[Repository Layout](#-repository-structure) •
[Contributing](CONTRIBUTING.md)

</div>

---

## 📌 Overview & Research Statement

**Topolens** explores a fundamental research question at the boundary of Computer Vision and Graph Learning:

> *Can a generic 2D Convolutional Neural Network (CNN) estimate topological properties—specifically node count $|V|$ and edge count $|E|$—purely from rendered 2D bitmap representations of graphs, and how does it compare to graph-native GNN architectures?*

Graph Neural Networks (GNNs) operate natively on graph adjacency matrices and node features via spatial or spectral message passing. However, in many real-world scenarios—such as legacy scientific diagrams, scanned figures, UI wireframes, and biological pathways—only rendered 2D images are available. **Topolens** formulates graph counting as a visual regression task over 224×224 RGB images, benchmarking a 4-block CNN (`CustomCNNRegressor`) against a standard 3-layer Graph Convolutional Network (`GraphCountGCN`) and a non-learned density heuristic.

---

## 📊 Key Experimental Findings

### 1. In-Distribution vs. Held-Out Generalization (MAE)

Evaluated on **1,676 test and held-out graphs** across synthetic generators and real-world biological benchmark datasets (MUTAG & PROTEINS):

| Model | Split | Vertex MAE ($|V|$) | Edge MAE ($|E|$) | Key Characteristics |
|---|---|---|---|---|
| **Custom CNN** | Synthetic Test | **3.20** | **25.39** | Highly accurate in-distribution |
| **GCN Baseline** | Synthetic Test | 19.05 | 76.21 | Degree-normalized mean pooling |
| **Heuristic** | Synthetic Test | 0.00* | 342.10 | *Trivial node lookup; edge formula fails on scale |
| **Custom CNN** | Held-Out (Real) | **10.62** | **28.03** | Robust on small-to-medium real graphs |
| **GCN Baseline** | Held-Out (Real) | 21.15 | 32.50 | Stable mean-pooling embedding |
| **Heuristic** | Held-Out (Real) | 0.00* | 4927.70 | Catastrophic failure on large graphs |

### 2. Shortcut-Learning & Probe Interventions

Controlled intervention probes isolate *what* visual features the CNN relies upon:

* **Constant Node Size Probe**: When node circle radii are locked to a constant size (removing the $r = \max(15, 4000/N)$ scaling cue), CNN Vertex MAE increases significantly ($3.43 \to 10.48, p = 3.6 \times 10^{-6}$ via Wilcoxon signed-rank test), confirming apparent dot coverage acts as a primary counting proxy.
* **Layout Sensitivity Probe**: Switching rendering layout from Graphviz `sfdp` to `Kamada-Kawai` produced no statistically significant change in vertex error ($p = 0.499$) or edge error ($p = 0.898$), ruling out layout-specific spatial overfitting.
* **Pixel-Only Ink Coverage Baseline**: Simple linear regression of ink fraction $\to |E|$ explains **82% of edge variance** on synthetic renders ($r = 0.905$), but collapses on real molecule/protein graphs ($r = 0.059$). This confirms the CNN's real-world performance relies on visual structural features rather than trivial ink counting.

---

## 🏗️ System Architecture

### Pipeline Workflow

```mermaid
flowchart LR
    subgraph Data Generation
        A1[NetworkX Synthetic Generators] --> B[GraphML Raw Graphs]
        A2[PyG TUDataset MUTAG/PROTEINS] --> B
    end

    subgraph Rendering Pipeline
        B --> C{Graphviz sfdp Backend}
        C -- Success --> D[224x224 RGB PNG]
        C -- Fallback --> C2[NetworkX Spring Layout] --> D
    end

    subgraph Inference & Benchmarking
        D --> E1[Custom CNN Regressor]
        B --> E2[GraphCount GCN]
        B --> E3[Density Heuristic]
    end

    subgraph Web & Evaluation
        E1 --> F[Streamlit Web App]
        E2 --> F
        E3 --> F
        E1 --> G[Grad-CAM Attention Maps]
    end
```

### Core Components

1. **Synthetic & Real Dataset Engine** (`data/`):
   - **Synthetic**: 2,500 graphs across 5 generator families (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, Random Trees, Dense ER) × 4 node tiers (tiny $\le 10$, small $10\text{–}25$, medium $25\text{–}50$, large $50\text{–}100$).
   - **Real-World**: MUTAG (188 graphs) & PROTEINS (1,113 graphs), including an out-of-distribution "xlarge" tier up to 620 nodes.
2. **Custom CNN Regressor** (`models/cnn_model.py`):
   - 4-Block Conv2D stack with BatchNorm, ReLU, and MaxPool2D ($3 \to 16 \to 32 \to 64 \to 128$ channels).
   - Adaptive Average Pooling ($3 \times 3$) $\to$ Linear MLP head ($1152 \to 128 \to 2$).
   - Log-transformed targets ($\log(1 + y)$) with MSE loss and Adam optimizer.
3. **Graph Neural Network Baseline** (`models/gnn_baseline.py`):
   - 3-Layer GCN architecture using PyTorch Geometric.
   - Normalized degree node features + `global_mean_pool` size-invariant embeddings.
4. **Grad-CAM Interpretation Engine** (`models/gradcam.py`):
   - Captures activation maps at `features[3]` (last spatial convolutional block) for target-specific (vertex vs. edge) visual attention analysis.

---

## 🚀 Quickstart

### Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/whozahm3d/topolens.git
cd topolens

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> **Optional Backend**: Graphviz `sfdp` rendering requires system-level Graphviz binaries installed on your system PATH (`dot`/`sfdp`). If unavailable, `render_graphs.py` gracefully falls back to NetworkX spring layout with an automated `[WARN]` notification.

### Running the Interactive Web App

Launch the production Streamlit application locally:

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 💻 Interactive Web Application

The Streamlit web application (`app/app.py`) provides an interactive interface for model inspection, prediction, and visual interpretability:

- **Multi-Format Ingestion**: Upload raw graph structures (`.graphml`, `.csv`, `.txt`) or rendered visual graph images (`.png`, `.jpg`).
- **Live Multi-Model Comparison**: Evaluates uploaded topology simultaneously through the CNN, GCN, and Statistical Heuristic models.
- **Dynamic Warning Banner**: Fires context-aware low-confidence alerts when node counts exceed the empirical training distribution ($|V| > 100$) or when density enters the clutter regime ($\text{density} \ge 0.40$).
- **Grad-CAM Heatmap Overlay**: Real-time side-by-side rendering of spatial attention for vertex counting vs. edge counting.
- **Research Insights Suite**: Interactive dashboards covering shortcut probes, layout sensitivity, and ink coverage correlations.

---

## 🧪 Training & Evaluation

### Training Models

All models support automatic CUDA GPU detection (used for Colab T4 training) and fall back to CPU execution:

```bash
# Train CNN Regressor (smoke test: 2 epochs on tiny subset)
python models/train_cnn.py --smoke-test

# Train CNN Regressor (full training)
python models/train_cnn.py

# Train GCN Baseline
python models/train_gnn.py
```

### Running Evaluation & Diagnostics

```bash
# Evaluate checkpoints against test and held-out splits
python evaluation/evaluate.py

# Run error analysis & generate scatter plots
python evaluation/error_analysis.py

# Run failure-case taxonomy classification
python evaluation/failure_taxonomy.py

# Execute controlled shortcut and layout probes
python render/render_probe_variants.py
python evaluation/shortcut_probe.py

# Compute ink coverage & component stats
python evaluation/ink_coverage_analysis.py
```

---

## 📂 Repository Structure

```
topolens/
├── app/
│   ├── app.py                     # Main Streamlit multi-page web application
│   └── .streamlit/config.toml     # UI theme styling & configuration
├── data/
│   ├── raw/                       # Raw GraphML synthetic & real graph files
│   ├── images/                    # Rendered 224x224 graph images
│   ├── processed/                 # Dataset manifests & labels (labels_processed.csv)
│   ├── splits/                    # Train, val, test, and held-out CSV splits
│   ├── generate_synthetic.py      # NetworkX synthetic graph generator
│   ├── load_tu_datasets.py        # PyG MUTAG / PROTEINS dataset importer
│   └── make_splits.py             # Stratified dataset split script
├── render/
│   ├── render_graphs.py           # Core Graphviz/NetworkX graph rendering engine
│   ├── render_probe_variants.py   # Controlled probe variant generator
│   └── novel_uploads/             # User-uploaded application testing images & manifest
├── models/
│   ├── cnn_model.py               # Custom 4-block CNN Regressor architecture
│   ├── gnn_baseline.py            # PyG 3-layer GraphCountGCN model
│   ├── graph_statistic_baseline.py# Density-based statistical baseline
│   ├── dataset.py                 # PyTorch TopolensImageDataset & normalization
│   ├── gradcam.py                 # Grad-CAM attention heatmap generator
│   ├── train_cnn.py               # CNN training loop with checkpointing
│   ├── train_gnn.py               # GCN baseline training loop
│   └── checkpoints/               # Trained PyTorch model checkpoints (.pt)
├── evaluation/
│   ├── evaluate.py                # Multi-model test & held-out evaluation runner
│   ├── error_analysis.py          # Error distribution & faceted scatter plots
│   ├── failure_taxonomy.py        # Out-of-distribution & clutter failure classifier
│   ├── shortcut_probe.py          # Node-size & layout intervention analyzer
│   ├── ink_coverage_analysis.py   # Pixel coverage & connected component analysis
│   └── results/                   # Computed evaluation CSV metrics & summaries
├── notebooks/                     # Exploratory analysis & visualization notebooks
├── report/
│   ├── FINAL_REPORT.md            # Comprehensive research report & submission
│   ├── log.md                     # Technical execution log
│   └── figures/                   # Generated figures, Grad-CAM grids, and plots
├── config.py                      # Central Python environment & path constants
├── config.yaml                    # Global YAML parameter configuration
├── progress_log.md                # Narrative project development log
├── requirements.txt               # Pinned Python package dependencies
├── CONTRIBUTING.md                # Open-source contribution guidelines
└── LICENSE                        # MIT License
```

---

## 🛡️ Reproducibility & Experimental Protocol

- **Deterministic Random Seeds**: All stochastic processes (graph generation, split partitioning, weight initialization, probe sampling) are seeded via `config.yaml` (`project.seed = 42`).
- **Graph ID Hashing**: Individual synthetic graph parameters are derived deterministically using SHA-256 hashes of `graph_id` strings.
- **Checkpoint Compatibility**: Model checkpoints saved during GPU training explicitly support CPU execution (`map_location="cpu"`) and safe dict unpickling (`weights_only=False`).

---

## 📜 License & Citation

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

If you use **Topolens** in your research or project, please cite:

```bibtex
@misc{topolens2026,
  author = {Whozahm3d},
  title = {Topolens: Visual Topology Regression via Deep Convolutional Networks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/whozahm3d/topolens}
}
```
