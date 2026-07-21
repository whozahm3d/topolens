# config.py
# Source of truth for dataset generation/rendering constants (NODE_TIERS,
# GENERATORS, paths, IMAGE_SIZE/DPI, RANDOM_SEED). These values are mirrored
# in config.yaml's dataset:/render: blocks for human-readable documentation
# only — config.yaml is NOT read for these; keep both in sync by hand if
# either changes. Training hyperparameters (batch size, epochs, lr, GCN
# hidden dim) live only in config.yaml and are loaded via load_yaml_config().

RANDOM_SEED = 42

NODE_TIERS = {
    "tiny":   (5, 10),
    "small":  (10, 25),
    "medium": (25, 50),
    "large":  (50, 100),
}

GENERATORS = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "random_tree", "dense"]
GRAPHS_PER_CELL = 125   # 5 * 4 * 125 = 2500 synthetic graphs

IMAGE_SIZE = 224
IMAGE_DPI = 100

DATA_DIR = "data"
RAW_SYNTHETIC_DIR = "data/raw/synthetic"
RAW_REAL_DIR = "data/raw/real"
IMAGES_DIR = "data/images"
GEPHI_DEMO_DIR = "data/gephi_demo"
# Legacy path — kept only so render_graphs.py can detect and delete the old
# file on re-render. Do not use this for reading labels; use PROCESSED_LABELS_CSV.
LABELS_CSV = "data/labels.csv"
PROCESSED_LABELS_CSV = "data/processed/labels_processed.csv"

TU_DATASETS = ["MUTAG", "PROTEINS"]

CHECKPOINTS_DIR = "models/checkpoints"
RESULTS_DIR = "evaluation/results"
REPORT_FIGURES_DIR = "report/figures"
