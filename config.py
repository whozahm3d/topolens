# config.py
# Global configuration file for Topolens Phase 1.
# Imports from this file ensure consistency across dataset generation, loading, rendering, and validation.

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
LABELS_CSV = "data/labels.csv"

TU_DATASETS = ["MUTAG", "PROTEINS"]
