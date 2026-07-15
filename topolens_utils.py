"""Shared helpers for Topolens Phase 2 scripts.

Keep this module small and dependency-light so data, model, and evaluation
scripts can import the same deterministic path/seed/config helpers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_YAML = PROJECT_ROOT / "config.yaml"


def load_yaml_config() -> Dict[str, Any]:
    """Load the Phase 2 YAML config from the project root."""
    with CONFIG_YAML.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(*parts: str) -> Path:
    """Build an absolute path inside the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_parent_dir(path: Path | str) -> Path:
    """Create the parent directory for a file path if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_density(num_nodes: int, num_edges: int) -> float:
    """Compute simple undirected graph density with a guard for tiny graphs."""
    if num_nodes < 2:
        return 0.0
    return float(2 * num_edges / (num_nodes * (num_nodes - 1)))


def density_bucket(density: float) -> str:
    """Map density to a stable, human-readable bucket.

    Fixed cut points keep the split logic reproducible and avoid letting the
    held-out data influence the bucket boundaries.
    """
    if density < 0.15:
        return "sparse"
    if density < 0.40:
        return "medium"
    return "dense"


def coerce_path(value: Any) -> str:
    """Normalize a path-like value to a forward-slash string."""
    return str(Path(value)).replace("\\", "/")


def to_2d_tensor(values: Iterable[float]) -> torch.Tensor:
    """Convert a short iterable to a 1D float tensor."""
    return torch.tensor(list(values), dtype=torch.float32)
