"""Dataset utilities for Topolens image-based regression."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from topolens_utils import load_yaml_config


def _get_input_size() -> Tuple[int, int]:
    cfg = load_yaml_config()
    input_size = cfg["model"]["cnn"]["input_size"]
    return int(input_size[0]), int(input_size[1])


def build_image_transform(mean: Sequence[float], std: Sequence[float], input_size: Optional[Tuple[int, int]] = None):
    """Build the standard resize -> tensor -> normalize pipeline."""
    size = input_size or _get_input_size()
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_tensor_only_transform(input_size: Optional[Tuple[int, int]] = None):
    """Resize and convert to tensor without normalization."""
    size = input_size or _get_input_size()
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor()])


@lru_cache(maxsize=8)
def compute_normalization_stats(split_csv: str, images_dir: str, input_size: Tuple[int, int]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute RGB mean/std from the provided split.

    The stats are computed from the split passed in, which lets training scripts
    compute the values once on the training set and reuse them for val/test.
    """
    df = pd.read_csv(split_csv)
    tensor_transform = build_tensor_only_transform(input_size)
    channel_sums = torch.zeros(3, dtype=torch.float64)
    channel_squared_sums = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for _, row in df.iterrows():
        image_path = Path(images_dir) / f"{row['graph_id']}.png"
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            tensor = tensor_transform(image).to(torch.float64)
        channel_sums += tensor.sum(dim=(1, 2))
        channel_squared_sums += (tensor ** 2).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    if pixel_count == 0:
        raise ValueError(f"No images found while computing stats from {split_csv}")

    mean = channel_sums / pixel_count
    variance = channel_squared_sums / pixel_count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


class TopolensImageDataset(Dataset):
    """Image regression dataset returning normalized tensors and log targets."""

    def __init__(
        self,
        split_csv: str | Path,
        images_dir: str | Path,
        transform=None,
        normalize_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.images_dir = Path(images_dir)
        self.df = pd.read_csv(self.split_csv).reset_index(drop=True)
        self.input_size = _get_input_size()

        if transform is not None:
            self.transform = transform
            self.normalize_stats = normalize_stats
        else:
            if normalize_stats is None:
                normalize_stats = compute_normalization_stats(
                    str(self.split_csv),
                    str(self.images_dir),
                    self.input_size,
                )
            self.normalize_stats = normalize_stats
            self.transform = build_image_transform(normalize_stats[0], normalize_stats[1], self.input_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self.images_dir / f"{row['graph_id']}.png"
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        target = torch.tensor(
            [np.log1p(float(row["num_nodes"])), np.log1p(float(row["num_edges"]))],
            dtype=torch.float32,
        )
        return image, target

    @staticmethod
    def inverse_transform(pred_tensor):
        """Map log1p outputs back to rounded raw counts."""
        tensor = torch.as_tensor(pred_tensor, dtype=torch.float32)
        raw = torch.expm1(tensor).round().clamp_min(0)
        if raw.ndim == 1:
            return int(raw[0].item()), int(raw[1].item())
        return raw.to(torch.int64)
