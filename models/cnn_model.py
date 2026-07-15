"""Lightweight CNN regressor for Topolens."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNNRegressor(nn.Module):
    """Small CNN for 2D graph-image regression."""

    def __init__(self, in_channels: int = 3, output_dim: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_cnn_model(architecture: str, **kwargs) -> nn.Module:
    """Factory seam for later architecture swaps."""
    if architecture != "custom_cnn":
        raise ValueError(f"Unsupported CNN architecture: {architecture}")
    return CustomCNNRegressor(**kwargs)
