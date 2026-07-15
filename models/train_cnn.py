"""Train the Topolens CNN regressor."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from models.cnn_model import build_cnn_model
from models.dataset import TopolensImageDataset, compute_normalization_stats
from topolens_utils import ensure_dir, load_yaml_config, set_random_seeds


def build_subset(dataset: TopolensImageDataset, size: int, seed: int) -> Subset:
    """Select a deterministic subset for smoke tests."""
    size = min(size, len(dataset))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return Subset(dataset, indices[:size].tolist())


def get_base_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def build_dataloaders(smoke_test: bool, seed: int):
    cfg = load_yaml_config()
    batch_size = int(cfg["model"]["cnn"]["batch_size"])
    input_size = tuple(cfg["model"]["cnn"]["input_size"])

    train_csv = Path(config.DATA_DIR) / "splits" / "train.csv"
    val_csv = Path(config.DATA_DIR) / "splits" / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Split CSVs are missing. Run data/make_splits.py first.")

    images_dir = Path(config.IMAGES_DIR)
    normalize_stats = compute_normalization_stats(str(train_csv), str(images_dir), input_size)

    train_base = TopolensImageDataset(train_csv, images_dir, normalize_stats=normalize_stats)
    val_base = TopolensImageDataset(val_csv, images_dir, normalize_stats=normalize_stats)

    if smoke_test:
        train_dataset = build_subset(train_base, 40, seed)
        val_dataset = build_subset(val_base, 10, seed + 1)
    else:
        train_dataset = train_base
        val_dataset = val_base

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, normalize_stats


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_loss: float, normalize_stats, cfg) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_val_loss": val_loss,
            "normalize_stats": normalize_stats,
            "architecture": cfg["model"]["cnn"]["architecture"],
            "input_size": cfg["model"]["cnn"]["input_size"],
            "config": cfg,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Topolens CNN.")
    parser.add_argument("--smoke-test", action="store_true", help="Train on a tiny deterministic subset for 2 epochs.")
    args = parser.parse_args()

    cfg = load_yaml_config()
    seed = int(cfg["project"]["seed"])
    set_random_seeds(seed)

    if cfg["model"]["cnn"]["optimizer"].lower() != "adam":
        raise ValueError("Only Adam is supported for the CNN in this phase.")
    if cfg["model"]["cnn"]["loss"].lower() != "mse":
        raise ValueError("Only MSE loss is supported for the CNN in this phase.")

    train_loader, val_loader, normalize_stats = build_dataloaders(args.smoke_test, seed)
    model = build_cnn_model(cfg["model"]["cnn"]["architecture"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["model"]["cnn"]["learning_rate"]))
    epochs = 2 if args.smoke_test else int(cfg["model"]["cnn"]["epochs"])

    checkpoint_path = Path(config.DATA_DIR).parent / "models" / "checkpoints" / "cnn_best.pt"
    log_path = Path(config.DATA_DIR).parent / "evaluation" / "results" / "cnn_training_log.csv"
    ensure_dir(log_path.parent)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    training_start = time.perf_counter()
    best_val_loss = float("inf")
    rows = []

    print(f"CNN parameter count: {model.num_parameters():,}")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        epoch_seconds = time.perf_counter() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_path, model, epoch, best_val_loss, normalize_stats, cfg)

        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_seconds": epoch_seconds,
                "best_val_loss": best_val_loss,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"time={epoch_seconds:.1f}s"
        )

    training_seconds = time.perf_counter() - training_start
    pd.DataFrame(rows).to_csv(log_path, index=False)

    if not checkpoint_path.exists():
        save_checkpoint(checkpoint_path, model, epochs, best_val_loss, normalize_stats, cfg)

    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Total training time: {training_seconds:.1f}s")


if __name__ == "__main__":
    main()
