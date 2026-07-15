"""Create reproducible train/val/test splits for Topolens.

Synthetic graphs are used for train/validation/test, stratified by generator.
Real graphs are written to a separate held-out file and never mixed into the
trainable pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from topolens_utils import compute_density, density_bucket, ensure_dir, load_yaml_config, set_random_seeds


def add_density_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach density metadata used later for analysis and stratified reporting."""
    out = df.copy()
    out["density"] = out.apply(lambda row: compute_density(int(row["num_nodes"]), int(row["num_edges"])), axis=1)
    out["density_bucket"] = out["density"].map(density_bucket)
    return out


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate exact split counts while preserving the requested proportions."""
    raw_counts = np.array(ratios, dtype=float) * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - counts.sum()
    if remainder > 0:
        fractional = raw_counts - counts
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def split_synthetic_pool(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split synthetic graphs 70/15/15 with generator stratification."""
    synthetic = df[df["source"] == "synthetic"].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)

    train_parts = []
    val_parts = []
    test_parts = []

    ratios = (0.70, 0.15, 0.15)
    for generator, group in synthetic.groupby("generator", sort=True):
        group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        n_train, n_val, n_test = allocate_counts(len(group), ratios)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val : n_train + n_val + n_test])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df


def write_split(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def main() -> None:
    cfg = load_yaml_config()
    seed = int(cfg["project"]["seed"])
    set_random_seeds(seed)

    labels_path = Path(config.PROCESSED_LABELS_CSV)
    if not labels_path.exists():
        labels_path = Path(config.LABELS_CSV)
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} does not exist. Run rendering first.")

    labels = pd.read_csv(labels_path)
    labels = add_density_columns(labels)

    processed_dir = Path(config.DATA_DIR) / "processed"
    ensure_dir(processed_dir)
    labels.to_csv(processed_dir / "labels_processed.csv", index=False)

    train_df, val_df, test_df = split_synthetic_pool(labels, seed)
    held_out_df = labels[labels["source"] == "real"].copy().reset_index(drop=True)

    out_dir = Path(config.DATA_DIR) / "splits"
    ensure_dir(out_dir)

    write_split(train_df, out_dir / "train.csv")
    write_split(val_df, out_dir / "val.csv")
    write_split(test_df, out_dir / "test.csv")
    write_split(held_out_df, out_dir / "held_out.csv")

    print("Split summary")
    print(f"  train: {len(train_df)}")
    print(f"  val: {len(val_df)}")
    print(f"  test: {len(test_df)}")
    print(f"  held_out: {len(held_out_df)}")

    print("\nPer-generator counts")
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"  {split_name}:")
        print(split_df["generator"].value_counts().sort_index().to_string())

    print("\nDensity buckets")
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df), ("held_out", held_out_df)):
        print(f"  {split_name}:")
        print(split_df["density_bucket"].value_counts().reindex(["sparse", "medium", "dense"], fill_value=0).to_string())


if __name__ == "__main__":
    main()
