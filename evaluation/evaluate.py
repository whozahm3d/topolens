"""Evaluate Topolens models on test and held-out splits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from models.cnn_model import CustomCNNRegressor
from models.dataset import TopolensImageDataset
from models.gnn_baseline import GraphCountGCN, graphml_to_pyg_data
from models.graph_statistic_baseline import build_density_table, predict_edges
from topolens_utils import ensure_dir, load_yaml_config

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception as exc:  # pragma: no cover - import guard for environments without PyG
    raise ImportError("torch_geometric is required for evaluation.py") from exc


TARGET_COLUMNS = ["num_vertices", "num_edges"]
TARGET_ALIASES = {"num_vertices": "num_nodes", "num_edges": "num_edges"}


def load_split(split_name: str) -> pd.DataFrame:
    csv_path = Path(config.DATA_DIR) / "splits" / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    return pd.read_csv(csv_path)


def raw_targets(df: pd.DataFrame) -> np.ndarray:
    return df[["num_nodes", "num_edges"]].to_numpy(dtype=np.float32)


def safe_counts(preds: np.ndarray) -> np.ndarray:
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.expm1(preds)
    raw = np.nan_to_num(raw, nan=0.0, posinf=1_000_000.0, neginf=0.0)
    return np.rint(np.clip(raw, 0.0, 1_000_000.0)).astype(np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = y_pred - y_true
    return {
        "mae": np.abs(errors).mean(axis=0),
        "rmse": np.sqrt(np.square(errors).mean(axis=0)),
    }


def metrics_records(model_name: str, split_name: str, df: pd.DataFrame, pred_counts: np.ndarray) -> List[dict]:
    y_true = raw_targets(df)
    records: List[dict] = []

    def add_block(breakdown: str, group_value: str, sub_df: pd.DataFrame, pred_subset: np.ndarray):
        metrics = compute_metrics(raw_targets(sub_df), pred_subset)
        for metric_name, values in metrics.items():
            for idx, target in enumerate(TARGET_COLUMNS):
                records.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "breakdown": breakdown,
                        "group_value": group_value,
                        "target": target,
                        "metric": metric_name,
                        "value": float(values[idx]),
                        "n": int(len(sub_df)),
                    }
                )

    add_block("overall", "all", df, pred_counts)

    generator_column = "generator"
    for generator, sub_df in df.groupby(generator_column, sort=True):
        add_block("generator", str(generator), sub_df, pred_counts[sub_df.index.to_numpy()])

    for bucket, sub_df in df.groupby("density_bucket", sort=True):
        add_block("density_bucket", str(bucket), sub_df, pred_counts[sub_df.index.to_numpy()])

    if split_name == "held_out":
        for tier, sub_df in df.groupby("tier", sort=True):
            add_block("tier", str(tier), sub_df, pred_counts[sub_df.index.to_numpy()])

    return records


def load_cnn_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CustomCNNRegressor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["normalize_stats"]


def predict_cnn(split_csv: Path, normalize_stats, model):
    cfg = load_yaml_config()
    batch_size = int(cfg["model"]["cnn"]["batch_size"])
    dataset = TopolensImageDataset(split_csv, config.IMAGES_DIR, normalize_stats=normalize_stats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    outputs = []
    with torch.no_grad():
        for images, _ in loader:
            preds = model(images)
            outputs.append(preds.cpu())
    raw_outputs = torch.cat(outputs, dim=0)
    return safe_counts(raw_outputs.numpy())


def load_gnn_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = GraphCountGCN(hidden_dim=int(cfg["model"]["baseline"]["hidden_dim"]), layers=int(cfg["model"]["baseline"]["layers"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_graph_samples(split_csv: Path):
    df = pd.read_csv(split_csv)
    samples = []
    for _, row in df.iterrows():
        graph = nx.read_graphml(row["graph_path"])
        data = graphml_to_pyg_data(graph)
        data.y = torch.tensor([[np.log1p(float(row["num_nodes"])), np.log1p(float(row["num_edges"]))]], dtype=torch.float32)
        samples.append(data)
    return samples


def predict_gnn(split_csv: Path, model):
    cfg = load_yaml_config()
    batch_size = int(cfg["model"]["cnn"]["batch_size"])
    samples = load_graph_samples(split_csv)
    loader = PyGDataLoader(samples, batch_size=batch_size, shuffle=False)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            preds = model(batch)
            outputs.append(preds.cpu())
    raw_outputs = torch.cat(outputs, dim=0)
    return safe_counts(raw_outputs.numpy())

def predict_graph_statistic(df: pd.DataFrame, density_table: dict, fallback_density: float) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        num_nodes = int(row["num_nodes"])
        pred_edges = predict_edges(num_nodes, row["generator"], density_table, fallback_density)
        preds.append((num_nodes, pred_edges))
    return np.array(preds, dtype=np.int64)


def maybe_alias_breakdown(name: str) -> str:
    return {"generator_type": "generator", "density_bucket": "density_bucket"}.get(name, name)


def save_metrics_table(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)


def main() -> None:
    results_dir = Path(config.DATA_DIR).parent / "evaluation" / "results"
    ensure_dir(results_dir)

    splits = {
        "test": load_split("test"),
        "held_out": load_split("held_out"),
    }

    cnn_checkpoint = Path(config.DATA_DIR).parent / "models" / "checkpoints" / "cnn_best.pt"
    gnn_checkpoint = Path(config.DATA_DIR).parent / "models" / "checkpoints" / "gnn_best.pt"
    if not cnn_checkpoint.exists():
        raise FileNotFoundError(f"Missing CNN checkpoint: {cnn_checkpoint}")
    if not gnn_checkpoint.exists():
        raise FileNotFoundError(f"Missing GNN checkpoint: {gnn_checkpoint}")

    model_cnn, normalize_stats = load_cnn_model(cnn_checkpoint)
    model_gnn = load_gnn_model(gnn_checkpoint)
    density_table, fallback_density = build_density_table(Path(config.DATA_DIR) / "splits" / "train.csv")

    cnn_tables = []
    gnn_tables = []
    baseline_tables = []

    for split_name, df in splits.items():
        cnn_preds = predict_cnn(Path(config.DATA_DIR) / "splits" / f"{split_name}.csv", normalize_stats, model_cnn)
        gnn_preds = predict_gnn(Path(config.DATA_DIR) / "splits" / f"{split_name}.csv", model_gnn)
        baseline_preds = predict_graph_statistic(df, density_table, fallback_density)

        cnn_tables.append(pd.DataFrame(metrics_records("cnn", split_name, df, cnn_preds)))
        gnn_tables.append(pd.DataFrame(metrics_records("gcn", split_name, df, gnn_preds)))
        baseline_tables.append(pd.DataFrame(metrics_records("graph_statistic", split_name, df, baseline_preds)))

    cnn_metrics = pd.concat(cnn_tables, ignore_index=True)
    gnn_metrics = pd.concat(gnn_tables, ignore_index=True)
    baseline_metrics = pd.concat(baseline_tables, ignore_index=True)

    save_metrics_table(cnn_metrics, results_dir / "cnn_metrics.csv")
    save_metrics_table(gnn_metrics, results_dir / "gnn_metrics.csv")
    save_metrics_table(baseline_metrics, results_dir / "graph_statistic_metrics.csv")

    summary = pd.concat([cnn_metrics, gnn_metrics, baseline_metrics], ignore_index=True)
    summary_wide = (
        summary.pivot_table(
            index=["split", "breakdown", "group_value", "target", "metric"],
            columns="model",
            values="value",
        )
        .reset_index()
        .sort_values(["split", "breakdown", "group_value", "target", "metric"])
    )
    summary_wide.columns.name = None
    save_metrics_table(summary_wide, results_dir / "summary_comparison.csv")
    print(f"Saved evaluation tables to {results_dir}")


if __name__ == "__main__":
    main()
