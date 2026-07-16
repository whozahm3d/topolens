"""Train the Topolens GCN baseline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from topolens_utils import ensure_dir, load_yaml_config, set_random_seeds

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as exc:  # pragma: no cover - import guard for environments without PyG
    raise ImportError("torch_geometric is required for models/gnn_baseline.py") from exc


def graphml_to_pyg_data(graph: nx.Graph) -> Data:
    """Convert a NetworkX graph to a PyG Data object with degree features."""
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    degrees = torch.tensor([[float(graph.degree(node))] for node in nodes], dtype=torch.float32)
    degrees = degrees / (degrees.max() + 1e-6)

    edges = []
    for source, target in graph.edges():
        src_idx = node_to_idx[source]
        tgt_idx = node_to_idx[target]
        edges.append((src_idx, tgt_idx))
        edges.append((tgt_idx, src_idx))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=degrees, edge_index=edge_index, num_nodes=num_nodes)


def load_graph_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        graph = nx.read_graphml(row["graph_path"])
        data = graphml_to_pyg_data(graph)
        data.y = torch.tensor(
            [[np.log1p(float(row["num_nodes"])), np.log1p(float(row["num_edges"]))]],
            dtype=torch.float32,
        )
        data.graph_id = row["graph_id"]
        samples.append(data)
    return samples


def build_subset(samples, size: int, seed: int):
    size = min(size, len(samples))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    return [samples[i] for i in indices[:size].tolist()]


class GraphCountGCN(nn.Module):
    def __init__(self, hidden_dim: int = 64, layers: int = 3, output_dim: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = 1
        for layer_idx in range(layers):
            out_channels = hidden_dim
            self.convs.append(GCNConv(in_channels, out_channels))
            self.norms.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0

    for data in loader:
        data = data.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(data)
        loss = criterion(outputs, data.y)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = data.y.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_loss: float, cfg) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_val_loss": val_loss,
            "architecture": "gcn",
            "config": cfg,
        },
        path,
    )


def build_loaders(smoke_test: bool, seed: int):
    cfg = load_yaml_config()
    batch_size = int(cfg["model"]["cnn"]["batch_size"])

    train_csv = Path(config.DATA_DIR) / "splits" / "train.csv"
    val_csv = Path(config.DATA_DIR) / "splits" / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Split CSVs are missing. Run data/make_splits.py first.")

    train_samples = load_graph_dataset(train_csv)
    val_samples = load_graph_dataset(val_csv)

    if smoke_test:
        train_samples = build_subset(train_samples, 40, seed)
        val_samples = build_subset(val_samples, 10, seed + 1)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Topolens GCN baseline.")
    parser.add_argument("--smoke-test", action="store_true", help="Train on a tiny deterministic subset for 2 epochs.")
    args = parser.parse_args()

    cfg = load_yaml_config()
    seed = int(cfg["project"]["seed"])
    set_random_seeds(seed)

    if cfg["model"]["baseline"]["type"].lower() != "gcn":
        raise ValueError("config.yaml model.baseline.type must be 'gcn' for this script.")

    train_loader, val_loader = build_loaders(args.smoke_test, seed)
    model = GraphCountGCN(hidden_dim=int(cfg["model"]["baseline"]["hidden_dim"]), layers=int(cfg["model"]["baseline"]["layers"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["model"]["cnn"]["learning_rate"]))
    epochs = 2 if args.smoke_test else int(cfg["model"]["baseline"]["epochs"])

    checkpoint_path = Path(config.DATA_DIR).parent / "models" / "checkpoints" / "gnn_best.pt"
    log_path = Path(config.DATA_DIR).parent / "evaluation" / "results" / "gnn_training_log.csv"
    ensure_dir(log_path.parent)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    training_start = time.perf_counter()
    best_val_loss = float("inf")
    rows = []

    print(f"GCN parameter count: {model.num_parameters():,}")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        epoch_seconds = time.perf_counter() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_path, model, epoch, best_val_loss, cfg)

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
        save_checkpoint(checkpoint_path, model, epochs, best_val_loss, cfg)

    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Total training time: {training_seconds:.1f}s")


if __name__ == "__main__":
    main()
