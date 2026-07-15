"""Direct graph-statistic baseline that predicts counts from GraphML."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import networkx as nx

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from topolens_utils import ensure_dir


def predict_from_graphml(graph_path: str | Path) -> tuple[int, int]:
    """Return the exact node and edge counts from a GraphML file."""
    graph = nx.read_graphml(graph_path)
    return graph.number_of_nodes(), graph.number_of_edges()


def load_rows(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    data_dir = Path(config.DATA_DIR) / "splits"
    test_df = load_rows(data_dir / "test.csv")
    held_out_df = load_rows(data_dir / "held_out.csv")

    rows = []
    for split_name, split_df in (("test", test_df), ("held_out", held_out_df)):
        for _, row in split_df.iterrows():
            pred_nodes, pred_edges = predict_from_graphml(row["graph_path"])
            rows.append(
                {
                    "graph_id": row["graph_id"],
                    "pred_num_vertices": pred_nodes,
                    "pred_num_edges": pred_edges,
                }
            )

    out_path = Path(config.DATA_DIR).parent / "evaluation" / "results" / "baseline_predictions.csv"
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved baseline predictions to {out_path}")


if __name__ == "__main__":
    main()
