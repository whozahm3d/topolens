"""Graph-statistic heuristic baseline: exact node count (structurally
trivial to observe) + a non-learned density-formula estimate for edges.

Edge predictions come from a per-generator average density, computed once
from the TRAIN split only, applied to the observed node count:
pred_edges ~= density * N * (N-1) / 2. The real edge count for the graph
being predicted is never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from topolens_utils import ensure_dir


def build_density_table(train_csv: Path) -> tuple[dict[str, float], float]:
    """Return {generator: mean_density} from TRAIN only, plus a global fallback.

    The fallback covers generators with no training rows (MUTAG, PROTEINS
    only ever appear in held_out).
    """
    train_df = pd.read_csv(train_csv)
    per_generator = train_df.groupby("generator")["density"].mean().to_dict()
    global_fallback = float(train_df["density"].mean())
    return per_generator, global_fallback


def predict_edges(num_nodes: int, generator: str, density_table: dict[str, float], fallback_density: float) -> int:
    """Estimate edge count from node count + a training-derived average density."""
    density = density_table.get(generator, fallback_density)
    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    return int(round(density * max_possible_edges))


def load_rows(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    data_dir = Path(config.DATA_DIR) / "splits"
    test_df = load_rows(data_dir / "test.csv")
    held_out_df = load_rows(data_dir / "held_out.csv")

    density_table, fallback_density = build_density_table(data_dir / "train.csv")

    rows = []
    for split_name, split_df in (("test", test_df), ("held_out", held_out_df)):
        for _, row in split_df.iterrows():
            num_nodes = int(row["num_nodes"])
            pred_edges = predict_edges(num_nodes, row["generator"], density_table, fallback_density)
            rows.append(
                {
                    "graph_id": row["graph_id"],
                    "pred_num_vertices": num_nodes,
                    "pred_num_edges": pred_edges,
                }
            )

    out_path = Path(config.DATA_DIR).parent / "evaluation" / "results" / "baseline_predictions.csv"
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved baseline predictions to {out_path}")


if __name__ == "__main__":
    main()