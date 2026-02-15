"""Summarize OOF prediction files with benchmark-style corr metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agents.code.metrics import numerai_metrics

AGENTS_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark-style summary: mean/std/sharpe/max_drawdown."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=AGENTS_DIR / "results",
        help="Directory containing results JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Experiment output directory (uses <output-dir>/results).",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="prediction",
        help="Prediction column in parquet files.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Target column in parquet files.",
    )
    parser.add_argument(
        "--era-col",
        type=str,
        default="era",
        help="Era column in parquet files.",
    )
    return parser.parse_args()


def _resolve_predictions_path(results_dir: Path, results_file: Path) -> Path:
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    rel_path = payload.get("output", {}).get("predictions_file")
    if rel_path:
        candidate = (results_dir.parent / rel_path).resolve()
        if candidate.exists():
            return candidate
    fallback = results_dir.parent / "predictions" / f"{results_file.stem}.parquet"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Predictions file not found for {results_file.name}: {fallback}"
        )
    return fallback


def _summarize_predictions(
    predictions_path: Path,
    pred_col: str,
    target_col: str,
    era_col: str,
) -> dict[str, float]:
    cols = [era_col, target_col, pred_col]
    df = pd.read_parquet(predictions_path, columns=cols)
    per_era = numerai_metrics.per_era_corr(df, [pred_col], target_col, era_col=era_col)
    summary = numerai_metrics.summarize_scores(per_era).loc[pred_col]
    return {
        "mean": float(summary["mean"]),
        "std": float(summary["std"]),
        "sharpe": float(summary["sharpe"]),
        "max_drawdown": float(summary["max_drawdown"]),
    }


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if args.output_dir is not None:
        results_dir = args.output_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows = []
    for results_file in sorted(results_dir.glob("*.json")):
        predictions_path = _resolve_predictions_path(results_dir, results_file)
        summary = _summarize_predictions(
            predictions_path,
            pred_col=args.pred_col,
            target_col=args.target_col,
            era_col=args.era_col,
        )
        summary["model"] = results_file.stem
        rows.append(summary)

    if not rows:
        print(f"No results found in {results_dir}")
        return

    out = pd.DataFrame(rows).set_index("model")
    print("\n=== BENCHMARK SUMMARY ===")
    print(out)


if __name__ == "__main__":
    main()
