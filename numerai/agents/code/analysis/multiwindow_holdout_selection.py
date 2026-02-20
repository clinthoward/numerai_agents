from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics


AGENTS_DIR = Path(__file__).resolve().parents[2]
NUMERAI_DIR = AGENTS_DIR.parent
REPO_ROOT = NUMERAI_DIR.parent
DEFAULT_EXPERIMENT_DIR = AGENTS_DIR / "experiments" / "arrowstreet_integration"
DEFAULT_CANDIDATE_RUNS = [
    "d02_confirm_ranked128_twostage_w050_full",
    "g11_60e03_40d02",
    "g12_50e03_50d02",
    "g13_40e03_60d02",
    "g14_30e03_70d02",
]
DEFAULT_BENCHMARK_MODEL = "v52_lgbm_ender20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-window trailing holdout robustness checks for precomputed blend candidates."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Path to experiment directory containing predictions/.",
    )
    parser.add_argument(
        "--candidate-runs",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_RUNS),
        help="Comma-separated run names (prediction parquet stems).",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default="d02_confirm_ranked128_twostage_w050_full",
        help="Baseline run for deltas and fallback slot.",
    )
    parser.add_argument(
        "--benchmark-model",
        type=str,
        default=DEFAULT_BENCHMARK_MODEL,
        help="Benchmark model column used for BMC scoring.",
    )
    parser.add_argument(
        "--benchmark-data-path",
        type=str,
        default="v5.2/full_benchmark_models.parquet",
        help="Path to benchmark models parquet.",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=3,
        help="Number of trailing era windows.",
    )
    parser.add_argument(
        "--window-size-eras",
        type=int,
        default=192,
        help="Era count per holdout window.",
    )
    parser.add_argument(
        "--corr-floor-delta",
        type=float,
        default=-0.0010,
        help="Preferred worst-window corr delta floor for primary balanced slot.",
    )
    parser.add_argument(
        "--corr-hard-fail-delta",
        type=float,
        default=-0.0015,
        help="Hard worst-window corr delta floor for secondary slot.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="v7_multiwindow_robustness",
        help="Prefix for output artifacts.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] in {NUMERAI_DIR.name, AGENTS_DIR.name}:
        return (REPO_ROOT / candidate).resolve()
    return (NUMERAI_DIR / candidate).resolve()


def _sort_eras(eras: pd.Series) -> list[Any]:
    def _key(value: Any) -> tuple[int, str]:
        try:
            return (0, f"{int(value):012d}")
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted(pd.unique(eras), key=_key)


def _evaluate_prediction(
    df: pd.DataFrame,
    pred_col: str,
    benchmark_col: str,
    *,
    target_col: str = "target",
    era_col: str = "era",
) -> dict[str, float]:
    per_corr = numerai_metrics.per_era_corr(df, [pred_col], target_col, era_col)[pred_col]
    per_bmc = numerai_metrics.per_era_bmc(
        df, [pred_col], benchmark_col, target_col, era_col
    )[pred_col]
    corr_summary = numerai_metrics.score_summary(per_corr)
    bmc_summary = numerai_metrics.score_summary(per_bmc)
    bmc_last = numerai_metrics.score_summary(per_bmc.tail(200))
    return {
        "corr_mean": float(corr_summary["mean"]),
        "corr_sharpe": float(corr_summary["sharpe"]),
        "bmc_mean": float(bmc_summary["mean"]),
        "bmc_sharpe": float(bmc_summary["sharpe"]),
        "bmc_last200_mean": float(bmc_last["mean"]),
        "bmc_last200_sharpe": float(bmc_last["sharpe"]),
    }


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    numeric_cols = [
        col for col in display.columns if pd.api.types.is_numeric_dtype(display[col])
    ]
    for col in numeric_cols:
        display[col] = display[col].map(lambda v: f"{v:.6f}")
    cols = list(display.columns)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in cols) + " |"
        for _, row in display.iterrows()
    ]
    path.write_text("\n".join([header, separator, *rows]) + "\n", encoding="utf-8")


def _load_candidates(predictions_dir: Path, runs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    loaded: list[str] = []
    for run in runs:
        path = predictions_dir / f"{run}.parquet"
        if not path.exists():
            continue
        cols = numerai_metrics._parquet_columns(path)  # noqa: SLF001
        required = [col for col in ["id", "era", "target", "prediction"] if col in cols]
        if set(required) != {"id", "era", "target", "prediction"}:
            continue
        frame = pd.read_parquet(path, columns=required).rename(columns={"prediction": run})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[["id", "era", run]], on=["id", "era"], how="inner")
        loaded.append(run)
    if merged is None or not loaded:
        raise FileNotFoundError("No compatible candidate prediction files found.")
    return merged, loaded


def _build_windows(eras: list[Any], num_windows: int, window_size: int) -> list[list[Any]]:
    windows: list[list[Any]] = []
    n = len(eras)
    for i in range(num_windows):
        end = n - i * window_size
        start = max(0, end - window_size)
        if start >= end:
            break
        window = eras[start:end]
        if not window:
            continue
        windows.append(window)
    return windows


def main() -> None:
    args = parse_args()
    experiment_dir = _resolve_repo_path(args.experiment_dir)
    predictions_dir = experiment_dir / "predictions"
    candidate_runs = [item.strip() for item in args.candidate_runs.split(",") if item.strip()]
    baseline_run = args.baseline_run

    base_df, loaded_runs = _load_candidates(predictions_dir, candidate_runs)
    if baseline_run not in loaded_runs:
        baseline_run = loaded_runs[0]

    benchmark_path = _resolve_repo_path(args.benchmark_data_path)
    benchmark, benchmark_col = numerai_metrics.load_benchmark_predictions_from_path(
        benchmark_path,
        args.benchmark_model,
        era_col="era",
        id_col="id",
    )
    base_df = numerai_metrics.attach_benchmark_predictions(
        base_df, benchmark, benchmark_col, era_col="era", id_col="id"
    )

    eras = _sort_eras(base_df["era"])
    windows = _build_windows(
        eras,
        num_windows=max(1, int(args.num_windows)),
        window_size=max(1, int(args.window_size_eras)),
    )
    if not windows:
        raise ValueError("No windows were generated; adjust num-windows/window-size-eras.")

    window_rows: list[dict[str, Any]] = []
    for w_idx, window_eras in enumerate(windows):
        window_df = base_df[base_df["era"].isin(window_eras)].copy()
        metrics_by_run: dict[str, dict[str, float]] = {}
        for run in loaded_runs:
            metrics_by_run[run] = _evaluate_prediction(window_df, run, benchmark_col)
        base_metrics = metrics_by_run[baseline_run]
        for run in loaded_runs:
            m = metrics_by_run[run]
            window_rows.append(
                {
                    "window": f"w{w_idx}",
                    "window_eras": len(window_eras),
                    "window_first_era": str(window_eras[0]),
                    "window_last_era": str(window_eras[-1]),
                    "run": run,
                    **m,
                    f"delta_corr_vs_{baseline_run}": m["corr_mean"] - base_metrics["corr_mean"],
                    f"delta_bmc_vs_{baseline_run}": m["bmc_mean"] - base_metrics["bmc_mean"],
                    f"delta_bmc200_vs_{baseline_run}": m["bmc_last200_mean"]
                    - base_metrics["bmc_last200_mean"],
                }
            )

    window_summary = pd.DataFrame(window_rows).sort_values(["window", "run"])
    delta_corr_col = f"delta_corr_vs_{baseline_run}"
    delta_bmc_col = f"delta_bmc200_vs_{baseline_run}"

    agg = (
        window_summary.groupby("run")
        .agg(
            mean_corr=("corr_mean", "mean"),
            mean_bmc=("bmc_mean", "mean"),
            mean_bmc200=("bmc_last200_mean", "mean"),
            std_corr=("corr_mean", "std"),
            std_bmc=("bmc_mean", "std"),
            std_bmc200=("bmc_last200_mean", "std"),
            mean_delta_corr=(delta_corr_col, "mean"),
            mean_delta_bmc200=(delta_bmc_col, "mean"),
            min_delta_corr=(delta_corr_col, "min"),
            min_delta_bmc200=(delta_bmc_col, "min"),
            positive_bmc200_windows=(delta_bmc_col, lambda s: int((s > 0).sum())),
            total_windows=(delta_bmc_col, "count"),
        )
        .reset_index()
    )
    agg["positive_bmc200_ratio"] = agg["positive_bmc200_windows"] / agg["total_windows"]
    agg = agg.sort_values("mean_delta_bmc200", ascending=False)

    promotions: list[dict[str, Any]] = []
    base_row = agg[agg["run"] == baseline_run].iloc[0]
    promotions.append(
        {
            "slot": "fallback_stable",
            "run": baseline_run,
            "mean_delta_bmc200": float(base_row["mean_delta_bmc200"]),
            "mean_delta_corr": float(base_row["mean_delta_corr"]),
            "min_delta_corr": float(base_row["min_delta_corr"]),
            "positive_bmc200_ratio": float(base_row["positive_bmc200_ratio"]),
            "rule": "baseline_control",
        }
    )

    non_base = agg[agg["run"] != baseline_run].copy()
    primary_pool = non_base[
        (non_base["min_delta_corr"] >= float(args.corr_floor_delta))
        & (non_base["min_delta_bmc200"] > 0.0)
    ].copy()
    if not primary_pool.empty:
        primary_pool = primary_pool.sort_values(
            ["mean_delta_bmc200", "mean_delta_corr"], ascending=[False, False]
        )
        row = primary_pool.iloc[0]
        promotions.append(
            {
                "slot": "primary_balanced",
                "run": row["run"],
                "mean_delta_bmc200": float(row["mean_delta_bmc200"]),
                "mean_delta_corr": float(row["mean_delta_corr"]),
                "min_delta_corr": float(row["min_delta_corr"]),
                "positive_bmc200_ratio": float(row["positive_bmc200_ratio"]),
                "rule": "all_windows_corr_floor_and_positive_bmc200",
            }
        )

    secondary_pool = non_base[
        (non_base["min_delta_corr"] >= float(args.corr_hard_fail_delta))
        & (non_base["mean_delta_bmc200"] > 0.0)
    ].copy()
    chosen = {row["run"] for row in promotions}
    secondary_pool = secondary_pool[~secondary_pool["run"].isin(chosen)]
    if not secondary_pool.empty:
        secondary_pool = secondary_pool.sort_values(
            ["mean_delta_bmc200", "mean_delta_corr"], ascending=[False, False]
        )
        row = secondary_pool.iloc[0]
        promotions.append(
            {
                "slot": "secondary_high_bmc",
                "run": row["run"],
                "mean_delta_bmc200": float(row["mean_delta_bmc200"]),
                "mean_delta_corr": float(row["mean_delta_corr"]),
                "min_delta_corr": float(row["min_delta_corr"]),
                "positive_bmc200_ratio": float(row["positive_bmc200_ratio"]),
                "rule": "max_mean_bmc200_with_hard_corr_guardrail",
            }
        )

    promotion_df = pd.DataFrame(promotions)
    desired_slots = ["primary_balanced", "secondary_high_bmc", "fallback_stable"]
    if not promotion_df.empty:
        promotion_df["slot"] = pd.Categorical(
            promotion_df["slot"], categories=desired_slots, ordered=True
        )
        promotion_df = promotion_df.sort_values("slot")

    summary_csv = experiment_dir / f"{args.output_prefix}_window_summary.csv"
    summary_md = experiment_dir / f"{args.output_prefix}_window_summary.md"
    aggregate_csv = experiment_dir / f"{args.output_prefix}_aggregate_summary.csv"
    aggregate_md = experiment_dir / f"{args.output_prefix}_aggregate_summary.md"
    promotion_csv = experiment_dir / f"{args.output_prefix}_promotion_table.csv"
    promotion_md = experiment_dir / f"{args.output_prefix}_promotion_table.md"

    window_summary.to_csv(summary_csv, index=False)
    agg.to_csv(aggregate_csv, index=False)
    promotion_df.to_csv(promotion_csv, index=False)
    _write_markdown_table(window_summary, summary_md)
    _write_markdown_table(agg, aggregate_md)
    _write_markdown_table(promotion_df, promotion_md)

    print("Loaded runs:", ",".join(loaded_runs))
    print("Windows:", ",".join([f"w{i}:{len(w)}" for i, w in enumerate(windows)]))
    print("Saved window summary CSV:", summary_csv)
    print("Saved aggregate summary CSV:", aggregate_csv)
    print("Saved promotion CSV:", promotion_csv)
    print(
        agg[
            [
                "run",
                "mean_delta_bmc200",
                "mean_delta_corr",
                "min_delta_corr",
                "positive_bmc200_ratio",
            ]
        ].to_string(index=False)
    )
    if not promotion_df.empty:
        print("Promotion slots:")
        print(promotion_df.to_string(index=False))


if __name__ == "__main__":
    main()
