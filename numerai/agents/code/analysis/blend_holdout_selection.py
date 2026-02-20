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
    "e03_penalized_low_benchmark_corr",
    "g11_60e03_40d02",
    "g12_50e03_50d02",
    "g13_40e03_60d02",
    "g14_30e03_70d02",
]
DEFAULT_BENCHMARK_MODEL = "v52_lgbm_ender20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate era-block holdout Corr/BMC trade-offs for precomputed blend candidates "
            "and emit promotion slots."
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
        help="Baseline run for delta calculations and fallback slot.",
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
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of latest eras used as holdout.",
    )
    parser.add_argument(
        "--corr-floor-delta",
        type=float,
        default=-0.0010,
        help="Preferred corr delta floor vs baseline for primary balanced slot.",
    )
    parser.add_argument(
        "--corr-hard-fail-delta",
        type=float,
        default=-0.0015,
        help="Hard corr floor vs baseline for high-BMC slot.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="v6_holdout_selection",
        help="Prefix for output summary and promotion files.",
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
    per_bench_corr = numerai_metrics.per_era_pred_corr(
        df, [pred_col], benchmark_col, era_col
    )[pred_col]
    corr_summary = numerai_metrics.score_summary(per_corr)
    bmc_summary = numerai_metrics.score_summary(per_bmc)
    bmc_last = numerai_metrics.score_summary(per_bmc.tail(200))
    return {
        "corr_mean": float(corr_summary["mean"]),
        "corr_sharpe": float(corr_summary["sharpe"]),
        "bmc_mean": float(bmc_summary["mean"]),
        "bmc_sharpe": float(bmc_summary["sharpe"]),
        "bmc_avg_corr_benchmark": float(per_bench_corr.mean()),
        "bmc_last200_mean": float(bmc_last["mean"]),
        "bmc_last200_sharpe": float(bmc_last["sharpe"]),
    }


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    numeric_cols = [
        col
        for col in display.columns
        if pd.api.types.is_numeric_dtype(display[col])
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


def _load_candidates(
    predictions_dir: Path,
    runs: list[str],
) -> tuple[pd.DataFrame, list[str]]:
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
        frame = pd.read_parquet(path, columns=required)
        frame = frame.rename(columns={"prediction": run})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame[["id", "era", run]], on=["id", "era"], how="inner")
        loaded.append(run)
    if merged is None or not loaded:
        raise FileNotFoundError("No compatible prediction files found for requested runs.")
    return merged, loaded


def _append_deltas(summary: pd.DataFrame, baseline_run: str) -> pd.DataFrame:
    baseline_row = summary[summary["run"] == baseline_run]
    if baseline_row.empty:
        return summary
    baseline = baseline_row.iloc[0]
    out = summary.copy()
    for prefix in ("", "train_", "holdout_"):
        corr_col = f"{prefix}corr_mean"
        bmc_col = f"{prefix}bmc_mean"
        bmc200_col = f"{prefix}bmc_last200_mean"
        out[f"{prefix}delta_corr_vs_{baseline_run}"] = out[corr_col] - float(baseline[corr_col])
        out[f"{prefix}delta_bmc_vs_{baseline_run}"] = out[bmc_col] - float(baseline[bmc_col])
        out[f"{prefix}delta_bmc200_vs_{baseline_run}"] = (
            out[bmc200_col] - float(baseline[bmc200_col])
        )
    return out


def _build_promotion_table(
    summary: pd.DataFrame,
    *,
    baseline_run: str,
    corr_floor_delta: float,
    corr_hard_fail_delta: float,
) -> pd.DataFrame:
    if baseline_run not in set(summary["run"]):
        baseline_run = str(summary.iloc[0]["run"])
    delta_corr_col = f"holdout_delta_corr_vs_{baseline_run}"
    delta_bmc_col = f"holdout_delta_bmc200_vs_{baseline_run}"

    rows: list[dict[str, Any]] = []
    baseline = summary[summary["run"] == baseline_run].iloc[0]
    rows.append(
        {
            "slot": "fallback_stable",
            "run": baseline["run"],
            "holdout_corr_mean": float(baseline["holdout_corr_mean"]),
            "holdout_bmc_last200_mean": float(baseline["holdout_bmc_last200_mean"]),
            "delta_corr_vs_baseline": float(baseline[delta_corr_col]),
            "delta_bmc200_vs_baseline": float(baseline[delta_bmc_col]),
            "rule": "baseline_control",
        }
    )

    non_baseline = summary[summary["run"] != baseline_run].copy()

    primary_pool = non_baseline[
        (non_baseline[delta_corr_col] >= corr_floor_delta)
        & (non_baseline[delta_bmc_col] > 0.0)
    ].copy()
    if not primary_pool.empty:
        primary_pool = primary_pool.sort_values(
            [delta_bmc_col, "holdout_corr_mean"], ascending=[False, False]
        )
        primary = primary_pool.iloc[0]
        rows.append(
            {
                "slot": "primary_balanced",
                "run": primary["run"],
                "holdout_corr_mean": float(primary["holdout_corr_mean"]),
                "holdout_bmc_last200_mean": float(primary["holdout_bmc_last200_mean"]),
                "delta_corr_vs_baseline": float(primary[delta_corr_col]),
                "delta_bmc200_vs_baseline": float(primary[delta_bmc_col]),
                "rule": "corr_floor_constrained",
            }
        )

    secondary_pool = non_baseline[
        (non_baseline[delta_corr_col] >= corr_hard_fail_delta)
        & (non_baseline[delta_bmc_col] > 0.0)
    ].copy()
    if not secondary_pool.empty:
        chosen_runs = {row["run"] for row in rows}
        secondary_pool = secondary_pool[~secondary_pool["run"].isin(chosen_runs)]
        if not secondary_pool.empty:
            secondary_pool = secondary_pool.sort_values(
                [delta_bmc_col, "holdout_corr_mean"], ascending=[False, False]
            )
            secondary = secondary_pool.iloc[0]
            rows.append(
                {
                    "slot": "secondary_high_bmc",
                    "run": secondary["run"],
                    "holdout_corr_mean": float(secondary["holdout_corr_mean"]),
                    "holdout_bmc_last200_mean": float(
                        secondary["holdout_bmc_last200_mean"]
                    ),
                    "delta_corr_vs_baseline": float(secondary[delta_corr_col]),
                    "delta_bmc200_vs_baseline": float(secondary[delta_bmc_col]),
                    "rule": "max_bmc_with_hard_corr_guardrail",
                }
            )

    promotion = pd.DataFrame(rows)
    desired_slots = ["primary_balanced", "secondary_high_bmc", "fallback_stable"]
    if not promotion.empty:
        promotion["slot"] = pd.Categorical(
            promotion["slot"], categories=desired_slots, ordered=True
        )
        promotion = promotion.sort_values("slot")
    return promotion


def main() -> None:
    args = parse_args()
    experiment_dir = _resolve_repo_path(args.experiment_dir)
    predictions_dir = experiment_dir / "predictions"
    candidate_runs = [item.strip() for item in args.candidate_runs.split(",") if item.strip()]

    base_df, loaded_runs = _load_candidates(predictions_dir, candidate_runs)
    benchmark_path = _resolve_repo_path(args.benchmark_data_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark data file not found: {benchmark_path}")

    benchmark, benchmark_col = numerai_metrics.load_benchmark_predictions_from_path(
        benchmark_path,
        args.benchmark_model,
        era_col="era",
        id_col="id",
    )
    base_df = numerai_metrics.attach_benchmark_predictions(
        base_df,
        benchmark,
        benchmark_col,
        era_col="era",
        id_col="id",
    )

    era_order = _sort_eras(base_df["era"])
    holdout_count = int(round(len(era_order) * float(args.holdout_fraction)))
    holdout_count = max(1, min(holdout_count, max(1, len(era_order) - 1)))
    train_cut = len(era_order) - holdout_count
    train_eras = set(era_order[:train_cut])
    holdout_eras = set(era_order[train_cut:])
    train_df = base_df[base_df["era"].isin(train_eras)].copy()
    holdout_df = base_df[base_df["era"].isin(holdout_eras)].copy()

    summary_rows: list[dict[str, Any]] = []
    for run in loaded_runs:
        full_metrics = _evaluate_prediction(base_df, run, benchmark_col)
        train_metrics = _evaluate_prediction(train_df, run, benchmark_col)
        holdout_metrics = _evaluate_prediction(holdout_df, run, benchmark_col)
        summary_rows.append(
            {
                "run": run,
                **full_metrics,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
                "train_eras": len(train_eras),
                "holdout_eras": len(holdout_eras),
            }
        )

    summary = pd.DataFrame(summary_rows)
    baseline_run = args.baseline_run if args.baseline_run in set(summary["run"]) else loaded_runs[0]
    summary = _append_deltas(summary, baseline_run)
    summary = summary.sort_values("holdout_bmc_last200_mean", ascending=False)

    summary_csv = experiment_dir / f"{args.output_prefix}_summary.csv"
    summary_md = experiment_dir / f"{args.output_prefix}_summary.md"
    summary.to_csv(summary_csv, index=False)
    _write_markdown_table(summary, summary_md)

    promotion = _build_promotion_table(
        summary,
        baseline_run=baseline_run,
        corr_floor_delta=float(args.corr_floor_delta),
        corr_hard_fail_delta=float(args.corr_hard_fail_delta),
    )
    promotion_csv = experiment_dir / f"{args.output_prefix}_promotion_table.csv"
    promotion_md = experiment_dir / f"{args.output_prefix}_promotion_table.md"
    promotion.to_csv(promotion_csv, index=False)
    _write_markdown_table(promotion, promotion_md)

    print("Loaded runs:", ",".join(loaded_runs))
    print("Saved summary CSV:", summary_csv)
    print("Saved summary MD:", summary_md)
    print("Saved promotion CSV:", promotion_csv)
    print("Saved promotion MD:", promotion_md)
    print(
        summary[
            ["run", "holdout_bmc_last200_mean", "holdout_bmc_mean", "holdout_corr_mean"]
        ].to_string(index=False)
    )
    if not promotion.empty:
        print("Promotion slots:")
        print(promotion[["slot", "run", "delta_bmc200_vs_baseline", "delta_corr_vs_baseline"]].to_string(index=False))


if __name__ == "__main__":
    main()
