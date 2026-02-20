from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet

from agents.code.metrics import numerai_metrics


AGENTS_DIR = Path(__file__).resolve().parents[2]
NUMERAI_DIR = AGENTS_DIR.parent
REPO_ROOT = NUMERAI_DIR.parent
DEFAULT_EXPERIMENT_DIR = AGENTS_DIR / "experiments" / "arrowstreet_integration"
DEFAULT_CANDIDATE_RUNS = [
    "c00_scale_ranked128_full",
    "c01_scale_ranked128_twostage_w040_full",
    "d00_confirm_ranked128_twostage_w030_full",
    "d01_confirm_ranked128_twostage_w040_full",
    "d02_confirm_ranked128_twostage_w050_full",
]
DEFAULT_BENCHMARK_MODEL = "v52_lgbm_ender20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an OOF ensemble frontier from existing experiment predictions."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Path to experiment directory containing predictions/ and results/.",
    )
    parser.add_argument(
        "--candidate-runs",
        type=str,
        default=",".join(DEFAULT_CANDIDATE_RUNS),
        help="Comma-separated run names (prediction/result stems).",
    )
    parser.add_argument(
        "--benchmark-model",
        type=str,
        default=DEFAULT_BENCHMARK_MODEL,
        help="Benchmark model column name to use for BMC/corr benchmarking.",
    )
    parser.add_argument(
        "--benchmark-data-path",
        type=str,
        default=None,
        help="Optional path to benchmark models parquet. If omitted, inferred from results JSON.",
    )
    parser.add_argument(
        "--include-benchmark-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include benchmark model predictions as an additional ensemble candidate.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ensemble_frontier",
        help="Prefix for summary artifacts.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of latest eras reserved for era-block holdout validation.",
    )
    parser.add_argument(
        "--corr-floor-baseline-run",
        type=str,
        default="d02_confirm_ranked128_twostage_w050_full",
        help="Run name used as corr/BMC baseline for constrained blending and deltas.",
    )
    parser.add_argument(
        "--corr-floor-delta",
        type=float,
        default=-0.0010,
        help="Required corr delta floor vs baseline run for balanced promotion logic.",
    )
    parser.add_argument(
        "--corr-hard-fail-delta",
        type=float,
        default=-0.0015,
        help="Hard corr delta floor vs baseline run for high-BMC promotion logic.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic optimization fallbacks.",
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


def _rank_by_era(values: pd.Series, eras: pd.Series) -> np.ndarray:
    ranked = values.groupby(eras).rank(method="average", pct=True)
    return ranked.to_numpy(dtype=np.float64, copy=False)


def _simplex_optimize(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    corr_lambda: float = 0.0,
    benchmark: np.ndarray | None = None,
    corr_floor: float | None = None,
) -> np.ndarray:
    n_features = X.shape[1]
    if n_features == 1:
        return np.array([1.0], dtype=np.float64)

    y_centered = y - y.mean()
    y_ss = float(np.dot(y_centered, y_centered))
    bench_centered = None
    bench_ss = None
    if benchmark is not None:
        bench_centered = benchmark - benchmark.mean()
        bench_ss = float(np.dot(bench_centered, bench_centered))

    def _objective(weights: np.ndarray) -> float:
        pred = X @ weights
        err = y_centered - pred
        mse = float(np.mean(err * err))
        ridge = float(alpha * np.dot(weights, weights))
        corr_penalty = 0.0
        if corr_lambda > 0.0 and bench_centered is not None and bench_ss and bench_ss > 0:
            pred_centered = pred - pred.mean()
            pred_ss = float(np.dot(pred_centered, pred_centered))
            if pred_ss > 0.0:
                corr = float(np.dot(pred_centered, bench_centered) / np.sqrt(pred_ss * bench_ss))
                corr_penalty = corr_lambda * corr * corr
        return mse + ridge + corr_penalty

    def _target_corr(weights: np.ndarray) -> float:
        if y_ss <= 0.0:
            return -1.0
        pred = X @ weights
        pred_centered = pred - pred.mean()
        pred_ss = float(np.dot(pred_centered, pred_centered))
        if pred_ss <= 0.0:
            return -1.0
        return float(np.dot(pred_centered, y_centered) / np.sqrt(pred_ss * y_ss))

    x0 = np.full(n_features, 1.0 / n_features, dtype=np.float64)
    bounds = [(0.0, 1.0)] * n_features
    constraints: list[dict[str, Any]] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if corr_floor is not None:
        constraints.append({"type": "ineq", "fun": lambda w: _target_corr(w) - corr_floor})
    result = minimize(
        _objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not result.success:
        return x0
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return x0
    return weights / total


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


def _infer_benchmark_path(results_dir: Path, candidate_runs: list[str]) -> Path:
    for run in candidate_runs:
        path = results_dir / f"{run}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmark_file = payload.get("benchmark", {}).get("file")
        if benchmark_file:
            return _resolve_repo_path(benchmark_file)
    raise FileNotFoundError(
        "Could not infer benchmark data path from results JSON. "
        "Provide --benchmark-data-path."
    )


def _load_base_oof_frame(predictions_dir: Path, runs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    loaded_runs: list[str] = []
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
            # Keep a canonical target from the first loaded frame and merge
            # additional model predictions strictly on (id, era). This allows
            # blending models trained on different targets.
            merged = merged.merge(
                frame[["id", "era", run]],
                on=["id", "era"],
                how="inner",
            )
        loaded_runs.append(run)

    if merged is None or not loaded_runs:
        raise FileNotFoundError("No compatible prediction files found for candidate runs.")
    return merged, loaded_runs


def _load_run_metrics(results_dir: Path, runs: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        path = results_dir / f"{run}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        bmc = metrics.get("bmc", {})
        bmc_last = metrics.get("bmc_last_200_eras", {})
        corr = metrics.get("corr", {})
        out[run] = {
            "corr_mean": float(corr.get("mean", np.nan)),
            "bmc_mean": float(bmc.get("mean", np.nan)),
            "bmc_last200_mean": float(bmc_last.get("mean", np.nan)),
            "bmc_last200_sharpe": float(bmc_last.get("sharpe", np.nan)),
        }
    return out


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


def main() -> None:
    args = parse_args()
    experiment_dir = _resolve_repo_path(args.experiment_dir)
    predictions_dir = experiment_dir / "predictions"
    results_dir = experiment_dir / "results"
    weights_dir = experiment_dir / "ensemble_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    candidate_runs = [
        item.strip()
        for item in args.candidate_runs.split(",")
        if item.strip()
    ]
    base_df, loaded_runs = _load_base_oof_frame(predictions_dir, candidate_runs)
    if args.benchmark_data_path:
        benchmark_path = _resolve_repo_path(args.benchmark_data_path)
    else:
        benchmark_path = _infer_benchmark_path(results_dir, loaded_runs)
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

    baseline_col = f"baseline__{args.benchmark_model}"
    if args.include_benchmark_baseline:
        base_df[baseline_col] = base_df[benchmark_col].astype(np.float64, copy=False)

    # Evaluate each candidate directly on the canonical scoring target in base_df.
    candidate_metrics: dict[str, dict[str, float]] = {}
    for run in loaded_runs:
        candidate_metrics[run] = _evaluate_prediction(base_df, run, benchmark_col)
    if args.include_benchmark_baseline:
        candidate_metrics[baseline_col] = _evaluate_prediction(
            base_df, baseline_col, benchmark_col
        )

    era_order = _sort_eras(base_df["era"])
    if not era_order:
        raise ValueError("No eras available in merged base frame.")
    holdout_count = int(round(len(era_order) * float(args.holdout_fraction)))
    holdout_count = max(1, min(holdout_count, max(1, len(era_order) - 1)))
    train_cut = len(era_order) - holdout_count
    train_eras = set(era_order[:train_cut])
    holdout_eras = set(era_order[train_cut:])
    train_mask = base_df["era"].isin(train_eras).to_numpy()
    holdout_mask = base_df["era"].isin(holdout_eras).to_numpy()

    y_all = _rank_by_era(base_df["target"], base_df["era"])
    y_train = y_all[train_mask]

    ranked_cols: list[str] = []
    for run in loaded_runs:
        ranked_col = f"rank__{run}"
        base_df[ranked_col] = _rank_by_era(base_df[run], base_df["era"])
        ranked_cols.append(ranked_col)
    if args.include_benchmark_baseline:
        ranked_col = f"rank__{baseline_col}"
        base_df[ranked_col] = _rank_by_era(base_df[baseline_col], base_df["era"])
        ranked_cols.append(ranked_col)

    corr_floor_run = args.corr_floor_baseline_run
    if corr_floor_run not in loaded_runs:
        corr_floor_run = "d02_confirm_ranked128_twostage_w050_full"
    if corr_floor_run not in loaded_runs and loaded_runs:
        corr_floor_run = loaded_runs[0]
    corr_floor_rank_col = f"rank__{corr_floor_run}"
    if corr_floor_rank_col not in base_df.columns:
        corr_floor_rank_col = ranked_cols[0]
        corr_floor_run = corr_floor_rank_col.replace("rank__", "")
    baseline_rank_train = base_df[corr_floor_rank_col].to_numpy(dtype=np.float64, copy=False)[
        train_mask
    ]
    y_train_centered = y_train - y_train.mean()
    baseline_centered = baseline_rank_train - baseline_rank_train.mean()
    y_ss = float(np.dot(y_train_centered, y_train_centered))
    baseline_ss = float(np.dot(baseline_centered, baseline_centered))
    baseline_corr_train = 0.0
    if y_ss > 0.0 and baseline_ss > 0.0:
        baseline_corr_train = float(
            np.dot(y_train_centered, baseline_centered) / np.sqrt(y_ss * baseline_ss)
        )
    corr_floor_target = baseline_corr_train + float(args.corr_floor_delta)

    sortable_runs = sorted(
        loaded_runs,
        key=lambda r: candidate_metrics.get(r, {}).get("bmc_last200_mean", -np.inf),
        reverse=True,
    )
    top3_runs = sortable_runs[: min(3, len(sortable_runs))]
    top5_runs = sortable_runs[: min(5, len(sortable_runs))]

    blend_specs: list[tuple[str, list[str], np.ndarray]] = []

    # e00: equal-weight top3
    e00_cols = [f"rank__{run}" for run in top3_runs]
    e00_weights = np.full(len(e00_cols), 1.0 / len(e00_cols), dtype=np.float64)
    blend_specs.append(("e00_equal_weight_top3", e00_cols, e00_weights))

    # e01: non-negative ridge top5
    e01_cols = [f"rank__{run}" for run in top5_runs]
    X_e01 = base_df[e01_cols].to_numpy(dtype=np.float64, copy=False)
    w_e01 = _simplex_optimize(X_e01[train_mask], y_train, alpha=1e-3)
    blend_specs.append(("e01_nonneg_ridge_top5", e01_cols, w_e01))

    # e02: elastic-net top5
    X_e02 = X_e01
    enet = ElasticNet(
        alpha=5e-5,
        l1_ratio=0.7,
        fit_intercept=False,
        positive=True,
        max_iter=10000,
        random_state=args.random_state,
    )
    enet.fit(X_e02[train_mask], y_train)
    w_e02 = np.clip(np.asarray(enet.coef_, dtype=np.float64), 0.0, None)
    if w_e02.sum() <= 0:
        w_e02 = np.full(len(e01_cols), 1.0 / len(e01_cols), dtype=np.float64)
    else:
        w_e02 = w_e02 / w_e02.sum()
    blend_specs.append(("e02_elasticnet_top5", e01_cols, w_e02))

    # e03: penalized low-benchmark-corr blend
    e03_runs = list(top5_runs)
    if args.include_benchmark_baseline:
        e03_runs.append(baseline_col)
    e03_cols = [f"rank__{run}" for run in e03_runs]
    X_e03 = base_df[e03_cols].to_numpy(dtype=np.float64, copy=False)
    bench_train = base_df[f"rank__{baseline_col}"].to_numpy(dtype=np.float64, copy=False)[
        train_mask
    ] if args.include_benchmark_baseline else None
    w_e03 = _simplex_optimize(
        X_e03[train_mask],
        y_train,
        alpha=1e-3,
        corr_lambda=0.20,
        benchmark=bench_train,
    )
    blend_specs.append(("e03_penalized_low_benchmark_corr", e03_cols, w_e03))

    # e04: stability-weighted blend
    e04_runs = list(top5_runs)
    if args.include_benchmark_baseline:
        e04_runs.append(baseline_col)
    e04_cols = [f"rank__{run}" for run in e04_runs]
    e04_scores = []
    for run in e04_runs:
        stats = candidate_metrics.get(run, {})
        score = max(0.0, stats.get("bmc_last200_mean", 0.0)) * max(
            0.0, stats.get("bmc_last200_sharpe", 0.0)
        )
        e04_scores.append(score)
    w_e04 = np.asarray(e04_scores, dtype=np.float64)
    if w_e04.sum() <= 0:
        w_e04 = np.full(len(e04_cols), 1.0 / len(e04_cols), dtype=np.float64)
    else:
        w_e04 = w_e04 / w_e04.sum()
    blend_specs.append(("e04_stability_weighted_blend", e04_cols, w_e04))

    # e05: corr-floor constrained top5 blend (explicit corr guardrail).
    e05_cols = [f"rank__{run}" for run in top5_runs]
    X_e05 = base_df[e05_cols].to_numpy(dtype=np.float64, copy=False)
    w_e05 = _simplex_optimize(
        X_e05[train_mask],
        y_train,
        alpha=1e-3,
        corr_floor=corr_floor_target,
    )
    blend_specs.append(("e05_corr_floor_constrained_top5", e05_cols, w_e05))

    summary_rows: list[dict[str, Any]] = []
    summary_candidates = list(loaded_runs)
    if args.include_benchmark_baseline:
        summary_candidates.append(baseline_col)

    for run in summary_candidates:
        full_metrics = _evaluate_prediction(base_df, run, benchmark_col)
        train_metrics = _evaluate_prediction(base_df.loc[train_mask], run, benchmark_col)
        holdout_metrics = _evaluate_prediction(base_df.loc[holdout_mask], run, benchmark_col)
        summary_rows.append(
            {
                "run": run,
                "kind": "candidate",
                **full_metrics,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
                "models": run,
            }
        )

    for run_name, cols, weights in blend_specs:
        X = base_df[cols].to_numpy(dtype=np.float64, copy=False)
        pred = X @ weights
        pred_col = run_name
        base_df[pred_col] = pred.astype(np.float32)

        prediction_out = base_df[["id", "era", "target", pred_col]].rename(
            columns={pred_col: "prediction"}
        )
        prediction_out.to_parquet(predictions_dir / f"{run_name}.parquet", index=False)

        weight_map = {col.replace("rank__", ""): float(w) for col, w in zip(cols, weights)}
        (weights_dir / f"{run_name}.json").write_text(
            json.dumps(
                {
                    "run": run_name,
                    "weights": weight_map,
                    "train_era_count": len(train_eras),
                    "total_era_count": len(era_order),
                    "holdout_era_count": len(holdout_eras),
                    "corr_floor_baseline_run": corr_floor_run,
                    "corr_floor_target_train": corr_floor_target,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        metrics = _evaluate_prediction(base_df, pred_col, benchmark_col)
        train_metrics = _evaluate_prediction(base_df.loc[train_mask], pred_col, benchmark_col)
        holdout_metrics = _evaluate_prediction(
            base_df.loc[holdout_mask], pred_col, benchmark_col
        )
        summary_rows.append(
            {
                "run": run_name,
                "kind": "blend",
                **metrics,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"holdout_{k}": v for k, v in holdout_metrics.items()},
                "models": ";".join(weight_map.keys()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    baseline_row = summary[summary["run"] == corr_floor_run]
    if not baseline_row.empty:
        baseline = baseline_row.iloc[0]
        for prefix in ("", "train_", "holdout_"):
            corr_col = f"{prefix}corr_mean"
            bmc_col = f"{prefix}bmc_mean"
            bmc200_col = f"{prefix}bmc_last200_mean"
            if corr_col in summary.columns:
                summary[f"{prefix}delta_corr_vs_{corr_floor_run}"] = (
                    summary[corr_col] - float(baseline[corr_col])
                )
            if bmc_col in summary.columns:
                summary[f"{prefix}delta_bmc_vs_{corr_floor_run}"] = (
                    summary[bmc_col] - float(baseline[bmc_col])
                )
            if bmc200_col in summary.columns:
                summary[f"{prefix}delta_bmc200_vs_{corr_floor_run}"] = (
                    summary[bmc200_col] - float(baseline[bmc200_col])
                )

    summary = summary.sort_values("holdout_bmc_last200_mean", ascending=False)
    summary_csv = experiment_dir / f"{args.output_prefix}_summary.csv"
    summary_md = experiment_dir / f"{args.output_prefix}_summary.md"
    summary.to_csv(summary_csv, index=False)
    _write_markdown_table(summary, summary_md)

    delta_corr_col = f"holdout_delta_corr_vs_{corr_floor_run}"
    delta_bmc_col = f"holdout_delta_bmc200_vs_{corr_floor_run}"
    if delta_corr_col not in summary.columns:
        summary[delta_corr_col] = np.nan
    if delta_bmc_col not in summary.columns:
        summary[delta_bmc_col] = np.nan
    promotions: list[dict[str, Any]] = []

    fallback_run = corr_floor_run if corr_floor_run in set(summary["run"]) else summary.iloc[0]["run"]
    fallback_row = summary[summary["run"] == fallback_run].iloc[0]
    promotions.append(
        {
            "slot": "fallback_stable",
            "run": fallback_row["run"],
            "kind": fallback_row["kind"],
            "holdout_corr_mean": float(fallback_row.get("holdout_corr_mean", np.nan)),
            "holdout_bmc_last200_mean": float(
                fallback_row.get("holdout_bmc_last200_mean", np.nan)
            ),
            "delta_corr_vs_baseline": float(
                fallback_row.get(delta_corr_col, 0.0)
            ),
            "delta_bmc200_vs_baseline": float(
                fallback_row.get(delta_bmc_col, 0.0)
            ),
            "rule": "baseline_control",
        }
    )

    non_baseline = summary[summary["run"] != fallback_run].copy()
    primary_pool = non_baseline[
        (non_baseline[delta_corr_col] >= float(args.corr_floor_delta))
        & (non_baseline[delta_bmc_col] > 0.0)
    ].copy()
    if not primary_pool.empty:
        primary_pool = primary_pool.sort_values(
            [delta_bmc_col, "holdout_corr_mean"], ascending=[False, False]
        )
        primary = primary_pool.iloc[0]
        promotions.append(
            {
                "slot": "primary_balanced",
                "run": primary["run"],
                "kind": primary["kind"],
                "holdout_corr_mean": float(primary.get("holdout_corr_mean", np.nan)),
                "holdout_bmc_last200_mean": float(
                    primary.get("holdout_bmc_last200_mean", np.nan)
                ),
                "delta_corr_vs_baseline": float(primary.get(delta_corr_col, np.nan)),
                "delta_bmc200_vs_baseline": float(primary.get(delta_bmc_col, np.nan)),
                "rule": "corr_floor_constrained",
            }
        )

    secondary_pool = non_baseline[
        (non_baseline[delta_corr_col] >= float(args.corr_hard_fail_delta))
        & (non_baseline[delta_bmc_col] > 0.0)
    ].copy()
    if not secondary_pool.empty:
        chosen_runs = {row["run"] for row in promotions}
        secondary_pool = secondary_pool[~secondary_pool["run"].isin(chosen_runs)]
        if not secondary_pool.empty:
            secondary_pool = secondary_pool.sort_values(
                [delta_bmc_col, "holdout_corr_mean"], ascending=[False, False]
            )
            secondary = secondary_pool.iloc[0]
            promotions.append(
                {
                    "slot": "secondary_high_bmc",
                    "run": secondary["run"],
                    "kind": secondary["kind"],
                    "holdout_corr_mean": float(secondary.get("holdout_corr_mean", np.nan)),
                    "holdout_bmc_last200_mean": float(
                        secondary.get("holdout_bmc_last200_mean", np.nan)
                    ),
                    "delta_corr_vs_baseline": float(
                        secondary.get(delta_corr_col, np.nan)
                    ),
                    "delta_bmc200_vs_baseline": float(
                        secondary.get(delta_bmc_col, np.nan)
                    ),
                    "rule": "max_bmc_with_hard_corr_guardrail",
                }
            )

    promotion_df = pd.DataFrame(promotions)
    desired_slots = ["primary_balanced", "secondary_high_bmc", "fallback_stable"]
    if not promotion_df.empty:
        promotion_df["slot"] = pd.Categorical(
            promotion_df["slot"], categories=desired_slots, ordered=True
        )
        promotion_df = promotion_df.sort_values("slot")
    promotion_csv = experiment_dir / f"{args.output_prefix}_promotion_table.csv"
    promotion_md = experiment_dir / f"{args.output_prefix}_promotion_table.md"
    promotion_df.to_csv(promotion_csv, index=False)
    _write_markdown_table(promotion_df, promotion_md)

    print("Saved frontier summary CSV:", summary_csv)
    print("Saved frontier summary MD:", summary_md)
    print("Saved promotion CSV:", promotion_csv)
    print("Saved promotion MD:", promotion_md)
    print(
        summary[
            [
                "run",
                "kind",
                "holdout_bmc_last200_mean",
                "holdout_bmc_mean",
                "holdout_corr_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
