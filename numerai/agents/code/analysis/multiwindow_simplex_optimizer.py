from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.code.metrics import numerai_metrics


AGENTS_DIR = Path(__file__).resolve().parents[2]
NUMERAI_DIR = AGENTS_DIR.parent
REPO_ROOT = NUMERAI_DIR.parent
DEFAULT_EXPERIMENT_DIR = AGENTS_DIR / "experiments" / "arrowstreet_integration"
DEFAULT_BASELINE_RUN = "d02_confirm_ranked128_twostage_w050_full"
DEFAULT_SPECIALIST_RUNS = [
    "f10_full_target_main_orth_beta075",
    "f01_full_target_main_orth_twostage",
]
DEFAULT_COMPARISON_RUNS = [
    "g11_60e03_40d02",
    "g12_50e03_50d02",
    "g13_40e03_60d02",
    "g14_30e03_70d02",
    "h20_multiwindow_opt_primary_balanced_f10_full_target_main_orth_beta075_w008",
]
DEFAULT_BENCHMARK_MODEL = "v52_lgbm_ender20"
DEFAULT_LINEAGE_MANIFEST = (
    DEFAULT_EXPERIMENT_DIR / "blend_lineage_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Constrained multi-window simplex optimization for baseline + "
            "multiple specialist heads."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Path to experiment directory containing predictions/.",
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default=DEFAULT_BASELINE_RUN,
        help="Baseline run name (prediction parquet stem).",
    )
    parser.add_argument(
        "--specialist-runs",
        type=str,
        default=",".join(DEFAULT_SPECIALIST_RUNS),
        help="Comma-separated specialist run names to blend with baseline.",
    )
    parser.add_argument(
        "--comparison-runs",
        type=str,
        default=",".join(DEFAULT_COMPARISON_RUNS),
        help="Optional comma-separated runs to compare against optimized blends.",
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
        help="Number of trailing windows.",
    )
    parser.add_argument(
        "--window-size-eras",
        type=int,
        default=192,
        help="Era count per trailing window.",
    )
    parser.add_argument(
        "--window-spec-path",
        type=str,
        default=None,
        help=(
            "Optional explicit window spec path (.json/.csv). "
            "If set, overrides --num-windows and --window-size-eras."
        ),
    )
    parser.add_argument(
        "--coarse-step",
        type=float,
        default=0.05,
        help="Coarse simplex step size.",
    )
    parser.add_argument(
        "--max-coarse-points",
        type=int,
        default=0,
        help=(
            "Optional cap on coarse simplex points. "
            "0 means auto (uncapped for <=2 specialists, capped for >=3 specialists)."
        ),
    )
    parser.add_argument(
        "--enable-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run local refinement around coarse selection seeds.",
    )
    parser.add_argument(
        "--refine-step",
        type=float,
        default=0.02,
        help="Local refinement step size.",
    )
    parser.add_argument(
        "--refine-radius",
        type=float,
        default=0.08,
        help="Local refinement radius around each seed weight.",
    )
    parser.add_argument(
        "--max-refine-points",
        type=int,
        default=0,
        help=(
            "Optional cap on refinement points. "
            "0 means auto (uncapped for <=2 specialists, capped for >=3 specialists)."
        ),
    )
    parser.add_argument(
        "--corr-floor-delta",
        type=float,
        default=-0.0010,
        help="Primary worst-window corr delta floor vs baseline.",
    )
    parser.add_argument(
        "--corr-hard-fail-delta",
        type=float,
        default=-0.0015,
        help="Secondary worst-window corr delta floor vs baseline.",
    )
    parser.add_argument(
        "--corr-delta-std-max",
        type=float,
        default=0.0010,
        help="Maximum std of corr deltas across windows.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="v9_multiwindow_simplex",
        help="Prefix for output artifacts.",
    )
    parser.add_argument(
        "--save-selected-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save parquet predictions for selected slots.",
    )
    parser.add_argument(
        "--lineage-manifest-path",
        type=str,
        default=str(DEFAULT_LINEAGE_MANIFEST),
        help="Path to CSV append-only lineage manifest.",
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


def _load_runs(
    predictions_dir: Path, runs: list[str]
) -> tuple[pd.DataFrame, list[str], dict[str, Path]]:
    merged: pd.DataFrame | None = None
    loaded: list[str] = []
    run_paths: dict[str, Path] = {}
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
        run_paths[run] = path
    if merged is None or not loaded:
        raise FileNotFoundError("No compatible candidate prediction files found.")
    return merged, loaded, run_paths


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


def _match_era_value(value: Any, era_set: set[Any]) -> Any:
    if value in era_set:
        return value
    candidates = [value]
    try:
        candidates.append(int(value))
    except (TypeError, ValueError):
        pass
    try:
        candidates.append(str(int(value)))
    except (TypeError, ValueError):
        pass
    candidates.append(str(value))
    for candidate in candidates:
        if candidate in era_set:
            return candidate
    raise ValueError(f"Era '{value}' not found in available eras.")


def _load_windows_from_spec(path: Path, all_eras: list[Any]) -> list[list[Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Window spec file not found: {path}")
    era_set = set(all_eras)
    suffix = path.suffix.lower()
    windows: list[list[Any]] = []
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if "era" not in frame.columns:
            raise ValueError("CSV window spec must include an 'era' column.")
        group_col = "window"
        if group_col not in frame.columns:
            group_col = "window_id" if "window_id" in frame.columns else None
        if group_col is None:
            frame = frame.copy()
            frame["window"] = "w0"
            group_col = "window"
        for _, group in frame.groupby(group_col, sort=False):
            window = [_match_era_value(v, era_set) for v in group["era"].tolist()]
            window = [era for era in all_eras if era in set(window)]
            if window:
                windows.append(window)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_windows = payload["windows"] if isinstance(payload, dict) and "windows" in payload else payload
        if not isinstance(raw_windows, list):
            raise ValueError("JSON window spec must be a list or {'windows': [...]} object.")
        for item in raw_windows:
            if isinstance(item, dict):
                if "eras" in item:
                    raw_eras = item["eras"]
                    if not isinstance(raw_eras, list):
                        raise ValueError("Window 'eras' must be a list.")
                    window = [_match_era_value(v, era_set) for v in raw_eras]
                elif "start_era" in item and "end_era" in item:
                    start = _match_era_value(item["start_era"], era_set)
                    end = _match_era_value(item["end_era"], era_set)
                    start_idx = all_eras.index(start)
                    end_idx = all_eras.index(end)
                    lo, hi = sorted((start_idx, end_idx))
                    window = all_eras[lo : hi + 1]
                else:
                    raise ValueError(
                        "JSON dict windows must include either 'eras' or 'start_era'+'end_era'."
                    )
            elif isinstance(item, list):
                window = [_match_era_value(v, era_set) for v in item]
            else:
                raise ValueError("Window entries must be lists or dicts.")
            window = [era for era in all_eras if era in set(window)]
            if window:
                windows.append(window)
    else:
        raise ValueError("Unsupported window spec format. Use .json or .csv.")
    if not windows:
        raise ValueError("Window spec produced zero non-empty windows.")
    return windows


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _infer_prediction_prefix(output_prefix: str) -> str:
    # Maps v8->h20, v9->h30, v10->h40 style naming used in this experiment line.
    tag = output_prefix.split("_", 1)[0]
    if tag.startswith("v"):
        try:
            major = int(tag[1:])
            if major >= 8:
                return f"h{(major - 6) * 10}_multiwindow_simplex"
        except ValueError:
            pass
    return "h30_multiwindow_simplex"


def _window_spec_summary(windows: list[list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, window in enumerate(windows):
        out.append(
            {
                "window": f"w{idx}",
                "count": len(window),
                "first_era": str(window[0]),
                "last_era": str(window[-1]),
            }
        )
    return out


def _grid_values(step: float) -> np.ndarray:
    if step <= 0.0:
        return np.array([0.0, 1.0], dtype=np.float64)
    raw = np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)
    vals = np.unique(np.clip(np.round(raw, 8), 0.0, 1.0))
    if vals[0] != 0.0:
        vals = np.insert(vals, 0, 0.0)
    if vals[-1] != 1.0:
        vals = np.append(vals, 1.0)
    return vals


def _simplex_grid(n_dims: int, step: float) -> list[np.ndarray]:
    vals = _grid_values(step)
    out: list[np.ndarray] = []

    def _recurse(prefix: list[float], depth: int, remaining: float) -> None:
        if depth == n_dims:
            out.append(np.array(prefix, dtype=np.float64))
            return
        max_val = remaining
        for value in vals:
            if value > max_val + 1e-12:
                break
            _recurse(prefix + [float(value)], depth + 1, remaining - float(value))

    _recurse([], 0, 1.0)
    return out


def _local_simplex_grid(seed: np.ndarray, step: float, radius: float) -> list[np.ndarray]:
    n_dims = int(seed.shape[0])
    vals = _grid_values(step)
    candidates: list[list[float]] = []
    for i in range(n_dims):
        lo = max(0.0, float(seed[i] - radius))
        hi = min(1.0, float(seed[i] + radius))
        v = [float(x) for x in vals if lo - 1e-12 <= x <= hi + 1e-12]
        if not v:
            v = [float(seed[i])]
        candidates.append(v)

    out: list[np.ndarray] = []

    def _recurse(prefix: list[float], depth: int, running_sum: float) -> None:
        if depth == n_dims:
            if running_sum <= 1.0 + 1e-12:
                out.append(np.array(prefix, dtype=np.float64))
            return
        for value in candidates[depth]:
            nxt = running_sum + value
            if nxt > 1.0 + 1e-12:
                continue
            _recurse(prefix + [float(value)], depth + 1, nxt)

    _recurse([], 0, 0.0)
    return out


def _weight_key(weights: np.ndarray) -> tuple[float, ...]:
    return tuple(float(np.round(x, 8)) for x in weights.tolist())


def _auto_max_points(n_dims: int, explicit: int, stage: str) -> int | None:
    if explicit > 0:
        return explicit
    if n_dims <= 2:
        return None
    if stage == "coarse":
        return 120
    if stage == "refine":
        return 180
    return None


def _downsample_weight_vectors(
    weights: list[np.ndarray],
    *,
    max_points: int | None,
    force_include: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    if max_points is None or max_points <= 0 or len(weights) <= max_points:
        return weights
    by_key: dict[tuple[float, ...], np.ndarray] = {}
    for w in weights:
        by_key[_weight_key(w)] = w
    ordered = list(by_key.values())

    selected: list[np.ndarray] = []
    selected_keys: set[tuple[float, ...]] = set()
    for w in force_include or []:
        key = _weight_key(w)
        if key in by_key and key not in selected_keys:
            selected.append(by_key[key])
            selected_keys.add(key)
    remaining = [w for w in ordered if _weight_key(w) not in selected_keys]
    slots = max(0, max_points - len(selected))
    if slots <= 0:
        return selected[:max_points]
    if len(remaining) <= slots:
        return selected + remaining
    idx = np.linspace(0, len(remaining) - 1, num=slots, dtype=int)
    for i in idx.tolist():
        w = remaining[i]
        key = _weight_key(w)
        if key in selected_keys:
            continue
        selected.append(w)
        selected_keys.add(key)
    if len(selected) < max_points:
        for w in remaining:
            key = _weight_key(w)
            if key in selected_keys:
                continue
            selected.append(w)
            selected_keys.add(key)
            if len(selected) >= max_points:
                break
    return selected[:max_points]


def _select_best(
    df: pd.DataFrame,
    *,
    corr_floor: float,
    corr_std_max: float,
) -> pd.Series | None:
    eligible = df[
        (df["min_delta_corr"] >= corr_floor)
        & (df["std_delta_corr"] <= corr_std_max)
        & (df["min_delta_bmc200"] > 0.0)
    ].copy()
    if eligible.empty:
        return None
    eligible = eligible.sort_values(
        ["mean_delta_bmc200", "mean_delta_corr"],
        ascending=[False, False],
    )
    return eligible.iloc[0]


def _run_tag(run_name: str) -> str:
    return run_name.split("_", 1)[0]


def main() -> None:
    args = parse_args()
    experiment_dir = _resolve_repo_path(args.experiment_dir)
    predictions_dir = experiment_dir / "predictions"
    manifest_path = _resolve_repo_path(args.lineage_manifest_path)

    specialist_runs = [item.strip() for item in args.specialist_runs.split(",") if item.strip()]
    if not specialist_runs:
        raise ValueError("At least one specialist run is required.")
    comparison_runs = [item.strip() for item in args.comparison_runs.split(",") if item.strip()]

    run_list = [args.baseline_run, *specialist_runs, *comparison_runs]
    deduped_runs = list(dict.fromkeys(run_list))
    base_df, loaded_runs, run_paths = _load_runs(predictions_dir, deduped_runs)

    if args.baseline_run not in loaded_runs:
        raise ValueError(f"Baseline run not loaded: {args.baseline_run}")
    missing_specialists = [run for run in specialist_runs if run not in loaded_runs]
    if missing_specialists:
        raise ValueError(f"Specialist runs not loaded: {','.join(missing_specialists)}")
    loaded_comparison = [run for run in comparison_runs if run in loaded_runs]
    optimization_runs = [args.baseline_run, *specialist_runs]
    optimization_paths = {run: run_paths[run] for run in optimization_runs}
    optimization_sha = {run: _sha256_file(path) for run, path in optimization_paths.items()}

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
    if args.window_spec_path:
        window_spec_path = _resolve_repo_path(args.window_spec_path)
        windows = _load_windows_from_spec(window_spec_path, eras)
    else:
        window_spec_path = None
        windows = _build_windows(
            eras,
            num_windows=max(1, int(args.num_windows)),
            window_size=max(1, int(args.window_size_eras)),
        )
    if not windows:
        raise ValueError("No windows were generated; adjust num-windows/window-size-eras.")

    window_frames: list[pd.DataFrame] = []
    baseline_by_window: list[dict[str, float]] = []
    for window_eras in windows:
        window_df = base_df[base_df["era"].isin(window_eras)].copy()
        window_frames.append(window_df)
        baseline_by_window.append(
            _evaluate_prediction(window_df, args.baseline_run, benchmark_col)
        )

    seen: set[tuple[float, ...]] = set()
    grid_rows: list[dict[str, Any]] = []

    def _evaluate_weight_vectors(weights_list: list[np.ndarray], stage: str) -> None:
        for spec_weights in weights_list:
            key = _weight_key(spec_weights)
            if key in seen:
                continue
            seen.add(key)
            specialist_sum = float(np.sum(spec_weights))
            baseline_weight = float(1.0 - specialist_sum)
            if baseline_weight < -1e-12:
                continue
            baseline_weight = max(0.0, baseline_weight)

            bmc_deltas: list[float] = []
            corr_deltas: list[float] = []
            corr_vals: list[float] = []
            bmc_vals: list[float] = []
            bmc200_vals: list[float] = []

            for w_idx, window_df in enumerate(window_frames):
                baseline_metrics = baseline_by_window[w_idx]
                blend_raw = baseline_weight * window_df[args.baseline_run]
                for run, weight in zip(specialist_runs, spec_weights):
                    if float(weight) > 0.0:
                        blend_raw = blend_raw + float(weight) * window_df[run]

                ranked_blend = _rank_by_era(blend_raw, window_df["era"])
                eval_df = window_df[["id", "era", "target", benchmark_col]].copy()
                eval_df["prediction"] = ranked_blend
                blend_metrics = _evaluate_prediction(eval_df, "prediction", benchmark_col)

                bmc_delta = (
                    blend_metrics["bmc_last200_mean"] - baseline_metrics["bmc_last200_mean"]
                )
                corr_delta = blend_metrics["corr_mean"] - baseline_metrics["corr_mean"]
                bmc_deltas.append(float(bmc_delta))
                corr_deltas.append(float(corr_delta))
                corr_vals.append(float(blend_metrics["corr_mean"]))
                bmc_vals.append(float(blend_metrics["bmc_mean"]))
                bmc200_vals.append(float(blend_metrics["bmc_last200_mean"]))

            row: dict[str, Any] = {
                "stage": stage,
                "weight_baseline": baseline_weight,
                "mean_corr": float(np.mean(corr_vals)),
                "mean_bmc": float(np.mean(bmc_vals)),
                "mean_bmc200": float(np.mean(bmc200_vals)),
                "mean_delta_corr": float(np.mean(corr_deltas)),
                "mean_delta_bmc200": float(np.mean(bmc_deltas)),
                "min_delta_corr": float(np.min(corr_deltas)),
                "min_delta_bmc200": float(np.min(bmc_deltas)),
                "std_delta_corr": float(np.std(corr_deltas, ddof=0)),
                "positive_bmc200_windows": int(np.sum(np.asarray(bmc_deltas) > 0.0)),
                "total_windows": int(len(bmc_deltas)),
                "positive_bmc200_ratio": float(
                    np.sum(np.asarray(bmc_deltas) > 0.0) / max(1, len(bmc_deltas))
                ),
            }
            for run, weight in zip(specialist_runs, spec_weights):
                row[f"weight_{_run_tag(run)}"] = float(weight)
            grid_rows.append(row)

    n_specialists = len(specialist_runs)
    coarse_weights_full = _simplex_grid(n_specialists, float(args.coarse_step))
    force_points = [np.zeros(n_specialists, dtype=np.float64)]
    for idx in range(n_specialists):
        point = np.zeros(n_specialists, dtype=np.float64)
        point[idx] = 1.0
        force_points.append(point)
    coarse_weights = _downsample_weight_vectors(
        coarse_weights_full,
        max_points=_auto_max_points(n_specialists, int(args.max_coarse_points), "coarse"),
        force_include=force_points,
    )
    _evaluate_weight_vectors(coarse_weights, stage="coarse")
    grid_df = pd.DataFrame(grid_rows)
    coarse_df = grid_df.copy()

    seed_rows: list[pd.Series] = []
    if not coarse_df.empty:
        seed_rows.append(
            coarse_df.sort_values(
                ["mean_delta_bmc200", "mean_delta_corr"], ascending=[False, False]
            ).iloc[0]
        )
        coarse_primary = _select_best(
            coarse_df,
            corr_floor=float(args.corr_floor_delta),
            corr_std_max=float(args.corr_delta_std_max),
        )
        if coarse_primary is not None:
            seed_rows.append(coarse_primary)
        coarse_secondary = _select_best(
            coarse_df,
            corr_floor=float(args.corr_hard_fail_delta),
            corr_std_max=float(args.corr_delta_std_max),
        )
        if coarse_secondary is not None:
            seed_rows.append(coarse_secondary)

    refine_eval_count = 0
    if bool(args.enable_refine) and seed_rows:
        refine_weights: list[np.ndarray] = []
        for row in seed_rows:
            seed = np.array([float(row[f"weight_{_run_tag(run)}"]) for run in specialist_runs])
            refine_weights.extend(
                _local_simplex_grid(
                    seed,
                    step=float(args.refine_step),
                    radius=float(args.refine_radius),
                )
            )
        refine_weights = _downsample_weight_vectors(
            refine_weights,
            max_points=_auto_max_points(
                n_specialists, int(args.max_refine_points), "refine"
            ),
            force_include=[np.array([float(row[f"weight_{_run_tag(run)}"]) for run in specialist_runs]) for row in seed_rows],
        )
        refine_eval_count = len(refine_weights)
        _evaluate_weight_vectors(refine_weights, stage="refine")

    if not grid_rows:
        raise ValueError("No candidate simplex weights were evaluated.")

    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values("mean_delta_bmc200", ascending=False).reset_index(drop=True)
    best_unconstrained = grid_df.iloc[0]
    best_primary = _select_best(
        grid_df,
        corr_floor=float(args.corr_floor_delta),
        corr_std_max=float(args.corr_delta_std_max),
    )
    best_secondary = _select_best(
        grid_df,
        corr_floor=float(args.corr_hard_fail_delta),
        corr_std_max=float(args.corr_delta_std_max),
    )

    selected_rows: list[dict[str, Any]] = []

    def _selection_row(slot: str, row: pd.Series, rule: str) -> dict[str, Any]:
        out = {
            "slot": slot,
            "stage": row["stage"],
            "mean_delta_bmc200": float(row["mean_delta_bmc200"]),
            "mean_delta_corr": float(row["mean_delta_corr"]),
            "min_delta_corr": float(row["min_delta_corr"]),
            "std_delta_corr": float(row["std_delta_corr"]),
            "weight_baseline": float(row["weight_baseline"]),
            "rule": rule,
        }
        for run in specialist_runs:
            out[f"weight_{_run_tag(run)}"] = float(row[f"weight_{_run_tag(run)}"])
        return out

    selected_rows.append(
        _selection_row(
            "best_unconstrained",
            best_unconstrained,
            "max_mean_delta_bmc200",
        )
    )
    if best_primary is not None:
        selected_rows.append(
            _selection_row(
                "primary_balanced",
                best_primary,
                "strict_corr_floor_and_corr_vol",
            )
        )
    if best_secondary is not None:
        duplicate = False
        if best_primary is not None:
            duplicate = _weight_key(
                np.array(
                    [float(best_primary[f"weight_{_run_tag(run)}"]) for run in specialist_runs]
                )
            ) == _weight_key(
                np.array(
                    [float(best_secondary[f"weight_{_run_tag(run)}"]) for run in specialist_runs]
                )
            )
        if not duplicate:
            selected_rows.append(
                _selection_row(
                    "secondary_high_bmc",
                    best_secondary,
                    "hard_corr_floor_and_corr_vol",
                )
            )

    selected_df = pd.DataFrame(selected_rows)

    comparison_rows: list[dict[str, Any]] = []
    for run in [args.baseline_run, *loaded_comparison]:
        corr_deltas: list[float] = []
        bmc_deltas: list[float] = []
        corr_vals: list[float] = []
        bmc_vals: list[float] = []
        bmc200_vals: list[float] = []
        for w_idx, window_df in enumerate(window_frames):
            baseline_metrics = baseline_by_window[w_idx]
            metrics = _evaluate_prediction(window_df, run, benchmark_col)
            corr_delta = metrics["corr_mean"] - baseline_metrics["corr_mean"]
            bmc_delta = metrics["bmc_last200_mean"] - baseline_metrics["bmc_last200_mean"]
            corr_deltas.append(float(corr_delta))
            bmc_deltas.append(float(bmc_delta))
            corr_vals.append(float(metrics["corr_mean"]))
            bmc_vals.append(float(metrics["bmc_mean"]))
            bmc200_vals.append(float(metrics["bmc_last200_mean"]))
        comparison_rows.append(
            {
                "run": run,
                "mean_corr": float(np.mean(corr_vals)),
                "mean_bmc": float(np.mean(bmc_vals)),
                "mean_bmc200": float(np.mean(bmc200_vals)),
                "mean_delta_corr": float(np.mean(corr_deltas)),
                "mean_delta_bmc200": float(np.mean(bmc_deltas)),
                "min_delta_corr": float(np.min(corr_deltas)),
                "min_delta_bmc200": float(np.min(bmc_deltas)),
                "std_delta_corr": float(np.std(corr_deltas, ddof=0)),
                "positive_bmc200_windows": int(np.sum(np.asarray(bmc_deltas) > 0.0)),
                "total_windows": int(len(bmc_deltas)),
                "positive_bmc200_ratio": float(
                    np.sum(np.asarray(bmc_deltas) > 0.0) / max(1, len(bmc_deltas))
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "mean_delta_bmc200", ascending=False
    )

    output_grid_csv = experiment_dir / f"{args.output_prefix}_grid_summary.csv"
    output_grid_md = experiment_dir / f"{args.output_prefix}_grid_summary.md"
    output_selected_csv = experiment_dir / f"{args.output_prefix}_selection_table.csv"
    output_selected_md = experiment_dir / f"{args.output_prefix}_selection_table.md"
    output_compare_csv = experiment_dir / f"{args.output_prefix}_comparison_table.csv"
    output_compare_md = experiment_dir / f"{args.output_prefix}_comparison_table.md"

    grid_df.to_csv(output_grid_csv, index=False)
    selected_df.to_csv(output_selected_csv, index=False)
    comparison_df.to_csv(output_compare_csv, index=False)
    _write_markdown_table(grid_df, output_grid_md)
    _write_markdown_table(selected_df, output_selected_md)
    _write_markdown_table(comparison_df, output_compare_md)

    saved_predictions: list[tuple[str, str, Path]] = []
    manifest_rows: list[dict[str, Any]] = []
    prediction_prefix = _infer_prediction_prefix(args.output_prefix)
    git_head = _get_git_head()
    window_spec = _window_spec_summary(windows)
    constraints = {
        "corr_floor_delta": float(args.corr_floor_delta),
        "corr_hard_fail_delta": float(args.corr_hard_fail_delta),
        "corr_delta_std_max": float(args.corr_delta_std_max),
        "coarse_step": float(args.coarse_step),
        "refine_step": float(args.refine_step),
        "refine_radius": float(args.refine_radius),
        "enable_refine": bool(args.enable_refine),
        "window_spec_path": str(window_spec_path) if window_spec_path else None,
        "num_windows": int(args.num_windows),
        "window_size_eras": int(args.window_size_eras),
    }
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    if bool(args.save_selected_predictions):
        selection_map = {
            row["slot"]: row
            for _, row in selected_df.iterrows()
            if row["slot"] in {"primary_balanced", "secondary_high_bmc"}
        }
        for slot, row in selection_map.items():
            blend = base_df[["id", "era", "target"]].copy()
            baseline_weight = float(row["weight_baseline"])
            blend_raw = baseline_weight * base_df[args.baseline_run]
            code_parts = [f"{_run_tag(args.baseline_run)}{int(round(baseline_weight * 100)):03d}"]
            for run in specialist_runs:
                w = float(row[f"weight_{_run_tag(run)}"])
                if w > 0.0:
                    blend_raw = blend_raw + w * base_df[run]
                code_parts.append(f"{_run_tag(run)}{int(round(w * 100)):03d}")
            blend["prediction"] = _rank_by_era(blend_raw, base_df["era"])
            run_name = f"{prediction_prefix}_{slot}_{'_'.join(code_parts)}"
            out_path = predictions_dir / f"{run_name}.parquet"
            blend.to_parquet(out_path, index=False)
            saved_predictions.append((slot, run_name, out_path))
            weights_payload = {"baseline": baseline_weight}
            for run in specialist_runs:
                weights_payload[run] = float(row[f"weight_{_run_tag(run)}"])
            manifest_rows.append(
                {
                    "timestamp_utc": timestamp_utc,
                    "output_run_name": run_name,
                    "baseline_run": args.baseline_run,
                    "specialist_runs": json.dumps(specialist_runs, separators=(",", ":")),
                    "weights_json": json.dumps(weights_payload, separators=(",", ":")),
                    "input_prediction_paths": json.dumps(
                        {run: str(path) for run, path in optimization_paths.items()},
                        separators=(",", ":"),
                    ),
                    "input_sha256_json": json.dumps(optimization_sha, separators=(",", ":")),
                    "window_spec": json.dumps(window_spec, separators=(",", ":")),
                    "constraints_json": json.dumps(constraints, separators=(",", ":")),
                    "script_path": str(Path(__file__).resolve()),
                    "git_head": git_head or "",
                }
            )
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        append_mode = manifest_path.exists()
        manifest_df.to_csv(
            manifest_path,
            mode="a" if append_mode else "w",
            header=not append_mode,
            index=False,
        )

    print("Loaded runs:", ",".join(loaded_runs))
    print("Specialist runs:", ",".join(specialist_runs))
    print("Windows:", ",".join([f"w{i}:{len(w)}" for i, w in enumerate(windows)]))
    print("Coarse simplex points (generated):", len(coarse_weights_full))
    print("Coarse simplex points (evaluated):", len(coarse_weights))
    print("Refine simplex points (evaluated):", refine_eval_count)
    print("Evaluated simplex points:", len(grid_df))
    print("Saved grid summary CSV:", output_grid_csv)
    print("Saved selection CSV:", output_selected_csv)
    print("Saved comparison CSV:", output_compare_csv)
    if manifest_rows:
        print("Appended lineage rows:", len(manifest_rows))
        print("Lineage manifest path:", manifest_path)
    if saved_predictions:
        for slot, run_name, out_path in saved_predictions:
            print(f"Saved selected prediction ({slot}): {run_name} -> {out_path}")
    print("Top grid rows by mean_delta_bmc200:")
    print(
        grid_df.head(10)[
            [
                "stage",
                "weight_baseline",
                *[f"weight_{_run_tag(run)}" for run in specialist_runs],
                "mean_delta_bmc200",
                "mean_delta_corr",
                "min_delta_corr",
                "std_delta_corr",
            ]
        ].to_string(index=False)
    )
    print("Selection table:")
    print(selected_df.to_string(index=False))
    print("Comparison table:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
