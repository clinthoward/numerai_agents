#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
NUMERAI_ROOT = REPO_ROOT / "numerai"
if str(NUMERAI_ROOT) not in sys.path:
    sys.path.insert(0, str(NUMERAI_ROOT))

from agents.code.modeling.models.arrowstreet_regressor import (  # noqa: E402
    ArrowstreetRegressor,
)
import agents.code.modeling.models.arrowstreet_components as arrow_components  # noqa: E402
import agents.code.modeling.models.arrowstreet_regressor as arrow_regressor  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    config_name: str
    weight: float


DEFAULT_SPECS = (
    ModelSpec(config_name="f10_full_target_main_orth_beta075", weight=0.70),
    ModelSpec(config_name="d02_confirm_ranked128_twostage_w050_full", weight=0.30),
)
DEFAULT_BLEND_NAME = "g00_70f10_30d02_upload"
DEFAULT_BENCHMARK_COL = "v52_lgbm_ender20"
DEFAULT_ERA_COL = "era"
DEFAULT_ID_COL = "id"
DEFAULT_TARGET_COL = "target"

CONFIGS_DIR = SCRIPT_PATH.parent / "configs"
UPLOAD_ARTIFACTS_DIR = SCRIPT_PATH.parent / "upload_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Numerai Classic upload pickle for Arrowstreet."
    )
    parser.add_argument(
        "--blend-name",
        type=str,
        default=DEFAULT_BLEND_NAME,
        help="Name used in artifact filenames and metadata.",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.2",
        help="Numerai dataset version directory under numerai/.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="medium",
        help="Feature set key from features.json.",
    )
    parser.add_argument(
        "--benchmark-col",
        type=str,
        default=DEFAULT_BENCHMARK_COL,
        help="Benchmark model column used for stage2 orthogonalization.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help=(
            "Model spec in the form '<config_name>:<weight>'. "
            "Repeat for multi-model blends, e.g. "
            "--spec f10_full_target_main_orth_beta075:0.7 --spec d02_confirm_ranked128_twostage_w050_full:0.3. "
            "If omitted, defaults are used."
        ),
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=None,
        help="Output pickle path. Defaults to upload_artifacts/<blend_name>.pkl",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=None,
        help="Output metadata json path. Defaults to upload_artifacts/<blend_name>.json",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick smoke builds. Omit for full-data training.",
    )
    parser.add_argument(
        "--stage2-lgbm-n-jobs",
        type=int,
        default=None,
        help=(
            "Optional override for stage2_lgbm_params['n_jobs'] in each config. "
            "Use small values (e.g., 1-2) to reduce memory pressure."
        ),
    )
    return parser.parse_args()


def parse_model_specs(spec_args: list[str] | None) -> list[ModelSpec]:
    if not spec_args:
        return list(DEFAULT_SPECS)
    specs: list[ModelSpec] = []
    for raw in spec_args:
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            config_name, weight_str = item.split(":", 1)
            config_name = config_name.strip()
            try:
                weight = float(weight_str.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid spec weight: '{raw}'") from exc
        else:
            config_name = item
            weight = 1.0
        if not config_name:
            raise ValueError(f"Invalid spec config name: '{raw}'")
        if not np.isfinite(weight):
            raise ValueError(f"Invalid non-finite spec weight: '{raw}'")
        specs.append(ModelSpec(config_name=config_name, weight=weight))
    if not specs:
        raise ValueError("At least one --spec value is required when --spec is provided.")
    return specs


def load_config(config_name: str) -> dict[str, Any]:
    config_path = CONFIGS_DIR / f"{config_name}.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    spec = importlib.util.spec_from_file_location(config_name, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must define a CONFIG dict.")
    return config


def load_feature_list(data_version: str, feature_set: str) -> list[str]:
    features_path = NUMERAI_ROOT / data_version / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing features metadata: {features_path}. Build/download datasets first."
        )
    with features_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    feature_sets = metadata.get("feature_sets", {})
    if feature_set not in feature_sets:
        raise KeyError(
            f"feature_set '{feature_set}' not found in {features_path}. "
            f"Available: {sorted(feature_sets.keys())[:10]}..."
        )
    return list(feature_sets[feature_set])


def _normalize_id_column(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col in df.columns:
        return df
    if df.index.name == id_col:
        return df.reset_index()
    reset = df.reset_index()
    if id_col in reset.columns:
        return reset
    if "index" in reset.columns:
        reset = reset.rename(columns={"index": id_col})
        return reset
    raise ValueError(f"Could not recover id column '{id_col}' from dataframe.")


def load_training_frame(
    *,
    data_version: str,
    feature_set: str,
    benchmark_col: str,
    era_col: str,
    id_col: str,
    target_col: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    features = load_feature_list(data_version, feature_set)
    full_path = NUMERAI_ROOT / data_version / "full.parquet"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full dataset: {full_path}")

    base_columns = [id_col, era_col, target_col, *features]
    full = pd.read_parquet(full_path, columns=base_columns)
    full = _normalize_id_column(full, id_col)

    benchmark_path = NUMERAI_ROOT / data_version / "full_benchmark_models.parquet"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing full benchmark models: {benchmark_path}")

    bench_cols = [id_col, era_col, benchmark_col]
    try:
        benchmark = pd.read_parquet(benchmark_path, columns=bench_cols)
    except Exception:
        benchmark = pd.read_parquet(benchmark_path)
    benchmark = _normalize_id_column(benchmark, id_col)
    if benchmark_col not in benchmark.columns:
        raise KeyError(f"Benchmark column '{benchmark_col}' not found in {benchmark_path}")
    keep_cols = [id_col, benchmark_col]
    if era_col in benchmark.columns:
        keep_cols.insert(1, era_col)
    benchmark = benchmark[keep_cols]

    full_indexed = full.set_index(id_col)
    benchmark_indexed = benchmark.set_index(id_col)
    common_ids = full_indexed.index.intersection(benchmark_indexed.index, sort=False)
    if common_ids.empty:
        raise ValueError("No overlapping ids between full data and benchmark models.")
    full_indexed = full_indexed.loc[common_ids]
    benchmark_indexed = benchmark_indexed.loc[common_ids]

    if era_col in benchmark_indexed.columns:
        full_eras = full_indexed[era_col].astype(str).to_numpy()
        bench_eras = benchmark_indexed[era_col].astype(str).to_numpy()
        if not np.array_equal(full_eras, bench_eras):
            raise ValueError("Era mismatch between full data and benchmark models.")

    full_indexed[benchmark_col] = (
        pd.to_numeric(benchmark_indexed[benchmark_col], errors="coerce")
        .fillna(0.0)
        .astype("float32")
    )
    full = full_indexed.reset_index()

    if max_rows is not None:
        full = full.iloc[: int(max_rows)].copy()

    return full, features


def make_predict_function(
    *,
    models: list[ArrowstreetRegressor],
    weights: list[float],
    feature_cols: list[str],
    era_col: str,
    benchmark_col: str,
):
    feature_cols = list(feature_cols)
    weights_arr = np.asarray(weights, dtype=np.float32)
    model_refs = tuple(models)

    def predict(
        live_features: pd.DataFrame,
        live_benchmark_models: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(live_features, pd.DataFrame):
            raise TypeError("live_features must be a pandas DataFrame.")
        if era_col not in live_features.columns:
            raise ValueError(f"live_features is missing required column '{era_col}'.")
        missing_features = [col for col in feature_cols if col not in live_features.columns]
        if missing_features:
            sample = missing_features[:5]
            suffix = "..." if len(missing_features) > 5 else ""
            raise ValueError(f"Missing required features: {sample}{suffix}")

        X_live = live_features[feature_cols + [era_col]].copy()
        if benchmark_col in live_features.columns:
            benchmark_series = live_features[benchmark_col]
        else:
            if live_benchmark_models is None:
                raise ValueError(
                    "live_benchmark_models is required when benchmark column is not present in live_features."
                )
            if benchmark_col not in live_benchmark_models.columns:
                raise ValueError(
                    f"live_benchmark_models missing required column '{benchmark_col}'."
                )
            benchmark_series = live_benchmark_models.reindex(live_features.index)[
                benchmark_col
            ]
        X_live[benchmark_col] = (
            pd.to_numeric(benchmark_series, errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )

        blended = np.zeros(len(X_live), dtype=np.float32)
        for weight, model in zip(weights_arr, model_refs):
            blended += weight * np.asarray(model.predict(X_live), dtype=np.float32)

        pred_series = pd.Series(blended, index=live_features.index, dtype=np.float32)
        ranked = pred_series.groupby(
            live_features[era_col], group_keys=False
        ).rank(method="first", pct=True)
        return pd.DataFrame({"prediction": ranked.astype(np.float32)}, index=live_features.index)

    return predict


def main() -> None:
    args = parse_args()
    blend_name = args.blend_name
    output_pkl = (args.output_pkl or (UPLOAD_ARTIFACTS_DIR / f"{blend_name}.pkl")).resolve()
    output_meta = (args.output_meta or (UPLOAD_ARTIFACTS_DIR / f"{blend_name}.json")).resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    specs = parse_model_specs(args.spec)
    if not specs:
        raise ValueError("At least one model spec is required.")

    full, feature_cols = load_training_frame(
        data_version=args.data_version,
        feature_set=args.feature_set,
        benchmark_col=args.benchmark_col,
        era_col=DEFAULT_ERA_COL,
        id_col=DEFAULT_ID_COL,
        target_col=DEFAULT_TARGET_COL,
        max_rows=args.max_rows,
    )
    print(
        "Training frame loaded:",
        f"rows={len(full):,}",
        f"features={len(feature_cols):,}",
        f"eras={full[DEFAULT_ERA_COL].nunique():,}",
    )

    X_train = full[feature_cols + [DEFAULT_ERA_COL, args.benchmark_col]]
    y_train = full[DEFAULT_TARGET_COL].astype("float32", copy=False)

    models: list[ArrowstreetRegressor] = []
    weights: list[float] = []
    for spec in specs:
        cfg = load_config(spec.config_name)
        params = dict(cfg["model"]["params"])
        if args.stage2_lgbm_n_jobs is not None:
            lgbm_params = dict(params.get("stage2_lgbm_params", {}))
            if lgbm_params:
                lgbm_params["n_jobs"] = int(args.stage2_lgbm_n_jobs)
                params["stage2_lgbm_params"] = lgbm_params
        params["feature_cols"] = list(feature_cols)

        model = ArrowstreetRegressor(**params)
        print(f"Fitting {spec.config_name} with weight={spec.weight:.2f} ...")
        model.fit(X_train, y_train)
        models.append(model)
        weights.append(float(spec.weight))

    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value.")
    weights = [w / total_weight for w in weights]

    predict_fn = make_predict_function(
        models=models,
        weights=weights,
        feature_cols=feature_cols,
        era_col=DEFAULT_ERA_COL,
        benchmark_col=args.benchmark_col,
    )

    # Ensure repo-local model modules are serialized by value to avoid
    # `ImportError: No module named 'agents'` in Numerai's runtime.
    cloudpickle.register_pickle_by_value(arrow_components)
    cloudpickle.register_pickle_by_value(arrow_regressor)

    # Local smoke-check with in-sample rows to validate callable output shape/type.
    smoke_n = min(5000, len(full))
    smoke_features = full.iloc[:smoke_n][feature_cols + [DEFAULT_ERA_COL]].copy()
    smoke_benchmarks = full.iloc[:smoke_n][[args.benchmark_col]].copy()
    smoke_preds = predict_fn(smoke_features, smoke_benchmarks)
    if "prediction" not in smoke_preds.columns:
        raise RuntimeError("Smoke test failed: missing 'prediction' output column.")
    if len(smoke_preds) != smoke_n:
        raise RuntimeError("Smoke test failed: output row count mismatch.")
    if not np.isfinite(smoke_preds["prediction"].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Smoke test failed: non-finite predictions.")

    with output_pkl.open("wb") as f:
        cloudpickle.dump(predict_fn, f)

    metadata = {
        "blend_name": blend_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": args.data_version,
        "feature_set": args.feature_set,
        "benchmark_col": args.benchmark_col,
        "target_col": DEFAULT_TARGET_COL,
        "rows_trained": int(len(full)),
        "eras_trained": int(full[DEFAULT_ERA_COL].nunique()),
        "feature_count": int(len(feature_cols)),
        "weights": {
            spec.config_name: float(weight)
            for spec, weight in zip(specs, weights)
        },
        "configs": [spec.config_name for spec in specs],
        "smoke_test_rows": int(smoke_n),
        "output_pkl": str(output_pkl),
    }
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Wrote pickle: {output_pkl}")
    print(f"Wrote metadata: {output_meta}")


if __name__ == "__main__":
    main()
