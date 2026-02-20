#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
NUMERAI_ROOT = REPO_ROOT / "numerai"
if str(NUMERAI_ROOT) not in sys.path:
    sys.path.insert(0, str(NUMERAI_ROOT))

import agents.code.modeling.models.arrowstreet_components as arrow_components  # noqa: E402
import agents.code.modeling.models.arrowstreet_regressor as arrow_regressor  # noqa: E402


DEFAULT_ERA_COL = "era"
DEFAULT_ID_COL = "id"
DEFAULT_BENCHMARK_COL = "v52_lgbm_ender20"
DEFAULT_BLEND_NAME = "arrowstreet_blend_from_components"
UPLOAD_ARTIFACTS_DIR = SCRIPT_PATH.parent / "upload_artifacts"


@dataclass(frozen=True)
class WeightedConfig:
    config_name: str
    weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Numerai upload pickle by blending pre-trained component pickles."
    )
    parser.add_argument(
        "--blend-name",
        type=str,
        default=DEFAULT_BLEND_NAME,
        help="Output blend name (used in default output filenames).",
    )
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help=(
            "Blend spec in the form '<config_name>:<weight>'. "
            "Repeat for each config in the blend."
        ),
    )
    parser.add_argument(
        "--component-pkl",
        action="append",
        required=True,
        help=(
            "Component source in the form '<config_name>:<path_to_single_model_pickle>'. "
            "Each referenced config in --spec must be provided."
        ),
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.2",
        help="Data version folder under numerai/ used for smoke test rows.",
    )
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=5000,
        help="Rows used for local smoke-check. Set 0 to disable smoke-check.",
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
        help="Output metadata path. Defaults to upload_artifacts/<blend_name>.json",
    )
    return parser.parse_args()


def _parse_weighted_configs(items: list[str]) -> list[WeightedConfig]:
    specs: list[WeightedConfig] = []
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --spec value '{raw}'. Expected <config_name>:<weight>.")
        name, weight_str = raw.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --spec value '{raw}'. Missing config name.")
        try:
            weight = float(weight_str.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid --spec weight in '{raw}'.") from exc
        if not np.isfinite(weight):
            raise ValueError(f"Invalid non-finite --spec weight in '{raw}'.")
        specs.append(WeightedConfig(config_name=name, weight=weight))
    if not specs:
        raise ValueError("At least one --spec value is required.")
    return specs


def _parse_component_paths(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in items:
        if ":" not in raw:
            raise ValueError(
                f"Invalid --component-pkl value '{raw}'. Expected <config_name>:<path>."
            )
        name, path_str = raw.split(":", 1)
        name = name.strip()
        path = Path(path_str.strip()).expanduser().resolve()
        if not name:
            raise ValueError(f"Invalid --component-pkl value '{raw}'. Missing config name.")
        if not path.exists():
            raise FileNotFoundError(f"Component pickle not found: {path}")
        out[name] = path
    if not out:
        raise ValueError("At least one --component-pkl value is required.")
    return out


def _closure_map(fn) -> dict[str, object]:  # noqa: ANN001
    names = fn.__code__.co_freevars
    values = fn.__closure__ or ()
    return {name: cell.cell_contents for name, cell in zip(names, values)}


def _load_component(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        fn = cloudpickle.load(f)
    if not callable(fn):
        raise TypeError(f"Component artifact is not callable: {path}")
    closure = _closure_map(fn)

    required = {"feature_cols", "era_col", "benchmark_col", "model_refs"}
    missing = required - set(closure.keys())
    if missing:
        raise ValueError(
            f"Component artifact missing required closure fields {sorted(missing)}: {path}"
        )

    model_refs = closure["model_refs"]
    if not isinstance(model_refs, (tuple, list)) or len(model_refs) != 1:
        raise ValueError(
            "Component pickle must contain exactly one underlying model in 'model_refs'. "
            f"Found len={len(model_refs) if isinstance(model_refs, (tuple, list)) else 'n/a'} "
            f"for {path}."
        )

    return {
        "path": str(path),
        "feature_cols": list(closure["feature_cols"]),
        "era_col": str(closure["era_col"]),
        "benchmark_col": str(closure["benchmark_col"]),
        "model": model_refs[0],
    }


def _normalize_weights(specs: list[WeightedConfig]) -> tuple[list[str], list[float]]:
    names = [s.config_name for s in specs]
    weights = np.asarray([s.weight for s in specs], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value.")
    weights = weights / total
    return names, weights.astype(np.float32).tolist()


def _make_predict_function(
    *,
    models: list[object],
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
                    "live_benchmark_models is required when benchmark column is not "
                    "present in live_features."
                )
            if benchmark_col not in live_benchmark_models.columns:
                raise ValueError(
                    f"live_benchmark_models missing required column '{benchmark_col}'."
                )
            benchmark_series = live_benchmark_models.reindex(live_features.index)[benchmark_col]

        X_live[benchmark_col] = (
            pd.to_numeric(benchmark_series, errors="coerce").fillna(0.0).astype("float32")
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


def _normalize_id_column(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col in df.columns:
        return df
    if df.index.name == id_col:
        return df.reset_index()
    reset = df.reset_index()
    if id_col in reset.columns:
        return reset
    if "index" in reset.columns:
        return reset.rename(columns={"index": id_col})
    raise ValueError(f"Could not recover id column '{id_col}' from dataframe.")


def _run_smoke_check(
    *,
    predict_fn,
    data_version: str,
    feature_cols: list[str],
    era_col: str,
    benchmark_col: str,
    smoke_rows: int,
) -> int:
    if smoke_rows <= 0:
        return 0

    full_path = NUMERAI_ROOT / data_version / "full.parquet"
    benchmark_path = NUMERAI_ROOT / data_version / "full_benchmark_models.parquet"
    if not full_path.exists() or not benchmark_path.exists():
        raise FileNotFoundError(
            f"Smoke-check data files missing: {full_path} and/or {benchmark_path}"
        )

    base_cols = [DEFAULT_ID_COL, era_col, *feature_cols]
    full = pd.read_parquet(full_path, columns=base_cols)
    full = _normalize_id_column(full, DEFAULT_ID_COL)
    full = full.iloc[:smoke_rows].copy()

    bench = pd.read_parquet(benchmark_path, columns=[DEFAULT_ID_COL, benchmark_col])
    bench = _normalize_id_column(bench, DEFAULT_ID_COL).set_index(DEFAULT_ID_COL)
    smoke_features = full.set_index(DEFAULT_ID_COL)
    smoke_benchmarks = bench.reindex(smoke_features.index)
    smoke_preds = predict_fn(
        smoke_features.reset_index(drop=True),
        smoke_benchmarks.reset_index(drop=True),
    )

    if "prediction" not in smoke_preds.columns:
        raise RuntimeError("Smoke-check failed: missing 'prediction' column.")
    if len(smoke_preds) != len(smoke_features):
        raise RuntimeError("Smoke-check failed: row count mismatch.")
    if not np.isfinite(smoke_preds["prediction"].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Smoke-check failed: non-finite predictions.")
    return len(smoke_features)


def main() -> None:
    args = parse_args()
    specs = _parse_weighted_configs(args.spec)
    component_paths = _parse_component_paths(args.component_pkl)
    output_pkl = (
        args.output_pkl
        or (UPLOAD_ARTIFACTS_DIR / f"{args.blend_name}.pkl")
    ).resolve()
    output_meta = (
        args.output_meta
        or (UPLOAD_ARTIFACTS_DIR / f"{args.blend_name}.json")
    ).resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    component_names, weights = _normalize_weights(specs)
    missing_components = [name for name in component_names if name not in component_paths]
    if missing_components:
        raise ValueError(
            f"Missing --component-pkl entries for configs: {sorted(missing_components)}"
        )

    loaded = {name: _load_component(component_paths[name]) for name in component_names}
    reference = loaded[component_names[0]]
    feature_cols = reference["feature_cols"]
    era_col = reference["era_col"]
    benchmark_col = reference["benchmark_col"]
    for name, comp in loaded.items():
        if comp["feature_cols"] != feature_cols:
            raise ValueError(
                f"Feature mismatch for component '{name}'. "
                "All components must share identical feature ordering."
            )
        if comp["era_col"] != era_col:
            raise ValueError(f"Era column mismatch for component '{name}'.")
        if comp["benchmark_col"] != benchmark_col:
            raise ValueError(f"Benchmark column mismatch for component '{name}'.")

    models = [loaded[name]["model"] for name in component_names]
    predict_fn = _make_predict_function(
        models=models,
        weights=weights,
        feature_cols=feature_cols,
        era_col=era_col,
        benchmark_col=benchmark_col,
    )

    cloudpickle.register_pickle_by_value(arrow_components)
    cloudpickle.register_pickle_by_value(arrow_regressor)

    smoke_rows_run = _run_smoke_check(
        predict_fn=predict_fn,
        data_version=args.data_version,
        feature_cols=feature_cols,
        era_col=era_col,
        benchmark_col=benchmark_col,
        smoke_rows=int(args.smoke_rows),
    )

    with output_pkl.open("wb") as f:
        cloudpickle.dump(predict_fn, f)

    metadata = {
        "blend_name": args.blend_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": args.data_version,
        "feature_count": int(len(feature_cols)),
        "feature_set": "medium",
        "era_col": era_col,
        "benchmark_col": benchmark_col,
        "components": {
            name: {
                "weight": float(weight),
                "source_pkl": loaded[name]["path"],
            }
            for name, weight in zip(component_names, weights)
        },
        "configs": component_names,
        "smoke_test_rows": int(smoke_rows_run),
        "output_pkl": str(output_pkl),
    }
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Wrote pickle: {output_pkl}")
    print(f"Wrote metadata: {output_meta}")


if __name__ == "__main__":
    main()
