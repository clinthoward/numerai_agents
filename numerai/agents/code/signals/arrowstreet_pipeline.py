from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from agents.code.metrics import numerai_metrics
from agents.code.modeling.models.arrowstreet_regressor import ArrowstreetRegressor
from agents.code.modeling.models.lgbm_regressor import LGBMRegressor
from agents.code.modeling.utils.numerai_cv import era_cv_splits

from .artifacts import load_variant_artifacts, save_variant_artifacts
from .postprocess import (
    IsotonicEraCalibrator,
    neutralize_predictions,
    to_submission_scores,
)
from .target_ensemble import (
    average_target_predictions,
    discover_target_columns,
    select_ensemble_targets,
)
from .variants import VariantSpec, resolve_variant


AGENTS_DIR = Path(__file__).resolve().parents[2]
NUMERAI_DIR = AGENTS_DIR.parent
DEFAULT_DATA_DIR = NUMERAI_DIR
DEFAULT_OUTPUT_DIR = AGENTS_DIR / "experiments" / "arrowstreet_integration" / "variant_runs"
DEFAULT_ARTIFACTS_ROOT = AGENTS_DIR / "signals_artifacts"
REPO_ROOT = NUMERAI_DIR.parent

_LEGACY_LGBM_MODEL_KEYS = {
    "n_estimators",
    "learning_rate",
    "max_depth",
    "num_leaves",
    "colsample_bytree",
    "subsample",
    "min_data_in_leaf",
    "n_jobs",
    "verbosity",
    "device_type",
}
_ARROWSTREET_MODEL_KEYS = {
    "ridge_alpha",
    "indirect_max_base_features",
    "basket_cluster_sizes",
    "linkage_k",
    "linkage_stats",
    "use_baskets",
    "use_linkages",
    "model_variant",
    "stage2_model_type",
    "stage2_lgbm_params",
    "stage2_weight",
    "stage2_target_mode",
    "stage2_benchmark_col",
    "stage2_benchmark_beta",
    "dtype_float",
    "random_state",
    "era_col",
    "group_sets",
    "indirect_feature_selection",
    "indirect_feature_ranking_target",
    "indirect_feature_ranking_min_eras",
    "embedding_mode",
    "embedding_pca_components",
}


@dataclass(frozen=True)
class SignalsRunSettings:
    data_version: str = "v5.2"
    feature_set: str = "medium"
    target_col: str = "target"
    era_col: str = "era"
    id_col: str | None = "id"
    data_dir: Path = DEFAULT_DATA_DIR
    train_path: Path | None = None
    validation_path: Path | None = None
    live_path: Path | None = None
    features_json: Path | None = None
    dtype_float: str = "float32"
    max_train_eras: int | None = 150
    downsample_stride: int = 1
    cv_n_splits: int = 5
    cv_embargo: int = 13
    cv_mode: str = "expanding"
    cv_min_train_size: int = 1
    random_state: int = 42
    output_dir: Path = DEFAULT_OUTPUT_DIR
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT


def settings_from_dict(config: dict[str, Any]) -> SignalsRunSettings:
    data = config.get("data", {})
    training = config.get("training", {})
    output = config.get("output", {})

    def _repo_path(value: str | Path | None, default: Path) -> Path:
        if value is None:
            return default
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        return (REPO_ROOT / candidate).resolve()

    return SignalsRunSettings(
        data_version=str(data.get("data_version", "v5.2")),
        feature_set=str(data.get("feature_set", "medium")),
        target_col=str(data.get("target_col", "target")),
        era_col=str(data.get("era_col", "era")),
        id_col=data.get("id_col", "id"),
        data_dir=_repo_path(data.get("data_dir"), DEFAULT_DATA_DIR),
        train_path=Path(data["train_path"]).expanduser() if data.get("train_path") else None,
        validation_path=Path(data["validation_path"]).expanduser()
        if data.get("validation_path")
        else None,
        live_path=Path(data["live_path"]).expanduser() if data.get("live_path") else None,
        features_json=Path(data["features_json"]).expanduser()
        if data.get("features_json")
        else None,
        dtype_float=str(data.get("dtype_float", "float32")),
        max_train_eras=training.get("max_train_eras", 150),
        downsample_stride=int(training.get("downsample_stride", 1)),
        cv_n_splits=int(training.get("cv_n_splits", 5)),
        cv_embargo=int(training.get("cv_embargo", 13)),
        cv_mode=str(training.get("cv_mode", "expanding")),
        cv_min_train_size=int(training.get("cv_min_train_size", 1)),
        random_state=int(training.get("random_state", 42)),
        output_dir=_repo_path(output.get("output_dir"), DEFAULT_OUTPUT_DIR),
        artifacts_root=_repo_path(output.get("artifacts_root"), DEFAULT_ARTIFACTS_ROOT),
    )


def _resolve_version_dir(settings: SignalsRunSettings) -> Path:
    if settings.data_dir.name == settings.data_version:
        return settings.data_dir
    return settings.data_dir / settings.data_version


def resolve_paths(settings: SignalsRunSettings) -> tuple[Path, Path, Path, Path]:
    version_dir = _resolve_version_dir(settings)
    train_path = settings.train_path or (version_dir / "train.parquet")
    validation_path = settings.validation_path or (version_dir / "validation.parquet")
    features_json = settings.features_json or (version_dir / "features.json")
    live_path = settings.live_path or (version_dir / "live.parquet")
    return train_path, validation_path, features_json, live_path


def _read_feature_set(features_json: Path, feature_set: str) -> list[str]:
    meta = json.loads(features_json.read_text(encoding="utf-8"))
    if "feature_sets" not in meta or feature_set not in meta["feature_sets"]:
        raise KeyError(
            f"Feature set '{feature_set}' not found in {features_json}."
        )
    return list(meta["feature_sets"][feature_set])


def _read_split(
    path: Path,
    columns: list[str],
    *,
    validation_only: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    available_cols = set(pq.ParquetFile(path).schema.names)
    read_cols = [col for col in columns if col in available_cols]
    if validation_only and "data_type" in available_cols:
        read_cols = [*read_cols, "data_type"]

    df = pd.read_parquet(path, columns=read_cols)
    if validation_only and "data_type" in df.columns:
        df = df[df["data_type"] == "validation"]
        df = df.drop(columns=["data_type"], errors="ignore")

    # Numerai parquet splits commonly store `id` as the DataFrame index rather than
    # a materialized column. Preserve it when the caller requested it.
    if df.index.name in columns and df.index.name not in df.columns:
        df[df.index.name] = df.index.to_numpy()

    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = np.nan

    if list(df.columns) != columns:
        df = df.reindex(columns=columns, copy=False)
    return df


def _sort_eras(eras: pd.Series) -> list[Any]:
    def _key(value: Any) -> tuple[int, str]:
        try:
            return (0, f"{int(value):012d}")
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted(pd.unique(eras), key=_key)


def _downsample_eras(df: pd.DataFrame, era_col: str, stride: int) -> pd.DataFrame:
    stride = max(int(stride), 1)
    if stride <= 1:
        return df
    eras = _sort_eras(df[era_col])
    selected = set(eras[::stride])
    return df[df[era_col].isin(selected)]


def _restrict_last_train_eras(
    df: pd.DataFrame,
    era_col: str,
    max_train_eras: int | None,
) -> pd.DataFrame:
    if max_train_eras is None:
        return df
    max_train_eras = int(max_train_eras)
    if max_train_eras <= 0:
        return df

    eras = _sort_eras(df[era_col])
    if len(eras) <= max_train_eras:
        return df
    keep = set(eras[-max_train_eras:])
    return df[df[era_col].isin(keep)]


def _cast_numeric(df: pd.DataFrame, cols: list[str], dtype_float: str) -> pd.DataFrame:
    target_dtype = np.dtype(dtype_float)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if np.issubdtype(series.dtype, np.number) and series.dtype == target_dtype:
            continue
        # Cast one column at a time to avoid creating a massive temporary block
        # when working with full-data Signals tables.
        df[col] = series.astype(target_dtype, copy=False)
    return df


def _load_datasets(
    settings: SignalsRunSettings,
    *,
    include_validation_targets: bool,
    target_cols_override: list[str] | None = None,
) -> dict[str, Any]:
    train_path, validation_path, features_json, _ = resolve_paths(settings)
    feature_cols = _read_feature_set(features_json, settings.feature_set)
    if target_cols_override is None:
        target_cols = discover_target_columns(train_path)
    else:
        target_cols = list(dict.fromkeys(target_cols_override))

    if settings.target_col not in target_cols:
        target_cols = [settings.target_col, *target_cols]

    available_train_cols = set(pq.ParquetFile(train_path).schema.names)
    available_validation_cols = set(pq.ParquetFile(validation_path).schema.names)
    target_cols = [col for col in target_cols if col in available_train_cols]
    if settings.target_col not in target_cols:
        raise ValueError(
            f"Primary target column '{settings.target_col}' not found in {train_path}"
        )

    train_id_cols: list[str] = []
    validation_id_cols: list[str] = []
    if settings.id_col:
        if settings.id_col in available_train_cols:
            train_id_cols = [settings.id_col]
        if settings.id_col in available_validation_cols:
            validation_id_cols = [settings.id_col]

    train_cols = list(dict.fromkeys([*train_id_cols, settings.era_col, *feature_cols, *target_cols]))
    train = _read_split(train_path, train_cols, validation_only=False)
    train = train.dropna(subset=[settings.target_col])
    train = _restrict_last_train_eras(train, settings.era_col, settings.max_train_eras)
    train = _downsample_eras(train, settings.era_col, settings.downsample_stride)
    train = _cast_numeric(train, [*feature_cols, *target_cols], settings.dtype_float)

    validation_targets = target_cols if include_validation_targets else [settings.target_col]
    validation_cols = list(
        dict.fromkeys([*validation_id_cols, settings.era_col, *feature_cols, *validation_targets])
    )
    validation = _read_split(validation_path, validation_cols, validation_only=True)
    if settings.target_col in validation.columns:
        validation = validation.dropna(subset=[settings.target_col])
    validation = _downsample_eras(validation, settings.era_col, settings.downsample_stride)
    validation = _cast_numeric(
        validation,
        [*feature_cols, *validation_targets],
        settings.dtype_float,
    )

    return {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "train": train,
        "validation": validation,
        "train_path": train_path,
        "validation_path": validation_path,
        "features_json": features_json,
    }


def _build_model(
    variant: VariantSpec,
    settings: SignalsRunSettings,
    feature_cols: list[str],
):
    params = dict(variant.model_params)

    if variant.toggles.base_model_type == "LGBMRegressor":
        params.setdefault("random_state", settings.random_state)
        return LGBMRegressor(feature_cols=feature_cols, **params)

    if variant.toggles.base_model_type != "ArrowstreetRegressor":
        raise ValueError(
            "Unsupported base_model_type. Expected LGBMRegressor or ArrowstreetRegressor, "
            f"got {variant.toggles.base_model_type}."
        )

    params.setdefault("random_state", settings.random_state)
    params.setdefault("era_col", settings.era_col)
    params.setdefault("model_variant", variant.toggles.model_variant)
    params.setdefault("stage2_model_type", variant.toggles.stage2_model_type)
    params.setdefault("use_baskets", variant.toggles.use_baskets)
    params.setdefault("use_linkages", variant.toggles.use_linkages)
    if not variant.toggles.use_linkages:
        params["linkage_stats"] = []

    unknown = sorted(set(params.keys()) - _ARROWSTREET_MODEL_KEYS)
    legacy = [key for key in unknown if key in _LEGACY_LGBM_MODEL_KEYS]
    for key in legacy:
        params.pop(key, None)
    remaining_unknown = sorted(set(params.keys()) - _ARROWSTREET_MODEL_KEYS)
    if remaining_unknown:
        raise ValueError(
            "Unsupported Arrowstreet parameters in variant model_params: "
            + ", ".join(remaining_unknown)
        )

    return ArrowstreetRegressor(feature_cols=feature_cols, **params)


def _build_X(
    df: pd.DataFrame,
    *,
    base_model_type: str,
    feature_cols: list[str],
    era_col: str,
) -> pd.DataFrame:
    if base_model_type == "ArrowstreetRegressor":
        return df[[era_col, *feature_cols]]
    return df[feature_cols]


def _train_single_target(
    *,
    variant: VariantSpec,
    settings: SignalsRunSettings,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None,
    feature_cols: list[str],
    target_name: str,
) -> dict[str, Any]:
    oof = pd.Series(index=train_df.index, dtype=settings.dtype_float)
    fold_cols = [settings.era_col, *feature_cols, target_name]

    eras = _sort_eras(train_df[settings.era_col])
    try:
        splits = era_cv_splits(
            eras,
            n_splits=settings.cv_n_splits,
            embargo=settings.cv_embargo,
            mode=settings.cv_mode,
            min_train_size=settings.cv_min_train_size,
        )
    except ValueError as exc:
        if "train split too small" not in str(exc):
            raise
        print(
            "CV configuration too strict for available eras; retrying with "
            "min_train_size=0 for this variant."
        )
        splits = era_cv_splits(
            eras,
            n_splits=settings.cv_n_splits,
            embargo=settings.cv_embargo,
            mode=settings.cv_mode,
            min_train_size=0,
        )

    fold_meta: list[dict[str, Any]] = []
    for fold_idx, (train_eras, val_eras) in enumerate(splits):
        train_block = train_df.loc[
            train_df[settings.era_col].isin(train_eras), fold_cols
        ]
        val_block = train_df.loc[
            train_df[settings.era_col].isin(val_eras), fold_cols
        ]
        if val_block.empty:
            continue
        train_block = train_block.dropna(subset=[target_name])
        if train_block.empty:
            continue

        model = _build_model(variant, settings, feature_cols)
        X_train = _build_X(
            train_block,
            base_model_type=variant.toggles.base_model_type,
            feature_cols=feature_cols,
            era_col=settings.era_col,
        )
        y_train = train_block[target_name]
        model.fit(X_train, y_train)

        X_val = _build_X(
            val_block,
            base_model_type=variant.toggles.base_model_type,
            feature_cols=feature_cols,
            era_col=settings.era_col,
        )
        preds = model.predict(X_val)
        oof.loc[val_block.index] = np.asarray(preds, dtype=np.float32)

        fold_meta.append(
            {
                "fold": fold_idx,
                "target": target_name,
                "train_eras": len(train_eras),
                "val_eras": len(val_eras),
                "train_rows": int(train_block.shape[0]),
                "val_rows": int(val_block.shape[0]),
            }
        )

    full_fit_df = train_df.loc[:, fold_cols].dropna(subset=[target_name])
    if full_fit_df.empty:
        raise ValueError(
            f"Target '{target_name}' has no non-null rows in training data."
        )
    full_model = _build_model(variant, settings, feature_cols)
    X_full = _build_X(
        full_fit_df,
        base_model_type=variant.toggles.base_model_type,
        feature_cols=feature_cols,
        era_col=settings.era_col,
    )
    full_model.fit(X_full, full_fit_df[target_name])

    holdout_pred = None
    if holdout_df is not None and not holdout_df.empty:
        X_holdout = _build_X(
            holdout_df,
            base_model_type=variant.toggles.base_model_type,
            feature_cols=feature_cols,
            era_col=settings.era_col,
        )
        holdout_pred = pd.Series(
            np.asarray(full_model.predict(X_holdout), dtype=np.float32),
            index=holdout_df.index,
            dtype=np.float32,
            name=target_name,
        )

    return {
        "model": full_model,
        "oof": oof,
        "holdout": holdout_pred,
        "fold_meta": fold_meta,
    }


def _build_summary(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str,
    era_col: str,
) -> dict[str, float]:
    if frame.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    per_era = numerai_metrics.per_era_corr(
        frame,
        [pred_col],
        target_col,
        era_col=era_col,
    )
    summary = numerai_metrics.summarize_scores(per_era).loc[pred_col]
    mean = float(summary["mean"])
    std = float(summary["std"])
    sharpe = float(summary["sharpe"])
    max_drawdown = float(summary["max_drawdown"])
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std):
        std = 0.0
    if not np.isfinite(sharpe):
        sharpe = 0.0
    if not np.isfinite(max_drawdown):
        max_drawdown = 0.0
    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _apply_postprocess(
    *,
    variant: VariantSpec,
    settings: SignalsRunSettings,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None,
    feature_cols: list[str],
    oof_raw: pd.Series,
    holdout_raw: pd.Series | None,
) -> tuple[pd.Series, pd.Series | None, IsotonicEraCalibrator | None, list[str]]:
    oof_pred = oof_raw.astype(np.float32)
    holdout_pred = None if holdout_raw is None else holdout_raw.astype(np.float32)

    neutralized_features: list[str] = []
    method = variant.neutralization.method
    if method != "none":
        train_neut = train_df[[settings.era_col, *feature_cols]].copy()
        train_neut["prediction"] = oof_pred
        oof_pred, neutralized_features = neutralize_predictions(
            train_neut,
            pred_col="prediction",
            feature_cols=feature_cols,
            method=method,
            proportion=variant.neutralization.proportion,
            top_n=variant.neutralization.top_n,
            era_col=settings.era_col,
        )

        if holdout_pred is not None and holdout_df is not None:
            val_neut = holdout_df[[settings.era_col, *feature_cols]].copy()
            val_neut["prediction"] = holdout_pred
            holdout_pred, _ = neutralize_predictions(
                val_neut,
                pred_col="prediction",
                feature_cols=feature_cols,
                method=method,
                proportion=variant.neutralization.proportion,
                top_n=variant.neutralization.top_n,
                era_col=settings.era_col,
            )

    calibrator: IsotonicEraCalibrator | None = None
    if variant.calibration.enabled and variant.toggles.use_calibration:
        calibrator = IsotonicEraCalibrator()
        cal_train = train_df[[settings.era_col, settings.target_col]].copy()
        cal_train["prediction"] = oof_pred
        cal_train = cal_train.dropna(subset=["prediction", settings.target_col])
        if not cal_train.empty:
            calibrator.fit(
                cal_train,
                pred_col="prediction",
                target_col=settings.target_col,
                era_col=settings.era_col,
            )
            oof_pred = calibrator.apply(
                cal_train,
                pred_col="prediction",
                era_col=settings.era_col,
            ).reindex(train_df.index)
            if holdout_pred is not None and holdout_df is not None:
                cal_holdout = holdout_df[[settings.era_col]].copy()
                cal_holdout["prediction"] = holdout_pred
                holdout_pred = calibrator.apply(
                    cal_holdout,
                    pred_col="prediction",
                    era_col=settings.era_col,
                )

    return (
        oof_pred.astype(np.float32),
        None if holdout_pred is None else holdout_pred.astype(np.float32),
        calibrator,
        neutralized_features,
    )


def _build_prediction_frame(
    *,
    df: pd.DataFrame,
    settings: SignalsRunSettings,
    prediction: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if settings.id_col and settings.id_col in df.columns:
        out[settings.id_col] = df[settings.id_col]
    out[settings.era_col] = df[settings.era_col]
    if settings.target_col in df.columns:
        out[settings.target_col] = df[settings.target_col]
    out["prediction"] = prediction.astype(np.float32)
    return out


def run_variant_training(
    *,
    variant_name: str,
    settings: SignalsRunSettings,
    mode: str,
    variants: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in {"benchmark", "full_train"}:
        raise ValueError("mode must be one of: benchmark, full_train")

    variant = resolve_variant(variant_name, variants=variants)
    include_validation_targets = mode == "full_train"
    load_all_targets = (
        variant.toggles.use_target_ensemble and variant.target_ensemble.enabled
    )
    target_cols_override = None if load_all_targets else [settings.target_col]
    data = _load_datasets(
        settings,
        include_validation_targets=include_validation_targets,
        target_cols_override=target_cols_override,
    )

    feature_cols: list[str] = data["feature_cols"]
    target_cols: list[str] = data["target_cols"]
    train_df: pd.DataFrame = data["train"]
    validation_df: pd.DataFrame = data["validation"]

    if mode == "full_train":
        fit_df = pd.concat([train_df, validation_df], axis=0, ignore_index=False)
        holdout_df = None
    else:
        fit_df = train_df
        holdout_df = validation_df

    if variant.toggles.use_target_ensemble and variant.target_ensemble.enabled:
        selected_targets, target_corrs = select_ensemble_targets(
            fit_df,
            primary_target=settings.target_col,
            candidate_targets=target_cols,
            correlation_threshold=variant.target_ensemble.correlation_threshold,
            max_models=variant.target_ensemble.max_models,
            duplicate_corr_cutoff=variant.target_ensemble.duplicate_corr_cutoff,
        )
    else:
        selected_targets = [settings.target_col]
        target_corrs = fit_df[[settings.target_col]].corrwith(fit_df[settings.target_col])

    models_by_target: dict[str, Any] = {}
    oof_by_target: dict[str, pd.Series] = {}
    holdout_by_target: dict[str, pd.Series] = {}
    folds: list[dict[str, Any]] = []

    for target_name in selected_targets:
        if target_name not in fit_df.columns:
            continue
        target_fit = _train_single_target(
            variant=variant,
            settings=settings,
            train_df=fit_df,
            holdout_df=holdout_df,
            feature_cols=feature_cols,
            target_name=target_name,
        )
        models_by_target[target_name] = target_fit["model"]
        oof_by_target[target_name] = target_fit["oof"].rename(target_name)
        if target_fit["holdout"] is not None:
            holdout_by_target[target_name] = target_fit["holdout"].rename(target_name)
        folds.extend(target_fit["fold_meta"])

    if not models_by_target:
        raise ValueError(
            f"No models were trained for variant '{variant_name}'. Check target availability."
        )

    oof_raw = average_target_predictions(oof_by_target)
    holdout_raw = None
    if holdout_df is not None and holdout_by_target:
        holdout_raw = average_target_predictions(holdout_by_target)

    oof_final, holdout_final, calibrator, neutralized_features = _apply_postprocess(
        variant=variant,
        settings=settings,
        train_df=fit_df,
        holdout_df=holdout_df,
        feature_cols=feature_cols,
        oof_raw=oof_raw,
        holdout_raw=holdout_raw,
    )

    predictions_dir = settings.output_dir / "predictions"
    results_dir = settings.output_dir / "results"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    oof_frame = _build_prediction_frame(df=fit_df, settings=settings, prediction=oof_final)
    oof_path = predictions_dir / f"{variant_name}_oof.parquet"
    oof_frame.to_parquet(oof_path, index=False)

    holdout_path = None
    holdout_summary = None
    if holdout_df is not None and holdout_final is not None:
        holdout_frame = _build_prediction_frame(
            df=holdout_df,
            settings=settings,
            prediction=holdout_final,
        )
        holdout_path = predictions_dir / f"{variant_name}_validation.parquet"
        holdout_frame.to_parquet(holdout_path, index=False)

        holdout_summary = _build_summary(
            holdout_frame,
            pred_col="prediction",
            target_col=settings.target_col,
            era_col=settings.era_col,
        )

    model_bundle = {
        "models_by_target": models_by_target,
        "feature_cols": feature_cols,
        "era_col": settings.era_col,
        "target_col": settings.target_col,
        "variant": variant.to_dict(),
        "selected_targets": selected_targets,
    }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant.to_dict(),
        "data": {
            "data_version": settings.data_version,
            "feature_set": settings.feature_set,
            "target_col": settings.target_col,
            "era_col": settings.era_col,
            "dtype_float": settings.dtype_float,
            "mode": mode,
            "downsample_stride": settings.downsample_stride,
            "max_train_eras": settings.max_train_eras,
        },
        "ensemble": {
            "selected_targets": selected_targets,
            "target_correlations": {k: float(v) for k, v in target_corrs.to_dict().items()},
        },
        "postprocess": {
            "calibration_enabled": bool(calibrator is not None),
            "neutralization_method": variant.neutralization.method,
            "neutralized_features": neutralized_features,
        },
    }

    artifact_dir = save_variant_artifacts(
        artifacts_root=settings.artifacts_root,
        variant_name=variant_name,
        model_bundle=model_bundle,
        metadata=metadata,
        feature_cols=feature_cols,
        calibrator=calibrator,
    )

    results = {
        "variant": variant.to_dict(),
        "mode": mode,
        "paths": {
            "oof_predictions": str(oof_path),
            "validation_predictions": str(holdout_path) if holdout_path else None,
            "artifacts_dir": str(artifact_dir),
        },
        "metrics": {
            "validation_corr": holdout_summary,
        },
        "training": {
            "train_rows": int(fit_df.shape[0]),
            "train_eras": int(fit_df[settings.era_col].nunique()),
            "folds": folds,
        },
    }

    results_path = results_dir / f"{variant_name}.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    results["paths"]["results"] = str(results_path)
    return results


def predict_with_loaded_artifacts(
    *,
    live_df: pd.DataFrame,
    model_bundle: dict[str, Any],
    calibrator: IsotonicEraCalibrator | None,
) -> pd.Series:
    feature_cols = list(model_bundle["feature_cols"])
    era_col = str(model_bundle["era_col"])
    variant_dict = model_bundle["variant"]
    base_model_type = variant_dict["toggles"]["base_model_type"]

    models_by_target: dict[str, Any] = model_bundle["models_by_target"]
    predictions_by_target: dict[str, pd.Series] = {}
    for target_name, model in models_by_target.items():
        X_live = _build_X(
            live_df,
            base_model_type=base_model_type,
            feature_cols=feature_cols,
            era_col=era_col,
        )
        pred = model.predict(X_live)
        predictions_by_target[target_name] = pd.Series(
            np.asarray(pred, dtype=np.float32),
            index=live_df.index,
            dtype=np.float32,
            name=target_name,
        )

    pred = average_target_predictions(predictions_by_target).astype(np.float32)

    neutral_cfg = variant_dict["postprocess"]["neutralization"]
    if neutral_cfg["method"] != "none":
        frame = live_df[[era_col, *feature_cols]].copy()
        frame["prediction"] = pred
        pred, _ = neutralize_predictions(
            frame,
            pred_col="prediction",
            feature_cols=feature_cols,
            method=str(neutral_cfg["method"]),
            proportion=float(neutral_cfg.get("proportion", 1.0)),
            top_n=neutral_cfg.get("top_n", 0.10),
            era_col=era_col,
        )

    calibration_enabled = bool(variant_dict["postprocess"]["calibration"]["enabled"])
    if calibration_enabled and calibrator is not None:
        cal_frame = live_df[[era_col]].copy()
        cal_frame["prediction"] = pred
        pred = calibrator.apply(cal_frame, pred_col="prediction", era_col=era_col)

    return pred.astype(np.float32)


def generate_live_submission(
    *,
    variant_name: str,
    settings: SignalsRunSettings,
    output_file: Path | None = None,
) -> Path:
    artifact_dir = settings.artifacts_root / variant_name
    loaded = load_variant_artifacts(artifact_dir)
    model_bundle = loaded["model_bundle"]
    calibrator = loaded["calibrator"]

    _, _, _, live_path = resolve_paths(settings)
    if not live_path.exists():
        raise FileNotFoundError(f"Live dataset not found: {live_path}")

    feature_cols = list(model_bundle["feature_cols"])
    era_col = str(model_bundle["era_col"])
    read_cols = [era_col, *feature_cols]
    available = set(pq.ParquetFile(live_path).schema.names)
    read_cols = [col for col in read_cols if col in available]

    live = pd.read_parquet(live_path, columns=read_cols)
    missing_features = [col for col in feature_cols if col not in live.columns]
    if missing_features:
        raise ValueError(
            "Live dataset is missing required features: "
            f"{missing_features[:8]}"
            + ("..." if len(missing_features) > 8 else "")
        )
    live[feature_cols] = live[feature_cols].astype(settings.dtype_float)

    pred = predict_with_loaded_artifacts(
        live_df=live,
        model_bundle=model_bundle,
        calibrator=calibrator,
    )

    pred_frame = pd.DataFrame(index=live.index)
    pred_frame[era_col] = live[era_col]
    pred_frame["prediction"] = pred
    submission = to_submission_scores(pred_frame, pred_col="prediction", era_col=era_col)

    out = pd.DataFrame(index=live.index)
    out["prediction"] = submission

    if output_file is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        predictions_dir = settings.output_dir / "live_predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_file = predictions_dir / f"{variant_name}_{stamp}.csv"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_file, index=True)
    return output_file.resolve()


def load_variant_results(results_path: Path) -> dict[str, Any]:
    return json.loads(results_path.read_text(encoding="utf-8"))


def settings_to_jsonable(settings: SignalsRunSettings) -> dict[str, Any]:
    payload = asdict(settings)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
