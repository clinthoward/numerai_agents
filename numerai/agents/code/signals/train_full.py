"""Train an Arrowstreet model on Numerai Signals train+validation and persist artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agents.code.modeling.models.arrowstreet_regressor import ArrowstreetRegressor


AGENTS_DIR = Path(__file__).resolve().parents[2]
NUMERAI_DIR = AGENTS_DIR.parent
DEFAULT_DATA_DIR = NUMERAI_DIR / "data"
DEFAULT_ARTIFACTS_DIR = AGENTS_DIR / "signals_artifacts" / "arrowstreet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Arrowstreet model on full Signals data and save artifacts."
    )
    parser.add_argument("--data-version", type=str, default="v5.2")
    parser.add_argument("--feature-set", type=str, default="medium")
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--era-col", type=str, default="era")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--validation-path", type=Path, default=None)
    parser.add_argument("--features-json", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)

    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--indirect-max-base-features", type=int, default=64)
    parser.add_argument("--basket-cluster-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--linkage-k", type=int, default=10)
    parser.add_argument(
        "--linkage-stats",
        nargs="+",
        default=["mean", "std", "min", "max"],
    )
    parser.add_argument(
        "--model-variant",
        choices=["standard", "residual_two_stage"],
        default="standard",
    )
    parser.add_argument(
        "--stage2-model-type",
        choices=["ridge", "lgbm"],
        default="ridge",
    )
    parser.add_argument("--dtype-float", type=str, default="float32")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    version_dir = args.data_dir / args.data_version
    train_path = args.train_path or (version_dir / "train.parquet")
    validation_path = args.validation_path or (version_dir / "validation.parquet")
    features_json = args.features_json or (version_dir / "features.json")
    return train_path, validation_path, features_json


def _load_feature_set(features_json: Path, feature_set: str) -> list[str]:
    metadata = json.loads(features_json.read_text(encoding="utf-8"))
    try:
        return list(metadata["feature_sets"][feature_set])
    except KeyError as exc:
        raise KeyError(
            f"Feature set '{feature_set}' not found in {features_json}."
        ) from exc


def _read_split(
    path: Path,
    feature_cols: list[str],
    era_col: str,
    target_col: str,
    *,
    validation_only: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    cols = [era_col, target_col, *feature_cols]
    read_cols = list(cols)
    try:
        candidate = pd.read_parquet(path, columns=[*read_cols, "data_type"])
        has_data_type = True
    except Exception:
        candidate = pd.read_parquet(path, columns=read_cols)
        has_data_type = False
    if validation_only and has_data_type and "data_type" in candidate.columns:
        candidate = candidate[candidate["data_type"] == "validation"].copy()
    return candidate[cols]


def _load_training_frame(
    train_path: Path,
    validation_path: Path,
    feature_cols: list[str],
    era_col: str,
    target_col: str,
    dtype_float: str,
) -> pd.DataFrame:
    train_df = _read_split(
        train_path,
        feature_cols,
        era_col,
        target_col,
        validation_only=False,
    )
    valid_df = _read_split(
        validation_path,
        feature_cols,
        era_col,
        target_col,
        validation_only=True,
    )
    df = pd.concat([train_df, valid_df], axis=0, ignore_index=False)
    df = df.dropna(subset=[target_col, era_col]).copy()
    df[feature_cols] = df[feature_cols].astype(dtype_float)
    return df


def main() -> None:
    args = parse_args()
    train_path, validation_path, features_json = _resolve_paths(args)
    feature_cols = _load_feature_set(features_json, args.feature_set)
    train_df = _load_training_frame(
        train_path,
        validation_path,
        feature_cols,
        args.era_col,
        args.target_col,
        args.dtype_float,
    )

    model = ArrowstreetRegressor(
        feature_cols=feature_cols,
        ridge_alpha=args.ridge_alpha,
        indirect_max_base_features=args.indirect_max_base_features,
        basket_cluster_sizes=args.basket_cluster_sizes,
        linkage_k=args.linkage_k,
        linkage_stats=args.linkage_stats,
        model_variant=args.model_variant,
        stage2_model_type=args.stage2_model_type,
        dtype_float=args.dtype_float,
        random_state=args.random_state,
        era_col=args.era_col,
    )
    X = train_df[[args.era_col, *feature_cols]]
    y = train_df[args.target_col]
    model.fit(X, y)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.pkl"
    metadata_path = args.artifacts_dir / "metadata.json"

    with model_path.open("wb") as fp:
        pickle.dump(model, fp)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": args.data_version,
        "feature_set": args.feature_set,
        "feature_cols": feature_cols,
        "target_col": args.target_col,
        "era_col": args.era_col,
        "model_params": {
            "ridge_alpha": args.ridge_alpha,
            "indirect_max_base_features": args.indirect_max_base_features,
            "basket_cluster_sizes": args.basket_cluster_sizes,
            "linkage_k": args.linkage_k,
            "linkage_stats": args.linkage_stats,
            "model_variant": args.model_variant,
            "stage2_model_type": args.stage2_model_type,
            "dtype_float": args.dtype_float,
            "random_state": args.random_state,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model artifacts to {args.artifacts_dir.resolve()}")
    print(f"Model: {model_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
