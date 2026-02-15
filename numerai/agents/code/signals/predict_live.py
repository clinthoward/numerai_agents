"""Generate Numerai Signals live predictions from trained Arrowstreet artifacts."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


AGENTS_DIR = Path(__file__).resolve().parents[2]
NUMERAI_DIR = AGENTS_DIR.parent
DEFAULT_DATA_DIR = NUMERAI_DIR / "data"
DEFAULT_ARTIFACTS_DIR = AGENTS_DIR / "signals_artifacts" / "arrowstreet"
DEFAULT_PREDICTIONS_DIR = AGENTS_DIR / "signals_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate live Signals predictions from saved model artifacts."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--data-version", type=str, default="v5.2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--live-path", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--era-col", type=str, default=None)
    parser.add_argument("--dtype-float", type=str, default="float32")
    return parser.parse_args()


def _era_rank(values: pd.Series) -> pd.Series:
    denom = values.count()
    if denom == 0:
        return pd.Series(np.nan, index=values.index, dtype=np.float32)
    return ((values.rank(method="average") - 0.5) / denom).astype(np.float32)


def _load_artifacts(artifacts_dir: Path) -> tuple[object, dict]:
    model_path = artifacts_dir / "model.pkl"
    metadata_path = artifacts_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found: {metadata_path}")
    with model_path.open("rb") as fp:
        model = pickle.load(fp)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def generate_predictions(
    *,
    artifacts_dir: Path,
    data_dir: Path,
    data_version: str,
    live_path: Path | None,
    output_file: Path | None,
    era_col_override: str | None,
    dtype_float: str = "float32",
) -> Path:
    model, metadata = _load_artifacts(artifacts_dir)
    feature_cols = list(metadata["feature_cols"])
    era_col = era_col_override or metadata.get("era_col", "era")
    resolved_live_path = live_path or (data_dir / data_version / "live.parquet")
    if not resolved_live_path.exists():
        raise FileNotFoundError(f"Live parquet not found: {resolved_live_path}")

    read_cols = [era_col, *feature_cols]
    live = pd.read_parquet(resolved_live_path, columns=read_cols)
    live[feature_cols] = live[feature_cols].astype(dtype_float)

    X_live = live[[era_col, *feature_cols]]
    raw_pred = pd.Series(model.predict(X_live), index=live.index, name="prediction")
    submission = live[[era_col]].copy()
    submission["prediction"] = raw_pred
    submission["prediction"] = submission.groupby(era_col)["prediction"].transform(
        _era_rank
    )

    if output_file is None:
        DEFAULT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_file = DEFAULT_PREDICTIONS_DIR / f"signals_prediction_{stamp}.csv"
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    submission[["prediction"]].to_csv(output_file, index=True)
    return output_file.resolve()


def main() -> None:
    args = parse_args()
    output_file = generate_predictions(
        artifacts_dir=args.artifacts_dir,
        data_dir=args.data_dir,
        data_version=args.data_version,
        live_path=args.live_path,
        output_file=args.output_file,
        era_col_override=args.era_col,
        dtype_float=args.dtype_float,
    )
    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
