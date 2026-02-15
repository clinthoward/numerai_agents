"""Upload Numerai Signals predictions, optionally generating them first."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from numerapi import NumerAPI

from agents.code.signals.predict_live import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_DIR,
    generate_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Signals predictions via NumerAPI."
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
        help="Path to an existing predictions CSV. If omitted, predictions are generated first.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Signals model id. Falls back to NUMERAI_SIGNALS_MODEL_ID.",
    )
    parser.add_argument("--public-id", type=str, default=None)
    parser.add_argument("--secret-key", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--data-version", type=str, default="v5.2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--live-path", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--era-col", type=str, default=None)
    return parser.parse_args()


def _build_api(public_id: str | None, secret_key: str | None) -> NumerAPI:
    if public_id and secret_key:
        return NumerAPI(public_id=public_id, secret_key=secret_key)

    env_public = os.getenv("NUMERAI_PUBLIC_ID")
    env_secret = os.getenv("NUMERAI_SECRET_KEY")
    if env_public and env_secret:
        return NumerAPI(public_id=env_public, secret_key=env_secret)

    return NumerAPI()


def main() -> None:
    args = parse_args()
    model_id = (
        args.model_id
        or os.getenv("NUMERAI_SIGNALS_MODEL_ID")
        or os.getenv("NUMERAI_MODEL_ID")
    )
    if not model_id:
        raise ValueError(
            "Model id is required. Provide --model-id or set NUMERAI_SIGNALS_MODEL_ID."
        )

    predictions_file = args.predictions_file
    if predictions_file is None:
        predictions_file = generate_predictions(
            artifacts_dir=args.artifacts_dir,
            data_dir=args.data_dir,
            data_version=args.data_version,
            live_path=args.live_path,
            output_file=args.output_file,
            era_col_override=args.era_col,
        )
        print(f"Generated predictions file: {predictions_file}")

    predictions_file = predictions_file.resolve()
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    if args.dry_run:
        print("Dry run enabled; skipping upload.")
        print(f"Would upload: {predictions_file}")
        print(f"Model id: {model_id}")
        return

    api = _build_api(args.public_id, args.secret_key)
    api.upload_predictions(str(predictions_file), model_id=model_id)
    print(f"Uploaded {predictions_file} to model {model_id}")


if __name__ == "__main__":
    main()
