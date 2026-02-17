"""One-command daily Signals generation + upload runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from numerapi import NumerAPI

from agents.code.signals.arrowstreet_pipeline import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
)
from agents.code.signals.predict_live import generate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and upload daily Signals predictions for a variant."
    )
    parser.add_argument("--variant", type=str, default="v99_production")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--public-id", type=str, default=None)
    parser.add_argument("--secret-key", type=str, default=None)
    parser.add_argument("--data-version", type=str, default="v5.2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--live-path", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--dtype-float", type=str, default="float32")
    parser.add_argument("--dry-run", action="store_true")
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

    predictions_file = generate_predictions(
        variant=args.variant,
        data_version=args.data_version,
        data_dir=args.data_dir,
        live_path=args.live_path,
        output_file=args.output_file,
        output_dir=args.output_dir,
        artifacts_root=args.artifacts_root,
        dtype_float=args.dtype_float,
    )
    print(f"Generated predictions file: {predictions_file}")

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
