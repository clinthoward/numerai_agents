"""Generate Numerai Signals live predictions from variant artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from agents.code.signals.arrowstreet_pipeline import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    SignalsRunSettings,
    generate_live_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate live Signals predictions from saved variant artifacts."
    )
    parser.add_argument("--variant", type=str, default="v99_production")
    parser.add_argument("--data-version", type=str, default="v5.2")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--live-path", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--dtype-float", type=str, default="float32")
    return parser.parse_args()


def generate_predictions(
    *,
    variant: str,
    data_version: str,
    data_dir: Path,
    live_path: Path | None,
    output_file: Path | None,
    output_dir: Path,
    artifacts_root: Path,
    dtype_float: str,
) -> Path:
    settings = SignalsRunSettings(
        data_version=data_version,
        data_dir=data_dir,
        live_path=live_path,
        output_dir=output_dir,
        artifacts_root=artifacts_root,
        dtype_float=dtype_float,
    )
    return generate_live_submission(
        variant_name=variant,
        settings=settings,
        output_file=output_file,
    )


def main() -> None:
    args = parse_args()
    output_path = generate_predictions(
        variant=args.variant,
        data_version=args.data_version,
        data_dir=args.data_dir,
        live_path=args.live_path,
        output_file=args.output_file,
        output_dir=args.output_dir,
        artifacts_root=args.artifacts_root,
        dtype_float=args.dtype_float,
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
