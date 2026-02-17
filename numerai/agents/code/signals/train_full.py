"""Train a selected Signals variant and persist a reusable artifact bundle."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from agents.code.signals.arrowstreet_pipeline import (
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    SignalsRunSettings,
    run_variant_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Signals variant on train+validation and save artifacts."
    )
    parser.add_argument("--variant", type=str, default="v99_production")
    parser.add_argument(
        "--variant-config",
        type=Path,
        default=None,
        help="Optional .py/.json file defining variant_definitions or signals.variant_definitions.",
    )
    parser.add_argument("--data-version", type=str, default="v5.2")
    parser.add_argument("--feature-set", type=str, default="medium")
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--era-col", type=str, default="era")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--validation-path", type=Path, default=None)
    parser.add_argument("--features-json", type=Path, default=None)
    parser.add_argument("--dtype-float", type=str, default="float32")
    parser.add_argument("--max-train-eras", type=int, default=150)
    parser.add_argument("--downsample-stride", type=int, default=1)
    parser.add_argument("--cv-n-splits", type=int, default=5)
    parser.add_argument("--cv-embargo", type=int, default=13)
    parser.add_argument("--cv-mode", choices=["expanding", "blocked"], default="expanding")
    parser.add_argument("--cv-min-train-size", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    return parser.parse_args()


def _load_variant_definitions(config_path: Path | None) -> dict | None:
    if config_path is None:
        return None
    if config_path.suffix == ".py":
        module_vars = runpy.run_path(str(config_path))
        if isinstance(module_vars.get("variant_definitions"), dict):
            return module_vars["variant_definitions"]
        config = module_vars.get("CONFIG") or module_vars.get("config")
        if isinstance(config, dict):
            signals = config.get("signals", {})
            if isinstance(signals.get("variant_definitions"), dict):
                return signals["variant_definitions"]
        raise ValueError(
            f"{config_path} must define variant_definitions or CONFIG.signals.variant_definitions."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("variant_definitions"), dict):
        return payload["variant_definitions"]
    signals = payload.get("signals", {})
    if isinstance(signals.get("variant_definitions"), dict):
        return signals["variant_definitions"]
    raise ValueError(
        f"{config_path} must define variant_definitions or signals.variant_definitions."
    )


def main() -> None:
    args = parse_args()
    variant_definitions = _load_variant_definitions(args.variant_config)
    settings = SignalsRunSettings(
        data_version=args.data_version,
        feature_set=args.feature_set,
        target_col=args.target_col,
        era_col=args.era_col,
        id_col=args.id_col,
        data_dir=args.data_dir,
        train_path=args.train_path,
        validation_path=args.validation_path,
        features_json=args.features_json,
        dtype_float=args.dtype_float,
        max_train_eras=args.max_train_eras,
        downsample_stride=max(args.downsample_stride, 1),
        cv_n_splits=max(args.cv_n_splits, 2),
        cv_embargo=max(args.cv_embargo, 0),
        cv_mode=args.cv_mode,
        cv_min_train_size=max(args.cv_min_train_size, 1),
        random_state=args.random_state,
        output_dir=args.output_dir,
        artifacts_root=args.artifacts_root,
    )

    result = run_variant_training(
        variant_name=args.variant,
        settings=settings,
        mode="full_train",
        variants=variant_definitions,
    )

    artifact_dir = result["paths"]["artifacts_dir"]
    oof_path = result["paths"]["oof_predictions"]
    print(f"Saved artifacts to {artifact_dir}")
    print(f"Saved OOF predictions to {oof_path}")


if __name__ == "__main__":
    main()
