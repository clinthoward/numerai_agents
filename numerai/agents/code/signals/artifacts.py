from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_variant_artifacts(
    *,
    artifacts_root: Path,
    variant_name: str,
    model_bundle: dict[str, Any],
    metadata: dict[str, Any],
    feature_cols: list[str],
    calibrator: object | None,
) -> Path:
    variant_dir = artifacts_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_path = variant_dir / "model.pkl"
    metadata_path = variant_dir / "metadata.json"
    features_path = variant_dir / "features.txt"
    manifest_path = variant_dir / "manifest.json"

    with model_path.open("wb") as fp:
        pickle.dump(model_bundle, fp)

    calibrator_path = None
    if calibrator is not None:
        calibrator_path = variant_dir / "calibrator.pkl"
        with calibrator_path.open("wb") as fp:
            pickle.dump(calibrator, fp)

    features_path.write_text("\n".join(feature_cols), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    files = {
        "model": {
            "file": model_path.name,
            "sha256": _file_sha256(model_path),
        },
        "metadata": {
            "file": metadata_path.name,
            "sha256": _file_sha256(metadata_path),
        },
        "features": {
            "file": features_path.name,
            "sha256": _file_sha256(features_path),
        },
    }
    if calibrator_path is not None:
        files["calibrator"] = {
            "file": calibrator_path.name,
            "sha256": _file_sha256(calibrator_path),
        }

    manifest = {
        "variant": variant_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return variant_dir


def load_variant_artifacts(variant_dir: Path) -> dict[str, Any]:
    model_path = variant_dir / "model.pkl"
    metadata_path = variant_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found: {metadata_path}")

    with model_path.open("rb") as fp:
        model_bundle = pickle.load(fp)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    calibrator = None
    calibrator_path = variant_dir / "calibrator.pkl"
    if calibrator_path.exists():
        with calibrator_path.open("rb") as fp:
            calibrator = pickle.load(fp)

    return {
        "model_bundle": model_bundle,
        "metadata": metadata,
        "calibrator": calibrator,
        "variant_dir": variant_dir,
    }
