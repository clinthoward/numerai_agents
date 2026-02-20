#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from numerapi import NumerAPI


@dataclass(frozen=True)
class UploadItem:
    model_id: str
    pickle_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one or more Numerai model pickles to specific model IDs."
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Upload item in the form '<model_id>:<pickle_path>'. Repeat per model slot.",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.2",
        help="Numerai upload data version (name or ID). Default: v5.2",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="numerai_predict_py_3_12",
        help=(
            "Numerai upload docker image (name or ID). "
            "Default: numerai_predict_py_3_12"
        ),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional JSON report path for upload IDs and current model upload status.",
    )
    return parser.parse_args()


def _parse_items(items: list[str]) -> list[UploadItem]:
    out: list[UploadItem] = []
    for raw in items:
        if ":" not in raw:
            raise ValueError(f"Invalid --item value '{raw}'. Expected <model_id>:<pickle_path>.")
        model_id, path_str = raw.split(":", 1)
        model_id = model_id.strip()
        pickle_path = Path(path_str.strip()).expanduser().resolve()
        if not model_id:
            raise ValueError(f"Invalid --item value '{raw}'. Missing model_id.")
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle path not found: {pickle_path}")
        if pickle_path.suffix != ".pkl":
            raise ValueError(f"Pickle path must end with .pkl: {pickle_path}")
        out.append(UploadItem(model_id=model_id, pickle_path=pickle_path))
    if not out:
        raise ValueError("At least one --item is required.")
    return out


def _parse_auth_token(raw: str) -> tuple[str, str]:
    token = raw.strip()
    if token.lower().startswith("token "):
        token = token[6:].strip()
    if "$" not in token:
        raise ValueError("NUMERAI_MCP_AUTH must be in the form 'Token PUBLIC_ID$SECRET_KEY'.")
    public_id, secret_key = token.split("$", 1)
    if not public_id or not secret_key:
        raise ValueError("Malformed NUMERAI_MCP_AUTH; missing public or secret segment.")
    return public_id, secret_key


def _load_api_from_env() -> NumerAPI:
    raw = os.environ.get("NUMERAI_MCP_AUTH")
    if not raw:
        raise EnvironmentError(
            "NUMERAI_MCP_AUTH is not set. Source your shell profile first."
        )
    public_id, secret_key = _parse_auth_token(raw)
    return NumerAPI(public_id=public_id, secret_key=secret_key, verbosity="info")


def _fetch_model_status(api: NumerAPI, model_ids: set[str]) -> list[dict[str, object]]:
    query = """
    query {
      account {
        models {
          id
          name
          computePickleUpload {
            id
            filename
            validationStatus
            triggerStatus
            insertedAt
          }
        }
      }
    }
    """
    data = api.raw_query(query, {}, authorization=True)
    models = data["data"]["account"]["models"]
    status_rows: list[dict[str, object]] = []
    for m in models:
        if m["id"] not in model_ids:
            continue
        upload = m.get("computePickleUpload")
        status_rows.append(
            {
                "model_id": m["id"],
                "model_name": m.get("name"),
                "compute_pickle_upload": upload,
            }
        )
    return status_rows


def main() -> None:
    args = parse_args()
    items = _parse_items(args.item)
    api = _load_api_from_env()

    docker_images = api.model_upload_docker_images()
    data_versions = api.model_upload_data_versions()
    if args.data_version not in data_versions and len(args.data_version) != 36:
        raise ValueError(
            f"Unknown data version '{args.data_version}'. "
            f"Known: {sorted(data_versions.keys())}"
        )
    if args.docker_image not in docker_images and len(args.docker_image) != 36:
        raise ValueError(
            f"Unknown docker image '{args.docker_image}'. "
            f"Known: {sorted(docker_images.keys())}"
        )

    uploads: list[dict[str, str]] = []
    for item in items:
        upload_id = api.model_upload(
            file_path=str(item.pickle_path),
            model_id=item.model_id,
            data_version=args.data_version,
            docker_image=args.docker_image,
        )
        row = {
            "model_id": item.model_id,
            "pickle_path": str(item.pickle_path),
            "upload_id": upload_id,
        }
        uploads.append(row)
        print(json.dumps(row, sort_keys=True))

    model_ids = {x.model_id for x in items}
    model_status = _fetch_model_status(api, model_ids=model_ids)
    for row in model_status:
        print(json.dumps(row, sort_keys=True))

    if args.output_report is not None:
        report_path = args.output_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "uploads": uploads,
            "model_status": model_status,
            "data_version": args.data_version,
            "docker_image": args.docker_image,
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
