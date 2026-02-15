"""Build full dataset and benchmark parquet files for Numerai."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from numerapi import NumerAPI
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from agents.code.modeling.utils.constants import NUMERAI_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full.parquet and full_benchmark_models.parquet."
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default="v5.2",
        help="Numerai data version (default: v5.2).",
    )
    parser.add_argument(
        "--downsample-eras-step",
        type=int,
        default=4,
        help="Keep every Nth era when building downsampled_full (default: 4).",
    )
    parser.add_argument(
        "--downsample-eras-offset",
        type=int,
        default=0,
        help="Offset when selecting every Nth era (default: 0).",
    )
    parser.add_argument(
        "--skip-downsample",
        action="store_true",
        help="Skip building downsampled_full datasets.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild full datasets even if they already exist.",
    )
    return parser.parse_args()


def _download_if_missing(napi: NumerAPI, remote_name: str, local_path: Path) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    napi.download_dataset(remote_name, dest_path=str(local_path))


def _iter_parquet_tables(path: Path, batch_size: int = 131_072) -> Iterable[pa.Table]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield pa.Table.from_batches([batch])


def _drop_columns_if_present(table: pa.Table, drop_cols: Iterable[str]) -> pa.Table:
    drop_set = set(drop_cols)
    keep_cols = [col for col in table.column_names if col not in drop_set]
    return table.select(keep_cols)


def _normalize_id_column(table: pa.Table, id_col: str = "id") -> pa.Table:
    names = list(table.column_names)
    if id_col in names:
        extras = [col for col in ("__index_level_0__", "index") if col in names]
        if extras:
            table = _drop_columns_if_present(table, extras)
        return table

    if "__index_level_0__" in names:
        names = [id_col if name == "__index_level_0__" else name for name in names]
        return table.rename_columns(names)

    if "index" in names:
        names = [id_col if name == "index" else name for name in names]
        return table.rename_columns(names)

    return table


def _filter_data_type(table: pa.Table, data_type: str) -> pa.Table:
    if "data_type" not in table.column_names:
        return table
    mask = pc.equal(table["data_type"], pa.scalar(data_type))
    return table.filter(mask)


def _coerce_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    table_cols = set(table.column_names)
    for field in schema:
        if field.name not in table_cols:
            table = table.append_column(
                field.name,
                pa.nulls(table.num_rows, type=field.type),
            )
    table = table.select(schema.names)
    if table.schema != schema:
        table = table.cast(schema, safe=False)
    return table


def _write_tables_to_parquet(path: Path, tables: Iterable[pa.Table]) -> None:
    temp_path = Path(str(path) + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    writer: pq.ParquetWriter | None = None
    try:
        for table in tables:
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema)
            else:
                table = _coerce_to_schema(table, writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"No rows produced when building parquet at {path}.")
    os.replace(temp_path, path)


def build_features_metadata(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    features_path = (NUMERAI_DIR / data_version / "features.json").resolve()
    if reuse_existing and features_path.exists():
        return features_path
    _download_if_missing(napi, f"{data_version}/features.json", features_path)
    return features_path


def build_full_dataset(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    full_path = (NUMERAI_DIR / data_version / "full.parquet").resolve()
    if reuse_existing and full_path.exists():
        return full_path
    train_path = (NUMERAI_DIR / data_version / "train.parquet").resolve()
    validation_path = (NUMERAI_DIR / data_version / "validation.parquet").resolve()
    _download_if_missing(napi, f"{data_version}/train.parquet", train_path)
    _download_if_missing(napi, f"{data_version}/validation.parquet", validation_path)

    def _tables() -> Iterable[pa.Table]:
        for table in _iter_parquet_tables(train_path):
            table = _normalize_id_column(table, id_col="id")
            table = _drop_columns_if_present(table, {"data_type"})
            yield table

        for table in _iter_parquet_tables(validation_path):
            table = _filter_data_type(table, "validation")
            if table.num_rows == 0:
                continue
            table = _normalize_id_column(table, id_col="id")
            table = _drop_columns_if_present(table, {"data_type"})
            yield table

    _write_tables_to_parquet(full_path, _tables())
    return full_path


def _collect_validation_ids(validation_data_path: Path) -> set:
    ids: set = set()
    for table in _iter_parquet_tables(validation_data_path):
        table = _normalize_id_column(table, id_col="id")
        if "id" not in table.column_names:
            raise ValueError(f"{validation_data_path} missing 'id' column.")
        table = _filter_data_type(table, "validation")
        if table.num_rows == 0:
            continue
        id_chunk = table["id"].to_pylist()
        ids.update(value for value in id_chunk if value is not None)
    if not ids:
        raise ValueError("No validation ids found in validation.parquet.")
    return ids


def build_full_benchmark(
    napi: NumerAPI, data_version: str, reuse_existing: bool = True
) -> Path:
    full_path = (NUMERAI_DIR / data_version / "full_benchmark_models.parquet").resolve()
    if reuse_existing and full_path.exists():
        return full_path
    train_path = (NUMERAI_DIR / data_version / "train_benchmark_models.parquet").resolve()
    validation_path = (
        NUMERAI_DIR / data_version / "validation_benchmark_models.parquet"
    ).resolve()
    validation_data_path = (NUMERAI_DIR / data_version / "validation.parquet").resolve()
    _download_if_missing(
        napi,
        f"{data_version}/train_benchmark_models.parquet",
        train_path,
    )
    _download_if_missing(
        napi,
        f"{data_version}/validation_benchmark_models.parquet",
        validation_path,
    )
    _download_if_missing(napi, f"{data_version}/validation.parquet", validation_data_path)

    validation_ids = _collect_validation_ids(validation_data_path)
    validation_ids_array = pa.array(list(validation_ids))

    def _tables() -> Iterable[pa.Table]:
        for table in _iter_parquet_tables(train_path):
            yield _normalize_id_column(table, id_col="id")

        for table in _iter_parquet_tables(validation_path):
            table = _normalize_id_column(table, id_col="id")
            if "id" not in table.column_names:
                raise ValueError(f"{validation_path} missing 'id' column.")
            mask = pc.is_in(table["id"], value_set=validation_ids_array)
            table = table.filter(mask)
            if table.num_rows > 0:
                yield table

    _write_tables_to_parquet(full_path, _tables())
    return full_path


def _sorted_unique_values(values: set) -> list:
    try:
        return sorted(values, key=lambda value: int(value))
    except (TypeError, ValueError):
        return sorted(values, key=lambda value: str(value))


def build_downsampled_full_dataset(
    full_path: Path,
    data_version: str,
    era_step: int,
    era_offset: int,
) -> Path:
    if era_step < 2:
        raise ValueError("downsample-eras-step must be >= 2.")
    if era_offset < 0 or era_offset >= era_step:
        raise ValueError("downsample-eras-offset must be in [0, downsample-eras-step).")
    downsampled_path = (NUMERAI_DIR / data_version / "downsampled_full.parquet").resolve()
    era_col = "era"
    unique_eras_set: set = set()
    for table in _iter_parquet_tables(full_path):
        if era_col not in table.column_names:
            raise ValueError(f"{full_path} missing '{era_col}' column.")
        unique_eras_set.update(table[era_col].to_pylist())

    unique_eras = _sorted_unique_values(unique_eras_set)
    keep_eras = {
        era for idx, era in enumerate(unique_eras) if idx % era_step == era_offset
    }
    keep_eras_array = pa.array(list(keep_eras))

    def _tables() -> Iterable[pa.Table]:
        for table in _iter_parquet_tables(full_path):
            mask = pc.is_in(table[era_col], value_set=keep_eras_array)
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                yield filtered

    _write_tables_to_parquet(downsampled_path, _tables())
    return downsampled_path


def _collect_ids(path: Path, id_col: str = "id") -> set:
    values: set = set()
    for table in _iter_parquet_tables(path):
        table = _normalize_id_column(table, id_col=id_col)
        if id_col not in table.column_names:
            raise ValueError(f"{path} missing '{id_col}' column.")
        values.update(value for value in table[id_col].to_pylist() if value is not None)
    return values


def build_downsampled_full_benchmark(
    full_benchmark_path: Path,
    downsampled_full_path: Path,
    data_version: str,
) -> Path:
    downsampled_path = (
        NUMERAI_DIR / data_version / "downsampled_full_benchmark_models.parquet"
    ).resolve()
    id_values = _collect_ids(downsampled_full_path, id_col="id")
    id_values_array = pa.array(list(id_values))

    def _tables() -> Iterable[pa.Table]:
        for table in _iter_parquet_tables(full_benchmark_path):
            table = _normalize_id_column(table, id_col="id")
            if "id" not in table.column_names:
                raise ValueError(f"{full_benchmark_path} missing 'id' column.")
            mask = pc.is_in(table["id"], value_set=id_values_array)
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                yield filtered

    _write_tables_to_parquet(downsampled_path, _tables())
    return downsampled_path


def main() -> None:
    args = parse_args()
    data_version = args.data_version
    napi = NumerAPI()
    reuse_existing = not args.rebuild

    features_path = build_features_metadata(
        napi, data_version, reuse_existing=reuse_existing
    )
    full_data = build_full_dataset(napi, data_version, reuse_existing=reuse_existing)
    full_benchmark = build_full_benchmark(
        napi, data_version, reuse_existing=reuse_existing
    )

    print(f"Built {features_path}")
    print(f"Built {full_data}")
    print(f"Built {full_benchmark}")
    if not args.skip_downsample:
        downsampled_full = build_downsampled_full_dataset(
            full_data,
            data_version,
            args.downsample_eras_step,
            args.downsample_eras_offset,
        )
        downsampled_benchmark = build_downsampled_full_benchmark(
            full_benchmark,
            downsampled_full,
            data_version,
        )
        print(f"Built {downsampled_full}")
        print(f"Built {downsampled_benchmark}")


if __name__ == "__main__":
    main()
