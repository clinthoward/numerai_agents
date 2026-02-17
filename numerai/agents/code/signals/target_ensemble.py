from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def discover_target_columns(parquet_path: Path, target_prefix: str = "target") -> list[str]:
    columns = pq.ParquetFile(parquet_path).schema.names
    return sorted([col for col in columns if col.startswith(target_prefix)])


def _pairwise_corr(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> float:
    pair = df[[col_a, col_b]].dropna()
    if pair.empty:
        return 0.0
    corr = pair[col_a].corr(pair[col_b])
    if pd.isna(corr):
        return 0.0
    return float(corr)


def select_ensemble_targets(
    df_train: pd.DataFrame,
    *,
    primary_target: str,
    candidate_targets: list[str],
    correlation_threshold: float,
    max_models: int,
    duplicate_corr_cutoff: float,
) -> tuple[list[str], pd.Series]:
    if primary_target not in candidate_targets:
        candidate_targets = [primary_target, *candidate_targets]

    available_targets = [
        col
        for col in candidate_targets
        if col in df_train.columns and df_train[col].notna().any()
    ]
    if primary_target not in available_targets:
        raise ValueError(
            f"Primary target '{primary_target}' is not available in training dataframe."
        )

    corr_to_primary = (
        df_train[available_targets].corrwith(df_train[primary_target]).fillna(0.0)
    )

    ranked = [
        col
        for col in corr_to_primary.sort_values(ascending=False).index.tolist()
        if col != primary_target
    ]

    selected = [primary_target]
    for col in ranked:
        if corr_to_primary[col] < correlation_threshold:
            continue

        too_similar = False
        for kept in selected:
            if abs(_pairwise_corr(df_train, col, kept)) >= duplicate_corr_cutoff:
                too_similar = True
                break
        if too_similar:
            continue

        selected.append(col)
        if len(selected) >= max_models:
            break

    return selected, corr_to_primary.sort_values(ascending=False)


def average_target_predictions(predictions_by_target: dict[str, pd.Series]) -> pd.Series:
    if not predictions_by_target:
        raise ValueError("No target predictions provided for ensemble averaging.")
    aligned = pd.concat(predictions_by_target.values(), axis=1)
    return aligned.mean(axis=1)
