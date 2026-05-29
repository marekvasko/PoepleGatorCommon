"""Shared utility functions for People Gator pipelines."""

from __future__ import annotations

import logging

import pandas as pd

from people_gator_common.filesystem import ensure_parent_dir, open_file


def csv_escape(value: str) -> str:
    """Escape a value for safe inclusion in a CSV field."""
    if '"' in value:
        value = value.replace('"', '""')
    if ',' in value or '"' in value or '\n' in value:
        value = f'"{value}"'
    return value


def read_column_from_csv(path: str, column: str) -> list[str]:
    """Read a single string column from a CSV, dropping nulls and blank values."""
    with open_file(path, "r") as f:
        df = pd.read_csv(f)
    if column not in df.columns:
        raise ValueError(f"CSV file {path!r} must contain {column!r} column")
    values = df[column].dropna().astype(str).str.strip()
    return [v for v in values if v]


def unique_values_from_csv(
    source_csv: str,
    target_csv: str,
    values_column: str = "value",
    new_column_name: str = "entity",
) -> pd.DataFrame:
    """Extract unique values from one column of a CSV and write them to a new CSV."""
    logger = logging.getLogger(__name__ + "." + unique_values_from_csv.__name__)
    with open_file(source_csv, "r") as f:
        df = pd.read_csv(f)
    logger.debug("Read %d rows from source CSV %s", len(df), source_csv)
    unique_values_df = (
        df[[values_column]]
        .drop_duplicates()
        .sort_values(by=values_column)
        .reset_index(drop=True)
    )
    unique_values_df.columns = pd.Index([new_column_name])
    logger.debug(
        "Extracted %d unique values from column %s in source CSV %s",
        len(unique_values_df),
        values_column,
        source_csv,
    )
    ensure_parent_dir(target_csv)
    with open_file(target_csv, "w", encoding="utf-8") as f:
        unique_values_df.to_csv(f, index=False)
    logger.info("Saved %d unique values to %s", len(unique_values_df), target_csv)
    return unique_values_df


__all__ = [
    "csv_escape",
    "read_column_from_csv",
    "unique_values_from_csv",
]
