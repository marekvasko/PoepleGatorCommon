from __future__ import annotations

from pathlib import Path

import pandas as pd


def extract_nlcr_aut_ids_from_csv(
    csv_path: Path,
    *,
    id_column: str = "entity",
    deduplicate: bool = True,
    limit: int | None = None,
) -> list[str]:
    source_df = pd.read_csv(csv_path)
    if id_column not in source_df.columns:
        raise ValueError(f"CSV file {csv_path} must contain '{id_column}' column")

    ids_series = source_df[id_column].dropna().astype(str).str.strip()
    ids_series = ids_series[ids_series != ""]
    if deduplicate:
        ids_series = ids_series.drop_duplicates()

    ids = ids_series.tolist()
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        ids = ids[:limit]

    return ids
