from __future__ import annotations
from collections.abc import Callable
from typing import Literal
import logging
from pathlib import Path
import pandas as pd
from SPARQLWrapper import SPARQLWrapper
from people_gator_common.filesystem import open_file, ensure_parent_dir

from people_gator_common.transport.sparql import (
    rate_limited_query_all,
    rate_limited_query_all_cursor,
    rate_limited_query_entities,
)
from ..rate_limiter import RateLimiter
from .queries import (
    build_people_of_property_query,
    build_people_of_property_cursor_query,
    build_property_query_builder,
    build_text_query_builder,
)

def extract_entity_id(entity_uri: str) -> str:
    return entity_uri.rsplit("/", maxsplit=1)[-1]

def extract_entity_list_from_csv(csv_path: str) -> list[str]:
    with open_file(csv_path, "r") as f:
        df = pd.read_csv(f)
    if "entity" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'entity' column")
    return df["entity"].apply(extract_entity_id).tolist()

def save_to_csv_if_applicable(df: pd.DataFrame, output_file: str | None) -> None:
    if output_file is not None:
        ensure_parent_dir(output_file)
        with open_file(output_file, "w", encoding="utf-8") as f:
            df.to_csv(f, index=False)

def fetch_property_values_for_entities(
    property_id: str,
    entities_csv_path: str,
    output_file: str,
    *,
    client: SPARQLWrapper,
    batch_size: int,
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + fetch_property_values_for_entities.__name__)

    entities = extract_entity_list_from_csv(entities_csv_path)
    query_builder = build_property_query_builder(property_id)
    properties_df = rate_limited_query_entities(
        entities=entities,
        query_builder=query_builder,
        client=client,
        batch_size=batch_size,
        rate_limiter=rate_limiter,
    ).sort_values(
        ["entity", "value"]
    ).drop_duplicates().reset_index(drop=True)
    save_to_csv_if_applicable(properties_df, output_file)

    logger.info(
        "Fetched properties for %d entities for property %s",
        len(properties_df),
        property_id,
    )
    return properties_df


def fetch_texts_of_entities(
    text_property: Literal["labels", "descriptions", "alt_labels"],
    entities_csv_path: str,
    output_file: str,
    *,
    client: SPARQLWrapper,
    batch_size: int,
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + fetch_texts_of_entities.__name__)
    
    entities = extract_entity_list_from_csv(entities_csv_path)
    query_builder: Callable[[list[str]], str] = build_text_query_builder(text_property)
    labels_df = rate_limited_query_entities(
        entities=entities,
        query_builder=query_builder,
        client=client,
        batch_size=batch_size,
        rate_limiter=rate_limiter,
    ).sort_values(
        ["entity", "entityTextLanguage", "entityText"]
    ).drop_duplicates().reset_index(drop=True)
    save_to_csv_if_applicable(labels_df, output_file)

    logger.info(
        "Fetched %d labels for %d entities avgerage %.2f labels/entity",
        len(labels_df),
        len(labels_df["entity"].unique()),
        labels_df.groupby("entity").size().mean() if not labels_df.empty else 0.0,
    )
    return labels_df


def fetch_people_of_property(
    property_id: str,
    value_id: str,
    output_file: str,
    *,
    client: SPARQLWrapper,
    batch_size: int,
    use_cursor: bool,
    rate_limiter: RateLimiter,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__ + "." + fetch_people_of_property.__name__)

    if use_cursor:
        df = rate_limited_query_all_cursor(
            lambda after_id, limit: build_people_of_property_cursor_query(
                property_id=property_id,
                value_id=value_id,
                after_id=after_id,
                limit=limit,
            ),
            client=client,
            limit=batch_size,
            rate_limiter=rate_limiter,
        )
    else:
        df = rate_limited_query_all(
            lambda limit, offset: build_people_of_property_query(
                property_id=property_id,
                value_id=value_id,
                limit=limit,
                offset=offset,
            ),
            client=client,
            limit=batch_size,
            initial_offset=0,
            rate_limiter=rate_limiter,
        )
    df = df.sort_values(by="entity").drop_duplicates().reset_index(drop=True)
    save_to_csv_if_applicable(df, output_file)
    
    logger.info(
        "Fetched %d entities for %s=%s (use_cursor=%s)",
        len(df),
        property_id,
        value_id,
        use_cursor,
    )
    return df
