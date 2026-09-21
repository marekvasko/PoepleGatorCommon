"""Wikidata v2 pipeline orchestration.

Implements the same data-gathering workflow as v1 (collect → enrich properties
→ deduplicate → enrich texts) using the manifest-based, resumable collector
and enricher modules from ``people_gator_common.wikidata_v2``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Literal

from SPARQLWrapper import SPARQLWrapper

from people_gator_common.headers import CLIENT_USER_AGENT
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.filesystem import exists, makedirs
from people_gator_common.utils import unique_values_from_csv
from people_gator_common.wikidata_v2.collectors import run_generic_collector
from people_gator_common.wikidata_v2.enrichers import (
    build_alt_label_enricher_query,
    build_description_enricher_query,
    build_label_enricher_query,
    build_property_enricher_query,
    extract_entity_id,
    run_generic_enricher,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wikidata property / entity constants
# ---------------------------------------------------------------------------

IS_INSTANCE_RELATION_ID = "wdt:P31"
IS_INSTANCE_OF_ANY_SUBCLASS_RELATION_ID = "wdt:P31/wdt:P279*"

PERSON_ENTITY_ID = "wd:Q5"
CHARACTER_ENTITY_ID = "wd:Q95074"

COUNTRY_PROPERTY_ID = "P27"
LANGUAGE_PROPERTY_ID = "P1412"
IMAGE_PROPERTY_ID = "P18"
VIAF_ID_PROPERTY_ID = "P214"
NLCR_AUT_ID_PROPERTY_ID = "P691"
DATE_OF_BIRTH_PROPERTY_ID = "P569"
DATE_OF_DEATH_PROPERTY_ID = "P570"

PROPERTY_TO_NAME: dict[str, str] = {
    COUNTRY_PROPERTY_ID: "country",
    LANGUAGE_PROPERTY_ID: "language",
    IMAGE_PROPERTY_ID: "image",
    VIAF_ID_PROPERTY_ID: "VIAF ID",
    NLCR_AUT_ID_PROPERTY_ID: "NLCR AUT ID",
    DATE_OF_BIRTH_PROPERTY_ID: "date of birth",
    DATE_OF_DEATH_PROPERTY_ID: "date of death",
}

PEOPLE_ENRICHMENT_PROPERTIES: tuple[str, ...] = (
    IMAGE_PROPERTY_ID,
    COUNTRY_PROPERTY_ID,
    LANGUAGE_PROPERTY_ID,
    VIAF_ID_PROPERTY_ID,
    NLCR_AUT_ID_PROPERTY_ID,
    DATE_OF_BIRTH_PROPERTY_ID,
    DATE_OF_DEATH_PROPERTY_ID,
)

CHARACTERS_ENRICHMENT_PROPERTIES: tuple[str, ...] = (
    IMAGE_PROPERTY_ID,
    COUNTRY_PROPERTY_ID,
    LANGUAGE_PROPERTY_ID,
    VIAF_ID_PROPERTY_ID,
    NLCR_AUT_ID_PROPERTY_ID,
)

DEDUPLICATABLE_PROPERTIES: tuple[str, ...] = (
    COUNTRY_PROPERTY_ID,
    LANGUAGE_PROPERTY_ID,
    VIAF_ID_PROPERTY_ID,
    NLCR_AUT_ID_PROPERTY_ID,
)



def _extract_entity_ids_from_csv(csv_path: str) -> list[str]:
    """Read a CSV with an ``entity`` column and return extracted Q-IDs."""
    import pandas as pd
    from people_gator_common.filesystem import open_file

    with open_file(csv_path, "r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    if "entity" not in df.columns:
        raise ValueError(f"CSV file {csv_path} must contain 'entity' column")
    return df["entity"].apply(extract_entity_id).tolist()


def _run_entity_property_enrichment(
    all_entity_file: str,
    output_dir: str,
    tmp_dir: str,
    prefix: str,
    enrich_by: tuple[str, ...],
    *,
    send_batch_size: int,
    force: bool,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    """Phase 2: for each property, enrich all entities with property values."""
    logger.info("Starting Phase 2: property enrichment for %s", prefix)
    entities = _extract_entity_ids_from_csv(all_entity_file)

    for property_id in enrich_by:
        label = PROPERTY_TO_NAME.get(property_id, property_id)
        output_file = os.path.join(output_dir, f"{prefix}_{property_id}.csv")
        logger.info("Enriching %s with property %s (%s)", prefix, property_id, label)
        run_generic_enricher(
            output_file=output_file,
            tmp_dir=tmp_dir,
            task_name=f"enricher__property__{property_id}",
            query_builder=lambda batch, pid=property_id: build_property_enricher_query(pid, batch),
            entities=entities,
            batch_size=send_batch_size,
            force=force,
            client=client,
            rate_limiter=rate_limiter,
        )


def _run_text_enrichment(
    all_entities_file: str,
    output_dir: str,
    tmp_dir: str,
    prefix: str,
    *,
    send_batch_size: int,
    force: bool,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    """Phase 4: fetch labels, descriptions, and alt_labels for all entities."""
    logger.info("Starting Phase 4: text enrichment for %s", prefix)
    entities = _extract_entity_ids_from_csv(all_entities_file)

    text_query_builders: dict[Literal["labels", "descriptions", "alt_labels"], Callable[[list[str]], str]] = {
        "labels": build_label_enricher_query,
        "descriptions": build_description_enricher_query,
        "alt_labels": build_alt_label_enricher_query,
    }

    for text_property, query_builder in text_query_builders.items():
        output_file = os.path.join(output_dir, f"{prefix}_{text_property}.csv")
        if exists(output_file) and not force:
            logger.info("Output file %s already exists, skipping text enrichment for %s", output_file, text_property)
            continue
        logger.info("Fetching %s for %s", text_property, prefix)
        run_generic_enricher(
            output_file=output_file,
            tmp_dir=tmp_dir,
            task_name=f"enricher__text__{text_property}",
            query_builder=query_builder,
            entities=entities,
            batch_size=send_batch_size,
            force=force,
            client=client,
            rate_limiter=rate_limiter,
        )


def _run_property_value_deduplication(
    output_dir: str,
    prefix: str,
    property_keys: tuple[str, ...],
    *,
    force: bool = False,
) -> None:
    """Phase 3: extract unique property values from enriched CSVs."""
    logger.info("Starting Phase 3: deduplication for %s", prefix)

    for property_key in property_keys:
        source_csv = os.path.join(output_dir, f"{prefix}_{property_key}.csv")
        target_csv = os.path.join(output_dir, f"{prefix}_unique_{property_key}.csv")
        label = PROPERTY_TO_NAME.get(property_key, property_key)
        if exists(target_csv) and not force:
            logger.info("Output file %s already exists, skipping deduplication for %s", target_csv, label)
            continue
        logger.info("Deduplicating %s values from %s", label, source_csv)
        unique_values_from_csv(source_csv=source_csv, target_csv=target_csv)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_wikidata_v2_pipeline(
    output_dir: str,
    tmp_dir: str,
    *,
    receive_batch_size: int,
    send_batch_size: int,
    force: bool,
    rate_limiter: RateLimiter,
) -> None:
    """Run the full Wikidata v2 collection and enrichment pipeline.

    Phases:
        1. Collect entity IDs (all people Q5, all characters Q95074)
        2. Enrich entities with property values (P18, P27, P1412, …)
        3. Deduplicate property values (P27, P1412, P214, P691)
        4. Enrich entities with text metadata (labels, descriptions, alt_labels)
    """
    client = SPARQLWrapper("https://query.wikidata.org/sparql", agent=CLIENT_USER_AGENT)
    client.setMethod("POST")

    makedirs(output_dir, exist_ok=True)

    people_prefix = "people_of_all"
    people_file = os.path.join(output_dir, f"{people_prefix}.csv")

    characters_prefix = "characters_of_all"
    characters_file = os.path.join(output_dir, f"{characters_prefix}.csv")

    # ── Phase 1: Collection ─────────────────────────────────────────────
    logger.info("Starting Phase 1: entity collection")
    # Collect all people (P31=Q5)
    run_generic_collector(
        output_file=os.path.join(output_dir, f"{people_prefix}.csv"),
        tmp_dir=tmp_dir,
        relationship=IS_INSTANCE_RELATION_ID,
        to_what=PERSON_ENTITY_ID,
        batch_size=receive_batch_size,
        force=force,
        rate_limiter=rate_limiter,
        client=client,
    )
    # Collect all characters (P31=Q95074)
    run_generic_collector(
        output_file=os.path.join(output_dir, f"{characters_prefix}.csv"),
        tmp_dir=tmp_dir,
        relationship=IS_INSTANCE_OF_ANY_SUBCLASS_RELATION_ID,
        to_what=CHARACTER_ENTITY_ID,
        batch_size=receive_batch_size,
        force=force,
        rate_limiter=rate_limiter,
        client=client,
    )

    # ── Phase 2: Property Enrichment ────────────────────────────────────
    logger.info("Starting Phase 2: property enrichment")
    _run_entity_property_enrichment(
        all_entity_file=people_file,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        prefix=people_prefix,
        enrich_by=PEOPLE_ENRICHMENT_PROPERTIES,
        send_batch_size=send_batch_size,
        force=force,
        client=client,
        rate_limiter=rate_limiter,
    )
    _run_entity_property_enrichment(
        all_entity_file=characters_file,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        prefix=characters_prefix,
        enrich_by=CHARACTERS_ENRICHMENT_PROPERTIES,
        send_batch_size=send_batch_size,
        force=force,
        client=client,
        rate_limiter=rate_limiter,
    )

    # ── Phase 3: Deduplication ──────────────────────────────────────────
    logger.info("Starting Phase 3: property value deduplication")
    _run_property_value_deduplication(
        output_dir=output_dir,
        prefix=people_prefix,
        property_keys=DEDUPLICATABLE_PROPERTIES,
        force=force,
    )
    _run_property_value_deduplication(
        output_dir=output_dir,
        prefix=characters_prefix,
        property_keys=DEDUPLICATABLE_PROPERTIES,
        force=force,
    )

    # ── Phase 4: Text Enrichment ────────────────────────────────────────
    logger.info("Starting Phase 4: text enrichment")
    _run_text_enrichment(
        all_entities_file=people_file,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        prefix=people_prefix,
        send_batch_size=send_batch_size // 4,
        force=force,
        client=client,
        rate_limiter=rate_limiter,
    )
    _run_text_enrichment(
        all_entities_file=characters_file,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        prefix=characters_prefix,
        send_batch_size=send_batch_size // 4,
        force=force,
        client=client,
        rate_limiter=rate_limiter,
    )

    logger.info("Wikidata v2 pipeline completed successfully")
