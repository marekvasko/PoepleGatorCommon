from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from SPARQLWrapper import SPARQLWrapper

from people_gator_common.headers import CLIENT_USER_AGENT
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.wikidata import create_people_of_all, fetch_property_values_for_entities
from people_gator_common.wikidata.fetchers import build_all_people_cursor_query, build_all_people_query, fetch_all_entities_by_query, fetch_people_of_property, fetch_texts_of_entities
from people_gator_common.utils import unique_values_from_csv
from people_gator_common.filesystem import makedirs
from os.path import join

from people_gator_common.wikidata.queries import build_all_character_cursor_query, build_all_character_query


COUNTRY_PROPERTY_ID = "P27"
LANGUAGE_PROPERTY_ID = "P1412"
IMAGE_PROPERTY_ID = "P18"
VIAF_ID_PROPERTY_ID = "P214"
NLCR_AUT_ID_PROPERTY_ID = "P691"
DATE_OF_BIRTH_PROPERTY_ID = "P569"
DATE_OF_DEATH_PROPERTY_ID = "P570"

PROPERTY_TO_NAME = {
    COUNTRY_PROPERTY_ID: "country",
    LANGUAGE_PROPERTY_ID: "language",
    IMAGE_PROPERTY_ID: "image",
    VIAF_ID_PROPERTY_ID: "VIAF ID",
    NLCR_AUT_ID_PROPERTY_ID: "NLCR AUT ID",
    DATE_OF_BIRTH_PROPERTY_ID: "date of birth",
    DATE_OF_DEATH_PROPERTY_ID: "date of death",
}

CZECHIA_ID = "Q213"
CZECHOSLOVAKIA_ID = "Q33946"
SLOVAKIA_ID = "Q214"
AUSTRIA_HUNGARY_ID = "Q28513"
CZECH_LANGUAGE_ID = "Q9067"
SLOVAK_LANGUAGE_ID = "Q9058"


def _run_people_collectors(
    output_dir: str,
    *,
    client: SPARQLWrapper,
    receive_batch_size: int,
    rate_limiter: RateLimiter,
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_people_collectors.__name__)
    #tasks: list[tuple[str, str, str]] = [
    #    (
    #        COUNTRY_PROPERTY_ID, 
    #        CZECHIA_ID, 
    #        join(output_dir, f"people_of_{COUNTRY_PROPERTY_ID}={CZECHIA_ID}.csv")),
    #    (
    #        COUNTRY_PROPERTY_ID,
    #        CZECHOSLOVAKIA_ID,
    #        join(output_dir, f"people_of_{COUNTRY_PROPERTY_ID}={CZECHOSLOVAKIA_ID}.csv"),
    #    ),
    #    (
    #        COUNTRY_PROPERTY_ID, 
    #        SLOVAKIA_ID, 
    #        join(output_dir, f"people_of_{COUNTRY_PROPERTY_ID}={SLOVAKIA_ID}.csv")),
    #    (
    #        COUNTRY_PROPERTY_ID,
    #        AUSTRIA_HUNGARY_ID,
    #        join(output_dir, f"people_of_{COUNTRY_PROPERTY_ID}={AUSTRIA_HUNGARY_ID}.csv"),  
    #    ),
    #    (
    #        LANGUAGE_PROPERTY_ID,
    #        CZECH_LANGUAGE_ID,
    #        join(output_dir, f"people_of_{LANGUAGE_PROPERTY_ID}={CZECH_LANGUAGE_ID}.csv"),
    #    ),
    #    (
    #        LANGUAGE_PROPERTY_ID,
    #        SLOVAK_LANGUAGE_ID,
    #        join(output_dir, f"people_of_{LANGUAGE_PROPERTY_ID}={SLOVAK_LANGUAGE_ID}.csv"),
    #    ),
    #]
    #for property_id, value_id, output_file in tasks:
    #    logger.info("Starting to fetch people of %s=%s", property_id, value_id)
    #    fetch_people_of_property(
    #        property_id=property_id,
    #        value_id=value_id,
    #        client=client,
    #        batch_size=receive_batch_size,
    #        use_cursor=True,
    #        rate_limiter=rate_limiter,
    #        output_file=output_file,
    #    )
    fetch_all_entities_by_query(
        output_file=join(output_dir, "people_of_all.csv"),
        partial_results_dir=join(output_dir, "partial_people_of_all"),
        query_builder=lambda limit, offset: build_all_people_query(limit=limit, offset=offset),
        cursor_query_builder=lambda after_id, limit: build_all_people_cursor_query(after_id=after_id, limit=limit),
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
    )
    fetch_all_entities_by_query(
        output_file=join(output_dir, "characters_of_all.csv"),
        partial_results_dir=join(output_dir, "partial_characters_of_all"),
        query_builder=lambda limit, offset: build_all_character_query(limit=limit, offset=offset),
        cursor_query_builder=lambda after_id, limit: build_all_character_cursor_query(after_id=after_id, limit=limit),
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
    )
    


def _run_entity_property_enrichment(
    all_entity_file: str,
    output_dir: str,
    prefix: str = "people_all",
    enrich_by: tuple[str, ...] = (
        IMAGE_PROPERTY_ID, 
        COUNTRY_PROPERTY_ID, 
        LANGUAGE_PROPERTY_ID, 
        VIAF_ID_PROPERTY_ID, 
        NLCR_AUT_ID_PROPERTY_ID, 
        DATE_OF_BIRTH_PROPERTY_ID, 
        DATE_OF_DEATH_PROPERTY_ID
    ),
    *,
    send_batch_size: int,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_entity_property_enrichment.__name__)
    tasks: list[tuple[str, str, str]] = [
        (property_id, join(output_dir, f"{prefix}_{property_id}.csv"), PROPERTY_TO_NAME.get(property_id, property_id))
        for property_id in enrich_by
    ]
    for property_id, output_file, label in tasks:
        logger.info("Starting to fetch property values for all people in %s for %s", all_entity_file, label)
        fetch_property_values_for_entities(
            property_id=property_id,
            partial_results_dir=join(output_dir, f"partial_{prefix}_{property_id}"),
            entities_csv_path=all_entity_file,
            client=client,
            batch_size=send_batch_size,
            rate_limiter=rate_limiter,
            output_file=output_file,
        )


def _run_property_value_deduplication(
    output_dir: str, 
    prefix: str = "people_all",
    property_keys: tuple[str, ...] = (
        COUNTRY_PROPERTY_ID,
        LANGUAGE_PROPERTY_ID,
        VIAF_ID_PROPERTY_ID,
        NLCR_AUT_ID_PROPERTY_ID,
    ),
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_property_value_deduplication.__name__)
    tasks: list[tuple[str, str, str]] = [
        (join(output_dir, f"{prefix}_{property_key}.csv"), join(output_dir, f"{prefix}_unique_{property_key}.csv"), property_key)
        for property_key in property_keys
    ]
    for source_csv, target_csv, label in tasks:
        logger.info("Starting to deduplicate %s values from %s", label, source_csv)
        unique_values_from_csv(source_csv=source_csv, target_csv=target_csv)


def _run_text_enrichment(
    all_entities_file: str,
    output_dir: str,
    prefix: str = "people_all",
    *,
    send_batch_size: int,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_text_enrichment.__name__)
    tasks: list[tuple[str, Literal["labels", "descriptions", "alt_labels"], str, str]] = [
        (all_entities_file, "labels", join(output_dir, f"{prefix}_labels.csv"), f"All {prefix} labels"),
        (all_entities_file, "descriptions", join(output_dir, f"{prefix}_descriptions.csv"), f"All {prefix} descriptions"),
        (all_entities_file, "alt_labels", join(output_dir, f"{prefix}_alt_labels.csv"), f"All {prefix} alt labels"),
    ]
    for entities_csv, text_property, output_file, label in tasks:
        logger.info("Starting to fetch %s for %s in %s", text_property, label, entities_csv)
        fetch_texts_of_entities(
            entities_csv_path=entities_csv,
            partial_results_dir=join(output_dir, f"partial_{prefix}_{text_property}"),
            text_property=text_property,
            client=client,
            batch_size=send_batch_size,
            rate_limiter=rate_limiter,
            output_file=output_file,
        )


def run_wikidata_pipeline(
    output_dir: str,
    *,
    receive_batch_size: int,
    send_batch_size: int,
    rate_limiter: RateLimiter,
) -> None:
    client = SPARQLWrapper("https://query.wikidata.org/sparql", agent=CLIENT_USER_AGENT)
    client.setMethod("POST")

    makedirs(output_dir, exist_ok=True)
    people_prefix = "people_of_all"
    people_file = join(output_dir, f"{people_prefix}.csv")
    people_enrichement_properties = (
        IMAGE_PROPERTY_ID, 
        COUNTRY_PROPERTY_ID, 
        LANGUAGE_PROPERTY_ID, 
        VIAF_ID_PROPERTY_ID,
        NLCR_AUT_ID_PROPERTY_ID,
        DATE_OF_BIRTH_PROPERTY_ID,
        DATE_OF_DEATH_PROPERTY_ID,
    )
    characters_prefix = "characters_of_all"
    characters_file = join(output_dir, f"{characters_prefix}.csv")
    characters_enrichement_properties = (
        IMAGE_PROPERTY_ID, 
        COUNTRY_PROPERTY_ID, 
        LANGUAGE_PROPERTY_ID, 
        VIAF_ID_PROPERTY_ID,
        NLCR_AUT_ID_PROPERTY_ID,
    )
    
    fetch_all_entities_by_query(
        output_file=people_file,
        partial_results_dir=join(output_dir, "partial_people_of_all"),
        query_builder=lambda limit, offset: build_all_people_query(limit=limit, offset=offset),
        cursor_query_builder=lambda after_id, limit: build_all_people_cursor_query(after_id=after_id, limit=limit),
        client=client,
        batch_size=receive_batch_size,
        use_cursor=False,
        rate_limiter=rate_limiter,
    )
    fetch_all_entities_by_query(
        output_file=characters_file,
        partial_results_dir=join(output_dir, "partial_characters_of_all"),
        query_builder=lambda limit, offset: build_all_character_query(limit=limit, offset=offset),
        cursor_query_builder=lambda after_id, limit: build_all_character_cursor_query(after_id=after_id, limit=limit),
        client=client,
        batch_size=receive_batch_size,
        use_cursor=False,
        rate_limiter=rate_limiter,
    )
    #create_people_of_all(
    #    output_dir=output_dir,
    #    input_csvs_glob="people_of_P*.csv",
    #    output_file=people_of_all_file,
    #)
    _run_entity_property_enrichment(
        all_entity_file=people_file,
        output_dir=output_dir,
        prefix=people_prefix,
        enrich_by=people_enrichement_properties,
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )
    _run_entity_property_enrichment(
        all_entity_file=characters_file,
        output_dir=output_dir,
        prefix=characters_prefix,
        enrich_by=characters_enrichement_properties,
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )
    _run_property_value_deduplication(
        output_dir=output_dir,
        prefix=people_prefix,
        property_keys=people_enrichement_properties,
    )
    _run_property_value_deduplication(
        output_dir=output_dir,
        prefix=characters_prefix,
        property_keys=characters_enrichement_properties,
    )
    _run_text_enrichment(
        all_entities_file=people_file,
        output_dir=output_dir,
        prefix=people_prefix,
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )
    _run_text_enrichment(
        all_entities_file=characters_file,
        output_dir=output_dir,
        prefix=characters_prefix,
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )
