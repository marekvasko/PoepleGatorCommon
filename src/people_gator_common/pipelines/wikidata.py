from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from SPARQLWrapper import SPARQLWrapper

from people_gator_common.headers import CLIENT_USER_AGENT
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.wikidata import create_people_of_all, fetch_property_values_for_entities
from people_gator_common.wikidata.fetchers import fetch_people_of_property, fetch_texts_of_entities
from people_gator_common.wikidata.processors import unique_values_from_csv

COUNTRY_PROPERTY_ID = "P27"
LANGUAGE_PROPERTY_ID = "P1412"
IMAGE_PROPERTY_ID = "P18"
VIAF_ID_PROPERTY_ID = "P214"
NLCR_AUT_ID_PROPERTY_ID = "P691"
DATE_OF_BIRTH_PROPERTY_ID = "P569"
DATE_OF_DEATH_PROPERTY_ID = "P570"

CZECHIA_ID = "Q213"
CZECHOSLOVAKIA_ID = "Q33946"
SLOVAKIA_ID = "Q214"
CZECH_LANGUAGE_ID = "Q9067"
SLOVAK_LANGUAGE_ID = "Q9058"


def _run_people_collectors(
    output_dir: Path,
    *,
    client: SPARQLWrapper,
    receive_batch_size: int,
    rate_limiter: RateLimiter,
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_people_collectors.__name__)
    tasks: list[tuple[str, str, Path]] = [
        (COUNTRY_PROPERTY_ID, CZECHIA_ID, output_dir / f"people_of_{COUNTRY_PROPERTY_ID}={CZECHIA_ID}.csv"),
        (
            COUNTRY_PROPERTY_ID,
            CZECHOSLOVAKIA_ID,
            output_dir / f"people_of_{COUNTRY_PROPERTY_ID}={CZECHOSLOVAKIA_ID}.csv",
        ),
        (COUNTRY_PROPERTY_ID, SLOVAKIA_ID, output_dir / f"people_of_{COUNTRY_PROPERTY_ID}={SLOVAKIA_ID}.csv"),
        (
            LANGUAGE_PROPERTY_ID,
            CZECH_LANGUAGE_ID,
            output_dir / f"people_of_{LANGUAGE_PROPERTY_ID}={CZECH_LANGUAGE_ID}.csv",
        ),
        (
            LANGUAGE_PROPERTY_ID,
            SLOVAK_LANGUAGE_ID,
            output_dir / f"people_of_{LANGUAGE_PROPERTY_ID}={SLOVAK_LANGUAGE_ID}.csv",
        ),
    ]
    for property_id, value_id, output_file in tasks:
        logger.info("Starting to fetch people of %s=%s", property_id, value_id)
        fetch_people_of_property(
            property_id=property_id,
            value_id=value_id,
            client=client,
            batch_size=receive_batch_size,
            use_cursor=True,
            rate_limiter=rate_limiter,
            output_file=output_file,
        )


def _run_people_property_enrichment(
    people_of_all_file: Path,
    output_dir: Path,
    *,
    send_batch_size: int,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_people_property_enrichment.__name__)
    tasks: list[tuple[str, Path, str]] = [
        (IMAGE_PROPERTY_ID, output_dir / f"people_all_{IMAGE_PROPERTY_ID}.csv", "image"),
        (COUNTRY_PROPERTY_ID, output_dir / f"people_all_{COUNTRY_PROPERTY_ID}.csv", "country"),
        (LANGUAGE_PROPERTY_ID, output_dir / f"people_all_{LANGUAGE_PROPERTY_ID}.csv", "language"),
        (VIAF_ID_PROPERTY_ID, output_dir / f"people_all_{VIAF_ID_PROPERTY_ID}.csv", "VIAF ID"),
        (NLCR_AUT_ID_PROPERTY_ID, output_dir / f"people_all_{NLCR_AUT_ID_PROPERTY_ID}.csv", "NLCR AUT ID"),
        (DATE_OF_BIRTH_PROPERTY_ID, output_dir / f"people_all_{DATE_OF_BIRTH_PROPERTY_ID}.csv", "date of birth"),
        (DATE_OF_DEATH_PROPERTY_ID, output_dir / f"people_all_{DATE_OF_DEATH_PROPERTY_ID}.csv", "date of death"),
    ]
    for property_id, output_file, label in tasks:
        logger.info("Starting to fetch property values for all people in %s for %s", people_of_all_file, label)
        fetch_property_values_for_entities(
            property_id=property_id,
            entities_csv_path=people_of_all_file,
            client=client,
            batch_size=send_batch_size,
            rate_limiter=rate_limiter,
            output_file=output_file,
        )


def _run_property_value_deduplication(output_dir: Path) -> None:
    logger = logging.getLogger(__name__ + "." + _run_property_value_deduplication.__name__)
    tasks: list[tuple[Path, Path, str]] = [
        (output_dir / f"people_all_{COUNTRY_PROPERTY_ID}.csv", output_dir / "unique_countries.csv", "country"),
        (output_dir / f"people_all_{LANGUAGE_PROPERTY_ID}.csv", output_dir / "unique_languages.csv", "language"),
        (output_dir / f"people_all_{VIAF_ID_PROPERTY_ID}.csv", output_dir / "unique_viaf_id.csv", "VIAF ID"),
        (output_dir / f"people_all_{NLCR_AUT_ID_PROPERTY_ID}.csv", output_dir / "unique_nl_cr_aut_id.csv", "NLCR AUT ID"),
    ]
    for source_csv, target_csv, label in tasks:
        logger.info("Starting to deduplicate %s values from %s", label, source_csv)
        unique_values_from_csv(source_csv=source_csv, target_csv=target_csv)


def _run_text_enrichment(
    people_of_all_file: Path,
    unique_countries_file: Path,
    unique_languages_file: Path,
    *,
    send_batch_size: int,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
    output_dir: Path,
) -> None:
    logger = logging.getLogger(__name__ + "." + _run_text_enrichment.__name__)
    tasks: list[tuple[Path, Literal["labels", "descriptions", "alt_labels"], Path, str]] = [
        (people_of_all_file, "labels", output_dir / "people_of_all_labels.csv", "all people"),
        (people_of_all_file, "descriptions", output_dir / "people_of_all_descriptions.csv", "all people"),
        (people_of_all_file, "alt_labels", output_dir / "people_of_all_alt_labels.csv", "all people"),
        (unique_countries_file, "labels", output_dir / "unique_countries_labels.csv", "all unique countries"),
        (
            unique_countries_file,
            "descriptions",
            output_dir / "unique_countries_descriptions.csv",
            "all unique countries",
        ),
        (
            unique_countries_file,
            "alt_labels",
            output_dir / "unique_countries_alt_labels.csv",
            "all unique countries",
        ),
        (unique_languages_file, "labels", output_dir / "unique_languages_labels.csv", "all unique languages"),
        (
            unique_languages_file,
            "descriptions",
            output_dir / "unique_languages_descriptions.csv",
            "all unique languages",
        ),
        (
            unique_languages_file,
            "alt_labels",
            output_dir / "unique_languages_alt_labels.csv",
            "all unique languages",
        ),
    ]
    for entities_csv, text_property, output_file, label in tasks:
        logger.info("Starting to fetch %s for %s in %s", text_property, label, entities_csv)
        fetch_texts_of_entities(
            entities_csv_path=entities_csv,
            text_property=text_property,
            client=client,
            batch_size=send_batch_size,
            rate_limiter=rate_limiter,
            output_file=output_file,
        )


def run_wikidata_pipeline(
    output_dir: Path,
    *,
    receive_batch_size: int,
    send_batch_size: int,
    rate_limit_interval: float,
    rate_limit_retries: int,
    rate_limit_backoff: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    client = SPARQLWrapper("https://query.wikidata.org/sparql", agent=CLIENT_USER_AGENT)
    client.setMethod("POST")
    rate_limiter = RateLimiter(
        min_request_interval=rate_limit_interval,
        max_retries=rate_limit_retries,
        retry_backoff_seconds=rate_limit_backoff,
    )

    _run_people_collectors(
        output_dir=output_dir,
        client=client,
        receive_batch_size=receive_batch_size,
        rate_limiter=rate_limiter,
    )

    people_of_all_file = output_dir / "people_of_all.csv"
    create_people_of_all(
        output_dir=output_dir,
        input_csvs_glob="people_of_P*.csv",
        output_file=people_of_all_file,
    )

    _run_people_property_enrichment(
        people_of_all_file=people_of_all_file,
        output_dir=output_dir,
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )

    _run_property_value_deduplication(output_dir=output_dir)
    _run_text_enrichment(
        people_of_all_file=people_of_all_file,
        unique_countries_file=output_dir / "unique_countries.csv",
        unique_languages_file=output_dir / "unique_languages.csv",
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
        output_dir=output_dir,
    )
