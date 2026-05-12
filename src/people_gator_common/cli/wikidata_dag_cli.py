import argparse
import logging
import sys
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper
from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.headers import CLIENT_USER_AGENT
from people_gator_common.wikidata import create_people_of_all, fetch_property_values_for_entities
from people_gator_common.wikidata.fetchers import fetch_people_of_property, fetch_texts_of_entities
from people_gator_common.wikidata.processors import unique_values_from_csv


country_property_id = "P27" # country of citizenship
language_property_id = "P1412" # language spoken, written or signed
image_property_id = "P18" # image
viaf_id_property_id = "P214" # VIAF ID
nlcr_aut_id_property_id = "P691" # NLCR AUT ID
date_of_birth_property_id = "P569" # Date of birth
date_of_death_property_id = "P570" # Date of death

czechia_id = "Q213" # Czech Republic (Q213)
czechoslovakia_id = "Q33946" # Czechoslovakia (Q33946) is a historical country that existed from 1918 to 1992
slovakia_id = "Q214" # Slovakia (Q214)
czech_language_id = "Q9067" # Czech language (Q9067)
slovak_language_id = "Q9058" # Slovak language (Q9058)

def parse_args():
    parser = argparse.ArgumentParser(description="CLI for Wikidata DAG pipelines.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Directory to save output files."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--receive-batch-size",
        type=int,
        default=40_000,
        help="Batch size for SPARQL queries (default: 40,000).",
    )
    parser.add_argument(
        "--send-batch-size",
        type=int,
        default=10_000,
        help="Batch size for SPARQL queries (default: 10,000).",
    )
    parser.add_argument(
        "--rate-limit-interval",
        type=float,
        default=5.0,
        help="Minimum interval between SPARQL requests in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=5,
        help="Maximum number of retries for SPARQL requests (default: 5).",
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=float,
        default=15.0,
        help="Backoff time in seconds between retries for SPARQL requests (default: 15.0).",
    )
    return parser.parse_args()

def people_collectors_main(
    czech_people_file: Path,
    czechoslovakia_people_file: Path,
    slovakia_people_file: Path,
    czech_language_people_file: Path,
    slovak_language_people_file: Path,
    *,
    client: SPARQLWrapper,
    receive_batch_size: int,
    rate_limiter: RateLimiter,
):
    logger = logging.getLogger(__name__ + "." + people_collectors_main.__name__)

    logger.info("Starting to fetch people of %s=%s", country_property_id, czechia_id)
    fetch_people_of_property(
        property_id=country_property_id,
        value_id=czechia_id,
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
        output_file=czech_people_file,
    )
    logger.info("Starting to fetch people of %s=%s", country_property_id, czechoslovakia_id)
    fetch_people_of_property(
        property_id=country_property_id,
        value_id=czechoslovakia_id,
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
        output_file=czechoslovakia_people_file,
    )
    logger.info("Starting to fetch people of %s=%s", country_property_id, slovakia_id)
    fetch_people_of_property(
        property_id=country_property_id,
        value_id=slovakia_id,
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
        output_file=slovakia_people_file,
    )
    logger.info("Starting to fetch people of %s=%s", language_property_id, czech_language_id)
    fetch_people_of_property(
        property_id=language_property_id,
        value_id=czech_language_id,
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
        output_file=czech_language_people_file,
    )
    logger.info("Starting to fetch people of %s=%s", language_property_id, slovak_language_id)
    fetch_people_of_property(
        property_id=language_property_id,
        value_id=slovak_language_id,
        client=client,
        batch_size=receive_batch_size,
        use_cursor=True,
        rate_limiter=rate_limiter,
        output_file=slovak_language_people_file,
    )

def people_property_enrichment_main(
    people_of_all_file: Path,
    people_all_images_file: Path,
    people_all_countries_file: Path,
    people_all_languages_file: Path,
    people_all_viaf_ids_file: Path,
    people_all_nlcr_aut_ids_file: Path,
    people_all_dates_of_birth_file: Path,
    people_all_dates_of_death_file: Path,
    *,
    send_batch_size: int,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
):
    logger = logging.getLogger(__name__ + "." + people_property_enrichment_main.__name__)
    
    logger.info("Starting to fetch property values for all people in %s for image", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=image_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_images_file,
    )
    logger.info("Starting to fetch property values for all people in %s for country", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=country_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_countries_file,
    )
    logger.info("Starting to fetch property values for all people in %s for language", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=language_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_languages_file,
    )
    logger.info("Starting to fetch property values for all people in %s for VIAF ID", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=viaf_id_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_viaf_ids_file,
    )
    logger.info("Starting to fetch property values for all people in %s for NLCR AUT ID", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=nlcr_aut_id_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_nlcr_aut_ids_file,
    )
    logger.info("Starting to fetch property values for all people in %s for date of birth", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=date_of_birth_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_dates_of_birth_file,
    )
    logger.info("Starting to fetch property values for all people in %s for date of death", people_of_all_file)
    fetch_property_values_for_entities(
        property_id=date_of_death_property_id,
        entities_csv_path=people_of_all_file,
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_all_dates_of_death_file,
    )

def property_valued_deduplication_main(
    people_all_countries_file: Path,
    people_all_languages_file: Path,
    people_all_viaf_ids_file: Path,
    people_all_nlcr_aut_ids_file: Path,
    unique_countries_file: Path,
    unique_languages_file: Path,
    unique_viaf_id_file: Path,
    unique_nlcr_aut_id_file: Path,
):
    logger = logging.getLogger(__name__ + "." + property_valued_deduplication_main.__name__)
    
    logger.info("Starting to deduplicate country values from %s", people_all_countries_file)
    unique_values_from_csv(
        source_csv=people_all_countries_file,
        target_csv=unique_countries_file,
    )
    logger.info("Starting to deduplicate language values from %s", people_all_languages_file)
    unique_values_from_csv(
        source_csv=people_all_languages_file,
        target_csv=unique_languages_file,
    )
    logger.info("Starting to deduplicate VIAF ID values from %s", people_all_viaf_ids_file)
    unique_values_from_csv(
        source_csv=people_all_viaf_ids_file,
        target_csv=unique_viaf_id_file,
    )
    logger.info("Starting to deduplicate NLCR AUT ID values from %s", people_all_nlcr_aut_ids_file)
    unique_values_from_csv(
        source_csv=people_all_nlcr_aut_ids_file,
        target_csv=unique_nlcr_aut_id_file,
    )

def labels_descriptions_alt_labels_fetching_main(
    people_of_all_file: Path,
    unique_countries_file: Path,
    unique_languages_file: Path,
    
    people_of_all_labels_file: Path,
    people_of_all_descriptions_file: Path,
    people_of_all_alt_labels_file: Path,
    unique_countries_labels_file: Path,
    unique_countries_descriptions_file: Path,
    unique_countries_alt_labels_file: Path,
    unique_languages_labels_file: Path,
    unique_languages_descriptions_file: Path,
    unique_languages_alt_labels_file: Path,
    *,
    send_batch_size: int,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
):
    logger = logging.getLogger(__name__ + "." + labels_descriptions_alt_labels_fetching_main.__name__)
    
    logger.info("Starting to fetch labels for all people in %s", people_of_all_file)
    fetch_texts_of_entities(
        entities_csv_path=people_of_all_file,
        text_property="labels",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_of_all_labels_file,
    )
    logger.info("Starting to fetch descriptions for all people in %s", people_of_all_file)
    fetch_texts_of_entities(
        entities_csv_path=people_of_all_file,
        text_property="descriptions",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_of_all_descriptions_file,
    )
    logger.info("Starting to fetch alt labels for all people in %s", people_of_all_file)
    fetch_texts_of_entities(
        entities_csv_path=people_of_all_file,
        text_property="alt_labels",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=people_of_all_alt_labels_file,
    )
    logger.info("Starting to fetch labels for all unique countries in %s", unique_countries_file)
    fetch_texts_of_entities(
        entities_csv_path=unique_countries_file,
        text_property="labels",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=unique_countries_labels_file,
    )
    logger.info("Starting to fetch descriptions for all unique countries in %s", unique_countries_file)
    fetch_texts_of_entities(
        entities_csv_path=unique_countries_file,
        text_property="descriptions",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=unique_countries_descriptions_file,
    )
    logger.info("Starting to fetch alt labels for all unique countries in %s", unique_countries_file)
    fetch_texts_of_entities(
        entities_csv_path=unique_countries_file,
        text_property="alt_labels",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=unique_countries_alt_labels_file,
    )
    logger.info("Starting to fetch labels for all unique languages in %s", unique_languages_file    )
    fetch_texts_of_entities(
        entities_csv_path=unique_languages_file,
        text_property="labels",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=unique_languages_labels_file,
    )
    logger.info("Starting to fetch descriptions for all unique languages in %s", unique_languages_file)
    fetch_texts_of_entities(
        entities_csv_path=unique_languages_file,
        text_property="descriptions",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=unique_languages_descriptions_file,
    )
    logger.info("Starting to fetch alt labels for all unique languages in %s", unique_languages_file)
    fetch_texts_of_entities(
        entities_csv_path=unique_languages_file,
        text_property="alt_labels",
        client=client,
        batch_size=send_batch_size,
        rate_limiter=rate_limiter,
        output_file=unique_languages_alt_labels_file,
    )

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    receive_batch_size = int(args.receive_batch_size)
    send_batch_size = int(args.send_batch_size)
    rate_limit_interval = float(args.rate_limit_interval)
    rate_limit_retries = int(args.rate_limit_retries)
    rate_limit_backoff = float(args.rate_limit_backoff)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    client = SPARQLWrapper(
        "https://query.wikidata.org/sparql", 
        agent=CLIENT_USER_AGENT
    )
    client.setMethod("POST")
    rate_limiter = RateLimiter(
        min_request_interval=rate_limit_interval,
        max_retries=rate_limit_retries,
        retry_backoff_seconds=rate_limit_backoff
    )
    

    czech_people_file = output_dir / f"people_of_{country_property_id}={czechia_id}.csv"
    czechoslovakia_people_file = output_dir / f"people_of_{country_property_id}={czechoslovakia_id}.csv"
    slovakia_people_file = output_dir / f"people_of_{country_property_id}={slovakia_id}.csv"
    czech_language_people_file = output_dir / f"people_of_{language_property_id}={czech_language_id}.csv"
    slovak_language_people_file = output_dir / f"people_of_{language_property_id}={slovak_language_id}.csv"
    people_collectors_main(
        czech_people_file=czech_people_file,
        czechoslovakia_people_file=czechoslovakia_people_file,
        slovakia_people_file=slovakia_people_file,
        czech_language_people_file=czech_language_people_file,
        slovak_language_people_file=slovak_language_people_file,
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

    people_all_images_file = output_dir / f"people_all_{image_property_id}.csv"
    people_all_countries_file = output_dir / f"people_all_{country_property_id}.csv"
    people_all_languages_file = output_dir / f"people_all_{language_property_id}.csv"
    people_all_viaf_ids_file = output_dir / f"people_all_{viaf_id_property_id}.csv"
    people_all_nlcr_aut_ids_file = output_dir / f"people_all_{nlcr_aut_id_property_id}.csv"
    people_all_dates_of_birth_file = output_dir / f"people_all_{date_of_birth_property_id}.csv"
    people_all_dates_of_death_file = output_dir / f"people_all_{date_of_death_property_id}.csv"
    people_property_enrichment_main(
        people_of_all_file=people_of_all_file,
        people_all_images_file=people_all_images_file,
        people_all_countries_file=people_all_countries_file,
        people_all_languages_file=people_all_languages_file,
        people_all_viaf_ids_file=people_all_viaf_ids_file,
        people_all_nlcr_aut_ids_file=people_all_nlcr_aut_ids_file,
        people_all_dates_of_birth_file=people_all_dates_of_birth_file,
        people_all_dates_of_death_file=people_all_dates_of_death_file,
        
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )
    
    unique_countries_file = output_dir / "unique_countries.csv"
    unique_languages_file = output_dir / "unique_languages.csv"
    unique_nlcr_aut_id_file = output_dir / "unique_nl_cr_aut_id.csv"
    unique_viaf_id_file = output_dir / "unique_viaf_id.csv"
    property_valued_deduplication_main(
        people_all_countries_file=people_all_countries_file,
        people_all_languages_file=people_all_languages_file,
        people_all_viaf_ids_file=people_all_viaf_ids_file,
        people_all_nlcr_aut_ids_file=people_all_nlcr_aut_ids_file,
        unique_countries_file=unique_countries_file,
        unique_languages_file=unique_languages_file,
        unique_viaf_id_file=unique_viaf_id_file,
        unique_nlcr_aut_id_file=unique_nlcr_aut_id_file,
    )

    people_of_all_labels_file = output_dir / "people_of_all_labels.csv"
    people_of_all_descriptions_file = output_dir / "people_of_all_descriptions.csv"
    people_of_all_alt_labels_file = output_dir / "people_of_all_alt_labels.csv"


    unique_countries_labels_file = output_dir / "unique_countries_labels.csv"
    unique_countries_descriptions_file = output_dir / "unique_countries_descriptions.csv"
    unique_countries_alt_labels_file = output_dir / "unique_countries_alt_labels.csv"

    unique_languages_labels_file = output_dir / "unique_languages_labels.csv"
    unique_languages_descriptions_file = output_dir / "unique_languages_descriptions.csv"
    unique_languages_alt_labels_file = output_dir / "unique_languages_alt_labels.csv"

    labels_descriptions_alt_labels_fetching_main(
        people_of_all_file=people_of_all_file,
        unique_countries_file=unique_countries_file,
        unique_languages_file=unique_languages_file,
        people_of_all_labels_file=people_of_all_labels_file,
        people_of_all_descriptions_file=people_of_all_descriptions_file,
        people_of_all_alt_labels_file=people_of_all_alt_labels_file,
        unique_countries_labels_file=unique_countries_labels_file,
        unique_countries_descriptions_file=unique_countries_descriptions_file,
        unique_countries_alt_labels_file=unique_countries_alt_labels_file,
        unique_languages_labels_file=unique_languages_labels_file,
        unique_languages_descriptions_file=unique_languages_descriptions_file,
        unique_languages_alt_labels_file=unique_languages_alt_labels_file,
        send_batch_size=send_batch_size,
        client=client,
        rate_limiter=rate_limiter,
    )


if __name__ == "__main__":
    main()
