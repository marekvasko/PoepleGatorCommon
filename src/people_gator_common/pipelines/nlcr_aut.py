from __future__ import annotations
import logging

from people_gator_common.nlcr_aut import (
    download_and_extract_marc2021, 
    marc_to_jsonl,
    tags_and_subfields_in_jsonl,
    jsonl_information_extractor,
    build_field_data_extractor,
    build_subfield_data_extractor,
)
from people_gator_common.utils import unique_values_from_csv
from people_gator_common.filesystem import makedirs
from os.path import join

def run_nlcr_pipeline(
    output_dir: str,
    *,
    personal_authorities_link: str = "https://aleph.nkp.cz/data/aut_ja.xml.gz",
) -> None:
    logger = logging.getLogger(__name__)
    makedirs(output_dir)

    downloaded_xml_path = join(output_dir, "nlcr_aut_marc21.xml")
    indexed_marc_file_path = join(output_dir, "nlcr_aut_marc21.jsonl")
    
    tags_and_subfields_json_path = join(output_dir, "nlcr_aut__tags_and_subfields.json")

    control_numbers_csv_path = join(output_dir, "nlcr_aut__control_numbers.csv")
    person_names_csv_path = join(output_dir, "nlcr_aut__person_names.csv")
    descriptions_csv_path = join(output_dir, "nlcr_aut__descriptions.csv")
    also_known_as_csv_path = join(output_dir, "nlcr_aut__also_known_as.csv")
    death_dates_csv_path = join(output_dir, "nlcr_aut__death_dates.csv")
    birth_dates_csv_path = join(output_dir, "nlcr_aut__birth_dates.csv")
    unique_control_numbers_csv_path = join(output_dir, "unique_control_numbers.csv")

    logger.info("Downloading NLCR personal authorities from %s", personal_authorities_link)
    download_and_extract_marc2021(personal_authorities_link, downloaded_xml_path)

    # Convert to jsonl
    logger.info("Converting MARC XML to JSONL %s -> %s", downloaded_xml_path, indexed_marc_file_path)
    marc_to_jsonl(downloaded_xml_path, indexed_marc_file_path)

    logger.info("Extracting tags and subfields from JSONL to %s", tags_and_subfields_json_path)
    tags_and_subfields_in_jsonl(indexed_marc_file_path, tags_and_subfields_json_path)

    # Extract CSVs
    logger.info("Extract control numbers and other fields to CSVs")
    jsonl_information_extractor(
        input_jsonl_path=indexed_marc_file_path,
        output_csv_path=control_numbers_csv_path,
        information_extractor=build_field_data_extractor("001"),
    )
    tasks = [
        ("person names", person_names_csv_path, build_subfield_data_extractor("100", "a")),
        ("descriptions", descriptions_csv_path, build_subfield_data_extractor("678", "a")),
        ("also known as", also_known_as_csv_path, build_subfield_data_extractor("400", "a")),
        ("birth dates", birth_dates_csv_path, build_subfield_data_extractor("046", "f")),
        ("death dates", death_dates_csv_path, build_subfield_data_extractor("046", "g")),
    ]
    for field_name, output_csv_path, extractor in tasks:
        logger.info("Extracting %s to %s", field_name, output_csv_path)
        jsonl_information_extractor(
            input_jsonl_path=indexed_marc_file_path,
            output_csv_path=output_csv_path,
            information_extractor=extractor,
        )

    # Deduplicate control numbers
    logger.info("Creating unique control numbers CSV")
    unique_values_from_csv(control_numbers_csv_path, unique_control_numbers_csv_path)
