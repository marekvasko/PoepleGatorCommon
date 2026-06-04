from __future__ import annotations

import json
from pathlib import Path
from people_gator_common.filesystem import open_file
import tqdm
from collections.abc import Callable
from pymarc import (
    Record as PymarcRecord, 
    parse_xml_to_array
)
from people_gator_common.nlcr_aut.types import (
    FieldCodeDataDictType,
    ExtractedInformation,
)
from people_gator_common.nlcr_aut.parsers import convert_record_to_field_code_dict
from people_gator_common.utils import csv_escape

def marc_to_jsonl(
    input_marc_xml_path: str, 
    output_jsonl_path: str
) -> None:
    with open_file(input_marc_xml_path, 'r') as f:
        records: list[PymarcRecord] = parse_xml_to_array(f)
    with open_file(output_jsonl_path, "w") as f:
        for record in tqdm.tqdm(records, desc="Converting to JSONL"):
            field_code_dict = convert_record_to_field_code_dict(record)
            f.write(json.dumps(field_code_dict) + "\n")

def sort_subfields(subfields: set[str] | None) -> list[str] | None:
    if subfields is None:
        return None
    return sorted(subfields)

def tags_and_subfields_in_jsonl(
    input_jsonl_path: str,
    output_tags_and_subfields_json_path: str,
) -> None:
    tags: dict[str, None | set[str]] = {}
    with open_file(input_jsonl_path, 'r') as f:
        for line in tqdm.tqdm(f, desc="Gathering tags and subfields from JSONL"):
            field_code_dict: FieldCodeDataDictType = json.loads(line)
            for tag, values in field_code_dict.items():
                is_data_field = all(isinstance(value, str) for value in values)
                is_subfield_field = all(isinstance(value, dict) for value in values)
                if is_data_field and tag not in tags:
                    tags[tag] = None
                if is_data_field and tag in tags and tags[tag] is not None:
                    raise ValueError(f"Expected only one of data or subfields for field code {tag}, but got both")
                if is_subfield_field and tag not in tags:
                    tags[tag] = set()
                if is_subfield_field and tag in tags and tags[tag] is None:
                    raise ValueError(f"Expected only one of data or subfields for field code {tag}, but got both")
                if is_subfield_field and tag in tags and tags[tag] is not None:
                    for value in values:
                        if not isinstance(value, dict):
                            raise ValueError(f"Expected dict value for subfields of field code {tag}, but got {type(value)}")
                        for subfield_code in value.keys():
                            subfield_set = tags[tag]
                            if subfield_set is None:
                                raise ValueError(f"Expected set for subfields of field code {tag}, but got None")
                            subfield_set.add(subfield_code)
    tags_sorted = {tag: sort_subfields(tags[tag]) for tag in sorted(tags.keys())}
    with open_file(output_tags_and_subfields_json_path, 'w') as f:
        json.dump(tags_sorted, f, indent=2)
    

def jsonl_information_extractor(
    input_jsonl_path: str, 
    output_csv_path: str, 
    information_extractor: Callable[[FieldCodeDataDictType], list[ExtractedInformation]]
) -> None:
    with open_file(input_jsonl_path, 'r') as fi, open_file(output_csv_path, 'w') as fo:
        fo.write(f"entity,value\n")
        for line in tqdm.tqdm(fi, desc="Extracting Information from JSONL"):
            field_code_dict: FieldCodeDataDictType = json.loads(line)
            extracted_information = information_extractor(field_code_dict)
            for info in extracted_information:
                fo.write(f"{csv_escape(info['entity'])},{csv_escape(info['value'])}\n")

def csv_reparser(
    input_csv_path: str,
    output_csv_path: str,
    csv_reparser_function: Callable[[str], str],
) -> None:
    with open_file(input_csv_path, 'r') as fi, open_file(output_csv_path, 'w') as fo:
        header = fi.readline()
        fo.write(header)
        for line in tqdm.tqdm(fi, desc="Reparsing CSV"):
            entity, value = line.strip().split(",", 1)
            reparsed_value = csv_reparser_function(value)
            fo.write(
                f"{entity},{csv_escape(reparsed_value)}\n"
            )
