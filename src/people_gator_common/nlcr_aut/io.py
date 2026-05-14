from __future__ import annotations

import json
from pathlib import Path
import tqdm
from collections.abc import Callable
from pymarc import Record as PymarcRecord, parse_xml_to_array


def to_subfield_code_dict(subfields: list[dict[str, str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for subfield in subfields:
        key_list = list(subfield.keys())
        if len(key_list) != 1:
            raise ValueError(f"Expected only one key in {subfield}, but got {len(key_list)}")
        key, value = next(iter(subfield.items()))
        if not isinstance(value, str):
            raise ValueError(f"Expected string value for subfield code {key}, but got {type(value)}")
        if key not in result:
            result[key] = []
        result[key].append(value)
    return result


def to_field_code_dict(record: dict) -> dict[str, list[dict | str]]:
    result: dict[str, list[dict | str]] = {}
    for field in record["fields"]:
        key_list = list(field.keys())
        if len(key_list) != 1:
            raise ValueError(f"Expected only one key in {field}, but got {len(key_list)}")
        key, value = next(iter(field.items()))
        if key not in result:
            result[key] = []
        if isinstance(value, str):
            result[key].append(value)
            continue
        if isinstance(value, dict):
            subfields = value.get("subfields", [])
            result[key].append(to_subfield_code_dict(subfields))
            continue
        raise ValueError(f"Expected string or dict value for field code {key}, but got {type(value)}")
    return result


def get_control_number(field_code_dict: dict[str, list]) -> str:
    if "001" not in field_code_dict:
        raise ValueError(f"Expected '001' field in {field_code_dict}, but it is not present")
    if len(field_code_dict["001"]) != 1:
        raise ValueError(f"Expected only one '001' field in {field_code_dict}, but got {len(field_code_dict['001'])}")
    value = field_code_dict["001"][0]
    if not isinstance(value, str):
        raise ValueError(f"Expected string value for '001' field, but got {type(value)}")
    return value


def data_list_to_exported_records(control_number: str, data_list: list[str]) -> list[dict]:
    return [{"entity": control_number, "value": data} for data in data_list]


def build_field_data_extractor(field_code: str) -> Callable[[dict], list[dict]]:
    def extractor(field_code_dict: dict) -> list[dict]:
        result: list[str] = []
        if field_code not in field_code_dict:
            return []
        for value in field_code_dict[field_code]:
            if not isinstance(value, str):
                raise ValueError(f"Expected string value for field code {field_code}, but got {type(value)}")
            result.append(value)
        return data_list_to_exported_records(get_control_number(field_code_dict), result)

    return extractor


def build_subfield_data_extractor(field_code: str, subfield_code: str) -> Callable[[dict], list[dict]]:
    def extractor(field_code_dict: dict) -> list[dict]:
        result: list[str] = []
        if field_code not in field_code_dict:
            return []
        for value in field_code_dict[field_code]:
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict value for field code {field_code}, but got {type(value)}")
            if subfield_code not in value:
                continue
            subfield_value = value[subfield_code]
            if not isinstance(subfield_value, list):
                raise ValueError(f"Expected list value for subfield code {subfield_code} in field code {field_code}, but got {type(subfield_value)}")
            for subfield_value_item in subfield_value:
                if not isinstance(subfield_value_item, str):
                    raise ValueError(f"Expected string value for subfield code {subfield_code} in field code {field_code}, but got {type(subfield_value_item)}")
                result.append(subfield_value_item)
        return data_list_to_exported_records(get_control_number(field_code_dict), result)

    return extractor


def marc_to_jsonl(input_marc_xml_path: Path, output_jsonl_path: Path) -> None:
    records: list[PymarcRecord] = parse_xml_to_array(str(input_marc_xml_path))
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for record in tqdm.tqdm(records, desc="Converting to JSONL"):
            record_dict = record.as_dict()
            field_code_dict = to_field_code_dict(record_dict)  # type: ignore
            f.write(json.dumps(field_code_dict) + "\n")


def csv_escape(value: str) -> str:
    if '"' in value:
        value = value.replace('"', '""')
    if ',' in value or '"' in value or '\n' in value:
        value = f'"{value}"'
    return value


def jsonl_information_extractor(input_jsonl_path: Path, output_csv_path: Path, information_extractor: Callable[[dict], list[dict]]) -> None:
    with open(input_jsonl_path, 'r', encoding='utf-8') as fi, open(output_csv_path, 'w', encoding='utf-8') as fo:
        fo.write(f"entity,value\n")
        for line in tqdm.tqdm(fi, desc="Extracting Information from JSONL"):
            field_code_dict: dict = json.loads(line)
            extracted_information = information_extractor(field_code_dict)
            for info in extracted_information:
                fo.write(f"{csv_escape(info['entity'])},{csv_escape(info['value'])}\n")
