from __future__ import annotations
from people_gator_common.nlcr_aut.types import (
    SubfieldCodeDictType,
    FieldCodeDataDictType,
    FieldCodeSubfieldDictType,
    FieldCodeDictType,
    ExtractedInformation,
)
from pymarc import Record as PymarcRecord, Field as PymarcField, Subfield as PymarcSubfield
from collections.abc import Callable


def convert_to_subfield_code_dict(subfields: list[PymarcSubfield]) -> dict[str, list[str]]:
    result: SubfieldCodeDictType = {}
    for subfield in subfields:
        code = subfield.code
        value = subfield.value
        if code in result:
            result[code].append(value)
        else:
            result[code] = [value]
    return result

def convert_fields_to_field_code_dict(fields: list[PymarcField]) -> FieldCodeDictType:
    results_data: FieldCodeDataDictType = {}
    results_subfield: FieldCodeSubfieldDictType = {}
    for field in fields:
        tag = field.tag
        data = field.data
        subfields = field.subfields
        if data is not None and subfields:
            raise ValueError(f"Expected either data or subfields for field code {tag}, but got both")
        if data is not None:
            if not isinstance(data, str):
                raise ValueError(f"Expected string value for field code {tag}, but got {type(data)}")
            if tag in results_subfield:
                raise ValueError(f"Expected only one of data or subfields for field code {tag}, but got both")
            if tag in results_data:
                results_data[tag].append(data)
            else:
                results_data[tag] = [data]
            continue
        if subfields:
            if not isinstance(subfields, list):
                raise ValueError(f"Expected list value for subfields of field code {tag}, but got {type(subfields)}")
            if tag in results_data:
                raise ValueError(f"Expected only one of data or subfields for field code {tag}, but got both")
            subfield_code_dict = convert_to_subfield_code_dict(subfields)
            if tag in results_subfield:
                results_subfield[tag].append(subfield_code_dict)
            else:
                results_subfield[tag] = [subfield_code_dict]
            continue
        raise ValueError(f"Expected either data or subfields for field code {tag}, but got neither")
    
    result: FieldCodeDictType = {}
    for tag, data_list in results_data.items():
        result[tag] = data_list
    for tag, subfield_list in results_subfield.items():
        if tag in result:
            raise ValueError(f"Expected only one of data or subfields for field code {tag}, but got both")
        result[tag] = subfield_list
    return result

def convert_record_to_field_code_dict(record: PymarcRecord) -> FieldCodeDictType:
    return convert_fields_to_field_code_dict(record.fields)

def get_control_number(field_code_dict: dict[str, list]) -> str:
    if "001" not in field_code_dict:
        raise ValueError(f"Expected '001' field in {field_code_dict}, but it is not present")
    if len(field_code_dict["001"]) != 1:
        raise ValueError(f"Expected only one '001' field in {field_code_dict}, but got {len(field_code_dict['001'])}")
    value = field_code_dict["001"][0]
    if not isinstance(value, str):
        raise ValueError(f"Expected string value for '001' field, but got {type(value)}")
    return value


def data_list_to_exported_records(control_number: str, data_list: list[str]) -> list[ExtractedInformation]:
    return [{"entity": control_number, "value": data} for data in data_list]


def build_field_data_extractor(field_code: str) -> Callable[[dict], list[ExtractedInformation]]:
    def extractor(field_code_dict: dict) -> list[ExtractedInformation]:
        result: list[str] = []
        if field_code not in field_code_dict:
            return []
        for value in field_code_dict[field_code]:
            if not isinstance(value, str):
                raise ValueError(f"Expected string value for field code {field_code}, but got {type(value)}")
            result.append(value)
        return data_list_to_exported_records(get_control_number(field_code_dict), result)

    return extractor


def build_subfield_data_extractor(field_code: str, subfield_code: str) -> Callable[[dict], list[ExtractedInformation]]:
    def extractor(field_code_dict: dict) -> list[ExtractedInformation]:
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


# csv_escape has moved to people_gator_common.utils
from people_gator_common.utils import csv_escape as csv_escape  # re-export for backward compat