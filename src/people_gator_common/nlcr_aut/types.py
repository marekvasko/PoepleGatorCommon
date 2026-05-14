from typing import TypedDict

type SubfieldCodeDictType = dict[str, list[str]]
type FieldCodeDataDictType = dict[str, list[str]]
type FieldCodeSubfieldDictType = dict[str, list[SubfieldCodeDictType]]
type FieldCodeDictType = dict[str, list[SubfieldCodeDictType] | list[str]]

class ExtractedInformation(TypedDict):
    entity: str
    value: str
