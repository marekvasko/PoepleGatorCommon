from __future__ import annotations

# Tag/subfield maps
indexed_tags = {
    '001': 'Control Number',
    '003': 'Control Number Identifier',
    '005': 'Date and Time of Latest Transaction',
    '008': 'Fixed-Length Data Elements',
    '010': 'Library of Congress Control Number',
    '024': 'Other Standard Identifier',
    '040': 'Cataloging Source',
    '046': 'Special Coded Dates',
    '100': 'Heading - Personal Name',
    '368': 'Other Attributes of Person or Corporate Body',
    '370': 'Associated Place',
    '372': 'Field of Activity',
    '373': 'Associated Group',
    '374': 'Occupation',
    '375': 'Gender',
    '376': 'Family Information',
    '377': 'Associated Language',
    '378': 'Fuller Form of Personal Name',
    '400': 'See From Tracing - Personal Name',
    '410': 'See From Tracing - Corporate Name',
    '500': 'See Also From Tracing - Personal Name',
    '510': 'See Also From Tracing - Corporate Name',
    '511': 'See Also From Tracing - Meeting Name',
    '665': 'History Reference',
    '667': 'Nonpublic General Note',
    '670': 'Source Data Found',
    '675': 'Source Data Not Found',
    '677': 'Definition',
    '678': 'Biographical or Historical Data',
    '680': 'Public General Note',
    '856': 'Electronic Location and Access',
    '998': ''
}
indexed_subfields =   {
  "010" : {
    "a": "LC control number",
   },
  "024" : {
    "2": "Source",
    "a": "Standard number or code",
    "d": "Additional codes following the standard number or code",
   },
  "040" : {
    "a": "Original cataloging agency",
    "b": "Language of cataloging",
    "c": "Transcribing agency",
    "d": "Modifying agency",
    "e": "Description conventions",
   },
  "046" : {
    "f": "Birth date",
    "g": "Death date",
    "s": "Start period",
    "t": "End period",
   },
  "100" : {
    "7": "Data provenance",
    "a": "Personal name",
    "b": "Numeration",
    "c": "Titles and other words associated with a name",
    "d": "Dates associated with a name",
    "q": "Fuller form of name",
   },
  "368" : {
    "a": "Type of corporate body",
    "b": "Type of jurisdiction",
    "c": "Other designation",
    "d": "Title of person",
    "s": "Start period",
    "t": "End period",
   },
  "370" : {
    "a": "Place of birth",
    "b": "Place of death",
    "c": "Associated country",
    "e": "Place of residence/headquarters",
    "f": "Other associated place",
    "g": "Place of origin of work or expression",
    "s": "Start period",
    "t": "End period",
   },
  "372" : {
    "a": "Field of activity",
    "s": "Start period",
    "t": "End period",
   },
  "373" : {
    "2": "Source of term",
    "a": "Associated group",
    "s": "Start period",
    "t": "End period",
   },
  "374" : {
    "a": "Occupation",
    "s": "Start period",
    "t": "End period",
   },
  "375" : {
    "a": "Gender",
    "s": "Start period",
    "t": "End period",
   },
  "376" : {
    "a": "Type of family",
    "c": "Hereditary title",
    "s": "Start period",
    "t": "End period",
   },
  "377" : {
    "a": "Language code",
   },
  "378" : {
    "q": "Fuller form of personal name",
   },
  "400" : {
    "0": "",
    "2": "",
    "7": "Data provenance",
    "a": "Personal name",
    "b": "Numeration",
    "c": "Titles and other words associated with a name",
    "d": "Dates associated with a name",
    "i": "Relationship information",
    "q": "Name of part/section of a work",
    "w": "Control subfield",
   },
  "410" : {
    "0": "",
    "a": "Corporate name or jurisdiction name as entry element",
   },
  "500" : {
    "7": "Data provenance",
    "a": "Personal name",
    "b": "Numeration",
    "c": "Titles and other words associated with a name",
    "d": "Dates associated with a name",
    "i": "Relationship information",
    "q": "Fuller form of name",
    "w": "Control subfield",
   },
  "510" : {
    "7": "Data provenance",
    "a": "Corporate name or jurisdiction name as entry element",
    "b": "Subordinate unit",
    "w": "Control subfield",
   },
  "511" : {
    "7": "Data provenance",
    "a": "Meeting name or jurisdiction name as entry element",
   },
  "665" : {
    "a": "History reference",
   },
  "667" : {
    "a": "Nonpublic general note",
   },
  "670" : {
    "a": "Source citation",
    "b": "Information found",
    "u": "Uniform Resource Identifier",
   },
  "675" : {
    "a": "Source citation",
    "b": "",
   },
  "677" : {
    "a": "Definition",
   },
  "678" : {
    "a": "Biographical or historical data",
    "b": "Expansion",
    "u": "Uniform Resource Identifier",
   },
  "680" : {
    "a": "Heading or subdivision term",
    "i": "Explanatory text",
   },
  "856" : {
    "3": "Materials specified",
    "4": "",
    "u": "Uniform Resource Identifier",
    "y": "Link text",
    "z": "Public note",
   },
  "998" : {
    "a": "",
   },
}

def print_tags_and_subfields(tags: list[str], subfields: dict[str, list[str]]) -> None:
    for tag in tags:
        indexed_tag = indexed_tags.get(tag)
        indexed_tag = "[UNKNOWN TAG]" if indexed_tag is None or indexed_tag == "" else indexed_tag
        print(f"[{tag}] {indexed_tag}")
        for subfield_code in subfields.get(tag, []):
            subfield_name = indexed_subfields.get(tag, {}).get(subfield_code)
            subfield_name = "[UNKNOWN SUBFIELD]" if subfield_name is None or subfield_name == "" else subfield_name
            print(f"  [{subfield_code}] {subfield_name}")

def field_subfield_to_names(tag: str, subfield_code: str | None) -> str:
    tag_name = indexed_tags.get(tag)
    tag_name = "[UNKNOWN TAG]" if tag_name is None or tag_name == "" else tag_name
    if subfield_code is None:
        return tag_name
    subfield_name = indexed_subfields.get(tag, {}).get(subfield_code)
    subfield_name = "[UNKNOWN SUBFIELD]" if subfield_name is None or subfield_name == "" else subfield_name
    return f"{tag_name} -> {subfield_name}"
