from __future__ import annotations
from collections.abc import Callable
from typing import Literal

def build_historical_european_countries_query(*, limit: int, offset: int) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity WHERE {{
  ?entity wdt:P31 wd:Q3024240 ;
          wdt:P30 wd:Q46 .
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def build_european_country_query(*, limit: int, offset: int) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity WHERE {{
  ?entity wdt:P31 wd:Q6256 ;
           wdt:P30 wd:Q46 .
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def build_people_of_property_query(*, property_id: str, value_id: str, limit: int, offset: int) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity WHERE {{
  ?entity wdt:P31 wd:Q5 ;
          wdt:{property_id} wd:{value_id} .
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def build_people_of_property_cursor_query(*, property_id: str, value_id: str, after_id: str, limit: int) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity WHERE {{
  ?entity wdt:P31 wd:Q5 ;
          wdt:{property_id} wd:{value_id} .
  FILTER (STR(?entity) > "{after_id}")
}}
ORDER BY ?entity
LIMIT {limit}
""".strip()


def build_property_query_builder(property_id: str, value_var: str = "value") -> Callable[[list[str]], str]:
    def builder(entity_ids: list[str]) -> str:
        values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity ?{value_var} WHERE {{
  VALUES ?entity {{ {values} }}
  ?entity wdt:{property_id} ?{value_var} .
}}
""".strip()

    return builder


def build_label_query(entity_ids: list[str]) -> str:
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?entity ?entityText (lang(?entityText) AS ?entityTextLanguage) WHERE {{
VALUES ?entity {{ {values} }}
?entity rdfs:label ?entityText .
}}
""".strip()


def build_description_query(entity_ids: list[str]) -> str:
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX schema: <http://schema.org/>
SELECT ?entity ?entityText (lang(?entityText) AS ?entityTextLanguage) WHERE {{
VALUES ?entity {{ {values} }}
?entity schema:description ?entityText .
}}
""".strip()


def build_also_known_as_query(entity_ids: list[str]) -> str:
    values = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?entity ?entityText (lang(?entityText) AS ?entityTextLanguage) WHERE {{
VALUES ?entity {{ {values} }}
?entity skos:altLabel ?entityText .
}}
""".strip()

def build_text_query_builder(text_property: Literal["labels", "descriptions", "alt_labels"]) -> Callable[[list[str]], str]:
    if text_property == "labels":
        return build_label_query
    elif text_property == "descriptions":
        return build_description_query
    elif text_property == "alt_labels":
        return build_also_known_as_query
    else:
        raise ValueError(f"Unsupported text property: {text_property}")