"""Manifest-based, resumable WikiData entity enrichment.

Enrichers fetch *property values* or *text fields* (labels, descriptions,
alt-labels) for a known set of entities and merge the results into a CSV.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable

from SPARQLWrapper import SPARQLWrapper

from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.wikidata_v2.common import (
    TaskModel,
    ManifestModel,
    cleanup,
    collect_results,
    execute_tasks,
    get_manifest_tasks as _get_manifest_tasks,
    get_pending_tasks as _get_pending_tasks,
    init_manifest,
    run_pipeline,
)


logger = logging.getLogger(__name__)

PROPERTY_IMAGE_WIKIDATA_ID = "wdt:P18"
PROPERTY_DATE_OF_BIRTH_WIKIDATA_ID = "wdt:P569"
PROPERTY_DATE_OF_DEATH_WIKIDATA_ID = "wdt:P570"
PROPERTY_GENDER_WIKIDATA_ID = "wdt:P21"
PROPERTY_OCCUPATION_WIKIDATA_ID = "wdt:P106"
PROPERTY_DESCRIPTION = "schema:description"
PROPERTY_LABEL = "rdfs:label"
PROPERTY_ALT_LABEL = "skos:altLabel"


# ---------------------------------------------------------------------------
# Task & Manifest models
# ---------------------------------------------------------------------------


class EnricherTask(TaskModel):
    total: int
    batch_start: int
    batch_end: int
    entities: list[str]

    @property
    def frame_name(self) -> str:
        batch_start_str = str(self.batch_start).zfill(len(str(self.total)))
        batch_end_str = str(self.batch_end).zfill(len(str(self.total)))
        entity_hash = hashlib.sha256(",".join(sorted(self.entities)).encode()).hexdigest()[:16]
        return f"{self.name}__total={self.total}__batch_start={batch_start_str}__batch_end={batch_end_str}__hash={entity_hash}.csv"


class EnricherManifest(ManifestModel):
    tasks: list[EnricherTask]


# ---------------------------------------------------------------------------
# Convenience re-exports of generic helpers with concrete types
# ---------------------------------------------------------------------------


def get_manifest_tasks(directory: str) -> dict[str, EnricherTask]:
    """Read the enricher manifest from *directory*/manifest.json."""
    return _get_manifest_tasks(directory, EnricherManifest, lambda t: t.frame_name)


def get_pending_tasks(directory: str) -> list[EnricherTask]:
    """Return enricher tasks whose frames are not yet completed."""
    return _get_pending_tasks(directory, EnricherManifest, lambda t: t.frame_name)


# ---------------------------------------------------------------------------
# Query builders — produce SPARQL strings from a batch of entity IDs
# ---------------------------------------------------------------------------


def build_property_enricher_query(
    property_id: str,
    entities: list[str],
) -> str:
    """Build a SPARQL query that fetches ``wdt:{property_id}`` values."""
    values = " ".join(f"wd:{entity}" for entity in entities)
    return f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity ?value WHERE {{
    VALUES ?entity {{ {values} }}
    ?entity wdt:{property_id} ?value .
}}
""".strip()


def build_label_enricher_query(entities: list[str]) -> str:
    """Build a SPARQL query to fetch rdfs:labels for a batch of entities."""
    values = " ".join(f"wd:{entity}" for entity in entities)
    return f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?entity ?entityText (lang(?entityText) AS ?entityTextLanguage) WHERE {{
    VALUES ?entity {{ {values} }}
    ?entity rdfs:label ?entityText .
}}
""".strip()


def build_description_enricher_query(entities: list[str]) -> str:
    """Build a SPARQL query to fetch schema:descriptions for a batch of entities."""
    values = " ".join(f"wd:{entity}" for entity in entities)
    return f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX schema: <http://schema.org/>
SELECT ?entity ?entityText (lang(?entityText) AS ?entityTextLanguage) WHERE {{
    VALUES ?entity {{ {values} }}
    ?entity schema:description ?entityText .
}}
""".strip()


def build_alt_label_enricher_query(entities: list[str]) -> str:
    """Build a SPARQL query to fetch skos:altLabels for a batch of entities."""
    values = " ".join(f"wd:{entity}" for entity in entities)
    return f"""PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?entity ?entityText (lang(?entityText) AS ?entityTextLanguage) WHERE {{
    VALUES ?entity {{ {values} }}
    ?entity skos:altLabel ?entityText .
}}
""".strip()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def extract_entity_id(entity_uri: str) -> str:
    """Extract the WikiData entity ID (e.g. ``Q123``) from a full URI."""
    return entity_uri.rsplit("/", maxsplit=1)[-1]


# ---------------------------------------------------------------------------
# Manifest init / execute / collect / cleanup  (delegating to common)
# ---------------------------------------------------------------------------

def init_enricher_manifest(
    tmp_dir: str,
    task_name: str,
    query_builder: Callable[[list[str]], str],
    entities: list[str],
    batch_size: int,
    force: bool,
) -> None:
    """Initialise an enricher manifest for *task_name* using *query_builder*.

    ``query_builder`` receives a list of entity ID strings and must return
    a complete SPARQL SELECT query string.  This single entry-point handles
    both property enrichment (e.g. ``wdt:P18``) and text enrichment
    (labels, descriptions, alt_labels).
    """
    tmp_dir_path = os.path.join(tmp_dir, task_name)

    tasks = [
        EnricherTask(
            name=task_name,
            command=query_builder(entities[i : i + batch_size]),
            total=len(entities),
            batch_start=i,
            batch_end=i + len(entities[i : i + batch_size]) - 1,
            entities=entities[i : i + batch_size],
        )
        for i in range(0, len(entities), batch_size)
    ]
    manifest = EnricherManifest(
        output_file=f"{task_name}.csv",
        tasks=tasks,
    )
    init_manifest(tmp_dir_path, manifest, force=force)


def execute_enricher_tasks(
    tmp_dir: str,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    execute_tasks(
        tmp_dir=tmp_dir,
        manifest_type=EnricherManifest,
        task_key=lambda t: t.frame_name,
        client=client,
        rate_limiter=rate_limiter,
    )


def collect_enricher_tasks(
    tmp_dir: str,
    output_file: str,
) -> None:
    collect_results(
        tmp_dir=tmp_dir,
        output_file=output_file,
        manifest_type=EnricherManifest,
        task_key=lambda t: t.frame_name,
    )


def cleanup_enricher(
    tmp_dir_path: str,
) -> None:
    cleanup(tmp_dir_path)


# ---------------------------------------------------------------------------
# Public run function  (single entry-point for all enrichment)
# ---------------------------------------------------------------------------

def run_generic_enricher(
    output_file: str,
    tmp_dir: str,
    task_name: str,
    query_builder: Callable[[list[str]], str],
    entities: list[str],
    batch_size: int,
    force: bool,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    """Run the full init → execute → collect → cleanup cycle.

    *query_builder* is a ``Callable[[list[str]], str]`` that turns a batch
    of entity IDs into a SPARQL query.  Use the provided query-builder
    helpers (``build_property_enricher_query``, ``build_label_enricher_query``,
    etc.) or supply a custom one.
    """
    task_tmp_dir = os.path.join(tmp_dir, task_name)

    def _init() -> None:
        init_enricher_manifest(
            tmp_dir=tmp_dir,
            task_name=task_name,
            query_builder=query_builder,
            entities=entities,
            batch_size=batch_size,
            force=force,
        )

    run_pipeline(
        output_file=output_file,
        tmp_dir=tmp_dir,
        manifest_type=EnricherManifest,
        task_key=lambda t: t.frame_name,
        init_fn=_init,
        task_tmp_dir=task_tmp_dir,
        force=force,
        client=client,
        rate_limiter=rate_limiter,
    )