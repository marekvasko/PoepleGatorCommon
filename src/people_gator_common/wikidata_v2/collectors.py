"""Manifest-based, resumable WikiData entity collectors.

Collectors discover *which* entities satisfy a relationship (e.g.
``wdt:P27 wd:Q213``) and persist the resulting entity list as a CSV.
"""

from __future__ import annotations

import logging
import os

from SPARQLWrapper import SPARQLWrapper

from people_gator_common.rate_limiter import RateLimiter
from people_gator_common.transport.sparql import execute_sparql_query, parse_sparql_results
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


# ---------------------------------------------------------------------------
# Task & Manifest models
# ---------------------------------------------------------------------------


class CollectorTask(TaskModel):
    total: int
    offset: int
    limit: int

    @property
    def frame_name(self) -> str:
        num_leading_zeros = len(str(self.total))
        offset_str = str(self.offset).zfill(num_leading_zeros)
        limit_str = str(self.limit).zfill(num_leading_zeros)
        return f"{self.name}__total={self.total}__offset={offset_str}__limit={limit_str}.csv"


class CollectorManifest(ManifestModel):
    tasks: list[CollectorTask]


# ---------------------------------------------------------------------------
# Convenience re-exports of generic helpers with concrete types
# ---------------------------------------------------------------------------


def get_manifest_tasks(directory: str) -> dict[str, CollectorTask]:
    """Read the collector manifest from *directory*/manifest.json."""
    return _get_manifest_tasks(directory, CollectorManifest, lambda t: t.frame_name)


def get_pending_tasks(directory: str) -> list[CollectorTask]:
    """Return collector tasks whose frames are not yet completed."""
    return _get_pending_tasks(directory, CollectorManifest, lambda t: t.frame_name)

def build_generic_collector_count_query(
    relationship: str,
    to_what: str,
) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT (COUNT(*) AS ?count)
WHERE {{
    ?entity {relationship} {to_what} .
}}
"""

def build_generic_collector_query(
    relationship: str,
    to_what: str,
    limit: int,
    offset: int,
) -> str:
    return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?entity WHERE {{
    ?entity {relationship} {to_what} .
}}
LIMIT {limit}
OFFSET {offset}
""".strip()

def get_task_name(relationship: str, to_what: str) -> str:
    sanitized_relationship = relationship.replace(" ", "_").replace("/", "_").replace("*", "(star)")
    sanitized_to_what = to_what.replace(" ", "_").replace("/", "_").replace("*", "(star)")
    return f"collector__{sanitized_relationship}__{sanitized_to_what}"

def init_generic_collector_query(
    output_file: str,
    tmp_dir: str,
    relationship: str,
    to_what: str,
    batch_size: int,
    force: bool,
    rate_limiter: RateLimiter,
    client: SPARQLWrapper,
) -> None:
    task_name = get_task_name(relationship, to_what)
    tmp_dir_path = os.path.join(tmp_dir, task_name)

    count_query = build_generic_collector_count_query(relationship, to_what)
    result = parse_sparql_results(execute_sparql_query(client=client, query=count_query, rate_limiter=rate_limiter))
    count = int(result["count"][0])

    manifest = CollectorManifest(
        output_file=output_file,
        tasks=[
            CollectorTask(
                name=get_task_name(relationship, to_what),
                command=build_generic_collector_query(relationship, to_what, batch_size, offset),
                total=count,
                offset=offset,
                limit=batch_size,
            )
            for offset in range(0, count, batch_size)
        ],
    )
    init_manifest(tmp_dir_path, manifest, force=force)


def execute_collector_tasks(
    tmp_dir: str,
    relationship: str,
    to_what: str,
    *,
    client: SPARQLWrapper,
    rate_limiter: RateLimiter,
) -> None:
    tmp_dir_path = os.path.join(tmp_dir, get_task_name(relationship, to_what))
    execute_tasks(
        tmp_dir=tmp_dir_path,
        manifest_type=CollectorManifest,
        task_key=lambda t: t.frame_name,
        client=client,
        rate_limiter=rate_limiter,
    )


def collect_collector_results(
    output_file: str,
    tmp_dir: str,
    relationship: str,
    to_what: str,
) -> None:
    task_name = get_task_name(relationship, to_what)
    tmp_dir_path = os.path.join(tmp_dir, task_name)
    collect_results(
        tmp_dir=tmp_dir_path,
        output_file=output_file,
        manifest_type=CollectorManifest,
        task_key=lambda t: t.frame_name,
        sort_by="entity",
    )


def cleanup_collector(
    tmp_dir: str,
    relationship: str,
    to_what: str,
) -> None:
    task_name = get_task_name(relationship, to_what)
    tmp_dir_path = os.path.join(tmp_dir, task_name)
    cleanup(tmp_dir_path)


def run_generic_collector(
    output_file: str,
    tmp_dir: str,
    relationship: str,
    to_what: str,
    batch_size: int,
    force: bool,
    rate_limiter: RateLimiter,
    client: SPARQLWrapper,
) -> None:
    task_tmp_dir = os.path.join(tmp_dir, get_task_name(relationship, to_what))

    def _init() -> None:
        init_generic_collector_query(
            output_file=output_file,
            tmp_dir=tmp_dir,
            relationship=relationship,
            to_what=to_what,
            batch_size=batch_size,
            force=force,
            rate_limiter=rate_limiter,
            client=client,
        )

    run_pipeline(
        output_file=output_file,
        tmp_dir=tmp_dir,
        manifest_type=CollectorManifest,
        task_key=lambda t: t.frame_name,
        init_fn=_init,
        task_tmp_dir=task_tmp_dir,
        sort_by="entity",
        force=force,
        client=client,
        rate_limiter=rate_limiter,
    )
