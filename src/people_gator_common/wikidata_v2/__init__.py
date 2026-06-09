"""Wikidata v2 collection pipeline — manifest-based, resumable."""

from __future__ import annotations

from people_gator_common.wikidata_v2.common import (
    ManifestModel,
    TaskModel,
    cleanup as cleanup_pipeline,
    collect_results,
    execute_tasks,
    get_completed_frames,
    get_manifest_tasks as generic_get_manifest_tasks,
    get_pending_tasks as generic_get_pending_tasks,
    init_manifest,
    run_pipeline,
)

from people_gator_common.wikidata_v2.collectors import (
    CollectorManifest,
    CollectorTask,
    build_generic_collector_count_query,
    build_generic_collector_query,
    cleanup_collector,
    collect_collector_results,
    execute_collector_tasks,
    get_manifest_tasks as get_collector_manifest_tasks,
    get_pending_tasks as get_collector_pending_tasks,
    get_task_name as get_collector_task_name,
    init_generic_collector_query,
    run_generic_collector,
)

from people_gator_common.wikidata_v2.enrichers import (
    EnricherManifest,
    EnricherTask,
    build_alt_label_enricher_query,
    build_description_enricher_query,
    build_label_enricher_query,
    build_property_enricher_query,
    cleanup_enricher,
    collect_enricher_tasks,
    execute_enricher_tasks,
    extract_entity_id,
    get_manifest_tasks as get_enricher_manifest_tasks,
    get_pending_tasks as get_enricher_pending_tasks,
    init_enricher_manifest,
    run_generic_enricher,
)

__all__ = [
    # Common infrastructure
    "ManifestModel",
    "TaskModel",
    "cleanup_pipeline",
    "collect_results",
    "execute_tasks",
    "generic_get_manifest_tasks",
    "generic_get_pending_tasks",
    "get_completed_frames",
    "init_manifest",
    "run_pipeline",
    # Collectors
    "CollectorManifest",
    "CollectorTask",
    "build_generic_collector_count_query",
    "build_generic_collector_query",
    "cleanup_collector",
    "collect_collector_results",
    "execute_collector_tasks",
    "get_collector_manifest_tasks",
    "get_collector_pending_tasks",
    "get_collector_task_name",
    "init_generic_collector_query",
    "run_generic_collector",
    # Enrichers
    "EnricherManifest",
    "EnricherTask",
    "build_alt_label_enricher_query",
    "build_description_enricher_query",
    "build_label_enricher_query",
    "build_property_enricher_query",
    "cleanup_enricher",
    "collect_enricher_tasks",
    "execute_enricher_tasks",
    "extract_entity_id",
    "get_enricher_manifest_tasks",
    "get_enricher_pending_tasks",
    "init_enricher_manifest",
    "run_generic_enricher",
]
