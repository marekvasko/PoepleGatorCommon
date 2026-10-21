"""Public NER entity-linking API.

Implementation lives in :mod:`people_gator_common.pipelines.ner_linking`.
"""

from .ner_linking.api import (
    _link_application_complete as _link_application_complete,
    apply_linked_entity_result,
    link_source_cluster,
)
from .ner_linking.llm import llm_decide_entity_link, llm_decide_entity_link_async
from .ner_linking.models import (
    DEFAULT_API_BASE_URL,
    DEFAULT_DOCUMENT_INPUT_PREFIX,
    DEFAULT_DOCUMENT_OUTPUT_PREFIX,
    DEFAULT_INPUT_PREFIX,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    EntityCandidate,
    EntityLinkDecision,
    EntityLinkingApiError,
    EntityLinkingConfig,
    EntityLinkingRollbackSummary,
    EntitySnapshot,
    LinkableEntity,
    LinkedEntityResult,
    MutationAudit,
    NerEntityLinkingRunSummary,
    PageNerEntityLinks,
    ProposedEntityLink,
)
from .ner_linking.rollback import (
    _latest_rollbackable_output as _latest_rollbackable_output,
    rollback_latest_entity_linking_run,
)
from .ner_linking.pipeline import run_ner_entity_linking_pipeline
from .ner_linking.run import iter_page_source_clusters
from .ner_linking.transport import (
    authenticate_entity_linking_api,
    lookup_page_source,
    search_entity_candidates,
)
from .ner_linking.source import build_search_queries, source_entity_from_cluster

__all__ = [
    "DEFAULT_API_BASE_URL",
    "DEFAULT_DOCUMENT_INPUT_PREFIX",
    "DEFAULT_DOCUMENT_OUTPUT_PREFIX",
    "DEFAULT_INPUT_PREFIX",
    "DEFAULT_OUTPUT_PREFIX",
    "DEFAULT_PROGRESS_INTERVAL_SECONDS",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "EntityCandidate",
    "EntityLinkDecision",
    "EntityLinkingApiError",
    "EntityLinkingConfig",
    "EntityLinkingRollbackSummary",
    "EntitySnapshot",
    "LinkableEntity",
    "LinkedEntityResult",
    "MutationAudit",
    "NerEntityLinkingRunSummary",
    "PageNerEntityLinks",
    "ProposedEntityLink",
    "apply_linked_entity_result",
    "authenticate_entity_linking_api",
    "build_search_queries",
    "iter_page_source_clusters",
    "link_source_cluster",
    "llm_decide_entity_link",
    "llm_decide_entity_link_async",
    "lookup_page_source",
    "rollback_latest_entity_linking_run",
    "run_ner_entity_linking_pipeline",
    "search_entity_candidates",
    "source_entity_from_cluster",
]
