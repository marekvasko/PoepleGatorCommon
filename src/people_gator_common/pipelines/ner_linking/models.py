"""Data contracts for NER entity linking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
from people_gator_common.pipelines.llm_ner import LLMBase
from people_gator_common.pipelines.ner_clusters import DisambiguatedNamedEntity

DEFAULT_API_BASE_URL = "https://dev.people-gator.capturemyhand.com"
DEFAULT_INPUT_PREFIX = "page-named-entities-clusters"
DEFAULT_OUTPUT_PREFIX = "page-named-entities-links"
MODIFICATION_LOG_NAME = "modifications.jsonl"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class EntityLinkingConfig:
    """Connection settings for the People Gator API."""

    base_url: str = DEFAULT_API_BASE_URL
    bearer_token: str | None = None
    username: str | None = None
    password: str | None = None
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class NerEntityLinkingRunSummary:
    processed_pages: int
    skipped_pages: int
    source_entities: int
    merge_decisions: int
    create_decisions: int
    entity_mutations: int = 0
    uploaded_mentions: int = 0


@dataclass(frozen=True)
class EntityLinkingRollbackSummary:
    audit_path: str
    rolled_back_mutations: int


class EntityLinkingApiError(RuntimeError):
    """People Gator API failure with a machine-readable HTTP status."""

    def __init__(self, method: str, path: str, status_code: int, body: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.body = body
        super().__init__(f"{method} {path} failed with status {status_code}: {body}")


class LinkableEntity(BaseModel):
    """Normalized fields used to search for and describe one person."""

    name: str
    aliases: list[str]
    short_description: str
    long_description: str


class EntityCandidate(LinkableEntity):
    """Full database entity detail corresponding to one search result."""

    entity_id: str
    entity_type: str | None = None
    matching_score: float


class EntitySnapshot(BaseModel):
    """API entity state retained for audit and conflict-safe rollback."""

    entity_id: str
    entity_type: str
    name: str
    description: str | None = None
    short_description: str | None = None
    aliases: list[str] | None = None
    wikidata_id: str | None = None
    nlcr_authority_id: str | None = None
    user_id: str | None = None


class MutationAudit(BaseModel):
    """One durable API mutation and the information needed to reverse it."""

    operation: Literal["create_entity", "update_entity", "create_mention"]
    resource_id: str
    endpoint: str
    before: dict[str, Any] | None = None
    after: dict[str, Any]
    applied_at: str
    rollback_status: Literal["pending", "rolled_back", "failed"] = "pending"
    rollback_error: str | None = None
    rolled_back_at: str | None = None


class ProposedEntityLink(BaseModel):
    """Strict schema for the small decision surface exposed to the LLM."""

    action: Literal["merge", "create"]
    candidate_index: int | None = None
    short_description: str = ""
    long_description: str = ""

    @model_validator(mode="after")
    def _validate_candidate_index(self) -> ProposedEntityLink:
        if self.action == "merge" and self.candidate_index is None:
            raise ValueError("a merge decision requires candidate_index")
        if self.action == "create" and self.candidate_index is not None:
            raise ValueError("a create decision must not select a candidate")
        return self


class EntityLinkDecision(LinkableEntity):
    """Validated, deterministic proposed operation for one source cluster."""

    action: Literal["merge", "create"]
    candidate_index: int | None = None
    description_language: Literal["cs"] = "cs"


class LinkedEntityResult(BaseModel):
    """Auditable proposed link for an input cluster."""

    source_cluster: DisambiguatedNamedEntity
    source: LinkableEntity
    candidates: list[EntityCandidate]
    decision: EntityLinkDecision
    selected_entity_id: str | None = None
    selected_entity_before: EntitySnapshot | None = None
    selected_entity_after: EntitySnapshot | None = None
    unapplied_fields: list[str] = Field(default_factory=list)
    llm: LLMBase | None = None
    mutations: list[MutationAudit] = Field(default_factory=list)


class PageNerEntityLinks(BaseModel):
    """All saved entity-linking decisions for one People Gator page."""

    library: str
    document_id: str
    page_id: str
    links: list[LinkedEntityResult]
    mode: Literal["decision-only", "apply"] = "decision-only"
    source_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    rolled_back_at: str | None = None


class _ApiEntityDetail(BaseModel):
    """Subset of ``EntityResponse`` used by this pipeline."""

    entity_id: str
    entity_type: str | None = None
    name: str | None = None
    aliases: list[str] | None = None
    description: str | None = None
    short_description: str | None = None
    wikidata_id: str | None = None
    nlcr_authority_id: str | None = None
    user_id: str | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
