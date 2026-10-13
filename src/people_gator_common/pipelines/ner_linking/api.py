"""Apply entity decisions and record their mutations."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
import httpx
from pydantic import ValidationError
from people_gator_common.client.pydantic_gen import (
    EntityCreateRequest,
    EntityPatchRequest,
    TextMentionCreateRequest,
)
from people_gator_common.pipelines.llm_ner import ReasoningEffort
from people_gator_common.pipelines.ner_clusters import (
    DEFAULT_MODEL,
    DisambiguatedNamedEntity,
)
from .llm import llm_decide_entity_link_async
from .models import (
    EntityLinkDecision,
    EntitySnapshot,
    LinkedEntityResult,
    MutationAudit,
    _utc_now,
)
from .source import source_entity_from_cluster
from .transport import _request_json, search_entity_candidates

logger = logging.getLogger(__name__)


def _entity_snapshot(payload: dict[str, Any]) -> EntitySnapshot:
    try:
        return EntitySnapshot.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid entity API response: {exc}") from exc


def _entity_write_payload(decision: EntityLinkDecision) -> dict[str, Any]:
    return {
        "name": decision.name,
        "aliases": decision.aliases,
        "description": decision.long_description or None,
        "short_description": decision.short_description or None,
    }


def _mutable_entity_state(entity: EntitySnapshot) -> dict[str, Any]:
    return {
        "name": entity.name,
        "aliases": entity.aliases or [],
        "description": entity.description,
        "short_description": entity.short_description,
        "wikidata_id": entity.wikidata_id,
        "nlcr_authority_id": entity.nlcr_authority_id,
    }


def _mention_payloads(
    cluster: DisambiguatedNamedEntity,
    *,
    entity_id: str,
    source_id: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity in cluster.entities or []:
        for mention in entity.mentions:
            name = (mention.exact_name or mention.name).strip()
            context = mention.exact_context or mention.context
            key = (name, context)
            if not name or key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "name": name,
                    "context": context or None,
                    "entity_id": entity_id,
                    "entity_link_state": "automatic",
                    "source_id": source_id,
                }
            )
    return output


def _link_application_complete(link: LinkedEntityResult, *, source_id: str) -> bool:
    """Return whether a journaled link has its entity and all mentions applied."""
    if link.selected_entity_id is None:
        return False
    if link.decision.action == "merge":
        entity_complete = link.selected_entity_after is not None
    else:
        entity_complete = any(
            mutation.operation == "create_entity" for mutation in link.mutations
        )
    if not entity_complete:
        return False
    recorded_mentions = {
        (
            str(mutation.after.get("name", "")),
            str(mutation.after.get("context") or ""),
            str(mutation.after.get("entity_id", "")),
            str(mutation.after.get("source_id", "")),
        )
        for mutation in link.mutations
        if mutation.operation == "create_mention"
    }
    expected_mentions = {
        (
            str(payload["name"]),
            str(payload["context"] or ""),
            str(payload["entity_id"]),
            str(payload["source_id"]),
        )
        for payload in _mention_payloads(
            link.source_cluster,
            entity_id=link.selected_entity_id,
            source_id=source_id,
        )
    }
    return expected_mentions <= recorded_mentions


async def apply_linked_entity_result(
    link: LinkedEntityResult,
    *,
    source_id: str,
    api_client: httpx.AsyncClient,
    headers: dict[str, str],
    checkpoint: Callable[[], Awaitable[None]],
) -> None:
    """Apply one decision, checkpointing after every durable API mutation."""
    entity_payload = _entity_write_payload(link.decision)
    if link.decision.action == "merge":
        if link.selected_entity_id is None:
            raise RuntimeError("merge decision has no selected entity ID")
        entity_already_applied = link.selected_entity_after is not None or any(
            mutation.operation == "update_entity" for mutation in link.mutations
        )
        if not entity_already_applied:
            path = f"/api/v1/entities/{link.selected_entity_id}"
            before_payload = await _request_json(
                api_client, "GET", path, headers=headers
            )
            before = _entity_snapshot(before_payload)
            link.selected_entity_before = before
            await checkpoint()
            before_state = _mutable_entity_state(before)
            changed_payload = {
                key: value
                for key, value in entity_payload.items()
                if before_state[key] != value
            }
            requested_aliases = changed_payload.get("aliases")
            if isinstance(requested_aliases, list) and set(
                before.aliases or []
            ).intersection(requested_aliases):
                # The current PostgreSQL repository replaces alias rows by
                # inserting before deleting. Re-sending an existing alias
                # violates its unique (entity_id, name) constraint.
                changed_payload.pop("aliases")
                if "aliases" not in link.unapplied_fields:
                    link.unapplied_fields.append("aliases")
                logger.warning(
                    "Deferring alias update for entity %s because the requested "
                    "aliases overlap existing aliases",
                    before.entity_id,
                )
            if not changed_payload:
                link.selected_entity_after = before
                await checkpoint()
            else:
                after_payload = await _request_json(
                    api_client,
                    "PATCH",
                    path,
                    headers=headers,
                    json_body=EntityPatchRequest.model_validate(
                        changed_payload
                    ).model_dump(mode="json", exclude_unset=True),
                )
                after = _entity_snapshot(after_payload)
                link.selected_entity_after = after
                link.mutations.append(
                    MutationAudit(
                        operation="update_entity",
                        resource_id=after.entity_id,
                        endpoint=path,
                        before=before.model_dump(),
                        after=after.model_dump(),
                        applied_at=_utc_now(),
                    )
                )
                await checkpoint()
    else:
        create_mutation = next(
            (
                mutation
                for mutation in link.mutations
                if mutation.operation == "create_entity"
            ),
            None,
        )
        if create_mutation is not None:
            link.selected_entity_id = create_mutation.resource_id
            if link.selected_entity_after is None:
                link.selected_entity_after = _snapshot_from_audit(
                    create_mutation.after, "after"
                )
        else:
            create_payload = {
                "entity_type": "person",
                **entity_payload,
                "wikidata_id": None,
                "nlcr_authority_id": None,
            }
            after_payload = await _request_json(
                api_client,
                "POST",
                "/api/v1/entities",
                headers=headers,
                json_body=EntityCreateRequest.model_validate(create_payload).model_dump(
                    mode="json"
                ),
            )
            after = _entity_snapshot(after_payload)
            link.selected_entity_id = after.entity_id
            link.selected_entity_after = after
            link.mutations.append(
                MutationAudit(
                    operation="create_entity",
                    resource_id=after.entity_id,
                    endpoint="/api/v1/entities",
                    before=None,
                    after=after.model_dump(),
                    applied_at=_utc_now(),
                )
            )
            await checkpoint()

    assert link.selected_entity_id is not None
    uploaded_mention_keys = {
        (
            str(mutation.after.get("name", "")),
            str(mutation.after.get("context") or ""),
            str(mutation.after.get("entity_id", "")),
            str(mutation.after.get("source_id", "")),
        )
        for mutation in link.mutations
        if mutation.operation == "create_mention"
    }
    for mention_payload in _mention_payloads(
        link.source_cluster,
        entity_id=link.selected_entity_id,
        source_id=source_id,
    ):
        mention_key = (
            mention_payload["name"],
            mention_payload["context"] or "",
            mention_payload["entity_id"],
            mention_payload["source_id"],
        )
        if mention_key in uploaded_mention_keys:
            continue
        mention_response = await _request_json(
            api_client,
            "POST",
            "/api/v1/text-mentions",
            headers=headers,
            json_body=TextMentionCreateRequest.model_validate(
                mention_payload
            ).model_dump(mode="json"),
        )
        mention_id = mention_response.get("mention_id")
        if not isinstance(mention_id, str) or not mention_id:
            raise RuntimeError("text-mentions response does not contain mention_id")
        link.mutations.append(
            MutationAudit(
                operation="create_mention",
                resource_id=mention_id,
                endpoint="/api/v1/text-mentions",
                before=None,
                after=mention_response,
                applied_at=_utc_now(),
            )
        )
        await checkpoint()


async def link_source_cluster(
    source_cluster: DisambiguatedNamedEntity,
    *,
    api_client: httpx.AsyncClient,
    headers: dict[str, str],
    llm_client: Any,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
) -> LinkedEntityResult:
    """Run all deterministic and model-backed steps for one source cluster."""
    source = source_entity_from_cluster(source_cluster)
    candidates = await search_entity_candidates(api_client, headers, source)
    decision, llm = await llm_decide_entity_link_async(
        source,
        candidates,
        client=llm_client,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    selected_entity_id = None
    if decision.action == "merge" and decision.candidate_index is not None:
        selected_entity_id = candidates[decision.candidate_index - 1].entity_id
    return LinkedEntityResult(
        source_cluster=source_cluster,
        source=source,
        candidates=candidates,
        decision=decision,
        selected_entity_id=selected_entity_id,
        llm=llm,
    )


def _snapshot_from_audit(payload: dict[str, Any] | None, label: str) -> EntitySnapshot:
    if payload is None:
        raise RuntimeError(f"Mutation audit has no {label} entity state")
    try:
        return EntitySnapshot.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"Mutation audit has invalid {label} entity state: {exc}"
        ) from exc
