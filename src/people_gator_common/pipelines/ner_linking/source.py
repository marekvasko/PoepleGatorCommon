"""Deterministic source normalization and decision mapping."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError
from people_gator_common.pipelines.ner_clusters import DisambiguatedNamedEntity
from .models import (
    EntityCandidate,
    EntityLinkDecision,
    LinkableEntity,
    ProposedEntityLink,
)


def _deduplicate_strings(values: Sequence[str]) -> list[str]:
    """Strip and case-insensitively deduplicate strings in stable order."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def source_entity_from_cluster(
    cluster: DisambiguatedNamedEntity,
) -> LinkableEntity:
    """Expose one disambiguated cluster as a normalized linking input."""
    observed_names: list[str] = [cluster.name]
    for entity in cluster.entities or []:
        observed_names.append(entity.name)
        observed_names.extend(mention.name for mention in entity.mentions)

    names = _deduplicate_strings(observed_names)
    canonical_name = cluster.name.strip() or (names[0] if names else "")
    aliases = [name for name in names if name.casefold() != canonical_name.casefold()]
    return LinkableEntity(
        name=canonical_name,
        aliases=aliases,
        short_description=cluster.short_description.strip(),
        long_description=cluster.long_description.strip(),
    )


def build_search_queries(source: LinkableEntity) -> list[str]:
    """Build the repeated ``q`` values accepted by ``search-entities``."""
    return _deduplicate_strings(
        [
            source.name,
            *source.aliases,
            source.short_description,
            source.long_description,
        ]
    )


def _format_resolution_input(
    source: LinkableEntity,
    candidates: Sequence[EntityCandidate],
) -> str:
    payload = {
        "source": source.model_dump(),
        "candidates": [
            {"index": index, **candidate.model_dump()}
            for index, candidate in enumerate(candidates, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _create_decision(
    source: LinkableEntity,
    proposed: ProposedEntityLink,
) -> EntityLinkDecision:
    return EntityLinkDecision(
        action="create",
        candidate_index=None,
        name=source.name,
        aliases=source.aliases,
        short_description=proposed.short_description.strip(),
        long_description=proposed.long_description.strip(),
    )


def _combined_aliases(
    source: LinkableEntity,
    candidate: EntityCandidate,
    canonical_name: str,
) -> list[str]:
    aliases = _deduplicate_strings(
        [
            *candidate.aliases,
            candidate.name,
            source.name,
            *source.aliases,
        ]
    )
    return [alias for alias in aliases if alias.casefold() != canonical_name.casefold()]


def _merge_decision(
    proposed: ProposedEntityLink,
    source: LinkableEntity,
    candidates: Sequence[EntityCandidate],
) -> EntityLinkDecision | None:
    index = proposed.candidate_index
    if index is None or not 1 <= index <= len(candidates):
        return None
    candidate = candidates[index - 1]
    canonical_name = candidate.name.strip() or source.name
    return EntityLinkDecision(
        action="merge",
        candidate_index=index,
        name=canonical_name,
        aliases=_combined_aliases(source, candidate, canonical_name),
        short_description=proposed.short_description.strip(),
        long_description=proposed.long_description.strip(),
    )


def _decision_from_response(
    response: dict[str, Any],
    source: LinkableEntity,
    candidates: Sequence[EntityCandidate],
) -> EntityLinkDecision | None:
    nested_decision = response.get("decision")
    payload = nested_decision if isinstance(nested_decision, dict) else response
    try:
        proposed = ProposedEntityLink.model_validate(payload)
    except ValidationError:
        return None
    if proposed.action == "create":
        evidence_has_description = bool(
            source.short_description or source.long_description
        )
        if evidence_has_description and not (
            proposed.short_description.strip() and proposed.long_description.strip()
        ):
            return None
        return _create_decision(source, proposed)

    index = proposed.candidate_index
    if index is None or not 1 <= index <= len(candidates):
        return None
    candidate = candidates[index - 1]
    evidence_has_description = bool(
        source.short_description
        or source.long_description
        or candidate.short_description
        or candidate.long_description
    )
    if evidence_has_description and not (
        proposed.short_description.strip() and proposed.long_description.strip()
    ):
        return None
    return _merge_decision(proposed, source, candidates)
