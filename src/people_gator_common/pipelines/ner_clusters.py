"""Reusable entity clustering and LLM disambiguation over document NER output.

This module is deliberately free of filesystem, library, and page concerns.
It consumes the generic :class:`DocumentNerBase` hierarchy produced by
:mod:`people_gator_common.pipelines.llm_ner` (or anything structurally
compatible) and provides two composable operations:

1. :func:`cluster_document_entities` — group entities that share names or
   name variants into :class:`NamedEntityCluster` records.
2. :func:`disambiguate_ner` — verify or split one cluster with an LLM and
   return the resulting list of clusters.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import os
import posixpath
import re
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError
import rapidfuzz
from sklearn.cluster import DBSCAN

from people_gator_common.filesystem import (
    glob as storage_glob,
    read_text as storage_read_text,
    write_text as storage_write_text,
)
from people_gator_common.pipelines.llm_ner import (
    ChunkNerBase,
    DocumentNerBase,
    LLMBase,
    NamedEntity,
    PageInput,
    PageNer,
    ReasoningEffort,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_CLUSTER_THRESHOLD = 92.0
DEFAULT_INPUT_PREFIX = "page-named-entities"
DEFAULT_OUTPUT_PREFIX = "page-named-entities-clusters"

DEFAULT_NAME_WEIGHT = 0.7
DEFAULT_MENTION_WEIGHT = 0.3

T = TypeVar("T", bound=ChunkNerBase)


@dataclass(frozen=True)
class NerClustersRunSummary:
    processed_pages: int
    skipped_pages: int
    clusters: int


DISAMBIGUATION_PROMPT = """\
You verify clusters of named entities extracted from a single document.

You receive one tentative cluster: entities were grouped because their names
look similar, but the grouping MAY BE WRONG — similar-looking names can belong
to different real-world referents.

Each entity in the cluster is numbered. Decide which entities truly refer to
the SAME real-world referent and output the corrected clusters. The result may
be one cluster (the proposal was right), several clusters, or all singletons.

# Decision rules
Treat two entities as the same referent when:
- They are spelling, OCR, or transcription variants of the same name.
- They are inflected forms of the same name (e.g. grammatical declension).
- One is an abbreviation, initialism, or shortened form of the other.
- They differ only by titles, honorifics, or a missing middle name.
- The ordering of the name components is swapped.

Split entities into separate clusters ONLY when there is clear evidence they
are different referents (incompatible facts, clearly different names, or
clearly different points in time). When the evidence is ambiguous or merely
missing, KEEP the entities together: splitting the same referent is the more
costly error.

# Output requirements
- Every input entity index must appear in exactly one output cluster. Do not
  invent, drop, or duplicate indices.
- `name`: the most complete canonical name, without titles.
- `aliases`: other name variants observed in the entities, excluding `name`.
- `short_description`: basic reduced information useful for entity
  identification (e.g. for people: occupation, gender, citizenship).
- `long_description`: a few sentences in the language of the source text,
  merging the non-contradictory information from all member entities.
- Output ONLY a single JSON object. No explanations, no text before or after.

# Example output
```json
{{
  "clusters": [
    {{
      "name": "<canonical_name>",
      "aliases": ["<variant>", "<variant>"],
      "short_description": "<basic_identification_info>",
      "long_description": "<merged_information>",
      "entity_indices": [1, 2, 5]
    }},
    {{
      "name": "<canonical_name>",
      "aliases": [],
      "short_description": "<basic_identification_info>",
      "long_description": "<restated_information>",
      "entity_indices": [3]
    }}
  ]
}}
```

# Tentative cluster
```json
{entities}
```
"""

class NamedEntityShort(BaseModel):
    """A single named entity mention, derived from a full NamedEntity."""

    @classmethod
    def from_named_entity(cls, index: int, entity: NamedEntity) -> "NamedEntityShort":
        return cls(
            index=index,
            name=entity.name,
            aliases=[
                mention.name
                for mention in entity.mentions
                if mention.name != entity.name
            ],
            information=entity.info,
        )
    index: int
    name: str
    aliases: list[str]
    information: str

class DisambiguatedNamedEntity(BaseModel):
    """One output cluster from the disambiguation LLM."""
    name: str
    aliases: list[str] = []
    short_description: str = ""
    long_description: str = ""
    entity_indices: list[int]
    entities: list[NamedEntity] | None = None
    
    @classmethod
    def from_named_entity(
        cls,
        named_entity: NamedEntity,
    ) -> "DisambiguatedNamedEntity":
        return cls(
            name=named_entity.name,
            aliases=[
                mention.name
                for mention in named_entity.mentions
                if mention.name != named_entity.name
            ],
            short_description="",
            long_description="",
            entity_indices=[],
            entities=[named_entity],
        )

class DisambiguationOutput(LLMBase):
    """Pure disambiguation result for one cluster, with its LLM record."""
    clusters: list[DisambiguatedNamedEntity]

class DisambiguationResult(BaseModel):
    """Validated schema of the disambiguation LLM response."""
    clusters: list[DisambiguatedNamedEntity]

class PageNerClusters(BaseModel):
    """Clustered and disambiguated entities for one PeopleGator page."""
    library: str
    document_id: str
    page_id: str
    clusters: list[DisambiguationOutput]


def parse_json_object(response: str) -> dict[str, Any]:
    """Parse a JSON object from a plain or Markdown-fenced model response."""
    candidates: list[str] = [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            re.IGNORECASE | re.DOTALL,
        )
    ]
    candidates.append(response.strip())

    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(response[first_brace : last_brace + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return {}


def _format_entities(entities: list[NamedEntityShort]) -> str:
    """Render numbered entities for the disambiguation prompt."""
    return json.dumps(
        {
            "Entities": [
                {
                    "EntityId": entity.index,
                    "Name": entity.name,
                    "Aliases": entity.aliases,
                    "Description": entity.information,
                }
                for entity in entities
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _clusters_from_response(
    response: dict[str, Any],
    named_entities: list[NamedEntity],
    llm: LLMBase,
) -> tuple[list[DisambiguatedNamedEntity], LLMBase] | None:
    """Map a validated LLM response back to entity-based clusters."""
    try:
        result = DisambiguationResult.model_validate(response)
    except ValidationError:
        return None

    entity_count = len(named_entities)
    used_indices: set[int] = set()
    output: list[DisambiguatedNamedEntity] = []
    for disambiguated in result.clusters:
        valid_indices = [
            index
            for index in disambiguated.entity_indices
            if 1 <= index <= entity_count and index not in used_indices
        ]
        if not valid_indices:
            continue
        used_indices.update(valid_indices)
        output.append(
            disambiguated.model_copy(
                update={
                    "entity_indices": valid_indices,
                    "entities": [named_entities[index - 1] for index in valid_indices],
                }
            )
        )

    # Any entity the LLM dropped stays in a residual cluster.
    missing_indices = [
        index
        for index in range(1, entity_count + 1)
        if index not in used_indices
    ]
    if missing_indices:
        logger.warning(
            "Disambiguation dropped %d of %d entities; keeping them in a "
            "residual cluster",
            len(missing_indices),
            entity_count,
        )
        residual_entities = [named_entities[index - 1] for index in missing_indices]
        output.append(
            DisambiguatedNamedEntity(
                name=residual_entities[0].name if residual_entities else "",
                aliases=[entity.name for entity in residual_entities[1:]],
                short_description="",
                long_description="",
                entity_indices=missing_indices,
                entities=residual_entities,
            )
        )
    if not output:
        return None
    return output, llm

def _fit_to_size[T](
    item: list[T] | T,
    size: int,
) -> list[T]:
    """Broadcast a scalar or validate a list to exactly ``size`` elements."""
    if isinstance(item, list):
        if len(item) != size:
            raise ValueError(
                f"List must have exactly {size} elements, but has {len(item)}."
            )
        return item
    return [item] * size

def fuzzy_score_form_aliases(
    candidates_1: list[str],
    candidates_2: list[str],
    candidates_1_weight: list[float] | float = 1.0,
    candidates_2_weight: list[float] | float = 1.0,
    aggregation: Literal["max", "mean", "min"] = "max",
) -> float:
    """Return the best fuzzy similarity score between two lists of names.

    Each pairwise score is weighted by the sum of both candidate weights,
    normalized by the maximum possible weight sum, so an exact match between
    the highest-weighted candidates still scores 100.
    """
    _candidates_1_weight = _fit_to_size(candidates_1_weight, len(candidates_1))
    _candidates_2_weight = _fit_to_size(candidates_2_weight, len(candidates_2))
    max_weight_sum = max(_candidates_1_weight) + max(_candidates_2_weight)
    if max_weight_sum <= 0:
        return 0.0
    all_scores: list[float] = []
    for i, name_1 in enumerate(candidates_1):
        for j, name_2 in enumerate(candidates_2):
            score = rapidfuzz.fuzz.partial_token_sort_ratio(name_1, name_2)
            weight = (
                _candidates_1_weight[i] + _candidates_2_weight[j]
            ) / max_weight_sum
            all_scores.append(score * weight)
    if not all_scores:
        return 0.0
    if aggregation == "max":
        return max(all_scores)
    elif aggregation == "mean":
        return sum(all_scores) / len(all_scores)
    elif aggregation == "min":
        return min(all_scores)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation}")

def score_entities(
    entities: list[NamedEntity], 
    mention_weight: float, 
    name_weight: float,
    *,
    aggregation: Literal["max", "mean", "min"] = "max"
) -> list[list[float]]:
    entity_scores: list[list[float]] = []
    normalized_mention_weight = mention_weight / (mention_weight + name_weight)
    normalized_name_weight = name_weight / (mention_weight + name_weight)
    for entity_1 in entities:
        scores: list[float] = []
        for entity_2 in entities:
            score = fuzzy_score_form_aliases(
                candidates_1=[entity_1.name, *[m.name for m in entity_1.mentions]],
                candidates_2=[entity_2.name, *[m.name for m in entity_2.mentions]],
                candidates_1_weight=(
                    [
                        normalized_name_weight,
                        *[normalized_mention_weight] * len(entity_1.mentions),
                    ]
                ),
                candidates_2_weight=(
                    [
                        normalized_name_weight,
                        *[normalized_mention_weight] * len(entity_2.mentions),
                    ]
                ),
                aggregation=aggregation,
            )
            scores.append(score)
        entity_scores.append(scores)
    return entity_scores

def cluster_entities(
    entities: list[NamedEntity],
    *,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    per_entity_aggregation: Literal["max", "mean", "min"] = "max",
    mention_weight: float = DEFAULT_MENTION_WEIGHT,
    name_weight: float = DEFAULT_NAME_WEIGHT,
) -> list[list[NamedEntity]]:
    """Group entities into tentative clusters using DBSCAN over fuzzy scores.

    Pairwise similarity is computed by :func:`score_entities`; entities whose
    weighted name/mention similarity is at least ``threshold`` are clustered
    together (DBSCAN with ``eps = 100 - threshold`` on the precomputed
    distance matrix). Each resulting group becomes one
    :class:`DisambiguationInput` whose ``entities`` are numbered
    :class:`NamedEntityShort` records ready for the disambiguation prompt.
    """
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if not entities:
        return []

    entity_scores = score_entities(
        entities=entities,
        mention_weight=mention_weight,
        name_weight=name_weight,
        aggregation=per_entity_aggregation,
    )

    distance_matrix = np.array(
        [[100.0 - score for score in row] for row in entity_scores]
    )
    clustering = DBSCAN(eps=100.0 - threshold, min_samples=1, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    grouped: dict[int, list[NamedEntity]] = {}
    for entity_index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(entities[entity_index])
    clusters = list(grouped.values())
    return clusters


def llm_disambiguate_entities(
    named_entities: list[NamedEntity],
    *,
    client: Any,  # openrouter client: third-party package without type stubs
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
) -> tuple[list[DisambiguatedNamedEntity], LLMBase | None]:
    """Verify or split one cluster with an LLM.

    On any LLM, parsing, or validation failure the input cluster is returned
    as one trivial disambiguated cluster per entity so callers never lose
    entities. The returned :class:`DisambiguationOutput` carries the raw
    prompt, reasoning, and model output for observability.
    """
    if not named_entities:
        raise ValueError("named_entities must not be empty")
    if len(named_entities) == 1:
        return [DisambiguatedNamedEntity.from_named_entity(named_entities[0])], None
    short_named_entities = [
        NamedEntityShort.from_named_entity(i, entity) 
        for i, entity in enumerate(named_entities)
    ]
    text = _format_entities(short_named_entities)
    prompt = DISAMBIGUATION_PROMPT.format(
        entities=text,
    )
    response = client.chat.send(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning={"effort": reasoning_effort},
    )
    message = response.choices[0].message
    raw_content = message.content
    raw_reasoning = getattr(message, "reasoning", "") or ""
    if not isinstance(raw_reasoning, str):
        raw_reasoning = str(raw_reasoning)
    llm = LLMBase(
        text=text,
        prompt=prompt,
        reasoning=raw_reasoning,
        raw_output=raw_content if isinstance(raw_content, str) else "",
    )
    if not isinstance(raw_content, str):
        logger.warning("Disambiguation returned no text content; keeping cluster")
        return _trivial_output(named_entities, llm)

    parsed = parse_json_object(raw_content)
    if not parsed:
        logger.warning("Disambiguation response is not valid JSON; keeping cluster")
        return _trivial_output(named_entities, llm)

    result = _clusters_from_response(parsed, named_entities, llm)
    if result is None:
        logger.warning("Disambiguation response failed validation; keeping cluster")
        return _trivial_output(named_entities, llm)
    return result


def _trivial_clusters(
    named_entities: list[NamedEntity],
) -> list[DisambiguatedNamedEntity]:
    """One singleton disambiguated cluster per input entity."""
    return [
        DisambiguatedNamedEntity.from_named_entity(entity)
        for index, entity in enumerate(named_entities, start=1)
    ]


def _trivial_output(
    named_entities: list[NamedEntity],
    llm: LLMBase | None = None,
) -> tuple[list[DisambiguatedNamedEntity], LLMBase | None]:
    """Fallback output with one singleton cluster per entity."""
    return (
        _trivial_clusters(named_entities),
        llm,
    )


def document_entities_clustering_pipeline(
    document_ner: DocumentNerBase[T],
    *,
    client: Any,  # openrouter client: third-party package without type stubs
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    per_entity_aggregation: Literal["max", "mean", "min"] = "max",
    mention_weight: float = DEFAULT_MENTION_WEIGHT,
    name_weight: float = DEFAULT_NAME_WEIGHT,
) -> list[DisambiguationOutput]:
    """Cluster a document's entities and disambiguate every cluster."""
    indexed_entities = [
        (chunk.chunk_id, entity)
        for chunk in document_ner.chunks
        for entity in chunk.valid_entities
        if isinstance(entity, NamedEntity)
    ]
    for chunk_id, entity in indexed_entities:
        for mention in entity.mentions:
            mention.chunk_id = chunk_id
    entities = [entity for _, entity in indexed_entities]
    entity_clusters = cluster_entities(
        entities,
        threshold=threshold,
        per_entity_aggregation=per_entity_aggregation,
        mention_weight=mention_weight,
        name_weight=name_weight,
    )
    disambiguated_entities = [
        llm_disambiguate_entities(
            entities,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        for entities in entity_clusters
    ]
    return [
        DisambiguationOutput(
            text="",
            prompt=llm.prompt if llm is not None else "",
            reasoning=llm.reasoning if llm is not None else "",
            raw_output=llm.raw_output if llm is not None else "",
            clusters=clusters,
        )
        for clusters, llm in disambiguated_entities
    ]


# ---------------------------------------------------------------------------
# Configured-storage adapter
# ---------------------------------------------------------------------------


def _iter_ner_pages(input_prefix: str) -> Iterator[PageInput]:
    """Yield saved LLM NER page outputs below a storage prefix.

    The expected key structure is ``<prefix>/<library>/<document-id>/<page-id>.json``
    as written by :func:`people_gator_common.pipelines.llm_ner.run_llm_ner_pipeline`.
    """
    input_prefix = posixpath.normpath(input_prefix)
    pattern = posixpath.join(input_prefix, "*", "*", "*.json")
    page_paths = sorted(storage_glob(pattern))
    if not page_paths:
        raise ValueError(
            f"No LLM NER outputs found below storage prefix: {input_prefix}"
        )

    for page_path in page_paths:
        relative_path = posixpath.relpath(page_path, input_prefix)
        parts = relative_path.split("/")
        if len(parts) != 3:
            logger.warning("Ignoring unexpected NER output key: %s", page_path)
            continue
        library, document_id, page_name = parts
        if not page_name.endswith(".json"):
            continue
        yield PageInput(
            library=library,
            document_id=document_id,
            page_id=page_name.removesuffix(".json"),
            path=page_path,
        )


def _run_with_client(
    *,
    input_prefix: str,
    output_prefix: str,
    client: Any,  # openrouter client: third-party package without type stubs
    model: str,
    reasoning_effort: ReasoningEffort,
    threshold: float,
    mention_weight: float,
    name_weight: float,
    force: bool,
    max_pages: int | None,
) -> NerClustersRunSummary:
    processed_pages = 0
    skipped_pages = 0
    total_clusters = 0
    page_inputs = list(_iter_ner_pages(input_prefix))

    existing_outputs: set[str] = set()
    if not force:
        output_pattern = posixpath.join(output_prefix, "*", "*", "*.json")
        existing_outputs = set(storage_glob(output_pattern))

    logger.info(
        "Found %d NER pages and %d existing cluster outputs",
        len(page_inputs),
        len(existing_outputs),
    )

    for page_input in page_inputs:
        if max_pages is not None and processed_pages >= max_pages:
            break
        output_path = posixpath.join(
            output_prefix,
            page_input.library,
            page_input.document_id,
            f"{page_input.page_id}.json",
        )
        if output_path in existing_outputs:
            skipped_pages += 1
            logger.debug("Skipping existing cluster output %s", output_path)
            continue

        logger.info(
            "Processing %s/%s/%s",
            page_input.library,
            page_input.document_id,
            page_input.page_id,
        )
        page_ner = PageNer.model_validate_json(
            storage_read_text(page_input.path)
        )
        clusters = document_entities_clustering_pipeline(
            page_ner,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            threshold=threshold,
            mention_weight=mention_weight,
            name_weight=name_weight,
        )
        output = PageNerClusters(
            library=page_input.library,
            document_id=page_input.document_id,
            page_id=page_input.page_id,
            clusters=clusters,
        )
        storage_write_text(output_path, output.model_dump_json(indent=2))
        processed_pages += 1
        total_clusters += len(clusters)

    return NerClustersRunSummary(
        processed_pages=processed_pages,
        skipped_pages=skipped_pages,
        clusters=total_clusters,
    )


def run_ner_clusters_pipeline(
    *,
    input_prefix: str | os.PathLike[str] = DEFAULT_INPUT_PREFIX,
    output_prefix: str | os.PathLike[str] = DEFAULT_OUTPUT_PREFIX,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    mention_weight: float = DEFAULT_MENTION_WEIGHT,
    name_weight: float = DEFAULT_NAME_WEIGHT,
    force: bool = False,
    max_pages: int | None = None,
    api_key: str | None = None,
    client: Any | None = None,  # openrouter client: no type stubs
) -> NerClustersRunSummary:
    """Cluster and disambiguate saved LLM NER page outputs.

    Reads page NER JSON written by
    :func:`people_gator_common.pipelines.llm_ner.run_llm_ner_pipeline`,
    clusters each page's entities, disambiguates the clusters with an LLM, and
    writes one cluster JSON per page below ``output_prefix``. Existing outputs
    are skipped unless ``force`` is true. When ``client`` is omitted, an
    OpenRouter client is made using ``api_key`` or ``OPENROUTER_API_KEY``.
    """
    input_prefix = posixpath.normpath(os.fspath(input_prefix))
    output_prefix = posixpath.normpath(os.fspath(output_prefix))
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")

    run_kwargs = {
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "threshold": threshold,
        "mention_weight": mention_weight,
        "name_weight": name_weight,
        "force": force,
        "max_pages": max_pages,
    }
    if client is not None:
        return _run_with_client(client=client, **run_kwargs)

    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for NER clustering")
    with OpenRouter(api_key=resolved_api_key) as openrouter_client:
        return _run_with_client(client=openrouter_client, **run_kwargs)


__all__ = [
    "DEFAULT_CLUSTER_THRESHOLD",
    "DEFAULT_INPUT_PREFIX",
    "DEFAULT_MENTION_WEIGHT",
    "DEFAULT_MODEL",
    "DEFAULT_NAME_WEIGHT",
    "DEFAULT_OUTPUT_PREFIX",
    "DisambiguatedNamedEntity",
    "DisambiguationOutput",
    "DisambiguationResult",
    "NamedEntityShort",
    "NerClustersRunSummary",
    "PageNerClusters",
    "cluster_entities",
    "llm_disambiguate_entities",
    "fuzzy_score_form_aliases",
    "parse_json_object",
    "document_entities_clustering_pipeline",
    "run_ner_clusters_pipeline",
    "score_entities",
]
