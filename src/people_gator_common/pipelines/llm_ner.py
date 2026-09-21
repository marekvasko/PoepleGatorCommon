"""LLM-based named-entity extraction for PeopleGator page layouts.

The text/LLM functions in this module deliberately know nothing about files,
libraries, documents, pages, or OCR identifiers.  The PeopleGator adapter at
the bottom of the module adds those concerns around the text result.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import json
import logging
import os
import posixpath
import re
from typing import Any, Literal
import uuid

from fuzzysearch import find_near_matches
from openrouter import OpenRouter
from pydantic import AliasChoices, BaseModel, Field, ValidationError, model_validator

from people_gator_common.filesystem import (
    glob as storage_glob,
    read_text as storage_read_text,
    write_text as storage_write_text,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_MIN_CHARACTERS_IN_CHUNK = 2048
DEFAULT_OVERLAP_LINES = 1
DEFAULT_INPUT_PREFIX = "page-layouts"
DEFAULT_OUTPUT_PREFIX = "page-named-entities"
DEFAULT_POSTPROCESSED_OUTPUT_PREFIX = "page-named-entities-exact-v4"
DEFAULT_MENTION_MATCH_SCORE_CUTOFF = 80.0

# Stable namespace for chunk UUIDs derived from chunk text (uuid5/SHA-1).
CHUNK_ID_NAMESPACE = uuid.UUID("d4f9a3c2-7b1e-4e6a-9f2d-8c5b1a0e4d77")

PROMPT = """\
You extract named entities of type beeing from the supplied text. This includes
people, mythical creatures, film or cartoon characters, gods, and other
personal entities.

Return each entity once. Its top-level `name` may be a canonical name deduced
from the text. Write `info` in the language of the source text.

Every item in `mentions` must be extractive:
- Copy `mentions[].name` and `mentions[].context` character-for-character
  from the supplied text.
- `context` must be one contiguous substring of the supplied text.
- `name` must be one contiguous substring of `context`.
- Preserve capitalization, punctuation, OCR spelling, hyphenation, spaces, and
  line breaks exactly. Encode source line breaks as `\\n` in JSON.
- Never paraphrase, normalize, repair OCR, join separated passages, or omit
  intervening text.
- Never insert `...` or `…` unless those characters occur literally in the
  supplied text.
- Choose the shortest contiguous context that contains the mention and enough
  local text to identify it.
- Include every mention that satisfies these rules. If an exact extractive
  mention cannot be located, omit it rather than inventing or rewriting it.

The top-level entity `name` may be canonicalized, but `mentions[].name` must
always retain the exact source spelling.

Return only a JSON array in this format:
```json
[
  {{
    "name": "<canonical_entity_name>",
    "type": "beeing",
    "mentions": [
      {{
        "name": "<verbatim_contiguous_source_name>",
        "context": "<verbatim_contiguous_source_context>"
      }}
    ],
    "info": "<relevant_info>"
  }}
]
```

Here is the text:
```
{text}
```
"""

type ReasoningEffort = Literal["low", "medium", "high"]
type Span = tuple[int, int]


class EntityType(StrEnum):
    """Entity labels emitted by the extraction prompt.

    ``beeing`` intentionally preserves the existing notebook/output contract.
    """

    BEEING = "beeing"


class ExactMatchMethod(StrEnum):
    CONTEXT_EXACT_NAME_EXACT = "context_exact_name_exact"
    CONTEXT_EXACT_NAME_FUZZY = "context_exact_name_fuzzy"
    CONTEXT_FUZZY_NAME_EXACT = "context_fuzzy_name_exact"
    CONTEXT_FUZZY_NAME_FUZZY = "context_fuzzy_name_fuzzy"
    NAME_EXACT_FALLBACK = "name_exact_fallback"
    NAME_NORMALIZED_FALLBACK = "name_normalized_fallback"
    NAME_FUZZY_FALLBACK = "name_fuzzy_fallback"
    UNMATCHED = "unmatched"


class NamedEntityMention(BaseModel):
    context: str
    name: str
    exact_context: str | None = None
    exact_name: str | None = None
    exact_method: ExactMatchMethod | None = None
    chunk_id: uuid.UUID | None = None


class NamedEntity(BaseModel):
    """An entity extracted from text, without page or storage metadata."""
    name: str
    type: EntityType
    mentions: list[NamedEntityMention]
    info: str


class ChunkNerBase(BaseModel):
    """Base class for text chunk outputs from the LLM."""

    chunk_id: uuid.UUID | None = None
    text: str = Field(validation_alias=AliasChoices("text", "chunk_text"))
    valid_entities: list[NamedEntity]
    invalid_entities: list[Any]

    @model_validator(mode="after")
    def _assign_stable_chunk_id(self) -> ChunkNerBase:
        """Derive a stable chunk UUID from the chunk text via SHA-1 (uuid5).

        Re-running the pipeline on the same text yields the same ``chunk_id``,
        so saved outputs are reproducible and idempotent. Chunks with
        identical text share the same identity by design.
        """
        if self.chunk_id is None:
            self.chunk_id = uuid.uuid5(CHUNK_ID_NAMESPACE, self.text)
        return self

class ChunkMetadataBase(BaseModel):
    """Line range for the chunk at the same list index."""
    start_line: int # Inclusive start line of the chunk.
    end_line: int # Exclusive end line of the chunk.

class DocumentNerBase[T: ChunkNerBase](BaseModel):
    """Base class for document-level metadata."""

    text: str
    chunks: list[T]
    chunk_metadata: list[ChunkMetadataBase]

class LLMBase(BaseModel):
    """Generic LLM output for text chunks."""
    text: str
    prompt: str
    reasoning: str
    raw_output: str

class ChunkNerLLM(ChunkNerBase, LLMBase):
    """Pure text/LLM result for one chunk."""


class PageLineMapping(BaseModel):
    region_id: str
    line_id: str
    
class PageCharMapping(BaseModel):
    region_id: str
    line_id: str
    char_index: int


class PageChunkMetadata(ChunkMetadataBase):
    """Page-specific metadata for the chunk at the same list index."""
    line_mapping: list[PageLineMapping]



class PageNer(DocumentNerBase[ChunkNerLLM]):
    """Text NER output enriched only with PeopleGator page metadata."""
    library: str
    document_id: str
    page_id: str
    chunks: list[ChunkNerLLM]
    chunk_metadata: list[PageChunkMetadata]


@dataclass(frozen=True)
class PageInput:
    library: str
    document_id: str
    page_id: str
    path: str


@dataclass(frozen=True)
class LlmNerRunSummary:
    processed_pages: int
    skipped_pages: int


@dataclass(frozen=True)
class LlmNerPostprocessSummary:
    processed_pages: int
    skipped_pages: int
    mentions: int
    matched_contexts: int
    matched_names: int


# ---------------------------------------------------------------------------
# Pure text/LLM processing
# ---------------------------------------------------------------------------


def lines_to_text(lines: list[str]) -> str:
    return "\n".join(lines)


def chunk_lines(
    lines: list[str],
    *,
    min_characters_in_chunk: int = DEFAULT_MIN_CHARACTERS_IN_CHUNK,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Split text lines and return half-open source-line ranges per chunk."""
    if min_characters_in_chunk <= 0:
        raise ValueError("min_characters_in_chunk must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")

    chunks: list[str] = []
    chunk_line_ranges: list[tuple[int, int]] = []
    current_lines: list[str] = []
    current_length = 0
    start_line = 0

    for line in lines:
        if current_lines and current_length >= min_characters_in_chunk:
            end_line = start_line + len(current_lines)
            chunks.append(lines_to_text(current_lines))
            chunk_line_ranges.append((start_line, end_line))

            overlap = min(overlap_lines, len(current_lines))
            current_lines = current_lines[-overlap:] if overlap else []
            start_line = end_line - overlap
            current_length = sum(len(item) for item in current_lines)

        current_lines.append(line)
        current_length += len(line)

    if current_lines:
        end_line = start_line + len(current_lines)
        chunks.append(lines_to_text(current_lines))
        chunk_line_ranges.append((start_line, end_line))

    return chunks, chunk_line_ranges


def parse_json_result(response: str) -> list[Any]:
    """Parse a JSON array from a plain or Markdown-fenced model response."""
    candidates: list[str] = [
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            re.IGNORECASE | re.DOTALL,
        )
    ]
    candidates.append(response.strip())

    first_bracket = response.find("[")
    last_bracket = response.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        candidates.append(response[first_bracket : last_bracket + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, list):
            return result
    return []


def _find_exact_span(text: str, value: str) -> Span | None:
    """Return the first exact occurrence of ``value`` in ``text``."""
    if not value:
        return None
    start = text.find(value)
    if start < 0:
        return None
    return start, start + len(value)


def _find_fuzzy_span(
    text: str,
    value: str,
    *,
    score_cutoff: float,
) -> Span | None:
    """Return the closest non-exact occurrence of ``value`` in ``text``."""
    if not value:
        return None
    max_l_dist = int(len(value) * (100 - score_cutoff) / 100)
    if max_l_dist <= 0:
        return None
    matches = [
        match
        for match in find_near_matches(value, text, max_l_dist=max_l_dist)
        if match.dist > 0
    ]
    if not matches:
        return None
    match = min(
        matches,
        key=lambda item: (
            item.dist,
            abs((item.end - item.start) - len(value)),
            item.start,
        ),
    )
    return match.start, match.end


def normalize_with_offsets(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize case, whitespace, and line hyphenation while retaining offsets."""
    normalized: list[str] = []
    offsets: list[tuple[int, int]] = []
    index = 0
    while index < len(value):
        if value[index] == "-":
            after_hyphen = index + 1
            while after_hyphen < len(value) and value[after_hyphen] in " \t\r":
                after_hyphen += 1
            if after_hyphen < len(value) and value[after_hyphen] == "\n":
                index = after_hyphen + 1
                while index < len(value) and value[index] in " \t":
                    index += 1
                continue

        if value[index].isspace():
            whitespace_start = index
            while index < len(value) and value[index].isspace():
                index += 1
            if normalized and normalized[-1] != " ":
                normalized.append(" ")
                offsets.append((whitespace_start, index))
            continue

        folded = value[index].casefold()
        normalized.extend(folded)
        offsets.extend([(index, index + 1)] * len(folded))
        index += 1

    if normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized), offsets


def _line_context_span(text: str, start: int, end: int) -> Span:
    """Return the smallest complete source line range containing a span."""
    context_start = text.rfind("\n", 0, start) + 1
    context_end = text.find("\n", end)
    if context_end < 0:
        context_end = len(text)
    return context_start, context_end


def _find_name_in_context(
    text: str,
    context_span: Span | None,
    name: str,
    *,
    fuzzy: bool,
    score_cutoff: float,
) -> Span | None:
    """Find a name inside an already matched context span."""
    if context_span is None:
        return None
    context_start, context_end = context_span
    context_text = text[context_start:context_end]
    relative_span = (
        _find_fuzzy_span(context_text, name, score_cutoff=score_cutoff)
        if fuzzy
        else _find_exact_span(context_text, name)
    )
    if relative_span is None:
        return None
    return context_start + relative_span[0], context_start + relative_span[1]


def _find_hierarchical_spans(
    text: str,
    context: str,
    name: str,
    *,
    fuzzy_context: bool,
    fuzzy_name: bool,
    score_cutoff: float,
) -> tuple[Span | None, Span | None]:
    """Find one context/name pair using exactly one matching strategy."""
    context_span = (
        _find_fuzzy_span(text, context, score_cutoff=score_cutoff)
        if fuzzy_context
        else _find_exact_span(text, context)
    )
    name_span = _find_name_in_context(
        text,
        context_span,
        name,
        fuzzy=fuzzy_name,
        score_cutoff=score_cutoff,
    )
    return context_span, name_span


def _find_normalized_span(
    text: str,
    value: str,
    *,
    score_cutoff: float,
) -> Span | None:
    """Find an exact or fuzzy normalized value and map it to source offsets."""
    normalized_text, text_offsets = normalize_with_offsets(text)
    normalized_value, _ = normalize_with_offsets(value)
    if not normalized_value:
        return None

    normalized_span = _find_exact_span(normalized_text, normalized_value)
    if normalized_span is None:
        normalized_span = _find_fuzzy_span(
            normalized_text,
            normalized_value,
            score_cutoff=score_cutoff,
        )
    if normalized_span is None:
        return None

    start, end = normalized_span
    return text_offsets[start][0], text_offsets[end - 1][1]


def find_exact_context_and_name(
    text: str,
    context: str,
    name: str,
    *,
    score_cutoff: float = DEFAULT_MENTION_MATCH_SCORE_CUTOFF,
) -> tuple[str | None, str | None, ExactMatchMethod]:
    """Map an LLM mention through a simple, ordered strategy waterfall."""
    if not 0 <= score_cutoff <= 100:
        raise ValueError("score_cutoff must be between 0 and 100")
    if not text:
        return None, None, ExactMatchMethod.UNMATCHED

    context_span: Span | None = None
    name_span: Span | None = None
    exact_method = ExactMatchMethod.UNMATCHED

    if name_span is None and context and name:
        context_span, name_span = _find_hierarchical_spans(
            text,
            context,
            name,
            fuzzy_context=False,
            fuzzy_name=False,
            score_cutoff=score_cutoff,
        )
        if name_span is not None:
            exact_method = ExactMatchMethod.CONTEXT_EXACT_NAME_EXACT

    if name_span is None and context and name:
        context_span, name_span = _find_hierarchical_spans(
            text,
            context,
            name,
            fuzzy_context=False,
            fuzzy_name=True,
            score_cutoff=score_cutoff,
        )
        if name_span is not None:
            exact_method = ExactMatchMethod.CONTEXT_EXACT_NAME_FUZZY

    if name_span is None and context and name:
        context_span, name_span = _find_hierarchical_spans(
            text,
            context,
            name,
            fuzzy_context=True,
            fuzzy_name=False,
            score_cutoff=score_cutoff,
        )
        if name_span is not None:
            exact_method = ExactMatchMethod.CONTEXT_FUZZY_NAME_EXACT

    if name_span is None and context and name:
        context_span, name_span = _find_hierarchical_spans(
            text,
            context,
            name,
            fuzzy_context=True,
            fuzzy_name=True,
            score_cutoff=score_cutoff,
        )
        if name_span is not None:
            exact_method = ExactMatchMethod.CONTEXT_FUZZY_NAME_FUZZY

    if name_span is None and name:
        name_span = _find_exact_span(text, name)
        if name_span is not None:
            exact_method = ExactMatchMethod.NAME_EXACT_FALLBACK

    if name_span is None and name:
        name_span = _find_fuzzy_span(text, name, score_cutoff=score_cutoff)
        if name_span is not None:
            exact_method = ExactMatchMethod.NAME_FUZZY_FALLBACK

    if name_span is None and name:
        name_span = _find_normalized_span(
            text,
            name,
            score_cutoff=score_cutoff,
        )
        if name_span is not None:
            exact_method = ExactMatchMethod.NAME_NORMALIZED_FALLBACK

    if name_span is not None and exact_method in {
        ExactMatchMethod.NAME_EXACT_FALLBACK,
        ExactMatchMethod.NAME_FUZZY_FALLBACK,
        ExactMatchMethod.NAME_NORMALIZED_FALLBACK,
    }:
        context_span = _line_context_span(text, *name_span)

    exact_context = (
        text[context_span[0] : context_span[1]] if context_span is not None else None
    )
    exact_name = text[name_span[0] : name_span[1]] if name_span is not None else None
    return exact_context, exact_name, exact_method


def _enrich_exact_matches(
    output: dict[str, Any],
    *,
    score_cutoff: float,
) -> tuple[int, int, int]:
    mentions = 0
    matched_contexts = 0
    matched_names = 0
    chunks = output.get("chunks", [])
    if not isinstance(chunks, list):
        return mentions, matched_contexts, matched_names

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_text = chunk.get("chunk_text", "")
        if not isinstance(chunk_text, str):
            continue
        entities = chunk.get("valid_entities", [])
        if not isinstance(entities, list):
            continue
        invalid_entities = chunk.get("invalid_entities", [])
        if not isinstance(invalid_entities, list):
            invalid_entities = []
        matched_entities: list[Any] = []
        for entity in entities:
            if not isinstance(entity, dict):
                invalid_entities.append(entity)
                continue
            entity_mentions = entity.get("mentions", [])
            if not isinstance(entity_mentions, list):
                invalid_entities.append(entity)
                continue
            matched_mentions: list[Any] = []
            unmatched_mentions: list[Any] = []
            for mention in entity_mentions:
                if not isinstance(mention, dict):
                    unmatched_mentions.append(mention)
                    continue
                context = mention.get("context", "")
                name = mention.get("name", "")
                if not isinstance(context, str) or not isinstance(name, str):
                    unmatched_mentions.append(mention)
                    continue
                exact_context, exact_name, exact_method = find_exact_context_and_name(
                    chunk_text,
                    context,
                    name,
                    score_cutoff=score_cutoff,
                )
                mention["exact_context"] = exact_context
                mention["exact_name"] = exact_name
                mention["exact_method"] = exact_method.value
                mentions += 1
                matched_contexts += exact_context is not None
                matched_names += exact_name is not None
                if exact_context is not None and exact_name is not None:
                    matched_mentions.append(mention)
                else:
                    unmatched_mentions.append(mention)

            if matched_mentions:
                matched_entity = deepcopy(entity)
                matched_entity["mentions"] = matched_mentions
                matched_entities.append(matched_entity)
            if unmatched_mentions:
                unmatched_entity = deepcopy(entity)
                unmatched_entity["mentions"] = unmatched_mentions
                invalid_entities.append(unmatched_entity)

        chunk["valid_entities"] = matched_entities
        chunk["invalid_entities"] = invalid_entities

    return mentions, matched_contexts, matched_names


def postprocess_exact_matches(
    output: Mapping[str, Any],
    *,
    score_cutoff: float = DEFAULT_MENTION_MATCH_SCORE_CUTOFF,
) -> dict[str, Any]:
    """Return a copy of a saved LLM NER output enriched with exact matches."""
    if not 0 <= score_cutoff <= 100:
        raise ValueError("score_cutoff must be between 0 and 100")
    enriched = deepcopy(dict(output))
    _enrich_exact_matches(enriched, score_cutoff=score_cutoff)
    return enriched


def process_chunk(
    text: str,
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
) -> ChunkNerLLM:
    """Extract and validate entities from one string of text."""
    prompt = PROMPT.format(text=text)
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
    if not isinstance(raw_content, str):
        return ChunkNerLLM(
            text=text,
            prompt=prompt,
            reasoning=raw_reasoning,
            raw_output="",
            valid_entities=[],
            invalid_entities=[],
        )

    valid_entities: list[NamedEntity] = []
    invalid_entities: list[Any] = []
    for entity_data in parse_json_result(raw_content):
        try:
            entity = NamedEntity.model_validate(entity_data)
        except ValidationError:
            invalid_entities.append(entity_data)
            continue
        matched_mentions: list[NamedEntityMention] = []
        unmatched_mentions: list[NamedEntityMention] = []
        for mention in entity.mentions:
            (
                mention.exact_context,
                mention.exact_name,
                mention.exact_method,
            ) = find_exact_context_and_name(
                text,
                mention.context,
                mention.name,
            )
            if mention.exact_context is not None and mention.exact_name is not None:
                matched_mentions.append(mention)
            else:
                unmatched_mentions.append(mention)

        if matched_mentions:
            valid_entities.append(entity.model_copy(update={"mentions": matched_mentions}))
        if unmatched_mentions:
            invalid_entity = entity.model_copy(update={"mentions": unmatched_mentions})
            invalid_entities.append(invalid_entity.model_dump(mode="json"))

    return ChunkNerLLM(
        text=text,
        prompt=prompt,
        reasoning=raw_reasoning,
        raw_output=raw_content,
        valid_entities=valid_entities,
        invalid_entities=invalid_entities,
    )


def process_text_lines(
    lines: list[str],
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    min_characters_in_chunk: int = DEFAULT_MIN_CHARACTERS_IN_CHUNK,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> DocumentNerBase[ChunkNerLLM]:
    """Chunk text lines and run LLM NER without any page or I/O concerns."""
    chunks, chunk_line_ranges = chunk_lines(
        lines,
        min_characters_in_chunk=min_characters_in_chunk,
        overlap_lines=overlap_lines,
    )
    return DocumentNerBase(
        text=lines_to_text(lines),
        chunks=[
            process_chunk(
                chunk,
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            for chunk in chunks
        ],
        chunk_metadata=[
            ChunkMetadataBase(start_line=start_line, end_line=end_line)
            for start_line, end_line in chunk_line_ranges
        ],
    )


# ---------------------------------------------------------------------------
# PeopleGator page and filesystem adapter
# ---------------------------------------------------------------------------


def iter_peoplegator_pages(input_prefix: str | os.PathLike[str]) -> Iterator[PageInput]:
    """Yield layout objects below a configured-storage prefix.

    The expected key structure is
    ``<prefix>/<library>/<document-id>/<page-id>.json``. Returned
    paths are storage-relative and can be opened with
    :mod:`people_gator_common.filesystem`.
    """
    input_prefix = posixpath.normpath(os.fspath(input_prefix))
    pattern = posixpath.join(input_prefix, "*", "*", "*.json")
    page_paths = sorted(storage_glob(pattern))
    if not page_paths:
        raise ValueError(
            f"No PeopleGator page layouts found below storage prefix: "
            f"{input_prefix}"
        )

    for page_path in page_paths:
        relative_path = posixpath.relpath(page_path, input_prefix)
        parts = relative_path.split("/")
        if len(parts) != 3:
            logger.warning("Ignoring unexpected layout key: %s", page_path)
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


def page_to_lines_and_identifiers(
    page: Mapping[str, Any],
) -> tuple[list[str], list[PageLineMapping]]:
    """Adapt a PeopleGator page layout to aligned text lines and identifiers."""
    lines: list[str] = []
    identifiers: list[PageLineMapping] = []
    regions = page.get("regions", [])
    if not isinstance(regions, list):
        return lines, identifiers

    for region in regions:
        if not isinstance(region, Mapping):
            continue
        region_id = str(region.get("id", ""))
        region_lines = region.get("lines", [])
        if not isinstance(region_lines, list):
            continue
        for line in region_lines:
            if not isinstance(line, Mapping):
                continue
            transcription = line.get("transcription", "")
            lines.append("" if transcription is None else str(transcription))
            identifiers.append(
                PageLineMapping(
                    region_id=region_id,
                    line_id=str(line.get("id", "")),
                )
            )
    return lines, identifiers


def process_page(
    page: Mapping[str, Any],
    *,
    library: str,
    document_id: str,
    page_id: str,
    client: Any,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    min_characters_in_chunk: int = DEFAULT_MIN_CHARACTERS_IN_CHUNK,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> PageNer:
    """Adapt one PeopleGator layout around the pure text processor."""
    lines, line_identifiers = page_to_lines_and_identifiers(page)
    text_output = process_text_lines(
        lines,
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        min_characters_in_chunk=min_characters_in_chunk,
        overlap_lines=overlap_lines,
    )
    chunk_metadata = [
        PageChunkMetadata(
            start_line=metadata.start_line,
            end_line=metadata.end_line,
            line_mapping=line_identifiers[
                metadata.start_line : metadata.end_line
            ],
        )
        for metadata in text_output.chunk_metadata
    ]
    return PageNer(
        library=library,
        document_id=document_id,
        page_id=page_id,
        text=text_output.text,
        chunk_metadata=chunk_metadata,
        chunks=text_output.chunks,
    )


def _run_with_client(
    *,
    input_prefix: str,
    output_prefix: str,
    client: Any,
    model: str,
    reasoning_effort: ReasoningEffort,
    min_characters_in_chunk: int,
    overlap_lines: int,
    force: bool,
    max_pages: int | None,
) -> LlmNerRunSummary:
    processed_pages = 0
    skipped_pages = 0
    page_inputs = list(iter_peoplegator_pages(input_prefix))

    existing_outputs: set[str] = set()
    if not force:
        output_pattern = posixpath.join(output_prefix, "*", "*", "*.json")
        existing_outputs = set(storage_glob(output_pattern))

    logger.info(
        "Found %d layout pages and %d existing NER outputs",
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
            logger.debug("Skipping existing page output %s", output_path)
            continue

        logger.info(
            "Processing %s/%s/%s",
            page_input.library,
            page_input.document_id,
            page_input.page_id,
        )
        page = json.loads(storage_read_text(page_input.path))
        if not isinstance(page, Mapping):
            raise ValueError(f"Page layout must be a JSON object: {page_input.path}")

        page_output = process_page(
            page,
            library=page_input.library,
            document_id=page_input.document_id,
            page_id=page_input.page_id,
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            min_characters_in_chunk=min_characters_in_chunk,
            overlap_lines=overlap_lines,
        )
        storage_write_text(output_path, page_output.model_dump_json(indent=2))
        processed_pages += 1

    return LlmNerRunSummary(
        processed_pages=processed_pages,
        skipped_pages=skipped_pages,
    )


def run_llm_ner_postprocessing(
    *,
    input_prefix: str | os.PathLike[str] = DEFAULT_OUTPUT_PREFIX,
    output_prefix: str | os.PathLike[str] = DEFAULT_POSTPROCESSED_OUTPUT_PREFIX,
    score_cutoff: float = DEFAULT_MENTION_MATCH_SCORE_CUTOFF,
    force: bool = False,
    max_pages: int | None = None,
) -> LlmNerPostprocessSummary:
    """Enrich saved LLM NER JSON without making any LLM calls.

    The input's JSON structure and unknown fields are preserved.
    Exact values and ``exact_method`` are added or refreshed, and mentions
    which still cannot be matched are moved from ``valid_entities`` to
    ``invalid_entities``. By default, results are written to a separate storage
    prefix so the original model outputs remain unchanged.
    """
    input_prefix = posixpath.normpath(os.fspath(input_prefix))
    output_prefix = posixpath.normpath(os.fspath(output_prefix))
    if not 0 <= score_cutoff <= 100:
        raise ValueError("score_cutoff must be between 0 and 100")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")

    input_pattern = posixpath.join(input_prefix, "*", "*", "*.json")
    input_paths = sorted(storage_glob(input_pattern))
    if not input_paths:
        raise ValueError(f"No LLM NER outputs found below storage prefix: {input_prefix}")

    existing_outputs: set[str] = set()
    if not force:
        output_pattern = posixpath.join(output_prefix, "*", "*", "*.json")
        existing_outputs = set(storage_glob(output_pattern))

    processed_pages = 0
    skipped_pages = 0
    mentions = 0
    matched_contexts = 0
    matched_names = 0
    for input_path in input_paths:
        if max_pages is not None and processed_pages >= max_pages:
            break
        relative_path = posixpath.relpath(input_path, input_prefix)
        output_path = posixpath.join(output_prefix, relative_path)
        if output_path in existing_outputs:
            skipped_pages += 1
            continue

        parsed = json.loads(storage_read_text(input_path))
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM NER output must be a JSON object: {input_path}")
        enriched = deepcopy(parsed)
        page_mentions, page_contexts, page_names = _enrich_exact_matches(
            enriched,
            score_cutoff=score_cutoff,
        )
        storage_write_text(
            output_path,
            json.dumps(enriched, ensure_ascii=False, indent=2),
        )
        processed_pages += 1
        mentions += page_mentions
        matched_contexts += page_contexts
        matched_names += page_names

    return LlmNerPostprocessSummary(
        processed_pages=processed_pages,
        skipped_pages=skipped_pages,
        mentions=mentions,
        matched_contexts=matched_contexts,
        matched_names=matched_names,
    )


def run_llm_ner_pipeline(
    *,
    input_prefix: str | os.PathLike[str] = DEFAULT_INPUT_PREFIX,
    output_prefix: str | os.PathLike[str] = DEFAULT_OUTPUT_PREFIX,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    min_characters_in_chunk: int = DEFAULT_MIN_CHARACTERS_IN_CHUNK,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
    force: bool = False,
    max_pages: int | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> LlmNerRunSummary:
    """Extract named entities between configured-storage prefixes.

    All layout keys and existing output keys are listed once. Existing outputs
    are skipped unless ``force`` is true, then each missing page is read and
    written through :mod:`people_gator_common.filesystem`. When ``client`` is
    omitted, an OpenRouter client is made using ``api_key`` or
    ``OPENROUTER_API_KEY``.
    """
    input_prefix = posixpath.normpath(os.fspath(input_prefix))
    output_prefix = posixpath.normpath(os.fspath(output_prefix))
    if min_characters_in_chunk <= 0:
        raise ValueError("min_characters_in_chunk must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")

    run_kwargs = {
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "min_characters_in_chunk": min_characters_in_chunk,
        "overlap_lines": overlap_lines,
        "force": force,
        "max_pages": max_pages,
    }
    if client is not None:
        return _run_with_client(client=client, **run_kwargs)

    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for LLM NER extraction")
    with OpenRouter(api_key=resolved_api_key) as openrouter_client:
        return _run_with_client(client=openrouter_client, **run_kwargs)


__all__ = [
    "DEFAULT_INPUT_PREFIX",
    "DEFAULT_MENTION_MATCH_SCORE_CUTOFF",
    "DEFAULT_MIN_CHARACTERS_IN_CHUNK",
    "DEFAULT_OUTPUT_PREFIX",
    "DEFAULT_POSTPROCESSED_OUTPUT_PREFIX",
    "DEFAULT_MODEL",
    "DEFAULT_OVERLAP_LINES",
    "ChunkMetadataBase",
    "EntityType",
    "ExactMatchMethod",
    "PageLineMapping",
    "LlmNerPostprocessSummary",
    "LlmNerRunSummary",
    "NamedEntity",
    "NamedEntityMention",
    "PageChunkMetadata",
    "PageInput",
    "PageNer",
    "ChunkNerLLM",
    "ChunkNerBase",
    "chunk_lines",
    "normalize_with_offsets",
    "find_exact_context_and_name",
    "iter_peoplegator_pages",
    "page_to_lines_and_identifiers",
    "parse_json_result",
    "postprocess_exact_matches",
    "process_chunk",
    "process_page",
    "process_text_lines",
    "run_llm_ner_pipeline",
    "run_llm_ner_postprocessing",
]
