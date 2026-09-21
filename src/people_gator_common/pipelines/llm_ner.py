"""LLM-based named-entity extraction for PeopleGator page layouts.

The text/LLM functions in this module deliberately know nothing about files,
libraries, documents, pages, or OCR identifiers.  The PeopleGator adapter at
the bottom of the module adds those concerns around the text result.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
import logging
import os
import posixpath
import re
from typing import Any, Literal

from fuzzysearch import find_near_matches
from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError

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
DEFAULT_MENTION_MATCH_SCORE_CUTOFF = 80.0

PROMPT = """\
You are an AI assistant that extracts a single type of named entity from text.
Extract every entity of type beeing. This includes people, mythical creatures,
film or cartoon characters, gods, and other personal entities.
Use the canonical name that can be deduced from the text. Include relevant
information about the entity in the language of the source text. Include every
mention from the text, with its exact spelling and the smallest useful matching
context. Represent line breaks in JSON strings as \\n.
Return only JSON in this format:
```json
[
  {{
    "name": "<entity_name>",
    "type": "beeing",
    "mentions": [
      {{
        "name": "<exact_mention>",
        "context": "<minimal_context>"
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

ReasoningEffort = Literal["low", "medium", "high"]


class EntityType(StrEnum):
    """Entity labels emitted by the extraction prompt.

    ``beeing`` intentionally preserves the existing notebook/output contract.
    """

    BEEING = "beeing"


class NamedEntityMention(BaseModel):
    context: str
    name: str
    exact_context: str | None = None
    exact_name: str | None = None


class NamedEntity(BaseModel):
    """An entity extracted from text, without page or storage metadata."""

    name: str
    type: EntityType
    mentions: list[NamedEntityMention]
    info: str


class TextChunkNerOutput(BaseModel):
    """Pure text/LLM result for one chunk."""

    chunk_text: str
    prompt: str
    reasoning: str
    raw_output: str
    valid_entities: list[NamedEntity]
    invalid_entities: list[Any]


class ChunkMetadata(BaseModel):
    """Line range for the chunk at the same list index."""

    start_line: int
    end_line: int


class TextNerOutput(BaseModel):
    """Pure text/LLM output with no page or storage metadata."""

    text: str
    chunks: list[TextChunkNerOutput]
    chunk_metadata: list[ChunkMetadata]


class LineMapping(BaseModel):
    region_id: str
    line_id: str
    
class LineCharMapping(BaseModel):
    region_id: str
    line_id: str
    char_index: int


class PageChunkMetadata(ChunkMetadata):
    """Page-specific metadata for the chunk at the same list index."""

    line_mapping: list[LineMapping]


class PageNersOutput(TextNerOutput):
    """Text NER output enriched only with PeopleGator page metadata."""

    library: str
    document_id: str
    page_id: str
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


def _find_closest_substring(
    text: str,
    value: str,
    *,
    score_cutoff: float,
) -> str | None:
    """Return the closest qualifying span as a literal slice of ``text``."""
    exact_start = text.find(value)
    if exact_start >= 0:
        return value

    max_l_dist = int(len(value) * (100 - score_cutoff) / 100)
    if max_l_dist == 0:
        return None
    matches = find_near_matches(value, text, max_l_dist=max_l_dist)
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
    return text[match.start : match.end]


def find_exact_context_and_name(
    text: str,
    context: str,
    name: str,
    *,
    score_cutoff: float = DEFAULT_MENTION_MATCH_SCORE_CUTOFF,
) -> tuple[str | None, str | None]:
    """Map an LLM mention back to literal substrings of its source text.

    The context is aligned first so that it can disambiguate repeated names.
    The mention name is then aligned only within that matched context. Each
    non-``None`` result is sliced from its containing source, which guarantees
    ``exact_context in text`` and ``exact_name in exact_context``.
    """
    if not 0 <= score_cutoff <= 100:
        raise ValueError("score_cutoff must be between 0 and 100")
    if not text or not context:
        return None, None

    exact_context = _find_closest_substring(
        text,
        context,
        score_cutoff=score_cutoff,
    )
    if exact_context is None:
        return None, None

    if not name:
        return exact_context, None

    exact_name = _find_closest_substring(
        exact_context,
        name,
        score_cutoff=score_cutoff,
    )
    return exact_context, exact_name


def process_chunk(
    text: str,
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
) -> TextChunkNerOutput:
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
        return TextChunkNerOutput(
            chunk_text=text,
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
        for mention in entity.mentions:
            mention.exact_context, mention.exact_name = find_exact_context_and_name(
                text,
                mention.context,
                mention.name,
            )
        valid_entities.append(entity)

    return TextChunkNerOutput(
        chunk_text=text,
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
) -> TextNerOutput:
    """Chunk text lines and run LLM NER without any page or I/O concerns."""
    chunks, chunk_line_ranges = chunk_lines(
        lines,
        min_characters_in_chunk=min_characters_in_chunk,
        overlap_lines=overlap_lines,
    )
    return TextNerOutput(
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
            ChunkMetadata(start_line=start_line, end_line=end_line)
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
) -> tuple[list[str], list[LineMapping]]:
    """Adapt a PeopleGator page layout to aligned text lines and identifiers."""
    lines: list[str] = []
    identifiers: list[LineMapping] = []
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
                LineMapping(
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
) -> PageNersOutput:
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
    return PageNersOutput(
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
    "DEFAULT_MODEL",
    "DEFAULT_OVERLAP_LINES",
    "ChunkMetadata",
    "EntityType",
    "LineMapping",
    "LlmNerRunSummary",
    "NamedEntity",
    "NamedEntityMention",
    "PageChunkMetadata",
    "PageInput",
    "PageNersOutput",
    "TextChunkNerOutput",
    "TextNerOutput",
    "chunk_lines",
    "find_exact_context_and_name",
    "iter_peoplegator_pages",
    "page_to_lines_and_identifiers",
    "parse_json_result",
    "process_chunk",
    "process_page",
    "process_text_lines",
    "run_llm_ner_pipeline",
]
