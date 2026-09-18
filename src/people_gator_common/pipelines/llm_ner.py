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
from pathlib import Path
import re
from typing import Any, Literal

from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_MIN_CHARACTERS_IN_CHUNK = 2048
DEFAULT_OVERLAP_LINES = 1

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


class LineIdentifier(BaseModel):
    region_id: str
    line_id: str


class PageChunkMetadata(ChunkMetadata):
    """Page-specific metadata for the chunk at the same list index."""

    line_identifiers: list[LineIdentifier]


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
    path: Path


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
            valid_entities.append(NamedEntity.model_validate(entity_data))
        except ValidationError:
            invalid_entities.append(entity_data)

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


def iter_peoplegator_pages(dataset_dir: str | Path) -> Iterator[PageInput]:
    """Yield page files from a PeopleGator dataset in deterministic order."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise ValueError(f"PeopleGator dataset directory does not exist: {root}")

    for library_dir in sorted(root.iterdir()):
        if not library_dir.is_dir():
            continue
        for layout_dir in sorted(library_dir.iterdir()):
            if not layout_dir.is_dir() or not layout_dir.name.endswith(".layouts"):
                continue
            document_id = layout_dir.name.removesuffix(".layouts")
            for page_path in sorted(layout_dir.glob("*.json")):
                yield PageInput(
                    library=library_dir.name,
                    document_id=document_id,
                    page_id=page_path.stem,
                    path=page_path,
                )


def page_to_lines_and_identifiers(
    page: Mapping[str, Any],
) -> tuple[list[str], list[LineIdentifier]]:
    """Adapt a PeopleGator page layout to aligned text lines and identifiers."""
    lines: list[str] = []
    identifiers: list[LineIdentifier] = []
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
                LineIdentifier(
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
            line_identifiers=line_identifiers[
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
    dataset_dir: Path,
    output_dir: Path,
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

    for page_input in iter_peoplegator_pages(dataset_dir):
        output_path = (
            output_dir
            / page_input.library
            / page_input.document_id
            / f"{page_input.page_id}.json"
        )
        if output_path.exists() and not force:
            skipped_pages += 1
            logger.debug("Skipping existing page output %s", output_path)
            continue
        if max_pages is not None and processed_pages >= max_pages:
            break

        logger.info(
            "Processing %s/%s/%s",
            page_input.library,
            page_input.document_id,
            page_input.page_id,
        )
        with page_input.path.open("r", encoding="utf-8") as page_file:
            page = json.load(page_file)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            page_output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        processed_pages += 1

    return LlmNerRunSummary(
        processed_pages=processed_pages,
        skipped_pages=skipped_pages,
    )


def run_llm_ner_pipeline(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    min_characters_in_chunk: int = DEFAULT_MIN_CHARACTERS_IN_CHUNK,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
    force: bool = False,
    max_pages: int | None = None,
    api_key: str | None = None,
    client: Any | None = None,
) -> LlmNerRunSummary:
    """Extract named entities from a local PeopleGator dataset.

    Existing page outputs are skipped unless ``force`` is true. Writes are
    atomic per page. When ``client`` is omitted, an OpenRouter client is made
    using ``api_key`` or ``OPENROUTER_API_KEY``.
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    if not dataset_path.is_dir():
        raise ValueError(
            f"PeopleGator dataset directory does not exist: {dataset_path}"
        )
    if min_characters_in_chunk <= 0:
        raise ValueError("min_characters_in_chunk must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")

    run_kwargs = {
        "dataset_dir": dataset_path,
        "output_dir": output_path,
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
    "DEFAULT_MIN_CHARACTERS_IN_CHUNK",
    "DEFAULT_MODEL",
    "DEFAULT_OVERLAP_LINES",
    "ChunkMetadata",
    "EntityType",
    "LineIdentifier",
    "LlmNerRunSummary",
    "NamedEntity",
    "NamedEntityMention",
    "PageChunkMetadata",
    "PageInput",
    "PageNersOutput",
    "TextChunkNerOutput",
    "TextNerOutput",
    "chunk_lines",
    "iter_peoplegator_pages",
    "page_to_lines_and_identifiers",
    "parse_json_result",
    "process_chunk",
    "process_page",
    "process_text_lines",
    "run_llm_ner_pipeline",
]
