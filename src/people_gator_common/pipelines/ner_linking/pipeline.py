"""Storage selection and public NER entity-linking entry point."""

from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Iterator
from typing import Any, Literal
import httpx
from openrouter import OpenRouter
from people_gator_common.filesystem import (
    glob as storage_glob,
    read_text as storage_read_text,
)
from people_gator_common.pipelines.llm_ner import PageInput, ReasoningEffort
from people_gator_common.pipelines.ner_clusters import DEFAULT_MODEL
from .journal import _link_output_path
from .models import (
    DEFAULT_INPUT_PREFIX,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    EntityLinkingConfig,
    NerEntityLinkingRunSummary,
    PageNerEntityLinks,
)
from .run import LinkRun
from .transport import authenticate_entity_linking_api

logger = logging.getLogger(__name__)


def _iter_cluster_pages(input_prefix: str) -> Iterator[PageInput]:
    input_prefix = posixpath.normpath(input_prefix)
    pattern = posixpath.join(input_prefix, "*", "*", "*.json")
    page_paths = sorted(storage_glob(pattern))
    if not page_paths:
        raise ValueError(
            f"No NER cluster outputs found below storage prefix: {input_prefix}"
        )

    for page_path in page_paths:
        relative_path = posixpath.relpath(page_path, input_prefix)
        parts = relative_path.split("/")
        if len(parts) != 3 or not parts[2].endswith(".json"):
            logger.warning("Ignoring unexpected NER cluster key: %s", page_path)
            continue
        yield PageInput(
            library=parts[0],
            document_id=parts[1],
            page_id=parts[2].removesuffix(".json"),
            path=page_path,
        )


def _select_link_pages(
    input_prefix: str,
    output_prefix: str,
    *,
    force: bool,
    max_pages: int | None,
    mode: Literal["decision-only", "apply"],
) -> tuple[list[PageInput], dict[str, PageNerEntityLinks], int]:
    existing = (
        set(storage_glob(posixpath.join(output_prefix, "*", "*", "*.json")))
        if not force
        else set()
    )
    selected: list[PageInput] = []
    resumable: dict[str, PageNerEntityLinks] = {}
    skipped = 0
    for page in _iter_cluster_pages(input_prefix):
        if max_pages is not None and len(selected) >= max_pages:
            break
        path = _link_output_path(output_prefix, page)
        if path in existing:
            previous = PageNerEntityLinks.model_validate_json(storage_read_text(path))
            if (
                mode == "apply"
                and previous.mode == "apply"
                and previous.completed_at is None
                and previous.rolled_back_at is None
            ):
                resumable[path] = previous
            else:
                skipped += 1
                continue
        selected.append(page)
    return selected, resumable, skipped


async def _run_with_clients(
    *,
    input_prefix: str,
    output_prefix: str,
    api_client: httpx.AsyncClient,
    llm_client: Any,
    config: EntityLinkingConfig,
    model: str,
    reasoning_effort: ReasoningEffort,
    force: bool,
    max_pages: int | None,
    progress_interval_seconds: float,
    mode: Literal["decision-only", "apply"],
    parallel_pages: int,
) -> NerEntityLinkingRunSummary:
    """Run concurrent decisions and apply them through one ordered commit stream."""
    scopes = ["people-gator:browse"]
    if mode == "apply":
        scopes.extend(["people-gator:annotate", "people-gator:ingest"])
    token = await authenticate_entity_linking_api(api_client, config, scopes=scopes)
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    pages, resumable, skipped = _select_link_pages(
        input_prefix, output_prefix, force=force, max_pages=max_pages, mode=mode
    )
    counts = await LinkRun(
        pages=pages,
        resumable=resumable,
        output_prefix=output_prefix,
        api_client=api_client,
        headers=headers,
        llm_client=llm_client,
        model=model,
        reasoning_effort=reasoning_effort,
        mode=mode,
        force=force,
        parallel_entities=parallel_pages,
        progress_interval_seconds=progress_interval_seconds,
    ).run()
    return NerEntityLinkingRunSummary(
        processed_pages=counts.processed_pages,
        skipped_pages=skipped,
        source_entities=counts.source_entities,
        merge_decisions=counts.merge_decisions,
        create_decisions=counts.create_decisions,
        entity_mutations=counts.entity_mutations,
        uploaded_mentions=counts.uploaded_mentions,
    )


async def run_ner_entity_linking_pipeline(
    *,
    config: EntityLinkingConfig,
    input_prefix: str | os.PathLike[str] = DEFAULT_INPUT_PREFIX,
    output_prefix: str | os.PathLike[str] = DEFAULT_OUTPUT_PREFIX,
    model: str = DEFAULT_MODEL,
    reasoning_effort: ReasoningEffort = "low",
    force: bool = False,
    max_pages: int | None = None,
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    openrouter_api_key: str | None = None,
    api_client: httpx.AsyncClient | None = None,
    llm_client: Any | None = None,
    mode: Literal["decision-only", "apply"] = "apply",
    parallel_pages: int = 4,
) -> NerEntityLinkingRunSummary:
    """Resolve NER clusters and optionally apply the resulting API mutations."""
    normalized_input = posixpath.normpath(os.fspath(input_prefix))
    normalized_output = posixpath.normpath(os.fspath(output_prefix))
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")
    if progress_interval_seconds < 0:
        raise ValueError("progress_interval_seconds must be non-negative")
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if mode not in ("decision-only", "apply"):
        raise ValueError("mode must be 'decision-only' or 'apply'")
    if parallel_pages < 1:
        raise ValueError("parallel_pages must be at least 1")

    resolved_openrouter_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    if llm_client is None and not resolved_openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is required for NER entity linking")

    async def run(api: httpx.AsyncClient, llm: Any) -> NerEntityLinkingRunSummary:
        return await _run_with_clients(
            input_prefix=normalized_input,
            output_prefix=normalized_output,
            api_client=api,
            llm_client=llm,
            config=config,
            model=model,
            reasoning_effort=reasoning_effort,
            force=force,
            max_pages=max_pages,
            progress_interval_seconds=progress_interval_seconds,
            mode=mode,
            parallel_pages=parallel_pages,
        )

    if api_client is not None and llm_client is not None:
        return await run(api_client, llm_client)
    if api_client is not None:
        async with OpenRouter(api_key=resolved_openrouter_key) as created_llm:
            return await run(api_client, created_llm)
    if llm_client is not None:
        timeout = httpx.Timeout(config.timeout_seconds)
        async with httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"), timeout=timeout
        ) as created_api:
            return await run(created_api, llm_client)

    timeout = httpx.Timeout(config.timeout_seconds)
    async with httpx.AsyncClient(
        base_url=config.base_url.rstrip("/"), timeout=timeout
    ) as created_api:
        async with OpenRouter(api_key=resolved_openrouter_key) as created_llm:
            return await run(created_api, created_llm)
