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
from people_gator_common.pipelines.ner_clusters import DEFAULT_MODEL, NerClustersDump
from .journal import _link_output_path
from .models import (
    DEFAULT_INPUT_PREFIX,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_DOCUMENT_INPUT_PREFIX,
    DEFAULT_DOCUMENT_OUTPUT_PREFIX,
    DEFAULT_PROGRESS_INTERVAL_SECONDS,
    EntityLinkingConfig,
    NerEntityLinkingRunSummary,
    PageNerEntityLinks,
    DocumentInput,
)
from .run import LinkRun
from .source import document_input_digest
from .transport import _authenticate_headers

logger = logging.getLogger(__name__)


def _iter_cluster_pages(input_prefix: str) -> Iterator[PageInput]:
    input_prefix = posixpath.normpath(input_prefix)
    pattern = posixpath.join(input_prefix, "*", "*", "*.json")
    logger.info("Listing page cluster dumps using %s", pattern)
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


def _iter_cluster_documents(input_prefix: str) -> Iterator[DocumentInput]:
    input_prefix = posixpath.normpath(input_prefix)
    pattern = posixpath.join(input_prefix, "*", "*.json")
    logger.info("Listing document cluster dumps using %s", pattern)
    paths = sorted(storage_glob(pattern))
    if not paths:
        raise ValueError(
            f"No document cluster dumps found below storage prefix: {input_prefix}"
        )
    for path in paths:
        parts = posixpath.relpath(path, input_prefix).split("/")
        if len(parts) != 2 or not parts[1].endswith(".json"):
            logger.warning("Ignoring unexpected document cluster key: %s", path)
            continue
        yield DocumentInput(
            library=parts[0], document_id=parts[1].removesuffix(".json"), path=path
        )


def _select_link_documents(
    input_prefix: str,
    output_prefix: str,
    *,
    force: bool,
    max_documents: int | None,
    mode: Literal["decision-only", "apply"],
) -> tuple[list[DocumentInput], dict[str, PageNerEntityLinks], int, int]:
    selected: list[DocumentInput] = []
    resumable: dict[str, PageNerEntityLinks] = {}
    skipped_pages = skipped_documents = 0
    for document in _iter_cluster_documents(input_prefix):
        if max_documents is not None and len(selected) >= max_documents:
            break
        logger.info("Loading document cluster dump %s", document.path)
        source = NerClustersDump.model_validate_json(storage_read_text(document.path))
        if (source.scope, source.library, source.document_id) != (
            "document",
            document.library,
            document.document_id,
        ):
            raise ValueError(
                f"Document input identity does not match path: {document.path}"
            )
        logger.info(
            "Loaded document %s/%s: %d pages, %d clusters, %d entity records",
            source.library,
            source.document_id,
            len(source.inputs),
            len(source.clusters),
            sum(len(cluster.members) for cluster in source.clusters),
        )
        path = _link_output_path(output_prefix, document)
        previous = None
        if not force:
            try:
                previous = PageNerEntityLinks.model_validate_json(
                    storage_read_text(path)
                )
            except FileNotFoundError:
                pass
        if previous is not None:
            if (
                previous.scope,
                previous.library,
                previous.document_id,
                previous.page_id,
            ) != ("document", source.library, source.document_id, None):
                raise ValueError(
                    f"Document journal identity does not match input: {path}"
                )
            if previous.input_digest != document_input_digest(source):
                raise ValueError(
                    f"Document cluster evidence changed since {path}; "
                    "use a new output prefix or roll back the existing journal"
                )
            if (
                previous.mode == mode
                and previous.completed_at is None
                and previous.rolled_back_at is None
            ):
                resumable[path] = previous
                logger.info(
                    "Resuming document journal %s: %d saved cluster decisions",
                    path,
                    len(previous.links),
                )
            else:
                skipped_documents += 1
                skipped_pages += len(source.inputs)
                logger.info(
                    "Skipping existing document journal %s (mode=%s, complete=%s, rolled_back=%s)",
                    path,
                    previous.mode,
                    previous.completed_at is not None,
                    previous.rolled_back_at is not None,
                )
                continue
        selected.append(document)
    logger.info(
        "Selected %d documents (%d resumable); skipped %d documents covering %d pages",
        len(selected),
        len(resumable),
        skipped_documents,
        skipped_pages,
    )
    return selected, resumable, skipped_pages, skipped_documents


def _select_link_pages(
    input_prefix: str,
    output_prefix: str,
    *,
    force: bool,
    max_pages: int | None,
    mode: Literal["decision-only", "apply"],
) -> tuple[list[PageInput], dict[str, PageNerEntityLinks], int]:
    selected: list[PageInput] = []
    resumable: dict[str, PageNerEntityLinks] = {}
    skipped = 0
    for page in _iter_cluster_pages(input_prefix):
        if max_pages is not None and len(selected) >= max_pages:
            break
        path = _link_output_path(output_prefix, page)
        logger.info("Checking page link journal %s for input %s", path, page.path)
        previous = None
        if not force:
            # Read the exact checkpoint rather than relying on directory-listing
            # caches or the path spelling returned by an object-store glob.
            try:
                content = storage_read_text(path)
            except FileNotFoundError:
                pass
            else:
                previous = PageNerEntityLinks.model_validate_json(content)
        if previous is not None:
            if (
                previous.mode == mode
                and previous.completed_at is None
                and previous.rolled_back_at is None
            ):
                resumable[path] = previous
                logger.info(
                    "Resuming %s from checkpoint: %d saved decisions",
                    path,
                    len(previous.links),
                )
            else:
                skipped += 1
                logger.info("Skipping existing page link journal %s", path)
                continue
        selected.append(page)
    logger.info(
        "Selected %d pages (%d resumable); skipped %d existing pages",
        len(selected),
        len(resumable),
        skipped,
    )
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
    scope: Literal["page", "document"],
    max_documents: int | None,
) -> NerEntityLinkingRunSummary:
    """Run concurrent decisions and apply them through one ordered commit stream."""
    scopes = ["people-gator:browse"]
    if mode == "apply":
        scopes.extend(["people-gator:annotate", "people-gator:ingest"])
    headers = await _authenticate_headers(api_client, config, scopes=scopes)
    skipped_documents = 0
    pages: list[PageInput | DocumentInput]
    if scope == "document":
        pages, resumable, skipped, skipped_documents = _select_link_documents(
            input_prefix,
            output_prefix,
            force=force,
            max_documents=max_documents,
            mode=mode,
        )
    else:
        pages, resumable, skipped = _select_link_pages(
            input_prefix, output_prefix, force=force, max_pages=max_pages, mode=mode
        )
    logger.info(
        "Starting %s entity linking: mode=%s, model=%s, parallel entities=%d, output=%s",
        scope,
        mode,
        model,
        parallel_pages,
        output_prefix,
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
        processed_documents=counts.processed_documents,
        skipped_documents=skipped_documents,
    )


async def run_ner_entity_linking_pipeline(
    *,
    config: EntityLinkingConfig,
    input_prefix: str | os.PathLike[str] | None = None,
    output_prefix: str | os.PathLike[str] | None = None,
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
    scope: Literal["page", "document"] = "page",
    max_documents: int | None = None,
) -> NerEntityLinkingRunSummary:
    """Resolve NER clusters and optionally apply the resulting API mutations."""
    if scope not in ("page", "document"):
        raise ValueError("scope must be 'page' or 'document'")
    if input_prefix is None:
        input_prefix = (
            DEFAULT_DOCUMENT_INPUT_PREFIX
            if scope == "document"
            else DEFAULT_INPUT_PREFIX
        )
    if output_prefix is None:
        output_prefix = (
            DEFAULT_DOCUMENT_OUTPUT_PREFIX
            if scope == "document"
            else DEFAULT_OUTPUT_PREFIX
        )
    normalized_input = posixpath.normpath(os.fspath(input_prefix))
    normalized_output = posixpath.normpath(os.fspath(output_prefix))
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when provided")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")
    if scope == "document" and max_pages is not None:
        raise ValueError("max_pages requires page scope; use max_documents")
    if scope == "page" and max_documents is not None:
        raise ValueError("max_documents requires document scope")
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
            scope=scope,
            max_documents=max_documents,
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
