"""Coordinate bounded decisions and ordered commits."""

from __future__ import annotations

import asyncio
import logging
import json
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal
import httpx
from people_gator_common.filesystem import (
    exists as storage_exists,
    read_text as storage_read_text,
)
from people_gator_common.pipelines.llm_ner import PageInput, ReasoningEffort
from people_gator_common.pipelines.ner_clusters import (
    DisambiguatedNamedEntity,
    PageNerClusters,
    NerClustersDump,
    load_page_ner_clusters,
)
from people_gator_common.progress import LoggingProgress
from .api import _link_application_complete, apply_linked_entity_result
from .journal import _LinkPage, _MutationJournal, _link_output_path
from .llm import llm_decide_entity_link_async
from .models import (
    EntityCandidate,
    DocumentInput,
    LinkableEntity,
    LinkedEntityResult,
    PageNerEntityLinks,
    _utc_now,
)
from .source import (
    document_input_digest,
    document_source_clusters,
    source_entity_from_cluster,
)
from .transport import lookup_page_source, search_entity_candidates

logger = logging.getLogger(__name__)


def iter_page_source_clusters(
    page: PageNerClusters,
) -> Iterator[DisambiguatedNamedEntity]:
    """Flatten model-call groups while preserving saved cluster order."""
    for output in page.clusters:
        yield from output.clusters


@dataclass
class _QueuedLink:
    page: _LinkPage
    index: int
    cluster: DisambiguatedNamedEntity
    task: asyncio.Task[_PreparedLink] | None = None
    candidates: list[EntityCandidate] = dataclass_field(default_factory=list)


@dataclass
class _PreparedLink:
    link: LinkedEntityResult
    revision: int


@dataclass
class _RunCounts:
    processed_pages: int = 0
    source_entities: int = 0
    merge_decisions: int = 0
    create_decisions: int = 0
    entity_mutations: int = 0
    uploaded_mentions: int = 0
    processed_documents: int = 0
    processed_units: int = 0


class _CandidateReservations:
    """Reference counts for candidates held by speculative decisions."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def acquire(self, job: _QueuedLink, candidates: list[EntityCandidate]) -> None:
        job.candidates = candidates
        for candidate in candidates:
            key = candidate.entity_id
            self.counts[key] = self.counts.get(key, 0) + 1

    def release(self, job: _QueuedLink) -> None:
        for candidate in job.candidates:
            key = candidate.entity_id
            count = self.counts[key] - 1
            if count:
                self.counts[key] = count
            else:
                del self.counts[key]
        job.candidates = []

    def contains(self, job: _QueuedLink, entity_id: str) -> bool:
        if entity_id not in self.counts:
            return False
        return any(candidate.entity_id == entity_id for candidate in job.candidates)


class LinkRun:
    """Bounded speculative LLM work with one ordered API commit stream."""

    def __init__(
        self,
        *,
        pages: list[PageInput | DocumentInput],
        resumable: dict[str, PageNerEntityLinks],
        output_prefix: str,
        api_client: httpx.AsyncClient,
        headers: dict[str, str],
        llm_client: Any,
        model: str,
        reasoning_effort: ReasoningEffort,
        mode: Literal["decision-only", "apply"],
        force: bool,
        parallel_entities: int,
        progress_interval_seconds: float,
    ) -> None:
        self.pages = pages
        self.resumable = resumable
        self.output_prefix = output_prefix
        self.api_client = api_client
        self.headers = headers
        self.llm_client = llm_client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.mode = mode
        self.force = force
        self.parallel_entities = parallel_entities
        self.progress_interval_seconds = progress_interval_seconds
        self.journal = _MutationJournal(output_prefix)
        self.api_lock = asyncio.Lock()
        self.llm_semaphore = asyncio.Semaphore(parallel_entities)
        self.revision = 0
        self.reservations = _CandidateReservations()
        self.created_entities: list[EntityCandidate] = []
        self.counts = _RunCounts()
        self.pending: deque[_QueuedLink] = deque()

    def _load_pages(self) -> Iterator[_LinkPage]:
        for page_input in self.pages:
            path = _link_output_path(self.output_prefix, page_input)
            if self.force and storage_exists(path):
                previous = PageNerEntityLinks.model_validate_json(
                    storage_read_text(path)
                )
                if any(
                    m.rollback_status != "rolled_back"
                    for link in previous.links
                    for m in link.mutations
                ):
                    raise RuntimeError(
                        f"Refusing to overwrite mutation journal {path}; "
                        "roll it back first or use a different output prefix"
                    )
            logger.info("Loading cluster input %s", page_input.path)
            content = storage_read_text(page_input.path)
            payload = json.loads(content)
            dump = (
                NerClustersDump.model_validate(payload)
                if "schema_version" in payload
                else None
            )
            if isinstance(page_input, DocumentInput):
                if dump is None:
                    raise ValueError(
                        f"Document linking requires a scoped cluster dump: {page_input.path}"
                    )
                clusters = document_source_clusters(dump)
                source = dump
            else:
                source = load_page_ner_clusters(content)
                clusters = list(iter_page_source_clusters(source))
            if (source.library, source.document_id, source.page_id) != (
                page_input.library,
                page_input.document_id,
                page_input.page_id,
            ):
                raise RuntimeError(
                    f"Input identity does not match path: {page_input.path}"
                )
            scope = "document" if isinstance(page_input, DocumentInput) else "page"
            input_pages = (
                [page.page_id for page in dump.inputs]
                if dump is not None
                else [page_input.page_id]
            )
            cluster_members = (
                {str(cluster.cluster_id): cluster.members for cluster in dump.clusters}
                if dump is not None
                else {}
            )
            digest = document_input_digest(dump) if scope == "document" else None
            logger.info(
                "Loaded %s %s: %d pages, %d clusters, %d entity records",
                scope,
                page_input.path,
                len(input_pages),
                len(clusters),
                sum(len(cluster.entities or []) for cluster in clusters),
            )
            output = self.resumable.get(path) or PageNerEntityLinks(
                library=source.library,
                document_id=source.document_id,
                page_id=source.page_id,
                links=[],
                mode=self.mode,
                started_at=_utc_now(),
                scope=scope,
                input_pages=input_pages,
                input_digest=digest,
            )
            if (output.library, output.document_id, output.page_id) != (
                source.library,
                source.document_id,
                source.page_id,
            ):
                raise RuntimeError(f"Audit identity does not match input page: {path}")
            if output.scope != scope:
                raise RuntimeError(f"Audit scope does not match input: {path}")
            if scope == "document" and (
                output.input_digest != digest or output.input_pages != input_pages
            ):
                raise RuntimeError(
                    f"Document evidence no longer matches journal: {path}"
                )
            if len(output.links) > len(clusters):
                raise RuntimeError(
                    f"Audit has more decisions than input clusters: {path}"
                )
            for index, link in enumerate(output.links):
                if link.source_cluster != clusters[index]:
                    raise RuntimeError(
                        f"Saved decision {index} does not match input cluster in {path}"
                    )
                expected_members = cluster_members.get(
                    str(clusters[index].cluster_id), []
                )
                if (
                    scope == "document" or link.members
                ) and link.members != expected_members:
                    raise RuntimeError(
                        f"Saved member provenance no longer matches input cluster in {path}"
                    )
                if (
                    scope == "document"
                    and link.cluster_id != clusters[index].cluster_id
                ):
                    raise RuntimeError(
                        f"Saved cluster ID no longer matches input in {path}"
                    )
                if not link.members and scope == "page":
                    link.members = expected_members
                    link.cluster_id = clusters[index].cluster_id
            yield _LinkPage(
                page_input, path, clusters, output, cluster_members=cluster_members
            )

    def _complete_unit(self, page: _LinkPage) -> None:
        page.output.completed_at = _utc_now()
        self.journal.checkpoint(page)
        self.counts.processed_units += 1
        self.counts.processed_pages += len(page.output.input_pages) or 1
        if page.output.scope == "document":
            self.counts.processed_documents += 1
        logger.info(
            "Saved completed %s link journal %s: %d cluster decisions, %d pages",
            page.output.scope,
            page.path,
            len(page.output.links),
            len(page.output.input_pages) or 1,
        )

    def _entities(self) -> Iterator[_QueuedLink]:
        for page in self._load_pages():
            if not page.clusters:
                self._complete_unit(page)
            for index, cluster in enumerate(page.clusters):
                yield _QueuedLink(page, index, cluster)

    async def _search(self, source: LinkableEntity) -> list[EntityCandidate]:
        """Add same-run creates while the remote search index catches up."""
        logger.info(
            "Searching entity candidates for %s (%d aliases)",
            source.name,
            len(source.aliases),
        )
        candidates = await search_entity_candidates(
            self.api_client, self.headers, source
        )
        seen = {candidate.entity_id for candidate in candidates}
        names = {name.casefold() for name in [source.name, *source.aliases]}
        for candidate in self.created_entities:
            if candidate.entity_id in seen:
                continue
            if names.intersection(
                name.casefold() for name in [candidate.name, *candidate.aliases]
            ):
                candidates.append(candidate)
        logger.info("Found %d candidate entities for %s", len(candidates), source.name)
        return candidates

    async def _decide(
        self,
        cluster: DisambiguatedNamedEntity,
        candidates: list[EntityCandidate],
    ) -> LinkedEntityResult:
        source = source_entity_from_cluster(cluster)
        async with self.llm_semaphore:
            logger.info(
                "Resolving cluster %s (%s): %d candidates, model=%s",
                cluster.cluster_id or "legacy",
                source.name,
                len(candidates),
                self.model,
            )
            decision, llm = await llm_decide_entity_link_async(
                source,
                candidates,
                client=self.llm_client,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
        selected_id = (
            candidates[decision.candidate_index - 1].entity_id
            if decision.action == "merge" and decision.candidate_index is not None
            else None
        )
        logger.info(
            "Decision for cluster %s (%s): %s%s",
            cluster.cluster_id or "legacy",
            source.name,
            decision.action,
            f" entity {selected_id}" if selected_id else " a new entity",
        )
        return LinkedEntityResult(
            cluster_id=cluster.cluster_id,
            source_cluster=cluster,
            source=source,
            candidates=candidates,
            decision=decision,
            selected_entity_id=selected_id,
            llm=llm,
        )

    async def _prepare(self, job: _QueuedLink) -> _PreparedLink:
        if job.index < len(job.page.output.links):
            logger.info(
                "Reusing saved decision for cluster %s from %s",
                job.cluster.cluster_id or job.index,
                job.page.path,
            )
            return _PreparedLink(job.page.output.links[job.index], self.revision)
        logger.info(
            "Preparing cluster %d/%d (%s) from %s",
            job.index + 1,
            len(job.page.clusters),
            job.cluster.name,
            job.page.input.path,
        )
        source = source_entity_from_cluster(job.cluster)
        async with self.api_lock:
            candidates = await self._search(source)
            searched_at = self.revision
            self.reservations.acquire(job, candidates)
        try:
            return _PreparedLink(
                await self._decide(job.cluster, candidates), searched_at
            )
        except BaseException:
            self.reservations.release(job)
            raise

    def _schedule(self, job: _QueuedLink) -> None:
        job.task = asyncio.create_task(self._prepare(job))

    async def _cancel_jobs(self, jobs: list[_QueuedLink]) -> None:
        for job in jobs:
            assert job.task is not None
            job.task.cancel()
        await asyncio.gather(*(job.task for job in jobs), return_exceptions=True)
        for job in jobs:
            self.reservations.release(job)

    async def _invalidate(self, entity_id: str, *, created: bool) -> None:
        """Restart pending work that can include the changed candidate."""
        affected = [
            job
            for job in self.pending
            if created or self.reservations.contains(job, entity_id)
        ]
        if affected:
            await self._cancel_jobs(affected)
            for job in affected:
                self._schedule(job)

    async def _apply(self, job: _QueuedLink, link: LinkedEntityResult) -> None:
        page = job.page
        output = page.output
        async with self.api_lock:
            if output.scope == "document":
                for page_id in dict.fromkeys(member.page_id for member in link.members):
                    if page_id in output.page_source_ids:
                        continue
                    logger.info(
                        "Resolving page source %s/%s/%s for cluster %s",
                        page.input.library,
                        page.input.document_id,
                        page_id,
                        link.cluster_id,
                    )
                    output.page_source_ids[page_id] = await lookup_page_source(
                        self.api_client,
                        self.headers,
                        library=page.input.library,
                        document_id=page.input.document_id,
                        page_id=page_id,
                    )
                    self.journal.checkpoint(page)
            elif output.source_id is None:
                logger.info(
                    "Resolving page source %s/%s/%s",
                    page.input.library,
                    page.input.document_id,
                    page.input.page_id,
                )
                output.source_id = await lookup_page_source(
                    self.api_client,
                    self.headers,
                    library=page.input.library,
                    document_id=page.input.document_id,
                    page_id=page.input.page_id,
                )
                self.journal.checkpoint(page)
            if output.scope == "page":
                assert output.source_id is not None
            page_sources = (
                output.page_source_ids if output.scope == "document" else None
            )
            changes = 0
            if not _link_application_complete(
                link, source_id=output.source_id, page_source_ids=page_sources
            ):
                before = len(link.mutations)

                async def checkpoint() -> None:
                    self.journal.checkpoint(page)

                await apply_linked_entity_result(
                    link,
                    source_id=output.source_id,
                    page_source_ids=page_sources,
                    api_client=self.api_client,
                    headers=self.headers,
                    checkpoint=checkpoint,
                )
                new = link.mutations[before:]
                changes = sum(m.operation != "create_mention" for m in new)
                self.counts.entity_mutations += changes
                self.counts.uploaded_mentions += len(new) - changes
                if changes:
                    self.revision += 1
                    assert link.selected_entity_id is not None
                    await self._invalidate(
                        link.selected_entity_id,
                        created=link.decision.action == "create",
                    )
            if (
                link.decision.action == "create"
                and link.selected_entity_after is not None
            ):
                entity = link.selected_entity_after
                if all(
                    item.entity_id != entity.entity_id for item in self.created_entities
                ):
                    self.created_entities.append(
                        EntityCandidate(
                            entity_id=entity.entity_id,
                            entity_type=entity.entity_type,
                            name=entity.name,
                            aliases=entity.aliases or [],
                            short_description=entity.short_description or "",
                            long_description=entity.description or "",
                            matching_score=1.0,
                        )
                    )
                    if not changes:
                        self.revision += 1
                        await self._invalidate(entity.entity_id, created=True)

    async def _commit(self, job: _QueuedLink, prepared: _PreparedLink) -> None:
        page = job.page
        output = page.output
        link = prepared.link
        saved = job.index < len(output.links)
        applied = saved and bool(link.mutations or link.selected_entity_after)
        if self.mode == "apply" and not saved and prepared.revision != self.revision:
            # Search ranking and result contents can change after any mutation.
            async with self.api_lock:
                fresh = await self._search(link.source)
            if fresh != link.candidates:
                logger.info(
                    "Candidates changed for %s; repeating LLM decision",
                    link.source.name,
                )
                link = await self._decide(job.cluster, fresh)
        if not saved:
            link.members = page.cluster_members.get(str(job.cluster.cluster_id), [])
            output.links.append(link)
            self.journal.checkpoint(page)
            logger.info(
                "Saved cluster %s decision to %s",
                link.cluster_id or job.index,
                page.path,
            )
        elif not applied:
            output.links[job.index] = link
            self.journal.checkpoint(page)

        self.counts.source_entities += 1
        if link.decision.action == "merge":
            self.counts.merge_decisions += 1
        else:
            self.counts.create_decisions += 1
        if self.mode == "apply":
            await self._apply(job, link)
        page.completed += 1
        if page.completed == len(page.clusters):
            self._complete_unit(page)

    async def run(self) -> _RunCounts:
        stream = iter(self._entities())
        with LoggingProgress(
            logger,
            total=len(self.pages),
            description="NER entity linking",
            unit=(
                "documents"
                if self.pages and isinstance(self.pages[0], DocumentInput)
                else "document-pages"
            ),
            min_interval_seconds=self.progress_interval_seconds,
        ) as progress:
            reported_units = 0
            try:
                while True:
                    while len(self.pending) < self.parallel_entities:
                        try:
                            job = next(stream)
                        except StopIteration:
                            break
                        self._schedule(job)
                        self.pending.append(job)
                    if self.counts.processed_units > reported_units:
                        progress.update(self.counts.processed_units - reported_units)
                        reported_units = self.counts.processed_units
                    if not self.pending:
                        break
                    job = self.pending.popleft()
                    assert job.task is not None
                    prepared = await job.task
                    try:
                        await self._commit(job, prepared)
                    finally:
                        self.reservations.release(job)
                    if self.counts.processed_units > reported_units:
                        progress.update(self.counts.processed_units - reported_units)
                        reported_units = self.counts.processed_units
            except BaseException:
                await self._cancel_jobs(list(self.pending))
                raise
        return self.counts
