from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid4, uuid5

from fsspec.implementations.local import LocalFileSystem
import httpx
import pytest

import people_gator_common.filesystem as storage
from people_gator_common.cli import ner_entity_linking_pipeline_cli as cli
from people_gator_common.pipelines.llm_ner import (
    LLMBase,
    NamedEntity,
    NamedEntityMention,
)
from people_gator_common.pipelines.ner_clusters import (
    ClusterInputPage,
    ClusterMember,
    NerClustersDump,
    ResolvedEntityCluster,
)
from people_gator_common.pipelines.ner_entity_linking import (
    EntityLinkingApiError,
    EntityLinkingConfig,
    PageNerEntityLinks,
    rollback_latest_entity_linking_run,
    run_ner_entity_linking_pipeline,
)


def uid(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, value))


DOCUMENT = uid("document")
PAGES = [uid("page-a"), uid("page-b")]


@pytest.fixture
def local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(storage, "fs", LocalFileSystem())
    monkeypatch.setattr(storage, "prefix", str(tmp_path))
    return tmp_path


def dump_document(document_id=DOCUMENT, *, empty=False):
    chunk_id = uuid5(NAMESPACE_URL, "identical-ocr")
    cluster = ResolvedEntityCluster(
        cluster_id=uuid5(NAMESPACE_URL, f"cluster/{document_id}"),
        name="Ada Lovelace",
        short_description="English mathematician and Analytical Engine author.",
        long_description="Ada Lovelace was an English mathematician who wrote about the Analytical Engine.",
        members=[
            ClusterMember(
                record_id=uuid5(NAMESPACE_URL, f"record/{document_id}/{page_id}"),
                page_id=page_id,
                chunk_id=chunk_id,
                chunk_index=0,
                entity_index=0,
                entity=NamedEntity(
                    name="Ada Lovelace",
                    type="beeing",
                    info=information,
                    mentions=[
                        NamedEntityMention(
                            name="Ada Lovelace",
                            context="Ada Lovelace appeared here.",
                            chunk_id=chunk_id,
                        )
                    ],
                ),
            )
            for page_id, information in zip(
                PAGES,
                ["English mathematician.", "Analytical Engine author."],
                strict=True,
            )
        ],
        disambiguation_index=0,
        entity_indices=[1, 2],
    )
    dump = NerClustersDump(
        scope="document",
        library="mzk",
        document_id=document_id,
        inputs=[
            ClusterInputPage(
                page_id=page_id,
                path=f"page-named-entities/mzk/{document_id}/{page_id}.json",
            )
            for page_id in PAGES
        ],
        input_fingerprint="synthetic-inputs",
        run_id=uuid4(),
        clusters=[] if empty else [cluster],
        disambiguations=(
            []
            if empty
            else [
                LLMBase(
                    text="",
                    prompt="Synthetic clustering audit.",
                    reasoning="",
                    raw_output="",
                )
            ]
        ),
    )
    storage.write_text(
        f"document-named-entities-clusters/mzk/{document_id}.json",
        dump.model_dump_json(),
    )
    return dump


def journal(document_id=DOCUMENT):
    return PageNerEntityLinks.model_validate_json(
        storage.read_text(f"document-named-entities-links/mzk/{document_id}.json")
    )


class Decider:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        evidence = json.loads(prompt.split("Resolution input:\n", 1)[1])
        candidates = evidence["candidates"]
        decision = {
            "action": "merge" if candidates else "create",
            "candidate_index": 1 if candidates else None,
            "short_description": "Anglická matematička a autorka poznámek k analytickému stroji.",
            "long_description": "Ada Lovelace byla anglická matematička a psala o analytickém stroji.",
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(decision), reasoning="")
                )
            ]
        )


class Api:
    def __init__(self, *, existing=False):
        self.entities = {}
        self.mentions = {}
        self.lookups = []
        self.requests = []
        self.created = 0
        self.patched = 0
        self.failed_page = None
        self.failed = False
        self.existing = existing
        if existing:
            entity_id = uid("existing-entity")
            self.entities[entity_id] = {
                "entity_id": entity_id,
                "entity_type": "person",
                "name": "Ada Lovelace",
                "aliases": [],
                "description": "English mathematician.",
                "short_description": "Mathematician.",
            }

    def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        method, path = request.method, request.url.path
        if method == "GET" and path == "/api/v1/search-entities":
            hits = (
                [{"entity_id": entity_id, "score": 0.95} for entity_id in self.entities]
                if self.existing
                else []
            )
            return httpx.Response(200, json={"items": hits})
        if method == "GET" and path == "/api/v1/pages/lookup":
            page_id = request.url.params["page_id"]
            self.lookups.append(page_id)
            return httpx.Response(
                200,
                json={
                    "library": request.url.params["library"],
                    "document_id": request.url.params["document_id"],
                    "page_id": page_id,
                    "source_id": uid(
                        f"source/{request.url.params['document_id']}/{page_id}"
                    ),
                },
            )
        if method == "POST" and path == "/api/v1/entities":
            self.created += 1
            entity_id = uid(f"created/{self.created}")
            self.entities[entity_id] = {**body, "entity_id": entity_id}
            return httpx.Response(201, json=self.entities[entity_id])
        if method == "POST" and path == "/api/v1/text-mentions":
            if body["source_id"] == self.failed_page and not self.failed:
                self.failed = True
                return httpx.Response(503, json={"detail": "Temporary mention failure"})
            mention_id = uid(f"mention/{len(self.mentions)}")
            self.mentions[mention_id] = {**body, "mention_id": mention_id}
            return httpx.Response(201, json=self.mentions[mention_id])
        if path.startswith("/api/v1/entities/"):
            entity_id = path.rsplit("/", 1)[1]
            if method == "GET":
                return httpx.Response(200, json=self.entities[entity_id])
            if method == "PATCH":
                self.patched += 1
                self.entities[entity_id].update(body)
                return httpx.Response(200, json=self.entities[entity_id])
            if method == "DELETE":
                assert not any(
                    mention["entity_id"] == entity_id
                    for mention in self.mentions.values()
                )
                del self.entities[entity_id]
                return httpx.Response(204)
        if method == "DELETE" and path.startswith("/api/v1/text-mentions/"):
            del self.mentions[path.rsplit("/", 1)[1]]
            return httpx.Response(204)
        raise AssertionError(f"Unexpected request: {method} {path}")


def run_linker(api, decider, **kwargs):
    async def run():
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(api.handle)
        ) as client:
            return await run_ner_entity_linking_pipeline(
                config=EntityLinkingConfig(bearer_token="test-token"),
                api_client=client,
                llm_client=decider,
                scope="document",
                **kwargs,
            )

    return asyncio.run(run())


def test_document_cluster_is_linked_once_and_equal_mentions_stay_on_exact_pages(
    local_storage,
):
    dump = dump_document()
    api, decider = Api(), Decider()

    summary = run_linker(api, decider)

    assert summary.processed_documents == 1
    assert summary.processed_pages == 2
    assert summary.source_entities == 1
    assert summary.entity_mutations == 1
    assert summary.uploaded_mentions == 2
    assert len(decider.prompts) == 1
    assert "English mathematician" in decider.prompts[0]
    assert "Analytical Engine" in decider.prompts[0]
    assert api.created == 1
    assert len(api.mentions) == 2
    assert api.lookups == PAGES
    saved = journal()
    assert saved.scope == "document"
    assert saved.page_id is None
    assert saved.source_id is None
    assert saved.input_pages == PAGES
    assert saved.page_source_ids == {
        page: uid(f"source/{DOCUMENT}/{page}") for page in PAGES
    }
    assert len(saved.links) == 1
    link = saved.links[0]
    assert link.cluster_id == dump.clusters[0].cluster_id
    assert link.source_cluster.cluster_id == link.cluster_id
    assert link.members == dump.clusters[0].members
    assert {mention["entity_id"] for mention in api.mentions.values()} == {
        link.selected_entity_id
    }
    assert {mention["source_id"] for mention in api.mentions.values()} == set(
        saved.page_source_ids.values()
    )
    assert all(
        mention["entity_link_state"] == "automatic" for mention in api.mentions.values()
    )
    assert saved.completed_at is not None
    assert sorted(storage.glob("document-named-entities-links/**/*.json")) == [
        f"document-named-entities-links/mzk/{DOCUMENT}.json"
    ]

    skipped = run_linker(api, decider)
    assert skipped.skipped_documents == 1
    assert skipped.skipped_pages == 2
    assert skipped.processed_documents == 0
    assert len(decider.prompts) == 1
    assert api.created == 1
    with pytest.raises(RuntimeError, match="Refusing to overwrite mutation journal"):
        run_linker(api, decider, force=True)
    assert api.created == 1


@pytest.mark.parametrize("decision_only", [False, True])
def test_document_merge_uses_existing_entity_for_every_page(
    local_storage, decision_only
):
    dump_document()
    api, decider = Api(existing=True), Decider()

    summary = run_linker(
        api, decider, mode="decision-only" if decision_only else "apply"
    )

    assert summary.merge_decisions == 1
    assert summary.create_decisions == 0
    assert len(decider.prompts) == 1
    assert api.created == 0
    assert journal().links[0].selected_entity_id == uid("existing-entity")
    if decision_only:
        assert api.patched == 0
        assert not api.mentions
        assert not api.lookups
        assert not journal().page_source_ids
        assert all(method == "GET" for method, _, _ in api.requests)
    else:
        assert api.patched == 1
        assert len(api.mentions) == 2
        assert {mention["entity_id"] for mention in api.mentions.values()} == {
            uid("existing-entity")
        }


def test_document_resume_reuses_decision_entity_and_completed_page_mentions(
    local_storage,
):
    dump_document()
    api, decider = Api(), Decider()
    api.failed_page = uid(f"source/{DOCUMENT}/{PAGES[1]}")
    with pytest.raises(EntityLinkingApiError):
        run_linker(api, decider)
    prior = journal()
    assert prior.completed_at is None
    assert len(prior.links[0].mutations) == 2
    assert len(api.mentions) == 1
    assert api.created == 1

    summary = run_linker(api, decider)

    assert summary.processed_documents == 1
    assert summary.processed_pages == 2
    assert summary.entity_mutations == 0
    assert summary.uploaded_mentions == 1
    assert api.created == 1
    assert len(api.mentions) == 2
    assert len(decider.prompts) == 1
    assert api.lookups == PAGES
    saved = journal()
    assert saved.completed_at is not None
    assert saved.links[0].selected_entity_id == prior.links[0].selected_entity_id
    assert len(saved.links[0].mutations) == 3
    operations = [
        json.loads(line)["operation"]
        for line in storage.read_text(
            "document-named-entities-links/modifications.jsonl"
        ).splitlines()
    ]
    assert operations == ["create_entity", "create_mention", "create_mention"]


def test_changed_document_evidence_cannot_silently_reuse_prior_journal(local_storage):
    dump = dump_document()
    api, decider = Api(), Decider()
    run_linker(api, decider)
    dump.run_id = uuid4()
    dump.input_fingerprint = "new-generation-same-evidence"
    dump.disambiguations[0].prompt = "New audit for equivalent evidence."
    path = f"document-named-entities-clusters/mzk/{DOCUMENT}.json"
    storage.write_text(path, dump.model_dump_json())
    assert run_linker(api, decider).skipped_documents == 1

    dump.clusters[0].long_description = "New evidence requiring review."
    storage.write_text(path, dump.model_dump_json())
    before = len(api.requests)
    with pytest.raises(ValueError, match="cluster evidence changed"):
        run_linker(api, decider)
    assert len(api.requests) == before
    assert api.created == 1


def test_document_rollback_removes_all_page_mentions_before_entity(local_storage):
    dump_document()
    api, decider = Api(), Decider()
    run_linker(api, decider)

    async def rollback():
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(api.handle)
        ) as client:
            return await rollback_latest_entity_linking_run(
                config=EntityLinkingConfig(bearer_token="test-token"),
                output_prefix="document-named-entities-links",
                api_client=client,
            )

    result = asyncio.run(rollback())

    assert result.audit_path == f"document-named-entities-links/mzk/{DOCUMENT}.json"
    assert result.rolled_back_mutations == 3
    assert not api.mentions
    assert not api.entities
    saved = journal()
    assert saved.rolled_back_at is not None
    assert all(
        mutation.rollback_status == "rolled_back"
        for mutation in saved.links[0].mutations
    )
    assert run_linker(api, decider, force=True).processed_documents == 1
    assert api.created == 2


def test_document_limit_counts_whole_documents_and_excludes_skipped_ones(local_storage):
    document_ids = sorted([DOCUMENT, uid("other-document")])
    for document_id in document_ids:
        dump_document(document_id)
    api, decider = Api(), Decider()

    first = run_linker(api, decider, max_documents=1, mode="decision-only")
    assert first.processed_documents == 1
    assert first.processed_pages == 2
    assert not storage.exists(
        f"document-named-entities-links/mzk/{document_ids[1]}.json"
    )

    second = run_linker(api, decider, max_documents=1, mode="decision-only")
    assert second.skipped_documents == 1
    assert second.skipped_pages == 2
    assert second.processed_documents == 1
    assert second.processed_pages == 2
    assert len(decider.prompts) == 2


def test_multiple_document_clusters_share_page_lookup_cache_but_keep_entity_assignments_separate(
    local_storage,
):
    dump = dump_document()
    ada = dump.clusters[0]
    grace = ada.model_copy(deep=True)
    grace.cluster_id = uuid5(NAMESPACE_URL, "cluster/grace")
    grace.name = "Grace Hopper"
    grace.short_description = "American computer scientist."
    grace.long_description = "Grace Hopper was an American computer scientist."
    grace.disambiguation_index = 1
    for member in grace.members:
        member.record_id = uuid5(NAMESPACE_URL, f"grace/{member.page_id}")
        member.entity_index = 1
        member.entity.name = "Grace Hopper"
        member.entity.info = grace.short_description
        member.entity.mentions[0].name = "Grace Hopper"
        member.entity.mentions[0].context = "Grace Hopper appeared here."
    dump.clusters.append(grace)
    dump.disambiguations.append(dump.disambiguations[0])
    storage.write_text(
        f"document-named-entities-clusters/mzk/{DOCUMENT}.json", dump.model_dump_json()
    )
    api, decider = Api(), Decider()

    summary = run_linker(api, decider, parallel_pages=1)

    assert summary.source_entities == 2
    assert summary.entity_mutations == 2
    assert summary.uploaded_mentions == 4
    assert api.created == 2
    assert len(decider.prompts) == 2
    assert api.lookups == PAGES
    by_name = {link.source.name: link for link in journal().links}
    assert by_name["Ada Lovelace"].cluster_id == ada.cluster_id
    assert by_name["Grace Hopper"].cluster_id == grace.cluster_id
    assert (
        by_name["Ada Lovelace"].selected_entity_id
        != by_name["Grace Hopper"].selected_entity_id
    )
    for mention in api.mentions.values():
        assert mention["entity_id"] == by_name[mention["name"]].selected_entity_id


def test_empty_document_is_checkpointed_without_llm_or_page_mutations(local_storage):
    dump_document(empty=True)
    api, decider = Api(), Decider()

    summary = run_linker(api, decider)

    assert summary.processed_documents == 1
    assert summary.processed_pages == 2
    assert summary.source_entities == 0
    assert not api.requests
    assert not decider.prompts
    assert journal().completed_at is not None
    assert journal().links == []


def test_document_input_identity_is_checked_before_mutations(local_storage):
    dump = dump_document()
    dump.document_id = uid("wrong-document")
    storage.write_text(
        f"document-named-entities-clusters/mzk/{DOCUMENT}.json", dump.model_dump_json()
    )
    api, decider = Api(), Decider()
    with pytest.raises(ValueError, match="identity does not match path"):
        run_linker(api, decider)
    assert not api.requests


def test_document_linking_logs_loaded_cluster_search_decision_pages_and_output(
    local_storage, caplog
):
    dump = dump_document()
    api, decider = Api(), Decider()
    with caplog.at_level(
        logging.INFO, logger="people_gator_common.pipelines.ner_linking"
    ):
        run_linker(api, decider)
    messages = caplog.text
    assert "Listing document cluster dumps using" in messages
    assert (
        f"Loading document cluster dump document-named-entities-clusters/mzk/{DOCUMENT}.json"
        in messages
    )
    assert "2 pages, 1 clusters, 2 entity records" in messages
    assert "Searching entity candidates for Ada Lovelace" in messages
    assert f"Resolving cluster {dump.clusters[0].cluster_id}" in messages
    assert f"Decision for cluster {dump.clusters[0].cluster_id}" in messages
    assert "Uploading mention Ada Lovelace to page source" in messages
    assert (
        f"Saved completed document link journal document-named-entities-links/mzk/{DOCUMENT}.json"
        in messages
    )


def test_document_cli_wiring_selects_scope_and_reports_document_summary(
    local_storage, monkeypatch, capsys
):
    dump_document()
    api, decider = Api(), Decider()
    run = run_ner_entity_linking_pipeline

    async def linked(**kwargs):
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(api.handle)
        ) as client:
            return await run(api_client=client, llm_client=decider, **kwargs)

    monkeypatch.setattr(cli, "run_ner_entity_linking_pipeline", linked)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "people-gator-ner-entity-linking",
            "--scope",
            "document",
            "--decision-only",
            "--max-documents",
            "1",
            "--bearer-token",
            "test-token",
        ],
    )

    assert cli.main() == 0

    assert "1 documents processed" in capsys.readouterr().out
    assert journal().scope == "document"
    assert not api.lookups


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scope", "document", "--max-pages", "1"],
        ["--max-documents", "1"],
        ["--scope", "document", "--max-documents", "0"],
    ],
)
def test_cli_rejects_limits_for_wrong_scope(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["people-gator-ner-entity-linking", *arguments])
    with pytest.raises(SystemExit) as exc:
        cli.parse_args()
    assert exc.value.code == 2
