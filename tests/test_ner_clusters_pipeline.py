from __future__ import annotations

import json
import asyncio
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

from fsspec.implementations.local import LocalFileSystem
import pytest
import httpx

import people_gator_common.filesystem as storage
from people_gator_common.cli import ner_clusters_pipeline_cli as cli
from people_gator_common.pipelines import ner_clusters as pipeline
from people_gator_common.pipelines.llm_ner import PageNer
from people_gator_common.pipelines.ner_linking.api import _mention_payloads
from people_gator_common.pipelines.ner_linking.run import iter_page_source_clusters


class Disambiguator:
    """Exercise real clustering and dump serialization without an external LLM."""

    def __init__(self, *, split: bool = False) -> None:
        self.records: list[list[dict]] = []
        self.split = split
        self.reverse_groups = False
        self.drop_last = False
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **kwargs) -> SimpleNamespace:
        prompt = kwargs["messages"][0]["content"]
        records = json.loads(prompt.rsplit("Input records:\n", 1)[1])["entities"]
        self.records.append(records)
        if self.drop_last:
            records = records[:-1]
        groups = [[record] for record in records] if self.split else [records]
        if self.reverse_groups:
            groups.reverse()
        clusters = [
            {
                "name": group[0]["name"],
                "short_description": " ".join(
                    record["information"] for record in group
                ),
                "long_description": " ".join(record["information"] for record in group),
                "entity_indices": [record["index"] for record in group],
            }
            for group in groups
        ]
        message = SimpleNamespace(
            content=json.dumps({"clusters": clusters}), reasoning=""
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(storage, "fs", LocalFileSystem())
    monkeypatch.setattr(storage, "prefix", str(tmp_path))
    return tmp_path


def write_page(
    page_id: str,
    information: str,
    *,
    library: str = "library",
    document_id: str = "document",
    chunk_text: str | None = None,
    empty: bool = False,
    name: str = "Ada Lovelace",
) -> PageNer:
    entity = {
        "name": name,
        "type": "beeing",
        "info": information,
        "mentions": [{"name": name, "context": information}],
    }
    page = PageNer.model_validate(
        {
            "library": library,
            "document_id": document_id,
            "page_id": page_id,
            "text": chunk_text if chunk_text is not None else information,
            "chunks": [
                {
                    "text": chunk_text if chunk_text is not None else information,
                    "valid_entities": [] if empty else [entity],
                    "invalid_entities": [],
                    "prompt": "",
                    "reasoning": "",
                    "raw_output": "",
                }
            ],
            "chunk_metadata": [{"start_line": 0, "end_line": 1, "line_mapping": []}],
        }
    )
    storage.write_text(
        f"page-named-entities/{library}/{document_id}/{page_id}.json",
        page.model_dump_json(),
    )
    return page


def read_dump(
    *,
    page_id: str | None = None,
    library: str = "library",
    document_id: str = "document",
) -> pipeline.NerClustersDump:
    if page_id is None:
        path = f"document-named-entities-clusters/{library}/{document_id}.json"
    else:
        path = f"page-named-entities-clusters/{library}/{document_id}/{page_id}.json"
    return pipeline.NerClustersDump.model_validate_json(storage.read_text(path))


def test_document_scope_writes_one_complete_dump_with_exact_page_members(storage_root):
    first = write_page("page-a", "English mathematician.", chunk_text="Repeated OCR.")
    second = write_page(
        "page-b", "Wrote about the Analytical Engine.", chunk_text="Repeated OCR."
    )
    assert first.chunks[0].chunk_id == second.chunks[0].chunk_id
    client = Disambiguator()

    summary = pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    assert summary.processed_documents == 1
    assert summary.processed_pages == 2
    assert summary.clusters == 1
    assert len(client.records) == 1
    assert sorted(storage.glob("document-named-entities-clusters/**/*.json")) == [
        "document-named-entities-clusters/library/document.json"
    ]
    dump = read_dump()
    assert dump.scope == "document"
    assert dump.page_id is None
    assert [page.page_id for page in dump.inputs] == ["page-a", "page-b"]
    assert [page.path for page in dump.inputs] == [
        "page-named-entities/library/document/page-a.json",
        "page-named-entities/library/document/page-b.json",
    ]
    assert len(dump.clusters) == 1
    cluster = dump.clusters[0]
    assert cluster.cluster_id
    assert (
        cluster.short_description
        == "English mathematician. Wrote about the Analytical Engine."
    )
    assert len(cluster.members) == 2
    assert len({member.record_id for member in cluster.members}) == 2
    for member, original in zip(cluster.members, [first, second], strict=True):
        assert member.page_id == original.page_id
        assert member.chunk_id == original.chunks[0].chunk_id
        assert member.chunk_index == 0
        assert member.entity_index == 0
        expected = original.chunks[0].valid_entities[0]
        assert member.entity.name == expected.name
        assert member.entity.info == expected.info
        assert member.entity.mentions[0].context == expected.mentions[0].context
        assert member.entity.mentions[0].chunk_id == member.chunk_id
    assert (
        "English mathematician."
        in dump.disambiguations[cluster.disambiguation_index].prompt
    )
    assert (
        "Wrote about the Analytical Engine."
        in dump.disambiguations[cluster.disambiguation_index].prompt
    )
    assert len(cluster.entity_indices) == len(cluster.members)


def test_page_scope_writes_one_dump_per_page_and_remains_linker_compatible(
    storage_root,
):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()

    summary = pipeline.run_ner_clusters_pipeline(client=client)

    assert summary.processed_pages == 2
    assert not client.records
    assert sorted(storage.glob("page-named-entities-clusters/**/*.json")) == [
        "page-named-entities-clusters/library/document/page-a.json",
        "page-named-entities-clusters/library/document/page-b.json",
    ]
    for page_id, context in [
        ("page-a", "English mathematician."),
        ("page-b", "Wrote about the Analytical Engine."),
    ]:
        dump = read_dump(page_id=page_id)
        assert dump.scope == "page"
        assert dump.page_id == page_id
        assert len(dump.inputs) == 1
        assert len(dump.clusters) == 1
        cluster = dump.clusters[0]
        assert len(cluster.members) == 1
        assert cluster.members[0].page_id == page_id
        assert cluster.short_description == context
        adapted = pipeline.load_page_ner_clusters(dump.model_dump_json())
        link_cluster = list(iter_page_source_clusters(adapted))[0]
        assert link_cluster.cluster_id == cluster.cluster_id
        mentions = _mention_payloads(
            link_cluster, entity_id="entity", source_id=page_id
        )
        assert [mention["context"] for mention in mentions] == [context]


def test_page_linker_reads_new_dump_and_retains_cluster_id_in_journal(storage_root):
    from people_gator_common.pipelines.ner_entity_linking import (
        EntityLinkingConfig,
        PageNerEntityLinks,
        run_ner_entity_linking_pipeline,
    )

    write_page("page-a", "English mathematician.")
    pipeline.run_ner_clusters_pipeline(client=Disambiguator())

    def search(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/search-entities"
        return httpx.Response(200, json={"items": []})

    def decide(**kwargs):
        response = {
            "action": "create",
            "candidate_index": None,
            "short_description": "Anglická matematička.",
            "long_description": "Ada Lovelace byla anglická matematička.",
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(response), reasoning="")
                )
            ]
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(search)
        ) as client:
            return await run_ner_entity_linking_pipeline(
                config=EntityLinkingConfig(bearer_token="test-token"),
                api_client=client,
                llm_client=SimpleNamespace(chat=SimpleNamespace(send=decide)),
                mode="decision-only",
            )

    summary = asyncio.run(run())
    assert summary.processed_pages == 1
    journal = PageNerEntityLinks.model_validate_json(
        storage.read_text("page-named-entities-links/library/document/page-a.json")
    )
    assert (
        journal.links[0].source_cluster.cluster_id
        == read_dump(page_id="page-a").clusters[0].cluster_id
    )


def test_legacy_page_outputs_can_be_read_and_are_upgraded_to_cluster_dumps(
    storage_root,
):
    original = write_page("page-a", "English mathematician.")
    cluster = pipeline.DisambiguatedNamedEntity.from_named_entity(
        1, original.chunks[0].valid_entities[0]
    )
    legacy = pipeline.PageNerClusters(
        library="library",
        document_id="document",
        page_id="page-a",
        clusters=[
            pipeline.DisambiguationOutput(
                text="", prompt="", reasoning="", raw_output="", clusters=[cluster]
            )
        ],
    )
    path = "page-named-entities-clusters/library/document/page-a.json"
    storage.write_text(path, legacy.model_dump_json())
    assert pipeline.load_page_ner_clusters(storage.read_text(path)) == legacy

    summary = pipeline.run_ner_clusters_pipeline(client=Disambiguator())

    assert summary.processed_pages == 1
    assert read_dump(page_id="page-a").clusters[0].cluster_id
    assert pipeline.run_ner_clusters_pipeline(client=Disambiguator()).skipped_pages == 1


def test_document_scope_isolates_libraries_documents_and_retains_empty_page_references(
    storage_root,
):
    write_page("page-a", "First document.")
    write_page("page-b", "Second document.", document_id="other")
    write_page("page-c", "Other library.", library="other")
    write_page("page-empty", "", empty=True)
    write_page("page-only-empty", "", empty=True, document_id="empty")
    client = Disambiguator()

    summary = pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    assert summary.processed_documents == 4
    assert summary.processed_pages == 5
    assert summary.clusters == 3
    assert not client.records
    assert read_dump(document_id="empty").clusters == []
    assert [page.page_id for page in read_dump().inputs] == ["page-a", "page-empty"]
    dumps = [read_dump(), read_dump(document_id="other"), read_dump(library="other")]
    assert len({dump.clusters[0].cluster_id for dump in dumps}) == 3
    assert [dump.clusters[0].short_description for dump in dumps] == [
        "First document.",
        "Second document.",
        "Other library.",
    ]


def test_cluster_ids_survive_unchanged_reruns_and_change_on_split_or_merge(
    storage_root,
):
    write_page("page-a", "A mathematician.")
    write_page("page-b", "A fictional namesake.")
    client = Disambiguator()
    pipeline.run_ner_clusters_pipeline(scope="document", client=client)
    original = read_dump()
    original_records = {member.record_id for member in original.clusters[0].members}

    pipeline.run_ner_clusters_pipeline(scope="document", client=client, force=True)
    repeated = read_dump()
    assert repeated.run_id != original.run_id
    assert repeated.clusters[0].cluster_id == original.clusters[0].cluster_id

    client.split = True
    pipeline.run_ner_clusters_pipeline(scope="document", client=client, force=True)
    split = read_dump()
    assert len(split.clusters) == 2
    assert original.clusters[0].cluster_id not in {
        cluster.cluster_id for cluster in split.clusters
    }
    assert {
        member.record_id for cluster in split.clusters for member in cluster.members
    } == original_records
    assert {cluster.short_description for cluster in split.clusters} == {
        "A mathematician.",
        "A fictional namesake.",
    }

    client.reverse_groups = True
    pipeline.run_ner_clusters_pipeline(scope="document", client=client, force=True)
    assert {cluster.cluster_id for cluster in read_dump().clusters} == {
        cluster.cluster_id for cluster in split.clusters
    }

    client.split = False
    pipeline.run_ner_clusters_pipeline(scope="document", client=client, force=True)
    assert read_dump().clusters[0].cluster_id == original.clusters[0].cluster_id


def test_adding_an_unrelated_person_does_not_change_existing_cluster_id(storage_root):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    pipeline.run_ner_clusters_pipeline(scope="document", client=client)
    original = read_dump().clusters[0]
    write_page("page-c", "American computer scientist.", name="Grace Hopper")

    pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    clusters = {cluster.name: cluster for cluster in read_dump().clusters}
    assert len(clusters) == 2
    assert clusters["Ada Lovelace"].cluster_id == original.cluster_id
    assert clusters["Grace Hopper"].members[0].page_id == "page-c"


def test_dump_retains_every_record_when_llm_omits_a_member(storage_root):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    client.drop_last = True

    pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    dump = read_dump()
    assert len(dump.clusters) == 2
    assert {
        member.page_id for cluster in dump.clusters for member in cluster.members
    } == {"page-a", "page-b"}
    assert sum(len(cluster.members) for cluster in dump.clusters) == 2


def test_document_rerun_skips_complete_dump_preserves_edits_and_rebuilds_missing_dump(
    storage_root,
):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    pipeline.run_ner_clusters_pipeline(scope="document", client=client)
    original = read_dump()
    original.clusters[0].short_description = "Reviewed description."
    storage.write_text(
        "document-named-entities-clusters/library/document.json",
        original.model_dump_json(),
    )

    skipped = pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    assert skipped.skipped_documents == 1
    assert skipped.skipped_pages == 2
    assert skipped.processed_pages == 0
    assert len(client.records) == 1
    assert read_dump().run_id == original.run_id
    assert read_dump().clusters[0].short_description == "Reviewed description."
    storage.delete("document-named-entities-clusters/library/document.json")

    resumed = pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    assert resumed.processed_pages == 2
    assert len(client.records) == 2
    assert read_dump().run_id != original.run_id
    assert read_dump().clusters[0].cluster_id == original.clusters[0].cluster_id


@pytest.mark.parametrize("change", ["new-page", "edited-page", "model", "threshold"])
def test_document_rerun_rebuilds_when_inputs_or_settings_change(storage_root, change):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    pipeline.run_ner_clusters_pipeline(scope="document", client=client)
    original = read_dump()
    kwargs = {}
    if change == "new-page":
        write_page("page-c", "Published notes.")
    elif change == "edited-page":
        write_page("page-b", "Revised evidence.")
    elif change == "model":
        kwargs["model"] = "another-model"
    else:
        kwargs["threshold"] = 90

    summary = pipeline.run_ner_clusters_pipeline(
        scope="document", client=client, **kwargs
    )

    expected_pages = 3 if change == "new-page" else 2
    assert summary.processed_documents == 1
    assert summary.processed_pages == expected_pages
    assert len(client.records) == 2
    current = read_dump()
    assert len(current.inputs) == expected_pages
    assert len(current.clusters[0].members) == expected_pages
    assert current.input_fingerprint != original.input_fingerprint
    assert current.run_id != original.run_id
    if change == "new-page":
        assert "Published notes." in current.clusters[0].long_description
        assert current.clusters[0].cluster_id != original.clusters[0].cluster_id
    elif change == "edited-page":
        assert "Revised evidence." in current.clusters[0].long_description
    else:
        assert current.clusters[0].cluster_id == original.clusters[0].cluster_id


def test_interrupted_dump_write_is_regenerated(storage_root, monkeypatch):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    pipeline.run_ner_clusters_pipeline(scope="document", client=client)
    write = pipeline.storage_write_text

    def fail_write(path, content):
        write(path, '{"schema_version": 1, "scope":')
        raise OSError("Interrupted output write")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "storage_write_text", fail_write)
        with pytest.raises(OSError, match="Interrupted output write"):
            pipeline.run_ner_clusters_pipeline(
                scope="document", client=client, force=True
            )

    summary = pipeline.run_ner_clusters_pipeline(scope="document", client=client)

    assert summary.processed_pages == 2
    assert len(client.records) == 3
    assert len(read_dump().clusters[0].members) == 2


def test_scope_paths_are_separate_even_under_one_output_prefix(storage_root):
    write_page("page-a", "English mathematician.")
    client = Disambiguator()
    pipeline.run_ner_clusters_pipeline(output_prefix="shared", client=client)
    pipeline.run_ner_clusters_pipeline(
        scope="document", output_prefix="shared", client=client
    )

    assert sorted(storage.glob("shared/**/*.json")) == [
        "shared/library/document.json",
        "shared/library/document/page-a.json",
    ]


def test_document_limit_keeps_documents_whole_and_counts_only_processed_documents(
    storage_root,
):
    write_page("page-a", "First page.", document_id="a")
    write_page("page-b", "Second page.", document_id="a")
    write_page("page-c", "Other document.", document_id="b")
    client = Disambiguator()

    first = pipeline.run_ner_clusters_pipeline(
        scope="document", max_documents=1, client=client
    )

    assert first.processed_documents == 1
    assert first.processed_pages == 2
    assert not storage.exists("document-named-entities-clusters/library/b.json")
    assert len(read_dump(document_id="a").clusters[0].members) == 2

    second = pipeline.run_ner_clusters_pipeline(
        scope="document", max_documents=1, client=client
    )

    assert second.skipped_documents == 1
    assert second.skipped_pages == 2
    assert second.processed_documents == 1
    assert second.processed_pages == 1
    assert read_dump(document_id="b").scope == "document"


@pytest.mark.parametrize(
    "scope,limits",
    [
        ("document", {"max_pages": 1}),
        ("page", {"max_documents": 1}),
        ("document", {"max_documents": 0}),
        ("document", {"max_documents": -1}),
        ("unknown", {}),
    ],
)
def test_invalid_scope_and_limits_fail_before_storage_access(scope, limits):
    with pytest.raises(ValueError):
        pipeline.run_ner_clusters_pipeline(
            scope=scope, client=Disambiguator(), **limits
        )


def test_logs_identify_inputs_current_group_and_saved_dump(storage_root, caplog):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    with caplog.at_level(logging.INFO, logger=pipeline.__name__):
        pipeline.run_ner_clusters_pipeline(scope="document", client=client)
        pipeline.run_ner_clusters_pipeline(scope="document", client=client)
    messages = caplog.text
    assert "Found 2 NER page files below page-named-entities" in messages
    assert (
        "Loading page 1/2 from page-named-entities/library/document/page-a.json"
        in messages
    )
    assert "Loaded page page-b: 1 chunks, 1 entity records" in messages
    assert (
        "Processing document library/document: 2 pages, 2 chunks, 2 entity records"
        in messages
    )
    assert "Fuzzy clustering produced 1 candidate groups" in messages
    assert "Disambiguating group 1/1: Ada Lovelace (2 entity records" in messages
    assert (
        "Saved document cluster dump document-named-entities-clusters/library/document.json"
        in messages
    )
    assert "Skipping unchanged document library/document" in messages


def test_dump_validation_rejects_unknown_page_and_duplicate_members(storage_root):
    write_page("page-a", "English mathematician.")
    pipeline.run_ner_clusters_pipeline(scope="document", client=Disambiguator())
    dump = read_dump().model_dump(mode="json")
    dump["clusters"][0]["members"][0]["page_id"] = "foreign-page"
    with pytest.raises(ValueError, match="unknown page"):
        pipeline.NerClustersDump.model_validate(dump)
    dump["clusters"][0]["members"][0]["page_id"] = "page-a"
    dump["clusters"][0]["members"].append(dump["clusters"][0]["members"][0])
    dump["clusters"][0]["entity_indices"].append(2)
    with pytest.raises(ValueError, match="exactly one cluster"):
        pipeline.NerClustersDump.model_validate(dump)


def test_document_dump_requires_document_level_consumer(storage_root):
    write_page("page-a", "English mathematician.")
    pipeline.run_ner_clusters_pipeline(scope="document", client=Disambiguator())
    with pytest.raises(ValueError, match="page-scoped"):
        pipeline.load_page_ner_clusters(read_dump().model_dump_json())


def test_cli_passes_document_scope_and_reports_documents(
    storage_root, monkeypatch, capsys
):
    write_page("page-a", "English mathematician.")
    write_page("page-b", "Wrote about the Analytical Engine.")
    client = Disambiguator()
    run = pipeline.run_ner_clusters_pipeline
    monkeypatch.setattr(
        cli, "run_ner_clusters_pipeline", lambda **kwargs: run(client=client, **kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["people-gator-ner-clusters", "--scope", "document", "--max-documents", "1"],
    )

    assert cli.main() == 0

    assert "1 documents processed" in capsys.readouterr().out
    assert len(read_dump().clusters[0].members) == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--scope", "document", "--max-pages", "1"],
        ["--max-documents", "1"],
        ["--scope", "document", "--max-documents", "0"],
    ],
)
def test_cli_rejects_incompatible_limits(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["people-gator-ner-clusters", *arguments])
    with pytest.raises(SystemExit) as exc:
        cli.parse_args()
    assert exc.value.code == 2
