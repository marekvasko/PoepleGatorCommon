from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

import people_gator_common.pipelines.ner_entity_linking as entity_linking
import people_gator_common.pipelines.ner_linking.journal as linking_journal
import people_gator_common.pipelines.ner_linking.pipeline as linking_pipeline
import people_gator_common.pipelines.ner_linking.rollback as linking_rollback
import people_gator_common.pipelines.ner_linking.run as linking_run
from people_gator_common.pipelines.llm_ner import NamedEntity, NamedEntityMention
from people_gator_common.pipelines.ner_clusters import DisambiguatedNamedEntity
from people_gator_common.pipelines.ner_entity_linking import (
    EntityCandidate,
    EntityLinkDecision,
    EntityLinkingConfig,
    LinkableEntity,
    LinkedEntityResult,
    MutationAudit,
    PageNerEntityLinks,
    apply_linked_entity_result,
    authenticate_entity_linking_api,
    link_source_cluster,
    llm_decide_entity_link,
    lookup_page_source,
    rollback_latest_entity_linking_run,
    search_entity_candidates,
    source_entity_from_cluster,
)


class FakeOpenRouter:
    def __init__(self, responses: str | list[str]) -> None:
        self._responses = responses if isinstance(responses, list) else [responses]
        self.calls = 0
        self.prompt = ""
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **kwargs: object) -> SimpleNamespace:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.prompt = messages[-1]["content"]
        self.prompts.append(self.prompt)
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        message = SimpleNamespace(content=response, reasoning="")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeMessageOpenRouter:
    def __init__(
        self,
        *,
        content: object,
        reasoning: object = "",
        reasoning_details: object = None,
    ) -> None:
        self._content = content
        self._reasoning = reasoning
        self._reasoning_details = reasoning_details
        self.calls = 0
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        message = SimpleNamespace(
            content=self._content,
            reasoning=self._reasoning,
            reasoning_details=self._reasoning_details,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _source() -> LinkableEntity:
    return LinkableEntity(
        name="Ada Lovelace",
        aliases=["A. Lovelace"],
        short_description="English mathematician.",
        long_description="Ada Lovelace wrote about the Analytical Engine.",
    )


def _candidate() -> EntityCandidate:
    return EntityCandidate(
        entity_id="11111111-1111-1111-1111-111111111111",
        entity_type="person",
        name="Augusta Ada King",
        aliases=["Ada Lovelace"],
        short_description="English mathematician and writer.",
        long_description="She was an English mathematician and writer.",
        matching_score=0.91,
    )


def _cluster() -> DisambiguatedNamedEntity:
    entity = NamedEntity(
        name="Ada Lovelace",
        type="beeing",
        mentions=[
            NamedEntityMention(
                name="A. Lovelace",
                context="Text about A. Lovelace.",
            )
        ],
        info="English mathematician.",
    )
    return DisambiguatedNamedEntity(
        name="Ada Lovelace",
        short_description="English mathematician.",
        long_description="Ada Lovelace wrote about the Analytical Engine.",
        entity_indices=[1],
        entities=[entity],
    )


def test_cluster_is_exposed_as_deterministic_search_input() -> None:
    source = source_entity_from_cluster(_cluster())

    assert source.name == "Ada Lovelace"
    assert source.aliases == ["A. Lovelace"]
    assert source.short_description == "English mathematician."


def test_search_uses_repeated_query_text_and_hydrates_entity_details() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer token"
        if request.url.path == "/api/v1/search-entities":
            assert request.url.params.multi_items() == [
                ("query.text", "Ada Lovelace"),
                ("query.text", "A. Lovelace"),
                ("query.text", "English mathematician."),
                (
                    "query.text",
                    "Ada Lovelace wrote about the Analytical Engine.",
                ),
            ]
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "entity_id": _candidate().entity_id,
                            "matching_score": 0.91,
                        }
                    ],
                    "total_count": 1,
                    "limit": 20,
                },
            )
        assert request.url.path == f"/api/v1/entities/{_candidate().entity_id}"
        return httpx.Response(
            200,
            json={
                "entity_id": _candidate().entity_id,
                "entity_type": "person",
                "name": "Augusta Ada King",
                "aliases": ["Ada Lovelace"],
                "short_description": "English mathematician and writer.",
                "description": "She was an English mathematician and writer.",
            },
        )

    async def run() -> list[EntityCandidate]:
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await search_entity_candidates(
                client,
                {"Authorization": "Bearer token"},
                _source(),
            )

    candidates = asyncio.run(run())

    assert len(requests) == 2
    assert candidates == [_candidate()]


def test_merge_decision_uses_candidate_index_and_czech_descriptions() -> None:
    client = FakeOpenRouter(
        """{
            "action": "merge",
            "candidate_index": 1,
            "short_description": "Anglická matematička a spisovatelka.",
            "long_description": "Ada Lovelace byla anglická matematička a spisovatelka."
        }"""
    )

    decision, llm = llm_decide_entity_link(
        _source(),
        [_candidate()],
        client=client,
    )

    assert decision.action == "merge"
    assert decision.candidate_index == 1
    assert decision.description_language == "cs"
    assert decision.name == "Augusta Ada King"
    assert decision.aliases == ["Ada Lovelace", "A. Lovelace"]
    assert decision.short_description == "Anglická matematička a spisovatelka."
    assert llm is not None
    assert '"index": 1' in client.prompt
    assert "must be written in Czech (`cs`)" in client.prompt
    assert r"ASCII double quotes as `\"`" in client.prompt
    assert r"backslashes as `\\`" in client.prompt
    assert r"JSON escapes (`\n`, `\r`, `\t`" in client.prompt


def test_create_decision_also_calls_llm_to_force_czech() -> None:
    client = FakeOpenRouter(
        """{
            "action": "create",
            "candidate_index": null,
            "short_description": "Anglická matematička.",
            "long_description": "Ada Lovelace psala o analytickém stroji."
        }"""
    )

    decision, _ = llm_decide_entity_link(_source(), [], client=client)

    assert client.calls == 1
    assert decision.action == "create"
    assert decision.description_language == "cs"
    assert decision.short_description == "Anglická matematička."
    assert decision.long_description == "Ada Lovelace psala o analytickém stroji."


def test_text_content_blocks_are_parsed() -> None:
    response = (
        '{"action":"create","candidate_index":null,'
        '"short_description":"Anglická matematička.",'
        '"long_description":"Ada Lovelace byla matematička."}'
    )
    client = FakeMessageOpenRouter(
        content=[SimpleNamespace(type="text", text=response)]
    )

    decision, _ = llm_decide_entity_link(_source(), [], client=client)

    assert client.calls == 1
    assert decision.action == "create"


def test_reasoning_channel_json_is_used_when_content_is_empty() -> None:
    response = (
        '{"action":"create","candidate_index":null,'
        '"short_description":"Anglická matematička.",'
        '"long_description":"Ada Lovelace byla matematička."}'
    )
    client = FakeMessageOpenRouter(content=None, reasoning=response)

    decision, llm = llm_decide_entity_link(_source(), [], client=client)

    assert client.calls == 1
    assert decision.action == "create"
    assert llm.raw_output == response


def test_invalid_first_response_is_repaired_once() -> None:
    invalid = (
        '{"action":"create","candidate_index":null,'
        '"short_description":"","long_description":""}'
    )
    valid = (
        '{"action":"create","candidate_index":null,'
        '"short_description":"Anglická matematička.",'
        '"long_description":"Ada Lovelace psala o analytickém stroji."}'
    )
    client = FakeOpenRouter([invalid, valid])

    decision, llm = llm_decide_entity_link(_source(), [], client=client)

    assert client.calls == 2
    assert "previous response did not satisfy" in client.prompt
    assert decision.action == "create"
    assert decision.description_language == "cs"
    assert "initial" in llm.raw_output
    assert "repair" in llm.raw_output


def test_nested_decision_object_is_accepted_without_retry() -> None:
    client = FakeOpenRouter(
        '{"decision":{'
        '"action":"create","candidate_index":null,'
        '"short_description":"Anglická matematička.",'
        '"long_description":"Ada Lovelace byla matematička."}}'
    )

    decision, _ = llm_decide_entity_link(_source(), [], client=client)

    assert client.calls == 1
    assert decision.action == "create"


def test_missing_translated_descriptions_are_rejected() -> None:
    client = FakeOpenRouter(
        '{"action":"create","candidate_index":null,'
        '"short_description":"","long_description":""}'
    )

    with pytest.raises(RuntimeError, match="after one repair attempt") as exc_info:
        llm_decide_entity_link(_source(), [], client=client)

    message = str(exc_info.value)
    assert client.calls == 2
    assert "Initial response:" in message
    assert "Repair response:" in message
    assert "short_description" in message


def test_link_source_cluster_saves_decision_without_entity_mutation() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/api/v1/search-entities":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "entity_id": _candidate().entity_id,
                            "matching_score": 0.91,
                        }
                    ],
                    "total_count": 1,
                    "limit": 20,
                },
            )
        if request.url.path == f"/api/v1/entities/{_candidate().entity_id}":
            return httpx.Response(
                200,
                json={
                    "entity_id": _candidate().entity_id,
                    "entity_type": "person",
                    "name": _candidate().name,
                    "aliases": _candidate().aliases,
                    "short_description": _candidate().short_description,
                    "description": _candidate().long_description,
                },
            )
        raise AssertionError(
            f"Unexpected mutating request: {request.method} {request.url}"
        )

    llm = FakeOpenRouter(
        """{
            "action":"merge",
            "candidate_index":1,
            "short_description":"Anglická matematička.",
            "long_description":"Ada Lovelace byla anglická matematička."
        }"""
    )

    async def run():
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await link_source_cluster(
                _cluster(),
                api_client=client,
                headers={"Authorization": "Bearer token"},
                llm_client=llm,
            )

    result = asyncio.run(run())

    assert methods == ["GET", "GET"]
    assert result.selected_entity_id == _candidate().entity_id
    assert result.source_cluster == _cluster()


def test_apply_merge_updates_entity_and_uploads_mentions_with_audit() -> None:
    entity_id = _candidate().entity_id
    checkpoint_count = 0

    def entity_payload(short_description: str) -> dict[str, object]:
        return {
            "entity_id": entity_id,
            "entity_type": "person",
            "name": "Augusta Ada King",
            "aliases": ["Ada Lovelace"],
            "short_description": short_description,
            "description": "She was an English mathematician and writer.",
            "wikidata_id": "Q7259",
            "nlcr_authority_id": None,
            "user_id": "22222222-2222-2222-2222-222222222222",
        }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=entity_payload("Old description."))
        if request.method == "PATCH":
            body = request.read().decode()
            assert '"short_description":"Anglická matematička."' in body
            assert '"aliases"' not in body
            return httpx.Response(200, json=entity_payload("Anglická matematička."))
        assert request.url.path == "/api/v1/text-mentions"
        return httpx.Response(
            201,
            json={
                "mention_id": "33333333-3333-3333-3333-333333333333",
                "name": "A. Lovelace",
                "context": "Text about A. Lovelace.",
                "entity_id": entity_id,
                "source_id": "44444444-4444-4444-4444-444444444444",
            },
        )

    result = LinkedEntityResult(
        source_cluster=_cluster(),
        source=_source(),
        candidates=[_candidate()],
        decision=EntityLinkDecision(
            action="merge",
            candidate_index=1,
            name="Augusta Ada King",
            aliases=["Ada Lovelace", "Countess of Lovelace"],
            short_description="Anglická matematička.",
            long_description="She was an English mathematician and writer.",
        ),
        selected_entity_id=entity_id,
    )

    async def run() -> None:
        nonlocal checkpoint_count

        async def checkpoint() -> None:
            nonlocal checkpoint_count
            checkpoint_count += 1

        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            await apply_linked_entity_result(
                result,
                source_id="44444444-4444-4444-4444-444444444444",
                api_client=client,
                headers={"Authorization": "Bearer token"},
                checkpoint=checkpoint,
            )
            await apply_linked_entity_result(
                result,
                source_id="44444444-4444-4444-4444-444444444444",
                api_client=client,
                headers={"Authorization": "Bearer token"},
                checkpoint=checkpoint,
            )

    asyncio.run(run())

    assert checkpoint_count == 3
    assert result.selected_entity_before is not None
    assert result.selected_entity_before.wikidata_id == "Q7259"
    assert result.unapplied_fields == ["aliases"]
    assert [item.operation for item in result.mutations] == [
        "update_entity",
        "create_mention",
    ]
    assert entity_linking._link_application_complete(
        result,
        source_id="44444444-4444-4444-4444-444444444444",
    )
    incomplete = result.model_copy(deep=True)
    incomplete.mutations = incomplete.mutations[:1]
    assert not entity_linking._link_application_complete(
        incomplete,
        source_id="44444444-4444-4444-4444-444444444444",
    )


def test_apply_create_posts_person_before_mentions() -> None:
    entity_id = "55555555-5555-5555-5555-555555555555"
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        body = request.read().decode()
        if request.url.path == "/api/v1/entities":
            assert '"entity_type":"person"' in body
            assert '"wikidata_id":null' in body
            return httpx.Response(
                201,
                json={
                    "entity_id": entity_id,
                    "entity_type": "person",
                    "name": "Ada Lovelace",
                    "aliases": ["A. Lovelace"],
                    "short_description": "Anglická matematička.",
                    "description": "Ada Lovelace byla matematička.",
                    "user_id": "22222222-2222-2222-2222-222222222222",
                },
            )
        assert f'"entity_id":"{entity_id}"' in body
        return httpx.Response(
            201,
            json={
                "mention_id": "33333333-3333-3333-3333-333333333333",
                "name": "A. Lovelace",
                "context": "Text about A. Lovelace.",
                "entity_id": entity_id,
                "source_id": "44444444-4444-4444-4444-444444444444",
            },
        )

    result = LinkedEntityResult(
        source_cluster=_cluster(),
        source=_source(),
        candidates=[],
        decision=EntityLinkDecision(
            action="create",
            name="Ada Lovelace",
            aliases=["A. Lovelace"],
            short_description="Anglická matematička.",
            long_description="Ada Lovelace byla matematička.",
        ),
    )

    async def run() -> None:
        async def checkpoint() -> None:
            pass

        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            await apply_linked_entity_result(
                result,
                source_id="44444444-4444-4444-4444-444444444444",
                api_client=client,
                headers={"Authorization": "Bearer token"},
                checkpoint=checkpoint,
            )
            await apply_linked_entity_result(
                result,
                source_id="44444444-4444-4444-4444-444444444444",
                api_client=client,
                headers={"Authorization": "Bearer token"},
                checkpoint=checkpoint,
            )

    asyncio.run(run())

    assert requests == ["/api/v1/entities", "/api/v1/text-mentions"]
    assert result.selected_entity_id == entity_id
    assert [item.operation for item in result.mutations] == [
        "create_entity",
        "create_mention",
    ]
    assert entity_linking._link_application_complete(
        result,
        source_id="44444444-4444-4444-4444-444444444444",
    )


def test_missing_page_source_is_created_after_lookup_404() -> None:
    source_id = "44444444-4444-4444-4444-444444444444"
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(
                404,
                json={"detail": {"code": "not_found", "message": "Page not found"}},
            )
        assert request.url.path == "/api/v1/pages"
        assert request.read().decode() == (
            '{"library":"d","document_id":"11111111-1111-1111-1111-111111111111",'
            '"page_id":"22222222-2222-2222-2222-222222222222"}'
        )
        return httpx.Response(
            201,
            json={
                "source_id": source_id,
                "source_type": "page",
                "url": "https://example.test/page",
                "library": "d",
                "document_id": "11111111-1111-1111-1111-111111111111",
                "page_id": "22222222-2222-2222-2222-222222222222",
            },
        )

    async def run() -> str:
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await lookup_page_source(
                client,
                {"Authorization": "Bearer token"},
                library="d",
                document_id="11111111-1111-1111-1111-111111111111",
                page_id="22222222-2222-2222-2222-222222222222",
            )

    assert asyncio.run(run()) == source_id
    assert requests == [
        ("GET", "/api/v1/pages/lookup"),
        ("POST", "/api/v1/pages"),
    ]


def test_page_creation_conflict_retries_lookup() -> None:
    source_id = "44444444-4444-4444-4444-444444444444"
    lookup_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal lookup_calls
        if request.method == "POST":
            return httpx.Response(
                409,
                json={"detail": {"code": "already_exists"}},
            )
        lookup_calls += 1
        if lookup_calls == 1:
            return httpx.Response(404, json={"detail": {"code": "not_found"}})
        return httpx.Response(
            200,
            json={
                "source_id": source_id,
                "source_type": "page",
                "url": "https://example.test/page",
                "library": "d",
                "document_id": "11111111-1111-1111-1111-111111111111",
                "page_id": "22222222-2222-2222-2222-222222222222",
            },
        )

    async def run() -> str:
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await lookup_page_source(
                client,
                {},
                library="d",
                document_id="11111111-1111-1111-1111-111111111111",
                page_id="22222222-2222-2222-2222-222222222222",
            )

    assert asyncio.run(run()) == source_id
    assert lookup_calls == 2


def test_password_flow_requests_only_browse_scope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/token/login"
        form = parse_qs(request.content.decode())
        assert form["scope"] == ["people-gator:browse"]
        return httpx.Response(200, json={"access_token": "access-token"})

    async def run() -> str:
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await authenticate_entity_linking_api(
                client,
                EntityLinkingConfig(username="user", password="secret"),
            )

    assert asyncio.run(run()) == "access-token"


def test_rollback_reverses_mentions_then_restores_entity(monkeypatch) -> None:
    entity_id = _candidate().entity_id
    mention_id = "33333333-3333-3333-3333-333333333333"
    user_id = "22222222-2222-2222-2222-222222222222"

    def state(short_description: str) -> dict[str, object]:
        return {
            "entity_id": entity_id,
            "entity_type": "person",
            "name": "Augusta Ada King",
            "description": "Biography.",
            "short_description": short_description,
            "aliases": ["Ada Lovelace"],
            "wikidata_id": "Q7259",
            "nlcr_authority_id": None,
            "user_id": user_id,
        }

    link = LinkedEntityResult(
        source_cluster=_cluster(),
        source=_source(),
        candidates=[_candidate()],
        decision=EntityLinkDecision(
            action="merge",
            candidate_index=1,
            name="Augusta Ada King",
            aliases=["Ada Lovelace"],
            short_description="New.",
            long_description="Biography.",
        ),
        selected_entity_id=entity_id,
        mutations=[
            MutationAudit(
                operation="update_entity",
                resource_id=entity_id,
                endpoint=f"/api/v1/entities/{entity_id}",
                before=state("Old."),
                after=state("New."),
                applied_at="2026-01-01T00:00:00+00:00",
            ),
            MutationAudit(
                operation="create_mention",
                resource_id=mention_id,
                endpoint="/api/v1/text-mentions",
                after={"mention_id": mention_id},
                applied_at="2026-01-01T00:00:01+00:00",
            ),
        ],
    )
    output = PageNerEntityLinks(
        library="library",
        document_id="document",
        page_id="page",
        links=[link],
        mode="apply",
    )
    monkeypatch.setattr(
        linking_rollback,
        "_latest_rollbackable_output",
        lambda _prefix: ("audit.json", output),
    )
    monkeypatch.setattr(linking_rollback, "storage_write_text", lambda *_args: None)
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "GET":
            return httpx.Response(200, json=state("New."))
        assert request.method == "PATCH"
        assert request.read().decode().find('"short_description":"Old."') >= 0
        return httpx.Response(200, json=state("Old."))

    async def run():
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await rollback_latest_entity_linking_run(
                config=EntityLinkingConfig(bearer_token="token"),
                api_client=client,
            )

    summary = asyncio.run(run())

    assert summary.rolled_back_mutations == 2
    assert requests == [
        ("DELETE", f"/api/v1/text-mentions/{mention_id}"),
        ("GET", f"/api/v1/entities/{entity_id}"),
        ("PATCH", f"/api/v1/entities/{entity_id}"),
    ]
    assert all(item.rollback_status == "rolled_back" for item in link.mutations)


@pytest.mark.parametrize("split_pages", [False, True])
def test_parallel_entities_revalidate_after_ordered_create(
    tmp_path, monkeypatch, split_pages
) -> None:
    from people_gator_common.pipelines.ner_clusters import (
        DisambiguationOutput,
        PageNerClusters,
    )
    from people_gator_common.pipelines.llm_ner import PageInput

    cluster = _cluster().model_copy(update={"entities": None})
    input_prefix = tmp_path / "input"
    output_prefix = tmp_path / "output"
    page_ids = ["page-a", "page-b"] if split_pages else ["page"]
    page_inputs = []
    for page_id in page_ids:
        page_path = input_prefix / "lib" / "doc" / f"{page_id}.json"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text(
            PageNerClusters(
                library="lib",
                document_id="doc",
                page_id=page_id,
                clusters=[
                    DisambiguationOutput(
                        text="",
                        prompt="",
                        reasoning="",
                        raw_output="",
                        clusters=[cluster] if split_pages else [cluster, cluster],
                    )
                ],
            ).model_dump_json()
        )
        page_inputs.append(
            PageInput(
                library="lib",
                document_id="doc",
                page_id=page_id,
                path=str(page_path),
            )
        )
    monkeypatch.setattr(
        linking_pipeline, "_iter_cluster_pages", lambda _prefix: iter(page_inputs)
    )
    monkeypatch.setattr(linking_pipeline, "storage_glob", lambda _pattern: [])

    def read(path):
        return __import__("pathlib").Path(path).read_text()

    monkeypatch.setattr(linking_run, "storage_read_text", read)
    monkeypatch.setattr(linking_pipeline, "storage_read_text", read)
    monkeypatch.setattr(linking_journal, "storage_read_text", read)

    def write(path: str, value: str) -> None:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    monkeypatch.setattr(linking_journal, "storage_write_text", write)
    monkeypatch.setattr(
        linking_journal,
        "storage_ensure_parent_dir",
        lambda path: (
            __import__("pathlib").Path(path).parent.mkdir(parents=True, exist_ok=True)
        ),
    )
    monkeypatch.setattr(linking_journal, "storage_open_file", open)

    def exists(path):
        return __import__("pathlib").Path(path).exists()

    monkeypatch.setattr(linking_run, "storage_exists", exists)
    monkeypatch.setattr(linking_journal, "storage_exists", exists)

    created = False
    search_calls = 0
    decision_calls = 0
    cancelled_decisions = 0
    both_started = asyncio.Event()
    candidate = _candidate().model_copy(update={"name": "Ada Lovelace"})

    async def search(_client, _headers, _source):
        nonlocal search_calls
        search_calls += 1
        return []

    async def decide(source, candidates, **_kwargs):
        nonlocal decision_calls, cancelled_decisions
        decision_calls += 1
        if decision_calls == 2:
            both_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled_decisions += 1
                raise
        await both_started.wait()
        return (
            EntityLinkDecision(
                action="merge" if candidates else "create",
                candidate_index=1 if candidates else None,
                name=source.name,
                aliases=source.aliases,
                short_description="Český popis.",
                long_description="Český popis.",
            ),
            None,
        )

    async def apply(link, *, checkpoint, **_kwargs):
        nonlocal created
        if link.decision.action == "create":
            assert not created
            created = True
            link.selected_entity_id = candidate.entity_id
            link.selected_entity_after = entity_linking.EntitySnapshot(
                entity_id=candidate.entity_id,
                entity_type="person",
                name=candidate.name,
                aliases=candidate.aliases,
            )
            link.mutations.append(
                MutationAudit(
                    operation="create_entity",
                    resource_id=candidate.entity_id,
                    endpoint="/api/v1/entities",
                    after={"entity_id": candidate.entity_id},
                    applied_at="2026-01-01T00:00:00+00:00",
                )
            )
        else:
            assert created
            link.mutations.append(
                MutationAudit(
                    operation="update_entity",
                    resource_id=candidate.entity_id,
                    endpoint=f"/api/v1/entities/{candidate.entity_id}",
                    before={"entity_id": candidate.entity_id},
                    after={"entity_id": candidate.entity_id},
                    applied_at="2026-01-01T00:00:01+00:00",
                )
            )
        await checkpoint()

    async def lookup(*_args, **_kwargs):
        return "source-id"

    monkeypatch.setattr(linking_run, "search_entity_candidates", search)
    monkeypatch.setattr(linking_run, "llm_decide_entity_link_async", decide)
    monkeypatch.setattr(linking_run, "apply_linked_entity_result", apply)
    monkeypatch.setattr(linking_run, "lookup_page_source", lookup)

    async def run():
        async with httpx.AsyncClient() as api:
            return await entity_linking.run_ner_entity_linking_pipeline(
                config=EntityLinkingConfig(bearer_token="token"),
                input_prefix=input_prefix,
                output_prefix=output_prefix,
                api_client=api,
                llm_client=object(),
                parallel_pages=2,
            )

    summary = asyncio.run(run())
    actions = []
    for page_id in page_ids:
        output = PageNerEntityLinks.model_validate_json(
            (output_prefix / "lib" / "doc" / f"{page_id}.json").read_text()
        )
        actions.extend(link.decision.action for link in output.links)
    assert actions == ["create", "merge"]
    assert search_calls == 3
    assert decision_calls == 3
    assert cancelled_decisions == 1
    assert summary.processed_pages == len(page_ids)
    assert summary.entity_mutations == 2
    import json

    records = [
        json.loads(line)
        for line in (output_prefix / "modifications.jsonl").read_text().splitlines()
    ]
    assert [record["operation"] for record in records] == [
        "create_entity",
        "update_entity",
    ]
    assert records[0]["before"] is None
    assert records[1]["before"] == {"entity_id": candidate.entity_id}
    assert records[1]["after"] == {"entity_id": candidate.entity_id}
    output_paths = [
        str(output_prefix / "lib" / "doc" / f"{page_id}.json") for page_id in page_ids
    ]
    monkeypatch.setattr(linking_pipeline, "storage_glob", lambda _pattern: output_paths)
    resumed = asyncio.run(run())
    assert resumed.skipped_pages == len(page_ids)
    assert len((output_prefix / "modifications.jsonl").read_text().splitlines()) == 2


def test_empty_cluster_page_is_checkpointed_without_llm(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from people_gator_common.pipelines.llm_ner import PageInput
    from people_gator_common.pipelines.ner_clusters import PageNerClusters

    source_path = tmp_path / "input" / "lib" / "doc" / "page.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        PageNerClusters(
            library="lib", document_id="doc", page_id="page", clusters=[]
        ).model_dump_json()
    )
    monkeypatch.setattr(
        linking_pipeline,
        "_iter_cluster_pages",
        lambda _prefix: iter(
            [
                PageInput(
                    library="lib",
                    document_id="doc",
                    page_id="page",
                    path=str(source_path),
                )
            ]
        ),
    )
    monkeypatch.setattr(linking_pipeline, "storage_glob", lambda _pattern: [])
    monkeypatch.setattr(linking_run, "storage_exists", lambda path: Path(path).exists())
    monkeypatch.setattr(
        linking_journal, "storage_exists", lambda path: Path(path).exists()
    )
    monkeypatch.setattr(
        linking_run, "storage_read_text", lambda path: Path(path).read_text()
    )
    monkeypatch.setattr(
        linking_pipeline, "storage_read_text", lambda path: Path(path).read_text()
    )
    monkeypatch.setattr(
        linking_journal, "storage_read_text", lambda path: Path(path).read_text()
    )

    def write(path: str, value: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    monkeypatch.setattr(linking_journal, "storage_write_text", write)

    async def run():
        async with httpx.AsyncClient() as api:
            return await entity_linking.run_ner_entity_linking_pipeline(
                config=EntityLinkingConfig(bearer_token="token"),
                input_prefix=tmp_path / "input",
                output_prefix=tmp_path / "output",
                mode="decision-only",
                api_client=api,
                llm_client=object(),
            )

    summary = asyncio.run(run())
    output = PageNerEntityLinks.model_validate_json(
        (tmp_path / "output" / "lib" / "doc" / "page.json").read_text()
    )
    assert summary.processed_pages == 1
    assert summary.source_entities == 0
    assert output.links == []
    assert output.completed_at is not None


def test_incomplete_page_resumes_recorded_create_without_repeating_api_call(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    from people_gator_common.pipelines.llm_ner import PageInput
    from people_gator_common.pipelines.ner_clusters import (
        DisambiguationOutput,
        PageNerClusters,
    )

    cluster = _cluster().model_copy(update={"entities": None})
    source_path = tmp_path / "input" / "lib" / "doc" / "page.json"
    output_path = tmp_path / "output" / "lib" / "doc" / "page.json"
    source_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    source_path.write_text(
        PageNerClusters(
            library="lib",
            document_id="doc",
            page_id="page",
            clusters=[
                DisambiguationOutput(
                    text="", prompt="", reasoning="", raw_output="", clusters=[cluster]
                )
            ],
        ).model_dump_json()
    )
    entity_id = _candidate().entity_id
    prior = PageNerEntityLinks(
        library="lib",
        document_id="doc",
        page_id="page",
        mode="apply",
        source_id="44444444-4444-4444-4444-444444444444",
        links=[
            LinkedEntityResult(
                source_cluster=cluster,
                source=_source(),
                candidates=[],
                decision=EntityLinkDecision(
                    action="create",
                    name="Ada Lovelace",
                    aliases=[],
                    short_description="Český popis.",
                    long_description="Český popis.",
                ),
                selected_entity_id=entity_id,
                selected_entity_after=entity_linking.EntitySnapshot(
                    entity_id=entity_id, entity_type="person", name="Ada Lovelace"
                ),
                mutations=[
                    MutationAudit(
                        operation="create_entity",
                        resource_id=entity_id,
                        endpoint="/api/v1/entities",
                        after={"entity_id": entity_id},
                        applied_at="2026-01-01T00:00:00+00:00",
                    )
                ],
            )
        ],
    )
    output_path.write_text(prior.model_dump_json())
    monkeypatch.setattr(
        linking_pipeline,
        "_iter_cluster_pages",
        lambda _prefix: iter(
            [
                PageInput(
                    library="lib",
                    document_id="doc",
                    page_id="page",
                    path=str(source_path),
                )
            ]
        ),
    )
    monkeypatch.setattr(
        linking_pipeline, "storage_glob", lambda _pattern: [str(output_path)]
    )
    monkeypatch.setattr(linking_run, "storage_exists", lambda path: Path(path).exists())
    monkeypatch.setattr(
        linking_journal, "storage_exists", lambda path: Path(path).exists()
    )
    monkeypatch.setattr(
        linking_run, "storage_read_text", lambda path: Path(path).read_text()
    )
    monkeypatch.setattr(
        linking_pipeline, "storage_read_text", lambda path: Path(path).read_text()
    )
    monkeypatch.setattr(
        linking_journal, "storage_read_text", lambda path: Path(path).read_text()
    )
    monkeypatch.setattr(
        linking_journal,
        "storage_ensure_parent_dir",
        lambda path: Path(path).parent.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(linking_journal, "storage_open_file", open)

    def write(path: str, value: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    monkeypatch.setattr(linking_journal, "storage_write_text", write)

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("recorded create must not be repeated")

    monkeypatch.setattr(linking_run, "search_entity_candidates", unexpected)
    monkeypatch.setattr(linking_run, "llm_decide_entity_link_async", unexpected)
    monkeypatch.setattr(linking_run, "apply_linked_entity_result", unexpected)

    async def run():
        async with httpx.AsyncClient() as api:
            return await entity_linking.run_ner_entity_linking_pipeline(
                config=EntityLinkingConfig(bearer_token="token"),
                input_prefix=tmp_path / "input",
                output_prefix=tmp_path / "output",
                api_client=api,
                llm_client=object(),
            )

    summary = asyncio.run(run())
    resumed = PageNerEntityLinks.model_validate_json(output_path.read_text())
    assert summary.processed_pages == 1
    assert summary.entity_mutations == 0
    assert resumed.completed_at is not None
    assert (
        len((tmp_path / "output" / "modifications.jsonl").read_text().splitlines()) == 1
    )
