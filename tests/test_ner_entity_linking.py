from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

from people_gator_common.pipelines.llm_ner import NamedEntity, NamedEntityMention
from people_gator_common.pipelines.ner_clusters import DisambiguatedNamedEntity
from people_gator_common.pipelines.ner_entity_linking import (
    EntityCandidate,
    EntityLinkingConfig,
    LinkableEntity,
    authenticate_entity_linking_api,
    link_source_cluster,
    llm_decide_entity_link,
    search_entity_candidates,
    source_entity_from_cluster,
)


class FakeOpenRouter:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0
        self.prompt = ""
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.prompt = messages[0]["content"]
        message = SimpleNamespace(content=self._response, reasoning="")
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


def test_search_uses_only_repeated_q_and_hydrates_entity_details() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer token"
        if request.url.path == "/api/v1/search-entities":
            assert request.url.params.multi_items() == [
                ("q", "Ada Lovelace"),
                ("q", "A. Lovelace"),
                ("q", "English mathematician."),
                (
                    "q",
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


def test_missing_translated_descriptions_are_rejected() -> None:
    client = FakeOpenRouter(
        '{"action":"create","candidate_index":null,'
        '"short_description":"","long_description":""}'
    )

    with pytest.raises(RuntimeError, match="omitted Czech descriptions"):
        llm_decide_entity_link(_source(), [], client=client)


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
