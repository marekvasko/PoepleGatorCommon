from __future__ import annotations

from types import SimpleNamespace

from people_gator_common.pipelines.llm_ner import NamedEntity
from people_gator_common.pipelines.ner_clusters import llm_disambiguate_entities


class FakeOpenRouter:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompt = ""
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **kwargs: object) -> SimpleNamespace:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.prompt = messages[0]["content"]
        message = SimpleNamespace(content=self._response, reasoning="")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _entity(name: str, info: str) -> NamedEntity:
    return NamedEntity(
        name=name,
        type="beeing",
        mentions=[],
        info=info,
    )


def test_singleton_copies_information_and_uses_its_exact_index() -> None:
    client = FakeOpenRouter("")

    clusters, llm = llm_disambiguate_entities(
        [_entity("Ada Lovelace", "English mathematician.")],
        client=client,
    )

    assert llm is None
    assert len(clusters) == 1
    assert clusters[0].entity_indices == [1]
    assert clusters[0].short_description == "English mathematician."
    assert clusters[0].long_description == "English mathematician."
    assert "aliases" not in clusters[0].model_dump()


def test_prompt_and_response_use_the_same_one_based_entity_ids() -> None:
    client = FakeOpenRouter(
        """{
            "clusters": [{
                "name": "Ada Lovelace",
                "short_description": "English mathematician.",
                "long_description": "Ada Lovelace was an English mathematician.",
                "entity_indices": [1, 2]
            }]
        }"""
    )
    first = _entity("Ada Lovelace", "English mathematician.")
    second = _entity("A. Lovelace", "Wrote about the Analytical Engine.")

    clusters, _ = llm_disambiguate_entities([first, second], client=client)

    assert '"index": 1' in client.prompt
    assert '"index": 2' in client.prompt
    assert '"index": 0' not in client.prompt
    assert clusters[0].entity_indices == [1, 2]
    assert clusters[0].entities == [first, second]


def test_duplicate_and_missing_ids_are_recovered_as_singletons() -> None:
    client = FakeOpenRouter(
        """{
            "clusters": [{
                "name": "Ada Lovelace",
                "aliases": ["invented output alias"],
                "short_description": "Incorrect replacement.",
                "long_description": "Incorrect replacement.",
                "entity_indices": [1, 1]
            }]
        }"""
    )
    first = _entity("Ada Lovelace", "English mathematician.")
    second = _entity("Grace Hopper", "American computer scientist.")

    clusters, _ = llm_disambiguate_entities([first, second], client=client)

    assert [cluster.entity_indices for cluster in clusters] == [[1], [2]]
    assert clusters[0].short_description == first.info
    assert clusters[0].long_description == first.info
    assert clusters[1].short_description == second.info
    assert clusters[1].long_description == second.info
    assert all("aliases" not in cluster.model_dump() for cluster in clusters)
