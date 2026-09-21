from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fsspec.implementations.local import LocalFileSystem

import people_gator_common.filesystem as storage
from people_gator_common.pipelines.llm_ner import (
    chunk_lines,
    find_exact_context_and_name,
    parse_json_result,
    PageNersOutput,
    process_page,
    process_text_lines,
    TextNerOutput,
    run_llm_ner_pipeline,
)


class FakeOpenRouter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.chat = SimpleNamespace(send=self.send)

    def send(self, **_: object) -> SimpleNamespace:
        message = SimpleNamespace(content=next(self._responses), reasoning="")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _page(*transcriptions: str) -> dict[str, object]:
    return {
        "id": "page-1",
        "regions": [
            {
                "id": "region-1",
                "lines": [
                    {"id": f"line-{index}", "transcription": transcription}
                    for index, transcription in enumerate(transcriptions)
                ],
            }
        ],
    }


def _entity(name: str) -> str:
    return json.dumps(
        [
            {
                "name": name,
                "type": "beeing",
                "mentions": [{"name": name, "context": name}],
                "info": "Person in the source text.",
            }
        ]
    )


def test_chunk_lines_tracks_overlap_indices() -> None:
    chunks, indices = chunk_lines(
        ["aaa", "bbb", "ccc"],
        min_characters_in_chunk=5,
        overlap_lines=1,
    )

    assert chunks == ["aaa\nbbb", "bbb\nccc"]
    assert indices == [(0, 2), (1, 3)]


def test_parse_json_result_accepts_fenced_json_and_surrounding_text() -> None:
    assert parse_json_result("result:\n```json\n[{\"name\": \"Ada\"}]\n```\ndone") == [
        {"name": "Ada"}
    ]


def test_find_exact_context_and_name_returns_source_substrings() -> None:
    text = "Before Ada Lovelace wrote notes. After"

    exact_context, exact_name = find_exact_context_and_name(
        text,
        "Ada Lovelace wrote note.",
        "Ava Lovelace",
    )

    assert exact_context == "Ada Lovelace wrote notes."
    assert exact_name == "Ada Lovelace"
    assert exact_context in text
    assert exact_name in exact_context


def test_find_exact_context_and_name_rejects_unrelated_text() -> None:
    assert find_exact_context_and_name("Ada was here.", "unrelated", "Nobody") == (
        None,
        None,
    )


def test_text_processing_has_no_page_or_source_metadata() -> None:
    result = process_text_lines(
        ["Ada"],
        client=FakeOpenRouter([_entity("Ada")]),
    )

    assert isinstance(result, TextNerOutput)
    dumped_chunk = result.chunks[0].model_dump()
    dumped_entity = dumped_chunk["valid_entities"][0]
    assert "line_mapping" not in dumped_chunk
    assert "source" not in dumped_entity
    assert set(dumped_entity) == {"name", "type", "mentions", "info"}
    assert dumped_entity["mentions"] == [
        {
            "context": "Ada",
            "name": "Ada",
            "exact_context": "Ada",
            "exact_name": "Ada",
        }
    ]


def test_page_keeps_metadata_top_level_and_chunk_responses_separate() -> None:
    client = FakeOpenRouter([_entity("Ada"), _entity("Grace")])

    result = process_page(
        _page("aaa", "bbb", "ccc"),
        library="library",
        document_id="document",
        page_id="page-1",
        client=client,
        min_characters_in_chunk=5,
        overlap_lines=1,
    )

    assert isinstance(result, PageNersOutput)
    assert isinstance(result, TextNerOutput)
    assert [chunk.raw_output for chunk in result.chunks] == [
        _entity("Ada"),
        _entity("Grace"),
    ]
    assert [line.line_id for line in result.chunk_metadata[0].line_mapping] == [
        "line-0",
        "line-1",
    ]
    assert [line.line_id for line in result.chunk_metadata[1].line_mapping] == [
        "line-1",
        "line-2",
    ]
    assert "line_mapping" not in result.chunks[0].model_dump()
    assert set(result.chunk_metadata[0].line_mapping[0].model_dump()) == {
        "region_id",
        "line_id",
    }


def test_pipeline_uses_storage_prefixes_and_skips_listed_outputs(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    storage_root = tmp_path / "storage"
    page_path = (
        storage_root
        / "layout-prefix"
        / "library"
        / "document"
        / "page.json"
    )
    page_path.parent.mkdir(parents=True)
    page_path.write_text(json.dumps(_page("Ada")), encoding="utf-8")
    monkeypatch.setattr(storage, "fs", LocalFileSystem())
    monkeypatch.setattr(storage, "prefix", str(storage_root))

    first_summary = run_llm_ner_pipeline(
        input_prefix="layout-prefix",
        output_prefix="ner-prefix",
        client=FakeOpenRouter([_entity("Ada")]),
    )
    output_path = storage_root / "ner-prefix" / "library" / "document" / "page.json"

    assert first_summary.processed_pages == 1
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["chunks"][0]["valid_entities"][0]["name"] == "Ada"
    assert "source" not in output["chunks"][0]["valid_entities"][0]
    assert output["chunk_metadata"][0]["line_mapping"][0]["line_id"] == "line-0"

    second_summary = run_llm_ner_pipeline(
        input_prefix="layout-prefix",
        output_prefix="ner-prefix",
        client=FakeOpenRouter([]),
    )
    assert second_summary.processed_pages == 0
    assert second_summary.skipped_pages == 1
