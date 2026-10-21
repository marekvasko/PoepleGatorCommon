from __future__ import annotations

import json
from pathlib import Path

from fsspec.implementations.local import LocalFileSystem

from people_gator_common import filesystem
from people_gator_common.pipelines.ner_postprocess import soft_link_ner


def test_soft_link_keeps_unmatched_entities_in_separate_clusters(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(filesystem, "fs", LocalFileSystem())
    monkeypatch.setattr(filesystem, "prefix", None)
    records = [
        {
            "document_id": "doc",
            "id": str(index),
            "entity": name,
            "all_name_variants": [],
            "type": "being",
            "information": name,
        }
        for index, name in enumerate(
            ["Ada Lovelace", "Ada Lovelace", "Isaac Newton", "Marie Curie"],
            start=1,
        )
    ]
    input_path = tmp_path / "ner_normalized.jsonl"
    input_path.write_text("".join(json.dumps(record) + "\n" for record in records))

    output_dir = tmp_path / "clusters"
    soft_link_ner(str(input_path), str(output_dir), n_jobs=1)

    clusters = [
        json.loads(line)
        for line in (output_dir / "doc__being.jsonl").read_text().splitlines()
    ]
    assert sorted(sorted(entity["id"] for entity in cluster) for cluster in clusters) == [
        ["1", "2"], ["3"], ["4"],
    ]
