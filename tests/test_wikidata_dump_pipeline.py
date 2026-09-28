from __future__ import annotations

import csv
import gzip
import json
import logging
from pathlib import Path

import pytest

from people_gator_common.pipelines.wikidata_dump import (
    WikidataDumpPipelineConfig,
    load_wikidata_dump_config,
    run_wikidata_dump_pipeline,
)


def _claim(property_id: str, value: str, datatype: str, rank: str = "normal") -> dict:
    data_value: dict[str, object]
    if datatype == "wikibase-item":
        data_value = {
            "value": {"entity-type": "item", "id": value},
            "type": "wikibase-entityid",
        }
    else:
        data_value = {"value": value, "type": "string"}
    return {
        "mainsnak": {
            "snaktype": "value",
            "property": property_id,
            "datavalue": data_value,
            "datatype": datatype,
        },
        "type": "statement",
        "id": f"Q42${property_id}-{value}",
        "rank": rank,
    }


def _item() -> dict:
    return {
        "id": "Q42",
        "type": "item",
        "labels": {"en": {"language": "en", "value": "Douglas Adams"}},
        "descriptions": {"en": {"language": "en", "value": "English writer"}},
        "aliases": {
            "en": [
                {"language": "en", "value": "DNA"},
                {"language": "en", "value": "Douglas Noel Adams"},
            ]
        },
        "claims": {
            "P31": [_claim("P31", "Q5", "wikibase-item")],
            "P27": [
                _claim("P27", "Q213", "wikibase-item"),
                _claim("P27", "Q213", "wikibase-item"),
            ],
            "P214": [_claim("P214", "113230702", "external-id")],
        },
        "sitelinks": {},
    }


@pytest.mark.parametrize("workers", [1, 2])
def test_configured_pipeline_writes_v2_compatible_outputs(
    tmp_path: Path, workers: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    dump = tmp_path / "dump.json.gz"
    with gzip.open(dump, "wt", encoding="utf-8") as stream:
        stream.write("[\n")
        stream.write(json.dumps(_item()) + "\n")
        stream.write("]\n")

    output = tmp_path / "output"
    config = WikidataDumpPipelineConfig.model_validate(
        {
            "input_path": str(dump),
            "output_directory": str(output),
            "workers": workers,
            "progress_every_items": 1,
            "groups": {
                "people": {
                    "filters": [{"parameter": "P31", "value": "Q5"}],
                    "entities_output": "people_of_all.csv",
                    "claims": [
                        {
                            "claim": "P27",
                            "output": "people_of_all_P27.csv",
                            "unique_output": "people_of_all_unique_P27.csv",
                        },
                        {"claim": "P214", "output": "people_of_all_P214.csv"},
                    ],
                    "text": {
                        "labels": "people_of_all_labels.csv",
                        "descriptions": "people_of_all_descriptions.csv",
                        "aliases": "people_of_all_alt_labels.csv",
                    },
                }
            },
        }
    )

    stats = run_wikidata_dump_pipeline(config)

    assert stats.items_seen == 1
    assert stats.matched_by_group == {"people": 1}
    assert "Starting Wikidata dump extraction" in caplog.text
    assert "Dump progress: items=1, last_entity=Q42" in caplog.text
    assert "Wikidata dump extraction completed" in caplog.text
    with (output / "people_of_all.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["entity"],
            ["http://www.wikidata.org/entity/Q42"],
        ]
    with (output / "people_of_all_P27.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["entity", "value"],
            [
                "http://www.wikidata.org/entity/Q42",
                "http://www.wikidata.org/entity/Q213",
            ],
        ]
    with (output / "people_of_all_unique_P27.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["entity"],
            ["http://www.wikidata.org/entity/Q213"],
        ]
    with (output / "people_of_all_labels.csv").open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["entity", "entityText", "entityTextLanguage"],
            ["http://www.wikidata.org/entity/Q42", "Douglas Adams", "en"],
        ]
    with (output / "people_of_all_alt_labels.csv").open(newline="") as stream:
        assert list(csv.reader(stream))[1:] == [
            ["http://www.wikidata.org/entity/Q42", "DNA", "en"],
            [
                "http://www.wikidata.org/entity/Q42",
                "Douglas Noel Adams",
                "en",
            ],
        ]


def test_repository_v2_compatibility_config_is_valid() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "wikidata_dump_v2.yaml"
    config = load_wikidata_dump_config(config_path)

    assert [claim.claim for claim in config.groups["people"].claims] == [
        "P18",
        "P27",
        "P1412",
        "P214",
        "P691",
        "P569",
        "P570",
    ]
    assert config.groups["characters"].text.aliases == (
        "characters_of_all_alt_labels.csv"
    )


def test_pipeline_resumes_inclusively_and_rebuilds_unique_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    first = _item()
    second = _item()
    second["id"] = "Q43"
    second["claims"]["P27"] = [_claim("P27", "Q30", "wikibase-item")]
    dump = tmp_path / "dump.json.gz"
    with gzip.open(dump, "wt", encoding="utf-8") as stream:
        stream.write("[\n")
        stream.write(json.dumps(first) + ",\n")
        stream.write(json.dumps(second) + "\n")
        stream.write("]\n")

    output = tmp_path / "output"
    output.mkdir()
    (output / "people.csv").write_text(
        "entity\nhttp://www.wikidata.org/entity/Q42\n", encoding="utf-8"
    )
    (output / "countries.csv").write_text(
        "entity,value\n"
        "http://www.wikidata.org/entity/Q42,http://www.wikidata.org/entity/Q213\n",
        encoding="utf-8",
    )
    config = WikidataDumpPipelineConfig.model_validate(
        {
            "input_path": str(dump),
            "output_directory": str(output),
            "resume_from_line": 3,
            "groups": {
                "people": {
                    "filters": [{"parameter": "P31", "value": "Q5"}],
                    "entities_output": "people.csv",
                    "claims": [
                        {
                            "claim": "P27",
                            "output": "countries.csv",
                            "unique_output": "unique_countries.csv",
                        }
                    ],
                }
            },
        }
    )

    stats = run_wikidata_dump_pipeline(config)

    assert stats.items_seen == 1
    assert "Fast-skipping dump content before physical line 3" in caplog.text
    with (output / "people.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["entity"],
            ["http://www.wikidata.org/entity/Q42"],
            ["http://www.wikidata.org/entity/Q43"],
        ]
    with (output / "unique_countries.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        assert list(csv.reader(stream)) == [
            ["entity"],
            ["http://www.wikidata.org/entity/Q213"],
            ["http://www.wikidata.org/entity/Q30"],
        ]
