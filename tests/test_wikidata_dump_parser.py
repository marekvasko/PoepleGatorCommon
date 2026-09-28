from __future__ import annotations

import csv
import gzip
import io
import json
from datetime import UTC, datetime

import pytest

from people_gator_common.wikidata.dump_parser import (
    FilterCondition,
    extract_parameter_value,
    extract_parameter_values,
    filter_wikidata_item,
    gather_claim,
    parse_wikidata_dump,
    parse_wikidata_item,
    save_parameter_value,
)

ITEM_LINE = r"""{"id":"Q42","type":"item","labels":{},"descriptions":{},"aliases":{},"claims":{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datavalue":{"value":{"entity-type":"item","numeric-id":5,"id":"Q5"},"type":"wikibase-entityid"},"datatype":"wikibase-item"},"type":"statement","id":"Q42$1","rank":"normal"}],"P569":[{"mainsnak":{"snaktype":"value","property":"P569","datavalue":{"value":{"time":"+1952-03-11T00:00:00Z","timezone":0,"before":0,"after":0,"precision":11,"calendarmodel":"http://www.wikidata.org/entity/Q1985727"},"type":"time"},"datatype":"time"},"type":"statement","id":"Q42$2","rank":"normal"}],"P214":[{"mainsnak":{"snaktype":"value","property":"P214","datavalue":{"value":"113230702","type":"string"},"datatype":"external-id"},"type":"statement","id":"Q42$3","rank":"normal"},{"mainsnak":{"snaktype":"value","property":"P214","datavalue":{"value":"999","type":"string"},"datatype":"external-id"},"type":"statement","id":"Q42$4","rank":"normal"}],"P2048":[{"mainsnak":{"snaktype":"value","property":"P2048","datavalue":{"value":{"amount":"+1.85","unit":"http://www.wikidata.org/entity/Q11573"},"type":"quantity"},"datatype":"quantity"},"type":"statement","id":"Q42$5","rank":"normal"}],"P570":[{"mainsnak":{"snaktype":"novalue","property":"P570","datatype":"time"},"type":"statement","id":"Q42$6","rank":"normal"}]},"sitelinks":{},"pageid":1,"ns":0,"title":"Q42","lastrevid":2,"modified":"2020-01-01T00:00:00Z"},"""


def test_parse_dump_line_and_extract_typed_values() -> None:
    item = parse_wikidata_item(ITEM_LINE)

    assert item.id == "Q42"
    assert extract_parameter_value(item, "P31") == "Q5"
    assert extract_parameter_values(item, "P214") == ["113230702", "999"]
    assert extract_parameter_value(item, "P2048") == 1.85
    assert extract_parameter_value(item, "P569") == datetime(1952, 3, 11, tzinfo=UTC)
    assert extract_parameter_value(item, "P570") is None
    assert extract_parameter_value(item, "P999") is None


def test_filter_matches_any_claim_and_can_be_inverted() -> None:
    item = parse_wikidata_item(ITEM_LINE)

    assert filter_wikidata_item(item, FilterCondition.from_expression("P31=Q5"))
    assert filter_wikidata_item(item, FilterCondition(parameter="P214", value="999"))
    assert filter_wikidata_item(
        item, FilterCondition(parameter="P31", value="Q215627", include=False)
    )
    assert not filter_wikidata_item(
        item, FilterCondition(parameter="P31", value="Q5", include=False)
    )
    assert filter_wikidata_item(item, "P31", "Q5")
    assert gather_claim(item, "P214") == "113230702"


def test_save_parameter_value_writes_valid_csv_with_one_header(tmp_path) -> None:
    path = tmp_path / "P214.csv"
    save_parameter_value(path, "Q42", "113,230\n702")
    save_parameter_value(path, "Q1", 42)

    with path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["wikidata_id", "parameter_value"],
            ["Q42", "113,230\n702"],
            ["Q1", "42"],
        ]


def test_save_parameter_value_accepts_open_stream() -> None:
    stream = io.StringIO()
    save_parameter_value(stream, "Q42", {"latitude": 1.0, "longitude": 2.0})

    stream.seek(0)
    assert list(csv.reader(stream)) == [
        ["wikidata_id", "parameter_value"],
        ["Q42", '{"latitude":1.0,"longitude":2.0}'],
    ]


@pytest.mark.parametrize("line", ["", "[", "]", b"  "])
def test_parse_rejects_non_entity_lines(line: str | bytes) -> None:
    with pytest.raises(ValueError, match="does not contain"):
        parse_wikidata_item(line)


def test_parse_tolerates_claim_with_missing_datatype() -> None:
    raw = json.loads(ITEM_LINE.rstrip(","))
    del raw["claims"]["P214"][0]["mainsnak"]["datatype"]

    item = parse_wikidata_item(json.dumps(raw))

    assert item.claims["P214"][0].mainsnak.datatype is None
    assert extract_parameter_value(item, "P214") == {
        "value": "113230702",
        "type": "string",
    }


@pytest.mark.parametrize("workers", [1, 2])
def test_parse_dump_streams_claims_and_localized_text(tmp_path, workers: int) -> None:
    first = json.loads(ITEM_LINE.rstrip(","))
    first["labels"] = {
        "en": {"language": "en", "value": "Douglas Adams"},
        "cs": {"language": "cs", "value": "Douglas Adams"},
    }
    first["descriptions"] = {"en": {"language": "en", "value": "English writer"}}
    first["aliases"] = {
        "en": [
            {"language": "en", "value": "DNA"},
            {"language": "en", "value": "Douglas Noel Adams"},
        ]
    }
    second = json.loads(ITEM_LINE.rstrip(","))
    second["id"] = "Q1"
    second["claims"]["P31"][0]["mainsnak"]["datavalue"]["value"]["id"] = "Q215627"

    dump_path = tmp_path / "wikidata.json.gz"
    with gzip.open(dump_path, "wt", encoding="utf-8") as stream:
        stream.write("[\n")
        stream.write(json.dumps(first, ensure_ascii=False) + ",\n")
        stream.write(json.dumps(second, ensure_ascii=False) + "\n")
        stream.write("]\n")

    output = tmp_path / "out"
    stats = parse_wikidata_dump(
        dump_path,
        output,
        ["P214", "P569"],
        filters=[FilterCondition(parameter="P31", value="Q5")],
        workers=workers,
    )

    assert stats.items_seen == 2
    assert stats.items_matched == 1
    assert stats.rows_written == {
        "P214": 2,
        "P569": 1,
        "labels": 2,
        "descriptions": 1,
        "aliases": 2,
    }
    with (output / "P214.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["entity", "value"],
            ["Q42", "113230702"],
            ["Q42", "999"],
        ]
    with (output / "labels.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["entity", "language", "value"],
            ["Q42", "en", "Douglas Adams"],
            ["Q42", "cs", "Douglas Adams"],
        ]
    with (output / "aliases.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["entity", "language", "value"],
            ["Q42", "en", "DNA"],
            ["Q42", "en", "Douglas Noel Adams"],
        ]
