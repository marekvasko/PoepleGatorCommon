from __future__ import annotations

from people_gator_common.pipelines.authority_import import (
    AuthorityRecord,
    SourceKind,
    _build_entity_create_payload,
    _build_source_create_payload,
)


def test_build_source_create_payload_for_wikidata_record() -> None:
    record = AuthorityRecord(
        provider=SourceKind.wikidata,
        external_id="Q42",
        uri="https://www.wikidata.org/entity/Q42",
    )

    payload = _build_source_create_payload(record)

    assert payload["source_type"] == "wikidata"
    assert payload["uri"] == "https://www.wikidata.org/entity/Q42"
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["entity_id"] == "Q42"
    assert metadata["wikidata_url"] == "https://www.wikidata.org/entity/Q42"


def test_build_entity_create_payload_uses_preferred_name_and_description() -> None:
    record = AuthorityRecord(
        provider=SourceKind.nlcr,
        external_id="n123",
        uri="https://example.test/n123",
        names=["Ada Lovelace", "Ada"],
        aliases=["Countess of Lovelace"],
        descriptions=[("English mathematician", "en"), ("Mathematician", "en")],
    )

    payload = _build_entity_create_payload(record)

    assert payload["name"] == "Ada Lovelace"
    assert payload["description"] == "English mathematician"
    assert payload["entity_type"] == "person"
