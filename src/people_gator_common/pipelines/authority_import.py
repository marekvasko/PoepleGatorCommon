from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
import uuid

import fsspec
from pydantic import AnyUrl
from pydantic import BaseModel
from pydantic import TypeAdapter
import weaviate
from weaviate import WeaviateAsyncClient
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from weaviate.collections.classes.types import WeaviateField

from people_gator_common.filesystem import exists as storage_exists
from people_gator_common.filesystem import open_file as storage_open_file

logger = logging.getLogger(__name__)

type MutableProperties = dict[str, WeaviateField]

ANY_URL_ADAPTER = TypeAdapter(AnyUrl)

IMPORTER_NAMESPACE = uuid.UUID("dc40695e-61dc-4da5-b2f0-b6ccb7e6ed3d")
IMPORTER_PRINCIPAL_ID = uuid.uuid5(IMPORTER_NAMESPACE, "principal:authority-importer")

PRINCIPAL_COLLECTION = "Principal"
SOURCE_COLLECTION = "Source"
ENTITY_COLLECTION = "Entity"
ENTITY_NAME_COLLECTION = "EntityName"
DESCRIPTION_COLLECTION = "Description"


class PrincipalType(StrEnum):
    user = "user"
    service_agent = "service_agent"
    model_agent = "model_agent"


class SourceKind(StrEnum):
    wikidata = "wikidata"
    wikimedia = "wikimedia"
    nlcr = "nlcr"


class NlcrAuthoritySourceMetadata(BaseModel):
    nlcr_id: str


class WikidataSourceMetadata(BaseModel):
    entity_id: str
    wikidata_url: AnyUrl


class WikimediaImageSourceMetadata(BaseModel):
    wikimedia_url: AnyUrl


@dataclass(slots=True)
class AuthorityRecord:
    provider: SourceKind
    external_id: str
    uri: str
    names: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    descriptions: list[tuple[str, str]] = field(default_factory=list)
    wikimedia_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ImportStats:
    records_processed: int = 0
    sources_upserted: int = 0
    entities_upserted: int = 0
    names_upserted: int = 0
    descriptions_upserted: int = 0
    wikimedia_sources_upserted: int = 0


@dataclass(slots=True)
class WeaviateConnectionConfig:
    http_host: str = "localhost"
    http_port: int = 8080
    http_secure: bool = False
    grpc_host: str = "localhost"
    grpc_port: int = 50051
    grpc_secure: bool = False
    api_key: str | None = None


def _non_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _normalize_entity_id(raw_value: str) -> str:
    candidate = raw_value.strip()
    if "/" in candidate:
        return candidate.rsplit("/", maxsplit=1)[-1]
    return candidate


def _join_path(base_path: str, filename: str) -> str:
    base = base_path.rstrip("/")
    return f"{base}/{filename}" if base else filename


def _is_uri(path: str) -> bool:
    return "://" in path


def _path_exists(path: str) -> bool:
    if _is_uri(path):
        fs, normalized_path = fsspec.core.url_to_fs(path)
        return fs.exists(normalized_path)
    return storage_exists(path)


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    if _is_uri(path):
        with fsspec.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    with storage_open_file(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _extract_entity_value_row(row: dict[str, str]) -> tuple[str, str] | None:
    entity_raw = row.get("entity")
    value_raw = row.get("value")
    if entity_raw is None or value_raw is None:
        return None
    entity = _non_empty(entity_raw)
    value = _non_empty(value_raw)
    if entity is None or value is None:
        return None
    return _normalize_entity_id(entity), value


def _extract_entity_text_row(row: dict[str, str]) -> tuple[str, str, str] | None:
    entity_raw = row.get("entity")
    text_raw = row.get("entityText")
    language_raw = row.get("entityTextLanguage")
    if entity_raw is None or text_raw is None:
        return None
    entity = _non_empty(entity_raw)
    text = _non_empty(text_raw)
    language = _non_empty(language_raw) or "und"
    if entity is None or text is None:
        return None
    return _normalize_entity_id(entity), text, language


def _deduplicate_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        marker = normalized.lower()
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(normalized)
    return ordered


def _validated_url(value: str) -> AnyUrl:
    return ANY_URL_ADAPTER.validate_python(value)


def _is_valid_url(value: str) -> bool:
    try:
        _ = _validated_url(value)
        return True
    except ValueError:
        return False


def _build_wikidata_records(wikidata_dir: str, prefix: str) -> list[AuthorityRecord]:
    entities_path = _join_path(wikidata_dir, f"{prefix}.csv")
    labels_path = _join_path(wikidata_dir, f"{prefix}_labels.csv")
    descriptions_path = _join_path(wikidata_dir, f"{prefix}_descriptions.csv")
    alt_labels_path = _join_path(wikidata_dir, f"{prefix}_alt_labels.csv")
    image_path = _join_path(wikidata_dir, f"{prefix}_P18.csv")

    if not _path_exists(entities_path):
        raise FileNotFoundError(f"Missing Wikidata entities CSV: {entities_path}")

    by_entity: dict[str, AuthorityRecord] = {}
    for row in _read_csv_rows(entities_path):
        entity_raw = row.get("entity")
        if entity_raw is None:
            continue
        entity_id = _normalize_entity_id(entity_raw)
        if not entity_id:
            continue
        by_entity[entity_id] = AuthorityRecord(
            provider=SourceKind.wikidata,
            external_id=entity_id,
            uri=f"https://www.wikidata.org/entity/{entity_id}",
        )

    if _path_exists(labels_path):
        for row in _read_csv_rows(labels_path):
            extracted = _extract_entity_text_row(row)
            if extracted is None:
                continue
            entity_id, text, _language = extracted
            if entity_id in by_entity:
                by_entity[entity_id].names.append(text)

    if _path_exists(alt_labels_path):
        for row in _read_csv_rows(alt_labels_path):
            extracted = _extract_entity_text_row(row)
            if extracted is None:
                continue
            entity_id, text, _language = extracted
            if entity_id in by_entity:
                by_entity[entity_id].aliases.append(text)

    if _path_exists(descriptions_path):
        for row in _read_csv_rows(descriptions_path):
            extracted = _extract_entity_text_row(row)
            if extracted is None:
                continue
            entity_id, text, language = extracted
            if entity_id in by_entity:
                by_entity[entity_id].descriptions.append((text, language))

    if _path_exists(image_path):
        for row in _read_csv_rows(image_path):
            extracted = _extract_entity_value_row(row)
            if extracted is None:
                continue
            entity_id, image_value = extracted
            if entity_id not in by_entity:
                continue
            if image_value.startswith("http") and _is_valid_url(image_value):
                by_entity[entity_id].wikimedia_urls.append(image_value)

    records = list(by_entity.values())
    for record in records:
        record.names = _deduplicate_strings(record.names)
        record.aliases = _deduplicate_strings(record.aliases)
        record.wikimedia_urls = _deduplicate_strings(record.wikimedia_urls)
        deduped_descriptions: dict[tuple[str, str], None] = {}
        for text, language in record.descriptions:
            key = (text.strip(), language.strip() or "und")
            if not key[0]:
                continue
            deduped_descriptions[key] = None
        record.descriptions = list(deduped_descriptions.keys())
    return records


def _load_pair_csv(path: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    if not _path_exists(path):
        return values
    for row in _read_csv_rows(path):
        entity_raw = row.get("entity")
        value_raw = row.get("value")
        if entity_raw is None or value_raw is None:
            continue
        entity = _non_empty(entity_raw)
        value = _non_empty(value_raw)
        if entity is None or value is None:
            continue
        values[entity].append(value)
    return values


def _build_nlcr_records(nlcr_dir: str) -> list[AuthorityRecord]:
    control_numbers_path = _join_path(nlcr_dir, "unique_control_numbers.csv")
    names_path = _join_path(nlcr_dir, "nlcr_aut__person_names_reparsed.csv")
    aliases_path = _join_path(nlcr_dir, "nlcr_aut__also_known_as_reparsed.csv")
    descriptions_path = _join_path(nlcr_dir, "nlcr_aut__descriptions.csv")

    if not _path_exists(control_numbers_path):
        raise FileNotFoundError(f"Missing NLCR IDs CSV: {control_numbers_path}")

    names_by_entity = _load_pair_csv(names_path)
    aliases_by_entity = _load_pair_csv(aliases_path)
    descriptions_by_entity = _load_pair_csv(descriptions_path)

    records: list[AuthorityRecord] = []
    for row in _read_csv_rows(control_numbers_path):
        value_raw = row.get("value")
        if value_raw is None:
            continue
        nlcr_id = _non_empty(value_raw)
        if nlcr_id is None:
            continue

        record = AuthorityRecord(
            provider=SourceKind.nlcr,
            external_id=nlcr_id,
            uri=f"https://aleph.nkp.cz/F/?func=find-c&local_base=aut&ccl_term=ica={nlcr_id}",
            names=_deduplicate_strings(names_by_entity.get(nlcr_id, [])),
            aliases=_deduplicate_strings(aliases_by_entity.get(nlcr_id, [])),
            descriptions=[
                (item, "cs")
                for item in descriptions_by_entity.get(nlcr_id, [])
                if item.strip()
            ],
        )
        records.append(record)

    return records


def _source_uuid(provider: SourceKind, uri: str) -> uuid.UUID:
    return uuid.uuid5(IMPORTER_NAMESPACE, f"source:{provider.value}:{uri}")


def _entity_uuid(provider: SourceKind, external_id: str) -> uuid.UUID:
    return uuid.uuid5(IMPORTER_NAMESPACE, f"entity:{provider.value}:{external_id}")


def _entity_name_uuid(entity_id: uuid.UUID, name: str) -> uuid.UUID:
    return uuid.uuid5(IMPORTER_NAMESPACE, f"entity_name:{entity_id}:{name.lower()}")


def _description_uuid(entity_id: uuid.UUID, language: str, text: str) -> uuid.UUID:
    return uuid.uuid5(IMPORTER_NAMESPACE, f"description:{entity_id}:{language.lower()}:{text.lower()}")


async def _fetch_by_id(
    client: WeaviateAsyncClient,
    collection_name: str,
    id_property: str,
    object_id: uuid.UUID,
) -> object | None:
    response = await client.collections.get(collection_name).query.fetch_objects(
        filters=Filter.by_property(id_property).equal(str(object_id)),
        limit=1,
    )
    objects = getattr(response, "objects")
    if not objects:
        return None
    return objects[0]


async def _upsert(
    client: WeaviateAsyncClient,
    collection_name: str,
    object_uuid: uuid.UUID,
    *,
    id_property: str,
    properties: MutableProperties,
    references: dict[str, uuid.UUID] | None = None,
) -> None:
    collection = client.collections.get(collection_name)
    current = await _fetch_by_id(client, collection_name, id_property, object_uuid)
    if current is None:
        await collection.data.insert(
            uuid=object_uuid,
            properties=properties,
            references=references,
        )
        return
    await collection.data.replace(
        uuid=object_uuid,
        properties=properties,
        references=references,
    )


def _build_weaviate_client(config: WeaviateConnectionConfig) -> WeaviateAsyncClient:
    auth = Auth.api_key(config.api_key) if config.api_key else None
    return weaviate.use_async_with_custom(
        http_host=config.http_host,
        http_port=config.http_port,
        http_secure=config.http_secure,
        grpc_host=config.grpc_host,
        grpc_port=config.grpc_port,
        grpc_secure=config.grpc_secure,
        auth_credentials=auth,
    )


async def _ensure_importer_principal(client: WeaviateAsyncClient) -> uuid.UUID:
    existing = await _fetch_by_id(
        client,
        PRINCIPAL_COLLECTION,
        "principal_id",
        IMPORTER_PRINCIPAL_ID,
    )
    if existing is not None:
        return IMPORTER_PRINCIPAL_ID

    await _upsert(
        client,
        PRINCIPAL_COLLECTION,
        IMPORTER_PRINCIPAL_ID,
        id_property="principal_id",
        properties={
            "principal_id": str(IMPORTER_PRINCIPAL_ID),
            "principal_type": PrincipalType.service_agent.value,
            "display_name": "Authority Importer",
            "is_active": True,
            "external_subject": None,
            "external_issuer": None,
            "email": None,
            "annotation_type": None,
            "role": "authority-importer",
            "model_slug": None,
            "model_version": None,
            "registry_source": None,
            "registry_uri": None,
        },
    )
    return IMPORTER_PRINCIPAL_ID


async def _upsert_source_for_record(client: WeaviateAsyncClient, record: AuthorityRecord) -> uuid.UUID:
    source_id = _source_uuid(record.provider, record.uri)
    if record.provider is SourceKind.wikidata:
        metadata = WikidataSourceMetadata(
            entity_id=record.external_id,
            wikidata_url=_validated_url(record.uri),
        )
    elif record.provider is SourceKind.nlcr:
        metadata = NlcrAuthoritySourceMetadata(nlcr_id=record.external_id)
    else:
        raise ValueError(f"Unsupported provider for main record source: {record.provider.value}")

    await _upsert(
        client,
        SOURCE_COLLECTION,
        source_id,
        id_property="source_id",
        properties={
            "source_id": str(source_id),
            "source_type": record.provider.value,
            "uri": record.uri,
            "metadata": metadata.model_dump(mode="json"),
        },
    )
    return source_id


async def _upsert_wikimedia_source(client: WeaviateAsyncClient, image_url: str) -> None:
    source_id = _source_uuid(SourceKind.wikimedia, image_url)
    metadata = WikimediaImageSourceMetadata(wikimedia_url=_validated_url(image_url))
    await _upsert(
        client,
        SOURCE_COLLECTION,
        source_id,
        id_property="source_id",
        properties={
            "source_id": str(source_id),
            "source_type": SourceKind.wikimedia.value,
            "uri": image_url,
            "metadata": metadata.model_dump(mode="json"),
        },
    )


async def _upsert_entity(
    client: WeaviateAsyncClient,
    record: AuthorityRecord,
    principal_id: uuid.UUID,
) -> uuid.UUID:
    entity_id = _entity_uuid(record.provider, record.external_id)
    preferred_name = record.names[0] if record.names else (record.aliases[0] if record.aliases else None)
    preferred_description = record.descriptions[0][0] if record.descriptions else None

    wikidata_id: str | None = None
    nlcr_authority_id: str | None = None
    if record.provider is SourceKind.wikidata:
        wikidata_id = record.external_id
    elif record.provider is SourceKind.nlcr:
        nlcr_authority_id = record.external_id

    await _upsert(
        client,
        ENTITY_COLLECTION,
        entity_id,
        id_property="entity_id",
        properties={
            "entity_id": str(entity_id),
            "principal_id": str(principal_id),
            "entity_type": "person",
            "name": preferred_name,
            "description": preferred_description,
            "wikidata_id": wikidata_id,
            "nlcr_authority_id": nlcr_authority_id,
        },
        references={"principal": principal_id},
    )
    return entity_id


async def _upsert_entity_names(
    client: WeaviateAsyncClient,
    entity_id: uuid.UUID,
    source_id: uuid.UUID,
    names: list[str],
) -> int:
    inserted = 0
    for name in _deduplicate_strings(names):
        entity_name_id = _entity_name_uuid(entity_id, name)
        await _upsert(
            client,
            ENTITY_NAME_COLLECTION,
            entity_name_id,
            id_property="entity_name_id",
            properties={
                "entity_name_id": str(entity_name_id),
                "name": name,
                "source_id": str(source_id),
                "entity_id": str(entity_id),
            },
            references={"entity": entity_id, "source": source_id},
        )
        inserted += 1
    return inserted


async def _upsert_descriptions(
    client: WeaviateAsyncClient,
    entity_id: uuid.UUID,
    source_id: uuid.UUID,
    descriptions: list[tuple[str, str]],
) -> int:
    inserted = 0
    seen: set[tuple[str, str]] = set()
    for text, language in descriptions:
        normalized_text = text.strip()
        normalized_language = language.strip() or "und"
        key = (normalized_text.lower(), normalized_language.lower())
        if not normalized_text or key in seen:
            continue
        seen.add(key)

        description_id = _description_uuid(entity_id, normalized_language, normalized_text)
        await _upsert(
            client,
            DESCRIPTION_COLLECTION,
            description_id,
            id_property="description_id",
            properties={
                "description_id": str(description_id),
                "text": normalized_text,
                "language": normalized_language,
                "source_id": str(source_id),
                "entity_id": str(entity_id),
            },
            references={"entity": entity_id, "source": source_id},
        )
        inserted += 1
    return inserted


def _build_records(
    wikidata_dir: str | None,
    wikidata_prefix: str,
    nlcr_dir: str | None,
) -> list[AuthorityRecord]:
    records: list[AuthorityRecord] = []
    if wikidata_dir is not None:
        records.extend(_build_wikidata_records(wikidata_dir, wikidata_prefix))
    if nlcr_dir is not None:
        records.extend(_build_nlcr_records(nlcr_dir))
    if not records:
        raise ValueError("No records to import. Provide wikidata_dir and/or nlcr_dir.")
    return records


async def _import_records(
    client: WeaviateAsyncClient,
    records: list[AuthorityRecord],
    max_records: int | None,
) -> ImportStats:
    stats = ImportStats()
    principal_id = await _ensure_importer_principal(client)
    import_records = records[:max_records] if max_records is not None else records

    for index, record in enumerate(import_records, start=1):
        logger.info(
            "Importing record %d/%d: %s:%s",
            index,
            len(import_records),
            record.provider.value,
            record.external_id,
        )
        source_id = await _upsert_source_for_record(client, record)
        stats.sources_upserted += 1

        entity_id = await _upsert_entity(client, record, principal_id)
        stats.entities_upserted += 1

        combined_names = record.names + record.aliases
        stats.names_upserted += await _upsert_entity_names(client, entity_id, source_id, combined_names)
        stats.descriptions_upserted += await _upsert_descriptions(
            client,
            entity_id,
            source_id,
            record.descriptions,
        )

        for image_url in _deduplicate_strings(record.wikimedia_urls):
            if not _is_valid_url(image_url):
                continue
            await _upsert_wikimedia_source(client, image_url)
            stats.wikimedia_sources_upserted += 1

        stats.records_processed += 1

    return stats


async def run_authority_import(
    *,
    wikidata_dir: str | None,
    wikidata_prefix: str,
    nlcr_dir: str | None,
    max_records: int | None,
    weaviate: WeaviateConnectionConfig,
) -> ImportStats:
    records = _build_records(wikidata_dir, wikidata_prefix, nlcr_dir)
    logger.info("Prepared %d authority records for import", len(records))

    client = _build_weaviate_client(weaviate)
    await client.connect()
    try:
        stats = await _import_records(client, records, max_records)
    finally:
        await client.close()

    logger.info("Import finished")
    logger.info("records_processed=%d", stats.records_processed)
    logger.info("sources_upserted=%d", stats.sources_upserted)
    logger.info("entities_upserted=%d", stats.entities_upserted)
    logger.info("names_upserted=%d", stats.names_upserted)
    logger.info("descriptions_upserted=%d", stats.descriptions_upserted)
    logger.info("wikimedia_sources_upserted=%d", stats.wikimedia_sources_upserted)
    return stats
