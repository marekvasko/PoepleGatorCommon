"""Parse and extract values from line-oriented Wikidata JSON dumps.

Wikidata's ``latest-all.json`` dump is a JSON array with one entity per line.
The helpers in this module deliberately operate on one line/entity at a time so
that callers can process the dump without holding it in memory.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import json
import logging
import os
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
type ParameterValue = str | int | float | datetime | dict[str, JsonValue]
type DumpPath = os.PathLike[str] | str


class WikidataText(BaseModel):
    language: str
    value: str


class WikiDataDataValueBase(BaseModel):
    type: str


class WikiDataStringDataValueBase(WikiDataDataValueBase):
    value: str


class WikiDataDataValueExternalId(WikiDataStringDataValueBase):
    pass


class WikiDataQuantityValue(BaseModel):
    amount: str
    unit: str


class WikiDataDataValueQuantity(WikiDataDataValueBase):
    value: WikiDataQuantityValue


class WikiDataEntityIdValue(BaseModel):
    entity_type: str = Field(alias="entity-type")
    numeric_id: int | None = Field(default=None, alias="numeric-id")
    id: str


class WikiDataDataValueWikibaseItem(WikiDataDataValueBase):
    value: WikiDataEntityIdValue


class WikiDataGlobeCoordinateValue(BaseModel):
    latitude: float
    longitude: float
    altitude: float | None = None
    precision: float | int | None = None
    globe: str


class WikiDataDataValueGlobeCoordinate(WikiDataDataValueBase):
    value: WikiDataGlobeCoordinateValue


class WikiDataDataValueString(WikiDataStringDataValueBase):
    pass


class WikiDataMonolingualTextValue(BaseModel):
    text: str
    language: str


class WikiDataDataValueMonolingualText(WikiDataDataValueBase):
    value: WikiDataMonolingualTextValue


class WikiDataDataValueCommonsMedia(WikiDataStringDataValueBase):
    pass


class WikiDataTimeValue(BaseModel):
    time: str
    timezone: int
    before: int
    after: int
    precision: int
    calendarmodel: str


class WikiDataDataValueTime(WikiDataDataValueBase):
    value: WikiDataTimeValue


class WikiDataDataValueUrl(WikiDataStringDataValueBase):
    pass


class WikiDataDataValueGeoShape(WikiDataStringDataValueBase):
    pass


class WikiDataDataValueWikibaseLexeme(WikiDataDataValueBase):
    value: WikiDataEntityIdValue


class WikiDataDataValueTabularData(WikiDataStringDataValueBase):
    pass


class WikiDataDataValueWikibaseProperty(WikiDataDataValueBase):
    value: WikiDataEntityIdValue


class WikiDataDataValueMath(WikiDataStringDataValueBase):
    pass


class WikiDataEntitySchemaValue(BaseModel):
    id: str
    entity_type: str = Field(alias="entity-type")


class WikiDataDataValueEntitySchema(WikiDataDataValueBase):
    value: WikiDataEntitySchemaValue


class WikiDataDataValueMusicalNotation(WikiDataStringDataValueBase):
    pass


class WikiDataFormValue(BaseModel):
    entity_type: str = Field(alias="entity-type")
    id: str


class WikiDataDataValueWikibaseForm(WikiDataDataValueBase):
    value: WikiDataFormValue


class WikiDataDataValueWikibaseSense(WikiDataDataValueBase):
    value: WikiDataFormValue


type WikiDataDataValue = (
    WikiDataDataValueExternalId
    | WikiDataDataValueQuantity
    | WikiDataDataValueWikibaseItem
    | WikiDataDataValueGlobeCoordinate
    | WikiDataDataValueString
    | WikiDataDataValueMonolingualText
    | WikiDataDataValueCommonsMedia
    | WikiDataDataValueTime
    | WikiDataDataValueUrl
    | WikiDataDataValueGeoShape
    | WikiDataDataValueWikibaseLexeme
    | WikiDataDataValueTabularData
    | WikiDataDataValueWikibaseProperty
    | WikiDataDataValueMath
    | WikiDataDataValueEntitySchema
    | WikiDataDataValueMusicalNotation
    | WikiDataDataValueWikibaseForm
    | WikiDataDataValueWikibaseSense
)


WIKIDATA_DATA_VALUE_MODELS: dict[str, type[WikiDataDataValueBase]] = {
    "external-id": WikiDataDataValueExternalId,
    "quantity": WikiDataDataValueQuantity,
    "wikibase-item": WikiDataDataValueWikibaseItem,
    "globe-coordinate": WikiDataDataValueGlobeCoordinate,
    "string": WikiDataDataValueString,
    "monolingualtext": WikiDataDataValueMonolingualText,
    "commonsMedia": WikiDataDataValueCommonsMedia,
    "time": WikiDataDataValueTime,
    "url": WikiDataDataValueUrl,
    "geo-shape": WikiDataDataValueGeoShape,
    "wikibase-lexeme": WikiDataDataValueWikibaseLexeme,
    "tabular-data": WikiDataDataValueTabularData,
    "wikibase-property": WikiDataDataValueWikibaseProperty,
    "math": WikiDataDataValueMath,
    "entity-schema": WikiDataDataValueEntitySchema,
    "musical-notation": WikiDataDataValueMusicalNotation,
    "wikibase-form": WikiDataDataValueWikibaseForm,
    "wikibase-sense": WikiDataDataValueWikibaseSense,
}


class WikidataSnak(BaseModel):
    snaktype: str
    property: str
    datavalue: WikiDataDataValue | dict[str, JsonValue] | None = None
    # A small number of malformed statements in official dumps omit this
    # field. Keep their raw datavalue instead of rejecting the whole entity.
    datatype: str | None = None

    @model_validator(mode="before")
    @classmethod
    def parse_datavalue(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        model = WIKIDATA_DATA_VALUE_MODELS.get(data.get("datatype"))
        datavalue = data.get("datavalue")
        if model is None or not isinstance(datavalue, dict):
            return data
        parsed = data.copy()
        parsed["datavalue"] = model.model_validate(datavalue)
        return parsed


class WikiDataClaim(BaseModel):
    mainsnak: WikidataSnak
    type: str
    id: str
    rank: str
    references: list[dict[str, JsonValue]] | None = None


class WikidataSiteLink(BaseModel):
    site: str
    title: str
    badges: list[str] = Field(default_factory=list)


class WikiDataItem(BaseModel):
    """The useful top-level fields of an entity in a Wikidata JSON dump."""

    id: str
    type: str
    labels: dict[str, WikidataText] = Field(default_factory=dict)
    descriptions: dict[str, WikidataText] = Field(default_factory=dict)
    aliases: dict[str, list[WikidataText]] = Field(default_factory=dict)
    claims: dict[str, list[WikiDataClaim]] = Field(default_factory=dict)
    sitelinks: dict[str, WikidataSiteLink] = Field(default_factory=dict)
    pageid: int | None = None
    ns: int | None = None
    title: str | None = None
    lastrevid: int | None = None
    modified: str | None = None


class FilterCondition(BaseModel):
    """A property/value predicate, optionally inverted with ``include=False``."""

    model_config = ConfigDict(populate_by_name=True)

    parameter: str = Field(
        validation_alias=AliasChoices("parameter", "property", "property_id")
    )
    value: ParameterValue
    include: bool = True

    @classmethod
    def from_expression(
        cls, expression: str, *, include: bool = True
    ) -> FilterCondition:
        """Build a condition from an expression such as ``P31=Q5``."""
        parameter, separator, value = expression.partition("=")
        if not separator or not parameter.strip() or not value.strip():
            raise ValueError("A filter expression must have the form P…=value")
        return cls(parameter=parameter.strip(), value=value.strip(), include=include)


def parse_wikidata_item(line: str | bytes) -> WikiDataItem:
    """Parse one entity line from a Wikidata array dump.

    Lines in the dump end in a comma except for the last entity.  The array's
    opening and closing bracket are not entity lines and raise ``ValueError``.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    normalized = line.strip()
    if normalized in {"", "[", "]"}:
        raise ValueError("The line does not contain a Wikidata entity")
    if normalized.endswith(","):
        normalized = normalized[:-1].rstrip()
    return WikiDataItem.model_validate_json(normalized)


def _quantity_value(amount: str) -> int | float | str:
    try:
        number = Decimal(amount)
    except InvalidOperation:
        return amount
    if number == number.to_integral_value():
        return int(number)
    return float(number)


def _time_value(value: WikiDataTimeValue) -> datetime | str:
    # Python datetime only supports years 1..9999, whereas Wikidata also has
    # BCE and very large years. Preserve those losslessly as their source text.
    timestamp = value.time
    candidate = timestamp.removeprefix("+")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return timestamp


def _extract_datavalue(
    datavalue: WikiDataDataValue | dict[str, JsonValue],
) -> ParameterValue:
    if isinstance(datavalue, WikiDataStringDataValueBase):
        return datavalue.value
    if isinstance(datavalue, WikiDataDataValueQuantity):
        return _quantity_value(datavalue.value.amount)
    if isinstance(
        datavalue,
        (
            WikiDataDataValueWikibaseItem,
            WikiDataDataValueWikibaseLexeme,
            WikiDataDataValueWikibaseProperty,
            WikiDataDataValueWikibaseForm,
            WikiDataDataValueWikibaseSense,
        ),
    ):
        return datavalue.value.id
    if isinstance(datavalue, WikiDataDataValueEntitySchema):
        return datavalue.value.id
    if isinstance(datavalue, WikiDataDataValueTime):
        return _time_value(datavalue.value)
    if isinstance(datavalue, WikiDataDataValueMonolingualText):
        return datavalue.value.text
    if isinstance(datavalue, WikiDataDataValueGlobeCoordinate):
        return datavalue.value.model_dump(mode="json")
    return datavalue


def extract_parameter_values(
    item: WikiDataItem, parameter: str
) -> list[ParameterValue]:
    """Return every concrete value claimed for ``parameter``, in dump order."""
    values: list[ParameterValue] = []
    for claim in item.claims.get(parameter, []):
        datavalue = claim.mainsnak.datavalue
        if claim.mainsnak.snaktype == "value" and datavalue is not None:
            values.append(_extract_datavalue(datavalue))
    return values


def extract_parameter_value(
    item: WikiDataItem, parameter: str
) -> ParameterValue | None:
    """Return the first concrete value claimed for ``parameter``, if present."""
    values = extract_parameter_values(item, parameter)
    return values[0] if values else None


def extract_truthy_parameter_values(
    item: WikiDataItem, parameter: str
) -> list[ParameterValue]:
    """Return values corresponding to Wikidata's truthy ``wdt:`` predicate.

    Deprecated statements are omitted.  If preferred statements exist, normal
    statements are omitted as well.  Duplicate values are collapsed in dump
    order, matching the RDF graph behavior used by the former SPARQL pipeline.
    """
    candidates = [
        claim
        for claim in item.claims.get(parameter, [])
        if claim.rank != "deprecated"
        and claim.mainsnak.snaktype == "value"
        and claim.mainsnak.datavalue is not None
    ]
    preferred = [claim for claim in candidates if claim.rank == "preferred"]
    selected = preferred or candidates
    values: list[ParameterValue] = []
    fingerprints: set[str] = set()
    for claim in selected:
        datavalue = claim.mainsnak.datavalue
        assert datavalue is not None
        value = _extract_datavalue(datavalue)
        fingerprint = json.dumps(
            _csv_value(value), ensure_ascii=False, sort_keys=True, default=str
        )
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            values.append(value)
    return values


_MISSING = object()


def filter_wikidata_item(
    item: WikiDataItem,
    claim: str | FilterCondition,
    value: object = _MISSING,
    *,
    include: bool = True,
) -> bool:
    """Return whether an entity should pass a property/value condition.

    A property with several claims matches when *any* concrete claim equals the
    requested value.  ``include=False`` negates the result for exclusion rules.

    ``FilterCondition`` is accepted for configuration-driven callers, while
    ``filter_wikidata_item(item, "P31", "Q5")`` provides the compact interface
    used by dump processing code.
    """
    if isinstance(claim, FilterCondition):
        if value is not _MISSING:
            raise TypeError("value must not be supplied with FilterCondition")
        where = claim
    else:
        if value is _MISSING:
            raise TypeError("value is required when claim is a property id")
        where = FilterCondition(parameter=claim, value=value, include=include)

    expected = (
        where.value.isoformat() if isinstance(where.value, datetime) else where.value
    )
    matched = any(
        (value.isoformat() if isinstance(value, datetime) else value) == expected
        for value in extract_parameter_values(item, where.parameter)
    )
    return matched if where.include else not matched


def gather_claim(item: WikiDataItem, claim: str) -> ParameterValue | None:
    """Return the first concrete value for ``claim``, or ``None`` if absent."""
    return extract_parameter_value(item, claim)


def _csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def save_parameter_value(
    path: os.PathLike[str] | str | IO[str], wikidata_id: str, value: object
) -> None:
    """Append one ``wikidata_id,parameter_value`` row to a CSV stream.

    A header is written when the destination path is new/empty, or when an
    opened seekable stream is positioned at an empty file.
    """
    if hasattr(path, "write"):
        stream = path
        if not hasattr(stream, "write"):
            raise TypeError("path must be a filesystem path or writable text stream")
        write_header = False
        try:
            write_header = stream.tell() == 0
        except (AttributeError, OSError):
            pass
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(("wikidata_id", "parameter_value"))
        writer.writerow((wikidata_id, _csv_value(value)))
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(("wikidata_id", "parameter_value"))
        writer.writerow((wikidata_id, _csv_value(value)))


class ClaimCsvWriter:
    """A bound per-property CSV sink exposing the requested save interface."""

    def __init__(self, stream: IO[str]) -> None:
        self._writer = csv.writer(stream)
        self._writer.writerow(("entity", "value"))

    def save_claim(self, item_id: str, claim_value_str: object) -> None:
        """Write one entity/value row to this claim's output file."""
        self._writer.writerow((item_id, _csv_value(claim_value_str)))


@dataclass(slots=True)
class WikidataDumpStats:
    """Counters returned by :func:`parse_wikidata_dump`."""

    items_seen: int = 0
    items_matched: int = 0
    rows_written: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _ExtractedItem:
    item_id: str
    claims: dict[str, list[ParameterValue]]
    labels: list[tuple[str, str]]
    descriptions: list[tuple[str, str]]
    aliases: list[tuple[str, str]]


def _extract_dump_item(
    line: bytes,
    claims: tuple[str, ...],
    filters: tuple[FilterCondition, ...],
    include_labels: bool,
    include_descriptions: bool,
    include_aliases: bool,
) -> _ExtractedItem | None:
    """Parse and reduce one entity; kept at module scope for process workers."""
    item = parse_wikidata_item(line)
    if not all(filter_wikidata_item(item, condition) for condition in filters):
        return None

    return _ExtractedItem(
        item_id=item.id,
        claims={claim: extract_parameter_values(item, claim) for claim in claims},
        labels=(
            [(language, text.value) for language, text in item.labels.items()]
            if include_labels
            else []
        ),
        descriptions=(
            [(language, text.value) for language, text in item.descriptions.items()]
            if include_descriptions
            else []
        ),
        aliases=(
            [
                (language, alias.value)
                for language, language_aliases in item.aliases.items()
                for alias in language_aliases
            ]
            if include_aliases
            else []
        ),
    )


@contextmanager
def _open_dump(path: Path) -> Iterator[IO[bytes]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            yield stream
    elif path.suffix == ".bz2":
        with bz2.open(path, "rb") as stream:
            yield stream
    else:
        with path.open("rb") as stream:
            yield stream


def _iter_dump_lines(
    path: Path, *, start_line: int = 1
) -> Iterator[tuple[int, bytes]]:
    """Yield entity lines from Wikidata's line-oriented top-level JSON array."""
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    with _open_dump(path) as stream:
        started = False
        ended = False
        if start_line > 1:
            logger.info(
                "Fast-skipping dump content before physical line %d without parsing",
                start_line,
            )
        for line_number, line in enumerate(stream, start=1):
            if not started:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped != b"[":
                    raise ValueError(
                        f"Expected '[' at {path}:{line_number}, got entity content"
                    )
                started = True
                continue
            # Official dumps have one entity on every line. During a resume,
            # avoid even normalizing those large byte strings: gzip/bzip2 must
            # still decompress them, but JSON decoding and validation are zero.
            if line_number < start_line:
                if line in (b"]", b"]\n", b"]\r\n"):
                    ended = True
                if line_number % 10_000_000 == 0:
                    logger.info(
                        "Resume scan reached line %d of %d",
                        line_number,
                        start_line,
                    )
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == b"]":
                ended = True
                continue
            if ended:
                raise ValueError(
                    f"Unexpected content after ']' at {path}:{line_number}"
                )
            yield line_number, line
        if not started:
            raise ValueError(f"Wikidata dump is empty: {path}")
        if not ended:
            raise ValueError(f"Wikidata dump has no closing ']': {path}")


def _parsed_items(
    lines: Iterable[tuple[int, bytes]],
    claims: tuple[str, ...],
    filters: tuple[FilterCondition, ...],
    include_labels: bool,
    include_descriptions: bool,
    include_aliases: bool,
    workers: int,
    max_pending: int,
) -> Iterator[tuple[int, _ExtractedItem | None]]:
    args = (
        claims,
        filters,
        include_labels,
        include_descriptions,
        include_aliases,
    )
    if workers == 1:
        for line_number, line in lines:
            try:
                yield line_number, _extract_dump_item(line, *args)
            except Exception as error:
                raise ValueError(
                    f"Failed to parse Wikidata entity at line {line_number}"
                ) from error
        return

    # ProcessPoolExecutor.map eagerly consumes iterables on Python 3.12.  This
    # explicit queue keeps memory bounded while retaining deterministic order.
    pending: deque[tuple[int, Future[_ExtractedItem | None]]] = deque()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        try:
            for line_number, line in lines:
                future = executor.submit(_extract_dump_item, line, *args)
                pending.append((line_number, future))
                if len(pending) >= max_pending:
                    pending_line, pending_future = pending.popleft()
                    try:
                        yield pending_line, pending_future.result()
                    except Exception as error:
                        raise ValueError(
                            f"Failed to parse Wikidata entity at line {pending_line}"
                        ) from error
            while pending:
                pending_line, pending_future = pending.popleft()
                try:
                    yield pending_line, pending_future.result()
                except Exception as error:
                    raise ValueError(
                        f"Failed to parse Wikidata entity at line {pending_line}"
                    ) from error
        except BaseException:
            for _, future in pending:
                future.cancel()
            raise


def iter_wikidata_items(
    input_path: DumpPath,
    *,
    workers: int = 1,
    max_pending: int | None = None,
    start_line: int = 1,
) -> Iterator[WikiDataItem]:
    """Stream typed items from a dump with optionally parallel validation.

    ``start_line`` is an inclusive, one-based physical dump line. Earlier
    lines are read as raw bytes (and decompressed when needed), but are not
    JSON-decoded or validated.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_pending is None:
        max_pending = max(1, workers * 2)
    if max_pending < workers:
        raise ValueError("max_pending must be at least workers")

    path = Path(input_path)
    logger.info("Opening Wikidata dump %s", path)
    lines = _iter_dump_lines(path, start_line=start_line)
    if workers == 1:
        logger.info("Parsing Wikidata entities in the main process")
        for line_number, line in lines:
            try:
                yield parse_wikidata_item(line)
            except Exception as error:
                raise ValueError(
                    f"Failed to parse Wikidata entity at line {line_number}"
                ) from error
        return

    pending: deque[tuple[int, Future[WikiDataItem]]] = deque()
    logger.info(
        "Starting parser process pool with %d workers and at most %d queued items",
        workers,
        max_pending,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        try:
            for line_number, line in lines:
                pending.append(
                    (line_number, executor.submit(parse_wikidata_item, line))
                )
                if len(pending) >= max_pending:
                    pending_line, future = pending.popleft()
                    try:
                        yield future.result()
                    except Exception as error:
                        raise ValueError(
                            f"Failed to parse Wikidata entity at line {pending_line}"
                        ) from error
            while pending:
                pending_line, future = pending.popleft()
                try:
                    yield future.result()
                except Exception as error:
                    raise ValueError(
                        f"Failed to parse Wikidata entity at line {pending_line}"
                    ) from error
        except BaseException:
            for _, future in pending:
                future.cancel()
            raise


def parse_wikidata_dump(
    input_path: DumpPath,
    output_directory: DumpPath,
    claims: Sequence[str],
    *,
    filters: Sequence[FilterCondition] = (),
    include_labels: bool = True,
    include_descriptions: bool = True,
    include_aliases: bool = True,
    workers: int = 1,
    max_pending: int | None = None,
) -> WikidataDumpStats:
    """Stream a Wikidata dump into one CSV per selected claim and text field.

    Input may be an uncompressed JSON path or a ``.gz``/``.bz2`` path.  Each
    property is written to ``<property>.csv`` with ``entity,value`` columns.
    Labels, descriptions, and aliases are normalized into corresponding CSVs
    with ``entity,language,value`` columns.  All configured filters must match.

    When ``workers`` is greater than one, JSON decoding, Pydantic validation,
    filtering, and value gathering run in child processes.  ``max_pending``
    bounds queued entities and defaults to twice the worker count.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_pending is None:
        max_pending = max(1, workers * 2)
    if max_pending < workers:
        raise ValueError("max_pending must be at least workers")

    claim_names = tuple(dict.fromkeys(claims))
    if any(not claim for claim in claim_names):
        raise ValueError("claim ids must not be empty")
    input_file = Path(input_path)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = WikidataDumpStats(
        rows_written={claim: 0 for claim in claim_names}
        | {
            name: 0
            for name, enabled in (
                ("labels", include_labels),
                ("descriptions", include_descriptions),
                ("aliases", include_aliases),
            )
            if enabled
        }
    )

    with ExitStack() as stack:
        claim_writers: dict[str, ClaimCsvWriter] = {}
        for claim in claim_names:
            stream = stack.enter_context(
                (output_dir / f"{claim}.csv").open("w", encoding="utf-8", newline="")
            )
            claim_writers[claim] = ClaimCsvWriter(stream)

        text_writers: dict[str, csv.writer] = {}
        for name, enabled in (
            ("labels", include_labels),
            ("descriptions", include_descriptions),
            ("aliases", include_aliases),
        ):
            if enabled:
                stream = stack.enter_context(
                    (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="")
                )
                writer = csv.writer(stream)
                writer.writerow(("entity", "language", "value"))
                text_writers[name] = writer

        parsed = _parsed_items(
            _iter_dump_lines(input_file),
            claim_names,
            tuple(filters),
            include_labels,
            include_descriptions,
            include_aliases,
            workers,
            max_pending,
        )
        for _, item in parsed:
            stats.items_seen += 1
            if item is None:
                continue
            stats.items_matched += 1
            for claim, values in item.claims.items():
                for value in values:
                    claim_writers[claim].save_claim(item.item_id, value)
                    stats.rows_written[claim] += 1
            for name, values in (
                ("labels", item.labels),
                ("descriptions", item.descriptions),
                ("aliases", item.aliases),
            ):
                writer = text_writers.get(name)
                if writer is None:
                    continue
                for language, value in values:
                    writer.writerow((item.item_id, language, value))
                    stats.rows_written[name] += 1

    return stats


__all__ = [
    "ClaimCsvWriter",
    "FilterCondition",
    "ParameterValue",
    "WikiDataClaim",
    "WikiDataDataValue",
    "WikiDataItem",
    "WikidataDumpStats",
    "WikidataSiteLink",
    "WikidataSnak",
    "WikidataText",
    "extract_parameter_value",
    "extract_parameter_values",
    "extract_truthy_parameter_values",
    "filter_wikidata_item",
    "gather_claim",
    "iter_wikidata_items",
    "parse_wikidata_dump",
    "parse_wikidata_item",
    "save_parameter_value",
]
