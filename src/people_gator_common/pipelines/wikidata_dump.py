"""YAML-configured, local-file Wikidata dump extraction pipeline."""

from __future__ import annotations

import csv
import json
import logging
import re
import sqlite3
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import IO, Literal
from urllib.parse import quote

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from people_gator_common.wikidata.dump_parser import (
    FilterCondition,
    ParameterValue,
    WikiDataItem,
    extract_truthy_parameter_values,
    iter_wikidata_items,
)

logger = logging.getLogger(__name__)


class ClaimOutputConfig(BaseModel):
    """One Wikidata property output and its optional unique-value output."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(pattern=r"^P[1-9][0-9]*$")
    output: str
    unique_output: str | None = None
    value_format: Literal["auto", "plain", "wikidata_entity", "commons_media"] = "auto"


class TextOutputConfig(BaseModel):
    """Output names for localized entity text."""

    model_config = ConfigDict(extra="forbid")

    labels: str | None = None
    descriptions: str | None = None
    aliases: str | None = None


class EntityGroupConfig(BaseModel):
    """An independently filtered entity collection and its output files."""

    model_config = ConfigDict(extra="forbid")

    filters: list[FilterCondition]
    entities_output: str
    claims: list[ClaimOutputConfig]
    text: TextOutputConfig = Field(default_factory=TextOutputConfig)

    @model_validator(mode="after")
    def claims_are_unique(self) -> EntityGroupConfig:
        ids = [claim.claim for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim ids must be unique within an entity group")
        return self


class WikidataDumpPipelineConfig(BaseModel):
    """Complete local dump extraction configuration."""

    model_config = ConfigDict(extra="forbid")

    input_path: str
    output_directory: str
    workers: int = Field(default=1, ge=1)
    max_pending: int | None = Field(default=None, ge=1)
    progress_every_items: int = Field(default=10_000, ge=1)
    progress_every_seconds: float = Field(default=30.0, gt=0)
    overwrite: bool = False
    resume_from_line: int | None = Field(default=None, ge=2)
    entity_uri_prefix: str | None = "http://www.wikidata.org/entity/"
    groups: dict[str, EntityGroupConfig]

    @model_validator(mode="after")
    def validate_pipeline(self) -> WikidataDumpPipelineConfig:
        if not self.groups:
            raise ValueError("at least one entity group is required")
        if self.max_pending is not None and self.max_pending < self.workers:
            raise ValueError("max_pending must be at least workers")
        if self.overwrite and self.resume_from_line is not None:
            raise ValueError("overwrite and resume_from_line cannot be used together")
        return self


class WikidataDumpPipelineStats(BaseModel):
    items_seen: int = 0
    matched_by_group: dict[str, int] = Field(default_factory=dict)
    rows_by_output: dict[str, int] = Field(default_factory=dict)


def load_wikidata_dump_config(path: str | Path) -> WikidataDumpPipelineConfig:
    """Load and validate a YAML pipeline configuration."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return WikidataDumpPipelineConfig.model_validate(raw)


def _safe_output_path(output_directory: Path, name: str) -> Path:
    if not name or Path(name).is_absolute():
        raise ValueError(f"Output name must be a relative path: {name!r}")
    root = output_directory.resolve()
    output = (output_directory / name).resolve()
    if not output.is_relative_to(root):
        raise ValueError(f"Output escapes output_directory: {name!r}")
    return output


def _entity_value(item_id: str, prefix: str | None) -> str:
    return f"{prefix}{item_id}" if prefix else item_id


def _format_value(
    value: ParameterValue,
    value_format: Literal["auto", "plain", "wikidata_entity", "commons_media"],
    entity_uri_prefix: str | None,
) -> str:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    if value_format == "commons_media":
        filename = quote(text.replace(" ", "_"), safe="()!,'-._~")
        return f"http://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
    if value_format == "wikidata_entity" or (
        value_format == "auto" and re.fullmatch(r"[QPL][1-9][0-9]*", text)
    ):
        return _entity_value(text, entity_uri_prefix)
    return text


def _matches(item: WikiDataItem, filters: list[FilterCondition]) -> bool:
    for condition in filters:
        expected = condition.value
        matched = expected in extract_truthy_parameter_values(item, condition.parameter)
        if matched != condition.include:
            return False
    return True


class _CsvSink:
    def __init__(
        self, stream: IO[str], header: tuple[str, ...], *, write_header: bool = True
    ) -> None:
        self.writer = csv.writer(stream)
        if write_header:
            self.writer.writerow(header)
        self.rows = 0

    def write(self, *values: object) -> None:
        self.writer.writerow(values)
        self.rows += 1


def _configured_output_names(config: WikidataDumpPipelineConfig) -> list[str]:
    names: list[str] = []
    for group in config.groups.values():
        names.append(group.entities_output)
        for claim in group.claims:
            names.append(claim.output)
            if claim.unique_output:
                names.append(claim.unique_output)
        names.extend(
            output
            for output in (
                group.text.labels,
                group.text.descriptions,
                group.text.aliases,
            )
            if output
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Output names must be unique: {duplicates}")
    return names


def _streaming_output_headers(
    config: WikidataDumpPipelineConfig,
) -> dict[str, tuple[str, ...]]:
    headers: dict[str, tuple[str, ...]] = {}
    for group in config.groups.values():
        headers[group.entities_output] = ("entity",)
        headers.update({claim.output: ("entity", "value") for claim in group.claims})
        for output in (
            group.text.labels,
            group.text.descriptions,
            group.text.aliases,
        ):
            if output:
                headers[output] = ("entity", "entityText", "entityTextLanguage")
    return headers


def _validate_csv_header(path: Path, expected: tuple[str, ...]) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        actual = next(csv.reader(stream), None)
    if actual != list(expected):
        raise ValueError(
            f"Cannot resume into {path}: expected CSV header {expected}, got {actual}"
        )


def _seed_unique_values(
    connection: sqlite3.Connection,
    source: Path,
    sink_name: str,
) -> int:
    """Restore one uniqueness index from a partial entity/value output."""
    inserted = 0
    batch: list[tuple[str, str]] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["entity", "value"]:
            raise ValueError(
                f"Cannot rebuild unique values from {source}: invalid header {header}"
            )
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"Cannot rebuild unique values from {source}:{row_number}: "
                    f"expected 2 columns, got {len(row)}"
                )
            batch.append((sink_name, row[1]))
            if len(batch) >= 10_000:
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO unique_value VALUES (?, ?)", batch
                )
                inserted += connection.total_changes - before
                batch.clear()
        if batch:
            before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO unique_value VALUES (?, ?)", batch
            )
            inserted += connection.total_changes - before
    return inserted


def run_wikidata_dump_pipeline(
    config: WikidataDumpPipelineConfig,
) -> WikidataDumpPipelineStats:
    """Run every configured entity group in one bounded pass over the dump."""
    started_at = time.monotonic()
    output_directory = Path(config.output_directory)
    input_path = Path(config.input_path)
    input_size = input_path.stat().st_size
    logger.info(
        "Starting Wikidata dump extraction: input=%s (%.2f GiB), output=%s, "
        "workers=%d, max_pending=%s, resume_from_line=%s",
        input_path,
        input_size / (1024**3),
        output_directory,
        config.workers,
        config.max_pending or config.workers * 2,
        config.resume_from_line,
    )
    logger.info(
        "Configured groups: %s; progress interval: every %d items or %.1f seconds",
        ", ".join(config.groups),
        config.progress_every_items,
        config.progress_every_seconds,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_names = _configured_output_names(config)
    output_paths = {
        name: _safe_output_path(output_directory, name) for name in output_names
    }
    streaming_headers = _streaming_output_headers(config)
    streaming_paths = {name: output_paths[name] for name in streaming_headers}
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not config.overwrite and config.resume_from_line is None:
        logger.error("Refusing to replace %d existing output files", len(existing))
        raise FileExistsError(
            "Configured outputs already exist; enable overwrite to replace them: "
            + ", ".join(existing)
        )
    if config.resume_from_line is not None:
        missing = [str(path) for path in streaming_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot resume because streaming output files are missing: "
                + ", ".join(missing)
            )
        for name, header in streaming_headers.items():
            _validate_csv_header(output_paths[name], header)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    stats = WikidataDumpPipelineStats(
        matched_by_group=dict.fromkeys(config.groups, 0),
        rows_by_output=dict.fromkeys(output_names, 0),
    )
    unique_claims = [
        (group_name, claim)
        for group_name, group in config.groups.items()
        for claim in group.claims
        if claim.unique_output
    ]
    temporary_db: Path | None = None
    connection: sqlite3.Connection | None = None

    try:
        if unique_claims:
            logger.info(
                "Creating disk-backed unique-value index for %d outputs",
                len(unique_claims),
            )
            with tempfile.NamedTemporaryFile(
                prefix="wikidata-unique-", suffix=".sqlite3", dir=output_directory
            ) as temporary:
                temporary_db = Path(temporary.name)
            connection = sqlite3.connect(temporary_db)
            connection.execute(
                "CREATE TABLE unique_value "
                "(sink TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (sink, value))"
            )
            if config.resume_from_line is not None:
                logger.info(
                    "Rebuilding unique-value indexes from existing claim outputs"
                )
                for _, claim in unique_claims:
                    assert claim.unique_output is not None
                    restored = _seed_unique_values(
                        connection,
                        output_paths[claim.output],
                        claim.unique_output,
                    )
                    logger.info(
                        "Restored %d unique values for %s",
                        restored,
                        claim.unique_output,
                    )

        with ExitStack() as stack:
            sinks: dict[str, _CsvSink] = {}
            resumed = config.resume_from_line is not None
            for name, header in streaming_headers.items():
                stream = stack.enter_context(
                    output_paths[name].open(
                        "a" if resumed else "w", encoding="utf-8", newline=""
                    )
                )
                sinks[name] = _CsvSink(stream, header, write_header=not resumed)

            logger.info(
                "Opened %d streaming CSV outputs; beginning dump scan", len(sinks)
            )

            items = iter_wikidata_items(
                config.input_path,
                workers=config.workers,
                max_pending=config.max_pending,
                start_line=config.resume_from_line or 1,
            )
            last_report_at = started_at
            last_report_items = 0
            for item in items:
                stats.items_seen += 1
                for group_name, group in config.groups.items():
                    if not _matches(item, group.filters):
                        continue
                    stats.matched_by_group[group_name] += 1
                    entity = _entity_value(item.id, config.entity_uri_prefix)
                    sinks[group.entities_output].write(entity)

                    for claim_config in group.claims:
                        values = extract_truthy_parameter_values(
                            item, claim_config.claim
                        )
                        for value in values:
                            formatted = _format_value(
                                value,
                                claim_config.value_format,
                                config.entity_uri_prefix,
                            )
                            sinks[claim_config.output].write(entity, formatted)
                            if claim_config.unique_output and connection:
                                connection.execute(
                                    "INSERT OR IGNORE INTO unique_value VALUES (?, ?)",
                                    (claim_config.unique_output, formatted),
                                )

                    for output, texts in (
                        (group.text.labels, item.labels),
                        (group.text.descriptions, item.descriptions),
                    ):
                        if output:
                            for language, text in texts.items():
                                sinks[output].write(entity, text.value, language)
                    if group.text.aliases:
                        for language, aliases in item.aliases.items():
                            for alias in aliases:
                                sinks[group.text.aliases].write(
                                    entity, alias.value, language
                                )

                now = time.monotonic()
                report_due = (
                    stats.items_seen - last_report_items >= config.progress_every_items
                    or now - last_report_at >= config.progress_every_seconds
                )
                if stats.items_seen == 1 or report_due:
                    elapsed = max(now - started_at, 0.001)
                    logger.info(
                        "Dump progress: items=%d, last_entity=%s, rate=%.1f items/s, "
                        "matched=%s, rows=%d, elapsed=%.1f min",
                        stats.items_seen,
                        item.id,
                        stats.items_seen / elapsed,
                        stats.matched_by_group,
                        sum(sink.rows for sink in sinks.values()),
                        elapsed / 60,
                    )
                    last_report_at = now
                    last_report_items = stats.items_seen

            for name, sink in sinks.items():
                stats.rows_by_output[name] = sink.rows

            logger.info(
                "Finished dump scan after %.1f minutes; finalizing unique-value files",
                (time.monotonic() - started_at) / 60,
            )

        if connection:
            connection.commit()
            for _, claim in unique_claims:
                assert claim.unique_output is not None
                logger.info("Writing unique values to %s", claim.unique_output)
                with output_paths[claim.unique_output].open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("entity",))
                    rows = 0
                    for (value,) in connection.execute(
                        "SELECT value FROM unique_value WHERE sink = ? ORDER BY value",
                        (claim.unique_output,),
                    ):
                        writer.writerow((value,))
                        rows += 1
                    stats.rows_by_output[claim.unique_output] = rows
                logger.info(
                    "Finished %s with %d unique values", claim.unique_output, rows
                )
    finally:
        if connection:
            connection.close()
        if temporary_db and temporary_db.exists():
            temporary_db.unlink()

    logger.info(
        "Wikidata dump extraction completed in %.1f minutes: %s",
        (time.monotonic() - started_at) / 60,
        stats.model_dump(),
    )
    return stats


__all__ = [
    "ClaimOutputConfig",
    "EntityGroupConfig",
    "TextOutputConfig",
    "WikidataDumpPipelineConfig",
    "WikidataDumpPipelineStats",
    "load_wikidata_dump_config",
    "run_wikidata_dump_pipeline",
]
