"""Page checkpoints and incremental mutation records."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from typing import Any
from people_gator_common.filesystem import (
    ensure_parent_dir as storage_ensure_parent_dir,
    exists as storage_exists,
    open_file as storage_open_file,
    read_text as storage_read_text,
    write_text as storage_write_text,
)
from people_gator_common.pipelines.llm_ner import PageInput
from people_gator_common.pipelines.ner_clusters import DisambiguatedNamedEntity
from .models import MODIFICATION_LOG_NAME, PageNerEntityLinks


def _link_output_path(output_prefix: str, page: PageInput) -> str:
    return posixpath.join(
        output_prefix, page.library, page.document_id, f"{page.page_id}.json"
    )


@dataclass
class _LinkPage:
    input: PageInput
    path: str
    clusters: list[DisambiguatedNamedEntity]
    output: PageNerEntityLinks
    completed: int = 0


class _MutationJournal:
    """Keep page checkpoints and one incremental before/after mutation log."""

    def __init__(self, output_prefix: str) -> None:
        self.path = posixpath.join(output_prefix, MODIFICATION_LOG_NAME)
        self.logged_keys: set[str] = set()
        self.checked_mutations: dict[str, list[int]] = {}
        if storage_exists(self.path):
            for line in storage_read_text(self.path).splitlines():
                if line.strip():
                    self.logged_keys.add(json.loads(line)["key"])

    def checkpoint(self, page: _LinkPage) -> None:
        storage_write_text(page.path, page.output.model_dump_json(indent=2))
        records = self._new_records(page)
        if not records:
            self._remember_page(page)
            return
        lines = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
        storage_ensure_parent_dir(self.path)
        try:
            with storage_open_file(self.path, "a", encoding="utf-8") as log:
                log.write(lines)
        except (OSError, ValueError, NotImplementedError):
            # Some configured object stores do not support append mode.
            previous = storage_read_text(self.path) if storage_exists(self.path) else ""
            existing = {
                json.loads(line)["key"]
                for line in previous.splitlines()
                if line.strip()
            }
            missing = (record for record in records if record["key"] not in existing)
            storage_write_text(
                self.path,
                previous
                + "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in missing
                ),
            )
        self.logged_keys.update(record["key"] for record in records)
        self._remember_page(page)

    def _remember_page(self, page: _LinkPage) -> None:
        self.checked_mutations[page.path] = [
            len(link.mutations) for link in page.output.links
        ]

    def _new_records(self, page: _LinkPage) -> list[dict[str, Any]]:
        records = []
        checked = self.checked_mutations.get(page.path, [])
        for index, link in enumerate(page.output.links):
            start = checked[index] if index < len(checked) else 0
            for mutation in link.mutations[start:]:
                key = "|".join(
                    (
                        page.path,
                        str(index),
                        mutation.operation,
                        mutation.resource_id,
                        mutation.applied_at,
                    )
                )
                if key in self.logged_keys:
                    continue
                records.append(
                    {
                        "key": key,
                        "source_path": page.path,
                        "source_index": index,
                        "operation": mutation.operation,
                        "resource_id": mutation.resource_id,
                        "applied_at": mutation.applied_at,
                        "before": mutation.before,
                        "after": mutation.after,
                    }
                )
        return records
