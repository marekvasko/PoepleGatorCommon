"""Page checkpoints and incremental mutation records."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass, field
from typing import Any
from botocore.exceptions import HTTPClientError
from people_gator_common.filesystem import (
    exists as storage_exists,
    read_text as storage_read_text,
    write_text as storage_write_text,
)
from people_gator_common.settings import PeopleGatorSettings
from people_gator_common.pipelines.llm_ner import PageInput
from people_gator_common.pipelines.ner_clusters import (
    ClusterMember,
    DisambiguatedNamedEntity,
)
from .models import DocumentInput, MODIFICATION_LOG_NAME, PageNerEntityLinks


def _link_output_path(output_prefix: str, page: PageInput | DocumentInput) -> str:
    if page.page_id is None:
        return posixpath.join(output_prefix, page.library, f"{page.document_id}.json")
    return posixpath.join(
        output_prefix, page.library, page.document_id, f"{page.page_id}.json"
    )


def _log_keys(content: str) -> set[str]:
    return {json.loads(line)["key"] for line in content.splitlines() if line.strip()}


def _read_s3_log_without_aiohttp(path: str) -> str:
    """Read a journal whose oversized response header aiohttp cannot parse."""
    from botocore.config import Config
    from botocore.session import get_session

    s3 = PeopleGatorSettings().storage.s3
    if not s3.enabled or not s3.bucket:
        raise RuntimeError("S3 journal recovery requires configured S3 storage")
    client = get_session().create_client(
        "s3",
        endpoint_url=s3.endpoint_url,
        aws_access_key_id=s3.access_key_id,
        aws_secret_access_key=s3.secret_access_key,
        config=Config(request_checksum_calculation="when_required"),
    )
    try:
        response = client.get_object(Bucket=s3.bucket, Key=path)
        with response["Body"] as body:
            return body.read().decode("utf-8")
    finally:
        client.close()


def _read_log(path: str) -> str:
    try:
        return storage_read_text(path) if storage_exists(path) else ""
    except HTTPClientError as exc:
        message = str(exc)
        if "Got more than" not in message or "aws-chunked" not in message:
            raise
        content = _read_s3_log_without_aiohttp(path)
        _log_keys(content)  # Validate every record before replacing the object.
        storage_write_text(path, content)
        return content


@dataclass
class _LinkPage:
    input: PageInput | DocumentInput
    path: str
    clusters: list[DisambiguatedNamedEntity]
    output: PageNerEntityLinks
    completed: int = 0
    cluster_members: dict[str, list[ClusterMember]] = field(default_factory=dict)


class _MutationJournal:
    """Keep page checkpoints and one incremental before/after mutation log."""

    def __init__(self, output_prefix: str) -> None:
        self.path = posixpath.join(output_prefix, MODIFICATION_LOG_NAME)
        self.checked_mutations: dict[str, list[int]] = {}
        self.logged_keys: set[str] = _log_keys(_read_log(self.path))

    def checkpoint(self, page: _LinkPage) -> None:
        storage_write_text(page.path, page.output.model_dump_json(indent=2))
        records = self._new_records(page)
        if not records:
            self._remember_page(page)
            return
        # S3 append preserves the old object's Content-Encoding. With checksum
        # trailers, that can repeatedly accumulate aws-chunked until the
        # response header exceeds aiohttp's limit. A fresh write also repairs
        # journals whose metadata already contains that encoding.
        previous = _read_log(self.path)
        existing = _log_keys(previous)
        missing = [record for record in records if record["key"] not in existing]
        if missing:
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
