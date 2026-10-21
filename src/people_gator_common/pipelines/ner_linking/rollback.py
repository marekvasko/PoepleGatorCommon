"""Reverse journaled linking mutations."""

from __future__ import annotations

import logging
import os
import posixpath
import httpx
from pydantic import ValidationError
from people_gator_common.client.pydantic_gen import EntityPatchRequest
from people_gator_common.filesystem import (
    glob as storage_glob,
    read_text as storage_read_text,
    write_text as storage_write_text,
)
from .api import _entity_snapshot, _mutable_entity_state, _snapshot_from_audit
from .transport import (
    _request_json,
    _request_no_content,
    _authenticate_headers,
)
from .models import (
    DEFAULT_OUTPUT_PREFIX,
    EntityLinkingConfig,
    EntityLinkingRollbackSummary,
    PageNerEntityLinks,
    _utc_now,
)

logger = logging.getLogger(__name__)


def _latest_rollbackable_output(output_prefix: str) -> tuple[str, PageNerEntityLinks]:
    latest: tuple[str, str, PageNerEntityLinks] | None = None
    prefix = posixpath.normpath(output_prefix)
    paths = set(storage_glob(posixpath.join(prefix, "*", "*", "*.json")))
    paths.update(storage_glob(posixpath.join(prefix, "*", "*.json")))
    logger.info(
        "Checking %d page/document link journals for rollback below %s",
        len(paths),
        prefix,
    )
    for path in sorted(paths):
        try:
            output = PageNerEntityLinks.model_validate_json(storage_read_text(path))
        except (ValidationError, ValueError):
            logger.warning("Ignoring invalid entity-link audit file %s", path)
            continue
        pending = [
            mutation
            for link in output.links
            for mutation in link.mutations
            if mutation.rollback_status != "rolled_back"
        ]
        if not pending:
            continue
        timestamp = max(mutation.applied_at for mutation in pending)
        if latest is None or timestamp > latest[0]:
            latest = (timestamp, path, output)
    if latest is None:
        raise ValueError(f"No rollbackable mutations found below {output_prefix}")
    return latest[1], latest[2]


async def rollback_latest_entity_linking_run(
    *,
    config: EntityLinkingConfig,
    output_prefix: str | os.PathLike[str] = DEFAULT_OUTPUT_PREFIX,
    force: bool = False,
    api_client: httpx.AsyncClient | None = None,
) -> EntityLinkingRollbackSummary:
    """Reverse the newest page/document journal, refusing changed entities by default."""
    audit_path, output = _latest_rollbackable_output(os.fspath(output_prefix))
    logger.info("Rolling back %s link journal %s", output.scope, audit_path)

    async def run(client: httpx.AsyncClient) -> EntityLinkingRollbackSummary:
        headers = await _authenticate_headers(
            client,
            config,
            scopes=(
                "people-gator:browse",
                "people-gator:annotate",
                "people-gator:admin",
            ),
        )
        mutations = [
            mutation
            for link in output.links
            for mutation in link.mutations
            if mutation.rollback_status != "rolled_back"
        ]
        rolled_back = 0
        for mutation in reversed(mutations):
            mutation.rollback_error = None
            try:
                if mutation.operation == "create_mention":
                    await _request_no_content(
                        client,
                        "DELETE",
                        f"/api/v1/text-mentions/{mutation.resource_id}",
                        headers=headers,
                    )
                else:
                    path = f"/api/v1/entities/{mutation.resource_id}"
                    current = _entity_snapshot(
                        await _request_json(client, "GET", path, headers=headers)
                    )
                    expected = _snapshot_from_audit(mutation.after, "after")
                    if not force and _mutable_entity_state(
                        current
                    ) != _mutable_entity_state(expected):
                        raise RuntimeError(
                            f"Refusing to roll back {mutation.resource_id}: "
                            "the entity changed after this run (use --force-rollback to override)"
                        )
                    if mutation.operation == "create_entity":
                        await _request_no_content(
                            client, "DELETE", path, headers=headers
                        )
                    else:
                        before = _snapshot_from_audit(mutation.before, "before")
                        await _request_json(
                            client,
                            "PATCH",
                            path,
                            headers=headers,
                            json_body=EntityPatchRequest.model_validate(
                                _mutable_entity_state(before)
                            ).model_dump(mode="json", exclude_unset=True),
                        )
                mutation.rollback_status = "rolled_back"
                mutation.rolled_back_at = _utc_now()
                rolled_back += 1
            except Exception as exc:
                mutation.rollback_status = "failed"
                mutation.rollback_error = str(exc)
                storage_write_text(audit_path, output.model_dump_json(indent=2))
                raise
            storage_write_text(audit_path, output.model_dump_json(indent=2))
        output.rolled_back_at = _utc_now()
        storage_write_text(audit_path, output.model_dump_json(indent=2))
        return EntityLinkingRollbackSummary(audit_path, rolled_back)

    if api_client is not None:
        return await run(api_client)
    timeout = httpx.Timeout(config.timeout_seconds)
    async with httpx.AsyncClient(
        base_url=config.base_url.rstrip("/"), timeout=timeout
    ) as created_api:
        return await run(created_api)
