"""Sequential async API reads, search, and source lookup."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
import httpx
from pydantic import ValidationError
from people_gator_common.client.pydantic_gen import PageCreateRequest
from .models import (
    EntityCandidate,
    EntityLinkingApiError,
    EntityLinkingConfig,
    LinkableEntity,
    _ApiEntityDetail,
)
from .source import _deduplicate_strings, build_search_queries

logger = logging.getLogger(__name__)


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: httpx.QueryParams | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = await client.request(
            method,
            path,
            headers=headers,
            params=params,
            json=json_body,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EntityLinkingApiError(
            method,
            path,
            exc.response.status_code,
            exc.response.text,
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"{method} {path} timed out") from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected response from {path}: {payload}")
    return payload


async def _request_no_content(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    headers: dict[str, str],
) -> None:
    try:
        response = await client.request(method, path, headers=headers)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"{method} {path} failed with status "
            f"{exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"{method} {path} timed out") from exc


async def authenticate_entity_linking_api(
    client: httpx.AsyncClient,
    config: EntityLinkingConfig,
    *,
    scopes: Sequence[str] = ("people-gator:browse",),
) -> str:
    """Resolve a supplied bearer token or obtain one via OAuth2 password flow."""
    if config.bearer_token:
        return config.bearer_token
    if not config.username or config.password is None:
        raise ValueError("provide a bearer token or both API username and password")

    try:
        response = await client.post(
            "/token/login",
            data={
                "username": config.username,
                "password": config.password,
                "scope": " ".join(scopes),
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            "POST /token/login failed with status "
            f"{exc.response.status_code}: {exc.response.text}"
        ) from exc
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(
        payload.get("access_token"), str
    ):
        raise TypeError("Token response does not contain access_token")
    return payload["access_token"]


async def search_entity_candidates(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    source: LinkableEntity,
) -> list[EntityCandidate]:
    """Search using repeated ``query.text`` and hydrate unique results."""
    queries = build_search_queries(source)
    if not queries:
        return []
    search_response = await _request_json(
        client,
        "GET",
        "/api/v1/search-entities",
        headers=headers,
        params=httpx.QueryParams({"query.text": queries}),
    )
    items = search_response.get("items")
    if not isinstance(items, list):
        raise TypeError("search-entities response does not contain an items list")

    unique_hits: list[tuple[str, float]] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Invalid search-entities item: {item}")
        entity_id = item.get("entity_id")
        if not isinstance(entity_id, str):
            raise TypeError(f"Search result has no entity_id: {item}")
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        unique_hits.append(
            (entity_id, float(item.get("score", item.get("matching_score", 0.0))))
        )

    details = []
    for entity_id, _ in unique_hits:
        details.append(
            await _request_json(
                client, "GET", f"/api/v1/entities/{entity_id}", headers=headers
            )
        )
    candidates: list[EntityCandidate] = []
    for (entity_id, matching_score), detail_payload in zip(
        unique_hits, details, strict=True
    ):
        try:
            detail = _ApiEntityDetail.model_validate(detail_payload)
        except ValidationError as exc:
            raise RuntimeError(
                f"Invalid entity detail response for {entity_id}: {exc}"
            ) from exc
        if detail.entity_id != entity_id:
            raise RuntimeError(
                f"Entity detail ID {detail.entity_id} does not match search ID {entity_id}"
            )
        candidates.append(
            EntityCandidate(
                entity_id=detail.entity_id,
                entity_type=detail.entity_type,
                name=(detail.name or "").strip(),
                aliases=_deduplicate_strings(detail.aliases or []),
                short_description=(detail.short_description or "").strip(),
                long_description=(detail.description or "").strip(),
                matching_score=matching_score,
            )
        )
    return candidates


async def lookup_page_source(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    *,
    library: str,
    document_id: str,
    page_id: str,
) -> str:
    """Resolve a page source, creating it when lookup returns HTTP 404."""
    identity = {
        "library": library,
        "document_id": document_id,
        "page_id": page_id,
    }
    params = httpx.QueryParams(identity)
    try:
        payload = await _request_json(
            client,
            "GET",
            "/api/v1/pages/lookup",
            headers=headers,
            params=params,
        )
    except EntityLinkingApiError as exc:
        if exc.status_code != 404:
            raise
        logger.info(
            "Page source not found; creating %s/%s/%s",
            library,
            document_id,
            page_id,
        )
        try:
            payload = await _request_json(
                client,
                "POST",
                "/api/v1/pages",
                headers=headers,
                json_body=PageCreateRequest.model_validate(identity).model_dump(
                    mode="json"
                ),
            )
        except EntityLinkingApiError as create_exc:
            if create_exc.status_code != 409:
                raise
            # Another worker registered the page between lookup and create.
            payload = await _request_json(
                client,
                "GET",
                "/api/v1/pages/lookup",
                headers=headers,
                params=params,
            )

    for field, expected in identity.items():
        if str(payload.get(field)) != expected:
            raise RuntimeError(
                f"Page source {field} {payload.get(field)!r} does not match "
                f"requested value {expected!r}"
            )
    source_id = payload.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise RuntimeError("pages/lookup response does not contain source_id")
    return source_id
