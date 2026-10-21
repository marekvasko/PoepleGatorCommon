from __future__ import annotations

import asyncio

import httpx
import pytest

from people_gator_common.pipelines.ner_linking.models import (
    EntityLinkingApiError,
    EntityLinkingConfig,
)
from people_gator_common.pipelines.ner_linking.transport import (
    _authenticate_headers,
    _request_json,
    _request_no_content,
)


def test_login_refresh_rotation_and_mutation_retry():
    requests = []
    refreshed = 0

    def handle(request):
        nonlocal refreshed
        requests.append(request)
        if request.url.path == "/token/login":
            assert (
                b"scope=people-gator%3Abrowse+people-gator%3Aingest" in request.content
            )
            return httpx.Response(
                200, json={"access_token": "old", "refresh_token": "refresh-0"}
            )
        if request.url.path == "/token/refresh":
            assert (
                request.content == f'{{"refresh_token":"refresh-{refreshed}"}}'.encode()
            )
            refreshed += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"new-{refreshed}",
                    "refresh_token": f"refresh-{refreshed}",
                },
            )
        if (
            request.headers["Authorization"] != f"Bearer new-{refreshed}"
            or refreshed == 0
        ):
            return httpx.Response(401, json={"detail": "expired_token"})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"entity_id": "created"})

    async def run():
        async with httpx.AsyncClient(
            base_url="https://api.test", transport=httpx.MockTransport(handle)
        ) as client:
            headers = await _authenticate_headers(
                client,
                EntityLinkingConfig(username="user", password="pw"),
                scopes=["people-gator:browse", "people-gator:ingest"],
            )
            assert await _request_json(
                client,
                "POST",
                "/api/v1/entities",
                headers=headers,
                json_body={"name": "Person"},
            ) == {"entity_id": "created"}
            headers["Authorization"] = "Bearer expired-again"
            await _request_no_content(
                client, "DELETE", "/api/v1/entities/created", headers=headers
            )
            assert headers["Authorization"] == "Bearer new-2"
            assert headers.refresh_token == "refresh-2"

    asyncio.run(run())
    assert refreshed == 2
    posts = [r for r in requests if r.url.path == "/api/v1/entities"]
    assert len(posts) == 2
    assert posts[0].content == posts[1].content


def test_concurrent_expiry_refreshes_once():
    refreshes = 0
    expired_requests = 0
    both_expired = asyncio.Event()

    async def handle(request):
        nonlocal refreshes, expired_requests
        if request.url.path == "/token/refresh":
            refreshes += 1
            return httpx.Response(
                200, json={"access_token": "new", "refresh_token": "rotated"}
            )
        if request.headers["Authorization"] == "Bearer old":
            expired_requests += 1
            if expired_requests == 2:
                both_expired.set()
            await both_expired.wait()
            return httpx.Response(401)
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(
            base_url="https://api.test", transport=httpx.MockTransport(handle)
        ) as client:
            headers = await _authenticate_headers(
                client, EntityLinkingConfig(bearer_token="old", refresh_token="refresh")
            )
            results = await asyncio.gather(
                *[
                    _request_json(client, "GET", "/api/v1/entities", headers=headers)
                    for _ in range(2)
                ]
            )
            assert results == [{"ok": True}, {"ok": True}]

    asyncio.run(run())
    assert refreshes == 1


@pytest.mark.parametrize(
    "refresh_token,refresh_status,retry_status,expected_path",
    [
        (None, 200, 401, "/api/v1/entities"),
        ("refresh", 401, 200, "/token/refresh"),
        ("refresh", 200, 401, "/api/v1/entities"),
    ],
)
def test_refresh_failures_and_retry_limit(
    refresh_token, refresh_status, retry_status, expected_path
):
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/token/refresh":
            return httpx.Response(
                refresh_status, json={"access_token": "new", "refresh_token": "rotated"}
            )
        return httpx.Response(
            401 if request.headers["Authorization"] == "Bearer old" else retry_status,
            json={"detail": "unauthorized"},
        )

    async def run():
        async with httpx.AsyncClient(
            base_url="https://api.test", transport=httpx.MockTransport(handle)
        ) as client:
            headers = await _authenticate_headers(
                client,
                EntityLinkingConfig(bearer_token="old", refresh_token=refresh_token),
            )
            with pytest.raises(EntityLinkingApiError) as exc:
                await _request_json(
                    client,
                    "POST",
                    "/api/v1/entities",
                    headers=headers,
                    json_body={"name": "Person"},
                )
            assert exc.value.path == expected_path

    asyncio.run(run())
    assert len(calls) <= 3
