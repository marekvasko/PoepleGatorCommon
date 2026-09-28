import asyncio
import json
from typing import Any

import httpx

from people_gator_common.models import PeopleGatorDetection
from people_gator_common.pipelines.face_import import (
    _existing_face_identifiers,
    _register_sources_and_upload_batch,
)


def _detection(*, crop_name: str = "page__image_0__face_0.jpg") -> PeopleGatorDetection:
    return PeopleGatorDetection(
        library="mzk",
        document="11111111-1111-1111-1111-111111111111",
        page="22222222-2222-2222-2222-222222222222.jpg",
        page_height=2000,
        page_width=1000,
        image_name="page__image_0.jpg",
        crop_name=crop_name,
        image_left=1,
        image_top=2,
        page_left=101,
        page_top=202,
        width=30,
        height=40,
        image_keypoints=[(1.0, 2.0)] * 5,
        page_keypoints=[(10.0 + index, 20.0 + index) for index in range(5)],
        confidence=0.95,
    )


def test_source_is_registered_before_faces_and_reused_within_batch() -> None:
    requests: list[tuple[str, str, Any]] = []
    source_id = "33333333-3333-3333-3333-333333333333"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append((request.method, request.url.path, payload))
        if request.url.path == "/api/v1/pages:batch":
            assert payload == [
                {
                    "library": "mzk",
                    "document_id": "11111111-1111-1111-1111-111111111111",
                    "page_id": "22222222-2222-2222-2222-222222222222",
                }
            ]
            return httpx.Response(
                201,
                json=[
                    {
                        **payload[0],
                        "source_id": source_id,
                        "source_type": "page",
                        "url": "https://example.test/page",
                    }
                ],
            )
        assert request.url.path == "/api/v1/faces:batch"
        assert len(payload) == 2
        assert {item["source_id"] for item in payload} == {source_id}
        assert all("source" not in item for item in payload)
        return httpx.Response(
            201,
            json=[{"detection_id": str(index)} for index in range(len(payload))],
        )

    async def run() -> tuple[int, int]:
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await _register_sources_and_upload_batch(
                client,
                {"Authorization": "Bearer token"},
                [
                    _detection(),
                    _detection(crop_name="page__image_0__face_1.jpg"),
                ],
            )

    assert asyncio.run(run()) == (1, 2)
    assert [(method, path) for method, path, _ in requests] == [
        ("POST", "/api/v1/pages:batch"),
        ("POST", "/api/v1/faces:batch"),
    ]


def test_existing_face_scan_uses_opaque_cursor() -> None:
    seen_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        if "cursor" not in request.url.params:
            return httpx.Response(
                200,
                json={
                    "items": [{"identifier": "first"}],
                    "cursor": "opaque-next-page",
                    "limit": 1000,
                    "total_count": None,
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [{"identifier": "second"}],
                "cursor": None,
                "limit": 1000,
                "total_count": None,
            },
        )

    async def run() -> set[str]:
        async with httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await _existing_face_identifiers(client, {})

    assert asyncio.run(run()) == {"first", "second"}
    assert seen_params == [
        {"limit": "1000"},
        {"limit": "1000", "cursor": "opaque-next-page"},
    ]
