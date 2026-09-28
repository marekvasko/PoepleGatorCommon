"""Import historical raw face detections into the People Gator API."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import numpy as np
from numpy.typing import NDArray

from people_gator_common.models import PeopleGatorDetection


logger = logging.getLogger(__name__)

KEYPOINT_TYPES = (
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth_corner",
    "right_mouth_corner",
)


@dataclass(slots=True)
class FaceImportConfig:
    base_url: str = "https://dev.people-gator.capturemyhand.com"
    bearer_token: str | None = None
    username: str | None = None
    password: str | None = None
    batch_size: int = 100
    timeout_seconds: float = 120.0


@dataclass(slots=True)
class FaceImportStats:
    files: int = 0
    detections_read: int = 0
    already_present: int = 0
    sources_registered: int = 0
    uploaded: int = 0
    batches: int = 0


@dataclass(frozen=True, slots=True)
class FaceEmbeddingArtifact:
    embeddings: NDArray[np.float32]
    indices_by_identifier: dict[str, int]
    model_identifier: str

    def vector(self, identifier: str) -> list[float]:
        try:
            index = self.indices_by_identifier[identifier]
        except KeyError as exc:
            raise KeyError(f"no embedding found for face {identifier!r}") from exc
        return self.embeddings[index].tolist()


@dataclass(slots=True)
class FaceEmbeddingBackfillStats:
    available_embeddings: int = 0
    faces_found: int = 0
    embeddings_updated: int = 0
    batches: int = 0


def detection_identifier(detection: PeopleGatorDetection) -> str:
    """Return a stable identifier for one historical face crop."""
    return f"{detection.library}/{detection.document}/{detection.crop_name}"


def _artifact_face_identifier(face_path: str) -> str:
    path = PurePosixPath(face_path.replace("\\", "/"))
    if len(path.parts) < 3:
        raise ValueError(f"invalid artifact face path: {face_path!r}")
    crop_directory = path.parts[-2]
    suffix = ".peoplegator_aligned_crops"
    if not crop_directory.endswith(suffix):
        raise ValueError(
            f"artifact path does not identify an aligned-crop directory: {face_path!r}"
        )
    document = crop_directory.removesuffix(suffix)
    return f"{path.parts[0]}/{document}/{path.name}"


def load_face_embedding_artifact(
    embeddings_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> FaceEmbeddingArtifact:
    """Load precomputed vectors without decoding or transforming face images."""
    embeddings_file = Path(embeddings_path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path is not None
        else embeddings_file.with_suffix(".manifest.jsonl")
    )
    metadata_file = (
        Path(metadata_path)
        if metadata_path is not None
        else embeddings_file.with_suffix(".metadata.json")
    )
    embeddings = np.load(embeddings_file, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(
            f"expected a [count, 512] embedding array, got {embeddings.shape}"
        )
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError(f"expected floating-point embeddings, got {embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding artifact contains non-finite values")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    model_identifier = str(metadata["model_source"]["backbone_name"]).rsplit(
        "/", maxsplit=1
    )[-1]
    if metadata.get("count") != embeddings.shape[0]:
        raise ValueError("embedding metadata count does not match the array")

    indices: dict[str, int] = {}
    with manifest_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record["index"])
            if not 0 <= index < embeddings.shape[0]:
                raise ValueError(
                    f"embedding index out of range at {manifest_file}:{line_number}"
                )
            identifier = _artifact_face_identifier(str(record["face"]))
            if identifier in indices:
                raise ValueError(f"duplicate face in embedding manifest: {identifier}")
            indices[identifier] = index
    if len(indices) != embeddings.shape[0]:
        raise ValueError("embedding manifest count does not match the array")
    return FaceEmbeddingArtifact(
        embeddings=embeddings,
        indices_by_identifier=indices,
        model_identifier=model_identifier,
    )


def detection_to_page_payload(detection: PeopleGatorDetection) -> dict[str, str]:
    """Map a historical detection to its canonical page source descriptor."""
    return {
        "library": detection.library,
        "document_id": detection.document,
        "page_id": Path(detection.page).stem,
    }


def _page_identity(page: dict[str, Any]) -> tuple[str, str, str]:
    return (str(page["library"]), str(page["document_id"]), str(page["page_id"]))


def detection_to_face_payload(
    detection: PeopleGatorDetection, *, source_id: str
) -> dict[str, Any]:
    """Map page-relative detection geometry to the API face contract."""
    if len(detection.page_keypoints) != len(KEYPOINT_TYPES):
        raise ValueError(
            f"{detection_identifier(detection)} has "
            f"{len(detection.page_keypoints)} keypoints; expected {len(KEYPOINT_TYPES)}"
        )

    return {
        "identifier": detection_identifier(detection),
        "source_id": source_id,
        "bounding_box": {
            "x": detection.page_left,
            "y": detection.page_top,
            "width": detection.width,
            "height": detection.height,
        },
        "keypoints": [
            {"type": keypoint_type, "x": point[0], "y": point[1]}
            for keypoint_type, point in zip(
                KEYPOINT_TYPES, detection.page_keypoints, strict=True
            )
        ],
        "confidence": detection.confidence,
    }


def iter_detection_files(input_root: str | Path) -> Iterator[Path]:
    root = Path(input_root)
    yield from sorted(root.rglob("*.people_gator.jsonl"))


def iter_detections(
    input_root: str | Path,
) -> Iterator[tuple[Path, PeopleGatorDetection]]:
    for path in iter_detection_files(input_root):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    detection = PeopleGatorDetection.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(f"invalid detection at {path}:{line_number}: {exc}") from exc
                yield path, detection


async def _authenticate(
    client: httpx.AsyncClient, config: FaceImportConfig
) -> str:
    if config.bearer_token:
        return config.bearer_token
    if not config.username or config.password is None:
        raise ValueError("provide a bearer token or both API username and password")

    response = await client.post(
        "/token/login",
        data={
            "username": config.username,
            "password": config.password,
            "scope": "people-gator:ingest people-gator:browse",
        },
    )
    response.raise_for_status()
    return str(response.json()["access_token"])


async def _existing_face_identifiers(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> set[str]:
    identifiers: set[str] = set()
    cursor: str | None = None
    while True:
        params: dict[str, str | int] = {"limit": 1_000}
        if cursor is not None:
            params["cursor"] = cursor
        response = await client.get("/api/v1/faces", params=params, headers=headers)
        response.raise_for_status()
        body = response.json()
        identifiers.update(
            item["identifier"]
            for item in body["items"]
            if item.get("identifier") is not None
        )
        cursor = body.get("cursor")
        if cursor is None:
            return identifiers


async def _existing_faces(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> dict[str, str]:
    faces: dict[str, str] = {}
    cursor: str | None = None
    while True:
        params: dict[str, str | int] = {"limit": 1_000}
        if cursor is not None:
            params["cursor"] = cursor
        response = await client.get("/api/v1/faces", params=params, headers=headers)
        response.raise_for_status()
        body = response.json()
        for item in body["items"]:
            identifier = item.get("identifier")
            if identifier is not None:
                if identifier in faces:
                    raise ValueError(f"duplicate API face identifier: {identifier}")
                faces[identifier] = item["detection_id"]
        cursor = body.get("cursor")
        if cursor is None:
            return faces


async def _register_page_sources(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    detections: list[PeopleGatorDetection],
) -> dict[tuple[str, str, str], str]:
    pages_by_identity: dict[tuple[str, str, str], dict[str, str]] = {}
    for detection in detections:
        page = detection_to_page_payload(detection)
        pages_by_identity.setdefault(_page_identity(page), page)

    response = await client.post(
        "/api/v1/pages:batch",
        json=list(pages_by_identity.values()),
        headers=headers,
    )
    response.raise_for_status()
    created = response.json()
    if not isinstance(created, list) or len(created) != len(pages_by_identity):
        raise RuntimeError(
            "page batch returned "
            f"{len(created) if isinstance(created, list) else 'a non-list'} "
            f"for {len(pages_by_identity)} inputs"
        )

    source_ids: dict[tuple[str, str, str], str] = {}
    for page in created:
        if not isinstance(page, dict) or "source_id" not in page:
            raise RuntimeError("page batch response is missing source_id")
        identity = _page_identity(page)
        if identity not in pages_by_identity:
            raise RuntimeError(f"page batch returned an unexpected page: {identity!r}")
        source_ids[identity] = str(page["source_id"])
    if len(source_ids) != len(pages_by_identity):
        raise RuntimeError("page batch response did not resolve every requested page")
    return source_ids


async def _upload_batch(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    payloads: list[dict[str, Any]],
) -> int:
    response = await client.post(
        "/api/v1/faces:batch", json=payloads, headers=headers
    )
    response.raise_for_status()
    created = response.json()
    if not isinstance(created, list) or len(created) != len(payloads):
        raise RuntimeError(
            f"face batch returned {len(created) if isinstance(created, list) else 'a non-list'} "
            f"for {len(payloads)} inputs"
        )
    return len(created)


async def _register_sources_and_upload_batch(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    detections: list[PeopleGatorDetection],
) -> tuple[int, int]:
    source_ids = await _register_page_sources(client, headers, detections)
    payloads = [
        detection_to_face_payload(
            detection,
            source_id=source_ids[_page_identity(detection_to_page_payload(detection))],
        )
        for detection in detections
    ]
    uploaded = await _upload_batch(client, headers, payloads)
    return len(source_ids), uploaded


async def run_face_import(
    input_root: str | Path,
    *,
    config: FaceImportConfig,
    max_records: int | None = None,
    skip_existing: bool = True,
) -> FaceImportStats:
    """Upload raw, unassigned face detections and return import statistics."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_records is not None and max_records < 0:
        raise ValueError("max_records cannot be negative")

    stats = FaceImportStats()
    stats.files = sum(1 for _ in iter_detection_files(input_root))
    base_url = config.base_url.rstrip("/")
    timeout = httpx.Timeout(config.timeout_seconds)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        token = await _authenticate(client, config)
        headers = {"Authorization": f"Bearer {token}"}
        existing = (
            await _existing_face_identifiers(client, headers)
            if skip_existing
            else set()
        )
        logger.info("Found %d existing face identifiers", len(existing))

        batch: list[PeopleGatorDetection] = []
        for _path, detection in iter_detections(input_root):
            if max_records is not None and stats.uploaded + len(batch) >= max_records:
                break
            stats.detections_read += 1
            if detection_identifier(detection) in existing:
                stats.already_present += 1
                continue
            batch.append(detection)
            if len(batch) < config.batch_size:
                continue
            registered, uploaded = await _register_sources_and_upload_batch(
                client, headers, batch
            )
            stats.sources_registered += registered
            stats.uploaded += uploaded
            stats.batches += 1
            logger.info("Uploaded %d faces", stats.uploaded)
            batch = []

        if batch:
            registered, uploaded = await _register_sources_and_upload_batch(
                client, headers, batch
            )
            stats.sources_registered += registered
            stats.uploaded += uploaded
            stats.batches += 1
    logger.info(
        "Face import complete: uploaded=%d already_present=%d files=%d",
        stats.uploaded,
        stats.already_present,
        stats.files,
    )
    return stats


async def run_face_embedding_backfill(
    embeddings_path: str | Path,
    *,
    config: FaceImportConfig,
    manifest_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> FaceEmbeddingBackfillStats:
    """Attach precomputed vectors to existing faces without image processing."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    artifact = load_face_embedding_artifact(
        embeddings_path,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    stats = FaceEmbeddingBackfillStats(
        available_embeddings=len(artifact.indices_by_identifier)
    )
    base_url = config.base_url.rstrip("/")
    timeout = httpx.Timeout(config.timeout_seconds)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        token = await _authenticate(client, config)
        headers = {"Authorization": f"Bearer {token}"}
        faces = await _existing_faces(client, headers)
        stats.faces_found = len(faces)
        missing = set(faces).difference(artifact.indices_by_identifier)
        if missing:
            examples = ", ".join(repr(value) for value in sorted(missing)[:3])
            raise ValueError(
                f"{len(missing)} API faces have no embedding; first: {examples}"
            )

        ordered_faces = sorted(faces.items())
        for start in range(0, len(ordered_faces), config.batch_size):
            batch = [
                {
                    "detection_id": detection_id,
                    "embeddings": [
                        {
                            "vector": artifact.vector(identifier),
                            "model_identifier": artifact.model_identifier,
                        }
                    ],
                }
                for identifier, detection_id in ordered_faces[
                    start : start + config.batch_size
                ]
            ]
            response = await client.patch(
                "/api/v1/faces:batch", json=batch, headers=headers
            )
            response.raise_for_status()
            updated = response.json()
            if (
                not isinstance(updated, list)
                or len(updated) != len(batch)
                or any(item is None for item in updated)
            ):
                raise RuntimeError(
                    "embedding batch did not update every requested face"
                )
            stats.embeddings_updated += len(updated)
            stats.batches += 1
            logger.info("Updated %d face embeddings", stats.embeddings_updated)
    return stats


__all__ = [
    "FaceImportConfig",
    "FaceEmbeddingArtifact",
    "FaceEmbeddingBackfillStats",
    "FaceImportStats",
    "detection_identifier",
    "detection_to_face_payload",
    "detection_to_page_payload",
    "iter_detection_files",
    "iter_detections",
    "load_face_embedding_artifact",
    "run_face_embedding_backfill",
    "run_face_import",
]
