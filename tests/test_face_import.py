import json

import numpy as np

from people_gator_common.models import PeopleGatorDetection
from people_gator_common.pipelines.face_import import (
    detection_to_face_payload,
    detection_to_page_payload,
    load_face_embedding_artifact,
)


def test_detection_to_face_payload_uses_page_coordinates_and_page_source() -> None:
    detection = PeopleGatorDetection(
        library="mzk",
        document="11111111-1111-1111-1111-111111111111",
        page="22222222-2222-2222-2222-222222222222.jpg",
        page_height=2000,
        page_width=1000,
        image_name="page__image_0.jpg",
        crop_name="page__image_0__face_0.jpg",
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

    source_id = "33333333-3333-3333-3333-333333333333"
    payload = detection_to_face_payload(detection, source_id=source_id)

    assert payload["identifier"] == (
        "mzk/11111111-1111-1111-1111-111111111111/page__image_0__face_0.jpg"
    )
    assert detection_to_page_payload(detection) == {
        "library": "mzk",
        "document_id": "11111111-1111-1111-1111-111111111111",
        "page_id": "22222222-2222-2222-2222-222222222222",
    }
    assert payload["source_id"] == source_id
    assert "source" not in payload
    assert payload["bounding_box"] == {
        "x": 101,
        "y": 202,
        "width": 30,
        "height": 40,
    }
    assert [point["type"] for point in payload["keypoints"]] == [
        "left_eye",
        "right_eye",
        "nose",
        "left_mouth_corner",
        "right_mouth_corner",
    ]
    assert "entity_id" not in payload
    assert "embedding" not in payload


def test_load_face_embedding_artifact_maps_aligned_crop_to_identifier(tmp_path) -> None:
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.ones((1, 512), dtype=np.float32))
    (tmp_path / "embeddings.manifest.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "face": (
                    "mzk/11111111-1111-1111-1111-111111111111."
                    "peoplegator_aligned_crops/page__image_0__face_0.jpg"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "embeddings.metadata.json").write_text(
        json.dumps(
            {
                "count": 1,
                "model_source": {
                    "backbone_name": (
                        "hf_hub:gaunernst/"
                        "vit_small_patch8_gap_112.cosface_ms1mv3"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    artifact = load_face_embedding_artifact(embeddings_path)

    identifier = (
        "mzk/11111111-1111-1111-1111-111111111111/"
        "page__image_0__face_0.jpg"
    )
    assert artifact.model_identifier == "vit_small_patch8_gap_112.cosface_ms1mv3"
    assert artifact.vector(identifier) == [1.0] * 512
