"""Canonical People Gator data models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PeopleGatorAnnotation(BaseModel):
    """Annotated people record used in the People Gator project."""

    model_config = ConfigDict(extra="forbid")

    person_name: str
    annotator: str
    library: str
    document: str
    page: str
    crop_name: str
    face: str
    page_width: int
    page_height: int
    page_left: int
    page_top: int
    width: int
    height: int
    page_keypoints: list[tuple[float, float]]


class PeopleGatorDetection(BaseModel):
    """Detected face record used in the People Gator project."""

    model_config = ConfigDict(extra="forbid")

    library: str
    document: str
    page: str
    page_height: int
    page_width: int
    image_name: str
    crop_name: str
    image_left: int
    image_top: int
    page_left: int
    page_top: int
    width: int
    height: int
    image_keypoints: list[tuple[float, float]]
    page_keypoints: list[tuple[float, float]]
    confidence: float = Field(ge=0.0, le=1.0)


class PeopleGatorNamedEntitySpan(BaseModel):
    """Span for a named entity detected in OCR text."""

    model_config = ConfigDict(extra="forbid")

    text: str
    line_id: str
    region_id: str
    page_id: str
    start_index: int
    end_index: int
    baseline: list[tuple[int, int]]


class PeopleGatorNamedEntity(BaseModel):
    """Named entity detected in the People Gator project."""

    model_config = ConfigDict(extra="forbid")

    library: str
    document: str
    page: str
    label: str
    text: str
    formated_text: str
    lematized_text: str | None
    probability: float = Field(ge=0.0, le=1.0)
    page_height: int
    page_width: int
    spans: list[PeopleGatorNamedEntitySpan]


class PeroLine(BaseModel):
    """Single text line from PeroLayout output."""

    model_config = ConfigDict(extra="forbid")

    id: str
    baseline: list[tuple[float, float]]
    polygon: list[tuple[float, float]]
    heights: tuple[float, float]
    transcription: str
    transcription_confidence: float = Field(ge=0.0, le=1.0)
    index: int


class PeroRegion(BaseModel):
    """Text or image region from PeroLayout output."""

    model_config = ConfigDict(extra="forbid")

    id: str
    polygon: list[tuple[float, float]]
    region_type: str | None
    transcription: str
    lines: list[PeroLine]


class PeroLayout(BaseModel):
    """Page layout model from PeroLayout output."""

    model_config = ConfigDict(extra="forbid")

    id: str
    page_size: tuple[int, int]
    regions: list[PeroRegion]
    reading_order: dict[str, list[str] | dict[str, int]]