#!/usr/bin/env -S uv run python
"""Render a reviewable extraction report for the checked-in drawing samples.

For each sample this writes one composite image whose left half shows the
geometry the pipeline extracted, drawn over a faded copy of the photograph,
and whose right half shows the hand-authored reference DXF for the same
sketch. The reference is an interpreted redrawing, not a pixel-level answer,
so the pair is meant to be judged by eye rather than scored.
"""

import argparse
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from math import cos, radians, sin
from pathlib import Path
from typing import Final, Protocol, cast

import ezdxf
from PIL import Image, ImageDraw

from cadraw.models import (
    ArcEntity,
    CircleEntity,
    DrawingAnalysis,
    GeometryEntity,
    LineEntity,
    PolylineEntity,
)
from cadraw.pipeline import PipelineConfig, analyze_image

_PROJECT_ROOT: Final = Path(__file__).parents[1]
_SAMPLE_DIRECTORY: Final = _PROJECT_ROOT / "sample" / "pure_background"
_REFERENCE_DIRECTORY: Final = _PROJECT_ROOT / "sample" / "dxf"
_DEFAULT_OUTPUT: Final = _PROJECT_ROOT / "docs" / "experiments" / "sample-extraction"
_READ_FUNCTION: Final = "readfile"
_NOTE_LAYER: Final = "NOTE"
_PANEL_HEIGHT: Final = 760
_PANEL_MARGIN: Final = 18
_PANEL_GAP: Final = 2
_FADE: Final = 0.74
_ARC_SEGMENTS: Final = 48
_CIRCLE_STEP_DEGREES: Final = 6
_MIN_DRAWABLE_POINTS: Final = 2
_FULL_TURN_DEGREES: Final = 360.0

_LINE_COLOR: Final = (206, 30, 30)
_POLYLINE_COLOR: Final = (20, 104, 200)
_CIRCLE_COLOR: Final = (14, 146, 62)
_ARC_COLOR: Final = (204, 112, 12)
_REFERENCE_COLOR: Final = (56, 56, 56)
_WHITE: Final = (255, 255, 255)
_DIVIDER_COLOR: Final = (196, 196, 196)

type Stroke = tuple[tuple[float, float], ...]
type Color = tuple[int, int, int]


class ReferenceAttributesLike(Protocol):
    """DXF attributes read while rendering a reference drawing."""

    layer: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    center: tuple[float, float, float]
    radius: float
    start_angle: float
    end_angle: float


class ReferenceEntityLike(Protocol):
    """DXF entity surface needed to flatten reference geometry."""

    dxf: ReferenceAttributesLike
    closed: bool

    def dxftype(self) -> str:
        """Return the DXF type name."""
        ...

    def get_points(self, point_format: str) -> Iterator[tuple[float, float]]:
        """Return lightweight-polyline vertices."""
        ...


class ReferenceModelspaceLike(Protocol):
    """Modelspace iteration surface used while rendering."""

    def __iter__(self) -> Iterator[ReferenceEntityLike]:
        """Iterate modelspace entities."""
        ...


class ReferenceDocumentLike(Protocol):
    """DXF document surface used while rendering."""

    def modelspace(self) -> ReferenceModelspaceLike:
        """Return modelspace entities."""
        ...


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One sample's extraction evidence, ready to tabulate."""

    index: int
    selected_variant: str
    status: str
    entity_counts: dict[str, int]
    reference_counts: dict[str, int]
    warning_codes: tuple[str, ...]
    image_name: str


def _read_reference(index: int) -> ReferenceDocumentLike:
    """Open one reference drawing through ezdxf's dynamic top-level reader."""
    return cast(
        "ReferenceDocumentLike",
        getattr(ezdxf, _READ_FUNCTION)(str(_REFERENCE_DIRECTORY / f"{index}.dxf")),
    )


def _arc_points(
    center: tuple[float, float],
    radius: float,
    start_degrees: float,
    end_degrees: float,
) -> Stroke:
    """Flatten one arc into a polyline in its own coordinate space."""
    sweep = end_degrees - start_degrees
    return tuple(
        (
            center[0]
            + radius * cos(radians(start_degrees + sweep * step / _ARC_SEGMENTS)),
            center[1]
            + radius * sin(radians(start_degrees + sweep * step / _ARC_SEGMENTS)),
        )
        for step in range(_ARC_SEGMENTS + 1)
    )


def _circle_points(center: tuple[float, float], radius: float) -> Stroke:
    """Flatten one full circle into a closed polyline."""
    return tuple(
        (
            center[0] + radius * cos(radians(angle)),
            center[1] + radius * sin(radians(angle)),
        )
        for angle in range(0, 361, _CIRCLE_STEP_DEGREES)
    )


def _reference_stroke(entity: ReferenceEntityLike) -> Stroke:
    """Flatten one reference DXF entity into a polyline, or nothing."""
    attributes = entity.dxf
    match entity.dxftype():
        case "LINE":
            return (
                (attributes.start[0], attributes.start[1]),
                (attributes.end[0], attributes.end[1]),
            )
        case "LWPOLYLINE":
            points = tuple(entity.get_points("xy"))
            return (*points, points[0]) if entity.closed and points else points
        case "CIRCLE":
            return _circle_points(
                (attributes.center[0], attributes.center[1]),
                attributes.radius,
            )
        case "ARC":
            start_angle = attributes.start_angle
            end_angle = attributes.end_angle
            if end_angle < start_angle:
                end_angle += _FULL_TURN_DEGREES
            return _arc_points(
                (attributes.center[0], attributes.center[1]),
                attributes.radius,
                start_angle,
                end_angle,
            )
        case _:
            return ()


def _reference_strokes(document: ReferenceDocumentLike) -> tuple[Stroke, ...]:
    """Flatten every geometric reference entity outside the note layer."""
    return tuple(
        stroke
        for entity in document.modelspace()
        if entity.dxf.layer != _NOTE_LAYER and (stroke := _reference_stroke(entity))
    )


def _reference_counts(document: ReferenceDocumentLike) -> dict[str, int]:
    """Count reference geometry by DXF type, ignoring annotation."""
    return dict(
        Counter(
            entity.dxftype()
            for entity in document.modelspace()
            if entity.dxf.layer != _NOTE_LAYER
        ),
    )


def _entity_stroke(entity: GeometryEntity) -> tuple[Stroke, Color]:
    """Flatten one detected entity into a polyline plus its review color."""
    match entity:
        case LineEntity(start=start, end=end):
            return ((start.x, start.y), (end.x, end.y)), _LINE_COLOR
        case PolylineEntity(points=points, closed=closed):
            flattened = tuple((point.x, point.y) for point in points)
            return (
                (*flattened, flattened[0]) if closed and flattened else flattened
            ), _POLYLINE_COLOR
        case CircleEntity(center=center, radius=radius):
            return _circle_points((center.x, center.y), radius), _CIRCLE_COLOR
        case ArcEntity(
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
        ):
            return _arc_points(
                (center.x, center.y),
                radius,
                start_angle,
                end_angle,
            ), _ARC_COLOR


def _panel_width(width: float, height: float) -> int:
    """Fit one drawing's aspect ratio into the shared panel height."""
    inner = _PANEL_HEIGHT - 2 * _PANEL_MARGIN
    return max(1, int(width * inner / max(height, 1.0))) + 2 * _PANEL_MARGIN


def _overlay_panel(image_path: Path, analysis: DrawingAnalysis) -> Image.Image:
    """Draw extracted geometry over a faded copy of the source photograph."""
    source = Image.open(image_path).convert("RGB")
    panel_width = _panel_width(*source.size)
    faded = Image.blend(source, Image.new("RGB", source.size, _WHITE), _FADE).resize(
        (panel_width - 2 * _PANEL_MARGIN, _PANEL_HEIGHT - 2 * _PANEL_MARGIN),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (panel_width, _PANEL_HEIGHT), _WHITE)
    panel.paste(faded, (_PANEL_MARGIN, _PANEL_MARGIN))
    scale = faded.size[0] / source.size[0]
    draw = ImageDraw.Draw(panel)
    for entity in analysis.entities:
        stroke, color = _entity_stroke(entity)
        if len(stroke) < _MIN_DRAWABLE_POINTS:
            continue
        draw.line(
            [(_PANEL_MARGIN + x * scale, _PANEL_MARGIN + y * scale) for x, y in stroke],
            fill=color,
            width=3,
        )
    return panel


def _reference_panel(strokes: tuple[Stroke, ...]) -> Image.Image:
    """Draw the reference DXF on its own panel at the shared height."""
    points = [point for stroke in strokes for point in stroke]
    if not points:
        return Image.new("RGB", (_PANEL_HEIGHT, _PANEL_HEIGHT), _WHITE)
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    panel_width = _panel_width(max(max_x - min_x, 1.0), max(max_y - min_y, 1.0))
    panel = Image.new("RGB", (panel_width, _PANEL_HEIGHT), _WHITE)
    draw = ImageDraw.Draw(panel)
    scale = (_PANEL_HEIGHT - 2 * _PANEL_MARGIN) / max(max_y - min_y, 1.0)
    for stroke in strokes:
        if len(stroke) < _MIN_DRAWABLE_POINTS:
            continue
        draw.line(
            [
                (
                    _PANEL_MARGIN + (x - min_x) * scale,
                    _PANEL_HEIGHT - _PANEL_MARGIN - (y - min_y) * scale,
                )
                for x, y in stroke
            ],
            fill=_REFERENCE_COLOR,
            width=2,
        )
    return panel


def _compose(left: Image.Image, right: Image.Image) -> Image.Image:
    """Place the extraction and its reference side by side."""
    canvas = Image.new(
        "RGB",
        (left.size[0] + _PANEL_GAP + right.size[0], _PANEL_HEIGHT),
        _WHITE,
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.size[0] + _PANEL_GAP, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line(
        [(left.size[0], 0), (left.size[0], _PANEL_HEIGHT)],
        fill=_DIVIDER_COLOR,
        width=_PANEL_GAP,
    )
    return canvas


def _analysis_config() -> PipelineConfig:
    """Return the shipped defaults plus closed-shape detection."""
    config = PipelineConfig.from_detection_defaults()
    return replace(
        config,
        detection=replace(config.detection, detect_closed_shapes=True),
    )


def _render_sample(index: int, output: Path) -> SampleRecord:
    """Analyze one sample and write its side-by-side review image."""
    image_path = _SAMPLE_DIRECTORY / f"{index}.png"
    analysis = analyze_image(image_path, _analysis_config())
    document = _read_reference(index)
    composite = _compose(
        _overlay_panel(image_path, analysis),
        _reference_panel(_reference_strokes(document)),
    )
    image_name = f"{index}.png"
    composite.save(output / image_name, optimize=True)
    return SampleRecord(
        index=index,
        selected_variant=analysis.selected_preprocessing_variant or "원본",
        status=analysis.status.value,
        entity_counts=dict(
            Counter(entity.entity_type.value for entity in analysis.entities),
        ),
        reference_counts=_reference_counts(document),
        warning_codes=tuple(warning.code.value for warning in analysis.warnings),
        image_name=image_name,
    )


def _counts_text(counts: dict[str, int]) -> str:
    """Format a count map for a markdown cell."""
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "-"


def _markdown(records: tuple[SampleRecord, ...]) -> str:
    """Build the reviewable report around the rendered images."""
    header = (
        "# 샘플 추출 결과",
        "",
        "`scripts/render_sample_report.py`가 생성합니다. 각 이미지는 왼쪽이",
        "파이프라인이 추출한 geometry를 원본 사진 위에 겹친 것이고, 오른쪽이",
        "같은 스케치의 레퍼런스 DXF입니다. 레퍼런스는 다중 뷰 재배치와 해석된",
        "치수를 포함한 완성형 해석본이므로 구조를 비교하는 용도이며 픽셀 단위",
        "정답지가 아닙니다.",
        "",
        "선 색: 빨강 `LINE`, 파랑 `LWPOLYLINE`, 초록 `CIRCLE`, 주황 `ARC`.",
        "",
        "```powershell",
        "uv run python scripts/render_sample_report.py",
        "```",
        "",
        "## 요약",
        "",
        "| 샘플 | 선택 variant | 상태 | 추출 | 레퍼런스 |",
        "| --- | --- | --- | --- | --- |",
    )
    rows = tuple(
        " | ".join(
            (
                f"| {record.index}",
                f"`{record.selected_variant}`",
                f"`{record.status}`",
                _counts_text(record.entity_counts),
                f"{_counts_text(record.reference_counts)} |",
            ),
        )
        for record in records
    )
    body: list[str] = []
    for record in records:
        warnings = ", ".join(f"`{code}`" for code in record.warning_codes) or "없음"
        body.extend(
            (
                "",
                f"## 샘플 {record.index}",
                "",
                f"![샘플 {record.index}](sample-extraction/{record.image_name})",
                "",
                f"- 선택 variant: `{record.selected_variant}`",
                f"- 상태: `{record.status}`",
                f"- 추출: {_counts_text(record.entity_counts)}",
                f"- 레퍼런스: {_counts_text(record.reference_counts)}",
                f"- 경고: {warnings}",
            ),
        )
    return "\n".join((*header, *rows, *body, ""))


def main() -> int:
    """Render every sample and write the markdown report beside the images."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    options = parser.parse_args()
    output = cast("Path", options.output)
    output.mkdir(parents=True, exist_ok=True)
    indices = sorted(
        int(path.stem)
        for path in _REFERENCE_DIRECTORY.glob("*.dxf")
        if path.stem.isdigit() and (_SAMPLE_DIRECTORY / f"{path.stem}.png").exists()
    )
    records = tuple(_render_sample(index, output) for index in indices)
    report = output.parent / "sample-extraction.md"
    _ = report.write_text(_markdown(records), encoding="utf-8")
    print(f"wrote {len(records)} sample images to {output} and {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
