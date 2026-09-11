"""Composition of image detection and geometry normalization."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from cadraw.detection import (
    DetectionConfig,
    GrayImage,
    detect_line_candidates,
    load_grayscale_image,
)
from cadraw.models import (
    AnalysisWarning,
    DetectedEntity,
    DrawingAnalysis,
    LineEntity,
    Point,
    WarningCode,
)
from cadraw.normalization import (
    NormalizationConfig,
    build_connections,
    normalize_lines,
    select_largest_connected_component,
)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for the deterministic Phase 1 analysis."""

    detection: DetectionConfig
    normalization: NormalizationConfig
    source_region: tuple[int, int, int, int] | None = None

    @classmethod
    def from_detection_defaults(cls) -> "PipelineConfig":
        """Create camera-sketch defaults that favor structural line chains."""
        return cls(
            detection=DetectionConfig(
                minimum_line_length=60.0,
                hough_threshold=25,
                maximum_line_gap=28,
                maximum_candidates=64,
            ),
            normalization=NormalizationConfig(
                horizontal_angle_tolerance_degrees=7.0,
                vertical_angle_tolerance_degrees=7.0,
                endpoint_snap_distance=18.0,
                line_merge_tolerance=18.0,
                minimum_line_length=60.0,
                keep_largest_connected_component=True,
            ),
        )


def _restore_source_coordinates(
    lines: tuple[LineEntity, ...],
    offset: Point,
) -> tuple[LineEntity, ...]:
    """Translate cropped line coordinates back to the source image."""
    return tuple(
        replace(
            line,
            raw_start=line.raw_start.translated(offset.x, offset.y),
            raw_end=line.raw_end.translated(offset.x, offset.y),
            start=line.start.translated(offset.x, offset.y),
            end=line.end.translated(offset.x, offset.y),
        )
        for line in lines
    )


def analyze_image(image_path: Path, config: PipelineConfig) -> DrawingAnalysis:
    """Analyze one image without assigning any physical dimension."""
    source_image = load_grayscale_image(image_path)
    image_shape = cast("tuple[int, int]", source_image.shape[:2])
    analysis_image: GrayImage = source_image
    offset = Point(0.0, 0.0)
    if config.source_region is not None:
        x, y, width, height = config.source_region
        analysis_image = source_image[y : y + height, x : x + width]
        offset = Point(float(x), float(y))
    candidates = detect_line_candidates(analysis_image, config.detection)
    lines = normalize_lines(candidates, config.normalization)
    if config.normalization.keep_largest_connected_component:
        lines = select_largest_connected_component(
            lines,
            config.normalization.endpoint_snap_distance,
        )
    lines = _restore_source_coordinates(lines, offset)
    warnings: list[AnalysisWarning] = [
        AnalysisWarning(
            code=WarningCode.UNKNOWN_DIMENSION,
            message="실제 치수가 아직 확인되지 않았습니다.",
        ),
    ]
    if not lines:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.LOW_CONFIDENCE_GEOMETRY,
                message="선 후보를 찾지 못했습니다. 사진을 확인해 주세요.",
            ),
        )
    return DrawingAnalysis(
        unit="mm",
        image_size=(image_shape[1], image_shape[0]),
        entities=tuple(lines),
        dimensions=(),
        connections=build_connections(
            lines,
            config.normalization.endpoint_snap_distance,
        ),
        warnings=tuple(warnings),
        detected_entities=tuple(
            DetectedEntity(
                entity_id=line.entity_id,
                geometry=line,
                confidence=line.confidence,
                source_image_reference=str(image_path),
            )
            for line in lines
        ),
    )
