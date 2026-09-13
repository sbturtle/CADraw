"""Composition of image detection and geometry normalization."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from cadraw.detection import (
    DetectionConfig,
    GrayImage,
    automatic_source_region,
    detect_line_candidates,
    detect_shape_candidates,
    load_grayscale_image,
)
from cadraw.models import (
    AnalysisWarning,
    ArcEntity,
    CircleCandidate,
    CircleEntity,
    DetectedEntity,
    DrawingAnalysis,
    GeometryEntity,
    LineEntity,
    PerspectiveTransform,
    Point,
    PolylineCandidate,
    PolylineEntity,
    ShapeCandidate,
    TextCandidate,
    WarningCode,
)
from cadraw.normalization import (
    NormalizationConfig,
    build_connections,
    normalize_lines,
    select_largest_connected_component,
)
from cadraw.ocr import OcrBackend, discover_tesseract_backend, recognize_dimensions
from cadraw.perspective import (
    detect_page_quadrilateral_with_region_fallback,
    transform_point,
    warp_page,
)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for the deterministic Phase 1 analysis."""

    detection: DetectionConfig
    normalization: NormalizationConfig
    source_region: tuple[int, int, int, int] | None = None
    auto_source_region: bool = False
    auto_region_margin_ratio: float = 0.06
    perspective_correction: bool = False
    ocr_enabled: bool = False

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
    entities: tuple[GeometryEntity, ...],
    offset: Point,
    inverse_matrix: tuple[float, ...] | None = None,
) -> tuple[GeometryEntity, ...]:
    """Map analyzed geometry back to source-image coordinates."""
    return tuple(
        _restore_source_geometry(entity, offset, inverse_matrix) for entity in entities
    )


def _restore_point(
    point: Point,
    offset: Point,
    inverse_matrix: tuple[float, ...] | None,
) -> Point:
    """Restore one analyzed point through crop translation and optional warp."""
    translated = point.translated(offset.x, offset.y)
    if inverse_matrix is None:
        return translated
    return transform_point(translated, inverse_matrix)


def _restore_source_geometry(
    entity: GeometryEntity,
    offset: Point,
    inverse_matrix: tuple[float, ...] | None,
) -> GeometryEntity:
    """Map one image-space geometry entity to source coordinates."""
    match entity:
        case LineEntity(
            raw_start=raw_start,
            raw_end=raw_end,
            start=start,
            end=end,
        ):
            return replace(
                entity,
                raw_start=_restore_point(raw_start, offset, inverse_matrix),
                raw_end=_restore_point(raw_end, offset, inverse_matrix),
                start=_restore_point(start, offset, inverse_matrix),
                end=_restore_point(end, offset, inverse_matrix),
            )
        case PolylineEntity(points=points):
            return replace(
                entity,
                points=tuple(
                    _restore_point(point, offset, inverse_matrix) for point in points
                ),
            )
        case CircleEntity(center=center):
            return replace(
                entity,
                center=_restore_point(center, offset, inverse_matrix),
            )
        case ArcEntity(center=center):
            return replace(
                entity,
                center=_restore_point(center, offset, inverse_matrix),
            )


def _shape_candidate_to_entity(candidate: ShapeCandidate) -> GeometryEntity:
    """Convert one detected closed-shape candidate to domain geometry."""
    match candidate:
        case CircleCandidate(
            entity_id=entity_id,
            center=center,
            radius=radius,
            confidence=confidence,
            role=role,
        ):
            return CircleEntity(
                entity_id,
                center,
                radius,
                confidence,
                role=role,
            )
        case PolylineCandidate(
            entity_id=entity_id,
            points=points,
            closed=closed,
            confidence=confidence,
            role=role,
        ):
            return PolylineEntity(
                entity_id,
                points,
                closed,
                confidence,
                role=role,
            )


def _restore_text_candidate(
    candidate: TextCandidate,
    offset: Point,
    inverse_matrix: tuple[float, ...] | None,
) -> TextCandidate:
    """Map an OCR box back to source-image coordinates."""
    x, y, width, height = candidate.bounds
    corners = (
        _restore_point(Point(x, y), offset, inverse_matrix),
        _restore_point(Point(x + width, y), offset, inverse_matrix),
        _restore_point(Point(x + width, y + height), offset, inverse_matrix),
        _restore_point(Point(x, y + height), offset, inverse_matrix),
    )
    min_x = min(point.x for point in corners)
    min_y = min(point.y for point in corners)
    max_x = max(point.x for point in corners)
    max_y = max(point.y for point in corners)
    return replace(
        candidate,
        bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
    )


def analyze_image(
    image_path: Path,
    config: PipelineConfig,
    ocr_backend: OcrBackend | None = None,
) -> DrawingAnalysis:
    """Analyze one image without inventing physical dimensions."""
    source_image = load_grayscale_image(image_path)
    image_shape = cast("tuple[int, int]", source_image.shape[:2])
    analysis_image: GrayImage = source_image
    offset = Point(0.0, 0.0)
    inverse_matrix: tuple[float, ...] | None = None
    perspective_transform: PerspectiveTransform | None = None
    warnings: list[AnalysisWarning] = []
    source_region = config.source_region
    if source_region is None and config.auto_source_region:
        source_region = automatic_source_region(
            source_image,
            config.auto_region_margin_ratio,
        )
    if config.perspective_correction:
        page_quad = detect_page_quadrilateral_with_region_fallback(
            source_image,
            source_region,
        )
        if page_quad is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.PERSPECTIVE_NOT_FOUND,
                    message="페이지 사변형을 찾지 못해 원본 좌표로 분석합니다.",
                ),
            )
        else:
            warp = warp_page(source_image, page_quad)
            analysis_image = warp.image
            inverse_matrix = warp.inverse_matrix
            perspective_transform = warp.transform
    if source_region is not None and perspective_transform is None:
        x, y, width, height = source_region
        analysis_image = analysis_image[y : y + height, x : x + width]
        offset = Point(float(x), float(y))
    candidates = detect_line_candidates(analysis_image, config.detection)
    lines = normalize_lines(candidates, config.normalization)
    if config.normalization.keep_largest_connected_component:
        lines = select_largest_connected_component(
            lines,
            config.normalization.endpoint_snap_distance,
        )
    shape_entities = tuple(
        _shape_candidate_to_entity(candidate)
        for candidate in detect_shape_candidates(analysis_image, config.detection)
    )
    restored_lines = tuple(
        entity
        for entity in _restore_source_coordinates(
            lines,
            offset,
            inverse_matrix,
        )
        if isinstance(entity, LineEntity)
    )
    restored_shapes = _restore_source_coordinates(
        shape_entities,
        offset,
        inverse_matrix,
    )
    entities = (*restored_lines, *restored_shapes)
    ocr_result = None
    if config.ocr_enabled:
        backend = (
            ocr_backend if ocr_backend is not None else discover_tesseract_backend()
        )
        ocr_result = recognize_dimensions(analysis_image, backend)
        if not ocr_result.available:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.OCR_UNAVAILABLE,
                    message="OCR 엔진을 찾지 못해 문자 후보를 만들지 못했습니다.",
                ),
            )
    if ocr_result is None or not ocr_result.dimensions:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.UNKNOWN_DIMENSION,
                message="실제 치수가 아직 확인되지 않았습니다.",
            ),
        )
    if not entities:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.LOW_CONFIDENCE_GEOMETRY,
                message="도형 후보를 찾지 못했습니다. 사진을 확인해 주세요.",
            ),
        )
    restored_text_candidates = (
        ()
        if ocr_result is None
        else tuple(
            _restore_text_candidate(candidate, offset, inverse_matrix)
            for candidate in ocr_result.candidates
        )
    )
    return DrawingAnalysis(
        unit="mm",
        image_size=(image_shape[1], image_shape[0]),
        entities=entities,
        dimensions=() if ocr_result is None else ocr_result.dimensions,
        connections=build_connections(
            restored_lines,
            config.normalization.endpoint_snap_distance,
        ),
        warnings=tuple(warnings),
        detected_entities=tuple(
            DetectedEntity(
                entity_id=entity.entity_id,
                geometry=entity,
                confidence=entity.confidence,
                source_image_reference=str(image_path),
            )
            for entity in entities
        ),
        source_region=source_region,
        text_candidates=restored_text_candidates,
        perspective_transform=perspective_transform,
    )
