"""Composition of image detection and geometry normalization."""

from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from cadraw.detection import (
    DetectionConfig,
    GrayImage,
    ShapeDetectionEvidence,
    automatic_source_region,
    detect_line_candidates,
    detect_shape_candidates_with_evidence,
    load_grayscale_image,
)
from cadraw.models import (
    AnalysisStatus,
    AnalysisWarning,
    ArcEntity,
    CircleCandidate,
    CircleEntity,
    DetectedEntity,
    DrawingAnalysis,
    GeometryEntity,
    LineCandidate,
    LineEntity,
    PerspectiveTransform,
    Point,
    PolylineCandidate,
    PolylineEntity,
    PreprocessingPreview,
    PreprocessingReport,
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
    PageQuadConfig,
    detect_page_quadrilateral_scored,
    transform_point,
    warp_page,
)
from cadraw.preprocessing import (
    CandidateScore,
    PreprocessingConfig,
    PreprocessingResult,
    PreprocessingVariant,
    preprocess_variants,
    score_candidates,
    write_debug_outputs,
)

type VariantEvaluation = tuple[
    PreprocessingResult,
    tuple[LineCandidate, ...],
    CandidateScore,
]
_VARIANT_NEAR_TIE = 0.05
_HIGH_NOISE_RATIO = 0.80
type AnalysisFrame = tuple[
    GrayImage,
    tuple[int, int, int, int] | None,
    Point,
    tuple[float, ...] | None,
    PerspectiveTransform | None,
]


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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    preprocessing_debug_directory: Path | None = None
    page_quad: PageQuadConfig = field(default_factory=PageQuadConfig)

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
            perspective_correction=True,
            preprocessing=PreprocessingConfig(enabled=True),
        )


@dataclass(frozen=True, slots=True)
class _PreprocessingSelection:
    """Internal selected stream plus reviewer-facing variant evidence."""

    candidates: tuple[LineCandidate, ...]
    detection_image: GrayImage
    reports: tuple[PreprocessingReport, ...]
    previews: tuple[PreprocessingPreview, ...]
    selected_variant: str | None
    score: CandidateScore | None


type DetectionStream = tuple[
    _PreprocessingSelection | None,
    tuple[LineCandidate, ...],
    GrayImage,
]


@dataclass(frozen=True, slots=True)
class _AnalysisQualityContext:
    """Quality evidence used to derive the final analysis status."""

    selection: _PreprocessingSelection | None
    shape_evidence: ShapeDetectionEvidence | None
    minimum_score: float
    explicit_variant: bool


_PAGE_EDGE_DOMINANCE_RATIO = 0.50
_MIN_GRID_LIKE_SHAPES = 8
_MIN_SHAPES_PER_LINE = 4


def _select_detection_stream(
    image: GrayImage,
    detection_config: DetectionConfig,
    config: PipelineConfig,
    requested_variant: str | None,
) -> DetectionStream:
    """Return the selected line raster and its optional preprocessing evidence."""
    if config.preprocessing.enabled or requested_variant is not None:
        selection = _select_preprocessing_stream(
            image,
            detection_config,
            config,
            requested_variant,
        )
        return selection, selection.candidates, selection.detection_image
    return None, detect_line_candidates(image, detection_config), image


def _point_segment_distance(
    point: Point,
    start: Point,
    end: Point,
) -> float:
    """Return the shortest distance from a point to a segment."""
    segment_x = end.x - start.x
    segment_y = end.y - start.y
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared == 0.0:
        return point.distance_to(start)
    projection = (
        (point.x - start.x) * segment_x + (point.y - start.y) * segment_y
    ) / segment_length_squared
    clamped = max(0.0, min(1.0, projection))
    nearest = Point(
        start.x + clamped * segment_x,
        start.y + clamped * segment_y,
    )
    return point.distance_to(nearest)


def _page_edge_ratio(
    candidates: tuple[LineCandidate, ...],
    quad: tuple[Point, Point, Point, Point],
    image_shape: tuple[int, int],
) -> float:
    """Measure how strongly a source stream consists of page-border lines."""
    if not candidates:
        return 0.0
    tolerance = max(8.0, min(image_shape) * 0.015)
    edges = tuple(zip(quad, (*quad[1:], quad[0]), strict=True))
    edge_candidates = sum(
        any(
            max(
                _point_segment_distance(candidate.start, start, end),
                _point_segment_distance(candidate.end, start, end),
            )
            <= tolerance
            for start, end in edges
        )
        for candidate in candidates
    )
    return edge_candidates / len(candidates)


def _stream_quality_key(
    stream: DetectionStream,
    normalization_config: NormalizationConfig,
) -> tuple[float, ...]:
    """Return a frame-quality tuple where larger values are safer."""
    selection, candidates, _ = stream
    if selection is not None and selection.score is not None:
        score = selection.score
        return (
            score.score,
            float(score.meaningful_line_count),
            score.connected_endpoint_ratio,
            score.geometry_coverage,
            -score.short_noise_ratio,
        )
    lines = normalize_lines(candidates, normalization_config)
    connections = build_connections(
        lines,
        normalization_config.endpoint_snap_distance,
    )
    return (
        float(len(lines)),
        float(len(connections)),
        sum(line.length for line in lines),
    )


def _should_reject_perspective(
    source_stream: DetectionStream,
    warped_stream: DetectionStream,
    source_image: GrayImage,
    page_quad: tuple[Point, Point, Point, Point],
    normalization_config: NormalizationConfig,
) -> bool:
    """Reject a warp unless it improves evidence over the unwarped stream."""
    if (
        _page_edge_ratio(
            source_stream[1],
            page_quad,
            cast("tuple[int, int]", source_image.shape[:2]),
        )
        >= _PAGE_EDGE_DOMINANCE_RATIO
    ):
        return False
    return _stream_quality_key(source_stream, normalization_config) >= (
        _stream_quality_key(warped_stream, normalization_config)
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


def _effective_detection_config(config: PipelineConfig) -> DetectionConfig:
    """Disable field discovery when an explicit region already narrows input."""
    detection_config = config.detection
    if config.source_region is not None and detection_config.drawing_field.enabled:
        return replace(
            detection_config,
            drawing_field=replace(
                detection_config.drawing_field,
                enabled=False,
            ),
        )
    return detection_config


def _preview_bytes(result: PreprocessingResult) -> bytes:
    """Encode a variant's binary ink view for the optional UI picker."""
    buffer = BytesIO()
    preview = Image.fromarray(
        np.asarray(255 - result.mask, dtype=np.uint8),
        mode="L",
    )
    preview.thumbnail((640, 480), Image.Resampling.BILINEAR)
    preview.save(buffer, format="PNG")
    return buffer.getvalue()


def _evaluate_variants(
    image: GrayImage,
    detection_config: DetectionConfig,
    preprocessing_config: PreprocessingConfig,
    normalization_config: NormalizationConfig,
) -> tuple[VariantEvaluation, ...]:
    """Prepare and independently detect geometry for every variant."""
    results = preprocess_variants(image, preprocessing_config)
    evaluations = []
    for result in results:
        candidates = detect_line_candidates(
            result.detection_image,
            detection_config,
        )
        score = score_candidates(
            result.variant,
            candidates,
            result.mask,
            detection_config.minimum_line_length,
            normalization_config.endpoint_snap_distance,
        )
        evaluations.append((result, candidates, score))
    return tuple(evaluations)


def _variant_selection_key(evaluation: VariantEvaluation) -> tuple[float, float, float]:
    """Prefer meaningful low-noise geometry when composite scores are close."""
    score = evaluation[2]
    meaningful_ratio = min(1.0, score.meaningful_line_count / 8.0)
    return (
        meaningful_ratio - 0.50 * score.short_noise_ratio,
        score.average_line_length,
        score.score,
    )


def _select_variant(
    evaluations: tuple[VariantEvaluation, ...],
) -> VariantEvaluation:
    """Select the best score, with a conservative meaningful-line near-tie."""
    best_score = max(evaluation[2].score for evaluation in evaluations)
    near_ties = tuple(
        evaluation
        for evaluation in evaluations
        if best_score - evaluation[2].score <= _VARIANT_NEAR_TIE
    )
    return max(near_ties, key=_variant_selection_key)


def _select_preprocessing_stream(
    image: GrayImage,
    detection_config: DetectionConfig,
    config: PipelineConfig,
    requested_variant: str | None,
) -> _PreprocessingSelection:
    """Evaluate variants and return one geometry stream plus evidence."""
    preprocessing_config = config.preprocessing
    evaluations = _evaluate_variants(
        image,
        detection_config,
        preprocessing_config,
        config.normalization,
    )
    requested = (
        None if requested_variant is None else PreprocessingVariant(requested_variant)
    )
    if requested is None:
        chosen = _select_variant(evaluations)
    else:
        chosen = next(
            evaluation
            for evaluation in evaluations
            if evaluation[0].variant is requested
        )
    chosen_score = chosen[2]
    automatic_confident = (
        requested is not None
        or chosen_score.score >= preprocessing_config.minimum_selection_score
    )
    candidates = (
        chosen[1]
        if automatic_confident
        else detect_line_candidates(image, detection_config)
    )
    selected_variant = chosen[0].variant.value if automatic_confident else None
    reports = tuple(
        PreprocessingReport(
            variant=result.variant.value,
            score=score.score,
            meaningful_line_count=score.meaningful_line_count,
            average_line_length=score.average_line_length,
            connected_endpoint_ratio=score.connected_endpoint_ratio,
            geometry_coverage=score.geometry_coverage,
            duplicate_line_ratio=score.duplicate_line_ratio,
            short_noise_ratio=score.short_noise_ratio,
            component_quality=score.component_quality,
            processing_time_ms=result.processing_time_ms,
            selected=result is chosen[0],
        )
        for result, _, score in evaluations
    )
    previews = tuple(
        PreprocessingPreview(
            variant=result.variant.value,
            image_png=_preview_bytes(result),
        )
        for result, _, _ in evaluations[:3]
    )
    if config.preprocessing_debug_directory is not None:
        write_debug_outputs(
            image,
            tuple(result for result, _, _ in evaluations),
            config.preprocessing_debug_directory,
        )
    return _PreprocessingSelection(
        candidates=candidates,
        detection_image=chosen[0].detection_image if automatic_confident else image,
        reports=reports,
        previews=previews,
        selected_variant=selected_variant,
        score=chosen_score,
    )


def _add_analysis_quality_warnings(
    warnings: list[AnalysisWarning],
    entities: tuple[GeometryEntity, ...],
    context: _AnalysisQualityContext,
) -> AnalysisStatus:
    """Convert detection evidence into explicit, user-visible states."""
    status = AnalysisStatus.READY
    selection = context.selection
    shape_evidence = context.shape_evidence
    minimum_score = context.minimum_score
    explicit_variant = context.explicit_variant
    selected_variant = None if selection is None else selection.selected_variant
    selected_score = None if selection is None else selection.score
    if selected_score is not None and selected_variant is None:
        status = AnalysisStatus.PREPROCESSING_UNCERTAIN
        warnings.append(
            AnalysisWarning(
                code=WarningCode.PREPROCESSING_UNCERTAIN,
                message="전처리 후보가 불확실해 원본 영상으로 계속합니다.",
            ),
        )
    elif (
        selected_score is not None
        and selected_variant is not None
        and selected_score.score < minimum_score
    ):
        warnings.append(
            AnalysisWarning(
                code=WarningCode.VERY_LOW_CONFIDENCE,
                message="선택한 도면 후보의 신뢰도가 낮습니다.",
            ),
        )
        if not explicit_variant:
            status = AnalysisStatus.VERY_LOW_CONFIDENCE
    if not entities:
        status = AnalysisStatus.NO_GEOMETRY_DETECTED
        warnings.extend(
            (
                AnalysisWarning(
                    code=WarningCode.LOW_CONFIDENCE_GEOMETRY,
                    message="도형 후보를 찾지 못했습니다. 사진을 확인해 주세요.",
                ),
                AnalysisWarning(
                    code=WarningCode.NO_GEOMETRY_DETECTED,
                    message="도면 선을 충분히 인식하지 못했습니다.",
                ),
            ),
        )
    if (
        entities
        and selected_score is not None
        and selected_score.short_noise_ratio > _HIGH_NOISE_RATIO
        and selected_score.score < minimum_score
    ):
        status = AnalysisStatus.TOO_MUCH_NOISE
        warnings.append(
            AnalysisWarning(
                code=WarningCode.TOO_MUCH_NOISE,
                message="배경 잡음이 많아 도면 후보를 확인해야 합니다.",
            ),
        )
    shape_dominated = (
        shape_evidence is not None
        and len(shape_evidence.candidates) >= _MIN_GRID_LIKE_SHAPES
        and shape_evidence.grid_like_count >= _MIN_GRID_LIKE_SHAPES
        and len(shape_evidence.candidates)
        >= _MIN_SHAPES_PER_LINE
        * max(
            1,
            sum(isinstance(entity, LineEntity) for entity in entities),
        )
    )
    if shape_evidence is not None and (shape_evidence.grid_detected or shape_dominated):
        shape_entity_ids = tuple(
            entity.entity_id
            for entity in entities
            if isinstance(entity, (PolylineEntity, CircleEntity))
        )
        warnings.append(
            AnalysisWarning(
                code=WarningCode.GRAPH_PAPER_DETECTED,
                message="규칙 격자 배경이 감지되어 닫힌 도형 후보를 검토해야 합니다.",
                entity_ids=shape_entity_ids,
            ),
        )
        if entities:
            status = AnalysisStatus.TOO_MUCH_NOISE
    return status


def _prepare_analysis_frame(
    source_image: GrayImage,
    config: PipelineConfig,
    warnings: list[AnalysisWarning],
) -> AnalysisFrame:
    """Isolate an optional page/crop while retaining source coordinates."""
    analysis_image: GrayImage = source_image
    offset = Point(0.0, 0.0)
    inverse_matrix: tuple[float, ...] | None = None
    perspective_transform: PerspectiveTransform | None = None
    source_region = config.source_region
    if source_region is None and config.auto_source_region:
        source_region = automatic_source_region(
            source_image,
            config.auto_region_margin_ratio,
        )
    if config.perspective_correction:
        page_detection = detect_page_quadrilateral_scored(
            source_image,
            source_region,
            config.page_quad,
        )
        if page_detection.candidate is None:
            warnings.extend(
                (
                    AnalysisWarning(
                        code=WarningCode.PERSPECTIVE_NOT_FOUND,
                        message="페이지 사변형을 찾지 못해 원본 좌표로 분석합니다.",
                    ),
                    AnalysisWarning(
                        code=WarningCode.DOCUMENT_NOT_FOUND,
                        message="문서 경계를 확정하지 못해 원본 사진을 사용합니다.",
                    ),
                ),
            )
        else:
            page_candidate = page_detection.candidate
            warp = warp_page(source_image, page_candidate.quad)
            analysis_image = warp.image
            inverse_matrix = warp.inverse_matrix
            perspective_transform = replace(
                warp.transform,
                confidence=page_candidate.confidence,
            )
    if source_region is not None and perspective_transform is None:
        x, y, width, height = source_region
        analysis_image = analysis_image[y : y + height, x : x + width]
        offset = Point(float(x), float(y))
    return (
        analysis_image,
        source_region,
        offset,
        inverse_matrix,
        perspective_transform,
    )


def analyze_image(
    image_path: Path,
    config: PipelineConfig,
    ocr_backend: OcrBackend | None = None,
    preprocessing_variant: str | None = None,
) -> DrawingAnalysis:
    """Analyze one image without inventing physical dimensions."""
    source_image = load_grayscale_image(image_path)
    image_shape = cast("tuple[int, int]", source_image.shape[:2])
    warnings: list[AnalysisWarning] = []
    (
        analysis_image,
        source_region,
        offset,
        inverse_matrix,
        perspective_transform,
    ) = _prepare_analysis_frame(source_image, config, warnings)
    detection_config = _effective_detection_config(config)
    detection_stream = _select_detection_stream(
        analysis_image,
        detection_config,
        config,
        preprocessing_variant,
    )
    if perspective_transform is not None:
        source_stream = _select_detection_stream(
            source_image,
            detection_config,
            replace(config, preprocessing_debug_directory=None),
            preprocessing_variant,
        )
        if _should_reject_perspective(
            source_stream,
            detection_stream,
            source_image,
            perspective_transform.source_quad,
            config.normalization,
        ):
            analysis_image = source_image
            offset = Point(0.0, 0.0)
            inverse_matrix = None
            perspective_transform = None
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.PERSPECTIVE_REJECTED,
                    message=(
                        "원근 보정이 선 구조를 개선하지 않아 원본 영상을 사용합니다."
                    ),
                ),
            )
            detection_stream = (
                _select_detection_stream(
                    source_image,
                    detection_config,
                    config,
                    preprocessing_variant,
                )
                if config.preprocessing_debug_directory is not None
                else source_stream
            )
    preprocessing_selection, candidates, detection_image = detection_stream
    preprocessing_reports = (
        () if preprocessing_selection is None else preprocessing_selection.reports
    )
    preprocessing_previews = (
        () if preprocessing_selection is None else preprocessing_selection.previews
    )
    selected_preprocessing_variant = (
        None
        if preprocessing_selection is None
        else preprocessing_selection.selected_variant
    )
    lines = normalize_lines(candidates, config.normalization)
    if (
        config.normalization.keep_largest_connected_component
        and not detection_config.drawing_field.enabled
    ):
        lines = select_largest_connected_component(
            lines,
            config.normalization.endpoint_snap_distance,
        )
    shape_detection = detect_shape_candidates_with_evidence(
        detection_image,
        detection_config,
    )
    shape_entities = tuple(
        _shape_candidate_to_entity(candidate)
        for candidate in shape_detection.candidates
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
    status = _add_analysis_quality_warnings(
        warnings,
        entities,
        _AnalysisQualityContext(
            selection=preprocessing_selection,
            shape_evidence=shape_detection,
            minimum_score=config.preprocessing.minimum_selection_score,
            explicit_variant=preprocessing_variant is not None,
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
        status=status,
        preprocessing_reports=preprocessing_reports,
        preprocessing_previews=preprocessing_previews,
        selected_preprocessing_variant=selected_preprocessing_variant,
    )
