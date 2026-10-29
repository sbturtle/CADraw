"""Composition of image detection and geometry normalization."""

import math
import os
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from cadraw.association import suggest_dimension_target
from cadraw.detection import (
    DetectionConfig,
    GrayImage,
    ShapeDetectionEvidence,
    automatic_source_region,
    detect_face_shapes_with_evidence,
    detect_line_candidates,
    detect_shape_candidates_with_evidence,
    is_page_frame_rule,
    load_grayscale_image,
    settle_shape_roles,
    stroke_intensity,
    stroke_wobble,
)
from cadraw.duplicate_filter import (
    without_lines_through_features,
    without_polygon_edge_lines,
)
from cadraw.faces import FaceConfig, drawing_ink
from cadraw.furniture_filter import (
    RulingRails,
    without_printed_rules,
    without_ruling_remnants,
    without_table_edge_lines,
)
from cadraw.local_ocr import discover_ocr_backend
from cadraw.models import (
    AnalysisStatus,
    AnalysisWarning,
    ArcCandidate,
    ArcEntity,
    CircleCandidate,
    CircleEntity,
    DetectedEntity,
    Dimension,
    DimensionStatus,
    DrawingAnalysis,
    GeometryEntity,
    GeometryRole,
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
    InkBetween,
    NormalizationConfig,
    build_connections,
    normalize_lines,
    resolution_adjusted,
    select_significant_components,
)
from cadraw.ocr import (
    OcrBackend,
    OcrResult,
    TextReader,
    recognize_dimensions,
)
from cadraw.ollama_reader import OllamaTextReader
from cadraw.page_cleanup import PageCleanupConfig, clean_page
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
    ScoreContext,
    illumination_flattened,
    preprocess_variants,
    score_candidates,
    write_debug_outputs,
)
from cadraw.stroke_evidence import ink_between_factory, ink_mask_from_grayscale
from cadraw.text_filter import TextStrokeConfig, TextStrokeRemoval, without_text_strokes

type VariantEvaluation = tuple[
    PreprocessingResult,
    tuple[LineCandidate, ...],
    CandidateScore,
]
_VISION_MODEL_ENV = "CADRAW_VISION_MODEL"
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
    page_cleanup: PageCleanupConfig = field(default_factory=PageCleanupConfig)
    text_strokes: TextStrokeConfig = field(default_factory=TextStrokeConfig)
    ocr_cross_check_model: str | None = None

    @classmethod
    def from_detection_defaults(cls) -> "PipelineConfig":
        """Create camera-sketch defaults that favor structural line chains.

        Closed shapes are detected here because a cut plate's outline is the
        part, not an extra. They were opt-in while ruled paper and printed
        form furniture still reached the candidate list and outnumbered the
        drawing; those are now proved away before detection.
        """
        return cls(
            detection=DetectionConfig(
                minimum_line_length=60.0,
                hough_threshold=25,
                maximum_line_gap=28,
                maximum_candidates=64,
                detect_closed_shapes=True,
                faces=FaceConfig(enabled=True),
            ),
            normalization=NormalizationConfig(
                horizontal_angle_tolerance_degrees=7.0,
                vertical_angle_tolerance_degrees=7.0,
                endpoint_snap_distance=18.0,
                line_merge_tolerance=8.0,
                line_merge_tolerance_ratio=0.009,
                minimum_line_length=60.0,
                keep_largest_connected_component=True,
                evidence_gap_distance=90.0,
                component_length_ratio=0.20,
            ),
            perspective_correction=True,
            preprocessing=PreprocessingConfig(enabled=True),
        )

    @classmethod
    def for_workbench(cls) -> "PipelineConfig":
        """Return the detection defaults plus the local OCR the app ships.

        OCR stays off in `from_detection_defaults` so that analyses built for
        tests and scripts never depend on a model download. The workbench reads
        labels with the bundled PP-OCR reader, which also lets it drop the pen
        strokes of those labels before geometry is built. Setting
        `CADRAW_VISION_MODEL` to a local Ollama vision model (for example
        `qwen3-vl:4b-instruct`) adds a second reading of every value.
        """
        return replace(
            cls.from_detection_defaults(),
            ocr_enabled=True,
            ocr_cross_check_model=os.environ.get(_VISION_MODEL_ENV) or None,
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
    ink_mask: GrayImage | None = None


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
_RAIL_REMNANT_PERIOD_RATIO = 0.25
_POLYGON_EDGE_TOLERANCE = 12.0
_FEATURE_PENETRATION_MARGIN = 3.0
_PRINTED_RULE_LENGTH_RATIO = 0.15
_POLYGON_EDGE_TOLERANCE_RATIO = 0.012
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
        case ArcCandidate(
            entity_id=entity_id,
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
            confidence=confidence,
            role=role,
        ):
            return ArcEntity(
                entity_id,
                center,
                radius,
                start_angle,
                end_angle,
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
    quad = candidate.quad
    return replace(
        candidate,
        bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
        quad=None
        if quad is None
        else (
            _restore_point(quad[0], offset, inverse_matrix),
            _restore_point(quad[1], offset, inverse_matrix),
            _restore_point(quad[2], offset, inverse_matrix),
            _restore_point(quad[3], offset, inverse_matrix),
        ),
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
    """Prepare and independently detect geometry for every variant.

    Closed shapes are detected per variant only when the caller asked for
    them, so the line-only path costs exactly what it did before. Where they
    are wanted, how much paper structure a variant manufactured is the one
    signal that separates a variant which recovered the drawing from one that
    recovered the stationery.
    """
    results = preprocess_variants(image, preprocessing_config)
    evaluations = []
    for result in results:
        candidates = detect_line_candidates(
            result.detection_image,
            detection_config,
        )
        shapes = detect_shape_candidates_with_evidence(
            result.detection_image,
            detection_config,
        )
        score = score_candidates(
            result.variant,
            candidates,
            result.mask,
            ScoreContext(
                minimum_line_length=detection_config.minimum_line_length,
                endpoint_tolerance=normalization_config.endpoint_snap_distance,
                paper_shape_count=(
                    shapes.grid_candidate_count + shapes.table_candidate_count
                ),
            ),
        )
        evaluations.append((result, candidates, score))
    return tuple(evaluations)


def _variant_selection_key(
    evaluation: VariantEvaluation,
) -> tuple[float, float, float, float]:
    """Prefer meaningful low-noise geometry when composite scores are close.

    Printed ruling yields long, page-spanning candidates, so the grid share is
    ranked ahead of average length to stop a tie being settled by the very
    artifact the drawing needs removed.
    """
    score = evaluation[2]
    meaningful_ratio = min(1.0, score.meaningful_line_count / 8.0)
    return (
        meaningful_ratio - 0.50 * score.short_noise_ratio,
        -score.periodic_grid_ratio,
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
            periodic_grid_ratio=score.periodic_grid_ratio,
            paper_shape_count=score.paper_shape_count,
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
        ink_mask=chosen[0].mask if automatic_confident else None,
    )


def _ink_mask_for(stream: DetectionStream) -> GrayImage:
    """Return the binary ink the selected raster's detector saw.

    The selected variant's own binary mask is the natural evidence; when no
    variant was trusted, the raw frame is binarized once so later tests still
    see the same ink the detector saw.
    """
    selection, _, detection_image = stream
    mask = None if selection is None else selection.ink_mask
    return ink_mask_from_grayscale(detection_image) if mask is None else mask


def _ink_between_for(
    ink_mask: GrayImage,
    config: PipelineConfig,
) -> InkBetween | None:
    """Return the gap-bridging predicate for the selected raster, if wanted."""
    if config.normalization.evidence_gap_distance <= 0.0:
        return None
    return ink_between_factory(ink_mask)


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
    # Holes and slots drawn inside a part are that part's features, so a
    # plate with a ring of slots is not a swarm however few lines it has.
    loose_shapes = (
        0
        if shape_evidence is None
        else sum(
            candidate.role is not GeometryRole.HOLE
            for candidate in shape_evidence.candidates
        )
    )
    shape_dominated = (
        shape_evidence is not None
        and loose_shapes >= _MIN_GRID_LIKE_SHAPES
        and shape_evidence.grid_like_count >= _MIN_GRID_LIKE_SHAPES
        and loose_shapes
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
        removed = shape_evidence.grid_candidate_count
        warnings.append(
            AnalysisWarning(
                code=WarningCode.GRAPH_PAPER_DETECTED,
                message=(
                    f"규칙 격자 배경에서 칸 후보 {removed}개를 제외했습니다."
                    if removed
                    else "규칙 격자 배경이 감지되어 닫힌 도형 후보를 검토해야 합니다."
                ),
                entity_ids=shape_entity_ids,
            ),
        )
        # A proved grid has had its cells removed, so only an unproved
        # shape swarm still needs the reviewer to arbitrate.
        if entities and shape_dominated:
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


def _largest_shape_perimeter(entities: tuple[GeometryEntity, ...]) -> float:
    """Return the longest closed-shape perimeter as the drawing's yardstick.

    Once a part's outline closes into a polygon or circle, its edges leave the
    line pool, and the longest remaining line group is often lettering. The
    outline's perimeter says how large the drawing really is.
    """
    perimeters: list[float] = [0.0]
    for entity in entities:
        if isinstance(entity, PolylineEntity) and entity.closed and entity.points:
            points = entity.points
            perimeters.append(
                sum(
                    points[index].distance_to(points[(index + 1) % len(points)])
                    for index in range(len(points))
                ),
            )
        elif isinstance(entity, CircleEntity):
            perimeters.append(2.0 * math.pi * entity.radius)
    return max(perimeters)


def _is_isolated_line(
    entity: LineEntity,
    lines: tuple[LineEntity, ...],
    snap_distance: float,
) -> bool:
    """Return whether no other line meets this one at an endpoint."""
    return not any(
        other is not entity
        and any(
            point.distance_to(other_point) <= snap_distance
            for point in (entity.start, entity.end)
            for other_point in (other.start, other.end)
        )
        for other in lines
    )


def _without_page_frame_rules(
    lines: tuple[LineEntity, ...],
    image: GrayImage,
    detection_config: DetectionConfig,
    snap_distance: float,
) -> tuple[LineEntity, ...]:
    """Drop printed frame rules that survived as normalized geometry.

    Candidate-level rejection is not enough on its own: a form's closing rule
    is often detected as two shorter collinear pieces, which near-collinear
    merging then rejoins into one page-spanning line.

    A rule is only dropped when it is also isolated. A drawn member that runs
    the width of the sheet is joined to the rest of the outline, so requiring
    isolation keeps the geometry of a part that fills its page while still
    removing the form's own closing rule.
    """
    height, width = cast("tuple[int, int]", image.shape[:2])
    return tuple(
        entity
        for entity in lines
        if not (
            is_page_frame_rule(
                entity.start,
                entity.end,
                width,
                height,
                detection_config,
            )
            and _is_isolated_line(entity, lines, snap_distance)
        )
    )


def _without_printed_rules(
    lines: tuple[LineEntity, ...],
    source_image: GrayImage,
    snap_distance: float,
    *,
    shapes: tuple[GeometryEntity, ...],
) -> tuple[LineEntity, ...]:
    """Drop isolated lines as straight as print, measured on the source page.

    A line touching a shape's outline is part of the drawing. A form's rule
    is furniture beside a drawing, so when removing the rules would leave the
    sheet with no geometry at all, the lines are the drawing, however
    straight, and all of them stay.
    """
    height, width = cast("tuple[int, int]", source_image.shape[:2])
    flattened = illumination_flattened(source_image)
    kept = without_printed_rules(
        lines,
        tuple(
            shape
            for shape in shapes
            if isinstance(shape, CircleEntity)
            or (isinstance(shape, PolylineEntity) and shape.closed)
        ),
        snap_distance,
        min(width, height) * _PRINTED_RULE_LENGTH_RATIO,
        lambda start, end: stroke_wobble(flattened, start, end),
    )
    return kept if kept or shapes else lines


def _suggest_ocr_dimension_targets(
    ocr_result: OcrResult | None,
    entities: tuple[GeometryEntity, ...],
    offset: Point,
    inverse_matrix: tuple[float, ...] | None,
) -> tuple[Dimension, ...]:
    """Pre-fill OCR dimension targets only when exactly one entity is near."""
    if ocr_result is None:
        return ()
    if not ocr_result.dimensions:
        return ocr_result.dimensions
    restored_sources = tuple(
        _restore_text_candidate(candidate, offset, inverse_matrix)
        for candidate in ocr_result.dimension_sources
    )
    return tuple(
        suggest_dimension_target(dimension, source.bounds, entities)
        for dimension, source in zip(
            ocr_result.dimensions,
            restored_sources,
            strict=True,
        )
    )


def _second_reader_for(
    config: PipelineConfig,
    second_reader: TextReader | None,
) -> TextReader | None:
    """Return the injected cross-check reader, or the configured local model."""
    if second_reader is not None:
        return second_reader
    model = config.ocr_cross_check_model
    return None if model is None else OllamaTextReader(model=model)


def _read_text(
    analysis_image: GrayImage,
    config: PipelineConfig,
    ocr_backend: OcrBackend | None,
    second_reader: TextReader | None,
    warnings: list[AnalysisWarning],
) -> OcrResult:
    """Read labels in the analysis frame and report what could not run."""
    reader = _second_reader_for(config, second_reader)
    result = recognize_dimensions(
        analysis_image,
        ocr_backend if ocr_backend is not None else discover_ocr_backend(),
        second_reader=reader,
    )
    if not result.available:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.OCR_UNAVAILABLE,
                message="OCR 엔진을 찾지 못해 문자 후보를 만들지 못했습니다.",
            ),
        )
    elif result.second_reader_available is False:
        warnings.append(
            AnalysisWarning(
                code=WarningCode.SECOND_READER_UNAVAILABLE,
                message=(
                    "보조 판독 모델에 연결하지 못해 일부 치수를 교차 확인하지 "
                    "못했습니다."
                ),
            ),
        )
    return result


def _without_lettering(
    candidates: tuple[LineCandidate, ...],
    shape_detection: ShapeDetectionEvidence,
    ocr_result: OcrResult | None,
    ink_mask: GrayImage,
    config: TextStrokeConfig,
) -> tuple[tuple[LineCandidate, ...], ShapeDetectionEvidence, TextStrokeRemoval]:
    """Drop candidates proved to be the pen strokes of a text label."""
    if ocr_result is None or not ocr_result.candidates:
        return (
            candidates,
            shape_detection,
            TextStrokeRemoval(
                candidates,
                shape_detection.candidates,
            ),
        )
    removal = without_text_strokes(
        candidates,
        shape_detection.candidates,
        ocr_result.candidates,
        ink_mask,
        config,
    )
    return (
        removal.lines,
        replace(shape_detection, candidates=removal.shapes),
        removal,
    )


def _lettering_warning(
    removals: tuple[TextStrokeRemoval, ...],
) -> AnalysisWarning | None:
    """Report how many candidates lettering removal took out, if any."""
    removed = sum(len(removal.removed_ids) for removal in removals)
    if not removed:
        return None
    regions = max(removal.region_count for removal in removals)
    return AnalysisWarning(
        code=WarningCode.TEXT_STROKES_REMOVED,
        message=(
            f"숫자로 읽힌 글자 영역 {regions}곳 안에만 있는 "
            f"획 후보 {removed}개를 도형에서 제외했습니다."
        ),
    )


@dataclass(frozen=True, slots=True)
class _ShapeContext:
    """Everything closed-shape detection and its lettering removal need."""

    analysis_image: GrayImage
    detection_image: GrayImage
    detection: DetectionConfig
    ink_mask: GrayImage
    ocr_result: OcrResult | None
    text_strokes: TextStrokeConfig
    snap_distance: float
    lines: tuple[LineEntity, ...] = ()


def _joins_lines(arc: ArcCandidate, context: _ShapeContext) -> bool:
    """Return whether both ends of an open arc meet the end of a line.

    An open curve that belongs to a part, such as a rounded corner, runs from
    one straight edge to another. A curve standing alone, such as part of a
    written number or a scribble around a price, meets no line.
    """
    reach = 2.0 * context.snap_distance
    ends = tuple(point for line in context.lines for point in (line.start, line.end))
    return all(
        any(point.distance_to(end) <= reach for end in ends)
        for point in _arc_endpoints(arc)
    )


def _arc_endpoints(arc: ArcCandidate) -> tuple[Point, Point]:
    """Return the start and end points of an arc in image coordinates."""
    start = math.radians(arc.start_angle_degrees)
    end = math.radians(arc.end_angle_degrees)
    return (
        Point(
            arc.center.x + arc.radius * math.cos(start),
            arc.center.y + arc.radius * math.sin(start),
        ),
        Point(
            arc.center.x + arc.radius * math.cos(end),
            arc.center.y + arc.radius * math.sin(end),
        ),
    )


def _lettering_free_shapes(
    context: _ShapeContext,
) -> tuple[ShapeDetectionEvidence, TextStrokeRemoval]:
    """Detect closed shapes, drop lettering, and settle the hole roles."""
    _, shapes, removal = _without_lettering(
        (),
        _closed_shapes(context),
        context.ocr_result,
        context.ink_mask,
        context.text_strokes,
    )
    if not context.detection.faces.enabled:
        return shapes, removal
    height, width = cast("tuple[int, int]", context.analysis_image.shape[:2])
    return (
        replace(
            shapes,
            candidates=settle_shape_roles(
                shapes.candidates,
                min(width, height) * context.detection.faces.isolated_mark_ratio,
            ),
        ),
        removal,
    )


def _closed_shapes(context: _ShapeContext) -> ShapeDetectionEvidence:
    """Detect closed shapes from enclosed faces, or from variant contours."""
    config = context.detection
    contours = detect_shape_candidates_with_evidence(context.detection_image, config)
    if not config.faces.enabled:
        return contours
    return detect_face_shapes_with_evidence(
        context.analysis_image,
        config,
        context.snap_distance,
        ()
        if context.ocr_result is None
        else tuple(candidate.bounds for candidate in context.ocr_result.candidates),
        tuple(
            candidate
            for candidate in contours.candidates
            if isinstance(candidate, ArcCandidate) and _joins_lines(candidate, context)
        ),
    )


def _uncertain_reading_warnings(
    dimensions: tuple[Dimension, ...],
    ocr_result: OcrResult | None,
) -> tuple[AnalysisWarning, ...]:
    """Name every OCR value a second reader saw differently, with both readings.

    Showing the second reading matters: on the samples it was the right one
    for several values the primary got wrong (`590` against `510`), so the
    reviewer gets a concrete alternative to check rather than only a flag.
    """
    sources = () if ocr_result is None else ocr_result.dimension_sources
    uncertain = tuple(
        (dimension, source)
        for dimension, source in zip(dimensions, sources, strict=True)
        if dimension.status is DimensionStatus.UNCERTAIN
    )
    if not uncertain:
        return ()
    readings = ", ".join(
        f"{dimension.raw_text} / {source.alternate_text or '읽지 못함'}"
        for dimension, source in uncertain
    )
    return (
        AnalysisWarning(
            code=WarningCode.UNCERTAIN_TEXT,
            message=(
                f"두 판독기가 다르게 읽은 치수 {len(uncertain)}개({readings})는 "
                "원본 사진과 대조해 주세요."
            ),
            entity_ids=tuple(
                entity_id
                for dimension, _ in uncertain
                for entity_id in dimension.target_entity_ids
            ),
        ),
    )


def analyze_image(
    image_path: Path,
    config: PipelineConfig,
    ocr_backend: OcrBackend | None = None,
    preprocessing_variant: str | None = None,
    second_reader: TextReader | None = None,
) -> DrawingAnalysis:
    """Analyze one image without inventing physical dimensions."""
    source_image = load_grayscale_image(image_path)
    image_shape = cast("tuple[int, int]", source_image.shape[:2])
    warnings: list[AnalysisWarning] = []
    cleanup = clean_page(source_image, config.page_cleanup)
    if cleanup.applied:
        source_image = cleanup.image
        period = 0.0 if cleanup.grid_period is None else cleanup.grid_period
        warnings.append(
            AnalysisWarning(
                code=WarningCode.PRINTED_RULING_REMOVED,
                message=(
                    f"모눈 주기 {period:.0f}px로 증명된 인쇄 괘선"
                    f"(가로 {cleanup.horizontal_rule_count}, "
                    f"세로 {cleanup.vertical_rule_count})을 제외했습니다."
                ),
            ),
        )
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
    frame_height, frame_width = cast("tuple[int, int]", analysis_image.shape[:2])
    normalization = resolution_adjusted(
        config.normalization,
        frame_width,
        frame_height,
    )
    ink_mask = (
        drawing_ink(analysis_image, detection_config.faces)
        if detection_config.faces.enabled
        else _ink_mask_for(detection_stream)
    )
    ocr_result = (
        _read_text(analysis_image, config, ocr_backend, second_reader, warnings)
        if config.ocr_enabled
        else None
    )
    candidates, _, line_lettering = _without_lettering(
        candidates,
        ShapeDetectionEvidence(()),
        ocr_result,
        ink_mask,
        config.text_strokes,
    )
    lines = _without_page_frame_rules(
        normalize_lines(
            candidates,
            normalization,
            _ink_between_for(ink_mask, config),
        ),
        analysis_image,
        detection_config,
        normalization.endpoint_snap_distance,
    )
    shape_detection, shape_lettering = _lettering_free_shapes(
        _ShapeContext(
            analysis_image=analysis_image,
            detection_image=detection_image,
            detection=detection_config,
            ink_mask=ink_mask,
            ocr_result=ocr_result,
            text_strokes=config.text_strokes,
            snap_distance=normalization.endpoint_snap_distance,
            lines=lines,
        ),
    )
    lettering_warning = _lettering_warning((line_lettering, shape_lettering))
    if lettering_warning is not None:
        warnings.append(lettering_warning)
    lines = without_table_edge_lines(
        lines,
        shape_detection.table_cell_bounds,
        max(
            normalization.endpoint_snap_distance,
            min(frame_width, frame_height)
            * detection_config.printed_table.edge_line_tolerance_ratio,
        ),
    )
    shape_entities = tuple(
        _shape_candidate_to_entity(candidate)
        for candidate in shape_detection.candidates
    )
    lines = without_lines_through_features(
        without_polygon_edge_lines(
            lines,
            tuple(
                entity
                for entity in shape_entities
                if isinstance(entity, PolylineEntity)
            ),
            max(
                _POLYGON_EDGE_TOLERANCE,
                min(frame_width, frame_height) * _POLYGON_EDGE_TOLERANCE_RATIO,
            ),
        ),
        tuple(
            entity
            for entity in shape_entities
            if isinstance(entity, (CircleEntity, PolylineEntity))
            and entity.role is GeometryRole.HOLE
        ),
        _FEATURE_PENETRATION_MARGIN,
    )
    if (
        normalization.keep_largest_connected_component
        and not detection_config.drawing_field.enabled
    ):
        lines = select_significant_components(
            lines,
            normalization.endpoint_snap_distance,
            normalization.component_length_ratio,
            _largest_shape_perimeter(shape_entities),
        )
    restored_shapes = _restore_source_coordinates(
        shape_entities,
        offset,
        inverse_matrix,
    )
    restored_lines = _without_printed_rules(
        without_ruling_remnants(
            tuple(
                entity
                for entity in _restore_source_coordinates(
                    lines,
                    offset,
                    inverse_matrix,
                )
                if isinstance(entity, LineEntity)
            ),
            RulingRails(
                horizontal=cleanup.horizontal_rails,
                vertical=cleanup.vertical_rails,
                # A skewed scan drifts a rail off its median row by more than the
                # erasure band over the sheet, so remnants are matched within a
                # quarter period; the ink test keeps a stroke drawn on a rail.
                band_width=max(
                    config.page_cleanup.rail_band_width,
                    _RAIL_REMNANT_PERIOD_RATIO * (cleanup.grid_period or 0.0),
                ),
                maximum_intensity=(
                    detection_config.arbitration.grid.maximum_grid_intensity
                ),
            ),
            normalization.endpoint_snap_distance,
            lambda start, end: stroke_intensity(source_image, start, end),
        ),
        source_image,
        normalization.endpoint_snap_distance,
        shapes=restored_shapes,
    )
    entities = (*restored_lines, *restored_shapes)
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
    suggested_dimensions = _suggest_ocr_dimension_targets(
        ocr_result,
        entities,
        offset,
        inverse_matrix,
    )
    warnings.extend(_uncertain_reading_warnings(suggested_dimensions, ocr_result))
    return DrawingAnalysis(
        unit="mm",
        image_size=(image_shape[1], image_shape[0]),
        entities=entities,
        dimensions=suggested_dimensions,
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
