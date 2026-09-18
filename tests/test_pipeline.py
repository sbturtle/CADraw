from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cadraw.detection import DetectionConfig
from cadraw.models import (
    AnalysisStatus,
    CircleEntity,
    DimensionKind,
    DimensionStatus,
    LineEntity,
    PolylineEntity,
    TextCandidate,
    WarningCode,
)
from cadraw.normalization import NormalizationConfig
from cadraw.pipeline import PipelineConfig, analyze_image


class _FakeOcrBackend:
    def recognize(self, image: np.ndarray) -> tuple[TextCandidate, ...]:
        assert image.ndim == 2
        return (
            TextCandidate(
                text="150",
                bounds=(20.0, 20.0, 40.0, 16.0),
                confidence=0.9,
            ),
        )


def test_blank_photo_reports_no_geometry_instead_of_success(tmp_path: Path) -> None:
    # Given a blank source image
    image_path = tmp_path / "blank.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)

    # When the robust pipeline analyzes it
    analysis = analyze_image(image_path, PipelineConfig.from_detection_defaults())

    # Then the failure is explicit and machine-readable
    assert analysis.status is AnalysisStatus.NO_GEOMETRY_DETECTED
    assert any(
        warning.code is WarningCode.NO_GEOMETRY_DETECTED
        for warning in analysis.warnings
    )
    assert any(
        warning.code is WarningCode.DOCUMENT_NOT_FOUND for warning in analysis.warnings
    )


def test_low_score_photo_exposes_uncertainty_and_explicit_variant_warning(
    tmp_path: Path,
) -> None:
    # Given a deliberately sparse one-stroke source
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    image_path = tmp_path / "sparse.png"
    Image.fromarray(image).save(image_path)
    config = PipelineConfig.from_detection_defaults()

    # When automatic selection is uncertain
    automatic = analyze_image(image_path, config)

    # Then it falls back and asks for a human preprocessing choice
    assert automatic.status is AnalysisStatus.PREPROCESSING_UNCERTAIN
    assert automatic.selected_preprocessing_variant is None
    assert any(
        warning.code is WarningCode.PREPROCESSING_UNCERTAIN
        for warning in automatic.warnings
    )

    # When a reviewer explicitly selects one visible variant
    chosen = analyze_image(
        image_path,
        config,
        preprocessing_variant="grayscale_adaptive",
    )

    # Then detection is not silent, even if confidence remains low
    assert chosen.selected_preprocessing_variant == "grayscale_adaptive"
    assert any(
        warning.code is WarningCode.VERY_LOW_CONFIDENCE for warning in chosen.warnings
    )


def test_proved_paper_grid_is_removed_and_reported_instead_of_blocking(
    tmp_path: Path,
) -> None:
    # Given many regularly spaced grid cells and one dark profile
    image = np.full((720, 720), 255, dtype=np.uint8)
    for x in range(20, 690, 44):
        for y in range(20, 690, 44):
            _ = cv2.rectangle(image, (x, y), (x + 32, y + 32), 160, 1)
    _ = cv2.rectangle(image, (160, 160), (560, 560), 0, 7)
    image_path = tmp_path / "grid-profile.png"
    Image.fromarray(image).save(image_path)
    config = replace(
        PipelineConfig.from_detection_defaults(),
        perspective_correction=False,
        preprocessing=replace(
            PipelineConfig.from_detection_defaults().preprocessing,
            enabled=False,
        ),
        detection=DetectionConfig(
            detect_closed_shapes=True,
            minimum_shape_area=800.0,
            minimum_shape_span=30.0,
            maximum_shape_candidates=64,
            border_margin_ratio=0.0,
        ),
    )

    # When the production analysis pipeline processes the page
    analysis = analyze_image(image_path, config)

    # Then the ruling is removed with evidence and the profile stays usable
    assert "GRAPH_PAPER_DETECTED" in tuple(
        warning.code.value for warning in analysis.warnings
    )
    assert analysis.status is AnalysisStatus.READY
    assert any(
        isinstance(entity, PolylineEntity) and entity.closed
        for entity in analysis.entities
    )


def test_analysis_keeps_missing_dimensions_explicit(tmp_path: Path) -> None:
    # Given a saved sketch with two connected structural strokes
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    _ = cv2.line(image, (40, 50), (40, 210), 0, 5)
    image_path = tmp_path / "sketch.png"
    Image.fromarray(image).save(image_path)

    # When the production analysis pipeline runs
    analysis = analyze_image(
        image_path,
        PipelineConfig.from_detection_defaults(),
    )

    # Then geometry is present but no physical value is invented
    assert len(analysis.entities) >= 2
    assert analysis.dimensions == ()
    assert any(
        warning.code.value == "UNKNOWN_DIMENSION" for warning in analysis.warnings
    )


def test_optional_region_preserves_original_image_coordinates(
    tmp_path: Path,
) -> None:
    # Given a drawing inside a configured crop region
    image = np.full((300, 400), 255, dtype=np.uint8)
    _ = cv2.line(image, (140, 120), (340, 120), 0, 5)
    image_path = tmp_path / "region.png"
    Image.fromarray(image).save(image_path)
    config = PipelineConfig(
        detection=DetectionConfig(
            minimum_line_length=60.0,
            hough_threshold=20,
            maximum_line_gap=12,
            maximum_candidates=12,
            border_margin_ratio=0.0,
        ),
        normalization=NormalizationConfig(
            minimum_line_length=40.0,
        ),
        source_region=(100, 80, 260, 160),
        auto_source_region=True,
        auto_region_margin_ratio=0.2,
    )

    # When only that region is analyzed
    analysis = analyze_image(image_path, config)

    # Then overlays stay aligned with the original photograph
    assert analysis.image_size == (400, 300)
    assert analysis.source_region == (100, 80, 260, 160)
    line_entities = tuple(
        entity for entity in analysis.entities if isinstance(entity, LineEntity)
    )
    assert line_entities
    assert (
        min(point.x for entity in line_entities for point in (entity.start, entity.end))
        >= 100.0
    )


def test_auto_region_restores_mixed_closed_shape_coordinates(
    tmp_path: Path,
) -> None:
    # Given a drawing with a rectangle and circle inside a photographed inset
    image = np.full((360, 480), 255, dtype=np.uint8)
    _ = cv2.rectangle(image, (100, 100), (220, 190), 0, 5)
    _ = cv2.circle(image, (350, 220), 42, 0, 5)
    image_path = tmp_path / "mixed-shapes.png"
    Image.fromarray(image).save(image_path)
    config = PipelineConfig(
        detection=DetectionConfig(
            detect_closed_shapes=True,
            minimum_shape_area=500.0,
            minimum_shape_radius=20.0,
            maximum_shape_candidates=12,
            border_margin_ratio=0.0,
        ),
        normalization=NormalizationConfig(minimum_line_length=20.0),
        auto_source_region=True,
        auto_region_margin_ratio=0.1,
    )

    # When automatic source-region analysis runs
    analysis = analyze_image(image_path, config)

    # Then mixed geometry is preserved in original-image coordinates
    assert analysis.source_region == (48, 36, 384, 288)
    assert any(isinstance(entity, CircleEntity) for entity in analysis.entities)
    assert any(
        isinstance(entity, PolylineEntity) and entity.closed
        for entity in analysis.entities
    )
    assert all(
        point.x >= 48.0 and point.y >= 36.0
        for entity in analysis.entities
        for point in (
            (entity.center,)
            if isinstance(entity, CircleEntity)
            else entity.points
            if isinstance(entity, PolylineEntity)
            else (entity.start, entity.end)
            if isinstance(entity, LineEntity)
            else ()
        )
    )


def test_pipeline_integrates_reviewable_ocr_dimensions(tmp_path: Path) -> None:
    # Given a blank image and an injected OCR adapter
    image_path = tmp_path / "ocr.png"
    Image.fromarray(np.full((120, 180), 255, dtype=np.uint8)).save(image_path)
    config = replace(PipelineConfig.from_detection_defaults(), ocr_enabled=True)

    # When the production pipeline receives OCR candidates
    analysis = analyze_image(image_path, config, _FakeOcrBackend())

    # Then parsed dimensions and source boxes reach the analysis model
    assert tuple(dimension.kind for dimension in analysis.dimensions) == (
        DimensionKind.LENGTH,
    )
    assert analysis.text_candidates[0].text == "150"
    assert analysis.text_candidates[0].bounds == (20.0, 20.0, 40.0, 16.0)


class _AdjacentTextOcrBackend:
    def recognize(self, image: np.ndarray) -> tuple[TextCandidate, ...]:
        assert image.ndim == 2
        return (
            TextCandidate(
                text="150",
                bounds=(150.0, 30.0, 40.0, 16.0),
                confidence=0.9,
            ),
        )


class _AmbiguousTextOcrBackend:
    def recognize(self, image: np.ndarray) -> tuple[TextCandidate, ...]:
        assert image.ndim == 2
        return (
            TextCandidate(
                text="150",
                bounds=(150.0, 87.0, 40.0, 16.0),
                confidence=0.9,
            ),
        )


def _single_line_config() -> PipelineConfig:
    """Return a deterministic config that skips preprocessing and perspective."""
    defaults = PipelineConfig.from_detection_defaults()
    return replace(
        defaults,
        perspective_correction=False,
        preprocessing=replace(defaults.preprocessing, enabled=False),
        ocr_enabled=True,
    )


def test_pipeline_suggests_an_unambiguous_ocr_dimension_target(
    tmp_path: Path,
) -> None:
    # Given one detected line and an OCR label positioned right beside it
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    image_path = tmp_path / "labeled-line.png"
    Image.fromarray(image).save(image_path)

    # When the production pipeline runs with OCR enabled
    analysis = analyze_image(
        image_path, _single_line_config(), _AdjacentTextOcrBackend()
    )

    # Then the recognized length is pre-linked to the single nearby line
    line_ids = {
        entity.entity_id
        for entity in analysis.entities
        if isinstance(entity, LineEntity)
    }
    assert analysis.dimensions
    dimension = analysis.dimensions[0]
    assert dimension.target_entity_ids
    assert dimension.target_entity_ids[0] in line_ids
    assert dimension.source == "ocr_suggested"
    assert dimension.status is DimensionStatus.NEEDS_REVIEW


def test_pipeline_leaves_ambiguous_ocr_dimension_untargeted(
    tmp_path: Path,
) -> None:
    # Given two parallel lines equally close to one OCR label
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    _ = cv2.line(image, (40, 140), (280, 140), 0, 5)
    image_path = tmp_path / "ambiguous-line.png"
    Image.fromarray(image).save(image_path)

    # When the production pipeline runs with OCR enabled
    config = _single_line_config()
    config = replace(
        config,
        normalization=replace(
            config.normalization,
            keep_largest_connected_component=False,
        ),
    )
    analysis = analyze_image(image_path, config, _AmbiguousTextOcrBackend())

    # Then no target is guessed between the two equally plausible lines
    assert analysis.dimensions
    assert analysis.dimensions[0].target_entity_ids == ()


def test_pipeline_records_perspective_metadata(tmp_path: Path) -> None:
    # Given a page photographed against a contrasting background
    image = np.full((520, 720), 30, dtype=np.uint8)
    quad = np.array(
        [(80, 40), (640, 70), (665, 470), (50, 440)],
        dtype=np.int32,
    )
    _ = cv2.fillConvexPoly(image, quad, 240)
    _ = cv2.polylines(image, [quad], isClosed=True, color=0, thickness=6)
    image_path = tmp_path / "perspective.png"
    Image.fromarray(image).save(image_path)
    config = replace(
        PipelineConfig.from_detection_defaults(),
        perspective_correction=True,
    )

    # When perspective correction is enabled
    analysis = analyze_image(image_path, config)

    # Then the source page and inverse map remain inspectable
    assert analysis.perspective_transform is not None
    assert len(analysis.perspective_transform.inverse_matrix) == 9
    assert analysis.perspective_transform.target_size[0] > 300
    assert analysis.perspective_transform.target_size[1] > 300


def test_pipeline_retries_page_detection_inside_auto_region(
    tmp_path: Path,
) -> None:
    # Given a page boundary that is too small to qualify in the full photograph
    image = np.full((400, 600), 30, dtype=np.uint8)
    _ = cv2.rectangle(image, (150, 100), (420, 270), 240, thickness=-1)
    _ = cv2.rectangle(image, (150, 100), (420, 270), 0, thickness=5)
    image_path = tmp_path / "inset-page.png"
    Image.fromarray(image).save(image_path)
    config = PipelineConfig(
        detection=DetectionConfig(border_margin_ratio=0.0),
        normalization=NormalizationConfig(
            minimum_line_length=20.0,
            keep_largest_connected_component=False,
        ),
        auto_source_region=True,
        auto_region_margin_ratio=0.1,
        perspective_correction=True,
    )

    # When perspective analysis runs with automatic region selection
    analysis = analyze_image(image_path, config)

    # Then the page is rectified and the selected region remains provenance
    assert analysis.perspective_transform is not None
    assert analysis.source_region == (60, 40, 480, 320)
