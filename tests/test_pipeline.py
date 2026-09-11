from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cadraw.detection import DetectionConfig
from cadraw.models import (
    CircleEntity,
    DimensionKind,
    LineEntity,
    PolylineEntity,
    TextCandidate,
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
