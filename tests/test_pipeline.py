from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cadraw.detection import DetectionConfig
from cadraw.models import LineEntity
from cadraw.normalization import NormalizationConfig
from cadraw.pipeline import PipelineConfig, analyze_image


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
    )

    # When only that region is analyzed
    analysis = analyze_image(image_path, config)

    # Then overlays stay aligned with the original photograph
    assert analysis.image_size == (400, 300)
    line_entities = tuple(
        entity for entity in analysis.entities if isinstance(entity, LineEntity)
    )
    assert line_entities
    assert (
        min(point.x for entity in line_entities for point in (entity.start, entity.end))
        >= 100.0
    )
