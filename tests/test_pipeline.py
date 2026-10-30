import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from cadraw.detection import DetectionConfig
from cadraw.models import (
    AnalysisStatus,
    ArcEntity,
    CircleEntity,
    DimensionKind,
    DimensionStatus,
    EntityId,
    LineEntity,
    Point,
    PolylineEntity,
    TextCandidate,
    WarningCode,
)
from cadraw.normalization import NormalizationConfig
from cadraw.perspective import transform_point
from cadraw.pipeline import PipelineConfig, analyze_image, restore_source_geometry
from cadraw.text_filter import TextStrokeConfig


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


def test_a_plate_with_many_slots_is_not_treated_as_noise(tmp_path: Path) -> None:
    # Given one plate outline holding two rows of equal slots and nothing else
    image = np.full((900, 1200), 236, dtype=np.uint8)
    _ = cv2.rectangle(image, (280, 230), (920, 670), 30, 3)
    for row in (330, 530):
        for left in range(340, 840, 100):
            _ = cv2.rectangle(image, (left, row), (left + 50, row + 60), 30, 2)
    image_path = tmp_path / "slotted-plate.png"
    Image.fromarray(image).save(image_path)

    # When the shipped detection settings analyze it as an already flat page
    analysis = analyze_image(
        image_path,
        replace(PipelineConfig.from_detection_defaults(), perspective_correction=False),
    )

    # Then the slots are the plate's holes and export is not held back
    holes = [
        entity
        for entity in analysis.entities
        if isinstance(entity, PolylineEntity) and entity.role.value == "hole"
    ]
    assert len(holes) == 10
    assert analysis.status is not AnalysisStatus.TOO_MUCH_NOISE


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


def test_isolated_printed_frame_rule_is_dropped(tmp_path: Path) -> None:
    # Given a form whose closing rule crosses the sheet below the drawing
    image = np.full((900, 700), 252, dtype=np.uint8)
    _ = cv2.line(image, (200, 300), (500, 300), 0, 5)
    _ = cv2.line(image, (200, 300), (200, 600), 0, 5)
    _ = cv2.line(image, (30, 800), (670, 800), 0, 2)
    image_path = tmp_path / "framed.png"
    Image.fromarray(image).save(image_path)

    # When the production analysis pipeline runs
    analysis = analyze_image(image_path, PipelineConfig.from_detection_defaults())

    # Then the drawing's own strokes are kept and the rule is not
    lines = [entity for entity in analysis.entities if isinstance(entity, LineEntity)]
    assert lines
    assert all(abs(entity.end.x - entity.start.x) < 600.0 for entity in lines)


def test_a_page_spanning_stroke_joined_to_the_drawing_is_kept(
    tmp_path: Path,
) -> None:
    # Given a part whose baseline runs the width of the sheet near its edge
    image = np.full((900, 700), 252, dtype=np.uint8)
    _ = cv2.line(image, (70, 800), (630, 800), 0, 5)
    _ = cv2.line(image, (70, 800), (70, 480), 0, 5)
    _ = cv2.line(image, (630, 800), (630, 480), 0, 5)
    image_path = tmp_path / "wide-part.png"
    Image.fromarray(image).save(image_path)

    # When the production analysis pipeline runs
    analysis = analyze_image(image_path, PipelineConfig.from_detection_defaults())

    # Then the baseline survives, because it is joined to the rest of the part
    lines = [entity for entity in analysis.entities if isinstance(entity, LineEntity)]
    assert any(abs(entity.end.x - entity.start.x) > 500.0 for entity in lines)


class _StrokeLabelBackend:
    """Reports the box of a `117` label drawn with pen-width strokes."""

    def __init__(self, box: tuple[int, int, int, int]) -> None:
        self._box: tuple[int, int, int, int] = box

    def recognize(self, image: np.ndarray) -> tuple[TextCandidate, ...]:
        del image
        left, top, right, bottom = self._box
        return (
            TextCandidate(
                text="117",
                bounds=(left, top, right - left, bottom - top),
                confidence=0.95,
                quad=(
                    Point(left, top),
                    Point(right, top),
                    Point(right, bottom),
                    Point(left, bottom),
                ),
            ),
        )


def _labelled_outline(tmp_path: Path) -> tuple[Path, tuple[int, int, int, int]]:
    image = np.full((600, 800), 255, dtype=np.uint8)
    _ = cv2.line(image, (100, 100), (700, 100), 0, 5)
    _ = cv2.line(image, (100, 100), (100, 500), 0, 5)
    (width, height), _ = cv2.getTextSize("117", cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
    _ = cv2.putText(image, "117", (330, 330), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
    image_path = tmp_path / "labelled-outline.png"
    Image.fromarray(image).save(image_path)
    return image_path, (330, 330 - height, 330 + width, 330)


def _entity_points(entity: object) -> tuple[Point, ...]:
    if isinstance(entity, LineEntity):
        return (entity.start, entity.end)
    if isinstance(entity, PolylineEntity):
        return entity.points
    return ()


def test_label_strokes_are_removed_while_the_outline_stays(tmp_path: Path) -> None:
    # Given an L outline and, apart from it, a pen-drawn `117` label
    image_path, box = _labelled_outline(tmp_path)
    config = replace(
        _single_line_config(),
        normalization=replace(
            _single_line_config().normalization,
            keep_largest_connected_component=False,
        ),
    )
    backend = _StrokeLabelBackend(box)
    left, top, right, bottom = box

    def inside_label(entity: object) -> bool:
        points = _entity_points(entity)
        return bool(points) and all(
            left - 5 <= point.x <= right + 5 and top - 5 <= point.y <= bottom + 5
            for point in points
        )

    # When the pipeline runs with, and then without, lettering removal
    kept = analyze_image(
        image_path,
        replace(config, text_strokes=TextStrokeConfig(enabled=False)),
        backend,
    )
    filtered = analyze_image(image_path, config, backend)

    # Then only the label's own strokes leave the geometry
    assert any(inside_label(entity) for entity in kept.entities)
    assert not any(inside_label(entity) for entity in filtered.entities)
    long_lines = [
        entity
        for entity in filtered.entities
        if isinstance(entity, LineEntity) and entity.length > 350.0
    ]
    assert len(long_lines) == 2
    assert WarningCode.TEXT_STROKES_REMOVED in {
        warning.code for warning in filtered.warnings
    }
    assert filtered.text_candidates[0].quad is not None


class _DisagreeingReader:
    def read(self, crop: np.ndarray) -> str | None:
        assert crop.size
        return "180"


def test_a_value_read_differently_twice_is_flagged_on_its_line(
    tmp_path: Path,
) -> None:
    # Given one line, its `150` label, and a second reader that sees `180`
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    image_path = tmp_path / "second-reading.png"
    Image.fromarray(image).save(image_path)

    # When the pipeline cross-checks OCR
    analysis = analyze_image(
        image_path,
        _single_line_config(),
        _AdjacentTextOcrBackend(),
        second_reader=_DisagreeingReader(),
    )

    # Then the suggestion is uncertain and the warning names its line
    dimension = analysis.dimensions[0]
    assert dimension.status is DimensionStatus.UNCERTAIN
    warning = next(
        warning
        for warning in analysis.warnings
        if warning.code is WarningCode.UNCERTAIN_TEXT
    )
    assert warning.entity_ids == dimension.target_entity_ids
    assert "150 / 180" in warning.message
    assert analysis.text_candidates[0].alternate_text == "180"


def test_workbench_config_reads_labels_and_the_optional_vision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given no vision model, then a configured one
    monkeypatch.delenv("CADRAW_VISION_MODEL", raising=False)
    plain = PipelineConfig.for_workbench()
    monkeypatch.setenv("CADRAW_VISION_MODEL", "qwen3-vl:2b-instruct")
    checked = PipelineConfig.for_workbench()

    # Then OCR is on in the app while tests' defaults stay model-free
    assert not PipelineConfig.from_detection_defaults().ocr_enabled
    assert plain.ocr_enabled
    assert plain.ocr_cross_check_model is None
    assert checked.ocr_cross_check_model == "qwen3-vl:2b-instruct"


def test_a_circle_on_a_skewed_page_keeps_its_photographed_radius(
    tmp_path: Path,
) -> None:
    # Given a page photographed steeply, with a 30 px circle near its far edge
    image = np.full((560, 760), 30, dtype=np.uint8)
    quad = np.array([(250, 40), (510, 40), (700, 520), (60, 520)], dtype=np.int32)
    _ = cv2.fillConvexPoly(image, quad, 240)
    _ = cv2.polylines(image, [quad], isClosed=True, color=0, thickness=6)
    _ = cv2.circle(image, (380, 140), 30, 0, 3)
    _ = cv2.rectangle(image, (230, 300), (530, 460), 0, 4)
    image_path = tmp_path / "steep-page.png"
    Image.fromarray(image).save(image_path)
    config = replace(
        PipelineConfig.from_detection_defaults(),
        perspective_correction=True,
    )

    # When the page is flattened, analyzed, and mapped back to the photograph
    analysis = analyze_image(image_path, config)

    # Then the circle has the size it has in the photograph, not the flat page
    assert analysis.perspective_transform is not None
    circles = [
        entity for entity in analysis.entities if isinstance(entity, CircleEntity)
    ]
    assert len(circles) == 1
    assert circles[0].center.distance_to(Point(380.0, 140.0)) <= 4.0
    assert abs(circles[0].radius - 30.0) <= 3.0


def test_a_restored_arc_ends_where_its_mapped_ends_land() -> None:
    # Given an arc on a flattened page and a steep warp back to the photograph
    arc = ArcEntity(
        entity_id=EntityId("arc"),
        center=Point(200.0, 200.0),
        radius=80.0,
        start_angle_degrees=0.0,
        end_angle_degrees=90.0,
        confidence=0.9,
    )
    warp = (0.6, -0.2, 140.0, 0.05, 0.9, 30.0, 0.0006, 0.0004, 1.0)

    # When it is restored to source coordinates
    restored = restore_source_geometry(arc, Point(0.0, 0.0), warp)

    # Then both of its ends lie where the page's arc ends map to, so a line
    # that met an end on the page still meets it in the photograph
    assert isinstance(restored, ArcEntity)
    for angle, original in (
        (restored.start_angle_degrees, Point(280.0, 200.0)),
        (restored.end_angle_degrees, Point(200.0, 280.0)),
    ):
        mapped = transform_point(original, warp)
        end = Point(
            restored.center.x + restored.radius * math.cos(math.radians(angle)),
            restored.center.y + restored.radius * math.sin(math.radians(angle)),
        )
        assert end.distance_to(mapped) <= 0.01
