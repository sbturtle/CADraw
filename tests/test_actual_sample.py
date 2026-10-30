from math import isfinite
from pathlib import Path

import pytest
from scripts.run_poc import PocOptions, run

from cadraw.detection import DetectionConfig
from cadraw.export import validate_dxf
from cadraw.models import CircleEntity, LineEntity, Orientation, Point
from cadraw.normalization import NormalizationConfig
from cadraw.pipeline import PipelineConfig, analyze_image

_PROJECT_ROOT = Path(__file__).parents[1]
_SAMPLE = next(
    path
    for path in (
        _PROJECT_ROOT / "sample" / "KakaoTalk_20260909_141552817.jpg",
        _PROJECT_ROOT / "sample" / "1.jpg",
    )
    if path.exists()
)
_PURE_BACKGROUND_SAMPLE = _PROJECT_ROOT / "sample" / "pure_background" / "1.png"
_PEN2STEP = _PROJECT_ROOT / "samples" / "pen2step"
_RETRACE_TOLERANCE = 12.0
_REAL_SAMPLES = tuple(
    sorted(
        {
            path
            for directory in (
                _PROJECT_ROOT / "samples" / "real",
                _SAMPLE.parent,
            )
            for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
            for path in directory.glob(pattern)
        },
    ),
)


def test_default_pipeline_keeps_the_structural_chain_from_camera_noise() -> None:
    # Given the supplied camera photograph with no special per-image settings
    analysis = analyze_image(_SAMPLE, PipelineConfig.from_detection_defaults())

    # Then the application defaults keep the four-stroke structural chain
    assert len(analysis.entities) == 4
    assert len(analysis.connections) == 3


def test_default_pipeline_records_deterministic_preprocessing_evidence() -> None:
    # Given the supplied camera photograph
    analysis = analyze_image(_SAMPLE, PipelineConfig.from_detection_defaults())

    # Then all independent variants expose finite selection evidence
    assert tuple(report.variant for report in analysis.preprocessing_reports) == (
        "grayscale_adaptive",
        "clahe_adaptive",
        "shadow_removed_adaptive",
        "blur_otsu",
        "flattened_global",
    )
    assert analysis.selected_preprocessing_variant in {
        report.variant for report in analysis.preprocessing_reports
    }
    assert all(
        report.processing_time_ms >= 0.0 and report.score == report.score
        for report in analysis.preprocessing_reports
    )


def test_perspective_is_rejected_when_unwarped_geometry_is_stronger() -> None:
    # Given the clean first sample whose page warp loses structural strokes
    config = PipelineConfig.from_detection_defaults()

    # When automatic perspective correction compares both geometry streams
    analysis = analyze_image(_PURE_BACKGROUND_SAMPLE, config)

    # Then the source-coordinate analysis keeps the stronger unwarped result
    assert analysis.perspective_transform is None
    lines = {
        entity.entity_id: entity
        for entity in analysis.entities
        if isinstance(entity, LineEntity)
    }
    connected = {
        frozenset((connection.first_entity_id, connection.second_entity_id))
        for connection in analysis.connections
    }

    def joined(first: LineEntity, second: LineEntity) -> bool:
        return frozenset((first.entity_id, second.entity_id)) in connected

    # And the sketch's four members are present as one connected chain: the
    # 1050 stroke is whole even though Hough broke it, the 100 horizontal
    # meets its top, the bend meets its bottom, and the 150 stroke follows.
    tall = max(
        (line for line in lines.values() if line.orientation is Orientation.VERTICAL),
        key=lambda line: line.length,
    )
    assert tall.length > 400.0
    top = min(tall.start.y, tall.end.y)
    horizontal = next(
        line
        for line in lines.values()
        if line.orientation is Orientation.HORIZONTAL and joined(line, tall)
    )
    assert abs(horizontal.start.y - top) < 5.0
    bend = next(
        line
        for line in lines.values()
        if line.orientation is Orientation.DIAGONAL
        and joined(line, tall)
        and max(line.start.y, line.end.y) > top + 400.0
    )
    assert any(
        line.orientation is Orientation.VERTICAL
        and line is not tall
        and joined(line, bend)
        for line in lines.values()
    )


@pytest.mark.parametrize(
    "sample_path",
    _REAL_SAMPLES,
)
def test_all_real_samples_record_repeatable_preprocessing_evidence(
    sample_path: Path,
) -> None:
    # Given each supplied real photograph
    config = PipelineConfig.from_detection_defaults()

    # When the deterministic preprocessing/detection pipeline runs twice
    first = analyze_image(sample_path, config)
    second = analyze_image(sample_path, config)

    # Then selection/status/geometry are repeatable and timings are measurable
    assert len(first.preprocessing_reports) == 5
    assert first.selected_preprocessing_variant in {
        report.variant for report in first.preprocessing_reports
    }
    assert all(
        isfinite(report.processing_time_ms) and report.processing_time_ms >= 0.0
        for report in first.preprocessing_reports
    )
    assert (
        first.status,
        first.selected_preprocessing_variant,
        tuple(entity.entity_id for entity in first.entities),
    ) == (
        second.status,
        second.selected_preprocessing_variant,
        tuple(entity.entity_id for entity in second.entities),
    )


def test_supplied_hand_drawing_reduces_to_four_connected_strokes() -> None:
    # Given the supplied camera photograph and its documented ROI
    config = PipelineConfig(
        detection=DetectionConfig(
            minimum_line_length=70.0,
            hough_threshold=25,
            maximum_line_gap=28,
            maximum_candidates=64,
            border_margin_ratio=0.0,
        ),
        normalization=NormalizationConfig(
            horizontal_angle_tolerance_degrees=7.0,
            vertical_angle_tolerance_degrees=7.0,
            endpoint_snap_distance=45.0,
            line_merge_tolerance=45.0,
            minimum_line_length=60.0,
            keep_largest_connected_component=True,
        ),
        source_region=(500, 680, 550, 1150),
    )

    # When the production image pipeline analyzes it
    analysis = analyze_image(_SAMPLE, config)

    # Then the structural chain is separated from the handwritten labels
    assert analysis.image_size == (1920, 2560)
    assert len(analysis.entities) == 4
    assert tuple(
        detected.entity_id for detected in analysis.detected_entities
    ) == tuple(entity.entity_id for entity in analysis.entities)
    assert len(analysis.connections) == 3
    assert {
        entity.orientation.value
        for entity in analysis.entities
        if isinstance(entity, LineEntity)
    } == {
        "horizontal",
        "vertical",
        "diagonal",
    }


def test_supplied_sample_runner_exports_all_confirmed_values(
    tmp_path: Path,
) -> None:
    # Given the supplied photograph and the four user-confirmed values
    summary = run(
        PocOptions(
            output_directory=tmp_path,
            input_path=_SAMPLE,
            dimension_texts=("100", "1050", "20°", "150"),
            confirm_length=None,
            render_preview=True,
        ),
    )

    # Then each confirmed label is exported without fabricating geometry
    assert summary.dimension_texts == ("100", "1050", "20°", "150")
    assert summary.dxf_entities == 5
    dxf_path = tmp_path / "sample_drawing.dxf"
    assert dxf_path.exists()
    assert validate_dxf(dxf_path).text_values == (
        "100 mm",
        "1050 mm",
        "20°",
        "150 mm",
    )


def test_supplied_sample_reconstructs_entered_segment_lengths(
    tmp_path: Path,
) -> None:
    # Given the same camera sample with four explicit segment lengths
    summary = run(
        PocOptions(
            output_directory=tmp_path,
            input_path=_SAMPLE,
            dimension_texts=("100", "1050", "20", "150"),
            confirm_length=None,
            render_preview=True,
        ),
    )

    # When the CAD-space DXF is reopened
    validation = validate_dxf(tmp_path / "sample_drawing.dxf")

    # Then every entered segment length survives the real sample pipeline
    lengths = tuple(round(length, 6) for length in validation.geometry_lengths)
    assert tuple(sorted(lengths)) == (20.0, 100.0, 150.0, 1050.0)
    assert "UNKNOWN_ANGLE" in summary.warnings


@pytest.mark.parametrize(
    ("name", "radii"),
    [
        ("real_11_mv.jpg", (91.0, 63.0, 24.0)),
        ("real_17_mv.jpg", (95.0, 56.0)),
        ("real_18_mv.jpg", (138.0, 49.0)),
    ],
)
def test_an_end_view_keeps_every_concentric_circle_once(
    name: str,
    radii: tuple[float, ...],
) -> None:
    # Given another hand's end view of concentric circles crossed by centre lines
    analysis = analyze_image(_PEN2STEP / name, PipelineConfig.from_detection_defaults())
    circles = tuple(
        entity for entity in analysis.entities if isinstance(entity, CircleEntity)
    )
    rim = max(circles, key=lambda circle: circle.radius)

    # Then each drawn circle is found around the rim's centre
    for radius in radii:
        assert any(
            circle.center.distance_to(rim.center) <= 0.15 * rim.radius
            and abs(circle.radius - radius) <= 0.06 * radius
            for circle in circles
        ), radius

    # And no detector line re-traces a circle's stroke as a chord
    def on_rim(point: Point, circle: CircleEntity) -> bool:
        return abs(point.distance_to(circle.center) - circle.radius) <= (
            _RETRACE_TOLERANCE
        )

    for line in analysis.entities:
        if isinstance(line, LineEntity):
            midpoint = Point(
                (line.start.x + line.end.x) / 2.0,
                (line.start.y + line.end.y) / 2.0,
            )
            assert not any(
                all(on_rim(point, circle) for point in (line.start, midpoint, line.end))
                for circle in circles
            )
