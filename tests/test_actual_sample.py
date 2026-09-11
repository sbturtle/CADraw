from pathlib import Path

from scripts.run_poc import PocOptions, run

from cadraw.detection import DetectionConfig
from cadraw.export import validate_dxf
from cadraw.models import LineEntity
from cadraw.normalization import NormalizationConfig
from cadraw.pipeline import PipelineConfig, analyze_image

_SAMPLE = Path(__file__).parents[1] / "sample" / "KakaoTalk_20260909_141552817.jpg"


def test_default_pipeline_keeps_the_structural_chain_from_camera_noise() -> None:
    # Given the supplied camera photograph with no special per-image settings
    analysis = analyze_image(_SAMPLE, PipelineConfig.from_detection_defaults())

    # Then the application defaults keep the four-stroke structural chain
    assert len(analysis.entities) == 4
    assert len(analysis.connections) == 3


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
