from pathlib import Path

from PIL import Image, ImageDraw
from scripts.run_poc import PocOptions, run

from cadraw.export import validate_dxf


def test_reproducible_poc_runner_writes_expected_artifacts(tmp_path: Path) -> None:
    # Given an isolated artifact directory
    # When the checked-in Phase 1 runner executes with one confirmed length
    summary = run(
        PocOptions(
            output_directory=tmp_path,
            input_path=None,
            dimension_texts=None,
            confirm_length=500.0,
            render_preview=True,
        ),
    )

    # Then the deterministic four-segment slice becomes one open polyline
    assert summary.geometry_entities == 4
    assert summary.dxf_entities == 2
    assert set(summary.orientations) == {"horizontal", "vertical", "diagonal"}
    assert (tmp_path / "sample_hand_drawing.png").exists()
    assert (tmp_path / "sample_preview.png").exists()
    assert (tmp_path / "sample_foreground.png").exists()
    assert (tmp_path / "sample_drawing.dxf").exists()
    assert (tmp_path / "sample_summary.json").exists()


def test_runner_does_not_reuse_legacy_region_for_arbitrary_input(
    tmp_path: Path,
) -> None:
    # Given an arbitrary source image that is not the legacy sample
    image = Image.new("L", (320, 240), color=255)
    draw = ImageDraw.Draw(image)
    draw.line((80, 60, 260, 60), fill=0, width=5)
    image_path = tmp_path / "arbitrary.png"
    image.save(image_path)

    # When the real runner analyzes that image without a user ROI
    summary = run(
        PocOptions(
            output_directory=tmp_path / "output",
            input_path=image_path,
            dimension_texts=None,
            confirm_length=None,
            render_preview=False,
        ),
    )

    # Then it does not silently crop to coordinates from another photograph
    assert summary.source_region is None


def test_enhanced_runner_exports_detected_closed_shapes(tmp_path: Path) -> None:
    # Given a photographed-style input with one rectangle and one circle
    image = Image.new("L", (480, 360), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 220, 190), outline=0, width=5)
    draw.ellipse((308, 178, 392, 262), outline=0, width=5)
    image_path = tmp_path / "mixed-shapes.png"
    image.save(image_path)

    # When the enhanced runner opts into automatic region and shape detection
    summary = run(
        PocOptions(
            output_directory=tmp_path / "output",
            input_path=image_path,
            dimension_texts=None,
            confirm_length=None,
            render_preview=True,
            auto_region=True,
            detect_shapes=True,
        ),
    )

    # Then the closed shapes survive into the real DXF output
    validation = validate_dxf(tmp_path / "output" / "sample_drawing.dxf")
    assert "circle" in summary.entity_types
    assert "polyline" in summary.entity_types
    assert validation.count("CIRCLE") == 1
    assert validation.count("LWPOLYLINE") >= 1
