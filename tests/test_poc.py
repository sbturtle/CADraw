from pathlib import Path

from scripts.run_poc import PocOptions, run


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

    # Then the deterministic four-line slice and DXF are reproduced
    assert summary.geometry_entities == 4
    assert summary.dxf_entities == 5
    assert set(summary.orientations) == {"horizontal", "vertical", "diagonal"}
    assert (tmp_path / "sample_hand_drawing.png").exists()
    assert (tmp_path / "sample_preview.png").exists()
    assert (tmp_path / "sample_drawing.dxf").exists()
    assert (tmp_path / "sample_summary.json").exists()
