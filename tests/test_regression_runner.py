from pathlib import Path

from scripts.run_preprocessing_regression import sample_paths


def test_regression_runner_accepts_png_and_jpeg_sample_directories(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "2.png").write_bytes(b"png")
    _ = (tmp_path / "1.JPG").write_bytes(b"jpg")
    _ = (tmp_path / "ignore.txt").write_bytes(b"text")

    assert sample_paths((tmp_path,)) == (
        (tmp_path / "1.JPG").resolve(),
        (tmp_path / "2.png").resolve(),
    )
