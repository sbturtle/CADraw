#!/usr/bin/env -S uv run python
"""Compare legacy and robust preprocessing on real drawing photographs."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Final, cast

from cadraw.models import DrawingAnalysis
from cadraw.pipeline import PipelineConfig, analyze_image

_DEFAULT_OUTPUT: Final = Path("artifacts") / "preprocessing_regression.json"
_SAMPLE_DIRECTORIES: Final = (Path("samples/real"), Path("sample"))
_IMAGE_PATTERNS: Final = (
    "*.jpg",
    "*.jpeg",
    "*.JPG",
    "*.JPEG",
    "*.png",
    "*.PNG",
)


def sample_paths(
    directories: tuple[Path, ...] = _SAMPLE_DIRECTORIES,
) -> tuple[Path, ...]:
    """Collect real photos from the preferred and compatibility directories."""
    paths = {
        path.resolve()
        for directory in directories
        if directory.exists()
        for pattern in _IMAGE_PATTERNS
        for path in directory.glob(pattern)
    }
    return tuple(sorted(paths))


def _sha256(path: Path) -> str:
    """Return an input fingerprint for reproducible sample identity."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(analysis: DrawingAnalysis) -> int:
    """Count line entities without changing geometry semantics."""
    return sum(entity.entity_type.value == "line" for entity in analysis.entities)


def _record(path: Path, *, detect_shapes: bool = False) -> dict[str, object]:
    """Run both paths and serialize observed regression evidence."""
    config = PipelineConfig.from_detection_defaults()
    if detect_shapes:
        config = replace(
            config,
            detection=replace(config.detection, detect_closed_shapes=True),
        )
    legacy_config = replace(
        config,
        perspective_correction=False,
        preprocessing=replace(config.preprocessing, enabled=False),
    )
    legacy_started = perf_counter()
    legacy = analyze_image(path, legacy_config)
    legacy_time = (perf_counter() - legacy_started) * 1000.0
    improved_started = perf_counter()
    improved = analyze_image(path, config)
    improved_time = (perf_counter() - improved_started) * 1000.0
    return {
        "image": str(path),
        "input_sha256": _sha256(path),
        "legacy": {
            "detected_line_count": _line_count(legacy),
            "normalized_line_count": len(legacy.entities),
            "status": legacy.status.value,
            "processing_time_ms": legacy_time,
        },
        "improved": {
            "detected_line_count": _line_count(improved),
            "normalized_line_count": len(improved.entities),
            "selected_variant": improved.selected_preprocessing_variant,
            "status": improved.status.value,
            "warning_codes": tuple(warning.code.value for warning in improved.warnings),
            "processing_time_ms": improved_time,
            "variant_reports": tuple(
                {
                    "variant": report.variant,
                    "score": report.score,
                    "processing_time_ms": report.processing_time_ms,
                }
                for report in improved.preprocessing_reports
            ),
        },
    }


def main() -> int:
    """Write a JSON comparison for every discovered real sample."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    _ = parser.add_argument(
        "--sample-dir",
        type=Path,
        action="append",
        help="sample directory to scan (repeatable)",
    )
    _ = parser.add_argument(
        "--detect-shapes",
        action="store_true",
        help="include the existing closed-shape detector",
    )
    options = parser.parse_args()
    output = cast("Path", options.output)
    requested_directories = cast("list[Path] | None", options.sample_dir)
    detect_shapes = cast("bool", options.detect_shapes)
    directories = (
        tuple(requested_directories)
        if requested_directories is not None
        else _SAMPLE_DIRECTORIES
    )
    records = tuple(
        _record(
            path,
            detect_shapes=detect_shapes,
        )
        for path in sample_paths(directories)
    )
    _ = output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(records)} sample records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
