#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "ezdxf>=1.3.0",
#   "numpy>=2.0.0",
#   "opencv-python>=4.10.0",
#   "pillow>=10.4.0",
# ]
# ///
# ─── How to run ───
# uv run scripts/run_poc.py

"""Generate and execute the reproducible Phase 1 fixture."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, cast, final, override

from PIL import Image, ImageDraw, ImageFont

from cadraw.calculator import calculate_document
from cadraw.detection import DetectionConfig
from cadraw.dimensions import (
    associate_dimension,
    confirm_dimension_text,
    parse_dimension_text,
)
from cadraw.export import export_dxf, validate_dxf
from cadraw.models import Dimension, DimensionId, DrawingAnalysis, LineEntity
from cadraw.normalization import NormalizationConfig
from cadraw.pipeline import PipelineConfig, analyze_image
from cadraw.preview import PreviewConfig, render_preview

_SAMPLE_DIMENSION_COUNT: Final = 4
_MIN_VERTICAL_TARGETS: Final = 2
_ARGUMENTS_ERROR: Final = "arguments"
_DIMENSIONS_ERROR: Final = "dimensions"


@dataclass(frozen=True, slots=True)
class PocOptions:
    """Command-line options for the local fixture runner."""

    output_directory: Path
    input_path: Path | None
    dimension_texts: tuple[str, ...] | None
    confirm_length: float | None
    render_preview: bool


@dataclass(frozen=True, slots=True)
class PocSummary:
    """JSON-safe machine summary of one fixture run."""

    image: str
    dimension_texts: tuple[str, ...]
    geometry_entities: int
    connections: int
    orientations: tuple[str, ...]
    scale_mm_per_source_unit: float | None
    dxf_entities: int
    dxf_types: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]


@final
class PocUsageError(Exception):
    """Raised for invalid fixture-runner options."""

    def __init__(self, option: str) -> None:
        """Store the invalid option for a concise CLI message."""
        super().__init__()
        self.option = option

    @override
    def __str__(self) -> str:
        return f"알 수 없는 옵션 또는 값 누락: {self.option}"


def _parse_options(argv: Sequence[str]) -> PocOptions:
    """Parse the small fixture runner command surface."""
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    _ = parser.add_argument("--input", type=Path)
    _ = parser.add_argument("--dimensions")
    _ = parser.add_argument("--confirm-length", type=float)
    _ = parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    _ = parser.add_argument("--no-preview", action="store_true")
    try:
        parsed = parser.parse_args(list(argv))
    except (argparse.ArgumentError, SystemExit) as error:
        raise PocUsageError(_ARGUMENTS_ERROR) from error
    input_path = cast("Path | None", parsed.input)
    dimensions_option = cast("str | None", parsed.dimensions)
    confirm_length = cast("float | None", parsed.confirm_length)
    output_directory = cast("Path", parsed.output_dir)
    no_preview = cast("bool", parsed.no_preview)
    if dimensions_option is not None and confirm_length is not None:
        raise PocUsageError(_DIMENSIONS_ERROR)
    dimension_texts = (
        tuple(text.strip() for text in dimensions_option.split(",") if text.strip())
        if dimensions_option is not None
        else None
    )
    if dimension_texts is not None and len(dimension_texts) != _SAMPLE_DIMENSION_COUNT:
        raise PocUsageError(_DIMENSIONS_ERROR)
    return PocOptions(
        output_directory=output_directory,
        input_path=input_path,
        dimension_texts=dimension_texts,
        confirm_length=confirm_length,
        render_preview=not no_preview,
    )


def _fixture_image() -> Image.Image:
    """Build a slightly skewed and interrupted four-stroke sketch."""
    image = Image.new("L", (1200, 900), color=250)
    draw = ImageDraw.Draw(image)
    ink = 25
    stroke_width = 8
    segments = (
        ((160, 140), (930, 146)),
        ((165, 148), (165, 400)),
        ((165, 412), (165, 690)),
        ((165, 690), (400, 594)),
        ((412, 589), (650, 480)),
        ((650, 480), (650, 590)),
        ((650, 603), (650, 730)),
    )
    for start, end in segments:
        draw.line((start, end), fill=ink, width=stroke_width)
    font = ImageFont.load_default()
    draw.text((380, 105), "500", fill=ink, font=font)
    draw.text((88, 360), "1050", fill=ink, font=font)
    draw.text((400, 610), "45 deg", fill=ink, font=font)
    draw.text((672, 600), "30", fill=ink, font=font)
    return image


def _add_manual_dimension(
    analysis: DrawingAnalysis,
    value: float,
) -> DrawingAnalysis:
    """Attach one explicitly confirmed length to the longest detected line."""
    lines = tuple(
        entity for entity in analysis.entities if isinstance(entity, LineEntity)
    )
    if not lines:
        return analysis
    target = max(lines, key=lambda entity: entity.length)
    dimension = associate_dimension(
        confirm_dimension_text(
            parse_dimension_text(str(value), DimensionId("manual_length")),
            str(value),
        ),
        target.entity_id,
    )
    return replace(analysis, dimensions=(dimension,))


def _confirmed_dimension(
    text: str,
    target: LineEntity,
    index: int,
) -> Dimension:
    """Create one user-confirmed value attached to a selected line."""
    candidate = parse_dimension_text(text, DimensionId(f"manual_{index}"))
    return associate_dimension(
        confirm_dimension_text(candidate, text),
        target.entity_id,
    )


def _add_sample_dimensions(
    analysis: DrawingAnalysis,
    texts: tuple[str, ...],
) -> DrawingAnalysis:
    """Map four explicit sample values by geometry role, not image ratio."""
    lines = tuple(
        entity for entity in analysis.entities if isinstance(entity, LineEntity)
    )
    horizontals = tuple(
        line for line in lines if line.orientation.value == "horizontal"
    )
    verticals = tuple(
        sorted(
            (line for line in lines if line.orientation.value == "vertical"),
            key=lambda line: min(line.start.y, line.end.y),
        ),
    )
    diagonals = tuple(line for line in lines if line.orientation.value == "diagonal")
    if not horizontals or len(verticals) < _MIN_VERTICAL_TARGETS or not diagonals:
        return analysis
    targets = (horizontals[0], verticals[0], diagonals[0], verticals[-1])
    return replace(
        analysis,
        dimensions=tuple(
            _confirmed_dimension(text, target, index)
            for index, (text, target) in enumerate(zip(texts, targets, strict=True))
        ),
    )


def _pipeline_config(input_path: Path | None) -> PipelineConfig:
    """Use a documented crop for the supplied camera sample."""
    if input_path is None:
        return PipelineConfig(
            detection=DetectionConfig(
                minimum_line_length=120.0,
                hough_threshold=45,
                maximum_line_gap=24,
                maximum_candidates=48,
            ),
            normalization=NormalizationConfig(
                endpoint_snap_distance=32.0,
                line_merge_tolerance=60.0,
                minimum_line_length=80.0,
            ),
        )
    return PipelineConfig(
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


def run(options: PocOptions) -> PocSummary:
    """Generate the fixture, execute the pipeline, and write artifacts."""
    options.output_directory.mkdir(parents=True, exist_ok=True)
    image_path = options.output_directory / "sample_hand_drawing.png"
    preview_path = options.output_directory / "sample_preview.png"
    dxf_path = options.output_directory / "sample_drawing.dxf"
    summary_path = options.output_directory / "sample_summary.json"
    if options.input_path is None:
        _fixture_image().save(image_path)
    else:
        image_path = options.input_path

    config = _pipeline_config(options.input_path)
    analysis = analyze_image(image_path, config)
    if options.dimension_texts is not None:
        analysis = _add_sample_dimensions(analysis, options.dimension_texts)
    elif options.confirm_length is not None:
        analysis = _add_manual_dimension(analysis, options.confirm_length)
    document = calculate_document(analysis)
    if options.render_preview:
        render_preview(
            document,
            PreviewConfig(width=1200, height=900),
        ).save(preview_path)
    _ = export_dxf(document, dxf_path)
    validation = validate_dxf(dxf_path)
    summary = PocSummary(
        image=str(image_path),
        dimension_texts=tuple(dimension.raw_text for dimension in analysis.dimensions),
        geometry_entities=len(document.entities),
        connections=len(analysis.connections),
        orientations=tuple(
            entity.orientation.value
            for entity in document.entities
            if isinstance(entity, LineEntity)
        ),
        scale_mm_per_source_unit=document.scale_mm_per_source_unit,
        dxf_entities=validation.total_entities,
        dxf_types=validation.entity_counts,
        warnings=tuple(warning.code.value for warning in document.warnings),
    )
    _ = summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture from command-line arguments."""
    arguments = sys.argv[1:] if argv is None else argv
    if "--help" in arguments or "-h" in arguments:
        print(
            "Usage: uv run scripts/run_poc.py {options}".format(
                options=(
                    "[--input IMAGE] [--dimensions A,B,C,D] "
                    "[--confirm-length MM] "
                    "[--output-dir DIR] [--no-preview]"
                ),
            ),
        )
        return 0
    try:
        options = _parse_options(arguments)
    except PocUsageError as error:
        print(str(error), file=sys.stderr)
        return 2
    summary = run(options)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
