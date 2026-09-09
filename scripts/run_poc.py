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

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from cadraw.calculator import calculate_document
from cadraw.dimensions import (
    associate_dimension,
    confirm_dimension_text,
    parse_dimension_text,
)
from cadraw.export import export_dxf, validate_dxf
from cadraw.models import (
    DimensionId,
    DrawingAnalysis,
    LineEntity,
)
from cadraw.detection import DetectionConfig
from cadraw.normalization import NormalizationConfig
from cadraw.pipeline import PipelineConfig, analyze_image
from cadraw.preview import PreviewConfig, render_preview


@dataclass(frozen=True, slots=True)
class PocOptions:
    """Command-line options for the local fixture runner."""

    output_directory: Path
    confirm_length: float | None
    render_preview: bool


@dataclass(frozen=True, slots=True)
class PocSummary:
    """JSON-safe machine summary of one fixture run."""

    image: str
    geometry_entities: int
    connections: int
    orientations: tuple[str, ...]
    scale_mm_per_source_unit: float | None
    dxf_entities: int
    dxf_types: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]


def _parse_options(argv: Sequence[str]) -> PocOptions:
    """Parse the small fixture runner command surface."""
    output_directory = Path("artifacts")
    confirm_length: float | None = None
    render_preview = True
    arguments = iter(argv)
    for argument in arguments:
        if argument == "--no-preview":
            render_preview = False
        elif argument == "--confirm-length":
            confirm_length = float(next(arguments))
        elif argument == "--output-dir":
            output_directory = Path(next(arguments))
        else:
            raise ValueError(f"알 수 없는 옵션: {argument}")
    return PocOptions(
        output_directory=output_directory,
        confirm_length=confirm_length,
        render_preview=render_preview,
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
    lines = tuple(entity for entity in analysis.entities if isinstance(entity, LineEntity))
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


def run(options: PocOptions) -> PocSummary:
    """Generate the fixture, execute the pipeline, and write artifacts."""
    options.output_directory.mkdir(parents=True, exist_ok=True)
    image_path = options.output_directory / "sample_hand_drawing.png"
    preview_path = options.output_directory / "sample_preview.png"
    dxf_path = options.output_directory / "sample_drawing.dxf"
    summary_path = options.output_directory / "sample_summary.json"
    _fixture_image().save(image_path)

    config = PipelineConfig(
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
    analysis = analyze_image(image_path, config)
    if options.confirm_length is not None:
        analysis = _add_manual_dimension(analysis, options.confirm_length)
    document = calculate_document(analysis)
    if options.render_preview:
        render_preview(
            document,
            PreviewConfig(width=1200, height=900),
        ).save(preview_path)
    export_dxf(document, dxf_path)
    validation = validate_dxf(dxf_path)
    summary = PocSummary(
        image=str(image_path),
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
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture from command-line arguments."""
    arguments = sys.argv[1:] if argv is None else argv
    options = _parse_options(arguments)
    summary = run(options)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
