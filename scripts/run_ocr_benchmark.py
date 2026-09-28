#!/usr/bin/env -S uv run python
"""Measure local OCR on the hand-labelled sample dimensions.

Every label in `docs/experiments/ocr-ground-truth.json` is matched to the
parsed dimension whose detector box contains the label's centre. A label is
read correctly when that dimension's value equals the written value. With
`--cross-check MODEL`, each value is read again by a local Ollama vision model
and the script also reports how often the two readers agreed and how often an
agreed value was right.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, cast

import cv2
import numpy as np

from cadraw.detection import load_grayscale_image
from cadraw.local_ocr import RapidOcrBackend
from cadraw.models import Dimension, DimensionStatus, TextCandidate
from cadraw.ocr import TextReader, recognize_dimensions
from cadraw.ollama_reader import OllamaTextReader

_GROUND_TRUTH: Final = Path("docs/experiments/ocr-ground-truth.json")
_DEFAULT_OUTPUT: Final = Path("artifacts") / "ocr_benchmark.json"
_NUMBER: Final = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class LabelResult:
    """How one hand-labelled value was read."""

    image: str
    expected: str
    read: str | None
    correct: bool
    status: str | None
    alternate: str | None


def _expected_value(text: str) -> float:
    """Return the numeric value a ground-truth label writes."""
    match = _NUMBER.search(text)
    if match is None:
        message = f"ground-truth label has no number: {text!r}"
        raise ValueError(message)
    return float(match.group())


def _contains(candidate: TextCandidate, center: tuple[float, float]) -> bool:
    """Return whether a detector box contains a label centre."""
    if candidate.quad is None:
        x, y, width, height = candidate.bounds
        return x <= center[0] <= x + width and y <= center[1] <= y + height
    polygon = np.asarray(
        [(point.x, point.y) for point in candidate.quad],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(polygon, center, measureDist=False) >= 0.0


def _match(
    dimensions: tuple[Dimension, ...],
    sources: tuple[TextCandidate, ...],
    center: tuple[float, float],
) -> tuple[Dimension, TextCandidate] | None:
    """Return the parsed dimension read from the box around a centre."""
    return next(
        (
            (dimension, source)
            for dimension, source in zip(dimensions, sources, strict=True)
            if _contains(source, center)
        ),
        None,
    )


def _sample_results(
    image_path: Path,
    labels: list[dict[str, object]],
    reader: TextReader | None,
) -> tuple[tuple[LabelResult, ...], float, bool | None]:
    """Read one photo and score each of its labels."""
    image = load_grayscale_image(image_path)
    start = perf_counter()
    result = recognize_dimensions(image, RapidOcrBackend(), second_reader=reader)
    elapsed = perf_counter() - start
    records: list[LabelResult] = []
    for label in labels:
        expected = cast("str", label["text"])
        x, y = cast("list[float]", label["center"])
        matched = _match(result.dimensions, result.dimension_sources, (x, y))
        dimension = None if matched is None else matched[0]
        records.append(
            LabelResult(
                image=str(image_path),
                expected=expected,
                read=None if dimension is None else dimension.raw_text,
                correct=dimension is not None
                and dimension.parsed_value == _expected_value(expected),
                status=None if dimension is None else dimension.status.value,
                alternate=None if matched is None else matched[1].alternate_text,
            ),
        )
    return tuple(records), elapsed, result.second_reader_available


def _summary(
    records: tuple[LabelResult, ...], seconds: list[float]
) -> dict[str, object]:
    """Aggregate the per-label results into the reported figures."""
    agreed = tuple(
        record
        for record in records
        if record.status == DimensionStatus.NEEDS_REVIEW.value
    )
    uncertain = tuple(
        record for record in records if record.status == DimensionStatus.UNCERTAIN.value
    )
    return {
        "labels": len(records),
        "read_correctly": sum(record.correct for record in records),
        "reviewable": len(agreed),
        "reviewable_correct": sum(record.correct for record in agreed),
        "uncertain": len(uncertain),
        "uncertain_correct": sum(record.correct for record in uncertain),
        "seconds_per_image": round(sum(seconds) / max(1, len(seconds)), 2),
    }


def main() -> int:
    """Run the benchmark and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--cross-check", metavar="MODEL", default=None)
    _ = parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parsed = parser.parse_args()
    model = cast("str | None", parsed.cross_check)
    output = cast("Path", parsed.output)
    reader = None if model is None else OllamaTextReader(model=model)
    document = cast(
        "dict[str, list[dict[str, object]]]",
        json.loads(_GROUND_TRUTH.read_text(encoding="utf-8")),
    )
    records: list[LabelResult] = []
    seconds: list[float] = []
    for sample in document["samples"]:
        sample_records, elapsed, reader_available = _sample_results(
            Path(cast("str", sample["image"])),
            cast("list[dict[str, object]]", sample["labels"]),
            reader,
        )
        records.extend(sample_records)
        seconds.append(elapsed)
        if reader_available is False:
            message = f"cross-check model {model!r} was unreachable"
            raise SystemExit(message)
    summary = _summary(tuple(records), seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        json.dumps(
            {
                "cross_check_model": model,
                "summary": summary,
                "misreads": [
                    {
                        "image": record.image,
                        "expected": record.expected,
                        "read": record.read,
                        "alternate": record.alternate,
                    }
                    for record in records
                    if not record.correct
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
