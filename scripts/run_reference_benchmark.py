#!/usr/bin/env -S uv run python
"""Score extracted geometry against the traced reference ground truth.

`docs/experiments/sample-geometry-ground-truth.json` traces each scanned
sample's part outlines, holes, slots, and cutouts in scan pixels. This runner
analyzes every sample with the shipped workbench configuration and reports:

- boundary precision, recall, and F1: the share of extracted geometry lying
  within a tolerance of a traced boundary, and the share of traced boundary
  that some extracted geometry covers;
- outlines found: traced outlines matched by one extracted closed shape whose
  filled area overlaps it by at least `_OUTLINE_IOU`, or for an open profile,
  covered along `_OPEN_COVERAGE` of its length;
- holes, slots, and cutouts found, matched one to one;
- false closed shapes: extracted closed shapes that match nothing traced and
  do not sit in an ignore region.

The tolerance scales with the image so the same drawn wobble is judged the
same at any scan resolution.
"""

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Final, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.detection import load_grayscale_image
from cadraw.models import (
    ArcEntity,
    CircleEntity,
    DrawingAnalysis,
    GeometryEntity,
    LineEntity,
    PolylineEntity,
)
from cadraw.pipeline import PipelineConfig, analyze_image

type Mask = NDArray[np.uint8]
type Polygon = NDArray[np.int32]

_GROUND_TRUTH: Final = Path("docs/experiments/sample-geometry-ground-truth.json")
_DEFAULT_OUTPUT: Final = Path("artifacts") / "reference_benchmark.json"
_TOLERANCE_RATIO: Final = 0.006
_MIN_TOLERANCE: Final = 6.0
_OUTLINE_IOU: Final = 0.80
_FEATURE_IOU: Final = 0.50
_OPEN_COVERAGE: Final = 0.90
_HOLE_RADIUS_RATIO: Final = (0.6, 1.6)
_ARC_STEP_DEGREES: Final = 4.0


@dataclass(frozen=True, slots=True)
class SampleScore:
    """Every figure reported for one sample."""

    image: str
    precision: float
    recall: float
    outlines_found: int
    outlines: int
    holes_found: int
    holes: int
    features_found: int
    features: int
    false_closed_shapes: int
    entities: int
    seconds: float

    @property
    def f1(self) -> float:
        """Return the harmonic mean of boundary precision and recall."""
        total = self.precision + self.recall
        return 0.0 if total == 0.0 else 2.0 * self.precision * self.recall / total


def _polygon(points: list[list[float]]) -> Polygon:
    """Convert JSON points to an OpenCV polygon."""
    return np.asarray(
        [[round(x), round(y)] for x, y in points],
        dtype=np.int32,
    ).reshape(-1, 1, 2)


def _arc_points(entity: ArcEntity) -> Polygon:
    """Sample an arc in image coordinates, where detection measured it."""
    sweep = (entity.end_angle_degrees - entity.start_angle_degrees) % 360.0 or 360.0
    count = max(2, math.ceil(sweep / _ARC_STEP_DEGREES) + 1)
    points = [
        [
            round(
                entity.center.x
                + entity.radius
                * math.cos(
                    math.radians(entity.start_angle_degrees + sweep * i / (count - 1)),
                ),
            ),
            round(
                entity.center.y
                + entity.radius
                * math.sin(
                    math.radians(entity.start_angle_degrees + sweep * i / (count - 1)),
                ),
            ),
        ]
        for i in range(count)
    ]
    return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)


def _draw_entity(mask: Mask, entity: GeometryEntity) -> None:
    """Rasterize one analysis entity as a one-pixel boundary."""
    match entity:
        case LineEntity(start=start, end=end):
            _ = cv2.line(
                mask,
                (round(start.x), round(start.y)),
                (round(end.x), round(end.y)),
                255,
                1,
            )
        case PolylineEntity(points=points, closed=closed):
            polygon = np.asarray(
                [[round(point.x), round(point.y)] for point in points],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            _ = cv2.polylines(mask, [polygon], isClosed=closed, color=255, thickness=1)
        case CircleEntity(center=center, radius=radius):
            _ = cv2.circle(
                mask,
                (round(center.x), round(center.y)),
                max(1, round(radius)),
                255,
                1,
            )
        case ArcEntity():
            _ = cv2.polylines(
                mask,
                [_arc_points(entity)],
                isClosed=False,
                color=255,
                thickness=1,
            )


def _filled(entity: GeometryEntity, shape: tuple[int, int]) -> Mask | None:
    """Return the filled area of a closed extracted shape, if it is one."""
    mask = np.zeros(shape, dtype=np.uint8)
    match entity:
        case PolylineEntity(points=points, closed=True) if len(points) >= 3:  # noqa: PLR2004
            polygon = np.asarray(
                [[round(point.x), round(point.y)] for point in points],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            _ = cv2.fillPoly(mask, [polygon], 255)
        case CircleEntity(center=center, radius=radius):
            _ = cv2.circle(
                mask,
                (round(center.x), round(center.y)),
                max(1, round(radius)),
                255,
                -1,
            )
        case _:
            return None
    return mask


def _iou(first: Mask, second: Mask) -> float:
    """Return the intersection over union of two filled masks."""
    inter = int(np.count_nonzero(np.logical_and(first > 0, second > 0)))
    union = int(np.count_nonzero(np.logical_or(first > 0, second > 0)))
    return 0.0 if union == 0 else inter / union


def _distance_to(mask: Mask) -> NDArray[np.float32]:
    """Return each pixel's distance to the nearest set pixel of a mask."""
    return np.asarray(
        cv2.distanceTransform(
            np.where(mask > 0, 0, 255).astype(np.uint8),
            cv2.DIST_L2,
            3,
        ),
        dtype=np.float32,
    )


@dataclass(frozen=True, slots=True)
class _Truth:
    """Rasterized ground truth for one sample."""

    boundary: Mask
    ignore: Mask
    outlines: tuple[tuple[Mask, bool], ...]
    holes: tuple[tuple[float, float, float], ...]
    features: tuple[Mask, ...]


def _truth(elements: list[dict[str, object]], shape: tuple[int, int]) -> _Truth:
    """Rasterize the traced elements of one sample."""
    boundary = np.zeros(shape, dtype=np.uint8)
    ignore = np.zeros(shape, dtype=np.uint8)
    outlines: list[tuple[Mask, bool]] = []
    holes: list[tuple[float, float, float]] = []
    features: list[Mask] = []
    for element in elements:
        kind = cast("str", element["kind"])
        if kind == "hole":
            cx, cy = cast("list[float]", element["center"])
            radius = cast("float", element["radius"])
            _ = cv2.circle(boundary, (round(cx), round(cy)), round(radius), 255, 1)
            holes.append((cx, cy, radius))
            continue
        polygon = _polygon(cast("list[list[float]]", element["points"]))
        closed = cast("bool", element.get("closed", True))
        if kind == "ignore":
            _ = cv2.fillPoly(ignore, [polygon], 255)
            continue
        _ = cv2.polylines(boundary, [polygon], isClosed=closed, color=255, thickness=1)
        region = np.zeros(shape, dtype=np.uint8)
        if closed:
            _ = cv2.fillPoly(region, [polygon], 255)
        else:
            _ = cv2.polylines(region, [polygon], isClosed=False, color=255, thickness=1)
        if kind == "outline":
            outlines.append((region, closed))
        else:
            features.append(region)
    return _Truth(boundary, ignore, tuple(outlines), tuple(holes), tuple(features))


def _hole_matches(
    holes: tuple[tuple[float, float, float], ...],
    circles: list[CircleEntity],
) -> set[int]:
    """Match traced holes to extracted circles one to one by centre and size.

    Returns the indices of the circles that were matched.
    """
    used: set[int] = set()
    for cx, cy, radius in holes:
        best: tuple[float, int] | None = None
        for index, circle in enumerate(circles):
            if index in used:
                continue
            distance = math.hypot(circle.center.x - cx, circle.center.y - cy)
            ratio = circle.radius / max(1.0, radius)
            if (
                distance <= max(4.0, 0.5 * radius)
                and _HOLE_RADIUS_RATIO[0] <= ratio <= _HOLE_RADIUS_RATIO[1]
                and (best is None or distance < best[0])
            ):
                best = (distance, index)
        if best is not None:
            used.add(best[1])
    return used


def _best_unmatched(
    region: Mask,
    closed_shapes: list[tuple[GeometryEntity, Mask]],
    matched: set[int],
) -> tuple[float, int]:
    """Return the best-overlapping closed shape not yet claimed."""
    return max(
        (
            (_iou(region, filled), index)
            for index, (_, filled) in enumerate(closed_shapes)
            if index not in matched
        ),
        default=(0.0, -1),
    )


def _boundary_scores(
    extracted: Mask,
    truth: _Truth,
    tolerance: float,
) -> tuple[float, float, NDArray[np.float32]]:
    """Return boundary precision, recall, and the distance to extraction."""
    to_truth = _distance_to(truth.boundary)
    to_extracted = _distance_to(extracted)
    near_truth: NDArray[np.bool_] = np.less_equal(to_truth, tolerance)
    outside_ignore: NDArray[np.bool_] = np.equal(truth.ignore, 0)
    counted: NDArray[np.bool_] = np.logical_and(
        np.greater(extracted, 0),
        outside_ignore,
    )
    true_positive = int(np.count_nonzero(np.logical_and(counted, near_truth)))
    precision = true_positive / max(1, int(np.count_nonzero(counted)))
    on_truth = np.logical_and(truth.boundary > 0, outside_ignore)
    covered = np.logical_and(on_truth, np.less_equal(to_extracted, tolerance))
    recall = int(np.count_nonzero(covered)) / max(1, int(np.count_nonzero(on_truth)))
    return precision, recall, to_extracted


def _false_closed_count(
    closed_shapes: list[tuple[GeometryEntity, Mask]],
    claimed: set[int],
    ignore: Mask,
) -> int:
    """Count closed shapes that match nothing and sit outside ignore regions."""
    height, width = cast("tuple[int, int]", ignore.shape[:2])
    false_closed = 0
    for index, (_, filled) in enumerate(closed_shapes):
        if index in claimed:
            continue
        moments = cv2.moments(filled, binaryImage=True)
        if moments["m00"] == 0:
            continue
        cx = round(moments["m10"] / moments["m00"])
        cy = round(moments["m01"] / moments["m00"])
        if 0 <= cy < height and 0 <= cx < width and ignore[cy, cx] > 0:
            continue
        false_closed += 1
    return false_closed


def score_sample(
    analysis: DrawingAnalysis,
    elements: list[dict[str, object]],
    shape: tuple[int, int],
) -> tuple[float, float, int, int, int, int]:
    """Score one analysis: boundary P/R and the structural match counts."""
    height, width = shape
    tolerance = max(_MIN_TOLERANCE, _TOLERANCE_RATIO * math.hypot(width, height))
    truth = _truth(elements, shape)
    extracted = np.zeros(shape, dtype=np.uint8)
    for entity in analysis.entities:
        _draw_entity(extracted, entity)
    precision, recall, to_extracted = _boundary_scores(extracted, truth, tolerance)
    closed_shapes = [
        (entity, filled)
        for entity in analysis.entities
        if (filled := _filled(entity, shape)) is not None
    ]
    matched: set[int] = set()
    outlines_found = 0
    for region, closed in truth.outlines:
        if not closed:
            covered = np.logical_and(region > 0, np.less_equal(to_extracted, tolerance))
            outlines_found += int(np.count_nonzero(covered)) >= (
                _OPEN_COVERAGE * int(np.count_nonzero(region))
            )
            continue
        overlap, index = _best_unmatched(region, closed_shapes, matched)
        if overlap >= _OUTLINE_IOU:
            matched.add(index)
            outlines_found += 1
    features_found = 0
    for region in truth.features:
        overlap, index = _best_unmatched(region, closed_shapes, matched)
        if overlap >= _FEATURE_IOU:
            matched.add(index)
            features_found += 1
    circles = [
        (index, entity)
        for index, (entity, _) in enumerate(closed_shapes)
        if index not in matched and isinstance(entity, CircleEntity)
    ]
    hole_circles = _hole_matches(truth.holes, [circle for _, circle in circles])
    holes_found = len(hole_circles)
    claimed = matched | {circles[position][0] for position in hole_circles}
    return (
        precision,
        recall,
        outlines_found,
        holes_found,
        features_found,
        _false_closed_count(closed_shapes, claimed, truth.ignore),
    )


def _workbench_config() -> PipelineConfig:
    """Return what the app ships, without the optional vision cross-check."""
    return replace(PipelineConfig.for_workbench(), ocr_cross_check_model=None)


def run(
    config: PipelineConfig,
    ground_truth: Path = _GROUND_TRUTH,
) -> tuple[SampleScore, ...]:
    """Analyze and score every traced sample."""
    document = cast(
        "dict[str, list[dict[str, object]]]",
        json.loads(ground_truth.read_text(encoding="utf-8")),
    )
    scores: list[SampleScore] = []
    for sample in document["samples"]:
        image_path = Path(cast("str", sample["image"]))
        elements = cast("list[dict[str, object]]", sample["elements"])
        shape = cast("tuple[int, int]", load_grayscale_image(image_path).shape[:2])
        started = perf_counter()
        analysis = analyze_image(image_path, config)
        elapsed = perf_counter() - started
        precision, recall, outlines, holes, features, false_closed = score_sample(
            analysis,
            elements,
            shape,
        )
        kinds = [cast("str", element["kind"]) for element in elements]
        scores.append(
            SampleScore(
                image=str(image_path),
                precision=precision,
                recall=recall,
                outlines_found=outlines,
                outlines=kinds.count("outline"),
                holes_found=holes,
                holes=kinds.count("hole"),
                features_found=features,
                features=kinds.count("slot") + kinds.count("cutout"),
                false_closed_shapes=false_closed,
                entities=len(analysis.entities),
                seconds=elapsed,
            ),
        )
    return tuple(scores)


def _row(score: SampleScore) -> dict[str, object]:
    """Serialize one sample score."""
    return {
        "image": score.image,
        "precision": round(score.precision, 3),
        "recall": round(score.recall, 3),
        "f1": round(score.f1, 3),
        "outlines": f"{score.outlines_found}/{score.outlines}",
        "holes": f"{score.holes_found}/{score.holes}",
        "slots_cutouts": f"{score.features_found}/{score.features}",
        "false_closed_shapes": score.false_closed_shapes,
        "entities": score.entities,
        "seconds": round(score.seconds, 2),
    }


def main() -> int:
    """Run the benchmark, print a table, and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--ground-truth", type=Path, default=_GROUND_TRUTH)
    _ = parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parsed = parser.parse_args()
    output = cast("Path", parsed.output)
    scores = run(_workbench_config(), cast("Path", parsed.ground_truth))
    rows = [_row(score) for score in scores]
    outlines = (sum(s.outlines_found for s in scores), sum(s.outlines for s in scores))
    holes = (sum(s.holes_found for s in scores), sum(s.holes for s in scores))
    features = (sum(s.features_found for s in scores), sum(s.features for s in scores))
    total = {
        "f1_mean": round(sum(score.f1 for score in scores) / len(scores), 3),
        "outlines": f"{outlines[0]}/{outlines[1]}",
        "holes": f"{holes[0]}/{holes[1]}",
        "slots_cutouts": f"{features[0]}/{features[1]}",
        "false_closed_shapes": sum(s.false_closed_shapes for s in scores),
    }
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps({"total": total}, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(
        json.dumps({"samples": rows, "total": total}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
