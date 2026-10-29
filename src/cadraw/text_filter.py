"""Removal of stroke candidates that are proved to be handwritten lettering.

Dimension labels, quantities, and prices are drawn with the same pen as the
part, so Hough and contour detection turn every `100` into a few short lines
and a closed blob. A text detector can say where lettering is, but a box alone
is not proof: a label is written right against the edge it measures, and a
drawn notch can look like a glyph. Two further tests make the removal safe.

1. Only a box whose reading holds at least two digits counts as lettering. A
   hole reads as a lone `0` or `O` and a notch as box-drawing characters, so
   neither can prove a region.
2. A stroke is removed only when every piece of ink it sits on lies inside
   that region. A label touching the outline shares its ink component with the
   outline, which then reaches outside the box and keeps the stroke.

A circle is judged differently, because a drilled hole is the one closed
shape that sits inside a label legitimately, as in `o←Ø14`, where it is the
thing labelled and OCR reads it as the label's first glyph. A small circle
inside a proved label is a digit's loop unless it leads the reading order.
"""

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import (
    ArcCandidate,
    CircleCandidate,
    EntityId,
    LineCandidate,
    Point,
    PolylineCandidate,
    ShapeCandidate,
    TextCandidate,
)

type GrayImage = NDArray[np.uint8]
type Quad = tuple[Point, Point, Point, Point]
type Region = NDArray[np.float32]

_SAMPLE_SPACING = 3.0
# Closed loops a handwritten glyph encloses. A letter `O` is left out: a
# drilled hole drawn after a label is read as one.
_GLYPH_LOOPS = {"0": 1, "6": 1, "8": 2, "9": 1, "Ø": 1, "ø": 1, "φ": 1, "Φ": 1}
_ARC_SAMPLE_DEGREES = 10.0


@dataclass(frozen=True, slots=True)
class TextStrokeConfig:
    """Thresholds for treating an OCR box as proved lettering."""

    enabled: bool = True
    minimum_confidence: float = 0.6
    minimum_digits: int = 2
    region_padding_ratio: float = 0.25
    minimum_region_padding: float = 4.0
    ink_search_band: int = 2
    leading_glyph_fraction: float = 0.3


@dataclass(frozen=True, slots=True)
class TextStrokeRemoval:
    """Candidates that survived, plus what was removed and why."""

    lines: tuple[LineCandidate, ...]
    shapes: tuple[ShapeCandidate, ...]
    removed_ids: tuple[EntityId, ...] = ()
    region_count: int = 0


def _digit_count(text: str) -> int:
    """Count decimal digits after compatibility normalization."""
    return sum(
        character.isdecimal() for character in unicodedata.normalize("NFKC", text)
    )


def _candidate_quad(candidate: TextCandidate) -> Quad:
    """Return the detector quad, or the axis-aligned box when there is none."""
    if candidate.quad is not None:
        return candidate.quad
    x, y, width, height = candidate.bounds
    return (
        Point(x, y),
        Point(x + width, y),
        Point(x + width, y + height),
        Point(x, y + height),
    )


def _expanded_region(quad: Quad, config: TextStrokeConfig) -> Region:
    """Grow a quad's rotated rectangle outward by a margin on every side."""
    points = np.asarray([(point.x, point.y) for point in quad], dtype=np.float32)
    (center_x, center_y), (width, height), angle = cv2.minAreaRect(points)
    margin = max(
        config.minimum_region_padding,
        config.region_padding_ratio * min(width, height),
    )
    corners = cv2.boxPoints(
        ((center_x, center_y), (width + 2.0 * margin, height + 2.0 * margin), angle),
    )
    return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)


def proved_text_regions(
    candidates: tuple[TextCandidate, ...],
    config: TextStrokeConfig,
) -> tuple[Region, ...]:
    """Return padded regions of every box that proves it holds lettering."""
    return tuple(
        _expanded_region(_candidate_quad(candidate), config)
        for candidate in candidates
        if candidate.confidence >= config.minimum_confidence
        and _digit_count(candidate.text) >= config.minimum_digits
    )


def _reading_axis(quad: Quad) -> tuple[Point, Point]:
    """Return the midpoints of a text quad's leading and trailing edges."""
    top_left, top_right, bottom_right, bottom_left = quad
    return (
        Point((top_left.x + bottom_left.x) / 2.0, (top_left.y + bottom_left.y) / 2.0),
        Point(
            (top_right.x + bottom_right.x) / 2.0,
            (top_right.y + bottom_right.y) / 2.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class _ReadLabel:
    """A proved label region, its reading axis, and the text read there."""

    region: Region
    leading: Point
    trailing: Point
    text: str


def _read_labels(
    candidates: tuple[TextCandidate, ...],
    config: TextStrokeConfig,
) -> tuple[_ReadLabel, ...]:
    """Pair each proved region with its reading axis and text."""
    return tuple(
        _ReadLabel(
            _expanded_region(quad, config),
            *_reading_axis(quad),
            text=candidate.text,
        )
        for candidate in candidates
        if candidate.confidence >= config.minimum_confidence
        and _digit_count(candidate.text) >= config.minimum_digits
        for quad in (_candidate_quad(candidate),)
    )


def _position_along(circle: CircleCandidate, label: _ReadLabel) -> float | None:
    """Return how far along a label's reading axis a circle inside it sits."""
    center = circle.center
    corners = (
        Point(center.x - circle.radius, center.y - circle.radius),
        Point(center.x + circle.radius, center.y - circle.radius),
        Point(center.x + circle.radius, center.y + circle.radius),
        Point(center.x - circle.radius, center.y + circle.radius),
    )
    if not all(_inside(label.region, corner) for corner in corners):
        return None
    axis_x = label.trailing.x - label.leading.x
    axis_y = label.trailing.y - label.leading.y
    length_squared = axis_x * axis_x + axis_y * axis_y
    if length_squared <= 0.0:
        return None
    return (
        (center.x - label.leading.x) * axis_x + (center.y - label.leading.y) * axis_y
    ) / length_squared


def _glyph_loop_ids(
    shapes: tuple[ShapeCandidate, ...],
    labels: tuple[_ReadLabel, ...],
    config: TextStrokeConfig,
) -> set[EntityId]:
    """Return the small circles that are the loops of a label's own digits.

    OCR gives no position per glyph, so each glyph of the reading is given an
    equal share of the label along its reading axis. A circle goes only when
    it sits in the share of a looped digit, and no more of them than that
    digit has loops: in `160` the circles in the `6` and `0` shares go. The
    first glyph and the leading part of the label are never taken, because a
    drilled hole is usually labelled by an arrow and a size written right
    after it, so OCR reads hole and label as one string such as `0149`. A
    hole read as the letter `O`, as in `100O`, sits in a share no digit
    explains and stays.
    """
    circles = tuple(shape for shape in shapes if isinstance(shape, CircleCandidate))
    removed: set[EntityId] = set()
    for label in labels:
        count = len(label.text)
        placed = tuple(
            (position, circle.entity_id)
            for circle in circles
            if (position := _position_along(circle, label)) is not None
            and position > config.leading_glyph_fraction
        )
        for index, character in enumerate(label.text):
            loops = _GLYPH_LOOPS.get(character, 0)
            if index == 0 or not loops:
                continue
            low, high = index / count, (index + 1) / count
            in_share = sorted(
                (abs(position - (low + high) / 2.0), entity_id)
                for position, entity_id in placed
                if low <= position <= high and entity_id not in removed
            )
            removed.update(entity_id for _, entity_id in in_share[:loops])
    return removed


def _inside(region: Region, point: Point) -> bool:
    """Return whether a point lies in or on a region polygon."""
    return cv2.pointPolygonTest(region, (point.x, point.y), measureDist=False) >= 0.0


def _segment_samples(start: Point, end: Point) -> tuple[Point, ...]:
    """Return evenly spaced points along a segment, both ends included."""
    count = max(2, math.ceil(start.distance_to(end) / _SAMPLE_SPACING) + 1)
    return tuple(
        Point(
            start.x + (end.x - start.x) * index / (count - 1),
            start.y + (end.y - start.y) * index / (count - 1),
        )
        for index in range(count)
    )


def _outline_samples(candidate: LineCandidate | ShapeCandidate) -> tuple[Point, ...]:
    """Return points that trace the ink a candidate claims to follow."""
    match candidate:
        case LineCandidate(start=start, end=end):
            return _segment_samples(start, end)
        case PolylineCandidate(points=points, closed=closed):
            ends = (*points[1:], points[0]) if closed else points[1:]
            return tuple(
                sample
                for start, end in zip(points, ends, strict=False)
                for sample in _segment_samples(start, end)
            )
        case ArcCandidate(
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
        ):
            sweep = (end_angle - start_angle) % 360.0 or 360.0
            count = max(2, math.ceil(sweep / _ARC_SAMPLE_DEGREES) + 1)
            # Detection measures arc angles with `atan2` in the y-down image
            # itself, so they map straight back to pixels without a mirror.
            return tuple(
                Point(
                    center.x + radius * math.cos(math.radians(angle)),
                    center.y + radius * math.sin(math.radians(angle)),
                )
                for angle in (
                    start_angle + sweep * index / (count - 1) for index in range(count)
                )
            )
        case CircleCandidate():
            return ()


class _InkComponents:
    """Connected ink components and a cached inside-region test per label."""

    def __init__(self, mask: GrayImage, band: int) -> None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(mask > 0, dtype=np.uint8),
            connectivity=8,
        )
        self._count: int = int(count)
        self._labels: NDArray[np.int32] = np.asarray(labels, dtype=np.int32)
        self._stats: NDArray[np.int32] = np.asarray(stats, dtype=np.int32)
        self._band: int = band
        self._inside_cache: dict[tuple[int, int], bool] = {}

    def labels_under(self, samples: Iterable[Point]) -> frozenset[int]:
        """Return the non-background labels found near the sampled points."""
        height, width = cast("tuple[int, int]", self._labels.shape[:2])
        found: set[int] = set()
        band = self._band
        for sample in samples:
            x = round(sample.x)
            y = round(sample.y)
            window = self._labels[
                max(0, y - band) : min(height, y + band + 1),
                max(0, x - band) : min(width, x + band + 1),
            ]
            labels = cast("list[int]", np.unique(window).tolist())
            found.update(label for label in labels if label > 0)
        return frozenset(found)

    def lies_inside(self, label: int, region_index: int, region: Region) -> bool:
        """Return whether every pixel of one ink component is in the region."""
        key = (label, region_index)
        cached = self._inside_cache.get(key)
        if cached is not None:
            return cached
        left, top, width, height = cast(
            "list[int]",
            self._stats[label, :4].tolist(),
        )
        region_left, region_top, region_width, region_height = cv2.boundingRect(
            region,
        )
        inside = (
            left >= region_left
            and top >= region_top
            and left + width <= region_left + region_width
            and top + height <= region_top + region_height
        )
        if inside:
            sub_labels = self._labels[top : top + height, left : left + width]
            window: NDArray[np.bool_] = np.equal(sub_labels, label)
            rows, columns = np.nonzero(window)
            pixels = np.column_stack((columns + left, rows + top)).astype(np.float32)
            hull = cv2.convexHull(pixels.reshape(-1, 1, 2))
            inside = all(
                _inside(region, Point(float(x), float(y)))
                for x, y in np.asarray(hull, dtype=np.float32).reshape(-1, 2).tolist()
            )
        self._inside_cache[key] = inside
        return inside


def _is_lettering(
    samples: tuple[Point, ...],
    regions: tuple[Region, ...],
    ink: _InkComponents,
) -> bool:
    """Return whether a stroke and all ink under it lie in one text region."""
    if not samples:
        return False
    containing = tuple(
        (index, region)
        for index, region in enumerate(regions)
        if all(_inside(region, sample) for sample in samples)
    )
    if not containing:
        return False
    labels = ink.labels_under(samples)
    return bool(labels) and any(
        all(ink.lies_inside(label, index, region) for label in labels)
        for index, region in containing
    )


def without_text_strokes(
    lines: tuple[LineCandidate, ...],
    shapes: tuple[ShapeCandidate, ...],
    text_candidates: tuple[TextCandidate, ...],
    ink_mask: GrayImage,
    config: TextStrokeConfig,
) -> TextStrokeRemoval:
    """Drop line and shape candidates that are proved to be lettering."""
    regions = proved_text_regions(text_candidates, config) if config.enabled else ()
    if not regions:
        return TextStrokeRemoval(lines=lines, shapes=shapes)
    ink = _InkComponents(ink_mask, config.ink_search_band)
    removed: list[EntityId] = []
    kept_lines: list[LineCandidate] = []
    for line in lines:
        if _is_lettering(_outline_samples(line), regions, ink):
            removed.append(line.entity_id)
        else:
            kept_lines.append(line)
    loops = _glyph_loop_ids(shapes, _read_labels(text_candidates, config), config)
    kept_shapes: list[ShapeCandidate] = []
    for shape in shapes:
        lettering = (
            shape.entity_id in loops
            if isinstance(shape, CircleCandidate)
            else _is_lettering(_outline_samples(shape), regions, ink)
        )
        if lettering:
            removed.append(shape.entity_id)
        else:
            kept_shapes.append(shape)
    return TextStrokeRemoval(
        lines=tuple(kept_lines),
        shapes=tuple(kept_shapes),
        removed_ids=tuple(removed),
        region_count=len(regions),
    )
