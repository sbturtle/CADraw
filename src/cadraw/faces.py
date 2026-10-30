"""Closed shapes read from the paper regions a drawing's strokes enclose.

A part outline drawn with a pen encloses a blank region of paper, and so does
every drilled hole and slot drawn inside it. The boundary of that region is
the outline itself, already closed, already free of the extension lines and
lettering drawn around the outside, and already separated from a neighbouring
face such as an angle bracket's flange. Reading shapes from these faces,
instead of from the outer contour of the ink, keeps each drawn part distinct.

Two things decide whether a face exists at all:

- The ink mask must keep thin strokes whole. A pen line two pixels wide is
  cut into dashes by any morphological opening, and faint pencil is lost to a
  single global threshold. Hysteresis keeps faint ink only where it continues
  a confidently dark stroke, so paper texture and show-through stay out.
- Small gaps must close. A long stroke that stops short of a corner, of the
  stroke facing it, or of the line it was meant to meet is run on along its
  own direction for a short reach, found from the free ends of the strokes'
  centre lines. Faces are then taken with a fine closing that keeps small
  drilled holes open, and with a closing proportional to the sheet that
  seals what remains of a pen lift in a part's outline.

These bridges and closings only shape the faces. No bridge becomes a line,
and a closed shape is still the boundary of paper the drawing left blank.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import (
    CircleCandidate,
    EntityId,
    GeometryRole,
    Point,
    PolylineCandidate,
    ShapeCandidate,
)
from cadraw.preprocessing import illumination_flattened

type GrayImage = NDArray[np.uint8]
type Contour = NDArray[np.int32]
type Bounds = tuple[float, float, float, float]

_INK = 255
_PENDING_ID = EntityId("shape_raw_pending")
_MINIMUM_POLYGON_POINTS = 3
_MINIMUM_FIT_POINTS = 6
_MINIMUM_STROKE_WIDTH = 1.0
_MAXIMUM_STROKE_WIDTH = 8.0
_COARSE_APPROXIMATION = 0.04
_QUADRILATERAL_VERTICES = 4
_MINIMUM_CORNER_SINE = 0.34
_CORNER_BACKTRACK_RATIO = 0.25
_FACING_COSINE = 0.87
_SQUARE_FACING_COSINE = 0.95
_LABEL_GAP_START = 0.2
_RING_RAYS = 24
_MINIMUM_OUTLINE_SAMPLES = 12
_INTERSTITIAL_SHARE = 0.75
_SAME_FEATURE_AREA_RATIO = 2.0
_RING_SEARCH_MARGIN = 6.0
_RING_SEARCH_START = 0.8
_MAD_TO_SIGMA = 1.4826
_PERIODICITY_DETREND = 61
_PERIODICITY_MINIMUM_LAG = 15
_PERIODICITY_MAXIMUM_LAG = 120
_RULED_PERIODICITY = 0.45
_THINNING_MINIMUM_NEIGHBOURS = 2
_THINNING_MAXIMUM_NEIGHBOURS = 6
_SPUR_PRUNING_ITERATIONS = 3
_MINIMUM_REDUNDANT_NEIGHBOURS = 2


@dataclass(frozen=True, slots=True)
class FaceConfig:
    """Settings for reading closed shapes from enclosed paper regions."""

    enabled: bool = False
    weak_ink_fraction: float = 0.4
    clean_paper_sigma: float = 2.0
    clean_paper_sigmas: float = 6.0
    clean_paper_percentile: float = 3.0
    clean_paper_margin: float = 4.0
    feature_close_size: int = 3
    part_close_ratio: float = 0.006
    minimum_face_area: float = 15.0
    approximation_ratio: float = 0.004
    minimum_approximation: float = 2.0
    corner_merge_degrees: float = 12.0
    circle_circularity: float = 0.78
    circle_aspect_ratio: float = 0.72
    circle_residual_ratio: float = 0.10
    circle_maximum_rectangularity: float = 0.88
    small_feature_ratio: float = 0.03
    small_circle_circularity: float = 0.6
    small_circle_residual_ratio: float = 0.18
    tiny_feature_ratio: float = 0.015
    tiny_circle_aspect_ratio: float = 0.5
    tiny_maximum_rectangularity: float = 0.93
    tiny_circle_residual_ratio: float = 0.25
    maximum_vertices: int = 48
    sliver_aspect_ratio: float = 5.0
    sliver_width: float = 10.0
    maximum_extent_ratio: float = 0.92
    border_margin_ratio: float = 0.015
    isolated_mark_ratio: float = 0.04
    bridge_thickness: int = 2
    bridge_reach_ratio: float = 0.025
    facing_reach_ratio: float = 0.10
    bridge_branch_ratio: float = 0.02
    bridge_direction_span: int = 10


@dataclass(frozen=True, slots=True)
class Face:
    """One enclosed region, grown out to the centre of its bounding strokes."""

    contour: Contour
    area: float


def _odd(value: int) -> int:
    """Return an odd kernel size of at least three."""
    bounded = max(3, value)
    return bounded if bounded % 2 else bounded + 1


def _profile_periodicity(profile: NDArray[np.float64]) -> float:
    """Return the strongest normalized autocorrelation at a grid-like lag."""
    window = _PERIODICITY_DETREND
    if len(profile) <= 2 * window + _PERIODICITY_MAXIMUM_LAG:
        return 0.0
    trend: NDArray[np.float64] = np.convolve(
        profile,
        np.ones(window) / window,
        mode="same",
    )
    detrended: NDArray[np.float64] = np.subtract(profile, trend)[window:-window]
    centered: NDArray[np.float64] = np.subtract(detrended, float(np.mean(detrended)))
    squares: NDArray[np.float64] = np.multiply(centered, centered)
    energy = float(np.mean(squares)) * len(squares)
    if energy <= 0.0:
        return 0.0
    correlations: list[float] = []
    for lag in range(_PERIODICITY_MINIMUM_LAG, _PERIODICITY_MAXIMUM_LAG):
        products: NDArray[np.float64] = np.multiply(centered[:-lag], centered[lag:])
        correlations.append(float(np.mean(products)) * len(products) / energy)
    return max(correlations)


def ruled_paper(flattened: GrayImage) -> bool:
    """Return whether the page carries a periodic ruling on either axis.

    Graph paper darkens every column and row at its period, so the mean
    intensity profiles of the page centre repeat; a drawing's few lines do
    not. This needs no ruling to be proved rail by rail, which a scanner's
    faint, resampled grid often defeats.
    """
    height, width = cast("tuple[int, int]", flattened.shape[:2])
    core = np.asarray(
        flattened[height // 6 : 5 * height // 6, width // 6 : 5 * width // 6],
        dtype=np.float64,
    )
    return (
        max(
            _profile_periodicity(np.asarray(core.mean(axis=0), dtype=np.float64)),
            _profile_periodicity(np.asarray(core.mean(axis=1), dtype=np.float64)),
        )
        >= _RULED_PERIODICITY
    )


def _weak_cut(
    flattened: GrayImage,
    strong_cut: float,
    config: FaceConfig,
    *,
    sensitive: bool,
) -> float:
    """Return the lighter cut that admits the faint run of a drawn stroke.

    On ruled paper the cut stays halfway towards paper, because the ruling is
    as dark as faint pencil and hysteresis would grow into every rail a line
    crosses. On clean, evenly lit paper the cut follows the paper itself: a
    few noise widths below its typical level, which keeps a faint arc whole.
    """
    conservative = strong_cut + (_INK - strong_cut) * config.weak_ink_fraction
    paper = np.asarray(flattened[flattened > strong_cut], dtype=np.float64)
    if not sensitive or paper.size == 0:
        return conservative
    median = float(np.median(paper))
    deviations: NDArray[np.float64] = np.abs(np.subtract(paper, median))
    sigma = float(np.median(deviations)) * _MAD_TO_SIGMA
    if sigma > config.clean_paper_sigma or ruled_paper(flattened):
        return conservative
    paper_cut = min(
        median - config.clean_paper_sigmas * sigma,
        float(np.percentile(paper, config.clean_paper_percentile))
        - config.clean_paper_margin,
    )
    return max(conservative, paper_cut)


def drawing_ink(
    image: GrayImage,
    config: FaceConfig,
    *,
    sensitive: bool = False,
) -> GrayImage:
    """Binarize the drawing's strokes without breaking thin or faint lines.

    Otsu's cut on the flattened page marks confident ink. A second, lighter
    cut admits the faint run of a pencil line, but only the parts of it
    connected to confident ink, so a stroke keeps its whole length while
    isolated paper grain and show-through stay paper. `sensitive` lowers that
    second cut further on clean paper; it keeps a long faint outline whole,
    but would also fill the few blank pixels inside a small drilled hole.
    """
    return flattened_ink(illumination_flattened(image), config, sensitive=sensitive)


def flattened_ink(
    flattened: GrayImage,
    config: FaceConfig,
    *,
    sensitive: bool = False,
) -> GrayImage:
    """Binarize strokes on an already illumination-flattened page."""
    strong_cut, _ = cv2.threshold(
        flattened,
        0,
        _INK,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    weak_cut = _weak_cut(flattened, strong_cut, config, sensitive=sensitive)
    strong: NDArray[np.bool_] = np.less_equal(flattened, strong_cut)
    weak = np.asarray(np.less_equal(flattened, weak_cut), dtype=np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    label_image = np.asarray(labels, dtype=np.int32)
    keep = np.zeros(int(count), dtype=np.bool_)
    keep[np.unique(label_image[strong])] = True
    keep[0] = False
    return np.asarray(keep[label_image], dtype=np.uint8) * _INK


def stroke_width(mask: GrayImage) -> float:
    """Estimate the drawn stroke width as twice the ink area per boundary."""
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(
        cv2.arcLength(cast("Contour", contour), closed=True) for contour in contours
    )
    area = float(np.count_nonzero(mask))
    width = 2.0 * area / max(1.0, perimeter)
    return max(_MINIMUM_STROKE_WIDTH, min(_MAXIMUM_STROKE_WIDTH, width))


def _draw(mask: GrayImage, start: Point, end: Point, thickness: int) -> None:
    """Rasterize one bridge segment into a mask."""
    _ = cv2.line(
        mask,
        (round(start.x), round(start.y)),
        (round(end.x), round(end.y)),
        _INK,
        thickness,
    )


def thinned(mask: GrayImage, maximum_iterations: int | None = None) -> GrayImage:
    """Return the one-pixel skeleton of a binary mask (Zhang and Suen).

    Each iteration peels one layer off every stroke, so a pen stroke is down
    to its centre line within a few iterations of its own width. A dark blob
    such as the backdrop around a photographed page would take hundreds;
    `maximum_iterations` leaves it thick, which yields no stroke ends anyway.
    """
    image = np.asarray(np.greater(mask, 0), dtype=np.uint8)
    changed = True
    iterations = 0
    while changed and (maximum_iterations is None or iterations < maximum_iterations):
        iterations += 1
        changed = False
        for table in _THINNING_TABLES:
            removable = np.logical_and(
                np.equal(image, 1),
                table[_neighbour_codes(image)],
            )
            if bool(np.any(removable)):
                image[removable] = 0
                changed = True
    return np.asarray(image * _INK, dtype=np.uint8)


def _neighbour_codes(image: GrayImage) -> NDArray[np.uint8]:
    """Encode each pixel's eight neighbours as one byte, north first, clockwise."""
    codes = cv2.filter2D(
        image,
        cv2.CV_16U,
        _NEIGHBOUR_WEIGHTS,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.asarray(codes, dtype=np.uint8)


def _thinning_table(step: int) -> NDArray[np.bool_]:
    """Return, per neighbour code, whether one Zhang-Suen sub-iteration removes."""
    table = np.zeros(256, dtype=np.bool_)
    for code in range(256):
        ring = tuple((code >> bit) & 1 for bit in range(8))
        north, _, east, _, south, _, west, _ = ring
        neighbours = code.bit_count()
        transitions = sum(
            1
            for index, current in enumerate(ring)
            if current == 0 and ring[(index + 1) % len(ring)] == 1
        )
        if step == 0:
            clear = north * east * south == 0 and east * south * west == 0
        else:
            clear = north * east * west == 0 and north * south * west == 0
        table[code] = (
            _THINNING_MINIMUM_NEIGHBOURS <= neighbours <= _THINNING_MAXIMUM_NEIGHBOURS
            and transitions == 1
            and clear
        )
    return table


# Weights of the neighbours in `_neighbour_codes`, as offsets from the centre:
# north is bit 0, then clockwise to north-west at bit 7.
_NEIGHBOUR_WEIGHTS = np.asarray(
    [[128.0, 1.0, 2.0], [64.0, 0.0, 4.0], [32.0, 16.0, 8.0]],
    dtype=np.float32,
)
_THINNING_TABLES = (_thinning_table(0), _thinning_table(1))
# Row and column offsets of the neighbour bits, north first, clockwise.
_RING_OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
_NEIGHBOUR_COUNT = np.asarray(
    [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)


_NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _skeleton_neighbours(
    skeleton: NDArray[np.bool_],
    y: int,
    x: int,
) -> list[tuple[int, int]]:
    """Return the skeleton pixels touching one pixel."""
    height, width = cast("tuple[int, int]", skeleton.shape[:2])
    return [
        (y + dy, x + dx)
        for dy, dx in _NEIGHBOUR_OFFSETS
        if 0 <= y + dy < height and 0 <= x + dx < width and skeleton[y + dy, x + dx]
    ]


@dataclass(frozen=True, slots=True)
class StrokeEnd:
    """A free end of a drawn stroke, with the direction the pen was moving."""

    point: Point
    direction: tuple[float, float]
    branch_length: int


def _branch_from(
    present: NDArray[np.bool_],
    start: tuple[int, int],
    limit: int,
) -> list[tuple[int, int]]:
    """Trace a skeleton branch from a free end until a junction or `limit`."""
    path = [start]
    previous: tuple[int, int] | None = None
    current = start
    while len(path) <= limit:
        following = [
            pixel
            for pixel in _skeleton_neighbours(present, *current)
            if pixel != previous and pixel not in path[-3:]
        ]
        if len(following) != 1:
            break
        previous, current = current, following[0]
        path.append(current)
    return path


def _free_ends(present: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Return the skeleton pixels with exactly one skeleton neighbour."""
    neighbours = cv2.filter2D(
        np.asarray(present, dtype=np.uint8),
        cv2.CV_16U,
        _NEIGHBOUR_COUNT,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.logical_and(present, np.equal(neighbours, 1))


def _ring_clusters(code: int) -> int:
    """Return how many 8-connected groups a neighbour code's pixels form."""
    members = [_RING_OFFSETS[bit] for bit in range(8) if (code >> bit) & 1]
    groups = 0
    unvisited = set(members)
    while unvisited:
        groups += 1
        frontier = [unvisited.pop()]
        while frontier:
            row, column = frontier.pop()
            touching = {
                other
                for other in unvisited
                if max(abs(other[0] - row), abs(other[1] - column)) <= 1
            }
            unvisited -= touching
            frontier.extend(touching)
    return groups


_REDUNDANT_TABLE = np.asarray(
    [
        code.bit_count() >= _MINIMUM_REDUNDANT_NEIGHBOURS and _ring_clusters(code) == 1
        for code in range(256)
    ],
    dtype=np.bool_,
)


def _minimal(present: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Drop skeleton pixels whose removal keeps every neighbour connected.

    Zhang-Suen leaves a staircase pixel wherever a line steps diagonally, and
    there the line has two touching neighbours of one pixel instead of one. A
    stroke's tip then shows two neighbours and a trace along it sees a false
    junction. Removing such pixels in four interleaved passes, so no two
    removed pixels ever touch, leaves a line exactly one pixel wide.
    """
    minimal = np.asarray(present, dtype=np.uint8)
    changed = True
    while changed:
        changed = False
        for row_phase, column_phase in ((0, 0), (0, 1), (1, 0), (1, 1)):
            removable = np.logical_and(
                np.equal(minimal, 1),
                _REDUNDANT_TABLE[_neighbour_codes(minimal)],
            )
            phase = np.zeros_like(removable)
            phase[row_phase::2, column_phase::2] = True
            removable = np.logical_and(removable, phase)
            if bool(np.any(removable)):
                minimal[removable] = 0
                changed = True
    return np.greater(minimal, 0)


def _pruned(present: NDArray[np.bool_], iterations: int) -> NDArray[np.bool_]:
    """Peel the free ends off a skeleton a few times to drop its short spurs.

    A wobbling pen edge thins to a centre line with one- and two-pixel spurs,
    and each spur is a junction that would stop a stroke's trace long before
    its real end. Peeling shortens every true end by the same few pixels,
    which a bridge drawn from it still covers.
    """
    pruned = present.copy()
    for _ in range(iterations):
        pruned[_free_ends(pruned)] = False
    return pruned


def _restored_tip(
    minimal: NDArray[np.bool_],
    pruned: NDArray[np.bool_],
    start: tuple[int, int],
) -> tuple[int, int]:
    """Walk back out along the pixels pruning peeled off a stroke's end."""
    previous: tuple[int, int] | None = None
    current = start
    for _ in range(_SPUR_PRUNING_ITERATIONS):
        options = [
            pixel
            for pixel in _skeleton_neighbours(minimal, *current)
            if pixel != previous and not pruned[pixel]
        ]
        if len(options) != 1:
            break
        previous, current = current, options[0]
    return current


def stroke_ends(
    skeleton: GrayImage,
    minimum_branch: int,
    direction_span: int,
) -> tuple[StrokeEnd, ...]:
    """Return the free ends of skeleton branches at least `minimum_branch` long.

    Each end is traced back along its branch until a junction, so a short
    spur, a tick mark, or a stroke of handwriting never qualifies; only the
    end of a long stroke, such as an outline side, is a candidate to bridge.
    """
    minimal = _minimal(np.greater(skeleton, 0))
    present = _pruned(minimal, _SPUR_PRUNING_ITERATIONS)
    ys, xs = np.nonzero(_free_ends(present))
    ends: list[StrokeEnd] = []
    rows: list[int] = ys.tolist()
    columns: list[int] = xs.tolist()
    for pruned_y, pruned_x in zip(rows, columns, strict=True):
        path = _branch_from(
            present,
            (pruned_y, pruned_x),
            max(minimum_branch, direction_span),
        )
        if len(path) < minimum_branch:
            continue
        y, x = _restored_tip(minimal, present, (pruned_y, pruned_x))
        back_y, back_x = path[min(len(path) - 1, direction_span)]
        delta_x = float(x - back_x)
        delta_y = float(y - back_y)
        length = math.hypot(delta_x, delta_y)
        if length == 0.0:
            continue
        ends.append(
            StrokeEnd(
                point=Point(float(x), float(y)),
                direction=(delta_x / length, delta_y / length),
                branch_length=len(path),
            ),
        )
    return tuple(ends)


def _rays_meet(first: StrokeEnd, second: StrokeEnd, reach: float) -> Point | None:
    """Return where two stroke ends' continuations meet within reach of both.

    Ends heading across each other meet at the crossing of their lines; ends
    heading along one line towards each other meet halfway between them.
    """
    first_x, first_y = first.direction
    second_x, second_y = second.direction
    cross = first_x * second_y - first_y * second_x
    delta_x = second.point.x - first.point.x
    delta_y = second.point.y - first.point.y
    if abs(cross) < _MINIMUM_CORNER_SINE:
        distance = math.hypot(delta_x, delta_y)
        if distance == 0.0 or distance > reach:
            return None
        facing = (delta_x * first_x + delta_y * first_y) / distance
        returning = -(delta_x * second_x + delta_y * second_y) / distance
        if min(facing, returning) < _FACING_COSINE:
            return None
        return Point(
            (first.point.x + second.point.x) / 2.0,
            (first.point.y + second.point.y) / 2.0,
        )
    along_first = (delta_x * second_y - delta_y * second_x) / cross
    along_second = (delta_x * first_y - delta_y * first_x) / cross
    backtrack = -_CORNER_BACKTRACK_RATIO * reach
    if not (backtrack <= along_first <= reach and backtrack <= along_second <= reach):
        return None
    return Point(
        first.point.x + first_x * along_first,
        first.point.y + first_y * along_first,
    )


def _ink_ahead(
    mask: GrayImage,
    end: StrokeEnd,
    reach: float,
    skip: float,
) -> Point | None:
    """Return the first ink a stroke end's continuation reaches, if any."""
    height, width = cast("tuple[int, int]", mask.shape[:2])
    direction_x, direction_y = end.direction
    distance = skip
    while distance <= reach:
        x = round(end.point.x + direction_x * distance)
        y = round(end.point.y + direction_y * distance)
        if not (0 <= x < width and 0 <= y < height):
            return None
        if mask[y, x] > 0:
            return Point(float(x), float(y))
        distance += 1.0
    return None


def _closest_partner(
    ends: tuple[StrokeEnd, ...],
    index: int,
    bridged: set[int],
    runs: tuple[float, ...],
) -> tuple[int, Point] | None:
    """Return the unbridged end whose continuation meets this one soonest.

    `runs` bounds how far each end may travel: its bridge reach, or less when
    its continuation hits ink first, since the pen would have stopped there.
    """
    first = ends[index]
    best: tuple[float, int, Point] | None = None
    for other_index in range(index + 1, len(ends)):
        if other_index in bridged:
            continue
        second = ends[other_index]
        meeting = _rays_meet(first, second, max(runs[index], runs[other_index]))
        if meeting is None:
            continue
        first_run = first.point.distance_to(meeting)
        second_run = second.point.distance_to(meeting)
        if first_run > runs[index] or second_run > runs[other_index]:
            continue
        span = max(first_run, second_run)
        if best is None or span < best[0]:
            best = (span, other_index, meeting)
    return None if best is None else (best[1], best[2])


def _faces_squarely(first: StrokeEnd, second: StrokeEnd, reach: float) -> bool:
    """Return whether two ends point straight at each other within `reach`."""
    delta_x = second.point.x - first.point.x
    delta_y = second.point.y - first.point.y
    distance = math.hypot(delta_x, delta_y)
    if distance == 0.0 or distance > reach:
        return False
    first_x, first_y = first.direction
    second_x, second_y = second.direction
    facing = (delta_x * first_x + delta_y * first_y) / distance
    returning = -(delta_x * second_x + delta_y * second_y) / distance
    return min(facing, returning) >= _SQUARE_FACING_COSINE


def _paper_between(mask: GrayImage, start: Point, end: Point, skip: float) -> bool:
    """Return whether the straight run between two points crosses no ink."""
    distance = start.distance_to(end)
    steps = max(1, math.ceil(distance))
    for step in range(steps + 1):
        along = distance * step / steps
        if along < skip or along > distance - skip:
            continue
        x = round(start.x + (end.x - start.x) * step / steps)
        y = round(start.y + (end.y - start.y) * step / steps)
        if mask[y, x] > 0:
            return False
    return True


@dataclass(frozen=True, slots=True)
class FacingReach:
    """How far squarely facing stroke ends may be joined, and where not.

    `labels` are the boxes of text read on the sheet. A dimension line is
    broken to make room for its value, which is written across the middle of
    the gap, so a gap with a label centred on it keeps its break. A label by
    either end of the gap, such as the size of a corner, does not.
    """

    reach: float = 0.0
    labels: tuple[Bounds, ...] = ()

    def labelled(self, start: Point, end: Point) -> bool:
        """Return whether a label is centred on the gap between two ends."""
        length = start.distance_to(end)
        if length == 0.0:
            return False
        direction_x = (end.x - start.x) / length
        direction_y = (end.y - start.y) / length
        for left, top, width, height in self.labels:
            offset_x = left + width / 2.0 - start.x
            offset_y = top + height / 2.0 - start.y
            along = (offset_x * direction_x + offset_y * direction_y) / length
            across = abs(offset_x * direction_y - offset_y * direction_x)
            if (
                _LABEL_GAP_START <= along <= 1.0 - _LABEL_GAP_START
                and across <= length / 2.0 + max(width, height) / 2.0
            ):
                return True
        return False


def _facing_bridges(
    mask: GrayImage,
    ends: tuple[StrokeEnd, ...],
    bridged: set[int],
    facing: FacingReach,
    skip: float,
) -> list[tuple[int, int]]:
    """Pair the ends that point straight at each other across blank paper.

    A faint stretch of one stroke drops out of the ink mask as a gap far
    longer than a pen lift, and the two ends left on either side still point
    squarely at each other. Nothing may lie between them: two lines meeting a
    third between them keep their gap, and so does a dimension line broken
    for a value written in or beside the gap.
    """
    pairs: list[tuple[int, int]] = []
    taken = set(bridged)
    for index, first in enumerate(ends):
        if index in taken:
            continue
        candidates = [
            (first.point.distance_to(second.point), other_index)
            for other_index, second in enumerate(ends)
            if other_index != index
            and other_index not in taken
            and _faces_squarely(first, second, facing.reach)
            and _paper_between(mask, first.point, second.point, skip)
            and not facing.labelled(first.point, second.point)
        ]
        if candidates:
            _, other_index = min(candidates)
            pairs.append((index, other_index))
            taken.update((index, other_index))
    return pairs


def with_stroke_bridges(
    mask: GrayImage,
    ends: tuple[StrokeEnd, ...],
    reach: float,
    thickness: int,
    facing: FacingReach | None = None,
) -> GrayImage:
    """Close the gaps a drawing's long strokes stop just short of.

    Bridges go into the face mask only; they never become geometry.

    - Two stroke ends whose straight continuations cross within `reach` of
      both are run to the crossing, the corner the pen was heading for; two
      ends facing each other along one line are joined directly.
    - Two ends pointing squarely at each other across blank paper are joined
      up to the longer `facing` reach, closing a faint stretch of one stroke,
      unless a text label sits by the gap.
    - A remaining end whose continuation meets other ink within `reach` is
      run to it, closing a T junction drawn a little short.
    """
    guided = mask.copy()
    skip = 2.0 * thickness + 2.0
    hits = tuple(_ink_ahead(mask, end, reach, skip) for end in ends)
    runs = tuple(
        reach if hit is None else end.point.distance_to(hit)
        for end, hit in zip(ends, hits, strict=True)
    )
    bridged: set[int] = set()
    for index in range(len(ends)):
        if index in bridged:
            continue
        partner = _closest_partner(ends, index, bridged, runs)
        if partner is None:
            continue
        other_index, meeting = partner
        _draw(guided, ends[index].point, meeting, thickness)
        _draw(guided, ends[other_index].point, meeting, thickness)
        bridged.update((index, other_index))
    for index, other_index in _facing_bridges(
        mask,
        ends,
        bridged,
        facing or FacingReach(),
        skip,
    ):
        _draw(guided, ends[index].point, ends[other_index].point, thickness)
        bridged.update((index, other_index))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    label_image = np.asarray(labels, dtype=np.int32)
    stat_table = np.asarray(stats, dtype=np.int64)
    extents = cast(
        "list[int]",
        np.maximum(stat_table[:, 2], stat_table[:, 3]).tolist(),
    )
    del count
    for index, (end, hit) in enumerate(zip(ends, hits, strict=True)):
        if index in bridged or hit is None:
            continue
        # A leader line pointing at a drilled hole stops just short of it;
        # joining the two would make the hole part of the outline.
        label = int(cast("int", label_image[round(hit.y), round(hit.x)]))
        if extents[label] < 2.0 * reach:
            continue
        _draw(guided, end.point, hit, thickness)
    return guided


def enclosed_faces(
    mask: GrayImage,
    close_size: int,
    minimum_area: float,
    grow: int,
) -> Iterator[Face]:
    """Yield every bounded paper region after closing gaps of a given size."""
    closed = (
        mask
        if close_size <= 1
        else np.asarray(
            cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (_odd(close_size), _odd(close_size)),
                ),
            ),
            dtype=np.uint8,
        )
    )
    paper = np.asarray(np.equal(closed, 0), dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    label_image = np.asarray(labels, dtype=np.int32)
    stat_rows = cast("list[list[int]]", np.asarray(stats).tolist())
    height, width = cast("tuple[int, int]", mask.shape[:2])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    # A leader or centre line drawn in from the outline leaves a slit one
    # stroke wide in the face; sealing it keeps the outline from following the
    # slit in to whatever the line points at.
    seal_size = _odd(4 * grow + 3)
    seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seal_size, seal_size))
    for label in range(1, int(count)):
        x, y, box_width, box_height, area = stat_rows[label]
        if (
            area < minimum_area
            or x == 0
            or y == 0
            or x + box_width >= width
            or y + box_height >= height
        ):
            continue
        pad = grow + seal_size
        region = np.zeros((box_height + 2 * pad, box_width + 2 * pad), dtype=np.uint8)
        region[pad : pad + box_height, pad : pad + box_width] = np.equal(
            label_image[y : y + box_height, x : x + box_width],
            label,
        )
        filled = _filled(
            np.asarray(
                cv2.morphologyEx(_filled(region), cv2.MORPH_CLOSE, seal), dtype=np.uint8
            ),
        )
        if grow > 0:
            filled = np.asarray(cv2.dilate(filled, kernel), dtype=np.uint8)
        contour = _largest_contour(filled)
        if contour is None:
            continue
        shifted = np.asarray(contour + np.array([x - pad, y - pad]), dtype=np.int32)
        yield Face(contour=shifted, area=float(cv2.contourArea(shifted)))


def _filled(region: GrayImage) -> GrayImage:
    """Fill a region's islands so lettering inside it does not dent it."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(region)
    _ = cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


def _largest_contour(region: GrayImage) -> Contour | None:
    """Return the outer contour of the largest connected part of a region."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return cast(
        "Contour",
        max(contours, key=lambda item: cv2.contourArea(cast("Contour", item))),
    )


def _fitted_circle(points: NDArray[np.float64]) -> tuple[Point, float, float] | None:
    """Fit a circle by least squares and return it with its RMS residual."""
    if len(points) < _MINIMUM_FIT_POINTS:
        return None
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((x, y, np.ones_like(x)))
    target = -(x * x + y * y)
    solution = cast(
        "list[float]",
        np.linalg.lstsq(design, target, rcond=None)[0].tolist(),
    )
    center_x = -solution[0] / 2.0
    center_y = -solution[1] / 2.0
    radius_squared = center_x * center_x + center_y * center_y - solution[2]
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        return None
    radius = math.sqrt(radius_squared)
    distances = np.hypot(x - center_x, y - center_y)
    residual = math.sqrt(float(np.mean((distances - radius) ** 2)))
    return Point(center_x, center_y), radius, residual


def _turn_degrees(previous: Point, current: Point, following: Point) -> float:
    """Return how far a polygon turns at one vertex, in degrees."""
    first = (current.x - previous.x, current.y - previous.y)
    second = (following.x - current.x, following.y - current.y)
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length == 0.0 or second_length == 0.0:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def merge_straight_corners(
    points: tuple[Point, ...],
    angle_degrees: float,
) -> tuple[Point, ...]:
    """Drop vertices where a closed polygon barely turns.

    A freehand edge wobbles, so the polygon approximation leaves a vertex part
    way along a side that the pen meant to be straight. Removing every vertex
    that turns less than the threshold keeps real corners, notches, and
    chamfers, whose turns are far larger.
    """
    kept = list(points)
    changed = True
    while changed and len(kept) > _MINIMUM_POLYGON_POINTS:
        changed = False
        turns = [
            _turn_degrees(kept[index - 1], kept[index], kept[(index + 1) % len(kept)])
            for index in range(len(kept))
        ]
        weakest = min(range(len(kept)), key=lambda index: turns[index])
        if turns[weakest] < angle_degrees:
            del kept[weakest]
            changed = True
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class _Roundness:
    """The measurements that decide whether a face is a drilled hole."""

    circularity: float
    rectangularity: float
    aspect: float
    coarse_vertices: int
    residual_ratio: float
    extent_ratio: float


def _reads_as_circle(roundness: _Roundness, config: FaceConfig) -> bool:
    """Return whether a face's measurements make it a circle.

    A face only a few stroke widths across has too few pixels to show its
    corners or its roundness, and on a fabrication sketch a mark that small
    inside a part is a drilled hole, so it needs only to be roughly as wide
    as it is tall.
    """
    if roundness.extent_ratio <= config.tiny_feature_ratio:
        # A small square cutout still fills its bounding box; a rough round
        # hole leaves the box's corners empty. At a radius of a few pixels the
        # pixel staircase alone strays a fifth of the radius from any circle.
        return (
            roundness.aspect >= config.tiny_circle_aspect_ratio
            and roundness.residual_ratio <= config.tiny_circle_residual_ratio
            and roundness.rectangularity <= config.tiny_maximum_rectangularity
        )
    if (
        roundness.aspect < config.circle_aspect_ratio
        or roundness.coarse_vertices <= _QUADRILATERAL_VERTICES
    ):
        return False
    if (
        roundness.circularity >= config.circle_circularity
        and roundness.rectangularity <= config.circle_maximum_rectangularity
        and roundness.residual_ratio <= config.circle_residual_ratio
    ):
        return True
    return (
        roundness.extent_ratio <= config.small_feature_ratio
        and roundness.circularity >= config.small_circle_circularity
        and roundness.residual_ratio <= config.small_circle_residual_ratio
    )


def ring_radius(mask: GrayImage, center: Point, radius: float) -> float:
    """Return the radius of the centre line of the ink ring around a hole.

    A hole's face is the paper inside its ring, so its fitted radius falls
    short of the drawn circle by half the ring's width, which on a small hole
    drawn with a heavy pen is half the radius itself. Rays from the centre
    find where the ring's ink starts and ends; the median midpoint is the
    drawn radius. Without a clear ring the fitted radius stands.
    """
    height, width = cast("tuple[int, int]", mask.shape[:2])
    limit = 2.0 * radius + _RING_SEARCH_MARGIN
    midpoints: list[float] = []
    for step in range(_RING_RAYS):
        angle = 2.0 * math.pi * step / _RING_RAYS
        cosine, sine = math.cos(angle), math.sin(angle)
        start: float | None = None
        distance = _RING_SEARCH_START * radius
        while distance <= limit:
            x = round(center.x + cosine * distance)
            y = round(center.y + sine * distance)
            if not (0 <= x < width and 0 <= y < height):
                break
            inked = int(cast("int", mask[y, x])) > 0
            if start is None and inked:
                start = distance
            elif start is not None and not inked:
                midpoints.append((start + distance - 1.0) / 2.0)
                break
            distance += 1.0
    if 2 * len(midpoints) < _RING_RAYS:
        return radius
    return float(np.median(np.asarray(midpoints, dtype=np.float64)))


def face_shape(
    face: Face,
    config: FaceConfig,
    minimum_side: int,
) -> tuple[float, ShapeCandidate] | None:
    """Read one face as a circle or a closed polygon.

    A face is a circle only when it fits a circle and does not reduce to four
    corners: a hand-drawn square is round enough by circularity alone to pass
    for one. A small face is held to a looser fit, because a quickly drawn
    drilled hole is rarely a true circle and is still meant as one.
    """
    contour = face.contour
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 0.0 or face.area <= 0.0:
        return None
    (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
    long_side = max(rect_width, rect_height)
    short_side = min(rect_width, rect_height)
    if long_side <= 0.0:
        return None
    # Two strokes drawn almost on top of each other leave a sliver of paper
    # between them, which is a doubled line and not a feature.
    if (
        long_side >= config.sliver_aspect_ratio * short_side
        and short_side <= config.sliver_width
    ):
        return None
    circularity = 4.0 * math.pi * face.area / (perimeter * perimeter)
    rectangularity = face.area / max(1.0, rect_width * rect_height)
    fit = _fitted_circle(contour.reshape(-1, 2).astype(np.float64))
    if fit is not None and _reads_as_circle(
        _Roundness(
            circularity=circularity,
            rectangularity=rectangularity,
            aspect=short_side / long_side,
            coarse_vertices=len(
                cv2.approxPolyDP(
                    contour,
                    _COARSE_APPROXIMATION * perimeter,
                    closed=True,
                ),
            ),
            residual_ratio=fit[2] / max(fit[1], 1e-9),
            extent_ratio=long_side / max(1, minimum_side),
        ),
        config,
    ):
        center, radius, _ = fit
        return face.area, CircleCandidate(
            entity_id=_PENDING_ID,
            center=center,
            radius=radius,
            confidence=min(0.99, circularity),
        )
    epsilon = max(
        config.minimum_approximation,
        config.approximation_ratio * minimum_side,
    )
    approximated: list[list[int]] = (
        np.asarray(cv2.approxPolyDP(contour, epsilon, closed=True), dtype=np.int32)
        .reshape(-1, 2)
        .tolist()
    )
    vertices = merge_straight_corners(
        tuple(Point(float(x), float(y)) for x, y in approximated),
        config.corner_merge_degrees,
    )
    if not _MINIMUM_POLYGON_POINTS <= len(vertices) <= config.maximum_vertices:
        return None
    return face.area, PolylineCandidate(
        entity_id=_PENDING_ID,
        points=vertices,
        closed=True,
        confidence=min(0.95, max(0.5, rectangularity)),
    )


def _outline_samples(
    candidate: CircleCandidate | PolylineCandidate,
) -> NDArray[np.float64]:
    """Return points every couple of pixels along a closed shape's outline."""
    if isinstance(candidate, CircleCandidate):
        count = max(
            _MINIMUM_OUTLINE_SAMPLES, round(2.0 * math.pi * candidate.radius / 2.0)
        )
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        return np.column_stack(
            (
                candidate.center.x + candidate.radius * np.cos(angles),
                candidate.center.y + candidate.radius * np.sin(angles),
            ),
        )
    corners = candidate.points
    samples: list[tuple[float, float]] = []
    for index, start in enumerate(corners):
        end = corners[(index + 1) % len(corners)]
        steps = max(1, round(start.distance_to(end) / 2.0))
        samples.extend(
            (
                start.x + (end.x - start.x) * step / steps,
                start.y + (end.y - start.y) * step / steps,
            )
            for step in range(steps)
        )
    return np.asarray(samples, dtype=np.float64)


def _distance_to_outline(
    points: NDArray[np.float64],
    candidate: CircleCandidate | PolylineCandidate,
) -> NDArray[np.float64]:
    """Return each point's distance to a closed shape's outline."""
    if isinstance(candidate, CircleCandidate):
        offsets = points - np.asarray([candidate.center.x, candidate.center.y])
        return np.abs(np.hypot(offsets[:, 0], offsets[:, 1]) - candidate.radius)
    corners = np.asarray([[point.x, point.y] for point in candidate.points])
    starts = corners
    ends = np.roll(corners, -1, axis=0)
    edges = ends - starts
    lengths = np.maximum(np.sum(edges * edges, axis=1), 1e-12)
    relative = points[:, None, :] - starts[None, :, :]
    along = np.clip(np.sum(relative * edges[None, :, :], axis=2) / lengths, 0.0, 1.0)
    nearest = starts[None, :, :] + along[:, :, None] * edges[None, :, :]
    gaps = points[:, None, :] - nearest
    distances: NDArray[np.float64] = np.min(
        np.hypot(gaps[:, :, 0], gaps[:, :, 1]), axis=1
    )
    return distances


def _shape_area(candidate: CircleCandidate | PolylineCandidate) -> float:
    """Return the area a closed shape encloses."""
    if isinstance(candidate, CircleCandidate):
        return math.pi * candidate.radius * candidate.radius
    points = candidate.points
    twice = sum(
        points[index].x * points[(index + 1) % len(points)].y
        - points[(index + 1) % len(points)].x * points[index].y
        for index in range(len(points))
    )
    return abs(twice) / 2.0


def _shape_centre(candidate: CircleCandidate | PolylineCandidate) -> Point:
    """Return a closed shape's centre: a circle's, or a polygon's mean corner."""
    if isinstance(candidate, CircleCandidate):
        return candidate.center
    count = len(candidate.points)
    return Point(
        sum(point.x for point in candidate.points) / count,
        sum(point.y for point in candidate.points) / count,
    )


def _encloses_point(
    candidate: CircleCandidate | PolylineCandidate, point: Point
) -> bool:
    """Return whether a point lies inside a closed shape."""
    if isinstance(candidate, CircleCandidate):
        return point.distance_to(candidate.center) < candidate.radius
    contour = np.asarray(
        [[corner.x, corner.y] for corner in candidate.points],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (point.x, point.y), measureDist=False) > 0.0


def without_interstitial_holes(
    candidates: tuple[ShapeCandidate, ...],
    near: float,
) -> tuple[ShapeCandidate, ...]:
    """Drop hole faces walled in almost entirely by larger holes' outlines.

    Where slots radiate from a ring with a size written inside it, the
    strokes of the slots, the ring, and the lettering wall off small pockets
    of paper. Each pocket is a face, and inside the part it reads as a hole,
    but its outline is made of other features' walls: most of it runs within
    `near` of a larger hole's outline. A drilled hole or slot stands in the
    part's own paper, so little of its outline runs along another hole's.
    """
    holes = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, (CircleCandidate, PolylineCandidate))
        and candidate.role is GeometryRole.HOLE
        and (not isinstance(candidate, PolylineCandidate) or candidate.closed)
    )
    areas = {id(hole): _shape_area(hole) for hole in holes}
    walled: set[int] = set()
    for hole in holes:
        # A slightly larger reading of the same feature encloses it, and its
        # outline runs right along this one's; that is no wall.
        larger = tuple(
            other
            for other in holes
            if areas[id(other)] > areas[id(hole)]
            and not (
                areas[id(other)] <= _SAME_FEATURE_AREA_RATIO * areas[id(hole)]
                and _encloses_point(other, _shape_centre(hole))
            )
        )
        if not larger:
            continue
        samples = _outline_samples(hole)
        nearest = np.min(
            np.vstack([_distance_to_outline(samples, other) for other in larger]),
            axis=0,
        )
        if float(np.mean(nearest <= near)) >= _INTERSTITIAL_SHARE:
            walled.add(id(hole))
    return tuple(candidate for candidate in candidates if id(candidate) not in walled)
