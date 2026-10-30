"""Recognition of drawn dimension lines and the geometry they measure.

A formally drafted sketch states a size on its own line beside the part: a
straight stroke closed by a short tick across each end, with the value
written along it. The line detector sees that stroke like any edge. Kept as
geometry, it becomes a member of the part, and the value written beside it
attaches to it, the nearest line, instead of to the circle or edge it
measures.

A line is read as a dimension line only when three things agree: a tick
crosses each end, the stroke stops at the tick, and a parsed value lies
beside it. Several values along one line make a chain whose ends measure
their sum, so such a line gets no target. A plate's short end edge looks
like a tick too, but the long edge it closes runs on past the tick length;
a hole's centre line has no ticks.

The ends of the line are what it measures, so a circle whose two poles
along the line fall on those ends, or a single edge that spans them, is the
target, as is a closed outline whose overall width or height they span. A
dimension is drawn beside what it measures, so the target must lie within
one line length of it.
Anything else leaves the value unresolved for the reviewer; it is never
handed to the nearest line instead.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cadraw.models import (
    CircleEntity,
    Dimension,
    DimensionKind,
    EntityId,
    GeometryEntity,
    LineEntity,
    Point,
    PolylineEntity,
)

type InkMask = NDArray[np.uint8]
type Bounds = tuple[float, float, float, float]

_MINIMUM_TICK_WIDTHS = 4.0
_MAXIMUM_TICK_WIDTHS = 12.0
_MAXIMUM_OVERRUN_WIDTHS = 3.0
_TICK_SETBACK_WIDTHS = (0.0, 2.0, 4.0)
_INK_GAP = 2
_TEXT_ALONG_SLACK = 0.1
_TEXT_ACROSS_HEIGHTS = 1.5
_TARGET_SLACK_WIDTHS = 3.0
_TARGET_SLACK_RATIO = 0.1
_PARALLEL_DEGREES = 8.0
_TARGET_REACH_LENGTHS = 1.0


@dataclass(frozen=True, slots=True)
class DimensionLine:
    """One recognized dimension line with the value and target it carries.

    `dimension_index` points into the OCR dimensions the value came from.
    `target_id` is `None` when nothing, or more than one thing, matches the
    line's ends. `kind` is what the target takes: a circle's diameter, an
    edge's length, or a closed outline's width or height.
    """

    line_id: EntityId
    dimension_index: int
    target_id: EntityId | None
    kind: DimensionKind


@dataclass(frozen=True, slots=True)
class DimensionText:
    """A parsed OCR value with the box it was read from."""

    dimension: Dimension
    bounds: Bounds


def _ink_run(
    mask: InkMask,
    start: Point,
    direction: tuple[float, float],
    limit: float,
) -> int:
    """Return how far ink runs from a point, bridging pixel-sized breaks."""
    height, width = cast("tuple[int, int]", mask.shape[:2])
    last = 0
    gap = 0
    for step in range(1, math.floor(limit) + 1):
        x = round(start.x + direction[0] * step)
        y = round(start.y + direction[1] * step)
        if not (0 <= x < width and 0 <= y < height):
            break
        if int(cast("int", mask[y, x])) > 0:
            last = step
            gap = 0
            continue
        gap += 1
        if gap > _INK_GAP:
            break
    return last


def _band_run(
    mask: InkMask,
    point: Point,
    direction: tuple[float, float],
    limit: float,
    half_width: float,
) -> int:
    """Return the longest ink run over a band of parallel rays a stroke wide."""
    normal = (-direction[1], direction[0])
    steps = max(0, math.floor(half_width))
    return max(
        _ink_run(
            mask,
            Point(point.x + normal[0] * offset, point.y + normal[1] * offset),
            direction,
            limit,
        )
        for offset in range(-steps, steps + 1)
    )


def _tick_sides(
    mask: InkMask,
    end: Point,
    inward: tuple[float, float],
    normal: tuple[float, float],
    width: float,
) -> frozenset[int] | None:
    """Return the sides of a line a short tick crosses at one end, if any.

    The sides are +1 along the line's `normal` and -1 against it, so both
    ends of one line report them alike. A hand-drawn tick may cross a
    little inside the end, so the test also looks a couple of pen widths
    back along the line. `None` means no tick: too little ink across the
    end, a stroke running on past it, or ink across it longer than a tick.
    """
    longest = _MAXIMUM_TICK_WIDTHS * width
    outward = (-inward[0], -inward[1])
    if _band_run(mask, end, outward, longest, width / 2.0) > (
        _MAXIMUM_OVERRUN_WIDTHS * width
    ):
        return None
    for setback in _TICK_SETBACK_WIDTHS:
        point = Point(
            end.x + inward[0] * setback * width,
            end.y + inward[1] * setback * width,
        )
        runs = {
            side: _band_run(
                mask,
                point,
                (normal[0] * side, normal[1] * side),
                1.5 * longest,
                width,
            )
            for side in (1, -1)
        }
        if (
            sum(runs.values()) >= _MINIMUM_TICK_WIDTHS * width
            and max(runs.values()) <= longest
        ):
            return frozenset(side for side, run in runs.items() if run >= width)
    return None


def _ticked(mask: InkMask, line: LineEntity, width: float) -> bool:
    """Return whether a tick closes each end of a line.

    A tick crosses the line, so it shows on both sides. A freehand line can
    meet its tick off centre, and then one end may show it on one side only;
    the other end then shows it on the other side or both. Short strokes
    leaving both ends on the same side are the corners of an open outline,
    such as a channel's flanges, and not ticks.
    """
    direction = _direction(line)
    if direction is None:
        return False
    normal = (-direction[1], direction[0])
    start = _tick_sides(mask, line.start, direction, normal, width)
    end = _tick_sides(mask, line.end, (-direction[0], -direction[1]), normal, width)
    if not start or not end:
        return False
    return not (len(start) == 1 and start == end)


def _direction(line: LineEntity) -> tuple[float, float] | None:
    """Return a line's unit direction from start to end."""
    length = line.length
    if length == 0.0:
        return None
    return ((line.end.x - line.start.x) / length, (line.end.y - line.start.y) / length)


def _beside(line: LineEntity, direction: tuple[float, float], bounds: Bounds) -> float:
    """Return how far a text box's centre lies off a line, or infinity."""
    x, y, box_width, box_height = bounds
    centre_x = x + box_width / 2.0
    centre_y = y + box_height / 2.0
    along = (centre_x - line.start.x) * direction[0] + (
        centre_y - line.start.y
    ) * direction[1]
    across = abs(
        (centre_x - line.start.x) * direction[1]
        - (centre_y - line.start.y) * direction[0],
    )
    length = line.length
    reach = _TEXT_ACROSS_HEIGHTS * min(box_width, box_height) + 0.5 * max(
        box_width,
        box_height,
    )
    if not (-_TEXT_ALONG_SLACK * length <= along <= (1.0 + _TEXT_ALONG_SLACK) * length):
        return math.inf
    return across if across <= reach else math.inf


def _project(point: Point, origin: Point, direction: tuple[float, float]) -> float:
    """Return a point's position along a line through an origin."""
    return (point.x - origin.x) * direction[0] + (point.y - origin.y) * direction[1]


def _spans(
    line: LineEntity,
    first: float,
    second: float,
    slack: float,
) -> bool:
    """Return whether two positions along a line fall on its two ends."""
    low, high = sorted((first, second))
    return abs(low) <= slack and abs(high - line.length) <= slack


def _parallel(direction: tuple[float, float], other: tuple[float, float]) -> bool:
    """Return whether two directions agree within the parallel tolerance."""
    cosine = abs(direction[0] * other[0] + direction[1] * other[1])
    return cosine >= math.cos(math.radians(_PARALLEL_DEGREES))


def _across(point: Point, line: LineEntity, direction: tuple[float, float]) -> float:
    """Return how far a point lies off a line's supporting axis."""
    return abs(
        (point.x - line.start.x) * direction[1]
        - (point.y - line.start.y) * direction[0],
    )


def _gap(
    line: LineEntity,
    direction: tuple[float, float],
    entity: GeometryEntity,
) -> float:
    """Return how far across from a dimension line a shape's nearest side is."""
    if isinstance(entity, CircleEntity):
        return max(0.0, _across(entity.center, line, direction) - entity.radius)
    if isinstance(entity, LineEntity):
        return min(
            _across(point, line, direction) for point in (entity.start, entity.end)
        )
    if isinstance(entity, PolylineEntity) and entity.points:
        return min(_across(point, line, direction) for point in entity.points)
    return math.inf


def _axis_kind(direction: tuple[float, float]) -> DimensionKind | None:
    """Return the outline extent a horizontal or vertical line measures."""
    limit = math.cos(math.radians(_PARALLEL_DEGREES))
    if abs(direction[0]) >= limit:
        return DimensionKind.WIDTH
    if abs(direction[1]) >= limit:
        return DimensionKind.HEIGHT
    return None


def _measured(
    line: LineEntity,
    direction: tuple[float, float],
    geometry: tuple[GeometryEntity, ...],
    width: float,
) -> tuple[tuple[GeometryEntity, DimensionKind], ...]:
    """Return everything whose extent matches the line's ends, with its kind.

    A circle's two poles along the line give its diameter, a parallel edge
    spanning the ends its length, and a closed outline whose overall extent
    along a horizontal or vertical line matches gives its width or height.
    """
    slack = max(_TARGET_SLACK_WIDTHS * width, _TARGET_SLACK_RATIO * line.length)
    extent_kind = _axis_kind(direction)
    matches: list[tuple[GeometryEntity, DimensionKind]] = []
    for entity in geometry:
        if _gap(line, direction, entity) > _TARGET_REACH_LENGTHS * line.length:
            continue
        if isinstance(entity, CircleEntity):
            centre = _project(entity.center, line.start, direction)
            if _spans(line, centre - entity.radius, centre + entity.radius, slack):
                matches.append((entity, DimensionKind.DIAMETER))
        elif isinstance(entity, LineEntity) and entity.entity_id != line.entity_id:
            other = _direction(entity)
            if (
                other is not None
                and _parallel(direction, other)
                and _spans(
                    line,
                    _project(entity.start, line.start, direction),
                    _project(entity.end, line.start, direction),
                    slack,
                )
            ):
                matches.append((entity, DimensionKind.LENGTH))
        elif (
            isinstance(entity, PolylineEntity)
            and entity.closed
            and entity.points
            and extent_kind is not None
        ):
            positions = [
                _project(point, line.start, direction) for point in entity.points
            ]
            if _spans(line, min(positions), max(positions), slack):
                matches.append((entity, extent_kind))
    return tuple(matches)


def _measuring(
    line: LineEntity,
    direction: tuple[float, float],
    geometry: tuple[GeometryEntity, ...],
    width: float,
    text: DimensionText,
) -> tuple[EntityId | None, DimensionKind]:
    """Return the one target a value beside a line measures, and its kind.

    A value carrying its own symbol, such as `Ø40` or `R5`, keeps it, and
    only a target of that kind may take it.
    """
    stated = text.dimension.kind
    matches = tuple(
        (entity, kind)
        for entity, kind in _measured(line, direction, geometry, width)
        if stated is DimensionKind.LENGTH or kind is stated
    )
    if len(matches) != 1:
        return None, stated
    entity, kind = matches[0]
    return entity.entity_id, kind


def find_dimension_lines(
    lines: tuple[LineEntity, ...],
    texts: tuple[DimensionText, ...],
    geometry: tuple[GeometryEntity, ...],
    ink: InkMask,
    width: float,
) -> tuple[DimensionLine, ...]:
    """Return the drawn dimension lines, one record per value they carry.

    `width` is the pen width in the same pixels as `ink`. Each value goes to
    the ticked line it sits closest beside. A line given one value measures
    what its ends span; a line given several is a chain, whose ends bound
    the sum of its values, so none of them gets a target. Dimension lines
    are annotation, so none is a target of another.
    """
    ticked = tuple(line for line in lines if _ticked(ink, line, width))
    carried: dict[EntityId, list[int]] = {}
    for index, text in enumerate(texts):
        if text.dimension.parsed_value is None:
            continue
        nearest = min(
            (
                (distance, str(line.entity_id), line.entity_id)
                for line in ticked
                if (direction := _direction(line)) is not None
                and math.isfinite(distance := _beside(line, direction, text.bounds))
            ),
            default=None,
        )
        if nearest is not None:
            carried.setdefault(nearest[2], []).append(index)
    measurable = tuple(entity for entity in geometry if entity.entity_id not in carried)
    found: list[DimensionLine] = []
    for line in ticked:
        indices = carried.get(line.entity_id, [])
        direction = _direction(line)
        if not indices or direction is None:
            continue
        for index in indices:
            target, kind = (
                (None, texts[index].dimension.kind)
                if len(indices) > 1
                else _measuring(line, direction, measurable, width, texts[index])
            )
            found.append(
                DimensionLine(
                    line_id=line.entity_id,
                    dimension_index=index,
                    target_id=target,
                    kind=kind,
                ),
            )
    return tuple(found)
