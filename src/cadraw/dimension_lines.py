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


def _ticked(
    mask: InkMask, end: Point, inward: tuple[float, float], width: float
) -> bool:
    """Return whether a short tick crosses a line's end and the line stops there.

    A hand-drawn tick may cross a little inside the end, so the test also
    looks a couple of pen widths back along the line.
    """
    across = (-inward[1], inward[0])
    longest = _MAXIMUM_TICK_WIDTHS * width
    outward = (-inward[0], -inward[1])
    if _band_run(mask, end, outward, longest, width / 2.0) > (
        _MAXIMUM_OVERRUN_WIDTHS * width
    ):
        return False
    for setback in _TICK_SETBACK_WIDTHS:
        point = Point(
            end.x + inward[0] * setback * width,
            end.y + inward[1] * setback * width,
        )
        one_side = _band_run(mask, point, across, 1.5 * longest, width)
        other_side = _band_run(
            mask,
            point,
            (-across[0], -across[1]),
            1.5 * longest,
            width,
        )
        if (
            one_side + other_side >= _MINIMUM_TICK_WIDTHS * width
            and max(one_side, other_side) <= longest
        ):
            return True
    return False


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

    `width` is the pen width in the same pixels as `ink`. Values and ticked
    lines are paired closest first. A line takes one value, unless several
    are written along it: that is a chain, whose ends bound the sum of its
    values, so it takes all of them and gives none a target.
    """
    pairs: list[tuple[float, str, int, LineEntity, tuple[float, float]]] = []
    chained: set[EntityId] = set()
    for line in lines:
        direction = _direction(line)
        if direction is None:
            continue
        beside = [
            (distance, index)
            for index, text in enumerate(texts)
            if text.dimension.parsed_value is not None
            and math.isfinite(distance := _beside(line, direction, text.bounds))
        ]
        if (
            beside
            and _ticked(ink, line.start, direction, width)
            and _ticked(ink, line.end, (-direction[0], -direction[1]), width)
        ):
            pairs.extend(
                (distance, str(line.entity_id), index, line, direction)
                for distance, index in beside
            )
            if len(beside) > 1:
                chained.add(line.entity_id)
    found: list[DimensionLine] = []
    used_lines: set[EntityId] = set()
    used_texts: set[int] = set()
    for _, _, index, line, direction in sorted(pairs, key=lambda item: item[:3]):
        if index in used_texts or (
            line.entity_id in used_lines and line.entity_id not in chained
        ):
            continue
        used_lines.add(line.entity_id)
        used_texts.add(index)
        target, kind = (
            (None, texts[index].dimension.kind)
            if line.entity_id in chained
            else _measuring(line, direction, geometry, width, texts[index])
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
