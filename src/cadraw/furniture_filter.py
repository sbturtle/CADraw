"""Line-level removal of printed form furniture that earlier proofs identified.

Keeping only the longest connected stroke group used to hide two kinds of
printed marks by accident: the rules of a form's header table, and the pieces
of graph-paper ruling that survive erasure where a rail drifted off its row.
Once every part on a sheet is kept, those marks come back as isolated lines.

Nothing here is removed on position or length alone. A line goes only when it
lies on the edge of a table cell that `table_filter` already proved by tiling,
or on a ruling rail that `page_cleanup` already proved periodic on both axes
and the ink under it is as light as printed ruling. A drawn stroke that
happens to follow a rail is dark, joined to the rest of the drawing, or both,
and stays.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from cadraw.models import (
    ArcEntity,
    CircleEntity,
    LineEntity,
    Orientation,
    Point,
    PolylineEntity,
)

type Bounds = tuple[float, float, float, float]
type StrokeIntensity = Callable[[Point, Point], float]

_MIN_EDGE_OVERLAP_RATIO = 0.50
_PRINTED_RULE_WOBBLE = 0.6
_RECTANGLE_CORNERS = 4
_PRINTED_ROW_FILL = 0.95
_PRINTED_ROW_ASPECT = 4.0
_PRINTED_ROW_SPAN = 0.8
_JUDGED_EDGE_RATIO = 0.25


@dataclass(frozen=True, slots=True)
class RulingRails:
    """Proved ruling rail positions in the coordinates of the lines."""

    horizontal: tuple[float, ...]
    vertical: tuple[float, ...]
    band_width: float
    maximum_intensity: float


def _span(first: float, second: float) -> tuple[float, float]:
    """Return an ordered interval."""
    return (first, second) if first <= second else (second, first)


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return the overlap length of two ordered intervals."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _lies_on_cell_edge(line: LineEntity, cell: Bounds, tolerance: float) -> bool:
    """Return whether an axis-aligned line runs along one edge of a cell."""
    min_x, min_y, max_x, max_y = cell
    length = line.length
    if length == 0.0:
        return False
    if line.orientation is Orientation.HORIZONTAL:
        y = (line.start.y + line.end.y) / 2.0
        if min(abs(y - min_y), abs(y - max_y)) > tolerance:
            return False
        along = _span(line.start.x, line.end.x)
        return _overlap(along, (min_x, max_x)) / length >= _MIN_EDGE_OVERLAP_RATIO
    if line.orientation is Orientation.VERTICAL:
        x = (line.start.x + line.end.x) / 2.0
        if min(abs(x - min_x), abs(x - max_x)) > tolerance:
            return False
        along = _span(line.start.y, line.end.y)
        return _overlap(along, (min_y, max_y)) / length >= _MIN_EDGE_OVERLAP_RATIO
    return False


def without_table_edge_lines(
    lines: tuple[LineEntity, ...],
    cells: tuple[Bounds, ...],
    tolerance: float,
) -> tuple[LineEntity, ...]:
    """Drop axis-aligned lines that trace the edges of proved table cells.

    The cells were proved as a printed table by tiling, so a line that shares
    one of their edges for at least half its own length is that table's rule
    seen by the line detector rather than by the contour detector.
    """
    if not cells:
        return lines
    return tuple(
        line
        for line in lines
        if not any(_lies_on_cell_edge(line, cell, tolerance) for cell in cells)
    )


def _is_isolated(
    line: LineEntity,
    lines: tuple[LineEntity, ...],
    snap_distance: float,
) -> bool:
    """Return whether no other line meets this one at an endpoint."""
    return not any(
        other is not line
        and any(
            point.distance_to(other_point) <= snap_distance
            for point in (line.start, line.end)
            for other_point in (other.start, other.end)
        )
        for other in lines
    )


def _on_rail(line: LineEntity, rails: RulingRails) -> bool:
    """Return whether an axis-aligned line sits on a proved ruling rail."""
    if line.orientation is Orientation.HORIZONTAL:
        offset = (line.start.y + line.end.y) / 2.0
        family = rails.horizontal
    elif line.orientation is Orientation.VERTICAL:
        offset = (line.start.x + line.end.x) / 2.0
        family = rails.vertical
    else:
        return False
    return any(abs(offset - rail) <= rails.band_width for rail in family)


def _segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return the distance from a point to a line segment."""
    delta_x = end.x - start.x
    delta_y = end.y - start.y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        return point.distance_to(start)
    along = (
        (point.x - start.x) * delta_x + (point.y - start.y) * delta_y
    ) / length_squared
    along = max(0.0, min(1.0, along))
    return point.distance_to(
        Point(start.x + along * delta_x, start.y + along * delta_y),
    )


def _on_arc(point: Point, arc: ArcEntity, snap_distance: float) -> bool:
    """Return whether a point lies on an open arc's curve."""
    if abs(point.distance_to(arc.center) - arc.radius) > snap_distance:
        return False
    angle = math.degrees(
        math.atan2(point.y - arc.center.y, point.x - arc.center.x),
    )
    sweep = (arc.end_angle_degrees - arc.start_angle_degrees) % 360.0
    # The snap distance also reaches a little past either end of the arc.
    slack = math.degrees(snap_distance / max(arc.radius, 1.0))
    return (angle - arc.start_angle_degrees + slack) % 360.0 <= sweep + 2.0 * slack


def _touches_shape(
    point: Point,
    shapes: tuple[PolylineEntity | CircleEntity | ArcEntity, ...],
    snap_distance: float,
) -> bool:
    """Return whether a point lies on a shape's outline or on an arc."""
    for shape in shapes:
        if isinstance(shape, ArcEntity):
            if _on_arc(point, shape, snap_distance):
                return True
            continue
        if isinstance(shape, CircleEntity):
            if abs(point.distance_to(shape.center) - shape.radius) <= snap_distance:
                return True
            continue
        corners = shape.points
        if any(
            _segment_distance(
                point, corners[index], corners[(index + 1) % len(corners)]
            )
            <= snap_distance
            for index in range(len(corners))
        ):
            return True
    return False


def without_printed_rules(
    lines: tuple[LineEntity, ...],
    shapes: tuple[PolylineEntity | CircleEntity | ArcEntity, ...],
    snap_distance: float,
    minimum_length: float,
    wobble: StrokeIntensity,
) -> tuple[LineEntity, ...]:
    """Drop long, isolated lines whose ink is as straight as print.

    A form's header and footer rules can sit anywhere on the sheet, so the
    page-edge test misses most of them. What gives them away is their
    straightness: a printed rule stays within a fraction of a pixel of one
    line along its whole length, while a hand-drawn stroke of that length
    wanders by a pixel or more even when drawn against a ruler. A line joined
    to the rest of the drawing stays, however straight: a printed rule meets
    no drawn stroke at its ends, neither another line, a closed outline, nor
    an arc such as a rounded corner.
    """
    return tuple(
        line
        for line in lines
        if not (
            line.length >= minimum_length
            and not _meets_another_line(line, lines, snap_distance)
            and not any(
                _touches_shape(point, shapes, snap_distance)
                for point in (line.start, line.end)
            )
            and wobble(line.start, line.end) <= _PRINTED_RULE_WOBBLE
        )
    )


def _meets_another_line(
    line: LineEntity,
    lines: tuple[LineEntity, ...],
    snap_distance: float,
) -> bool:
    """Return whether any other line touches this one anywhere.

    A branch drawn from the middle of a long edge leaves the edge unsplit, so
    meeting only at shared endpoints would miss the joint of a T. Either
    line's end lying on the other line counts.
    """
    return any(
        other is not line
        and (
            any(
                _segment_distance(point, line.start, line.end) <= snap_distance
                for point in (other.start, other.end)
            )
            or any(
                _segment_distance(point, other.start, other.end) <= snap_distance
                for point in (line.start, line.end)
            )
        )
        for other in lines
    )


def is_printed_row(
    points: tuple[Point, ...],
    page_width: float,
    wobble: StrokeIntensity,
) -> bool:
    """Return whether a closed polygon is a printed form row spanning the page.

    A work-order form's remarks row is one long ruled cell nearly as wide as
    the sheet. All of these must hold, and none alone suffices: the polygon
    fills its own oriented bounding box (a rectangle, whatever small corner
    cut tracing left), the band is at least `_PRINTED_ROW_ASPECT` times as
    long as it is deep and `_PRINTED_ROW_SPAN` of the page wide, and every
    edge long enough to judge is as straight as print. A part drawn against
    a ruler can be that straight, but it has chamfers or notches, or does not
    run across the whole sheet.
    """
    if len(points) < _RECTANGLE_CORNERS:
        return False
    edges = tuple(
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )
    length, depth = _oriented_extent(points, max(edges, key=_edge_length))
    xs = [point.x for point in points]
    if (
        depth <= 0.0
        or _polygon_area(points) < _PRINTED_ROW_FILL * length * depth
        or length < _PRINTED_ROW_ASPECT * depth
        or max(xs) - min(xs) < _PRINTED_ROW_SPAN * page_width
    ):
        return False
    judged = tuple(
        edge for edge in edges if _edge_length(edge) >= _JUDGED_EDGE_RATIO * depth
    )
    return len(judged) >= _RECTANGLE_CORNERS and all(
        wobble(start, end) <= _PRINTED_RULE_WOBBLE for start, end in judged
    )


def _edge_length(edge: tuple[Point, Point]) -> float:
    """Return the length of one polygon edge."""
    return edge[0].distance_to(edge[1])


def _oriented_extent(
    points: tuple[Point, ...],
    axis: tuple[Point, Point],
) -> tuple[float, float]:
    """Return a polygon's extent along an edge's direction and across it."""
    size = max(_edge_length(axis), 1e-9)
    direction_x = (axis[1].x - axis[0].x) / size
    direction_y = (axis[1].y - axis[0].y) / size
    along = [point.x * direction_x + point.y * direction_y for point in points]
    across = [point.y * direction_x - point.x * direction_y for point in points]
    return max(along) - min(along), max(across) - min(across)


def _polygon_area(points: tuple[Point, ...]) -> float:
    """Return the area a closed polygon encloses (shoelace formula)."""
    twice = sum(
        points[index].x * points[(index + 1) % len(points)].y
        - points[(index + 1) % len(points)].x * points[index].y
        for index in range(len(points))
    )
    return abs(twice) / 2.0


def without_ruling_remnants(
    lines: tuple[LineEntity, ...],
    rails: RulingRails,
    snap_distance: float,
    intensity: StrokeIntensity,
) -> tuple[LineEntity, ...]:
    """Drop light, isolated lines lying on rails that ruling removal proved.

    Erasure keeps only the straight run of each rail, so where a printed rail
    wanders off its row a short piece survives and the line detector finds it.
    Three conditions must hold at once: the line sits on a proved rail, no
    other line meets it, and its ink is no darker than printed ruling. A drawn
    stroke along a rail fails the last test, and a member of the drawing fails
    the second.
    """
    if not rails.horizontal and not rails.vertical:
        return lines
    return tuple(
        line
        for line in lines
        if not (
            _on_rail(line, rails)
            and _is_isolated(line, lines, snap_distance)
            and intensity(line.start, line.end) >= rails.maximum_intensity
        )
    )
