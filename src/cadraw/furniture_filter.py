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
_PRINTED_RULE_SKEW = 3.0
_RECTANGLE_CORNERS = 4
_PRINTED_ROW_FILL = 0.95
_PRINTED_ROW_ASPECT = 4.0
_PRINTED_ROW_SPAN = 0.8
_JUDGED_EDGE_RATIO = 0.25
_ATTACHED_CELLS = 2
_ATTACHED_CELL_SLACK = 4.0


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


@dataclass(frozen=True, slots=True)
class RuleLengths:
    """How long a printed rule must be, alone or among others it touches.

    `single` is the least length of a lone rule. A group of touching rules
    needs one at least `spanning` long: a form's ruled T crosses the form,
    while the straight corner of a part drawn against a ruler does not.
    """

    single: float
    spanning: float


def without_printed_rules(
    lines: tuple[LineEntity, ...],
    shapes: tuple[PolylineEntity | CircleEntity | ArcEntity, ...],
    snap_distance: float,
    lengths: RuleLengths,
    wobble: StrokeIntensity,
) -> tuple[LineEntity, ...]:
    """Drop groups of touching lines that are all as straight as print.

    A form's header and footer rules can sit anywhere on the sheet, so the
    page-edge test misses most of them. What gives them away is their
    straightness: a printed rule stays within a fraction of a pixel of one
    line along its whole length, while a hand-drawn stroke of that length
    wanders by a pixel or more even when drawn against a ruler.

    Lines are judged in groups that touch anywhere, end to end or at a T, so
    a form's ruled T goes as one piece. A group goes only when every line in
    it is as straight as print and runs along the page's axes, none of them
    meets a closed outline or an arc, and it is long enough for print: a lone
    line `lengths.single` long, or a group with a line `lengths.spanning`
    long. One wavering stroke or one slanted edge makes the group part of the
    drawing, and so does a straight corner of lines too short to cross the
    form.
    """
    straight = {
        id(line): wobble(line.start, line.end) <= _PRINTED_RULE_WOBBLE
        and _axis_aligned(line)
        for line in lines
    }
    printed: set[int] = set()
    for group in _touching_groups(lines, snap_distance):
        needed = lengths.single if len(group) == 1 else lengths.spanning
        if (
            all(straight[id(line)] for line in group)
            and any(line.length >= needed for line in group)
            and not any(
                _touches_shape(point, shapes, snap_distance)
                for line in group
                for point in (line.start, line.end)
            )
        ):
            printed.update(id(line) for line in group)
    return tuple(line for line in lines if id(line) not in printed)


def _axis_aligned(line: LineEntity) -> bool:
    """Return whether a line runs along the page's axes, as printed rules do."""
    angle = math.degrees(
        math.atan2(line.end.y - line.start.y, line.end.x - line.start.x)
    )
    deviation = abs(angle) % 90.0
    return min(deviation, 90.0 - deviation) <= _PRINTED_RULE_SKEW


def _lines_touch(first: LineEntity, second: LineEntity, snap_distance: float) -> bool:
    """Return whether either line's end lies on the other line."""
    return any(
        _segment_distance(point, first.start, first.end) <= snap_distance
        for point in (second.start, second.end)
    ) or any(
        _segment_distance(point, second.start, second.end) <= snap_distance
        for point in (first.start, first.end)
    )


def _touching_groups(
    lines: tuple[LineEntity, ...],
    snap_distance: float,
) -> tuple[tuple[LineEntity, ...], ...]:
    """Group lines that touch one another, directly or through others."""
    unvisited = list(lines)
    groups: list[tuple[LineEntity, ...]] = []
    while unvisited:
        group = [unvisited.pop()]
        frontier = list(group)
        while frontier:
            current = frontier.pop()
            touching = [
                other
                for other in unvisited
                if _lines_touch(current, other, snap_distance)
            ]
            for other in touching:
                unvisited.remove(other)
            group.extend(touching)
            frontier.extend(touching)
        groups.append(tuple(group))
    return tuple(groups)


def is_printed_row(
    points: tuple[Point, ...],
    page_width: float,
    wobble: StrokeIntensity,
    proved_cells: tuple[Bounds, ...],
) -> bool:
    """Return whether a closed polygon is a printed form row spanning the page.

    A work-order form's remarks row is one long ruled cell nearly as wide as
    the sheet, ruled directly under the header table. All of these must hold,
    and none alone suffices: the polygon fills its own oriented bounding box
    (a rectangle, whatever small corner cut tracing left), the band is at
    least `_PRINTED_ROW_ASPECT` times as long as it is deep and
    `_PRINTED_ROW_SPAN` of the page wide, every edge long enough to judge is
    as straight as print, and at least `_ATTACHED_CELLS` cells of a table
    already proved by tiling sit against it. A long bar drawn against a ruler
    can be as straight and as wide, but no proved table is ruled onto it.
    """
    if len(points) < _RECTANGLE_CORNERS or not _attached_to_table(
        points,
        proved_cells,
    ):
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


def _attached_to_table(
    points: tuple[Point, ...],
    proved_cells: tuple[Bounds, ...],
) -> bool:
    """Return whether enough proved table cells sit against a polygon.

    A cell counts when its box overlaps the polygon's box, grown by a few
    pixels, while its centre lies outside the polygon: it is ruled onto the
    row from beside it, not drawn inside it.
    """
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    low_x = min(xs) - _ATTACHED_CELL_SLACK
    high_x = max(xs) + _ATTACHED_CELL_SLACK
    low_y = min(ys) - _ATTACHED_CELL_SLACK
    high_y = max(ys) + _ATTACHED_CELL_SLACK
    attached = sum(
        1
        for cell_low_x, cell_low_y, cell_high_x, cell_high_y in proved_cells
        if cell_low_x <= high_x
        and cell_high_x >= low_x
        and cell_low_y <= high_y
        and cell_high_y >= low_y
        and not _inside_polygon(
            Point((cell_low_x + cell_high_x) / 2.0, (cell_low_y + cell_high_y) / 2.0),
            points,
        )
    )
    return attached >= _ATTACHED_CELLS


def _inside_polygon(point: Point, points: tuple[Point, ...]) -> bool:
    """Return whether a point lies inside a polygon (even-odd rule)."""
    inside = False
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        if (first.y > point.y) != (second.y > point.y):
            crossing = first.x + (point.y - first.y) * (second.x - first.x) / (
                second.y - first.y
            )
            if point.x < crossing:
                inside = not inside
    return inside


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
