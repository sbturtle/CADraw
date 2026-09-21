"""Removal of open lines that re-trace the edge of a detected closed outline.

The line detector and the contour detector see the same ink. Where a cut
plate's outline closes into a polygon, its straight edges also come back as
Hough lines, so the same physical edge reaches the reviewer twice: once as a
segment with its own length row and once as part of the profile. The
reference drawings carry each edge once, and so should the export.

A line is treated as a re-trace only when its two ends and its midpoint all
lie within a small distance of the same polygon's boundary. A line that
starts on an edge and leaves it, such as a dimension tick or a member that
continues past the outline, keeps one of those three points away from the
boundary and stays.
"""

from cadraw.models import LineEntity, Point, PolylineEntity


def _segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return the distance from a point to a segment."""
    delta_x = end.x - start.x
    delta_y = end.y - start.y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        return point.distance_to(start)
    projection = (
        (point.x - start.x) * delta_x + (point.y - start.y) * delta_y
    ) / length_squared
    clamped = max(0.0, min(1.0, projection))
    return point.distance_to(
        Point(start.x + clamped * delta_x, start.y + clamped * delta_y),
    )


def _boundary_distance(point: Point, polygon: PolylineEntity) -> float:
    """Return the distance from a point to the closed polygon's boundary."""
    points = polygon.points
    return min(
        _segment_distance(point, points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _retraces(line: LineEntity, polygon: PolylineEntity, tolerance: float) -> bool:
    """Return whether the whole line lies along the polygon's boundary."""
    midpoint = Point(
        (line.start.x + line.end.x) / 2.0,
        (line.start.y + line.end.y) / 2.0,
    )
    return all(
        _boundary_distance(point, polygon) <= tolerance
        for point in (line.start, midpoint, line.end)
    )


def without_polygon_edge_lines(
    lines: tuple[LineEntity, ...],
    polygons: tuple[PolylineEntity, ...],
    tolerance: float,
) -> tuple[LineEntity, ...]:
    """Drop lines whose ends and midpoint all hug one closed polygon's edge."""
    closed = tuple(polygon for polygon in polygons if polygon.closed and polygon.points)
    if not closed:
        return lines
    return tuple(
        line
        for line in lines
        if not any(_retraces(line, polygon, tolerance) for polygon in closed)
    )
