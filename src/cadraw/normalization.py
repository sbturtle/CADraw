"""Deterministic line orientation, snapping, merging, and connectivity."""

from dataclasses import dataclass, replace
from math import atan2, degrees
from typing import Final

from cadraw.models import (
    Connection,
    LineCandidate,
    LineEntity,
    Orientation,
    Point,
)

_FULL_TURN: Final[float] = 180.0


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Tolerances controlling every line normalization decision."""

    horizontal_angle_tolerance_degrees: float = 4.0
    vertical_angle_tolerance_degrees: float = 4.0
    endpoint_snap_distance: float = 18.0
    line_merge_tolerance: float = 12.0
    minimum_line_length: float = 30.0
    keep_largest_connected_component: bool = False


def classify_orientation(
    start: Point,
    end: Point,
    config: NormalizationConfig,
) -> Orientation:
    """Classify a line without inventing a missing angle."""
    angle = degrees(atan2(end.y - start.y, end.x - start.x)) % _FULL_TURN
    horizontal_distance = min(angle, _FULL_TURN - angle)
    vertical_distance = abs(angle - 90.0)
    if horizontal_distance <= config.horizontal_angle_tolerance_degrees:
        return Orientation.HORIZONTAL
    if vertical_distance <= config.vertical_angle_tolerance_degrees:
        return Orientation.VERTICAL
    return Orientation.DIAGONAL


def _axis_snap(
    start: Point,
    end: Point,
    orientation: Orientation,
) -> tuple[Point, Point]:
    """Make an axis-aligned line exact while preserving its midpoint."""
    if orientation is Orientation.HORIZONTAL:
        y = (start.y + end.y) / 2.0
        return Point(start.x, y), Point(end.x, y)
    if orientation is Orientation.VERTICAL:
        x = (start.x + end.x) / 2.0
        return Point(x, start.y), Point(x, end.y)
    return start, end


def _candidate_to_line(
    candidate: LineCandidate,
    config: NormalizationConfig,
) -> LineEntity | None:
    """Parse one raw candidate into a normalized line."""
    if candidate.start.distance_to(candidate.end) < config.minimum_line_length:
        return None
    orientation = classify_orientation(candidate.start, candidate.end, config)
    start, end = _axis_snap(candidate.start, candidate.end, orientation)
    return LineEntity(
        entity_id=candidate.entity_id,
        raw_start=candidate.start,
        raw_end=candidate.end,
        start=start,
        end=end,
        orientation=orientation,
        confidence=candidate.confidence,
    )


def _projection(point: Point, origin: Point, unit_x: float, unit_y: float) -> float:
    """Project a point onto a line axis."""
    return (point.x - origin.x) * unit_x + (point.y - origin.y) * unit_y


def _perpendicular_distance(
    point: Point,
    origin: Point,
    unit_x: float,
    unit_y: float,
) -> float:
    """Return the absolute distance from a point to a line axis."""
    return abs((point.x - origin.x) * unit_y - (point.y - origin.y) * unit_x)


def _mergeable(first: LineEntity, second: LineEntity, tolerance: float) -> bool:
    """Check whether two normalized lines describe one physical stroke."""
    if first.orientation is not second.orientation:
        return False
    length = first.length
    if length == 0.0:
        return False
    unit_x = (first.end.x - first.start.x) / length
    unit_y = (first.end.y - first.start.y) / length
    points = (first.start, first.end, second.start, second.end)
    distances = tuple(
        _perpendicular_distance(point, first.start, unit_x, unit_y) for point in points
    )
    if max(distances) > tolerance:
        return False
    projections = tuple(
        _projection(point, first.start, unit_x, unit_y) for point in points
    )
    first_min, first_max = sorted(projections[:2])
    second_min, second_max = sorted(projections[2:])
    gap = max(first_min - second_max, second_min - first_max, 0.0)
    return gap <= tolerance


def _merged_line(first: LineEntity, second: LineEntity) -> LineEntity:
    """Return a stable merged line spanning both inputs."""
    points = (first.start, first.end, second.start, second.end)
    if first.orientation is Orientation.HORIZONTAL:
        y = sum(point.y for point in points) / len(points)
        start = Point(min(point.x for point in points), y)
        end = Point(max(point.x for point in points), y)
    elif first.orientation is Orientation.VERTICAL:
        x = sum(point.x for point in points) / len(points)
        start = Point(x, min(point.y for point in points))
        end = Point(x, max(point.y for point in points))
    else:
        length = first.length
        unit_x = (first.end.x - first.start.x) / length
        unit_y = (first.end.y - first.start.y) / length
        projections = tuple(
            _projection(point, first.start, unit_x, unit_y) for point in points
        )
        start_projection, end_projection = min(projections), max(projections)
        start = Point(
            first.start.x + unit_x * start_projection,
            first.start.y + unit_y * start_projection,
        )
        end = Point(
            first.start.x + unit_x * end_projection,
            first.start.y + unit_y * end_projection,
        )
    return replace(
        first,
        start=start,
        end=end,
        raw_end=second.raw_end,
        confidence=min(first.confidence, second.confidence),
    )


def _merge_collinear(
    lines: list[LineEntity],
    tolerance: float,
) -> list[LineEntity]:
    """Iteratively merge neighboring pieces until stable."""
    merged = list(lines)
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(merged):
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                if not _mergeable(first, second, tolerance):
                    continue
                merged[first_index] = _merged_line(first, second)
                del merged[second_index]
                changed = True
                break
            if changed:
                break
    return merged


def merge_collinear_lines(
    lines: tuple[LineEntity, ...],
    tolerance: float,
) -> tuple[LineEntity, ...]:
    """Merge only the supplied near-collinear line pieces."""
    return tuple(_merge_collinear(list(lines), tolerance))


def _join_point(first: LineEntity, second: LineEntity, first_point: Point) -> Point:
    """Choose a shared endpoint while preserving axis alignment."""
    second_points = (second.start, second.end)
    second_point = min(second_points, key=first_point.distance_to)
    midpoint = Point(
        (first_point.x + second_point.x) / 2.0,
        (first_point.y + second_point.y) / 2.0,
    )
    joined = midpoint
    match first.orientation:
        case Orientation.HORIZONTAL:
            if second.orientation is Orientation.VERTICAL:
                joined = Point(second_point.x, first_point.y)
            else:
                joined = Point(midpoint.x, first_point.y)
        case Orientation.VERTICAL:
            if second.orientation is Orientation.HORIZONTAL:
                joined = Point(first_point.x, second_point.y)
            else:
                joined = Point(first_point.x, midpoint.y)
        case Orientation.DIAGONAL | Orientation.UNKNOWN:
            if second.orientation is Orientation.HORIZONTAL:
                joined = Point(midpoint.x, second_point.y)
            elif second.orientation is Orientation.VERTICAL:
                joined = Point(second_point.x, midpoint.y)
    return joined


def _replace_endpoint(
    line: LineEntity,
    endpoint_index: int,
    point: Point,
) -> LineEntity:
    """Replace one endpoint in a frozen line model."""
    if endpoint_index == 0:
        return replace(line, start=point)
    return replace(line, end=point)


def _snap_nearby_endpoints(
    lines: list[LineEntity],
    distance: float,
) -> list[LineEntity]:
    """Snap the closest endpoint pair for each line pair."""
    snapped = list(lines)
    for first_index in range(len(snapped)):
        for second_index in range(first_index + 1, len(snapped)):
            first_line = snapped[first_index]
            second = snapped[second_index]
            endpoints = (
                (0, first_line.start, 0, second.start),
                (0, first_line.start, 1, second.end),
                (1, first_line.end, 0, second.start),
                (1, first_line.end, 1, second.end),
            )
            pair = min(endpoints, key=lambda item: item[1].distance_to(item[3]))
            if pair[1].distance_to(pair[3]) > distance:
                continue
            joined = _join_point(first_line, second, pair[1])
            snapped[first_index] = _replace_endpoint(first_line, pair[0], joined)
            snapped[second_index] = _replace_endpoint(second, pair[2], joined)
            first_line = snapped[first_index]
    return snapped


def normalize_lines(
    candidates: tuple[LineCandidate, ...],
    config: NormalizationConfig,
) -> tuple[LineEntity, ...]:
    """Normalize raw candidates into stable, connected-ready line entities."""
    lines = [
        line
        for candidate in sorted(candidates, key=lambda item: item.entity_id)
        if (line := _candidate_to_line(candidate, config)) is not None
    ]
    merged = _merge_collinear(lines, config.line_merge_tolerance)
    snapped = _snap_nearby_endpoints(merged, config.endpoint_snap_distance)
    return tuple(sorted(snapped, key=lambda item: item.entity_id))


def build_connections(
    lines: tuple[LineEntity, ...],
    snap_distance: float,
) -> tuple[Connection, ...]:
    """Build stable pairwise connections from nearby line endpoints."""
    connections: list[Connection] = []
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1 :]:
            endpoints = (
                (first.start, second.start),
                (first.start, second.end),
                (first.end, second.start),
                (first.end, second.end),
            )
            first_point, second_point = min(
                endpoints,
                key=lambda pair: pair[0].distance_to(pair[1]),
            )
            if first_point.distance_to(second_point) > snap_distance:
                continue
            connections.append(
                Connection(
                    first_entity_id=first.entity_id,
                    second_entity_id=second.entity_id,
                    point=Point(
                        (first_point.x + second_point.x) / 2.0,
                        (first_point.y + second_point.y) / 2.0,
                    ),
                ),
            )
    return tuple(connections)


def select_largest_connected_component(
    lines: tuple[LineEntity, ...],
    snap_distance: float,
) -> tuple[LineEntity, ...]:
    """Keep the strongest connected stroke group when a photo has clutter."""
    if len(lines) <= 1:
        return lines
    adjacency = [set[int]() for _ in lines]
    for first_index, first in enumerate(lines):
        for second_index in range(first_index + 1, len(lines)):
            second = lines[second_index]
            distances = (
                first.start.distance_to(second.start),
                first.start.distance_to(second.end),
                first.end.distance_to(second.start),
                first.end.distance_to(second.end),
            )
            if min(distances) <= snap_distance:
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)

    components: list[tuple[int, ...]] = []
    unseen = set(range(len(lines)))
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))

    selected = max(
        components,
        key=lambda component: (
            sum(lines[index].length for index in component),
            len(component),
            -min(component),
        ),
    )
    return tuple(
        sorted(
            (lines[index] for index in selected),
            key=lambda line: line.entity_id,
        ),
    )
