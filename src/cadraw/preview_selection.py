"""Pure selection geometry for the interactive CAD preview."""

from dataclasses import dataclass

from cadraw.models import CadDocument, EntityId, LineEntity, Point
from cadraw.preview import PreviewConfig, preview_bounds, preview_mapper

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class PreviewSelectionWindow:
    """One drag rectangle in rendered image coordinates."""

    start: Point
    end: Point
    left_to_right: bool


def _point_in_rectangle(
    point: Point,
    rectangle: tuple[float, float, float, float],
) -> bool:
    """Return whether a point is inside or on a selection rectangle."""
    min_x, min_y, max_x, max_y = rectangle
    return min_x <= point.x <= max_x and min_y <= point.y <= max_y


def _cross(first: Point, second: Point, third: Point) -> float:
    """Return the signed cross product of two vectors."""
    return (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (
        third.x - first.x
    )


def _on_segment(
    first: Point,
    second: Point,
    point: Point,
) -> bool:
    """Return whether a collinear point lies on the segment."""
    return (
        min(first.x, second.x) - _EPSILON
        <= point.x
        <= max(first.x, second.x) + _EPSILON
        and min(first.y, second.y) - _EPSILON
        <= point.y
        <= max(first.y, second.y) + _EPSILON
    )


def _segments_intersect(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> bool:
    """Return whether two closed line segments touch or cross."""
    first_orientation = _cross(first_start, first_end, second_start)
    second_orientation = _cross(first_start, first_end, second_end)
    third_orientation = _cross(second_start, second_end, first_start)
    fourth_orientation = _cross(second_start, second_end, first_end)
    if (
        (first_orientation > _EPSILON and second_orientation < -_EPSILON)
        or (first_orientation < -_EPSILON and second_orientation > _EPSILON)
    ) and (
        (third_orientation > _EPSILON and fourth_orientation < -_EPSILON)
        or (third_orientation < -_EPSILON and fourth_orientation > _EPSILON)
    ):
        return True
    return (
        (
            abs(first_orientation) <= _EPSILON
            and _on_segment(first_start, first_end, second_start)
        )
        or (
            abs(second_orientation) <= _EPSILON
            and _on_segment(first_start, first_end, second_end)
        )
        or (
            abs(third_orientation) <= _EPSILON
            and _on_segment(second_start, second_end, first_start)
        )
        or (
            abs(fourth_orientation) <= _EPSILON
            and _on_segment(second_start, second_end, first_end)
        )
    )


def _segment_intersects_rectangle(
    start: Point,
    end: Point,
    rectangle: tuple[float, float, float, float],
) -> bool:
    """Return whether a line segment touches a rectangle."""
    if _point_in_rectangle(start, rectangle) or _point_in_rectangle(
        end,
        rectangle,
    ):
        return True
    min_x, min_y, max_x, max_y = rectangle
    corners = (
        Point(min_x, min_y),
        Point(max_x, min_y),
        Point(max_x, max_y),
        Point(min_x, max_y),
    )
    return any(
        _segments_intersect(start, end, edge_start, edge_end)
        for edge_start, edge_end in zip(
            corners,
            (*corners[1:], corners[0]),
            strict=True,
        )
    )


def select_preview_entities(
    document: CadDocument,
    config: PreviewConfig,
    selection: PreviewSelectionWindow,
) -> tuple[EntityId, ...]:
    """Select lines by CAD window or crossing-window semantics."""
    bounds = preview_bounds(document.entities)
    map_point, _ = preview_mapper(config, bounds)
    rectangle = (
        min(selection.start.x, selection.end.x),
        min(selection.start.y, selection.end.y),
        max(selection.start.x, selection.end.x),
        max(selection.start.y, selection.end.y),
    )
    selected: list[EntityId] = []
    for entity in document.entities:
        if not isinstance(entity, LineEntity):
            continue
        start = map_point(entity.start)
        end = map_point(entity.end)
        start_point = Point(*start)
        end_point = Point(*end)
        is_selected = (
            _point_in_rectangle(start_point, rectangle)
            and _point_in_rectangle(end_point, rectangle)
            if selection.left_to_right
            else _segment_intersects_rectangle(start_point, end_point, rectangle)
        )
        if is_selected:
            selected.append(entity.entity_id)
    return tuple(selected)
