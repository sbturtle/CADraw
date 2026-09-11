"""Pure Pillow renderer for the normalized CAD preview."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import hypot

from PIL import Image, ImageDraw, ImageFont

from cadraw.models import (
    ArcEntity,
    CadDocument,
    CircleEntity,
    DimensionKind,
    EntityId,
    GeometryEntity,
    LineEntity,
    Point,
    PolylineEntity,
)

_BACKGROUND = (255, 255, 255)
_GEOMETRY = (20, 20, 19)
_DIMENSION = (201, 100, 66)
_BORDER = (232, 230, 220)
_SELECTED = (217, 119, 87)


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    """Raster settings for the embedded preview."""

    width: int = 720
    height: int = 520
    padding: int = 32
    line_width: int = 3
    selection_line_width: int = 5
    highlight_entity_ids: tuple[EntityId, ...] = ()
    invert_y: bool = False


class PreviewMode(StrEnum):
    """Coordinate space shown by the interactive preview."""

    DETECTED = "detected"
    CAD = "cad"


def _display_entities(document: CadDocument) -> tuple[GeometryEntity, ...]:
    """Return source entities so every line remains directly selectable."""
    return document.entities


def _entity_points(entity: GeometryEntity) -> tuple[Point, ...]:
    """Return points sufficient to compute a visible entity bound."""
    match entity:
        case LineEntity(start=start, end=end):
            return (start, end)
        case PolylineEntity(points=points):
            return points
        case CircleEntity(center=center, radius=radius):
            return (
                Point(center.x - radius, center.y - radius),
                Point(center.x + radius, center.y + radius),
            )
        case ArcEntity(center=center, radius=radius):
            return (
                Point(center.x - radius, center.y - radius),
                Point(center.x + radius, center.y + radius),
            )


def preview_bounds(
    entities: tuple[GeometryEntity, ...],
) -> tuple[float, float, float, float]:
    """Compute a padded-independent drawing bound."""
    if not entities:
        return (0.0, 0.0, 100.0, 100.0)
    points = tuple(point for entity in entities for point in _entity_points(entity))
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    if min_x == max_x:
        min_x -= 50.0
        max_x += 50.0
    if min_y == max_y:
        min_y -= 50.0
        max_y += 50.0
    return min_x, min_y, max_x, max_y


def _anchor(entity: GeometryEntity) -> Point:
    """Choose a stable label anchor for a geometry entity."""
    match entity:
        case LineEntity(start=start, end=end):
            return Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
        case PolylineEntity(points=points):
            return points[len(points) // 2]
        case CircleEntity(center=center):
            return center
        case ArcEntity(center=center):
            return center


def preview_mapper(
    config: PreviewConfig,
    bounds: tuple[float, float, float, float],
) -> tuple[Callable[[Point], tuple[float, float]], float]:
    """Create a source-to-preview coordinate mapper."""
    min_x, min_y, max_x, max_y = bounds
    usable_width = max(1, config.width - 2 * config.padding)
    usable_height = max(1, config.height - 2 * config.padding)
    scale = min(
        usable_width / (max_x - min_x),
        usable_height / (max_y - min_y),
    )

    def map_point(point: Point) -> tuple[float, float]:
        """Map a local point into the image-oriented preview frame."""
        return (
            config.padding + (point.x - min_x) * scale,
            (
                config.padding + (point.y - min_y) * scale
                if not config.invert_y
                else config.padding + (max_y - point.y) * scale
            ),
        )

    return map_point, scale


def _draw_entity(
    draw: ImageDraw.ImageDraw,
    entity: GeometryEntity,
    map_point: Callable[[Point], tuple[float, float]],
    scale: float,
    config: PreviewConfig,
) -> None:
    """Render one tagged geometry variant."""
    stroke = _SELECTED if entity.entity_id in config.highlight_entity_ids else _GEOMETRY
    line_width = (
        config.selection_line_width
        if entity.entity_id in config.highlight_entity_ids
        else config.line_width
    )
    match entity:
        case LineEntity(start=start, end=end):
            draw.line(
                (map_point(start), map_point(end)),
                fill=stroke,
                width=line_width,
            )
        case PolylineEntity(points=points, closed=closed):
            mapped = tuple(map_point(point) for point in points)
            draw.line(
                (*mapped, mapped[0]) if closed else mapped,
                fill=stroke,
                width=line_width,
                joint="curve",
            )
        case CircleEntity(center=center, radius=radius):
            mapped_center = map_point(center)
            radius_px = radius * scale
            draw.ellipse(
                (
                    mapped_center[0] - radius_px,
                    mapped_center[1] - radius_px,
                    mapped_center[0] + radius_px,
                    mapped_center[1] + radius_px,
                ),
                outline=stroke,
                width=line_width,
            )
        case ArcEntity(
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
        ):
            mapped_center = map_point(center)
            radius_px = radius * scale
            draw.arc(
                (
                    mapped_center[0] - radius_px,
                    mapped_center[1] - radius_px,
                    mapped_center[0] + radius_px,
                    mapped_center[1] + radius_px,
                ),
                start=start_angle,
                end=end_angle,
                fill=stroke,
                width=line_width,
            )


def render_preview(document: CadDocument, config: PreviewConfig) -> Image.Image:
    """Render CAD geometry and associated known dimensions into an image."""
    image = Image.new("RGB", (config.width, config.height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (0, 0, config.width - 1, config.height - 1),
        outline=_BORDER,
        width=1,
    )
    display_entities = _display_entities(document)
    bounds = preview_bounds(display_entities)
    map_point, scale = preview_mapper(config, bounds)
    for entity in display_entities:
        _draw_entity(draw, entity, map_point, scale, config)

    entities_by_id = {entity.entity_id: entity for entity in document.entities}
    font = ImageFont.load_default()
    for dimension in document.dimensions:
        if dimension.parsed_value is None or not dimension.target_entity_ids:
            continue
        target = entities_by_id.get(dimension.target_entity_ids[0])
        if target is None:
            continue
        position = map_point(_anchor(target))
        label = (
            dimension.raw_text
            if dimension.kind is DimensionKind.ANGLE
            else f"{dimension.raw_text} {dimension.unit}"
        )
        draw.text(
            (position[0] + 6, position[1] - 16),
            label,
            fill=_DIMENSION,
            font=font,
        )
    return image


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return the shortest distance from a pixel to a segment."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    clamped = max(0.0, min(1.0, projection))
    nearest = (start[0] + clamped * dx, start[1] + clamped * dy)
    return hypot(point[0] - nearest[0], point[1] - nearest[1])


def hit_test_preview(
    document: CadDocument,
    config: PreviewConfig,
    image_point: tuple[float, float],
) -> EntityId | None:
    """Return the nearest selectable line at a rendered image point."""
    bounds = preview_bounds(_display_entities(document))
    map_point, _ = preview_mapper(config, bounds)
    candidates = tuple(
        entity for entity in document.entities if isinstance(entity, LineEntity)
    )
    nearest = min(
        (
            (
                _point_segment_distance(
                    image_point,
                    map_point(entity.start),
                    map_point(entity.end),
                ),
                entity.entity_id,
            )
            for entity in candidates
        ),
        default=None,
    )
    if nearest is None or nearest[0] > max(12.0, config.selection_line_width + 6.0):
        return None
    return nearest[1]
