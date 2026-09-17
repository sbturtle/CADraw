"""Conservative proximity suggestions linking OCR text to detected geometry."""

from dataclasses import replace

from cadraw.models import (
    ArcEntity,
    CircleEntity,
    Dimension,
    DimensionKind,
    GeometryEntity,
    LineEntity,
    Point,
)

_MAX_SUGGESTION_DISTANCE = 140.0
_MIN_AMBIGUITY_MARGIN_RATIO = 1.35
_SUGGESTED_SOURCE = "ocr_suggested"


def _text_anchor(bounds: tuple[float, float, float, float]) -> Point:
    """Return the center point of an OCR text box in source coordinates."""
    x, y, width, height = bounds
    return Point(x + width / 2.0, y + height / 2.0)


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return the shortest distance from a point to a line segment."""
    segment_x = end.x - start.x
    segment_y = end.y - start.y
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared == 0.0:
        return point.distance_to(start)
    projection = (
        (point.x - start.x) * segment_x + (point.y - start.y) * segment_y
    ) / segment_length_squared
    clamped = max(0.0, min(1.0, projection))
    nearest = Point(
        start.x + clamped * segment_x,
        start.y + clamped * segment_y,
    )
    return point.distance_to(nearest)


def _entity_distance(point: Point, entity: GeometryEntity) -> float | None:
    """Return proximity from a text anchor to one candidate target entity."""
    if isinstance(entity, LineEntity):
        return _point_segment_distance(point, entity.start, entity.end)
    if isinstance(entity, (CircleEntity, ArcEntity)):
        return abs(point.distance_to(entity.center) - entity.radius)
    return None


def _candidate_entities(
    kind: DimensionKind,
    entities: tuple[GeometryEntity, ...],
) -> tuple[GeometryEntity, ...]:
    """Return only the entity families a dimension kind may physically target."""
    if kind in (DimensionKind.LENGTH, DimensionKind.ANGLE):
        return tuple(entity for entity in entities if isinstance(entity, LineEntity))
    if kind is DimensionKind.DIAMETER:
        return tuple(entity for entity in entities if isinstance(entity, CircleEntity))
    if kind is DimensionKind.RADIUS:
        return tuple(
            entity
            for entity in entities
            if isinstance(entity, (CircleEntity, ArcEntity))
        )
    return ()


def suggest_dimension_target(
    dimension: Dimension,
    text_bounds: tuple[float, float, float, float],
    entities: tuple[GeometryEntity, ...],
) -> Dimension:
    """Attach the single unambiguous nearby entity, otherwise leave it untargeted.

    The dimension keeps its confirmation status untouched: this only pre-fills
    the target a person still has to review and confirm, matching the
    documented AI boundary that never invents a connection when it is
    unclear which geometry a recognized value belongs to.
    """
    if dimension.parsed_value is None or dimension.target_entity_ids:
        return dimension
    anchor = _text_anchor(text_bounds)
    ranked = sorted(
        (
            (distance, entity.entity_id)
            for entity in _candidate_entities(dimension.kind, entities)
            if (distance := _entity_distance(anchor, entity)) is not None
        ),
        key=lambda item: item[0],
    )
    if not ranked or ranked[0][0] > _MAX_SUGGESTION_DISTANCE:
        return dimension
    if len(ranked) > 1 and ranked[1][0] <= ranked[0][0] * _MIN_AMBIGUITY_MARGIN_RATIO:
        return dimension
    return replace(
        dimension,
        target_entity_ids=(ranked[0][1],),
        source=_SUGGESTED_SOURCE,
    )
