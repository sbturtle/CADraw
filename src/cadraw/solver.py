"""Deterministic reconstruction of detected lines into CAD coordinates."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import cos, radians, sin

from cadraw.models import (
    ArcEntity,
    CadEntity,
    CadResolutionStatus,
    CircleEntity,
    Constraint,
    ConstraintKind,
    Dimension,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
    GeometryEntity,
    LineEntity,
    Orientation,
    Point,
    PolylineEntity,
)
from cadraw.topology import connected_line_components


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """CAD geometry plus the scale used for preview fallback values."""

    entities: tuple[CadEntity, ...]
    scale_mm_per_source_unit: float | None


def _confirmed(
    dimensions: Iterable[Dimension],
    entity_id: EntityId,
) -> tuple[Dimension, ...]:
    """Return confirmed dimensions attached to one source entity."""
    return tuple(
        dimension
        for dimension in dimensions
        if (
            dimension.status is DimensionStatus.CONFIRMED
            and entity_id in dimension.target_entity_ids
            and dimension.parsed_value is not None
        )
    )


def _value(
    dimensions: Iterable[Dimension],
    entity_id: EntityId,
    kind: DimensionKind,
) -> float | None:
    """Return the first confirmed value of one dimension kind."""
    return next(
        (
            dimension.parsed_value
            for dimension in _confirmed(dimensions, entity_id)
            if dimension.kind is kind and dimension.parsed_value is not None
        ),
        None,
    )


def _source_extent(
    target: GeometryEntity | None,
    kind: DimensionKind,
) -> float | None:
    """Return the detected size a confirmed value of this kind describes."""
    if isinstance(target, LineEntity) and kind is DimensionKind.LENGTH:
        return target.length
    if isinstance(target, PolylineEntity) and target.closed:
        if kind is DimensionKind.WIDTH:
            return target.width
        if kind is DimensionKind.HEIGHT:
            return target.height
    if isinstance(target, CircleEntity) and kind is DimensionKind.DIAMETER:
        return target.radius * 2.0
    if isinstance(target, (CircleEntity, ArcEntity)) and kind is DimensionKind.RADIUS:
        return target.radius
    return None


def _scale_from(
    analysis: DrawingAnalysis,
    kinds: tuple[DimensionKind, ...],
) -> float | None:
    """Derive a scale from the first confirmed value of the given kinds."""
    entities = {entity.entity_id: entity for entity in analysis.entities}
    for dimension in analysis.dimensions:
        if (
            dimension.status is not DimensionStatus.CONFIRMED
            or dimension.kind not in kinds
            or dimension.parsed_value is None
            or len(dimension.target_entity_ids) != 1
        ):
            continue
        extent = _source_extent(
            entities.get(dimension.target_entity_ids[0]),
            dimension.kind,
        )
        if extent is not None and extent > 0.0:
            return dimension.parsed_value / extent
    return None


def _first_scale(analysis: DrawingAnalysis) -> float | None:
    """Find the fallback scale from the first confirmed physical size.

    Line lengths are consulted first so an existing drawing keeps the exact
    scale it had. A sheet whose only confirmable feature is a cut outline or a
    drilled hole could otherwise never be calibrated at all, which left the
    profile and circle rows unable to produce CAD millimetres.
    """
    for kinds in (
        (DimensionKind.LENGTH,),
        (DimensionKind.WIDTH, DimensionKind.HEIGHT),
        (DimensionKind.DIAMETER, DimensionKind.RADIUS),
    ):
        scale = _scale_from(analysis, kinds)
        if scale is not None:
            return scale
    return None


def _line_vector(
    line: LineEntity,
    length: float,
    angle: float | None,
    reverse: bool,
) -> tuple[float, float]:
    """Return one source-oriented segment vector in CAD millimetres."""
    source_length = line.length
    source_x = (line.end.x - line.start.x) / source_length if source_length else 0.0
    source_y = (line.end.y - line.start.y) / source_length if source_length else 0.0
    if line.orientation is Orientation.HORIZONTAL:
        direction = (1.0 if source_x >= 0.0 else -1.0, 0.0)
    elif line.orientation is Orientation.VERTICAL:
        direction = (0.0, -1.0 if source_y >= 0.0 else 1.0)
    elif angle is None:
        direction = (source_x, source_y)
    else:
        theta = radians(angle)
        direction = (
            cos(theta) * (1.0 if source_x >= 0.0 else -1.0),
            sin(theta) * (-1.0 if source_y >= 0.0 else 1.0),
        )
    vector = (length * direction[0], length * direction[1])
    return (-vector[0], -vector[1]) if reverse else vector


def _component_geometry(
    component: tuple[LineEntity, ...],
    dimensions: tuple[Dimension, ...],
    scale: float | None,
    component_index: int,
) -> dict[EntityId, LineEntity]:
    """Propagate CAD positions from the first line's start anchor."""
    root = min(component, key=lambda line: line.entity_id)
    graph: dict[Point, list[tuple[int, Point]]] = {}
    for index, line in enumerate(component):
        graph.setdefault(line.start, []).append((index, line.end))
        graph.setdefault(line.end, []).append((index, line.start))
    positions = {
        root.start: (
            Point(0.0, 0.0) if component_index == 0 else root.start.scaled(scale or 1.0)
        ),
    }
    root_length = _value(dimensions, root.entity_id, DimensionKind.LENGTH)
    root_length = (
        root_length if root_length is not None else root.length * (scale or 1.0)
    )
    root_angle = _value(dimensions, root.entity_id, DimensionKind.ANGLE)
    positions[root.end] = positions[root.start].translated(
        *_line_vector(root, root_length, root_angle, reverse=False),
    )
    queue = deque((root.start, root.end))
    visited = {root.entity_id}
    while queue:
        source_point = queue.popleft()
        for index, opposite_point in graph.get(source_point, ()):
            line = component[index]
            if line.entity_id in visited:
                continue
            length = _value(dimensions, line.entity_id, DimensionKind.LENGTH)
            length = length if length is not None else line.length * (scale or 1.0)
            angle = _value(dimensions, line.entity_id, DimensionKind.ANGLE)
            reverse = line.end == source_point
            positions[opposite_point] = positions[source_point].translated(
                *_line_vector(line, length, angle, reverse),
            )
            visited.add(line.entity_id)
            queue.append(opposite_point)
    return {
        line.entity_id: replace(
            line,
            start=positions.get(line.start, Point(0.0, 0.0)),
            end=positions.get(line.end, Point(0.0, 0.0)),
        )
        for line in component
    }


_CONSTRAINT_KINDS = {
    DimensionKind.LENGTH: ConstraintKind.LENGTH,
    DimensionKind.WIDTH: ConstraintKind.WIDTH,
    DimensionKind.HEIGHT: ConstraintKind.HEIGHT,
    DimensionKind.ANGLE: ConstraintKind.ANGLE,
    DimensionKind.DIAMETER: ConstraintKind.DIAMETER,
    DimensionKind.RADIUS: ConstraintKind.RADIUS,
    DimensionKind.UNKNOWN: ConstraintKind.LENGTH,
}


def _constraints(attached: tuple[Dimension, ...]) -> tuple[Constraint, ...]:
    """Record the confirmed values behind one entity's CAD geometry."""
    return tuple(
        Constraint(
            kind=_CONSTRAINT_KINDS[dimension.kind],
            value=dimension.parsed_value,
            target_entity_ids=dimension.target_entity_ids,
            status=dimension.status,
        )
        for dimension in attached
    )


def _own_factors(
    source: GeometryEntity,
    dimensions: tuple[Dimension, ...],
) -> tuple[float, float] | None:
    """Return the x and y factors this entity's own confirmed sizes imply.

    A sketch carries one overall size per part, not one for the sheet: the
    reference drawings label four brackets `W=500 H=510`, `W=400 H=650` and so
    on. So each outline is sized by its own confirmed extents, and a part whose
    traced aspect ratio differs from the real one still comes out at the width
    and height a person actually wrote down.
    """
    if isinstance(source, PolylineEntity) and source.closed:
        width = _value(dimensions, source.entity_id, DimensionKind.WIDTH)
        height = _value(dimensions, source.entity_id, DimensionKind.HEIGHT)
        x_factor = width / source.width if width and source.width > 0.0 else None
        y_factor = height / source.height if height and source.height > 0.0 else None
        if x_factor is None and y_factor is None:
            return None
        return (x_factor or y_factor or 1.0, y_factor or x_factor or 1.0)
    if isinstance(source, (CircleEntity, ArcEntity)) and source.radius > 0.0:
        diameter = (
            _value(dimensions, source.entity_id, DimensionKind.DIAMETER)
            if isinstance(source, CircleEntity)
            else None
        )
        radius = _value(dimensions, source.entity_id, DimensionKind.RADIUS)
        confirmed_radius = (diameter / 2.0) if diameter else radius
        if confirmed_radius:
            factor = confirmed_radius / source.radius
            return (factor, factor)
    return None


def _sized_non_line(
    source: GeometryEntity,
    factors: tuple[float, float],
    scale: float | None,
) -> GeometryEntity:
    """Resize one entity by its own factors, keeping the sheet's layout."""
    layout = scale if scale is not None else 1.0
    if isinstance(source, PolylineEntity):
        origin_x = min(point.x for point in source.points)
        origin_y = min(point.y for point in source.points)
        return replace(
            source,
            points=tuple(
                Point(
                    origin_x * layout + (point.x - origin_x) * factors[0],
                    origin_y * layout + (point.y - origin_y) * factors[1],
                )
                for point in source.points
            ),
        )
    if isinstance(source, (CircleEntity, ArcEntity)):
        return replace(
            source,
            center=source.center.scaled(layout),
            radius=source.radius * factors[0],
        )
    return source


def _non_line_cad_entity(
    source: GeometryEntity,
    dimensions: tuple[Dimension, ...],
    scale: float | None,
) -> CadEntity:
    """Resolve one closed shape from its own confirmed size, or leave it open.

    A global scale taken from somewhere else on the sheet is a preview aid, not
    a measurement of this shape. Treating it as one used to mark every detected
    hole and outline `RESOLVED` the moment any one value was confirmed, which
    exported sizes that came from the photograph's proportions rather than from
    a person.
    """
    factors = _own_factors(source, dimensions)
    resolved = factors is not None
    geometry = (
        _sized_non_line(source, factors, scale)
        if factors is not None
        else _scaled_for_preview(source, scale)
    )
    return CadEntity(
        entity_id=source.entity_id,
        source_entity_id=source.entity_id,
        geometry=geometry,
        constraints=_constraints(_confirmed(dimensions, source.entity_id)),
        status=(
            CadResolutionStatus.RESOLVED if resolved else CadResolutionStatus.UNRESOLVED
        ),
        inferred_for_preview=not resolved,
    )


def _scaled_for_preview(
    source: GeometryEntity,
    scale: float | None,
) -> GeometryEntity:
    """Place unconfirmed geometry for preview only, never for export."""
    if scale is None:
        return source
    return _sized_non_line(source, (scale, scale), scale)


def _cad_entity(
    source: GeometryEntity,
    geometry: GeometryEntity,
    dimensions: tuple[Dimension, ...],
    scale: float | None,
) -> CadEntity:
    """Build one source-to-CAD relation and resolution state."""
    if not isinstance(source, LineEntity):
        return _non_line_cad_entity(source, dimensions, scale)
    has_length = _value(dimensions, source.entity_id, DimensionKind.LENGTH) is not None
    has_angle = _value(dimensions, source.entity_id, DimensionKind.ANGLE) is not None
    status = (
        CadResolutionStatus.RESOLVED if has_length else CadResolutionStatus.UNRESOLVED
    )
    if source.orientation is Orientation.DIAGONAL and has_length and not has_angle:
        status = CadResolutionStatus.UNKNOWN_ANGLE
    return CadEntity(
        entity_id=source.entity_id,
        source_entity_id=source.entity_id,
        geometry=geometry,
        constraints=_constraints(_confirmed(dimensions, source.entity_id)),
        status=status,
        inferred_for_preview=status is not CadResolutionStatus.RESOLVED,
    )


def reconstruct_geometry(analysis: DrawingAnalysis) -> ReconstructionResult:
    """Reconstruct all detected geometry using explicit dimensions first."""
    scale = _first_scale(analysis)
    lines = tuple(
        entity for entity in analysis.entities if isinstance(entity, LineEntity)
    )
    if scale is None:
        return ReconstructionResult(
            entities=tuple(
                _cad_entity(entity, entity, analysis.dimensions, scale)
                for entity in analysis.entities
            ),
            scale_mm_per_source_unit=None,
        )
    reconstructed: dict[EntityId, LineEntity] = {}
    for index, component in enumerate(connected_line_components(lines)):
        reconstructed.update(
            _component_geometry(
                component,
                analysis.dimensions,
                scale,
                index,
            ),
        )
    return ReconstructionResult(
        entities=tuple(
            _cad_entity(
                source,
                reconstructed.get(source.entity_id, source),
                analysis.dimensions,
                scale,
            )
            for source in analysis.entities
        ),
        scale_mm_per_source_unit=scale,
    )
