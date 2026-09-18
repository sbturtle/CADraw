"""Dimension-row construction helpers for the verification panel."""

from cadraw.models import (
    ArcEntity,
    CircleEntity,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    EntityId,
    GeometryEntity,
    Orientation,
    PolylineEntity,
)


def _circular_dimension(
    entity: CircleEntity | ArcEntity,
    kind: DimensionKind,
    existing_by_key: dict[tuple[EntityId, DimensionKind], Dimension],
) -> Dimension:
    """Return a stable diameter/radius row for one circle or arc entity."""
    suffix = "diameter" if kind is DimensionKind.DIAMETER else "radius"
    return existing_by_key.get(
        (entity.entity_id, kind),
        Dimension(
            dimension_id=DimensionId(f"manual_{entity.entity_id}_{suffix}"),
            raw_text="",
            kind=kind,
            parsed_value=None,
            unit="mm",
            target_entity_ids=(entity.entity_id,),
            confidence=0.0,
            status=DimensionStatus.UNSET,
        ),
    )


def _profile_dimension(
    entity: PolylineEntity,
    kind: DimensionKind,
    existing_by_key: dict[tuple[EntityId, DimensionKind], Dimension],
) -> Dimension:
    """Return a stable overall-extent row for one closed profile outline."""
    suffix = "width" if kind is DimensionKind.WIDTH else "height"
    return existing_by_key.get(
        (entity.entity_id, kind),
        Dimension(
            dimension_id=DimensionId(f"manual_{entity.entity_id}_{suffix}"),
            raw_text="",
            kind=kind,
            parsed_value=None,
            unit="mm",
            target_entity_ids=(entity.entity_id,),
            confidence=0.0,
            status=DimensionStatus.UNSET,
        ),
    )


def manual_dimensions(
    entities: tuple[GeometryEntity, ...],
    existing_dimensions: tuple[Dimension, ...] = (),
) -> tuple[Dimension, ...]:
    """Create stable length/angle/diameter/radius rows for every entity."""
    existing_by_key = {
        (dimension.target_entity_ids[0], dimension.kind): dimension
        for dimension in existing_dimensions
        if len(dimension.target_entity_ids) == 1
    }
    dimensions: list[Dimension] = []
    for entity in entities:
        if isinstance(entity, CircleEntity):
            dimensions.append(
                _circular_dimension(
                    entity,
                    DimensionKind.DIAMETER,
                    existing_by_key,
                ),
            )
            continue
        if isinstance(entity, ArcEntity):
            dimensions.append(
                _circular_dimension(
                    entity,
                    DimensionKind.RADIUS,
                    existing_by_key,
                ),
            )
            continue
        if isinstance(entity, PolylineEntity):
            # A closed outline is dimensioned by its overall extents. An open
            # chain is assembled from lines that already carry their own rows.
            if entity.closed:
                dimensions.extend(
                    _profile_dimension(entity, kind, existing_by_key)
                    for kind in (DimensionKind.WIDTH, DimensionKind.HEIGHT)
                )
            continue
        length_key = (entity.entity_id, DimensionKind.LENGTH)
        dimensions.append(
            existing_by_key.get(
                length_key,
                Dimension(
                    dimension_id=DimensionId(
                        f"manual_{entity.entity_id}_length",
                    ),
                    raw_text="",
                    kind=DimensionKind.LENGTH,
                    parsed_value=None,
                    unit="mm",
                    target_entity_ids=(entity.entity_id,),
                    confidence=0.0,
                    status=DimensionStatus.UNSET,
                ),
            ),
        )
        axis_angle = {
            Orientation.HORIZONTAL: 0.0,
            Orientation.VERTICAL: 90.0,
        }.get(entity.orientation)
        angle_key = (entity.entity_id, DimensionKind.ANGLE)
        existing_angle = existing_by_key.get(angle_key)
        if existing_angle is not None and existing_angle.source == "axis_inference":
            existing_angle = None
        angle = (
            Dimension(
                dimension_id=DimensionId(f"manual_{entity.entity_id}_angle"),
                raw_text=f"{axis_angle:g}°",
                kind=DimensionKind.ANGLE,
                parsed_value=axis_angle,
                unit="deg",
                target_entity_ids=(entity.entity_id,),
                confidence=1.0,
                status=DimensionStatus.CONFIRMED,
                source="axis_inference",
            )
            if axis_angle is not None
            else existing_angle
            or Dimension(
                dimension_id=DimensionId(f"manual_{entity.entity_id}_angle"),
                raw_text="",
                kind=DimensionKind.ANGLE,
                parsed_value=None,
                unit="deg",
                target_entity_ids=(entity.entity_id,),
                confidence=0.0,
                status=DimensionStatus.UNSET,
            )
        )
        dimensions.append(angle)
    return tuple(dimensions)
