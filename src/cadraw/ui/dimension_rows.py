"""Dimension-row construction helpers for the verification panel."""

from cadraw.models import (
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    GeometryEntity,
    LineEntity,
    Orientation,
)


def manual_dimensions(
    entities: tuple[GeometryEntity, ...],
    existing_dimensions: tuple[Dimension, ...] = (),
) -> tuple[Dimension, ...]:
    """Create stable length/angle rows without discarding confirmed values."""
    existing_by_key = {
        (dimension.target_entity_ids[0], dimension.kind): dimension
        for dimension in existing_dimensions
        if len(dimension.target_entity_ids) == 1
    }
    dimensions: list[Dimension] = []
    for entity in entities:
        if not isinstance(entity, LineEntity):
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
