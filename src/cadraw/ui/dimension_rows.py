"""Dimension-row construction helpers for the verification panel."""

from cadraw.models import (
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    GeometryEntity,
    LineEntity,
)


def manual_dimensions(
    entities: tuple[GeometryEntity, ...],
) -> tuple[Dimension, ...]:
    """Create empty length rows for detected lines, never inferred values."""
    dimensions: list[Dimension] = []
    for entity in entities:
        if not isinstance(entity, LineEntity):
            continue
        dimensions.append(
            Dimension(
                dimension_id=DimensionId(f"manual_{entity.entity_id}"),
                raw_text="",
                kind=DimensionKind.LENGTH,
                parsed_value=None,
                unit="mm",
                target_entity_ids=(entity.entity_id,),
                confidence=0.0,
                status=DimensionStatus.UNKNOWN,
            ),
        )
    return tuple(dimensions)
