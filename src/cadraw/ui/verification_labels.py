"""Human-readable labels for verification rows."""

from cadraw.models import (
    Dimension,
    DimensionStatus,
    DrawingAnalysis,
    LineEntity,
    Orientation,
)

_ORIENTATION_LABELS = {
    Orientation.HORIZONTAL: "수평",
    Orientation.VERTICAL: "수직",
    Orientation.DIAGONAL: "사선",
    Orientation.UNKNOWN: "방향 미확인",
}
_STATUS_LABELS = {
    DimensionStatus.CONFIRMED: "확정",
    DimensionStatus.NEEDS_REVIEW: "검토 필요",
    DimensionStatus.UNKNOWN: "확인 필요",
}


def dimension_label(
    analysis: DrawingAnalysis | None,
    dimension: Dimension,
    row: int,
) -> str:
    """Describe a verification target in workshop language."""
    target_id = dimension.target_entity_ids[0] if dimension.target_entity_ids else None
    if analysis is None or target_id is None:
        return f"항목 {row + 1} · 대상 미확인"
    target = next(
        (entity for entity in analysis.entities if entity.entity_id == target_id),
        None,
    )
    if isinstance(target, LineEntity):
        orientation = _ORIENTATION_LABELS[target.orientation]
        return f"선 {row + 1} · {orientation} · 로컬 {target.length:g}"
    return f"항목 {row + 1} · geometry"


def status_text(dimension: Dimension) -> str:
    """Return a textual review state for the table."""
    return _STATUS_LABELS[dimension.status]


def selected_dimension_label(
    analysis: DrawingAnalysis | None,
    dimension: Dimension,
) -> str:
    """Describe the currently focused dimension without a row ordinal."""
    target_id = dimension.target_entity_ids[0] if dimension.target_entity_ids else None
    if analysis is None or target_id is None:
        return "대상 미확인"
    target = next(
        (entity for entity in analysis.entities if entity.entity_id == target_id),
        None,
    )
    if isinstance(target, LineEntity):
        orientation = _ORIENTATION_LABELS[target.orientation]
        return f"선 · {orientation} · 로컬 {target.length:g}"
    return "geometry"
