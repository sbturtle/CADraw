"""Human-readable labels for verification rows."""

from cadraw.models import (
    CadDocument,
    CadResolutionStatus,
    Dimension,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
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
    DimensionStatus.UNSET: "미입력",
    DimensionStatus.CONFIRMED: "확정",
    DimensionStatus.NEEDS_REVIEW: "검토 필요",
    DimensionStatus.UNKNOWN: "확인 필요",
    DimensionStatus.AI_DETECTED: "AI 제안",
    DimensionStatus.UNCERTAIN: "불확실",
    DimensionStatus.INVALID: "입력 오류",
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
        if dimension.kind is DimensionKind.ANGLE:
            return f"선 {row + 1} · {orientation} · 각도"
        return f"선 {row + 1} · {orientation} · 로컬 {target.length:g}"
    return f"항목 {row + 1} · geometry"


def status_text(dimension: Dimension) -> str:
    """Return a textual review state for the table."""
    if dimension.source == "axis_inference":
        return "자동 확정"
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
        if dimension.kind is DimensionKind.ANGLE:
            return f"선 · {orientation} · 각도"
        return f"선 · {orientation} · 로컬 {target.length:g}"
    return "geometry"


def selected_entity_info(
    analysis: DrawingAnalysis,
    document: CadDocument,
    entity_id: EntityId,
) -> str:
    """Describe a selected source entity and its CAD resolution state."""
    source = next(
        (entity for entity in analysis.entities if entity.entity_id == entity_id),
        None,
    )
    cad_entity = next(
        (entity for entity in document.cad_entities if entity.entity_id == entity_id),
        None,
    )
    if not isinstance(source, LineEntity) or cad_entity is None:
        return f"선택된 entity · ID {entity_id}"
    cad_geometry = cad_entity.geometry
    actual_length = (
        cad_geometry.length
        if isinstance(cad_geometry, LineEntity)
        and cad_entity.status is CadResolutionStatus.RESOLVED
        else None
    )
    actual_text = f"{actual_length:g} mm" if actual_length is not None else "미확정"
    return "\n".join(
        (
            f"종류: {_ORIENTATION_LABELS[source.orientation]} Line",
            f"ID: {entity_id}",
            f"상태: {cad_entity.status.value}",
            f"감지 pixel 길이: {source.length:g} px",
            f"실제 길이: {actual_text}",
        ),
    )
