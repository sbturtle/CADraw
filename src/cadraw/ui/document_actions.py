"""Document-level calculation and export captions for the UI shell."""

from collections.abc import Callable
from pathlib import Path

from cadraw.export import export_dxf, validate_dxf
from cadraw.models import AnalysisStatus, CadDocument

type CaptionSetter = Callable[[str], None]

_BLOCKED_STATUSES = frozenset(
    {
        AnalysisStatus.NO_GEOMETRY_DETECTED,
        AnalysisStatus.VERY_LOW_CONFIDENCE,
        AnalysisStatus.TOO_MUCH_NOISE,
        AnalysisStatus.DOCUMENT_NOT_FOUND,
        AnalysisStatus.PREPROCESSING_UNCERTAIN,
    },
)


def calculation_caption(document: CadDocument) -> str:
    """Return the status message for a recalculated document."""
    if document.scale_mm_per_source_unit is None:
        return "스케일 미확인: 치수를 하나 이상 확정하면 mm로 다시 계산합니다."
    geometry_text = f"{len(document.entities)}개 geometry"
    warning_text = f" · 경고 {len(document.warnings)}개" if document.warnings else ""
    return f"계산 완료 · {geometry_text}{warning_text}"


def save_document(
    document: CadDocument | None,
    output_path: Path,
    set_caption: CaptionSetter,
) -> bool:
    """Export and validate a document while updating the UI caption."""
    if document is None:
        set_caption("먼저 손도면 사진을 불러오세요.")
        return False
    if document.analysis_status in _BLOCKED_STATUSES:
        status_label = f"({document.analysis_status.value})"
        caption = "DXF 저장 보류: 분석 상태를 확인해 주세요 "
        set_caption(caption + status_label + ".")
        return False
    if not document.entities:
        set_caption("DXF 저장 보류: 도면 geometry가 없습니다.")
        return False
    try:
        summary = export_dxf(document, output_path)
        validation = validate_dxf(output_path)
    except OSError as error:
        set_caption(f"DXF 저장 실패: {error}")
        return False
    geometry_text = f"geometry {summary.geometry_entity_count}개"
    entity_text = f"entity {validation.total_entities}개"
    omitted_text = (
        f" · 미확정 geometry {summary.omitted_unresolved_count}개 제외"
        if summary.omitted_unresolved_count
        else ""
    )
    set_caption(f"DXF 저장 완료 · {geometry_text} · {entity_text}{omitted_text}")
    return True
