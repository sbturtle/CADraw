"""Source-image rendering bridge for the workbench shell."""

from PySide6.QtGui import QPixmap

from cadraw.models import DrawingAnalysis, LineEntity, Point
from cadraw.ui.workbench_view import WorkbenchView


def refresh_source_view(
    view: WorkbenchView,
    pixmap: QPixmap | None,
    analysis: DrawingAnalysis | None,
) -> None:
    """Render the source image and all selected raw line overlays."""
    if pixmap is None or pixmap.isNull():
        return
    selected_entity_ids = set(view.selected_entity_ids)
    selection_lines: tuple[tuple[Point, Point], ...] = ()
    if analysis is not None:
        selection_lines = tuple(
            (entity.raw_start, entity.raw_end)
            for entity in analysis.entities
            if isinstance(entity, LineEntity)
            and entity.entity_id in selected_entity_ids
        )
    view.set_source_pixmap(pixmap, selection_lines)
