"""Source-image rendering bridge for the workbench shell."""

from io import BytesIO

from PIL import Image
from PySide6.QtGui import QImage, QPixmap

from cadraw.models import DrawingAnalysis, LineEntity, Point
from cadraw.preview import render_foreground_overlay
from cadraw.ui.workbench_view import WorkbenchView


def render_foreground_pixmap(analysis: DrawingAnalysis) -> QPixmap:
    """Convert the transparent geometry overlay into a readable Qt pixmap."""
    overlay = render_foreground_overlay(analysis)
    background = Image.new("RGBA", overlay.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, overlay).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return QPixmap.fromImage(QImage.fromData(buffer.getvalue()))


def pixmap_from_png(image_png: bytes) -> QPixmap:
    """Decode an encoded preprocessing preview without changing its pixels."""
    return QPixmap.fromImage(QImage.fromData(image_png))


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
