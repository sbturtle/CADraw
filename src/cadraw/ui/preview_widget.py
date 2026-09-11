"""Qt widget that displays a Pillow-rendered CAD preview."""

from io import BytesIO
from typing import final, override

from PIL import Image
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from cadraw.models import CadDocument, EntityId, Point
from cadraw.preview import PreviewConfig, hit_test_preview
from cadraw.preview_selection import (
    PreviewSelectionWindow,
    select_preview_entities,
)

_CLICK_TOLERANCE = 4


@final
class PreviewWidget(QWidget):
    """A fit-to-frame preview with no geometry editing responsibilities."""

    selection_changed = Signal(tuple)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize an empty preview surface."""
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._document: CadDocument | None = None
        self._preview_config = PreviewConfig()
        self._selected_entity_ids: tuple[EntityId, ...] = ()
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._drag_modifiers = Qt.KeyboardModifier.NoModifier
        self.setMinimumSize(QSize(160, 120))
        self.setAutoFillBackground(True)

    def set_interaction_context(
        self,
        document: CadDocument,
        config: PreviewConfig,
    ) -> None:
        """Store the CAD document used for click-to-entity hit testing."""
        self._document = document
        self._preview_config = config
        self._selected_entity_ids = config.highlight_entity_ids

    def set_image(self, image: Image.Image) -> None:
        """Replace the displayed preview image."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        qimage = QImage.fromData(buffer.getvalue())
        self._pixmap = QPixmap.fromImage(qimage)
        self.update()

    def clear_image(self) -> None:
        """Remove the current preview."""
        self._pixmap = None
        self._document = None
        self._selected_entity_ids = ()
        self.update()

    def _ordered_entity_ids(
        self,
        entity_ids: set[EntityId],
    ) -> tuple[EntityId, ...]:
        """Return selected IDs in the document's stable drawing order."""
        if self._document is None:
            return ()
        return tuple(
            entity.entity_id
            for entity in self._document.entities
            if entity.entity_id in entity_ids
        )

    def _emit_selection(self, entity_ids: tuple[EntityId, ...]) -> None:
        """Publish a complete preview selection to the workbench."""
        self._selected_entity_ids = entity_ids
        self.selection_changed.emit(tuple(str(entity_id) for entity_id in entity_ids))

    def _image_point(self, position_x: int, position_y: int) -> Point | None:
        """Convert a widget click into the fixed preview image coordinates."""
        pixmap = self._pixmap
        if pixmap is None or pixmap.isNull():
            return None
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        left = (self.width() - scaled.width()) // 2
        top = (self.height() - scaled.height()) // 2
        if not (left <= position_x < left + scaled.width()):
            return None
        if not (top <= position_y < top + scaled.height()):
            return None
        return Point(
            (position_x - left) * pixmap.width() / scaled.width(),
            (position_y - top) * pixmap.height() / scaled.height(),
        )

    def _select_click(
        self,
        position: QPoint,
        modifiers: Qt.KeyboardModifier,
    ) -> None:
        """Apply single-click or Ctrl-click toggle selection."""
        if self._document is None:
            return
        image_point = self._image_point(position.x(), position.y())
        additive = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        entity_id = (
            hit_test_preview(
                self._document,
                self._preview_config,
                (image_point.x, image_point.y),
            )
            if image_point is not None
            else None
        )
        if entity_id is None:
            if not additive:
                self._emit_selection(())
            return
        if not additive:
            self._emit_selection((entity_id,))
            return
        selected = set(self._selected_entity_ids)
        if entity_id in selected:
            selected.remove(entity_id)
        else:
            selected.add(entity_id)
        self._emit_selection(self._ordered_entity_ids(selected))

    def _finish_drag(self) -> None:
        """Convert a drag gesture into a window or crossing selection."""
        start = self._drag_start
        end = self._drag_current
        modifiers = self._drag_modifiers
        self._drag_start = None
        self._drag_current = None
        self.update()
        if start is None or end is None or self._document is None:
            return
        if (
            abs(end.x() - start.x()) <= _CLICK_TOLERANCE
            and abs(end.y() - start.y()) <= _CLICK_TOLERANCE
        ):
            self._select_click(start, modifiers)
            return
        start_image = self._image_point(start.x(), start.y())
        end_image = self._image_point(end.x(), end.y())
        if start_image is None or end_image is None:
            return
        selected = select_preview_entities(
            self._document,
            self._preview_config,
            PreviewSelectionWindow(
                start=start_image,
                end=end_image,
                left_to_right=end.x() >= start.x(),
            ),
        )
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            selected = self._ordered_entity_ids(
                set(self._selected_entity_ids).union(selected),
            )
        self._emit_selection(selected)

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start a click or directional selection-window gesture."""
        if event.button() is Qt.MouseButton.LeftButton and self._document is not None:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            self._drag_modifiers = event.modifiers()
            event.accept()
            return
        super().mousePressEvent(event)

    @override
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track the visible selection rectangle while dragging."""
        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    @override
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Commit a click or selection rectangle on left-button release."""
        if event.button() is Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self._finish_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint geometry and the active selection rectangle."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        if self._drag_start is not None and self._drag_current is not None:
            painter.setPen(
                QPen(
                    QColor("#C96442"),
                    1,
                    Qt.PenStyle.DashLine,
                ),
            )
            painter.setBrush(QColor(201, 100, 66, 32))
            painter.drawRect(
                QRect(self._drag_start, self._drag_current).normalized(),
            )
        _ = painter.end()
