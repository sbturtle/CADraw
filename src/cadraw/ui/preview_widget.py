"""Qt widget that displays a Pillow-rendered CAD preview."""

from io import BytesIO
from math import cos, radians, sin
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
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from cadraw.models import CadDocument, EntityId, Point
from cadraw.preview import PreviewConfig, hit_test_preview
from cadraw.preview_selection import (
    PreviewSelectionWindow,
    select_preview_entities,
)

_CLICK_TOLERANCE = 4
_VIEW_MARGIN = 0
_MIN_ZOOM = 0.25
_MAX_ZOOM = 4.0
_ZOOM_STEP = 1.25
_ZOOM_EPSILON = 0.001


@final
class PreviewWidget(QWidget):
    """A fit-to-frame preview with no geometry editing responsibilities."""

    selection_changed = Signal(tuple)
    view_changed = Signal()

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
        self._zoom_factor = 1.0
        self._rotation_degrees = 0
        self.setMinimumSize(QSize(160, 120))
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setAutoFillBackground(True)

    @property
    def zoom_factor(self) -> float:
        """Return the current multiplier over the fit-to-frame scale."""
        return self._zoom_factor

    @property
    def rotation_degrees(self) -> int:
        """Return the clockwise preview rotation in quarter turns."""
        return self._rotation_degrees

    @property
    def view_state_label(self) -> str:
        """Return a compact state label for the preview toolbar."""
        return f"{round(self._zoom_factor * 100)}% · {self._rotation_degrees}°"

    def set_zoom_factor(self, factor: float) -> None:
        """Clamp and apply a view-only zoom multiplier."""
        clamped = max(_MIN_ZOOM, min(_MAX_ZOOM, factor))
        if abs(clamped - self._zoom_factor) < _ZOOM_EPSILON:
            return
        self._zoom_factor = clamped
        self.view_changed.emit()
        self.update()

    def zoom_in(self) -> None:
        """Increase the view-only zoom around the preview center."""
        self.set_zoom_factor(self._zoom_factor * _ZOOM_STEP)

    def zoom_out(self) -> None:
        """Decrease the view-only zoom around the preview center."""
        self.set_zoom_factor(self._zoom_factor / _ZOOM_STEP)

    def rotate_left(self) -> None:
        """Rotate the preview counter-clockwise by 90 degrees."""
        self._rotation_degrees = (self._rotation_degrees - 90) % 360
        self.view_changed.emit()
        self.update()

    def rotate_right(self) -> None:
        """Rotate the preview clockwise by 90 degrees."""
        self._rotation_degrees = (self._rotation_degrees + 90) % 360
        self.view_changed.emit()
        self.update()

    def reset_view(self) -> None:
        """Return the preview to a fit-to-frame, upright view."""
        changed = self._zoom_factor != 1.0 or self._rotation_degrees != 0
        self._zoom_factor = 1.0
        self._rotation_degrees = 0
        if changed:
            self.view_changed.emit()
            self.update()

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
        self.reset_view()
        self.update()

    def _fit_scale(self) -> float:
        """Return the scale that fits the rotated image into the widget."""
        pixmap = self._pixmap
        if pixmap is None or pixmap.isNull():
            return 1.0
        angle = radians(self._rotation_degrees)
        width = abs(pixmap.width() * cos(angle)) + abs(pixmap.height() * sin(angle))
        height = abs(pixmap.width() * sin(angle)) + abs(pixmap.height() * cos(angle))
        available_width = max(1, self.width() - 2 * _VIEW_MARGIN)
        available_height = max(1, self.height() - 2 * _VIEW_MARGIN)
        return min(available_width / width, available_height / height)

    def _view_scale(self) -> float:
        """Return the current fit scale multiplied by the user zoom."""
        return self._fit_scale() * self._zoom_factor

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
        scale = self._view_scale()
        if scale <= 0.0:
            return None
        delta_x = (position_x - self.width() / 2.0) / scale
        delta_y = (position_y - self.height() / 2.0) / scale
        angle = radians(self._rotation_degrees)
        source_x = delta_x * cos(angle) + delta_y * sin(angle)
        source_y = -delta_x * sin(angle) + delta_y * cos(angle)
        image_x = source_x + pixmap.width() / 2.0
        image_y = source_y + pixmap.height() / 2.0
        if not (0.0 <= image_x < pixmap.width()):
            return None
        if not (0.0 <= image_y < pixmap.height()):
            return None
        return Point(image_x, image_y)

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
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom the view with the mouse wheel while keeping geometry unchanged."""
        if self._pixmap is None or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint geometry and the active selection rectangle."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        if self._pixmap is not None and not self._pixmap.isNull():
            painter.save()
            painter.translate(self.width() / 2.0, self.height() / 2.0)
            painter.rotate(self._rotation_degrees)
            painter.scale(self._view_scale(), self._view_scale())
            painter.drawPixmap(
                -(self._pixmap.width() // 2),
                -(self._pixmap.height() // 2),
                self._pixmap,
            )
            painter.restore()
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
