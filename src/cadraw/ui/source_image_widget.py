"""Interactive source-image surface for selection and manual line drawing."""

from enum import StrEnum
from typing import Final, final, override

from PySide6.QtCore import QLineF, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPen, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QWidget

from cadraw.models import Point

_SELECTION_COLOR: Final = "#D97757"
_SELECTION_WIDTH: Final = 6
_REGION_COLOR: Final = "#1468C8"
_REGION_WIDTH: Final = 3

type SourceLine = tuple[Point, Point]
type SourceRegion = tuple[float, float, float, float]


class SourceDragMode(StrEnum):
    """What a drag on the source image is currently for."""

    NONE = "none"
    LINE = "line"
    REGION = "region"


@final
class SourceImageLabel(QLabel):
    """Fit a source image and emit source-space drag lines when enabled."""

    line_drawn = Signal(float, float, float, float)
    region_drawn = Signal(float, float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize a passive source image surface."""
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._selection_lines: tuple[SourceLine, ...] = ()
        self._region: SourceRegion | None = None
        self._drag_mode = SourceDragMode.NONE
        self._draw_start: Point | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(420, 220)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        """Replace the source image while preserving its source coordinates."""
        self._source_pixmap = pixmap
        self.setText("" if not pixmap.isNull() else "손도면 사진을 불러오세요.")
        self._refresh_display()

    def set_selection_lines(self, lines: tuple[SourceLine, ...]) -> None:
        """Replace the source-space selection overlay."""
        self._selection_lines = lines
        self._refresh_display()

    def set_region(self, region: SourceRegion | None) -> None:
        """Replace the drawing region a reviewer marked, in source coordinates."""
        self._region = region
        self._refresh_display()

    def set_drag_mode(self, mode: SourceDragMode) -> None:
        """Choose what the next drag on the source image produces.

        One mode at a time, so marking a region cannot also redraw a line.
        """
        self._drag_mode = mode
        self._draw_start = None
        self.setCursor(
            Qt.CursorShape.ArrowCursor
            if mode is SourceDragMode.NONE
            else Qt.CursorShape.CrossCursor,
        )

    def set_drawing_enabled(self, enabled: bool) -> None:
        """Toggle source drag input for manual line replacement or addition."""
        self.set_drag_mode(SourceDragMode.LINE if enabled else SourceDragMode.NONE)

    def _refresh_display(self) -> None:
        """Fit the image and paint the current selection overlay."""
        source_pixmap = self._source_pixmap
        if source_pixmap is None or source_pixmap.isNull():
            self.setPixmap(QPixmap())
            self.update()
            return
        scaled = source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        source_width = source_pixmap.width()
        source_height = source_pixmap.height()
        if source_width <= 0 or source_height <= 0:
            self.setPixmap(scaled)
            return
        x_scale = scaled.width() / source_width
        y_scale = scaled.height() / source_height
        painter = QPainter(scaled)
        _ = painter.setRenderHint(QPainter.RenderHint.Antialiasing, on=False)
        pen = QPen(_SELECTION_COLOR)
        _ = pen.setWidth(_SELECTION_WIDTH)
        _ = pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        _ = painter.setPen(pen)
        for start, end in self._selection_lines:
            _ = painter.drawLine(
                QLineF(
                    start.x * x_scale,
                    start.y * y_scale,
                    end.x * x_scale,
                    end.y * y_scale,
                ),
            )
        region = self._region
        if region is not None:
            region_pen = QPen(_REGION_COLOR)
            _ = region_pen.setWidth(_REGION_WIDTH)
            _ = painter.setPen(region_pen)
            _ = painter.drawRect(
                QRectF(
                    region[0] * x_scale,
                    region[1] * y_scale,
                    region[2] * x_scale,
                    region[3] * y_scale,
                ),
            )
        _ = painter.end()
        self.setPixmap(scaled)

    def _display_rect(self) -> QRect:
        """Return the centered fitted image rectangle in widget coordinates."""
        pixmap = self.pixmap()
        if pixmap.isNull():
            return QRect()
        return QRect(
            (self.width() - pixmap.width()) // 2,
            (self.height() - pixmap.height()) // 2,
            pixmap.width(),
            pixmap.height(),
        )

    def _source_point(self, position: QPoint) -> Point | None:
        """Convert a widget point into the original image coordinate system."""
        source_pixmap = self._source_pixmap
        display_rect = self._display_rect()
        if (
            source_pixmap is None
            or source_pixmap.isNull()
            or not display_rect.contains(position)
        ):
            return None
        return Point(
            (position.x() - display_rect.left())
            * source_pixmap.width()
            / display_rect.width(),
            (position.y() - display_rect.top())
            * source_pixmap.height()
            / display_rect.height(),
        )

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start a source-space drag when a drag mode is active."""
        if (
            self._drag_mode is SourceDragMode.NONE
            or event.button() is not Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return
        self._draw_start = self._source_point(event.position().toPoint())
        event.accept()

    @override
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish a source-space drag and emit it in original coordinates."""
        if (
            self._drag_mode is SourceDragMode.NONE
            or event.button() is not Qt.MouseButton.LeftButton
        ):
            super().mouseReleaseEvent(event)
            return
        start = self._draw_start
        self._draw_start = None
        end = self._source_point(event.position().toPoint())
        if start is not None and end is not None:
            self._emit_drag(start, end)
        event.accept()

    def _emit_drag(self, start: Point, end: Point) -> None:
        """Emit one completed drag according to the active mode."""
        if self._drag_mode is SourceDragMode.REGION:
            self.region_drawn.emit(
                min(start.x, end.x),
                min(start.y, end.y),
                abs(end.x - start.x),
                abs(end.y - start.y),
            )
            return
        self.line_drawn.emit(start.x, start.y, end.x, end.y)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the fitted pixmap and its overlay aligned after resizing."""
        super().resizeEvent(event)
        self._refresh_display()
