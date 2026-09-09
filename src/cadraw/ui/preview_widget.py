"""Qt widget that displays a Pillow-rendered CAD preview."""

from io import BytesIO
from typing import final, override

from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QWidget


@final
class PreviewWidget(QWidget):
    """A fit-to-frame preview with no geometry editing responsibilities."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize an empty preview surface."""
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setMinimumSize(QSize(320, 240))
        self.setAutoFillBackground(True)

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
        self.update()

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the preview centered and scaled without distortion."""
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
        _ = painter.end()
