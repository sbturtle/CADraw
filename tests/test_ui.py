from pathlib import Path
from typing import final

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QTableWidget,
)

from cadraw.models import (
    DrawingAnalysis,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)
from cadraw.normalization import build_connections
from cadraw.ui.main_window import MainWindow


def _application() -> QApplication:
    """Return the test process QApplication."""
    application = QApplication.instance()
    if isinstance(application, QApplication):
        return application
    return QApplication([])


def _line(entity_id: str, start: Point, end: Point) -> LineEntity:
    """Build a deterministic line fixture for UI edit actions."""
    orientation = Orientation.HORIZONTAL if start.y == end.y else Orientation.VERTICAL
    return LineEntity(
        entity_id=EntityId(entity_id),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=orientation,
        confidence=0.9,
    )


@final
class _StaticAnalyzer:
    """Return a fixed analysis so edit tests do not depend on OpenCV ordering."""

    _analysis: DrawingAnalysis

    def __init__(self, analysis: DrawingAnalysis) -> None:
        """Store the deterministic analysis fixture."""
        self._analysis = analysis

    def analyze(self, image_path: Path) -> DrawingAnalysis:
        """Return the fixture regardless of the source path."""
        _ = image_path
        return self._analysis


def _two_line_analysis() -> DrawingAnalysis:
    """Build two unresolved lines that can exercise all three actions."""
    lines = (
        _line("line_a", Point(40.0, 50.0), Point(140.0, 50.0)),
        _line("line_b", Point(144.0, 50.0), Point(240.0, 50.0)),
    )
    return DrawingAnalysis(
        unit="mm",
        image_size=(320, 240),
        entities=lines,
        dimensions=(),
        connections=build_connections(lines, 18.0),
        warnings=(),
    )


def _source_drag(label: QLabel) -> None:
    """Drag across the fitted source pixmap without relying on timing."""
    pixmap = label.pixmap()
    assert pixmap is not None
    left = (label.width() - pixmap.width()) // 2 + 20
    right = (label.width() - pixmap.width()) // 2 + pixmap.width() - 20
    middle = (label.height() - pixmap.height()) // 2 + pixmap.height() // 2
    QTest.mousePress(
        label,
        Qt.MouseButton.LeftButton,
        pos=QPoint(left, middle),
    )
    QTest.mouseRelease(
        label,
        Qt.MouseButton.LeftButton,
        pos=QPoint(right, middle),
    )


def _status_label(window: MainWindow) -> QLabel:
    """Return the named status caption used by action assertions."""
    caption = window.findChild(QLabel, "caption")
    assert caption is not None
    return caption


def test_main_window_loads_real_image_through_production_path(tmp_path: Path) -> None:
    # Given a saved source drawing and the real Qt application shell
    _ = _application()
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    image_path = tmp_path / "sketch.png"
    Image.fromarray(image).save(image_path)
    window = MainWindow()

    # When the same load path used by the file dialog is called
    loaded = window.load_path(image_path)

    # Then analysis, unresolved manual rows, and a recalculable document exist
    assert loaded is True
    assert window.analysis is not None
    assert window.analysis.dimensions
    assert window.document is not None


def test_manual_dimension_edit_recalculates_and_exports(tmp_path: Path) -> None:
    # Given a loaded drawing with an editable unknown dimension row
    _ = _application()
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    image_path = tmp_path / "sketch.png"
    Image.fromarray(image).save(image_path)
    window = MainWindow()
    assert window.load_path(image_path) is True

    # When the user enters a physical length in the real table widget
    window.set_dimension_text(0, "100")

    # Then the CAD document is recalibrated and can be exported
    assert window.document is not None
    assert window.document.scale_mm_per_source_unit is not None
    output_path = tmp_path / "edited.dxf"
    assert window.save_path(output_path) is True
    assert output_path.exists()


def test_selecting_dimension_row_identifies_line_and_highlights_source(
    tmp_path: Path,
) -> None:
    # Given a loaded drawing with one detected horizontal line
    _ = _application()
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    image_path = tmp_path / "sketch.png"
    Image.fromarray(image).save(image_path)
    window = MainWindow()
    assert window.load_path(image_path) is True

    # When the user selects its verification row
    table = window.findChild(QTableWidget, "dimension_table")
    assert table is not None
    table.selectRow(0)

    # Then the row exposes a human-readable orientation and selected dimension id
    assert window.selected_dimension_id is not None
    label = table.item(0, 0)
    assert label is not None
    assert "수평" in label.text()

    # And the original image contains the selection accent over the source line
    source_label = window.findChild(QLabel, "source_image")
    assert source_label is not None
    pixmap = source_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    source_image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
    selection_color = QColor("#D97757")
    assert any(
        source_image.pixelColor(x, y) == selection_color
        for y in range(source_image.height())
        for x in range(source_image.width())
    )

    # When the selected row receives a confirmed physical value
    window.set_dimension_text(0, "100")

    # Then the row remains linked after the table is recalculated
    assert window.selected_dimension_id is not None


def test_line_edit_buttons_merge_selected_rows(tmp_path: Path) -> None:
    # Given two detected line rows and the real edit toolbar
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    table = window.findChild(QTableWidget, "dimension_table")
    merge_button = window.findChild(QPushButton, "merge_lines")
    assert table is not None
    assert merge_button is not None
    assert merge_button.isEnabled() is False
    table.selectAll()
    assert window.selected_entity_ids == (
        EntityId("line_a"),
        EntityId("line_b"),
    )

    # When the user clicks merge
    merge_button.click()

    # Then split rows become one geometry and one target row
    assert window.analysis is not None
    assert len(window.analysis.entities) == 1
    assert table.rowCount() == 1
    assert "병합" in _status_label(window).text()
    window.set_dimension_text(0, "100")
    output_path = tmp_path / "merged.dxf"
    assert window.save_path(output_path) is True
    assert output_path.exists()


def test_line_edit_buttons_delete_selected_row(tmp_path: Path) -> None:
    # Given two detected line rows
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    table = window.findChild(QTableWidget, "dimension_table")
    delete_button = window.findChild(QPushButton, "delete_lines")
    assert table is not None
    assert delete_button is not None
    table.selectRow(0)

    # When the user deletes the selected line
    delete_button.click()

    # Then one line and one verification row remain
    assert window.analysis is not None
    assert tuple(entity.entity_id for entity in window.analysis.entities) == (
        EntityId("line_b"),
    )
    assert table.rowCount() == 1
    assert "삭제" in _status_label(window).text()


def test_line_edit_buttons_redraw_selected_line_on_source_image(
    tmp_path: Path,
) -> None:
    # Given one selected detected line and a loaded source image
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    window.resize(1100, 800)
    window.show()
    _application().processEvents()
    table = window.findChild(QTableWidget, "dimension_table")
    redraw_button = window.findChild(QPushButton, "redraw_line")
    source_label = window.findChild(QLabel, "source_image")
    assert table is not None
    assert redraw_button is not None
    assert source_label is not None
    table.selectRow(0)

    # When the user activates redraw and drags a replacement on the source
    redraw_button.click()
    _source_drag(source_label)

    # Then the selected identity remains but its source endpoints change
    assert window.analysis is not None
    replacement = window.analysis.entities[0]
    assert isinstance(replacement, LineEntity)
    assert replacement.entity_id == EntityId("line_a")
    assert replacement.raw_start != Point(40.0, 50.0)
    assert replacement.raw_end != Point(140.0, 50.0)
    assert "다시 그렸습니다" in _status_label(window).text()


def test_line_edit_buttons_redraw_selected_rows_collapses_them_into_one(
    tmp_path: Path,
) -> None:
    # Given two detected rows selected together and a loaded source image
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    window.resize(1100, 800)
    window.show()
    _application().processEvents()
    table = window.findChild(QTableWidget, "dimension_table")
    redraw_button = window.findChild(QPushButton, "redraw_line")
    source_label = window.findChild(QLabel, "source_image")
    assert table is not None
    assert redraw_button is not None
    assert source_label is not None
    table.selectAll()
    assert window.selected_entity_ids == (
        EntityId("line_a"),
        EntityId("line_b"),
    )

    # When the user activates redraw and drags one replacement stroke
    redraw_button.click()
    _source_drag(source_label)

    # Then both selected rows collapse into the single drawn line
    assert window.analysis is not None
    assert tuple(entity.entity_id for entity in window.analysis.entities) == (
        EntityId("line_a"),
    )
    assert table.rowCount() == 1
    assert "하나로 합쳐" in _status_label(window).text()
