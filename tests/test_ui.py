from math import atan2, degrees
from pathlib import Path
from typing import final

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
)

from cadraw.calculator import calculate_document
from cadraw.models import (
    CadResolutionStatus,
    DrawingAnalysis,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)
from cadraw.normalization import build_connections
from cadraw.preview import PreviewConfig, render_preview
from cadraw.ui.main_window import MainWindow
from cadraw.ui.preview_widget import PreviewWidget


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


def _diagonal_analysis() -> DrawingAnalysis:
    """Build one diagonal line that requires an angle decision."""
    line = LineEntity(
        entity_id=EntityId("diagonal"),
        raw_start=Point(40.0, 160.0),
        raw_end=Point(140.0, 60.0),
        start=Point(40.0, 160.0),
        end=Point(140.0, 60.0),
        orientation=Orientation.DIAGONAL,
        confidence=0.9,
    )
    return DrawingAnalysis(
        unit="mm",
        image_size=(320, 240),
        entities=(line,),
        dimensions=(),
        connections=(),
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


def _interactive_preview() -> PreviewWidget:
    """Build a real preview widget with two separated selectable lines."""
    document = calculate_document(_two_line_analysis())
    config = PreviewConfig(width=240, height=160)
    preview = PreviewWidget()
    preview.set_interaction_context(document, config)
    preview.set_image(render_preview(document, config))
    preview.resize(240, 160)
    preview.show()
    _application().processEvents()
    return preview


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


def test_axis_angles_are_read_only_and_auto_confirmed(tmp_path: Path) -> None:
    # Given a loaded workbench with horizontal and vertical detected lines
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    table = window.findChild(QTableWidget, "dimension_table")
    assert table is not None

    # When the generated axis-angle rows are displayed
    angle_rows: list[int] = []
    for row in range(table.rowCount()):
        label = table.item(row, 0)
        if label is not None and "각도" in label.text():
            angle_rows.append(row)

    # Then common axes are visible as confirmed, non-editable values
    assert len(angle_rows) == 2
    first_value = table.item(angle_rows[0], 1)
    second_value = table.item(angle_rows[1], 1)
    assert first_value is not None
    assert second_value is not None
    assert first_value.text() == "0°"
    assert second_value.text() == "0°"
    for row in angle_rows:
        value = table.item(row, 1)
        status = table.item(row, 3)
        assert value is not None
        assert status is not None
        assert not bool(value.flags() & Qt.ItemFlag.ItemIsEditable)
        assert status.text() == "자동 확정"


def test_diagonal_angle_requires_approval_before_geometry_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a loaded workbench with one diagonal line and empty angle input
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_diagonal_analysis()))
    assert window.load_path(image_path) is True
    table = window.findChild(QTableWidget, "dimension_table")
    apply_button = window.findChild(QPushButton, "apply_dimension")
    assert table is not None
    assert apply_button is not None
    angle_row = next(
        row
        for row in range(table.rowCount())
        if ((label := table.item(row, 0)) is not None and "각도" in label.text())
    )

    # When length is confirmed and an angle is typed but not yet approved
    window.set_dimension_text(0, "100")
    window.set_dimension_text(angle_row, "20")
    assert window.document is not None
    assert window.document.cad_entities[0].status is CadResolutionStatus.UNKNOWN_ANGLE
    table.selectRow(angle_row)
    assert apply_button.text() == "각도 승인 후 다시 그리기"

    # When the user explicitly approves the angle-based redraw
    def approve_angle(
        *_args: object,
        **_kwargs: object,
    ) -> QMessageBox.StandardButton:
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", approve_angle)
    apply_button.click()

    # Then the CAD geometry is rebuilt at the approved angle
    assert window.document.cad_entities[0].status is CadResolutionStatus.RESOLVED
    geometry = window.document.cad_entities[0].geometry
    assert isinstance(geometry, LineEntity)
    assert (
        round(
            abs(
                degrees(
                    atan2(
                        geometry.end.y - geometry.start.y,
                        geometry.end.x - geometry.start.x,
                    ),
                ),
            ),
            6,
        )
        == 20.0
    )
    assert geometry.end.y > geometry.start.y


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


def test_clicking_preview_line_selects_entity_and_shows_identity(
    tmp_path: Path,
) -> None:
    # Given a loaded drawing with a selectable source line
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    window.resize(1100, 800)
    window.show()
    _application().processEvents()
    preview = window.findChild(PreviewWidget, "cad_preview")
    info = window.findChild(QLabel, "selection_info")
    assert preview is not None
    assert info is not None
    # When the user clicks the first line in the rendered preview
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(preview.width() // 4, preview.height() * 3 // 8),
    )

    # Then the linked row and stable entity identity are selected
    assert window.selected_entity_ids == (EntityId("line_a"),)
    assert "line_a" in info.text()
    assert "pixel" in info.text()


def test_main_window_preview_ctrl_click_selects_multiple_rows(
    tmp_path: Path,
) -> None:
    # Given a loaded workbench with two selectable verification lines
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    window.resize(1100, 800)
    window.show()
    _application().processEvents()
    preview = window.findChild(PreviewWidget, "cad_preview")
    assert preview is not None

    # When the user clicks one line and Ctrl-clicks the second line
    line_y = preview.height() * 3 // 8
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(preview.width() // 4, line_y),
    )
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        QPoint(preview.width() * 3 // 4, line_y),
    )

    # Then both source IDs are selected in the verification table
    assert window.selected_entity_ids == (
        EntityId("line_a"),
        EntityId("line_b"),
    )


def test_preview_ctrl_click_toggles_multiple_lines() -> None:
    # Given a rendered preview with two selectable lines
    _ = _application()
    preview = _interactive_preview()
    selections: list[tuple[str, ...]] = []
    _ = preview.selection_changed.connect(selections.append)

    # When the user clicks one line and Ctrl-clicks a second line
    QTest.mouseClick(preview, Qt.MouseButton.LeftButton, pos=QPoint(70, 76))
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        QPoint(165, 76),
    )

    # Then both lines remain selected
    assert selections[-1] == ("line_a", "line_b")

    # When the user Ctrl-clicks the first line again
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        QPoint(70, 76),
    )

    # Then Ctrl-click toggles only that line off
    assert selections[-1] == ("line_b",)


def test_preview_drag_uses_window_and_crossing_selection() -> None:
    # Given a rendered preview with two selectable lines
    _ = _application()
    preview = _interactive_preview()
    selections: list[tuple[str, ...]] = []
    _ = preview.selection_changed.connect(selections.append)

    # When dragging left-to-right around only the first line
    QTest.mousePress(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(25, 65),
    )
    QTest.mouseMove(preview, QPoint(122, 88))
    QTest.mouseRelease(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(122, 88),
    )

    # Then the window selects only fully enclosed lines
    assert selections[-1] == ("line_a",)

    # When dragging right-to-left across both lines, touching the first
    QTest.mousePress(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(215, 88),
    )
    QTest.mouseMove(preview, QPoint(115, 65))
    QTest.mouseRelease(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(115, 65),
    )

    # Then the crossing window accepts partial intersections
    assert selections[-1] == ("line_a", "line_b")


def test_preview_selection_info_stays_below_canvas(tmp_path: Path) -> None:
    # Given a loaded preview with a multi-line selected-entity description
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    window.resize(960, 680)
    window.show()
    _application().processEvents()
    preview = window.findChild(PreviewWidget, "cad_preview")
    info = window.findChild(QLabel, "selection_info")
    assert preview is not None
    assert info is not None

    # When the selected entity info wraps to its real content height
    info.setText(
        """종류: 사선 Line
ID: line_a
상태: unresolved
감지 픽셀 길이: 85.2029 px
실제 길이: 미확정""",
    )
    _application().processEvents()

    # Then the information panel never covers the drawing canvas
    assert preview.geometry().bottom() < info.geometry().top()


def test_dimension_apply_button_and_preview_mode_toggle_are_available(
    tmp_path: Path,
) -> None:
    # Given a loaded drawing and the explicit dimension controls
    _ = _application()
    image_path = tmp_path / "sketch.png"
    Image.fromarray(np.full((240, 320), 255, dtype=np.uint8)).save(image_path)
    window = MainWindow(analyzer=_StaticAnalyzer(_two_line_analysis()))
    assert window.load_path(image_path) is True
    table = window.findChild(QTableWidget, "dimension_table")
    apply_button = window.findChild(QPushButton, "apply_dimension")
    mode_combo = window.findChild(QComboBox, "preview_mode")
    assert table is not None
    assert apply_button is not None
    assert mode_combo is not None

    # When a user enters a length and switches coordinate-space view
    table.selectRow(0)
    window.set_dimension_text(0, "100")
    apply_button.click()
    mode_combo.setCurrentIndex(1)
    _application().processEvents()

    # Then the CAD geometry remains calculated and the detected view is active
    assert window.document is not None
    result = window.document.entities[0]
    assert isinstance(result, LineEntity)
    assert result.length == 100.0
    assert mode_combo.currentText() == "검출 결과"


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

    # Then split rows become one geometry and its length/angle rows
    assert window.analysis is not None
    assert len(window.analysis.entities) == 1
    assert table.rowCount() == 2
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

    # Then one line and its length/angle verification rows remain
    assert window.analysis is not None
    assert tuple(entity.entity_id for entity in window.analysis.entities) == (
        EntityId("line_b"),
    )
    assert table.rowCount() == 2
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

    # Then both selected rows collapse into one line with two verification rows
    assert window.analysis is not None
    assert tuple(entity.entity_id for entity in window.analysis.entities) == (
        EntityId("line_a"),
    )
    assert table.rowCount() == 2
    assert "하나로 합쳐" in _status_label(window).text()
