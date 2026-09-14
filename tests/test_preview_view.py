"""Regression tests for the interactive CAD preview viewport."""

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QSplitter, QTableWidget

from cadraw.models import CadDocument, EntityId, LineEntity, Orientation, Point
from cadraw.preview import PreviewConfig, render_preview
from cadraw.ui.main_window import MainWindow
from cadraw.ui.preview_widget import PreviewWidget
from cadraw.ui.workbench_view import WorkbenchView


def _application() -> QApplication:
    """Return the test process QApplication."""
    application = QApplication.instance()
    if isinstance(application, QApplication):
        return application
    return QApplication([])


def _preview_document() -> CadDocument:
    """Build one selectable line for transformed hit testing."""
    line = LineEntity(
        entity_id=EntityId("line"),
        raw_start=Point(0.0, 0.0),
        raw_end=Point(100.0, 0.0),
        start=Point(0.0, 0.0),
        end=Point(100.0, 0.0),
        orientation=Orientation.HORIZONTAL,
        confidence=1.0,
    )
    return CadDocument(
        unit="mm",
        entities=(line,),
        dimensions=(),
        warnings=(),
        scale_mm_per_source_unit=None,
    )


def test_preview_view_transform_is_clamped_and_resettable() -> None:
    # Given a preview with a non-square raster
    _ = _application()
    preview = PreviewWidget()
    preview.resize(320, 240)
    preview.set_image(Image.new("RGB", (200, 100), "white"))

    # When the user rotates and zooms beyond the supported range
    preview.rotate_right()
    preview.set_zoom_factor(20.0)

    # Then the view changes without mutating the underlying raster settings
    assert preview.rotation_degrees == 90
    assert preview.zoom_factor == 4.0

    # When the user asks for a fit reset
    preview.reset_view()

    # Then the initial orientation and scale return
    assert preview.rotation_degrees == 0
    assert preview.zoom_factor == 1.0


def test_rotated_preview_keeps_entity_hit_testing_aligned() -> None:
    # Given a rendered selectable horizontal line
    _ = _application()
    preview = PreviewWidget()
    preview.resize(200, 200)
    config = PreviewConfig(width=200, height=200, padding=20)
    document = _preview_document()
    preview.set_interaction_context(document, config)
    preview.set_image(render_preview(document, config))
    selected: list[tuple[str, ...]] = []
    _ = preview.selection_changed.connect(selected.append)
    preview.show()
    _application().processEvents()

    # When the view rotates clockwise and the user clicks its transformed line
    preview.rotate_right()
    QTest.mouseClick(
        preview,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(100, 140),
    )

    # Then selection still resolves to the unchanged source entity
    assert selected[-1] == ("line",)


def test_preview_view_controls_are_visible_at_minimum_workbench_size() -> None:
    # Given the smallest supported desktop workbench
    _ = _application()
    view = WorkbenchView()
    view.resize(960, 680)
    view.show()
    _application().processEvents()

    # Then every view control remains discoverable and visible
    for object_name in (
        "preview_rotate_left",
        "preview_rotate_right",
        "preview_zoom_out",
        "preview_zoom_reset",
        "preview_zoom_in",
    ):
        button = view.findChild(QPushButton, object_name)
        assert button is not None
        assert button.isVisible()

    # When the workbench has room for a normal desktop layout
    view.resize(1240, 820)
    _application().processEvents()

    # Then the preview is allowed to consume the available vertical space
    preview = view.findChild(PreviewWidget)
    assert preview is not None
    assert preview.height() > 120

    # When the controls are activated through the real widget tree
    rotate_right = view.findChild(QPushButton, "preview_rotate_right")
    zoom_in = view.findChild(QPushButton, "preview_zoom_in")
    zoom_reset = view.findChild(QPushButton, "preview_zoom_reset")
    assert preview is not None
    assert rotate_right is not None
    assert zoom_in is not None
    assert zoom_reset is not None
    rotate_right.click()
    zoom_in.click()

    # Then the preview state changes and the reset action restores it
    assert preview.rotation_degrees == 90
    assert preview.zoom_factor > 1.0
    zoom_reset.click()
    assert preview.rotation_degrees == 0
    assert preview.zoom_factor == 1.0


def test_line_list_stays_scrollable_when_workbench_is_resized() -> None:
    # Given a real sample with more line rows than a compact table can show
    _ = _application()
    window = MainWindow()
    sample_path = (
        Path(__file__).resolve().parents[1]
        / "sample"
        / "KakaoTalk_20260909_141552817.jpg"
    )
    assert window.load_path(sample_path) is True
    window.resize(960, 600)
    window.show()
    _application().processEvents()
    view = window.centralWidget()
    table = window.findChild(QTableWidget, "dimension_table")
    panel = window.findChild(QSplitter, "workbench_splitter")
    assert view is not None
    assert table is not None
    assert panel is not None
    assert table.rowCount() > 4

    # When the window is compact
    compact_viewport = table.viewport()
    compact_visible = sum(
        table.visualRect(table.model().index(row, 0)).intersects(
            compact_viewport.rect(),
        )
        for row in range(table.rowCount())
    )

    # Then the panel remains inside the workbench and the table can scroll
    assert window.height() == 600
    assert panel.orientation() is Qt.Orientation.Vertical
    verification = table.parentWidget()
    assert verification is not None
    assert verification.geometry().bottom() <= view.height()
    assert compact_visible > 0
    assert table.verticalScrollBar().maximum() > 0

    # When the user expands the window
    window.resize(1240, 820)
    _application().processEvents()
    expanded_viewport = table.viewport()
    expanded_visible = sum(
        table.visualRect(table.model().index(row, 0)).intersects(
            expanded_viewport.rect(),
        )
        for row in range(table.rowCount())
    )

    # Then the list gets more room without losing its scroll contract
    assert expanded_visible >= compact_visible
    assert table.verticalScrollBar().maximum() >= 0
    _ = window.close()
