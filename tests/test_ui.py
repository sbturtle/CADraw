from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from cadraw.ui.main_window import MainWindow


def _application() -> QApplication:
    """Return the test process QApplication."""
    application = QApplication.instance()
    if isinstance(application, QApplication):
        return application
    return QApplication([])


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
