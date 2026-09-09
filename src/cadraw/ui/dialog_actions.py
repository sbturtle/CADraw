"""Native file-dialog actions for the workbench shell."""

from collections.abc import Callable
from pathlib import Path
from typing import final

from PySide6.QtWidgets import QFileDialog, QWidget

type ImageLoader = Callable[[Path], bool]
type DocumentSaver = Callable[[Path], bool]


@final
class DialogActions:
    """Keep file-picker mechanics outside the application state owner."""

    def __init__(
        self,
        parent: QWidget,
        load_image: ImageLoader,
        save_document: DocumentSaver,
    ) -> None:
        """Store the two explicit file operation callbacks."""
        self._parent = parent
        self._load_image = load_image
        self._save_document = save_document

    def choose_image(self) -> None:
        """Open the native image picker."""
        file_name, _ = QFileDialog.getOpenFileName(
            self._parent,
            "손도면 사진 선택",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if file_name:
            _ = self._load_image(Path(file_name))

    def choose_save_path(self) -> None:
        """Open the native DXF save picker."""
        file_name, _ = QFileDialog.getSaveFileName(
            self._parent,
            "DXF 저장",
            "drawing.dxf",
            "DXF files (*.dxf)",
        )
        if file_name:
            _ = self._save_document(Path(file_name))
