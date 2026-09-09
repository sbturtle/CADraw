"""CADraw application entry point."""

import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtWidgets import QApplication

from cadraw.ui.main_window import MainWindow
from cadraw.ui.theme import apply_theme


def main(argv: Sequence[str] | None = None) -> int:
    """Start the native workbench, optionally opening one image path."""
    arguments = list(sys.argv if argv is None else argv)
    application = QApplication(arguments)
    apply_theme(application)
    window = MainWindow()
    if len(arguments) > 1:
        _ = window.load_path(Path(arguments[1]))
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
