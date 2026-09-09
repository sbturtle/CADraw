"""CADraw native widget styling tokens."""

from PySide6.QtWidgets import QApplication

_STYLE_SHEET = """
QMainWindow, QWidget {
    background: #F5F4ED;
    color: #141413;
    font-family: "Segoe UI";
    font-size: 14px;
}
QLabel#title {
    font-size: 22px;
    font-weight: 600;
}
QLabel#subtitle, QLabel#caption {
    color: #5E5D59;
}
QGroupBox {
    background: #FAF9F5;
    border: 1px solid #E8E6DC;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #4D4C48;
}
QPushButton {
    min-height: 40px;
    padding: 4px 12px;
    border: 1px solid #D1CFC5;
    border-radius: 8px;
    background: #E8E6DC;
    color: #4D4C48;
}
QPushButton:hover {
    background: #FFFFFF;
    border-color: #C96442;
}
QPushButton:pressed {
    background: #D1CFC5;
}
QPushButton:focus, QLineEdit:focus, QTableWidget:focus {
    border: 2px solid #3898EC;
}
QPushButton#primary {
    background: #C96442;
    border-color: #C96442;
    color: #FAF9F5;
}
QPushButton#primary:hover {
    background: #D97757;
}
QLineEdit, QTableWidget {
    background: #FFFFFF;
    border: 1px solid #E8E6DC;
    border-radius: 8px;
    padding: 4px 8px;
}
QTableWidget {
    selection-background-color: #F0EEE6;
    selection-color: #141413;
}
QTableWidget::item:selected {
    background: #F0EEE6;
    color: #141413;
}
QHeaderView::section {
    background: #E8E6DC;
    color: #4D4C48;
    padding: 4px 8px;
    border: 0;
}
QStatusBar {
    background: #E8E6DC;
    color: #4D4C48;
}
"""


def apply_theme(application: QApplication) -> None:
    """Apply the documented warm workshop palette to the Qt app."""
    application.setStyleSheet(_STYLE_SHEET)
