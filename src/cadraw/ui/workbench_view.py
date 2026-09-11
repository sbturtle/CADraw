"""Native workbench composition for the CADraw main window."""

from typing import Final, final

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cadraw.models import CadDocument, DimensionId, DrawingAnalysis, EntityId, Point
from cadraw.preview import PreviewConfig, PreviewMode
from cadraw.ui.preview_widget import PreviewWidget
from cadraw.ui.source_image_widget import SourceImageLabel
from cadraw.ui.verification_panel import VerificationPanel

_SOURCE_STRETCH: Final = 5
_PREVIEW_STRETCH: Final = 4
_PREVIEW_SELECTION_HINT: Final = (
    "클릭: 한 선 · Ctrl+클릭: 복수 선택\n좌→우: 전체 포함 · 우→좌: 교차 포함"
)
_SUBTITLE_TEXT = """선은 자동으로 정리하고, 실제 치수는 입력된 값만 사용합니다.
모르는 값은 미확인 상태로 남습니다."""


@final
class WorkbenchView(QWidget):
    """Own the static Qt composition while MainWindow owns application state."""

    load_requested = Signal()
    recalculate_requested = Signal()
    save_requested = Signal()
    dimension_edited = Signal(int, str)
    selection_changed = Signal()
    merge_requested = Signal()
    delete_requested = Signal()
    redraw_toggled = Signal(bool)
    source_line_drawn = Signal(float, float, float, float)
    preview_selection_changed = Signal(tuple)
    dimension_apply_requested = Signal(int, str)
    preview_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the empty workbench view."""
        super().__init__(parent)
        self._source_label = SourceImageLabel(self)
        self._preview = PreviewWidget(self)
        self._verification = VerificationPanel(self)
        self._load_button: QPushButton = QPushButton(self)
        self._recalculate_button: QPushButton = QPushButton(self)
        self._save_button: QPushButton = QPushButton(self)
        self._caption = QLabel(self)
        self._preview_info = QLabel(self)
        self._mode_combo = QComboBox(self)
        self._build()

    @property
    def selected_dimension_id(self) -> DimensionId | None:
        """Return the currently focused verification row."""
        return self._verification.selected_dimension_id

    @property
    def selected_dimension_ids(self) -> tuple[DimensionId, ...]:
        """Return all selected verification rows."""
        return self._verification.selected_dimension_ids

    @property
    def selected_entity_ids(self) -> tuple[EntityId, ...]:
        """Return all geometry IDs linked to selected verification rows."""
        return self._verification.selected_entity_ids

    def set_analysis(
        self,
        analysis: DrawingAnalysis,
        selected_dimension_ids: tuple[DimensionId, ...],
    ) -> None:
        """Render the latest analysis and restore stable row selection."""
        self._verification.set_dimensions(analysis, selected_dimension_ids)

    def set_source_pixmap(
        self,
        pixmap: QPixmap,
        selection_lines: tuple[tuple[Point, Point], ...],
    ) -> None:
        """Render the source image and selected raw line overlay."""
        self._source_label.set_source_pixmap(pixmap)
        self._source_label.set_selection_lines(selection_lines)

    def set_preview_image(self, image: Image.Image) -> None:
        """Replace the normalized preview raster."""
        self._preview.set_image(image)

    def set_preview_context(
        self,
        document: CadDocument,
        config: PreviewConfig,
    ) -> None:
        """Provide the current coordinate space for preview hit testing."""
        self._preview.set_interaction_context(document, config)

    def set_selection_info(self, text: str) -> None:
        """Display stable identity and dimension state for the selection."""
        self._preview_info.setText(text)

    def select_entity(self, entity_id: EntityId) -> None:
        """Select the verification rows linked to a preview entity."""
        self.select_entities((entity_id,))

    def select_entities(self, entity_ids: tuple[EntityId, ...]) -> None:
        """Select the verification rows linked to preview entities."""
        self._verification.select_entities(entity_ids)

    @property
    def preview_mode(self) -> PreviewMode:
        """Return the currently displayed coordinate-space mode."""
        return (
            PreviewMode.CAD
            if self._mode_combo.currentIndex() == 0
            else PreviewMode.DETECTED
        )

    def set_main_actions_enabled(
        self,
        recalculate: bool,
        save: bool,
    ) -> None:
        """Set the enabled state of the primary toolbar actions."""
        _ = self._recalculate_button.setEnabled(recalculate)
        _ = self._save_button.setEnabled(save)

    def set_line_actions_enabled(
        self,
        merge: bool,
        delete: bool,
        redraw: bool,
    ) -> None:
        """Set the enabled state of line correction actions."""
        self._verification.set_line_actions_enabled(merge, delete, redraw)

    def set_redraw_checked(self, checked: bool) -> None:
        """Set the redraw toggle state."""
        self._verification.set_redraw_checked(checked)

    def set_source_drawing_enabled(self, enabled: bool) -> None:
        """Toggle source-image drag input for the redraw action."""
        self._source_label.set_drawing_enabled(enabled)

    def set_caption(self, text: str) -> None:
        """Display user-facing operation status."""
        self._caption.setText(text)

    def set_dimension_text(self, row: int, text: str) -> None:
        """Set a dimension value through the verification panel."""
        self._verification.set_dimension_text(row, text)

    def dimension_id_for_row(self, row: int) -> DimensionId | None:
        """Resolve a visible row to its stable dimension identity."""
        return self._verification.dimension_id_for_row(row)

    def _build(self) -> None:
        """Construct the stable desktop layout and signal bridge."""
        self.setMinimumSize(960, 680)
        self.resize(1240, 820)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 12)
        root_layout.setSpacing(12)
        self._add_header(root_layout)
        self._add_toolbar(root_layout)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._create_source_group())
        splitter.addWidget(self._create_preview_group())
        splitter.setStretchFactor(0, _SOURCE_STRETCH)
        splitter.setStretchFactor(1, _PREVIEW_STRETCH)
        root_layout.addWidget(splitter, 1)
        root_layout.addWidget(self._verification)
        self._caption.setObjectName("caption")
        self._caption.setText("사진을 불러오면 분석을 시작합니다.")
        root_layout.addWidget(self._caption)
        _ = self._verification.dimension_edited.connect(self.dimension_edited)
        _ = self._verification.selection_changed.connect(self.selection_changed)
        _ = self._verification.dimension_apply_requested.connect(
            self.dimension_apply_requested,
        )
        _ = self._verification.merge_requested.connect(self.merge_requested)
        _ = self._verification.delete_requested.connect(self.delete_requested)
        _ = self._verification.redraw_toggled.connect(self.redraw_toggled)
        _ = self._source_label.line_drawn.connect(self.source_line_drawn)
        _ = self._preview.selection_changed.connect(
            self.preview_selection_changed,
        )

    def _add_header(self, root_layout: QVBoxLayout) -> None:
        """Add title and safety guidance."""
        header = QHBoxLayout()
        title = QLabel("CADraw")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("사진 → 확인 → DXF"))
        root_layout.addLayout(header)
        subtitle = QLabel(_SUBTITLE_TEXT)
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root_layout.addWidget(subtitle)

    def _add_toolbar(self, root_layout: QVBoxLayout) -> None:
        """Add the three primary workbench actions."""
        toolbar = QHBoxLayout()
        self._load_button.setText("손도면 사진 불러오기")
        self._load_button.setObjectName("primary")
        _ = self._load_button.clicked.connect(self.load_requested)
        toolbar.addWidget(self._load_button)
        self._recalculate_button.setText("도면 다시 계산")
        _ = self._recalculate_button.clicked.connect(self.recalculate_requested)
        _ = self._recalculate_button.setEnabled(False)
        toolbar.addWidget(self._recalculate_button)
        self._save_button.setText("DXF 저장")
        _ = self._save_button.clicked.connect(self.save_requested)
        _ = self._save_button.setEnabled(False)
        toolbar.addWidget(self._save_button)
        toolbar.addStretch()
        root_layout.addLayout(toolbar)

    def _create_source_group(self) -> QGroupBox:
        """Create the source-image viewport."""
        group = QGroupBox("원본 손도면", self)
        layout = QVBoxLayout(group)
        self._source_label.setObjectName("source_image")
        layout.addWidget(self._source_label)
        return group

    def _create_preview_group(self) -> QGroupBox:
        """Create the coordinate-space preview viewport."""
        group = QGroupBox("Geometry Preview", self)
        layout = QVBoxLayout(group)
        mode_toolbar = QHBoxLayout()
        mode_toolbar.addWidget(QLabel("보기"))
        self._mode_combo.setObjectName("preview_mode")
        self._mode_combo.addItem("CAD 결과", PreviewMode.CAD.value)
        self._mode_combo.addItem("검출 결과", PreviewMode.DETECTED.value)
        _ = self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_toolbar.addWidget(self._mode_combo)
        mode_toolbar.addStretch()
        layout.addLayout(mode_toolbar)
        self._preview.setObjectName("cad_preview")
        layout.addWidget(self._preview)
        self._preview_info.setObjectName("selection_info")
        self._preview_info.setWordWrap(True)
        self._preview_info.setText(_PREVIEW_SELECTION_HINT)
        layout.addWidget(self._preview_info)
        return group

    def _on_mode_changed(self, index: int) -> None:
        """Forward the selected coordinate-space mode to MainWindow."""
        if index == 0:
            self.preview_mode_changed.emit(PreviewMode.CAD.value)
        elif index == 1:
            self.preview_mode_changed.emit(PreviewMode.DETECTED.value)
