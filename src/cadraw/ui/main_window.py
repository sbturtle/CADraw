"""Main human-in-the-loop workbench window."""

from dataclasses import replace
from pathlib import Path
from typing import override

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cadraw.calculator import calculate_document
from cadraw.detection import ImageReadError
from cadraw.dimensions import confirm_dimension_text
from cadraw.export import export_dxf, validate_dxf
from cadraw.models import (
    ArcEntity,
    CadDocument,
    CircleEntity,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    GeometryEntity,
    LineEntity,
    PolylineEntity,
)
from cadraw.pipeline import PipelineConfig
from cadraw.preview import PreviewConfig, render_preview
from cadraw.ui.preview_widget import PreviewWidget
from cadraw.vision import MockVisionAnalyzer, VisionAnalyzer

_VALUE_COLUMN = 1
_STATUS_COLUMN = 3


def _manual_dimensions(entities: tuple[GeometryEntity, ...]) -> tuple[Dimension, ...]:
    """Create empty length rows for detected lines, never inferred values."""
    dimensions: list[Dimension] = []
    for entity in entities:
        match entity:
            case LineEntity(entity_id=entity_id):
                dimensions.append(
                    Dimension(
                        dimension_id=DimensionId(f"manual_{entity_id}"),
                        raw_text="",
                        kind=DimensionKind.LENGTH,
                        parsed_value=None,
                        unit="mm",
                        target_entity_ids=(entity_id,),
                        confidence=0.0,
                        status=DimensionStatus.UNKNOWN,
                    ),
                )
            case PolylineEntity() | CircleEntity() | ArcEntity():
                continue
    return tuple(dimensions)


class MainWindow(QMainWindow):
    """Desktop shell for image analysis, verification, and DXF export."""

    def __init__(
        self,
        analyzer: VisionAnalyzer | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the review-first workbench."""
        super().__init__(parent)
        self._analyzer: VisionAnalyzer = analyzer or MockVisionAnalyzer(
            PipelineConfig.from_detection_defaults(),
        )
        self._analysis: DrawingAnalysis | None = None
        self._document: CadDocument | None = None
        self._source_pixmap: QPixmap | None = None
        self._dimension_row_by_id: dict[DimensionId, int] = {}
        self._load_button: QPushButton = QPushButton()
        self._recalculate_button: QPushButton = QPushButton()
        self._save_button: QPushButton = QPushButton()
        self._source_label: QLabel = QLabel()
        self._preview: PreviewWidget = PreviewWidget()
        self._dimension_table: QTableWidget = QTableWidget(0, 4)
        self._status_label: QLabel = QLabel()
        self._build_ui()

    @property
    def analysis(self) -> DrawingAnalysis | None:
        """Return the latest intermediate analysis for smoke tests and callers."""
        return self._analysis

    @property
    def document(self) -> CadDocument | None:
        """Return the latest recalculated CAD document."""
        return self._document

    def _build_ui(self) -> None:
        """Construct the stable workbench layout."""
        self.setWindowTitle("CADraw · 손도면 변환")
        self.setMinimumSize(960, 680)
        self.resize(1240, 820)

        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 12)
        root_layout.setSpacing(12)
        self._add_header(root_layout)
        self._add_toolbar(root_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._create_source_group())
        splitter.addWidget(self._create_preview_group())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        root_layout.addWidget(splitter, 1)
        self._add_verification(root_layout)
        self._status_label.setText("사진을 불러오면 분석을 시작합니다.")
        self._status_label.setObjectName("caption")
        root_layout.addWidget(self._status_label)
        self.setCentralWidget(root)

    def _add_header(self, root_layout: QVBoxLayout) -> None:
        """Add title and safety guidance."""
        header = QHBoxLayout()
        title = QLabel("CADraw")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel("사진 → 확인 → DXF"))
        root_layout.addLayout(header)
        first_subtitle = "선은 자동으로 정리하고, 실제 치수는 입력된 값만 사용합니다."
        second_subtitle = "모르는 값은 미확인 상태로 남습니다."
        subtitle = QLabel(
            f"{first_subtitle} {second_subtitle}",
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root_layout.addWidget(subtitle)

    def _add_toolbar(self, root_layout: QVBoxLayout) -> None:
        """Add the three primary workbench actions."""
        toolbar = QHBoxLayout()
        self._load_button.setText("손도면 사진 불러오기")
        self._load_button.setObjectName("primary")
        _ = self._load_button.clicked.connect(self._choose_image)
        toolbar.addWidget(self._load_button)
        self._recalculate_button.setText("도면 다시 계산")
        _ = self._recalculate_button.clicked.connect(self._recalculate)
        _ = self._recalculate_button.setEnabled(False)
        toolbar.addWidget(self._recalculate_button)
        self._save_button.setText("DXF 저장")
        _ = self._save_button.clicked.connect(self._choose_save_path)
        _ = self._save_button.setEnabled(False)
        toolbar.addWidget(self._save_button)
        toolbar.addStretch()
        root_layout.addLayout(toolbar)

    def _add_verification(self, root_layout: QVBoxLayout) -> None:
        """Add the manual dimension verification area."""
        verification_group = QGroupBox("치수 확인")
        verification_layout = QVBoxLayout(verification_group)
        first_guidance = "선택된 선의 실제 길이를 직접 입력하세요."
        second_guidance = "빈 값은 UNKNOWN으로 저장됩니다."
        verification_layout.addWidget(
            QLabel(f"{first_guidance} {second_guidance}"),
        )
        self._dimension_table.setHorizontalHeaderLabels(
            ["항목", "값", "단위", "상태"],
        )
        self._dimension_table.horizontalHeader().setStretchLastSection(True)
        _ = self._dimension_table.itemChanged.connect(self._on_dimension_edited)
        verification_layout.addWidget(self._dimension_table)
        root_layout.addWidget(verification_group)

    def _create_source_group(self) -> QGroupBox:
        """Create the source-image viewport."""
        group = QGroupBox("원본 손도면")
        layout = QVBoxLayout(group)
        self._source_label.setText("손도면 사진을 불러오세요.")
        self._source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_label.setWordWrap(True)
        self._source_label.setMinimumSize(420, 300)
        layout.addWidget(self._source_label)
        return group

    def _create_preview_group(self) -> QGroupBox:
        """Create the normalized-preview viewport."""
        group = QGroupBox("정규화된 CAD Preview")
        layout = QVBoxLayout(group)
        layout.addWidget(self._preview)
        return group

    def _choose_image(self) -> None:
        """Open the native image picker."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "손도면 사진 선택",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if file_name:
            _ = self.load_path(Path(file_name))

    def load_path(self, image_path: Path) -> bool:
        """Load and analyze one image through the production path."""
        try:
            analysis = self._analyzer.analyze(image_path)
        except ImageReadError as error:
            self._status_label.setText(str(error))
            return False
        self._analysis = replace(
            analysis,
            dimensions=_manual_dimensions(analysis.entities),
        )
        self._source_pixmap = QPixmap(str(image_path))
        self._refresh_source_pixmap()
        _ = self._recalculate_button.setEnabled(True)
        _ = self._save_button.setEnabled(True)
        self._refresh_dimension_table()
        self._recalculate()
        self._status_label.setText(
            f"선 {len(self._analysis.entities)}개를 찾았습니다. 치수를 확인해 주세요.",
        )
        return True

    def _refresh_source_pixmap(self) -> None:
        """Fit the source image into its current viewport."""
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        available = self._source_label.size()
        scaled = self._source_pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        _ = self._source_label.setPixmap(scaled)

    def _refresh_dimension_table(self) -> None:
        """Render immutable dimension state into editable rows."""
        if self._analysis is None:
            return
        block_signals = True
        _ = self._dimension_table.blockSignals(block_signals)
        self._dimension_table.setRowCount(0)
        self._dimension_row_by_id.clear()
        for row, dimension in enumerate(self._analysis.dimensions):
            self._dimension_table.insertRow(row)
            self._dimension_row_by_id[dimension.dimension_id] = row
            label = QTableWidgetItem(
                f"{dimension.target_entity_ids[0]} 길이",
            )
            value = QTableWidgetItem(dimension.raw_text)
            value.setToolTip("예: 100 또는 105.5")
            unit = QTableWidgetItem(dimension.unit)
            status = QTableWidgetItem(self._status_text(dimension))
            self._dimension_table.setItem(row, 0, label)
            self._dimension_table.setItem(row, _VALUE_COLUMN, value)
            self._dimension_table.setItem(row, 2, unit)
            self._dimension_table.setItem(row, _STATUS_COLUMN, status)
        block_signals = False
        _ = self._dimension_table.blockSignals(block_signals)

    def _status_text(self, dimension: Dimension) -> str:
        """Return a textual review state for the table."""
        match dimension.status:
            case DimensionStatus.CONFIRMED:
                return "확정"
            case DimensionStatus.NEEDS_REVIEW:
                return "검토 필요"
            case DimensionStatus.UNKNOWN:
                return "확인 필요"

    def set_dimension_text(self, row: int, text: str) -> None:
        """Set a verification value through the same table edit path."""
        self._dimension_table.setItem(row, _VALUE_COLUMN, QTableWidgetItem(text))

    def _on_dimension_edited(self, item: QTableWidgetItem) -> None:
        """Parse each user edit and keep invalid values visibly unresolved."""
        if item.column() != _VALUE_COLUMN or self._analysis is None:
            return
        dimension_id = next(
            (
                key
                for key, row in self._dimension_row_by_id.items()
                if row == item.row()
            ),
            None,
        )
        if dimension_id is None:
            return
        dimensions = tuple(
            confirm_dimension_text(dimension, item.text())
            if dimension.dimension_id == dimension_id
            else dimension
            for dimension in self._analysis.dimensions
        )
        self._analysis = replace(self._analysis, dimensions=dimensions)
        self._refresh_dimension_table()
        self._recalculate()

    def _recalculate(self) -> None:
        """Rebuild the preview from the latest verified intermediate model."""
        if self._analysis is None:
            return
        self._document = calculate_document(self._analysis)
        preview = render_preview(
            self._document,
            PreviewConfig(width=720, height=520),
        )
        self._preview.set_image(preview)
        if self._document.scale_mm_per_source_unit is None:
            self._status_label.setText(
                "스케일 미확인: 치수를 하나 이상 확정하면 mm로 다시 계산합니다.",
            )
        else:
            geometry_text = f"{len(self._document.entities)}개 geometry"
            warning_text = (
                f" · 경고 {len(self._document.warnings)}개"
                if self._document.warnings
                else ""
            )
            self._status_label.setText(
                f"계산 완료 · {geometry_text}{warning_text}",
            )

    def _choose_save_path(self) -> None:
        """Open the native DXF save picker."""
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "DXF 저장",
            "drawing.dxf",
            "DXF files (*.dxf)",
        )
        if file_name:
            _ = self.save_path(Path(file_name))

    def save_path(self, output_path: Path) -> bool:
        """Export and validate a DXF at an explicit path."""
        if self._document is None:
            self._status_label.setText("먼저 손도면 사진을 불러오세요.")
            return False
        try:
            summary = export_dxf(self._document, output_path)
            validation = validate_dxf(output_path)
        except OSError as error:
            self._status_label.setText(f"DXF 저장 실패: {error}")
            return False
        geometry_text = f"geometry {summary.geometry_entity_count}개"
        entity_text = f"entity {validation.total_entities}개"
        self._status_label.setText(
            f"DXF 저장 완료 · {geometry_text} · {entity_text}",
        )
        return True

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the source image fitted after a window resize."""
        super().resizeEvent(event)
        self._refresh_source_pixmap()
