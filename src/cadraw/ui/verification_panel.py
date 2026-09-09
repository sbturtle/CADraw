"""Human verification table and line-edit controls."""

from typing import Final, final

from PySide6.QtCore import QItemSelectionModel, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cadraw.models import (
    DimensionId,
    DrawingAnalysis,
    EntityId,
)
from cadraw.ui.verification_labels import dimension_label, status_text

_VALUE_COLUMN: Final = 1
_STATUS_COLUMN: Final = 3
_MIN_MERGE_SELECTION: Final = 2
_MIN_DELETE_SELECTION: Final = 1
_GUIDANCE_TEXT = """선택된 선의 실제 길이를 직접 입력하세요.
빈 값은 UNKNOWN으로 저장됩니다."""


@final
class VerificationPanel(QGroupBox):
    """Own the verification table without owning analysis mutations."""

    dimension_edited = Signal(int, str)
    selection_changed = Signal()
    merge_requested = Signal()
    delete_requested = Signal()
    redraw_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty verification panel."""
        super().__init__("치수 확인", parent)
        self._analysis: DrawingAnalysis | None = None
        self._dimension_row_by_id: dict[DimensionId, int] = {}
        self._dimension_table = QTableWidget(0, 4, self)
        self._merge_button = QPushButton("선 병합", self)
        self._delete_button = QPushButton("선 삭제", self)
        self._redraw_button = QPushButton("선 다시 그리기", self)
        self._build()

    @property
    def selected_dimension_id(self) -> DimensionId | None:
        """Return the current row's stable dimension identity."""
        row = self._current_row()
        return self.dimension_id_for_row(row)

    @property
    def selected_dimension_ids(self) -> tuple[DimensionId, ...]:
        """Return selected verification IDs in visible row order."""
        rows = sorted(
            index.row()
            for index in self._dimension_table.selectionModel().selectedRows()
        )
        return tuple(
            dimension_id
            for row in rows
            if (dimension_id := self.dimension_id_for_row(row)) is not None
        )

    @property
    def selected_entity_ids(self) -> tuple[EntityId, ...]:
        """Return unique source IDs associated with selected rows."""
        if self._analysis is None:
            return ()
        selected_dimensions = set(self.selected_dimension_ids)
        entity_ids: list[EntityId] = []
        for dimension in self._analysis.dimensions:
            if dimension.dimension_id not in selected_dimensions:
                continue
            for entity_id in dimension.target_entity_ids:
                if entity_id not in entity_ids:
                    entity_ids.append(entity_id)
        return tuple(entity_ids)

    def set_dimensions(
        self,
        analysis: DrawingAnalysis,
        selected_dimension_ids: tuple[DimensionId, ...],
    ) -> None:
        """Render immutable analysis dimensions and restore row selection."""
        self._analysis = analysis
        block_signals = True
        _ = self._dimension_table.blockSignals(block_signals)
        self._dimension_table.setRowCount(0)
        self._dimension_row_by_id.clear()
        for row, dimension in enumerate(analysis.dimensions):
            self._dimension_table.insertRow(row)
            self._dimension_row_by_id[dimension.dimension_id] = row
            label = QTableWidgetItem(
                dimension_label(self._analysis, dimension, row),
            )
            if dimension.target_entity_ids:
                _ = label.setToolTip(
                    f"대상 ID: {dimension.target_entity_ids[0]}",
                )
            value = QTableWidgetItem(dimension.raw_text)
            value.setToolTip("예: 100 또는 105.5")
            unit = QTableWidgetItem(dimension.unit)
            status = QTableWidgetItem(status_text(dimension))
            self._dimension_table.setItem(row, 0, label)
            self._dimension_table.setItem(row, _VALUE_COLUMN, value)
            self._dimension_table.setItem(row, 2, unit)
            self._dimension_table.setItem(row, _STATUS_COLUMN, status)
        block_signals = False
        _ = self._dimension_table.blockSignals(block_signals)
        block_signals = True
        _ = self._dimension_table.blockSignals(block_signals)
        selection_model = self._dimension_table.selectionModel()
        selection_model.clearSelection()
        selected_rows = tuple(
            self._dimension_row_by_id[dimension_id]
            for dimension_id in selected_dimension_ids
            if dimension_id in self._dimension_row_by_id
        )
        if selected_rows:
            self._dimension_table.setCurrentCell(selected_rows[0], 0)
            for row in selected_rows:
                index = self._dimension_table.model().index(row, 0)
                selection_model.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
        block_signals = False
        _ = self._dimension_table.blockSignals(block_signals)
        self.selection_changed.emit()

    def set_line_actions_enabled(
        self,
        merge: bool,
        delete: bool,
        redraw: bool,
    ) -> None:
        """Set action state from the owning window's analysis selection."""
        _ = self._merge_button.setEnabled(merge)
        _ = self._delete_button.setEnabled(delete)
        _ = self._redraw_button.setEnabled(redraw)

    def set_redraw_checked(self, checked: bool) -> None:
        """Set redraw mode through the native checkable action."""
        _ = self._redraw_button.setChecked(checked)

    def dimension_id_for_row(self, row: int) -> DimensionId | None:
        """Resolve a visible row to its stable dimension identity."""
        return next(
            (
                dimension_id
                for dimension_id, dimension_row in self._dimension_row_by_id.items()
                if dimension_row == row
            ),
            None,
        )

    def set_dimension_text(self, row: int, text: str) -> None:
        """Set a verification value through the real table widget."""
        self._dimension_table.setItem(row, _VALUE_COLUMN, QTableWidgetItem(text))

    def _build(self) -> None:
        """Build controls and connect only panel-local signals."""
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_GUIDANCE_TEXT))
        edit_toolbar = QHBoxLayout()
        self._merge_button.setObjectName("merge_lines")
        self._merge_button.setToolTip("선택한 같은 선 조각을 하나로 합칩니다.")
        _ = self._merge_button.clicked.connect(self.merge_requested)
        edit_toolbar.addWidget(self._merge_button)
        self._delete_button.setObjectName("delete_lines")
        self._delete_button.setToolTip("선택한 잘못 인식한 선을 제거합니다.")
        _ = self._delete_button.clicked.connect(self.delete_requested)
        edit_toolbar.addWidget(self._delete_button)
        self._redraw_button.setObjectName("redraw_line")
        self._redraw_button.setCheckable(True)
        self._redraw_button.setToolTip(
            "원본 이미지에서 드래그해 선택한 선을 교체하거나 새 선을 추가합니다.",
        )
        _ = self._redraw_button.toggled.connect(self.redraw_toggled)
        edit_toolbar.addWidget(self._redraw_button)
        edit_toolbar.addStretch()
        layout.addLayout(edit_toolbar)
        self._dimension_table.setObjectName("dimension_table")
        self._dimension_table.setHorizontalHeaderLabels(
            ["항목", "값", "단위", "상태"],
        )
        self._dimension_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows,
        )
        self._dimension_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        header = self._dimension_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            _VALUE_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            _STATUS_COLUMN,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        _ = self._dimension_table.itemChanged.connect(self._on_item_changed)
        _ = self._dimension_table.itemSelectionChanged.connect(
            self.selection_changed,
        )
        layout.addWidget(self._dimension_table)
        self._dimension_table.setMinimumHeight(140)

    def _current_row(self) -> int:
        """Return a selected row even when a range has no current index."""
        current_row = self._dimension_table.currentRow()
        if current_row >= 0:
            return current_row
        selected_ids = self.selected_dimension_ids
        if not selected_ids:
            return -1
        return self._dimension_row_by_id[selected_ids[0]]

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Forward only value-column edits to the owning window."""
        if item.column() == _VALUE_COLUMN:
            self.dimension_edited.emit(item.row(), item.text())
