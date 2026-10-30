"""Human verification table and line-edit controls."""

from typing import Final, final

from PySide6.QtCore import QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cadraw.models import (
    DimensionId,
    DimensionKind,
    DrawingAnalysis,
    EntityId,
)
from cadraw.ui.verification_labels import dimension_label, status_text

_VALUE_COLUMN: Final = 1
_STATUS_COLUMN: Final = 3
_MIN_MERGE_SELECTION: Final = 2
_MIN_DELETE_SELECTION: Final = 1
_AUTO_ANGLE_SOURCE: Final = "axis_inference"
_UNLINKED_VALUE_TIP: Final = (
    "읽힌 값이지만 잴 형상을 정하지 못해 참고용입니다. 해당 형상의 행에 입력하세요."
)
_DEFAULT_APPLY_TEXT: Final = "치수 적용"
_ANGLE_APPLY_TEXT: Final = "각도 승인 후 다시 그리기"
_GUIDANCE_TEXT = """선택된 선의 실제 길이를 직접 입력하세요.
빈 값은 UNKNOWN으로 저장됩니다."""
_REGION_TOOLTIP = """원본 이미지에서 도면이 있는 사각형 영역을 드래그해
지정하면 양식의 머리글과 푸터를 분석에서 제외합니다."""


@final
class VerificationPanel(QGroupBox):
    """Own the verification table without owning analysis mutations."""

    dimension_edited = Signal(int, str)
    selection_changed = Signal()
    merge_requested = Signal()
    delete_requested = Signal()
    redraw_toggled = Signal(bool)
    region_toggled = Signal(bool)
    region_cleared = Signal()
    dimension_apply_requested = Signal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty verification panel."""
        super().__init__("치수 확인", parent)
        self._analysis: DrawingAnalysis | None = None
        self._dimension_row_by_id: dict[DimensionId, int] = {}
        self._dimension_table = QTableWidget(0, 4, self)
        self._merge_button = QPushButton("선 병합", self)
        self._delete_button = QPushButton("선 삭제", self)
        self._redraw_button = QPushButton("선 다시 그리기", self)
        self._region_button = QPushButton("도면 영역 지정", self)
        self._region_clear_button = QPushButton("영역 해제", self)
        self._apply_button = QPushButton("치수 적용", self)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._build()

    @property
    def selected_dimension_id(self) -> DimensionId | None:
        """Return the current row's stable dimension identity."""
        if not self.selected_dimension_ids:
            return None
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
            value.setToolTip(
                "예: 45 또는 45°"
                if dimension.kind is DimensionKind.ANGLE
                else "예: 100 또는 105.5",
            )
            if dimension.source == _AUTO_ANGLE_SOURCE:
                value.setFlags(value.flags() & ~Qt.ItemFlag.ItemIsEditable)
                value.setToolTip("축 정렬로 자동 확정된 각도입니다.")
            elif not dimension.target_entity_ids:
                # A value read with nothing to measure has no geometry to
                # size, so confirming it here would change nothing.
                value.setFlags(value.flags() & ~Qt.ItemFlag.ItemIsEditable)
                value.setToolTip(_UNLINKED_VALUE_TIP)
            unit = QTableWidgetItem(dimension.unit)
            status = QTableWidgetItem(status_text(dimension))
            for metadata_item in (label, unit, status):
                metadata_item.setFlags(
                    metadata_item.flags() & ~Qt.ItemFlag.ItemIsEditable,
                )
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
        self._on_selection_changed()

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

    def set_region_checked(self, checked: bool) -> None:
        """Set region-marking mode through the checkable button."""
        _ = self._region_button.setChecked(checked)

    def set_region_actions_enabled(self, mark: bool, clear: bool) -> None:
        """Set region affordances from whether an image and region exist."""
        _ = self._region_button.setEnabled(mark)
        _ = self._region_clear_button.setEnabled(clear)

    def _on_selection_changed(self) -> None:
        """Refresh the apply affordance before publishing row selection."""
        dimension_id = self.selected_dimension_id
        dimension = (
            next(
                (
                    item
                    for item in self._analysis.dimensions
                    if item.dimension_id == dimension_id
                ),
                None,
            )
            if self._analysis is not None and dimension_id is not None
            else None
        )
        self._apply_button.setText(
            _ANGLE_APPLY_TEXT
            if (
                dimension is not None
                and dimension.kind is DimensionKind.ANGLE
                and dimension.source != _AUTO_ANGLE_SOURCE
            )
            else _DEFAULT_APPLY_TEXT,
        )
        self.selection_changed.emit()

    def select_entity(self, entity_id: EntityId) -> None:
        """Select every verification row attached to one source entity."""
        self.select_entities((entity_id,))

    def select_entities(self, entity_ids: tuple[EntityId, ...]) -> None:
        """Select every verification row attached to any source entity."""
        selected_entity_ids = set(entity_ids)
        rows = (
            tuple(
                row
                for dimension_id, row in self._dimension_row_by_id.items()
                if selected_entity_ids.intersection(
                    next(
                        (
                            dimension.target_entity_ids
                            for dimension in self._analysis.dimensions
                            if dimension.dimension_id == dimension_id
                        ),
                        (),
                    ),
                )
            )
            if self._analysis is not None
            else ()
        )
        selection_model = self._dimension_table.selectionModel()
        selection_model.clearSelection()
        self._dimension_table.setCurrentCell(-1, -1)
        if not rows:
            self._on_selection_changed()
            return
        self._dimension_table.setCurrentCell(rows[0], 0)
        for row in rows:
            index = self._dimension_table.model().index(row, 0)
            selection_model.select(
                index,
                QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        self.selection_changed.emit()

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
        self._region_button.setObjectName("mark_region")
        self._region_button.setCheckable(True)
        self._region_button.setToolTip(_REGION_TOOLTIP)
        _ = self._region_button.toggled.connect(self.region_toggled)
        edit_toolbar.addWidget(self._region_button)
        self._region_clear_button.setObjectName("clear_region")
        self._region_clear_button.setToolTip("지정한 도면 영역을 해제합니다.")
        _ = self._region_clear_button.setEnabled(False)
        _ = self._region_clear_button.clicked.connect(self.region_cleared)
        edit_toolbar.addWidget(self._region_clear_button)
        self._apply_button.setObjectName("apply_dimension")
        self._apply_button.setToolTip(
            "선택한 치수 값을 검증하고 CAD geometry를 다시 계산합니다.",
        )
        _ = self._apply_button.clicked.connect(self._apply_current_dimension)
        edit_toolbar.addWidget(self._apply_button)
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
            self._on_selection_changed,
        )
        self._dimension_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._dimension_table, 1)
        self._dimension_table.setMinimumHeight(88)

    def _apply_current_dimension(self) -> None:
        """Request validation of the currently selected dimension value."""
        row = self._current_row()
        if row < 0:
            return
        item = self._dimension_table.item(row, _VALUE_COLUMN)
        if item is not None:
            self.dimension_apply_requested.emit(row, item.text())

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
