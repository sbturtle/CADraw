"""Main human-in-the-loop workbench window."""

from dataclasses import replace
from pathlib import Path
from typing import override

from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QMainWindow, QMessageBox, QWidget

from cadraw.calculator import calculate_document
from cadraw.detection import ImageReadError
from cadraw.dimensions import confirm_dimension_text
from cadraw.models import (
    CadDocument,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
)
from cadraw.pipeline import PipelineConfig
from cadraw.preview import PreviewConfig, PreviewMode, render_preview
from cadraw.ui.dialog_actions import DialogActions
from cadraw.ui.dimension_rows import manual_dimensions
from cadraw.ui.document_actions import calculation_caption, save_document
from cadraw.ui.line_edit_controller import (
    LineEditCallbacks,
    LineEditController,
)
from cadraw.ui.source_rendering import refresh_source_view
from cadraw.ui.verification_labels import (
    selected_dimension_label,
    selected_entity_info,
)
from cadraw.ui.workbench_view import WorkbenchView
from cadraw.vision import MockVisionAnalyzer, VisionAnalyzer

_MIN_MERGE_SELECTION = 2
_MIN_DELETE_SELECTION = 1


class MainWindow(QMainWindow):
    """Coordinate domain state with the composed native workbench view."""

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
        self._selected_dimension_id: DimensionId | None = None
        self._preview_mode: PreviewMode = PreviewMode.CAD
        self._view: WorkbenchView = WorkbenchView(self)
        self._dialog_actions: DialogActions = DialogActions(
            self,
            self.load_path,
            self.save_path,
        )
        self._line_edit_controller: LineEditController = LineEditController(
            LineEditCallbacks(
                analysis_provider=lambda: self._analysis,
                entity_selection_provider=lambda: self._view.selected_entity_ids,
                install_analysis=self._apply_line_edit,
                set_caption=self._view.set_caption,
                set_redraw_checked=self._view.set_redraw_checked,
                set_source_drawing_enabled=self._view.set_source_drawing_enabled,
            ),
        )
        self._connect_view()
        self.setWindowTitle("CADraw · 손도면 변환")
        self.setCentralWidget(self._view)

    @property
    def analysis(self) -> DrawingAnalysis | None:
        """Return the latest intermediate analysis for smoke tests and callers."""
        return self._analysis

    @property
    def document(self) -> CadDocument | None:
        """Return the latest recalculated CAD document."""
        return self._document

    @property
    def selected_dimension_id(self) -> DimensionId | None:
        """Return the dimension row currently linked to the selection."""
        return self._selected_dimension_id

    @property
    def selected_entity_ids(self) -> tuple[EntityId, ...]:
        """Return all line IDs currently selected in the verification table."""
        return self._view.selected_entity_ids

    def _connect_view(self) -> None:
        """Connect view events to deterministic application actions."""
        _ = self._view.load_requested.connect(self._dialog_actions.choose_image)
        _ = self._view.recalculate_requested.connect(self._recalculate)
        _ = self._view.save_requested.connect(
            self._dialog_actions.choose_save_path,
        )
        _ = self._view.dimension_edited.connect(self._on_dimension_edited)
        _ = self._view.selection_changed.connect(self._on_dimension_selected)
        _ = self._view.preview_selection_changed.connect(
            self._on_entities_selected,
        )
        _ = self._view.dimension_apply_requested.connect(
            self._on_dimension_apply_requested,
        )
        _ = self._view.preview_mode_changed.connect(self._on_preview_mode_changed)
        _ = self._view.merge_requested.connect(
            self._line_edit_controller.merge_selected_lines,
        )
        _ = self._view.delete_requested.connect(
            self._line_edit_controller.delete_selected_lines,
        )
        _ = self._view.redraw_toggled.connect(
            self._line_edit_controller.set_redraw_mode,
        )
        _ = self._view.source_line_drawn.connect(
            self._line_edit_controller.install_drawn_line,
        )

    def load_path(self, image_path: Path) -> bool:
        """Load and analyze one image through the production path."""
        try:
            analysis = self._analyzer.analyze(image_path)
        except ImageReadError as error:
            self._view.set_caption(str(error))
            return False
        self._selected_dimension_id = None
        self._view.set_redraw_checked(False)
        self._analysis = replace(
            analysis,
            dimensions=manual_dimensions(
                analysis.entities,
                analysis.dimensions,
            ),
        )
        self._source_pixmap = QPixmap(str(image_path))
        self._refresh_source_pixmap()
        self._view.set_main_actions_enabled(recalculate=True, save=True)
        self._refresh_dimension_table()
        self._recalculate()
        pending_angle_count = sum(
            dimension.kind is DimensionKind.ANGLE
            and dimension.status is DimensionStatus.UNSET
            for dimension in self._analysis.dimensions
        )
        angle_prompt = (
            f" 대각선 각도 {pending_angle_count}개는 입력 후 승인해 주세요."
            if pending_angle_count
            else ""
        )
        load_caption = (
            f"선 {len(self._analysis.entities)}개를 찾았습니다. 치수를 확인해 주세요."
        )
        self._view.set_caption(
            f"{load_caption}{angle_prompt}",
        )
        return True

    def set_dimension_text(self, row: int, text: str) -> None:
        """Set a verification value through the real table widget."""
        self._view.set_dimension_text(row, text)

    def _refresh_source_pixmap(self) -> None:
        """Fit the source image and update its selected-line overlay."""
        refresh_source_view(self._view, self._source_pixmap, self._analysis)

    def _refresh_dimension_table(self) -> None:
        """Render analysis dimensions while retaining stable row selection."""
        if self._analysis is None:
            return
        selected_dimension_ids = self._view.selected_dimension_ids
        if (
            self._selected_dimension_id is not None
            and self._selected_dimension_id not in selected_dimension_ids
        ):
            selected_dimension_ids = (
                *selected_dimension_ids,
                self._selected_dimension_id,
            )
        self._view.set_analysis(self._analysis, selected_dimension_ids)
        self._update_edit_buttons()

    def _on_dimension_edited(self, row: int, text: str) -> None:
        """Stage angle input while applying lengths directly for continuity."""
        current = self._dimension_for_row(row)
        if current is None:
            return
        if current.kind is DimensionKind.ANGLE:
            self._view.set_caption(
                """각도 입력 대기 · 값을 입력한 뒤
'각도 승인 후 다시 그리기'를 누르세요.""",
            )
            return
        self._apply_dimension_text(current, text)

    def _on_dimension_apply_requested(self, row: int, text: str) -> None:
        """Ask explicit permission before rebuilding geometry from an angle."""
        current = self._dimension_for_row(row)
        if current is None:
            return
        if current.kind is DimensionKind.ANGLE:
            if current.source == "axis_inference":
                self._view.set_caption("축 정렬 각도는 자동 확정되어 있습니다.")
                return
            if not text.strip():
                self._view.set_caption("각도를 입력한 뒤 다시 그릴 수 있습니다.")
                return
            answer = QMessageBox.question(
                self,
                "각도 반영 확인",
                f"{text} 각도로 geometry를 다시 그릴까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer is not QMessageBox.StandardButton.Yes:
                self._refresh_dimension_table()
                self._view.set_caption("각도 적용을 취소했습니다.")
                return
        self._apply_dimension_text(current, text)

    def _dimension_for_row(self, row: int) -> Dimension | None:
        """Resolve the current table row to its immutable dimension."""
        if self._analysis is None:
            return None
        dimension_id = self._view.dimension_id_for_row(row)
        if dimension_id is None:
            return None
        return next(
            (
                dimension
                for dimension in self._analysis.dimensions
                if dimension.dimension_id == dimension_id
            ),
            None,
        )

    def _apply_dimension_text(
        self,
        current: Dimension,
        text: str,
    ) -> None:
        """Confirm one value and rebuild every dependent CAD surface."""
        if self._analysis is None:
            return
        dimension_id = current.dimension_id
        candidate_text = text
        if (
            current.kind is DimensionKind.ANGLE
            and candidate_text.strip()
            and not candidate_text.strip()
            .lower()
            .endswith(
                ("°", "deg", "degree", "degrees"),
            )
        ):
            candidate_text = f"{candidate_text}°"
        confirmed = confirm_dimension_text(current, candidate_text)
        if confirmed.status is DimensionStatus.INVALID:
            self._refresh_dimension_table()
            self._view.set_caption(
                "입력 오류 · 길이는 0보다 크고 각도는 0° 초과 180° 이하여야 합니다.",
            )
            return
        dimensions = tuple(
            confirmed if dimension.dimension_id == dimension_id else dimension
            for dimension in self._analysis.dimensions
        )
        self._analysis = replace(self._analysis, dimensions=dimensions)
        self._selected_dimension_id = dimension_id
        self._refresh_dimension_table()
        self._recalculate()
        self._refresh_selection_info()

    def _on_entities_selected(self, entity_ids: tuple[str, ...]) -> None:
        """Select verification rows linked to the complete preview selection."""
        self._view.select_entities(
            tuple(EntityId(entity_id) for entity_id in entity_ids),
        )

    def _on_preview_mode_changed(self, mode: str) -> None:
        """Switch rendering between image-space detection and CAD-space."""
        self._preview_mode = PreviewMode(mode)
        self._render_preview()
        self._view.set_caption(
            "CAD 결과 보기"
            if self._preview_mode is PreviewMode.CAD
            else "검출 결과 보기",
        )

    def _on_dimension_selected(self) -> None:
        """Link selected verification rows to their source geometry."""
        self._selected_dimension_id = self._view.selected_dimension_id
        self._render_preview()
        self._refresh_source_pixmap()
        self._refresh_selection_info()
        self._update_edit_buttons()
        if self._selected_dimension_id is None or self._analysis is None:
            return
        dimension = next(
            (
                item
                for item in self._analysis.dimensions
                if item.dimension_id == self._selected_dimension_id
            ),
            None,
        )
        if dimension is None:
            return
        selection_count = len(self._view.selected_entity_ids)
        selection_text = (
            f"{selection_count}개 선 선택"
            if selection_count > 1
            else selected_dimension_label(self._analysis, dimension)
        )
        self._view.set_caption(f"선택됨 · {selection_text}")

    def _refresh_selection_info(self) -> None:
        """Show the selected line's identity, source length, and CAD state."""
        if self._analysis is None or self._document is None:
            return
        selected_entity_ids = self._view.selected_entity_ids
        if not selected_entity_ids:
            self._view.set_selection_info(
                """클릭: 한 선 · Ctrl+클릭: 복수 선택
좌→우: 전체 포함 · 우→좌: 교차 포함""",
            )
            return
        if len(selected_entity_ids) != 1:
            selection_text = (
                f"{len(selected_entity_ids)}개 선이 선택되었습니다. "
                "하나의 선을 클릭하면 상세 정보가 표시됩니다."
            )
            self._view.set_selection_info(
                selection_text,
            )
            return
        self._view.set_selection_info(
            selected_entity_info(
                self._analysis,
                self._document,
                selected_entity_ids[0],
            ),
        )

    def _update_edit_buttons(self) -> None:
        """Enable line actions only when their selection contract is met."""
        selected_count = len(self._view.selected_entity_ids)
        has_analysis = self._analysis is not None
        self._view.set_line_actions_enabled(
            has_analysis and selected_count >= _MIN_MERGE_SELECTION,
            has_analysis and selected_count >= _MIN_DELETE_SELECTION,
            has_analysis,
        )

    def _apply_line_edit(
        self,
        analysis: DrawingAnalysis,
        status: str,
        selected_dimension_id: DimensionId | None = None,
    ) -> None:
        """Install an edited analysis and rebuild dependent surfaces."""
        self._analysis = replace(
            analysis,
            dimensions=manual_dimensions(
                analysis.entities,
                analysis.dimensions,
            ),
        )
        self._selected_dimension_id = selected_dimension_id
        self._refresh_dimension_table()
        self._recalculate()
        self._refresh_source_pixmap()
        self._update_edit_buttons()
        self._view.set_caption(status)

    def _render_preview(self) -> None:
        """Render the current document with linked geometry selected."""
        if self._document is None or self._analysis is None:
            return
        config = PreviewConfig(
            width=720,
            height=520,
            highlight_entity_ids=self._view.selected_entity_ids,
            invert_y=self._preview_mode is PreviewMode.CAD,
        )
        preview_document = self._document
        if self._preview_mode is PreviewMode.DETECTED:
            preview_document = CadDocument(
                unit=self._analysis.unit,
                entities=self._analysis.entities,
                dimensions=self._analysis.dimensions,
                warnings=self._analysis.warnings,
                scale_mm_per_source_unit=None,
            )
        preview = render_preview(
            preview_document,
            config,
        )
        self._view.set_preview_context(preview_document, config)
        self._view.set_preview_image(preview)

    def _recalculate(self) -> None:
        """Rebuild the preview from the latest verified intermediate model."""
        if self._analysis is None:
            return
        self._document = calculate_document(self._analysis)
        self._render_preview()
        self._view.set_caption(calculation_caption(self._document))

    def save_path(self, output_path: Path) -> bool:
        """Export and validate a DXF at an explicit path."""
        return save_document(
            self._document,
            output_path,
            self._view.set_caption,
        )

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the source image fitted after a window resize."""
        super().resizeEvent(event)
        self._refresh_source_pixmap()
