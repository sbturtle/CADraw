"""Main human-in-the-loop workbench window."""

from dataclasses import replace
from pathlib import Path
from typing import override

from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QMainWindow, QWidget

from cadraw.calculator import calculate_document
from cadraw.detection import ImageReadError
from cadraw.dimensions import confirm_dimension_text
from cadraw.models import (
    CadDocument,
    DimensionId,
    DrawingAnalysis,
    EntityId,
)
from cadraw.pipeline import PipelineConfig
from cadraw.preview import PreviewConfig, render_preview
from cadraw.ui.dialog_actions import DialogActions
from cadraw.ui.dimension_rows import manual_dimensions
from cadraw.ui.document_actions import calculation_caption, save_document
from cadraw.ui.line_edit_controller import (
    LineEditCallbacks,
    LineEditController,
)
from cadraw.ui.source_rendering import refresh_source_view
from cadraw.ui.verification_labels import selected_dimension_label
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
            dimensions=manual_dimensions(analysis.entities),
        )
        self._source_pixmap = QPixmap(str(image_path))
        self._refresh_source_pixmap()
        self._view.set_main_actions_enabled(recalculate=True, save=True)
        self._refresh_dimension_table()
        self._recalculate()
        self._view.set_caption(
            f"선 {len(self._analysis.entities)}개를 찾았습니다. 치수를 확인해 주세요.",
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
        """Parse each user edit and keep invalid values visibly unresolved."""
        if self._analysis is None:
            return
        dimension_id = self._view.dimension_id_for_row(row)
        if dimension_id is None:
            return
        dimensions = tuple(
            confirm_dimension_text(dimension, text)
            if dimension.dimension_id == dimension_id
            else dimension
            for dimension in self._analysis.dimensions
        )
        self._analysis = replace(self._analysis, dimensions=dimensions)
        self._selected_dimension_id = dimension_id
        self._refresh_dimension_table()
        self._recalculate()

    def _on_dimension_selected(self) -> None:
        """Link selected verification rows to their source geometry."""
        self._selected_dimension_id = self._view.selected_dimension_id
        self._render_preview()
        self._refresh_source_pixmap()
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
        self._analysis = analysis
        self._selected_dimension_id = selected_dimension_id
        self._refresh_dimension_table()
        self._recalculate()
        self._refresh_source_pixmap()
        self._update_edit_buttons()
        self._view.set_caption(status)

    def _render_preview(self) -> None:
        """Render the current document with linked geometry selected."""
        if self._document is None:
            return
        preview = render_preview(
            self._document,
            PreviewConfig(
                width=720,
                height=520,
                highlight_entity_ids=self._view.selected_entity_ids,
            ),
        )
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
