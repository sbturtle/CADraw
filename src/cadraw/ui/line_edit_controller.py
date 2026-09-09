"""Line-edit command coordination for the workbench view."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, final

from cadraw.line_editing import (
    LineEditError,
    delete_selected_lines,
    merge_selected_lines,
    redraw_selected_line,
)
from cadraw.models import DimensionId, DrawingAnalysis, EntityId, Point
from cadraw.normalization import NormalizationConfig

_MANUAL_EDIT_CONFIG: Final = NormalizationConfig()

type AnalysisProvider = Callable[[], DrawingAnalysis | None]
type EntitySelectionProvider = Callable[[], tuple[EntityId, ...]]
type AnalysisInstaller = Callable[
    [DrawingAnalysis, str, DimensionId | None],
    None,
]
type CaptionSetter = Callable[[str], None]
type BoolSetter = Callable[[bool], None]


@dataclass(frozen=True, slots=True)
class LineEditCallbacks:
    """Application callbacks needed by line-edit commands."""

    analysis_provider: AnalysisProvider
    entity_selection_provider: EntitySelectionProvider
    install_analysis: AnalysisInstaller
    set_caption: CaptionSetter
    set_redraw_checked: BoolSetter
    set_source_drawing_enabled: BoolSetter


@final
class LineEditController:
    """Run immutable line commands without coupling them to MainWindow."""

    def __init__(
        self,
        callbacks: LineEditCallbacks,
    ) -> None:
        """Store application callbacks at the UI boundary."""
        self._callbacks = callbacks

    def merge_selected_lines(self) -> None:
        """Merge selected near-collinear source lines."""
        analysis = self._callbacks.analysis_provider()
        selected_entity_ids = self._callbacks.entity_selection_provider()
        if analysis is None:
            return
        try:
            edited = merge_selected_lines(
                analysis,
                selected_entity_ids,
                _MANUAL_EDIT_CONFIG,
            )
        except LineEditError as error:
            self._callbacks.set_caption(f"병합할 수 없음 · {error}")
            return
        canonical_id = min(selected_entity_ids)
        selected_dimension_id = next(
            (
                dimension.dimension_id
                for dimension in edited.dimensions
                if canonical_id in dimension.target_entity_ids
            ),
            None,
        )
        self._callbacks.install_analysis(
            edited,
            f"선 {len(selected_entity_ids)}개를 병합했습니다.",
            selected_dimension_id,
        )

    def delete_selected_lines(self) -> None:
        """Delete selected source lines and unresolved rows."""
        analysis = self._callbacks.analysis_provider()
        selected_entity_ids = self._callbacks.entity_selection_provider()
        if analysis is None:
            return
        try:
            edited = delete_selected_lines(
                analysis,
                selected_entity_ids,
                _MANUAL_EDIT_CONFIG,
            )
        except LineEditError as error:
            self._callbacks.set_caption(f"삭제할 수 없음 · {error}")
            return
        self._callbacks.install_analysis(
            edited,
            f"선 {len(selected_entity_ids)}개를 삭제했습니다.",
            None,
        )

    def set_redraw_mode(self, checked: bool) -> None:
        """Enter or leave source-image drawing mode."""
        if self._callbacks.analysis_provider() is None:
            return
        self._callbacks.set_source_drawing_enabled(checked)
        if checked:
            self._callbacks.set_caption(
                "원본 이미지에서 선의 시작점과 끝점을 드래그하세요.",
            )
        else:
            self._callbacks.set_caption("다시 그리기를 취소했습니다.")

    def install_drawn_line(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> None:
        """Replace one selected line or add the freshly drawn line."""
        analysis = self._callbacks.analysis_provider()
        if analysis is None:
            return
        selected_entity_ids = self._callbacks.entity_selection_provider()
        try:
            edited = redraw_selected_line(
                analysis,
                selected_entity_ids,
                Point(start_x, start_y),
                Point(end_x, end_y),
                _MANUAL_EDIT_CONFIG,
            )
        except LineEditError as error:
            self._callbacks.set_caption(f"다시 그릴 수 없음 · {error}")
            return
        target_id = (
            min(selected_entity_ids)
            if selected_entity_ids
            else edited.entities[-1].entity_id
        )
        selected_dimension_id = next(
            (
                dimension.dimension_id
                for dimension in edited.dimensions
                if target_id in dimension.target_entity_ids
            ),
            None,
        )
        unchecked = False
        self._callbacks.set_redraw_checked(unchecked)
        status = (
            f"선 {len(selected_entity_ids)}개를 하나로 합쳐 다시 그렸습니다."
            if len(selected_entity_ids) > 1
            else "선 1개를 다시 그렸습니다."
        )
        self._callbacks.install_analysis(
            edited,
            status,
            selected_dimension_id,
        )
