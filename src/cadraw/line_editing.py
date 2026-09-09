"""User-directed edits for line geometry in an immutable analysis."""

from dataclasses import replace
from typing import Final

from cadraw.line_editing_support import (
    LineEditError,
    LineEditErrorCode,
    LineSelection,
    ensure_dimensions_unconfirmed,
    merge_dimensions,
    new_manual_dimension,
    next_manual_entity_id,
    resolve_lines,
    with_rebuilt_connections,
)
from cadraw.models import (
    Dimension,
    DrawingAnalysis,
    GeometryEntity,
    LineCandidate,
    Point,
)
from cadraw.normalization import (
    NormalizationConfig,
    merge_collinear_lines,
    normalize_lines,
)

_MANUAL_CONFIDENCE: Final = 1.0
_DEFAULT_CONFIG: Final = NormalizationConfig()

__all__ = [
    "LineEditError",
    "LineEditErrorCode",
    "LineSelection",
    "delete_selected_lines",
    "merge_selected_lines",
    "redraw_selected_line",
]


def merge_selected_lines(
    analysis: DrawingAnalysis,
    entity_ids: LineSelection,
    config: NormalizationConfig | None = None,
) -> DrawingAnalysis:
    """Merge selected collinear pieces into one canonical line."""
    effective_config = config if config is not None else _DEFAULT_CONFIG
    selected_lines = resolve_lines(analysis, entity_ids)
    ensure_dimensions_unconfirmed(analysis, entity_ids)

    merged_lines = merge_collinear_lines(
        tuple(sorted(selected_lines, key=lambda line: line.entity_id)),
        effective_config.line_merge_tolerance,
    )
    if len(merged_lines) != 1:
        raise LineEditError(
            LineEditErrorCode.NOT_MERGEABLE,
            "선택한 선이 하나의 같은 선으로 이어지지 않습니다.",
        )
    canonical_id = min(
        (line.entity_id for line in selected_lines),
    )
    merged_line = replace(merged_lines[0], entity_id=canonical_id)
    selected = set(entity_ids)
    edited_entities: list[GeometryEntity] = []
    for entity in analysis.entities:
        if entity.entity_id == canonical_id:
            edited_entities.append(merged_line)
        elif entity.entity_id not in selected:
            edited_entities.append(entity)
    edited_dimensions = merge_dimensions(
        analysis.dimensions,
        selected,
        canonical_id,
    )
    return with_rebuilt_connections(
        replace(analysis, dimensions=edited_dimensions),
        tuple(edited_entities),
        effective_config,
    )


def delete_selected_lines(
    analysis: DrawingAnalysis,
    entity_ids: LineSelection,
    config: NormalizationConfig | None = None,
) -> DrawingAnalysis:
    """Delete selected lines and their now-invalid manual dimensions."""
    effective_config = config if config is not None else _DEFAULT_CONFIG
    _ = resolve_lines(analysis, entity_ids)
    ensure_dimensions_unconfirmed(analysis, entity_ids)
    selected = set(entity_ids)
    edited_dimensions: list[Dimension] = []
    for dimension in analysis.dimensions:
        if not dimension.target_entity_ids:
            edited_dimensions.append(dimension)
            continue
        target_entity_ids = tuple(
            entity_id
            for entity_id in dimension.target_entity_ids
            if entity_id not in selected
        )
        if not target_entity_ids:
            continue
        edited_dimensions.append(
            replace(dimension, target_entity_ids=target_entity_ids),
        )
    edited_entities = tuple(
        entity for entity in analysis.entities if entity.entity_id not in selected
    )
    return with_rebuilt_connections(
        replace(analysis, dimensions=tuple(edited_dimensions)),
        edited_entities,
        effective_config,
    )


def redraw_selected_line(
    analysis: DrawingAnalysis,
    entity_ids: LineSelection,
    raw_start: Point,
    raw_end: Point,
    config: NormalizationConfig | None = None,
) -> DrawingAnalysis:
    """Replace selected line(s) with one drawn stroke, or add a new line.

    Selecting several lines before drawing collapses all of them into the
    single stroke the user just drew, even when they are not collinear
    enough for :func:`merge_selected_lines` to accept. This lets a source
    stroke that detection split into disconnected pieces be redrawn as one
    canonical line.
    """
    effective_config = config if config is not None else _DEFAULT_CONFIG
    selected_lines = resolve_lines(analysis, entity_ids) if entity_ids else ()
    if selected_lines:
        ensure_dimensions_unconfirmed(analysis, entity_ids)
    target_id = (
        min(line.entity_id for line in selected_lines)
        if selected_lines
        else next_manual_entity_id(analysis.entities)
    )
    candidate = LineCandidate(
        entity_id=target_id,
        start=raw_start,
        end=raw_end,
        confidence=_MANUAL_CONFIDENCE,
    )
    normalized = normalize_lines((candidate,), effective_config)
    if len(normalized) != 1:
        raise LineEditError(
            LineEditErrorCode.ZERO_LENGTH,
            "새 선은 최소 길이 이상으로 그려 주세요.",
        )
    redrawn = normalized[0]
    if selected_lines:
        selected = set(entity_ids)
        edited_entities = tuple(
            redrawn if entity.entity_id == target_id else entity
            for entity in analysis.entities
            if entity.entity_id == target_id or entity.entity_id not in selected
        )
        edited_dimensions = merge_dimensions(analysis.dimensions, selected, target_id)
    else:
        edited_entities = (*analysis.entities, redrawn)
        edited_dimensions = (
            *analysis.dimensions,
            new_manual_dimension(target_id),
        )
    return with_rebuilt_connections(
        replace(analysis, dimensions=edited_dimensions),
        edited_entities,
        effective_config,
    )
