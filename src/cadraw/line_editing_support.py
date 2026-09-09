"""Validation and immutable-state helpers for line editing commands."""

from dataclasses import replace
from enum import StrEnum
from typing import final, override

from cadraw.models import (
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
    GeometryEntity,
    LineEntity,
)
from cadraw.normalization import NormalizationConfig, build_connections

type LineSelection = tuple[EntityId, ...]


class LineEditErrorCode(StrEnum):
    """Expected reasons a user-directed line edit can be refused."""

    EMPTY_SELECTION = "empty_selection"
    DUPLICATE_SELECTION = "duplicate_selection"
    UNKNOWN_ENTITY = "unknown_entity"
    NON_LINE_ENTITY = "non_line_entity"
    NOT_MERGEABLE = "not_mergeable"
    CONFIRMED_DIMENSION = "confirmed_dimension"
    ZERO_LENGTH = "zero_length"


@final
class LineEditError(Exception):
    """Typed error for an edit that would violate geometry safety."""

    code: LineEditErrorCode
    message: str

    def __init__(self, code: LineEditErrorCode, message: str) -> None:
        """Store a machine-readable code and user-facing explanation."""
        super().__init__(message)
        self.code = code
        self.message = message

    @override
    def __str__(self) -> str:
        """Return the explanation suitable for the status bar."""
        return self.message


def resolve_lines(
    analysis: DrawingAnalysis,
    entity_ids: LineSelection,
) -> tuple[LineEntity, ...]:
    """Resolve and validate a stable selection of line IDs."""
    if not entity_ids:
        raise LineEditError(
            LineEditErrorCode.EMPTY_SELECTION,
            "편집할 선을 하나 이상 선택해 주세요.",
        )
    if len(set(entity_ids)) != len(entity_ids):
        raise LineEditError(
            LineEditErrorCode.DUPLICATE_SELECTION,
            "같은 선을 중복해서 편집할 수 없습니다.",
        )
    entities_by_id = {entity.entity_id: entity for entity in analysis.entities}
    lines: list[LineEntity] = []
    for entity_id in entity_ids:
        entity = entities_by_id.get(entity_id)
        if entity is None:
            raise LineEditError(
                LineEditErrorCode.UNKNOWN_ENTITY,
                "선택한 geometry를 찾을 수 없습니다.",
            )
        if not isinstance(entity, LineEntity):
            raise LineEditError(
                LineEditErrorCode.NON_LINE_ENTITY,
                "현재 편집 버튼은 선 geometry에만 사용할 수 있습니다.",
            )
        lines.append(entity)
    return tuple(lines)


def ensure_dimensions_unconfirmed(
    analysis: DrawingAnalysis,
    entity_ids: LineSelection,
) -> None:
    """Prevent merge/delete from silently discarding physical values."""
    selected = set(entity_ids)
    if any(
        dimension.parsed_value is not None
        and any(target_id in selected for target_id in dimension.target_entity_ids)
        for dimension in analysis.dimensions
    ):
        raise LineEditError(
            LineEditErrorCode.CONFIRMED_DIMENSION,
            "확정 치수가 있는 선은 값을 지운 뒤 편집할 수 있습니다.",
        )


def with_rebuilt_connections(
    analysis: DrawingAnalysis,
    entities: tuple[GeometryEntity, ...],
    config: NormalizationConfig,
) -> DrawingAnalysis:
    """Replace entities and derive connections from the edited line set."""
    lines = tuple(entity for entity in entities if isinstance(entity, LineEntity))
    return replace(
        analysis,
        entities=entities,
        connections=build_connections(lines, config.endpoint_snap_distance),
    )


def _remap_targets(
    target_entity_ids: tuple[EntityId, ...],
    selected: set[EntityId],
    canonical_id: EntityId,
) -> tuple[EntityId, ...]:
    """Replace selected IDs once while preserving unrelated target IDs."""
    remapped: list[EntityId] = []
    for entity_id in target_entity_ids:
        effective_id = canonical_id if entity_id in selected else entity_id
        if effective_id not in remapped:
            remapped.append(effective_id)
    return tuple(remapped)


def merge_dimensions(
    dimensions: tuple[Dimension, ...],
    selected: set[EntityId],
    canonical_id: EntityId,
) -> tuple[Dimension, ...]:
    """Collapse unresolved rows that now target the merged line."""
    merged_row_kept = False
    edited: list[Dimension] = []
    for dimension in dimensions:
        if not any(entity_id in selected for entity_id in dimension.target_entity_ids):
            edited.append(dimension)
            continue
        target_entity_ids = _remap_targets(
            dimension.target_entity_ids,
            selected,
            canonical_id,
        )
        if target_entity_ids == (canonical_id,):
            if merged_row_kept:
                continue
            merged_row_kept = True
        edited.append(replace(dimension, target_entity_ids=target_entity_ids))
    return tuple(edited)


def next_manual_entity_id(
    entities: tuple[GeometryEntity, ...],
) -> EntityId:
    """Choose the first unused deterministic identity for a new stroke."""
    used_ids = {entity.entity_id for entity in entities}
    for index in range(1, len(entities) + 2):
        candidate = EntityId(f"manual_line_{index:03d}")
        if candidate not in used_ids:
            return candidate
    raise AssertionError


def new_manual_dimension(entity_id: EntityId) -> Dimension:
    """Create an unresolved physical-length row for a redrawn line."""
    return Dimension(
        dimension_id=DimensionId(f"manual_{entity_id}"),
        raw_text="",
        kind=DimensionKind.LENGTH,
        parsed_value=None,
        unit="mm",
        target_entity_ids=(entity_id,),
        confidence=0.0,
        status=DimensionStatus.UNKNOWN,
    )
