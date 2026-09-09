"""Deterministic calibration, validation, and coordinate transformation."""

from dataclasses import dataclass, replace

from cadraw.models import (
    AnalysisWarning,
    ArcEntity,
    CadDocument,
    CircleEntity,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    GeometryEntity,
    LineEntity,
    PolylineEntity,
    WarningCode,
)


@dataclass(frozen=True, slots=True)
class CalculatorConfig:
    """Tolerances for validating explicit dimensions."""

    validation_tolerance_ratio: float = 0.02
    minimum_validation_tolerance_mm: float = 0.5


def _find_scale(
    analysis: DrawingAnalysis,
) -> float | None:
    """Find the first valid confirmed line-length calibration."""
    entities = {entity.entity_id: entity for entity in analysis.entities}
    for dimension in analysis.dimensions:
        if dimension.status is not DimensionStatus.CONFIRMED:
            continue
        if dimension.kind is not DimensionKind.LENGTH:
            continue
        if dimension.parsed_value is None or len(dimension.target_entity_ids) != 1:
            continue
        target = entities.get(dimension.target_entity_ids[0])
        if not isinstance(target, LineEntity) or target.length == 0.0:
            continue
        return dimension.parsed_value / target.length
    return None


def _scale_entity(entity: GeometryEntity, scale: float) -> GeometryEntity:
    """Scale one known entity while preserving its identity and provenance."""
    match entity:
        case LineEntity(start=start, end=end):
            return replace(entity, start=start.scaled(scale), end=end.scaled(scale))
        case PolylineEntity(points=points):
            return replace(
                entity,
                points=tuple(point.scaled(scale) for point in points),
            )
        case CircleEntity(center=center, radius=radius):
            return replace(
                entity,
                center=center.scaled(scale),
                radius=radius * scale,
            )
        case ArcEntity(center=center, radius=radius):
            return replace(
                entity,
                center=center.scaled(scale),
                radius=radius * scale,
            )


def _has_warning(
    warnings: list[AnalysisWarning],
    code: WarningCode,
) -> bool:
    """Check whether a warning code is already present."""
    return any(warning.code is code for warning in warnings)


def _append_dimension_warnings(
    analysis: DrawingAnalysis,
    warnings: list[AnalysisWarning],
) -> None:
    """Report unresolved and unassociated dimensions without guessing."""
    entities = {entity.entity_id for entity in analysis.entities}
    for dimension in analysis.dimensions:
        if dimension.parsed_value is None and not _has_warning(
            warnings,
            WarningCode.UNKNOWN_DIMENSION,
        ):
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.UNKNOWN_DIMENSION,
                    message="읽을 수 없는 치수는 사용자 확인이 필요합니다.",
                ),
            )
        if any(entity_id not in entities for entity_id in dimension.target_entity_ids):
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.UNCONNECTED_DIMENSION,
                    message="치수와 geometry 연결을 확인해 주세요.",
                    entity_ids=dimension.target_entity_ids,
                ),
            )


def _append_constraint_warnings(
    analysis: DrawingAnalysis,
    scale: float,
    config: CalculatorConfig,
    warnings: list[AnalysisWarning],
) -> None:
    """Compare every confirmed line length to the first scale."""
    entities = {entity.entity_id: entity for entity in analysis.entities}
    for dimension in analysis.dimensions:
        if dimension.status is not DimensionStatus.CONFIRMED:
            continue
        if dimension.kind is not DimensionKind.LENGTH:
            continue
        if dimension.parsed_value is None or len(dimension.target_entity_ids) != 1:
            continue
        target = entities.get(dimension.target_entity_ids[0])
        if not isinstance(target, LineEntity):
            continue
        expected = target.length * scale
        tolerance = max(
            config.minimum_validation_tolerance_mm,
            dimension.parsed_value * config.validation_tolerance_ratio,
        )
        if abs(expected - dimension.parsed_value) <= tolerance:
            continue
        warnings.append(
            AnalysisWarning(
                code=WarningCode.CONSTRAINT_MISMATCH,
                message="확정 치수와 geometry 길이가 일치하지 않습니다.",
                entity_ids=dimension.target_entity_ids,
            ),
        )


def calculate_document(
    analysis: DrawingAnalysis,
    config: CalculatorConfig | None = None,
) -> CadDocument:
    """Create exportable geometry from only explicit user-confirmed values."""
    warnings = list(analysis.warnings)
    effective_config = config if config is not None else CalculatorConfig()
    calibration = _find_scale(analysis)
    _append_dimension_warnings(analysis, warnings)
    if calibration is None:
        if not _has_warning(warnings, WarningCode.UNKNOWN_DIMENSION):
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.UNKNOWN_DIMENSION,
                    message="확정된 길이가 없어 local unit으로 유지합니다.",
                ),
            )
        return CadDocument(
            unit=analysis.unit,
            entities=analysis.entities,
            dimensions=analysis.dimensions,
            warnings=tuple(warnings),
            scale_mm_per_source_unit=None,
        )

    scale = calibration
    _append_constraint_warnings(analysis, scale, effective_config, warnings)
    return CadDocument(
        unit=analysis.unit,
        entities=tuple(_scale_entity(entity, scale) for entity in analysis.entities),
        dimensions=analysis.dimensions,
        warnings=tuple(warnings),
        scale_mm_per_source_unit=scale,
    )
