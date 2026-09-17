"""Deterministic calibration, validation, and coordinate transformation."""

from dataclasses import dataclass, replace

from cadraw.models import (
    AnalysisWarning,
    ArcEntity,
    CadDocument,
    CadEntity,
    CadResolutionStatus,
    CircleEntity,
    Dimension,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    GeometryEntity,
    LineEntity,
    PolylineEntity,
    WarningCode,
)
from cadraw.solver import reconstruct_geometry


@dataclass(frozen=True, slots=True)
class CalculatorConfig:
    """Tolerances for validating explicit dimensions."""

    validation_tolerance_ratio: float = 0.02
    minimum_validation_tolerance_mm: float = 0.5


def _scale_geometry(geometry: GeometryEntity, scale: float) -> GeometryEntity:
    """Keep legacy non-line geometry in the calibrated CAD coordinate space."""
    if isinstance(geometry, PolylineEntity):
        return replace(
            geometry,
            points=tuple(point.scaled(scale) for point in geometry.points),
        )
    if isinstance(geometry, (CircleEntity, ArcEntity)):
        return replace(
            geometry,
            center=geometry.center.scaled(scale),
            radius=geometry.radius * scale,
        )
    return geometry


def _scale_cad_entities(
    entities: tuple[CadEntity, ...],
    scale: float | None,
) -> tuple[CadEntity, ...]:
    """Apply the existing global fallback only to non-line geometry."""
    if scale is None:
        return entities
    return tuple(
        replace(
            entity,
            geometry=_scale_geometry(entity.geometry, scale),
        )
        for entity in entities
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
    connected_ids = {
        entity_id
        for connection in analysis.connections
        for entity_id in (
            connection.first_entity_id,
            connection.second_entity_id,
        )
    }
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
        if target.entity_id in connected_ids:
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


def _expected_circular_value(
    dimension: Dimension,
    target: CircleEntity | ArcEntity,
    scale: float,
) -> float:
    """Return the physical value implied by scaled circle/arc geometry."""
    diameter_factor = 2.0 if dimension.kind is DimensionKind.DIAMETER else 1.0
    return target.radius * scale * diameter_factor


def _append_circular_constraint_warnings(
    analysis: DrawingAnalysis,
    scale: float,
    config: CalculatorConfig,
    warnings: list[AnalysisWarning],
) -> None:
    """Compare confirmed diameter/radius dimensions to detected geometry."""
    entities = {entity.entity_id: entity for entity in analysis.entities}
    for dimension in analysis.dimensions:
        if dimension.status is not DimensionStatus.CONFIRMED:
            continue
        if dimension.kind not in (DimensionKind.DIAMETER, DimensionKind.RADIUS):
            continue
        if dimension.parsed_value is None or len(dimension.target_entity_ids) != 1:
            continue
        target = entities.get(dimension.target_entity_ids[0])
        if not isinstance(target, (CircleEntity, ArcEntity)):
            continue
        if dimension.kind is DimensionKind.DIAMETER and not isinstance(
            target,
            CircleEntity,
        ):
            continue
        expected = _expected_circular_value(dimension, target, scale)
        tolerance = max(
            config.minimum_validation_tolerance_mm,
            dimension.parsed_value * config.validation_tolerance_ratio,
        )
        if abs(expected - dimension.parsed_value) <= tolerance:
            continue
        warnings.append(
            AnalysisWarning(
                code=WarningCode.CONSTRAINT_MISMATCH,
                message="확정 치수와 geometry 크기가 일치하지 않습니다.",
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
    reconstruction = reconstruct_geometry(analysis)
    calibration = reconstruction.scale_mm_per_source_unit
    cad_entities = _scale_cad_entities(reconstruction.entities, calibration)
    _append_dimension_warnings(analysis, warnings)
    if any(
        entity.status is CadResolutionStatus.UNKNOWN_ANGLE
        for entity in reconstruction.entities
    ):
        warnings.append(
            AnalysisWarning(
                code=WarningCode.UNKNOWN_ANGLE,
                message="대각선 각도가 없어 원본 방향을 preview에만 사용합니다.",
            ),
        )
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
            entities=tuple(entity.geometry for entity in cad_entities),
            dimensions=analysis.dimensions,
            warnings=tuple(warnings),
            scale_mm_per_source_unit=None,
            cad_entities=cad_entities,
            analysis_status=analysis.status,
        )

    scale = calibration
    _append_constraint_warnings(analysis, scale, effective_config, warnings)
    _append_circular_constraint_warnings(analysis, scale, effective_config, warnings)
    return CadDocument(
        unit=analysis.unit,
        entities=tuple(entity.geometry for entity in cad_entities),
        dimensions=analysis.dimensions,
        warnings=tuple(warnings),
        scale_mm_per_source_unit=scale,
        cad_entities=cad_entities,
        analysis_status=analysis.status,
    )
