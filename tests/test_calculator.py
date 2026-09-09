from cadraw.calculator import calculate_document
from cadraw.models import (
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)


def _line() -> LineEntity:
    return LineEntity(
        entity_id=EntityId("line_1"),
        raw_start=Point(10.0, 20.0),
        raw_end=Point(110.0, 20.0),
        start=Point(10.0, 20.0),
        end=Point(110.0, 20.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.95,
    )


def test_confirmed_dimension_calibrates_all_geometry_deterministically() -> None:
    # Given a line and a user-confirmed physical length
    analysis = DrawingAnalysis(
        unit="mm",
        image_size=(160, 80),
        entities=(_line(),),
        dimensions=(
            Dimension(
                dimension_id=DimensionId("dim_1"),
                raw_text="50",
                kind=DimensionKind.LENGTH,
                parsed_value=50.0,
                unit="mm",
                target_entity_ids=(EntityId("line_1"),),
                confidence=1.0,
                status=DimensionStatus.CONFIRMED,
            ),
        ),
        connections=(),
        warnings=(),
    )

    # When the calculator receives the confirmed analysis
    document = calculate_document(analysis)

    # Then the first explicit dimension is the only scale source
    assert document.scale_mm_per_source_unit == 0.5
    assert isinstance(document.entities[0], LineEntity)
    assert document.entities[0].length == 50.0


def test_missing_dimension_does_not_use_picture_ratio() -> None:
    # Given geometry with no physical measurement
    analysis = DrawingAnalysis(
        unit="mm",
        image_size=(160, 80),
        entities=(_line(),),
        dimensions=(),
        connections=(),
        warnings=(),
    )

    # When the calculator runs
    document = calculate_document(analysis)

    # Then the source geometry stays unscaled and the missing state is visible
    assert document.scale_mm_per_source_unit is None
    assert isinstance(document.entities[0], LineEntity)
    assert document.entities[0].length == 100.0
    assert any(
        warning.code.value == "UNKNOWN_DIMENSION" for warning in document.warnings
    )


def test_conflicting_confirmed_dimension_is_reported() -> None:
    # Given two confirmed lengths that disagree with the first calibration
    second_line = LineEntity(
        entity_id=EntityId("line_2"),
        raw_start=Point(0.0, 10.0),
        raw_end=Point(80.0, 10.0),
        start=Point(0.0, 10.0),
        end=Point(80.0, 10.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.95,
    )
    analysis = DrawingAnalysis(
        unit="mm",
        image_size=(160, 80),
        entities=(_line(), second_line),
        dimensions=(
            Dimension(
                dimension_id=DimensionId("dim_1"),
                raw_text="50",
                kind=DimensionKind.LENGTH,
                parsed_value=50.0,
                unit="mm",
                target_entity_ids=(EntityId("line_1"),),
                confidence=1.0,
                status=DimensionStatus.CONFIRMED,
            ),
            Dimension(
                dimension_id=DimensionId("dim_2"),
                raw_text="100",
                kind=DimensionKind.LENGTH,
                parsed_value=100.0,
                unit="mm",
                target_entity_ids=(EntityId("line_2"),),
                confidence=1.0,
                status=DimensionStatus.CONFIRMED,
            ),
        ),
        connections=(),
        warnings=(),
    )

    # When the deterministic calculator validates both constraints
    document = calculate_document(analysis)

    # Then it keeps the first scale and exposes the mismatch
    assert document.scale_mm_per_source_unit == 0.5
    assert any(
        warning.code.value == "CONSTRAINT_MISMATCH" for warning in document.warnings
    )
