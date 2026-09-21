from cadraw.calculator import calculate_document
from cadraw.models import (
    CircleEntity,
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


def test_confirmed_line_scale_preserves_non_line_cad_scaling() -> None:
    # Given a line calibration and a detected circle in source coordinates
    analysis = DrawingAnalysis(
        unit="mm",
        image_size=(160, 80),
        entities=(
            _line(),
            CircleEntity(
                entity_id=EntityId("circle_1"),
                center=Point(20.0, 40.0),
                radius=10.0,
                confidence=1.0,
            ),
        ),
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

    # When the CAD document is recalculated
    document = calculate_document(analysis)

    # Then non-line geometry is placed through the same frame as the line:
    # the line's start is the origin, and the circle drawn below it on the
    # photograph sits at negative CAD y, ten pixels right and twenty down.
    circle = document.entities[1]
    assert isinstance(circle, CircleEntity)
    assert circle.center == Point(5.0, -10.0)
    assert circle.radius == 5.0


def _line_and_circle_analysis(diameter_dimension: Dimension) -> DrawingAnalysis:
    """Build a scaled line plus one circle with a confirmed diameter row."""
    return DrawingAnalysis(
        unit="mm",
        image_size=(160, 80),
        entities=(
            _line(),
            CircleEntity(
                entity_id=EntityId("circle_1"),
                center=Point(20.0, 40.0),
                radius=10.0,
                confidence=1.0,
            ),
        ),
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
            diameter_dimension,
        ),
        connections=(),
        warnings=(),
    )


def test_matching_confirmed_diameter_raises_no_mismatch_warning() -> None:
    # Given a confirmed diameter that agrees with the scaled circle geometry
    analysis = _line_and_circle_analysis(
        Dimension(
            dimension_id=DimensionId("dim_2"),
            raw_text="10",
            kind=DimensionKind.DIAMETER,
            parsed_value=10.0,
            unit="mm",
            target_entity_ids=(EntityId("circle_1"),),
            confidence=1.0,
            status=DimensionStatus.CONFIRMED,
        ),
    )

    # When the calculator validates the confirmed values
    document = calculate_document(analysis)

    # Then no mismatch is reported for the agreeing diameter
    assert not any(
        warning.code.value == "CONSTRAINT_MISMATCH" for warning in document.warnings
    )


def test_conflicting_confirmed_diameter_is_reported() -> None:
    # Given a confirmed diameter that disagrees with the scaled circle geometry
    analysis = _line_and_circle_analysis(
        Dimension(
            dimension_id=DimensionId("dim_2"),
            raw_text="40",
            kind=DimensionKind.DIAMETER,
            parsed_value=40.0,
            unit="mm",
            target_entity_ids=(EntityId("circle_1"),),
            confidence=1.0,
            status=DimensionStatus.CONFIRMED,
        ),
    )

    # When the calculator validates the confirmed values
    document = calculate_document(analysis)

    # Then the disagreement is surfaced instead of silently accepted
    assert any(
        warning.code.value == "CONSTRAINT_MISMATCH" for warning in document.warnings
    )
