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


def test_line_entity_exposes_normalized_length() -> None:
    # Given a normalized vertical line in local drawing coordinates
    line = LineEntity(
        entity_id=EntityId("line_1"),
        raw_start=Point(10.0, 2.0),
        raw_end=Point(12.0, 102.0),
        start=Point(11.0, 2.0),
        end=Point(11.0, 102.0),
        orientation=Orientation.VERTICAL,
        confidence=0.95,
    )

    # When its length is requested
    # Then the normalized distance is returned, not the raw skewed distance
    assert line.length == 100.0


def test_unknown_dimension_remains_unresolved() -> None:
    # Given a dimension with no readable value or target
    dimension = Dimension(
        dimension_id=DimensionId("dim_1"),
        raw_text="?",
        kind=DimensionKind.LENGTH,
        parsed_value=None,
        unit="mm",
        target_entity_ids=(),
        confidence=0.2,
        status=DimensionStatus.UNKNOWN,
    )

    # When it is placed into an analysis
    analysis = DrawingAnalysis(
        unit="mm",
        image_size=(640, 480),
        entities=(),
        dimensions=(dimension,),
        connections=(),
        warnings=(),
    )

    # Then the unknown state is preserved for human verification
    assert analysis.dimensions[0].parsed_value is None
    assert analysis.dimensions[0].status is DimensionStatus.UNKNOWN
