from cadraw.association import suggest_dimension_target
from cadraw.models import (
    CircleEntity,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)


def _length_dimension(dimension_id: str, raw_text: str, value: float) -> Dimension:
    """Build an unconfirmed OCR-parsed length dimension fixture."""
    return Dimension(
        dimension_id=DimensionId(dimension_id),
        raw_text=raw_text,
        kind=DimensionKind.LENGTH,
        parsed_value=value,
        unit="mm",
        target_entity_ids=(),
        confidence=0.98,
        status=DimensionStatus.NEEDS_REVIEW,
        source="ocr",
    )


def test_suggests_the_single_nearby_line() -> None:
    # Given one line and an OCR label positioned right beside it
    line = LineEntity(
        entity_id=EntityId("line_a"),
        raw_start=Point(40.0, 50.0),
        raw_end=Point(280.0, 50.0),
        start=Point(40.0, 50.0),
        end=Point(280.0, 50.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )
    dimension = _length_dimension("ocr_001", "150", 150.0)

    # When a target suggestion is requested
    suggested = suggest_dimension_target(
        dimension,
        (150.0, 30.0, 40.0, 16.0),
        (line,),
    )

    # Then the single unambiguous line is linked without confirming it
    assert suggested.target_entity_ids == (EntityId("line_a"),)
    assert suggested.source == "ocr_suggested"
    assert suggested.status is DimensionStatus.NEEDS_REVIEW


def test_leaves_two_equally_close_lines_untargeted() -> None:
    # Given two parallel lines equally close to one OCR label
    first = LineEntity(
        entity_id=EntityId("line_top"),
        raw_start=Point(40.0, 50.0),
        raw_end=Point(280.0, 50.0),
        start=Point(40.0, 50.0),
        end=Point(280.0, 50.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )
    second = LineEntity(
        entity_id=EntityId("line_bottom"),
        raw_start=Point(40.0, 140.0),
        raw_end=Point(280.0, 140.0),
        start=Point(40.0, 140.0),
        end=Point(280.0, 140.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )
    dimension = _length_dimension("ocr_001", "150", 150.0)

    # When a target suggestion is requested exactly between them
    suggested = suggest_dimension_target(
        dimension,
        (150.0, 87.0, 40.0, 16.0),
        (first, second),
    )

    # Then no connection is fabricated between equally plausible candidates
    assert suggested.target_entity_ids == ()
    assert suggested.source == "ocr"


def test_ignores_a_line_far_beyond_the_suggestion_radius() -> None:
    # Given one line far away from the OCR label
    line = LineEntity(
        entity_id=EntityId("line_far"),
        raw_start=Point(2000.0, 2000.0),
        raw_end=Point(2200.0, 2000.0),
        start=Point(2000.0, 2000.0),
        end=Point(2200.0, 2000.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )
    dimension = _length_dimension("ocr_001", "150", 150.0)

    # When a target suggestion is requested
    suggested = suggest_dimension_target(dimension, (0.0, 0.0, 40.0, 16.0), (line,))

    # Then no distant geometry is guessed as the target
    assert suggested.target_entity_ids == ()


def test_diameter_only_targets_circles() -> None:
    # Given a nearby line and a nearby circle for a diameter label
    line = LineEntity(
        entity_id=EntityId("line_a"),
        raw_start=Point(0.0, 0.0),
        raw_end=Point(100.0, 0.0),
        start=Point(0.0, 0.0),
        end=Point(100.0, 0.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )
    circle = CircleEntity(
        entity_id=EntityId("circle_a"),
        center=Point(50.0, 40.0),
        radius=20.0,
        confidence=0.9,
    )
    dimension = Dimension(
        dimension_id=DimensionId("ocr_002"),
        raw_text="Ø40",
        kind=DimensionKind.DIAMETER,
        parsed_value=40.0,
        unit="mm",
        target_entity_ids=(),
        confidence=0.98,
        status=DimensionStatus.NEEDS_REVIEW,
        source="ocr",
    )

    # When a target suggestion is requested near both entities
    suggested = suggest_dimension_target(
        dimension, (45.0, 15.0, 10.0, 10.0), (line, circle)
    )

    # Then only the physically meaningful circle target is suggested
    assert suggested.target_entity_ids == (EntityId("circle_a"),)


def test_never_overrides_an_existing_target() -> None:
    # Given a dimension a person already linked to one entity
    line = LineEntity(
        entity_id=EntityId("line_a"),
        raw_start=Point(0.0, 0.0),
        raw_end=Point(100.0, 0.0),
        start=Point(0.0, 0.0),
        end=Point(100.0, 0.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )
    dimension = Dimension(
        dimension_id=DimensionId("manual_line_a_length"),
        raw_text="100",
        kind=DimensionKind.LENGTH,
        parsed_value=100.0,
        unit="mm",
        target_entity_ids=(EntityId("line_a"),),
        confidence=1.0,
        status=DimensionStatus.CONFIRMED,
    )

    # When a suggestion is requested anyway
    suggested = suggest_dimension_target(dimension, (0.0, 0.0, 10.0, 10.0), (line,))

    # Then the existing human decision is left untouched
    assert suggested is dimension
