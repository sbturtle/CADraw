from cadraw.models import (
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)
from cadraw.ui.dimension_rows import manual_dimensions


def _line(
    entity_id: str,
    start: Point,
    end: Point,
    orientation: Orientation,
) -> LineEntity:
    """Build a line fixture for automatic angle rows."""
    return LineEntity(
        entity_id=EntityId(entity_id),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=orientation,
        confidence=0.9,
    )


def test_axis_angles_are_auto_confirmed_and_diagonal_angles_stay_open() -> None:
    # Given horizontal, vertical, and diagonal detected lines
    dimensions = manual_dimensions(
        (
            _line(
                "horizontal",
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Orientation.HORIZONTAL,
            ),
            _line(
                "vertical",
                Point(0.0, 0.0),
                Point(0.0, 100.0),
                Orientation.VERTICAL,
            ),
            _line(
                "diagonal",
                Point(0.0, 0.0),
                Point(60.0, 80.0),
                Orientation.DIAGONAL,
            ),
        ),
    )

    # When verification rows are created
    angles = tuple(
        dimension for dimension in dimensions if dimension.kind is DimensionKind.ANGLE
    )

    # Then axes are fixed automatically while only diagonal input is requested
    assert tuple(dimension.raw_text for dimension in angles) == ("0°", "90°", "")
    assert all(
        dimension.status is DimensionStatus.CONFIRMED for dimension in angles[:2]
    )
    assert all(dimension.source == "axis_inference" for dimension in angles[:2])
    assert angles[-1].status is DimensionStatus.UNSET
    assert angles[-1].source == "manual"
    diagonal_rows = tuple(
        dimension
        for dimension in dimensions
        if dimension.target_entity_ids == (EntityId("diagonal"),)
    )
    assert diagonal_rows[-1].kind is DimensionKind.ANGLE
    assert diagonal_rows[-1].status is DimensionStatus.UNSET


def test_a_value_read_without_a_target_keeps_its_own_row() -> None:
    # Given one line and a read value that measures nothing on the sheet
    unlinked = Dimension(
        dimension_id=DimensionId("ocr_001"),
        raw_text="40mm",
        kind=DimensionKind.LENGTH,
        parsed_value=40.0,
        unit="mm",
        target_entity_ids=(),
        confidence=0.98,
        status=DimensionStatus.NEEDS_REVIEW,
    )

    # When the review rows are built
    dimensions = manual_dimensions(
        (_line("edge", Point(0.0, 0.0), Point(100.0, 0.0), Orientation.HORIZONTAL),),
        (unlinked,),
    )

    # Then the value is still listed for review after the line's own rows
    assert dimensions[-1] == unlinked
    assert all(
        dimension.target_entity_ids == (EntityId("edge"),)
        for dimension in dimensions[:-1]
    )
