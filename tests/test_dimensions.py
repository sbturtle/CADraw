import pytest

from cadraw.dimensions import (
    associate_dimension,
    confirm_dimension_text,
    parse_dimension_text,
)
from cadraw.models import (
    DimensionId,
    DimensionKind,
    DimensionStatus,
    EntityId,
)


@pytest.mark.parametrize(
    ("raw_text", "expected_kind", "expected_value"),
    [
        ("100", DimensionKind.LENGTH, 100.0),
        ("Ø25", DimensionKind.DIAMETER, 25.0),
        ("R12.5", DimensionKind.RADIUS, 12.5),
        ("45°", DimensionKind.ANGLE, 45.0),
        ("400R", DimensionKind.RADIUS, 400.0),
        ("790Ø", DimensionKind.DIAMETER, 790.0),
        ("790φ", DimensionKind.DIAMETER, 790.0),
        ("φ20", DimensionKind.DIAMETER, 20.0),
    ],
)
def test_dimension_parser_reads_supported_notation(
    raw_text: str,
    expected_kind: DimensionKind,
    expected_value: float,
) -> None:
    # Given a supported handwritten-style text candidate
    # When it is parsed at the trust boundary
    dimension = parse_dimension_text(raw_text, DimensionId("dim_1"))

    # Then the value is typed but still awaits human confirmation
    assert dimension.kind is expected_kind
    assert dimension.parsed_value == expected_value
    assert dimension.status is DimensionStatus.NEEDS_REVIEW


def test_unreadable_text_becomes_unknown_instead_of_zero() -> None:
    # Given text that cannot be safely interpreted
    # When the parser processes it
    dimension = parse_dimension_text("1O0?", DimensionId("dim_2"))

    # Then no fabricated numeric value crosses the model boundary
    assert dimension.parsed_value is None
    assert dimension.status is DimensionStatus.UNKNOWN


def test_non_positive_length_and_angles_through_180_degrees_are_validated() -> None:
    # Given values that must never become fabrication constraints
    zero = parse_dimension_text("0", DimensionId("dim_zero"))
    angle = parse_dimension_text("180°", DimensionId("dim_angle"))

    # Then non-positive lengths are invalid while a straight angle is accepted
    assert zero.status is DimensionStatus.INVALID
    assert zero.parsed_value is None
    assert angle.status is DimensionStatus.NEEDS_REVIEW
    assert angle.parsed_value == 180.0


def test_confirmation_requires_explicit_target_and_user_value() -> None:
    # Given a parsed dimension candidate
    parsed = parse_dimension_text("100", DimensionId("dim_3"))
    associated = associate_dimension(parsed, EntityId("line_1"))

    # When a user confirms the value
    confirmed = confirm_dimension_text(associated, "105")

    # Then both the target and explicit value are retained
    assert confirmed.target_entity_ids == (EntityId("line_1"),)
    assert confirmed.parsed_value == 105.0
    assert confirmed.status is DimensionStatus.CONFIRMED


@pytest.mark.parametrize("raw_text", ["1,050", "1.050", "2,400mm"])
def test_thousands_grouping_is_refused_not_shrunk(raw_text: str) -> None:
    # Given a size typed with a thousands separator
    row = parse_dimension_text("100", DimensionId("dim_1"))

    # When the reviewer confirms it
    confirmed = confirm_dimension_text(row, raw_text)

    # Then it is refused, never confirmed a thousand times too small
    assert confirmed.parsed_value is None
    assert confirmed.status is DimensionStatus.INVALID
