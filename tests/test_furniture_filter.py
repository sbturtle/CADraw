from cadraw.furniture_filter import (
    RulingRails,
    without_ruling_remnants,
    without_table_edge_lines,
)
from cadraw.models import EntityId, LineEntity, Orientation, Point


def _line(name: str, start: Point, end: Point, orientation: Orientation) -> LineEntity:
    return LineEntity(
        entity_id=EntityId(name),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=orientation,
        confidence=0.9,
    )


def test_a_line_along_a_proved_table_cell_edge_is_dropped() -> None:
    # Given a proved header cell and one rule the line detector traced along it
    cell = (100.0, 40.0, 400.0, 90.0)
    rule = _line("rule", Point(110.0, 91.0), Point(380.0, 91.0), Orientation.HORIZONTAL)
    member = _line(
        "part", Point(120.0, 300.0), Point(380.0, 300.0), Orientation.HORIZONTAL
    )
    chord = _line(
        "chord", Point(150.0, 65.0), Point(350.0, 65.0), Orientation.HORIZONTAL
    )

    kept = without_table_edge_lines((rule, member, chord), (cell,), 6.0)

    # Then only the rule goes; a line elsewhere or through the cell stays
    assert tuple(line.entity_id for line in kept) == (
        EntityId("part"),
        EntityId("chord"),
    )


def test_a_line_that_barely_touches_a_cell_edge_is_kept() -> None:
    # Given a drawn member whose end merely overlaps a cell edge
    cell = (100.0, 40.0, 200.0, 90.0)
    member = _line(
        "part", Point(180.0, 90.0), Point(600.0, 90.0), Orientation.HORIZONTAL
    )

    assert without_table_edge_lines((member,), (cell,), 6.0) == (member,)


def _rails() -> RulingRails:
    return RulingRails(
        horizontal=(100.0, 134.0, 168.0),
        vertical=(50.0, 84.0, 118.0),
        band_width=8.0,
        maximum_intensity=135.0,
    )


def test_light_isolated_line_on_a_proved_rail_is_a_ruling_remnant() -> None:
    # Given a faint isolated line lying on a proved rail
    remnant = _line(
        "remnant", Point(84.0, 300.0), Point(84.0, 500.0), Orientation.VERTICAL
    )

    kept = without_ruling_remnants((remnant,), _rails(), 10.0, lambda _s, _e: 190.0)

    assert kept == ()


def test_dark_stroke_drawn_on_a_rail_is_kept() -> None:
    # Given a pen stroke that follows a rail exactly
    stroke = _line(
        "stroke", Point(84.0, 300.0), Point(84.0, 500.0), Orientation.VERTICAL
    )

    kept = without_ruling_remnants((stroke,), _rails(), 10.0, lambda _s, _e: 60.0)

    assert kept == (stroke,)


def test_light_line_joined_to_the_drawing_is_kept() -> None:
    # Given a faint line on a rail that meets another member at an endpoint
    faint = _line("faint", Point(84.0, 300.0), Point(84.0, 500.0), Orientation.VERTICAL)
    other = _line(
        "other", Point(84.0, 500.0), Point(300.0, 500.0), Orientation.HORIZONTAL
    )

    kept = without_ruling_remnants((faint, other), _rails(), 10.0, lambda _s, _e: 190.0)

    assert kept == (faint, other)


def test_light_line_off_every_rail_is_kept() -> None:
    # Given a faint isolated line between rails
    faint = _line("faint", Point(67.0, 300.0), Point(67.0, 500.0), Orientation.VERTICAL)

    kept = without_ruling_remnants((faint,), _rails(), 10.0, lambda _s, _e: 190.0)

    assert kept == (faint,)
