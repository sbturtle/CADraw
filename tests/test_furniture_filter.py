from cadraw.furniture_filter import (
    RulingRails,
    is_printed_row,
    without_printed_rules,
    without_ruling_remnants,
    without_table_edge_lines,
)
from cadraw.models import (
    ArcEntity,
    EntityId,
    LineEntity,
    Orientation,
    Point,
    PolylineEntity,
)


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


def test_a_straight_line_touching_a_closed_outline_is_kept() -> None:
    # Given a ruler-straight member drawn out from a plate's outline
    plate = PolylineEntity(
        entity_id=EntityId("plate"),
        points=(
            Point(100.0, 100.0),
            Point(400.0, 100.0),
            Point(400.0, 300.0),
            Point(100.0, 300.0),
        ),
        closed=True,
        confidence=0.9,
    )
    member = _line(
        "member", Point(400.0, 200.0), Point(800.0, 200.0), Orientation.HORIZONTAL
    )

    kept = without_printed_rules((member,), (plate,), 10.0, 150.0, lambda _s, _e: 0.2)

    # Then meeting the outline makes it part of the drawing
    assert kept == (member,)


_HEADER_CELLS = (
    (48.0, 103.0, 162.0, 136.0),
    (162.0, 106.0, 410.0, 142.0),
    (517.0, 114.0, 727.0, 152.0),
)
_REMARKS_ROW = (
    Point(47.0, 135.0),
    Point(1028.0, 155.0),
    Point(1030.0, 239.0),
    Point(55.0, 225.0),
    Point(47.0, 218.0),
)


def test_a_ruled_row_under_a_proved_table_is_a_printed_row() -> None:
    # Given a slightly skewed remarks row ruled under proved header cells
    assert is_printed_row(_REMARKS_ROW, 1077.0, lambda _s, _e: 0.4, _HEADER_CELLS)


def test_a_page_wide_ruled_bar_without_a_table_is_kept() -> None:
    # Given the same straight, page-wide band with no proved table against it
    assert not is_printed_row(_REMARKS_ROW, 1077.0, lambda _s, _e: 0.4, ())


def test_a_ruler_drawn_part_is_not_a_printed_row() -> None:
    # Given straight edges that do not make a page-wide band
    notched = (
        Point(100.0, 100.0),
        Point(980.0, 100.0),
        Point(980.0, 300.0),
        Point(700.0, 300.0),
        Point(700.0, 200.0),
        Point(100.0, 200.0),
    )
    short = (
        Point(300.0, 100.0),
        Point(700.0, 100.0),
        Point(700.0, 160.0),
        Point(300.0, 160.0),
    )

    assert not is_printed_row(notched, 1077.0, lambda _s, _e: 0.3, _HEADER_CELLS)
    assert not is_printed_row(short, 1077.0, lambda _s, _e: 0.3, _HEADER_CELLS)


def test_a_hand_drawn_band_across_the_page_is_not_print() -> None:
    # Given a page-wide band whose edges wander like a pen's
    assert not is_printed_row(_REMARKS_ROW, 1077.0, lambda _s, _e: 1.3, _HEADER_CELLS)


def test_a_straight_edge_with_a_hand_drawn_branch_is_kept() -> None:
    # Given a ruler-straight edge and a pen-drawn branch from its middle
    edge = _line(
        "edge", Point(100.0, 300.0), Point(800.0, 300.0), Orientation.HORIZONTAL
    )
    branch = _line(
        "branch", Point(450.0, 300.0), Point(450.0, 380.0), Orientation.VERTICAL
    )

    kept = without_printed_rules(
        (edge, branch),
        (),
        10.0,
        150.0,
        lambda start, _e: 1.4 if start == branch.start else 0.2,
    )

    # Then the hand-drawn branch makes the edge part of the drawing
    assert kept == (edge, branch)


def test_a_printed_t_of_rules_goes_together() -> None:
    # Given a long printed rule and a short printed divider ruled onto it
    rule = _line(
        "rule", Point(40.0, 300.0), Point(1000.0, 300.0), Orientation.HORIZONTAL
    )
    divider = _line(
        "divider", Point(450.0, 260.0), Point(450.0, 300.0), Orientation.VERTICAL
    )

    kept = without_printed_rules((rule, divider), (), 10.0, 150.0, lambda _s, _e: 0.2)

    # Then the joint proves nothing and both are print
    assert kept == ()


def test_straight_lines_meeting_a_rounded_corner_are_kept() -> None:
    # Given two ruler-straight edges joined by an open quarter-circle corner
    corner = ArcEntity(
        entity_id=EntityId("corner"),
        center=Point(400.0, 300.0),
        radius=50.0,
        start_angle_degrees=270.0,
        end_angle_degrees=360.0,
        confidence=0.9,
    )
    top = _line("top", Point(100.0, 250.0), Point(400.0, 250.0), Orientation.HORIZONTAL)
    side = _line("side", Point(450.0, 300.0), Point(450.0, 600.0), Orientation.VERTICAL)

    kept = without_printed_rules(
        (top, side), (corner,), 10.0, 150.0, lambda _s, _e: 0.2
    )

    # Then the arc joins both to the drawing
    assert kept == (top, side)


def test_a_long_isolated_line_as_straight_as_print_is_a_form_rule() -> None:
    # Given a header rule, measured as straight to a fraction of a pixel
    rule = _line(
        "rule", Point(10.0, 373.0), Point(870.0, 373.0), Orientation.HORIZONTAL
    )

    kept = without_printed_rules((rule,), (), 10.0, 150.0, lambda _s, _e: 0.25)

    assert kept == ()


def test_a_hand_drawn_line_of_the_same_length_is_kept() -> None:
    # Given an isolated stroke that wanders a pixel or more, as a pen does
    stroke = _line(
        "stroke", Point(10.0, 373.0), Point(870.0, 373.0), Orientation.HORIZONTAL
    )

    kept = without_printed_rules((stroke,), (), 10.0, 150.0, lambda _s, _e: 1.4)

    assert kept == (stroke,)


def test_straight_lines_joined_to_drawn_or_slanted_ones_and_short_ones_stay() -> None:
    # Given a straight edge meeting a pen-drawn side, a straight edge meeting a
    # ruler-drawn chamfer, and a short isolated tick
    edge = _line(
        "edge", Point(100.0, 300.0), Point(700.0, 300.0), Orientation.HORIZONTAL
    )
    side = _line("side", Point(700.0, 300.0), Point(700.0, 600.0), Orientation.VERTICAL)
    base = _line(
        "base", Point(100.0, 800.0), Point(600.0, 800.0), Orientation.HORIZONTAL
    )
    chamfer = _line(
        "chamfer", Point(600.0, 800.0), Point(700.0, 700.0), Orientation.DIAGONAL
    )
    tick = _line("tick", Point(900.0, 50.0), Point(980.0, 50.0), Orientation.HORIZONTAL)

    kept = without_printed_rules(
        (edge, side, base, chamfer, tick),
        (),
        10.0,
        150.0,
        lambda start, _e: 1.4 if start == side.start else 0.2,
    )

    # Then none of them is taken for print
    assert kept == (edge, side, base, chamfer, tick)
