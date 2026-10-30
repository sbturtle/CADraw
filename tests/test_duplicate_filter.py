from cadraw.duplicate_filter import (
    without_circle_chords,
    without_lines_through_features,
    without_polygon_edge_lines,
)
from cadraw.models import (
    CircleEntity,
    EntityId,
    GeometryRole,
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


def _outline() -> PolylineEntity:
    return PolylineEntity(
        entity_id=EntityId("plate"),
        points=(
            Point(100.0, 100.0),
            Point(500.0, 100.0),
            Point(500.0, 300.0),
            Point(100.0, 300.0),
        ),
        closed=True,
        confidence=0.9,
    )


def test_a_line_that_retraces_an_outline_edge_is_dropped() -> None:
    # Given the outline's top edge seen again by the line detector
    edge = _line(
        "edge", Point(120.0, 103.0), Point(480.0, 98.0), Orientation.HORIZONTAL
    )

    assert without_polygon_edge_lines((edge,), (_outline(),), 12.0) == ()


def test_a_line_that_leaves_the_outline_is_kept() -> None:
    # Given a member that starts on the edge and continues past the outline
    member = _line(
        "member", Point(300.0, 100.0), Point(300.0, 0.0), Orientation.VERTICAL
    )
    chord = _line(
        "chord", Point(100.0, 200.0), Point(500.0, 200.0), Orientation.HORIZONTAL
    )

    kept = without_polygon_edge_lines((member, chord), (_outline(),), 12.0)

    # Then both stay: one leaves the boundary, the other crosses the interior
    assert kept == (member, chord)


def _hole() -> CircleEntity:
    return CircleEntity(
        entity_id=EntityId("hole"),
        center=Point(300.0, 200.0),
        radius=20.0,
        confidence=0.9,
        role=GeometryRole.HOLE,
    )


def test_a_line_through_a_blank_hole_is_dropped() -> None:
    # Given a detector line bridging strokes straight across a hole
    crossing = _line(
        "crossing", Point(250.0, 200.0), Point(350.0, 200.0), Orientation.HORIZONTAL
    )

    # Then it cannot be ink: a stroke there would have split the hole's face
    assert without_lines_through_features((crossing,), (_hole(),), 3.0) == ()


def test_a_leader_touching_a_hole_edge_is_kept() -> None:
    # Given a leader line that ends on the hole's rim
    leader = _line(
        "leader", Point(400.0, 200.0), Point(321.0, 200.0), Orientation.HORIZONTAL
    )

    # Then it stays, since it never runs into the blank interior
    assert without_lines_through_features((leader,), (_hole(),), 3.0) == (leader,)


def test_a_line_through_a_slot_is_dropped() -> None:
    # Given a slot and a line drawn along its blank middle
    slot = PolylineEntity(
        entity_id=EntityId("slot"),
        points=(
            Point(200.0, 180.0),
            Point(320.0, 180.0),
            Point(320.0, 220.0),
            Point(200.0, 220.0),
        ),
        closed=True,
        confidence=0.9,
        role=GeometryRole.HOLE,
    )
    crossing = _line(
        "crossing", Point(150.0, 200.0), Point(370.0, 200.0), Orientation.HORIZONTAL
    )

    assert without_lines_through_features((crossing,), (slot,), 3.0) == ()


def test_an_open_polyline_is_not_an_outline() -> None:
    # Given an open chain instead of a closed profile
    chain = PolylineEntity(
        entity_id=EntityId("chain"),
        points=(Point(100.0, 100.0), Point(500.0, 100.0), Point(500.0, 300.0)),
        closed=False,
        confidence=0.9,
    )
    edge = _line(
        "edge", Point(120.0, 100.0), Point(480.0, 100.0), Orientation.HORIZONTAL
    )

    assert without_polygon_edge_lines((edge,), (chain,), 12.0) == (edge,)


def _washer() -> CircleEntity:
    return CircleEntity(
        entity_id=EntityId("washer"),
        center=Point(300.0, 200.0),
        radius=100.0,
        confidence=0.9,
    )


def test_chords_retracing_a_circle_stroke_are_dropped() -> None:
    # Given two chords the line detector found end to end along the stroke
    first = _line(
        "first", Point(400.0, 180.0), Point(394.0, 236.0), Orientation.VERTICAL
    )
    second = _line(
        "second", Point(394.0, 236.0), Point(370.0, 271.0), Orientation.DIAGONAL
    )

    # Then they are the circle's ink a second time, not members of their own
    assert without_circle_chords((first, second), (_washer(),), 12.0) == ()


def test_a_straight_edge_touching_a_circle_is_kept() -> None:
    # Given a short edge touching the rim that goes on into the outline
    tangent = _line(
        "tangent", Point(260.0, 100.0), Point(340.0, 100.0), Orientation.HORIZONTAL
    )
    side = _line("side", Point(340.0, 100.0), Point(340.0, 20.0), Orientation.VERTICAL)

    # Then the edge stays, since a chord only ever joins other chords
    kept = without_circle_chords((tangent, side), (_washer(),), 12.0)
    assert kept == (tangent, side)


def test_a_centre_line_through_a_circle_is_kept() -> None:
    # Given a centre line drawn across the circle and past its rim
    centre_line = _line(
        "centre", Point(300.0, 80.0), Point(300.0, 320.0), Orientation.VERTICAL
    )

    # Then it stays: its midpoint is the circle's centre, far from the rim
    assert without_circle_chords((centre_line,), (_washer(),), 12.0) == (centre_line,)


def test_a_chord_ending_where_a_centre_line_crosses_the_rim_is_dropped() -> None:
    # Given a chord whose end meets a centre line dash crossing the rim
    chord = _line(
        "chord", Point(300.0, 100.0), Point(360.0, 120.0), Orientation.DIAGONAL
    )
    dash = _line("dash", Point(300.0, 105.0), Point(300.0, 60.0), Orientation.VERTICAL)

    # Then the dash stays and the chord goes: a centre line aims at the centre
    assert without_circle_chords((chord, dash), (_washer(),), 12.0) == (dash,)
