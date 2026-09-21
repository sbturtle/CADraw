from cadraw.duplicate_filter import without_polygon_edge_lines
from cadraw.models import EntityId, LineEntity, Orientation, Point, PolylineEntity


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
