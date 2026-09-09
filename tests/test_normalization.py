from cadraw.models import EntityId, LineCandidate, Orientation, Point
from cadraw.normalization import (
    NormalizationConfig,
    build_connections,
    normalize_lines,
    select_largest_connected_component,
)


def test_almost_horizontal_and_vertical_lines_are_snapped() -> None:
    # Given candidates that are only slightly skewed
    candidates = (
        LineCandidate(EntityId("h"), Point(0.0, 0.0), Point(100.0, 5.0), 0.9),
        LineCandidate(EntityId("v"), Point(105.0, 4.0), Point(108.0, 104.0), 0.9),
    )

    # When the deterministic normalizer runs
    lines = normalize_lines(
        candidates,
        NormalizationConfig(
            horizontal_angle_tolerance_degrees=4.0,
            vertical_angle_tolerance_degrees=4.0,
            endpoint_snap_distance=6.0,
            line_merge_tolerance=5.0,
            minimum_line_length=1.0,
        ),
    )

    # Then the intended axes are exact and the diagonal is not guessed
    assert lines[0].orientation is Orientation.HORIZONTAL
    assert lines[0].start.y == lines[0].end.y
    assert lines[1].orientation is Orientation.VERTICAL
    assert lines[1].start.x == lines[1].end.x


def test_nearby_endpoints_are_snapped_and_connected() -> None:
    # Given two lines whose endpoints miss by a few source units
    candidates = (
        LineCandidate(EntityId("first"), Point(0.0, 0.0), Point(100.0, 0.0), 0.9),
        LineCandidate(EntityId("second"), Point(103.0, 2.0), Point(103.0, 80.0), 0.9),
    )

    # When endpoint normalization and connection detection run
    config = NormalizationConfig(
        endpoint_snap_distance=5.0,
        line_merge_tolerance=4.0,
        minimum_line_length=1.0,
    )
    lines = normalize_lines(candidates, config)
    connections = build_connections(lines, config.endpoint_snap_distance)

    # Then one connection identifies the shared snapped corner
    assert len(connections) == 1
    assert connections[0].point == Point(103.0, 0.0)
    assert lines[0].end == connections[0].point
    assert lines[1].start == connections[0].point


def test_near_collinear_segments_merge_without_changing_orientation() -> None:
    # Given two pieces of the same hand-drawn stroke
    candidates = (
        LineCandidate(EntityId("piece_a"), Point(0.0, 10.0), Point(50.0, 10.0), 0.8),
        LineCandidate(EntityId("piece_b"), Point(54.0, 11.0), Point(100.0, 10.0), 0.8),
    )

    # When same-line merging runs
    lines = normalize_lines(
        candidates,
        NormalizationConfig(
            endpoint_snap_distance=4.0,
            line_merge_tolerance=6.0,
            minimum_line_length=1.0,
        ),
    )

    # Then the result is one clean horizontal line spanning both pieces
    assert len(lines) == 1
    assert lines[0].orientation is Orientation.HORIZONTAL
    assert lines[0].length == 100.0


def test_largest_connected_component_discards_unconnected_text_strokes() -> None:
    # Given a connected structural chain and one isolated text-like stroke
    candidates = (
        LineCandidate(EntityId("a"), Point(0.0, 0.0), Point(100.0, 0.0), 0.9),
        LineCandidate(EntityId("b"), Point(100.0, 0.0), Point(100.0, 100.0), 0.9),
        LineCandidate(EntityId("c"), Point(100.0, 100.0), Point(150.0, 100.0), 0.9),
        LineCandidate(EntityId("text"), Point(300.0, 300.0), Point(350.0, 300.0), 0.4),
    )
    config = NormalizationConfig(
        endpoint_snap_distance=5.0,
        line_merge_tolerance=5.0,
        minimum_line_length=1.0,
    )

    # When component selection runs after normalization
    lines = normalize_lines(candidates, config)
    structural = select_largest_connected_component(lines, 5.0)

    # Then isolated lettering does not enter the structural CAD result
    assert tuple(line.entity_id for line in structural) == (
        EntityId("a"),
        EntityId("b"),
        EntityId("c"),
    )
