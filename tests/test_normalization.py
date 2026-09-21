from cadraw.models import EntityId, LineCandidate, Orientation, Point
from cadraw.normalization import (
    NormalizationConfig,
    build_connections,
    normalize_lines,
    resolution_adjusted,
    select_largest_connected_component,
    select_significant_components,
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


def test_every_part_of_a_sheet_is_kept_and_lettering_still_falls_away() -> None:
    # Given two drawn parts of different size and one stray text stroke
    candidates = (
        LineCandidate(EntityId("a1"), Point(0.0, 0.0), Point(400.0, 0.0), 0.9),
        LineCandidate(EntityId("a2"), Point(400.0, 0.0), Point(400.0, 300.0), 0.9),
        LineCandidate(EntityId("b1"), Point(600.0, 0.0), Point(800.0, 0.0), 0.9),
        LineCandidate(EntityId("b2"), Point(800.0, 0.0), Point(800.0, 100.0), 0.9),
        LineCandidate(EntityId("text"), Point(300.0, 600.0), Point(340.0, 600.0), 0.4),
    )
    config = NormalizationConfig(
        endpoint_snap_distance=5.0,
        line_merge_tolerance=5.0,
        minimum_line_length=1.0,
    )
    lines = normalize_lines(candidates, config)

    # When components below a fifth of the longest group are dropped
    kept = select_significant_components(lines, 5.0, 0.20)

    # Then both parts survive and the lettering does not
    assert tuple(line.entity_id for line in kept) == (
        EntityId("a1"),
        EntityId("a2"),
        EntityId("b1"),
        EntityId("b2"),
    )
    assert select_significant_components(lines, 5.0, 1.0) == kept[:2]


def test_a_wide_gap_is_bridged_only_when_the_ink_is_continuous() -> None:
    # Given one stroke that Hough broke into two pieces 50 units apart
    candidates = (
        LineCandidate(EntityId("upper"), Point(100.0, 0.0), Point(100.0, 200.0), 0.9),
        LineCandidate(EntityId("lower"), Point(101.0, 250.0), Point(101.0, 500.0), 0.9),
    )
    config = NormalizationConfig(
        endpoint_snap_distance=5.0,
        line_merge_tolerance=5.0,
        minimum_line_length=1.0,
        evidence_gap_distance=90.0,
    )

    # When the raster proves ink across the gap, the pieces become one line
    merged = normalize_lines(candidates, config, ink_between=lambda _s, _e: True)
    assert len(merged) == 1
    assert merged[0].length == 500.0

    # And geometry alone, or a gap with no ink, leaves them apart
    assert len(normalize_lines(candidates, config)) == 2
    assert (
        len(normalize_lines(candidates, config, ink_between=lambda _s, _e: False)) == 2
    )


def test_merge_tolerance_scales_with_the_shorter_image_side() -> None:
    config = NormalizationConfig(
        line_merge_tolerance=8.0, line_merge_tolerance_ratio=0.01
    )

    assert resolution_adjusted(config, 400, 300).line_merge_tolerance == 8.0
    assert resolution_adjusted(config, 2000, 1500).line_merge_tolerance == 15.0
