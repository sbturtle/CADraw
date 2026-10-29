from cadraw.models import CircleCandidate, EntityId, Point, PolylineCandidate
from cadraw.table_filter import TableFilterConfig, inspect_printed_table_shapes

_WIDTH = 1000
_HEIGHT = 1400


def _box(
    name: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> PolylineCandidate:
    """Build one axis-aligned closed polygon candidate."""
    return PolylineCandidate(
        entity_id=EntityId(name),
        points=(
            Point(left, top),
            Point(right, top),
            Point(right, bottom),
            Point(left, bottom),
        ),
        closed=True,
        confidence=0.9,
    )


def _header_row() -> tuple[PolylineCandidate, ...]:
    """Build a printed header row whose cells tile with a slight skew."""
    return (
        _box("cell_1", 40.0, 100.0, 240.0, 132.0),
        _box("cell_2", 242.0, 103.0, 520.0, 136.0),
        _box("cell_3", 522.0, 106.0, 700.0, 139.0),
        _box("cell_4", 702.0, 109.0, 960.0, 142.0),
    )


def test_tiling_cells_prove_a_printed_table_and_are_removed() -> None:
    # Given a tiling header row beside one hand-drawn profile
    profile = _box("profile", 150.0, 500.0, 820.0, 1000.0)
    candidates = (*_header_row(), profile)

    # When the printed-table proof runs
    result = inspect_printed_table_shapes(candidates, _WIDTH, _HEIGHT)

    # Then only the row is removed and the profile survives
    assert result.table_detected is True
    assert result.removed_count == len(_header_row())
    assert result.candidates == (profile,)


def test_the_table_outline_around_a_proved_row_is_removed_too() -> None:
    # Given the same row plus the table's own enclosing outline
    outline = _box("outline", 38.0, 98.0, 962.0, 240.0)
    profile = _box("profile", 150.0, 500.0, 820.0, 1000.0)

    # When the proof runs
    result = inspect_printed_table_shapes(
        (outline, *_header_row(), profile),
        _WIDTH,
        _HEIGHT,
    )

    # Then the outline goes with the cells it encloses
    assert result.candidates == (profile,)


def test_a_cell_missed_by_the_tiling_run_still_goes_with_its_band() -> None:
    # Given a further cell in the row's band that no run reaches
    detached = _box("detached", 1200.0, 104.0, 1400.0, 137.0)
    profile = _box("profile", 150.0, 500.0, 820.0, 1000.0)

    # When the proof runs
    result = inspect_printed_table_shapes(
        (*_header_row(), detached, profile),
        _WIDTH,
        _HEIGHT,
    )

    # Then the whole band is treated as one piece of stationery
    assert result.candidates == (profile,)


def test_two_adjacent_boxes_alone_prove_nothing() -> None:
    # Given only two tiling boxes, which a drawing can easily contain
    pair = _header_row()[:2]

    # When the proof runs
    result = inspect_printed_table_shapes(pair, _WIDTH, _HEIGHT)

    # Then nothing is removed
    assert result.table_detected is False
    assert result.candidates == pair


def test_nested_drawing_profiles_are_not_read_as_a_table() -> None:
    # Given overlapping profiles of different depths, as a part view has
    candidates = (
        _box("outer", 100.0, 300.0, 700.0, 900.0),
        _box("inner", 160.0, 360.0, 420.0, 620.0),
        _box("notch", 430.0, 360.0, 560.0, 500.0),
    )

    # When the proof runs
    result = inspect_printed_table_shapes(candidates, _WIDTH, _HEIGHT)

    # Then the drawing keeps all of its geometry
    assert result.table_detected is False
    assert result.candidates == candidates


def test_a_glyph_inside_a_cell_does_not_break_the_row_proof() -> None:
    # Given a printed row with a written character traced inside one cell
    glyph = _box("glyph", 300.0, 110.0, 330.0, 128.0)
    profile = _box("profile", 150.0, 500.0, 820.0, 1000.0)

    # When the proof runs
    result = inspect_printed_table_shapes(
        (*_header_row(), glyph, profile),
        _WIDTH,
        _HEIGHT,
    )

    # Then the row still proves itself and takes the glyph with it
    assert result.table_detected is True
    assert result.candidates == (profile,)


def test_features_inside_a_kept_outline_are_not_stationery() -> None:
    # Given a proved row and, in the same band, a slot inside a part outline
    outline = _box("outline", 1000.0, 60.0, 1400.0, 300.0)
    slot = _box("slot", 1100.0, 110.0, 1200.0, 130.0)

    # When the proof runs
    result = inspect_printed_table_shapes(
        (*_header_row(), outline, slot),
        _WIDTH,
        _HEIGHT,
    )

    # Then the row goes, while the part keeps the slot drawn inside it
    assert result.table_detected is True
    assert result.candidates == (outline, slot)


def test_circles_are_never_table_cells() -> None:
    # Given a proved row and a drilled hole sitting in the same band
    hole = CircleCandidate(
        entity_id=EntityId("hole"),
        center=Point(500.0, 120.0),
        radius=14.0,
        confidence=0.9,
    )

    # When the proof runs
    result = inspect_printed_table_shapes(
        (*_header_row(), hole),
        _WIDTH,
        _HEIGHT,
    )

    # Then only rectangles are eligible, so the hole is kept
    assert result.candidates == (hole,)


def test_the_proof_can_be_disabled() -> None:
    # Given a caller that opts out
    candidates = _header_row()

    # When the proof is disabled
    result = inspect_printed_table_shapes(
        candidates,
        _WIDTH,
        _HEIGHT,
        TableFilterConfig(enabled=False),
    )

    # Then the candidates are returned untouched
    assert result.table_detected is False
    assert result.candidates == candidates
