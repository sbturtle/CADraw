import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.grid_filter import GridFilterConfig, without_ruled_lattices
from cadraw.models import EntityId, Point, PolylineCandidate, ShapeCandidate

type InkMask = NDArray[np.uint8]

_SHEET = 1000
_PEN = 3


def _cell(name: str, left: float, top: float, size: float) -> PolylineCandidate:
    """Return a slightly skewed four-sided cell, as ruling looks at an angle."""
    return PolylineCandidate(
        entity_id=EntityId(name),
        points=(
            Point(left, top),
            Point(left + size, top + 1.0),
            Point(left + size + 1.0, top + size + 1.0),
            Point(left + 1.0, top + size),
        ),
        closed=True,
        confidence=0.9,
    )


def _ink(shapes: tuple[ShapeCandidate, ...], sheet: int = _SHEET) -> InkMask:
    """Draw each cell's outline, as the face it was read from was bounded."""
    ink = np.zeros((sheet, sheet), dtype=np.uint8)
    for shape in shapes:
        assert isinstance(shape, PolylineCandidate)
        corners = np.asarray(
            [[round(point.x), round(point.y)] for point in shape.points],
            dtype=np.int32,
        )
        _ = cv2.polylines(ink, [corners], isClosed=True, color=255, thickness=_PEN)
    return ink


def _lattice(columns: int, rows: int, size: float) -> tuple[ShapeCandidate, ...]:
    return tuple(
        _cell(f"cell_{row}_{column}", 100.0 + column * size, 100.0 + row * size, size)
        for row in range(rows)
        for column in range(columns)
    )


def test_small_cells_tiling_the_paper_are_dropped() -> None:
    # Given a three by two patch of graph-paper cells seen at an angle
    cells = _lattice(3, 2, 40.0)

    # When ruled lattices are removed
    kept, removed = without_ruled_lattices(cells, _ink(cells))

    # Then every cell goes
    assert kept == ()
    assert removed == len(cells)


def test_a_frame_of_large_cells_is_a_drawn_part() -> None:
    # Given a two by two window frame drawn a fifth of the sheet across
    frame = _lattice(2, 2, 200.0)

    # Then its cells are kept: graph paper cells are a few percent across
    assert without_ruled_lattices(frame, _ink(frame)) == (frame, 0)


def test_three_small_cells_side_by_side_are_kept() -> None:
    # Given three small adjacent cells, as a part with two partitions makes
    cells = _lattice(3, 1, 40.0)

    assert without_ruled_lattices(cells, _ink(cells)) == (cells, 0)


def test_small_cells_standing_apart_are_kept() -> None:
    # Given four small slots spread over a plate, none touching another
    slots = tuple(
        _cell(f"slot_{index}", 100.0 + index * 120.0, 300.0, 40.0) for index in range(4)
    )

    assert without_ruled_lattices(slots, _ink(slots)) == (slots, 0)


def test_slots_drawn_a_few_pixels_apart_are_not_a_lattice() -> None:
    # Given a large photograph with four square slots in a two by two block,
    # each drawn with its own stroke, their outlines 8 px apart: within the
    # side-matching tolerance there
    slots = tuple(
        _cell(f"slot_{row}_{column}", 100.0 + column * 58.0, 100.0 + row * 58.0, 50.0)
        for row in range(2)
        for column in range(2)
    )

    # Then the paper between their strokes keeps them apart
    assert without_ruled_lattices(slots, _ink(slots, 3000)) == (slots, 0)


def test_the_lattice_step_is_off_with_the_grid_filter() -> None:
    # Given graph-paper cells and a grid filter switched off
    cells = _lattice(3, 2, 40.0)

    # Then nothing is removed
    assert without_ruled_lattices(
        cells,
        _ink(cells),
        GridFilterConfig(enabled=False),
    ) == (cells, 0)
