import cv2
import numpy as np

from cadraw.models import EntityId, LineEntity, Orientation, Point
from cadraw.stroke_evidence import (
    ink_between_factory,
    ink_mask_from_grayscale,
    wide_crossing_share,
    without_lettering_crossings,
)


def _mask_with_vertical_stroke(gap: tuple[int, int] | None) -> np.ndarray:
    mask = np.zeros((200, 100), dtype=np.uint8)
    mask[20:180, 49:52] = 255
    if gap is not None:
        mask[gap[0] : gap[1], :] = 0
    return mask


def test_continuous_ink_across_a_gap_is_proved() -> None:
    # Given one stroke whose Hough trace broke into two pieces
    ink_between = ink_between_factory(_mask_with_vertical_stroke(None))

    # Then the ink between the facing endpoints is continuous
    assert ink_between(Point(50.0, 60.0), Point(50.0, 120.0))


def test_a_gap_with_no_ink_is_not_bridged() -> None:
    # Given two strokes that line up but leave paper between them
    ink_between = ink_between_factory(_mask_with_vertical_stroke((80, 110)))

    # Then the gap is not proved and the pieces stay separate
    assert not ink_between(Point(50.0, 60.0), Point(50.0, 120.0))


def test_a_stroke_that_drifts_a_pixel_still_counts() -> None:
    # Given a stroke that wanders one pixel off the straight gap
    mask = np.zeros((200, 100), dtype=np.uint8)
    for y in range(20, 180):
        mask[y, 50 + (1 if y % 2 else 0)] = 255
    ink_between = ink_between_factory(mask)

    assert ink_between(Point(50.0, 30.0), Point(50.0, 170.0))


def _line(name: str, start: Point, end: Point) -> LineEntity:
    return LineEntity(
        entity_id=EntityId(name),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=Orientation.HORIZONTAL,
        confidence=0.9,
    )


def _page_with_edge_and_lettering() -> np.ndarray:
    """Draw a part edge and, below it, a row of tall written glyphs."""
    mask = np.zeros((200, 600), dtype=np.uint8)
    _ = cv2.line(mask, (40, 50), (560, 50), 255, 3)
    for left in range(40, 560, 30):
        _ = cv2.line(mask, (left, 130), (left, 160), 255, 3)
        _ = cv2.line(mask, (left, 145), (left + 15, 145), 255, 3)
    return mask


def test_a_drawn_edge_crosses_only_its_own_stroke_width() -> None:
    mask = _page_with_edge_and_lettering()

    share = wide_crossing_share(mask, Point(40.0, 50.0), Point(560.0, 50.0), 9.0)

    assert share == 0.0


def test_a_line_through_a_row_of_glyphs_meets_tall_ink() -> None:
    mask = _page_with_edge_and_lettering()

    share = wide_crossing_share(mask, Point(40.0, 145.0), Point(560.0, 145.0), 9.0)

    assert share > 0.2


def test_lines_through_lettering_are_dropped_and_edges_kept() -> None:
    # Given the part edge and a line the detector ran through the glyphs
    mask = _page_with_edge_and_lettering()
    edge = _line("edge", Point(40.0, 50.0), Point(560.0, 50.0))
    through = _line("through", Point(40.0, 145.0), Point(560.0, 145.0))

    kept = without_lettering_crossings((edge, through), mask, 3.0)

    assert kept == (edge,)


def test_grayscale_is_binarized_into_ink() -> None:
    # Given a light page with one dark stroke
    image = np.full((60, 60), 240, dtype=np.uint8)
    image[10:50, 29:32] = 30

    mask = ink_mask_from_grayscale(image)

    assert mask[30, 30] == 255
    assert mask[5, 5] == 0
