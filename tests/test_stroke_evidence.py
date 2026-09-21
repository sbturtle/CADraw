import numpy as np

from cadraw.models import Point
from cadraw.stroke_evidence import ink_between_factory, ink_mask_from_grayscale


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


def test_grayscale_is_binarized_into_ink() -> None:
    # Given a light page with one dark stroke
    image = np.full((60, 60), 240, dtype=np.uint8)
    image[10:50, 29:32] = 30

    mask = ink_mask_from_grayscale(image)

    assert mask[30, 30] == 255
    assert mask[5, 5] == 0
