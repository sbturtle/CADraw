from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.page_cleanup import PageCleanupConfig, clean_page

_PITCH = 34
_RULING_VALUE = 170
_STROKE_VALUE = 20
_STROKE_LIMIT = 128


def _ruled_sheet() -> NDArray[np.uint8]:
    """Draw a faint periodic grid covering most of a light page."""
    image = np.full((680, 680), 252, dtype=np.uint8)
    for offset in range(40, 640, _PITCH):
        _ = cv2.line(image, (30, offset), (650, offset), _RULING_VALUE, 1)
        _ = cv2.line(image, (offset, 30), (offset, 650), _RULING_VALUE, 1)
    return image


def test_proved_ruling_is_erased_and_its_period_reported() -> None:
    # Given a sheet whose only marks are faint periodic ruling
    image = _ruled_sheet()

    # When the page is cleaned
    result = clean_page(image)

    # Then the ruling is gone and the proved period is reported
    assert result.applied is True
    assert result.removed_ink_ratio > 0.80
    assert result.grid_period is not None
    assert abs(result.grid_period - _PITCH) <= 2.0
    assert result.horizontal_rule_count >= 4
    assert result.vertical_rule_count >= 4


def test_drawn_strokes_survive_the_ruling_they_are_drawn_over() -> None:
    # Given a part outline drawn across the same ruled sheet
    image = _ruled_sheet()
    _ = cv2.rectangle(image, (180, 180), (520, 460), _STROKE_VALUE, 5)

    # When the page is cleaned
    result = clean_page(image)

    # Then the outline stays dark on every edge while the ruling is removed
    assert result.grid_period is not None
    for y, x in ((180, 350), (460, 350), (320, 180), (320, 520)):
        assert int(cast("int", result.image[y, x])) < _STROKE_LIMIT


def test_a_lone_straight_line_is_never_mistaken_for_printed_ruling() -> None:
    # Given one long thin straight line with no periodic family behind it
    image = np.full((680, 680), 252, dtype=np.uint8)
    _ = cv2.line(image, (40, 300), (640, 300), _STROKE_VALUE, 1)

    # When the page is cleaned
    result = clean_page(image)

    # Then nothing is erased, because a single rail proves no grid
    assert result.applied is False
    assert result.grid_period is None
    assert np.array_equal(result.image, image)


def test_ruling_proved_on_one_axis_alone_is_kept() -> None:
    # Given evenly spaced lines running in one direction only
    image = np.full((680, 680), 252, dtype=np.uint8)
    for offset in range(40, 640, _PITCH):
        _ = cv2.line(image, (30, offset), (650, offset), _RULING_VALUE, 1)

    # When the page is cleaned
    result = clean_page(image)

    # Then a one-way pattern is left alone, since a drawing may repeat too
    assert result.applied is False
    assert result.grid_period is None


def test_cleanup_can_be_disabled_without_touching_the_page() -> None:
    # Given a ruled sheet and a caller that opts out
    image = _ruled_sheet()

    # When cleanup is disabled
    result = clean_page(image, PageCleanupConfig(enabled=False))

    # Then the source page is returned unchanged
    assert result.applied is False
    assert np.array_equal(result.image, image)
