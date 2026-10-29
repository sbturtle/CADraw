"""Ink evidence that lets two line pieces prove they are one stroke.

Probabilistic Hough breaks a hand-drawn stroke wherever its edge wobbles, and
the pieces it leaves can sit further apart than the near-collinear merge
tolerance. Bridging such a gap on geometry alone would invent a connection, so
this module answers one narrow question from the raster instead: is there ink
along the whole gap between two facing endpoints? Only a gap that the paper
itself confirms is bridged.
"""

from collections.abc import Callable
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import LineEntity, Point

type GrayImage = NDArray[np.uint8]
type InkBetween = Callable[[Point, Point], bool]

_MIN_SAMPLES = 8
_SAMPLE_SPACING = 2.0
_SEARCH_BAND = 2
_MIN_INK_RATIO = 0.90
_CROSSING_SPACING = 3.0
_MIN_CROSSING_BAND = 20
_WIDE_STROKES = 3.0
_MAXIMUM_WIDE_SHARE = 0.12


def ink_mask_from_grayscale(image: GrayImage) -> GrayImage:
    """Binarize one grayscale image into an ink mask with Otsu's threshold."""
    _, inverted = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return np.asarray(inverted, dtype=np.uint8)


def ink_between_factory(mask: GrayImage) -> InkBetween:
    """Return a predicate that checks a gap for continuous ink in the mask.

    The segment between the two points is sampled every couple of pixels and
    each sample looks for any ink inside a small square window, so a stroke
    that drifts a pixel or two off the straight gap still counts. The gap is
    continuous when almost every sample finds ink.
    """
    present: NDArray[np.bool_] = np.asarray(mask) > 0
    height, width = cast("tuple[int, int]", present.shape[:2])

    def ink_between(start: Point, end: Point) -> bool:
        length = start.distance_to(end)
        if length == 0.0:
            return True
        sample_count = max(_MIN_SAMPLES, int(length / _SAMPLE_SPACING))
        hits = 0
        for fraction in np.linspace(0.0, 1.0, sample_count):
            x = round(start.x + (end.x - start.x) * float(fraction))
            y = round(start.y + (end.y - start.y) * float(fraction))
            x_low = max(0, x - _SEARCH_BAND)
            x_high = min(width, x + _SEARCH_BAND + 1)
            y_low = max(0, y - _SEARCH_BAND)
            y_high = min(height, y + _SEARCH_BAND + 1)
            if x_low >= x_high or y_low >= y_high:
                continue
            if bool(present[y_low:y_high, x_low:x_high].any()):
                hits += 1
        return hits / sample_count >= _MIN_INK_RATIO

    return ink_between


def _run_across(
    present: NDArray[np.bool_],
    centre: tuple[float, float],
    normal: tuple[float, float],
    half_band: int,
) -> int | None:
    """Return the length of the ink run crossing one point of a line.

    The run is the one nearest the line within a couple of pixels, measured
    along the line's normal; `None` when no ink lies under the point.
    """
    height, width = cast("tuple[int, int]", present.shape[:2])
    profile = [
        0 <= (x := round(centre[0] + normal[0] * offset)) < width
        and 0 <= (y := round(centre[1] + normal[1] * offset)) < height
        and bool(cast("bool", present[y, x]))
        for offset in range(-half_band, half_band + 1)
    ]
    middle = half_band
    start = next(
        (
            middle + step * sign
            for step in range(_SEARCH_BAND + 1)
            for sign in (1, -1)
            if profile[middle + step * sign]
        ),
        None,
    )
    if start is None:
        return None
    low = start
    while low > 0 and profile[low - 1]:
        low -= 1
    high = start
    while high < len(profile) - 1 and profile[high + 1]:
        high += 1
    return high - low + 1


def wide_crossing_share(
    mask: GrayImage,
    start: Point,
    end: Point,
    wide: float,
) -> float:
    """Return the share of a line's inked samples where the ink is too wide.

    A drawn line is a band one pen stroke wide all along. A line the detector
    ran through a row of lettering crosses glyph after glyph, and wherever it
    meets a glyph's upright the ink across it is as tall as the letter. The
    share of samples whose crossing ink run exceeds `wide` pixels tells the
    two apart; a drawn line crossing another stroke only now and then keeps
    the share small.
    """
    length = start.distance_to(end)
    if length == 0.0:
        return 0.0
    present: NDArray[np.bool_] = np.asarray(mask) > 0
    direction = ((end.x - start.x) / length, (end.y - start.y) / length)
    normal = (-direction[1], direction[0])
    half_band = max(_MIN_CROSSING_BAND, round(2.0 * wide))
    runs = [
        run
        for fraction in np.linspace(
            0.0,
            1.0,
            max(_MIN_SAMPLES, int(length / _CROSSING_SPACING)),
        )
        if (
            run := _run_across(
                present,
                (
                    start.x + (end.x - start.x) * float(fraction),
                    start.y + (end.y - start.y) * float(fraction),
                ),
                normal,
                half_band,
            )
        )
        is not None
    ]
    if not runs:
        return 0.0
    return sum(run > wide for run in runs) / len(runs)


def without_lettering_crossings(
    lines: tuple[LineEntity, ...],
    mask: GrayImage,
    stroke_width: float,
) -> tuple[LineEntity, ...]:
    """Drop lines that run through lettering rather than along a stroke.

    A line is kept unless more than `_MAXIMUM_WIDE_SHARE` of its inked
    samples cross ink wider than `_WIDE_STROKES` pen strokes, which on the
    samples separated the lines drawn through rows of printed or handwritten
    text from every traced part edge.
    """
    wide = _WIDE_STROKES * stroke_width
    return tuple(
        line
        for line in lines
        if wide_crossing_share(mask, line.start, line.end, wide) <= _MAXIMUM_WIDE_SHARE
    )
