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

from cadraw.models import Point

type GrayImage = NDArray[np.uint8]
type InkBetween = Callable[[Point, Point], bool]

_MIN_SAMPLES = 8
_SAMPLE_SPACING = 2.0
_SEARCH_BAND = 2
_MIN_INK_RATIO = 0.90


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
