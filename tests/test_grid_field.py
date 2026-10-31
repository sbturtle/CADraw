from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.grid_field import ink_above_ruling
from cadraw.preprocessing import illumination_flattened

type GrayImage = NDArray[np.uint8]

_PAPER = 236.0
_RULING_VALUE = 200
_RULING_PITCH = 30
_CONTRAST = 8.0


def _ruled_page(height: int = 480, width: int = 640) -> GrayImage:
    """Return evenly lit graph paper with a little sensor noise."""
    rng = np.random.default_rng(11)
    noisy = np.full((height, width), _PAPER) + rng.normal(0.0, 1.0, (height, width))
    image = np.asarray(np.clip(noisy, 0, 255), dtype=np.uint8)
    for offset in range(15, width, _RULING_PITCH):
        _ = cv2.line(image, (offset, 0), (offset, height - 1), _RULING_VALUE, 1)
    for offset in range(15, height, _RULING_PITCH):
        _ = cv2.line(image, (0, offset), (width - 1, offset), _RULING_VALUE, 1)
    return image


def _standing(image: GrayImage) -> NDArray[np.bool_]:
    flattened = illumination_flattened(image)
    height, width = cast("tuple[int, int]", flattened.shape[:2])
    return ink_above_ruling(
        flattened,
        _CONTRAST,
        np.zeros((height, width), dtype=np.bool_),
    )


def test_the_ruling_alone_does_not_stand_out() -> None:
    # Given graph paper with nothing drawn on it
    image = _ruled_page()

    # When the ink standing out of the ruling is read
    standing = _standing(image)

    # Then no rail is taken for ink
    assert int(np.count_nonzero(standing)) == 0


def test_a_faint_stroke_as_dark_as_the_ruling_stands_out() -> None:
    # Given a faint slanted pencil stroke drawn barely darker than the rails
    image = _ruled_page()
    _ = cv2.line(image, (120, 200), (520, 236), 190, 2)

    # When the ink standing out of the ruling is read
    standing = _standing(image)

    # Then most of the stroke is ink, crossing rails included
    along = [
        bool(np.any(standing[round(200 + (x - 120) * 0.09) - 3 :][:7, x]))
        for x in range(140, 500)
    ]
    assert sum(along) >= 0.9 * len(along)


def test_a_stroke_traced_over_a_rail_still_stands_out() -> None:
    # Given a pencil stroke drawn along one rail, darker than the ruling
    image = _ruled_page()
    _ = cv2.line(image, (100, 225), (540, 225), 150, 2)

    # When the ink standing out of the ruling is read
    standing = _standing(image)

    # Then the rail's own darkness is taken away, not the stroke
    assert int(np.count_nonzero(standing[222:229, 140:500].any(axis=0))) >= 340


def test_a_darker_stretch_of_one_rail_is_not_ink() -> None:
    # Given one rail printed a little darker along part of its length
    image = _ruled_page()
    _ = cv2.line(image, (255, 60), (255, 420), 185, 1)

    # When the ink standing out of the ruling is read
    standing = _standing(image)

    # Then the stretch, lying on the rail all along, stays ruling
    assert int(np.count_nonzero(standing[:, 250:261])) == 0
