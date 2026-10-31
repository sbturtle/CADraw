"""Faint pencil on graph paper, separated from the ruling it is drawn over.

Graph-paper rails are often as dark as a faint pencil stroke, so no single
cut on darkness tells them apart. What does is shape: a rail runs straight
across the sheet with the same darkness all along it, while a stroke crosses
it or runs beside it at a slant. Each rail's darkness is read as the median
along it and subtracted, leaving the strokes.

A photographed sheet is seen in perspective and bends, so the rails' angle
drifts across it. The page is read in overlapping tiles, each finding its
own two rail families' angles, and the rails' fields are blended.
"""

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

type GrayImage = NDArray[np.uint8]
type FloatImage = NDArray[np.float32]

_TILE = 384
_STEP = 192
_FAMILIES = (0.0, 90.0)
_BACKGROUND_SIGMA = 10.0
_ANGLE_CLIP_PERCENTILE = 97.0
_COARSE_SPAN = 6.0
_COARSE_STEP = 1.0
_FINE_SPAN = 0.6
_FINE_STEP = 0.15
_PROFILE_SIGMA = 3.0
_PROFILE_TRIM = 5
_RAIL_MARGIN = _TILE // 8
_RAIL_FLOOR = 0.5
_RAIL_SHARE = 0.5
_RAIL_PERCENTILE = 80.0
_MINIMUM_RAILS = 3
_RAIL_CAP = 1.5
_WINDOW_FLOOR = 1e-3
_ON_RAIL_DARKNESS = 2.0
_ON_RAIL_SHARE = 0.8
_REMNANT_LENGTH_RATIO = 0.06
_REMNANT_CONTRAST_RATIO = 2.0


def _rotated(
    tile: FloatImage,
    angle: float,
) -> tuple[FloatImage, NDArray[np.float64]]:
    """Rotate a tile about its centre, returning the image and its matrix."""
    height, width = cast("tuple[int, int]", tile.shape[:2])
    matrix = np.asarray(
        cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0),
        dtype=np.float64,
    )
    rotated = cv2.warpAffine(
        tile,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(rotated, dtype=np.float32), matrix


def _rail_energy(tile: FloatImage, angle: float) -> float:
    """Return how sharply the rotated tile's rows alternate rail and paper."""
    rotated, _ = _rotated(tile, angle)
    profile = np.asarray(rotated.mean(axis=1), dtype=np.float32)
    smooth = np.asarray(
        cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), _PROFILE_SIGMA),
        dtype=np.float32,
    ).ravel()
    trimmed = slice(_PROFILE_TRIM, -_PROFILE_TRIM)
    return float(np.var(profile[trimmed] - smooth[trimmed]))


def _halved(tile: FloatImage) -> FloatImage:
    """Return a tile at half its resolution."""
    return np.asarray(
        cv2.resize(tile, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        dtype=np.float32,
    )


def _family_angle(tile: FloatImage, base: float) -> float:
    """Return the rotation that lays one rail family along the rows.

    A long drawn line would otherwise win the search on its own darkness,
    so the tile's darkest ink is first clipped to the level of the rest.
    The angle is found coarsely at a quarter of the resolution, then
    refined at half.
    """
    clipped = np.minimum(tile, float(np.percentile(tile, _ANGLE_CLIP_PERCENTILE)))
    half = _halved(clipped)
    quarter = _halved(half)
    coarse = max(
        np.arange(base - _COARSE_SPAN, base + _COARSE_SPAN + 1e-6, _COARSE_STEP),
        key=lambda angle: _rail_energy(quarter, float(angle)),
    )
    return float(
        max(
            np.arange(coarse - _FINE_SPAN, coarse + _FINE_SPAN + 1e-6, _FINE_STEP),
            key=lambda angle: _rail_energy(half, float(angle)),
        ),
    )


def _rail_field(tile: FloatImage, angle: float) -> FloatImage:
    """Return each rail's own darkness, spread back along the rail.

    The median along a rail is the rail: a stroke crossing it, or running
    beside it at a slant, covers too little of its length to move it. A
    stroke traced over a rail does, so the field is capped at the tile's
    typical rail, and ink darker than the ruling stays ink. Without a few
    rails standing clear of the paper's grain, the tile has no ruling.
    """
    rotated, matrix = _rotated(tile, angle)
    profile = np.asarray(
        np.median(rotated[:, _RAIL_MARGIN:-_RAIL_MARGIN], axis=1),
        dtype=np.float32,
    )
    before = np.concatenate((profile[:1], profile[:-1]))
    after = np.concatenate((profile[1:], profile[-1:]))
    peaks = profile[(profile >= before) & (profile > after) & (profile > _RAIL_FLOOR)]
    if peaks.size < _MINIMUM_RAILS:
        return np.zeros_like(tile)
    rails = peaks[peaks >= _RAIL_SHARE * float(np.percentile(peaks, _RAIL_PERCENTILE))]
    if rails.size < _MINIMUM_RAILS:
        return np.zeros_like(tile)
    capped = np.minimum(profile, _RAIL_CAP * float(np.median(rails)))
    height, width = cast("tuple[int, int]", tile.shape[:2])
    restored = cv2.warpAffine(
        np.repeat(capped.reshape(-1, 1), width, axis=1),
        cv2.invertAffineTransform(matrix),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.asarray(restored, dtype=np.float32)


def _tile_starts(length: int) -> list[int]:
    """Return overlapping tile origins that cover one image axis."""
    return sorted({*range(0, length - _TILE + 1, _STEP), length - _TILE})


def _ruling(flattened: GrayImage) -> tuple[FloatImage, FloatImage]:
    """Return each pixel's darkness above the paper and the ruling's share.

    `flattened` is the frame divided by its paper background. A frame
    smaller than one tile is given no ruling.
    """
    darkness = 255.0 - np.asarray(flattened, dtype=np.float32)
    local = np.asarray(
        cv2.GaussianBlur(darkness, (0, 0), _BACKGROUND_SIGMA),
        dtype=np.float32,
    )
    ink = np.maximum(darkness - local, 0.0)
    height, width = cast("tuple[int, int]", ink.shape[:2])
    if min(height, width) < _TILE:
        return ink, np.zeros_like(ink)
    window = np.asarray(
        np.outer(np.hanning(_TILE), np.hanning(_TILE)) + _WINDOW_FLOOR,
        dtype=np.float32,
    )
    total = np.zeros_like(ink)
    weight = np.zeros_like(ink)
    for top in _tile_starts(height):
        for left in _tile_starts(width):
            rows = slice(top, top + _TILE)
            columns = slice(left, left + _TILE)
            tile = np.ascontiguousarray(ink[rows, columns])
            field = np.zeros_like(tile)
            for base in _FAMILIES:
                field = np.maximum(
                    field,
                    _rail_field(tile, _family_angle(tile, base)),
                )
            total[rows, columns] += field * window
            weight[rows, columns] += window
    return ink, np.asarray(total / weight, dtype=np.float32)


def ink_above_ruling(
    flattened: GrayImage,
    contrast: float,
    known: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return the faint ink, beyond `known`, that stands out of the ruling.

    A pixel stands out when its darkness exceeds the ruling's own there by
    `contrast`. A thin rail sampled at a slant is darker where it falls on
    a pixel's centre than its median, so a stretch of rail survives the
    subtraction now and then, barely. A run lying on a rail for longer
    than a stroke's piece crossing one, and standing out by less than twice
    `contrast`, is left out: pencil running beside a rail at a slant leaves
    it within a few cells, and pencil traced over it is darker than that.
    """
    ink, ruling = _ruling(flattened)
    residual = ink - ruling
    standing = np.asarray(
        np.logical_and(residual > contrast, np.logical_not(known)),
        dtype=np.uint8,
    )
    count, found, stats, _ = cv2.connectedComponentsWithStats(
        standing,
        connectivity=8,
    )
    labels = np.asarray(found, dtype=np.int64)
    flat_labels = labels.ravel()
    on_rail = np.bincount(
        flat_labels,
        weights=np.asarray(ruling > _ON_RAIL_DARKNESS, dtype=np.float64).ravel(),
        minlength=count,
    )
    sizes = np.maximum(np.bincount(flat_labels, minlength=count), 1)
    share = on_rail / sizes
    strength = (
        np.bincount(
            flat_labels,
            weights=np.asarray(residual, dtype=np.float64).ravel(),
            minlength=count,
        )
        / sizes
    )
    boxes = np.asarray(stats, dtype=np.int64)
    extent = np.maximum(
        boxes[:, cv2.CC_STAT_WIDTH],
        boxes[:, cv2.CC_STAT_HEIGHT],
    )
    height, width = cast("tuple[int, int]", flattened.shape[:2])
    remnant = (
        (share > _ON_RAIL_SHARE)
        & (extent >= _REMNANT_LENGTH_RATIO * min(height, width))
        & (strength < _REMNANT_CONTRAST_RATIO * contrast)
    )
    remnant[0] = True
    return np.logical_not(remnant)[labels]
