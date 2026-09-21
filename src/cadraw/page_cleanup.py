"""Deterministic removal of printed graph-paper ruling before detection.

Work-order sketches are routinely drawn on ruled paper. Ruling is far fainter
than the drawing, so a locally adaptive threshold promotes every rail to a
full-strength stroke and the sheet's grid then outnumbers the part outline by
an order of magnitude. Erasing it from the source raster keeps the choice of
preprocessing variant from deciding whether the paper or the drawing wins.

Nothing is erased on suspicion. A rail is removed only as a member of a
periodic family proved on *both* axes, which is what distinguishes ruled paper
from a drawing's own repeated features, and only where the ink is thinner than
a drawn stroke. A long straight hand-drawn line is therefore kept even though
it looks exactly like a rule in isolation, and the returned result reports the
period and rail counts behind every pixel that was taken out.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.grid_filter import base_period_rails

type GrayImage = NDArray[np.uint8]

_MIN_RAIL_KERNEL = 15
_PAPER_VALUE = 255


@dataclass(frozen=True, slots=True)
class PageCleanupConfig:
    """Evidence thresholds for erasing printed rules and periodic ruling."""

    enabled: bool = True
    grid_span_ratio: float = 0.35
    maximum_rule_thickness: int = 2
    rail_drift_tolerance: int = 5
    ink_value_cut: int = 225
    rail_band_width: float = 6.0
    background_kernel_ratio: float = 0.08
    minimum_background_kernel: int = 31
    maximum_background_kernel: int = 301


@dataclass(frozen=True, slots=True)
class PageCleanupResult:
    """A cleaned raster plus the evidence for every mark that was erased."""

    image: GrayImage
    removed_ink_ratio: float
    horizontal_rule_count: int
    vertical_rule_count: int
    grid_period: float | None
    horizontal_rails: tuple[float, ...] = ()
    vertical_rails: tuple[float, ...] = ()

    @property
    def applied(self) -> bool:
        """Return whether any printed mark was actually erased."""
        return self.removed_ink_ratio > 0.0


def _odd(value: int, minimum: int = 3) -> int:
    """Return an odd kernel size accepted by OpenCV."""
    bounded = max(minimum, value)
    return bounded if bounded % 2 else bounded + 1


def _background(image: GrayImage, config: PageCleanupConfig) -> GrayImage:
    """Estimate the slowly varying paper background behind the ink."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    scaled = int(min(height, width) * config.background_kernel_ratio)
    size = _odd(
        min(
            config.maximum_background_kernel,
            max(config.minimum_background_kernel, scaled),
        ),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return np.asarray(
        cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
        dtype=np.uint8,
    )


def _ink_mask(image: GrayImage, config: PageCleanupConfig) -> GrayImage:
    """Mark every pixel darker than the flattened paper, however faint."""
    background = _background(image, config)
    flattened = cv2.divide(image, np.maximum(background, 1), scale=_PAPER_VALUE)
    return np.asarray(
        (np.asarray(flattened) < config.ink_value_cut).astype(np.uint8) * _PAPER_VALUE,
        dtype=np.uint8,
    )


def _thin_rails(
    ink: GrayImage,
    span: int,
    config: PageCleanupConfig,
    *,
    horizontal: bool,
) -> GrayImage:
    """Extract straight thin rails of one orientation from an ink mask.

    An opening with a long one-pixel-wide kernel keeps only ink that stays on
    a single row or column for the whole span, which freehand strokes do not.
    The mask is first smeared across the rail, because a page photographed or
    fed a fraction of a degree askew drifts off its row long before the span
    is reached, and is mapped back onto real ink afterwards. Ink lying in a run
    thicker than printed ruling is then given back, so a drawn line that
    happens to be straight is not mistaken for a rule.
    """
    drift_size = (
        (1, config.rail_drift_tolerance)
        if horizontal
        else (config.rail_drift_tolerance, 1)
    )
    line_size = (span, 1) if horizontal else (1, span)
    thickness_size = (
        (1, config.maximum_rule_thickness + 1)
        if horizontal
        else (config.maximum_rule_thickness + 1, 1)
    )
    drift_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, drift_size)
    smeared = cv2.dilate(ink, drift_kernel)
    rails = cv2.morphologyEx(
        smeared,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, line_size),
    )
    on_ink = cv2.bitwise_and(cv2.dilate(rails, drift_kernel), ink)
    thick = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, thickness_size),
    )
    return np.asarray(
        cv2.bitwise_and(on_ink, cv2.bitwise_not(thick)),
        dtype=np.uint8,
    )


def _rail_offsets(rails: GrayImage, *, horizontal: bool) -> list[float]:
    """Return the row or column positions that carry rail ink."""
    axis = 1 if horizontal else 0
    present: NDArray[np.bool_] = np.asarray(rails) > 0
    carries: NDArray[np.bool_] = present.any(axis=axis)
    return [float(index) for index in np.nonzero(carries)[0].tolist()]


def _rail_span(dimension: int, ratio: float) -> int:
    """Return the straight-run length a rail must hold, in pixels."""
    return max(_MIN_RAIL_KERNEL, int(dimension * ratio))


@dataclass(frozen=True, slots=True)
class _ProvedRuling:
    """One periodic ruling family proved on both axes."""

    mask: GrayImage
    period: float
    horizontal_count: int
    vertical_count: int
    horizontal_rails: tuple[float, ...] = ()
    vertical_rails: tuple[float, ...] = ()


def _periodic_rails(
    ink: GrayImage,
    config: PageCleanupConfig,
) -> _ProvedRuling | None:
    """Return ruling rails only when a period is proved on both axes.

    Graph paper is periodic in both directions, so requiring both keeps a
    drawing's own repeated features, such as a row of slots, from being read
    as ruling and erased.
    """
    height, width = cast("tuple[int, int]", ink.shape[:2])
    horizontal = _thin_rails(
        ink,
        _rail_span(width, config.grid_span_ratio),
        config,
        horizontal=True,
    )
    vertical = _thin_rails(
        ink,
        _rail_span(height, config.grid_span_ratio),
        config,
        horizontal=False,
    )
    horizontal_rails = base_period_rails(
        _rail_offsets(horizontal, horizontal=True),
        config.rail_band_width,
    )
    vertical_rails = base_period_rails(
        _rail_offsets(vertical, horizontal=False),
        config.rail_band_width,
    )
    if not horizontal_rails or not vertical_rails:
        return None
    gaps = [second - first for first, second in pairwise(horizontal_rails)]
    return _ProvedRuling(
        mask=np.asarray(cv2.bitwise_or(horizontal, vertical), dtype=np.uint8),
        period=float(np.median(np.asarray(gaps, dtype=np.float64))),
        horizontal_count=len(horizontal_rails),
        vertical_count=len(vertical_rails),
        horizontal_rails=horizontal_rails,
        vertical_rails=vertical_rails,
    )


def clean_page(
    image: GrayImage,
    config: PageCleanupConfig | None = None,
) -> PageCleanupResult:
    """Erase proved printed rules and ruling from one grayscale page."""
    settings = config if config is not None else PageCleanupConfig()
    source = np.ascontiguousarray(image, dtype=np.uint8)
    if not settings.enabled:
        return PageCleanupResult(source, 0.0, 0, 0, None)
    ink = _ink_mask(source, settings)
    ruling = _periodic_rails(ink, settings)
    if ruling is None:
        return PageCleanupResult(source, 0.0, 0, 0, None)
    removal_mask = np.asarray(ruling.mask) > 0
    ink_mask = np.asarray(ink) > 0
    ink_pixels = max(1, int(np.count_nonzero(ink_mask)))
    cleaned = source.copy()
    cleaned[removal_mask] = _background(source, settings)[removal_mask]
    return PageCleanupResult(
        image=np.asarray(cleaned, dtype=np.uint8),
        removed_ink_ratio=int(np.count_nonzero(removal_mask & ink_mask)) / ink_pixels,
        horizontal_rule_count=ruling.horizontal_count,
        vertical_rule_count=ruling.vertical_count,
        grid_period=ruling.period,
        horizontal_rails=ruling.horizontal_rails,
        vertical_rails=ruling.vertical_rails,
    )
