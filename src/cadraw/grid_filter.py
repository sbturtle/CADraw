"""Strict periodic-grid filtering for image-space candidates."""

from dataclasses import dataclass
from math import atan2, degrees
from statistics import median
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cadraw.models import LineCandidate, PolylineCandidate, ShapeCandidate

_SHAPE_SQUARE_RATIO = 0.65
_SHAPE_CELL_EXTENT_RATIO = 0.08
_SHAPE_SIZE_TOLERANCE_RATIO = 0.30

type GrayImage = NDArray[np.uint8]

_HALF_TURN_DEGREES = 180.0
_RIGHT_ANGLE_DEGREES = 90.0


@dataclass(frozen=True, slots=True)
class GridFilterConfig:
    """Evidence thresholds for proving and removing a paper grid."""

    enabled: bool = True
    minimum_family_count: int = 6
    axis_tolerance_degrees: float = 5.0
    spacing_cv_limit: float = 0.20
    minimum_span_ratio: float = 0.55
    band_width: float = 6.0
    maximum_grid_intensity: float = 135.0


@dataclass(frozen=True, slots=True)
class ShapeGridFilterResult:
    """Shape candidates plus evidence that a periodic grid was removed."""

    candidates: tuple[ShapeCandidate, ...]
    grid_detected: bool = False
    removed_count: int = 0
    grid_like_count: int = 0


def _line_length(candidate: LineCandidate) -> float:
    """Return one candidate's Euclidean length."""
    return candidate.start.distance_to(candidate.end)


def _angle(candidate: LineCandidate) -> float:
    """Return an orientation-independent line angle."""
    return (
        degrees(
            atan2(
                candidate.end.y - candidate.start.y,
                candidate.end.x - candidate.start.x,
            ),
        )
        % _HALF_TURN_DEGREES
    )


def _axis_delta(candidate: LineCandidate, axis: float) -> float:
    """Return the smallest angle between a line and an axis."""
    difference = abs(_angle(candidate) - axis)
    return min(difference, _HALF_TURN_DEGREES - difference)


def _cluster_offsets(offsets: list[float], band_width: float) -> tuple[float, ...]:
    """Collapse duplicate Hough rails into stable normal offsets."""
    if not offsets:
        return ()
    clusters: list[list[float]] = [[offsets[0]]]
    for offset in offsets[1:]:
        if offset - clusters[-1][-1] <= band_width:
            clusters[-1].append(offset)
        else:
            clusters.append([offset])
    return tuple(float(median(cluster)) for cluster in clusters)


def _regular_family(
    offsets: list[float],
    config: GridFilterConfig,
) -> tuple[float, ...]:
    """Return offsets only when enough rails have regular spacing."""
    clustered = _cluster_offsets(sorted(offsets), config.band_width)
    if len(clustered) < config.minimum_family_count:
        return ()
    spacings = np.diff(np.asarray(clustered, dtype=np.float64))
    if len(spacings) == 0:
        return ()
    mean_spacing = float(np.mean(spacings))
    if mean_spacing <= 0.0:
        return ()
    spacing_cv = float(np.std(spacings) / mean_spacing)
    if spacing_cv > config.spacing_cv_limit:
        return ()
    return clustered


def _grid_families(
    image: GrayImage,
    candidates: tuple[LineCandidate, ...],
    config: GridFilterConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Prove orthogonal regular line families from long candidates."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    horizontal_offsets = [
        (candidate.start.y + candidate.end.y) / 2.0
        for candidate in candidates
        if (
            _line_length(candidate) >= width * config.minimum_span_ratio
            and _axis_delta(candidate, 0.0) <= config.axis_tolerance_degrees
        )
    ]
    vertical_offsets = [
        (candidate.start.x + candidate.end.x) / 2.0
        for candidate in candidates
        if (
            _line_length(candidate) >= height * config.minimum_span_ratio
            and _axis_delta(candidate, _RIGHT_ANGLE_DEGREES)
            <= config.axis_tolerance_degrees
        )
    ]
    horizontal = _regular_family(horizontal_offsets, config)
    vertical = _regular_family(vertical_offsets, config)
    if not horizontal or not vertical:
        return (), ()
    return horizontal, vertical


def _sample_intensity(
    image: GrayImage,
    candidate: LineCandidate,
) -> float:
    """Estimate the darkest ink value along a candidate's narrow band."""
    start = candidate.start
    end = candidate.end
    length = _line_length(candidate)
    if length == 0.0:
        return 255.0
    direction = ((end.x - start.x) / length, (end.y - start.y) / length)
    normal = (-direction[1], direction[0])
    sample_count = max(12, min(160, int(length / 4.0)))
    values: list[int] = []
    height, width = cast("tuple[int, int]", image.shape[:2])
    for fraction in np.linspace(0.0, 1.0, sample_count):
        center_x = start.x + (end.x - start.x) * float(fraction)
        center_y = start.y + (end.y - start.y) * float(fraction)
        band_values: list[int] = []
        for offset in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            x = min(width - 1, max(0, round(center_x + normal[0] * offset)))
            y = min(height - 1, max(0, round(center_y + normal[1] * offset)))
            band_values.append(int(cast("int", image[y, x])))
        values.append(min(band_values))
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _matches_family(
    candidate: LineCandidate,
    offsets: tuple[float, ...],
    horizontal: bool,
    config: GridFilterConfig,
) -> bool:
    """Return whether a candidate lies on a proved grid rail."""
    offset = (
        (candidate.start.y + candidate.end.y) / 2.0
        if horizontal
        else (candidate.start.x + candidate.end.x) / 2.0
    )
    return any(
        abs(offset - family_offset) <= config.band_width for family_offset in offsets
    )


def filter_periodic_grid_candidates(
    image: GrayImage,
    candidates: tuple[LineCandidate, ...],
    config: GridFilterConfig | None = None,
) -> tuple[LineCandidate, ...]:
    """Remove only light rails from a proven orthogonal periodic grid."""
    settings = config if config is not None else GridFilterConfig()
    if not settings.enabled or not candidates:
        return candidates
    horizontal, vertical = _grid_families(image, candidates, settings)
    if not horizontal or not vertical:
        return candidates
    retained: list[LineCandidate] = []
    for candidate in candidates:
        on_grid = (
            _matches_family(candidate, horizontal, horizontal=True, config=settings)
            and _axis_delta(candidate, 0.0) <= settings.axis_tolerance_degrees
        ) or (
            _matches_family(candidate, vertical, horizontal=False, config=settings)
            and _axis_delta(candidate, _RIGHT_ANGLE_DEGREES)
            <= settings.axis_tolerance_degrees
        )
        if (
            not on_grid
            or _sample_intensity(image, candidate) <= settings.maximum_grid_intensity
        ):
            retained.append(candidate)
    return tuple(retained)


def _shape_bounds(
    candidate: PolylineCandidate,
) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounds of one polygon candidate."""
    return (
        min(float(point.x) for point in candidate.points),
        min(float(point.y) for point in candidate.points),
        max(float(point.x) for point in candidate.points),
        max(float(point.y) for point in candidate.points),
    )


def _shape_size(
    candidate: PolylineCandidate,
) -> tuple[float, float]:
    """Return one polygon's width and height."""
    bounds = _shape_bounds(candidate)
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def _shape_center(
    candidate: PolylineCandidate,
) -> tuple[float, float]:
    """Return one polygon's center."""
    bounds = _shape_bounds(candidate)
    return (0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3]))


def _regular_shape_offsets(
    offsets: list[float],
    config: GridFilterConfig,
) -> tuple[float, ...]:
    """Return regularly spaced centers from a candidate coordinate family."""
    clustered = _cluster_offsets(sorted(offsets), config.band_width)
    if len(clustered) < config.minimum_family_count:
        return ()
    spacings = np.diff(np.asarray(clustered, dtype=np.float64))
    if len(spacings) == 0:
        return ()
    mean_spacing = float(np.mean(spacings))
    if mean_spacing <= config.band_width * 2.0:
        return ()
    spacing_cv = float(np.std(spacings) / mean_spacing)
    if spacing_cv > config.spacing_cv_limit:
        return ()
    return clustered


def inspect_periodic_grid_shapes(
    candidates: tuple[ShapeCandidate, ...],
    width: int,
    height: int,
    config: GridFilterConfig | None = None,
) -> ShapeGridFilterResult:
    """Report small polygons proven to form a two-dimensional paper grid."""
    settings = config if config is not None else GridFilterConfig()
    grid_like_count = 0
    grid_members: tuple[PolylineCandidate, ...] = ()
    if settings.enabled:
        polygons = tuple(
            candidate
            for candidate in candidates
            if isinstance(candidate, PolylineCandidate)
        )
        minimum_family_members = settings.minimum_family_count * 2
        if len(polygons) >= minimum_family_members:
            maximum_cell_extent = min(width, height) * _SHAPE_CELL_EXTENT_RATIO
            small_polygons = tuple(
                candidate
                for candidate in polygons
                if (
                    max(_shape_size(candidate)) <= maximum_cell_extent
                    and min(_shape_size(candidate)) / max(_shape_size(candidate))
                    >= _SHAPE_SQUARE_RATIO
                )
            )
            grid_like_count = len(small_polygons)
            if len(small_polygons) >= minimum_family_members:
                sizes = tuple(_shape_size(candidate) for candidate in small_polygons)
                reference_width = float(
                    np.median(np.asarray([size[0] for size in sizes])),
                )
                reference_height = float(
                    np.median(np.asarray([size[1] for size in sizes])),
                )
                size_tolerance = max(
                    settings.band_width,
                    max(reference_width, reference_height)
                    * _SHAPE_SIZE_TOLERANCE_RATIO,
                )
                same_size = tuple(
                    candidate
                    for candidate, (candidate_width, candidate_height) in zip(
                        small_polygons,
                        sizes,
                        strict=True,
                    )
                    if (
                        abs(candidate_width - reference_width) <= size_tolerance
                        and abs(candidate_height - reference_height) <= size_tolerance
                    )
                )
                grid_like_count = len(same_size)
                if len(same_size) >= minimum_family_members:
                    horizontal = _regular_shape_offsets(
                        [_shape_center(candidate)[1] for candidate in same_size],
                        settings,
                    )
                    vertical = _regular_shape_offsets(
                        [_shape_center(candidate)[0] for candidate in same_size],
                        settings,
                    )
                    if horizontal and vertical:
                        grid_members = tuple(
                            candidate
                            for candidate in same_size
                            if (
                                any(
                                    abs(_shape_center(candidate)[1] - offset)
                                    <= settings.band_width
                                    for offset in horizontal
                                )
                                and any(
                                    abs(_shape_center(candidate)[0] - offset)
                                    <= settings.band_width
                                    for offset in vertical
                                )
                            )
                        )
                        if len(grid_members) < minimum_family_members:
                            grid_members = ()
    return ShapeGridFilterResult(
        candidates,
        grid_detected=bool(grid_members),
        removed_count=len(grid_members),
        grid_like_count=grid_like_count,
    )
