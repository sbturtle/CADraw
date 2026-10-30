"""Deterministic, non-destructive preprocessing for photographed drawings."""

from dataclasses import dataclass
from enum import StrEnum
from math import atan2, degrees, hypot
from pathlib import Path
from time import perf_counter
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cadraw.grid_filter import periodic_grid_ratio
from cadraw.models import LineCandidate, Point

GrayImage = NDArray[np.uint8]

_MIN_ADAPTIVE_BLOCK_SIZE = 3
_DUPLICATE_ANGLE_TOLERANCE = 5.0
_DUPLICATE_LENGTH_RATIO = 0.70
_MIN_COMPONENT_AREA = 9
_PAPER_SHAPE_TOLERANCE = 12.0
_PAPER_SHAPE_SATURATION = 120.0
_BACKDROP_SCALE = 0.25
_BACKDROP_SMOOTHING = 15
_BACKDROP_DARKNESS = 0.5
_MINIMUM_BACKDROP_SHARE = 0.02
_DEFAULT_VARIANT_ORDER = (
    "grayscale_adaptive",
    "clahe_adaptive",
    "shadow_removed_adaptive",
    "blur_otsu",
    "flattened_global",
)


class PreprocessingVariant(StrEnum):
    """Independent image preparation strategies shown to reviewers."""

    GRAYSCALE_ADAPTIVE = "grayscale_adaptive"
    CLAHE_ADAPTIVE = "clahe_adaptive"
    SHADOW_REMOVED_ADAPTIVE = "shadow_removed_adaptive"
    BLUR_OTSU = "blur_otsu"
    FLATTENED_GLOBAL = "flattened_global"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configurable kernels and thresholds for each preprocessing variant."""

    enabled: bool = False
    minimum_selection_score: float = 0.45
    adaptive_block_size: int = 35
    adaptive_constant: float = 7.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    shadow_kernel_ratio: float = 0.08
    shadow_min_kernel_size: int = 31
    shadow_max_kernel_size: int = 301
    blur_kernel_size: int = 5
    morphology_close_size: int = 5
    morphology_open_size: int = 3


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """One inspectable preprocessing result and its timing metadata."""

    variant: PreprocessingVariant
    grayscale: GrayImage
    enhanced: GrayImage
    mask: GrayImage
    detection_image: GrayImage
    edges: GrayImage
    width: int
    height: int
    processing_time_ms: float

    @property
    def foreground(self) -> GrayImage:
        """Return the binary stroke view used for quality scoring."""
        return self.mask


@dataclass(frozen=True, slots=True)
class ScoreContext:
    """Detection thresholds and cross-stage evidence used when scoring.

    `paper_shape_count` is how many closed shapes the variant produced that
    were then proved to be paper structure — ruled cells and printed table
    cells. It prices in the failure that every other signal rewards: a variant
    that amplifies stationery gains long, well-connected, plentiful geometry.
    Counting *proved paper* rather than surviving shapes leaves a drawing that
    genuinely holds many closed shapes unpenalized.

    A work-order form's own header and footer contribute a handful of cells on
    every variant, so a tolerance keeps that from deciding anything; the term
    only bites once a variant has manufactured a ruled sheet's worth of them.
    """

    minimum_line_length: float
    endpoint_tolerance: float
    paper_shape_count: int = 0


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Explainable score used to compare candidates from variants."""

    variant: PreprocessingVariant
    score: float
    meaningful_line_count: int
    average_line_length: float
    connected_endpoint_ratio: float
    geometry_coverage: float
    duplicate_line_ratio: float
    short_noise_ratio: float
    component_quality: float
    periodic_grid_ratio: float
    paper_shape_count: int


def _odd_size(value: int, minimum: int = 3) -> int:
    """Return an odd kernel or block size accepted by OpenCV."""
    bounded = max(minimum, value)
    return bounded if bounded % 2 else bounded + 1


def _scaled_kernel_size(
    image: GrayImage,
    config: PreprocessingConfig,
) -> int:
    """Scale the shadow kernel to the shorter image dimension."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    scaled = int(min(height, width) * config.shadow_kernel_ratio)
    return _odd_size(
        min(
            config.shadow_max_kernel_size,
            max(config.shadow_min_kernel_size, scaled),
        ),
        minimum=3,
    )


def _contrast_stretch(image: GrayImage) -> GrayImage:
    """Stretch an image without clipping its source array in place."""
    stretched = np.empty_like(image)
    _ = cv2.normalize(image, stretched, 0, 255, cv2.NORM_MINMAX)
    return stretched


def _illumination_flattened(
    image: GrayImage,
    config: PreprocessingConfig,
) -> GrayImage:
    """Divide out slow illumination so absolute ink darkness stays comparable."""
    kernel_size = _scaled_kernel_size(image, config)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    safe_background = np.maximum(background, 1).astype(np.uint8)
    return np.asarray(
        cv2.divide(image, safe_background, scale=255),
        dtype=np.uint8,
    )


def illumination_flattened(
    image: GrayImage,
    config: PreprocessingConfig | None = None,
) -> GrayImage:
    """Return the image divided by its slowly varying paper background."""
    return _illumination_flattened(image, config or PreprocessingConfig())


def backdrop_mask(image: GrayImage) -> NDArray[np.bool_] | None:
    """Return where a photograph shows the desk around the paper, if anywhere.

    Paper, even in shadow, stays more than half as bright as the paper's
    typical level, while a desk, a bag, or a mug beside it is darker than
    that. The page is first smoothed far beyond a pen stroke, so ink never
    counts as backdrop. A scan, or a photo filled by paper, shows too little
    backdrop to report any.
    """
    height, width = cast("tuple[int, int]", image.shape[:2])
    small = np.asarray(
        cv2.resize(
            image,
            None,
            fx=_BACKDROP_SCALE,
            fy=_BACKDROP_SCALE,
            interpolation=cv2.INTER_AREA,
        ),
        dtype=np.uint8,
    )
    smoothed = np.asarray(cv2.medianBlur(small, _BACKDROP_SMOOTHING), dtype=np.uint8)
    otsu, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = smoothed[smoothed > otsu]
    if paper.size == 0:
        return None
    paper_level = float(np.median(np.asarray(paper, dtype=np.float64)))
    backdrop = np.asarray(
        np.less_equal(smoothed, _BACKDROP_DARKNESS * paper_level),
        dtype=np.uint8,
    )
    if int(np.count_nonzero(backdrop)) < _MINIMUM_BACKDROP_SHARE * backdrop.size:
        return None
    full = cv2.resize(backdrop, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.greater(np.asarray(full, dtype=np.uint8), 0)


def _shadow_removed(
    image: GrayImage,
    config: PreprocessingConfig,
) -> GrayImage:
    """Normalize slow illumination changes with a morphology background."""
    return _contrast_stretch(_illumination_flattened(image, config))


def _global_mask(image: GrayImage) -> GrayImage:
    """Threshold one illumination-flattened image on absolute ink darkness.

    A global cut keeps faint printed backgrounds such as graph-paper ruling
    below the ink decision, which any locally adaptive threshold would instead
    promote to full-strength strokes.
    """
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return np.asarray(cv2.bitwise_not(threshold), dtype=np.uint8)


def _adaptive_mask(
    image: GrayImage,
    config: PreprocessingConfig,
) -> GrayImage:
    """Extract dark strokes while retaining text pixels for later OCR."""
    block_size = _odd_size(config.adaptive_block_size, _MIN_ADAPTIVE_BLOCK_SIZE)
    threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        config.adaptive_constant,
    )
    return np.asarray(cv2.bitwise_not(threshold), dtype=np.uint8)


def _morphology(
    mask: GrayImage,
    config: PreprocessingConfig,
) -> GrayImage:
    """Join small stroke gaps while limiting isolated-pixel growth."""
    close_size = _odd_size(config.morphology_close_size)
    open_size = _odd_size(config.morphology_open_size)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_size, open_size),
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return np.asarray(opened, dtype=np.uint8)


def _prepare_variant(
    image: GrayImage,
    variant: PreprocessingVariant,
    config: PreprocessingConfig,
) -> tuple[GrayImage, GrayImage]:
    """Build one enhanced grayscale image and its foreground mask."""
    if variant is PreprocessingVariant.GRAYSCALE_ADAPTIVE:
        enhanced = image.copy()
        mask = _adaptive_mask(enhanced, config)
    elif variant is PreprocessingVariant.CLAHE_ADAPTIVE:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(
                config.clahe_tile_grid_size,
                config.clahe_tile_grid_size,
            ),
        )
        enhanced = np.asarray(clahe.apply(image), dtype=np.uint8)
        enhanced = _contrast_stretch(enhanced)
        mask = _adaptive_mask(enhanced, config)
    elif variant is PreprocessingVariant.SHADOW_REMOVED_ADAPTIVE:
        enhanced = _shadow_removed(image, config)
        mask = _adaptive_mask(enhanced, config)
    elif variant is PreprocessingVariant.FLATTENED_GLOBAL:
        enhanced = _illumination_flattened(image, config)
        mask = _global_mask(enhanced)
    else:
        blur_size = _odd_size(config.blur_kernel_size)
        enhanced = np.asarray(
            cv2.GaussianBlur(image, (blur_size, blur_size), 0),
            dtype=np.uint8,
        )
        _, threshold = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        mask = np.asarray(cv2.bitwise_not(threshold), dtype=np.uint8)
    return enhanced, _morphology(mask, config)


def preprocess_variant(
    image: GrayImage,
    variant: PreprocessingVariant,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Prepare one variant without mutating the source image."""
    selected_config = config or PreprocessingConfig()
    source = np.ascontiguousarray(image.copy(), dtype=np.uint8)
    started = perf_counter()
    enhanced, mask = _prepare_variant(source, variant, selected_config)
    detection_image = (
        np.asarray(cv2.bitwise_not(mask), dtype=np.uint8)
        if variant is PreprocessingVariant.FLATTENED_GLOBAL
        else enhanced.copy()
    )
    edges = np.asarray(
        cv2.Canny(detection_image, 50, 150),
        dtype=np.uint8,
    )
    height, width = cast("tuple[int, int]", source.shape[:2])
    elapsed = (perf_counter() - started) * 1000.0
    return PreprocessingResult(
        variant=variant,
        grayscale=source,
        enhanced=enhanced,
        mask=mask,
        detection_image=detection_image,
        edges=edges,
        width=width,
        height=height,
        processing_time_ms=elapsed,
    )


def preprocess_variants(
    image: GrayImage,
    config: PreprocessingConfig | None = None,
) -> tuple[PreprocessingResult, ...]:
    """Prepare all independent variants in a stable reviewer-facing order."""
    return tuple(
        preprocess_variant(image, variant, config) for variant in PreprocessingVariant
    )


def preprocess_gray_adaptive(
    image: GrayImage,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Prepare the grayscale adaptive-threshold variant."""
    return preprocess_variant(
        image,
        PreprocessingVariant.GRAYSCALE_ADAPTIVE,
        config,
    )


def preprocess_clahe_adaptive(
    image: GrayImage,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Prepare the CLAHE adaptive-threshold variant."""
    return preprocess_variant(
        image,
        PreprocessingVariant.CLAHE_ADAPTIVE,
        config,
    )


def preprocess_shadow_removed_adaptive(
    image: GrayImage,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Prepare the shadow-normalized adaptive-threshold variant."""
    return preprocess_variant(
        image,
        PreprocessingVariant.SHADOW_REMOVED_ADAPTIVE,
        config,
    )


def preprocess_blur_otsu(
    image: GrayImage,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Prepare the blur plus Otsu-threshold variant."""
    return preprocess_variant(
        image,
        PreprocessingVariant.BLUR_OTSU,
        config,
    )


def write_debug_outputs(
    original: GrayImage,
    results: tuple[PreprocessingResult, ...],
    output_directory: Path,
) -> None:
    """Write optional debug images without changing the source image."""
    output_directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(original, mode="L").save(output_directory / "original.png")
    for result in results:
        Image.fromarray(result.enhanced, mode="L").save(
            output_directory / f"{result.variant.value}_enhanced.png",
        )
        Image.fromarray(result.mask, mode="L").save(
            output_directory / f"{result.variant.value}_mask.png",
        )


def _line_angle(first: LineCandidate) -> float:
    """Return an undirected line angle in degrees."""
    angle = degrees(
        atan2(
            first.end.y - first.start.y,
            first.end.x - first.start.x,
        ),
    )
    return angle % 180.0


def _angle_delta(first: float, second: float) -> float:
    """Return the smallest undirected angle difference."""
    delta = abs(first - second)
    return min(delta, 180.0 - delta)


def _endpoint_connectivity(
    candidates: tuple[LineCandidate, ...],
    tolerance: float,
) -> float:
    """Measure how often candidate endpoints join another candidate."""
    if not candidates:
        return 0.0
    connected = 0
    for index, candidate in enumerate(candidates):
        for point in (candidate.start, candidate.end):
            if any(
                point.distance_to(other_point) <= tolerance
                for other_index, other in enumerate(candidates)
                if other_index != index
                for other_point in (other.start, other.end)
            ):
                connected += 1
    return connected / (2.0 * len(candidates))


def _duplicate_ratio(
    candidates: tuple[LineCandidate, ...],
    minimum_line_length: float,
) -> float:
    """Penalize repeated Hough segments describing one physical line."""
    if not candidates:
        return 0.0
    duplicate_indices: set[int] = set()
    for first_index, first in enumerate(candidates):
        first_length = first.start.distance_to(first.end)
        first_angle = _line_angle(first)
        first_center = Point(
            (first.start.x + first.end.x) / 2.0,
            (first.start.y + first.end.y) / 2.0,
        )
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            second_length = second.start.distance_to(second.end)
            second_center = Point(
                (second.start.x + second.end.x) / 2.0,
                (second.start.y + second.end.y) / 2.0,
            )
            if (
                _angle_delta(first_angle, _line_angle(second))
                <= _DUPLICATE_ANGLE_TOLERANCE
                and first_center.distance_to(second_center)
                <= max(6.0, minimum_line_length * 0.15)
                and min(first_length, second_length) / max(first_length, second_length)
                >= _DUPLICATE_LENGTH_RATIO
            ):
                duplicate_indices.update((first_index, second_index))
    return len(duplicate_indices) / len(candidates)


def score_candidates(
    variant: PreprocessingVariant,
    candidates: tuple[LineCandidate, ...],
    mask: GrayImage,
    context: ScoreContext,
) -> CandidateScore:
    """Score candidate quality using geometry and connected-component signals."""
    minimum_line_length = context.minimum_line_length
    endpoint_tolerance = context.endpoint_tolerance
    paper_shape_count = context.paper_shape_count
    lengths = tuple(
        candidate.start.distance_to(candidate.end) for candidate in candidates
    )
    height, width = cast("tuple[int, int]", mask.shape[:2])
    diagonal = max(1.0, hypot(float(width), float(height)))
    meaningful = sum(length >= minimum_line_length * 1.35 for length in lengths)
    average = sum(lengths) / len(lengths) if lengths else 0.0
    meaningful_score = min(1.0, meaningful / 8.0)
    average_score = min(1.0, average / (diagonal * 0.25))
    coverage = min(1.0, sum(lengths) / (2.0 * diagonal))
    short_noise = (
        sum(length < minimum_line_length * 1.35 for length in lengths) / len(lengths)
        if lengths
        else 1.0
    )
    raw_component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    component_count = raw_component_count
    foreground_pixels = max(
        1,
        int(np.count_nonzero(mask).item()),
    )
    statistics = np.asarray(stats, dtype=np.int64)
    area_column = 4
    component_areas = sorted(
        cast(
            "list[int]",
            statistics[1:component_count, area_column].tolist(),
        ),
        reverse=True,
    )
    component_quality = min(
        1.0,
        sum(area for area in component_areas[:8] if area >= _MIN_COMPONENT_AREA)
        / foreground_pixels,
    )
    connected = _endpoint_connectivity(candidates, endpoint_tolerance)
    duplicate = _duplicate_ratio(candidates, minimum_line_length)
    grid_ratio = periodic_grid_ratio(mask, candidates)
    paper_ratio = min(
        1.0,
        max(0.0, paper_shape_count - _PAPER_SHAPE_TOLERANCE) / _PAPER_SHAPE_SATURATION,
    )
    score = (
        0.35 * meaningful_score
        + 0.15 * average_score
        + 0.15 * connected
        + 0.10 * coverage
        + 0.10 * component_quality
        - 0.15 * duplicate
        - 0.25 * short_noise
        - 0.30 * grid_ratio
        - 0.25 * paper_ratio
    )
    return CandidateScore(
        variant=variant,
        score=score,
        meaningful_line_count=meaningful,
        average_line_length=average,
        connected_endpoint_ratio=connected,
        geometry_coverage=coverage,
        duplicate_line_ratio=duplicate,
        short_noise_ratio=short_noise,
        component_quality=component_quality,
        periodic_grid_ratio=grid_ratio,
        paper_shape_count=paper_shape_count,
    )
