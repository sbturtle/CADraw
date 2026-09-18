"""OpenCV image preparation and deterministic line candidate detection."""

from dataclasses import dataclass, field
from math import atan2, degrees, hypot, isfinite, pi, sqrt
from pathlib import Path
from typing import cast, final, override

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.arbitration import (
    CandidateArbitrationConfig,
    arbitrate_line_candidates,
)
from cadraw.grid_filter import (
    filter_periodic_grid_candidates,
    inspect_periodic_grid_shapes,
)
from cadraw.models import (
    ArcCandidate,
    CircleCandidate,
    EntityId,
    GeometryRole,
    LineCandidate,
    Point,
    PolylineCandidate,
    ShapeCandidate,
)
from cadraw.table_filter import (
    TableFilterConfig,
    inspect_printed_table_shapes,
)

GrayImage = NDArray[np.uint8]

_MIN_DUPLICATE_CIRCLE_DISTANCE = 8.0
_MIN_DUPLICATE_RADIUS_RATIO = 0.55
_MIN_CIRCLE_APPROX_POINTS = 6
_MIN_CIRCLE_CIRCULARITY = 0.68
_MIN_CIRCLE_ASPECT_RATIO = 0.65
_RECTANGLE_APPROX_POINTS = 4
_MIN_RECTANGULARITY = 0.35
_MAX_POLYGON_APPROX_POINTS = 24
_MIN_POLYGON_SOLIDITY = 0.45
_POLYGON_APPROX_PERIMETER_RATIO = 0.01
_MIN_POLYGON_APPROX_EPSILON = 3.0
_MAX_SHAPE_EXTENT_RATIO = 0.92
_MIN_DUPLICATE_SHAPE_EXTENT_RATIO = 0.55
_AXIS_TOLERANCE_DEGREES = 5.0
_MIN_SELECTED_CANDIDATES = 3
_MIN_LSTSQ_RANK = 3
_MIN_ARC_FIT_POINTS = 24
_ARC_RESIDUAL_PIXEL_FLOOR = 2.5
_ARC_RESIDUAL_TO_RADIUS_RATIO = 0.08
_MIN_ARC_SPAN_DEGREES = 50.0
_MAX_ARC_SPAN_DEGREES = 320.0
_MAX_ARC_TO_BOX_RADIUS_RATIO = 3.0
_FRAME_RULE_AXIS_SLOPE = 0.09


@dataclass(frozen=True, slots=True)
class DrawingFieldConfig:
    """Settings for rejecting page clutter before the candidate budget."""

    enabled: bool = False
    frame_margin_ratio: float = 0.10
    minimum_cell_size: float = 80.0
    cell_size_ratio: float = 0.03
    angle_bin_size_degrees: float = 15.0
    minimum_angle_bins: int = 2
    minimum_field_score_ratio: float = 0.05
    field_band_ratio: float = 0.08
    endpoint_closure_distance: float = 18.0
    parallel_min_length_ratio: float = 0.12
    parallel_min_overlap_ratio: float = 0.35
    parallel_min_offset_ratio: float = 0.15
    parallel_max_offset_ratio: float = 0.35
    parallel_axis_tolerance_degrees: float = 7.0
    local_contrast_offset: float = 16.0
    periodic_min_family_count: int = 6
    periodic_min_span_ratio: float = 0.15
    periodic_min_spacing: float = 12.0
    periodic_max_spacing_ratio: float = 0.06
    periodic_spacing_cv_limit: float = 0.30
    periodic_band_width: float = 8.0
    periodic_max_intensity: float = 135.0


@final
class ImageReadError(Exception):
    """Raised only when an image cannot cross the file boundary."""

    def __init__(self, path: Path) -> None:
        """Store the failing path for user feedback."""
        super().__init__()
        self.path: Path = path

    @override
    def __str__(self) -> str:
        return f"이미지를 읽을 수 없습니다: {self.path}"


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """OpenCV parameters for the Phase 1 detector."""

    minimum_line_length: float = 80.0
    hough_threshold: int = 40
    maximum_line_gap: int = 18
    maximum_candidates: int = 32
    blur_kernel_size: int = 5
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    border_margin_ratio: float = 0.08
    frame_rule_margin_ratio: float = 0.22
    frame_rule_span_ratio: float = 0.60
    detect_closed_shapes: bool = False
    minimum_shape_area: float = 800.0
    minimum_shape_span: float = 30.0
    minimum_shape_radius: float = 12.0
    maximum_shape_candidates: int = 64
    shape_threshold: int | None = None
    arbitration: CandidateArbitrationConfig = field(
        default_factory=CandidateArbitrationConfig,
    )
    drawing_field: DrawingFieldConfig = field(default_factory=DrawingFieldConfig)
    printed_table: TableFilterConfig = field(default_factory=TableFilterConfig)


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Grayscale and edge views used by the detector."""

    grayscale: GrayImage
    edges: GrayImage
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ShapeDetectionEvidence:
    """Closed-shape candidates plus periodic-grid rejection evidence."""

    candidates: tuple[ShapeCandidate, ...]
    grid_detected: bool = False
    grid_candidate_count: int = 0
    grid_like_count: int = 0
    table_detected: bool = False
    table_candidate_count: int = 0
    candidate_limit_reached: bool = False


def load_grayscale_image(path: Path) -> GrayImage:
    """Read a supported image path as grayscale pixels."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise ImageReadError(path) from error
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageReadError(path)
    return np.asarray(image, dtype=np.uint8)


def automatic_source_region(
    image: GrayImage,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """Return a deterministic inset that excludes the photo border."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    return (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y,
    )


def preprocess_image(image: GrayImage, config: DetectionConfig) -> PreprocessedImage:
    """Create a blurred grayscale image and Canny edge map."""
    kernel_size = max(3, config.blur_kernel_size | 1)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(
        blurred,
        config.canny_low_threshold,
        config.canny_high_threshold,
    )
    image_shape = cast("tuple[int, int]", image.shape[:2])
    height, width = image_shape
    return PreprocessedImage(
        grayscale=image,
        edges=np.asarray(edges, dtype=np.uint8),
        width=int(width),
        height=int(height),
    )


def _candidate_confidence(length: float, width: int, height: int) -> float:
    """Assign a conservative deterministic confidence to a line length."""
    image_diagonal = max(1.0, hypot(width, height))
    return min(0.99, 0.55 + 0.44 * length / image_diagonal)


def _is_border_candidate(
    start: Point,
    end: Point,
    width: int,
    height: int,
    margin_ratio: float,
) -> bool:
    """Reject page edges while retaining strokes inside the page."""
    x_margin = width * margin_ratio
    y_margin = height * margin_ratio
    on_left = start.x <= x_margin and end.x <= x_margin
    on_right = start.x >= width - x_margin and end.x >= width - x_margin
    on_top = start.y <= y_margin and end.y <= y_margin
    on_bottom = start.y >= height - y_margin and end.y >= height - y_margin
    return on_left or on_right or on_top or on_bottom


def is_page_frame_rule(
    start: Point,
    end: Point,
    width: int,
    height: int,
    config: DetectionConfig,
) -> bool:
    """Reject a printed frame rule that runs margin to margin along one edge.

    Preprinted forms close their drawing field with a rule that crosses almost
    the whole sheet just inside an edge, which the plain border test misses
    because the rule sits a little further in than the page boundary. All three
    conditions are required together: the candidate must be axis-aligned, cross
    most of the sheet, and hug an edge. A stroke that merely sits low on the
    page, or a long stroke through the middle of the drawing, keeps all of its
    own geometry.
    """
    margin_x = width * config.frame_rule_margin_ratio
    margin_y = height * config.frame_rule_margin_ratio
    span_x = abs(end.x - start.x)
    span_y = abs(end.y - start.y)
    horizontal_rule = (
        span_y <= span_x * _FRAME_RULE_AXIS_SLOPE
        and span_x >= width * config.frame_rule_span_ratio
        and (
            max(start.y, end.y) <= margin_y or min(start.y, end.y) >= height - margin_y
        )
    )
    vertical_rule = (
        span_x <= span_y * _FRAME_RULE_AXIS_SLOPE
        and span_y >= height * config.frame_rule_span_ratio
        and (max(start.x, end.x) <= margin_x or min(start.x, end.x) >= width - margin_x)
    )
    return horizontal_rule or vertical_rule


def _line_angle_degrees(candidate: LineCandidate) -> float:
    """Return an orientation-independent angle for one candidate."""
    return (
        degrees(
            atan2(
                candidate.end.y - candidate.start.y,
                candidate.end.x - candidate.start.x,
            ),
        )
        % 180.0
    )


def _line_intensity(image: GrayImage, candidate: LineCandidate) -> float:
    """Estimate the darkest grayscale value along a candidate stroke."""
    length = candidate.start.distance_to(candidate.end)
    if length == 0.0:
        return 255.0
    direction = (
        (candidate.end.x - candidate.start.x) / length,
        (candidate.end.y - candidate.start.y) / length,
    )
    normal = (-direction[1], direction[0])
    height, width = cast("tuple[int, int]", image.shape[:2])
    sample_count = max(12, min(160, int(length / 4.0)))
    values: list[int] = []
    for fraction in np.linspace(0.0, 1.0, sample_count):
        center_x = candidate.start.x + (candidate.end.x - candidate.start.x) * float(
            fraction
        )
        center_y = candidate.start.y + (candidate.end.y - candidate.start.y) * float(
            fraction
        )
        band_values = [
            int(
                cast(
                    "int",
                    image[
                        min(
                            height - 1,
                            max(0, round(center_y + normal[1] * offset)),
                        ),
                        min(
                            width - 1,
                            max(0, round(center_x + normal[0] * offset)),
                        ),
                    ],
                ),
            )
            for offset in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
        ]
        values.append(min(band_values))
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _line_local_contrast(
    image: GrayImage,
    candidate: LineCandidate,
    offset: float,
) -> float:
    """Compare stroke darkness with pixels just outside its narrow band."""
    length = candidate.start.distance_to(candidate.end)
    if length == 0.0:
        return 0.0
    direction = (
        (candidate.end.x - candidate.start.x) / length,
        (candidate.end.y - candidate.start.y) / length,
    )
    normal = (-direction[1], direction[0])
    height, width = cast("tuple[int, int]", image.shape[:2])
    sample_count = max(12, min(80, int(length / 8.0)))
    inside: list[int] = []
    outside: list[int] = []
    for fraction in np.linspace(0.1, 0.9, sample_count):
        center_x = candidate.start.x + (candidate.end.x - candidate.start.x) * float(
            fraction
        )
        center_y = candidate.start.y + (candidate.end.y - candidate.start.y) * float(
            fraction
        )
        band_values: list[int] = []
        for band_offset in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
            sample_x = min(
                width - 1,
                max(0, round(center_x + normal[0] * band_offset)),
            )
            sample_y = min(
                height - 1,
                max(0, round(center_y + normal[1] * band_offset)),
            )
            band_values.append(int(cast("int", image[sample_y, sample_x])))
        inside.append(min(band_values))
        for sample_offset in (-offset, offset):
            sample_x = min(
                width - 1,
                max(0, round(center_x + normal[0] * sample_offset)),
            )
            sample_y = min(
                height - 1,
                max(0, round(center_y + normal[1] * sample_offset)),
            )
            outside.append(int(cast("int", image[sample_y, sample_x])))
    return max(
        0.0,
        float(np.median(np.asarray(outside, dtype=np.float32)))
        - float(np.median(np.asarray(inside, dtype=np.float32))),
    )


def _touches_frame(
    candidate: LineCandidate,
    width: int,
    height: int,
    margin_ratio: float,
) -> bool:
    """Return whether a candidate enters the photograph frame margin."""
    x_margin = width * margin_ratio
    y_margin = height * margin_ratio
    for fraction in np.linspace(0.0, 1.0, 5):
        x = candidate.start.x + (candidate.end.x - candidate.start.x) * float(
            fraction,
        )
        y = candidate.start.y + (candidate.end.y - candidate.start.y) * float(
            fraction,
        )
        if (
            x <= x_margin
            or x >= width - x_margin
            or y <= y_margin
            or y >= height - y_margin
        ):
            return True
    return False


def _cluster_periodic_offsets(
    offsets: list[float],
    band_width: float,
) -> tuple[float, ...]:
    """Collapse nearby Hough offsets into one rail position."""
    if not offsets:
        return ()
    ordered = sorted(offsets)
    clusters: list[list[float]] = [[ordered[0]]]
    for offset in ordered[1:]:
        if offset - clusters[-1][-1] <= band_width:
            clusters[-1].append(offset)
        else:
            clusters.append([offset])
    return tuple(float(np.median(np.asarray(cluster))) for cluster in clusters)


def _periodic_family_offsets(
    candidates: tuple[LineCandidate, ...],
    width: int,
    height: int,
    config: DrawingFieldConfig,
    horizontal: bool,
) -> tuple[float, ...]:
    """Find regularly spaced horizontal or vertical background rails."""
    minimum_span = min(width, height) * config.periodic_min_span_ratio
    offsets = [
        (candidate.start.y + candidate.end.y) / 2.0
        if horizontal
        else (candidate.start.x + candidate.end.x) / 2.0
        for candidate in candidates
        if candidate.start.distance_to(candidate.end) >= minimum_span
        and (
            abs(_line_angle_degrees(candidate)) <= _AXIS_TOLERANCE_DEGREES
            if horizontal
            else abs(_line_angle_degrees(candidate) - 90.0) <= _AXIS_TOLERANCE_DEGREES
        )
    ]
    clustered = _cluster_periodic_offsets(
        offsets,
        config.periodic_band_width,
    )
    if len(clustered) < config.periodic_min_family_count:
        return ()
    spacings = np.diff(np.asarray(clustered, dtype=np.float64))
    if len(spacings) == 0:
        return ()
    mean_spacing = float(np.mean(spacings))
    if not (
        config.periodic_min_spacing
        <= mean_spacing
        <= min(width, height) * config.periodic_max_spacing_ratio
    ):
        return ()
    spacing_cv = float(np.std(spacings) / mean_spacing)
    if spacing_cv > config.periodic_spacing_cv_limit:
        return ()
    return clustered


def _remove_periodic_rails(
    image: GrayImage,
    candidates: tuple[LineCandidate, ...],
    config: DrawingFieldConfig,
) -> tuple[LineCandidate, ...]:
    """Remove only light candidates supported by a regular rail family."""
    if not candidates:
        return ()
    height, width = cast("tuple[int, int]", image.shape[:2])
    families = (
        (
            True,
            _periodic_family_offsets(
                candidates,
                width,
                height,
                config,
                horizontal=True,
            ),
        ),
        (
            False,
            _periodic_family_offsets(
                candidates,
                width,
                height,
                config,
                horizontal=False,
            ),
        ),
    )
    if not any(offsets for _, offsets in families):
        return candidates
    retained: list[LineCandidate] = []
    for candidate in candidates:
        angle = _line_angle_degrees(candidate)
        is_light_rail = False
        for horizontal, offsets in families:
            axis_delta = abs(angle) if horizontal else abs(angle - 90.0)
            if axis_delta > _AXIS_TOLERANCE_DEGREES or not offsets:
                continue
            offset = (
                (candidate.start.y + candidate.end.y) / 2.0
                if horizontal
                else (candidate.start.x + candidate.end.x) / 2.0
            )
            if (
                min(abs(offset - family_offset) for family_offset in offsets)
                <= config.periodic_band_width
                and _line_intensity(image, candidate) >= config.periodic_max_intensity
            ):
                is_light_rail = True
                break
        if not is_light_rail:
            retained.append(candidate)
    return tuple(retained)


def _axis_delta(candidate: LineCandidate, axis: float) -> float:
    """Return the smallest difference between a line and an image axis."""
    difference = abs(_line_angle_degrees(candidate) - axis)
    return min(difference, 180.0 - difference)


def _select_parallel_axis_field_candidates(
    image: GrayImage,
    candidates: tuple[LineCandidate, ...],
    config: DrawingFieldConfig,
) -> tuple[LineCandidate, ...]:
    """Recover a drawing region from an overlapping axis-line pair."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    minimum_length = min(width, height) * config.parallel_min_length_ratio
    axis_candidates = tuple(
        (
            candidate,
            0.0
            if _axis_delta(candidate, 0.0) <= config.parallel_axis_tolerance_degrees
            else 90.0,
        )
        for candidate in candidates
        if candidate.start.distance_to(candidate.end) >= minimum_length
        and (
            _axis_delta(candidate, 0.0) <= config.parallel_axis_tolerance_degrees
            or _axis_delta(candidate, 90.0) <= config.parallel_axis_tolerance_degrees
        )
    )
    best_pair: (
        tuple[
            float,
            float,
            LineCandidate,
            LineCandidate,
            float,
            float,
        ]
        | None
    ) = None
    for first_index, (first, first_axis) in enumerate(axis_candidates):
        for second, second_axis in axis_candidates[first_index + 1 :]:
            if first_axis != second_axis:
                continue
            if first_axis == 0.0:
                offset = (
                    abs((first.start.y + first.end.y) - (second.start.y + second.end.y))
                    / 2.0
                )
                first_projection = sorted((first.start.x, first.end.x))
                second_projection = sorted((second.start.x, second.end.x))
            else:
                offset = (
                    abs((first.start.x + first.end.x) - (second.start.x + second.end.x))
                    / 2.0
                )
                first_projection = sorted((first.start.y, first.end.y))
                second_projection = sorted((second.start.y, second.end.y))
            overlap = max(
                0.0,
                min(first_projection[1], second_projection[1])
                - max(first_projection[0], second_projection[0]),
            )
            shorter_length = min(
                first.start.distance_to(first.end),
                second.start.distance_to(second.end),
            )
            if (
                offset < min(width, height) * config.parallel_min_offset_ratio
                or offset > min(width, height) * config.parallel_max_offset_ratio
                or overlap < shorter_length * config.parallel_min_overlap_ratio
            ):
                continue
            score = (
                first.start.distance_to(first.end)
                + second.start.distance_to(second.end)
                + overlap
            )
            if best_pair is None or score > best_pair[0]:
                best_pair = (
                    score,
                    first_axis,
                    first,
                    second,
                    offset,
                    shorter_length,
                )
    if best_pair is None:
        return ()
    _, axis, first, second, offset, shorter_length = best_pair
    min_x = min(
        first.start.x,
        first.end.x,
        second.start.x,
        second.end.x,
    )
    max_x = max(
        first.start.x,
        first.end.x,
        second.start.x,
        second.end.x,
    )
    min_y = min(
        first.start.y,
        first.end.y,
        second.start.y,
        second.end.y,
    )
    max_y = max(
        first.start.y,
        first.end.y,
        second.start.y,
        second.end.y,
    )
    x_margin = 0.35 * shorter_length if axis == 0.0 else offset
    y_margin = offset
    y_margin_before = 0.35 * shorter_length
    y_margin_after = 0.80 * shorter_length
    if axis == 0.0:
        x_low = min_x - x_margin
        x_high = max_x + x_margin
        y_low = min_y - y_margin
        y_high = max_y + y_margin
    else:
        x_low = min_x - x_margin
        x_high = max_x + x_margin
        y_low = min_y - y_margin_before
        y_high = max_y + y_margin_after
    selected = tuple(
        candidate
        for candidate in candidates
        if (
            x_low <= (candidate.start.x + candidate.end.x) / 2.0 <= x_high
            and y_low <= (candidate.start.y + candidate.end.y) / 2.0 <= y_high
        )
    )
    return selected if len(selected) >= _MIN_SELECTED_CANDIDATES else ()


@dataclass(slots=True)
class _DrawingFieldCell:
    """Accumulated support for one coarse image cell."""

    weight: float = 0.0
    angle_bins: set[int] = field(default_factory=set)
    candidate_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _DrawingFieldComponent:
    """Connected coarse cells describing one candidate drawing field."""

    score: float
    angle_count: int
    cells: tuple[tuple[int, int], ...]
    center_x: float
    center_y: float


def _build_drawing_field_cells(
    image: GrayImage,
    candidates: tuple[LineCandidate, ...],
    config: DrawingFieldConfig,
    cell_size: float,
) -> dict[tuple[int, int], _DrawingFieldCell]:
    """Accumulate contrast-weighted candidates into coarse image cells."""
    cells: dict[tuple[int, int], _DrawingFieldCell] = {}
    for index, candidate in enumerate(candidates):
        length = candidate.start.distance_to(candidate.end)
        contrast = _line_local_contrast(
            image,
            candidate,
            config.local_contrast_offset,
        )
        weight = min(length, 2.0 * cell_size) * (0.30 + contrast / 255.0)
        midpoint_x = (candidate.start.x + candidate.end.x) / 2.0
        midpoint_y = (candidate.start.y + candidate.end.y) / 2.0
        key = (int(midpoint_x // cell_size), int(midpoint_y // cell_size))
        cell = cells.setdefault(key, _DrawingFieldCell())
        cell.weight += weight
        angle_bin = int(
            _line_angle_degrees(candidate) / config.angle_bin_size_degrees,
        )
        cell.angle_bins.add(angle_bin)
        cell.candidate_indices.append(index)
    return cells


def _collect_drawing_field_components(
    cells: dict[tuple[int, int], _DrawingFieldCell],
    candidates: tuple[LineCandidate, ...],
    config: DrawingFieldConfig,
) -> tuple[_DrawingFieldComponent, ...]:
    """Find connected coarse-cell fields and their spatial centers."""
    unseen = set(cells)
    components: list[_DrawingFieldComponent] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            component.append(current)
            cell_x, cell_y = current
            for delta_x in (-1, 0, 1):
                for delta_y in (-1, 0, 1):
                    neighbor = (cell_x + delta_x, cell_y + delta_y)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
        weight = sum(cells[key].weight for key in component)
        angle_bins: set[int] = set()
        for key in component:
            angle_bins.update(cells[key].angle_bins)
        score = weight * min(
            1.0,
            len(angle_bins) / max(1, config.minimum_angle_bins + 1),
        )
        candidate_indices = tuple(
            index for key in component for index in cells[key].candidate_indices
        )
        component_candidates = tuple(candidates[index] for index in candidate_indices)
        min_x = min(
            point.x
            for candidate in component_candidates
            for point in (candidate.start, candidate.end)
        )
        min_y = min(
            point.y
            for candidate in component_candidates
            for point in (candidate.start, candidate.end)
        )
        max_x = max(
            point.x
            for candidate in component_candidates
            for point in (candidate.start, candidate.end)
        )
        max_y = max(
            point.y
            for candidate in component_candidates
            for point in (candidate.start, candidate.end)
        )
        components.append(
            _DrawingFieldComponent(
                score=score,
                angle_count=len(angle_bins),
                cells=tuple(component),
                center_x=(min_x + max_x) / 2.0,
                center_y=(min_y + max_y) / 2.0,
            ),
        )
    return tuple(components)


def _accepted_drawing_field_cells(
    components: tuple[_DrawingFieldComponent, ...],
    config: DrawingFieldConfig,
    width: int,
    height: int,
) -> set[tuple[int, int]]:
    """Choose the strongest central field and its aligned companion fields."""
    if not components:
        return set()
    best_score = max(component.score for component in components)
    qualified = tuple(
        component
        for component in components
        if component.score >= best_score * config.minimum_field_score_ratio
        and (component.angle_count >= config.minimum_angle_bins or len(components) == 1)
    )
    if not qualified:
        return set()
    anchor = min(
        qualified,
        key=lambda component: (
            hypot(
                (component.center_x - width / 2.0) / max(1.0, width),
                (component.center_y - height / 2.0) / max(1.0, height),
            ),
            -component.score,
        ),
    )
    accepted_cells: set[tuple[int, int]] = set()
    for drawing_field in qualified:
        if (
            abs(drawing_field.center_y - anchor.center_y)
            <= height * config.field_band_ratio
            or drawing_field is anchor
        ):
            accepted_cells.update(drawing_field.cells)
    return accepted_cells


def _close_drawing_field_candidates(
    candidates: tuple[LineCandidate, ...],
    selected_indices: set[int],
    closure_distance: float,
) -> set[int]:
    """Include only candidates joined to the selected stroke endpoints."""
    while True:
        additions = {
            index
            for index, candidate in enumerate(candidates)
            if index not in selected_indices
            and any(
                first_point.distance_to(second_point) <= closure_distance
                for selected_index in selected_indices
                for first_point in (
                    candidates[selected_index].start,
                    candidates[selected_index].end,
                )
                for second_point in (candidate.start, candidate.end)
            )
        }
        if not additions:
            return selected_indices
        selected_indices.update(additions)


def _select_drawing_field_candidates(
    image: GrayImage,
    candidates: tuple[LineCandidate, ...],
    config: DrawingFieldConfig,
) -> tuple[LineCandidate, ...]:
    """Keep candidates belonging to multi-angle dark-ink fields."""
    if not candidates:
        return ()
    height, width = cast("tuple[int, int]", image.shape[:2])
    frame_candidates = tuple(
        candidate
        for candidate in candidates
        if not _touches_frame(
            candidate,
            width,
            height,
            config.frame_margin_ratio,
        )
    )
    filtered = _remove_periodic_rails(image, frame_candidates, config)
    if not filtered:
        return ()
    parallel_selected = _select_parallel_axis_field_candidates(
        image,
        filtered,
        config,
    )
    if parallel_selected:
        return parallel_selected
    cell_size = max(
        config.minimum_cell_size,
        min(width, height) * config.cell_size_ratio,
    )
    cells = _build_drawing_field_cells(image, filtered, config, cell_size)
    if not cells:
        return ()
    accepted_cells = _accepted_drawing_field_cells(
        _collect_drawing_field_components(cells, filtered, config),
        config,
        width,
        height,
    )
    if not accepted_cells:
        return ()
    selected_indices = {
        index for key in accepted_cells for index in cells[key].candidate_indices
    }
    selected_indices = _close_drawing_field_candidates(
        filtered,
        selected_indices,
        config.endpoint_closure_distance,
    )
    return tuple(
        candidate
        for index, candidate in enumerate(filtered)
        if index in selected_indices
    )


def detect_line_candidates(
    image: GrayImage,
    config: DetectionConfig,
) -> tuple[LineCandidate, ...]:
    """Detect long line candidates using probabilistic Hough transform."""
    prepared = preprocess_image(image, config)
    raw_lines = cv2.HoughLinesP(
        prepared.edges,
        1.0,
        np.pi / 180.0,
        threshold=config.hough_threshold,
        minLineLength=config.minimum_line_length,
        maxLineGap=config.maximum_line_gap,
    )
    if raw_lines is None:
        return ()

    candidates: list[tuple[float, Point, Point, float]] = []
    for raw_line in raw_lines.reshape(-1, 4):
        start = Point(
            float(cast("float", raw_line[0])),
            float(cast("float", raw_line[1])),
        )
        end = Point(
            float(cast("float", raw_line[2])),
            float(cast("float", raw_line[3])),
        )
        length = start.distance_to(end)
        if length < config.minimum_line_length or _is_border_candidate(
            start,
            end,
            prepared.width,
            prepared.height,
            config.border_margin_ratio,
        ):
            continue
        candidates.append(
            (
                length,
                start,
                end,
                _candidate_confidence(length, prepared.width, prepared.height),
            ),
        )

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1].y,
            item[1].x,
            item[2].y,
            item[2].x,
        ),
    )
    ranked = tuple(
        LineCandidate(
            entity_id=EntityId(f"line_candidate_{index + 1:04d}"),
            start=start,
            end=end,
            confidence=confidence,
        )
        for index, (_, start, end, confidence) in enumerate(candidates)
    )
    grid_filtered = filter_periodic_grid_candidates(
        image,
        ranked,
        config.arbitration.grid,
    )
    arbitrated = arbitrate_line_candidates(grid_filtered, config.arbitration)
    field_selected = (
        _select_drawing_field_candidates(
            image,
            arbitrated,
            config.drawing_field,
        )
        if config.drawing_field.enabled
        else arbitrated
    )
    limited = field_selected[: config.maximum_candidates]
    return tuple(
        LineCandidate(
            entity_id=EntityId(f"line_raw_{index + 1:03d}"),
            start=candidate.start,
            end=candidate.end,
            confidence=candidate.confidence,
        )
        for index, candidate in enumerate(limited)
    )


def _shape_bounds(
    candidate: ShapeCandidate,
) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounds of one closed-shape candidate."""
    match candidate:
        case (
            CircleCandidate(center=center, radius=radius)
            | ArcCandidate(
                center=center,
                radius=radius,
            )
        ):
            return (
                center.x - radius,
                center.y - radius,
                center.x + radius,
                center.y + radius,
            )
        case PolylineCandidate(points=points):
            return (
                min(point.x for point in points),
                min(point.y for point in points),
                max(point.x for point in points),
                max(point.y for point in points),
            )


def _shape_center(candidate: ShapeCandidate) -> Point:
    """Return a stable center used to remove duplicate contour rings."""
    min_x, min_y, max_x, max_y = _shape_bounds(candidate)
    return Point((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)


def _shape_duplicate(
    first: ShapeCandidate,
    second: ShapeCandidate,
) -> bool:
    """Check whether nested Canny contours describe one physical shape."""
    circular = (CircleCandidate, ArcCandidate)
    if isinstance(first, circular) and isinstance(second, circular):
        radius_ratio = min(first.radius, second.radius) / max(
            first.radius,
            second.radius,
        )
        return (
            first.center.distance_to(second.center)
            <= max(
                _MIN_DUPLICATE_CIRCLE_DISTANCE,
                0.25 * max(first.radius, second.radius),
            )
            and radius_ratio >= _MIN_DUPLICATE_RADIUS_RATIO
        )
    if not isinstance(first, PolylineCandidate) or not isinstance(
        second,
        PolylineCandidate,
    ):
        return False
    first_bounds = _shape_bounds(first)
    second_bounds = _shape_bounds(second)
    first_width = first_bounds[2] - first_bounds[0]
    first_height = first_bounds[3] - first_bounds[1]
    second_width = second_bounds[2] - second_bounds[0]
    second_height = second_bounds[3] - second_bounds[1]
    width_ratio = min(first_width, second_width) / max(first_width, second_width)
    height_ratio = min(first_height, second_height) / max(
        first_height,
        second_height,
    )
    span = max(first_width, first_height, 1.0)
    centers_close = _shape_center(first).distance_to(
        _shape_center(second),
    ) <= max(12.0, span * 0.2)
    overlaps = (
        first_bounds[0] <= second_bounds[2]
        and second_bounds[0] <= first_bounds[2]
        and first_bounds[1] <= second_bounds[3]
        and second_bounds[1] <= first_bounds[3]
    )
    return (
        centers_close
        and overlaps
        and width_ratio >= _MIN_DUPLICATE_SHAPE_EXTENT_RATIO
        and height_ratio >= _MIN_DUPLICATE_SHAPE_EXTENT_RATIO
    )


def _closed_contour_candidate(
    contour: NDArray[np.int32],
    config: DetectionConfig,
    area: float,
    perimeter: float,
    box_size: tuple[int, int],
) -> tuple[float, ShapeCandidate] | None:
    """Parse a sufficiently filled contour into a rectangle or circle."""
    box_width, box_height = box_size
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, closed=True)
    circularity = 4.0 * pi * area / (perimeter * perimeter)
    aspect_ratio = min(box_width, box_height) / max(box_width, box_height)
    is_circle = (
        len(approx) >= _MIN_CIRCLE_APPROX_POINTS
        and circularity >= _MIN_CIRCLE_CIRCULARITY
        and aspect_ratio >= _MIN_CIRCLE_ASPECT_RATIO
    )
    if is_circle:
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius >= config.minimum_shape_radius:
            return (
                area,
                CircleCandidate(
                    entity_id=EntityId("shape_raw_pending"),
                    center=Point(float(center_x), float(center_y)),
                    radius=float(radius),
                    confidence=min(0.99, max(_MIN_DUPLICATE_RADIUS_RATIO, circularity)),
                ),
            )
        return None
    if len(approx) == _RECTANGLE_APPROX_POINTS and cv2.isContourConvex(approx):
        rectangularity = area / max(1.0, float(box_width * box_height))
        if rectangularity >= _MIN_RECTANGULARITY:
            approx_array = cast("NDArray[np.int32]", approx)
            points = tuple(
                Point(
                    cast("float", approx_array[index, 0, 0]),
                    cast("float", approx_array[index, 0, 1]),
                )
                for index in range(len(approx_array))
            )
            return (
                area,
                PolylineCandidate(
                    entity_id=EntityId("shape_raw_pending"),
                    points=points,
                    closed=True,
                    confidence=min(
                        0.99,
                        max(_MIN_DUPLICATE_RADIUS_RATIO, rectangularity),
                    ),
                ),
            )
    return _closed_polygon_candidate(contour, area, perimeter)


def _closed_polygon_candidate(
    contour: NDArray[np.int32],
    area: float,
    perimeter: float,
) -> tuple[float, ShapeCandidate] | None:
    """Parse a fabricated profile outline that is not a plain rectangle.

    Cut plates are routinely T, L, or notched outlines, so restricting closed
    geometry to four-point rectangles drops the part's main profile entirely.
    Solidity against the convex hull separates such an outline from an open
    scribble without assuming how many corners it has. The corner tolerance is
    far finer than the rectangle test's, which would otherwise collapse a
    notched profile into a triangle.
    """
    approx = cv2.approxPolyDP(
        contour,
        max(
            _MIN_POLYGON_APPROX_EPSILON,
            _POLYGON_APPROX_PERIMETER_RATIO * perimeter,
        ),
        closed=True,
    )
    if not _RECTANGLE_APPROX_POINTS <= len(approx) <= _MAX_POLYGON_APPROX_POINTS:
        return None
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    if hull_area <= 0.0:
        return None
    solidity = area / hull_area
    if solidity < _MIN_POLYGON_SOLIDITY:
        return None
    approx_array = cast("NDArray[np.int32]", approx)
    points = tuple(
        Point(
            cast("float", approx_array[index, 0, 0]),
            cast("float", approx_array[index, 0, 1]),
        )
        for index in range(len(approx_array))
    )
    return (
        area,
        PolylineCandidate(
            entity_id=EntityId("shape_raw_pending"),
            points=points,
            closed=True,
            confidence=min(0.95, max(_MIN_DUPLICATE_RADIUS_RATIO, solidity)),
        ),
    )


def _fit_circle_to_points(points: NDArray[np.float64]) -> tuple[Point, float] | None:
    """Fit a circle to a point cloud with the deterministic Kasa method."""
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((x, y, np.ones_like(x)))
    target = -((x * x) + (y * y))
    solution, _residuals, rank, _singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    if int(rank.item()) < _MIN_LSTSQ_RANK:
        return None
    linear_x = float(cast("float", solution[0]))
    linear_y = float(cast("float", solution[1]))
    constant = float(cast("float", solution[2]))
    center_x = -linear_x / 2.0
    center_y = -linear_y / 2.0
    radius_squared = ((linear_x * linear_x) + (linear_y * linear_y)) / 4.0 - constant
    if not isfinite(radius_squared) or radius_squared <= 0.0:
        return None
    return Point(center_x, center_y), sqrt(radius_squared)


def _arc_angular_span(
    points: NDArray[np.float64],
    center: Point,
) -> tuple[float, float]:
    """Return the start angle and covered span of a contour's fitted arc."""
    angles = np.mod(
        np.degrees(np.arctan2(points[:, 1] - center.y, points[:, 0] - center.x)),
        360.0,
    )
    ordered = np.sort(angles)
    gaps = np.diff(ordered)
    first_angle = float(cast("float", ordered[0]))
    last_angle = float(cast("float", ordered[-1]))
    wrap_gap = 360.0 - last_angle + first_angle
    all_gaps = np.append(gaps, wrap_gap)
    largest_gap_index = int(np.argmax(all_gaps).item())
    largest_gap = float(cast("float", all_gaps[largest_gap_index]))
    span = 360.0 - largest_gap
    start_angle = (
        first_angle
        if largest_gap_index == len(ordered) - 1
        else float(cast("float", ordered[largest_gap_index + 1]))
    )
    return start_angle, span


def _contour_arc_candidate(
    contour: NDArray[np.int32],
    config: DetectionConfig,
    box_size: tuple[int, int],
) -> tuple[float, ShapeCandidate] | None:
    """Fit an open circular arc to a contour that is not a filled shape."""
    box_width, box_height = box_size
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < _MIN_ARC_FIT_POINTS:
        return None
    fit = _fit_circle_to_points(points)
    if fit is None:
        return None
    center, radius = fit
    if (
        radius < config.minimum_shape_radius
        or radius > _MAX_ARC_TO_BOX_RADIUS_RATIO * max(box_width, box_height)
    ):
        return None
    distances = np.hypot(points[:, 0] - center.x, points[:, 1] - center.y)
    residual = sqrt(float(np.mean((distances - radius) ** 2)))
    if residual > max(
        _ARC_RESIDUAL_PIXEL_FLOOR, radius * _ARC_RESIDUAL_TO_RADIUS_RATIO
    ):
        return None
    start_angle, span = _arc_angular_span(points, center)
    if span < _MIN_ARC_SPAN_DEGREES or span > _MAX_ARC_SPAN_DEGREES:
        return None
    confidence = min(0.95, max(0.5, 1.0 - residual / max(1.0, radius)))
    area_proxy = radius * radius * span
    return (
        area_proxy,
        ArcCandidate(
            entity_id=EntityId("shape_raw_pending"),
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=start_angle + span,
            confidence=confidence,
        ),
    )


def _contour_shape_candidate(
    contour: NDArray[np.int32],
    config: DetectionConfig,
    width: int,
    height: int,
) -> tuple[float, ShapeCandidate] | None:
    """Parse one contour into a rectangle, circle, or open-arc candidate."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 0.0:
        return None
    x, y, box_width, box_height = cv2.boundingRect(contour)
    valid_bounds = (
        x > 1
        and y > 1
        and x + box_width < width - 1
        and y + box_height < height - 1
        and min(box_width, box_height) >= config.minimum_shape_span
        and box_width <= width * _MAX_SHAPE_EXTENT_RATIO
        and box_height <= height * _MAX_SHAPE_EXTENT_RATIO
    )
    if not valid_bounds:
        return None
    if area >= config.minimum_shape_area:
        closed_candidate = _closed_contour_candidate(
            contour,
            config,
            area,
            perimeter,
            (box_width, box_height),
        )
        if closed_candidate is not None:
            return closed_candidate
    return _contour_arc_candidate(contour, config, (box_width, box_height))


def _with_shape_id(candidate: ShapeCandidate, index: int) -> ShapeCandidate:
    """Assign a stable candidate id after contour ordering."""
    entity_id = EntityId(f"shape_raw_{index + 1:03d}")
    match candidate:
        case CircleCandidate(center=center, radius=radius, confidence=confidence):
            return CircleCandidate(
                entity_id,
                center,
                radius,
                confidence,
                candidate.role,
            )
        case ArcCandidate(
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
            confidence=confidence,
        ):
            return ArcCandidate(
                entity_id,
                center,
                radius,
                start_angle,
                end_angle,
                confidence,
                candidate.role,
            )
        case PolylineCandidate(
            points=points,
            closed=closed,
            confidence=confidence,
        ):
            return PolylineCandidate(
                entity_id,
                points,
                closed,
                confidence,
                candidate.role,
            )


def _shape_contains(parent: ShapeCandidate, child: ShapeCandidate) -> bool:
    """Return whether a smaller candidate is geometrically inside a parent."""
    parent_min_x, parent_min_y, parent_max_x, parent_max_y = _shape_bounds(parent)
    child_center = _shape_center(child)
    child_min_x, child_min_y, child_max_x, child_max_y = _shape_bounds(child)
    child_extent = max(child_max_x - child_min_x, child_max_y - child_min_y)
    parent_extent = max(parent_max_x - parent_min_x, parent_max_y - parent_min_y)
    if parent_extent < child_extent * 1.8:
        return False
    if isinstance(parent, (CircleCandidate, ArcCandidate)):
        return (
            parent.center.distance_to(child_center) + child_extent / 2.0
            <= parent.radius * 0.95
        )
    contour = np.asarray(
        [(point.x, point.y) for point in parent.points],
        dtype=np.float32,
    )
    return (
        cv2.pointPolygonTest(
            contour,
            (child_center.x, child_center.y),
            measureDist=False,
        )
        >= 0.0
    )


def _with_shape_role(
    candidate: ShapeCandidate,
    role: GeometryRole,
) -> ShapeCandidate:
    """Return one candidate with a semantic role while preserving provenance."""
    if isinstance(candidate, CircleCandidate):
        return CircleCandidate(
            candidate.entity_id,
            candidate.center,
            candidate.radius,
            candidate.confidence,
            role,
        )
    if isinstance(candidate, ArcCandidate):
        return ArcCandidate(
            candidate.entity_id,
            candidate.center,
            candidate.radius,
            candidate.start_angle_degrees,
            candidate.end_angle_degrees,
            candidate.confidence,
            role,
        )
    return PolylineCandidate(
        candidate.entity_id,
        candidate.points,
        candidate.closed,
        candidate.confidence,
        role,
    )


def classify_shape_holes(
    candidates: tuple[ShapeCandidate, ...],
) -> tuple[ShapeCandidate, ...]:
    """Mark only contained, smaller closed shapes as holes."""
    return tuple(
        _with_shape_role(
            candidate,
            GeometryRole.HOLE
            if any(
                parent != candidate and _shape_contains(parent, candidate)
                for parent in candidates
            )
            else GeometryRole.STRUCTURE,
        )
        for candidate in candidates
    )


def _without_nested_duplicates(
    candidates: tuple[ShapeCandidate, ...],
) -> tuple[ShapeCandidate, ...]:
    """Collapse the nested contour rings that one physical stroke produces.

    This runs after paper structure has been proved away. Reconciling every
    contour first would spend a quadratic comparison on ruling that is about
    to be discarded: a ruled sheet yields around a thousand cell contours, and
    none of them is a duplicate of another.
    """
    unique: list[ShapeCandidate] = []
    for candidate in candidates:
        if any(_shape_duplicate(existing, candidate) for existing in unique):
            continue
        unique.append(candidate)
    return tuple(unique)


def detect_shape_candidates_with_evidence(
    image: GrayImage,
    config: DetectionConfig,
) -> ShapeDetectionEvidence:
    """Detect closed shapes while reporting periodic-grid rejection evidence."""
    if not config.detect_closed_shapes:
        return ShapeDetectionEvidence(())
    prepared = preprocess_image(image, config)
    threshold = config.shape_threshold
    if threshold is None:
        threshold, _ = cv2.threshold(
            prepared.grayscale,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    shape_mask = cv2.threshold(
        prepared.grayscale,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )[1]
    closed_edges = cv2.morphologyEx(
        shape_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates = [
        parsed
        for contour in contours
        if (
            parsed := _contour_shape_candidate(
                cast("NDArray[np.int32]", contour),
                config,
                prepared.width,
                prepared.height,
            )
        )
    ]
    candidates.sort(
        key=lambda item: (
            -item[0],
            _shape_center(item[1]).y,
            _shape_center(item[1]).x,
        ),
    )
    grid_evidence = inspect_periodic_grid_shapes(
        tuple(candidate for _, candidate in candidates),
        prepared.width,
        prepared.height,
        config.arbitration.grid,
    )
    table_evidence = inspect_printed_table_shapes(
        _without_nested_duplicates(grid_evidence.candidates),
        prepared.width,
        prepared.height,
        config.printed_table,
    )
    classified = classify_shape_holes(table_evidence.candidates)
    limited = classified[: config.maximum_shape_candidates]
    return ShapeDetectionEvidence(
        candidates=tuple(
            _with_shape_id(candidate, index) for index, candidate in enumerate(limited)
        ),
        grid_detected=grid_evidence.grid_detected,
        grid_candidate_count=grid_evidence.removed_count,
        grid_like_count=grid_evidence.grid_like_count,
        table_detected=table_evidence.table_detected,
        table_candidate_count=table_evidence.removed_count,
        candidate_limit_reached=len(classified) > config.maximum_shape_candidates,
    )


def detect_shape_candidates(
    image: GrayImage,
    config: DetectionConfig,
) -> tuple[ShapeCandidate, ...]:
    """Detect closed rectangles and circular marks from a grayscale image."""
    return detect_shape_candidates_with_evidence(image, config).candidates
