"""OpenCV image preparation and deterministic line candidate detection."""

from dataclasses import dataclass, field
from math import hypot, pi
from pathlib import Path
from typing import cast, final, override

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.arbitration import (
    CandidateArbitrationConfig,
    arbitrate_line_candidates,
)
from cadraw.grid_filter import filter_periodic_grid_candidates
from cadraw.models import (
    CircleCandidate,
    EntityId,
    GeometryRole,
    LineCandidate,
    Point,
    PolylineCandidate,
    ShapeCandidate,
)

GrayImage = NDArray[np.uint8]

_MIN_DUPLICATE_CIRCLE_DISTANCE = 8.0
_MIN_DUPLICATE_RADIUS_RATIO = 0.55
_MIN_CIRCLE_APPROX_POINTS = 6
_MIN_CIRCLE_CIRCULARITY = 0.68
_MIN_CIRCLE_ASPECT_RATIO = 0.65
_RECTANGLE_APPROX_POINTS = 4
_MIN_RECTANGULARITY = 0.35
_MIN_DUPLICATE_SHAPE_EXTENT_RATIO = 0.55


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
    detect_closed_shapes: bool = False
    minimum_shape_area: float = 800.0
    minimum_shape_span: float = 30.0
    minimum_shape_radius: float = 12.0
    maximum_shape_candidates: int = 64
    shape_threshold: int | None = None
    arbitration: CandidateArbitrationConfig = field(
        default_factory=CandidateArbitrationConfig,
    )


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Grayscale and edge views used by the detector."""

    grayscale: GrayImage
    edges: GrayImage
    width: int
    height: int


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
    limited = arbitrated[: config.maximum_candidates]
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
        case CircleCandidate(center=center, radius=radius):
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
    if isinstance(first, CircleCandidate) and isinstance(second, CircleCandidate):
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


def _contour_shape_candidate(
    contour: NDArray[np.int32],
    config: DetectionConfig,
    width: int,
    height: int,
) -> tuple[float, ShapeCandidate] | None:
    """Parse one contour into a rectangle or circle candidate."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    candidate: tuple[float, ShapeCandidate] | None = None
    if area >= config.minimum_shape_area and perimeter > 0.0:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        valid_bounds = (
            x > 1
            and y > 1
            and x + box_width < width - 1
            and y + box_height < height - 1
            and min(box_width, box_height) >= config.minimum_shape_span
            and box_width <= width * 0.7
            and box_height <= height * 0.7
        )
        if valid_bounds:
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
                    candidate = (
                        area,
                        CircleCandidate(
                            entity_id=EntityId("shape_raw_pending"),
                            center=Point(float(center_x), float(center_y)),
                            radius=float(radius),
                            confidence=min(
                                0.99,
                                max(_MIN_DUPLICATE_RADIUS_RATIO, circularity),
                            ),
                        ),
                    )
            elif len(approx) == _RECTANGLE_APPROX_POINTS and cv2.isContourConvex(
                approx
            ):
                rectangularity = area / max(
                    1.0,
                    float(box_width * box_height),
                )
                if rectangularity >= _MIN_RECTANGULARITY:
                    approx_array = cast("NDArray[np.int32]", approx)
                    points = tuple(
                        Point(
                            cast("float", approx_array[index, 0, 0]),
                            cast("float", approx_array[index, 0, 1]),
                        )
                        for index in range(len(approx_array))
                    )
                    candidate = (
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
    return candidate


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
    if isinstance(parent, CircleCandidate):
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


def detect_shape_candidates(
    image: GrayImage,
    config: DetectionConfig,
) -> tuple[ShapeCandidate, ...]:
    """Detect closed rectangles and circular marks from a grayscale image."""
    if not config.detect_closed_shapes:
        return ()
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
    unique: list[ShapeCandidate] = []
    for _, candidate in candidates:
        if any(_shape_duplicate(existing, candidate) for existing in unique):
            continue
        unique.append(candidate)
        if len(unique) >= config.maximum_shape_candidates:
            break
    classified = classify_shape_holes(tuple(unique))
    return tuple(
        _with_shape_id(candidate, index) for index, candidate in enumerate(classified)
    )
