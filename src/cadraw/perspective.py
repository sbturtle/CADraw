"""Deterministic page-boundary detection and perspective rectification."""

from dataclasses import dataclass
from math import hypot
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import PerspectiveTransform, Point

GrayImage = NDArray[np.uint8]

_MIN_PAGE_AREA_RATIO = 0.20
_MAX_PAGE_AREA_RATIO = 0.98
_MIN_FALLBACK_PAGE_AREA_RATIO = 0.45
_MIN_QUADRILATERAL_POINTS = 4
_FRAME_MARGIN = 2.0


@dataclass(frozen=True, slots=True)
class PerspectiveWarp:
    """Rectified image and metadata for mapping geometry back to the photo."""

    image: GrayImage
    source_quad: tuple[Point, Point, Point, Point]
    inverse_matrix: tuple[float, ...]
    target_size: tuple[int, int]

    @property
    def transform(self) -> PerspectiveTransform:
        """Return serializable transform metadata for analysis provenance."""
        return PerspectiveTransform(
            source_quad=self.source_quad,
            target_size=self.target_size,
            inverse_matrix=self.inverse_matrix,
        )


def _ordered_quad(points: NDArray[np.float32]) -> tuple[Point, Point, Point, Point]:
    """Order four points clockwise from the upper-left corner."""
    flattened = cast("NDArray[np.float32]", points.reshape(4, 2))
    values = [
        (
            float(cast("float", flattened[index, 0])),
            float(cast("float", flattened[index, 1])),
        )
        for index in range(4)
    ]
    top_left = min(values, key=lambda value: value[0] + value[1])
    bottom_right = max(values, key=lambda value: value[0] + value[1])
    top_right = max(values, key=lambda value: value[0] - value[1])
    bottom_left = min(values, key=lambda value: value[0] - value[1])
    return (
        Point(*top_left),
        Point(*top_right),
        Point(*bottom_right),
        Point(*bottom_left),
    )


def detect_page_quadrilateral(
    image: GrayImage,
) -> tuple[Point, Point, Point, Point] | None:
    """Find the largest plausible page quadrilateral in a photograph."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    edges = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(width * height)
    candidates: list[tuple[float, tuple[Point, Point, Point, Point]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if not _MIN_PAGE_AREA_RATIO <= area_ratio <= _MAX_PAGE_AREA_RATIO:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0.0:
            continue
        approximation = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            closed=True,
        )
        if len(approximation) != _MIN_QUADRILATERAL_POINTS or not cv2.isContourConvex(
            approximation
        ):
            continue
        ordered = _ordered_quad(cast("NDArray[np.float32]", approximation))
        margins = (
            min(point.x for point in ordered),
            min(point.y for point in ordered),
            width - max(point.x for point in ordered),
            height - max(point.y for point in ordered),
        )
        if max(margins) <= _FRAME_MARGIN:
            continue
        candidates.append((area, ordered))
    if not candidates:
        return _detect_bright_page_fallback(image)
    return max(candidates, key=lambda candidate: candidate[0])[1]


def detect_page_quadrilateral_in_region(
    image: GrayImage,
    region: tuple[int, int, int, int],
) -> tuple[Point, Point, Point, Point] | None:
    """Find a page boundary inside a configured source-image region."""
    x, y, width, height = region
    cropped = image[y : y + height, x : x + width]
    detected = detect_page_quadrilateral(cropped)
    if detected is None:
        return None
    offset = Point(float(x), float(y))
    return (
        detected[0].translated(offset.x, offset.y),
        detected[1].translated(offset.x, offset.y),
        detected[2].translated(offset.x, offset.y),
        detected[3].translated(offset.x, offset.y),
    )


def detect_page_quadrilateral_with_region_fallback(
    image: GrayImage,
    region: tuple[int, int, int, int] | None,
) -> tuple[Point, Point, Point, Point] | None:
    """Try the full image before a configured region."""
    detected = detect_page_quadrilateral(image)
    if detected is not None or region is None:
        return detected
    return detect_page_quadrilateral_in_region(image, region)


def _detect_bright_page_fallback(
    image: GrayImage,
) -> tuple[Point, Point, Point, Point] | None:
    """Use the dominant bright paper region when its border is clipped."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    threshold, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )[1]
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((25, 25), dtype=np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(width * height)
    contours = tuple(
        contour
        for contour in contours
        if cv2.contourArea(contour) / image_area >= _MIN_FALLBACK_PAGE_AREA_RATIO
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    rectangle = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rectangle)
    ordered = _ordered_quad(cast("NDArray[np.float32]", box))
    margins = (
        min(point.x for point in ordered),
        min(point.y for point in ordered),
        width - max(point.x for point in ordered),
        height - max(point.y for point in ordered),
    )
    if max(margins) <= _FRAME_MARGIN:
        return None
    return ordered


def warp_page(
    image: GrayImage,
    source_quad: tuple[Point, Point, Point, Point],
) -> PerspectiveWarp:
    """Warp a page quadrilateral into a rectangular grayscale image."""
    source = np.asarray(
        [(point.x, point.y) for point in source_quad],
        dtype=np.float32,
    )
    source_array = cast("NDArray[np.float32]", source)
    top_width = hypot(
        cast("float", source_array[1, 0] - source_array[0, 0]),
        cast("float", source_array[1, 1] - source_array[0, 1]),
    )
    bottom_width = hypot(
        cast("float", source_array[2, 0] - source_array[3, 0]),
        cast("float", source_array[2, 1] - source_array[3, 1]),
    )
    left_height = hypot(
        cast("float", source_array[3, 0] - source_array[0, 0]),
        cast("float", source_array[3, 1] - source_array[0, 1]),
    )
    right_height = hypot(
        cast("float", source_array[2, 0] - source_array[1, 0]),
        cast("float", source_array[2, 1] - source_array[1, 1]),
    )
    target_width = max(1, round(max(top_width, bottom_width)))
    target_height = max(1, round(max(left_height, right_height)))
    destination = np.asarray(
        [
            (0.0, 0.0),
            (target_width - 1.0, 0.0),
            (target_width - 1.0, target_height - 1.0),
            (0.0, target_height - 1.0),
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse = cv2.getPerspectiveTransform(destination, source)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
    return PerspectiveWarp(
        image=np.asarray(warped, dtype=np.uint8),
        source_quad=source_quad,
        inverse_matrix=tuple(float(value) for value in inverse.flat),
        target_size=(target_width, target_height),
    )


def transform_point(
    point: Point,
    inverse_matrix: tuple[float, ...],
) -> Point:
    """Map one rectified-image point back into source-image coordinates."""
    matrix = np.asarray(inverse_matrix, dtype=np.float32).reshape(3, 3)
    source = np.asarray([[[point.x, point.y]]], dtype=np.float32)
    transformed = cast(
        "NDArray[np.float32]",
        cv2.perspectiveTransform(source, matrix),
    )
    return Point(
        cast("float", transformed[0, 0, 0]),
        cast("float", transformed[0, 0, 1]),
    )
