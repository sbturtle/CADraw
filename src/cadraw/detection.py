"""OpenCV image preparation and deterministic line candidate detection."""

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import cast, final, override

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import EntityId, LineCandidate, Point

GrayImage = NDArray[np.uint8]


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
    limited = candidates[: config.maximum_candidates]
    return tuple(
        LineCandidate(
            entity_id=EntityId(f"line_raw_{index + 1:03d}"),
            start=start,
            end=end,
            confidence=confidence,
        )
        for index, (_, start, end, confidence) in enumerate(limited)
    )
