import cv2
import numpy as np

from cadraw.detection import (
    DetectionConfig,
    detect_line_candidates,
    detect_shape_candidates,
)
from cadraw.models import CircleCandidate, PolylineCandidate


def test_opencv_detects_long_structural_strokes() -> None:
    # Given a small grayscale sketch with one horizontal and one vertical stroke
    image = np.full((240, 320), 255, dtype=np.uint8)
    _ = cv2.line(image, (40, 50), (280, 50), 0, 5)
    _ = cv2.line(image, (40, 50), (40, 210), 0, 5)

    # When probabilistic Hough detection runs
    candidates = detect_line_candidates(
        image,
        DetectionConfig(
            minimum_line_length=80.0,
            hough_threshold=30,
            maximum_line_gap=12,
            maximum_candidates=12,
        ),
    )

    # Then long candidates are returned for downstream normalization
    assert len(candidates) >= 2
    assert all(
        candidate.start.distance_to(candidate.end) >= 80.0 for candidate in candidates
    )


def test_border_filter_ignores_page_edges_in_photographs() -> None:
    # Given a page-like frame and a drawing inside its margins
    image = np.full((320, 420), 255, dtype=np.uint8)
    _ = cv2.line(image, (8, 8), (8, 312), 0, 5)
    _ = cv2.line(image, (8, 8), (412, 8), 0, 5)
    _ = cv2.line(image, (120, 80), (340, 80), 0, 5)

    # When the detector applies its configurable border exclusion
    candidates = detect_line_candidates(
        image,
        DetectionConfig(
            minimum_line_length=80.0,
            hough_threshold=30,
            maximum_line_gap=12,
            maximum_candidates=12,
            border_margin_ratio=0.1,
        ),
    )

    # Then the inner stroke remains while page edges do not become CAD geometry
    assert candidates
    assert all(
        not (abs(candidate.start.x) < 42.0 and abs(candidate.end.x) < 42.0)
        for candidate in candidates
    )


def test_shape_detector_returns_circle_and_closed_rectangle_candidates() -> None:
    # Given a clean image containing the two supported closed drawing families
    image = np.full((360, 480), 255, dtype=np.uint8)
    _ = cv2.rectangle(image, (80, 80), (220, 190), 0, 5)
    _ = cv2.circle(image, (350, 220), 42, 0, 5)

    # When opt-in closed-shape detection runs
    candidates = detect_shape_candidates(
        image,
        DetectionConfig(
            detect_closed_shapes=True,
            minimum_shape_area=500.0,
            minimum_shape_radius=20.0,
            maximum_shape_candidates=12,
        ),
    )

    # Then each shape is represented by a typed image-space candidate
    assert any(isinstance(candidate, CircleCandidate) for candidate in candidates)
    assert any(
        isinstance(candidate, PolylineCandidate)
        and candidate.closed
        and len(candidate.points) == 4
        for candidate in candidates
    )
