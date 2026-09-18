import cv2
import numpy as np

from cadraw.detection import (
    DetectionConfig,
    DrawingFieldConfig,
    detect_line_candidates,
    detect_shape_candidates,
)
from cadraw.models import ArcCandidate, CircleCandidate, PolylineCandidate


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


def test_drawing_field_rejects_frame_edges_and_keeps_multiaxis_sketch() -> None:
    # Given a page with long frame rails and a compact hand-drawn profile
    image = np.full((480, 640), 255, dtype=np.uint8)
    _ = cv2.line(image, (0, 55), (639, 55), 0, 4)
    _ = cv2.line(image, (0, 425), (639, 425), 0, 4)
    _ = cv2.line(image, (190, 150), (430, 150), 0, 5)
    _ = cv2.line(image, (190, 150), (190, 350), 0, 5)
    _ = cv2.line(image, (190, 350), (430, 350), 0, 5)
    _ = cv2.line(image, (430, 150), (430, 350), 0, 5)
    _ = cv2.line(image, (260, 230), (360, 230), 0, 5)

    # When drawing-field selection runs before the candidate budget
    candidates = detect_line_candidates(
        image,
        DetectionConfig(
            minimum_line_length=50.0,
            hough_threshold=20,
            maximum_line_gap=12,
            maximum_candidates=64,
            border_margin_ratio=0.0,
            drawing_field=DrawingFieldConfig(enabled=True),
        ),
    )

    # Then the compact multi-axis profile remains and frame rails do not
    assert candidates
    assert all(
        100.0 < (candidate.start.y + candidate.end.y) / 2.0 < 380.0
        for candidate in candidates
    )
    assert all(
        120.0 < (candidate.start.x + candidate.end.x) / 2.0 < 500.0
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


def test_shape_detector_fits_a_partial_arc_stroke() -> None:
    # Given a hand-drawn partial arc stroke, such as a fillet or a cut hole
    image = np.full((360, 480), 255, dtype=np.uint8)
    _ = cv2.ellipse(image, (200, 180), (60, 60), 0, 30, 300, 0, 5)

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

    # Then the open stroke is fit as an arc rather than dropped or forced closed
    arcs = tuple(
        candidate for candidate in candidates if isinstance(candidate, ArcCandidate)
    )
    assert len(arcs) == 1
    arc = arcs[0]
    assert abs(arc.center.x - 200.0) < 3.0
    assert abs(arc.center.y - 180.0) < 3.0
    assert abs(arc.radius - 60.0) < 3.0
    assert 50.0 < (arc.end_angle_degrees - arc.start_angle_degrees) < 320.0
    assert not any(isinstance(candidate, CircleCandidate) for candidate in candidates)


def test_shape_detector_does_not_fit_a_full_circle_as_an_arc() -> None:
    # Given a complete circle stroke
    image = np.full((360, 480), 255, dtype=np.uint8)
    _ = cv2.circle(image, (200, 180), 60, 0, 5)

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

    # Then the closed loop is a circle, never a partial-arc candidate
    assert any(isinstance(candidate, CircleCandidate) for candidate in candidates)
    assert not any(isinstance(candidate, ArcCandidate) for candidate in candidates)


def test_notched_profile_outline_is_kept_as_a_closed_polygon() -> None:
    # Given a T-shaped plate outline, which has more corners than a rectangle
    image = np.full((360, 360), 255, dtype=np.uint8)
    outline = np.array(
        [
            [140, 40],
            [200, 40],
            [200, 150],
            [280, 150],
            [280, 210],
            [60, 210],
            [60, 150],
            [140, 150],
        ],
        dtype=np.int32,
    )
    _ = cv2.polylines(image, [outline], isClosed=True, color=0, thickness=4)

    # When closed-shape detection runs
    candidates = detect_shape_candidates(
        image,
        DetectionConfig(
            detect_closed_shapes=True,
            minimum_shape_area=800.0,
            minimum_shape_span=30.0,
            border_margin_ratio=0.0,
        ),
    )

    # Then the profile survives as a closed polygon that keeps its notches
    polygons = [
        candidate
        for candidate in candidates
        if isinstance(candidate, PolylineCandidate) and candidate.closed
    ]
    assert polygons
    profile = max(
        polygons,
        key=lambda candidate: (
            max(point.x for point in candidate.points)
            - min(point.x for point in candidate.points)
        ),
    )
    assert len(profile.points) > 4
    assert (
        max(point.x for point in profile.points)
        - min(point.x for point in profile.points)
        >= 200
    )


def _sheet_with_small_mark(width: int, height: int) -> np.ndarray:
    """Draw one 30x42 closed mark, the size of a handwritten character."""
    image = np.full((height, width), 255, dtype=np.uint8)
    _ = cv2.rectangle(image, (200, 200), (230, 242), 0, 3)
    return image


def test_a_small_mark_is_a_shape_on_a_small_sheet() -> None:
    # Given a mark that is large relative to a small fixture
    image = _sheet_with_small_mark(480, 640)

    # When closed-shape detection runs with the shipped floors
    candidates = detect_shape_candidates(
        image,
        DetectionConfig(detect_closed_shapes=True, border_margin_ratio=0.0),
    )

    # Then it is kept, because for this sheet it is drawing-sized
    assert candidates


def test_the_same_mark_is_handwriting_on_a_photograph() -> None:
    # Given the identical mark on a full-resolution photograph
    image = _sheet_with_small_mark(1920, 2560)

    # When closed-shape detection runs with the shipped floors
    candidates = detect_shape_candidates(
        image,
        DetectionConfig(detect_closed_shapes=True, border_margin_ratio=0.0),
    )

    # Then the extent floor scales with the sheet and drops it
    assert candidates == ()
