import cv2
import numpy as np

from cadraw.arbitration import (
    CandidateArbitrationConfig,
    arbitrate_line_candidates,
)
from cadraw.detection import (
    DetectionConfig,
    detect_line_candidates,
    detect_shape_candidates,
)
from cadraw.models import (
    EntityId,
    GeometryRole,
    LineCandidate,
    Point,
    PolylineCandidate,
)


def test_parallel_edge_candidates_are_reduced_to_one_representative() -> None:
    # Given two Hough rails produced by one thick hand-drawn stroke
    candidates = (
        LineCandidate(EntityId("edge_a"), Point(0.0, 0.0), Point(140.0, 0.0), 0.8),
        LineCandidate(EntityId("edge_b"), Point(0.0, 6.0), Point(140.0, 6.0), 0.8),
    )

    # When candidate arbitration runs before the candidate budget is applied
    retained = arbitrate_line_candidates(
        candidates,
        CandidateArbitrationConfig(
            duplicate_perpendicular_distance=8.0,
            duplicate_overlap_ratio=0.85,
        ),
    )

    # Then one physical stroke consumes one candidate slot
    assert len(retained) == 1
    assert retained[0].entity_id == EntityId("edge_a")


def test_nearby_parallel_strokes_are_not_collapsed() -> None:
    # Given two distinct parallel strokes that have only a small gap
    candidates = (
        LineCandidate(EntityId("first"), Point(0.0, 0.0), Point(140.0, 0.0), 0.8),
        LineCandidate(EntityId("second"), Point(0.0, 12.0), Point(140.0, 12.0), 0.8),
    )

    # When arbitration uses the strict physical-duplicate distance
    retained = arbitrate_line_candidates(
        candidates,
        CandidateArbitrationConfig(duplicate_perpendicular_distance=8.0),
    )

    # Then two intentionally separate strokes remain available for review
    assert tuple(candidate.entity_id for candidate in retained) == (
        EntityId("first"),
        EntityId("second"),
    )


def test_periodic_gray_grid_is_filtered_but_dark_structure_survives() -> None:
    # Given a gray graph-paper lattice and one dark L-shaped drawing
    image = np.full((720, 720), 255, dtype=np.uint8)
    for coordinate in range(40, 681, 60):
        _ = cv2.line(image, (coordinate, 20), (coordinate, 700), 170, 2)
        _ = cv2.line(image, (20, coordinate), (700, coordinate), 170, 2)
    _ = cv2.line(image, (130, 130), (380, 130), 0, 7)
    _ = cv2.line(image, (130, 130), (130, 430), 0, 7)

    # When the detector proves the periodic grid before applying its budget
    candidates = detect_line_candidates(
        image,
        DetectionConfig(
            minimum_line_length=120.0,
            hough_threshold=30,
            maximum_line_gap=8,
            maximum_candidates=64,
            border_margin_ratio=0.0,
        ),
    )

    # Then the two dark structural strokes remain and grid rails do not
    assert len(candidates) <= 4
    assert all(
        candidate.start.distance_to(candidate.end) < 400.0 for candidate in candidates
    )
    assert any(
        min(candidate.start.y, candidate.end.y) < 145.0
        and max(candidate.start.y, candidate.end.y) < 145.0
        and candidate.start.distance_to(candidate.end) > 200.0
        for candidate in candidates
    )
    assert any(
        min(candidate.start.x, candidate.end.x) < 145.0
        and max(candidate.start.x, candidate.end.x) < 145.0
        and candidate.start.distance_to(candidate.end) > 250.0
        for candidate in candidates
    )


def test_nested_rectangles_are_retained_and_inner_shape_is_a_hole() -> None:
    # Given an outer profile and a smaller closed rectangular hole
    image = np.full((360, 360), 255, dtype=np.uint8)
    _ = cv2.rectangle(image, (60, 60), (280, 280), 0, 5)
    _ = cv2.rectangle(image, (130, 130), (210, 210), 0, 5)

    # When closed-shape detection runs
    candidates = detect_shape_candidates(
        image,
        DetectionConfig(
            detect_closed_shapes=True,
            minimum_shape_area=500.0,
            minimum_shape_radius=20.0,
            maximum_shape_candidates=12,
        ),
    )

    # Then the nested geometry is not mistaken for one duplicate contour
    rectangles = tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, PolylineCandidate)
    )
    assert len(rectangles) == 2
    assert sum(candidate.role is GeometryRole.HOLE for candidate in rectangles) == 1
