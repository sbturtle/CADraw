from typing import cast

import cv2
import numpy as np

from cadraw.models import Point
from cadraw.perspective import (
    detect_page_quadrilateral,
    detect_page_quadrilateral_in_region,
    transform_point,
    warp_page,
)


def test_warp_page_rectifies_a_known_quadrilateral() -> None:
    # Given a page photographed as a skewed quadrilateral
    image = np.full((500, 700), 40, dtype=np.uint8)
    quad = (
        Point(90.0, 45.0),
        Point(610.0, 75.0),
        Point(650.0, 450.0),
        Point(55.0, 420.0),
    )
    polygon = np.array([(point.x, point.y) for point in quad], dtype=np.int32)
    _ = cv2.fillConvexPoly(image, polygon, 255)

    # When the known page corners are rectified
    warp = warp_page(image, quad)

    # Then the page becomes a rectangular analysis image with an inverse map
    assert warp.image.shape[1] > warp.image.shape[0]
    assert warp.image.shape[0] >= 300
    assert len(warp.inverse_matrix) == 9
    assert warp.source_quad == quad
    restored_center = transform_point(
        Point(
            cast("float", warp.image.shape[1]) / 2.0,
            cast("float", warp.image.shape[0]) / 2.0,
        ),
        warp.inverse_matrix,
    )
    assert 250.0 < restored_center.x < 400.0
    assert 180.0 < restored_center.y < 320.0


def test_page_detector_finds_the_dominant_page_boundary() -> None:
    # Given a dark background and one bright bordered page
    image = np.full((520, 720), 30, dtype=np.uint8)
    quad = np.array(
        [(80, 40), (640, 70), (665, 470), (50, 440)],
        dtype=np.int32,
    )
    _ = cv2.fillConvexPoly(image, quad, 240)
    _ = cv2.polylines(image, [quad], isClosed=True, color=0, thickness=6)

    # When page-boundary detection runs
    detected = detect_page_quadrilateral(image)

    # Then a four-corner page candidate is returned near the source page
    assert detected is not None
    assert len(detected) == 4
    assert min(point.x for point in detected) < 100.0
    assert max(point.x for point in detected) > 600.0


def test_page_detector_maps_a_boundary_found_inside_requested_region() -> None:
    # Given a page that is too small to qualify in the full photograph
    image = np.full((400, 600), 30, dtype=np.uint8)
    _ = cv2.rectangle(image, (150, 100), (420, 270), 240, thickness=-1)
    _ = cv2.rectangle(image, (150, 100), (420, 270), 0, thickness=5)
    region = (60, 40, 480, 320)

    # When page detection is retried inside the configured analysis region
    detected = detect_page_quadrilateral_in_region(image, region)

    # Then the resulting corners are restored to source-image coordinates
    assert detected is not None
    assert min(point.x for point in detected) > 140.0
    assert min(point.y for point in detected) > 90.0
    assert max(point.x for point in detected) < 430.0
    assert max(point.y for point in detected) < 280.0
