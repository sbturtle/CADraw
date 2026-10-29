from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.detection import DetectionConfig, detect_face_shapes_with_evidence
from cadraw.faces import (
    Face,
    FaceConfig,
    drawing_ink,
    enclosed_faces,
    face_shape,
    merge_straight_corners,
    ruled_paper,
    stroke_ends,
    thinned,
    with_stroke_bridges,
)
from cadraw.models import CircleCandidate, GeometryRole, Point, PolylineCandidate
from cadraw.preprocessing import illumination_flattened

type GrayImage = NDArray[np.uint8]

_PAPER = 236.0
_INK_VALUE = 30
_FAINT_VALUE = 200
_RULING_VALUE = 200
_RULING_PITCH = 30


def _page(height: int = 400, width: int = 600) -> GrayImage:
    """Return evenly lit paper with a little sensor noise."""
    rng = np.random.default_rng(7)
    noisy = np.full((height, width), _PAPER) + rng.normal(0.0, 1.0, (height, width))
    return np.asarray(np.clip(noisy, 0, 255), dtype=np.uint8)


def _ruled(image: GrayImage) -> GrayImage:
    """Rule a faint square grid across a page."""
    height, width = cast("tuple[int, int]", image.shape[:2])
    for offset in range(20, width, _RULING_PITCH):
        _ = cv2.line(image, (offset, 0), (offset, height - 1), _RULING_VALUE, 1)
    for offset in range(20, height, _RULING_PITCH):
        _ = cv2.line(image, (0, offset), (width - 1, offset), _RULING_VALUE, 1)
    return image


def _ink_count(mask: GrayImage, rows: slice, columns: slice) -> int:
    return int(np.count_nonzero(mask[rows, columns]))


def _plate(corner_gap: int) -> GrayImage:
    """Draw a plate with one drilled hole, one slot, and a pen lift at a corner."""
    image = _page()
    _ = cv2.line(image, (80, 80), (500, 80), _INK_VALUE, 3)
    _ = cv2.line(image, (500, 80), (500, 320), _INK_VALUE, 3)
    _ = cv2.line(image, (500, 320), (80, 320), _INK_VALUE, 3)
    _ = cv2.line(image, (80, 320), (80, 80 + corner_gap), _INK_VALUE, 3)
    _ = cv2.circle(image, (180, 200), 18, _INK_VALUE, 2)
    _ = cv2.rectangle(image, (300, 180), (400, 220), _INK_VALUE, 2)
    return image


def _face(contour: list[tuple[int, int]]) -> Face:
    points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    return Face(contour=points, area=float(cv2.contourArea(points)))


def _circle_contour(radius: int) -> list[tuple[int, int]]:
    mask = np.zeros((4 * radius, 4 * radius), dtype=np.uint8)
    _ = cv2.circle(mask, (2 * radius, 2 * radius), radius, 1, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = cast("NDArray[np.int32]", contours[0]).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in outline.tolist()]


def test_faint_ink_survives_only_where_it_continues_a_dark_stroke() -> None:
    # Given a dark outline whose stroke runs on faintly, and a faint mark alone
    image = _page()
    _ = cv2.rectangle(image, (60, 60), (300, 300), _INK_VALUE, 3)
    _ = cv2.line(image, (300, 180), (520, 180), _FAINT_VALUE, 2)
    _ = cv2.line(image, (340, 330), (540, 330), _FAINT_VALUE, 2)

    # When strokes are binarized, conservatively and for faint outlines
    conservative = drawing_ink(image, FaceConfig())
    sensitive = drawing_ink(image, FaceConfig(), sensitive=True)

    # Then only the sensitive cut keeps the faint run, and never the lone mark
    assert _ink_count(conservative, slice(176, 185), slice(400, 500)) == 0
    assert _ink_count(sensitive, slice(176, 185), slice(400, 500)) >= 100
    assert _ink_count(sensitive, slice(326, 335), slice(400, 500)) == 0
    assert _ink_count(sensitive, slice(58, 63), slice(100, 260)) >= 160


def test_ruled_paper_keeps_the_conservative_cut() -> None:
    # Given the same faint continuation drawn over graph paper
    image = _ruled(_page())
    _ = cv2.rectangle(image, (60, 60), (300, 300), _INK_VALUE, 3)
    _ = cv2.line(image, (300, 185), (520, 185), 180, 2)

    # When strokes are binarized for faint outlines
    ink = drawing_ink(image, FaceConfig(), sensitive=True)

    # Then the rulings, as dark as the faint stroke, do not become ink
    assert ruled_paper(illumination_flattened(image)) is True
    assert _ink_count(ink, slice(340, 360), slice(0, 600)) == 0
    assert _ink_count(ink, slice(183, 188), slice(400, 500)) == 0


def test_a_plain_sheet_is_not_ruled() -> None:
    # Given clean paper carrying a single drawn part
    image = _page()
    _ = cv2.rectangle(image, (60, 60), (300, 300), _INK_VALUE, 3)

    # Then no periodic ruling is reported
    assert ruled_paper(illumination_flattened(image)) is False


def test_thinning_reduces_a_thick_stroke_to_its_centre_line() -> None:
    # Given a stroke seven pixels wide
    mask = np.zeros((60, 200), dtype=np.uint8)
    _ = cv2.line(mask, (20, 30), (180, 30), 255, 7)

    # When it is thinned
    skeleton = thinned(mask)

    # Then every column along the stroke keeps a single centre pixel
    columns = cast(
        "list[int]",
        np.asarray(np.greater(skeleton[:, 40:160], 0), dtype=np.int64)
        .sum(axis=0)
        .tolist(),
    )
    assert set(columns) == {1}
    assert _ink_count(skeleton, slice(29, 32), slice(40, 160)) == 120


def test_stroke_ends_point_the_way_the_pen_was_moving() -> None:
    # Given an L-shaped stroke and a tick too short to be an outline side
    mask = np.zeros((200, 200), dtype=np.uint8)
    _ = cv2.line(mask, (40, 40), (40, 160), 255, 3)
    _ = cv2.line(mask, (40, 160), (160, 160), 255, 3)
    _ = cv2.line(mask, (120, 40), (124, 40), 255, 3)

    # When the free ends are traced
    ends = stroke_ends(thinned(mask), minimum_branch=12, direction_span=10)

    # Then the two arm ends are found, each pointing away from the corner
    assert len(ends) == 2
    by_position = sorted(ends, key=lambda end: end.point.x)
    top, right = by_position
    assert top.point.distance_to(Point(40.0, 40.0)) <= 3.0
    assert top.direction[1] < -0.9
    assert right.point.distance_to(Point(160.0, 160.0)) <= 3.0
    assert right.direction[0] > 0.9


def test_a_corner_drawn_short_is_closed_for_the_face_only() -> None:
    # Given a plate outline whose last side stops short of the first corner
    mask = drawing_ink(_plate(corner_gap=12), FaceConfig(), sensitive=True)

    # When faces are taken before and after bridging the stroke ends
    before = list(enclosed_faces(mask, 3, 15.0, 1))
    ends = stroke_ends(thinned(mask, 8), minimum_branch=8, direction_span=10)
    bridged = with_stroke_bridges(mask, ends, 20.0, 2)
    after = list(enclosed_faces(bridged, 3, 15.0, 1))

    # Then the plate becomes a face beside its hole and slot
    assert len(before) == 2
    assert len(after) == 3
    plate = max(after, key=lambda face: face.area)
    assert plate.area > 0.9 * 420 * 240


def test_a_leader_line_stopping_short_of_a_hole_is_not_joined_to_it() -> None:
    # Given a long leader line pointing at a small drilled hole
    mask = np.zeros((300, 400), dtype=np.uint8)
    _ = cv2.line(mask, (40, 150), (240, 150), 255, 3)
    _ = cv2.circle(mask, (262, 150), 12, 255, 2)

    # When stroke ends are bridged
    ends = stroke_ends(thinned(mask), minimum_branch=12, direction_span=10)
    bridged = with_stroke_bridges(mask, ends, 30.0, 2)

    # Then the gap to the small hole stays open
    assert np.array_equal(bridged, mask)


def test_enclosed_faces_fill_islands_and_skip_the_open_page() -> None:
    # Given a closed square with a ring drawn inside it
    mask = np.zeros((300, 300), dtype=np.uint8)
    _ = cv2.rectangle(mask, (50, 50), (250, 250), 255, 3)
    _ = cv2.circle(mask, (150, 150), 30, 255, 3)

    # When faces are read
    faces = sorted(enclosed_faces(mask, 3, 15.0, 1), key=lambda face: face.area)

    # Then the ring's inside and the square's inside are faces, not the page
    assert len(faces) == 2
    hole, square = faces
    ring_area = np.pi * 29.0 * 29.0
    assert abs(hole.area - ring_area) < 0.15 * ring_area
    assert abs(square.area - 200.0 * 200.0) < 0.05 * 200.0 * 200.0


def test_a_round_face_is_a_circle_and_a_square_face_is_not() -> None:
    # Given a round face and a square, which is round enough by area alone
    round_face = _face(_circle_contour(40))
    square = _face([(0, 0), (100, 0), (100, 100), (0, 100)])

    # When the faces are read as shapes
    circle = face_shape(round_face, FaceConfig(), 1000)
    polygon = face_shape(square, FaceConfig(), 1000)

    # Then the square keeps its four corners
    assert circle is not None
    assert isinstance(circle[1], CircleCandidate)
    assert abs(circle[1].radius - 40.0) < 2.0
    assert polygon is not None
    assert isinstance(polygon[1], PolylineCandidate)
    assert len(polygon[1].points) == 4


def test_a_sliver_between_doubled_strokes_is_no_shape() -> None:
    # Given the thin strip of paper left between two strokes drawn twice
    sliver = _face([(0, 0), (200, 0), (200, 6), (0, 6)])

    # Then it is not offered as a feature
    assert face_shape(sliver, FaceConfig(), 1000) is None


def test_straight_runs_lose_their_wobble_vertices() -> None:
    # Given a rectangle with a vertex part way along a wobbling side
    points = (
        Point(0.0, 0.0),
        Point(100.0, 2.0),
        Point(200.0, 0.0),
        Point(200.0, 100.0),
        Point(0.0, 100.0),
    )

    # When straight corners are merged
    merged = merge_straight_corners(points, 12.0)

    # Then only the four real corners remain
    assert merged == (
        Point(0.0, 0.0),
        Point(200.0, 0.0),
        Point(200.0, 100.0),
        Point(0.0, 100.0),
    )


def test_face_detection_reads_a_plate_with_its_hole_and_slot() -> None:
    # Given a plate whose outline has a pen lift a few pixels wide
    image = _plate(corner_gap=6)

    # When closed shapes are read from the enclosed faces
    evidence = detect_face_shapes_with_evidence(
        image,
        DetectionConfig(detect_closed_shapes=True, faces=FaceConfig(enabled=True)),
    )

    # Then the plate is structure and the hole and slot inside it are holes
    shapes = sorted(
        evidence.candidates,
        key=lambda candidate: (
            isinstance(candidate, CircleCandidate),
            candidate.role is GeometryRole.HOLE,
        ),
    )
    assert len(shapes) == 3
    plate, slot, hole = shapes
    assert isinstance(plate, PolylineCandidate)
    assert plate.role is GeometryRole.STRUCTURE
    assert isinstance(slot, PolylineCandidate)
    assert slot.role is GeometryRole.HOLE
    assert isinstance(hole, CircleCandidate)
    assert hole.role is GeometryRole.HOLE
    assert hole.center.distance_to(Point(180.0, 200.0)) <= 2.0


def test_face_detection_respects_the_closed_shape_switch() -> None:
    # Given closed-shape detection turned off
    evidence = detect_face_shapes_with_evidence(
        _plate(corner_gap=0),
        DetectionConfig(detect_closed_shapes=False, faces=FaceConfig(enabled=True)),
    )

    # Then no face is read
    assert evidence.candidates == ()
