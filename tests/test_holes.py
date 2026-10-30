from cadraw.detection import classify_shape_holes
from cadraw.models import (
    ArcCandidate,
    CircleCandidate,
    EntityId,
    GeometryRole,
    Point,
    PolylineCandidate,
)


def test_shape_inside_outer_profile_is_marked_as_hole() -> None:
    # Given an outer rectangular profile and an inner circular mark
    outer = PolylineCandidate(
        entity_id=EntityId("outer"),
        points=(
            Point(10.0, 10.0),
            Point(210.0, 10.0),
            Point(210.0, 160.0),
            Point(10.0, 160.0),
        ),
        closed=True,
        confidence=0.9,
    )
    inner = CircleCandidate(
        entity_id=EntityId("inner"),
        center=Point(110.0, 80.0),
        radius=12.0,
        confidence=0.8,
    )

    # When semantic shape classification runs
    classified = classify_shape_holes((outer, inner))

    # Then only the contained mark receives hole semantics
    assert classified[0].role is GeometryRole.STRUCTURE
    assert classified[1].role is GeometryRole.HOLE


def test_shape_outside_profile_is_not_called_a_hole() -> None:
    # Given two unrelated closed shapes
    outer = PolylineCandidate(
        entity_id=EntityId("outer"),
        points=(
            Point(10.0, 10.0),
            Point(110.0, 10.0),
            Point(110.0, 110.0),
            Point(10.0, 110.0),
        ),
        closed=True,
        confidence=0.9,
    )
    outside = CircleCandidate(
        entity_id=EntityId("outside"),
        center=Point(180.0, 180.0),
        radius=10.0,
        confidence=0.8,
    )

    # When semantic shape classification runs
    classified = classify_shape_holes((outer, outside))

    # Then no containment relationship is fabricated
    assert all(candidate.role is GeometryRole.STRUCTURE for candidate in classified)


def test_arc_inside_outer_profile_is_marked_as_hole() -> None:
    # Given an outer rectangular profile and an inner partial-hole arc
    outer = PolylineCandidate(
        entity_id=EntityId("outer"),
        points=(
            Point(10.0, 10.0),
            Point(210.0, 10.0),
            Point(210.0, 160.0),
            Point(10.0, 160.0),
        ),
        closed=True,
        confidence=0.9,
    )
    inner = ArcCandidate(
        entity_id=EntityId("inner"),
        center=Point(110.0, 80.0),
        radius=12.0,
        start_angle_degrees=10.0,
        end_angle_degrees=280.0,
        confidence=0.8,
    )

    # When semantic shape classification runs
    classified = classify_shape_holes((outer, inner))

    # Then only the contained arc receives hole semantics
    assert classified[0].role is GeometryRole.STRUCTURE
    assert classified[1].role is GeometryRole.HOLE


def test_a_hole_drilled_near_the_rim_of_a_round_plate_is_a_hole() -> None:
    # Given a round plate and a hole drawn just inside its rim
    plate = CircleCandidate(
        entity_id=EntityId("plate"),
        center=Point(391.0, 967.0),
        radius=239.0,
        confidence=0.9,
    )
    hole = CircleCandidate(
        entity_id=EntityId("hole"),
        center=Point(568.0, 1099.0),
        radius=11.0,
        confidence=0.8,
    )

    classified = classify_shape_holes((plate, hole))

    # Then nearness to the rim does not make it a separate part
    assert classified[1].role is GeometryRole.HOLE
