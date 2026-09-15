"""Conservative arbitration of duplicate image-space line candidates."""

from dataclasses import dataclass, field
from math import atan2, degrees

from cadraw.grid_filter import GridFilterConfig
from cadraw.models import LineCandidate, Point

_HALF_TURN_DEGREES = 180.0


@dataclass(frozen=True, slots=True)
class CandidateArbitrationConfig:
    """Tolerances used to identify two rails from one physical stroke."""

    duplicate_angle_tolerance_degrees: float = 2.0
    duplicate_perpendicular_distance: float = 8.0
    duplicate_overlap_ratio: float = 0.85
    grid: GridFilterConfig = field(default_factory=GridFilterConfig)


def _line_length(candidate: LineCandidate) -> float:
    """Return one candidate's Euclidean length."""
    return candidate.start.distance_to(candidate.end)


def _line_angle(candidate: LineCandidate) -> float:
    """Return an orientation-independent angle in the half-turn range."""
    return (
        degrees(
            atan2(
                candidate.end.y - candidate.start.y,
                candidate.end.x - candidate.start.x,
            ),
        )
        % _HALF_TURN_DEGREES
    )


def _angle_delta(first: LineCandidate, second: LineCandidate) -> float:
    """Return the smallest orientation difference between two lines."""
    difference = abs(_line_angle(first) - _line_angle(second))
    return min(difference, _HALF_TURN_DEGREES - difference)


def _unit_direction(candidate: LineCandidate) -> tuple[float, float] | None:
    """Return the unit direction of a non-degenerate candidate."""
    length = _line_length(candidate)
    if length == 0.0:
        return None
    return (
        (candidate.end.x - candidate.start.x) / length,
        (candidate.end.y - candidate.start.y) / length,
    )


def _perpendicular_distance(
    point: Point,
    origin: Point,
    direction: tuple[float, float],
) -> float:
    """Return the distance from a point to a supporting line."""
    return abs(
        (point.x - origin.x) * direction[1] - (point.y - origin.y) * direction[0],
    )


def _project(
    point: Point,
    origin: Point,
    direction: tuple[float, float],
) -> float:
    """Project a point onto a candidate's supporting axis."""
    return (point.x - origin.x) * direction[0] + (point.y - origin.y) * direction[1]


def _projected_overlap(
    first: LineCandidate,
    second: LineCandidate,
    direction: tuple[float, float],
) -> float:
    """Return the overlap length of two segments on the first axis."""
    first_projections = sorted(
        _project(point, first.start, direction) for point in (first.start, first.end)
    )
    second_projections = sorted(
        _project(point, first.start, direction) for point in (second.start, second.end)
    )
    return max(
        0.0,
        min(first_projections[1], second_projections[1])
        - max(first_projections[0], second_projections[0]),
    )


def _is_duplicate(
    first: LineCandidate,
    second: LineCandidate,
    config: CandidateArbitrationConfig,
) -> bool:
    """Check strict angle, offset, and overlap conditions for one duplicate."""
    if _angle_delta(first, second) > config.duplicate_angle_tolerance_degrees:
        return False
    direction = _unit_direction(first)
    if direction is None:
        return False
    offset = max(
        _perpendicular_distance(second.start, first.start, direction),
        _perpendicular_distance(second.end, first.start, direction),
    )
    if offset > config.duplicate_perpendicular_distance:
        return False
    shorter_length = min(_line_length(first), _line_length(second))
    if shorter_length == 0.0:
        return False
    overlap_ratio = _projected_overlap(first, second, direction) / shorter_length
    return overlap_ratio >= config.duplicate_overlap_ratio


def arbitrate_line_candidates(
    candidates: tuple[LineCandidate, ...],
    config: CandidateArbitrationConfig | None = None,
) -> tuple[LineCandidate, ...]:
    """Keep one deterministic representative for each physical stroke."""
    settings = config if config is not None else CandidateArbitrationConfig()
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -_line_length(candidate),
            -candidate.confidence,
            str(candidate.entity_id),
        ),
    )
    retained: list[LineCandidate] = []
    for candidate in ranked:
        if any(_is_duplicate(existing, candidate, settings) for existing in retained):
            continue
        retained.append(candidate)
    return tuple(retained)
