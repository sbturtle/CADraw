"""Immutable intermediate and CAD geometry models."""

from dataclasses import dataclass
from enum import StrEnum
from math import hypot
from typing import NewType

EntityId = NewType("EntityId", str)
DimensionId = NewType("DimensionId", str)


class Orientation(StrEnum):
    """Recognized line orientation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"
    UNKNOWN = "unknown"


class EntityType(StrEnum):
    """Supported CAD entity families."""

    LINE = "line"
    POLYLINE = "polyline"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ARC = "arc"


class DimensionKind(StrEnum):
    """Physical measurement kinds supported by the MVP."""

    LENGTH = "length"
    DIAMETER = "diameter"
    RADIUS = "radius"
    ANGLE = "angle"
    UNKNOWN = "unknown"


class DimensionStatus(StrEnum):
    """Human-verification state of a dimension."""

    UNSET = "unset"
    UNKNOWN = "unknown"
    NEEDS_REVIEW = "needs_review"
    USER_ENTERED = "needs_review"
    CONFIRMED = "confirmed"
    USER_CONFIRMED = "confirmed"
    AI_DETECTED = "ai_detected"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


class ConstraintKind(StrEnum):
    """Constraint families accepted by the deterministic geometry solver."""

    LENGTH = "length"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    ANGLE = "angle"
    COINCIDENT = "coincident"
    DIAMETER = "diameter"
    RADIUS = "radius"


class CadResolutionStatus(StrEnum):
    """Whether a CAD entity is safe to treat as resolved geometry."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    UNKNOWN_ANGLE = "unknown_angle"


class WarningCode(StrEnum):
    """Non-exceptional uncertainty reported by the pipeline."""

    LOW_CONFIDENCE_GEOMETRY = "LOW_CONFIDENCE_GEOMETRY"
    UNKNOWN_DIMENSION = "UNKNOWN_DIMENSION"
    UNCONNECTED_DIMENSION = "UNCONNECTED_DIMENSION"
    UNCERTAIN_TEXT = "UNCERTAIN_TEXT"
    CONSTRAINT_MISMATCH = "CONSTRAINT_MISMATCH"
    UNKNOWN_ANGLE = "UNKNOWN_ANGLE"
    IMAGE_READ_FAILED = "IMAGE_READ_FAILED"


@dataclass(frozen=True, slots=True)
class Point:
    """A point in the local drawing coordinate system."""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """Return Euclidean distance to another point."""
        return hypot(self.x - other.x, self.y - other.y)

    def scaled(self, factor: float) -> "Point":
        """Return this point scaled around the local origin."""
        return Point(self.x * factor, self.y * factor)

    def translated(self, dx: float, dy: float) -> "Point":
        """Return this point translated by a local offset."""
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True, slots=True)
class LineCandidate:
    """A raw line candidate emitted by image processing."""

    entity_id: EntityId
    start: Point
    end: Point
    confidence: float


@dataclass(frozen=True, slots=True)
class LineEntity:
    """A normalized line that preserves its raw endpoints."""

    entity_id: EntityId
    raw_start: Point
    raw_end: Point
    start: Point
    end: Point
    orientation: Orientation
    confidence: float

    @property
    def entity_type(self) -> EntityType:
        """Return the tagged type used by exporters."""
        return EntityType.LINE

    @property
    def length(self) -> float:
        """Return normalized line length."""
        return self.start.distance_to(self.end)


@dataclass(frozen=True, slots=True)
class PolylineEntity:
    """An ordered connected path."""

    entity_id: EntityId
    points: tuple[Point, ...]
    closed: bool
    confidence: float
    entity_type: EntityType = EntityType.POLYLINE


@dataclass(frozen=True, slots=True)
class CircleEntity:
    """A circle detected or supplied by a future vision adapter."""

    entity_id: EntityId
    center: Point
    radius: float
    confidence: float
    entity_type: EntityType = EntityType.CIRCLE


@dataclass(frozen=True, slots=True)
class ArcEntity:
    """A circular arc in degrees."""

    entity_id: EntityId
    center: Point
    radius: float
    start_angle_degrees: float
    end_angle_degrees: float
    confidence: float
    entity_type: EntityType = EntityType.ARC


type GeometryEntity = LineEntity | PolylineEntity | CircleEntity | ArcEntity


@dataclass(frozen=True, slots=True)
class DetectedEntity:
    """Image-space geometry and its source-image provenance."""

    entity_id: EntityId
    geometry: GeometryEntity
    confidence: float
    source_image_reference: str | None = None


@dataclass(frozen=True, slots=True)
class Constraint:
    """One typed geometric constraint supplied by a verified dimension."""

    kind: ConstraintKind
    value: float | None
    target_entity_ids: tuple[EntityId, ...]
    status: DimensionStatus
    inferred_for_preview: bool = False


@dataclass(frozen=True, slots=True)
class CadEntity:
    """CAD-space geometry linked back to one detected entity."""

    entity_id: EntityId
    source_entity_id: EntityId
    geometry: GeometryEntity
    constraints: tuple[Constraint, ...]
    status: CadResolutionStatus
    inferred_for_preview: bool = False


@dataclass(frozen=True, slots=True)
class Connection:
    """A relation between two line entities at a snapped endpoint."""

    first_entity_id: EntityId
    second_entity_id: EntityId
    point: Point


@dataclass(frozen=True, slots=True)
class Dimension:
    """A physical measurement that may still require human confirmation."""

    dimension_id: DimensionId
    raw_text: str
    kind: DimensionKind
    parsed_value: float | None
    unit: str
    target_entity_ids: tuple[EntityId, ...]
    confidence: float
    status: DimensionStatus
    source: str = "manual"


@dataclass(frozen=True, slots=True)
class AnalysisWarning:
    """A warning that is part of normal analysis output."""

    code: WarningCode
    message: str
    entity_ids: tuple[EntityId, ...] = ()


@dataclass(frozen=True, slots=True)
class DrawingAnalysis:
    """The complete intermediate representation for one source image."""

    unit: str
    image_size: tuple[int, int] | None
    entities: tuple[GeometryEntity, ...]
    dimensions: tuple[Dimension, ...]
    connections: tuple[Connection, ...]
    warnings: tuple[AnalysisWarning, ...]
    detected_entities: tuple[DetectedEntity, ...] = ()


@dataclass(frozen=True, slots=True)
class CadDocument:
    """Geometry ready for preview and deterministic DXF export."""

    unit: str
    entities: tuple[GeometryEntity, ...]
    dimensions: tuple[Dimension, ...]
    warnings: tuple[AnalysisWarning, ...]
    scale_mm_per_source_unit: float | None
    cad_entities: tuple[CadEntity, ...] = ()
