"""DXF serialization and round-trip validation."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import ezdxf

from cadraw.models import (
    ArcEntity,
    CadDocument,
    CircleEntity,
    Dimension,
    DimensionKind,
    EntityId,
    GeometryEntity,
    LineEntity,
    Point,
    PolylineEntity,
)

_GEOMETRY_LAYER = "GEOMETRY"
_DIMENSION_LAYER = "DIMENSION"
_TEXT_HEIGHT = 3.0
_MILLIMETRES = 4
_NEW_FUNCTION = "new"
_READ_FUNCTION = "readfile"


class DxfAttributesLike(Protocol):
    """Minimal attribute surface used by the exporter."""

    insert: tuple[float, float]
    text: str


class DxfEntityLike(Protocol):
    """Minimal entity surface needed for writing and validation."""

    dxf: DxfAttributesLike

    def dxftype(self) -> str:
        """Return the DXF type name."""
        ...


class ModelspaceLike(Protocol):
    """Minimal ezdxf modelspace surface used by this module."""

    def __iter__(self) -> Iterator[DxfEntityLike]:
        """Iterate modelspace entities."""
        ...

    def add_line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        dxfattribs: dict[str, str | float],
    ) -> DxfEntityLike:
        """Add a line."""
        ...

    def add_lwpolyline(
        self,
        points: list[tuple[float, float]],
        **kwargs: str | bool | dict[str, str | float],
    ) -> DxfEntityLike:
        """Add a lightweight polyline."""
        ...

    def add_circle(
        self,
        center: tuple[float, float],
        radius: float,
        *,
        dxfattribs: dict[str, str | float],
    ) -> DxfEntityLike:
        """Add a circle."""
        ...

    def add_arc(
        self,
        center: tuple[float, float],
        radius: float,
        start_angle: float,
        end_angle: float,
        *,
        dxfattribs: dict[str, str | float],
    ) -> DxfEntityLike:
        """Add an arc."""
        ...

    def add_text(
        self,
        text: str,
        *,
        dxfattribs: dict[str, str | float],
    ) -> DxfEntityLike:
        """Add text."""
        ...


class LayerTableLike(Protocol):
    """Minimal layer-table surface."""

    def add(self, name: str) -> DxfEntityLike:
        """Add a layer."""
        ...


class DxfDocumentLike(Protocol):
    """Minimal ezdxf document surface."""

    units: int
    layers: LayerTableLike

    def modelspace(self) -> ModelspaceLike:
        """Return modelspace."""
        ...

    def saveas(self, path: str) -> None:
        """Save the document."""
        ...


class EzdxfApi(Protocol):
    """Dynamic top-level ezdxf functions used by the exporter."""

    def new(self, version: str) -> DxfDocumentLike:
        """Create a new document."""
        ...

    def readfile(self, path: str) -> DxfDocumentLike:
        """Read a document."""
        ...


def _new_document(version: str) -> DxfDocumentLike:
    """Call ezdxf's dynamic top-level constructor."""
    return cast("DxfDocumentLike", getattr(ezdxf, _NEW_FUNCTION)(version))


def _read_document(path: str) -> DxfDocumentLike:
    """Call ezdxf's dynamic top-level reader."""
    return cast("DxfDocumentLike", getattr(ezdxf, _READ_FUNCTION)(path))


@dataclass(frozen=True, slots=True)
class ExportSummary:
    """Counts produced by one export operation."""

    path: Path
    geometry_entity_count: int
    dimension_text_count: int


@dataclass(frozen=True, slots=True)
class DxfValidation:
    """Machine-readable result of reopening an exported DXF."""

    path: Path
    entity_counts: tuple[tuple[str, int], ...]
    text_values: tuple[str, ...] = ()

    @property
    def total_entities(self) -> int:
        """Return the total modelspace entity count."""
        return sum(count for _, count in self.entity_counts)

    def count(self, entity_type: str) -> int:
        """Return the count for one DXF type."""
        return dict(self.entity_counts).get(entity_type, 0)


def _point_tuple(point: Point) -> tuple[float, float]:
    """Convert a domain point to an ezdxf 2D tuple."""
    return point.x, point.y


def _add_geometry(modelspace: ModelspaceLike, entity: GeometryEntity) -> None:
    """Write one known geometry variant to the geometry layer."""
    match entity:
        case LineEntity(start=start, end=end):
            _ = modelspace.add_line(
                _point_tuple(start),
                _point_tuple(end),
                dxfattribs={"layer": _GEOMETRY_LAYER},
            )
        case PolylineEntity(points=points, closed=closed):
            _ = modelspace.add_lwpolyline(
                [_point_tuple(point) for point in points],
                format="xy",
                close=closed,
                dxfattribs={"layer": _GEOMETRY_LAYER},
            )
        case CircleEntity(center=center, radius=radius):
            _ = modelspace.add_circle(
                _point_tuple(center),
                radius,
                dxfattribs={"layer": _GEOMETRY_LAYER},
            )
        case ArcEntity(
            center=center,
            radius=radius,
            start_angle_degrees=start_angle,
            end_angle_degrees=end_angle,
        ):
            _ = modelspace.add_arc(
                _point_tuple(center),
                radius,
                start_angle,
                end_angle,
                dxfattribs={"layer": _GEOMETRY_LAYER},
            )


def _entity_anchor(entity: GeometryEntity) -> Point:
    """Return a stable text anchor for a target entity."""
    match entity:
        case LineEntity(start=start, end=end):
            return Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
        case PolylineEntity(points=points):
            return points[len(points) // 2]
        case CircleEntity(center=center):
            return center
        case ArcEntity(center=center):
            return center


def _add_dimension_text(
    modelspace: ModelspaceLike,
    dimension: Dimension,
    entities_by_id: dict[EntityId, GeometryEntity],
) -> bool:
    """Write only known dimensions with an explicit target."""
    if dimension.parsed_value is None or not dimension.target_entity_ids:
        return False
    target = entities_by_id.get(dimension.target_entity_ids[0])
    if target is None:
        return False
    anchor = _entity_anchor(target)
    label = (
        dimension.raw_text
        if dimension.kind is DimensionKind.ANGLE
        else f"{dimension.raw_text} {dimension.unit}"
    )
    text = modelspace.add_text(
        label,
        dxfattribs={"layer": _DIMENSION_LAYER, "height": _TEXT_HEIGHT},
    )
    text.dxf.insert = _point_tuple(anchor)
    return True


def export_dxf(document: CadDocument, path: Path) -> ExportSummary:
    """Export known geometry and safe dimension labels to DXF."""
    dxf_document = _new_document("R2010")
    dxf_document.units = _MILLIMETRES
    _ = dxf_document.layers.add(_GEOMETRY_LAYER)
    _ = dxf_document.layers.add(_DIMENSION_LAYER)
    modelspace = dxf_document.modelspace()
    for entity in document.entities:
        _add_geometry(modelspace, entity)
    entities_by_id = {entity.entity_id: entity for entity in document.entities}
    dimension_count = sum(
        _add_dimension_text(modelspace, dimension, entities_by_id)
        for dimension in document.dimensions
    )
    dxf_document.saveas(str(path))
    return ExportSummary(
        path=path,
        geometry_entity_count=len(document.entities),
        dimension_text_count=dimension_count,
    )


def validate_dxf(path: Path) -> DxfValidation:
    """Reload a DXF and count its modelspace entities by type."""
    dxf_document = _read_document(str(path))
    counts: dict[str, int] = {}
    text_values: list[str] = []
    for entity in dxf_document.modelspace():
        entity_type = entity.dxftype()
        counts[entity_type] = counts.get(entity_type, 0) + 1
        if entity_type == "TEXT":
            text_values.append(entity.dxf.text)
    return DxfValidation(
        path=path,
        entity_counts=tuple(sorted(counts.items())),
        text_values=tuple(text_values),
    )
