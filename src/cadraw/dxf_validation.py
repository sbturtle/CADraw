"""DXF round-trip validation kept separate from geometry serialization."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import ezdxf

_READ_FUNCTION = "readfile"


class ValidationDxfAttributesLike(Protocol):
    """DXF attributes read during validation."""

    text: str
    layer: str


class ValidationDxfEntityLike(Protocol):
    """DXF entity surface needed for round-trip checks."""

    dxf: ValidationDxfAttributesLike
    closed: bool

    def dxftype(self) -> str:
        """Return the DXF type name."""
        ...

    def get_points(self, point_format: str) -> Iterator[tuple[float, float]]:
        """Return lightweight-polyline vertices."""
        ...


class ValidationModelspaceLike(Protocol):
    """Modelspace iteration surface used by validation."""

    def __iter__(self) -> Iterator[ValidationDxfEntityLike]:
        """Iterate modelspace entities."""
        ...


class ValidationDocumentLike(Protocol):
    """DXF document surface used by validation."""

    def modelspace(self) -> ValidationModelspaceLike:
        """Return modelspace entities."""
        ...


def _read_document(path: str) -> ValidationDocumentLike:
    """Call ezdxf's dynamic top-level reader."""
    return cast(
        "ValidationDocumentLike",
        getattr(ezdxf, _READ_FUNCTION)(path),
    )


@dataclass(frozen=True, slots=True)
class DxfValidation:
    """Machine-readable result of reopening an exported DXF."""

    path: Path
    entity_counts: tuple[tuple[str, int], ...]
    text_values: tuple[str, ...] = ()
    polyline_points: tuple[tuple[tuple[float, float], ...], ...] = ()
    polyline_closed: tuple[bool, ...] = ()
    polyline_layers: tuple[str, ...] = ()

    @property
    def total_entities(self) -> int:
        """Return the total modelspace entity count."""
        return sum(count for _, count in self.entity_counts)

    def count(self, entity_type: str) -> int:
        """Return the count for one DXF type."""
        return dict(self.entity_counts).get(entity_type, 0)


def validate_dxf(path: Path) -> DxfValidation:
    """Reload a DXF and count its modelspace entities by type."""
    dxf_document = _read_document(str(path))
    counts: dict[str, int] = {}
    text_values: list[str] = []
    polyline_points: list[tuple[tuple[float, float], ...]] = []
    polyline_closed: list[bool] = []
    polyline_layers: list[str] = []
    for entity in dxf_document.modelspace():
        entity_type = entity.dxftype()
        counts[entity_type] = counts.get(entity_type, 0) + 1
        if entity_type == "TEXT":
            text_values.append(entity.dxf.text)
        if entity_type == "LWPOLYLINE":
            polyline_points.append(tuple(entity.get_points("xy")))
            polyline_closed.append(entity.closed)
            polyline_layers.append(entity.dxf.layer)
    return DxfValidation(
        path=path,
        entity_counts=tuple(sorted(counts.items())),
        text_values=tuple(text_values),
        polyline_points=tuple(polyline_points),
        polyline_closed=tuple(polyline_closed),
        polyline_layers=tuple(polyline_layers),
    )
