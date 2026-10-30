import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.dimension_lines import DimensionText, find_dimension_lines
from cadraw.dimensions import parse_dimension_text
from cadraw.models import (
    CircleEntity,
    DimensionId,
    DimensionKind,
    EntityId,
    LineEntity,
    Orientation,
    Point,
    PolylineEntity,
)

type InkMask = NDArray[np.uint8]

_PEN = 3


def _line(name: str, start: Point, end: Point) -> LineEntity:
    return LineEntity(
        entity_id=EntityId(name),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=Orientation.VERTICAL,
        confidence=0.9,
    )


def _ink() -> InkMask:
    return np.zeros((400, 600), dtype=np.uint8)


def _stroke(ink: InkMask, start: tuple[int, int], end: tuple[int, int]) -> None:
    _ = cv2.line(ink, start, end, 255, _PEN)


def _ticked_line(ink: InkMask, x: int, top: int, bottom: int) -> LineEntity:
    """Draw a vertical dimension line closed by a short tick at each end."""
    _stroke(ink, (x, top), (x, bottom))
    _stroke(ink, (x - 9, top), (x + 9, top))
    _stroke(ink, (x - 9, bottom), (x + 9, bottom))
    return _line(
        "dimension", Point(float(x), float(top)), Point(float(x), float(bottom))
    )


def _value(text: str, bounds: tuple[float, float, float, float]) -> DimensionText:
    return DimensionText(
        dimension=parse_dimension_text(text, DimensionId("ocr_001")),
        bounds=bounds,
    )


def _rim() -> CircleEntity:
    return CircleEntity(
        entity_id=EntityId("rim"),
        center=Point(300.0, 200.0),
        radius=80.0,
        confidence=0.9,
    )


def test_a_ticked_line_beside_a_circle_measures_its_diameter() -> None:
    # Given a circle and, beside it, a ticked line spanning its two poles
    ink = _ink()
    _ = cv2.circle(ink, (300, 200), 80, 255, _PEN)
    dimension = _ticked_line(ink, 450, 120, 280)

    # When the value written along the line is read
    found = find_dimension_lines(
        (dimension,),
        (_value("40mm", (462.0, 185.0, 40.0, 30.0)),),
        (dimension, _rim()),
        ink,
        float(_PEN),
    )

    # Then the line is a dimension and the value sizes the circle
    assert len(found) == 1
    assert found[0].line_id == dimension.entity_id
    assert found[0].target_id == EntityId("rim")
    assert found[0].kind is DimensionKind.DIAMETER


def test_a_plate_edge_closed_by_long_edges_is_not_a_dimension() -> None:
    # Given a plate's side edge with a value beside it, joined to long edges
    ink = _ink()
    _stroke(ink, (450, 120), (450, 280))
    _stroke(ink, (300, 120), (580, 120))
    _stroke(ink, (300, 280), (580, 280))
    edge = _line("edge", Point(450.0, 120.0), Point(450.0, 280.0))

    found = find_dimension_lines(
        (edge,),
        (_value("40", (462.0, 185.0, 30.0, 30.0)),),
        (edge,),
        ink,
        float(_PEN),
    )

    # Then it stays geometry: what crosses its ends runs on past a tick
    assert found == ()


def test_a_ticked_line_without_a_value_stays_geometry() -> None:
    # Given a ticked line with nothing written beside it
    ink = _ink()
    dimension = _ticked_line(ink, 450, 120, 280)

    found = find_dimension_lines(
        (dimension,),
        (_value("40mm", (100.0, 30.0, 40.0, 30.0)),),
        (dimension,),
        ink,
        float(_PEN),
    )

    assert found == ()


def test_a_chain_of_values_along_one_line_leaves_the_target_open() -> None:
    # Given two sizes written along one ticked line beside a circle
    ink = _ink()
    _ = cv2.circle(ink, (300, 200), 80, 255, _PEN)
    dimension = _ticked_line(ink, 450, 120, 280)

    found = find_dimension_lines(
        (dimension,),
        (
            _value("10mm", (462.0, 130.0, 40.0, 30.0)),
            _value("30mm", (462.0, 220.0, 40.0, 30.0)),
        ),
        (dimension, _rim()),
        ink,
        float(_PEN),
    )

    # Then the line carries both values, and neither measures its ends
    assert sorted(record.dimension_index for record in found) == [0, 1]
    assert {record.line_id for record in found} == {dimension.entity_id}
    assert all(record.target_id is None for record in found)


def test_a_line_matching_two_circles_leaves_the_target_open() -> None:
    # Given equal circles either side of a ticked line spanning both
    ink = _ink()
    _ = cv2.circle(ink, (300, 200), 80, 255, _PEN)
    dimension = _ticked_line(ink, 450, 120, 280)
    twin = CircleEntity(
        entity_id=EntityId("twin"),
        center=Point(560.0, 200.0),
        radius=80.0,
        confidence=0.9,
    )

    found = find_dimension_lines(
        (dimension,),
        (_value("40mm", (462.0, 185.0, 40.0, 30.0)),),
        (dimension, _rim(), twin),
        ink,
        float(_PEN),
    )

    # Then the reviewer chooses: the value is not handed to either one
    assert len(found) == 1
    assert found[0].target_id is None
    assert found[0].kind is DimensionKind.LENGTH


def test_a_ticked_line_spanning_a_plate_measures_its_width() -> None:
    # Given a plate outline and a ticked line below it spanning its full width
    ink = _ink()
    _ = cv2.rectangle(ink, (100, 80), (400, 220), 255, _PEN)
    _stroke(ink, (100, 300), (400, 300))
    _stroke(ink, (100, 291), (100, 309))
    _stroke(ink, (400, 291), (400, 309))
    dimension = _line("dimension", Point(100.0, 300.0), Point(400.0, 300.0))
    plate = PolylineEntity(
        entity_id=EntityId("plate"),
        points=(
            Point(100.0, 80.0),
            Point(400.0, 80.0),
            Point(400.0, 220.0),
            Point(100.0, 220.0),
        ),
        closed=True,
        confidence=0.9,
    )

    found = find_dimension_lines(
        (dimension,),
        (_value("300", (230.0, 312.0, 40.0, 24.0)),),
        (dimension, plate),
        ink,
        float(_PEN),
    )

    # Then the value is the plate's overall width
    assert len(found) == 1
    assert found[0].target_id == EntityId("plate")
    assert found[0].kind is DimensionKind.WIDTH


def test_a_far_outline_of_the_same_width_is_not_the_target() -> None:
    # Given a ticked line, and far below it a plate of exactly its width
    ink = _ink()
    _stroke(ink, (100, 40), (400, 40))
    _stroke(ink, (100, 31), (100, 49))
    _stroke(ink, (400, 31), (400, 49))
    dimension = _line("dimension", Point(100.0, 40.0), Point(400.0, 40.0))
    far_plate = PolylineEntity(
        entity_id=EntityId("far_plate"),
        points=(
            Point(100.0, 360.0),
            Point(400.0, 360.0),
            Point(400.0, 900.0),
            Point(100.0, 900.0),
        ),
        closed=True,
        confidence=0.9,
    )

    found = find_dimension_lines(
        (dimension,),
        (_value("300", (230.0, 52.0, 40.0, 24.0)),),
        (dimension, far_plate),
        ink,
        float(_PEN),
    )

    # Then nothing ties the plate to the line, so the value stays open
    assert len(found) == 1
    assert found[0].target_id is None
