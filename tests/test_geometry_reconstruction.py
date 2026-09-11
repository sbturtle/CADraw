from pathlib import Path

from cadraw.calculator import calculate_document
from cadraw.export import export_dxf, validate_dxf
from cadraw.models import (
    CadResolutionStatus,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)
from cadraw.normalization import build_connections


def _line(
    entity_id: str,
    start: Point,
    end: Point,
    orientation: Orientation,
) -> LineEntity:
    return LineEntity(
        entity_id=EntityId(entity_id),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=orientation,
        confidence=1.0,
    )


def _dimension(
    dimension_id: str,
    kind: DimensionKind,
    value: float,
    entity_id: str,
) -> Dimension:
    return Dimension(
        dimension_id=DimensionId(dimension_id),
        raw_text=str(value),
        kind=kind,
        parsed_value=value,
        unit="mm",
        target_entity_ids=(EntityId(entity_id),),
        confidence=1.0,
        status=DimensionStatus.CONFIRMED,
    )


def _analysis(
    lines: tuple[LineEntity, ...],
    dimensions: tuple[Dimension, ...],
) -> DrawingAnalysis:
    return DrawingAnalysis(
        unit="mm",
        image_size=(800, 600),
        entities=lines,
        dimensions=dimensions,
        connections=build_connections(lines, 0.01),
        warnings=(),
    )


def test_horizontal_dimension_reconstructs_from_origin_in_mm() -> None:
    # Given a detected horizontal line and one confirmed length
    line = _line(
        "line_001",
        Point(124.0, 180.0),
        Point(471.0, 184.0),
        Orientation.HORIZONTAL,
    )
    analysis = _analysis(
        (line,),
        (_dimension("dim_001", DimensionKind.LENGTH, 100.0, "line_001"),),
    )

    # When the CAD document is reconstructed
    document = calculate_document(analysis)

    # Then the line is exactly 100 mm from the deterministic anchor
    result = document.entities[0]
    assert isinstance(result, LineEntity)
    assert result.start == Point(0.0, 0.0)
    assert result.end == Point(100.0, 0.0)
    assert result.length == 100.0


def test_vertical_downward_dimension_reconstructs_with_cad_y_down() -> None:
    # Given a detected downward vertical line and one confirmed length
    line = _line(
        "line_001",
        Point(30.0, 40.0),
        Point(30.0, 340.0),
        Orientation.VERTICAL,
    )
    analysis = _analysis(
        (line,),
        (_dimension("dim_001", DimensionKind.LENGTH, 300.0, "line_001"),),
    )

    # When the CAD document is reconstructed
    document = calculate_document(analysis)

    # Then downward source direction becomes negative CAD Y
    result = document.entities[0]
    assert isinstance(result, LineEntity)
    assert result.start == Point(0.0, 0.0)
    assert result.end == Point(0.0, -300.0)
    assert result.length == 300.0


def test_connected_l_shape_uses_each_confirmed_segment_length() -> None:
    # Given an exact connected horizontal and downward vertical path
    lines = (
        _line(
            "line_001", Point(10.0, 20.0), Point(110.0, 20.0), Orientation.HORIZONTAL
        ),
        _line(
            "line_002", Point(110.0, 20.0), Point(110.0, 320.0), Orientation.VERTICAL
        ),
    )
    analysis = _analysis(
        lines,
        (
            _dimension("dim_001", DimensionKind.LENGTH, 100.0, "line_001"),
            _dimension("dim_002", DimensionKind.LENGTH, 300.0, "line_002"),
        ),
    )

    # When both segment constraints are solved together
    document = calculate_document(analysis)

    # Then the shared endpoint is exact in CAD space
    first, second = document.entities
    assert isinstance(first, LineEntity)
    assert isinstance(second, LineEntity)
    assert first.start == Point(0.0, 0.0)
    assert first.end == Point(100.0, 0.0)
    assert second.start == Point(100.0, 0.0)
    assert second.end == Point(100.0, -300.0)


def test_diagonal_length_and_angle_use_trigonometric_reconstruction() -> None:
    # Given a diagonal line with explicit length and CAD angle
    line = _line(
        "line_001",
        Point(0.0, 0.0),
        Point(10.0, 10.0),
        Orientation.DIAGONAL,
    )
    analysis = _analysis(
        (line,),
        (
            _dimension("dim_length", DimensionKind.LENGTH, 100.0, "line_001"),
            _dimension("dim_angle", DimensionKind.ANGLE, 45.0, "line_001"),
        ),
    )

    # When the diagonal is reconstructed
    document = calculate_document(analysis)

    # Then the explicit polar constraint determines CAD y-up deltas
    result = document.entities[0]
    assert isinstance(result, LineEntity)
    assert result.start == Point(0.0, 0.0)
    assert result.end.x == 70.71067811865476
    assert result.end.y == -70.71067811865476
    assert result.length == 100.0


def test_missing_dimension_is_unresolved_without_crashing() -> None:
    # Given a detected line with no physical dimension
    line = _line(
        "line_001",
        Point(10.0, 20.0),
        Point(110.0, 20.0),
        Orientation.HORIZONTAL,
    )
    document = calculate_document(_analysis((line,), ()))

    # Then the local geometry remains visible but is not resolved CAD data
    assert document.entities == (line,)
    assert document.cad_entities[0].status is CadResolutionStatus.UNRESOLVED
    assert document.scale_mm_per_source_unit is None


def test_reconstructed_l_shape_round_trips_exact_segment_lengths(
    tmp_path: Path,
) -> None:
    # Given a connected L shape with confirmed CAD dimensions
    lines = (
        _line(
            "line_001", Point(10.0, 20.0), Point(110.0, 20.0), Orientation.HORIZONTAL
        ),
        _line(
            "line_002", Point(110.0, 20.0), Point(110.0, 320.0), Orientation.VERTICAL
        ),
    )
    document = calculate_document(
        _analysis(
            lines,
            (
                _dimension("dim_001", DimensionKind.LENGTH, 100.0, "line_001"),
                _dimension("dim_002", DimensionKind.LENGTH, 300.0, "line_002"),
            ),
        ),
    )
    output_path = tmp_path / "l-shape.dxf"

    # When CAD-space geometry is exported and reopened
    _ = export_dxf(document, output_path)
    validation = validate_dxf(output_path)

    # Then DXF segment lengths are the entered millimetres, not source pixels
    assert validation.polyline_segment_lengths == ((100.0, 300.0),)
