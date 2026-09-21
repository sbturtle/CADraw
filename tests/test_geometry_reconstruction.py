from pathlib import Path

from cadraw.calculator import calculate_document
from cadraw.export import export_dxf, validate_dxf
from cadraw.models import (
    ArcEntity,
    CadResolutionStatus,
    CircleEntity,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    DrawingAnalysis,
    EntityId,
    LineEntity,
    Orientation,
    Point,
    PolylineEntity,
    WarningCode,
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


def _confirmed_dimension(
    entity_id: str,
    kind: DimensionKind,
    value: float,
) -> Dimension:
    """Build one confirmed dimension attached to a single entity."""
    return Dimension(
        dimension_id=DimensionId(f"{entity_id}_{kind.value}"),
        raw_text=f"{value:g}",
        kind=kind,
        parsed_value=value,
        unit="mm",
        target_entity_ids=(EntityId(entity_id),),
        confidence=1.0,
        status=DimensionStatus.CONFIRMED,
    )


def _profile(entity_id: str) -> PolylineEntity:
    """Build a closed outline traced 400 by 200 in source pixels."""
    return PolylineEntity(
        entity_id=EntityId(entity_id),
        points=(
            Point(100.0, 100.0),
            Point(500.0, 100.0),
            Point(500.0, 300.0),
            Point(100.0, 300.0),
        ),
        closed=True,
        confidence=0.9,
    )


def _shape_analysis(
    entities: tuple[LineEntity | PolylineEntity | CircleEntity | ArcEntity, ...],
    dimensions: tuple[Dimension, ...],
) -> DrawingAnalysis:
    """Build a minimal analysis around explicit entities and dimensions."""
    return DrawingAnalysis(
        unit="mm",
        image_size=(800, 600),
        entities=entities,
        dimensions=dimensions,
        connections=(),
        warnings=(),
    )


def test_a_shape_nobody_confirmed_is_not_treated_as_measured() -> None:
    # Given a confirmed line length and a hole nobody gave a size
    line = _line(
        "line_a",
        Point(100.0, 500.0),
        Point(500.0, 500.0),
        Orientation.HORIZONTAL,
    )
    hole = CircleEntity(
        entity_id=EntityId("hole"),
        center=Point(300.0, 200.0),
        radius=50.0,
        confidence=0.9,
    )
    analysis = _shape_analysis(
        (line, hole),
        (_confirmed_dimension("line_a", DimensionKind.LENGTH, 800.0),),
    )

    # When the document is calculated
    document = calculate_document(analysis)

    # Then the hole stays unresolved, so the sheet's scale is not its size
    states = {entity.entity_id: entity.status for entity in document.cad_entities}
    assert states[EntityId("line_a")] is CadResolutionStatus.RESOLVED
    assert states[EntityId("hole")] is CadResolutionStatus.UNRESOLVED


def test_a_confirmed_hole_diameter_resolves_that_hole(tmp_path: Path) -> None:
    # Given a drilled hole whose diameter a reviewer confirmed
    hole = CircleEntity(
        entity_id=EntityId("hole"),
        center=Point(300.0, 200.0),
        radius=50.0,
        confidence=0.9,
    )
    analysis = _shape_analysis(
        (hole,),
        (_confirmed_dimension("hole", DimensionKind.DIAMETER, 40.0),),
    )

    # When the document is calculated and exported
    document = calculate_document(analysis)
    export_path = tmp_path / "hole.dxf"
    _ = export_dxf(document, export_path)

    # Then that one value both calibrates and sizes it
    assert document.scale_mm_per_source_unit is not None
    assert document.cad_entities[0].status is CadResolutionStatus.RESOLVED
    assert validate_dxf(export_path).count("CIRCLE") == 1


def test_both_confirmed_extents_size_a_profile_on_each_axis() -> None:
    # Given a 400x200 traced outline confirmed as a 600 by 140 part
    profile = _profile("plate")
    analysis = _shape_analysis(
        (profile,),
        (
            _confirmed_dimension("plate", DimensionKind.WIDTH, 600.0),
            _confirmed_dimension("plate", DimensionKind.HEIGHT, 140.0),
        ),
    )

    # When the document is calculated
    document = calculate_document(analysis)

    # Then the outline comes out at exactly the confirmed overall size
    geometry = document.cad_entities[0].geometry
    assert isinstance(geometry, PolylineEntity)
    assert geometry.width == 600.0
    assert geometry.height == 140.0


def test_one_confirmed_extent_keeps_the_traced_proportions() -> None:
    # Given only the overall width of a 400x200 traced outline
    profile = _profile("plate")
    analysis = _shape_analysis(
        (profile,),
        (_confirmed_dimension("plate", DimensionKind.WIDTH, 800.0),),
    )

    # When the document is calculated
    document = calculate_document(analysis)

    # Then the height follows the same factor rather than being invented
    geometry = document.cad_entities[0].geometry
    assert isinstance(geometry, PolylineEntity)
    assert geometry.width == 800.0
    assert geometry.height == 400.0


def test_stretching_a_profile_to_fit_is_reported() -> None:
    # Given extents that disagree with the traced proportions
    profile = _profile("plate")
    analysis = _shape_analysis(
        (profile,),
        (
            _confirmed_dimension("plate", DimensionKind.WIDTH, 600.0),
            _confirmed_dimension("plate", DimensionKind.HEIGHT, 140.0),
        ),
    )

    # When the document is calculated
    document = calculate_document(analysis)

    # Then the reviewer is told the inside of the outline is stretched
    assert any(
        warning.code is WarningCode.CONSTRAINT_MISMATCH for warning in document.warnings
    )


def test_matching_extents_are_not_reported_as_stretched() -> None:
    # Given extents that agree with the traced 2:1 proportions
    profile = _profile("plate")
    analysis = _shape_analysis(
        (profile,),
        (
            _confirmed_dimension("plate", DimensionKind.WIDTH, 800.0),
            _confirmed_dimension("plate", DimensionKind.HEIGHT, 400.0),
        ),
    )

    # When the document is calculated
    document = calculate_document(analysis)

    # Then nothing is flagged, because no stretching was needed
    assert not any(
        warning.code is WarningCode.CONSTRAINT_MISMATCH for warning in document.warnings
    )


def test_lines_and_shapes_share_one_cad_frame() -> None:
    # Given a stem drawn as a line rising from the top edge of a traced outline
    stem = _line(
        "stem",
        Point(300.0, 300.0),
        Point(300.0, 100.0),
        Orientation.VERTICAL,
    )
    outline = PolylineEntity(
        entity_id=EntityId("bar"),
        points=(
            Point(100.0, 300.0),
            Point(500.0, 300.0),
            Point(500.0, 400.0),
            Point(100.0, 400.0),
        ),
        closed=True,
        confidence=0.9,
    )
    hole = CircleEntity(
        entity_id=EntityId("hole"),
        center=Point(400.0, 350.0),
        radius=20.0,
        confidence=0.9,
    )
    analysis = _shape_analysis(
        (stem, outline, hole),
        (
            _confirmed_dimension("stem", DimensionKind.LENGTH, 100.0),
            _confirmed_dimension("bar", DimensionKind.WIDTH, 200.0),
        ),
    )

    # When the document is calculated
    document = calculate_document(analysis)
    geometry = {entity.entity_id: entity.geometry for entity in document.cad_entities}

    # Then the stem rises from the origin, the bar hangs below it in the same
    # y-up frame, and the hole sits inside the bar rather than mirrored above
    cad_stem = geometry[EntityId("stem")]
    cad_bar = geometry[EntityId("bar")]
    cad_hole = geometry[EntityId("hole")]
    assert isinstance(cad_stem, LineEntity)
    assert isinstance(cad_bar, PolylineEntity)
    assert isinstance(cad_hole, CircleEntity)
    assert cad_stem.start == Point(0.0, 0.0)
    assert cad_stem.end == Point(0.0, 100.0)
    assert max(point.y for point in cad_bar.points) == 0.0
    assert min(point.y for point in cad_bar.points) == -50.0
    assert min(point.x for point in cad_bar.points) == -100.0
    assert cad_bar.width == 200.0
    assert cad_hole.center == Point(50.0, -25.0)


def test_a_second_line_component_keeps_its_place_on_the_sheet() -> None:
    # Given two separate parts drawn side by side
    first = _line("a", Point(100.0, 500.0), Point(300.0, 500.0), Orientation.HORIZONTAL)
    second = _line(
        "b", Point(600.0, 200.0), Point(800.0, 200.0), Orientation.HORIZONTAL
    )
    analysis = _shape_analysis(
        (first, second),
        (_confirmed_dimension("a", DimensionKind.LENGTH, 100.0),),
    )

    # When the document is calculated
    document = calculate_document(analysis)
    geometry = {entity.entity_id: entity.geometry for entity in document.cad_entities}

    # Then the second part is to the right and above the first, as photographed
    cad_second = geometry[EntityId("b")]
    assert isinstance(cad_second, LineEntity)
    assert cad_second.start == Point(250.0, 150.0)
    assert cad_second.end == Point(350.0, 150.0)


def test_arc_sweep_is_mirrored_with_the_y_axis() -> None:
    # Given an arc that sweeps from +x toward +y on the photograph (downward)
    arc = ArcEntity(
        entity_id=EntityId("fillet"),
        center=Point(200.0, 200.0),
        radius=50.0,
        start_angle_degrees=0.0,
        end_angle_degrees=90.0,
        confidence=0.9,
    )
    analysis = _shape_analysis(
        (arc,),
        (_confirmed_dimension("fillet", DimensionKind.RADIUS, 25.0),),
    )

    # When the document is calculated
    document = calculate_document(analysis)
    geometry = document.cad_entities[0].geometry

    # Then in the y-up CAD frame the same arc sweeps from 270 back to 360
    assert isinstance(geometry, ArcEntity)
    assert geometry.radius == 25.0
    assert (geometry.start_angle_degrees, geometry.end_angle_degrees) == (270.0, 0.0)


def test_a_sheet_without_lines_is_anchored_at_its_bottom_left() -> None:
    # Given only a traced outline with one confirmed extent
    profile = _profile("plate")
    analysis = _shape_analysis(
        (profile,),
        (_confirmed_dimension("plate", DimensionKind.WIDTH, 800.0),),
    )

    # When the document is calculated
    document = calculate_document(analysis)
    geometry = document.cad_entities[0].geometry

    # Then the outline fills the positive quadrant from the origin
    assert isinstance(geometry, PolylineEntity)
    assert min(point.x for point in geometry.points) == 0.0
    assert min(point.y for point in geometry.points) == 0.0
    assert max(point.y for point in geometry.points) == 400.0
