from pathlib import Path

from cadraw.export import export_dxf, validate_dxf
from cadraw.models import (
    CadDocument,
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    EntityId,
    LineEntity,
    Orientation,
    Point,
    PolylineEntity,
)
from cadraw.topology import assemble_open_line_chains


def _line(
    entity_id: str,
    start: Point,
    end: Point,
    confidence: float = 0.9,
) -> LineEntity:
    """Build a normalized line fixture."""
    return LineEntity(
        entity_id=EntityId(entity_id),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=Orientation.HORIZONTAL
        if start.y == end.y
        else Orientation.VERTICAL,
        confidence=confidence,
    )


def test_open_chain_becomes_one_stable_ordered_polyline() -> None:
    # Given a connected chain with scrambled input order and directions
    lines = (
        _line("segment_c", Point(150.0, 100.0), Point(100.0, 100.0)),
        _line("segment_b", Point(100.0, 100.0), Point(100.0, 0.0)),
        _line("segment_a", Point(100.0, 0.0), Point(0.0, 0.0)),
    )

    # When export topology assembly runs
    assembled = assemble_open_line_chains(lines)

    # Then the chain is one open polyline from the smaller terminal
    assert len(assembled) == 1
    polyline = assembled[0]
    assert isinstance(polyline, PolylineEntity)
    assert polyline.entity_id == EntityId("polyline_segment_a")
    assert polyline.points == (
        Point(0.0, 0.0),
        Point(100.0, 0.0),
        Point(100.0, 100.0),
        Point(150.0, 100.0),
    )
    assert polyline.closed is False
    assert polyline.confidence == 0.9


def test_branch_cycle_and_near_gap_remain_as_lines() -> None:
    # Given topologies that are ambiguous or not exactly connected
    branch = (
        _line("branch_a", Point(0.0, 0.0), Point(100.0, 0.0)),
        _line("branch_b", Point(100.0, 0.0), Point(100.0, 100.0)),
        _line("branch_c", Point(100.0, 0.0), Point(100.0, -100.0)),
    )
    cycle = (
        _line("cycle_a", Point(0.0, 0.0), Point(100.0, 0.0)),
        _line("cycle_b", Point(100.0, 0.0), Point(100.0, 100.0)),
        _line("cycle_c", Point(100.0, 100.0), Point(0.0, 100.0)),
        _line("cycle_d", Point(0.0, 100.0), Point(0.0, 0.0)),
    )
    near_gap = (
        _line("gap_a", Point(0.0, 0.0), Point(100.0, 0.0)),
        _line("gap_b", Point(100.1, 0.0), Point(100.1, 100.0)),
    )

    # When assembly runs without another proximity snap
    assembled_branch = assemble_open_line_chains(branch)
    assembled_cycle = assemble_open_line_chains(cycle)
    assembled_gap = assemble_open_line_chains(near_gap)

    # Then no branch, closed loop, or guessed gap is changed
    assert assembled_branch == branch
    assert assembled_cycle == cycle
    assert assembled_gap == near_gap


def test_export_uses_polyline_but_keeps_line_dimension_anchor(
    tmp_path: Path,
) -> None:
    # Given a calibrated chain and a dimension targeting its source line
    first = _line("line_a", Point(0.0, 0.0), Point(100.0, 0.0))
    second = _line("line_b", Point(100.0, 0.0), Point(100.0, 50.0))
    document = CadDocument(
        unit="mm",
        entities=(first, second),
        dimensions=(
            Dimension(
                dimension_id=DimensionId("dimension_a"),
                raw_text="100",
                kind=DimensionKind.LENGTH,
                parsed_value=100.0,
                unit="mm",
                target_entity_ids=(EntityId("line_a"),),
                confidence=1.0,
                status=DimensionStatus.CONFIRMED,
            ),
        ),
        warnings=(),
        scale_mm_per_source_unit=1.0,
    )
    output_path = tmp_path / "chain.dxf"

    # When the real DXF exporter serializes and reloads the document
    summary = export_dxf(document, output_path)
    validation = validate_dxf(output_path)

    # Then one geometry entity contains the ordered open path and its label survives
    assert summary.geometry_entity_count == 1
    assert summary.dimension_text_count == 1
    assert validation.count("LINE") == 0
    assert validation.count("LWPOLYLINE") == 1
    assert validation.count("TEXT") == 1
    assert validation.polyline_layers == ("GEOMETRY",)
    assert validation.polyline_closed == (False,)
    assert validation.polyline_points == (
        (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 50.0),
        ),
    )
