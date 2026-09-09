from pathlib import Path

from cadraw.export import export_dxf, validate_dxf
from cadraw.models import (
    ArcEntity,
    CadDocument,
    CircleEntity,
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


def test_dxf_export_round_trips_supported_geometry(tmp_path: Path) -> None:
    # Given a document containing the MVP geometry families and one known dimension
    line = LineEntity(
        entity_id=EntityId("line_1"),
        raw_start=Point(0.0, 0.0),
        raw_end=Point(100.0, 0.0),
        start=Point(0.0, 0.0),
        end=Point(100.0, 0.0),
        orientation=Orientation.HORIZONTAL,
        confidence=1.0,
    )
    document = CadDocument(
        unit="mm",
        entities=(
            line,
            PolylineEntity(
                entity_id=EntityId("poly_1"),
                points=(Point(0.0, 0.0), Point(20.0, 20.0), Point(40.0, 0.0)),
                closed=False,
                confidence=0.9,
            ),
            CircleEntity(
                entity_id=EntityId("circle_1"),
                center=Point(50.0, 50.0),
                radius=10.0,
                confidence=0.9,
            ),
            ArcEntity(
                entity_id=EntityId("arc_1"),
                center=Point(80.0, 50.0),
                radius=12.0,
                start_angle_degrees=0.0,
                end_angle_degrees=90.0,
                confidence=0.9,
            ),
        ),
        dimensions=(
            Dimension(
                dimension_id=DimensionId("dim_1"),
                raw_text="100",
                kind=DimensionKind.LENGTH,
                parsed_value=100.0,
                unit="mm",
                target_entity_ids=(EntityId("line_1"),),
                confidence=1.0,
                status=DimensionStatus.CONFIRMED,
            ),
        ),
        warnings=(),
        scale_mm_per_source_unit=1.0,
    )
    output_path = tmp_path / "drawing.dxf"

    # When the real ezdxf exporter writes and reloads the file
    summary = export_dxf(document, output_path)
    validation = validate_dxf(output_path)

    # Then each geometry type survives as a concrete DXF entity
    assert summary.geometry_entity_count == 4
    assert validation.count("LINE") == 1
    assert validation.count("LWPOLYLINE") == 1
    assert validation.count("CIRCLE") == 1
    assert validation.count("ARC") == 1
    assert validation.count("TEXT") == 1
    assert validation.total_entities == 5


def test_unknown_dimension_is_not_exported_as_misleading_text(tmp_path: Path) -> None:
    # Given geometry with an unassociated unknown dimension
    document = CadDocument(
        unit="mm",
        entities=(),
        dimensions=(
            Dimension(
                dimension_id=DimensionId("dim_unknown"),
                raw_text="?",
                kind=DimensionKind.UNKNOWN,
                parsed_value=None,
                unit="mm",
                target_entity_ids=(),
                confidence=0.2,
                status=DimensionStatus.UNKNOWN,
            ),
        ),
        warnings=(),
        scale_mm_per_source_unit=None,
    )
    output_path = tmp_path / "unknown.dxf"

    # When the document is exported and reloaded
    _ = export_dxf(document, output_path)
    validation = validate_dxf(output_path)

    # Then no zero or placeholder dimension text is written
    assert validation.total_entities == 0
    assert validation.count("TEXT") == 0
