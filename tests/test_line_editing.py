import pytest

from cadraw import line_editing
from cadraw.models import (
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
    confidence: float = 0.9,
) -> LineEntity:
    """Build a normalized line fixture."""
    orientation = Orientation.HORIZONTAL if start.y == end.y else Orientation.VERTICAL
    return LineEntity(
        entity_id=EntityId(entity_id),
        raw_start=start,
        raw_end=end,
        start=start,
        end=end,
        orientation=orientation,
        confidence=confidence,
    )


def _dimension(entity_id: str, target_id: str) -> Dimension:
    """Build an unresolved manual dimension fixture."""
    return Dimension(
        dimension_id=DimensionId(entity_id),
        raw_text="",
        kind=DimensionKind.LENGTH,
        parsed_value=None,
        unit="mm",
        target_entity_ids=(EntityId(target_id),),
        confidence=0.0,
        status=DimensionStatus.UNKNOWN,
    )


def _analysis(
    lines: tuple[LineEntity, ...],
    dimensions: tuple[Dimension, ...] = (),
) -> DrawingAnalysis:
    """Build an analysis fixture with real connection derivation."""
    return DrawingAnalysis(
        unit="mm",
        image_size=(320, 240),
        entities=lines,
        dimensions=dimensions,
        connections=build_connections(lines, 18.0),
        warnings=(),
    )


def test_merge_selected_lines_collapses_split_geometry_and_dimensions() -> None:
    # Given two near-collinear pieces with unresolved manual dimensions
    first = _line("line_a", Point(0.0, 10.0), Point(50.0, 10.0))
    second = _line("line_b", Point(54.0, 11.0), Point(100.0, 11.0))
    analysis = _analysis(
        (first, second),
        (_dimension("dimension_a", "line_a"), _dimension("dimension_b", "line_b")),
    )

    # When the user merges both selected source lines
    edited = line_editing.merge_selected_lines(
        analysis,
        (EntityId("line_a"), EntityId("line_b")),
    )

    # Then one canonical line and one valid dimension remain
    assert tuple(entity.entity_id for entity in edited.entities) == (
        EntityId("line_a"),
    )
    merged = edited.entities[0]
    assert isinstance(merged, LineEntity)
    assert merged.length == 100.0
    assert len(edited.dimensions) == 1
    assert edited.dimensions[0].target_entity_ids == (EntityId("line_a"),)


def test_delete_selected_lines_removes_geometry_and_its_manual_rows() -> None:
    # Given connected lines with one manual row per line
    first = _line("line_a", Point(0.0, 0.0), Point(100.0, 0.0))
    second = _line("line_b", Point(100.0, 0.0), Point(100.0, 80.0))
    analysis = _analysis(
        (first, second),
        (_dimension("dimension_a", "line_a"), _dimension("dimension_b", "line_b")),
    )

    # When the user deletes the second line
    edited = line_editing.delete_selected_lines(
        analysis,
        (EntityId("line_b"),),
    )

    # Then its geometry, row, and connection are gone
    assert tuple(entity.entity_id for entity in edited.entities) == (
        EntityId("line_a"),
    )
    assert tuple(dimension.dimension_id for dimension in edited.dimensions) == (
        DimensionId("dimension_a"),
    )
    assert edited.connections == ()


def test_merge_selected_lines_refuses_non_collinear_geometry() -> None:
    # Given selected lines that do not describe one straight source stroke
    first = _line("line_a", Point(0.0, 0.0), Point(100.0, 0.0))
    second = _line("line_b", Point(0.0, 80.0), Point(0.0, 160.0))
    analysis = _analysis(
        (first, second),
        (_dimension("dimension_a", "line_a"), _dimension("dimension_b", "line_b")),
    )

    # When the user tries to merge the unrelated lines
    with pytest.raises(line_editing.LineEditError) as error:
        _ = line_editing.merge_selected_lines(
            analysis,
            (EntityId("line_a"), EntityId("line_b")),
        )

    # Then the unsafe geometry edit is refused
    assert error.value.code is line_editing.LineEditErrorCode.NOT_MERGEABLE


def test_redraw_selected_line_replaces_id_and_preserves_source_points() -> None:
    # Given one selected line with a dimension anchor
    original = _line("line_a", Point(0.0, 0.0), Point(50.0, 0.0))
    analysis = _analysis(
        (original,),
        (_dimension("dimension_a", "line_a"),),
    )

    # When the user redraws it from two source-image points
    edited = line_editing.redraw_selected_line(
        analysis,
        (EntityId("line_a"),),
        Point(10.0, 20.0),
        Point(110.0, 20.0),
    )

    # Then the same identity is replaced with exact source provenance
    replacement = edited.entities[0]
    assert isinstance(replacement, LineEntity)
    assert replacement.entity_id == EntityId("line_a")
    assert replacement.raw_start == Point(10.0, 20.0)
    assert replacement.raw_end == Point(110.0, 20.0)
    assert replacement.start == Point(10.0, 20.0)
    assert replacement.end == Point(110.0, 20.0)
    assert edited.dimensions[0].target_entity_ids == (EntityId("line_a"),)


def test_redraw_selected_lines_collapses_disconnected_pieces_into_one() -> None:
    # Given three detection fragments too misaligned for a collinear merge
    first = _line("line_a", Point(0.0, 0.0), Point(50.0, 0.0))
    second = _line("line_b", Point(52.0, 6.0), Point(100.0, 6.0))
    third = _line("line_c", Point(101.0, 40.0), Point(150.0, 40.0))
    analysis = _analysis(
        (first, second, third),
        (
            _dimension("dimension_a", "line_a"),
            _dimension("dimension_b", "line_b"),
            _dimension("dimension_c", "line_c"),
        ),
    )
    with pytest.raises(line_editing.LineEditError):
        _ = line_editing.merge_selected_lines(
            analysis,
            (EntityId("line_a"), EntityId("line_b"), EntityId("line_c")),
        )

    # When the user redraws over all three as one continuous stroke
    edited = line_editing.redraw_selected_line(
        analysis,
        (EntityId("line_a"), EntityId("line_b"), EntityId("line_c")),
        Point(0.0, 0.0),
        Point(150.0, 0.0),
    )

    # Then one canonical line and one unresolved dimension row remain
    assert tuple(entity.entity_id for entity in edited.entities) == (
        EntityId("line_a"),
    )
    replacement = edited.entities[0]
    assert isinstance(replacement, LineEntity)
    assert replacement.raw_start == Point(0.0, 0.0)
    assert replacement.raw_end == Point(150.0, 0.0)
    assert len(edited.dimensions) == 1
    assert edited.dimensions[0].target_entity_ids == (EntityId("line_a"),)


def test_redraw_selected_lines_refuses_when_a_target_dimension_is_confirmed() -> None:
    # Given two split fragments where one already has a confirmed length
    first = _line("line_a", Point(0.0, 0.0), Point(50.0, 0.0))
    second = _line("line_b", Point(52.0, 6.0), Point(100.0, 6.0))
    confirmed = Dimension(
        dimension_id=DimensionId("dimension_b"),
        raw_text="100",
        kind=DimensionKind.LENGTH,
        parsed_value=100.0,
        unit="mm",
        target_entity_ids=(EntityId("line_b"),),
        confidence=1.0,
        status=DimensionStatus.CONFIRMED,
    )
    analysis = _analysis((first, second), (confirmed,))

    # When the user tries to redraw over both selected fragments
    with pytest.raises(line_editing.LineEditError) as error:
        _ = line_editing.redraw_selected_line(
            analysis,
            (EntityId("line_a"), EntityId("line_b")),
            Point(0.0, 0.0),
            Point(100.0, 0.0),
        )

    # Then the confirmed physical value is protected from silent loss
    assert error.value.code is line_editing.LineEditErrorCode.CONFIRMED_DIMENSION


def test_redraw_without_selection_adds_a_manual_line_and_dimension() -> None:
    # Given an analysis whose detected line IDs do not include a manual ID
    original = _line("line_a", Point(0.0, 0.0), Point(50.0, 0.0))
    analysis = _analysis((original,), (_dimension("dimension_a", "line_a"),))

    # When the user draws a new source line without selecting a row
    edited = line_editing.redraw_selected_line(
        analysis,
        (),
        Point(10.0, 20.0),
        Point(110.0, 20.0),
    )

    # Then a collision-free manual identity and unresolved row are added
    assert tuple(entity.entity_id for entity in edited.entities) == (
        EntityId("line_a"),
        EntityId("manual_line_001"),
    )
    assert edited.dimensions[-1].target_entity_ids == (EntityId("manual_line_001"),)


def test_redraw_selected_line_rejects_a_stroke_shorter_than_minimum() -> None:
    # Given an empty analysis and a zero-length manual stroke
    analysis = _analysis(())

    # When the user releases without drawing a usable line
    with pytest.raises(line_editing.LineEditError) as error:
        _ = line_editing.redraw_selected_line(
            analysis,
            (),
            Point(10.0, 20.0),
            Point(10.0, 20.0),
        )

    # Then the source model remains protected by the minimum length rule
    assert error.value.code is line_editing.LineEditErrorCode.ZERO_LENGTH


def test_editing_confirmed_lines_raises_a_typed_conflict() -> None:
    # Given a selected line whose physical value is already confirmed
    line = _line("line_a", Point(0.0, 0.0), Point(100.0, 0.0))
    confirmed = Dimension(
        dimension_id=DimensionId("dimension_a"),
        raw_text="100",
        kind=DimensionKind.LENGTH,
        parsed_value=100.0,
        unit="mm",
        target_entity_ids=(EntityId("line_a"),),
        confidence=1.0,
        status=DimensionStatus.CONFIRMED,
    )
    analysis = _analysis((line,), (confirmed,))

    # When a destructive geometry edit would discard that physical value
    with pytest.raises(line_editing.LineEditError) as error:
        _ = line_editing.delete_selected_lines(
            analysis,
            (EntityId("line_a"),),
        )

    # Then the operation is refused with a machine-readable conflict code
    assert error.value.code is line_editing.LineEditErrorCode.CONFIRMED_DIMENSION
