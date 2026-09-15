from PIL import Image

from cadraw.models import (
    CadDocument,
    DetectedEntity,
    DrawingAnalysis,
    EntityId,
    LineEntity,
    Orientation,
    Point,
)
from cadraw.preview import (
    PreviewConfig,
    preview_mapper,
    render_foreground_overlay,
    render_preview,
)
from cadraw.preview_selection import PreviewSelectionWindow, select_preview_entities


def _selection_document() -> CadDocument:
    """Build two separated lines for window and crossing selection tests."""
    return CadDocument(
        unit="mm",
        entities=(
            LineEntity(
                entity_id=EntityId("line_a"),
                raw_start=Point(10.0, 10.0),
                raw_end=Point(30.0, 10.0),
                start=Point(10.0, 10.0),
                end=Point(30.0, 10.0),
                orientation=Orientation.HORIZONTAL,
                confidence=0.95,
            ),
            LineEntity(
                entity_id=EntityId("line_b"),
                raw_start=Point(40.0, 10.0),
                raw_end=Point(60.0, 10.0),
                start=Point(40.0, 10.0),
                end=Point(60.0, 10.0),
                orientation=Orientation.HORIZONTAL,
                confidence=0.95,
            ),
        ),
        dimensions=(),
        warnings=(),
        scale_mm_per_source_unit=1.0,
    )


def test_foreground_overlay_keeps_only_source_geometry_on_transparent_canvas() -> None:
    # Given accepted source-space geometry and unrelated source pixels
    line = LineEntity(
        entity_id=EntityId("line_1"),
        raw_start=Point(10.0, 20.0),
        raw_end=Point(90.0, 20.0),
        start=Point(10.0, 20.0),
        end=Point(90.0, 20.0),
        orientation=Orientation.HORIZONTAL,
        confidence=0.95,
    )
    analysis = DrawingAnalysis(
        unit="mm",
        image_size=(100, 60),
        entities=(line,),
        dimensions=(),
        connections=(),
        warnings=(),
        detected_entities=(
            DetectedEntity(
                entity_id=line.entity_id,
                geometry=line,
                confidence=line.confidence,
            ),
        ),
    )

    # When the explicit foreground overlay is rendered
    image = render_foreground_overlay(analysis)

    # Then only the accepted stroke is opaque and the background is removed
    assert image.mode == "RGBA"
    alpha = image.getchannel("A")
    assert alpha.getpixel((5, 5)) == 0
    assert alpha.getpixel((50, 20)) == 255


def test_preview_renders_geometry_inside_requested_frame() -> None:
    # Given a document with one normalized line
    document = CadDocument(
        unit="mm",
        entities=(
            LineEntity(
                entity_id=EntityId("line_1"),
                raw_start=Point(0.0, 0.0),
                raw_end=Point(100.0, 0.0),
                start=Point(0.0, 0.0),
                end=Point(100.0, 0.0),
                orientation=Orientation.HORIZONTAL,
                confidence=0.95,
            ),
        ),
        dimensions=(),
        warnings=(),
        scale_mm_per_source_unit=1.0,
    )

    # When the deterministic preview renderer runs
    preview = render_preview(document, PreviewConfig(width=240, height=160))

    # Then it returns a real image with visible non-background geometry
    assert isinstance(preview, Image.Image)
    assert preview.size == (240, 160)
    colors = preview.getcolors(maxcolors=240 * 160)
    assert colors is not None
    assert len(colors) > 1


def test_preview_highlights_selected_geometry_only() -> None:
    # Given a document with one selectable source line
    entity_id = EntityId("line_1")
    document = CadDocument(
        unit="mm",
        entities=(
            LineEntity(
                entity_id=entity_id,
                raw_start=Point(0.0, 0.0),
                raw_end=Point(100.0, 0.0),
                start=Point(0.0, 0.0),
                end=Point(100.0, 0.0),
                orientation=Orientation.HORIZONTAL,
                confidence=0.95,
            ),
        ),
        dimensions=(),
        warnings=(),
        scale_mm_per_source_unit=1.0,
    )

    # When the preview renders with and without that source id selected
    plain = render_preview(document, PreviewConfig(width=240, height=160))
    highlighted = render_preview(
        document,
        PreviewConfig(
            width=240,
            height=160,
            highlight_entity_ids=(entity_id,),
        ),
    )
    plain_colors = plain.getcolors(maxcolors=240 * 160)
    highlighted_colors = highlighted.getcolors(maxcolors=240 * 160)

    # Then only the selected render contains the reserved selection accent
    assert plain_colors is not None
    assert highlighted_colors is not None
    assert (217, 119, 87) not in {color for _, color in plain_colors}
    assert (217, 119, 87) in {color for _, color in highlighted_colors}


def test_cad_preview_inverts_y_axis_without_changing_source_preview() -> None:
    # Given one normalized drawing bound in source/image coordinates
    bounds = (0.0, 0.0, 100.0, 100.0)
    source_mapper, _ = preview_mapper(
        PreviewConfig(width=240, height=160),
        bounds,
    )
    cad_mapper, _ = preview_mapper(
        PreviewConfig(width=240, height=160, invert_y=True),
        bounds,
    )

    # When the same endpoints are mapped for each coordinate-space preview
    source_bottom = source_mapper(Point(0.0, 100.0))[1]
    source_top = source_mapper(Point(0.0, 0.0))[1]
    cad_bottom = cad_mapper(Point(0.0, 100.0))[1]
    cad_top = cad_mapper(Point(0.0, 0.0))[1]

    # Then CAD-space is displayed in y-up order while source stays y-down
    assert source_top < source_bottom
    assert cad_top > cad_bottom


def test_left_to_right_window_selection_requires_full_line_containment() -> None:
    # Given two separated lines in the rendered preview coordinate space
    document = _selection_document()

    # When a left-to-right window fully contains only the first line
    selected = select_preview_entities(
        document,
        PreviewConfig(width=240, height=160),
        PreviewSelectionWindow(
            start=Point(30.0, 70.0),
            end=Point(55.0, 90.0),
            left_to_right=True,
        ),
    )

    # Then only the fully enclosed line is selected
    assert selected == (EntityId("line_a"),)


def test_right_to_left_crossing_selection_accepts_partial_intersection() -> None:
    # Given two separated lines in the rendered preview coordinate space
    document = _selection_document()

    # When a right-to-left crossing window touches both lines
    selected = select_preview_entities(
        document,
        PreviewConfig(width=240, height=160),
        PreviewSelectionWindow(
            start=Point(75.0, 90.0),
            end=Point(45.0, 70.0),
            left_to_right=False,
        ),
    )

    # Then both partially or fully intersecting lines are selected
    assert selected == (EntityId("line_a"), EntityId("line_b"))
