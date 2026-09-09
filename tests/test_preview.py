from PIL import Image

from cadraw.models import CadDocument, EntityId, LineEntity, Orientation, Point
from cadraw.preview import PreviewConfig, render_preview


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
