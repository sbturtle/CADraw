import cv2
import numpy as np

from cadraw.models import (
    ArcCandidate,
    CircleCandidate,
    EntityId,
    LineCandidate,
    Point,
    PolylineCandidate,
    TextCandidate,
)
from cadraw.text_filter import (
    TextStrokeConfig,
    proved_text_regions,
    without_text_strokes,
)


def _label(text: str, confidence: float = 0.95) -> TextCandidate:
    return TextCandidate(
        text=text,
        bounds=(100.0, 100.0, 80.0, 40.0),
        confidence=confidence,
        quad=(
            Point(100.0, 100.0),
            Point(180.0, 100.0),
            Point(180.0, 140.0),
            Point(100.0, 140.0),
        ),
    )


def _mask_with(*strokes: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((300, 400), dtype=np.uint8)
    for start, end in strokes:
        _ = cv2.line(mask, start, end, 255, 3)
    return mask


def _rings_at(*centres: float, joined: bool) -> np.ndarray:
    """Draw small rings on the label's row, crossed into one stroke or apart."""
    mask = _mask_with()
    for x in centres:
        _ = cv2.circle(mask, (round(x), 120), 4, 255, 1)
    if joined:
        _ = cv2.line(mask, (round(centres[0]), 120), (round(centres[-1]), 120), 255, 1)
    return mask


def _line(identifier: str, start: Point, end: Point) -> LineCandidate:
    return LineCandidate(EntityId(identifier), start, end, 0.8)


def test_a_stroke_whose_ink_stays_inside_a_digit_label_is_removed() -> None:
    # Given the stroke of a written `100` that touches nothing else
    mask = _mask_with(((110, 110), (110, 135)))
    stroke = _line("glyph", Point(110.0, 110.0), Point(110.0, 135.0))

    # When candidates are checked against the proved label
    removal = without_text_strokes(
        (stroke,),
        (),
        (_label("100"),),
        mask,
        TextStrokeConfig(),
    )

    # Then the glyph is no longer offered as geometry
    assert removal.lines == ()
    assert removal.removed_ids == (EntityId("glyph"),)
    assert removal.region_count == 1


def test_a_stroke_joined_to_the_outline_is_kept_inside_a_label() -> None:
    # Given a short edge inside the label box whose ink runs on to the part
    mask = _mask_with(((120, 120), (160, 120)), ((160, 120), (380, 120)))
    fragment = _line("edge", Point(120.0, 120.0), Point(160.0, 120.0))

    # When candidates are checked against the proved label
    removal = without_text_strokes(
        (fragment,),
        (),
        (_label("100"),),
        mask,
        TextStrokeConfig(),
    )

    # Then the shared ink component proves it is part of the drawing
    assert removal.lines == (fragment,)
    assert removal.removed_ids == ()


def test_a_single_character_or_uncertain_box_proves_nothing() -> None:
    # Given a hole read as `0` and a label read with low confidence
    labels = (_label("0"), _label("100", confidence=0.3), _label("口引"))

    # When proof regions are derived
    regions = proved_text_regions(labels, TextStrokeConfig())

    # Then none of them may remove geometry
    assert regions == ()


def test_holes_survive_while_letter_blobs_and_arcs_are_removed() -> None:
    # Given a hole, a closed glyph contour, and a glyph curve in one label
    mask = _mask_with(((140, 105), (170, 105)), ((140, 130), (170, 130)))
    _ = cv2.circle(mask, (115, 120), 6, 255, 2)
    hole = CircleCandidate(EntityId("hole"), Point(115.0, 120.0), 6.0, 0.9)
    blob = PolylineCandidate(
        EntityId("blob"),
        (Point(140.0, 105.0), Point(170.0, 105.0), Point(170.0, 106.0)),
        closed=True,
        confidence=0.7,
    )
    curve = ArcCandidate(
        EntityId("curve"),
        Point(155.0, 130.0),
        15.0,
        start_angle_degrees=180.0,
        end_angle_degrees=360.0,
        confidence=0.6,
    )
    _ = cv2.ellipse(mask, (155, 130), (15, 15), 0.0, 180.0, 360.0, 255, 2)

    # When shapes are checked against the proved label
    removal = without_text_strokes(
        (),
        (hole, blob, curve),
        (_label("10←Ø14"),),
        mask,
        TextStrokeConfig(),
    )

    # Then only the drilled hole stays geometry
    assert removal.shapes == (hole,)
    assert set(removal.removed_ids) == {EntityId("blob"), EntityId("curve")}


def test_a_loop_past_the_leading_glyph_is_a_digit_not_a_hole() -> None:
    # Given the loop of the `0` in a written `100`, found as a small circle
    loop = CircleCandidate(EntityId("loop"), Point(165.0, 120.0), 6.0, 0.9)
    leading = CircleCandidate(EntityId("hole"), Point(112.0, 120.0), 6.0, 0.9)

    # When shapes are checked against the proved label
    removal = without_text_strokes(
        (),
        (loop, leading),
        (_label("100"),),
        _mask_with(),
        TextStrokeConfig(),
    )

    # Then the trailing loop goes and the leading glyph, a possible hole, stays
    assert removal.shapes == (leading,)
    assert removal.removed_ids == (EntityId("loop"),)


def test_a_hole_read_as_a_letter_after_the_label_is_kept() -> None:
    # Given a hole drawn after its size, read together with it as `14O`
    hole = CircleCandidate(EntityId("hole"), Point(165.0, 120.0), 6.0, 0.9)

    # When shapes are checked against the label
    removal = without_text_strokes(
        (),
        (hole,),
        (_label("14O"),),
        _mask_with(),
        TextStrokeConfig(),
    )

    # Then nothing in the reading accounts for a loop there, so it stays
    assert removal.shapes == (hole,)


def test_only_as_many_loops_as_the_digit_has_are_removed() -> None:
    # Given `10` and two small circles both in the share of its `0`
    circles = tuple(
        CircleCandidate(EntityId(f"circle_{index}"), Point(x, 120.0), 5.0, 0.9)
        for index, x in enumerate((145.0, 170.0))
    )

    # When shapes are checked against the label
    removal = without_text_strokes(
        (),
        circles,
        (_label("10"),),
        _mask_with(),
        TextStrokeConfig(),
    )

    # Then the one loop a zero has goes, the one nearest its share's middle
    assert removal.shapes == (circles[0],)


def test_a_loop_closed_across_its_own_stroke_goes_as_one_loop() -> None:
    # Given a `9` in `149` whose loop crossed itself into two touching circles,
    # one of them just past the end of the box OCR drew
    halves = tuple(
        CircleCandidate(EntityId(f"half_{index}"), Point(x, 120.0), 5.0, 0.9)
        for index, x in enumerate((174.0, 184.0))
    )

    # When shapes are checked against the label
    removal = without_text_strokes(
        (),
        halves,
        (_label("149"),),
        _rings_at(174.0, 184.0, joined=True),
        TextStrokeConfig(),
    )

    # Then both halves go as the nine's one loop
    assert removal.shapes == ()


def test_a_hole_drawn_just_after_the_last_digit_is_kept() -> None:
    # Given `100` with a hole drawn apart from it just past the box, its ring
    # read close enough to touch the last zero's loop
    zero = CircleCandidate(EntityId("zero"), Point(174.0, 120.0), 5.0, 0.9)
    hole = CircleCandidate(EntityId("hole"), Point(184.0, 120.0), 5.0, 0.9)

    # When shapes are checked against the label
    removal = without_text_strokes(
        (),
        (zero, hole),
        (_label("100"),),
        _rings_at(174.0, 184.0, joined=False),
        TextStrokeConfig(),
    )

    # Then the zero's loop goes and the hole, a stroke of its own, stays
    assert removal.shapes == (hole,)


def test_a_hole_touching_a_digit_loop_in_the_next_share_is_kept() -> None:
    # Given `100O` whose last zero's loop touches the hole read as `O`
    zero = CircleCandidate(EntityId("zero"), Point(155.0, 120.0), 5.0, 0.9)
    hole = CircleCandidate(EntityId("hole"), Point(166.0, 120.0), 5.0, 0.9)

    # When shapes are checked against the label
    removal = without_text_strokes(
        (),
        (zero, hole),
        (_label("100O"),),
        _mask_with(),
        TextStrokeConfig(),
    )

    # Then the zero's loop goes and the hole in the `O` share stays
    assert removal.shapes == (hole,)


def test_a_hole_read_after_a_number_keeps_its_place() -> None:
    # Given `100` and a hole drawn right after it, read together as `100O`
    zeros = tuple(
        CircleCandidate(EntityId(f"zero_{index}"), Point(x, 120.0), 5.0, 0.9)
        for index, x in enumerate((130.0, 150.0))
    )
    hole = CircleCandidate(EntityId("hole"), Point(170.0, 120.0), 5.0, 0.9)

    # When shapes are checked against the label
    removal = without_text_strokes(
        (),
        (*zeros, hole),
        (_label("100O"),),
        _mask_with(),
        TextStrokeConfig(),
    )

    # Then the zeros' loops go and the hole in the `O` share stays
    assert removal.shapes == (hole,)


def test_a_circle_under_an_uncertain_label_is_kept() -> None:
    # Given the same loop under a label read with low confidence
    loop = CircleCandidate(EntityId("loop"), Point(165.0, 120.0), 6.0, 0.9)

    # When shapes are checked
    removal = without_text_strokes(
        (),
        (loop,),
        (_label("100", confidence=0.3),),
        _mask_with(),
        TextStrokeConfig(),
    )

    # Then nothing is removed on an unproved reading
    assert removal.shapes == (loop,)


def test_disabled_filter_or_missing_ink_keeps_everything() -> None:
    # Given a stroke inside a label but no ink under it in the mask
    stroke = _line("phantom", Point(110.0, 110.0), Point(110.0, 135.0))
    empty_mask = np.zeros((300, 400), dtype=np.uint8)

    # When the filter runs without ink evidence, and then disabled
    without_ink = without_text_strokes(
        (stroke,),
        (),
        (_label("100"),),
        empty_mask,
        TextStrokeConfig(),
    )
    disabled = without_text_strokes(
        (stroke,),
        (),
        (_label("100"),),
        _mask_with(((110, 110), (110, 135))),
        TextStrokeConfig(enabled=False),
    )

    # Then nothing is removed on an unproved basis
    assert without_ink.lines == (stroke,)
    assert disabled.lines == (stroke,)
