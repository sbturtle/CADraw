from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import EntityId, LineCandidate, Point
from cadraw.preprocessing import (
    PreprocessingConfig,
    PreprocessingVariant,
    ScoreContext,
    backdrop_mask,
    excess_ruled_cells,
    illumination_flattened,
    preprocess_variants,
    ruled_cell_count,
    score_candidates,
    write_debug_outputs,
)


def test_preprocessing_variants_preserve_source_and_expose_dark_strokes() -> None:
    # Given an unevenly lit page with a faint structural stroke
    image = np.full((240, 320), 236, dtype=np.uint8)
    image[40:200, :] -= np.arange(160, dtype=np.uint8)[:, None] // 8
    image[118:122, 45:275] = 75
    source_copy = image.copy()

    # When all independent preprocessing variants are generated
    results = preprocess_variants(
        image,
        PreprocessingConfig(
            adaptive_block_size=31,
            morphology_close_size=5,
        ),
    )

    # Then the source remains unchanged and every variant exposes the stroke
    assert np.array_equal(image, source_copy)
    assert tuple(result.variant for result in results) == (
        PreprocessingVariant.GRAYSCALE_ADAPTIVE,
        PreprocessingVariant.CLAHE_ADAPTIVE,
        PreprocessingVariant.SHADOW_REMOVED_ADAPTIVE,
        PreprocessingVariant.BLUR_OTSU,
        PreprocessingVariant.FLATTENED_GLOBAL,
    )
    assert all(result.mask[120, 100] == 255 for result in results)
    assert all(result.mask[20, 20] == 0 for result in results)
    assert all(result.processing_time_ms >= 0.0 for result in results)


def test_debug_outputs_are_opt_in_and_include_each_variant(tmp_path: Path) -> None:
    # Given a small source image and explicitly generated variants
    image = np.full((80, 100), 240, dtype=np.uint8)
    results = preprocess_variants(image)

    # When the caller explicitly requests debug artifacts
    write_debug_outputs(image, results, tmp_path)

    # Then original, enhanced, and mask images are available per variant
    assert (tmp_path / "original.png").exists()
    for variant in PreprocessingVariant:
        assert (tmp_path / f"{variant.value}_enhanced.png").exists()
        assert (tmp_path / f"{variant.value}_mask.png").exists()


def test_global_variant_refuses_faint_ruling_that_adaptive_amplifies() -> None:
    # Given a dark stroke over faint periodic ruling on an unevenly lit page
    image = np.full((320, 320), 248, dtype=np.uint8)
    image[:, :] -= np.arange(320, dtype=np.uint8)[None, :] // 16
    for offset in range(20, 300, 20):
        image[offset, 10:310] = 208
        image[10:310, offset] = 208
    image[150:156, 40:280] = 40

    # When every variant prepares the same page
    masks = {
        result.variant: result.mask
        for result in preprocess_variants(image, PreprocessingConfig())
    }

    # Then the global variant keeps the stroke while dropping the ruling
    global_mask = masks[PreprocessingVariant.FLATTENED_GLOBAL]
    assert global_mask[152, 160] == 255
    assert global_mask[100, 100] == 0
    # And the adaptive variant it was added for promotes that same ruling
    assert masks[PreprocessingVariant.GRAYSCALE_ADAPTIVE][100, 100] == 255


def _cross() -> NDArray[np.uint8]:
    """Return the mask of two crossing strokes."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[198:202, 40:360] = 255
    mask[40:360, 198:202] = 255
    return mask


def _photographed_ruling(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Add a sheet of ruling seen at an angle, its cells shrinking upward."""
    ruled = mask.copy()
    for index in range(-12, 30):
        x = 10.0 + index * 16.0
        _ = cv2.line(ruled, (round(x), 399), (round(200 + (x - 200) * 0.8), 0), 255, 1)
    for index in range(30):
        y = 399.0 - index * (18.0 - index * 0.3)
        _ = cv2.line(ruled, (0, round(y)), (399, round(y - 20)), 255, 1)
    return ruled


def _scored(paper_shape_count: int) -> float:
    """Score one fixed candidate set against a given proved-paper count."""
    mask = _cross()
    candidates = (
        LineCandidate(
            entity_id=EntityId("a"),
            start=Point(40.0, 200.0),
            end=Point(360.0, 200.0),
            confidence=0.9,
        ),
        LineCandidate(
            entity_id=EntityId("b"),
            start=Point(200.0, 40.0),
            end=Point(200.0, 360.0),
            confidence=0.9,
        ),
    )
    return score_candidates(
        PreprocessingVariant.GRAYSCALE_ADAPTIVE,
        candidates,
        mask,
        ScoreContext(
            minimum_line_length=60.0,
            endpoint_tolerance=18.0,
            paper_shape_count=paper_shape_count,
        ),
    ).score


def test_a_form_header_worth_of_proved_paper_does_not_move_the_score() -> None:
    # Given the few printed cells every variant finds on a work-order form
    # Then the score is untouched, so it cannot decide the variant
    assert _scored(8) == _scored(0)


def test_a_ruled_sheet_worth_of_proved_paper_is_penalised() -> None:
    # Given a variant that turned a whole ruled sheet into closed shapes
    # Then it scores well below one that did not
    assert _scored(900) < _scored(0) - 0.2


def test_ruling_photographed_at_an_angle_is_charged_without_a_proof() -> None:
    # Given one variant that kept a sheet's ruling, photographed at an angle
    # so no shape proof recognised its cells, and one that kept the drawing
    ruled = _photographed_ruling(_cross())

    # When the variants' masks are compared
    excess = excess_ruled_cells((ruled, _cross()))

    # Then the ruling variant is charged with its cells, enough to score
    # well below the other
    assert excess[0] > 200
    assert excess[1] == 0
    assert _scored(excess[0]) < _scored(0) - 0.2


def test_a_lattice_every_variant_keeps_is_drawn_and_not_charged() -> None:
    # Given a part whose slots sit so close that their outlines merge into
    # one lattice, which every variant keeps alike
    lattice = _photographed_ruling(_cross())

    # Then no variant is charged for it
    assert excess_ruled_cells((lattice, lattice.copy(), lattice.copy())) == (
        0,
        0,
        0,
    )


def test_a_lattice_a_variant_partly_lost_is_not_charged() -> None:
    # Given a drawn lattice of cells, one variant losing a third of them to
    # a faint stroke
    lattice = _photographed_ruling(_cross())
    partial = lattice.copy()
    partial[:, :130] = 0
    kept, fewer = ruled_cell_count(lattice), ruled_cell_count(partial)
    assert fewer < kept <= 2 * fewer

    # Then the variant that kept them all is not charged for it
    assert excess_ruled_cells((lattice, partial)) == (0, 0)


def test_a_drawing_full_of_holes_is_not_ruling() -> None:
    # Given a plate with six bolt holes and a slot, all enclosed paper
    mask = _cross()
    for index in range(6):
        _ = cv2.circle(mask, (60 + index * 55, 100), 12, 255, 2)
    _ = cv2.rectangle(mask, (80, 280), (160, 310), 255, 2)

    # Then none of its holes faces two others across a single rule
    assert ruled_cell_count(mask) == 0


def test_a_plate_perforated_with_many_equal_slots_is_not_ruling() -> None:
    # Given a 1200 by 900 plate drawn with 132 equal 20 by 50 slots, each
    # traced with its own outline, a few millimetres of paper between them
    mask = np.zeros((1200, 900), dtype=np.uint8)
    for row in range(11):
        for column in range(12):
            left, top = 60 + column * 66, 80 + row * 96
            _ = cv2.rectangle(mask, (left, top), (left + 24, top + 54), 255, 2)

    # Then none of the slots faces another across a single rule
    assert ruled_cell_count(mask) == 0


def test_the_paper_penalty_saturates_instead_of_running_away() -> None:
    # Given two variants that both manufactured hundreds of cells
    # Then the penalty stops growing, so other evidence still decides
    assert _scored(900) == _scored(5000)


def test_the_desk_beside_a_photographed_sheet_is_backdrop() -> None:
    # Given a sheet photographed on a dark desk, half of it in shadow
    image = np.full((800, 600), 40, dtype=np.uint8)
    image[:, 150:] = 220
    image[:, 150:350] = 140
    image[400:404, 200:550] = 30

    mask = backdrop_mask(image)

    # Then only the desk is backdrop; shadowed paper and ink are not
    assert mask is not None
    flags = cast("list[list[bool]]", mask.tolist())
    assert flags[400][50]
    assert not flags[400][250]
    assert not flags[402][450]


def test_a_deep_shadow_fading_into_the_paper_is_not_backdrop() -> None:
    # Given a sheet on a dark desk, its lower right corner under a shadow
    # darker than half the paper's level that fades in over 80 px
    rows = np.arange(800, dtype=np.float64)[:, np.newaxis]
    columns = np.arange(600, dtype=np.float64)[np.newaxis, :]
    depth = np.clip(np.minimum((rows - 400) / 80.0, (columns - 200) / 80.0), 0, 1)
    image = np.asarray(220.0 - 140.0 * depth, dtype=np.uint8)
    image[:, :150] = 40
    image[600:604, 300:550] = 20

    mask = backdrop_mask(image)

    # Then the desk, whose edge is a step, is backdrop; the shadow is paper
    assert mask is not None
    flags = cast("list[list[bool]]", mask.tolist())
    assert flags[200][50]
    assert not flags[650][400]
    assert not flags[602][450]


def test_a_scan_shows_no_backdrop() -> None:
    # Given a full sheet of paper with a drawn rectangle
    image = np.full((800, 600), 245, dtype=np.uint8)
    image[200:204, 100:500] = 30
    image[200:600, 100:104] = 30

    assert backdrop_mask(image) is None


def test_a_repeated_flattening_matches_and_cannot_be_spoiled_by_a_caller() -> None:
    # Given a shaded page flattened once and then scribbled on by its caller
    image = np.full((120, 160), 230, dtype=np.uint8)
    image[:, 80:] -= 40
    image[60:63, 20:140] = 40
    first = illumination_flattened(image)
    expected = first.copy()
    first[:] = 0

    # When the same frame is flattened again
    second = illumination_flattened(image)

    # Then the result is the original one, not the caller's scribble
    assert np.array_equal(second, expected)
    assert not np.shares_memory(second, first)
