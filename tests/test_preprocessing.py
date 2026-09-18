from pathlib import Path

import numpy as np

from cadraw.models import EntityId, LineCandidate, Point
from cadraw.preprocessing import (
    PreprocessingConfig,
    PreprocessingVariant,
    ScoreContext,
    preprocess_variants,
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


def _scored(paper_shape_count: int) -> float:
    """Score one fixed candidate set against a given proved-paper count."""
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[198:202, 40:360] = 255
    mask[40:360, 198:202] = 255
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


def test_the_paper_penalty_saturates_instead_of_running_away() -> None:
    # Given two variants that both manufactured hundreds of cells
    # Then the penalty stops growing, so other evidence still decides
    assert _scored(900) == _scored(5000)
