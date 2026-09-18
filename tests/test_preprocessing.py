from pathlib import Path

import numpy as np

from cadraw.preprocessing import (
    PreprocessingConfig,
    PreprocessingVariant,
    preprocess_variants,
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
