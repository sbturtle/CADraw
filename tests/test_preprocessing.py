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
