from pathlib import Path

import numpy as np
from PIL import Image

from cadraw.pipeline import PipelineConfig
from cadraw.vision import MockVisionAnalyzer


def test_mock_vision_adapter_stays_provider_independent(tmp_path: Path) -> None:
    # Given a simple source image and the local pipeline configuration
    image = np.full((180, 240), 255, dtype=np.uint8)
    image[80:84, 30:210] = 0
    image_path = tmp_path / "source.png"
    Image.fromarray(image).save(image_path)

    # When the mock adapter analyzes the image
    analysis = MockVisionAnalyzer(PipelineConfig.from_detection_defaults()).analyze(
        image_path,
    )

    # Then it returns local geometry without pretending to have OCR
    assert analysis.entities
    assert analysis.dimensions == ()


def test_an_explicit_region_is_not_overridden_by_page_warping(
    tmp_path: Path,
) -> None:
    # Given a sheet with a clear page border that warping would latch onto
    image = np.full((900, 700), 250, dtype=np.uint8)
    image[20:24, 20:680] = 0
    image[876:880, 20:680] = 0
    image[20:880, 20:24] = 0
    image[20:880, 676:680] = 0
    image[400:405, 150:550] = 0
    image_path = tmp_path / "bordered.png"
    Image.fromarray(image).save(image_path)
    analyzer = MockVisionAnalyzer(PipelineConfig.from_detection_defaults())

    # When a reviewer names the drawing region explicitly
    analysis = analyzer.analyze(image_path, source_region=(120, 320, 460, 300))

    # Then the crop is what ran, instead of a warp of the whole sheet
    assert analysis.source_region == (120, 320, 460, 300)
    assert analysis.perspective_transform is None
