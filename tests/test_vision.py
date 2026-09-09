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
