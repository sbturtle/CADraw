"""Provider-neutral vision boundary for local and future AI analyzers."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from cadraw.models import DrawingAnalysis
from cadraw.pipeline import PipelineConfig, analyze_image


class VisionAnalyzer(Protocol):
    """Capability required by the application analysis service."""

    def analyze(self, image_path: Path) -> DrawingAnalysis:
        """Analyze one image and return its typed intermediate representation."""
        ...


@dataclass(frozen=True, slots=True)
class MockVisionAnalyzer:
    """Local adapter used before handwriting OCR is introduced."""

    config: PipelineConfig

    def analyze(
        self,
        image_path: Path,
        preprocessing_variant: str | None = None,
        source_region: tuple[int, int, int, int] | None = None,
    ) -> DrawingAnalysis:
        """Run OpenCV detection without a network provider or guessed text.

        A `source_region` supplied here is a reviewer pointing at the drawing,
        so it also switches page-warp detection off. Otherwise a successful
        warp would take precedence over the crop and quietly analyze the whole
        sheet again, which is the opposite of what the reviewer just asked for.
        """
        config = (
            self.config
            if source_region is None
            else replace(
                self.config,
                source_region=source_region,
                perspective_correction=False,
            )
        )
        return analyze_image(
            image_path,
            config,
            preprocessing_variant=preprocessing_variant,
        )
