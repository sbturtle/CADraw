"""Provider-neutral vision boundary for local and future AI analyzers."""

from dataclasses import dataclass
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
    ) -> DrawingAnalysis:
        """Run OpenCV detection without a network provider or guessed text."""
        return analyze_image(
            image_path,
            self.config,
            preprocessing_variant=preprocessing_variant,
        )
