from pathlib import Path

import numpy as np
import pytest

from cadraw.models import DimensionKind, DimensionStatus, TextCandidate
from cadraw.ocr import (
    TesseractOcrBackend,
    discover_tesseract_backend,
    recognize_dimensions,
)


class _FakeOcrBackend:
    def recognize(self, image: np.ndarray) -> tuple[TextCandidate, ...]:
        assert image.shape == (80, 120)
        return (
            TextCandidate(
                text="Ø30",
                bounds=(10.0, 10.0, 35.0, 20.0),
                confidence=0.91,
            ),
            TextCandidate(
                text="150 mm",
                bounds=(50.0, 35.0, 45.0, 20.0),
                confidence=0.88,
            ),
            TextCandidate(
                text="20°",
                bounds=(15.0, 55.0, 30.0, 18.0),
                confidence=0.84,
            ),
        )


def test_ocr_candidates_become_reviewable_dimensions(tmp_path: Path) -> None:
    # Given an OCR backend that returns common engineering labels
    image = np.full((80, 120), 255, dtype=np.uint8)

    # When OCR candidates are parsed at the analysis boundary
    result = recognize_dimensions(image, _FakeOcrBackend(), tmp_path)

    # Then values and semantic kinds are preserved but remain reviewable
    assert result.available
    assert tuple(dimension.kind for dimension in result.dimensions) == (
        DimensionKind.DIAMETER,
        DimensionKind.LENGTH,
        DimensionKind.ANGLE,
    )
    assert all(
        dimension.status is DimensionStatus.NEEDS_REVIEW
        for dimension in result.dimensions
    )
    assert all(not dimension.target_entity_ids for dimension in result.dimensions)


def test_ocr_does_not_fabricate_dimensions_when_backend_is_unavailable(
    tmp_path: Path,
) -> None:
    # Given no OCR backend and a blank image
    image = np.full((40, 40), 255, dtype=np.uint8)

    # When OCR is requested without an engine
    result = recognize_dimensions(image, None, tmp_path)

    # Then the pipeline reports unavailable OCR instead of inventing values
    assert not result.available
    assert result.dimensions == ()
    assert result.candidates == ()


def test_discovery_uses_configured_tesseract_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a local Tesseract executable configured outside PATH
    executable = r"C:\Tools\tesseract.exe"
    monkeypatch.setenv("CADRAW_TESSERACT", executable)

    def fake_which(_executable: str) -> str:
        return executable

    monkeypatch.setattr("cadraw.ocr.shutil.which", fake_which)

    # When the production OCR backend is discovered
    backend = discover_tesseract_backend()

    # Then the configured executable is preserved for real sample runs
    assert isinstance(backend, TesseractOcrBackend)
    assert backend.executable == executable
