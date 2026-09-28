from pathlib import Path

import numpy as np
import pytest

from cadraw.models import DimensionKind, DimensionStatus, TextCandidate
from cadraw.ocr import (
    OcrUnavailableError,
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


class _ListOcrBackend:
    def __init__(self, *texts: str) -> None:
        self._texts: tuple[str, ...] = texts

    def recognize(self, image: np.ndarray) -> tuple[TextCandidate, ...]:
        del image
        return tuple(
            TextCandidate(
                text=text,
                bounds=(10.0 + 30.0 * index, 10.0, 25.0, 15.0),
                confidence=0.9,
            )
            for index, text in enumerate(self._texts)
        )


class _ScriptedReader:
    def __init__(self, *answers: str | None) -> None:
        self._answers: list[str | None] = list(answers)
        self.crops: list[tuple[int, ...]] = []

    def read(self, crop: np.ndarray) -> str | None:
        self.crops.append(crop.shape)
        return self._answers.pop(0)


class _UnreachableReader:
    def __init__(self) -> None:
        self.calls: int = 0

    def read(self, crop: np.ndarray) -> str | None:
        del crop
        self.calls += 1
        message = "down"
        raise OcrUnavailableError(message)


def test_prices_dates_grouping_and_lone_glyphs_never_become_dimensions() -> None:
    # Given readings that are prices, dates, grouped thousands, or one glyph
    image = np.full((60, 400), 255, dtype=np.uint8)
    backend = _ListOcrBackend("100.000", "2021.319", "1,050", "1", "150", "R5")

    # When OCR candidates are parsed
    result = recognize_dimensions(image, backend)

    # Then only plausible handwritten sizes are suggested, all text is kept
    assert tuple(dimension.raw_text for dimension in result.dimensions) == (
        "150",
        "R5",
    )
    assert len(result.candidates) == 6


def test_a_value_both_readers_agree_on_stays_reviewable() -> None:
    # Given a second reader that sees the same value
    image = np.full((60, 200), 255, dtype=np.uint8)
    reader = _ScriptedReader("150")

    # When OCR runs with the cross-check
    result = recognize_dimensions(image, _ListOcrBackend("150"), second_reader=reader)

    # Then the value keeps its normal review state and the second reading
    dimension = result.dimensions[0]
    assert dimension.status is DimensionStatus.NEEDS_REVIEW
    assert result.dimension_sources[0].alternate_text == "150"
    assert result.candidates[0].alternate_text == "150"
    assert result.second_reader_available is True
    assert reader.crops


def test_a_value_the_second_reader_disagrees_with_is_marked_uncertain() -> None:
    # Given PP-OCR reading `790Ø` as a length and a second reader seeing `790°`
    image = np.full((60, 200), 255, dtype=np.uint8)

    # When OCR runs with the cross-check
    result = recognize_dimensions(
        image,
        _ListOcrBackend("7909"),
        second_reader=_ScriptedReader("790°"),
    )

    # Then the plausible-looking value is flagged instead of trusted
    dimension = result.dimensions[0]
    assert dimension.status is DimensionStatus.UNCERTAIN
    assert dimension.parsed_value == 7909.0
    assert dimension.confidence < 0.98
    assert result.dimension_sources[0].alternate_text == "790°"


def test_an_unreachable_second_reader_is_asked_once_and_reported() -> None:
    # Given a configured second reader whose server is down
    image = np.full((60, 200), 255, dtype=np.uint8)
    reader = _UnreachableReader()

    # When OCR runs with several parsed values
    result = recognize_dimensions(
        image,
        _ListOcrBackend("100", "200"),
        second_reader=reader,
    )

    # Then the primary readings remain reviewable and the outage is visible
    assert reader.calls == 1
    assert all(
        dimension.status is DimensionStatus.NEEDS_REVIEW
        for dimension in result.dimensions
    )
    assert result.second_reader_available is False


def test_leader_marks_the_second_reader_transcribes_are_not_disagreement() -> None:
    # Given a vision model that also transcribes the leader touching `200`
    image = np.full((60, 200), 255, dtype=np.uint8)

    # When OCR runs with the cross-check
    result = recognize_dimensions(
        image,
        _ListOcrBackend("200", "60"),
        second_reader=_ScriptedReader("-200-", "-160"),
    )

    # Then only the genuinely different number is flagged
    assert tuple(dimension.status for dimension in result.dimensions) == (
        DimensionStatus.NEEDS_REVIEW,
        DimensionStatus.UNCERTAIN,
    )
    assert result.dimension_sources[0].alternate_text == "-200-"
