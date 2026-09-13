"""Optional OCR boundary with safe dimension parsing."""

import csv
import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.dimensions import parse_dimension_text
from cadraw.models import Dimension, DimensionId, TextCandidate

GrayImage = NDArray[np.uint8]
_TESSERACT_NOT_FOUND: Final = "Tesseract 실행 파일을 찾을 수 없습니다."
_OCR_INPUT_ENCODING_FAILED: Final = "OCR 입력 이미지를 인코딩할 수 없습니다."
_TESSERACT_FAILED: Final = "Tesseract가 OCR에 실패했습니다."
_TESSERACT_EXECUTABLE_ENV: Final = "CADRAW_TESSERACT"


class OcrUnavailableError(RuntimeError):
    """Raised when the configured OCR executable cannot be used."""


class OcrBackend(Protocol):
    """Minimal OCR adapter contract."""

    def recognize(self, image: GrayImage) -> tuple[TextCandidate, ...]:
        """Return text candidates in image coordinates."""
        ...


@dataclass(frozen=True, slots=True)
class OcrResult:
    """OCR candidates and only the dimensions that parsed safely."""

    candidates: tuple[TextCandidate, ...]
    dimensions: tuple[Dimension, ...]
    available: bool
    backend_name: str | None = None


@dataclass(frozen=True, slots=True)
class TesseractOcrBackend:
    """TSV adapter for an installed Tesseract executable."""

    executable: str = "tesseract"
    language: str = "eng"
    page_segmentation_mode: int = 11
    timeout_seconds: float = 20.0

    def recognize(self, image: GrayImage) -> tuple[TextCandidate, ...]:
        """Run Tesseract through stdin without creating a persistent file."""
        executable = shutil.which(self.executable)
        if executable is None:
            raise OcrUnavailableError(_TESSERACT_NOT_FOUND)
        encoded_ok, encoded = cv2.imencode(".png", image)
        if not encoded_ok:
            raise OcrUnavailableError(_OCR_INPUT_ENCODING_FAILED)
        result = subprocess.run(  # noqa: S603
            [
                executable,
                "stdin",
                "stdout",
                "--psm",
                str(self.page_segmentation_mode),
                "-l",
                self.language,
                "tsv",
            ],
            input=encoded.tobytes(),
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise OcrUnavailableError(message or _TESSERACT_FAILED)
        return _parse_tesseract_tsv(
            result.stdout.decode("utf-8", errors="replace"),
        )


def _parse_tesseract_tsv(payload: str) -> tuple[TextCandidate, ...]:
    """Convert Tesseract TSV rows to typed text candidates."""
    candidates: list[TextCandidate] = []
    reader = csv.DictReader(io.StringIO(payload), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            left = float(row["left"] or 0.0)
            top = float(row["top"] or 0.0)
            width = float(row["width"] or 0.0)
            height = float(row["height"] or 0.0)
            confidence = float(row["conf"] or -1.0) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0.0 or height <= 0.0 or confidence < 0.0:
            continue
        candidates.append(
            TextCandidate(
                text=text,
                bounds=(left, top, width, height),
                confidence=min(1.0, confidence),
            ),
        )
    return tuple(candidates)


def discover_tesseract_backend() -> OcrBackend | None:
    """Return a system OCR backend only when the executable is available."""
    executable = os.environ.get(_TESSERACT_EXECUTABLE_ENV, "tesseract")
    return (
        TesseractOcrBackend(executable=executable) if shutil.which(executable) else None
    )


def recognize_dimensions(
    image: GrayImage,
    backend: OcrBackend | None,
    work_directory: Path | None = None,
) -> OcrResult:
    """Run OCR and parse safe dimension values without assigning targets."""
    del work_directory
    if backend is None:
        return OcrResult(candidates=(), dimensions=(), available=False)
    try:
        candidates = backend.recognize(image)
    except OcrUnavailableError:
        return OcrResult(candidates=(), dimensions=(), available=False)
    dimensions: list[Dimension] = []
    for index, candidate in enumerate(candidates):
        dimension = parse_dimension_text(
            candidate.text,
            DimensionId(f"ocr_{index + 1:03d}"),
        )
        if dimension.parsed_value is not None:
            dimensions.append(dimension)
    return OcrResult(
        candidates=candidates,
        dimensions=tuple(dimensions),
        available=True,
        backend_name=type(backend).__name__,
    )
