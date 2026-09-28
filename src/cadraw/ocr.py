"""Optional OCR boundary with safe dimension parsing."""

import csv
import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.dimensions import parse_dimension_text
from cadraw.models import Dimension, DimensionId, DimensionStatus, TextCandidate

GrayImage = NDArray[np.uint8]
_TESSERACT_NOT_FOUND: Final = "Tesseract 실행 파일을 찾을 수 없습니다."
_OCR_INPUT_ENCODING_FAILED: Final = "OCR 입력 이미지를 인코딩할 수 없습니다."
_TESSERACT_FAILED: Final = "Tesseract가 OCR에 실패했습니다."
_TESSERACT_EXECUTABLE_ENV: Final = "CADRAW_TESSERACT"
# A handwritten fabrication size never carries three decimals. A reading that
# does is a price (`100.000`), a date (`2021.319`), or thousands grouping
# (`1,050`) that the decimal parser would silently shrink a thousandfold.
_IMPLAUSIBLE_FRACTION = re.compile(r"\d[.,]\d{3,}")
# A lone digit is one glyph, and a tick mark, a drilled hole, or an arrowhead
# reads as `1`, `0`, or `7` just as readily. `R5` or `Ø8` carry a symbol that
# says they are sizes, so only the bare single digit is held back.
_BARE_SINGLE_DIGIT = re.compile(r"\s*\d\s*")
_UNCORROBORATED_CONFIDENCE_FACTOR: Final = 0.5
_VALUE_TOLERANCE: Final = 1e-9
# The minus, en dash, em dash, arrows, and multiplication sign are given by
# code point so they stay distinguishable from the ASCII marks they resemble.
_LEADER_MARKS: Final = " -_<>xX*\"'|/:;,." + "".join(
    chr(code)
    for code in (0x2212, 0x2013, 0x2014, 0x2190, 0x2192, 0x2191, 0x2193, 0x00D7)
)
_READER_CROP_PADDING_RATIO: Final = 0.1
_READER_CROP_MINIMUM_PADDING: Final = 4.0


class OcrUnavailableError(RuntimeError):
    """Raised when the configured OCR executable cannot be used."""


class OcrBackend(Protocol):
    """Minimal OCR adapter contract."""

    def recognize(self, image: GrayImage) -> tuple[TextCandidate, ...]:
        """Return text candidates in image coordinates."""
        ...


class TextReader(Protocol):
    """A second, independent reader for one text box the OCR already found.

    It is only ever shown a crop, so it cannot move a box or add one; all it
    can do is agree or disagree with what the primary reader saw.
    """

    def read(self, crop: GrayImage) -> str | None:
        """Return the text in the crop, or `None` when it cannot be read."""
        ...


@dataclass(frozen=True, slots=True)
class OcrResult:
    """OCR candidates and only the dimensions that parsed safely."""

    candidates: tuple[TextCandidate, ...]
    dimensions: tuple[Dimension, ...]
    available: bool
    backend_name: str | None = None
    dimension_sources: tuple[TextCandidate, ...] = ()
    second_reader_available: bool | None = None


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


def _reader_crop(
    image: GrayImage,
    bounds: tuple[float, float, float, float],
) -> GrayImage:
    """Cut the box a second reader is asked about, with the primary's margin."""
    x, y, width, height = bounds
    padding = max(
        _READER_CROP_MINIMUM_PADDING,
        _READER_CROP_PADDING_RATIO * min(width, height),
    )
    image_height, image_width = cast("tuple[int, int]", image.shape[:2])
    left = max(0, int(x - padding))
    top = max(0, int(y - padding))
    right = min(image_width, int(x + width + padding) + 1)
    bottom = min(image_height, int(y + height + padding) + 1)
    return np.ascontiguousarray(image[top:bottom, left:right])


def _agrees(dimension: Dimension, alternate_text: str | None) -> bool:
    """Return whether a second reading parses to the same kind and value.

    A vision model transcribes the leader line and arrowhead touching a label
    as well (`-200-`, `<159`, `200x`). Those marks are trimmed from the ends of
    the second reading before it is compared, so only a different number or
    symbol counts as disagreement; the reading shown to the reviewer is kept.
    """
    if alternate_text is None or dimension.parsed_value is None:
        return False
    other = parse_dimension_text(
        alternate_text.strip(_LEADER_MARKS),
        dimension.dimension_id,
    )
    return (
        other.kind is dimension.kind
        and other.parsed_value is not None
        and abs(other.parsed_value - dimension.parsed_value) < _VALUE_TOLERANCE
    )


def _cross_checked(
    dimension: Dimension,
    candidate: TextCandidate,
    image: GrayImage,
    reader: TextReader,
) -> tuple[Dimension, TextCandidate]:
    """Keep a reading merely reviewable only when a second reader agrees.

    Disagreement is the useful signal: PP-OCR reads a handwritten `790Ø` as
    the plausible length `7909`, while a vision-language model reads `790°`.
    Neither is right, and marking the row uncertain is what tells the reviewer
    that this particular value was not seen the same way twice.
    """
    alternate = reader.read(_reader_crop(image, candidate.bounds))
    checked = replace(candidate, alternate_text=alternate)
    if _agrees(dimension, alternate):
        return dimension, checked
    return (
        replace(
            dimension,
            confidence=dimension.confidence * _UNCORROBORATED_CONFIDENCE_FACTOR,
            status=DimensionStatus.UNCERTAIN,
        ),
        checked,
    )


def recognize_dimensions(
    image: GrayImage,
    backend: OcrBackend | None,
    work_directory: Path | None = None,
    second_reader: TextReader | None = None,
) -> OcrResult:
    """Run OCR and parse safe dimension values without assigning targets.

    With a `second_reader`, every parsed value is read again from its crop.
    A reader that becomes unreachable part-way stops being asked, and the
    values it did not see stay at their primary, still-reviewable status.
    """
    del work_directory
    if backend is None:
        return OcrResult(candidates=(), dimensions=(), available=False)
    try:
        candidates = backend.recognize(image)
    except OcrUnavailableError:
        return OcrResult(candidates=(), dimensions=(), available=False)
    dimensions: list[Dimension] = []
    dimension_sources: list[TextCandidate] = []
    read_candidates = list(candidates)
    reader = second_reader
    for index, candidate in enumerate(candidates):
        if _IMPLAUSIBLE_FRACTION.search(candidate.text) or (
            _BARE_SINGLE_DIGIT.fullmatch(candidate.text)
        ):
            continue
        dimension = parse_dimension_text(
            candidate.text,
            DimensionId(f"ocr_{index + 1:03d}"),
        )
        if dimension.parsed_value is None:
            continue
        source = candidate
        if reader is not None:
            try:
                dimension, source = _cross_checked(dimension, candidate, image, reader)
            except OcrUnavailableError:
                reader = None
        read_candidates[index] = source
        dimensions.append(dimension)
        dimension_sources.append(source)
    return OcrResult(
        candidates=tuple(read_candidates),
        dimensions=tuple(dimensions),
        available=True,
        backend_name=type(backend).__name__,
        dimension_sources=tuple(dimension_sources),
        second_reader_available=(None if second_reader is None else reader is not None),
    )
