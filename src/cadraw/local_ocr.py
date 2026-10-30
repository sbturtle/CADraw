"""Local handwriting OCR through RapidOCR's PP-OCR ONNX models.

PP-OCRv6 runs on the CPU through onnxruntime, needs no API key or GPU, and is
Apache-2.0 licensed. On the checked-in sketches it read 61 of 75 hand-labelled
dimension values; 2B and 4B vision-language models read 54-59 on their own,
so this is the primary reader. See `docs/experiments/local-ocr.md`.

Detection and recognition are run as two steps on purpose. The detector's
quad hugs the ink, and recognizing that tight crop clips the first and last
strokes of loose handwriting (a leading `1` or a trailing `0`). An
axis-aligned crop with a small margin read more labels; a wide margin read
fewer, because it pulls in the neighbouring strokes.
"""

import importlib.util
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from hashlib import blake2b
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from weakref import WeakKeyDictionary

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.models import Point, TextCandidate
from cadraw.ocr import OcrBackend, OcrUnavailableError, discover_tesseract_backend

if TYPE_CHECKING:
    from collections.abc import Sequence

type GrayImage = NDArray[np.uint8]
type Quad = tuple[Point, Point, Point, Point]

_ENGINE_UNAVAILABLE: Final = "RapidOCR 엔진을 준비하지 못했습니다."
_RECOGNITION_FAILED: Final = "RapidOCR가 문자 인식에 실패했습니다."
_LOG_LEVEL: Final = "error"
_GRAYSCALE_DIMENSIONS: Final = 2
_MINIMUM_CROP_SIDE: Final = 2
_LOCAL_OCR_MODULES: Final = ("rapidocr", "onnxruntime")
_INFERENCE_THREADS: Final = 4
_CACHED_READINGS: Final = 4
_DETECT_ONLY: Final = {"use_det": True, "use_cls": False, "use_rec": False}


class _Engine(Protocol):
    """The slice of `rapidocr.RapidOCR` this adapter relies on."""

    def __call__(
        self,
        img_content: NDArray[np.uint8],
        *,
        use_det: bool,
        use_cls: bool,
        use_rec: bool,
    ) -> object:
        """Run the requested stages on one image."""
        ...

    def preprocess_img(
        self,
        ori_img: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], object]:
        """Bring one image within the engine's side-length bounds."""
        ...

    def recognize_txt(self, img: list[NDArray[np.uint8]]) -> object:
        """Recognize a batch of single-line crops, in the order given."""
        ...


_READINGS: WeakKeyDictionary[
    "_Engine",
    OrderedDict[
        tuple[bytes, tuple[int, ...], "RapidOcrBackend"], tuple[TextCandidate, ...]
    ],
] = WeakKeyDictionary()


@cache
def _engine(model_directory: Path | None) -> _Engine:
    """Build one engine per model directory; RapidOCR downloads on first use."""
    try:
        from rapidocr import RapidOCR  # noqa: PLC0415 - optional heavy import

        params: dict[str, object] = {
            "Global.log_level": _LOG_LEVEL,
            # onnxruntime's default spins one thread per logical core. With
            # the UI or anything else busy, that oversubscription made the
            # same recognition eight times slower than four fixed threads.
            "EngineConfig.onnxruntime.intra_op_num_threads": _INFERENCE_THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
        if model_directory is not None:
            params["Global.model_root_dir"] = str(model_directory)
        return cast("_Engine", RapidOCR(params=params))
    # The engine can fail through an import, a model download, or an
    # onnxruntime session, each with its own exception family. Any of them
    # means the same thing to a reviewer: no OCR on this machine right now.
    except Exception as error:
        raise OcrUnavailableError(_ENGINE_UNAVAILABLE) from error


def _quads(output: object) -> tuple[Quad, ...]:
    """Read detector quads from a RapidOCR detection output."""
    boxes = getattr(output, "boxes", None)
    if boxes is None:
        return ()
    quads: list[Quad] = []
    for box in cast("Sequence[NDArray[np.float64]]", boxes):
        corners = np.asarray(box, dtype=np.float64).reshape(4, 2)
        first, second, third, fourth = (
            Point(float(x), float(y)) for x, y in corners.tolist()
        )
        quads.append((first, second, third, fourth))
    return tuple(quads)


def _readings(output: object, count: int) -> tuple[tuple[str, float] | None, ...]:
    """Pair each recognized crop with its text and score, in input order."""
    texts = cast("Sequence[str] | None", getattr(output, "txts", None))
    scores = cast("Sequence[float] | None", getattr(output, "scores", None))
    if texts is None or scores is None or len(texts) != count:
        return (None,) * count
    return tuple(
        (str(text).strip(), float(score)) if str(text).strip() else None
        for text, score in zip(texts, scores, strict=True)
    )


def padded_bounds(
    quad: Quad,
    image_shape: tuple[int, int],
    padding_ratio: float,
    minimum_padding: float,
) -> tuple[int, int, int, int]:
    """Return an axis-aligned crop around a quad with a margin, clipped.

    The margin scales with the box's shorter side, so a label keeps the same
    breathing room whether the photo is 1000 or 4000 pixels wide.
    """
    xs = tuple(point.x for point in quad)
    ys = tuple(point.y for point in quad)
    short_side = min(max(xs) - min(xs), max(ys) - min(ys))
    padding = max(minimum_padding, padding_ratio * short_side)
    height, width = image_shape
    left = max(0, int(min(xs) - padding))
    top = max(0, int(min(ys) - padding))
    right = min(width, int(max(xs) + padding) + 1)
    bottom = min(height, int(max(ys) + padding) + 1)
    return left, top, right, bottom


@dataclass(frozen=True, slots=True)
class RapidOcrBackend:
    """PP-OCRv6 text detection followed by batched padded-crop recognition."""

    model_directory: Path | None = None
    padding_ratio: float = 0.1
    minimum_padding: float = 4.0
    minimum_score: float = 0.5

    def recognize(self, image: GrayImage) -> tuple[TextCandidate, ...]:
        """Return every readable text box in the image's own coordinates.

        The workbench analyzes the same frame again when the reviewer picks
        another preprocessing variant, and reading it is the slowest step.
        The engine is deterministic, so the last few readings are kept per
        engine by the frame's digest and this reader's settings.
        """
        engine = _engine(self.model_directory)
        readings = _READINGS.setdefault(engine, OrderedDict())
        key = (
            blake2b(np.ascontiguousarray(image).tobytes(), digest_size=16).digest(),
            cast("tuple[int, ...]", image.shape),
            self,
        )
        cached = readings.get(key)
        if cached is not None:
            readings.move_to_end(key)
            return cached
        result = self._recognize(engine, image)
        readings[key] = result
        if len(readings) > _CACHED_READINGS:
            _ = readings.popitem(last=False)
        return result

    def _recognize(
        self,
        engine: _Engine,
        image: GrayImage,
    ) -> tuple[TextCandidate, ...]:
        """Detect text boxes in one frame and read each padded crop."""
        color: NDArray[np.uint8] = np.asarray(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == _GRAYSCALE_DIMENSIONS
            else image,
            dtype=np.uint8,
        )
        height, width = cast("tuple[int, int]", image.shape[:2])
        quads: list[Quad] = []
        crops: list[NDArray[np.uint8]] = []
        for quad in _quads(_guarded(lambda: engine(color, **_DETECT_ONLY))):
            left, top, right, bottom = padded_bounds(
                quad,
                (height, width),
                self.padding_ratio,
                self.minimum_padding,
            )
            if min(right - left, bottom - top) < _MINIMUM_CROP_SIDE:
                continue
            quads.append(quad)
            crops.append(
                np.ascontiguousarray(color[top:bottom, left:right], dtype=np.uint8),
            )
        if not crops:
            return ()
        readings = _readings(
            _guarded(
                lambda: engine.recognize_txt(
                    [engine.preprocess_img(crop)[0] for crop in crops],
                ),
            ),
            len(crops),
        )
        return tuple(
            _candidate(quad, reading)
            for quad, reading in zip(quads, readings, strict=True)
            if reading is not None and reading[1] >= self.minimum_score
        )


def _candidate(quad: Quad, reading: tuple[str, float]) -> TextCandidate:
    """Build one text candidate from a detector quad and its reading."""
    xs = tuple(point.x for point in quad)
    ys = tuple(point.y for point in quad)
    return TextCandidate(
        text=reading[0],
        bounds=(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
        confidence=min(1.0, reading[1]),
        source="rapidocr",
        quad=quad,
    )


def _guarded(step: Callable[[], object]) -> object:
    """Run one engine step, reporting any engine failure as unavailable OCR."""
    try:
        return step()
    except Exception as error:  # the same families as `_engine`
        raise OcrUnavailableError(_RECOGNITION_FAILED) from error


def discover_ocr_backend() -> OcrBackend | None:
    """Prefer the bundled local PP-OCR reader, then a system Tesseract.

    Tesseract's models are trained on print and read few of these handwritten
    labels, so it is only the fallback for an install that lacks onnxruntime.
    Finding the package is enough here: its models download on first use, and
    a failed download surfaces later as unavailable OCR.
    """
    if all(importlib.util.find_spec(name) is not None for name in _LOCAL_OCR_MODULES):
        return RapidOcrBackend()
    return discover_tesseract_backend()
