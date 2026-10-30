import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadraw import local_ocr
from cadraw.local_ocr import RapidOcrBackend, discover_ocr_backend, padded_bounds
from cadraw.models import Point
from cadraw.ocr import OcrUnavailableError, TesseractOcrBackend


class _FakeEngine:
    """Stands in for `rapidocr.RapidOCR` with scripted detector output."""

    def __init__(self, texts: tuple[str, ...], scores: tuple[float, ...]) -> None:
        self._texts: tuple[str, ...] = texts
        self._scores: tuple[float, ...] = scores
        self.batches: list[list[tuple[int, ...]]] = []

    def __call__(
        self,
        img_content: np.ndarray,
        *,
        use_det: bool,
        use_cls: bool,
        use_rec: bool,
    ) -> object:
        assert use_det
        assert not use_cls
        assert not use_rec
        assert img_content.ndim == 3
        return SimpleNamespace(
            boxes=np.asarray(
                [
                    [[20, 10], [60, 12], [59, 30], [19, 28]],
                    [[100, 40], [140, 40], [140, 60], [100, 60]],
                ],
                dtype=np.float64,
            ),
        )

    def preprocess_img(self, ori_img: np.ndarray) -> tuple[np.ndarray, object]:
        return ori_img, {}

    def recognize_txt(self, img: list[np.ndarray]) -> object:
        self.batches.append([crop.shape for crop in img])
        return SimpleNamespace(txts=self._texts, scores=self._scores)


def test_backend_keeps_detector_quads_and_drops_weak_readings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an engine that finds two boxes but reads one poorly
    engine = _FakeEngine(("150", "7x"), (0.93, 0.21))

    def fake_engine(_directory: Path | None) -> _FakeEngine:
        return engine

    monkeypatch.setattr(local_ocr, "_engine", fake_engine)

    # When a grayscale frame is recognized
    candidates = RapidOcrBackend().recognize(np.full((80, 160), 255, np.uint8))

    # Then both crops were read in one batch, and only the strong one is kept
    assert len(engine.batches) == 1
    assert len(engine.batches[0]) == 2
    assert tuple(candidate.text for candidate in candidates) == ("150",)
    candidate = candidates[0]
    assert candidate.quad is not None
    assert candidate.quad[0] == Point(20.0, 10.0)
    assert candidate.bounds == (19.0, 10.0, 41.0, 20.0)
    assert candidate.source == "rapidocr"


def test_engine_failures_surface_as_unavailable_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an engine whose onnxruntime session fails at inference time
    class _BrokenEngine:
        def __call__(self, img_content: np.ndarray, **stages: bool) -> object:
            del img_content, stages
            message = "session failed"
            raise RuntimeError(message)

    def broken_engine(_directory: Path | None) -> _BrokenEngine:
        return _BrokenEngine()

    monkeypatch.setattr(local_ocr, "_engine", broken_engine)

    # When recognition runs, then the pipeline sees the documented boundary
    with pytest.raises(OcrUnavailableError):
        _ = RapidOcrBackend().recognize(np.full((40, 40), 255, np.uint8))


def test_crop_margin_scales_with_the_box_and_is_clipped() -> None:
    # Given a 40x20 label near the top-left corner of a 100x100 frame
    quad = (Point(2.0, 3.0), Point(42.0, 3.0), Point(42.0, 23.0), Point(2.0, 23.0))

    # When its recognition crop is cut with a 10% margin
    bounds = padded_bounds(quad, (100, 100), 0.1, 4.0)

    # Then the minimum margin applies and the crop stays inside the image
    assert bounds == (0, 0, 47, 28)


def test_discovery_prefers_the_local_reader_and_falls_back_to_tesseract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an install first with, then without, onnxruntime
    def with_everything(name: str) -> ModuleSpec | None:
        return ModuleSpec(name, None)

    def without_onnxruntime(name: str) -> ModuleSpec | None:
        return None if name == "onnxruntime" else ModuleSpec(name, None)

    monkeypatch.setattr(importlib.util, "find_spec", with_everything)
    preferred = discover_ocr_backend()

    def fake_which(_name: str) -> str:
        return r"C:\t.exe"

    monkeypatch.setattr(importlib.util, "find_spec", without_onnxruntime)
    monkeypatch.setattr("cadraw.ocr.shutil.which", fake_which)
    fallback = discover_ocr_backend()

    # Then handwriting OCR is used whenever it can run
    assert isinstance(preferred, RapidOcrBackend)
    assert isinstance(fallback, TesseractOcrBackend)


def _installed_models() -> bool:
    spec = importlib.util.find_spec("rapidocr")
    if spec is None or spec.origin is None:
        return False
    models = Path(spec.origin).parent / "models"
    return all(
        (models / name).exists()
        for name in ("PP-OCRv6_det_small.onnx", "PP-OCRv6_rec_small.onnx")
    )


@pytest.mark.skipif(
    not _installed_models(),
    reason="PP-OCRv6 models download on first use; not fetched during tests",
)
def test_real_models_read_a_handwritten_sample_label() -> None:
    # Given the photographed sketch whose top label is a handwritten 100
    from cadraw.detection import load_grayscale_image  # noqa: PLC0415

    image = load_grayscale_image(Path("sample/1.jpg"))

    # When the shipped backend reads it
    texts = {candidate.text for candidate in RapidOcrBackend().recognize(image)}

    # Then the drawing's two clearest dimensions are among the readings
    assert {"100", "1050"} <= texts


def test_reading_the_same_frame_again_reuses_the_first_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an engine whose reads are counted
    engine = _FakeEngine(("150", "60"), (0.93, 0.91))

    def fake_engine(_directory: Path | None) -> _FakeEngine:
        return engine

    monkeypatch.setattr(local_ocr, "_engine", fake_engine)
    image = np.full((80, 160), 255, dtype=np.uint8)
    backend = RapidOcrBackend()

    # When the same frame is read twice, and then a changed frame
    first = backend.recognize(image)
    second = backend.recognize(image.copy())
    changed = image.copy()
    changed[0, 0] = 0
    _ = backend.recognize(changed)

    # Then the engine read the first frame once and the changed frame anew
    assert second == first
    assert len(engine.batches) == 2
