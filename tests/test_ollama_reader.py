import json
import urllib.request
from typing import Self, cast

import numpy as np
import pytest

from cadraw.ocr import OcrUnavailableError
from cadraw.ollama_reader import OllamaTextReader


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body: bytes = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_details: object) -> None:
        return None


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    requests: list[dict[str, object]],
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout > 0
        assert request.full_url == "http://127.0.0.1:11434/api/generate"
        body = request.data
        assert isinstance(body, bytes)
        requests.append(cast("dict[str, object]", json.loads(body)))
        return _Response(json.dumps({"response": answer}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_reader_sends_only_the_crop_and_returns_the_first_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a local model that answers with a label and trailing chatter
    requests: list[dict[str, object]] = []
    _serve(monkeypatch, " 400R\nThe label is a radius.", requests)
    reader = OllamaTextReader(model="qwen3-vl:2b-instruct")

    # When one crop is read
    text = reader.read(np.full((30, 60), 255, dtype=np.uint8))

    # Then one non-reasoning request carried exactly one image
    assert text == "400R"
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "qwen3-vl:2b-instruct"
    assert request["think"] is False
    images = request["images"]
    assert isinstance(images, list)
    assert len(images) == 1


def test_an_explicit_none_answer_is_no_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a model that cannot read the crop
    _serve(monkeypatch, "NONE", [])

    # When the crop is read, then no text is invented
    reader = OllamaTextReader(model="qwen3-vl:2b-instruct")
    assert reader.read(np.full((30, 60), 255, dtype=np.uint8)) is None


def test_a_down_server_or_foreign_scheme_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given no Ollama server listening, and a non-HTTP host
    def refuse(request: urllib.request.Request, timeout: float) -> _Response:
        del request, timeout
        message = "connection refused"
        raise ConnectionRefusedError(message)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    crop = np.full((30, 60), 255, dtype=np.uint8)

    # When a crop is read, then the OCR boundary's unavailable error is raised
    with pytest.raises(OcrUnavailableError):
        _ = OllamaTextReader(model="m").read(crop)
    with pytest.raises(OcrUnavailableError):
        _ = OllamaTextReader(model="m", host="file:///etc").read(crop)
