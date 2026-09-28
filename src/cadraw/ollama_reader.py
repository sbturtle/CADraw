"""Optional second reading of one OCR crop by a local Ollama vision model.

The reader never sees the whole drawing and is never asked where anything is:
the primary detector has already located the label, and the model is shown
only that crop. Its answer is compared with the primary reading and can only
lower a value's review status, never confirm it, as `docs/ai-boundary.md`
requires of every model output.

On the 75 hand-labelled sample values, `qwen3-vl:4b-instruct` (Apache-2.0,
3.5 GB of VRAM) flagged 8 of PP-OCR's 10 misreads as uncertain, and 49 of the
51 values it agreed on were right; `qwen3-vl:2b-instruct` (2.0 GB) flagged 6.
The `-instruct` tag matters: the plain tag is a reasoning build that spends
its token budget thinking and returns no text.
"""

import base64
import json
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlsplit

import cv2
import numpy as np
from numpy.typing import NDArray

from cadraw.ocr import OcrUnavailableError

if TYPE_CHECKING:
    from http.client import HTTPResponse

type GrayImage = NDArray[np.uint8]

_DEFAULT_HOST: Final = "http://127.0.0.1:11434"
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
_NO_READING: Final = "NONE"
_UNREACHABLE: Final = "로컬 비전 모델에 연결하지 못했습니다."
_ENCODING_FAILED: Final = "보조 판독용 이미지를 인코딩하지 못했습니다."
_INVALID_HOST: Final = "Ollama 주소는 http 또는 https여야 합니다."
PROMPT: Final = (
    "The image is a tight crop of ONE handwritten dimension label from a "
    "metalwork sketch. Pencil lines of the drawing may touch the crop border; "
    "ignore them. Read only the handwritten label: digits and, if written, "
    "the symbols R, Ø, °, x, -. Do not add decimal points that are not "
    "written. Answer with the label only, or NONE."
)


@dataclass(frozen=True, slots=True)
class OllamaTextReader:
    """Read one crop through Ollama's local `/api/generate` endpoint."""

    model: str
    host: str = _DEFAULT_HOST
    timeout_seconds: float = 30.0
    context_tokens: int = 4096
    maximum_answer_tokens: int = 24
    keep_alive: str = "10m"

    def read(self, crop: GrayImage) -> str | None:
        """Return the model's transcription, or `None` for an unreadable crop."""
        encoded_ok, encoded = cv2.imencode(".png", crop)
        if not encoded_ok:
            raise OcrUnavailableError(_ENCODING_FAILED)
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": PROMPT,
            "images": [base64.b64encode(np.asarray(encoded).tobytes()).decode()],
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": self.maximum_answer_tokens,
                "num_ctx": self.context_tokens,
            },
        }
        answer = self._generate(payload).strip()
        first_line = answer.splitlines()[0].strip() if answer else ""
        return None if not first_line or first_line == _NO_READING else first_line

    def _generate(self, payload: dict[str, object]) -> str:
        """POST one generation request and return the response text."""
        url = f"{self.host.rstrip('/')}/api/generate"
        if urlsplit(url).scheme not in _ALLOWED_SCHEMES:
            raise OcrUnavailableError(_INVALID_HOST)
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with cast(
                "HTTPResponse",
                urllib.request.urlopen(  # noqa: S310 - scheme checked above
                    request,
                    timeout=self.timeout_seconds,
                ),
            ) as response:
                body = cast("object", json.loads(response.read()))
        except (OSError, ValueError) as error:
            raise OcrUnavailableError(_UNREACHABLE) from error
        if not isinstance(body, dict):
            raise OcrUnavailableError(_UNREACHABLE)
        text = cast("dict[str, object]", body).get("response")
        return text if isinstance(text, str) else ""
