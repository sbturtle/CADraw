# AI 경계

## 현재 어댑터 계약

```python
class VisionAnalyzer(Protocol):
    def analyze(self, image_path: Path) -> DrawingAnalysis: ...
```

`MockVisionAnalyzer`는 로컬 결정적 파이프라인을 실행합니다. 설정이 OCR을 켜면
(`PipelineConfig.for_workbench()`, 앱 기본값) 아래의 로컬 OCR도 함께 돌지만,
네트워크나 API 키는 쓰지 않습니다. 테스트와 스크립트가 쓰는
`PipelineConfig.from_detection_defaults()`는 OCR을 끈 상태라 모델 없이 결정적으로
동작합니다.

## 로컬 OCR (ADR-0018)

```python
class OcrBackend(Protocol):
    def recognize(self, image: GrayImage) -> tuple[TextCandidate, ...]: ...


class TextReader(Protocol):
    def read(self, crop: GrayImage) -> str | None: ...
```

- `local_ocr.RapidOcrBackend`: RapidOCR의 PP-OCRv6(Apache-2.0)을 CPU에서
  onnxruntime으로 실행합니다. 검출 사변형(`TextCandidate.quad`)과 원문, 신뢰도를
  보존합니다. `discover_ocr_backend`는 이 판독기를 먼저 고르고, onnxruntime이
  없으면 Tesseract로 돌아갑니다.
- `ollama_reader.OllamaTextReader`(선택, `CADRAW_VISION_MODEL`): 로컬 Ollama
  비전 모델에 **이미 찾은 글자 상자의 크롭 하나**만 보내고, 그 안의 글자만
  묻습니다. 위치, 좌표, 누락 치수는 묻지 않습니다. 답은
  `TextCandidate.alternate_text`에만 남습니다.

두 판독의 역할은 비대칭입니다. 두 번째 판독은 1차 값을 바꾸지 않고, 일치하면
`NEEDS_REVIEW`를 유지하며, 다르면 `UNCERTAIN`으로 **낮출 뿐**입니다. 어느
쪽도 `CONFIRMED`를 만들 수 없습니다.

글자 영역은 도형을 지울 수 있는 유일한 모델 출력입니다. 그 경우에도 모델 출력
하나만으로는 부족합니다. 숫자 두 개 이상, 후보 전체가 영역 안, 밑의 잉크 성분
전체가 영역 안이라는 세 조건이 모두 필요하고, 원(구멍)은 지우지 않으며, 지운
개수는 `TEXT_STROKES_REMOVED`로 보고합니다.

## 향후 OpenAI 호환 어댑터

향후 어댑터는 OpenRouter 또는 다른 OpenAI 호환 엔드포인트를 호출할 수 있습니다. 이때 반드시 다음을 지켜야 합니다.

1. 자격 증명을 소스 코드가 아니라 사용자 설정이나 환경 변수에서 로드한다.
2. 응답을 경계에서 타입이 있는 스키마로 파싱한다.
3. 원본 텍스트, 신뢰도, 소스 영역을 보존한다.
4. 연결이 불확실하면 `target_entity_ids = ()`를 반환한다.
5. 사람이 확인하기 전까지 인식된 값을 `NEEDS_REVIEW`로 표시한다.
6. 절대 DXF를 작성하거나 최종 CAD 좌표를 내보내지 않는다.

결정적 정규화기, 계산기, 내보내기는 공급자와 무관하게 유지됩니다.

## 명시적 비목표

- 누락된 측정값을 모델에게 "완성"하라고 요청하는 프롬프트는 없습니다.
- 모델이 생성한 좌표 목록을 CAD 기하로 신뢰하지 않습니다.
- 확정 값 없이 이미지 비율이 물리 치수가 되는 일은 없습니다.
- 비상업 라이선스 모델(예: Qwen2.5-VL 3B의 Qwen Research License)은 기본값이나
  권장값으로 두지 않습니다.
