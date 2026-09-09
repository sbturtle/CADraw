# CADraw

손으로 그린 철물/금속 제조용 도면을 사람이 확인한 뒤 간단한 DXF로 변환하는 Windows 데스크톱 PoC입니다.

CADraw는 범용 CAD 편집기가 아닙니다. 사진에서 선 후보를 찾고, 기하를 정규화하고, 모르는 치수는 `UNKNOWN`으로 남긴 뒤, 사용자가 입력한 치수만 반영해 DXF를 만듭니다.

## 현재 구현

- JPG/PNG 이미지 열기
- OpenCV grayscale, blur, Canny, probabilistic Hough line detection
- 수평/수직 snap, endpoint snap, near-collinear line merge
- horizontal / vertical / diagonal 방향 분류
- 연결 관계와 낮은 신뢰도 경고를 갖는 typed intermediate representation
- 수동 치수 편집: 길이, `Ø` 직경, `R` 반지름, 각도
- 첫 번째 확정 길이로만 전역 mm 스케일을 계산
- Pillow 기반 CAD preview
- `LINE`, `LWPOLYLINE`, `CIRCLE`, `ARC`, `TEXT` 레이어 분리 DXF export
- DXF 재읽기 entity-count validation
- OpenAI-compatible Vision API와 분리 가능한 `VisionAnalyzer` Protocol

## 안전한 동작 원칙

사진상의 비율은 실제 치수로 확정하지 않습니다. 확정된 치수가 하나도 없으면 geometry는 pixel-like local unit으로 유지되고, UI와 DXF validation 결과에 스케일 미확인 상태가 표시됩니다. OCR/AI가 나중에 추가되더라도 결과는 `needs_review`로 들어가며 사용자가 확인하기 전에는 CAD 좌표를 만들지 않습니다.

## 시작하기

Windows PowerShell에서:

```powershell
uv sync
uv run cadraw
```

이미지를 바로 넣으려면:

```powershell
uv run cadraw -- C:\path\to\sketch.jpg
```

## 샘플 PoC

기본 실행은 재현 가능한 synthetic hand-drawing fixture를 생성합니다. 선은 약간
기울어지고 끊긴 상태로 만들며, 숫자는 OCR 확정값으로 사용하지 않습니다.

```powershell
uv run python scripts/run_poc.py
```

기본 산출물:

- `artifacts/sample_hand_drawing.png`
- `artifacts/sample_preview.png`
- `artifacts/sample_drawing.dxf`
- `artifacts/sample_summary.json`

제공된 실제 사진도 같은 runner로 실행할 수 있습니다. 사진 가장자리와 배경
잡음을 제외하기 위해 이 샘플에 한해 문서화된 ROI를 사용합니다. 아래 네 값은
사용자가 확인한 값이며 OCR 결과로 가장하지 않습니다. 위에서부터 `100 mm`,
`1050 mm`, `20°`, `150 mm`입니다.

```powershell
uv run python scripts/run_poc.py `
  --input sample\KakaoTalk_20260909_141552817.jpg `
  --dimensions "100,1050,20°,150" `
  --output-dir artifacts\actual_sample_confirmed
```

실제 샘플 결과는 4개 구조선, 3개 연결, 4개 `LINE`과 4개 `TEXT` DXF
entity로 확인됩니다. 첫 번째 확정 길이로만 전역 스케일을 계산하므로
그림 비율과 실제 치수가 다른 두 값은 `CONSTRAINT_MISMATCH`로 표시됩니다.

GUI smoke test 없이 headless로 파이프라인만 실행하려면:

```powershell
uv run python scripts/run_poc.py --no-preview
```

## 테스트와 정적 검사

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
```

## 구조

```text
CADraw/
├─ DESIGN.md
├─ README.md
├─ pyproject.toml
├─ docs/
│  ├─ architecture.md
│  ├─ pipeline.md
│  ├─ geometry-model.md
│  ├─ ai-boundary.md
│  ├─ decisions/
│  └─ experiments/
├─ src/cadraw/
│  ├─ app.py
│  ├─ models.py
│  ├─ detection.py
│  ├─ normalization.py
│  ├─ dimensions.py
│  ├─ calculator.py
│  ├─ preview.py
│  ├─ export.py
│  ├─ vision.py
│  └─ ui/
└─ tests/
```

## 현재 한계

- 손글씨 OCR과 자동 dimension-to-entity association은 아직 없습니다.
- 사진 원근 보정, rectangle/circle/hole 특화 검출, arc 복원은 다음 단계입니다.
- 실제 샘플은 현재 도면 주변 ROI를 수동 설정으로 검증했으며, 임의 사진의
  ROI 자동 추정은 아직 없습니다.
- 여러 확정 치수를 geometry 길이에 동시에 반영하는 제약 solver는 없습니다.
  첫 번째 확정 길이만 전역 scale 기준으로 사용하고 나머지는 검증 경고로 취급합니다.
- DXF는 1차 산출물이며 DWG는 아직 지원하지 않습니다.
