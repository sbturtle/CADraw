# CADraw

손으로 그린 철물/금속 제조용 도면을 사람이 확인한 뒤 간단한 DXF로 변환하는 Windows 데스크톱 PoC입니다.

CADraw는 범용 CAD 편집기가 아닙니다. 사진에서 선 후보를 찾고, 기하를 정규화하고, 모르는 치수는 `UNKNOWN`으로 남긴 뒤, 사용자가 입력한 치수만 반영해 DXF를 만듭니다.

## 현재 구현

- JPG/PNG 이미지 열기
- OpenCV grayscale, blur, Canny, probabilistic Hough line detection
- 카메라 손그림 기본값의 후보 확장과 최대 연결 성분 필터
- 수평/수직 snap, endpoint snap, near-collinear line merge
- horizontal / vertical / diagonal 방향 분류
- 연결 관계와 낮은 신뢰도 경고를 갖는 typed intermediate representation
- 정확히 연결된 unambiguous open chain을 순서 있는 `POLYLINE`으로 조립
- 수동 치수 편집: 길이, `Ø` 직경, `R` 반지름, 각도
- image-space 검출 geometry와 CAD-space mm geometry의 명시적 분리
- `DetectedEntity` → `CadEntity` stable ID relation과 resolved/unresolved state
- 연결된 선의 확정 길이·각도 기반 anchor reconstruction
- Preview 선 직접 선택, entity 정보, 검출 결과/CAD 결과 toggle
- Preview Ctrl 다중 선택, 좌→우 window/우→좌 crossing 선택
- 치수 행 선택 시 대응 선을 원본 사진과 CAD preview에서 highlight
- 치수 행에 선 방향과 로컬 길이를 사람이 읽을 수 있는 형태로 표시
- 선택한 선 조각 병합, 잘못 인식한 선 삭제, 원본 이미지 위 선 다시 그리기
- 첫 번째 확정 길이는 연결되지 않은 geometry의 preview fallback scale로만 사용
- Pillow 기반 CAD preview
- `LINE`, `LWPOLYLINE`, `CIRCLE`, `ARC`, `TEXT` 레이어 분리 DXF export
- DXF 재읽기 entity-count 및 실제 segment length validation
- 미확정 geometry는 기본 DXF에서 제외하며 fixture만 명시적으로 허용 가능
- OpenAI-compatible Vision API와 분리 가능한 `VisionAnalyzer` Protocol

## 안전한 동작 원칙

사진상의 비율은 입력 치수 대신 사용하지 않습니다. 연결 geometry에 확정 치수가 있으면
각 segment의 명시 길이와 각도를 우선해 anchor부터 CAD mm 좌표를 재구성합니다.
치수가 없거나 대각선 각도가 없으면 원본 방향을 preview용으로만 사용하고
`UNRESOLVED` 또는 `UNKNOWN_ANGLE` 상태를 유지합니다. OCR/AI가 나중에 추가되더라도
결과는 `needs_review`로 들어가며 사용자가 확인하기 전에는 확정 constraint가 되지 않습니다.

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

실제 샘플 결과는 4개 구조선, 3개 연결, 분석 모델의 4개 원본 `LINE`을
하나의 열린 `LWPOLYLINE`으로 조립하고 4개 `TEXT`를 추가한 총 5개 DXF
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
- polyline 조립은 정확히 동일한 endpoint를 가진 열린 단일 경로에만 적용되며,
  branch, 닫힌 loop, 가까울 뿐 일치하지 않는 gap은 안전하게 원본 `LINE`으로
  남습니다.
- 현재 highlight는 선을 대상으로 한 수동 치수 행에 적용되며, 원/호 전용 확인
  행은 아직 없습니다.
- 병합과 삭제는 확정 치수를 보호하기 위해 해당 값을 먼저 비워야 합니다.
- 선 다시 그리기는 원본 이미지에서 두 점을 드래그하며, 한 행을 선택하면
  기존 선을 교체하고 선택하지 않으면 새 미확인 선을 추가합니다. 여러 행을
  선택하고 다시 그리면 near-collinear 판정 없이 선택된 모든 조각이 방금 그린
  한 선으로 즉시 합쳐집니다. 검출이 한 획을 여러 조각으로 따로 인식하고 서로
  어긋나지 않은 선을 이렇게 정리할 수 있습니다.
- 편집 실행 취소/다시 실행은 아직 지원하지 않습니다.
- 실제 샘플은 현재 도면 주변 ROI를 수동 설정으로 검증했으며, 임의 사진의
  ROI 자동 추정은 아직 없습니다.
- 완전한 일반 constraint solver는 아직 없습니다. 현재는 connected open line graph에
  길이·각도·방향·anchor 규칙을 적용합니다.
- DXF는 1차 산출물이며 DWG는 아직 지원하지 않습니다.
