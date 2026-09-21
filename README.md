# CADraw

손으로 그린 철물/금속 제조용 도면을 사람이 확인한 뒤 간단한 DXF로 변환하는 Windows 데스크톱 PoC입니다.

CADraw는 범용 CAD 편집기가 아닙니다. 사진에서 선 후보를 찾고, 기하를 정규화하고, 모르는 치수는 `UNKNOWN`으로 남긴 뒤, 사용자가 입력한 치수만 반영해 DXF를 만듭니다.

## 현재 구현

- JPG/PNG 이미지 열기
- OpenCV grayscale, blur, Canny, probabilistic Hough line detection
- 양축에서 증명된 모눈 괘선만 검출 이전에 제거(주기·레일 수 보고)
- 고립된 페이지 테두리 괘선과 타일링으로 증명된 인쇄 표 칸 제거
- variant별 도형 검출로 증명된 종이 구조량을 선택 점수에 반영
- 조명 평탄화 + 전역 임계값 전처리 variant로 흐린 인쇄 배경 거부
- 카메라 손그림 기본값의 후보 확장과 비율 기준 연결 성분 필터(시트의 여러 부품 유지)
- 잉크가 연속임이 확인된 끊긴 획만 병합 허용치 너머로 다시 잇기
- 증명된 표 칸 모서리와 모눈 레일 위의 흐린 고립 선 제거
- 수평/수직 snap, endpoint snap, near-collinear line merge
- horizontal / vertical / diagonal 방향 분류
- 연결 관계와 낮은 신뢰도 경고를 갖는 typed intermediate representation
- 정확히 연결된 unambiguous open chain을 순서 있는 `POLYLINE`으로 조립
- 수동 치수 편집: 길이, `Ø` 직경, `R` 반지름, 각도, 프로파일 전체 폭·높이
- image-space 검출 geometry와 CAD-space mm geometry의 명시적 분리
- `DetectedEntity` → `CadEntity` stable ID relation과 resolved/unresolved state
- 연결된 선의 확정 길이·각도 기반 anchor reconstruction
- 원본 사진에서 도면 영역을 드래그로 지정해 양식 머리글·푸터를 제외
- Preview 선 직접 선택, entity 정보, 검출 결과/CAD 결과 toggle
- Preview Ctrl 다중 선택, 좌→우 window/우→좌 crossing 선택
- 치수 행 선택 시 대응 선을 원본 사진과 CAD preview에서 highlight
- 치수 행에 선 방향과 로컬 길이를 사람이 읽을 수 있는 형태로 표시
- 선택한 선 조각 병합, 잘못 인식한 선 삭제, 원본 이미지 위 선 다시 그리기
- 선 편집·치수 확정에 대한 실행 취소/다시 실행(`Ctrl+Z`/`Ctrl+Y`)
- OCR 인식값과 가까운 geometry가 유일하면 확인 대상만 미리 채우는 제안(자동 확정 아님)
- 원/호 전용 지름·반지름 확인 행과 확정값-실제 크기 불일치 경고
- 닫힌 사각형/원 외에 모따기·필렛 등 부분 호(open arc) 검출
- T·L·노치 등 모서리 개수를 가정하지 않는 일반 프로파일 폴리곤 검출
- 앱 기본값에서 닫힌 프로파일·구멍 검출(크기 하한은 해상도에 비례)
- 확정된 선 길이·프로파일 전체 치수·지름 중 무엇이든 전역 보정 근거가 됨
- 확인 행의 측정 종류가 입력 텍스트보다 우선(맨 숫자도 지름 행이면 지름)
- 도형은 자기 확정값으로만 크기가 정해지며, 확인 없는 도형은 preview에만 배치
- 프로파일 폭·높이를 모두 확정하면 축별로 적용해 전체 치수가 정확히 일치
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

추가 사진 샘플에서 중앙 사진 영역과 닫힌 기본 형상을 함께 추출하려면:

```powershell
uv run python scripts/run_poc.py `
  --input sample\KakaoTalk_20260911_193752753.jpg `
  --auto-region --detect-shapes `
  --output-dir artifacts\enhanced_sample
```

`--auto-region`은 사진 가장자리에서 일정 비율을 제외하는 결정적 inset이며,
`--detect-shapes`는 윤곽에서 닫힌 사각형과 원 후보를 추가합니다. 검출된
도형은 원본 사진 좌표로 복원되어 기존 DXF/preview 경로에 전달됩니다.

페이지 경계를 찾을 수 있는 사진은 실제 사변형 투시 보정도 적용할 수 있습니다.
OCR은 설치된 Tesseract를 사용할 때만 활성화되며, 엔진이 없으면 값을 추측하지
않고 `OCR_UNAVAILABLE` 경고를 남깁니다. Tesseract가 PATH에 없다면
`CADRAW_TESSERACT`에 실행 파일 경로를 지정할 수 있습니다.

```powershell
$env:CADRAW_TESSERACT = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

```powershell
uv run python scripts/run_poc.py `
  --input sample\KakaoTalk_20260911_193752753.jpg `
  --auto-region --detect-shapes --perspective --ocr `
  --output-dir artifacts\full_vision_sample
```

외곽 형상 안에 포함된 작은 닫힌 형상은 `HOLE` 의미로 보수적으로 분류합니다.
미리보기에서는 파란색으로, DXF에서는 `HOLE` 레이어로 구분됩니다.

## 샘플 추출 결과 검토

7개 샘플의 추출 결과를 레퍼런스 DXF와 나란히 놓은 이미지와 요약표를
생성합니다. 결과는 `docs/experiments/sample-extraction.md`에서 봅니다.

```powershell
uv run python scripts/render_sample_report.py
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
│  ├─ perspective.py
│  ├─ ocr.py
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

- 손글씨 의미 해석은 아직 제한적입니다. OCR-geometry 자동 연결은 유일하게
  가까운 후보가 있을 때만 확인 대상을 미리 채우는 제안이며, 모호하면
  비워 두고 값 확정은 항상 사람이 합니다.
- 페이지 경계가 사진에 보이지 않으면 실제 원근 보정을 적용할 수 없으며,
  `PERSPECTIVE_NOT_FOUND` 경고와 함께 원본 좌표로 안전하게 처리합니다.
- Tesseract가 설치되지 않은 환경에서는 OCR을 실행하지 않고
  `OCR_UNAVAILABLE` 경고를 남깁니다.
- 부분 호 검출은 면적 기준을 만족하는 닫힌 도형이 되지 못한 윤곽에만
  최소제곱 원 피팅을 시도하며, 자재 가장자리에 걸쳐 잘린 반원처럼 여백
  규칙에 걸리는 경우는 아직 제외됩니다.
- 모눈 괘선 제거는 양축에서 주기가 증명될 때만 동작합니다. 괘선이 연속
  레일로 뽑히지 않을 만큼 흐리면 증명이 실패해 격자가 남습니다.
- 페이지 테두리 괘선은 축 정렬·시트 60% 횡단·가장자리 22% 이내·다른 선과
  미연결을 모두 만족할 때만 제거합니다. 같은 괘선의 더 짧은 조각은 남습니다.
  글자 획 하나가 괘선에 닿기만 해도 미연결 조건이 깨져 괘선이 남습니다.
- 연결 성분 필터는 가장 긴 성분 대비 20% 이상인 성분을 모두 남기므로, 표로
  증명되지 않은 머리글·푸터 괘선이 고립 선으로 남을 수 있습니다. 이 경우
  도면 영역을 지정하는 것이 의도된 해결책입니다.
- 손글씨 숫자의 획이 도면 모서리에 닿으면 구조에 연결된 선으로 남습니다.
  기하만으로는 구분할 수 없어 사용자가 삭제해야 합니다.
- 푸터의 회사 로고 블록은 표가 아니라 글리프 묶음이어서 타일링 증명에
  걸리지 않고 남습니다.
- 허용치(표 12칸)와 포화값(120칸)은 제공된 양식에서 관측한 값이며, 머리글
  칸이 훨씬 많은 양식에서는 불필요한 감점이 생길 수 있습니다. 도면 영역을
  지정하면 이 판단들이 결과를 좌우하지 않습니다.
- 도면 영역은 축 정렬 사각형만 지원하며, 지정하면 원근 보정을 끕니다.
  사람이 가리킨 범위가 페이지 워프보다 우선합니다.
- 닫힌 다각형을 다루는 constraint solver는 없습니다. 노치 하나의 치수를 따로
  확정하거나 변 사이 각도를 구속하는 방법은 없습니다.
- 프로파일 전체 치수는 축 정렬 바운딩 박스 기준이므로, 기울어진 부품을 찍은
  사진에서는 바운딩 폭이 실제 폭보다 큽니다.
- 프로파일이 축별로 늘어나도 그 안의 구멍 위치는 전역 스케일로 배치됩니다.
  부품 하나를 잘라내는 치수는 맞지만 조립 관계를 보장하지는 않습니다.
- `WIDTH`/`HEIGHT`는 손글씨에 표준 접두어가 없어 OCR에서 파싱되지 않으며
  수동 행으로만 생성됩니다.
- 닫힌 도형은 윤곽 후보이므로 잡음이 많은 사진에서는 사용자가 결과를
  확인해야 합니다. 도형 크기 하한은 이미지 짧은 변의 2%로 함께 계산되어
  고해상도 사진의 손글씨 획이 도형으로 잡히지 않습니다.
- polyline 조립은 정확히 동일한 endpoint를 가진 열린 단일 경로에만 적용되며,
  branch, 닫힌 loop, 가까울 뿐 일치하지 않는 gap은 안전하게 원본 `LINE`으로
  남습니다.
- 병합과 삭제는 확정 치수를 보호하기 위해 해당 값을 먼저 비워야 합니다.
- 선 다시 그리기는 원본 이미지에서 두 점을 드래그하며, 한 행을 선택하면
  기존 선을 교체하고 선택하지 않으면 새 미확인 선을 추가합니다. 여러 행을
  선택하고 다시 그리면 near-collinear 판정 없이 선택된 모든 조각이 방금 그린
  한 선으로 즉시 합쳐집니다. 검출이 한 획을 여러 조각으로 따로 인식하고 서로
  어긋나지 않은 선을 이렇게 정리할 수 있습니다.
- 편집 실행 취소/다시 실행은 선 병합·삭제·다시 그리기와 치수 확정만
  다루며, 새 사진 분석이나 전처리 방식 전환은 히스토리를 초기화합니다.
- `--auto-region`은 문서 경계를 추정하거나 투시 변환하지 않고 사진 가장자리를
  자르는 보수적 경로입니다. 기존 실제 샘플의 수동 ROI 경로는 그대로 유지됩니다.
- 완전한 일반 constraint solver는 아직 없습니다. 현재는 connected open line graph에
  길이·각도·방향·anchor 규칙을 적용합니다.
- DXF는 1차 산출물이며 DWG는 아직 지원하지 않습니다.
