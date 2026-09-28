# 아키텍처

## 결정

CADraw는 네 개의 경계로 나뉜 작은 데스크톱 애플리케이션입니다.

```text
이미지 경로
  -> perspective.py / detection.py / local_ocr.py / ocr.py
  -> text_filter.py
  -> normalization.py
  -> models.DrawingAnalysis
  -> dimensions.py / calculator.py / solver.py
  -> line_editing.py
  -> topology.py
  -> preview.py 와 export.py
```

`ui/`는 이 흐름을 조합하기만 하고 기하 계산을 직접 하지 않습니다. Preview
click은 stable ID만 전달하고, dimension editor는 typed `Dimension`만 갱신합니다.
덕분에 PySide6가 설치되지 않은 환경에서도 도메인 테스트와 DXF 왕복
(round-trip) 테스트를 실행할 수 있습니다.

## 모듈 책임

| 모듈 | 책임 |
|---|---|
| `models.py` | 불변 좌표점, 엔티티, 치수, 경고, 분석 문서 |
| `detection.py` | 이미지 로딩, 전처리, 가장자리 필터링, OpenCV 선 후보 추출 |
| `perspective.py` | 페이지 사변형 검출, 투시 워프, 원본 좌표 역변환 |
| `ocr.py` | OCR·보조 판독기 Protocol, Tesseract 대체 경로, 안전한 dimension parser 연결과 교차 판독 |
| `local_ocr.py` | RapidOCR PP-OCRv6 로컬 판독기와 OCR 백엔드 탐색 |
| `ollama_reader.py` | 선택적 로컬 Ollama 비전 모델의 크롭 단위 두 번째 판독 |
| `text_filter.py` | 숫자로 증명된 글자 영역 안에만 있는 획 후보 제거 |
| `normalization.py` | 방향 분류, 스냅, 동일선 병합, 연결 정보 생성 |
| `dimensions.py` | 길이/지름/반지름/각도 텍스트의 안전한 파싱 |
| `calculator.py` | 분석 상태를 계산 문서로 조합하고 경고를 관리 |
| `solver.py` | 확정 length/angle constraint로 connected line CAD 좌표 재구성 |
| `line_editing.py` | 병합·삭제·수동 다시 그리기의 immutable 분석 변환 |
| `topology.py` | 정확히 연결된 열린 선 체인을 순서 있는 polyline으로 조립 |
| `preview.py` | CAD 엔티티의 순수 Pillow 렌더링 |
| `export.py` | ezdxf 직렬화와 재읽기 검증 |
| `vision.py` | `VisionAnalyzer` 프로토콜과 로컬 전용 어댑터 |
| `ui/` | 원본 이미지, 미리보기, 수동 확인 입력, 선 편집 액션 |
| `ui/workbench_view.py` | native toolbar, viewport, verification panel 조합 |
| `ui/verification_panel.py` | 다중 행 선택, 치수 행, 선 편집 버튼 |
| `ui/source_image_widget.py` | 원본 이미지 fit, 선택 overlay, source-space 드래그 입력 |
| `ui/line_edit_controller.py` | UI callback과 immutable 선 편집 명령 연결 |

## 데이터 흐름

`DrawingAnalysis`가 중간 표현입니다. 검출된 선 엔티티에는 원본 좌표를 그대로 보존하고, 미리보기와 내보내기에는 정규화된 좌표를 사용합니다. 치수 값은 없을 수 있으며(nullable), 각 치수는 확인 상태(review status)를 함께 가집니다.

사용자 선 편집은 `line_editing.py`가 `DrawingAnalysis`를 새 immutable 값으로
교체하는 별도 경계입니다. 병합은 선택한 near-collinear `LineEntity`를 하나의
canonical ID로 합치고 미확인 치수 행을 하나로 정리합니다. 삭제는 해당 geometry와
그 geometry에만 연결된 미확인 행을 제거합니다. 다시 그리기는 원본 좌표 두 점을
기존 선 교체 또는 새 선 추가로 정규화하며, 모든 경우 connections를 재생성한 뒤
기존 calculator/preview/export 흐름으로 돌아갑니다.

`topology.py`는 `DrawingAnalysis.entities` 또는 `CadDocument.entities`를
교체하지 않습니다. 정확히 같은 endpoint로 연결되고 degree-one terminal이
정확히 두 개인 열린 경로만 `PolylineEntity`로 만든 표시/내보내기용 스트림에
넣습니다. 따라서 원본 `LineEntity` ID를 대상으로 하는 치수 편집과 calibration은
그대로 유지되고, branch·loop·gap은 추측 없이 선으로 남습니다.

제공된 카메라 샘플의 경우, `PipelineConfig.source_region`이 분석 범위를 알려진 종이 도면 영역으로 제한하고 검출된 좌표를 원본 사진 좌표계로 복원합니다. `auto_source_region`을 켜면 사진 가장자리 inset을 결정적으로 선택할 수 있습니다. `perspective_correction`은 검출 가능한 페이지 사변형을 워프하고 역행렬을 provenance로 보존하며, 경계를 찾지 못하면 원본 경로로 안전하게 돌아갑니다. `detect_closed_shapes`는 선 정규화 경로와 분리된 윤곽 검출로 `PolylineEntity`와 `CircleEntity` 후보를 추가하고, 포함 관계가 명확한 후보에만 `HOLE` 역할을 부여합니다. `ocr_enabled`는 OCR 원문과 box를 보존하고 parser를 통과한 미확정 치수만 추가합니다. OCR이 켜지면 선 정규화 전에 `text_filter`가 숫자로 증명된 글자 영역 안에만 있는 획 후보를 빼고, `ocr_cross_check_model`이 있으면 각 값을 로컬 비전 모델로 다시 읽어 불일치를 `UNCERTAIN`으로 낮춥니다(ADR-0018).

AI/비전은 절대 DXF 좌표를 반환하지 않습니다. 향후 어댑터가 파싱된 엔티티와 텍스트 후보를 반환할 수는 있지만, `calculator.py`와 `export.py`는 결정적 소비자로 유지됩니다.

## 배포 판단

이 PoC는 일반 Python 환경과 PySide6를 사용합니다. 데스크톱 경로가 안정화된 뒤 `PyInstaller` 또는 동등한 Windows 패키징 단계로 설치 파일을 만들 수 있습니다. DWG 변환은 DXF 표현과 사용자 확인 흐름이 신뢰할 수 있게 될 때까지 미룹니다.
