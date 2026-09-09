# 아키텍처

## 결정

CADraw는 네 개의 경계로 나뉜 작은 데스크톱 애플리케이션입니다.

```text
이미지 경로
  -> detection.py
  -> normalization.py
  -> models.DrawingAnalysis
  -> dimensions.py / calculator.py
  -> preview.py 와 export.py
```

`ui/`는 이 흐름을 조합하기만 하고 기하 계산을 직접 하지 않습니다. 덕분에 PySide6가 설치되지 않은 환경에서도 도메인 테스트와 DXF 왕복(round-trip) 테스트를 실행할 수 있습니다.

## 모듈 책임

| 모듈 | 책임 |
|---|---|
| `models.py` | 불변 좌표점, 엔티티, 치수, 경고, 분석 문서 |
| `detection.py` | 이미지 로딩, 전처리, 가장자리 필터링, OpenCV 선 후보 추출 |
| `normalization.py` | 방향 분류, 스냅, 동일선 병합, 연결 정보 생성 |
| `dimensions.py` | 길이/지름/반지름/각도 텍스트의 안전한 파싱 |
| `calculator.py` | 확정 치수 기반 보정과 결정적 좌표 변환 |
| `preview.py` | CAD 엔티티의 순수 Pillow 렌더링 |
| `export.py` | ezdxf 직렬화와 재읽기 검증 |
| `vision.py` | `VisionAnalyzer` 프로토콜과 OCR 없는 목(mock) 어댑터 |
| `ui/` | 원본 이미지, 미리보기, 수동 확인 입력, 액션 |

## 데이터 흐름

`DrawingAnalysis`가 중간 표현입니다. 검출된 선 엔티티에는 원본 좌표를 그대로 보존하고, 미리보기와 내보내기에는 정규화된 좌표를 사용합니다. 치수 값은 없을 수 있으며(nullable), 각 치수는 확인 상태(review status)를 함께 가집니다.

제공된 카메라 샘플의 경우, `PipelineConfig.source_region`이 분석 범위를 알려진 종이 도면 영역으로 제한하고 검출된 좌표를 원본 사진 좌표계로 복원합니다. 이는 조정 가능한 실험 설정이며 자동 원근 보정이 아닙니다.

AI/비전은 절대 DXF 좌표를 반환하지 않습니다. 향후 어댑터가 파싱된 엔티티와 텍스트 후보를 반환할 수는 있지만, `calculator.py`와 `export.py`는 결정적 소비자로 유지됩니다.

## 배포 판단

이 PoC는 일반 Python 환경과 PySide6를 사용합니다. 데스크톱 경로가 안정화된 뒤 `PyInstaller` 또는 동등한 Windows 패키징 단계로 설치 파일을 만들 수 있습니다. DWG 변환은 DXF 표현과 사용자 확인 흐름이 신뢰할 수 있게 될 때까지 미룹니다.
