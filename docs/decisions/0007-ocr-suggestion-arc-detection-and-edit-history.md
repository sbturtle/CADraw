# 결정 0007: OCR 치수 제안, 부분 호 검출, 편집 히스토리

## 상태

채택

## 맥락

세 가지 알려진 한계가 실제 사용을 가로막고 있었습니다.

1. OCR이 인식한 치수는 항상 `target_entity_ids = ()`로 남아 어떤 geometry도
   가리키지 못했고, `manual_dimensions`는 대상 없는 행을 만들지 않으므로
   UI 확인 테이블에 아예 나타나지 않았습니다.
2. 닫힌 사각형/원 윤곽만 도형 후보로 변환되어, 모따기·필렛·잘린 구멍처럼
   사진에서 흔한 부분 호(open arc) 스트로크는 조용히 버려졌습니다.
3. 선 병합·삭제·다시 그리기, 치수 확정은 모두 immutable analysis를 교체하는
   동작이었지만 되돌릴 방법이 없어 실수한 편집을 취소하려면 처음부터
   다시 분석해야 했습니다.

## 결정

### OCR 치수 제안

`association.suggest_dimension_target`은 파싱된 값이 있고 아직 대상이 없는
치수에 한해, 같은 물리적 종류(`LENGTH`/`ANGLE`→선, `DIAMETER`→원,
`RADIUS`→원 또는 호)의 후보 중 텍스트 상자 중심과 가장 가까운 geometry를
찾습니다. 가장 가까운 후보가 `140px`보다 멀거나, 두 번째로 가까운 후보와
`35%` 이내로 근접해 모호하면 아무 것도 연결하지 않고 `target_entity_ids`를
비워 둡니다. 연결에 성공해도 `status`는 `NEEDS_REVIEW`를 유지하고
`source`만 `"ocr_suggested"`로 남겨, 값을 실제로 확정하는 것은 여전히
사람의 몫입니다. `manual_dimensions`는 이제 `CircleEntity`에는 `DIAMETER`,
`ArcEntity`에는 `RADIUS` 확인 행을 추가로 만들어 이 제안이 테이블에
나타날 자리를 제공합니다. `calculator._append_circular_constraint_warnings`는
확정된 지름/반지름을 스케일된 geometry와 비교해 기존 길이 검증과 동일한
`CONSTRAINT_MISMATCH` 규칙을 적용합니다.

### 부분 호 검출

`detection._contour_arc_candidate`는 면적 기준(`minimum_shape_area`)을
만족하지 못해 사각형/원 후보가 되지 못한 윤곽에 대해서만 Kasa 최소제곱
원 피팅을 시도합니다. 다음 조건을 모두 만족해야 `ArcCandidate`가 됩니다.

- 점 개수가 충분하고(`24` 이상), 피팅 반지름이 관측 범위에 비해 과도하게
  크지 않습니다.
- RMS 잔차가 `max(2.5px, 반지름 * 8%)` 이내로 원에 가깝습니다.
- 각도 범위가 `50°`~`320°`로, 완전한 원(따로 검출됨)도 잡음도 아닙니다.

`ArcCandidate`/`ArcEntity`는 `CircleCandidate`와 동일한 `role` 필드를 가지며
외곽 닫힌 도형 안에 있으면 `classify_shape_holes`가 동일하게 `HOLE`로
표시합니다. Preview는 `HOLE` 파란색과, CAD 모드(`invert_y`)에서 각도를
`-end, -start`로 반사해 실제 방향을 유지합니다. DXF export는 `HOLE` 역할일
때 원과 같은 `HOLE` 레이어에 씁니다.

### 편집 히스토리

`MainWindow`는 `_undo_stack`/`_redo_stack`에 편집 직전의 `DrawingAnalysis`
스냅샷을 보관합니다. 선 병합·삭제·다시 그리기(`_apply_line_edit`)와 치수
확정(`_apply_dimension_text`)만 스냅샷을 남기며, 새 사진 분석이나 전처리
방식 전환은 히스토리를 초기화합니다. 실행 취소/다시 실행은 저장된 analysis를
그대로 재설치할 뿐 `manual_dimensions`를 다시 계산하지 않아, 이미 정리된
행 식별자를 그대로 보존합니다. 버튼은 표준 `Ctrl+Z`/`Ctrl+Y` 단축키를
사용합니다.

## 이유

세 결정 모두 기존 경계를 그대로 유지합니다. OCR 제안은 자동 확정이 아니라
제안이며, 모호하면 침묵 대신 빈 대상으로 남습니다. 호 검출은 새로운 확률
모델이 아니라 결정적 최소제곱 피팅이며, 임계값을 넘지 못하면 후보를 만들지
않습니다. 편집 히스토리는 별도 command 패턴 없이 이미 immutable한
`DrawingAnalysis` 스냅샷을 재사용해 정확성 위험을 최소화합니다.

## 한계

- 제안 거리 임계값과 모호성 마진은 고정 상수이며, 사진 배율에 따라
  조정하지 않습니다.
- 호 검출은 페이지 경계에 닿은 구멍(자재 가장자리에 걸친 반원)은 다른
  닫힌 도형과 동일한 여백 규칙 때문에 여전히 제외합니다.
- 편집 히스토리는 병합·삭제·다시 그리기·치수 확정만 다루며, 전처리 방식
  전환이나 새 사진 불러오기는 취소 대상이 아닙니다.
