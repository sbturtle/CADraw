# 기하 모델

## 좌표 계약

Phase 1의 좌표는 원본 이미지의 픽셀 좌표계를 따르는 로컬 2D 좌표입니다. x는 오른쪽으로, y는 아래쪽으로 증가합니다. 사용자가 최소 하나의 길이를 확정하기 전까지는 물리 치수가 아닙니다. 이렇게 하면 도면의 시각적 비율을 제작 치수로 소리 없이 변환하는 일을 막을 수 있습니다.

## 엔티티

- `LineEntity`: 원본 끝점, 정규화 끝점, 방향, 신뢰도.
- `PolylineEntity`: 순서 있는 점들과 연결 경로의 닫힘 여부 플래그. 현재는
  정확히 연결된 열린 선 체인의 표시/내보내기 결과로 생성됩니다.
- `CircleEntity`: 중심과 반지름.
- `ArcEntity`: 중심, 반지름, 시작 각, 끝 각.

모델은 태그된 `EntityType` 열거형과 불변 데이터클래스를 사용합니다. 내보내기 코드는 알려진 모든 변형을 명시적으로 매칭합니다.

`DrawingAnalysis.entities`는 image-space 검출 결과이며 `raw_start/raw_end`는
원본 이미지 pixel 좌표를 보존합니다. `CadDocument.entities`는 같은 stable
`EntityId`를 가진 CAD-space geometry입니다. 두 좌표계 사이의 명시적 관계는
`DetectedEntity`와 `CadEntity`로 표현되며, `CadEntity.source_entity_id`가
원본을 가리킵니다.

`CadEntity.status`는 `RESOLVED`, `UNRESOLVED`, `UNKNOWN_ANGLE` 중 하나입니다.
`inferred_for_preview`가 true인 좌표는 원본 방향이나 fallback scale을 사용한
표시용 좌표이며, 사용자가 확정한 제작 치수와 동일하게 취급하지 않습니다.

## 치수와 제약

`Dimension`은 `dimension_id`, `target_entity_ids`, `kind`, `parsed_value`,
`unit`, `source`, `confidence`, `status`를 가집니다. 상태는 기존
`UNKNOWN/NEEDS_REVIEW/CONFIRMED`와 함께 `UNSET`, `USER_ENTERED`,
`USER_CONFIRMED`, `AI_DETECTED`, `UNCERTAIN`, `INVALID`를 표현할 수 있습니다.
`Constraint`는 확정된 `Dimension`의 solver 입력이며 현재 `LENGTH`와 `ANGLE`
line constraint를 우선 지원합니다. AI/OCR은 이 모델에 제안값을 넣을 수 있지만
solver는 source provider를 알지 못합니다.

## 연결 재구성

`solver.reconstruct_geometry`는 연결된 line graph의 첫 line 시작점을 `(0, 0)`
anchor로 둡니다. 수평선은 source 방향의 ±X, 수직 하향선은 -Y를 유지하고,
대각선은 확정 angle과 `dx=L*cos(theta)`, `dy=L*sin(theta)`를 사용합니다.
명시 length가 없는 선은 상태를 `UNRESOLVED`로 두고 preview용 fallback만
계산합니다. angle이 없는 길이 확정 대각선은 원본 방향을 임시 사용하며
`UNKNOWN_ANGLE` 상태를 유지합니다.

계산 문서가 `CadEntity` 상태를 갖고 있을 때 DXF export는 기본적으로
`RESOLVED` geometry만 기록합니다. fixture 재현처럼 미확정 geometry가
필요한 경우에만 `export_dxf(..., allow_unresolved=True)`를 명시합니다.

## 정규화

`NormalizationConfig`가 모든 허용 오차를 관리합니다.

- `horizontal_angle_tolerance_degrees`
- `vertical_angle_tolerance_degrees`
- `endpoint_snap_distance`
- `line_merge_tolerance`
- `minimum_line_length`
- `keep_largest_connected_component`

`PipelineConfig.source_region`은 카메라 실험을 위해 알려진 도면 영역을 선택적으로 잘라낼 수 있습니다. 검출된 좌표는 원본 사진 좌표계로 다시 변환되며, 이는 물리적 원근 보정이 아닙니다.

엔진은 먼저 거의 수평/수직인 좌표를 스냅하고, 그다음 동일선 세그먼트를 병합하고, 마지막으로 가까운 끝점을 스냅합니다. 이 과정에 언어 모델은 관여하지 않습니다.

정규화 뒤의 `topology.py`는 endpoint 좌표가 정확히 같은 선만 그래프로
검사합니다. terminal이 두 개이고 나머지 vertex degree가 두 개인 열린 체인만
원본 순서와 무관하게 lexicographically 작은 terminal에서 시작하는 polyline으로
조립합니다. proximity만으로 연결된 선, branch, cycle은 원본 선으로 남습니다.

## 스케일링

`calculator.py`는 대상이 선인 첫 번째 확정 길이 치수를 선택합니다. 다음을 계산합니다.

```text
mm_per_pixel = 확정값 / 정규화된_대상_길이
```

모든 엔티티는 정규화된 로컬 원점을 기준으로 그 값만큼 스케일됩니다. 추가로 확정된 값들은 결과를 검증하며, 서로 어긋나면 경고를 생성합니다. 두 번째 숨은 스케일을 만들거나 사용자의 첫 번째 보정을 덮어쓰지 않습니다.

polyline stream은 export/preview 단계에서만 만들어집니다. 따라서 원본
`LineEntity` ID를 사용하는 dimension anchor와 calibration은 조립 전 모델을
계속 참조합니다.

## 사용자 선 편집

`line_editing.py`는 `DrawingAnalysis`만 입력으로 받아 새 분석 값을 반환합니다.
선택한 조각 병합은 기존 normalization의 collinear tolerance를 사용하고,
사전순으로 가장 작은 `EntityId`를 canonical ID로 보존합니다. 병합 대상의
미확인 length row는 canonical 선을 가리키는 하나의 row로 축약됩니다.

삭제는 선택한 `LineEntity`와 그 선만 대상으로 하는 미확인 dimension row를
함께 제거하고 connections를 다시 계산합니다. 이미 확정된 값이 대상이면
데이터 손실을 막기 위해 편집을 거부합니다.

`MainWindow`는 선 편집(병합/삭제/다시 그리기)과 치수 확정 직전의
`DrawingAnalysis`를 undo 스택에 보관합니다. 실행 취소/다시 실행은 저장된
analysis를 그대로 재설치하며 `manual_dimensions`를 다시 계산하지 않아 기존
행 식별자를 보존합니다. 새 사진 분석이나 전처리 방식 전환은 히스토리를
초기화합니다.

다시 그리기는 source-space의 두 raw endpoint를 받습니다. 선택 선이 하나면
기존 `EntityId`와 dimension anchor를 유지한 채 geometry를 교체하고, 선택 선이
없으면 충돌하지 않는 `manual_line_NNN`을 생성해 미확인 length row를 추가합니다.
입력은 기존 normalization을 다시 통과하므로 최소 길이와 축 snap 계약을 공유합니다.

## 알 수 없는 값

`Dimension.parsed_value`는 `float | None`입니다. 없는 지름, 반지름, 각도, 대상은 `None`으로 남고 미확정 상태를 가집니다. 내보내기는 연결되지 않은 치수 텍스트를 임의 좌표에 잘못 배치하는 대신 건너뜁니다.

## 이미지에서 닫힌 형상

`--detect-shapes`가 켜지면 검출 계층은 윤곽 후보를 다음 도메인 형상으로
변환합니다.

- 사변형 윤곽 -> `PolylineEntity(closed=True)`
- 원형 윤곽 -> `CircleEntity`

자동 inset으로 분석한 경우에도 후보의 좌표는 `_restore_source_coordinates`
정책에 따라 원본 이미지 좌표로 복원됩니다. 이는 후보 형상 보존 단계이며,
외곽 안에 포함된 작은 후보는 `GeometryRole.HOLE`로 표시됩니다. 해당 역할은
미리보기 색상과 DXF `HOLE` 레이어까지 보존되지만, 자동 치수 확정이나
기계적 feature 의미를 추측하지는 않습니다.

## 투시 보정과 OCR

`PerspectiveTransform`은 검출된 원본 사변형, 직사각형 분석 크기, 역행렬을
보존합니다. 분석 geometry와 `TextCandidate.bounds`는 이 역행렬을 통해
원본 사진 공간으로 복원됩니다. 페이지 경계가 없으면 변환은 `None`이고
`PERSPECTIVE_NOT_FOUND` 경고가 추가됩니다.

`TextCandidate`는 OCR 원문·box·신뢰도와, 검출기가 준 경우 기울어진 글자를
감싸는 사변형 `quad`를 보존합니다. `quad`도 `bounds`와 같은 역행렬로 원본 사진
좌표로 복원됩니다. 두 번째 판독 결과는 `alternate_text`에만 남고 `text`를 바꾸지
않습니다. 기존 dimension parser를 통과한 값만 `Dimension`으로 변환합니다. OCR 치수는 `NEEDS_REVIEW`이고
`target_entity_ids`가 비어 있어 사용자가 확인하기 전에는 DXF TEXT로 내보내지지
않습니다.

`association.suggest_dimension_target`은 파싱된 값이 있고 아직 대상이 없는
치수에 한해, 텍스트 박스 중심과 물리적으로 같은 종류의 geometry(길이/각도는
선, 지름은 원, 반지름은 원 또는 호) 중 가장 가까운 후보를 찾습니다. 가장
가까운 후보가 너무 멀거나 두 번째 후보와 거리 차이가 충분하지 않으면(모호)
아무 것도 연결하지 않습니다. 연결에 성공해도 `status`는 그대로 두고
`source`만 `"ocr_suggested"`로 표시해, 이는 사람이 검토할 제안일 뿐 확정이
아님을 유지합니다.

## 부분 호(open arc)

닫힌 사각형/원과 달리, 필렛이나 잘린 구멍처럼 완전히 닫히지 않은 원호는
`detection._contour_arc_candidate`가 별도로 처리합니다. 면적 기준을 만족하는
닫힌 도형이 되지 못한 윤곽에만 결정적 최소제곱 원 피팅(Kasa method)을
시도하며, 잔차·각도 범위·반지름이 모두 임계값을 통과해야 `ArcCandidate` ->
`ArcEntity`가 됩니다. `ArcEntity`도 `CircleEntity`와 같은 `role` 필드를 가지며
외곽 닫힌 도형 안에 있으면 `HOLE`로 분류되어 preview 색상과 DXF `HOLE`
레이어에 반영됩니다.
