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

다시 그리기는 source-space의 두 raw endpoint를 받습니다. 선택 선이 하나면
기존 `EntityId`와 dimension anchor를 유지한 채 geometry를 교체하고, 선택 선이
없으면 충돌하지 않는 `manual_line_NNN`을 생성해 미확인 length row를 추가합니다.
입력은 기존 normalization을 다시 통과하므로 최소 길이와 축 snap 계약을 공유합니다.

## 알 수 없는 값

`Dimension.parsed_value`는 `float | None`입니다. 없는 지름, 반지름, 각도, 대상은 `None`으로 남고 미확정 상태를 가집니다. 내보내기는 연결되지 않은 치수 텍스트를 임의 좌표에 잘못 배치하는 대신 건너뜁니다.
