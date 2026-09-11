# 결정 0006: 확정 치수 우선 CAD 좌표 재구성

## 배경

OpenCV가 반환하는 선은 image pixel 좌표입니다. 이를 그대로 전역 scale만
곱하면 연결된 선마다 사용자가 입력한 실제 길이를 보장할 수 없습니다.
대각선 각도처럼 사람이 아직 확인하지 않은 값도 제작 데이터로 오인될 수
있습니다.

## 결정

`solver.py`는 `DimensionStatus.CONFIRMED`인 `Dimension`만
`Constraint`로 변환한다. connected line graph의 첫 line 시작점을
`(0, 0)`으로 고정하고, 각 edge는 확정 length와 angle로 전진한다.

- horizontal은 source 방향의 X축을 유지한다.
- vertical 하향은 CAD Y를 음수로 계산한다.
- diagonal은 확정 angle이 있으면 삼각함수를 사용한다.
- length가 없으면 `UNRESOLVED`다.
- diagonal length만 있고 angle이 없으면 원본 방향을 표시용으로만 사용하고
  `UNKNOWN_ANGLE`을 유지한다.

`CadEntity`는 `source_entity_id`, CAD geometry, constraint, resolution state를
담아 image-space `DetectedEntity`와 최종 geometry를 분리한다. AI/OCR은
향후 동일한 `Dimension` 경계에 `AI_DETECTED` 제안을 넣을 수 있지만 solver와
직접 결합하지 않는다.

## 결과

연결된 L shape의 각 segment가 서로 다른 mm 치수를 가져도 shared endpoint가
정확히 유지된다. CAD preview와 DXF는 재구성된 좌표를 사용하고, DXF
round-trip validation은 다시 읽은 실제 line/segment 길이를 확인한다.
