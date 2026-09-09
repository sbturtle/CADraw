# 결정 0003: 안전한 열린 선 체인만 polyline으로 조립

## 상태

채택

## 맥락

Phase 1은 손도면의 선을 정규화하고 가까운 끝점의 `Connection`을 기록하지만,
DXF에는 여전히 여러 개의 `LINE`이 기록됩니다. 사용자가 연결된 구조를 하나의
깨끗한 경로로 선택할 수 있도록 polyline 표현이 필요합니다.

## 결정

`topology.py`에 결정론적 열린 체인 조립기를 둡니다.

- 입력 endpoint 좌표가 정확히 같을 때만 연결로 인정합니다.
- 연결 성분에 선이 두 개 이상 있고 terminal이 정확히 두 개여야 합니다.
- 나머지 vertex degree가 모두 2인 경우에만 열린 polyline을 생성합니다.
- 시작점은 두 terminal 중 `(x, y)`가 사전순으로 작은 점으로 고정합니다.
- branch, cycle, 고립 선, near-gap은 원본 `LINE`으로 남깁니다.
- `DrawingAnalysis`의 원본 선과 치수 대상 ID는 교체하지 않습니다.
- 조립은 preview와 DXF geometry stream에서만 적용합니다.

## 이유

`Connection`의 거리 tolerance는 사람이 보기 위한 관계 기록이지, 없는 edge를
만들 권한이 아닙니다. exact endpoint graph와 degree 조건을 사용하면 방향,
입력 순서, 손그림의 불확실성이 topology 추측으로 번지지 않습니다.

원본 선을 보존하면 현재 수동 치수 편집과 첫 확정 길이 calibration이 계속
선 ID를 대상으로 할 수 있습니다. 동시에 최종 DXF와 preview에서는 중복 선 없이
하나의 열린 `LWPOLYLINE`을 제공합니다.

## 결과

합성 fixture와 제공된 실제 사진은 각각 하나의 열린 `LWPOLYLINE`으로
내보내집니다. 모호한 topology는 자동으로 바뀌지 않으며, 이후 UI에서 사용자가
확인할 수 있는 별도 상태로 남길 수 있습니다.
