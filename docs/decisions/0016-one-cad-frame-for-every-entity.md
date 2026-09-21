# 결정 0016: 시트의 모든 entity는 하나의 CAD 프레임으로 배치한다

## 상태

채택

## 맥락

전수 감사에서 4번 스캔에 줄기 선 길이 70과 외곽 폴리곤 폭 110을 함께 확정해
보았습니다.

```text
CAD line_raw_006  LINE (0.0,0.0)-(0.0,70.0)          resolved
CAD shape_raw_001 POLY x[28.2,138.2] y[70.1,170.1]  resolved
```

사진에서 줄기는 외곽선의 위 변에서 위로 솟아 있습니다. CAD에서는 줄기 선이
원점에서 y가 커지는 방향으로 놓이고, 외곽 폴리곤은 `픽셀 × 스케일` 위치에 y가
아래로 커지는 채로 놓였습니다. 같은 부품의 두 표현이 서로 뒤집혀 다른 곳에
있었습니다.

원인은 `solver`가 프레임을 두 개 쓰고 있었기 때문입니다.

- 선 성분: `_component_geometry`가 첫 성분의 root 시작점을 `(0, 0)`에 두고
  `_line_vector`가 y를 뒤집은 벡터(y-up)로 전파합니다. 두 번째 이후 성분은
  root를 `root.start.scaled(scale)`(image y-down)에 앵커한 뒤 y-up 벡터로
  뻗어, 자기 자리에서 뒤집힙니다.
- 도형: `_sized_non_line`이 `origin × scale`(image y-down)에 놓고 반지름만
  키웁니다. 호의 각도도 그대로였습니다.

`CAD 결과` preview가 y를 뒤집어 그리므로 선만 있는 문서에서는 드러나지 않았고,
ADR-0012로 닫힌 도형이 기본 검출되면서 문제가 되었습니다.

같은 preview 뒤집기 때문에, 확정값이 없어 스케일이 없는 문서(image 좌표
그대로)는 `CAD 결과`에서 상하가 뒤집혀 보였습니다.

## 결정

`_Frame(anchor, scale)` 하나가 문서 전체의 픽셀→mm 매핑입니다.

```python
to_cad(p) = ((p.x - anchor.x) * scale, (anchor.y - p.y) * scale)
```

- 선 성분의 root는 모두 `frame.to_cad(root.start)`에 앵커합니다. 첫 성분의
  root가 anchor이므로 기존 선 문서의 원점은 그대로입니다.
- 폴리곤은 왼쪽 위 모서리를 프레임으로 놓고, 자기 축 배율(ADR-0014)로 오른쪽·
  아래로 자랍니다(CAD에서는 y가 줄어드는 방향).
- 원·호의 중심은 프레임으로 놓습니다. 호의 각도는 y 반전에 맞춰
  `start' = -end`, `end' = -start`(mod 360)로 바꿉니다. 사진에서 시계 방향으로
  돌던 호는 CAD에서 반시계 방향입니다.
- 선이 하나도 없는 시트는 전체 bounding box의 왼쪽 아래를 anchor로 삼아 도면이
  1사분면에 놓입니다.
- 스케일이 없는 문서는 이전과 같이 image 좌표를 유지하며, `MainWindow`의
  `CAD 결과` preview는 스케일이 있을 때만 y를 뒤집습니다.

## 측정

같은 4번 스캔 입력으로:

```text
CAD line_raw_006  LINE (0.0,0.0)-(0.0,70.0)           resolved
CAD shape_raw_001 POLY x[-78.6,31.4] y[-25.6,74.4]    resolved
CAD shape_raw_002 CIRC c=(81.1,-21.4)                unresolved
```

줄기 위 끝 y=70과 외곽선의 위 변 y=74.4가 사진에서처럼 거의 같은 높이이고,
구멍은 아래쪽 가로대(y<0) 안에 있습니다.

## 한계

- 프로파일이 축별로 늘어나면 그 안의 구멍 위치는 여전히 전역 스케일로
  배치됩니다(ADR-0014의 한계 그대로).
- anchor는 첫 선 성분의 root 시작점이라 entity ID 순서에 의존합니다. 선을
  삭제하면 원점이 다른 선으로 옮겨질 수 있으며, 상대 위치는 유지됩니다.
