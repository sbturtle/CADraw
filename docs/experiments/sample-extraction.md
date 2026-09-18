# 샘플 추출 결과

`scripts/render_sample_report.py`가 생성합니다. 각 이미지는 왼쪽이
파이프라인이 추출한 geometry를 원본 사진 위에 겹친 것이고, 오른쪽이
같은 스케치의 레퍼런스 DXF입니다. 레퍼런스는 다중 뷰 재배치와 해석된
치수를 포함한 완성형 해석본이므로 구조를 비교하는 용도이며 픽셀 단위
정답지가 아닙니다.

선 색: 빨강 `LINE`, 파랑 `LWPOLYLINE`, 초록 `CIRCLE`, 주황 `ARC`.

```powershell
uv run python scripts/render_sample_report.py
```

## 요약

| 샘플 | 선택 variant | 상태 | 추출 | 레퍼런스 |
| --- | --- | --- | --- | --- |
| 1 | `blur_otsu` | `READY` | line 4 | LWPOLYLINE 1 |
| 2 | `blur_otsu` | `READY` | arc 1, line 2, polyline 6 | LWPOLYLINE 3 |
| 3 | `blur_otsu` | `READY` | line 9, polyline 2 | LWPOLYLINE 10 |
| 4 | `clahe_adaptive` | `READY` | circle 1, line 3, polyline 3 | ARC 2, CIRCLE 2, LINE 2, LWPOLYLINE 1 |
| 5 | `shadow_removed_adaptive` | `READY` | arc 1, circle 2, line 8, polyline 13 | ARC 24, CIRCLE 7, LINE 24, LWPOLYLINE 1 |
| 6 | `clahe_adaptive` | `READY` | line 1, polyline 4 | LWPOLYLINE 10 |
| 7 | `shadow_removed_adaptive` | `READY` | arc 1, line 4, polyline 9 | ARC 8, CIRCLE 13, LINE 8, LWPOLYLINE 4 |

## 샘플 1

![샘플 1](sample-extraction/1.png)

- 선택 variant: `blur_otsu`
- 상태: `READY`
- 추출: line 4
- 레퍼런스: LWPOLYLINE 1
- 경고: `PERSPECTIVE_REJECTED`, `UNKNOWN_DIMENSION`

## 샘플 2

![샘플 2](sample-extraction/2.png)

- 선택 variant: `blur_otsu`
- 상태: `READY`
- 추출: arc 1, line 2, polyline 6
- 레퍼런스: LWPOLYLINE 3
- 경고: `PERSPECTIVE_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `UNKNOWN_DIMENSION`

## 샘플 3

![샘플 3](sample-extraction/3.png)

- 선택 variant: `blur_otsu`
- 상태: `READY`
- 추출: line 9, polyline 2
- 레퍼런스: LWPOLYLINE 10
- 경고: `PERSPECTIVE_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `UNKNOWN_DIMENSION`

## 샘플 4

![샘플 4](sample-extraction/4.png)

- 선택 variant: `clahe_adaptive`
- 상태: `READY`
- 추출: circle 1, line 3, polyline 3
- 레퍼런스: ARC 2, CIRCLE 2, LINE 2, LWPOLYLINE 1
- 경고: `PRINTED_RULING_REMOVED`, `PERSPECTIVE_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `UNKNOWN_DIMENSION`

## 샘플 5

![샘플 5](sample-extraction/5.png)

- 선택 variant: `shadow_removed_adaptive`
- 상태: `READY`
- 추출: arc 1, circle 2, line 8, polyline 13
- 레퍼런스: ARC 24, CIRCLE 7, LINE 24, LWPOLYLINE 1
- 경고: `PERSPECTIVE_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `UNKNOWN_DIMENSION`

## 샘플 6

![샘플 6](sample-extraction/6.png)

- 선택 variant: `clahe_adaptive`
- 상태: `READY`
- 추출: line 1, polyline 4
- 레퍼런스: LWPOLYLINE 10
- 경고: `PERSPECTIVE_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `UNKNOWN_DIMENSION`

## 샘플 7

![샘플 7](sample-extraction/7.png)

- 선택 variant: `shadow_removed_adaptive`
- 상태: `READY`
- 추출: arc 1, line 4, polyline 9
- 레퍼런스: ARC 8, CIRCLE 13, LINE 8, LWPOLYLINE 4
- 경고: `PERSPECTIVE_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `UNKNOWN_DIMENSION`
