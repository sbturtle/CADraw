# 파이프라인

## Phase 1 수직 슬라이스

1. 사용자가 JPG/PNG를 엽니다.
2. OpenCV가 회색조 픽셀을 읽고, 설정 가능한 페이지 가장자리 후보를 제외한 뒤 블러 처리한 Canny 엣지를 만듭니다.
3. 확률적 허프(Hough) 검출이 원본 선 후보를 생성합니다.
4. 후보를 결정적으로 정렬하고, 거의 동일선상에 있으면 병합합니다.
5. 방향을 수평, 수직, 대각선으로 분류합니다.
6. 가까운 끝점을 스냅하고 연결 기록을 생성합니다.
7. 정확히 같은 endpoint로 연결된 단일 열린 체인만 `topology.py`가 순서 있는
   `PolylineEntity`로 조립합니다. 분석 모델의 원본 선과 치수 대상 ID는 보존합니다.
8. 조립된 geometry stream을 `DrawingAnalysis`의 깔끔한 미리보기와 나란히 보여줍니다.
9. OCR 값을 가정하지 않습니다. 수평/수직 선은 각각 `0°`/`90°`로
   자동 확정하고, 대각선의 표기 없는 angle만 수동 입력 대상으로 남깁니다.
10. 확인 행을 선택하면 대응 선을 원본 사진과 preview에서 highlight하고, 방향과
    로컬 길이를 표시합니다.
11. 사용자가 병합·삭제 버튼으로 선택한 선을 교정하거나, 원본 이미지에서
    드래그해 기존 선을 다시 그리고 새 선을 추가합니다. 편집 결과는 immutable
    analysis 교체 후 connections를 재생성합니다.
12. 사용자가 실제 길이를 입력하고 확정으로 표시합니다.
13. 확정된 line length/angle을 `solver.py`가 사용해 첫 entity 시작점을
    `(0, 0)`으로 하는 CAD-space mm 좌표를 재구성합니다. 연결된 선은 공유
    endpoint를 유지하며, 명시 치수가 없는 선은 `UNRESOLVED`로 남습니다.
14. angle이 없는 diagonal은 원본 방향을 preview에만 사용하고
    `UNKNOWN_ANGLE`과 `inferredForPreview`를 유지합니다.
15. 대각선 angle을 입력하면 `각도 승인 후 다시 그리기`를 눌러 명시적으로
    재작성 권한을 승인한 뒤 CAD geometry를 다시 계산합니다. `180°`까지
    허용하며, 취소하면 기존 geometry를 유지합니다.
16. 미리보기를 다시 계산합니다. `검출 결과`는 image-space, `CAD 결과`는
    실제 입력 치수와 CAD y-up 표시를 사용해 원본 방향을 보존합니다.
17. `ezdxf`가 CAD-space 기하를 `GEOMETRY`에, 확인된 치수 라벨을
    `DIMENSION`에, 보조 텍스트를 `TEXT`에 기록합니다.
18. 내보내기 후 파일을 다시 읽어 엔티티 수, polyline vertex 순서, 각
    line/segment 실제 길이를 검증한 다음 성공을 보고합니다.

계산 문서의 미확정 geometry는 일반 UI export에서 제외됩니다. `scripts/run_poc.py`
는 실제 샘플의 preview와 경고 동작을 재현하기 위해서만
`allow_unresolved=True`를 명시합니다.

앱 기본값은 끊긴 획을 연결할 후보를 넉넉히 수집하고, 축 정렬 허용오차와
endpoint 병합을 적용한 뒤 가장 큰 연결 선 성분만 유지합니다. 이렇게 하면
페이지/배경과 손글씨 라벨 획을 줄이면서 원본 사진 좌표를 보존합니다.
제공된 사진의 POC runner는 추가로 `source_region=(500, 680, 550, 1150)`을
사용합니다. 일반적인 원근/ROI 자동 검출은 아직 구현되지 않았습니다.

## 도메인 상태로서의 실패

파이프라인은 일반적인 불확실성에 대해 예외를 던지지 않습니다.

- 후보 선 없음 -> `LOW_CONFIDENCE_GEOMETRY`
- 보정 정보 없음 -> `UNKNOWN_DIMENSION`
- 대상 없는 치수 -> `UNCONNECTED_DIMENSION`
- 파서가 값을 읽지 못함 -> `UNCERTAIN_TEXT`
- 확정 치수가 스케일된 기하와 불일치 -> `CONSTRAINT_MISMATCH`

polyline 조립은 proximity connection을 다시 사용하지 않습니다. branch, cycle,
고립 선, near-gap처럼 열린 단일 경로가 아닌 topology는 변경하지 않습니다.

입출력 실패와 잘못된 DXF 파일은 각 경계에서 예외로 처리합니다. 알 수 없는 제작 정보는 예외가 아닙니다.

## 확장 샘플 경로

`--auto-region`은 입력 이미지의 바깥 여백을 결정적으로 제외하고, 검출된
좌표를 원본 이미지 좌표로 되돌립니다. 명시적인 `source_region`이 있으면
항상 명시 설정이 우선합니다. `--detect-shapes`는 이 영역의 이진 윤곽에서
닫힌 사각형은 `PolylineEntity`, 원은 `CircleEntity`로 추가합니다.

`--perspective`가 켜지면 전체 사진에서 페이지 사변형을 먼저 검출하고,
실패하면 명시적/자동 `source_region` 안에서 한 번 더 찾습니다. 찾은 사변형은
직사각형으로 워프하고, 역행렬로 모든 도형과 OCR box를 원본 사진 좌표로
되돌립니다. 워프 stream의 geometry evidence가 원본 stream보다 나쁘면
`PERSPECTIVE_REJECTED`를 남기고 원본 stream을 사용합니다. 페이지 경계를
찾지 못한 경우에는 `PERSPECTIVE_NOT_FOUND`와 `DOCUMENT_NOT_FOUND`를 남깁니다.

`--ocr`는 설치된 Tesseract TSV 결과를 `TextCandidate`로 보존하고, 기존
`parse_dimension_text`를 거친 값만 `NEEDS_REVIEW` 치수로 추가합니다.
`association.suggest_dimension_target`이 텍스트 상자와 같은 물리적 종류의
geometry 중 유일하게 가까운 후보가 있을 때만 `target_entity_ids`를 채우며,
`status`는 `NEEDS_REVIEW`로 유지하고 `source`만 `ocr_suggested`로 남깁니다.
후보가 없거나 두 후보가 비슷하게 가까우면 대상을 비워 두므로, 검토 전 DXF
TEXT가 생성되지 않는다는 원칙은 그대로입니다.

그래프 용지, 인쇄 표, 손글씨가 섞인 사진에서는 후보가 남을 수 있으므로
preview와 DXF를 확인한 뒤 치수를 수동으로 확정해야 합니다. 닫힌 형상
후보가 같은 크기 정사각형이 양축에서 한 주기로 반복되는 격자로 **증명되면**
그 칸들을 후보에서 제거하고, `GRAPH_PAPER_DETECTED` 경고에 제거 개수를
보고합니다. 증명된 격자는 더 이상 분석을 막지 않습니다. 반대로 증명되지
않은 채 형상이 선보다 과도하게 많기만 하면 후보를 그대로 보존한 상태로
`TOO_MUCH_NOISE`를 남기고 DXF 저장을 보류해, 사람이 판단하게 합니다.
외곽 닫힌 형상 안에 포함된 작은 형상은 `HOLE` 역할로 분류하고 DXF
`HOLE` 레이어에 씁니다.

완전히 닫힌 사각형/원 윤곽이 되지 못한 모따기·필렛·잘린 구멍 스트로크는
`detection._contour_arc_candidate`가 결정적 최소제곱 원 피팅으로 처리합니다.
점 개수, 피팅 잔차, 각도 범위(`50°`~`320°`)가 모두 임계값을 통과해야
`ArcEntity`가 되며, 완전한 원이나 잡음 곡선은 대상이 아닙니다. `ArcEntity`도
`CircleEntity`와 같은 `HOLE` 분류·색상·DXF 레이어 규칙을 공유합니다.

사각형도 원도 아닌 닫힌 윤곽은 `detection._closed_polygon_candidate`가 볼록
껍질 대비 충실도로 판정해 닫힌 `PolylineCandidate`로 만듭니다. 모서리 개수를
가정하지 않으므로 T·L·노치 프로파일이 살아남습니다. 코너 허용 오차는 사각형
판정용(`둘레의 3.5%`)보다 훨씬 촘촘한 `둘레의 1%`를 따로 쓰는데, 거친 값을
그대로 쓰면 노치 프로파일이 삼각형으로 뭉개집니다. 크기 게이트는 축당
`0.92`로, 페이지 경계는 테두리 접촉 조건으로 이미 배제되므로 도면의 주
외곽선이 크기만으로 탈락하지 않습니다.

인쇄된 양식 furniture는 두 가지 증명으로 제거합니다. `is_page_frame_rule`은
축에 정렬되고 시트의 `60%` 이상을 가로지르며 가장자리 `22%` 이내에 있고
다른 선과 끝점에서 만나지 않는 선만 버립니다. 고립 조건이 없으면 시트
폭만큼 긴 부재를 가진 부품의 geometry가 사라지므로, 네 조건은 모두
필요합니다. 판정은 정규화 이후에 하는데, 양식의 괘선이 흔히 짧은 동일선상
조각으로 검출되고 near-collinear 병합이 그것을 다시 하나의 긴 선으로
되돌리기 때문입니다.

`table_filter.inspect_printed_table_shapes`는 깊이가 비슷한 사각형 3개
이상이 한 축에서 모서리를 맞대고 타일링할 때만 표로 증명합니다. 타일링은
기울어진 스캔에서도 유지되는 signature이며, 상·하단 공유만으로는 무관한
두 도형이 우연히 같은 높이를 갖는 경우와 구분되지 않습니다. 체인은 띠마다
따로 만듭니다. 전체 후보를 한 번에 훑으면 시트 다른 곳의 도형이 표 칸
사이에 끼어들어 체인을 끊습니다. 증명 후에는 같은 띠의 나머지 사각형과
칸 전체를 감싸는 외곽선도 함께 제거하며, 원과 중첩 프로파일은 대상이
아닙니다.

## 사진 전처리와 검토 선택

variant 평가 전에 `page_cleanup.clean_page`가 원본 grayscale에서 증명된 모눈
괘선을 지웁니다. 괘선은 연필/펜보다 흐리므로 적응형 임계값이 이를 완전 강도
스트로크로 승격시키는데, 그 결정을 variant 선택에 맡기지 않기 위해 입력
단계에서 제거합니다. 제거 조건은 레일이 획보다 얇고, 기울기 보정 후 긴
직선 열림을 통과하며, **양축 모두**에서 기본 주기의 정수배로 주기 패밀리가
증명되는 것입니다. 한 방향만 규칙적인 반복은 도면 자체의 특징일 수 있어
건드리지 않으므로, 격자 없는 길고 곧은 손그림 직선은 보존됩니다. 제거가
일어나면 `PRINTED_RULING_REMOVED`가 주기와 가로·세로 레일 수를 보고합니다.

사진 분석은 원본 grayscale 배열과 원본 좌표를 보존한 채 다섯 가지 독립
variant를 평가합니다.

1. grayscale + adaptive threshold
2. CLAHE + adaptive threshold
3. morphology background shadow removal + adaptive threshold
4. blur + Otsu threshold
5. 조명 평탄화 + 전역 Otsu threshold (`flattened_global`)

`flattened_global`은 전역 절단으로 흐린 인쇄 배경을 잉크 판정 아래에 두어,
국소 적응형 임계값이 구조적으로 할 수 없는 일을 합니다. 이 variant만은
검출 입력으로 이진 잉크를 넘깁니다. 나머지는 enhanced grayscale을 넘기므로
Canny가 흐린 괘선의 엣지를 다시 찾아내기 때문입니다.

각 variant는 자체 enhanced raster에서 Hough geometry를 실행합니다. 선택 점수는
의미 있는 선 수와 평균 길이, endpoint 연결성, geometry coverage,
중복선 비율, 짧은 잡음 비율, connected-component 품질을 조합합니다.
선 개수만 많다고 선택하지 않으며, 점수가 가까우면 의미 있는 선 비율과
짧은 잡음이 적은 결과를 우선합니다. 선택된 variant와 각 점수 구성은
`DrawingAnalysis.preprocessing_reports`에 보존됩니다.

현재 heuristic은 다음과 같습니다.

```text
score =
  .35 * meaningful_line_score
  + .15 * average_length_score
  + .15 * connected_endpoint_ratio
  + .10 * geometry_coverage
  + .10 * component_quality
  - .15 * duplicate_line_ratio
  - .25 * short_noise_ratio
  - .30 * periodic_grid_ratio
  - .25 * paper_shape_ratio
```

`meaningful_line_score`는 최소 선 길이보다 충분히 긴 선의 비율이고,
`component_quality`는 foreground mask의 큰 연결 성분이 차지하는 비율입니다.
따라서 반복 Hough segment나 짧은 배경 선이 많아지는 결과는 자동으로
불리해집니다. `periodic_grid_ratio`는 후보 중 증명된 주기 레일에 놓인 비율로,
괘선을 증폭한 variant를 직접 감점합니다. 최고 composite score 차이가 `.05`
이내이면 긴 선의 수와 짧은 잡음 역비율을 보조 기준으로 사용해 희미한 실제
stroke가 빠지지 않게 하며, 그 다음 기준은 평균 선 길이보다 앞서
`periodic_grid_ratio`입니다. 괘선은 페이지를 가로지르는 긴 후보를 만들므로
그러지 않으면 동점이 바로 그 인공물로 결정됩니다.

`paper_shape_ratio`는 해당 variant가 만든 닫힌 도형 중 종이 구조로 증명되어
제거된 개수(격자 칸 + 표 칸)를 `min(1, max(0, paper - 12) / 120)`으로 환산한
값입니다. 살아남은 도형 수가 아니라 증명된 종이 수를 세므로, 실제로 닫힌
도형이 많은 도면은 불리해지지 않습니다. 허용치 12는 업무 양식의 머리글·푸터가
모든 variant에서 몇 칸씩 잡히기 때문에 필요하며, 이 값이 선택을 좌우하면
직선을 호로 잘못 맞추는 variant가 이깁니다. 같은 이유로 이 증거는 동점
기준에는 넣지 않습니다. 도형 검출은 `detect_closed_shapes`가 켜졌을 때만
variant마다 실행되어, 선 전용 경로의 비용은 그대로입니다.

자동 선택 점수가 임계값보다 낮으면 원본 분석 stream으로 안전하게 fallback하고
`PREPROCESSING_UNCERTAIN` 상태를 남깁니다. geometry가 없으면
`NO_GEOMETRY_DETECTED`, 잡음이 지배하면 `TOO_MUCH_NOISE`를 남기며, UI에서는
DXF 저장을 보류합니다. 사용자는 `1 표준 강화`, `2 CLAHE 강화`,
`3 그림자 보정` 세 개의 작은 미리보기 버튼 중 하나를 골라 재분석할 수
있습니다. 명시적으로 고른 결과가 낮은 점수이면 `VERY_LOW_CONFIDENCE`
경고를 계속 보이지만, 그 선택 자체를 사람의 검토 승인으로 취급합니다.
원본 사진과 geometry-only foreground view는 별도로 유지됩니다.

`PipelineConfig.preprocessing_debug_directory` 또는 CLI의 `--debug-dir DIR`을
명시한 경우에만 `original.png`, variant별 `*_enhanced.png`, `*_mask.png`를
기록합니다. 전처리 mask는 OCR 입력이나 원본 사진을 대체하지 않습니다.
페이지 사변형은 area/rectangularity/side/edge/contrast confidence가 낮으면
워프하지 않고 원본 좌표로 fallback하며 `DOCUMENT_NOT_FOUND`와
`PERSPECTIVE_NOT_FOUND`를 남깁니다. confidence가 충분해도 source/warped
geometry 비교에서 개선이 없으면 `PERSPECTIVE_REJECTED`를 남깁니다.

## 향후 단계

- Phase 2: 연결 line의 수동 length/angle reconstruction과 CAD-space export.
- Phase 3: 구조화된 Pydantic 경계 페이로드를 사용하는 비전/OCR 어댑터.
- Phase 4: 페이지 경계가 없는 사진의 원근 추정. 원/홀/부분 호(arc) 검출은
  1차로 반영되었으며, 자재 가장자리에 걸친 반원 등은 후속 과제로 남습니다.
- Phase 5: 확인 신뢰도 임계값과 소스 영역 강조.
- Phase 6: 실제 현장 이미지와 임계값 튜닝.
- Phase 7: Windows 설치 파일과 설정/API 키 저장.
