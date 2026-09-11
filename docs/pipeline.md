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

`--perspective`가 켜지면 페이지 사변형을 검출해 직사각형으로 워프한 뒤,
역행렬로 모든 도형과 OCR box를 원본 사진 좌표로 되돌립니다. 페이지 경계가
보이지 않으면 `PERSPECTIVE_NOT_FOUND`를 남기고 워프를 생략합니다.

`--ocr`는 설치된 Tesseract TSV 결과를 `TextCandidate`로 보존하고, 기존
`parse_dimension_text`를 거친 값만 `NEEDS_REVIEW` 치수로 추가합니다. 대상
geometry는 자동 연결하지 않으므로 검토 전 DXF TEXT가 생성되지 않습니다.

그래프 용지, 인쇄 표, 손글씨가 섞인 사진에서는 후보가 남을 수 있으므로
preview와 DXF를 확인한 뒤 치수를 수동으로 확정해야 합니다. 외곽 닫힌 형상
안에 포함된 작은 형상은 `HOLE` 역할로 분류하고 DXF `HOLE` 레이어에 씁니다.

## 향후 단계

- Phase 2: 연결 line의 수동 length/angle reconstruction과 CAD-space export.
- Phase 3: 구조화된 Pydantic 경계 페이로드를 사용하는 비전/OCR 어댑터.
- Phase 4: 페이지 경계가 없는 사진의 원근 추정, 원/홀/호(arc) 특화.
- Phase 5: 확인 신뢰도 임계값과 소스 영역 강조.
- Phase 6: 실제 현장 이미지와 임계값 튜닝.
- Phase 7: Windows 설치 파일과 설정/API 키 저장.
