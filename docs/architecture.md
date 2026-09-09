# Architecture

## 결정

CADraw는 네 경계로 나뉜 작은 desktop application입니다.

```text
Image path
  -> detection.py
  -> normalization.py
  -> models.DrawingAnalysis
  -> dimensions.py / calculator.py
  -> preview.py and export.py
```

`ui/`는 이 흐름을 조합하지만 geometry 계산을 직접 하지 않습니다. 따라서 PySide6가 설치되지 않은 환경에서도 domain tests와 DXF round-trip tests를 실행할 수 있습니다.

## 모듈 책임

| Module | Responsibility |
|---|---|
| `models.py` | immutable points, entities, dimensions, warnings, analysis documents |
| `detection.py` | image loading, preprocessing, border filtering, OpenCV line candidates |
| `normalization.py` | orientation classification, snaps, collinear merging, connections |
| `dimensions.py` | safe parsing of length/diameter/radius/angle text |
| `calculator.py` | explicit-dimension calibration and deterministic coordinate transform |
| `preview.py` | pure Pillow rendering of CAD entities |
| `export.py` | ezdxf serialization and reload validation |
| `vision.py` | `VisionAnalyzer` Protocol and no-OCR mock adapter |
| `ui/` | source image, preview, manual verification fields, actions |

## 데이터 흐름

`DrawingAnalysis` is the intermediate representation. Raw coordinates are preserved on detected line entities, while normalized coordinates are used for preview and export. Dimension values are nullable and each dimension carries a review status.

For the supplied camera sample, `PipelineConfig.source_region` limits analysis to
the known paper drawing area and restores detected coordinates to the original
photo frame. This is a tunable experiment setting, not an automatic perspective
correction.

AI/Vision never returns DXF coordinates. A future adapter may return parsed entities and text candidates, but `calculator.py` and `export.py` remain deterministic consumers.

## 배포 판단

The PoC uses a regular Python environment and PySide6. After the desktop path is stable, `PyInstaller` or an equivalent Windows packaging step can produce an installer. DWG conversion is postponed until the DXF representation and user review flow are trusted.
