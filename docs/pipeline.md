# Pipeline

## Phase 1 vertical slice

1. User opens a JPG/PNG.
2. OpenCV reads grayscale pixels, excludes configurable page-edge candidates, and
   creates blurred Canny edges.
3. Probabilistic Hough detection produces raw line candidates.
4. Candidates are sorted deterministically and merged when they are near-collinear.
5. Orientation is classified as horizontal, vertical, or diagonal.
6. Nearby endpoints are snapped and connection records are created.
7. `DrawingAnalysis` is shown beside a clean preview.
8. No OCR value is assumed. Verification rows start as unknown/manual.
9. User enters a physical length and marks it confirmed.
10. The first confirmed length determines one global mm-per-pixel scale. No dimension is inferred from image proportions.
11. Preview is recalculated.
12. `ezdxf` writes geometry to `GEOMETRY`, known dimension labels to `DIMENSION`, and supporting text to `TEXT`.
13. The exporter reloads the file and counts entities before reporting success.

For the supplied photograph, the runner uses
`source_region=(500, 680, 550, 1150)` and keeps the largest connected line
component. This removes page/background and handwritten-label strokes while
preserving source-photo coordinates. The region is intentionally explicit
because general perspective/ROI detection is not yet implemented.

## Failure as domain state

The pipeline does not throw for normal uncertainty:

- no candidate lines -> `LOW_CONFIDENCE_GEOMETRY`
- missing calibration -> `UNKNOWN_DIMENSION`
- dimension without target -> `UNCONNECTED_DIMENSION`
- parser cannot read a value -> `UNCERTAIN_TEXT`
- confirmed dimensions disagree with scaled geometry -> `CONSTRAINT_MISMATCH`

I/O failures and malformed DXF files are exceptions at their boundaries. Unknown fabrication information is not an exception.

## Future phases

- Phase 2: stronger manual dimension editor and rectangle/circle tools.
- Phase 3: Vision/OCR adapter with structured Pydantic boundary payloads.
- Phase 4: perspective correction, circle/hole/arc specialization.
- Phase 5: review confidence thresholds and source-region highlighting.
- Phase 6: real shop images and threshold tuning.
- Phase 7: Windows installer and settings/API key storage.
