# Phase 1 Fixture and Supplied Sample

`scripts/run_poc.py` supports both a reproducible synthetic sketch and the
supplied camera photograph.

The synthetic fixture contains:

- four structural strokes: top horizontal, long vertical, diagonal, and lower vertical;
- slight endpoint gaps and angle errors;
- nearby handwritten-style numeric labels that are deliberately not treated as OCR truth.

The supplied photograph is `sample/KakaoTalk_20260909_141552817.jpg`. Its
documented experiment configuration uses ROI `(500, 680, 550, 1150)`, 7-degree
axis tolerances, endpoint snap distance `45`, and largest-connected-component
selection. The user supplied four manual values in top-to-bottom order:
`100 mm`, `1050 mm`, `20°`, and `150 mm`. The observed result is four
structural lines, three connections, and a valid four-`LINE` plus four-`TEXT`
DXF. `CONSTRAINT_MISMATCH` is expected for lengths that cannot share the first
global image scale; it is surfaced rather than hidden.

This validates the deterministic path and the real-image boundary, not
handwriting recognition or perspective correction.
