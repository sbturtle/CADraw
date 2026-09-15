# Real drawing samples

Put additional father-hand-drawing photographs in this directory. The
regression runner scans `samples/real/*.jpg` and `*.jpeg` first; the checked-in
eight-image set remains under `sample/` for compatibility with existing tests.

Run:

```text
uv run python scripts/run_preprocessing_regression.py
uv run python scripts/run_preprocessing_regression.py --sample-dir sample/pure_background --detect-shapes
```

The output records the legacy and improved line counts, selected preprocessing
variant, status, warning codes, and measured processing time. Timing values are
observations from the current machine and are not golden assertions. The
`--detect-shapes` option intentionally includes the existing closed-shape
detector; use it to audit circles and rectangles separately from the default
line-only extraction.
