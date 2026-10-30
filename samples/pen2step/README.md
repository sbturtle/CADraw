# Pen2STEP end-view photographs

Three photographs of hand-drawn multi-view sheets from Pen2STEP-Bench. They are
here as test data for a second drawing hand and pen. Each one has an end view
of concentric circles crossed by centre lines:

| File | Part | End view |
|---|---|---|
| `real_11_mv.jpg` | stepped shaft | three concentric circles |
| `real_17_mv.jpg` | cone with a bore | two concentric circles, Ø40 written |
| `real_18_mv.jpg` | washer | rim and centre hole, 40 mm and 16 mm written |

- Source: [unes21/pen2step](https://github.com/unes21/pen2step), path
  `benchmark/real/images/`, commit `fbadbede6b4ebdf620e73178d9de7911c3ca90a9`.
- Authors: Cem Baytin, Ayberk Goker Unes, Selman Aslan, Onur Behzat Tokdemir.
  The paper is *Pen2STEP: A Deterministic Inference-Time Shell for Converting
  Photographed Hand-Drawn Multi-View Sketches into CadQuery Programs and STEP
  Solids without Additional Training*.
- Licence: the images are under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as the upstream
  README states.
- The files are unmodified.

Pen2STEP's ground truth is 3D CadQuery code, not 2D drawing geometry. The tests
here check only what the photographs show: which circles are drawn, and that no
extra line re-traces them.
