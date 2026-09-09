# Decision 0001: Keep Geometry Deterministic

## Context

Hand drawings contain gaps, skew, uncertain text, and missing dimensions. A generative model can produce plausible but unsafe coordinates.

## Decision

Vision is limited to recognition and semantic suggestions. Geometry normalization, scaling, validation, preview, and DXF serialization are deterministic Python code operating on typed models.

## Consequence

The application can be tested without a network or API key. Uncertainty is visible in the intermediate representation and can be corrected by a person. The trade-off is that Phase 1 does not automatically solve ambiguous geometry.
