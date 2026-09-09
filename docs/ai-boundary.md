# AI Boundary

## Current adapter contract

```python
class VisionAnalyzer(Protocol):
    def analyze(self, image_path: Path) -> DrawingAnalysis: ...
```

`MockVisionAnalyzer` runs the local deterministic pipeline and returns no OCR dimensions. It exists so the application can be developed and tested without an API key.

## Future OpenAI-compatible adapter

A future adapter may call OpenRouter or another OpenAI-compatible endpoint. It must:

1. Load credentials from user settings or environment variables, never source code.
2. Parse the response at the boundary into a typed schema.
3. Preserve raw text, confidence, and source regions.
4. Return `target_entity_ids = ()` when association is unclear.
5. Mark recognized values as `NEEDS_REVIEW` until a person confirms them.
6. Never write a DXF or emit final CAD coordinates.

The deterministic normalizer, calculator, and exporter remain provider-agnostic.

## Explicit non-goals

- No prompt asks a model to "complete" a missing measurement.
- No model-generated coordinate list is trusted as CAD geometry.
- No image ratio becomes a physical dimension without a confirmed value.
