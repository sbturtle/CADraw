# Decision 0002: DXF Before DWG

## Context

The first useful deliverable is a stable file that fabrication software can inspect. DWG support introduces proprietary format and deployment concerns.

## Decision

Support DXF with `ezdxf` first. Add DWG conversion only after DXF round-trip tests and real shop-image validation are reliable.

## Consequence

The exporter stays inspectable and testable in the open Python environment. A later DWG bridge can consume the same `CadDocument` without changing recognition or normalization.
