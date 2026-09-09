# Geometry Model

## Coordinate contract

Phase 1 coordinates are local 2D coordinates in the source image's pixel frame: x grows right and y grows down. They are not physical dimensions until a user confirms at least one length. This avoids silently converting a drawing's visual ratio into a fabrication measurement.

## Entities

- `LineEntity`: raw endpoints, normalized endpoints, orientation, confidence.
- `PolylineEntity`: ordered points and a closed flag for connected paths.
- `CircleEntity`: center and radius.
- `ArcEntity`: center, radius, start angle, end angle.

The model uses a tagged `EntityType` enum and immutable dataclasses. Export code matches every known variant explicitly.

## Normalization

`NormalizationConfig` owns all tolerances:

- `horizontal_angle_tolerance_degrees`
- `vertical_angle_tolerance_degrees`
- `endpoint_snap_distance`
- `line_merge_tolerance`
- `minimum_line_length`
- `keep_largest_connected_component`

`PipelineConfig.source_region` can optionally crop a known drawing area for a
camera experiment. Detected coordinates are translated back to the original
photo frame; this is not a physical perspective correction.
- `keep_largest_connected_component`

`PipelineConfig.source_region` can optionally crop a known drawing area for a
camera experiment. Detected coordinates are translated back to the original
photo frame; this is not a physical perspective correction.

The engine first snaps nearly horizontal/vertical coordinates, then merges same-line segments, then snaps nearby endpoints. No language model participates in these operations.

## Scaling

`calculator.py` chooses the first confirmed length dimension whose target is a line. It computes:

```text
mm_per_pixel = confirmed_value / normalized_target_length
```

All entities are scaled by that value around a normalized local origin. Additional confirmed values validate the result and create a warning if they disagree; they do not create a second hidden scale or override the user's first calibration.

## Unknown values

`Dimension.parsed_value` is `float | None`. A missing diameter, radius, angle, or target remains `None` and has a non-confirmed status. Export skips unassociated dimension text instead of placing misleading text at an arbitrary coordinate.
