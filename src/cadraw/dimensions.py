"""Safe parsing and explicit confirmation of dimension text."""

import re
import unicodedata
from dataclasses import replace

from cadraw.models import (
    Dimension,
    DimensionId,
    DimensionKind,
    DimensionStatus,
    EntityId,
)

_NUMBER = r"(?P<value>(?:\d+(?:[.,]\d*)?|[.,]\d+))"
_VALUE_PATTERN = re.compile(
    rf"^\s*{_NUMBER}\s*(?P<unit>mm|㎜)?\s*$",
    re.IGNORECASE,
)
_PREFIX_PATTERN = re.compile(r"^\s*(?P<prefix>[Ø⌀Rr])\s*(?P<body>.*)$")
_ANGLE_PATTERN = re.compile(
    rf"^\s*{_NUMBER}\s*(?P<unit>°|deg|degree|degrees)\s*$",
    re.IGNORECASE,
)


def _numeric_value(raw_value: str) -> float | None:
    """Parse a positive decimal using dot or comma notation."""
    try:
        value = float(raw_value.replace(",", "."))
    except ValueError:
        return None
    return value if value > 0.0 else None


def parse_dimension_text(raw_text: str, dimension_id: DimensionId) -> Dimension:
    """Parse a text candidate without marking it fabrication-safe."""
    normalized = unicodedata.normalize("NFKC", raw_text).strip()
    angle_match = _ANGLE_PATTERN.fullmatch(normalized)
    if angle_match is not None:
        value = _numeric_value(angle_match.group("value"))
        if value is not None:
            return Dimension(
                dimension_id=dimension_id,
                raw_text=raw_text,
                kind=DimensionKind.ANGLE,
                parsed_value=value,
                unit="deg",
                target_entity_ids=(),
                confidence=0.98,
                status=DimensionStatus.NEEDS_REVIEW,
            )

    prefix_match = _PREFIX_PATTERN.fullmatch(normalized)
    kind = DimensionKind.LENGTH
    body = normalized
    if prefix_match is not None:
        prefix = prefix_match.group("prefix").upper()
        kind = DimensionKind.DIAMETER if prefix in {"Ø", "⌀"} else DimensionKind.RADIUS
        body = prefix_match.group("body")

    value_match = _VALUE_PATTERN.fullmatch(body)
    value = None if value_match is None else _numeric_value(value_match.group("value"))
    if value is None:
        return Dimension(
            dimension_id=dimension_id,
            raw_text=raw_text,
            kind=DimensionKind.UNKNOWN,
            parsed_value=None,
            unit="mm",
            target_entity_ids=(),
            confidence=0.2,
            status=DimensionStatus.UNKNOWN,
        )
    return Dimension(
        dimension_id=dimension_id,
        raw_text=raw_text,
        kind=kind,
        parsed_value=value,
        unit="mm",
        target_entity_ids=(),
        confidence=0.98,
        status=DimensionStatus.NEEDS_REVIEW,
    )


def associate_dimension(dimension: Dimension, entity_id: EntityId) -> Dimension:
    """Attach a user-selected entity without changing the parsed value."""
    return replace(dimension, target_entity_ids=(entity_id,))


def confirm_dimension_text(dimension: Dimension, raw_text: str) -> Dimension:
    """Replace a candidate with the exact text a user confirmed."""
    parsed = parse_dimension_text(raw_text, dimension.dimension_id)
    return replace(
        parsed,
        target_entity_ids=dimension.target_entity_ids,
        confidence=1.0 if parsed.parsed_value is not None else parsed.confidence,
        status=(
            DimensionStatus.CONFIRMED
            if parsed.parsed_value is not None
            else DimensionStatus.UNKNOWN
        ),
    )
