"""Proof-based removal of preprinted form tables from closed-shape candidates.

A work-order form opens with a header table and often closes with a company
footer block. Their cells are high-contrast printed rectangles, so closed-shape
detection finds them far more reliably than it finds the pencil drawing, and
they then outnumber the part's own geometry.

The cells are removed only once they prove themselves as a table: three or more
boxes of similar depth whose edges tile consecutively along one axis, each
overlapping its neighbour's band. Tiling is the signature that survives a
skewed scan, where a row's top and bottom edges drift by more than the accidental
agreement between two unrelated shapes, and it is something hand-drawn geometry
does not do. Nothing is removed on position alone, because a part outline may
legitimately sit anywhere on the sheet.
"""

from dataclasses import dataclass

from cadraw.models import ArcCandidate, PolylineCandidate, ShapeCandidate

_MIN_BAND_OVERLAP_RATIO = 0.45
_ENCLOSURE_SLACK_RATIO = 0.10


@dataclass(frozen=True, slots=True)
class TableFilterConfig:
    """Evidence thresholds for proving a printed table row or column."""

    enabled: bool = True
    minimum_cell_count: int = 3
    tile_gap_ratio: float = 0.012
    depth_tolerance_ratio: float = 0.40
    maximum_cell_depth_ratio: float = 0.12
    edge_line_tolerance_ratio: float = 0.02


@dataclass(frozen=True, slots=True)
class TableFilterResult:
    """Retained candidates plus evidence for the table cells removed."""

    candidates: tuple[ShapeCandidate, ...]
    table_detected: bool = False
    removed_count: int = 0


@dataclass(frozen=True, slots=True)
class _Cell:
    """One candidate reduced to the box used for the tiling proof."""

    candidate: PolylineCandidate
    low: float
    high: float
    band_low: float
    band_high: float

    @property
    def depth(self) -> float:
        """Return the extent across the tiling axis."""
        return self.band_high - self.band_low


def _cells(
    candidates: tuple[ShapeCandidate, ...],
    *,
    horizontal: bool,
) -> tuple[_Cell, ...]:
    """Reduce polygon candidates to boxes oriented for one tiling axis."""
    cells: list[_Cell] = []
    for candidate in candidates:
        if not isinstance(candidate, PolylineCandidate) or not candidate.points:
            continue
        xs = [point.x for point in candidate.points]
        ys = [point.y for point in candidate.points]
        along = (min(xs), max(xs)) if horizontal else (min(ys), max(ys))
        across = (min(ys), max(ys)) if horizontal else (min(xs), max(xs))
        cells.append(
            _Cell(
                candidate=candidate,
                low=along[0],
                high=along[1],
                band_low=across[0],
                band_high=across[1],
            ),
        )
    return tuple(sorted(cells, key=lambda cell: cell.low))


def _encloses(outer: _Cell, inner: _Cell) -> bool:
    """Return whether one cell's box lies inside a strictly larger one.

    A glyph traced from inside a printed cell can poke a pixel or two past the
    cell's own traced edge, so the containment allows a sliver of the depth.
    """
    slack = _ENCLOSURE_SLACK_RATIO * outer.depth
    return (
        outer.low - slack <= inner.low
        and inner.high <= outer.high + slack
        and outer.band_low - slack <= inner.band_low
        and inner.band_high <= outer.band_high + slack
        and (outer.high - outer.low) * outer.depth
        > (inner.high - inner.low) * inner.depth
    )


def _shares_band(first: _Cell, second: _Cell) -> bool:
    """Return whether two cells sit in the same row or column band."""
    overlap = min(first.band_high, second.band_high) - max(
        first.band_low,
        second.band_low,
    )
    shallower = min(first.depth, second.depth)
    return shallower > 0.0 and overlap / shallower >= _MIN_BAND_OVERLAP_RATIO


def _tiles_after(first: _Cell, second: _Cell, gap: float) -> bool:
    """Return whether the second cell continues the first one's edge."""
    return abs(second.low - first.high) <= gap


def _bands(cells: tuple[_Cell, ...]) -> tuple[tuple[_Cell, ...], ...]:
    """Group cells into rows or columns that overlap across the tiling axis.

    Grouping runs against the previous member rather than a fixed
    representative, so a row on a skewed page stays one band even though its
    first and last cells no longer overlap each other.
    """
    groups: list[list[_Cell]] = []
    for cell in sorted(cells, key=lambda member: member.band_low):
        if groups and _shares_band(groups[-1][-1], cell):
            groups[-1].append(cell)
        else:
            groups.append([cell])
    return tuple(tuple(group) for group in groups)


def _proved_chains(
    cells: tuple[_Cell, ...],
    gap: float,
    maximum_depth: float,
    config: TableFilterConfig,
) -> tuple[_Cell, ...]:
    """Return cells belonging to a tiling run long enough to prove a table.

    Runs are sought inside one band at a time. Scanning every candidate in one
    pass would let a shape from elsewhere on the sheet land between two cells
    of a row and cut the run short.
    """
    shallow = tuple(cell for cell in cells if 0.0 < cell.depth <= maximum_depth)
    proved: list[_Cell] = []
    for band in _bands(shallow):
        chain: list[_Cell] = []
        for cell in sorted(band, key=lambda member: member.low):
            if (
                chain
                and _tiles_after(chain[-1], cell, gap)
                and abs(cell.depth - chain[-1].depth)
                <= config.depth_tolerance_ratio * max(cell.depth, chain[-1].depth)
            ):
                chain.append(cell)
                continue
            if len(chain) >= config.minimum_cell_count:
                proved.extend(chain)
            chain = [cell]
        if len(chain) >= config.minimum_cell_count:
            proved.extend(chain)
    return tuple(proved)


def _belongs_to_table(cell: _Cell, proved: tuple[_Cell, ...]) -> bool:
    """Return whether one cell is further furniture of an already proved table.

    Detection recovers a printed row unevenly, so some of its cells never join
    the tiling run, and the table's own outline encloses all of them. Once the
    run is proved, a box sitting inside that row's band, or a box enclosing the
    whole run, is the same piece of stationery.
    """
    along_low = min(member.low for member in proved)
    along_high = max(member.high for member in proved)
    band_low = min(member.band_low for member in proved)
    band_high = max(member.band_high for member in proved)
    margin = max(member.depth for member in proved)
    # A printed row runs across the sheet, so membership is judged by the band
    # rather than by the span of the run that happened to be detected.
    inside_row = (
        cell.band_low >= band_low - margin
        and cell.band_high <= band_high + margin
        and cell.depth <= 2.0 * margin
    )
    encloses_row = (
        cell.low <= along_low
        and cell.high >= along_high
        and cell.band_low <= band_low
        and cell.band_high >= band_high
    )
    return inside_row or encloses_row


def _table_member_ids(
    candidates: tuple[ShapeCandidate, ...],
    along_gap: float,
    maximum_depth: float,
    settings: TableFilterConfig,
    *,
    horizontal: bool,
) -> set[int]:
    """Return candidate ids proved to belong to a table on one axis."""
    cells = _cells(candidates, horizontal=horizontal)
    # A glyph traced inside a printed cell sits between two of the row's
    # cells in the run and breaks it, so a row that fails to prove itself is
    # tried again without the shapes enclosed by another. The first attempt
    # keeps them, because a table's own outline encloses whole cells.
    proved = _proved_chains(
        cells,
        along_gap,
        maximum_depth,
        settings,
    ) or _proved_chains(
        tuple(
            cell
            for cell in cells
            if not any(other is not cell and _encloses(other, cell) for other in cells)
        ),
        along_gap,
        maximum_depth,
        settings,
    )
    if not proved:
        return set()
    return {id(cell.candidate) for cell in cells if _belongs_to_table(cell, proved)}


def _box(candidate: ShapeCandidate) -> tuple[float, float, float, float]:
    """Return a candidate's axis-aligned bounding box."""
    if isinstance(candidate, PolylineCandidate):
        xs = [point.x for point in candidate.points]
        ys = [point.y for point in candidate.points]
        return min(xs), min(ys), max(xs), max(ys)
    center = candidate.center
    return (
        center.x - candidate.radius,
        center.y - candidate.radius,
        center.x + candidate.radius,
        center.y + candidate.radius,
    )


def _drawn_inside_retained(
    candidates: tuple[ShapeCandidate, ...],
    removed_ids: set[int],
) -> set[int]:
    """Return removed candidates that sit inside a shape the proof kept.

    Band membership reaches across the whole sheet, so on a photograph with a
    ruled page beside the drawing it can claim a part's slots. A box drawn
    inside a part outline is that part's feature; a printed cell never sits
    inside a shape that is not itself stationery. A kept shape enclosing every
    removed one is a frame around the sheet, not a part, and restores nothing.
    """
    removed = [candidate for candidate in candidates if id(candidate) in removed_ids]
    restored: set[int] = set()
    for outer in candidates:
        if (
            id(outer) in removed_ids
            or isinstance(outer, ArcCandidate)
            or (isinstance(outer, PolylineCandidate) and not outer.closed)
        ):
            continue
        inside = {
            id(candidate)
            for candidate in removed
            if _box_inside(_box(candidate), _box(outer))
        }
        if len(inside) < len(removed):
            restored |= inside
    return restored


def _box_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    """Return whether one box lies strictly inside another."""
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def inspect_printed_table_shapes(
    candidates: tuple[ShapeCandidate, ...],
    width: int,
    height: int,
    config: TableFilterConfig | None = None,
) -> TableFilterResult:
    """Remove closed-shape candidates proved to tile a printed form table."""
    settings = config if config is not None else TableFilterConfig()
    if not settings.enabled or not candidates:
        return TableFilterResult(candidates)
    removed_ids = _table_member_ids(
        candidates,
        width * settings.tile_gap_ratio,
        height * settings.maximum_cell_depth_ratio,
        settings,
        horizontal=True,
    ) | _table_member_ids(
        candidates,
        height * settings.tile_gap_ratio,
        width * settings.maximum_cell_depth_ratio,
        settings,
        horizontal=False,
    )
    removed_ids -= _drawn_inside_retained(candidates, removed_ids)
    if not removed_ids:
        return TableFilterResult(candidates)
    return TableFilterResult(
        candidates=tuple(
            candidate for candidate in candidates if id(candidate) not in removed_ids
        ),
        table_detected=True,
        removed_count=len(removed_ids),
    )
