"""Deterministic conversion of unambiguous line chains to polylines."""

from typing import Final

from cadraw.models import EntityId, GeometryEntity, LineEntity, Point, PolylineEntity

_MIN_POLYLINE_SEGMENTS: Final = 2
_OPEN_CHAIN_TERMINAL_COUNT: Final = 2
_MAX_CHAIN_VERTEX_DEGREE: Final = 2


def _line_graph(
    lines: tuple[LineEntity, ...],
) -> dict[Point, list[tuple[int, Point]]]:
    """Index each line by its exact, already-normalized endpoints."""
    graph: dict[Point, list[tuple[int, Point]]] = {}
    for index, line in enumerate(lines):
        if line.start == line.end:
            continue
        graph.setdefault(line.start, []).append((index, line.end))
        graph.setdefault(line.end, []).append((index, line.start))
    return graph


def _components(
    lines: tuple[LineEntity, ...],
    graph: dict[Point, list[tuple[int, Point]]],
) -> tuple[tuple[int, ...], ...]:
    """Find exact-endpoint-connected line components."""
    unseen = {index for index, line in enumerate(lines) if line.start != line.end}
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            line = lines[current]
            for endpoint in (line.start, line.end):
                for neighbor, _ in graph[endpoint]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def connected_line_components(
    lines: tuple[LineEntity, ...],
) -> tuple[tuple[LineEntity, ...], ...]:
    """Return exact-endpoint-connected line components in stable order."""
    graph = _line_graph(lines)
    return tuple(
        tuple(lines[index] for index in component)
        for component in _components(lines, graph)
    )


def _chain_points(
    component: tuple[int, ...],
    lines: tuple[LineEntity, ...],
    graph: dict[Point, list[tuple[int, Point]]],
) -> tuple[Point, ...] | None:
    """Return ordered vertices only for an unambiguous open chain."""
    if len(component) < _MIN_POLYLINE_SEGMENTS:
        return None
    component_edges = set(component)
    vertices = {
        point for index in component for point in (lines[index].start, lines[index].end)
    }
    terminals = tuple(
        sorted(
            (point for point in vertices if len(graph[point]) == 1),
            key=lambda point: (point.x, point.y),
        ),
    )
    if len(terminals) != _OPEN_CHAIN_TERMINAL_COUNT or any(
        len(graph[point]) not in {1, _MAX_CHAIN_VERTEX_DEGREE} for point in vertices
    ):
        return None

    start, finish = terminals
    points = [start]
    visited: set[int] = set()
    current = start
    while len(visited) < len(component):
        options = tuple(
            (edge_index, next_point)
            for edge_index, next_point in graph[current]
            if edge_index in component_edges and edge_index not in visited
        )
        if len(options) != 1:
            return None
        edge_index, next_point = options[0]
        visited.add(edge_index)
        points.append(next_point)
        current = next_point
    if current != finish:
        return None
    return tuple(points)


def _assemble_component(
    component: tuple[int, ...],
    lines: tuple[LineEntity, ...],
    graph: dict[Point, list[tuple[int, Point]]],
) -> PolylineEntity | None:
    """Create one stable polyline for a valid component."""
    points = _chain_points(component, lines, graph)
    if points is None:
        return None
    first_line = min(
        (lines[index] for index in component),
        key=lambda line: line.entity_id,
    )
    return PolylineEntity(
        entity_id=EntityId(f"polyline_{first_line.entity_id}"),
        points=points,
        closed=False,
        confidence=min(lines[index].confidence for index in component),
    )


def assemble_open_line_chains(
    entities: tuple[GeometryEntity, ...],
) -> tuple[GeometryEntity, ...]:
    """Replace only exact, unambiguous open line chains with polylines.

    Branches, loops, isolated lines, and near-but-not-equal endpoints remain
    untouched so topology is never invented from proximity alone.
    """
    ordered_entities = tuple(sorted(entities, key=lambda entity: entity.entity_id))
    lines = tuple(
        entity for entity in ordered_entities if isinstance(entity, LineEntity)
    )
    graph = _line_graph(lines)
    replacements: dict[EntityId, PolylineEntity] = {}
    consumed_ids: set[EntityId] = set()
    for component in _components(lines, graph):
        polyline = _assemble_component(component, lines, graph)
        if polyline is None:
            continue
        first_line = min(
            (lines[index] for index in component),
            key=lambda line: line.entity_id,
        )
        replacements[first_line.entity_id] = polyline
        consumed_ids.update(lines[index].entity_id for index in component)

    assembled: list[GeometryEntity] = []
    for entity in ordered_entities:
        replacement = replacements.get(entity.entity_id)
        if replacement is not None:
            assembled.append(replacement)
        elif entity.entity_id not in consumed_ids:
            assembled.append(entity)
    return tuple(assembled)
