"""Pure graph layout helpers for VIPP workflow canvases."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LayoutNode:
    """Node record used by the pure auto-layout helper."""

    id: str
    width: float = 220.0
    height: float = 180.0


@dataclass(frozen=True)
class LayoutEdge:
    """Directed edge record used by the pure auto-layout helper."""

    source_id: str
    target_id: str


Position = tuple[float, float]

DEFAULT_NODE_WIDTH = 220.0
DEFAULT_NODE_HEIGHT = 180.0
MIN_NODE_WIDTH = 80.0
MIN_NODE_HEIGHT = 60.0


def layout_layered_dag(
    nodes: Iterable[LayoutNode],
    edges: Iterable[LayoutEdge],
    *,
    current_positions: Mapping[str, Position] | None = None,
    origin: Position = (0.0, 20.0),
    horizontal_gap: float = 140.0,
    vertical_gap: float = 90.0,
    component_gap: float = 190.0,
) -> dict[str, Position]:
    """Return deterministic source-to-sink positions for an acyclic graph.

    The layout is intentionally one-shot and conservative: it assigns columns by
    dependency depth, orders nodes inside each column with stable barycentric
    sweeps, and places weakly disconnected components in separate vertical lanes.
    """

    node_list = list(nodes)
    if not node_list:
        return {}

    current_positions = current_positions or {}
    node_ids = [node.id for node in node_list]
    node_id_set = set(node_ids)
    node_order = {node_id: index for index, node_id in enumerate(node_ids)}
    sizes = {
        node.id: (
            _positive_size(node.width, DEFAULT_NODE_WIDTH, MIN_NODE_WIDTH),
            _positive_size(node.height, DEFAULT_NODE_HEIGHT, MIN_NODE_HEIGHT),
        )
        for node in node_list
    }
    valid_edges = [
        edge
        for edge in edges
        if edge.source_id in node_id_set
        and edge.target_id in node_id_set
        and edge.source_id != edge.target_id
    ]
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    undirected: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in valid_edges:
        successors[edge.source_id].add(edge.target_id)
        predecessors[edge.target_id].add(edge.source_id)
        undirected[edge.source_id].add(edge.target_id)
        undirected[edge.target_id].add(edge.source_id)

    positions: dict[str, Position] = {}
    y_cursor = float(origin[1])
    for component in _weak_components(node_ids, undirected, node_order):
        component_positions, component_height = _layout_component(
            component,
            predecessors,
            successors,
            sizes,
            current_positions,
            node_order,
            x_origin=float(origin[0]),
            y_origin=y_cursor,
            horizontal_gap=float(horizontal_gap),
            vertical_gap=float(vertical_gap),
        )
        positions.update(component_positions)
        y_cursor += component_height + float(component_gap)
    return positions


def _layout_component(
    component: list[str],
    predecessors: Mapping[str, set[str]],
    successors: Mapping[str, set[str]],
    sizes: Mapping[str, tuple[float, float]],
    current_positions: Mapping[str, Position],
    node_order: Mapping[str, int],
    *,
    x_origin: float,
    y_origin: float,
    horizontal_gap: float,
    vertical_gap: float,
) -> tuple[dict[str, Position], float]:
    layers = _assign_layers(component, predecessors, successors, node_order)
    layer_nodes: dict[int, list[str]] = defaultdict(list)
    for node_id in component:
        layer_nodes[layers[node_id]].append(node_id)

    orders = {
        layer: sorted(
            layer_node_ids,
            key=lambda node_id: (
                _position_y(current_positions, node_id),
                node_order[node_id],
            ),
        )
        for layer, layer_node_ids in layer_nodes.items()
    }
    _reduce_crossings(orders, layers, predecessors, successors, node_order)

    sorted_layers = sorted(orders)
    column_widths = {
        layer: max(sizes[node_id][0] for node_id in orders[layer])
        for layer in sorted_layers
    }
    column_heights = {
        layer: _stack_height(
            [sizes[node_id][1] for node_id in orders[layer]],
            vertical_gap,
        )
        for layer in sorted_layers
    }
    component_height = max(column_heights.values(), default=0.0)

    x_by_layer: dict[int, float] = {}
    x_cursor = x_origin
    for layer in sorted_layers:
        x_by_layer[layer] = x_cursor
        x_cursor += column_widths[layer] + horizontal_gap

    positions: dict[str, Position] = {}
    for layer in sorted_layers:
        y_cursor = y_origin + max(0.0, (component_height - column_heights[layer]) / 2)
        for node_id in orders[layer]:
            positions[node_id] = (x_by_layer[layer], y_cursor)
            y_cursor += sizes[node_id][1] + vertical_gap

    return positions, component_height


def _assign_layers(
    component: list[str],
    predecessors: Mapping[str, set[str]],
    successors: Mapping[str, set[str]],
    node_order: Mapping[str, int],
) -> dict[str, int]:
    component_set = set(component)
    topo = _topological_component_order(component, predecessors, successors, node_order)
    layers = {node_id: 0 for node_id in component}
    for node_id in topo:
        parent_layers = [
            layers[parent] + 1
            for parent in predecessors[node_id]
            if parent in component_set
        ]
        if parent_layers:
            layers[node_id] = max(layers[node_id], max(parent_layers))
    return layers


def _topological_component_order(
    component: list[str],
    predecessors: Mapping[str, set[str]],
    successors: Mapping[str, set[str]],
    node_order: Mapping[str, int],
) -> list[str]:
    component_set = set(component)
    indegree = {
        node_id: sum(1 for parent in predecessors[node_id] if parent in component_set)
        for node_id in component
    }
    ready = deque(
        sorted(
            [node_id for node_id in component if indegree[node_id] == 0],
            key=node_order.__getitem__,
        )
    )
    result: list[str] = []
    while ready:
        node_id = ready.popleft()
        result.append(node_id)
        for target_id in sorted(successors[node_id], key=node_order.__getitem__):
            if target_id not in component_set:
                continue
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)
        ready = deque(sorted(ready, key=node_order.__getitem__))

    if len(result) < len(component):
        result_set = set(result)
        result.extend(
            sorted(
                [node_id for node_id in component if node_id not in result_set],
                key=node_order.__getitem__,
            )
        )
    return result


def _reduce_crossings(
    orders: dict[int, list[str]],
    layers: Mapping[str, int],
    predecessors: Mapping[str, set[str]],
    successors: Mapping[str, set[str]],
    node_order: Mapping[str, int],
) -> None:
    sorted_layers = sorted(orders)
    for _ in range(4):
        for layer in sorted_layers[1:]:
            _sort_layer_by_neighbors(
                orders,
                layer,
                layers,
                predecessors,
                node_order,
            )
        for layer in reversed(sorted_layers[:-1]):
            _sort_layer_by_neighbors(
                orders,
                layer,
                layers,
                successors,
                node_order,
            )


def _sort_layer_by_neighbors(
    orders: dict[int, list[str]],
    layer: int,
    layers: Mapping[str, int],
    neighbors: Mapping[str, set[str]],
    node_order: Mapping[str, int],
) -> None:
    current_order = {node_id: index for index, node_id in enumerate(orders[layer])}
    neighbor_order: dict[str, int] = {}
    for neighbor_layer, node_ids in orders.items():
        if neighbor_layer == layer:
            continue
        for index, node_id in enumerate(node_ids):
            neighbor_order[node_id] = index

    def key(node_id: str) -> tuple[int, float, int, int]:
        indices = [
            neighbor_order[neighbor_id]
            for neighbor_id in neighbors[node_id]
            if neighbor_id in neighbor_order
            and layers.get(neighbor_id) != layers.get(node_id)
        ]
        if not indices:
            return (
                1,
                float(current_order[node_id]),
                current_order[node_id],
                node_order[node_id],
            )
        return (
            0,
            sum(indices) / len(indices),
            current_order[node_id],
            node_order[node_id],
        )

    orders[layer].sort(key=key)


def _weak_components(
    node_ids: list[str],
    undirected: Mapping[str, set[str]],
    node_order: Mapping[str, int],
) -> list[list[str]]:
    remaining = set(node_ids)
    components: list[list[str]] = []
    while remaining:
        seed = min(remaining, key=node_order.__getitem__)
        stack = [seed]
        component: set[str] = set()
        remaining.remove(seed)
        while stack:
            node_id = stack.pop()
            component.add(node_id)
            for neighbor_id in sorted(undirected[node_id], key=node_order.__getitem__):
                if neighbor_id in remaining:
                    remaining.remove(neighbor_id)
                    stack.append(neighbor_id)
        components.append(sorted(component, key=node_order.__getitem__))
    return components


def _positive_size(value: float, default: float, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(number) or number <= 0:
        return default
    return max(number, minimum)


def _position_y(
    positions: Mapping[str, Position],
    node_id: str,
) -> float:
    position = positions.get(node_id)
    if position is None:
        return 0.0
    try:
        return float(position[1])
    except (TypeError, ValueError, IndexError):
        return 0.0


def _stack_height(heights: list[float], vertical_gap: float) -> float:
    if not heights:
        return 0.0
    return sum(heights) + max(len(heights) - 1, 0) * vertical_gap
