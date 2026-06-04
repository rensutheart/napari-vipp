"""Editable prototype graph model and executor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from napari_vipp.core.operations import (
    binary_threshold,
    gaussian_blur,
    invert,
    max_intensity_projection,
    median_filter,
    otsu_threshold,
)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: float | int
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int = 0


@dataclass(frozen=True)
class OperationSpec:
    id: str
    title: str
    category: str
    input_type: str | None
    output_type: str
    parameters: tuple[ParameterSpec, ...] = ()
    function: Callable[..., Any] | None = None

    @property
    def has_input(self) -> bool:
        return self.input_type is not None


@dataclass
class GraphNode:
    id: str
    operation_id: str
    title: str
    category: str
    input_type: str | None
    output_type: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def has_input(self) -> bool:
        return self.input_type is not None


@dataclass(frozen=True)
class GraphConnection:
    source_id: str
    target_id: str


@dataclass(frozen=True)
class ConnectionResult:
    success: bool
    message: str
    removed: tuple[GraphConnection, ...] = ()


NODE_LIBRARY: tuple[OperationSpec, ...] = (
    OperationSpec("input", "Input Layer", "Input", None, "image"),
    OperationSpec(
        "gaussian_blur",
        "Gaussian Blur",
        "Filtering",
        "array",
        "image",
        (
            ParameterSpec("sigma", "Sigma", "float", 1.2, 0.0, 12.0, 0.1, 2),
        ),
        gaussian_blur,
    ),
    OperationSpec(
        "median_filter",
        "Median Filter",
        "Filtering",
        "array",
        "image",
        (
            ParameterSpec("size", "Size", "int", 3, 1, 21, 1),
        ),
        median_filter,
    ),
    OperationSpec(
        "mip",
        "Maximum Projection",
        "Projection",
        "array",
        "image",
        (
            ParameterSpec("axis", "Axis", "int", 0, 0, 5, 1),
        ),
        max_intensity_projection,
    ),
    OperationSpec(
        "otsu_threshold",
        "Otsu Threshold",
        "Segmentation",
        "image",
        "mask",
        (),
        otsu_threshold,
    ),
    OperationSpec(
        "binary_threshold",
        "Binary Threshold",
        "Segmentation",
        "image",
        "mask",
        (
            ParameterSpec("threshold", "Threshold", "float", 0.5, 0.0, 1.0, 0.01, 3),
        ),
        binary_threshold,
    ),
    OperationSpec("invert", "Invert", "Utility", "any", "any", (), invert),
)

NODE_LIBRARY_BY_ID = {spec.id: spec for spec in NODE_LIBRARY}
PALETTE_NODE_LIBRARY = tuple(spec for spec in NODE_LIBRARY if spec.id != "input")

PROTOTYPE_NODES = [
    GraphNode("input", "input", "Input Layer", "Input", None, "image"),
    GraphNode(
        "gaussian",
        "gaussian_blur",
        "Gaussian Blur",
        "Filtering",
        "array",
        "image",
        {"sigma": 1.2},
    ),
    GraphNode(
        "threshold",
        "otsu_threshold",
        "Otsu Threshold",
        "Segmentation",
        "image",
        "mask",
    ),
]
PROTOTYPE_CONNECTIONS = [
    GraphConnection("input", "gaussian"),
    GraphConnection("gaussian", "threshold"),
]


class PrototypePipeline:
    """A small executable single-input graph for the interaction prototype."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.connections: list[GraphConnection] = []
        self.outputs: dict[str, Any] = {}
        self._counters: Counter[str] = Counter()
        self.reset_starter_graph()

    def reset_starter_graph(self) -> None:
        self.nodes = {node.id: _clone_node(node) for node in PROTOTYPE_NODES}
        self.connections = list(PROTOTYPE_CONNECTIONS)
        self.outputs = {}
        self._counters = Counter()
        for node in self.nodes.values():
            self._counters[node.operation_id] += 1

    def add_node(self, operation_id: str) -> GraphNode:
        spec = self.operation_spec(operation_id)
        self._counters[operation_id] += 1
        node_id = _node_id(operation_id, self._counters[operation_id])
        while node_id in self.nodes:
            self._counters[operation_id] += 1
            node_id = _node_id(operation_id, self._counters[operation_id])

        node = GraphNode(
            node_id,
            spec.id,
            spec.title,
            spec.category,
            spec.input_type,
            spec.output_type,
            {param.name: param.default for param in spec.parameters},
        )
        self.nodes[node.id] = node
        self.outputs[node.id] = None
        return node

    def connect(self, source_id: str, target_id: str) -> ConnectionResult:
        if source_id not in self.nodes or target_id not in self.nodes:
            return ConnectionResult(False, "Cannot connect missing nodes.")
        if source_id == target_id:
            return ConnectionResult(False, "Cannot connect a node to itself.")

        source = self.nodes[source_id]
        target = self.nodes[target_id]
        if not target.has_input:
            return ConnectionResult(False, "That node does not accept an input.")
        if not self._types_compatible(source.output_type, target.input_type):
            return ConnectionResult(
                False,
                (
                    f"Cannot connect {source.output_type} output to "
                    f"{target.input_type} input."
                ),
            )
        connection = GraphConnection(source_id, target_id)
        if connection in self.connections:
            return ConnectionResult(True, "Those nodes are already connected.")
        if self._would_create_cycle(source_id, target_id):
            return ConnectionResult(False, "Cannot connect nodes in a cycle.")

        removed = tuple(
            existing
            for existing in self.connections
            if existing.target_id == target_id
        )
        self.connections = [
            existing
            for existing in self.connections
            if existing.target_id != target_id
        ]
        self.connections.append(connection)
        return ConnectionResult(True, "Connected nodes.", removed)

    def disconnect(self, source_id: str, target_id: str) -> bool:
        before = len(self.connections)
        self.connections = [
            connection
            for connection in self.connections
            if not (
                connection.source_id == source_id
                and connection.target_id == target_id
            )
        ]
        return len(self.connections) != before

    def set_param(self, node_id: str, name: str, value: Any) -> None:
        self.nodes[node_id].params[name] = value

    def operation_spec(self, operation_id: str) -> OperationSpec:
        return NODE_LIBRARY_BY_ID[operation_id]

    def node_parameter_specs(self, node_id: str) -> tuple[ParameterSpec, ...]:
        node = self.nodes[node_id]
        return self.operation_spec(node.operation_id).parameters

    def input_data_for_node(self, node_id: str):
        sources = self._input_sources(node_id)
        if not sources:
            return None
        return self.outputs.get(sources[0])

    def run(self, input_data) -> dict[str, Any]:
        self.outputs = {node_id: None for node_id in self.nodes}
        remaining = set(self.nodes)
        completed: set[str] = set()

        while remaining:
            runnable = [
                node_id
                for node_id in remaining
                if all(source in completed for source in self._input_sources(node_id))
            ]
            if not runnable:
                break

            for node_id in runnable:
                self.outputs[node_id] = self._run_node(node_id, input_data)
                remaining.remove(node_id)
                completed.add(node_id)
        return self.outputs

    def _run_node(self, node_id: str, input_data):
        node = self.nodes[node_id]
        spec = self.operation_spec(node.operation_id)
        if not spec.has_input:
            return input_data

        sources = self._input_sources(node_id)
        if not sources:
            return None
        source_output = self.outputs.get(sources[0])
        if source_output is None or spec.function is None:
            return None
        return spec.function(source_output, **node.params)

    def _input_sources(self, node_id: str) -> list[str]:
        return [
            connection.source_id
            for connection in self.connections
            if connection.target_id == node_id
        ]

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        downstream = {target_id}
        frontier = [target_id]
        while frontier:
            current = frontier.pop()
            for connection in self.connections:
                if connection.source_id != current:
                    continue
                if connection.target_id == source_id:
                    return True
                if connection.target_id not in downstream:
                    downstream.add(connection.target_id)
                    frontier.append(connection.target_id)
        return False

    def _types_compatible(self, output_type: str, input_type: str | None) -> bool:
        if input_type is None or input_type == "any" or output_type == "any":
            return True
        if input_type == "array":
            return output_type in {"array", "image", "mask"}
        return output_type == input_type


def grouped_palette_specs() -> dict[str, list[OperationSpec]]:
    groups: dict[str, list[OperationSpec]] = {}
    for spec in PALETTE_NODE_LIBRARY:
        groups.setdefault(spec.category, []).append(spec)
    return groups


def _node_id(operation_id: str, index: int) -> str:
    if operation_id == "gaussian_blur" and index == 1:
        return "gaussian"
    if operation_id == "otsu_threshold" and index == 1:
        return "threshold"
    return f"{operation_id}_{index}"


def _clone_node(node: GraphNode) -> GraphNode:
    return GraphNode(
        node.id,
        node.operation_id,
        node.title,
        node.category,
        node.input_type,
        node.output_type,
        dict(node.params),
    )


def connection_pairs(connections: Iterable[GraphConnection]) -> set[tuple[str, str]]:
    return {(connection.source_id, connection.target_id) for connection in connections}
