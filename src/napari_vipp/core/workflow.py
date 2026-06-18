"""Workflow persistence for the VIPP graph.

Serialize a :class:`PrototypePipeline` (nodes, parameters, connections) plus the
optional canvas node positions to a portable JSON document, and rebuild a graph
from such a document. Node titles/categories/port types are derived from the
operation library on load, so files stay compact and use one explicit schema.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from napari_vipp.core.pipeline import (
    NODE_LIBRARY_BY_ID,
    GraphConnection,
    GraphNode,
    PrototypePipeline,
)

WORKFLOW_VERSION = 1
WORKFLOW_TYPE = "napari-vipp-workflow"

Position = tuple[float, float]


def serialize_workflow(
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable dict describing the pipeline graph."""
    positions = positions or {}
    return {
        "type": WORKFLOW_TYPE,
        "version": WORKFLOW_VERSION,
        "nodes": [_node_to_dict(node) for node in pipeline.nodes.values()],
        "connections": [
            {
                "source": connection.source_id,
                "target": connection.target_id,
                "target_port": connection.target_port,
                "source_port": connection.source_port,
            }
            for connection in pipeline.connections
        ],
        "positions": {
            node_id: [float(x), float(y)] for node_id, (x, y) in positions.items()
        },
    }


def deserialize_workflow(data: Any) -> dict[str, Any]:
    """Rebuild nodes, connections, and positions from a workflow dict.

    Returns a dict with keys ``nodes`` (list[GraphNode]),
    ``connections`` (list[GraphConnection]), and ``positions``
    (dict[node_id, (x, y)]). Invalid versions, unknown operations, malformed
    nodes, and dangling connections are rejected with a clear error.
    """
    if not isinstance(data, dict):
        raise ValueError("Workflow file is not a valid object.")
    if data.get("type") != WORKFLOW_TYPE:
        raise ValueError("File is not a napari-vipp workflow.")
    if data.get("version") != WORKFLOW_VERSION:
        raise ValueError(
            f"Unsupported workflow version: {data.get('version')!r}. "
            f"Expected version {WORKFLOW_VERSION}."
        )

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Workflow must contain a non-empty nodes list.")
    nodes = [_node_from_dict(raw, index) for index, raw in enumerate(raw_nodes)]
    if not nodes:
        raise ValueError("Workflow contains no recognised nodes.")
    node_ids = [node.id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Workflow contains duplicate node ids.")

    node_id_set = set(node_ids)
    raw_connections = data.get("connections")
    if not isinstance(raw_connections, list):
        raise ValueError("Workflow connections must be a list.")
    connections: list[GraphConnection] = []
    occupied_inputs: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_connections):
        if not isinstance(raw, dict):
            raise ValueError(f"Connection {index} must be an object.")
        source = _required_text(raw, "source", f"connection {index}")
        target = _required_text(raw, "target", f"connection {index}")
        if source not in node_id_set or target not in node_id_set:
            raise ValueError(
                f"Connection {index} references a missing node: "
                f"{source!r} -> {target!r}."
            )
        target_port = _required_non_negative_int(
            raw, "target_port", f"connection {index}"
        )
        source_port = _required_non_negative_int(
            raw, "source_port", f"connection {index}"
        )
        target_slot = (target, target_port)
        if target_slot in occupied_inputs:
            raise ValueError(
                f"Multiple connections target {target!r} input {target_port}."
            )
        occupied_inputs.add(target_slot)
        connections.append(GraphConnection(source, target, target_port, source_port))

    positions: dict[str, Position] = {}
    raw_positions = data.get("positions")
    if not isinstance(raw_positions, dict):
        raise ValueError("Workflow positions must be an object.")
    for node_id, value in raw_positions.items():
        if node_id not in node_id_set:
            raise ValueError(f"Position references unknown node {node_id!r}.")
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Position for {node_id!r} must contain x and y.")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            for coordinate in value
        ):
            raise ValueError(f"Position for {node_id!r} must contain numeric x and y.")
        positions[node_id] = (float(value[0]), float(value[1]))

    return {"nodes": nodes, "connections": connections, "positions": positions}


def save_workflow(
    path: str | Path,
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
) -> Path:
    """Write the pipeline graph to ``path`` as a JSON workflow file."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = serialize_workflow(pipeline, positions)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def load_workflow(path: str | Path) -> dict[str, Any]:
    """Read a JSON workflow file and return deserialized graph parts."""
    source = Path(path).expanduser()
    data = json.loads(source.read_text(encoding="utf-8"))
    return deserialize_workflow(data)


def _node_to_dict(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "operation_id": node.operation_id,
        "params": dict(node.params),
    }


def _node_from_dict(raw: Any, index: int) -> GraphNode:
    if not isinstance(raw, dict):
        raise ValueError(f"Node {index} must be an object.")
    operation_id = _required_text(raw, "operation_id", f"node {index}")
    spec = NODE_LIBRARY_BY_ID.get(operation_id)
    if spec is None:
        raise ValueError(f"Node {index} uses unknown operation {operation_id!r}.")
    node_id = _required_text(raw, "id", f"node {index}")
    saved = raw.get("params")
    if not isinstance(saved, dict):
        raise ValueError(f"Parameters for node {node_id!r} must be an object.")
    required_params = {param.name for param in spec.parameters}
    missing_params = required_params - saved.keys()
    if missing_params:
        missing = ", ".join(sorted(missing_params))
        raise ValueError(f"Node {node_id!r} is missing required parameters: {missing}.")
    params = dict(saved)
    return GraphNode(
        node_id,
        spec.id,
        spec.title,
        spec.category,
        spec.input_type,
        spec.output_type,
        params,
        spec.max_inputs,
    )


def _required_text(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context.capitalize()} requires non-empty {key!r}.")
    return value


def _required_non_negative_int(
    data: dict[str, Any],
    key: str,
    context: str,
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context.capitalize()} {key!r} must be an integer.")
    if value < 0:
        raise ValueError(f"{context.capitalize()} {key!r} must be non-negative.")
    return value
