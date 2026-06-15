"""Workflow persistence for the VIPP graph.

Serialize a :class:`PrototypePipeline` (nodes, parameters, connections) plus the
optional canvas node positions to a portable JSON document, and rebuild a graph
from such a document. Node titles/categories/port types are derived from the
operation library on load, so files stay small and forward-compatible.
"""

from __future__ import annotations

import json
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
            node_id: [float(x), float(y)]
            for node_id, (x, y) in positions.items()
        },
    }


def deserialize_workflow(data: Any) -> dict[str, Any]:
    """Rebuild nodes, connections, and positions from a workflow dict.

    Returns a dict with keys ``nodes`` (list[GraphNode]),
    ``connections`` (list[GraphConnection]), and ``positions``
    (dict[node_id, (x, y)]). Unknown operations and dangling connections are
    skipped so partially compatible files still load.
    """
    if not isinstance(data, dict):
        raise ValueError("Workflow file is not a valid object.")
    if data.get("type") != WORKFLOW_TYPE:
        raise ValueError("File is not a napari-vipp workflow.")

    nodes: list[GraphNode] = []
    for raw in data.get("nodes", []):
        node = _node_from_dict(raw)
        if node is not None:
            nodes.append(node)
    if not nodes:
        raise ValueError("Workflow contains no recognised nodes.")

    node_ids = {node.id for node in nodes}
    connections: list[GraphConnection] = []
    for raw in data.get("connections", []):
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", ""))
        target = str(raw.get("target", ""))
        if source in node_ids and target in node_ids:
            try:
                target_port = int(raw.get("target_port", 0))
            except (TypeError, ValueError):
                target_port = 0
            try:
                source_port = int(raw.get("source_port", 0))
            except (TypeError, ValueError):
                source_port = 0
            connections.append(
                GraphConnection(
                    source,
                    target,
                    max(target_port, 0),
                    max(source_port, 0),
                )
            )

    positions: dict[str, Position] = {}
    raw_positions = data.get("positions") or {}
    if isinstance(raw_positions, dict):
        for node_id, value in raw_positions.items():
            if (
                node_id in node_ids
                and isinstance(value, (list, tuple))
                and len(value) == 2
            ):
                try:
                    positions[node_id] = (float(value[0]), float(value[1]))
                except (TypeError, ValueError):
                    continue

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


def _node_from_dict(raw: Any) -> GraphNode | None:
    if not isinstance(raw, dict):
        return None
    operation_id = str(raw.get("operation_id", ""))
    spec = NODE_LIBRARY_BY_ID.get(operation_id)
    if spec is None:
        return None
    node_id = str(raw.get("id") or operation_id)
    params: dict[str, Any] = {param.name: param.default for param in spec.parameters}
    saved = raw.get("params")
    if isinstance(saved, dict):
        params.update(saved)
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
