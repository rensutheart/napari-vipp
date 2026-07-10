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
    OutputTunnel,
    PrototypePipeline,
)

WORKFLOW_VERSION = 1
WORKFLOW_TYPE = "napari-vipp-workflow"

Position = tuple[float, float]


def serialize_workflow(
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
    notes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable dict describing the pipeline graph."""
    positions = positions or {}
    node_id_set = set(pipeline.nodes)
    unknown_positions = set(positions) - node_id_set
    if unknown_positions:
        unknown = ", ".join(repr(node_id) for node_id in sorted(unknown_positions))
        raise ValueError(f"Workflow positions reference unknown nodes: {unknown}.")
    document = {
        "type": WORKFLOW_TYPE,
        "version": WORKFLOW_VERSION,
        "nodes": [_node_to_dict(node) for node in pipeline.nodes.values()],
        "connections": [
            {
                "source": connection.source_id,
                "target": connection.target_id,
                "target_port": connection.target_port,
                "source_port": connection.source_port,
                **(
                    {"tunnel": connection.tunnel_name}
                    if connection.tunnel_name
                    else {}
                ),
            }
            for connection in pipeline.connections
        ],
        "tunnels": [
            {
                "name": tunnel.name,
                "source": tunnel.source_id,
                "source_port": tunnel.source_port,
            }
            for tunnel in pipeline.output_tunnel_list()
        ],
        "positions": {
            node_id: [float(x), float(y)] for node_id, (x, y) in positions.items()
        },
        "notes": [_note_to_dict(note) for note in notes or ()],
    }
    workflow_metadata = _workflow_metadata_to_dict(metadata, node_id_set)
    if workflow_metadata:
        document["metadata"] = workflow_metadata
    return document


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
    raw_tunnels = data.get("tunnels", [])
    if not isinstance(raw_tunnels, list):
        raise ValueError("Workflow tunnels must be a list.")
    output_tunnels: list[OutputTunnel] = []
    tunnel_names: set[str] = set()
    tunnel_sources: dict[str, tuple[str, int]] = {}
    occupied_outputs: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_tunnels):
        if not isinstance(raw, dict):
            raise ValueError(f"Tunnel {index} must be an object.")
        name = _required_text(raw, "name", f"tunnel {index}")
        key = _tunnel_key(name)
        if key in tunnel_names:
            raise ValueError(f"Workflow contains duplicate tunnel name {name!r}.")
        source = _required_text(raw, "source", f"tunnel {index}")
        if source not in node_id_set:
            raise ValueError(
                f"Tunnel {index} references missing source node {source!r}."
            )
        source_port = _required_non_negative_int(
            raw, "source_port", f"tunnel {index}"
        )
        source_slot = (source, source_port)
        if source_slot in occupied_outputs:
            raise ValueError(
                f"Multiple tunnels are assigned to {source!r} output {source_port}."
            )
        tunnel_names.add(key)
        occupied_outputs.add(source_slot)
        tunnel_sources[key] = source_slot
        output_tunnels.append(OutputTunnel(name.strip(), source, source_port))

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
        tunnel_name = ""
        if "tunnel" in raw:
            tunnel_name = _required_text(raw, "tunnel", f"connection {index}")
            tunnel_key = _tunnel_key(tunnel_name)
            if tunnel_key not in tunnel_sources:
                raise ValueError(
                    f"Connection {index} references unknown tunnel "
                    f"{tunnel_name!r}."
                )
            if tunnel_sources[tunnel_key] != (source, source_port):
                raise ValueError(
                    f"Connection {index} tunnel {tunnel_name!r} does not match "
                    "its declared source output."
                )
        target_slot = (target, target_port)
        if target_slot in occupied_inputs:
            raise ValueError(
                f"Multiple connections target {target!r} input {target_port}."
            )
        occupied_inputs.add(target_slot)
        connections.append(
            GraphConnection(source, target, target_port, source_port, tunnel_name)
        )

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

    notes = _notes_from_data(data.get("notes", []), node_id_set)
    metadata = _workflow_metadata_to_dict(data.get("metadata", {}), node_id_set)

    return {
        "nodes": nodes,
        "connections": connections,
        "positions": positions,
        "output_tunnels": output_tunnels,
        "notes": notes,
        "metadata": metadata,
    }


def save_workflow(
    path: str | Path,
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
    notes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the pipeline graph to ``path`` as a JSON workflow file."""
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError("Workflow save path cannot be blank.")
    target = Path(raw_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = serialize_workflow(pipeline, positions, notes, metadata)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def load_workflow(path: str | Path) -> dict[str, Any]:
    """Read a JSON workflow file and return deserialized graph parts."""
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError("Workflow path cannot be blank.")
    source = Path(raw_path).expanduser()
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


def _note_to_dict(note: dict[str, Any]) -> dict[str, Any]:
    position = note.get("position", (0.0, 0.0))
    x, y = tuple(position)
    attached_node = str(note.get("attached_node", "") or "").strip()
    result = {
        "id": str(note.get("id", "")).strip(),
        "text": str(note.get("text", "")),
        "position": [float(x), float(y)],
        "width": float(note.get("width", 240.0)),
    }
    if attached_node:
        result["attached_node"] = attached_node
    return result


def _notes_from_data(raw_notes: Any, node_id_set: set[str]) -> list[dict[str, Any]]:
    if raw_notes is None:
        return []
    if not isinstance(raw_notes, list):
        raise ValueError("Workflow notes must be a list.")
    notes: list[dict[str, Any]] = []
    note_ids: set[str] = set()
    for index, raw in enumerate(raw_notes):
        if not isinstance(raw, dict):
            raise ValueError(f"Note {index} must be an object.")
        note_id = _required_text(raw, "id", f"note {index}")
        key = note_id.casefold()
        if key in note_ids:
            raise ValueError(f"Workflow contains duplicate note id {note_id!r}.")
        note_ids.add(key)
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"Text for note {note_id!r} must be a string.")
        position = raw.get("position")
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"Position for note {note_id!r} must contain x and y.")
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            for coordinate in position
        ):
            raise ValueError(
                f"Position for note {note_id!r} must contain numeric x and y."
            )
        width = raw.get("width", 240.0)
        if (
            isinstance(width, bool)
            or not isinstance(width, (int, float))
            or not math.isfinite(width)
            or width <= 0
        ):
            raise ValueError(f"Width for note {note_id!r} must be a positive number.")
        attached_node = str(raw.get("attached_node", "") or "").strip()
        if attached_node and attached_node not in node_id_set:
            raise ValueError(
                f"Note {note_id!r} references missing attached node "
                f"{attached_node!r}."
            )
        note = {
            "id": note_id.strip(),
            "text": text,
            "position": (float(position[0]), float(position[1])),
            "width": float(width),
        }
        if attached_node:
            note["attached_node"] = attached_node
        notes.append(note)
    return notes


def _workflow_metadata_to_dict(
    raw_metadata: Any,
    node_id_set: set[str],
) -> dict[str, Any]:
    if raw_metadata is None:
        return {}
    if not isinstance(raw_metadata, dict):
        raise ValueError("Workflow metadata must be an object.")
    raw_vipp = raw_metadata.get("vipp", {})
    if raw_vipp is None:
        return {}
    if not isinstance(raw_vipp, dict):
        raise ValueError("VIPP workflow metadata must be an object.")

    vipp: dict[str, Any] = {}
    if "inspector" in raw_vipp:
        vipp["inspector"] = _inspector_metadata_to_dict(
            raw_vipp["inspector"],
            node_id_set,
        )
    if "thumbnails" in raw_vipp:
        vipp["thumbnails"] = _thumbnail_metadata_to_dict(
            raw_vipp["thumbnails"],
            node_id_set,
        )
    return {"vipp": vipp} if vipp else {}


def _inspector_metadata_to_dict(
    raw_inspector: Any,
    node_id_set: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_inspector, dict):
        raise ValueError("Workflow inspector metadata must be an object.")
    result: dict[str, Any] = {}
    if "selected_node_id" in raw_inspector:
        selected_node_id = _optional_node_id(
            raw_inspector,
            "selected_node_id",
            "workflow inspector metadata",
            node_id_set,
        )
        if selected_node_id:
            result["selected_node_id"] = selected_node_id
    if "right_panel_visible" in raw_inspector:
        right_panel_visible = raw_inspector.get("right_panel_visible")
        if not isinstance(right_panel_visible, bool):
            raise ValueError(
                "Workflow inspector metadata 'right_panel_visible' must be a "
                "boolean."
            )
        result["right_panel_visible"] = right_panel_visible
    return result


def _thumbnail_metadata_to_dict(
    raw_thumbnails: Any,
    node_id_set: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_thumbnails, dict):
        raise ValueError("Workflow thumbnail metadata must be an object.")
    raw_disabled = raw_thumbnails.get("disabled_node_ids", [])
    if raw_disabled is None:
        raw_disabled = []
    if not isinstance(raw_disabled, list):
        raise ValueError(
            "Workflow thumbnail metadata 'disabled_node_ids' must be a list."
        )
    disabled_node_ids: list[str] = []
    seen: set[str] = set()
    for value in raw_disabled:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Workflow thumbnail metadata disabled node ids must be "
                "non-empty strings."
            )
        node_id = value.strip()
        if node_id not in node_id_set:
            raise ValueError(
                f"Workflow thumbnail metadata references missing node "
                f"{node_id!r}."
            )
        key = node_id.casefold()
        if key in seen:
            raise ValueError(
                "Workflow thumbnail metadata contains duplicate disabled node "
                f"id {node_id!r}."
            )
        seen.add(key)
        disabled_node_ids.append(node_id)
    return {"disabled_node_ids": disabled_node_ids}


def _optional_node_id(
    data: dict[str, Any],
    key: str,
    context: str,
    node_id_set: set[str],
) -> str:
    value = data.get(key, "")
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context.capitalize()} {key!r} must be a node id string.")
    node_id = value.strip()
    if node_id not in node_id_set:
        raise ValueError(
            f"{context.capitalize()} references missing node {node_id!r}."
        )
    return node_id


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


def _tunnel_key(name: str) -> str:
    return str(name or "").strip().casefold()
