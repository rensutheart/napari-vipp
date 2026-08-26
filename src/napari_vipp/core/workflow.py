"""Workflow persistence for the VIPP graph.

Serialize a :class:`PrototypePipeline` (nodes, parameters, connections) plus the
optional canvas node positions to a portable JSON document, and rebuild a graph
from such a document. Node titles/categories/port types are derived from the
operation library on load, so files stay compact and use one explicit schema.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from napari_vipp.core.atomic_io import atomic_write_json
from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    NodeComputePreference,
)
from napari_vipp.core.pipeline import (
    GraphConnection,
    GraphNode,
    OutputTunnel,
    PrototypePipeline,
    graph_node_from_persisted_params,
)
from napari_vipp.core.snapshots import (
    GraphSnapshot,
    NodeSnapshot,
    WorkflowNoteSnapshot,
    WorkflowSnapshot,
)
from napari_vipp.core.source_item_persistence import (
    canonicalize_source_item_params,
)

WORKFLOW_VERSION = 5
LEGACY_COMPUTE_WORKFLOW_VERSION = 3
LEGACY_SOURCE_ITEM_WORKFLOW_VERSION = 4
WORKFLOW_TYPE = "napari-vipp-workflow"

Position = tuple[float, float]


def serialize_workflow(
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
    notes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    metadata: dict[str, Any] | None = None,
    compute_request: ComputeRequest | Mapping[str, object] | None = None,
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
        "execution": {
            "compute": _workflow_compute_to_dict(compute_request, node_id_set),
        },
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
    document_version = data.get("version")
    if type(document_version) is not int or document_version not in {
        LEGACY_COMPUTE_WORKFLOW_VERSION,
        LEGACY_SOURCE_ITEM_WORKFLOW_VERSION,
        WORKFLOW_VERSION,
    }:
        migration_guidance = ""
        if type(document_version) is int and document_version in {1, 2}:
            migration_guidance = (
                " Earlier schemas are not auto-migrated because the current "
                "schema requires explicit scientific axis, color, and intensity "
                "semantics. Use the VIPP release that created the workflow to "
                "inspect it, then recreate and verify it in the current release; "
                "do not edit the JSON version number alone."
            )
        raise ValueError(
            f"Unsupported workflow version: {document_version!r}. "
            f"Expected version {LEGACY_COMPUTE_WORKFLOW_VERSION}, "
            f"{LEGACY_SOURCE_ITEM_WORKFLOW_VERSION}, or {WORKFLOW_VERSION}."
            f"{migration_guidance}"
        )

    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Workflow must contain a non-empty nodes list.")
    nodes = [
        _node_from_dict(raw, index, document_version=document_version)
        for index, raw in enumerate(raw_nodes)
    ]
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
    compute_request = (
        ComputeRequest(mode=ComputeMode.CPU)
        if document_version == LEGACY_COMPUTE_WORKFLOW_VERSION
        else _compute_request_from_execution(data.get("execution"), node_id_set)
    )

    restored = {
        "nodes": nodes,
        "connections": connections,
        "positions": positions,
        "output_tunnels": output_tunnels,
        "notes": notes,
        "metadata": metadata,
        "compute_request": compute_request,
    }
    if "batch_config" in data:
        restored["batch_config"] = _batch_config_document_from_data(
            data.get("batch_config")
        )
    return restored


def workflow_snapshot_from_pipeline(
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
    notes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    metadata: dict[str, Any] | None = None,
    compute_request: ComputeRequest | Mapping[str, object] | None = None,
) -> WorkflowSnapshot:
    """Capture a validated workflow snapshot without exposing live mappings."""
    document = serialize_workflow(
        pipeline,
        positions,
        notes,
        metadata,
        compute_request,
    )
    if document["nodes"]:
        return workflow_snapshot_from_document(document)

    # Empty graphs are a valid transient editor/history state even though the
    # persisted workflow document contract deliberately requires at least one
    # node. Validate the remaining fields through the same helpers without
    # weakening ``workflow_snapshot_from_document`` or workflow loading.
    node_ids: set[str] = set()
    validated_notes = _notes_from_data(document["notes"], node_ids)
    validated_metadata = _workflow_metadata_to_dict(
        document.get("metadata", {}),
        node_ids,
    )
    return WorkflowSnapshot(
        GraphSnapshot.from_pipeline(pipeline),
        positions=document["positions"],
        notes=(WorkflowNoteSnapshot.from_mapping(note) for note in validated_notes),
        metadata=validated_metadata,
        compute_request=_portable_compute_request(compute_request),
    )


def workflow_snapshot_from_document(data: Any) -> WorkflowSnapshot:
    """Decode and structurally validate a current-schema workflow snapshot."""
    restored = deserialize_workflow(data)
    graph = GraphSnapshot(
        (
            NodeSnapshot(node.id, node.operation_id, node.params)
            for node in restored["nodes"]
        ),
        restored["connections"],
        restored["output_tunnels"],
    )
    # Deserialization validates the persisted records; materialization adds the
    # graph-level port, type, duplicate-input, and cycle validation performed by
    # the established pipeline restoration boundary.
    graph.to_pipeline()
    return WorkflowSnapshot(
        graph,
        positions=restored["positions"],
        notes=(WorkflowNoteSnapshot.from_mapping(note) for note in restored["notes"]),
        metadata=restored["metadata"],
        compute_request=restored["compute_request"],
    )


def workflow_document_from_snapshot(snapshot: WorkflowSnapshot) -> dict[str, Any]:
    """Materialize a snapshot as the canonical current workflow document."""
    pipeline = snapshot.graph.to_pipeline()
    return serialize_workflow(
        pipeline,
        positions=snapshot.positions_dict(),
        notes=[note.to_mapping() for note in snapshot.notes],
        metadata=snapshot.metadata,
        compute_request=snapshot.compute_request,
    )


def save_workflow(
    path: str | Path,
    pipeline: PrototypePipeline,
    positions: dict[str, Position] | None = None,
    notes: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    metadata: dict[str, Any] | None = None,
    compute_request: ComputeRequest | Mapping[str, object] | None = None,
) -> Path:
    """Write the pipeline graph to ``path`` as a JSON workflow file."""
    document = serialize_workflow(
        pipeline,
        positions,
        notes,
        metadata,
        compute_request,
    )
    return save_workflow_document(path, document)


def save_workflow_document(path: str | Path, document: object) -> Path:
    """Validate, migrate, and atomically write one canonical workflow document."""
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError("Workflow save path cannot be blank.")
    target = Path(raw_path).expanduser()
    canonical = canonical_workflow_document(document)
    return atomic_write_json(
        target,
        canonical,
        ensure_ascii=True,
        trailing_newline=False,
    )


def canonical_workflow_document(data: Any) -> dict[str, Any]:
    """Migrate any supported workflow to the one current write schema."""

    restored = deserialize_workflow(data)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored["output_tunnels"],
    )
    canonical = serialize_workflow(
        pipeline,
        positions=restored["positions"],
        notes=restored["notes"],
        metadata=restored["metadata"],
        compute_request=restored["compute_request"],
    )
    if "batch_config" in restored:
        canonical["batch_config"] = restored["batch_config"]
    return canonical


def load_workflow(path: str | Path) -> dict[str, Any]:
    """Read a JSON workflow file and return deserialized graph parts."""
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError("Workflow path cannot be blank.")
    source = Path(raw_path).expanduser()
    data = json.loads(source.read_text(encoding="utf-8"))
    return deserialize_workflow(data)


def _node_to_dict(node: GraphNode) -> dict[str, Any]:
    params = (
        canonicalize_source_item_params(node.params)
        if node.operation_id == "input"
        else dict(node.params)
    )
    return {
        "id": node.id,
        "operation_id": node.operation_id,
        "params": params,
    }


def _node_from_dict(
    raw: Any,
    index: int,
    *,
    document_version: int,
) -> GraphNode:
    if not isinstance(raw, dict):
        raise ValueError(f"Node {index} must be an object.")
    operation_id = raw.get("operation_id")
    saved_params = raw.get("params")
    if operation_id == "input" and isinstance(saved_params, Mapping):
        try:
            saved_params = canonicalize_source_item_params(saved_params)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Workflow v{document_version} node {index} contains an invalid "
                f"canonical SourceItem: {exc}"
            ) from exc
    return graph_node_from_persisted_params(
        raw.get("id"),
        operation_id,
        saved_params,
        index=index,
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
    if "compute_optimizer" in raw_vipp:
        vipp["compute_optimizer"] = _compute_optimizer_metadata_to_dict(
            raw_vipp["compute_optimizer"],
            node_id_set,
        )
    return {"vipp": vipp} if vipp else {}


def _portable_compute_request(
    value: ComputeRequest | Mapping[str, object] | None,
) -> ComputeRequest:
    """Return authored compute intent without machine-local run settings."""
    if value is None:
        request = ComputeRequest(mode=ComputeMode.CPU)
    elif isinstance(value, ComputeRequest):
        request = value
    elif isinstance(value, Mapping):
        if "precision_policy" in value or "workload_policy" in value:
            request = _compute_request_from_compute_block(value, None)
        else:
            request = ComputeRequest.from_dict(value)
    else:
        raise TypeError("Workflow compute intent must be a ComputeRequest or object.")
    return ComputeRequest(
        mode=request.mode,
        node_preferences=request.node_preferences,
        fallback_policy=request.fallback_policy,
        precision_policy_id=request.precision_policy_id,
        workload_policy_id=request.workload_policy_id,
    )


def _workflow_compute_to_dict(
    value: ComputeRequest | Mapping[str, object] | None,
    node_id_set: set[str],
) -> dict[str, object]:
    request = _portable_compute_request(value)
    unknown_node_ids = set(request.node_preferences) - node_id_set
    if unknown_node_ids:
        unknown = ", ".join(repr(node_id) for node_id in sorted(unknown_node_ids))
        raise ValueError(
            f"Workflow compute preferences reference unknown nodes: {unknown}."
        )
    return {
        "mode": request.mode.value,
        "fallback_policy": request.fallback_policy.value,
        "node_preferences": {
            node_id: _node_preference_text(preference)
            for node_id, preference in request.node_preferences.items()
        },
        "precision_policy": request.precision_policy_id,
        "workload_policy": request.workload_policy_id,
    }


def _compute_request_from_execution(
    raw_execution: Any,
    node_id_set: set[str],
) -> ComputeRequest:
    if not isinstance(raw_execution, dict):
        raise ValueError("Workflow execution must be an object.")
    unknown = set(raw_execution) - {"compute"}
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"Unknown workflow execution field(s): {names}.")
    if "compute" not in raw_execution:
        raise ValueError("Workflow execution requires a 'compute' object.")
    return _compute_request_from_compute_block(
        raw_execution["compute"],
        node_id_set,
    )


def _compute_request_from_compute_block(
    raw_compute: Any,
    node_id_set: set[str] | None,
) -> ComputeRequest:
    if not isinstance(raw_compute, Mapping):
        raise ValueError("Workflow execution compute must be an object.")
    required = {
        "mode",
        "fallback_policy",
        "node_preferences",
        "precision_policy",
        "workload_policy",
    }
    unknown = set(raw_compute) - required
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"Unknown workflow compute field(s): {names}.")
    missing = required - set(raw_compute)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Workflow compute is missing required field(s): {names}.")
    raw_preferences = raw_compute["node_preferences"]
    if not isinstance(raw_preferences, Mapping):
        raise ValueError("Workflow compute node_preferences must be an object.")
    preferences: dict[str, NodeComputePreference] = {}
    for raw_node_id, raw_preference in raw_preferences.items():
        if not isinstance(raw_node_id, str) or not raw_node_id.strip():
            raise ValueError(
                "Workflow compute preference node ids must be non-empty strings."
            )
        node_id = raw_node_id.strip()
        if node_id_set is not None and node_id not in node_id_set:
            raise ValueError(
                f"Workflow compute preference references missing node {node_id!r}."
            )
        if node_id in preferences:
            raise ValueError(
                "Workflow compute preferences contain duplicate normalized "
                f"node id {node_id!r}."
            )
        if not isinstance(raw_preference, str) or not raw_preference.strip():
            raise ValueError(
                f"Workflow compute preference for {node_id!r} must be a "
                "non-empty string."
            )
        preferences[node_id] = NodeComputePreference.parse(raw_preference)
    policies: dict[str, str] = {}
    for field_name in ("precision_policy", "workload_policy"):
        raw_policy = raw_compute[field_name]
        if not isinstance(raw_policy, str) or not raw_policy.strip():
            raise ValueError(
                f"Workflow compute {field_name} must be a non-empty string."
            )
        policies[field_name] = raw_policy.strip()
    return ComputeRequest(
        mode=raw_compute["mode"],
        node_preferences=preferences,
        fallback_policy=raw_compute["fallback_policy"],
        precision_policy_id=policies["precision_policy"],
        workload_policy_id=policies["workload_policy"],
    )


def _node_preference_text(preference: NodeComputePreference) -> str:
    if preference.value:
        return f"{preference.kind.value}:{preference.value}"
    return preference.kind.value


def _batch_config_document_from_data(raw_config: Any) -> dict[str, Any]:
    """Preserve an attached batch document without coupling workflow schemas."""
    if not isinstance(raw_config, dict):
        raise ValueError("Workflow batch_config must be an object.")
    try:
        encoded = json.dumps(
            raw_config,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Workflow batch_config must contain finite JSON values."
        ) from exc
    return json.loads(encoded)


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
    if "display_profiles" in raw_inspector:
        profiles = _inspect_display_profiles_to_list(
            raw_inspector.get("display_profiles"),
            node_id_set,
        )
        if profiles:
            result["display_profiles"] = profiles
    return result


def _inspect_display_profiles_to_list(
    raw_profiles: Any,
    node_id_set: set[str],
) -> list[dict[str, Any]]:
    if raw_profiles is None:
        return []
    if not isinstance(raw_profiles, list):
        raise ValueError(
            "Workflow inspector metadata 'display_profiles' must be a list."
        )
    profiles: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for index, raw in enumerate(raw_profiles):
        context = f"workflow Inspect display profile {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{context.capitalize()} must be an object.")
        node_id = _optional_node_id(raw, "node_id", context, node_id_set)
        if not node_id:
            raise ValueError(f"{context.capitalize()} requires 'node_id'.")
        output_port = _required_non_negative_int(raw, "output_port", context)
        data_kind = _required_text(raw, "data_kind", context).strip()
        display_kind = _required_text(raw, "display_kind", context).strip()
        display_ndim = _required_non_negative_int(raw, "display_ndim", context)
        if display_ndim < 1:
            raise ValueError(f"{context.capitalize()} display_ndim must be positive.")
        display_rgb = _required_profile_bool(raw, "display_rgb", context)
        display_rgb_as_channels = _required_profile_bool(
            raw,
            "display_rgb_as_channels",
            context,
        )
        channel_index = None
        if "display_rgb_channel_index" in raw:
            channel_index = _required_non_negative_int(
                raw,
                "display_rgb_channel_index",
                context,
            )
        if display_rgb_as_channels and not display_rgb:
            raise ValueError(
                f"{context.capitalize()} RGB channel surfaces require "
                "display_rgb to be true."
            )
        if display_rgb_as_channels != (channel_index is not None):
            raise ValueError(
                f"{context.capitalize()} display_rgb_channel_index is required "
                "exactly for RGB channel surfaces."
            )
        if channel_index is not None and channel_index not in {0, 1, 2}:
            raise ValueError(
                f"{context.capitalize()} display_rgb_channel_index must be 0, "
                "1, or 2."
            )
        key = (
            node_id,
            output_port,
            data_kind,
            display_kind,
            display_rgb,
            display_rgb_as_channels,
            channel_index,
            display_ndim,
        )
        if key in seen:
            raise ValueError(
                f"Workflow inspector metadata contains duplicate display profile "
                f"for node {node_id!r}."
            )
        seen.add(key)
        profile: dict[str, Any] = {
            "node_id": node_id,
            "output_port": output_port,
            "data_kind": data_kind,
            "display_kind": display_kind,
            "display_rgb": display_rgb,
            "display_rgb_as_channels": display_rgb_as_channels,
            "display_ndim": display_ndim,
        }
        if channel_index is not None:
            profile["display_rgb_channel_index"] = channel_index
        profile["settings"] = _inspect_general_display_settings_to_dict(
            raw.get("settings", {}),
            context,
        )
        intensity_settings = _inspect_intensity_settings_to_dict(
            raw.get("intensity_settings", {}),
            context,
        )
        if intensity_settings:
            profile["intensity_settings"] = intensity_settings
        profiles.append(profile)
    return profiles


def _inspect_general_display_settings_to_dict(
    raw_settings: Any,
    context: str,
) -> dict[str, Any]:
    if not isinstance(raw_settings, dict):
        raise ValueError(f"{context.capitalize()} settings must be an object.")
    result: dict[str, Any] = {}
    text_keys = (
        "colormap",
        "blending",
        "interpolation2d",
        "interpolation3d",
        "projection_mode",
        "rendering",
        "depiction",
    )
    for key in text_keys:
        if key not in raw_settings:
            continue
        value = raw_settings.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{context.capitalize()} setting {key!r} must be non-empty text."
            )
        result[key] = value.strip()
    if "visible" in raw_settings:
        result["visible"] = _required_profile_bool(
            raw_settings,
            "visible",
            f"{context} settings",
        )
    for key in ("opacity", "gamma", "attenuation"):
        if key not in raw_settings:
            continue
        value = _finite_profile_number(raw_settings.get(key), key, context)
        if key in {"opacity", "attenuation"} and not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{context.capitalize()} setting {key!r} must be between 0 and 1."
            )
        if key == "gamma" and value <= 0.0:
            raise ValueError(
                f"{context.capitalize()} setting 'gamma' must be positive."
            )
        result[key] = value
    return result


def _inspect_intensity_settings_to_dict(
    raw_by_dtype: Any,
    context: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_by_dtype, dict):
        raise ValueError(
            f"{context.capitalize()} intensity_settings must be an object."
        )
    result: dict[str, dict[str, Any]] = {}
    for raw_dtype, raw_settings in raw_by_dtype.items():
        if not isinstance(raw_dtype, str) or not raw_dtype.strip():
            raise ValueError(
                f"{context.capitalize()} intensity dtype keys must be non-empty "
                "text."
            )
        dtype = raw_dtype.strip()
        if not isinstance(raw_settings, dict):
            raise ValueError(
                f"{context.capitalize()} intensity settings for {dtype!r} must "
                "be an object."
            )
        settings: dict[str, Any] = {}
        if "iso_threshold" in raw_settings:
            settings["iso_threshold"] = _finite_profile_number(
                raw_settings.get("iso_threshold"),
                "iso_threshold",
                context,
            )
        if "contrast_limits" in raw_settings:
            limits = raw_settings.get("contrast_limits")
            if not isinstance(limits, list) or len(limits) != 2:
                raise ValueError(
                    f"{context.capitalize()} contrast_limits must contain two "
                    "finite numbers."
                )
            lower = _finite_profile_number(limits[0], "contrast_limits", context)
            upper = _finite_profile_number(limits[1], "contrast_limits", context)
            if lower >= upper:
                raise ValueError(
                    f"{context.capitalize()} contrast_limits must be increasing."
                )
            settings["contrast_limits"] = [lower, upper]
        result[dtype] = settings
    return result


def _required_profile_bool(data: dict[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context.capitalize()} {key!r} must be a boolean.")
    return value


def _finite_profile_number(value: Any, key: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{context.capitalize()} setting {key!r} must be a finite number."
        )
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{context.capitalize()} setting {key!r} must be a finite number."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"{context.capitalize()} setting {key!r} must be a finite number."
        )
    return number


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


def _compute_optimizer_metadata_to_dict(
    raw_optimizer: Any,
    node_id_set: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_optimizer, dict):
        raise ValueError("Workflow compute optimizer metadata must be an object.")
    unknown = set(raw_optimizer) - {"locked_node_ids"}
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(
            f"Unknown workflow compute optimizer metadata field(s): {names}."
        )
    raw_locked = raw_optimizer.get("locked_node_ids", [])
    if not isinstance(raw_locked, list):
        raise ValueError(
            "Workflow compute optimizer metadata locked_node_ids must be a list."
        )
    locked: list[str] = []
    for raw_node_id in raw_locked:
        if not isinstance(raw_node_id, str) or not raw_node_id.strip():
            raise ValueError(
                "Workflow compute optimizer lock IDs must be non-empty strings."
            )
        node_id = raw_node_id.strip()
        if node_id not in node_id_set:
            raise ValueError(
                "Workflow compute optimizer metadata references missing node "
                f"{node_id!r}."
            )
        locked.append(node_id)
    if len(set(locked)) != len(locked):
        raise ValueError("Workflow compute optimizer lock IDs must be unique.")
    return {"locked_node_ids": sorted(locked)}


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
