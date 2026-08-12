"""Portable, validated clipboard fragments for VIPP workflow graphs.

This module deliberately has no Qt dependency.  The graph view can put the
encoded bytes on any clipboard, while batch tools and tests can use the same
codec without constructing a GUI.

Fragments use source-local keys (``n0``, ``n1``, ...) instead of workflow node
IDs.  A paste operation must allocate new workflow IDs and remap these local
keys.  Only authored, portable state is captured: calculated values, cache
state, pins, runtime decisions, and accepted benchmark results never enter this
format.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

from napari_vipp.core.compute import (
    NodeComputePreference,
    NodePreferenceKind,
)
from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
    GraphConnection,
    GraphNode,
    OutputTunnel,
    PrototypePipeline,
    graph_node_from_persisted_params,
    optional_persisted_parameter_spec,
)

GRAPH_FRAGMENT_MIME_TYPE = "application/x-napari-vipp-graph-fragment+json"
GRAPH_FRAGMENT_KIND = "napari-vipp-graph-fragment"
GRAPH_FRAGMENT_VERSION = 1
MAX_GRAPH_FRAGMENT_BYTES = 1_000_000
MAX_GRAPH_FRAGMENT_NODES = 512
MAX_GRAPH_FRAGMENT_CONNECTIONS = 4_096
MAX_GRAPH_FRAGMENT_NOTES = 512
MAX_GRAPH_FRAGMENT_COORDINATE_ABS = 1_000_000.0
MAX_GRAPH_FRAGMENT_NOTE_WIDTH = 10_000.0

# Private node state is non-transferable by default.  This one setting is a
# genuine user choice rather than calculated/cache/UI state, so it is the only
# explicitly admitted private parameter.
TRANSFERABLE_PRIVATE_PARAMETER_NAMES = frozenset(
    {MANUAL_AUTO_RECALCULATE_PARAM}
)

# These names are accepted by workflow persistence for reconstruction or
# compatibility, but are inferred from connected data and must be recalculated
# at the paste destination.
NONTRANSFERABLE_OPTIONAL_PARAMETER_NAMES = frozenset(
    {"resolved_spatial_ndim"}
)

_LOCAL_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_MAX_JSON_DEPTH = 32


class GraphFragmentError(ValueError):
    """Raised when clipboard bytes do not describe a safe graph fragment."""


@dataclass(frozen=True, init=False, slots=True)
class GraphFragmentNode:
    """One node in a fragment, identified only within that fragment."""

    key: str
    operation_id: str
    position: tuple[float, float]
    compute_preference: NodeComputePreference | None
    optimizer_locked: bool
    _params: dict[str, Any] = field(repr=False)

    __hash__ = None

    def __init__(
        self,
        key: str,
        operation_id: str,
        params: Mapping[str, Any],
        position: Sequence[Real] = (0.0, 0.0),
        compute_preference: (
            NodeComputePreference | str | Mapping[str, object] | None
        ) = None,
        optimizer_locked: bool = False,
    ) -> None:
        normalized_params = _normalize_json_object(params, context="Node parameters")
        preference = _normalize_compute_preference(compute_preference)
        if not isinstance(optimizer_locked, bool):
            raise GraphFragmentError("Node optimizer_locked must be a boolean.")
        object.__setattr__(self, "key", _normalize_local_key(key, context="Node key"))
        object.__setattr__(
            self,
            "operation_id",
            _normalize_text(operation_id, context="Node operation_id"),
        )
        object.__setattr__(
            self,
            "position",
            _normalize_position(position, context=f"Node {key!r} position"),
        )
        object.__setattr__(self, "compute_preference", preference)
        object.__setattr__(self, "optimizer_locked", optimizer_locked)
        object.__setattr__(self, "_params", normalized_params)

    @property
    def params(self) -> dict[str, Any]:
        """Return detached transferable parameters."""
        return deepcopy(self._params)


@dataclass(frozen=True, slots=True)
class GraphFragmentConnection:
    """An edge whose endpoints are local fragment node keys."""

    source: str
    target: str
    target_port: int = 0
    source_port: int = 0
    tunnel: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _normalize_local_key(self.source, context="Connection source"),
        )
        object.__setattr__(
            self,
            "target",
            _normalize_local_key(self.target, context="Connection target"),
        )
        object.__setattr__(
            self,
            "target_port",
            _normalize_non_negative_int(
                self.target_port, context="Connection target_port"
            ),
        )
        object.__setattr__(
            self,
            "source_port",
            _normalize_non_negative_int(
                self.source_port, context="Connection source_port"
            ),
        )
        if not isinstance(self.tunnel, str):
            raise GraphFragmentError("Connection tunnel must be text.")
        object.__setattr__(self, "tunnel", self.tunnel)


@dataclass(frozen=True, slots=True)
class GraphFragmentTunnel:
    """A named internal tunnel whose subscribers are also in the fragment."""

    name: str
    source: str
    source_port: int = 0

    def __post_init__(self) -> None:
        name = _normalize_tunnel_name(self.name)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "source", _normalize_local_key(self.source, context="Tunnel source")
        )
        object.__setattr__(
            self,
            "source_port",
            _normalize_non_negative_int(self.source_port, context="Tunnel source_port"),
        )


@dataclass(frozen=True, slots=True)
class GraphFragmentNote:
    """A note attached to a copied node, positioned relative to the fragment."""

    key: str
    text: str
    position: tuple[float, float]
    width: float
    attached_node: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "key", _normalize_local_key(self.key, context="Note key")
        )
        if not isinstance(self.text, str):
            raise GraphFragmentError("Note text must be text.")
        object.__setattr__(
            self,
            "position",
            _normalize_position(self.position, context=f"Note {self.key!r} position"),
        )
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, Real)
            or not math.isfinite(float(self.width))
            or float(self.width) <= 0.0
            or float(self.width) > MAX_GRAPH_FRAGMENT_NOTE_WIDTH
        ):
            raise GraphFragmentError(
                "Note width must be a positive practical canvas size."
            )
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(
            self,
            "attached_node",
            _normalize_local_key(self.attached_node, context="Note attached_node"),
        )


@dataclass(frozen=True, init=False, slots=True)
class GraphFragment:
    """Detached immutable-by-convention representation of copied graph state."""

    nodes: tuple[GraphFragmentNode, ...]
    connections: tuple[GraphFragmentConnection, ...]
    tunnels: tuple[GraphFragmentTunnel, ...]
    notes: tuple[GraphFragmentNote, ...]

    __hash__ = None

    def __init__(
        self,
        nodes: Iterable[GraphFragmentNode],
        connections: Iterable[GraphFragmentConnection] = (),
        tunnels: Iterable[GraphFragmentTunnel] = (),
        notes: Iterable[GraphFragmentNote] = (),
    ) -> None:
        normalized_nodes = tuple(nodes)
        normalized_connections = tuple(connections)
        normalized_tunnels = tuple(tunnels)
        normalized_notes = tuple(notes)
        if any(not isinstance(node, GraphFragmentNode) for node in normalized_nodes):
            raise TypeError("nodes must contain GraphFragmentNode values.")
        if any(
            not isinstance(connection, GraphFragmentConnection)
            for connection in normalized_connections
        ):
            raise TypeError("connections must contain GraphFragmentConnection values.")
        if any(
            not isinstance(tunnel, GraphFragmentTunnel)
            for tunnel in normalized_tunnels
        ):
            raise TypeError("tunnels must contain GraphFragmentTunnel values.")
        if any(not isinstance(note, GraphFragmentNote) for note in normalized_notes):
            raise TypeError("notes must contain GraphFragmentNote values.")
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "connections", normalized_connections)
        object.__setattr__(self, "tunnels", normalized_tunnels)
        object.__setattr__(self, "notes", normalized_notes)

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached, canonical-shape JSON mapping."""
        return {
            "kind": GRAPH_FRAGMENT_KIND,
            "version": GRAPH_FRAGMENT_VERSION,
            "nodes": [
                {
                    "key": node.key,
                    "operation_id": node.operation_id,
                    "params": node.params,
                    "position": [node.position[0], node.position[1]],
                    "compute_preference": (
                        None
                        if node.compute_preference is None
                        else node.compute_preference.as_dict()
                    ),
                    "optimizer_locked": node.optimizer_locked,
                }
                for node in self.nodes
            ],
            "connections": [
                {
                    "source": connection.source,
                    "target": connection.target,
                    "target_port": connection.target_port,
                    "source_port": connection.source_port,
                    "tunnel": connection.tunnel,
                }
                for connection in self.connections
            ],
            "tunnels": [
                {
                    "name": tunnel.name,
                    "source": tunnel.source,
                    "source_port": tunnel.source_port,
                }
                for tunnel in self.tunnels
            ],
            "notes": [
                {
                    "key": note.key,
                    "text": note.text,
                    "position": [note.position[0], note.position[1]],
                    "width": note.width,
                    "attached_node": note.attached_node,
                }
                for note in self.notes
            ],
        }


def extract_transferable_parameters(
    operation_id: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Filter one live node's parameters down to authored portable state.

    All declared operation parameters and explicitly persisted optional authored
    fields are preserved.  Derived compatibility hints and arbitrary ``_vipp_*``
    state are omitted, except for the manual auto-recalculate user preference.
    """
    operation = _operation_spec(operation_id)
    if not isinstance(params, Mapping):
        raise GraphFragmentError("Node parameters must be an object.")
    required = {parameter.name for parameter in operation.parameters}
    result: dict[str, Any] = {}
    for raw_name, raw_value in params.items():
        if not isinstance(raw_name, str):
            raise GraphFragmentError("Node parameter names must be text.")
        name = raw_name
        if name in required:
            result[name] = _normalize_json_value(
                raw_value,
                context=f"Parameter {name!r}",
            )
            continue
        optional = optional_persisted_parameter_spec(operation, name)
        if optional is not None:
            if name not in NONTRANSFERABLE_OPTIONAL_PARAMETER_NAMES:
                result[name] = _normalize_json_value(
                    raw_value, context=f"Parameter {name!r}"
                )
            continue
        if name in NONTRANSFERABLE_OPTIONAL_PARAMETER_NAMES:
            continue
        if name == MANUAL_AUTO_RECALCULATE_PARAM:
            if operation.execution_policy == "manual":
                result[name] = _normalize_json_value(
                    raw_value, context=f"Parameter {name!r}"
                )
            continue
        if name.startswith("_vipp_"):
            continue
        raise GraphFragmentError(
            f"Operation {operation.id!r} has no transferable parameter {name!r}."
        )
    return validate_transferable_parameters(operation.id, result)


def validate_transferable_parameters(
    operation_id: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact transferable parameter set against today's registry."""
    operation = _operation_spec(operation_id)
    normalized = _normalize_json_object(params, context="Node parameters")
    required = {parameter.name for parameter in operation.parameters}
    for name, value in normalized.items():
        if name in required:
            continue
        optional = optional_persisted_parameter_spec(operation, name)
        if (
            optional is not None
            and name not in NONTRANSFERABLE_OPTIONAL_PARAMETER_NAMES
        ):
            continue
        if (
            name == MANUAL_AUTO_RECALCULATE_PARAM
            and operation.execution_policy == "manual"
        ):
            if not isinstance(value, bool):
                raise GraphFragmentError(
                    f"Parameter {MANUAL_AUTO_RECALCULATE_PARAM!r} must be a boolean."
                )
            continue
        raise GraphFragmentError(
            f"Operation {operation.id!r} has no transferable parameter {name!r}."
        )
    try:
        graph_node_from_persisted_params(
            "fragment-node",
            operation.id,
            normalized,
            index=0,
        )
    except (TypeError, ValueError) as exc:
        raise GraphFragmentError(str(exc)) from exc
    return deepcopy(normalized)


def prepare_paste_values(
    source: GraphFragmentNode,
    target_operation_id: str,
    target_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated parameter replacement for an exact-operation paste.

    When ``target_params`` is supplied, non-transferable target state is kept in
    place.  Every authored public/optional parameter is replaced by the source
    set; cache and derived fields are therefore preserved on the target rather
    than copied from the clipboard.
    """
    if not isinstance(source, GraphFragmentNode):
        raise TypeError("source must be a GraphFragmentNode.")
    target_operation = _normalize_text(
        target_operation_id, context="Target operation_id"
    )
    if source.operation_id != target_operation:
        raise GraphFragmentError(
            "Paste Values requires the same operation: "
            f"copied {source.operation_id!r}, target is {target_operation!r}."
        )
    source_values = validate_transferable_parameters(
        source.operation_id, source.params
    )
    if target_params is None:
        return source_values

    operation = _operation_spec(target_operation)
    normalized_target = _normalize_json_object(
        target_params, context="Target node parameters"
    )
    try:
        graph_node_from_persisted_params(
            "paste-target",
            operation.id,
            normalized_target,
            index=0,
        )
    except (TypeError, ValueError) as exc:
        raise GraphFragmentError(str(exc)) from exc

    required = {parameter.name for parameter in operation.parameters}
    # Auto-recalculate is authored execution intent and belongs with a complete
    # copied node, but Paste Values must retain the existing target node's
    # execution choice just like its compute preference.
    target_has_auto_recalculate = (
        MANUAL_AUTO_RECALCULATE_PARAM in normalized_target
    )
    target_auto_recalculate = normalized_target.get(MANUAL_AUTO_RECALCULATE_PARAM)
    merged = {
        name: value
        for name, value in normalized_target.items()
        if not (
            name in required
            or (
                optional_persisted_parameter_spec(operation, name) is not None
                and name not in NONTRANSFERABLE_OPTIONAL_PARAMETER_NAMES
            )
            or (
                name == MANUAL_AUTO_RECALCULATE_PARAM
                and operation.execution_policy == "manual"
            )
        )
    }
    merged.update(source_values)
    if target_has_auto_recalculate:
        merged[MANUAL_AUTO_RECALCULATE_PARAM] = target_auto_recalculate
    elif operation.execution_policy == "manual":
        # Absence is the persisted representation of the default ``False``.
        # Do not let an explicitly enabled source change that target choice.
        merged.pop(MANUAL_AUTO_RECALCULATE_PARAM, None)
    try:
        graph_node_from_persisted_params(
            "paste-target",
            operation.id,
            merged,
            index=0,
        )
    except (TypeError, ValueError) as exc:
        raise GraphFragmentError(str(exc)) from exc
    return deepcopy(merged)


def capture_graph_fragment(
    pipeline: PrototypePipeline,
    selected_node_ids: Iterable[str],
    *,
    positions: Mapping[str, Sequence[Real]] | None = None,
    notes: Iterable[Mapping[str, Any]] = (),
    node_preferences: Mapping[
        str, NodeComputePreference | str | Mapping[str, object]
    ]
    | None = None,
    optimizer_locked_node_ids: Iterable[str] = (),
) -> GraphFragment:
    """Capture selected nodes and wholly internal relationships.

    Node and note positions are expressed relative to the selected nodes'
    bounding-box centre.  Connections crossing the selection boundary are
    excluded.  A tunnel is included only when its source and at least one of its
    subscribers are selected, which keeps every fragment self-contained.
    """
    if not isinstance(pipeline, PrototypePipeline):
        raise TypeError("pipeline must be a PrototypePipeline.")
    requested = tuple(selected_node_ids)
    if not requested:
        raise GraphFragmentError("Copy requires at least one selected node.")
    if any(not isinstance(node_id, str) or not node_id for node_id in requested):
        raise GraphFragmentError("Selected node IDs must be non-empty strings.")
    selected = set(requested)
    unknown = selected - set(pipeline.nodes)
    if unknown:
        names = ", ".join(repr(node_id) for node_id in sorted(unknown))
        raise GraphFragmentError(f"Selection references missing nodes: {names}.")

    ordered_ids = [node_id for node_id in pipeline.nodes if node_id in selected]
    if len(ordered_ids) > MAX_GRAPH_FRAGMENT_NODES:
        raise GraphFragmentError(
            f"A fragment may contain at most {MAX_GRAPH_FRAGMENT_NODES} nodes."
        )
    local_key_by_id = {
        node_id: f"n{index}" for index, node_id in enumerate(ordered_ids)
    }
    raw_positions = positions or {}
    absolute_positions = {
        node_id: _normalize_position(
            raw_positions.get(node_id, (0.0, 0.0)),
            context=f"Node {node_id!r} position",
        )
        for node_id in ordered_ids
    }
    xs = [position[0] for position in absolute_positions.values()]
    ys = [position[1] for position in absolute_positions.values()]
    origin = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)

    preferences = node_preferences or {}
    locked_ids = set(optimizer_locked_node_ids)
    unknown_locks = locked_ids - selected
    if unknown_locks:
        names = ", ".join(repr(node_id) for node_id in sorted(unknown_locks))
        raise GraphFragmentError(
            f"Optimizer locks reference nodes outside the selection: {names}."
        )

    fragment_nodes: list[GraphFragmentNode] = []
    for node_id in ordered_ids:
        node = pipeline.nodes[node_id]
        preference = _normalize_compute_preference(preferences.get(node_id))
        if preference is not None and preference.kind is NodePreferenceKind.AUTO:
            preference = None
        relative = _relative_position(absolute_positions[node_id], origin)
        fragment_nodes.append(
            GraphFragmentNode(
                local_key_by_id[node_id],
                node.operation_id,
                extract_transferable_parameters(node.operation_id, node.params),
                relative,
                preference,
                node_id in locked_ids,
            )
        )

    internal_connections = [
        connection
        for connection in pipeline.connections
        if connection.source_id in selected and connection.target_id in selected
    ]
    if len(internal_connections) > MAX_GRAPH_FRAGMENT_CONNECTIONS:
        raise GraphFragmentError(
            "A fragment may contain at most "
            f"{MAX_GRAPH_FRAGMENT_CONNECTIONS} connections."
        )
    fragment_connections = tuple(
        GraphFragmentConnection(
            local_key_by_id[connection.source_id],
            local_key_by_id[connection.target_id],
            connection.target_port,
            connection.source_port,
            connection.tunnel_name,
        )
        for connection in internal_connections
    )

    referenced_tunnel_keys = {
        connection.tunnel_name.casefold()
        for connection in internal_connections
        if connection.tunnel_name
    }
    fragment_tunnels: list[GraphFragmentTunnel] = []
    for tunnel in pipeline.output_tunnel_list():
        if tunnel.name.casefold() not in referenced_tunnel_keys:
            continue
        if tunnel.source_id not in selected:
            raise GraphFragmentError(
                f"Internal tunnel {tunnel.name!r} has an external source."
            )
        fragment_tunnels.append(
            GraphFragmentTunnel(
                tunnel.name,
                local_key_by_id[tunnel.source_id],
                tunnel.source_port,
            )
        )
    if len(fragment_tunnels) != len(referenced_tunnel_keys):
        raise GraphFragmentError("A copied connection references an unknown tunnel.")

    fragment_notes: list[GraphFragmentNote] = []
    for raw_note in notes:
        if not isinstance(raw_note, Mapping):
            raise GraphFragmentError("Workflow notes must be objects.")
        attached = raw_note.get("attached_node", "")
        if attached not in selected:
            continue
        note_position = _normalize_position(
            raw_note.get("position", (0.0, 0.0)),
            context="Note position",
        )
        fragment_notes.append(
            GraphFragmentNote(
                f"note{len(fragment_notes)}",
                raw_note.get("text", ""),
                _relative_position(note_position, origin),
                raw_note.get("width", 240.0),
                local_key_by_id[attached],
            )
        )
    if len(fragment_notes) > MAX_GRAPH_FRAGMENT_NOTES:
        raise GraphFragmentError(
            f"A fragment may contain at most {MAX_GRAPH_FRAGMENT_NOTES} notes."
        )

    fragment = GraphFragment(
        fragment_nodes,
        fragment_connections,
        fragment_tunnels,
        fragment_notes,
    )
    return validate_graph_fragment(fragment)


def validate_graph_fragment(fragment: GraphFragment) -> GraphFragment:
    """Validate registry compatibility and all internal graph invariants."""
    if not isinstance(fragment, GraphFragment):
        raise TypeError("fragment must be a GraphFragment.")
    if not fragment.nodes:
        raise GraphFragmentError("A graph fragment must contain at least one node.")
    if len(fragment.nodes) > MAX_GRAPH_FRAGMENT_NODES:
        raise GraphFragmentError(
            f"A fragment may contain at most {MAX_GRAPH_FRAGMENT_NODES} nodes."
        )
    if len(fragment.connections) > MAX_GRAPH_FRAGMENT_CONNECTIONS:
        raise GraphFragmentError(
            "A fragment may contain at most "
            f"{MAX_GRAPH_FRAGMENT_CONNECTIONS} connections."
        )
    if len(fragment.notes) > MAX_GRAPH_FRAGMENT_NOTES:
        raise GraphFragmentError(
            f"A fragment may contain at most {MAX_GRAPH_FRAGMENT_NOTES} notes."
        )

    node_keys = [node.key for node in fragment.nodes]
    if len(set(node_keys)) != len(node_keys):
        raise GraphFragmentError("A graph fragment contains duplicate node keys.")
    node_key_set = set(node_keys)
    graph_nodes: list[GraphNode] = []
    for index, node in enumerate(fragment.nodes):
        params = validate_transferable_parameters(node.operation_id, node.params)
        if node.optimizer_locked and (
            node.compute_preference is None
            or node.compute_preference.kind is NodePreferenceKind.AUTO
        ):
            raise GraphFragmentError(
                f"Node {node.key!r} cannot be optimizer-locked without an "
                "explicit compute preference."
            )
        try:
            graph_nodes.append(
                graph_node_from_persisted_params(
                    node.key,
                    node.operation_id,
                    params,
                    index=index,
                )
            )
        except (TypeError, ValueError) as exc:
            raise GraphFragmentError(str(exc)) from exc

    graph_connections: list[GraphConnection] = []
    for index, connection in enumerate(fragment.connections):
        missing = {
            key
            for key in (connection.source, connection.target)
            if key not in node_key_set
        }
        if missing:
            names = ", ".join(repr(key) for key in sorted(missing))
            raise GraphFragmentError(
                f"Connection {index} references missing node key(s): {names}."
            )
        graph_connections.append(
            GraphConnection(
                connection.source,
                connection.target,
                connection.target_port,
                connection.source_port,
                connection.tunnel,
            )
        )

    tunnel_names: dict[str, GraphFragmentTunnel] = {}
    tunnel_slots: set[tuple[str, int]] = set()
    for tunnel in fragment.tunnels:
        key = tunnel.name.casefold()
        if key in tunnel_names:
            raise GraphFragmentError(
                f"A graph fragment contains duplicate tunnel name {tunnel.name!r}."
            )
        if tunnel.source not in node_key_set:
            raise GraphFragmentError(
                f"Tunnel {tunnel.name!r} references missing node {tunnel.source!r}."
            )
        slot = (tunnel.source, tunnel.source_port)
        if slot in tunnel_slots:
            raise GraphFragmentError(
                f"Multiple tunnels are assigned to {tunnel.source!r} output "
                f"{tunnel.source_port}."
            )
        tunnel_names[key] = tunnel
        tunnel_slots.add(slot)

    referenced_tunnels: set[str] = set()
    for index, connection in enumerate(fragment.connections):
        if not connection.tunnel:
            continue
        key = connection.tunnel.casefold()
        tunnel = tunnel_names.get(key)
        if tunnel is None:
            raise GraphFragmentError(
                f"Connection {index} references unknown tunnel "
                f"{connection.tunnel!r}."
            )
        if connection.tunnel != tunnel.name:
            raise GraphFragmentError(
                f"Connection {index} must use canonical tunnel name {tunnel.name!r}."
            )
        if (connection.source, connection.source_port) != (
            tunnel.source,
            tunnel.source_port,
        ):
            raise GraphFragmentError(
                f"Connection {index} tunnel {tunnel.name!r} does not match its "
                "declared source output."
            )
        referenced_tunnels.add(key)
    unreferenced = set(tunnel_names) - referenced_tunnels
    if unreferenced:
        names = ", ".join(
            repr(tunnel_names[key].name) for key in sorted(unreferenced)
        )
        raise GraphFragmentError(
            f"Internal tunnel(s) have no copied subscribers: {names}."
        )

    note_keys = [note.key for note in fragment.notes]
    if len(set(note_keys)) != len(note_keys):
        raise GraphFragmentError("A graph fragment contains duplicate note keys.")
    for note in fragment.notes:
        if note.attached_node not in node_key_set:
            raise GraphFragmentError(
                f"Note {note.key!r} references missing attached node "
                f"{note.attached_node!r}."
            )

    detached = PrototypePipeline()
    try:
        detached.restore_graph(
            graph_nodes,
            graph_connections,
            (
                OutputTunnel(tunnel.name, tunnel.source, tunnel.source_port)
                for tunnel in fragment.tunnels
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphFragmentError(f"Invalid graph fragment: {exc}") from exc
    return fragment


def graph_fragment_from_mapping(raw: Mapping[str, Any]) -> GraphFragment:
    """Decode a strict already-parsed JSON mapping."""
    top = _strict_object(
        raw,
        required={"kind", "version", "nodes", "connections", "tunnels", "notes"},
        context="Graph fragment",
    )
    if top["kind"] != GRAPH_FRAGMENT_KIND:
        raise GraphFragmentError(
            f"Unsupported graph fragment kind {top['kind']!r}."
        )
    if (
        isinstance(top["version"], bool)
        or not isinstance(top["version"], Integral)
        or int(top["version"]) != GRAPH_FRAGMENT_VERSION
    ):
        raise GraphFragmentError(
            f"Unsupported graph fragment version {top['version']!r}; "
            f"expected {GRAPH_FRAGMENT_VERSION}."
        )

    raw_nodes = _required_list(top["nodes"], context="Graph fragment nodes")
    raw_connections = _required_list(
        top["connections"], context="Graph fragment connections"
    )
    raw_tunnels = _required_list(top["tunnels"], context="Graph fragment tunnels")
    raw_notes = _required_list(top["notes"], context="Graph fragment notes")

    nodes: list[GraphFragmentNode] = []
    for index, value in enumerate(raw_nodes):
        item = _strict_object(
            value,
            required={
                "key",
                "operation_id",
                "params",
                "position",
                "compute_preference",
                "optimizer_locked",
            },
            context=f"Node {index}",
        )
        nodes.append(
            GraphFragmentNode(
                item["key"],
                item["operation_id"],
                item["params"],
                item["position"],
                item["compute_preference"],
                item["optimizer_locked"],
            )
        )

    connections: list[GraphFragmentConnection] = []
    for index, value in enumerate(raw_connections):
        item = _strict_object(
            value,
            required={"source", "target", "target_port", "source_port", "tunnel"},
            context=f"Connection {index}",
        )
        connections.append(
            GraphFragmentConnection(
                item["source"],
                item["target"],
                item["target_port"],
                item["source_port"],
                item["tunnel"],
            )
        )

    tunnels: list[GraphFragmentTunnel] = []
    for index, value in enumerate(raw_tunnels):
        item = _strict_object(
            value,
            required={"name", "source", "source_port"},
            context=f"Tunnel {index}",
        )
        tunnels.append(
            GraphFragmentTunnel(
                item["name"],
                item["source"],
                item["source_port"],
            )
        )

    parsed_notes: list[GraphFragmentNote] = []
    for index, value in enumerate(raw_notes):
        item = _strict_object(
            value,
            required={"key", "text", "position", "width", "attached_node"},
            context=f"Note {index}",
        )
        parsed_notes.append(
            GraphFragmentNote(
                item["key"],
                item["text"],
                item["position"],
                item["width"],
                item["attached_node"],
            )
        )

    return validate_graph_fragment(
        GraphFragment(nodes, connections, tunnels, parsed_notes)
    )


def encode_graph_fragment(
    fragment: GraphFragment,
    *,
    max_bytes: int = MAX_GRAPH_FRAGMENT_BYTES,
) -> bytes:
    """Encode a validated fragment as deterministic UTF-8 JSON bytes."""
    limit = _normalize_size_limit(max_bytes)
    validate_graph_fragment(fragment)
    try:
        encoded = json.dumps(
            fragment.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GraphFragmentError(f"Graph fragment is not finite JSON: {exc}") from exc
    if len(encoded) > limit:
        raise GraphFragmentError(
            f"Graph fragment is {len(encoded):,} bytes; the limit is {limit:,}."
        )
    return encoded


def decode_graph_fragment(
    payload: str | bytes | bytearray | memoryview,
    *,
    max_bytes: int = MAX_GRAPH_FRAGMENT_BYTES,
) -> GraphFragment:
    """Decode untrusted clipboard JSON with size and duplicate-key checks."""
    limit = _normalize_size_limit(max_bytes)
    if isinstance(payload, str):
        try:
            encoded = payload.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise GraphFragmentError("Graph fragment text is not valid UTF-8.") from exc
        text = payload
    elif isinstance(payload, (bytes, bytearray, memoryview)):
        encoded = bytes(payload)
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GraphFragmentError(
                "Graph fragment bytes are not valid UTF-8."
            ) from exc
    else:
        raise TypeError("payload must be JSON text or bytes.")
    if len(encoded) > limit:
        raise GraphFragmentError(
            f"Graph fragment is {len(encoded):,} bytes; the limit is {limit:,}."
        )
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except GraphFragmentError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError, ValueError) as exc:
        raise GraphFragmentError(f"Invalid graph fragment JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise GraphFragmentError("Graph fragment JSON must contain one object.")
    return graph_fragment_from_mapping(raw)


def _operation_spec(operation_id: str):
    operation = _normalize_text(operation_id, context="Operation ID")
    spec = NODE_LIBRARY_BY_ID.get(operation)
    if spec is None:
        raise GraphFragmentError(f"Unknown operation {operation!r}.")
    return spec


def _normalize_compute_preference(
    value: NodeComputePreference | str | Mapping[str, object] | None,
) -> NodeComputePreference | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        unknown = set(value) - {"kind", "value"}
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise GraphFragmentError(
                f"Unknown node compute preference field(s): {names}."
            )
        if "kind" not in value or not isinstance(value["kind"], str):
            raise GraphFragmentError(
                "Node compute preference requires a text 'kind'."
            )
        if "value" in value and not isinstance(value["value"], str):
            raise GraphFragmentError("Node compute preference 'value' must be text.")
    try:
        return NodeComputePreference.parse(value)
    except (TypeError, ValueError) as exc:
        raise GraphFragmentError(str(exc)) from exc


def _normalize_json_object(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    normalized = _normalize_json_value(value, context=context)
    if not isinstance(normalized, dict):
        raise GraphFragmentError(f"{context} must be an object.")
    return normalized


def _normalize_json_value(
    value: Any,
    *,
    context: str,
    depth: int = 0,
) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise GraphFragmentError(f"{context} is nested too deeply.")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise GraphFragmentError(f"{context} must not contain NaN or infinity.")
        return number
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise GraphFragmentError(f"{context} object keys must be text.")
            result[key] = _normalize_json_value(
                nested,
                context=context,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, context=context, depth=depth + 1)
            for item in value
        ]
    raise GraphFragmentError(
        f"{context} contains non-JSON value of type {type(value).__name__}."
    )


def _normalize_position(
    value: Sequence[Real],
    *,
    context: str,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GraphFragmentError(f"{context} must contain x and y.")
    if len(value) != 2:
        raise GraphFragmentError(f"{context} must contain x and y.")
    coordinates: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise GraphFragmentError(f"{context} must contain finite numbers.")
        coordinate = float(raw)
        if (
            not math.isfinite(coordinate)
            or abs(coordinate) > MAX_GRAPH_FRAGMENT_COORDINATE_ABS
        ):
            raise GraphFragmentError(
                f"{context} must contain practical finite canvas coordinates."
            )
        coordinates.append(coordinate)
    return coordinates[0], coordinates[1]


def _relative_position(
    position: tuple[float, float],
    origin: tuple[float, float],
) -> tuple[float, float]:
    relative = (position[0] - origin[0], position[1] - origin[1])
    if not all(math.isfinite(value) for value in relative):
        raise GraphFragmentError("Relative positions must remain finite.")
    return relative


def _normalize_text(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphFragmentError(f"{context} must be non-empty text.")
    return value.strip()


def _normalize_local_key(value: Any, *, context: str) -> str:
    text = _normalize_text(value, context=context)
    if _LOCAL_KEY.fullmatch(text) is None:
        raise GraphFragmentError(
            f"{context} must start with a letter and contain only letters, "
            "numbers, '_' or '-'."
        )
    return text


def _normalize_tunnel_name(value: Any) -> str:
    if not isinstance(value, str):
        raise GraphFragmentError("Tunnel name must be text.")
    normalized = re.sub(r"\s+", " ", value.strip())
    if not normalized:
        raise GraphFragmentError("Tunnel name must not be blank.")
    if normalized != value:
        raise GraphFragmentError("Tunnel names must use canonical whitespace.")
    return normalized


def _normalize_non_negative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise GraphFragmentError(f"{context} must be a non-negative integer.")
    return int(value)


def _normalize_size_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError("max_bytes must be a positive integer.")
    return int(value)


def _strict_object(
    value: Any,
    *,
    required: set[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphFragmentError(f"{context} must be an object.")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys):
        raise GraphFragmentError(f"{context} field names must be text.")
    missing = required - keys
    if missing:
        names = ", ".join(sorted(missing))
        raise GraphFragmentError(f"{context} is missing required field(s): {names}.")
    unknown = keys - required
    if unknown:
        names = ", ".join(sorted(unknown))
        raise GraphFragmentError(f"{context} has unknown field(s): {names}.")
    return value


def _required_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GraphFragmentError(f"{context} must be a list.")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraphFragmentError(f"Duplicate JSON field {key!r} is not allowed.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise GraphFragmentError(f"Non-finite JSON constant {value!r} is not allowed.")


__all__ = [
    "GRAPH_FRAGMENT_KIND",
    "GRAPH_FRAGMENT_MIME_TYPE",
    "GRAPH_FRAGMENT_VERSION",
    "MAX_GRAPH_FRAGMENT_BYTES",
    "NONTRANSFERABLE_OPTIONAL_PARAMETER_NAMES",
    "TRANSFERABLE_PRIVATE_PARAMETER_NAMES",
    "GraphFragment",
    "GraphFragmentConnection",
    "GraphFragmentError",
    "GraphFragmentNode",
    "GraphFragmentNote",
    "GraphFragmentTunnel",
    "capture_graph_fragment",
    "decode_graph_fragment",
    "encode_graph_fragment",
    "extract_transferable_parameters",
    "graph_fragment_from_mapping",
    "prepare_paste_values",
    "validate_graph_fragment",
    "validate_transferable_parameters",
]
