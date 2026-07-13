"""Immutable, defensively copied runtime snapshots of VIPP workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from napari_vipp.core.pipeline import (
    GraphConnection,
    GraphNode,
    OutputTunnel,
    PrototypePipeline,
    graph_node_from_persisted_params,
)

Position = tuple[float, float]
OrderedPositions = tuple[tuple[str, Position], ...]


@dataclass(frozen=True, init=False, slots=True)
class NodeSnapshot:
    """Persistable identity and parameters for one graph node."""

    id: str
    operation_id: str
    _params: dict[str, Any] = field(repr=False)

    __hash__ = None

    def __init__(
        self,
        id: str,
        operation_id: str,
        params: Mapping[str, Any],
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "_params", deepcopy(dict(params)))

    @property
    def params(self) -> dict[str, Any]:
        """Return a detached materialization of the saved parameters."""
        return deepcopy(self._params)

    def _to_graph_node(self, *, index: int) -> GraphNode:
        return graph_node_from_persisted_params(
            self.id,
            self.operation_id,
            self.params,
            index=index,
        )


@dataclass(frozen=True, init=False, slots=True)
class GraphSnapshot:
    """Ordered, immutable runtime representation of a pipeline graph."""

    nodes: tuple[NodeSnapshot, ...]
    connections: tuple[GraphConnection, ...]
    output_tunnels: tuple[OutputTunnel, ...]

    __hash__ = None

    def __init__(
        self,
        nodes: Iterable[NodeSnapshot],
        connections: Iterable[GraphConnection] = (),
        output_tunnels: Iterable[OutputTunnel] = (),
    ) -> None:
        node_copies = tuple(
            NodeSnapshot(node.id, node.operation_id, node.params) for node in nodes
        )
        connection_copies = tuple(
            GraphConnection(
                connection.source_id,
                connection.target_id,
                connection.target_port,
                connection.source_port,
                connection.tunnel_name,
            )
            for connection in connections
        )
        tunnel_copies = tuple(
            OutputTunnel(tunnel.name, tunnel.source_id, tunnel.source_port)
            for tunnel in output_tunnels
        )
        object.__setattr__(self, "nodes", node_copies)
        object.__setattr__(self, "connections", connection_copies)
        object.__setattr__(self, "output_tunnels", tunnel_copies)

    @classmethod
    def from_pipeline(cls, pipeline: PrototypePipeline) -> GraphSnapshot:
        """Capture the persistable graph state of ``pipeline``."""
        return cls(
            (
                NodeSnapshot(node.id, node.operation_id, node.params)
                for node in pipeline.nodes.values()
            ),
            pipeline.connections,
            pipeline.output_tunnel_list(),
        )

    def restore_into(self, pipeline: PrototypePipeline) -> None:
        """Validate and atomically replace ``pipeline`` with this graph."""
        nodes = [
            node._to_graph_node(index=index)
            for index, node in enumerate(self.nodes)
        ]
        pipeline.restore_graph(nodes, self.connections, self.output_tunnels)

    def to_pipeline(self) -> PrototypePipeline:
        """Return a validated, detached pipeline materialization."""
        pipeline = PrototypePipeline()
        self.restore_into(pipeline)
        return pipeline


@dataclass(frozen=True, init=False, slots=True)
class WorkflowNoteSnapshot:
    """Immutable persisted note state."""

    id: str
    text: str
    position: Position
    width: float
    attached_node: str

    __hash__ = None

    def __init__(
        self,
        id: str,
        text: str,
        position: Position,
        width: float = 240.0,
        attached_node: str = "",
    ) -> None:
        x, y = position
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "position", (float(x), float(y)))
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "attached_node", attached_node)

    @classmethod
    def from_mapping(cls, note: Mapping[str, Any]) -> WorkflowNoteSnapshot:
        """Capture one note returned by workflow deserialization."""
        return cls(
            id=note["id"],
            text=note["text"],
            position=note["position"],
            width=note["width"],
            attached_node=note.get("attached_node", ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a detached note representation accepted by the serializer."""
        note: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "position": self.position,
            "width": self.width,
        }
        if self.attached_node:
            note["attached_node"] = self.attached_node
        return note


@dataclass(frozen=True, init=False, slots=True)
class WorkflowSnapshot:
    """Graph plus ordered canvas and UI persistence state."""

    graph: GraphSnapshot
    positions: OrderedPositions
    notes: tuple[WorkflowNoteSnapshot, ...]
    _metadata: dict[str, Any] = field(repr=False)

    __hash__ = None

    def __init__(
        self,
        graph: GraphSnapshot,
        positions: Mapping[str, Position]
        | Iterable[tuple[str, Position]] = (),
        notes: Iterable[WorkflowNoteSnapshot] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        position_items = (
            positions.items() if isinstance(positions, Mapping) else positions
        )
        ordered_positions = tuple(
            (node_id, (float(position[0]), float(position[1])))
            for node_id, position in position_items
        )
        graph_copy = GraphSnapshot(
            graph.nodes,
            graph.connections,
            graph.output_tunnels,
        )
        note_copies = tuple(
            WorkflowNoteSnapshot(
                note.id,
                note.text,
                note.position,
                note.width,
                note.attached_node,
            )
            for note in notes
        )
        object.__setattr__(self, "graph", graph_copy)
        object.__setattr__(self, "positions", ordered_positions)
        object.__setattr__(self, "notes", note_copies)
        object.__setattr__(
            self,
            "_metadata",
            deepcopy(dict(metadata or {})),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return detached workflow metadata."""
        return deepcopy(self._metadata)

    def positions_dict(self) -> dict[str, Position]:
        """Materialize positions while retaining their snapshot order."""
        return dict(self.positions)


__all__ = [
    "GraphSnapshot",
    "NodeSnapshot",
    "WorkflowNoteSnapshot",
    "WorkflowSnapshot",
]
