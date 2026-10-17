"""Privacy-minimized evidence of one qualified failed assistant calculation.

This module grants no execution, rollback, session, sharing or retry authority.
The caller must establish those permissions before building or sending feedback.
Failed calculations never acquire successful output snapshots or assessments.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from napari_vipp.core.execution import PipelineExecutionFailure
from napari_vipp.core.mesh_refinement import safe_mesh_input_failure
from napari_vipp.core.pipeline import SPATIAL_OPERATIONS, PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.workflow_proposal_catalogue import (
    authorable_parameter_specs,
    optional_authorable_parameter_specs,
)
from napari_vipp.core.workflow_refinement import (
    REFINEMENT_SAFE_OPERATION_IDS,
    assert_refinement_graph_safe,
)

MAX_FAILURE_FEEDBACK_BYTES = 48 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_NODE_KEY = re.compile(r"n(?:[1-9]|[1-5][0-9]|6[0-4])\Z")
_STATUS = "failed_no_assessment"
_RULE = (
    "This calculation failed; no completed outputs or current assessment exist. "
    "The exact failed assistant addition was removed before this request. "
    "Propose a fresh candidate from the original goal and this failed recipe. "
    "Changing refinement settings cannot repair an invalid input mesh. "
    "Regenerate or change the upstream candidate explicitly; do not bypass "
    "input validation or claim a repair was run."
)


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _parameter_specs(operation_id):
    return {
        item.name: item
        for item in (
            *authorable_parameter_specs(operation_id),
            *optional_authorable_parameter_specs(operation_id),
        )
    }


def _safe_parameter(spec, value):
    if spec.kind == "bool":
        return type(value) is bool
    if spec.kind == "int":
        return type(value) is int
    if spec.kind == "float":
        try:
            return type(value) in {int, float} and math.isfinite(value)
        except OverflowError:
            return False
    # Dynamic choices may contain private axis or column names. Even valid
    # dynamic/free text is deliberately not sent by this failure boundary.
    return (
        spec.kind == "choice"
        and not spec.dynamic_choices
        and type(value) is str
        and value in spec.choices
    )


def _public_parameters(node):
    specs = _parameter_specs(node.operation_id)
    safe = {}
    redacted = []
    for name, value in node.params.items():
        spec = specs.get(name)
        if spec is None:
            # Private/derived/manual state is not a catalogue parameter; neither
            # its value nor an arbitrary untrusted key is public feedback.
            continue
        if _safe_parameter(spec, value):
            safe[name] = value
        else:
            redacted.append(name)
    return safe, sorted(redacted)


def _normalized(graph):
    nodes = []
    for node in graph.nodes:
        params = node.params
        if node.operation_id in SPATIAL_OPERATIONS:
            resolved = params.get("resolved_spatial_ndim")
            if resolved is None or (type(resolved) is int and resolved in {1, 2, 3}):
                params.pop("resolved_spatial_ndim", None)
        nodes.append(
            NodeSnapshot(node.id, node.operation_id, params, node.execution_mode)
        )
    return GraphSnapshot(nodes, graph.connections, graph.output_tunnels)


def _require_context(context):
    # Kept local: ProposalContext lazily validates this feedback's guard type.
    from napari_vipp.core.workflow_proposals import ProposalContext, _graph_fingerprint

    if type(context) is not ProposalContext:
        raise TypeError("Failure feedback needs an exact fresh ProposalContext.")
    if (
        _graph_fingerprint(context.baseline) != context.baseline_fingerprint
        or not isinstance(context.source_id, str)
        or not context.source_id
        or type(context.source_fingerprint) is not str
        or _FINGERPRINT.fullmatch(context.source_fingerprint) is None
        or context._result_feedback is not None
        or context._result_feedback_guard is not None
        or getattr(context, "_failure_feedback", None) is not None
        or getattr(context, "_failure_feedback_guard", None) is not None
    ):
        raise ValueError("Failure feedback needs an unmodified fresh rollback context.")


def _checked_payload(text, run_id):
    if type(text) is not str or len(text.encode("utf-8")) > MAX_FAILURE_FEEDBACK_BYTES:
        raise ValueError("Failure feedback exceeds its 48 KiB bound.")
    try:
        result = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Failure feedback must be canonical finite JSON.") from exc
    if _json(result) != text:
        raise ValueError("Failure feedback must be canonical finite JSON.")
    if type(result) is not dict or set(result) != {
        "kind",
        "run_id",
        "status",
        "failed_nodes",
        "failed_candidate",
        "rule",
    }:
        raise ValueError("Failure feedback contains unsupported fields.")
    if (
        result["kind"] != "calculation_failure"
        or result["run_id"] != run_id
        or result["status"] != _STATUS
        or result["rule"] != _RULE
    ):
        raise ValueError("Failure feedback has an invalid identity or status.")
    candidate = result["failed_candidate"]
    if type(candidate) is not dict or set(candidate) != {"nodes", "connections"}:
        raise ValueError("The failed candidate has unsupported fields.")
    nodes, connections = candidate["nodes"], candidate["connections"]
    if type(nodes) is not list or not 1 <= len(nodes) <= 64:
        raise ValueError("Failure feedback needs one to 64 candidate nodes.")
    aliases = {}
    for node in nodes:
        if type(node) is not dict or set(node) != {
            "key",
            "operation_id",
            "parameters",
            "redacted_parameters",
        }:
            raise ValueError("A failed candidate node has unsupported fields.")
        key, operation = node["key"], node["operation_id"]
        if (
            type(key) is not str
            or _NODE_KEY.fullmatch(key) is None
            or key in aliases
            or type(operation) is not str
            or operation not in REFINEMENT_SAFE_OPERATION_IDS
        ):
            raise ValueError("A failed candidate node has an invalid alias.")
        try:
            specs = _parameter_specs(operation)
        except KeyError as exc:
            raise ValueError("Unknown failed candidate operation.") from exc
        parameters, redacted = node["parameters"], node["redacted_parameters"]
        if (
            type(parameters) is not dict
            or type(redacted) is not list
            or redacted != sorted(set(redacted))
            or any(name not in specs for name in redacted)
            or set(parameters) & set(redacted)
            or any(
                name not in specs or not _safe_parameter(specs[name], value)
                for name, value in parameters.items()
            )
        ):
            raise ValueError("Failure feedback contains private or invalid parameters.")
        aliases[key] = operation
    if set(aliases) != {f"n{i}" for i in range(1, len(nodes) + 1)}:
        raise ValueError("Failed candidate aliases must be contiguous.")
    if type(connections) is not list or len(connections) > 128:
        raise ValueError("Failure feedback has too many connections.")
    for edge in connections:
        if (
            type(edge) is not dict
            or set(edge) != {"source", "target", "source_port", "target_port"}
            or edge["source"] not in {"input", *aliases}
            or edge["target"] not in aliases
            or any(
                type(edge[port]) is not int or not 0 <= edge[port] < 256
                for port in ("source_port", "target_port")
            )
        ):
            raise ValueError("Failure feedback contains an invalid connection.")
    failed = result["failed_nodes"]
    if type(failed) is not list or not 1 <= len(failed) <= len(nodes):
        raise ValueError("Failure feedback needs known failed candidate nodes.")
    seen = set()
    for item in failed:
        if type(item) is not dict or set(item) != {
            "node_key",
            "operation_id",
            "code",
            "safe_message",
            "object_id",
            "count",
        }:
            raise ValueError("Failure feedback contains an unsupported diagnosis.")
        key = item["node_key"]
        if key in seen or key not in aliases or aliases[key] != item["operation_id"]:
            raise ValueError("Failure feedback refers to another candidate.")
        safe = safe_mesh_input_failure(
            item["operation_id"], "MeshInputTopologyError", item["safe_message"]
        )
        if safe is None or any(
            type(item[name]) is not type(value) or item[name] != value
            for name, value in safe.items()
        ):
            raise ValueError("Failure feedback contains an unrecognized diagnosis.")
        seen.add(key)
    return result


@dataclass(frozen=True, slots=True)
class RunFailureFeedback:
    """Immutable public text bound to the fresh post-rollback private context."""

    run_id: str
    source_id: str
    baseline_fingerprint: str
    source_fingerprint: str
    _context: object = field(repr=False)
    _payload_json: str = field(repr=False)

    def __post_init__(self):
        _require_context(self._context)
        if (
            type(self.run_id) is not str
            or _TOKEN.fullmatch(self.run_id) is None
            or self.source_id != self._context.source_id
            or self.baseline_fingerprint != self._context.baseline_fingerprint
            or self.source_fingerprint != self._context.source_fingerprint
        ):
            raise ValueError("Failure feedback does not match its rollback context.")
        _checked_payload(self._payload_json, self.run_id)

    def public_feedback(self) -> dict[str, Any]:
        """Return only a fresh detached, privacy-validated JSON materialization."""
        return _checked_payload(self._payload_json, self.run_id)

    def assert_current(self, pipeline: PrototypePipeline) -> None:
        self._context.assert_current(pipeline)


def build_run_failure_feedback(
    context,
    failed_graph: GraphSnapshot,
    candidate_ids: tuple[str, ...],
    failed_node_ids: tuple[str, ...],
    run_id: str,
    failure: PipelineExecutionFailure,
) -> RunFailureFeedback:
    """Describe one qualified ordinary failure without copying partial outputs.

    ``context`` must be freshly captured after the exact failed Add was removed.
    Session/run ownership, cancellation, pending edits and sharing are checked
    by the composition root; these graph checks cannot establish those facts.
    """
    _require_context(context)
    if type(failed_graph) is not GraphSnapshot:
        raise TypeError("The failed workflow must be an exact graph snapshot.")
    if (
        type(candidate_ids) is not tuple
        or not 1 <= len(candidate_ids) <= 64
        or any(type(key) is not str or not key for key in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
        or type(failed_node_ids) is not tuple
        or not failed_node_ids
        or any(type(key) is not str or not key for key in failed_node_ids)
        or len(set(failed_node_ids)) != len(failed_node_ids)
        or not set(failed_node_ids) <= set(candidate_ids)
    ):
        raise ValueError("The failure must identify only this exact candidate.")
    if (
        type(failure) is not PipelineExecutionFailure
        or failure.kind != "execution_error"
        or failure.cleanup_succeeded is not True
        or failure.retryable
        or failure.runtime_id
        or failure.segment_id
        or failure.fallback_records
        or (failure.node_ids and set(failure.node_ids) != set(failed_node_ids))
    ):
        raise ValueError("This failure is not eligible for bounded calculation repair.")
    assert_refinement_graph_safe(failed_graph, context.source_id)
    assert_refinement_graph_safe(context.baseline, context.source_id)
    node_map = {node.id: node for node in failed_graph.nodes}
    baseline_ids = {node.id for node in context.baseline.nodes}
    selected = set(candidate_ids)
    if (
        len(node_map) != len(failed_graph.nodes)
        or not selected <= node_map.keys()
        or selected & baseline_ids
        or set(node_map) != baseline_ids | selected
        or any(tunnel.source_id in selected for tunnel in failed_graph.output_tunnels)
    ):
        raise ValueError("The failed addition does not match the rollback context.")
    remaining_edges = []
    candidate_edges = []
    for edge in failed_graph.connections:
        if edge.target_id in selected:
            if edge.source_id not in selected | {context.source_id} or edge.tunnel_name:
                raise ValueError("The failed candidate depends on an unrelated node.")
            candidate_edges.append(edge)
        elif edge.source_id in selected:
            raise ValueError("The failed candidate changed a pre-existing connection.")
        else:
            remaining_edges.append(edge)
    remaining = GraphSnapshot(
        (node for node in failed_graph.nodes if node.id not in selected),
        remaining_edges,
        failed_graph.output_tunnels,
    )
    if _normalized(remaining) != _normalized(context.baseline):
        raise ValueError("The source or baseline changed around the failed addition.")
    aliases = {key: f"n{index}" for index, key in enumerate(candidate_ids, 1)}
    aliases[context.source_id] = "input"
    failed_nodes = []
    for node_id in failed_node_ids:
        operation = node_map[node_id].operation_id
        safe = safe_mesh_input_failure(operation, failure.error_type, failure.message)
        if safe is None:
            raise ValueError("Only a known safe mesh-input failure may be shared.")
        failed_nodes.append(
            {"node_key": aliases[node_id], "operation_id": operation, **safe}
        )
    nodes = []
    for node_id in candidate_ids:
        node = node_map[node_id]
        parameters, redacted = _public_parameters(node)
        nodes.append(
            {
                "key": aliases[node_id],
                "operation_id": node.operation_id,
                "parameters": parameters,
                "redacted_parameters": redacted,
            }
        )
    payload = {
        "kind": "calculation_failure",
        "run_id": run_id,
        "status": _STATUS,
        "failed_nodes": failed_nodes,
        "failed_candidate": {
            "nodes": nodes,
            "connections": [
                {
                    "source": aliases[edge.source_id],
                    "target": aliases[edge.target_id],
                    "source_port": edge.source_port,
                    "target_port": edge.target_port,
                }
                for edge in candidate_edges
            ],
        },
        "rule": _RULE,
    }
    return RunFailureFeedback(
        run_id,
        context.source_id,
        context.baseline_fingerprint,
        context.source_fingerprint,
        context,
        _json(payload),
    )
