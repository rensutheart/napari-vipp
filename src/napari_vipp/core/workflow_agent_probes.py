"""Small, explicit diagnostic DAGs through the ordinary VIPP executor.

These are disposable experiments, not candidates and never an adoption path.
The parent session owns consent, authority, cancellation, counters and evidence
transmission. No numerical kernel, source loader or output writer lives here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

import numpy as np

from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.pipeline import EXECUTION_READY, PrototypePipeline, SourcePayload
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.core.workflow_assessment import (
    MAX_SNAPSHOT_BYTES,
    _capture_binding,
    _validate_image,
)
from napari_vipp.core.workflow_proposals import (
    PlanningPreferences,
    capture_proposal_context,
    validate_proposal,
)
from napari_vipp.core.workflow_refinement import assert_refinement_graph_safe

MAX_DIAGNOSTIC_PROBES = 6
MAX_PROBE_INPUT_VOXELS = 262_144
MAX_PROBE_NODES = 8
MAX_PROBE_OUTPUTS = 4
MAX_PROBE_SNAPSHOT_BYTES = MAX_SNAPSHOT_BYTES


class DiagnosticProbeError(ValueError):
    """An explicit probe exceeded its finite or scientific contract."""


class DiagnosticProbeStopped(DiagnosticProbeError):
    """Execution/cancellation/cleanup cannot safely yield probe evidence."""


@dataclass(frozen=True, slots=True)
class DiagnosticProbe:
    probe_id: str
    status: str
    _evidence: bytes = field(repr=False)
    outputs: tuple = field(default=(), repr=False)
    snapshot_bytes: int = 0

    @property
    def evidence(self):
        return json.loads(self._evidence)


def validate_probe_context(value):
    """Strict optional request-v2 allocation descriptor; grants no evidence."""
    fields = {
        "version",
        "attempts_used",
        "attempts_remaining",
        "maximum_attempts",
        "max_input_voxels",
        "max_nodes",
        "max_outputs",
        "snapshot_bytes_used",
        "snapshot_bytes_maximum",
        "index",
    }
    if type(value) is not dict or set(value) != fields:
        raise DiagnosticProbeError("Invalid diagnostic-probe allocation.")
    constants = {
        "version": 1,
        "maximum_attempts": MAX_DIAGNOSTIC_PROBES,
        "max_input_voxels": MAX_PROBE_INPUT_VOXELS,
        "max_nodes": MAX_PROBE_NODES,
        "max_outputs": MAX_PROBE_OUTPUTS,
        "snapshot_bytes_maximum": MAX_PROBE_SNAPSHOT_BYTES,
    }
    if any(type(value[k]) is not int or value[k] != v for k, v in constants.items()):
        raise DiagnosticProbeError("Diagnostic-probe hard limits cannot change.")
    used, remaining, size = (
        value["attempts_used"],
        value["attempts_remaining"],
        value["snapshot_bytes_used"],
    )
    if (
        type(used) is not int
        or not 0 <= used <= MAX_DIAGNOSTIC_PROBES
        or type(remaining) is not int
        or used + remaining != MAX_DIAGNOSTIC_PROBES
        or type(size) is not int
        or not 0 <= size <= MAX_PROBE_SNAPSHOT_BYTES
        or type(value["index"]) is not list
        or len(value["index"]) != used
    ):
        raise DiagnosticProbeError("Invalid diagnostic-probe consumption.")
    for index, row in enumerate(value["index"], 1):
        if (
            type(row) is not dict
            or set(row) != {"probe_id", "status", "full_record"}
            or row["probe_id"] != f"probe{index}"
            or type(row["status"]) is not str
            or row["status"] not in {"failed", "completed"}
            or type(row["full_record"]) is not dict
            or set(row["full_record"]) != {"record_id", "sha256"}
            or type(row["full_record"]["record_id"]) is not str
            or not re.fullmatch(
                r"record[1-9][0-9]{0,2}", row["full_record"]["record_id"]
            )
            or type(row["full_record"]["sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", row["full_record"]["sha256"])
        ):
            raise DiagnosticProbeError("Invalid diagnostic-probe record binding.")
    return value


def execute_diagnostic_probe(
    data,
    state,
    proposal,
    output_roles,
    *,
    context_id,
    compute_request,
    run_id,
    cancel_event,
    guard,
    remaining_snapshot_bytes,
):
    """Return immutable role bindings after one ordinary cleaned-up Run.

    ``data`` is the exact explicitly selected native ROI. Its translated grid
    and cut-edge limitations are carried by the session. Branches are ordinary
    graph edges; the caller/model, not this function, selects all parameters.
    """
    if type(data) is not np.ndarray or data.size > MAX_PROBE_INPUT_VOXELS:
        raise DiagnosticProbeError(
            "A probe input supports at most 262144 native voxels."
        )
    _validate_image(data, state)
    if type(proposal) is not dict or type(proposal.get("nodes")) is not list:
        raise DiagnosticProbeError("Use an ordinary explicit diagnostic proposal.")
    if not 1 <= len(proposal["nodes"]) <= MAX_PROBE_NODES:
        raise DiagnosticProbeError("A diagnostic probe needs one to eight nodes.")
    if (
        type(output_roles) is not list
        or not 1 <= len(output_roles) <= MAX_PROBE_OUTPUTS
    ):
        raise DiagnosticProbeError("Bind one to four explicit diagnostic outputs.")
    isolated = PrototypePipeline()
    isolated.reset_empty_graph()
    isolated.outputs["input"] = data
    isolated.output_states["input"] = state
    context = replace(
        capture_proposal_context(isolated, "input"),
        context_id=context_id,
        planning_preferences=PlanningPreferences("use_best_judgment"),
    )
    prepared = validate_proposal(proposal, context)
    if not prepared.can_apply:
        raise DiagnosticProbeError("Unanswered questions cannot execute a probe.")
    assert_refinement_graph_safe(prepared.candidate, "input")
    pipeline = prepared.candidate.to_pipeline()
    local_ids = {
        row["key"]: node
        for row, node in zip(proposal["nodes"], prepared.added_node_ids, strict=True)
    }
    roles, seen = [], set()
    for binding in output_roles:
        if type(binding) is not dict or set(binding) != {"role", "node", "port"}:
            raise DiagnosticProbeError("Each diagnostic output needs role/node/port.")
        role, port = binding["role"], binding["port"]
        node_id = local_ids[binding["node"]]
        if (
            type(role) is not str
            or not role.strip()
            or len(role) > 80
            or role in seen
            or type(port) is not int
            or not 0 <= port < len(pipeline.output_ports(node_id))
        ):
            raise DiagnosticProbeError(
                "Diagnostic roles must uniquely bind existing ports."
            )
        seen.add(role)
        roles.append((role, node_id, port))
    guard()
    request = PipelineRunRequest(
        run_id=run_id,
        workflow=serialize_workflow(pipeline, compute_request=compute_request),
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={"input": SourcePayload(data, image_state=state)},
        compute_request=compute_request,
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
        target_node_ids=frozenset(prepared.added_node_ids),
        retain_node_ids=frozenset(prepared.added_node_ids),
        cancel_event=cancel_event,
    )
    try:
        result = execute_pipeline_request(request)
    except Exception as exc:
        raise DiagnosticProbeStopped(
            "Ordinary diagnostic execution failed; review is required."
        ) from exc
    guard()
    if (
        result.cancelled
        or result.error
        or result.failure is not None
        or result.execution_report is None
        or result.execution_report.cleanup_succeeded is not True
        or result.pipeline is None
    ):
        raise DiagnosticProbeStopped(
            "Diagnostic execution was cancelled, failed or lacked cleanup proof; "
            "no output evidence exists."
        )
    calculated = result.pipeline
    if any(
        calculated.node_execution_states.get(node) != EXECUTION_READY
        or node not in calculated.completed_node_ids
        for node in prepared.added_node_ids
    ):
        raise DiagnosticProbeStopped(
            "Every diagnostic node must complete before evidence is captured."
        )
    budget, captured = [data.nbytes], []
    for role, node_id, port in roles:
        output = calculated.node_outputs[node_id][port]
        output_state = calculated.node_output_states[node_id][port]
        if type(output) is np.ndarray:
            _validate_image(output, output_state)
        binding = _capture_binding(node_id, port, output, output_state, budget)
        captured.append(
            (role, replace(binding, live=None), calculated.nodes[node_id].operation_id)
        )
    if budget[0] > remaining_snapshot_bytes:
        raise DiagnosticProbeError(
            "The session's 64 MiB diagnostic snapshot allocation is full."
        )
    return tuple(captured), budget[0]
