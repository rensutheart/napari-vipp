"""Finite, source-bound investigation through existing detached VIPP workflows.

This is an action interpreter, not a Python/shell tool runtime. It never changes
the accepted graph, loads a source, writes files, or calls a model. The caller
separately approves starting it and supplies current-authority/cancel callbacks.
All trial pixels come from the ordinary shared executor and existing nodes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from io import BytesIO
from numbers import Rational
from time import monotonic
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.diagnostics import (
    exact_finite_percentiles,
    exact_finite_stats,
    exact_histogram,
)
from napari_vipp.core.execution import (
    PipelineExecutionFailure,
    PipelineRunRequest,
    execute_pipeline_request,
)
from napari_vipp.core.grid import ImageGrid, compare_aligned_grids
from napari_vipp.core.mesh_refinement import safe_mesh_input_failure
from napari_vipp.core.pipeline import (
    _RESOLVED_SPATIAL_PARAMETER_OPERATION_IDS,
    EXECUTION_READY,
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.core.workflow_agent_budget import (
    HARD_ALLOWANCE,
    MAX_CONTINUATION_GRANTS,
    TIME_POLICY,
    allowance_dict,
)
from napari_vipp.core.workflow_agent_probes import (
    MAX_DIAGNOSTIC_PROBES,
    MAX_PROBE_INPUT_VOXELS,
    MAX_PROBE_NODES,
    MAX_PROBE_OUTPUTS,
    MAX_PROBE_SNAPSHOT_BYTES,
    DiagnosticProbe,
    DiagnosticProbeError,
    DiagnosticProbeStopped,
    execute_diagnostic_probe,
)
from napari_vipp.core.workflow_assessment import (
    _SAFE_TABLE_COLUMNS,
    MAX_SNAPSHOT_BYTES,
    AssessmentSnapshot,
    _capture_binding,
    _data_hash,
    _graph_fingerprint,
    _grid_summary,
    _image_summary,
    _mesh_summary,
    _MeshSnapshot,
    _Output,
    _state_hash,
    _table_bytes,
    _table_summary,
    _validate_image,
    capture_assessment_snapshot,
)
from napari_vipp.core.workflow_assessment_mesh import render_mesh_preview
from napari_vipp.core.workflow_inspection_metrics import (
    foreground_change_summary,
    public_role_binding,
)
from napari_vipp.core.workflow_investigation_context import (
    DecisionContextError,
    build_decision_context,
    validate_decision_context,
)
from napari_vipp.core.workflow_investigation_edits import (
    RecipeEditError,
    apply_recipe_edits,
)
from napari_vipp.core.workflow_investigation_experiments import (
    ExperimentError,
    ExperimentLedger,
)
from napari_vipp.core.workflow_investigation_history import (
    expand_request_history as expand_request_history,
)
from napari_vipp.core.workflow_investigation_notebook import (
    InvestigationNotebook,
    NotebookError,
    NotebookPreviewCache,
)
from napari_vipp.core.workflow_investigation_review_budget import review_reservation
from napari_vipp.core.workflow_investigation_topology import topology_review_payload
from napari_vipp.core.workflow_proposal_catalogue import (
    authorable_parameter_specs,
    build_catalogue,
    optional_authorable_parameter_specs,
)
from napari_vipp.core.workflow_proposal_images import (
    MAX_IMAGE_PREVIEWS,
    MAX_PREVIEW_EDGE,
    MAX_PREVIEW_TOTAL_BYTES,
    ImagePreview,
)
from napari_vipp.core.workflow_proposals import (
    MAX_PROPOSAL_BYTES,
    MAX_REQUEST_BYTES,
    PlanningPreferences,
    PreparedProposal,
    ProposalDialogue,
    ProposalError,
    _decode,
    _inspect_source,
    capture_proposal_context,
    validate_proposal,
)
from napari_vipp.core.workflow_refinement import (
    PRESERVED_OUTPUT_KEY,
    REFINEMENT_SAFE_OPERATION_IDS,
    PreservationConstraint,
    assert_refinement_graph_safe,
    preservation_mesh_creators,
)

ACTION_KIND = "napari-vipp-investigation-action"
MAX_NODE_DESCRIPTIONS = 8
MAX_NODE_INDEX_ENTRIES = 256
WORKING_REQUEST_TARGET_BYTES = 128 * 1024
_CLOSING_READ_ACTIONS = frozenset(
    {
        "read_notebook",
        "pin_notebook",
        "pin_previews",
        "inspect_source",
        "inspect_candidate",
        "inspect_probe",
        "compare_candidates",
        "describe_candidate",
    }
)
_PRIVATE_PATH = re.compile(
    r"[A-Za-z]:[\\/]|\\\\|file://|(?:^|[\s\"'])/(?:[A-Za-z0-9_.-]+/)+",
    re.IGNORECASE,
)
_ACTION_ARGUMENTS = {
    "read_notebook": ({"record_id", "sha256", "page"}, set()),
    "pin_notebook": ({"refs"}, set()),
    "pin_previews": ({"view_ids"}, set()),
    "find_nodes": ({"query"}, set()),
    "describe_node": ({"operation_id"}, set()),
    "describe_nodes": ({"operation_ids"}, set()),
    "describe_candidate": ({"candidate_id"}, set()),
    "plan_experiment": (
        {
            "baseline_candidate_id",
            "evidence_refs",
            "observed_failure",
            "hypothesis",
            "prior_assumptions",
            "predicted_geometric_change",
            "risks",
            "intent",
            "evaluation_checks",
        },
        set(),
    ),
    "assess_experiment": (
        {"experiment_id", "candidate_id", "check_results", "unresolved_criteria"},
        set(),
    ),
    "inspect_source": (set(), {"percentiles", "histogram", "views", "roi"}),
    "diagnostic_probe": ({"input", "roi", "proposal", "output_roles"}, set()),
    "inspect_probe": (
        {"probe_id", "role"},
        {"percentiles", "histogram", "views", "roi"},
    ),
    "trial_candidate": ({"proposal", "output_roles"}, {"experiment_id"}),
    "trial_edit": ({"candidate_id", "edits"}, {"experiment_id"}),
    "inspect_candidate": (
        {"candidate_id", "role"},
        {"percentiles", "histogram", "views", "roi"},
    ),
    "compare_candidates": ({"left", "right", "roles"}, {"roi"}),
    "retain_candidate": ({"candidate_id", "reason"}, set()),
    "finish": ({"candidate_id", "summary", "limitations"}, set()),
    "question": ({"question"}, set()),
    "record_observation": (
        {"evidence_refs", "interpretation", "hypothesis", "next_check"},
        set(),
    ),
}


class InvestigationError(ValueError):
    """A malformed action or unavailable bounded capability."""


class InvestigationStopped(InvestigationError):
    """Authority, cancellation, elapsed time, or a finite budget stopped work."""


class InvestigationPaused(InvestigationError):
    """An owner-resumable checkpoint, not cancellation or invalid evidence."""


def _json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvestigationError("Investigation values must be finite JSON.") from exc


def _text(value, name, maximum=2000):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise InvestigationError(
            f"{name} must be non-empty text of at most {maximum} characters."
        )
    return value


def _no_paths(value):
    if isinstance(value, str) and _PRIVATE_PATH.search(value):
        raise InvestigationError("Filesystem paths are not investigation actions.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _no_paths(key)
            _no_paths(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _no_paths(item)


@dataclass(frozen=True, slots=True)
class InvestigationLimits:
    max_attempts: int = 5
    max_actions: int = 32
    max_model_calls: int = 40
    max_seconds: float = 1800
    max_image_bytes: int = 10 * 1024 * 1024

    def __post_init__(self):
        for name, maximum in (
            ("max_attempts", 5),
            ("max_actions", 64),
            ("max_model_calls", 80),
            ("max_image_bytes", 20 * 1024 * 1024),
        ):
            if (
                type(getattr(self, name)) is not int
                or not 1 <= getattr(self, name) <= maximum
            ):
                raise InvestigationError(
                    f"{name} must be an integer from 1 to {maximum}."
                )
        if (
            type(self.max_seconds) not in {int, float}
            or not math.isfinite(self.max_seconds)
            or not 0 < self.max_seconds <= 1800
        ):
            raise InvestigationError(
                "Investigation time must be positive and at most 1800 seconds."
            )


@dataclass(frozen=True, init=False, slots=True)
class InvestigationAction:
    session_id: str
    action: str
    _arguments: bytes = field(repr=False)

    def __init__(self, session_id: str, action: str, arguments: Mapping[str, Any]):
        _text(session_id, "Session ID", 128)
        if (
            type(action) is not str
            or action not in _ACTION_ARGUMENTS
            or not isinstance(arguments, Mapping)
        ):
            raise InvestigationError(
                "Unknown investigation action or invalid arguments."
            )
        required, optional = _ACTION_ARGUMENTS[action]
        if (
            not required <= arguments.keys()
            or not arguments.keys() <= required | optional
        ):
            raise InvestigationError(f"Invalid arguments for {action}.")
        encoded = _json(dict(arguments))
        _no_paths(arguments)
        if len(encoded) > MAX_PROPOSAL_BYTES:
            raise InvestigationError("Action arguments exceed 128 KiB.")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "_arguments", encoded)

    @property
    def arguments(self) -> dict:
        return json.loads(self._arguments)

    def to_dict(self):
        return {
            "kind": ACTION_KIND,
            "version": 1,
            "session_id": self.session_id,
            "action": self.action,
            "arguments": self.arguments,
        }


def parse_investigation_action(raw, *, expected_session_id=None) -> InvestigationAction:
    try:
        value = _decode(raw)
    except (ValueError, TypeError) as exc:
        raise InvestigationError(str(exc)) from exc
    if (
        set(value) != {"kind", "version", "session_id", "action", "arguments"}
        or value["kind"] != ACTION_KIND
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise InvestigationError("Unsupported investigation action envelope.")
    result = InvestigationAction(
        value["session_id"], value["action"], value["arguments"]
    )
    if expected_session_id is not None and result.session_id != expected_session_id:
        raise InvestigationError("The action belongs to another investigation session.")
    return result


def action_schema() -> dict:
    """Small provider-neutral wire contract; no provider-native tool privilege."""
    return {
        "envelope": {
            "kind": ACTION_KIND,
            "version": 1,
            "session_id": "copy request session_id",
            "action": "one action below",
            "arguments": {},
        },
        "actions": {
            "read_notebook": {
                "record_id": "exact this-session notebook record ID",
                "sha256": "exact complete-record hash from record_index",
                "page": (
                    "zero-based integer page; authenticated exact canonical JSON "
                    "bytes in data_base64url with hash/Merkle proof, plus readable "
                    "ASCII text (~b means backslash, ~~ means literal tilde)"
                ),
            },
            "pin_notebook": {
                "refs": (
                    "0..4 exact {record_id,sha256} refs, total complete payloads "
                    "<=32 KiB; [] clears"
                ),
            },
            "pin_previews": {
                "view_ids": (
                    "0..7 exact view_id values from prior inspections; atomic "
                    "current PNG board, [] clears explicit pins; automatic "
                    "source/retained/latest comparison context can remain"
                ),
            },
            "find_nodes": {
                "query": (
                    "space-separated keywords matched together against ordinary "
                    "node title/category/id, or '*' for the bounded registry index"
                )
            },
            "describe_node": {"operation_id": "exact existing operation ID"},
            "describe_nodes": {
                "operation_ids": "1..8 unique exact existing operation IDs"
            },
            "describe_candidate": {
                "candidate_id": (
                    "this-session completed or failed candidate alias; retrieves "
                    "exact authored scientific parameters, wiring and role bindings"
                )
            },
            "inspect_source": {
                "percentiles": "optional list of 0..100, max 8",
                "histogram": "optional boolean",
                "views": "optional requested views",
                "roi": "optional exact diagnostic region; see roi_contract",
            },
            "trial_candidate": {
                "experiment_id": (
                    "prospective experiment alias from plan_experiment; required "
                    "when request.experiment_plan_required is true"
                ),
                "proposal": (
                    "ordinary strict workflow proposal using proposal_context_id; "
                    "no writers/sources/custom code"
                ),
                "output_roles": (
                    "1..8 objects {role,node,port}; node is proposal-local key, "
                    "port is integer"
                ),
            },
            "diagnostic_probe": {
                "input": (
                    "exact {kind:'source'} or {kind:'candidate',candidate_id,role}; "
                    "completed array role only, no new source"
                ),
                "roi": "required exact native ROI; at most 262144 input voxels",
                "proposal": (
                    "ordinary strict proposal using proposal_context_id; 1..8 safe "
                    "nodes, explicit parameters and wires; may branch to compare "
                    "methods/scales in one execution; input is only this declared ROI"
                ),
                "output_roles": (
                    "1..4 {role,node,port}; disposable diagnostic outputs, not "
                    "candidates and cannot be retained, edited or finished; recreate "
                    "a chosen hypothesis in a full checked trial. Cut ROI borders "
                    "are not whole-object shape/completeness evidence"
                ),
            },
            "inspect_probe": {
                "probe_id": "this-session completed diagnostic probe alias",
                "role": "explicit bound diagnostic output role",
                "percentiles": "optional list",
                "histogram": "optional boolean",
                "views": "optional views; indices address the probe output native grid",
                "roi": "optional exact diagnostic region within that probe output",
            },
            "trial_edit": {
                "candidate_id": (
                    "completed candidate alias, including accepted candidate0"
                ),
                "edits": (
                    "1..8 unique-node edits: {kind:'set_parameters',node:exact recipe "
                    "key,params:explicit changed parameter values} or "
                    "{kind:'replace_operation',node:exact key,operation_id:eligible ID,"
                    "params:ALL required replacement parameters}. Keys, wiring and "
                    "output roles are preserved; no insertion/deletion/rewiring. "
                    "The complete detached result passes ordinary validation."
                ),
                "experiment_id": "prospective plan alias, required when advertised",
            },
            "inspect_candidate": {
                "candidate_id": "completed candidate alias",
                "role": "explicit bound output role",
                "percentiles": "optional list",
                "histogram": "optional boolean",
                "views": "optional requested views",
                "roi": "optional exact diagnostic region; see roi_contract",
            },
            "compare_candidates": {
                "left": "completed candidate alias",
                "right": "completed candidate alias",
                "roles": (
                    "1..8 explicit roles common to both; requires identical "
                    "source, identical output dtypes and aligned image grids"
                ),
                "roi": "optional exact diagnostic region; see roi_contract",
            },
            "retain_candidate": {
                "candidate_id": "completed candidate alias",
                "reason": "evidence-based model interpretation, not accuracy proof",
            },
            "finish": {
                "candidate_id": "explicit completed candidate alias",
                "summary": "model interpretation",
                "limitations": "list of unresolved uncertainties",
            },
            "question": {
                "question": "one plain-language essential question; stops investigation"
            },
            "record_observation": {
                "evidence_refs": (
                    "1..8 already transmitted references: {kind:'preview',sha256:"
                    "exact preview manifest hash} or {kind:'numeric',sha256:exact "
                    "numeric evidence hash,alias:'input'|'candidateN/role'}"
                ),
                "interpretation": "brief model interpretation, max 800 characters",
                "hypothesis": (
                    "next-test hypothesis, not observed fact; max 800 characters"
                ),
                "next_check": "brief proposed next evidence check; max 400 characters",
            },
            "plan_experiment": {
                "baseline_candidate_id": "null for input or completed candidate alias",
                "evidence_refs": "1..8 previously transmitted exact evidence handles",
                "observed_failure": "evidence-bound interpretation, max 800 chars",
                "hypothesis": (
                    "selected test hypothesis and brief selection reason, max 800 chars"
                ),
                "prior_assumptions": "1..4 model-prior notes, each max 400 chars",
                "predicted_geometric_change": "predicted change, max 800 chars",
                "risks": "1..4 possible harms/tradeoffs, each max 400 chars",
                "intent": "reconstruction|cleanup|presentation",
                "evaluation_checks": (
                    "1..8 {check_id:unique [A-Za-z0-9_-] identifier max40,"
                    "kind:numeric|visual,criterion:"
                    "text max400}; at least one numeric check; plan before trial"
                ),
            },
            "assess_experiment": {
                "experiment_id": "exact prospectively linked experiment alias",
                "candidate_id": "its completed trial candidate alias",
                "check_results": (
                    "one {check_id,outcome:supports|contradicts|unresolved,"
                    "evidence_refs:[0..8 issued handles],assessment:text max600} "
                    "per planned check; resolved checks need candidate-owned "
                    "evidence of their declared kind; at least one candidate-owned "
                    "numeric handle overall; at most eight distinct handles "
                    "across the complete assessment"
                ),
                "unresolved_criteria": (
                    "0..8 remaining uncertainty notes max400 each; nonempty when "
                    "any check is unresolved; all assessments are model judgments"
                ),
            },
        },
        "proposal_contract": {
            "kind": "napari-vipp-workflow-proposal",
            "version": 1,
            "context_id": "copy request proposal_context_id",
            "summary": "plain-language candidate purpose",
            "assumptions": ["explicit scientific choices, not observed results"],
            "questions": [],
            "nodes": [
                {
                    "key": "unique local node key, not input",
                    "operation_id": "existing ID from describe_node",
                    "params": (
                        "object containing ALL required named parameters from "
                        "describe_node; only declared optional parameters"
                    ),
                    "reason": "why this node is proposed",
                }
            ],
            "connections": [
                {
                    "source": "input or new local key",
                    "source_port": 0,
                    "target": "new local key",
                    "target_port": 0,
                }
            ],
            "bounds": (
                "64 nodes, 128 connections; all nodes reachable from input; no "
                "new source, writer, code, path or external action"
            ),
        },
        "planning_preferences": (
            "Use best judgment for reversible candidate choices; ask only essential "
            "safety questions using question action. Unknown biological identity, "
            "accuracy and completeness remain explicit limitations."
        ),
        "mesh_view_contract": (
            "Mesh output views accepts one or two {view_index:0|1} objects for "
            "existing calibrated oblique cameras. No crop, intensity overlay, "
            "remeshing or topology repair is inferred."
        ),
        "roi_contract": (
            "Optional diagnostic-only {axis_order:'ZYX'|'YX', "
            "bounds:[[start,stop],...]} specifies EVERY native axis in carried "
            "order, using integer zero-based stop-exclusive bounds. No clipping, "
            "resampling, projection or scientific output crop. Measurements are "
            "exact over the requested region and include parent/region grids. "
            "Region components and border contacts are local, cut by its bounds, "
            "not whole-object completeness or biological accuracy. Views remain "
            "independently selected on the full original binding, not this region."
        ),
        "view_contract": (
            "Each image view: {axis:'Z'|'Y'|'X' (ZYX) or 'YX' (2D), "
            "indices:[zero-based integers, max 7], "
            "crop:[row_start,row_stop,column_start,column_stop] optional, "
            "overlay_source:boolean optional,montage:boolean optional}. Bounds "
            "are native selected-plane indices; stop exclusive. No projection, "
            "interpolation or implicit clipping. Source overlays require BOTH "
            "image consents and aligned grids."
        ),
    }


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    _public: bytes = field(repr=False)
    image_previews: tuple[ImagePreview, ...] = ()

    def public_result(self) -> dict:
        return json.loads(self._public)


@dataclass(frozen=True, slots=True)
class InvestigationCandidate:
    candidate_id: str
    run_id: str
    status: str
    graph: GraphSnapshot | None = field(repr=False)
    _recipe: bytes = field(repr=False)
    _evidence: bytes = field(repr=False)
    _outputs: tuple = field(default=(), repr=False)
    source_fingerprint: str = ""
    _role_bindings: bytes = field(default=b"[]", repr=False)
    _key_to_node: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    @property
    def recipe(self):
        return json.loads(self._recipe)

    @property
    def proposal(self):
        """Normal proposal JSON, without caches, for a separately checked Add."""
        return self.recipe

    @property
    def evidence(self):
        return json.loads(self._evidence)

    @property
    def output_roles(self):
        return json.loads(self._role_bindings)

    @property
    def key_to_node(self):
        """Exact private execution identities for recording, not public aliases."""
        return dict(self._key_to_node)


@dataclass(frozen=True, slots=True)
class CompletedCandidateBundle:
    """GUI-captured accepted evidence; workers never inspect a live UI pipeline."""

    _snapshot: AssessmentSnapshot = field(repr=False)
    _recipe: bytes = field(repr=False)
    _roles: bytes = field(repr=False)
    _key_to_node: tuple[tuple[str, str], ...] = field(repr=False)
    _binding: bytes = field(repr=False)
    _role_outputs: tuple = field(repr=False)
    preservation_constraint: PreservationConstraint | None = field(
        default=None, repr=False
    )
    protected_role: str | None = None
    _live_snapshot: AssessmentSnapshot | None = field(default=None, repr=False)

    @property
    def recipe(self):
        return json.loads(self._recipe)

    @property
    def output_roles(self):
        return json.loads(self._roles)

    @property
    def key_to_node(self):
        return dict(self._key_to_node)

    def public_binding(self):
        return json.loads(self._binding)

    @property
    def snapshot(self):
        return self._snapshot

    def assert_current(self, pipeline):
        """Call only against the real owned pipeline on its composition thread."""
        if self._live_snapshot is None:
            raise InvestigationError("A worker bundle has no live adoption authority.")
        self._live_snapshot.assert_current(pipeline)


def _binding_bytes(binding):
    data = binding.data
    if type(data) is np.ndarray:
        return data.nbytes
    if type(data) is _MeshSnapshot:
        return data.vertices.nbytes + data.faces.nbytes + data.face_object_ids.nbytes
    return _table_bytes(data)


def capture_completed_candidate(
    pipeline,
    snapshot,
    output_roles,
    *,
    preservation_constraint=None,
    protected_role=None,
):
    """Capture explicitly selected accepted ports on the GUI/composition thread.

    No calculation, source loading, name guessing or parameter inference occurs.
    This is a bounded subset, not permission to execute unrelated graph branches.
    The caller owns successful Run/session/dispatch authority before this call.
    """
    if type(snapshot) is not AssessmentSnapshot:
        raise InvestigationError("A current completed owned snapshot is required.")
    snapshot.assert_current(pipeline)
    if snapshot._uncalculated:
        raise InvestigationError("Accepted scientific nodes must all be calculated.")
    if type(output_roles) is not list or not 1 <= len(output_roles) <= 8:
        raise InvestigationError("Bind 1..8 explicit accepted output roles.")
    if (
        preservation_constraint is not None
        and type(preservation_constraint) is not PreservationConstraint
    ):
        raise InvestigationError("Use an explicitly captured preservation constraint.")
    if (preservation_constraint is None) != (protected_role is None):
        raise InvestigationError("Protection needs its explicit delivered output role.")
    keys = {node: f"b{i + 1}" for i, node in enumerate(snapshot.candidate_node_ids)}
    roles, outputs, seen, budget = [], [], set(), [snapshot._source.data.nbytes]
    known = {
        (out.node_id, out.port): out.binding
        for out in snapshot._outputs
        if out.binding is not None
    }
    retained = {}
    if preservation_constraint is not None:
        for reference in (
            preservation_constraint.primary_ref,
            preservation_constraint.region_ref,
        ):
            if reference is None:
                continue
            if reference in retained:
                continue
            if reference not in known:
                raise InvestigationError("Original protected ports are not captured.")
            binding = known[reference]
            retained[reference] = _Output(
                reference[0], reference[1], "original preservation selection", binding
            )
            budget[0] += _binding_bytes(binding)
    for item in output_roles:
        if type(item) is not dict or set(item) != {"role", "node", "port"}:
            raise InvestigationError("Each accepted role needs exactly role/node/port.")
        role, node, port = item["role"], item["node"], item["port"]
        _text(role, "Accepted output role", 80)
        if (
            role in seen
            or type(node) is not str
            or node not in keys
            or type(port) is not int
            or not 0 <= port < len(pipeline.output_ports(node))
        ):
            raise InvestigationError(
                "Accepted roles must bind unique names and existing ports."
            )
        seen.add(role)
        values, states = (
            pipeline.node_outputs.get(node),
            pipeline.node_output_states.get(node),
        )
        data = values[port] if values else pipeline.outputs.get(node)
        state = states[port] if states else pipeline.output_states.get(node)
        if data is None:
            raise InvestigationError("An accepted output is unavailable.")
        if type(data) is np.ndarray:
            _validate_image(data, state)
        reference = (node, port)
        if reference in retained:
            binding = retained[reference].binding
        elif reference in known:
            binding = known[reference]
            budget[0] += _binding_bytes(binding)
        else:
            binding = _capture_binding(node, port, data, state, budget)
        if budget[0] > MAX_SNAPSHOT_BYTES:
            raise InvestigationError(
                "Accepted source and selected ports exceed 64 MiB."
            )
        retained[reference] = _Output(node, port, role, binding)
        if len(retained) > 8:
            raise InvestigationError(
                "Accepted roles plus original protected ports exceed eight outputs."
            )
        outputs.append(_Output(node, port, role, binding))
        roles.append({"role": role, "node": node, "port": port})
        if role == protected_role:
            if port != 0:
                raise InvestigationError("The protected output must use port 0.")
            keys[node] = PRESERVED_OUTPUT_KEY
    if protected_role is not None and protected_role not in seen:
        raise InvestigationError("The protected role is not explicitly selected.")
    # Retain original captured constraint ports even when the explicit delivered
    # role is a different, equivalent mask (e.g. labels versus mesh input mask).
    expanded = replace(snapshot, _outputs=tuple(retained.values()), _omitted=())
    if preservation_constraint is not None:
        selected = next(row for row in roles if row["role"] == protected_role)
        check = preservation_constraint.check_detached(
            expanded,
            expanded,
            (selected["node"], selected["port"]),
            accepted_baseline=True,
        )
        if not check.passed:
            raise InvestigationError(
                "The accepted delivered role does not retain the protected region."
            )
    nodes, local_owned = [], []
    for node in snapshot.baseline.nodes:
        if node.id not in keys:
            continue
        if (
            node.operation_id not in REFINEMENT_SAFE_OPERATION_IDS
            or node.execution_mode != "run"
        ):
            raise InvestigationError(
                "Accepted nodes must be eligible ordinary run operations."
            )
        allowed = {
            p.name
            for p in (
                *authorable_parameter_specs(node.operation_id),
                *optional_authorable_parameter_specs(node.operation_id),
            )
        }
        params = node.params
        extras = set(params) - allowed
        for name in tuple(extras):
            value = params[name]
            if (
                name == "resolved_spatial_ndim"
                and node.operation_id in _RESOLVED_SPATIAL_PARAMETER_OPERATION_IDS
                and type(value) is int
                and value in {1, 2, 3}
            ):
                extras.remove(name)
                local_owned.append(
                    {
                        "node": keys[node.id],
                        "parameter": name,
                        "value": value,
                        "policy": "executor-derived; recomputed normally",
                    }
                )
            elif name == MANUAL_AUTO_RECALCULATE_PARAM and type(value) is bool:
                extras.remove(name)
                local_owned.append(
                    {
                        "node": keys[node.id],
                        "parameter": name,
                        "value": value,
                        "policy": (
                            "local execution preference; trial remains explicit manual"
                        ),
                    }
                )
        if extras:
            raise InvestigationError(
                "An accepted node has unsupported non-authorable parameters."
            )
        specs = {
            parameter.name: parameter
            for parameter in (
                *authorable_parameter_specs(node.operation_id),
                *optional_authorable_parameter_specs(node.operation_id),
            )
        }
        for name, value in params.items():
            if name not in allowed or type(value) is not str:
                continue
            parameter = specs[name]
            if parameter.kind == "choice" and value in parameter.choices:
                continue
            if value == "" and parameter.kind == "text":
                continue
            structural = False
            if parameter.kind == "text":
                if node.operation_id == "reorder_axes" and name == "order":
                    structural = re.fullmatch(r"[TZCYX]+", value) is not None
                elif (
                    node.operation_id == "calculate_weighted_image"
                    and name == "weights"
                ):
                    tokens = value.split(",")
                    structural = 1 <= len(tokens) <= 32 and all(
                        re.fullmatch(
                            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
                            token.strip(),
                        )
                        and math.isfinite(float(token))
                        for token in tokens
                    )
                elif name in {
                    "property_column",
                    "group_by",
                    "columns",
                    "value_columns",
                    "join_keys",
                    "x_column",
                    "y_column",
                    "group_column",
                    "image_column",
                    "sample_column",
                }:
                    structural = all(
                        token.strip()
                        in (
                            _SAFE_TABLE_COLUMNS
                            | {"auto", "area_pixels", "volume_voxels"}
                        )
                        for token in value.split(",")
                    )
                elif name == "statistics":
                    structural = all(
                        token.strip()
                        in {
                            "count",
                            "sum",
                            "mean",
                            "median",
                            "std",
                            "min",
                            "max",
                            "q25",
                            "q75",
                        }
                        for token in value.split(",")
                    )
                elif node.operation_id == "plot_results" and name in {
                    "x_tick_interval",
                    "y_tick_interval",
                }:
                    structural = value == "Auto"
            if not structural:
                raise InvestigationError(
                    "This branch has existing free-text settings that cannot be "
                    "shared safely. Choose a branch without private text settings; "
                    "nothing was changed or sent."
                )
        nodes.append(
            {
                "key": keys[node.id],
                "operation_id": node.operation_id,
                "params": {
                    name: value for name, value in params.items() if name in allowed
                },
                "reason": "Preserve the explicitly accepted existing operation.",
            }
        )
    edges = []
    for edge in snapshot.baseline.connections:
        if edge.target_id not in keys:
            continue
        if (
            edge.source_id not in keys
            and edge.source_id != snapshot.source_id
            or edge.tunnel_name
        ):
            raise InvestigationError(
                "Accepted nodes must use only the selected source "
                "and ordinary local wires."
            )
        edges.append(
            {
                "source": "input"
                if edge.source_id == snapshot.source_id
                else keys[edge.source_id],
                "source_port": edge.source_port,
                "target": keys[edge.target_id],
                "target_port": edge.target_port,
            }
        )
    recipe = {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": "accepted",
        "summary": "The explicitly accepted existing candidate; not anatomical truth.",
        "assumptions": ["Existing scientific choices are retained for review."],
        "questions": [],
        "nodes": nodes,
        "connections": edges,
    }
    public_roles = [{**row, "node": keys[row["node"]]} for row in roles]
    public = {
        "candidate_id": "candidate0",
        "run_id": snapshot.run_id,
        "source_sha256": snapshot._source.fingerprint,
        "source_grid": _grid_summary(snapshot._source.state),
        "role_bindings": [
            public_role_binding(
                row["role"],
                keys[row["node"]],
                f"n{snapshot.candidate_node_ids.index(row['node']) + 1}",
                row["port"],
                output.binding.fingerprint,
            )
            for row, output in zip(roles, outputs, strict=True)
        ],
        "local_execution_fields": local_owned,
        "meaning": (
            "Existing accepted inference imported without another calculation; "
            "not scientific accuracy."
        ),
    }
    if preservation_constraint is not None:
        public["preservation_check"] = check.public_result()
    _no_paths(recipe)
    _no_paths(public)
    snapshot.assert_current(pipeline)
    # GUI freshness authority stays separate. Workers receive private data only;
    # these pointers refer to owned readonly copies, never current GUI caches.
    private_snapshot = replace(
        expanded,
        _source=replace(expanded._source, live=expanded._source.data),
        _outputs=tuple(
            replace(out, binding=replace(out.binding, live=out.binding.data))
            for out in expanded._outputs
        ),
    )
    private_bindings = {
        (out.node_id, out.port): out.binding for out in private_snapshot._outputs
    }
    private_roles = tuple(
        replace(out, binding=private_bindings[(out.node_id, out.port)])
        for out in outputs
    )
    private_constraint = (
        replace(preservation_constraint, _snapshot=private_snapshot)
        if preservation_constraint is not None
        else None
    )
    return CompletedCandidateBundle(
        private_snapshot,
        _json(recipe),
        _json(public_roles),
        tuple((key, node) for node, key in keys.items()),
        _json(public),
        private_roles,
        private_constraint,
        protected_role,
        expanded,
    )


class _CancelEvent(threading.Event):
    def __init__(self, check):
        super().__init__()
        self._check = check

    def is_set(self):
        return super().is_set() or self._check()


class InvestigationSession:
    """One immutable-source, bounded in-session candidate journal.

    ``assert_current`` must check the caller's session/source/compute authority;
    the captured normal proposal context also checks original graph and pixels.
    Call ``record_model_call`` before every transport attempt, including repair.
    Only a separate normal checked Add may publish ``final_add_proposal()``.
    """

    def __init__(
        self,
        pipeline: PrototypePipeline,
        source_id: str,
        dialogue: str | ProposalDialogue,
        *,
        compute_request: ComputeRequest,
        limits: InvestigationLimits | None = None,
        include_input_images=False,
        include_output_images=False,
        require_experiment_plan=False,
        review_before_finish=False,
        continuation_enabled=False,
        assert_current: Callable[[], None],
        cancelled: Callable[[], bool] | None = None,
        clock: Callable[[], float] = monotonic,
    ):
        if (
            type(compute_request) is not ComputeRequest
            or type(include_input_images) is not bool
            or type(include_output_images) is not bool
            or type(require_experiment_plan) is not bool
            or type(review_before_finish) is not bool
            or type(continuation_enabled) is not bool
            or not callable(assert_current)
        ):
            raise InvestigationError(
                "Explicit compute policy, image choices and an authority "
                "callback are required."
            )
        self.session_id = uuid4().hex
        if limits is not None and type(limits) is not InvestigationLimits:
            raise InvestigationError("Use explicit finite InvestigationLimits.")
        if limits is not None:
            limits.__post_init__()
        self.limits = limits or InvestigationLimits()
        self._allowance_binding = _json(allowance_dict(self.limits))
        self._clock, self._authority, self._cancelled = clock, assert_current, cancelled
        started = clock()
        if type(started) not in {int, float} or not math.isfinite(started):
            raise InvestigationError("A finite monotonic elapsed clock is required.")
        self.deadline = started + self.limits.max_seconds
        self._started = self._last_clock = started
        self._continuation_enabled = continuation_enabled
        self._agent_tool_lock = threading.RLock()
        self._clock_lock = threading.Lock()
        self._pause_request_lock = threading.Lock()
        self._pause_request_generation = 0
        self._pause_requested = threading.Event()
        self._requested_pause = None
        self._pause_started = None
        self._paused_seconds = 0.0
        self._pause_kind = self._pause_reason = ""
        self._pending_question = None
        self._active_action = False
        self._allowance_grants = self._control_revision = 0
        self._pipeline, self._source_id = pipeline, source_id
        self._accepted_bundle = None
        self._seed_count = 0
        self._preservation = None
        self._protected_role = None
        self._original = capture_proposal_context(pipeline, source_id)
        self.dialogue = (
            dialogue
            if isinstance(dialogue, ProposalDialogue)
            else ProposalDialogue(dialogue)
        )
        self.compute_request = compute_request
        self.include_input_images, self.include_output_images = (
            include_input_images,
            include_output_images,
        )
        # Compatibility name: output sharing covers diagnostics AND PNGs, as in
        # the existing independent output-assessment permission.
        self._permission_binding = (include_input_images, include_output_images)
        source, _, _ = _inspect_source(pipeline, source_id)
        state = self._original._source_state
        if state.axis_order not in {"YX", "ZYX"}:
            raise InvestigationError(
                "Investigation requires explicit scalar YX or ZYX axes; "
                "no dimensional interpretation is inferred."
            )
        self._source = _capture_binding("input", 0, source, state, [0])
        isolated = PrototypePipeline()
        isolated.reset_empty_graph()
        isolated.outputs["input"] = self._source.data
        isolated.output_states["input"] = self._source.state
        self._context = replace(
            capture_proposal_context(isolated, "input"),
            planning_preferences=PlanningPreferences("use_best_judgment"),
        )
        self._candidates: list[InvestigationCandidate] = []
        self._probes: list[DiagnosticProbe] = []
        self._probe_snapshot_bytes = 0
        self._probe_records: dict[str, dict] = {}
        self._experiments = ExperimentLedger(max_experiments=self.limits.max_attempts)
        self._require_experiment_plan = require_experiment_plan
        self._review_before_finish = review_before_finish
        self._pending_finish_review: bytes | None = None
        self._finish_review_context_status = None
        self._continue_requests = 0
        self._latest_finish_feedback: bytes | None = None
        self._history: list[bytes] = []
        self._images: tuple[ImagePreview, ...] = ()
        self._image_bytes = self.actions_used = self.model_calls_used = 0
        self._sent_previews: set[str] = set()
        self._preview_owners: dict[int, frozenset[str]] = {}
        self._sent_preview_owners: dict[str, frozenset[str]] = {}
        self._sent_numeric: set[tuple[str, str]] = set()
        self._issued_numeric: set[tuple[str, str]] = set()
        self._numeric_owners: dict[tuple[str, str], frozenset[str]] = {}
        self._numeric_payloads: dict[tuple[str, str], bytes] = {}
        self._decision_numeric: dict[str, tuple[str, str]] = {}
        self._decision_contracts: dict[str, bytes] = {}
        self._recent_contract_ids: tuple[str, ...] = ()
        self._notebook = InvestigationNotebook(self.session_id)
        self._ledger_records: dict[str, dict] = {}
        self._candidate_records: dict[str, dict] = {}
        self._latest_action_record: dict | None = None
        self._last_transmission: tuple[bytes, tuple, tuple] | None = None
        self._inspection_cache: dict[str, bytes] = {}
        self._view_keys: dict[str, dict] = {}
        self._recent_view_ids: tuple[str, ...] = ()
        self._automatic_preview_omissions: list[dict] = []
        self._pinned_view_ids: tuple[str, ...] = ()
        self._preview_cache = NotebookPreviewCache(
            self.session_id, image_limit_bytes=self.limits.max_image_bytes
        )
        self._inspection_reused = False
        self._calculation_repairs = 0
        self.retained_candidate_id: str | None = None
        self.finished = False
        self.stop_reason = ""
        self._guard()
        self._notebook.append(
            "source_evidence",
            self._original.source_summary | {"sha256": self._source.fingerprint},
            owner={"alias": "input"},
        )
        self._notebook.append("node_index", self._node_index(safe_only=True))
        if self.include_input_images:
            indices = (
                sorted(
                    {0, self._source.data.shape[0] // 2, self._source.data.shape[0] - 1}
                )
                if state.axis_order == "ZYX"
                else [0]
            )
            self._images = self._render_views(
                self._source,
                "input",
                True,
                [
                    {
                        "axis": "Z" if state.axis_order == "ZYX" else "YX",
                        "indices": indices,
                        "montage": True,
                    }
                ],
            )
        self._initial_source_previews = self._images

    @classmethod
    def from_completed(cls, pipeline, bundle, dialogue, **kwargs):
        """Start from GUI-captured accepted evidence, with no baseline rerun.

        ``pipeline`` is the normal detached graph/source capture. The caller must
        recheck ``bundle.assert_current(live_pipeline)`` and owned Run authority
        on the composition thread before/after every worker step.
        """
        if type(bundle) is not CompletedCandidateBundle:
            raise InvestigationError(
                "Capture a completed candidate on its composition thread first."
            )
        if kwargs.get("include_output_images") is not True:
            raise InvestigationError(
                "Accepted result sharing requires explicit output consent."
            )
        if (
            _graph_fingerprint(GraphSnapshot.from_pipeline(pipeline))
            != bundle.snapshot.baseline_fingerprint
        ):
            raise InvestigationError(
                "The detached accepted workflow differs from its captured graph."
            )
        engine = cls(pipeline, bundle.snapshot.source_id, dialogue, **kwargs)
        if (
            engine._source.fingerprint != bundle.snapshot._source.fingerprint
            or engine._source.state_fingerprint
            != bundle.snapshot._source.state_fingerprint
        ):
            raise InvestigationError(
                "The detached accepted source differs from its captured source."
            )
        recipe = bundle.recipe
        recipe["context_id"] = engine.proposal_context_id
        prepared = validate_proposal(recipe, engine._context)
        assert_refinement_graph_safe(prepared.candidate, "input")
        engine._accepted_bundle = replace(bundle, _live_snapshot=None)
        engine._seed_count = 1
        engine._preservation = bundle.preservation_constraint
        engine._protected_role = bundle.protected_role
        bindings = tuple(
            (output.role, replace(output.binding, live=None))
            for output in bundle._role_outputs
        )
        operations = {
            node.id: node.operation_id for node in bundle.snapshot.baseline.nodes
        }
        evidence = bundle.public_binding() | {
            "status": "completed",
            "outputs": [
                {
                    "role": role,
                    "output_alias": f"candidate0/{role}",
                    "sha256": binding.fingerprint,
                    "diagnostics": engine._summary(
                        binding, operation_id=operations[binding.node_id]
                    ),
                }
                for role, binding in bindings
            ],
            "execution": {
                "already_completed": True,
                "new_calculation": False,
                "run_id": bundle.snapshot.run_id,
            },
        }
        engine._candidates.append(
            InvestigationCandidate(
                "candidate0",
                bundle.snapshot.run_id,
                "completed",
                bundle.snapshot.baseline,
                _json(recipe),
                _json(evidence),
                bindings,
                engine._original.source_fingerprint,
                _json(bundle.output_roles),
                bundle._key_to_node,
            )
        )
        engine.retained_candidate_id = "candidate0"
        engine._remember_candidate_context(engine._candidates[0])
        engine._sync_notebook_records()
        engine._guard()
        engine._add_protected_preview()
        return engine

    @property
    def attempted_trials(self):
        return len(self._candidates) - self._seed_count

    @property
    def attempted_probes(self):
        return len(self._probes)

    @property
    def diagnostic_probes(self):
        """Detached public records, never an alternative adoption inventory."""
        return tuple(probe.evidence for probe in self._probes)

    @property
    def warm_start_binding(self):
        return (
            self._accepted_bundle.public_binding()
            if self._accepted_bundle is not None
            else None
        )

    def _add_protected_preview(self):
        if self._preservation is None:
            return []
        self._images, omissions = self._preservation.with_previews(
            self._images, include_output_images=self.include_output_images
        )
        for preview in self._images:
            if preview.label.startswith("Protected prior foreground · "):
                self._preview_owners[id(preview)] = frozenset({"candidate0"})
        return omissions

    @property
    def candidates(self):
        return tuple(self._candidates)

    @property
    def selected_candidate(self):
        if self.retained_candidate_id is None:
            return None
        return self._candidate(self.retained_candidate_id)

    @property
    def experiment_ledger(self):
        """Detached read-only reasoning receipts, including terminal sessions."""
        return self._experiments.snapshots()

    @property
    def require_experiment_plan(self):
        return self._require_experiment_plan

    @property
    def review_before_finish(self):
        return self._review_before_finish

    def _remaining_review_budgets(self):
        return {
            "attempts_remaining": self.limits.max_attempts - self.attempted_trials,
            "actions_remaining": self.limits.max_actions - self.actions_used,
            "model_calls_remaining": self.limits.max_model_calls
            - self.model_calls_used,
            "seconds_remaining": self._seconds_remaining(),
            "image_bytes_remaining": self.limits.max_image_bytes - self._image_bytes,
        }

    def _review_reservation(self):
        if not self._review_before_finish:
            return None
        return review_reservation(
            self.limits,
            self._continue_requests,
            planned=self._require_experiment_plan,
            accepted_baseline=self._accepted_bundle is not None,
        )

    def _finish_continuation_status(self, *, check_context=True):
        remaining = self._remaining_review_budgets()
        if remaining["attempts_remaining"] <= 0:
            return False, "The original trial-attempt limit is reached."
        # A new linked test needs its plan, trial, assessment and another Finish.
        # The existing closing phase also reserves recovery room; user approval
        # cannot spend that reservation on a trial that cannot be assessed.
        logical_steps = 4 if self._require_experiment_plan else 3
        if remaining["actions_remaining"] < logical_steps + 1:
            return (
                False,
                "The original action budget leaves no complete trial/assessment room.",
            )
        if remaining["model_calls_remaining"] < 2 * logical_steps + 2:
            return (
                False,
                "The original model-call budget leaves no bounded "
                "assessment/Finish room.",
            )
        if self._closing_phase(transport_calls=2, include_review_reserve=False)[
            "active"
        ]:
            return (
                False,
                "The original closing-phase reservation prevents another trial.",
            )
        if (
            sum(len(image.png_bytes) for image in self._images)
            > remaining["image_bytes_remaining"]
        ):
            return (
                False,
                "The original transmitted-image budget cannot supply the next board.",
            )
        if check_context:
            key = (
                self._pending_finish_review,
                WORKING_REQUEST_TARGET_BYTES,
                MAX_REQUEST_BYTES,
            )
            cached = self._finish_review_context_status
            if self._pending_finish_review is not None and cached and cached[0] == key:
                context_allowed, context_reason = cached[1:]
            else:
                # Pending sessions cannot perform/send/pin/change evidence. Cache
                # only this quiescent context-fit result, not dynamic budgets or
                # source/candidate freshness. Continue checks the new prospective
                # user-decision/feedback request again after clearing the pause.
                context_allowed, context_reason = True, ""
                try:
                    request = self._public_request(
                        transport_calls=2,
                        finish_review=True,
                        include_review_reserve=False,
                    )
                    if request["closing_phase"]["active"]:
                        context_allowed, context_reason = (
                            False,
                            "The original context/closing reservation "
                            "prevents another trial.",
                        )
                except InvestigationStopped:
                    if self.stop_reason:
                        raise
                    context_allowed = False
                except (InvestigationError, NotebookError):
                    context_allowed = False
                if not context_allowed and not context_reason:
                    context_reason = (
                        "The unchanged context/notebook limits prevent another trial."
                    )
                if self._pending_finish_review is not None:
                    self._finish_review_context_status = (
                        key,
                        context_allowed,
                        context_reason,
                    )
            if not context_allowed:
                return False, context_reason
        return (
            True,
            "Another bounded test may be requested within the same remaining limits.",
        )

    @property
    def pending_finish_review(self):
        """Detached local nomination, not acceptance or scientific validation."""
        if self._pending_finish_review is None:
            return None
        self.assert_finish_review_current()
        result = json.loads(self._pending_finish_review)
        allowed, reason = self._finish_continuation_status()
        return {
            "candidate_id": result["retained_candidate_id"],
            "summary": result["model_interpretation"],
            "limitations": result["limitations"],
            "continuation_allowed": allowed,
            "continuation_reason": reason,
            "remaining_budgets": self._remaining_review_budgets(),
        }

    def assert_finish_review_current(self):
        """Check disposition freshness, without granting more elapsed work time."""
        if not self._review_before_finish or self._pending_finish_review is None:
            raise InvestigationError("No provisional Finish is awaiting user review.")
        self._assert_finish_review_authority()
        result = json.loads(self._pending_finish_review)
        candidate = self._candidate(result["retained_candidate_id"])
        if (
            result["recipe"] != candidate.recipe
            or result["evidence"] != candidate.evidence
        ):
            self.stop_reason = "The provisional completed candidate changed."
            raise InvestigationStopped(self.stop_reason)
        for _, binding in candidate._outputs:
            if (
                _data_hash(binding.data) != binding.fingerprint
                or _state_hash(binding.state) != binding.state_fingerprint
            ):
                self.stop_reason = "The provisional completed output evidence changed."
                raise InvestigationStopped(self.stop_reason)

    def _finish_decision(self, decision, feedback=""):
        result = json.loads(self._pending_finish_review)
        payload = {
            "decision": decision,
            "candidate_id": result["retained_candidate_id"],
            "feedback": feedback,
            "meaning": (
                "Explicit user disposition only, not scientific evidence "
                "or an accuracy certificate."
            ),
        }
        if self._continuation_enabled:
            payload["continuation_timing"] = {
                "active_seconds_used": self._active_seconds(),
                "paused_seconds": (
                    0
                    if self._pause_started is None
                    else self._now() - self._pause_started
                ),
                "time_policy": TIME_POLICY,
            }
        notebook = deepcopy(self._notebook)
        descriptor = notebook.append(
            "user_finish_decision",
            payload,
            owner={"candidate_id": payload["candidate_id"]},
        )
        return notebook, self._record_ref(descriptor), payload

    def continue_finish_review(self, feedback=""):
        with self._owner_lease():
            if self._continuation_enabled:
                now = self._assert_continuation_authority()
                if self._seconds_remaining() <= 0:
                    raise InvestigationError(
                        "Grant more active-time allowance before Keep improving."
                    )
            result = self._continue_finish_review(feedback)
            if self._continuation_enabled and self._pause_started is not None:
                self._paused_seconds += now - self._pause_started
                self._pause_started = None
                self._pause_reason = self._pause_kind = ""
                self.deadline = (
                    self._started + self.limits.max_seconds + self._paused_seconds
                )
                self._control_revision += 1
                self._settle_requested_pause()
            return result

    def _continue_finish_review(self, feedback=""):
        """Continue the same live session; no counter, deadline or evidence reset."""
        self.assert_finish_review_current()
        if type(feedback) is not str or len(feedback) > 1000:
            raise InvestigationError(
                "Continuation feedback must be at most 1000 characters."
            )
        if not feedback.strip():
            feedback = (
                "Continue considering the remaining goal requirements "
                "within the same limits."
            )
        _no_paths(feedback)
        allowed, reason = self._finish_continuation_status()
        if not allowed:
            raise InvestigationError(reason)
        if self._continue_requests >= self.limits.max_actions:
            raise InvestigationError(
                "The finite user-continuation record limit is reached."
            )
        notebook, reference, decision = self._finish_decision("continue", feedback)
        previous = (
            self._notebook,
            self._latest_action_record,
            self._pending_finish_review,
            self._continue_requests,
            self._latest_finish_feedback,
            self._finish_review_context_status,
        )
        self._notebook, self._latest_action_record = notebook, reference
        self._pending_finish_review = None
        self._finish_review_context_status = None
        self._continue_requests += 1
        self._latest_finish_feedback = _json(
            {"candidate_id": decision["candidate_id"], "feedback": feedback}
        )
        try:
            allowed, reason = self._finish_continuation_status()
            if not allowed:
                raise InvestigationError(reason)
        except Exception:
            (
                self._notebook,
                self._latest_action_record,
                self._pending_finish_review,
                self._continue_requests,
                self._latest_finish_feedback,
                self._finish_review_context_status,
            ) = previous
            raise
        self._last_transmission = None
        return deepcopy(decision)

    def accept_finish_review(self):
        with self._owner_lease():
            return self._accept_finish_review()

    def _accept_finish_review(self):
        """Accept the exact completed nomination, never Add/Run or certify accuracy."""
        self.assert_finish_review_current()
        self._require_assessed_experiments()
        result = json.loads(self._pending_finish_review)
        notebook, reference, _ = self._finish_decision("accept")
        self._notebook, self._latest_action_record = notebook, reference
        self.retained_candidate_id = result["retained_candidate_id"]
        self._pending_finish_review = None
        self._finish_review_context_status = None
        self._last_transmission = None
        self.finished = True
        return result

    @property
    def proposal_context_id(self):
        return self._context.context_id

    def preview_images(self):
        if self.stop_reason or self._permission_binding != (
            self.include_input_images,
            self.include_output_images,
        ):
            return ()
        return self._images

    def export_notebook(self):
        """Pure complete journal export, also available after a terminal stop.

        Exporting, listing, retrieving or pinning records grants no scientific
        or transmission authority and performs no new calculations.
        """
        return self._notebook.export()

    @staticmethod
    def _record_ref(descriptor):
        return {key: descriptor[key] for key in ("record_id", "sha256")}

    def _sync_notebook_records(self):
        for probe in self._probes:
            previous = self._probe_records.get(probe.probe_id)
            if previous is None or self._notebook.lookup(**previous) != probe.evidence:
                self._probe_records[probe.probe_id] = self._record_ref(
                    self._notebook.append(
                        "diagnostic_probe",
                        probe.evidence,
                        owner={"probe_id": probe.probe_id},
                    )
                )
        for entry in self.experiment_ledger:
            identifier = entry["experiment_id"]
            previous = self._ledger_records.get(identifier)
            if previous is None or self._notebook.lookup(**previous) != entry:
                self._ledger_records[identifier] = self._record_ref(
                    self._notebook.append(
                        "experiment_ledger", entry, owner={"experiment_id": identifier}
                    )
                )
        for candidate in self._candidates:
            previous = self._candidate_records.get(candidate.candidate_id)
            if (
                previous is None
                or self._notebook.lookup(**previous) != candidate.evidence
            ):
                self._candidate_records[candidate.candidate_id] = self._record_ref(
                    self._notebook.append(
                        "candidate_evidence",
                        candidate.evidence,
                        owner={"candidate_id": candidate.candidate_id},
                    )
                )

    def _pending_assessments(self):
        completed = {
            candidate.candidate_id
            for candidate in self._candidates
            if candidate.status in {"completed", "rejected"}
        }
        return [
            entry["experiment_id"]
            for entry in self.experiment_ledger
            if entry["candidate_id"] in completed and entry["assessment"] is None
        ]

    def _closing_phase(
        self, *, context_pressure=False, transport_calls=2, include_review_reserve=True
    ):
        pending = self._pending_assessments()
        reservation = self._review_reservation()
        held = bool(
            include_review_reserve
            and reservation
            and reservation["reserved_trial_attempts"]
        )
        future_actions = reservation["reserved_actions"] if held else 0
        future_calls = reservation["reserved_model_calls"] if held else 0
        # Every pending assessment and FINISH can each need the existing two
        # bounded format attempts. Reserve two more calls/actions for recovery,
        # without increasing either hard limit. Outgoing schemas anticipate the
        # transport's attempts; execution checks only the actual remaining room.
        reserved_actions = len(pending) + 3 + future_actions
        required_calls = 2 * (len(pending) + 1)
        reserved_calls = required_calls + 2 + future_calls
        actions = self.limits.max_actions - self.actions_used
        calls = self.limits.max_model_calls - self.model_calls_used - transport_calls
        seconds = self._seconds_remaining()
        active = bool(
            context_pressure
            or held
            and self.attempted_trials >= reservation["initial_trial_limit"]
            or actions <= reserved_actions
            or calls <= reserved_calls
            or seconds <= 180
        )
        allowed = set(_ACTION_ARGUMENTS)
        reason = "Exploration remains within its original finite limits."
        if active:
            allowed = {"question", "finish"}
            if pending:
                allowed.add("assess_experiment")
                allowed.discard("finish")
            if (
                actions > len(pending) + 1 + future_actions
                and calls >= required_calls + 2 + future_calls
            ):
                allowed.update(_CLOSING_READ_ACTIONS)
            reason = (
                "Assessment and explicit Finish room is reserved; no further trials, "
                "plans, discovery or free-form observations. Missing checks may remain "
                "honestly unresolved. A Finish is a model preference, not accuracy."
            )
            if held:
                reason = (
                    "One original trial and 7 actions/14 calls are reserved for "
                    "the first Keep improving decision. Complete genuine pending "
                    "assessments and propose Finish for review; no automatic "
                    "nomination or scientific verdict follows. Closing reads "
                    "cannot spend this future allowance. All hard limits remain."
                )
        return {
            "active": active,
            "reason": reason,
            "reserved_actions": reserved_actions,
            "reserved_model_calls": reserved_calls,
            "pending_experiment_ids": pending,
            "allowed_actions": sorted(allowed),
            "request_target_bytes": WORKING_REQUEST_TARGET_BYTES,
            "hard_request_bytes": MAX_REQUEST_BYTES,
            "ledger_scope": (
                "Latest full entry only; experiment_index preserves all checks/status; "
                "and exact full entries remain retrievable from this session notebook. "
                "Finish checks the unchanged complete authoritative internal ledger."
            ),
        }

    def _do_read_notebook(self, record_id, sha256, page):
        return self._notebook.read_page(record_id, sha256, page)

    def _do_pin_notebook(self, refs):
        self._notebook.set_pins(refs)
        return {
            "pinned_record_refs": self._notebook.pinned_record_refs,
            "meaning": (
                "Exact working-context records only; no evidence authority is issued."
            ),
        }

    def _deduplicated_board(self, previews):
        result, seen = [], set()
        for preview in previews:
            identity = _json(preview.manifest())
            if identity not in seen:
                result.append(preview)
                seen.add(identity)
        result = self._preview_cache.validate_board(tuple(result))
        if self._preservation is not None:
            result, omissions = self._preservation.with_previews(
                result, include_output_images=self.include_output_images
            )
            if omissions:
                raise InvestigationError(
                    "The visual board leaves no room for the protected prior; "
                    "unpin views or request fewer views. No pinned image was evicted."
                )
        return self._preview_cache.validate_board(tuple(result))

    def _install_inspection_board(self, previews):
        pinned = tuple(
            preview
            for identifier in self._pinned_view_ids
            for preview in (
                self._preview_cache.lookup(self._view_keys[identifier]) or ()
            )
        )
        board = self._deduplicated_board((*pinned, *previews))
        omissions = []
        # Reuse only exact already generated views, never render or infer which
        # structure is wanted. Explicit pins and this requested inspection are
        # mandatory; automatic source/retained/latest context is lower priority.
        selected = {candidate.candidate_id for candidate in self._decision_candidates()}
        source_ids = [
            identifier
            for identifier in self._recent_view_ids
            if self._view_keys[identifier]["alias"] == "input"
        ]
        automatic = (
            [
                (
                    source_ids[-1],
                    self._preview_cache.lookup(self._view_keys[source_ids[-1]]) or (),
                )
            ]
            if source_ids
            else [("initial_source", self._initial_source_previews)]
        )
        # Keep complete matched PNG pairs ahead of unmatched automatic views.
        # A cached inspection may contain several views: one exact native-view
        # pair can fit where both complete inspection tuples cannot. Pairing
        # never renders, measures, resamples or grants transmission authority.
        source_identifier, source_images = automatic[0]
        if source_images:
            try:
                board = self._deduplicated_board((*board, *source_images))
            except (InvestigationError, NotebookError, ValueError):
                omissions.append(
                    {
                        "kind": "previews",
                        "identity": source_identifier,
                        "reason": (
                            "Exact cached automatic source view excluded by the "
                            "unchanged board cap; explicit/requested views kept."
                        ),
                    }
                )
        pairs = self._cached_comparison_pairs()
        paired_manifests = {
            _json(member["preview"].manifest())
            for pair in pairs
            if len(pair["members"]) == 2
            for member in pair["members"]
        }
        for pair in pairs:
            if len(pair["members"]) != 2:
                continue
            try:
                board = self._deduplicated_board(
                    (*board, *(member["preview"] for member in pair["members"]))
                )
            except (InvestigationError, NotebookError, ValueError):
                omissions.append(
                    {
                        "kind": "preview_comparison",
                        "identity": pair["selection_sha256"],
                        "reason": (
                            "Whole exact cached comparison pair excluded by the "
                            "unchanged seven-image/2 MiB board; explicit/requested "
                            "views kept. No partial pair was automatically added."
                        ),
                    }
                )
        automatic = []
        grouped = set()
        for identifier in reversed(self._recent_view_ids):
            key = self._view_keys[identifier]
            alias = key["alias"]
            if alias == "input" or alias.split("/", 1)[0] not in selected:
                continue
            role = alias.split("/", 1)[1]
            selection = _json(
                {
                    "role": role,
                    "grid": key["grid"],
                    "views": key["views"],
                    "roi": key["roi"],
                }
            )
            if selection in grouped:
                continue
            grouped.add(selection)
            for candidate in self._decision_candidates():
                counterpart = next(
                    (
                        other
                        for other in reversed(self._recent_view_ids)
                        if self._view_keys[other]["alias"]
                        == f"{candidate.candidate_id}/{role}"
                        and _json(
                            {
                                "role": role,
                                "grid": self._view_keys[other]["grid"],
                                "views": self._view_keys[other]["views"],
                                "roi": self._view_keys[other]["roi"],
                            }
                        )
                        == selection
                    ),
                    None,
                )
                if counterpart is not None:
                    automatic.append(
                        (
                            counterpart,
                            self._preview_cache.lookup(self._view_keys[counterpart])
                            or (),
                        )
                    )
        # One exact view per (source/candidate, semantic role, native selection).
        # The most recently requested role/selection is considered first, with
        # its same-selection counterpart in the retained candidate next.
        seen = set()
        for identifier, images in automatic:
            # A complete cached pair is an atomic automatic selection even if
            # it did not fit. Do not sneak one omitted member back through the
            # unmatched-view fallback. Pins/requested members above are kept.
            images = tuple(
                image
                for image in images
                if _json(image.manifest()) not in paired_manifests
            )
            if not images:
                continue
            key = self._view_keys.get(identifier)
            identity = (
                _json({name: key[name] for name in ("alias", "views", "roi")})
                if key is not None
                else b"initial_source"
            )
            if identity in seen:
                continue
            seen.add(identity)
            try:
                board = self._deduplicated_board((*board, *images))
            except (InvestigationError, NotebookError, ValueError):
                omissions.append(
                    {
                        "kind": "previews",
                        "identity": identifier,
                        "reason": (
                            "Exact cached automatic view excluded by the unchanged "
                            "seven-image/2 MiB board; explicit/requested views kept."
                        ),
                    }
                )
        self._images = board
        self._automatic_preview_omissions = omissions[:32]

    def _comparison_preview_owners(self):
        """Choose evidence owners, not a scientific winner or preferred role."""
        candidates = self._decision_candidates()
        if not candidates:
            return ()
        current = candidates[0].candidate_id
        pairs = []
        if len(candidates) == 2:
            pairs.append((candidates[1].candidate_id, current))
        elif (
            self._accepted_bundle is not None
            and self._candidates[0].candidate_id != current
        ):
            # Retaining/finishing the current result must not erase its exact
            # accepted-before/current-after comparison owners.
            pairs.append((self._candidates[0].candidate_id, current))
        eligible = {
            candidate.candidate_id
            for candidate in self._candidates
            if candidate.status in {"completed", "rejected"}
        }
        for key in reversed(tuple(self._decision_numeric.values())):
            payload = json.loads(self._numeric_payloads[key])
            owners = (payload.get("left"), payload.get("right"))
            if (
                "roles" in payload
                and current in owners
                and len(set(owners)) == 2
                and all(owner in eligible for owner in owners)
            ):
                if owners not in pairs and owners[::-1] not in pairs:
                    pairs.append(owners)
                break
        return tuple(pairs)

    def _cached_comparison_pairs(self):
        return [
            pair
            for owners in self._comparison_preview_owners()
            for pair in self._cached_comparison_pairs_for(owners)
        ]

    def _cached_comparison_pairs_for(self, owners):
        """Exact cached per-PNG native selections; no scientific array access.

        A legacy/unknown render policy or a tuple whose member count cannot be
        mapped exactly to its explicit views is not a matching image. Distinct
        output fingerprints are expected; source revision, role, native grid,
        ROI, view and display policy must match. Full manifests, not PNG hashes
        alone, establish whether those exact members occupy the current board.
        """
        if len(owners) != 2 or not self.include_output_images:
            return []
        groups = {}
        for identifier in reversed(self._recent_view_ids):
            key = self._view_keys[identifier]
            alias = key["alias"]
            owner, separator, role = alias.partition("/")
            policy = key.get("display_policy")
            if (
                not separator
                or owner not in owners
                or type(policy) is not dict
                or policy.get("renderer") != "native-image-v1"
                or key.get("session_id") != self.session_id
                or key.get("source_sha256") != self._source.fingerprint
                or key.get("source_state_sha256") != self._source.state_fingerprint
                or key.get("permissions") != list(self._permission_binding)
            ):
                continue
            images = self._preview_cache.lookup(key)
            if not images:
                continue
            selections = []
            for view in key["views"]:
                if view.get("montage", False):
                    selections.append(view)
                else:
                    selections.extend(
                        view | {"indices": [index]} for index in view["indices"]
                    )
            if len(selections) != len(images):
                continue
            for view, preview in zip(selections, images, strict=True):
                selection = {
                    "role": role,
                    "grid": key["grid"],
                    "roi": key["roi"],
                    "view": view,
                    "display_policy": policy
                    | {
                        # The existing exact render description binds actual
                        # display extrema, native strides/crop and overlay.
                        # Comparing only a renderer name would falsely match
                        # independently stretched scalar intensity images.
                        "render_description_sha256": hashlib.sha256(
                            preview.description.encode("utf-8")
                        ).hexdigest(),
                    },
                    "source_sha256": key["source_sha256"],
                    "source_state_sha256": key["source_state_sha256"],
                }
                digest = hashlib.sha256(_json(selection)).hexdigest()
                group = groups.setdefault(
                    digest,
                    {
                        "candidate_ids": list(owners),
                        "selection_sha256": digest,
                        "selection": selection,
                        "by_owner": {},
                    },
                )
                group["by_owner"].setdefault(
                    owner,
                    {"candidate_id": owner, "view_id": identifier, "preview": preview},
                )
        return [
            {key: value for key, value in group.items() if key != "by_owner"}
            | {
                "members": [
                    group["by_owner"][owner]
                    for owner in owners
                    if owner in group["by_owner"]
                ]
            }
            for group in groups.values()
        ]

    def _comparison_preview_context(self):
        """Disclose actual board membership; descriptors never supply pixels."""
        supplied = {_json(preview.manifest()) for preview in self._images}
        result = []
        pairs = self._cached_comparison_pairs()

        def wholly_supplied(pair):
            return len(pair["members"]) == 2 and all(
                _json(member["preview"].manifest()) in supplied
                for member in pair["members"]
            )

        # Current actual-pixel groups must not vanish behind descriptors for
        # omitted/unavailable selections when the metadata allowance is full.
        pairs.sort(
            key=lambda pair: (not wholly_supplied(pair), len(pair["members"]) != 2)
        )
        for pair in pairs:
            members = pair["members"]
            complete = len(members) == 2
            on_board = wholly_supplied(pair)
            row = {
                "candidate_ids": pair["candidate_ids"],
                "role": pair["selection"]["role"],
                "selection": {
                    key: value
                    for key, value in pair["selection"].items()
                    if key != "role"
                },
                "views": [
                    {
                        "candidate_id": member["candidate_id"],
                        "view_id": member["view_id"],
                        "preview_sha256": member["preview"].manifest()["sha256"],
                    }
                    for member in members
                ],
                "status": (
                    "supplied" if on_board else "omitted" if complete else "unavailable"
                ),
                "reason": (
                    "Both exact cached PNG manifests are on the current board; "
                    "only an actual send supplies their pixels."
                    if on_board
                    else "The exact cached pair is not wholly on the current "
                    "board; its descriptor supplies no pixels."
                    if complete
                    else "No exact cached counterpart exists for this selection; "
                    "no mismatched or unseen image is substituted."
                ),
            }
            # Reserve a small omission notice before accepting the whole row;
            # never remove an accepted supplied pair merely to append prose.
            if len(_json([*result, row])) > 8 * 1024 - 256:
                break
            result.append(row)
            if len(result) >= 4:
                break
        if result and len(result) < len(pairs):
            result[-1]["reason"] += (
                f" {len(pairs) - len(result)} additional cached selection groups "
                "are excluded by the four-group/8 KiB metadata bound."
            )
        return result

    def _do_pin_previews(self, view_ids):
        if (
            type(view_ids) is not list
            or len(view_ids) > MAX_IMAGE_PREVIEWS
            or any(
                type(value) is not str or value not in self._view_keys
                for value in view_ids
            )
            or len(set(view_ids)) != len(view_ids)
        ):
            raise InvestigationError(
                "Choose zero to seven exact existing view_id values."
            )
        previews = tuple(
            preview
            for identifier in view_ids
            for preview in (
                self._preview_cache.lookup(self._view_keys[identifier]) or ()
            )
        )
        board = self._deduplicated_board(previews)
        self._pinned_view_ids, self._images = tuple(view_ids), board
        self._install_inspection_board(())
        return {
            "pinned_view_ids": view_ids,
            "image_previews": [preview.manifest() for preview in self._images],
            "meaning": (
                "Existing actual PNGs only; every send uses the original image budget."
            ),
        }

    @property
    def continuation_enabled(self):
        return self._continuation_enabled

    @property
    def paused(self):
        """True only after the current tool step has safely settled."""
        return (
            self._pause_started is not None
            and not self.finished
            and not self.stop_reason
        )

    @property
    def pending_question(self):
        return self._pending_question

    def _now(self):
        with self._clock_lock:
            now = self._clock()
            if (
                type(now) not in {int, float}
                or not math.isfinite(now)
                or now < self._last_clock
            ):
                self.stop_reason = "The finite monotonic investigation clock changed."
                raise InvestigationStopped(self.stop_reason)
            self._last_clock = now
            return now

    def _active_seconds(self, now=None):
        now = self._now() if now is None else now
        end = now if self._pause_started is None else self._pause_started
        return max(0.0, end - self._started - self._paused_seconds)

    def _seconds_remaining(self):
        if self._continuation_enabled:
            return max(0.0, self.limits.max_seconds - self._active_seconds())
        return max(0, self.deadline - self._clock())

    def _continuation_context(self):
        return {
            "version": 1,
            "allowance": allowance_dict(self.limits),
            "hard_limits": dict(HARD_ALLOWANCE),
            "grants_used": self._allowance_grants,
            "time_policy": TIME_POLICY,
        }

    @property
    def continuation_state(self):
        """Detached owner UI state; this does not send or grant model evidence."""
        now = self._now()
        state = (
            "stopped"
            if self.stop_reason
            else "finished"
            if self.finished
            else "finish_review"
            if self._pending_finish_review is not None
            else "question"
            if self._pending_question is not None
            else "paused"
            if self.paused
            else "pausing"
            if self._pause_requested.is_set()
            else "running"
        )
        return self._continuation_context() | {
            "enabled": self._continuation_enabled,
            "state": state,
            "reason": self.stop_reason or self._pause_reason,
            "question": self._pending_question,
            "paused_seconds": self._paused_seconds
            + (0 if self._pause_started is None else now - self._pause_started),
            "active_seconds_used": self._active_seconds(now),
            "remaining_budgets": self._remaining_review_budgets(),
        }

    @contextmanager
    def _owner_lease(self):
        if not self._agent_tool_lock.acquire(blocking=False):
            raise InvestigationError(
                "The shared investigation session is busy; "
                "pause after its current step."
            )
        try:
            yield
        finally:
            self._agent_tool_lock.release()

    def _assert_continuation_authority(self):
        if not self._continuation_enabled:
            raise InvestigationError("Same-session continuation was not enabled.")
        self._assert_approved_allowance()
        if self._permission_binding != (
            self.include_input_images,
            self.include_output_images,
        ):
            self.stop_reason = "Investigation image/evidence permissions changed."
        now = self._now()
        if (
            self.finished
            or self.stop_reason
            or self._cancelled is not None
            and self._cancelled()
            or self._active_seconds(now) >= HARD_ALLOWANCE["max_seconds"]
        ):
            self.stop_reason = self.stop_reason or (
                "Investigation finished, cancelled or reached "
                "its hard active-time ceiling."
            )
            raise InvestigationStopped(self.stop_reason)
        self._assert_current_authority()
        return now

    def _assert_approved_allowance(self):
        if (
            type(self.limits) is not InvestigationLimits
            or _json(allowance_dict(self.limits)) != self._allowance_binding
        ):
            self.stop_reason = (
                "The owner-approved investigation allowance changed outside its grant."
            )
            raise InvestigationStopped(self.stop_reason)

    def _usage(self):
        return {
            "attempted_trials": self.attempted_trials,
            "actions_used": self.actions_used,
            "model_calls_used": self.model_calls_used,
            "image_bytes_used": self._image_bytes,
            "attempted_probes": self.attempted_probes,
            "probe_snapshot_bytes_used": self._probe_snapshot_bytes,
            "active_seconds_used": self._active_seconds(),
        }

    def _pause_locked(self, reason, kind, question=None):
        if self._pause_started is not None or self.finished or self.stop_reason:
            return
        now = self._now()
        try:
            self._notebook.append(
                "session_pause",
                {
                    "reason": reason,
                    "kind": kind,
                    "question": question,
                    "usage": self._usage(),
                    "meaning": "Owner-resumable checkpoint; "
                    "no counters or permissions reset.",
                },
            )
        except NotebookError as exc:
            self.stop_reason = (
                "The finite notebook cannot record a safe continuation checkpoint."
            )
            raise InvestigationStopped(self.stop_reason) from exc
        self._pause_started = now
        self._pause_kind, self._pause_reason = kind, reason
        self._pending_question = question
        self._control_revision += 1
        self._last_transmission = None

    def _settle_requested_pause(self):
        if self._pause_requested.is_set() and not self._active_action:
            with self._pause_request_lock:
                request = self._requested_pause
                self._requested_pause = None
                self._pause_requested.clear()
            if request is not None:
                self._pause_locked(*request)

    def request_pause(self, reason="User requested a pause."):
        """Request a nonblocking pause after the current complete, journalled step.

        Unlike Stop, this does not cancel an ordinary executor midway through
        cleanup or discard its completed evidence. A disconnected MCP controller
        may call it concurrently without waiting for the shared tool lease.
        """
        if not self._continuation_enabled:
            raise InvestigationError("Same-session continuation was not enabled.")
        reason = _text(reason, "Pause reason", 500)
        _no_paths(reason)
        if self.finished or self.stop_reason:
            return False
        with self._pause_request_lock:
            if self._requested_pause is None:
                self._requested_pause = (reason, "owner", None)
            self._pause_request_generation += 1
            self._pause_requested.set()
        if self._agent_tool_lock.acquire(blocking=False):
            try:
                self._settle_requested_pause()
            finally:
                self._agent_tool_lock.release()
        return True

    pause = request_pause

    def _pause_allowance(self, reason, kind):
        if self._continuation_enabled:
            self._pause_locked(reason, kind)
            raise InvestigationPaused(reason)
        self.stop_reason = reason
        raise InvestigationStopped(reason)

    def _resume_locked(self, *, answer=None, limits=None, notebook=None):
        with self._pause_request_lock:
            pause_generation = self._pause_request_generation
        now = self._assert_continuation_authority()
        if self._active_action:
            raise InvestigationError("Wait for the current tool step to settle.")
        if self._pending_finish_review is not None:
            raise InvestigationError(
                "Use Keep improving or Accept for the pending Finish."
            )
        if not self.paused:
            raise InvestigationError(
                "No same-session checkpoint is awaiting continuation."
            )
        proposed = self.limits if limits is None else limits
        dialogue = self.dialogue
        if self._pending_question is not None:
            answer = _text(answer, "Answer", 2000)
            _no_paths(answer)
            dialogue = replace(
                dialogue,
                clarifications=(
                    *dialogue.clarifications,
                    (self._pending_question, answer),
                ),
            )
        elif answer is not None:
            raise InvestigationError(
                "An answer requires the exact pending assistant question."
            )
        if (
            proposed.max_actions <= self.actions_used
            or proposed.max_model_calls <= self.model_calls_used
            or proposed.max_seconds <= self._active_seconds(now)
            or proposed.max_image_bytes - self._image_bytes
            < sum(len(p.png_bytes) for p in self._images)
            or self._pause_kind == "trials"
            and proposed.max_attempts <= self.attempted_trials
        ):
            raise InvestigationError(
                "The approved allowance is exhausted; grant larger absolute "
                "totals within the hard ceilings."
            )
        duration = now - self._pause_started
        prospective = deepcopy(self._notebook) if notebook is None else notebook
        prospective.append(
            "session_resume",
            {
                "question": self._pending_question,
                "answer": answer,
                "paused_seconds": duration,
                "allowance": allowance_dict(proposed),
                "usage": self._usage(),
                "meaning": "Same source, candidates, notebook, permissions "
                "and consumed usage retained.",
            },
        )
        self._notebook = prospective
        self.dialogue = dialogue
        self._paused_seconds += duration
        self._pause_started = None
        self._pause_reason = self._pause_kind = ""
        self._pending_question = None
        with self._pause_request_lock:
            if self._pause_request_generation == pause_generation:
                self._requested_pause = None
                self._pause_requested.clear()
        self.deadline = self._started + proposed.max_seconds + self._paused_seconds
        self._control_revision += 1
        self._finish_review_context_status = self._last_transmission = None
        self._settle_requested_pause()

    def resume(self, answer=None):
        """Resume the same live session, optionally answering its exact Question."""
        with self._owner_lease():
            self._resume_locked(answer=answer)
            return self.continuation_state

    def grant_allowance(self, limits, *, resume=False, answer=None):
        """Approve nondecreasing ABSOLUTE totals; never replenish consumed usage.

        This is an owner/controller API, not a model tool. Only a quiescent
        checkpoint may be extended, and unsafe invalidations remain terminal.
        """
        with self._owner_lease():
            self._assert_continuation_authority()
            if type(limits) is not InvestigationLimits or type(resume) is not bool:
                raise InvestigationError(
                    "Use explicit finite InvestigationLimits and a resume choice."
                )
            # Frozen dataclasses are a convenience, not the admission boundary.
            # Revalidate even a caller-forged/mutated instance against constants.
            limits.__post_init__()
            if self._active_action or not (
                self.paused or self._pending_finish_review is not None
            ):
                raise InvestigationError(
                    "Pause after the current step before adjusting its allowance."
                )
            before, after = allowance_dict(self.limits), allowance_dict(limits)
            if any(after[k] < before[k] for k in before) or before == after:
                raise InvestigationError(
                    "A grant must increase absolute totals "
                    "without lowering any allowance."
                )
            if self._allowance_grants >= MAX_CONTINUATION_GRANTS:
                raise InvestigationError("The finite owner-grant count is reached.")
            if not resume and answer is not None:
                raise InvestigationError(
                    "An answer is applied only with an explicit resume."
                )
            prospective = deepcopy(self._notebook)
            prospective.append(
                "allowance_grant",
                {
                    "previous": before,
                    "approved": after,
                    "usage": self._usage(),
                    "grant_index": self._allowance_grants + 1,
                    "meaning": "Explicit owner approval of absolute totals; "
                    "no resource ceiling changes.",
                },
            )
            if resume:
                self._resume_locked(answer=answer, limits=limits, notebook=prospective)
            else:
                self._notebook = prospective
                self.deadline += limits.max_seconds - self.limits.max_seconds
            self.limits = limits
            self._allowance_binding = _json(allowance_dict(limits))
            self._experiments.max_experiments = limits.max_attempts
            self._preview_cache.image_limit_bytes = limits.max_image_bytes
            self._allowance_grants += 1
            self._control_revision += 1
            self._finish_review_context_status = self._last_transmission = None
            return self.continuation_state

    def cancel(self, *, user_decision=False):
        self.stop_reason = "Investigation cancelled."
        if self._pending_finish_review is not None:
            if user_decision is True:
                try:
                    notebook, reference, _ = self._finish_decision("stop")
                    self._notebook, self._latest_action_record = notebook, reference
                except NotebookError:
                    # Cancellation is never conditional on remaining journal room.
                    self.stop_reason += (
                        " The finite journal could not record the user disposition."
                    )
            self._pending_finish_review = None
            self._finish_review_context_status = None
            self._last_transmission = None

    def _cancel_check(self):
        with self._clock_lock:
            now = self._clock()
            if (
                type(now) not in {int, float}
                or not math.isfinite(now)
                or self._continuation_enabled
                and now < self._last_clock
            ):
                return True
            if self._continuation_enabled:
                self._last_clock = now
        time_reached = (
            self._active_seconds(now) >= HARD_ALLOWANCE["max_seconds"]
            if self._continuation_enabled
            else now >= self.deadline
        )
        return bool(
            self.stop_reason
            or time_reached
            or self._cancelled is not None
            and self._cancelled()
        )

    def _guard(self):
        if (
            self.include_input_images,
            self.include_output_images,
        ) != self._permission_binding:
            self.stop_reason = "Investigation image/evidence permissions changed."
        if self._continuation_enabled:
            self._assert_approved_allowance()
            self._now()
        if self.finished or self._cancel_check():
            self.stop_reason = self.stop_reason or (
                "Investigation finished."
                if self.finished
                else "Investigation cancelled or elapsed-time limit reached."
            )
            raise InvestigationStopped(self.stop_reason)
        self._assert_current_authority()
        if self._continuation_enabled and not self._active_action:
            self._settle_requested_pause()
            if self.paused:
                raise InvestigationPaused(self._pause_reason)
            if self._seconds_remaining() <= 0:
                self._pause_allowance(
                    "The approved active-time allowance is reached "
                    "after the completed step.",
                    "time",
                )

    def _assert_finish_review_authority(self):
        if not self._review_before_finish:
            raise InvestigationError(
                "Local Finish review was not enabled for this session."
            )
        if (
            self.include_input_images,
            self.include_output_images,
        ) != self._permission_binding:
            self.stop_reason = "Investigation image/evidence permissions changed."
        now = self._clock()
        if (
            self.finished
            or self.stop_reason
            or type(now) not in {int, float}
            or not math.isfinite(now)
            or self._cancelled is not None
            and self._cancelled()
        ):
            self.stop_reason = (
                self.stop_reason or "Investigation finished or cancelled."
            )
            raise InvestigationStopped(self.stop_reason)
        self._assert_current_authority()

    def _assert_current_authority(self):
        try:
            self._authority()
            self._original.assert_current(self._pipeline)
            if self._accepted_bundle is not None:
                for output in self._accepted_bundle.snapshot._outputs:
                    binding = output.binding
                    if binding is None:
                        continue
                    if (
                        _data_hash(binding.data) != binding.fingerprint
                        or _state_hash(binding.state) != binding.state_fingerprint
                    ):
                        raise InvestigationError(
                            "Frozen accepted result evidence changed."
                        )
                if self._preservation is not None:
                    self._preservation.assert_snapshot(self._accepted_bundle.snapshot)
        except Exception as exc:
            self.stop_reason = (
                "Investigation authority or source/workflow revision changed."
            )
            raise InvestigationStopped(self.stop_reason) from exc

    @staticmethod
    def _request_identity(request):
        detached = json.loads(_json(request))
        detached["budgets"]["seconds_remaining"] = 0
        return _json(detached)

    def _grant_candidate_outputs(self, evidence):
        for candidate in self._candidates:
            if candidate.status not in {"completed", "rejected"}:
                continue
            if evidence == candidate.evidence:
                for output in evidence.get("outputs", []):
                    key = (output["output_alias"], output["sha256"])
                    self._sent_numeric.add(key)
                    self._numeric_owners[key] = frozenset({candidate.candidate_id})

    def _grant_transmitted_numeric(self, request):
        # Only literal typed engine payloads in this exact actual request count.
        # An index, a quoted ref, a page fragment or a pinned old receipt cannot
        # turn a never-transmitted measurement into current evidence authority.
        if request["source"] == self._original.source_summary | {
            "sha256": self._source.fingerprint
        }:
            self._sent_numeric.add(("input", self._source.fingerprint))
        context = request.get("decision_context")
        if context is not None:
            context = validate_decision_context(context, session_id=self.session_id)
            for entry in context["numeric_evidence"]:
                reference = entry["evidence_ref"]
                key = (entry["alias"], reference["sha256"])
                if self._numeric_payloads.get(key) == _json(entry["payload"]):
                    self._sent_numeric.add(key)
        if self.include_output_images and "accepted_candidate" in request:
            self._grant_candidate_outputs(request["accepted_candidate"])
        for supplied in request["evidence_notebook"]["working_records"]:
            if self._latest_action_record is None or any(
                supplied[key] != self._latest_action_record[key]
                for key in ("record_id", "sha256")
            ):
                continue
            receipt = supplied["payload"]
            if receipt.get("status") != "ok":
                continue
            action, result = receipt.get("action"), receipt.get("result")
            if action == "diagnostic_probe" and self.include_output_images:
                # A completed probe is not a candidate. Its aggregate numeric
                # receipt only grants the exact separately issued probe handle.
                if type(result) is dict and type(result.get("evidence_ref")) is dict:
                    ref = result["evidence_ref"]
                    key = (ref.get("alias"), ref.get("sha256"))
                    payload = {k: v for k, v in result.items() if k != "evidence_ref"}
                    if self._numeric_payloads.get(key) == _json(payload):
                        self._sent_numeric.add(key)
            if (
                action in {"trial_candidate", "trial_edit"}
                and self.include_output_images
            ):
                self._grant_candidate_outputs(result)
            if action not in {
                "inspect_source",
                "inspect_candidate",
                "inspect_probe",
                "compare_candidates",
            }:
                continue
            if type(result) is not dict or type(result.get("evidence_ref")) is not dict:
                continue
            ref = result["evidence_ref"]
            if set(ref) != {"kind", "alias", "sha256"} or ref["kind"] != "numeric":
                continue
            key = (ref["alias"], ref["sha256"])
            payload = {
                name: value for name, value in result.items() if name != "evidence_ref"
            }
            if self._numeric_payloads.get(key) == _json(payload):
                self._sent_numeric.add(key)

    def _request_generation(self):
        return (
            self.actions_used,
            len(self._history),
            self.retained_candidate_id,
            self._control_revision,
            hashlib.sha256(
                _json(
                    {
                        "index": self._notebook.index(),
                        "pins": self._notebook.pinned_record_refs,
                        "ledger": list(self.experiment_ledger),
                        "candidates": [
                            candidate.evidence for candidate in self._candidates
                        ],
                    }
                )
            ).hexdigest(),
        )

    def record_model_call(self, count=1, *, request=None):
        if self._pending_finish_review is not None:
            raise InvestigationError(
                "A provisional Finish is awaiting the user's decision."
            )
        self._guard()
        if type(request) is not dict:
            raise InvestigationError(
                "Bind every actual model call to its exact outgoing request."
            )
        identity = self._request_identity(request)
        current = self._request_identity(self.public_request())
        images = tuple(_json(preview.manifest()) for preview in self._images)
        generation = self._request_generation()
        # One transport may make a bounded format repair with the SAME frozen
        # request. Only that already validated generation may retain its exact
        # original counters; no changed field or scientific state is ignored.
        repeated = self._last_transmission == (identity, generation, images)
        if identity != current and not repeated:
            raise InvestigationError(
                "The outgoing request or current PNG board changed."
            )
        if type(count) is not int or count < 1:
            if not self._continuation_enabled:
                self.stop_reason = "Investigation model-call budget reached."
                raise InvestigationStopped(self.stop_reason)
            raise InvestigationError(
                "Record a positive integer actual model-call count."
            )
        if self.model_calls_used + count > self.limits.max_model_calls:
            self._pause_allowance("Investigation model-call budget reached.", "calls")
        image_bytes = sum(len(p.png_bytes) for p in self._images) * count
        if self._image_bytes + image_bytes > self.limits.max_image_bytes:
            self._pause_allowance(
                "Investigation transmitted-image byte budget reached.", "images"
            )
        self._image_bytes += image_bytes
        self.model_calls_used += count
        self._sent_previews.update(p.manifest()["sha256"] for p in self._images)
        for preview in self._images:
            digest = preview.manifest()["sha256"]
            self._sent_preview_owners[digest] = self._sent_preview_owners.get(
                digest, frozenset()
            ) | self._preview_owners.get(id(preview), frozenset())
        self._grant_transmitted_numeric(request)
        self._last_transmission = (identity, generation, images)

    def public_request(self):
        return self._public_request(
            transport_calls=2, finish_review=self._pending_finish_review is not None
        )

    def _public_request(
        self, *, transport_calls, finish_review=False, include_review_reserve=True
    ):
        if finish_review:
            self._assert_finish_review_authority()
        else:
            self._guard()
        ledger = list(self.experiment_ledger)
        phase = self._closing_phase(
            transport_calls=transport_calls,
            include_review_reserve=include_review_reserve,
        )
        schema = action_schema()
        schema["actions"] = {
            name: value
            for name, value in schema["actions"].items()
            if name in phase["allowed_actions"]
        }
        index = []
        for entry in ledger:
            assessment = entry["assessment"]
            outcomes = (
                {}
                if assessment is None
                else {
                    row["check_id"]: row["outcome"]
                    for row in assessment["check_results"]
                }
            )
            index.append(
                {
                    "experiment_id": entry["experiment_id"],
                    "state": entry["state"],
                    "candidate_id": entry["candidate_id"],
                    "checks": [
                        {
                            "check_id": row["check_id"],
                            "kind": row["kind"],
                            "outcome": outcomes.get(row["check_id"], "not_assessed"),
                        }
                        for row in entry["plan"]["evaluation_checks"]
                    ],
                    "full_record": self._ledger_records[entry["experiment_id"]],
                }
            )
        working = []
        latest_is_page = False
        if self._latest_action_record is not None:
            descriptor = self._notebook.descriptor(
                self._latest_action_record["record_id"]
            )
            if descriptor["byte_count"] <= 64 * 1024:
                working.append(self._latest_action_record)
            latest_payload = self._notebook.lookup(**self._latest_action_record)
            latest_is_page = (
                latest_payload.get("action") == "read_notebook"
                and latest_payload.get("status") == "ok"
            )
            if latest_is_page and not working:
                raise InvestigationStopped(
                    "The requested page cannot fit the finite working view; "
                    "unpin notebook records and read it again."
                )
        try:
            notebook_view = self._notebook.view(working)
        except NotebookError:
            if latest_is_page:
                raise InvestigationStopped(
                    "The requested page cannot fit alongside the selected pins; "
                    "unpin notebook records and read it again."
                ) from None
            # Explicit unpinned latest payload may leave the working view, never
            # the immutable journal or the atomic user/model selected pins.
            notebook_view = self._notebook.view()
        request = {
            "kind": "napari-vipp-investigation-request",
            "version": 2,
            "session_id": self.session_id,
            "proposal_context_id": self.proposal_context_id,
            "objective": self.dialogue.objective,
            "clarifications": [
                {"question": q, "answer": a} for q, a in self.dialogue.clarifications
            ],
            "deferred_questions": list(self.dialogue.deferred_questions),
            "source": self._original.source_summary
            | {"sha256": self._source.fingerprint},
            "image_permissions": {
                "input_images": self.include_input_images,
                "output_images": self.include_output_images,
            },
            "budgets": {
                "attempts_remaining": self.limits.max_attempts - self.attempted_trials,
                "actions_remaining": self.limits.max_actions - self.actions_used,
                "model_calls_remaining": self.limits.max_model_calls
                - self.model_calls_used,
                "seconds_remaining": self._seconds_remaining(),
                "image_bytes_remaining": self.limits.max_image_bytes
                - self._image_bytes,
                "candidate_snapshot_bytes_per_attempt": MAX_SNAPSHOT_BYTES,
                "session_snapshot_bytes_maximum": (
                    self.limits.max_attempts + 1 + self._seed_count
                )
                * MAX_SNAPSHOT_BYTES,
            },
            "retained_candidate_id": self.retained_candidate_id,
            "experiment_plan_required": self._require_experiment_plan,
            "experiment_ledger": ledger[-1:],
            "experiment_index": index,
            "diagnostic_probes": {
                "version": 1,
                "attempts_used": self.attempted_probes,
                "attempts_remaining": MAX_DIAGNOSTIC_PROBES - self.attempted_probes,
                "maximum_attempts": MAX_DIAGNOSTIC_PROBES,
                "max_input_voxels": MAX_PROBE_INPUT_VOXELS,
                "max_nodes": MAX_PROBE_NODES,
                "max_outputs": MAX_PROBE_OUTPUTS,
                "snapshot_bytes_used": self._probe_snapshot_bytes,
                "snapshot_bytes_maximum": MAX_PROBE_SNAPSHOT_BYTES,
                "index": [
                    {
                        "probe_id": p.probe_id,
                        "status": p.status,
                        "full_record": self._probe_records[p.probe_id],
                    }
                    for p in self._probes
                ],
            },
            "response_schema": schema,
            "evidence_notebook": notebook_view,
            "closing_phase": phase,
            "decision_context": self._decision_context(),
            "image_previews": [p.manifest() for p in self._images],
            "comparison_previews": self._comparison_preview_context(),
            "rules": [
                "Use only finite actions and ordinary node contracts. No code, "
                "filesystem, arbitrary sources, writers or external tools.",
                "Inspect actual evidence, bind output roles, compare and "
                "nominate a completed candidate; latest is not retained by default.",
                "Model interpretations are not scientific validation. Never "
                "invent observed unseen pixels, accuracy or biological identity. "
                "An explicit ordinary-node reconstruction may be tested as a "
                "hypothesis; never present inferred values as observed anatomy.",
                "Trials do not change the user graph. Finish nominates immutable "
                "completed results for separate review and normal checked Add.",
                "Use describe_candidate to retrieve this session's exact tested "
                "parameters and connections before targeted revisions as needed. "
                "Failed recipes were authored, not successful scientific outputs.",
                "Use the compact node_index for ordinary-node discovery and "
                "describe_nodes for up to eight full contracts in one action. "
                "Discovery consumes the unchanged action/model/time budgets, "
                "not scientific trial attempts. ROI diagnostics measure only "
                "the explicit native region; whether foreground is wanted or "
                "missing remains a model interpretation, not measured truth.",
                "This is a finite working view, NOT complete inline history. "
                "Every action, result and full ledger snapshot is unchanged in "
                "this session's immutable notebook. Use read_notebook with exact "
                "record/hash/page, or pin_notebook for a small complete record. "
                "Indices, pages, quoted handles and pins create no evidence authority. "
                "Current image_previews alone supply actual PNG pixels; pin_previews "
                "may keep exact previous views together within the same cap. "
                "Clearing explicit pins does not clear automatic cached context.",
                "Use find_nodes/describe_nodes for the unchanged registry or retrieve "
                "its exact node_index notebook record. The complete experiment_index "
                "keeps all checks/outcomes visible; experiment_ledger is latest-only. "
                "Do not confuse a remembered or quoted judgment with new measurement.",
                "An unpinned latest result too large for working context remains "
                "exactly retrievable by pages. If selected pins leave no room for "
                "a requested page/result, explicitly unpin records and read again; "
                "no pinned record or complete journal entry is silently deleted.",
                "decision_context carries exact current recipes, whole node contracts "
                "and explicitly citable measured facts without manual pin/read setup. "
                "Its omitted list discloses complete entries not inlined. Numerical "
                "handles become available only when their literal typed payload is "
                "actually supplied; no accuracy claim or automatic winner follows.",
            ],
        }
        if self._continuation_enabled:
            request["continuation"] = self._continuation_context()
        if self._review_before_finish:
            request["completion_review"] = {
                "required": True,
                "pending": self._pending_finish_review is not None,
                "continue_requests": self._continue_requests,
                "latest_feedback": json.loads(self._latest_finish_feedback)
                if self._latest_finish_feedback is not None
                else None,
                "reservation": self._review_reservation(),
            }
        if self._accepted_bundle is not None:
            request["accepted_candidate"] = (
                self._accepted_candidate_descriptor()
                if self._has_completed_trial()
                else self._candidates[0].evidence
            )
            request["rules"].extend(
                [
                    "candidate0 is the explicitly accepted existing result, "
                    "not a new trial. Inspect it and keep it unless an explicitly "
                    "assessed replacement is preferable.",
                    "Use trial_edit for a controlled parameter or operation change "
                    "when practical; it preserves keys, wires and roles. Full "
                    "trial_candidate remains available for larger declared "
                    "hypotheses. Neither changes the accepted graph.",
                ]
            )
        if self._preservation is not None:
            request["preservation_constraint"] = self._preservation.public_contract()
            request["protected_role"] = self._protected_role
        # Optional convenience context must not force closing by itself. Exact
        # pins, the requested page/result and mandatory ledger/source have priority.
        desk = request.pop("decision_context")
        headroom = WORKING_REQUEST_TARGET_BYTES - len(_json(request)) - 32
        if headroom >= 512:
            request["decision_context"] = self._decision_context(
                byte_limit=min(64 * 1024, headroom)
            )
        elif len(_json(desk)) <= max(0, headroom):
            request["decision_context"] = desk
        else:
            desk_rule = next(
                index
                for index, rule in enumerate(request["rules"])
                if rule.startswith("decision_context carries")
            )
            request["rules"][desk_rule] = (
                "Optional decision context omitted at the finite request target; "
                "exact session records and ordinary lookups remain available."
            )
        if len(_json(request)) > WORKING_REQUEST_TARGET_BYTES:
            phase = self._closing_phase(
                context_pressure=True,
                transport_calls=transport_calls,
                include_review_reserve=include_review_reserve,
            )
            request["closing_phase"] = phase
            request["response_schema"]["actions"] = {
                name: value
                for name, value in action_schema()["actions"].items()
                if name in phase["allowed_actions"]
            }
            if self._accepted_bundle is not None:
                request["accepted_candidate"] = self._accepted_candidate_descriptor()
            supplied = request["evidence_notebook"]["working_records"]
            latest = next(
                (
                    row
                    for row in supplied
                    if self._latest_action_record is not None
                    and row["record_id"] == self._latest_action_record["record_id"]
                ),
                None,
            )
            numerical = latest is not None and latest["payload"].get("action") in {
                "inspect_source",
                "inspect_candidate",
                "compare_candidates",
                "trial_candidate",
                "trial_edit",
                "diagnostic_probe",
                "inspect_probe",
                "read_notebook",
            }
            if not numerical and len(_json(request)) > WORKING_REQUEST_TARGET_BYTES:
                request["evidence_notebook"] = self._notebook.view()
        if len(_json(request)) > MAX_REQUEST_BYTES:
            if finish_review:
                raise InvestigationError(
                    "The unchanged 192 KiB request cap prevents continuation."
                )
            self.stop_reason = (
                "Mandatory notebook context exceeds the unchanged 192 KiB request cap."
            )
            raise InvestigationStopped(self.stop_reason)
        return request

    def perform(
        self, action: InvestigationAction | Mapping | str
    ) -> InvestigationResult:
        with self._owner_lease():
            self._guard()
            self._active_action = True
            try:
                result = self._perform_action(action)
            finally:
                self._active_action = False
            if self._continuation_enabled and not (self.finished or self.stop_reason):
                if self._pending_finish_review is not None:
                    self._pause_locked(
                        "Review the completed nomination before continuing.",
                        "finish_review",
                    )
                else:
                    self._settle_requested_pause()
                    if not self.paused and self._seconds_remaining() <= 0:
                        self._pause_locked(
                            "The approved active-time allowance is reached "
                            "after the completed step.",
                            "time",
                        )
            return result

    def _perform_action(
        self, action: InvestigationAction | Mapping | str
    ) -> InvestigationResult:
        if self._pending_finish_review is not None:
            raise InvestigationError(
                "A provisional Finish is awaiting the user's decision."
            )
        self._guard()
        if not isinstance(action, InvestigationAction):
            action = parse_investigation_action(
                action, expected_session_id=self.session_id
            )
        if action.session_id != self.session_id:
            raise InvestigationError("The action belongs to another session.")
        if self.actions_used >= self.limits.max_actions:
            self._pause_allowance("Investigation action budget reached.", "actions")
        if (
            self._continuation_enabled
            and action.action in {"trial_candidate", "trial_edit"}
            and self.attempted_trials >= self.limits.max_attempts
        ):
            self._pause_allowance("Candidate attempt budget reached.", "trials")
        if (
            self._last_transmission is not None
            and self._last_transmission[1] == self._request_generation()
            and action.action
            not in json.loads(self._last_transmission[0])["closing_phase"][
                "allowed_actions"
            ]
        ):
            raise InvestigationError(
                "That action was not allowed by the exact outgoing request; "
                "assessment/Finish room is reserved."
            )
        if (
            action.action
            not in self._public_request(transport_calls=0)["closing_phase"][
                "allowed_actions"
            ]
        ):
            raise InvestigationError(
                "That action is unavailable while assessment/Finish room is reserved."
            )
        self.actions_used += 1
        self._notebook.append(
            "action_request",
            action.to_dict(),
            owner={"action_index": self.actions_used, "action": action.action},
        )
        prior_attempts = len(self._candidates)
        prior_probes = self.attempted_probes
        prior_images = self._images
        self._inspection_reused = False
        try:
            result = getattr(self, "_do_" + action.action)(**action.arguments)
            if not self.finished and not self.stop_reason:
                if self._pending_finish_review is not None:
                    self.assert_finish_review_current()
                else:
                    self._guard()
            public = {"action": action.action, "status": "ok", "result": result}
        except (InvestigationStopped, InvestigationPaused):
            raise
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self._images = prior_images
            # No arbitrary exception text: node/runtime messages may carry paths.
            public = {
                "action": action.action,
                "status": "error",
                "error": self._safe_error(exc),
                "error_type": type(exc).__name__,
            }
            if (
                action.action in {"trial_candidate", "trial_edit"}
                and len(self._candidates) > prior_attempts
            ):
                public["candidate_receipt"] = self._candidates[-1].evidence
            if (
                action.action == "diagnostic_probe"
                and self.attempted_probes > prior_probes
            ):
                public["probe_receipt"] = self._probes[-1].evidence
        omissions = self._add_protected_preview()
        if omissions:
            public["image_preview_omissions"] = omissions
        public["request"] = self._compact_action(action)
        public["image_previews"] = [p.manifest() for p in self._images]
        if action.action in {"inspect_source", "inspect_candidate", "inspect_probe"}:
            public["inspection_reused"] = self._inspection_reused
        if action.action == "read_notebook" and public["status"] == "ok":
            # This clone contains only immutable finite JSON bytes, not source
            # or scientific arrays. Check the EXACT prospective receipt/index
            # and existing atomic pins before promising a page on the next send.
            prospective = deepcopy(self._notebook)
            try:
                descriptor = prospective.append(
                    "action_receipt",
                    public,
                    owner={"action_index": self.actions_used, "action": action.action},
                )
                prospective.view([self._record_ref(descriptor)])
            except NotebookError:
                public = {
                    key: value
                    for key, value in public.items()
                    if key not in {"result", "status"}
                } | {
                    "status": "error",
                    "error_type": "NotebookError",
                    "error": (
                        "The requested page and selected pins/index exceed the "
                        "finite working view. Unpin notebook records and read "
                        "the page again; no pinned or journal bytes were changed."
                    ),
                }
        encoded = _json(public)
        self._history.append(encoded)
        try:
            self._latest_action_record = self._record_ref(
                self._notebook.append(
                    "action_receipt",
                    public,
                    owner={"action_index": self.actions_used, "action": action.action},
                )
            )
            self._sync_notebook_records()
        except NotebookError as exc:
            self.stop_reason = (
                "The complete local notebook exceeded its finite recording contract."
            )
            raise InvestigationStopped(self.stop_reason) from exc
        return InvestigationResult(encoded, self._images)

    @staticmethod
    def _safe_error(exc):
        if isinstance(
            exc,
            (
                InvestigationError,
                ProposalError,
                ExperimentError,
                RecipeEditError,
                DiagnosticProbeError,
                NotebookError,
                DecisionContextError,
            ),
        ):
            message = str(exc)
            if len(message) <= 2000 and not _PRIVATE_PATH.search(message):
                return message
        if isinstance(exc, KeyError):
            return (
                "An explicit output role, proposal node or required field "
                "was not found."
            )
        return (
            "The node, parameter, output or view failed its contract; "
            "inspect the ordinary node contract and correct the action."
        )

    @staticmethod
    def _compact_action(action):
        arguments = action.arguments
        if action.action == "trial_candidate":
            recipe = arguments["proposal"]
            nodes = recipe.get("nodes") if type(recipe) is dict else None
            if type(nodes) is not list:
                nodes = []
            summary = recipe.get("summary", "") if type(recipe) is dict else ""
            return {
                "recipe_sha256": hashlib.sha256(_json(recipe)).hexdigest(),
                "summary": summary if type(summary) is str else "[invalid summary]",
                "operations": [n.get("operation_id") for n in nodes if type(n) is dict],
                "output_roles": arguments["output_roles"],
            } | (
                {"experiment_id": arguments["experiment_id"]}
                if "experiment_id" in arguments
                else {}
            )
        return arguments

    @staticmethod
    def _node_index(*, safe_only=False):
        entries = [
            {k: entry[k] for k in ("operation_id", "title", "category")}
            | {"trial_allowed": entry["operation_id"] in REFINEMENT_SAFE_OPERATION_IDS}
            for entry in build_catalogue()["operations"]
            if not safe_only or entry["operation_id"] in REFINEMENT_SAFE_OPERATION_IDS
        ]
        if len(entries) > MAX_NODE_INDEX_ENTRIES or len(_json(entries)) > 48 * 1024:
            raise InvestigationError("The compact node index exceeds its bound.")
        return entries

    def _do_find_nodes(self, query):
        query = _text(query, "Node query", 200).casefold()
        keywords = query.split()
        return [
            entry
            for entry in self._node_index()
            if query == "*"
            or all(
                word
                in (
                    entry["operation_id"]
                    + " "
                    + entry["title"]
                    + " "
                    + entry["category"]
                ).casefold()
                for word in keywords
            )
        ]

    def _do_describe_node(self, operation_id):
        _text(operation_id, "Operation ID", 128)
        for entry in build_catalogue()["operations"]:
            if entry["operation_id"] == operation_id:
                contract = entry | {
                    "trial_allowed": operation_id in REFINEMENT_SAFE_OPERATION_IDS
                }
                self._decision_contracts[operation_id] = _json(contract)
                while (
                    len(self._decision_contracts) > 64
                    or sum(map(len, self._decision_contracts.values())) > 512 * 1024
                ):
                    self._decision_contracts.pop(next(iter(self._decision_contracts)))
                self._recent_contract_ids = (
                    operation_id,
                    *(
                        item
                        for item in self._recent_contract_ids
                        if item != operation_id
                    ),
                )[:64]
                return contract
        raise InvestigationError("Unknown existing node ID.")

    def _do_describe_nodes(self, operation_ids):
        if (
            type(operation_ids) is not list
            or not 1 <= len(operation_ids) <= MAX_NODE_DESCRIPTIONS
            or any(type(value) is not str for value in operation_ids)
            or len(set(operation_ids)) != len(operation_ids)
        ):
            raise InvestigationError("Use one to eight unique exact existing node IDs.")
        result = [
            self._do_describe_node(operation_id) for operation_id in operation_ids
        ]
        if len(_json(result)) > MAX_PROPOSAL_BYTES:
            raise InvestigationError("Requested node contracts exceed 128 KiB.")
        return result

    def _candidate(self, candidate_id, *, completed=True):
        candidate = next(
            (c for c in self._candidates if c.candidate_id == candidate_id), None
        )
        if candidate is None or completed and candidate.status != "completed":
            raise InvestigationError(
                "A fully completed immutable candidate is required."
            )
        return candidate

    def _do_describe_candidate(self, candidate_id):
        self._require_output_evidence()
        candidate = self._candidate(candidate_id, completed=False)
        authored = candidate.recipe
        if type(authored) is dict:
            nodes = authored.get("nodes")
            if type(nodes) is not list:
                nodes = []
            recipe = {
                "nodes": [
                    {
                        key: node[key]
                        for key in ("key", "operation_id", "params")
                        if key in node
                    }
                    for node in nodes
                    if type(node) is dict
                ],
                "connections": authored.get("connections", []),
            }
        else:
            recipe = {"invalid_authored_json": authored}
        return {
            "candidate_id": candidate.candidate_id,
            "run_id": candidate.run_id,
            "status": candidate.status,
            "recipe_sha256": hashlib.sha256(candidate._recipe).hexdigest(),
            "scientific_recipe": recipe,
            "output_roles": candidate.output_roles,
            "role_bindings": candidate.evidence.get("role_bindings", []),
            "meaning": (
                "Exact authored parameters and wiring from this session only. "
                "Failed candidates have no completed output evidence. Node "
                "explanation prose is omitted; scientific settings are not trimmed."
            ),
        }

    def _decision_candidates(self):
        eligible = [
            c for c in self._candidates if c.status in {"completed", "rejected"}
        ]
        identifiers = [eligible[-1].candidate_id] if eligible else []
        identifiers.append(self.retained_candidate_id)
        return [
            next(c for c in eligible if c.candidate_id == identifier)
            for identifier in dict.fromkeys(identifiers)
            if identifier is not None
            and any(c.candidate_id == identifier for c in eligible)
        ]

    def _accepted_candidate_descriptor(self):
        return {
            "candidate_id": "candidate0",
            "status": "completed",
            "full_record": self._candidate_records["candidate0"],
            "meaning": (
                "Accepted candidate descriptor only, not inline output evidence. "
                "Its exact full public record remains retrievable."
            ),
        }

    def _has_completed_trial(self):
        return any(
            candidate.status in {"completed", "rejected"}
            for candidate in self._candidates[self._seed_count :]
        )

    def _remember_candidate_context(self, candidate):
        """Register already measured immutable facts; do not issue or calculate."""
        evidence = candidate.evidence
        identifier = candidate.candidate_id
        common = {
            "candidate_id": identifier,
            "run_id": candidate.run_id,
            "source_sha256": self._source.fingerprint,
            "source_grid": _grid_summary(self._source.state),
        }
        status = common | {
            "status": candidate.status,
            "execution": evidence.get("execution", {}),
            "role_bindings": evidence.get("role_bindings", []),
            "meaning": "Actual completed calculation status, not scientific accuracy.",
        }
        self._numeric_evidence_ref(
            f"{identifier}/completed", status, candidate_ids=(identifier,)
        )
        if "preservation_check" in evidence:
            self._numeric_evidence_ref(
                f"{identifier}/preservation",
                common | {"preservation_check": evidence["preservation_check"]},
                candidate_ids=(identifier,),
            )
        for output in evidence.get("outputs", []):
            self._numeric_evidence_ref(
                f"{identifier}/measurements/{output['role']}",
                common
                | {
                    "role": output["role"],
                    "output_sha256": output["sha256"],
                    "diagnostics": output["diagnostics"],
                },
                candidate_ids=(identifier,),
            )
        baseline = next(
            (
                item
                for item in self._candidates
                if item.candidate_id == self.retained_candidate_id
                and item.status in {"completed", "rejected"}
            ),
            candidate,
        )
        topology = topology_review_payload(
            baseline.candidate_id, baseline.evidence, identifier, evidence
        )
        if topology is not None:
            self._numeric_evidence_ref(
                f"{baseline.candidate_id}:{identifier}/mesh_topology_review",
                common | topology,
                candidate_ids=(baseline.candidate_id, identifier),
            )
        for node in candidate.recipe.get("nodes", []):
            operation_id = node.get("operation_id")
            if operation_id not in self._decision_contracts:
                self._do_describe_node(operation_id)

    def _decision_context(self, *, byte_limit=64 * 1024):
        candidates = self._decision_candidates() if self.include_output_images else []
        identifiers = {c.candidate_id for c in candidates}
        recipes = [self._do_describe_candidate(c.candidate_id) for c in candidates]
        current = candidates[0] if candidates else None
        aliases, deferred_aliases = [], []
        if current is not None:
            identifier = current.candidate_id
            topology_alias = next(
                (
                    alias
                    for alias, key in reversed(tuple(self._decision_numeric.items()))
                    if alias.endswith("/mesh_topology_review")
                    and json.loads(self._numeric_payloads[key]).get("candidate_id")
                    == identifier
                ),
                None,
            )
            if topology_alias is not None:
                aliases.append(topology_alias)
            aliases.append(f"{identifier}/preservation")
            aliases.extend(
                f"{identifier}/measurements/{row['role']}"
                for row in current.evidence.get("outputs", [])
                if row["diagnostics"].get("kind") == "3D surface mesh"
            )
        # A fresh exact comparison or inspection is more useful to the pending
        # assessment than repeatedly supplying the accepted baseline's facts.
        recent = [
            (alias, json.loads(self._numeric_payloads[key]))
            for alias, key in reversed(tuple(self._decision_numeric.items()))
            if current is not None
            and current.candidate_id in self._numeric_owners.get(key, frozenset())
        ]
        comparison = next(
            (
                alias
                for alias, payload in recent
                if "left" in payload and "right" in payload and "roles" in payload
            ),
            None,
        )
        inspection = next(
            (
                alias
                for alias, payload in recent
                if payload.get("alias") == alias and "diagnostics" in payload
            ),
            None,
        )
        aliases.extend(alias for alias in (comparison, inspection) if alias is not None)
        for index, candidate in enumerate(candidates):
            identifier = candidate.candidate_id
            target = aliases if index == 0 else deferred_aliases
            target.extend((f"{identifier}/completed", f"{identifier}/preservation"))
            target.extend(
                f"{identifier}/measurements/{row['role']}"
                for row in candidate.evidence.get("outputs", [])
            )
        if "input" in self._decision_numeric:
            (aliases if current is None else deferred_aliases).append("input")
        other_recent = [
            alias
            for alias, key in reversed(tuple(self._decision_numeric.items()))
            if alias not in aliases
            and alias not in deferred_aliases
            and self._numeric_owners.get(key, frozenset()) <= identifiers
        ]
        deferred_aliases.extend(other_recent[:2])

        def numeric_entries(ordered_aliases):
            result = []
            for alias in dict.fromkeys(ordered_aliases):
                key = self._decision_numeric.get(alias)
                if key is None:
                    continue
                result.append(
                    {
                        "alias": alias,
                        "payload": json.loads(self._numeric_payloads[key]),
                        "evidence_ref": {
                            "kind": "numeric",
                            "alias": alias,
                            "sha256": key[1],
                        },
                    }
                )
            return result

        used = {
            node["operation_id"]
            for recipe in recipes
            for node in recipe["scientific_recipe"]["nodes"]
        }
        operation_ids = list(
            dict.fromkeys(
                (
                    *self._recent_contract_ids,
                    *sorted(used),
                )
            )
        )
        contracts = [
            json.loads(self._decision_contracts[identifier])
            for identifier in operation_ids
            if identifier in self._decision_contracts
        ]
        result = build_decision_context(
            session_id=self.session_id,
            recipes=recipes[:1],
            node_contracts=contracts,
            numeric_evidence=numeric_entries(aliases),
            deferred_recipes=recipes[1:],
            deferred_numeric_evidence=numeric_entries(deferred_aliases),
            byte_limit=byte_limit,
            omissions=self._automatic_preview_omissions,
        )
        return validate_decision_context(result, session_id=self.session_id)

    def _require_output_evidence(self):
        if not self.include_output_images:
            raise InvestigationError(
                "Output assessment sharing is off. Trial calculations and "
                "output diagnostics/images require its separate explicit consent."
            )

    def _require_assessed_experiments(self):
        for candidate in self._candidates:
            if candidate.status in {"completed", "rejected"}:
                self._experiments.require_assessed(candidate.candidate_id)

    def _do_plan_experiment(self, baseline_candidate_id, **arguments):
        self._require_output_evidence()
        self._require_assessed_experiments()
        if baseline_candidate_id is not None:
            self._candidate(baseline_candidate_id)
        return self._experiments.plan(
            baseline_candidate_id=baseline_candidate_id,
            validate_evidence_refs=self._validate_evidence_refs,
            **arguments,
        )

    def _do_assess_experiment(self, candidate_id, **arguments):
        self._require_output_evidence()
        candidate = self._candidate(candidate_id, completed=False)
        if candidate.status not in {"completed", "rejected"}:
            raise InvestigationError(
                "Assess only an actual completed scientific trial."
            )
        return self._experiments.assess(
            candidate_id=candidate_id,
            validate_evidence_refs=self._validate_evidence_refs,
            **arguments,
        )

    def _reserve_trial(self, proposal, output_roles, experiment_id):
        self._require_output_evidence()
        self._require_assessed_experiments()
        reservation = self._review_reservation()
        if (
            reservation
            and reservation["reserved_trial_attempts"]
            and self.attempted_trials >= reservation["initial_trial_limit"]
        ):
            raise InvestigationError(
                "One original trial attempt is reserved for Keep improving. "
                "Assess completed tests and propose a genuine Finish first."
            )
        if self._require_experiment_plan and experiment_id is None:
            raise InvestigationError(
                "This investigation requires a prospective plan_experiment "
                "and its exact experiment_id before every trial."
            )
        if experiment_id is not None:
            if self._experiments.get(experiment_id)["state"] != "planned":
                raise InvestigationError("Use an unbound prospective experiment plan.")
        if self.attempted_trials >= self.limits.max_attempts:
            raise InvestigationError("Candidate attempt budget reached.")
        candidate_id = f"candidate{self.attempted_trials + 1}"
        run_id = f"{self.session_id}_{candidate_id}"
        # Invalid/failed proposals consume an attempt too, never a retry loophole.
        recipe = _json(proposal)
        pending = InvestigationCandidate(
            candidate_id,
            run_id,
            "failed",
            None,
            recipe,
            _json(
                {
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "status": "failed",
                    "reason": (
                        "Candidate validation or ordinary calculation did not "
                        "complete; no output evidence exists."
                    ),
                }
            ),
            source_fingerprint=self._original.source_fingerprint,
            _role_bindings=_json(output_roles),
        )
        self._candidates.append(pending)
        if experiment_id is not None:
            self._experiments.bind_trial(experiment_id, candidate_id)
        return pending

    def _do_trial_edit(self, candidate_id, edits, experiment_id=None):
        self._require_output_evidence()
        parent = self._candidate(candidate_id)
        try:
            proposal = apply_recipe_edits(
                parent.recipe,
                edits,
                context_id=self.proposal_context_id,
                safe_operation_ids=REFINEMENT_SAFE_OPERATION_IDS,
            )
        except (ValueError, TypeError, KeyError):
            self._reserve_trial(
                {"invalid_micro_edit": edits}, parent.output_roles, experiment_id
            )
            raise
        return self._do_trial_candidate(
            proposal,
            parent.output_roles,
            experiment_id,
            _edit_receipt={
                "parent_candidate_id": candidate_id,
                "parent_recipe_sha256": hashlib.sha256(parent._recipe).hexdigest(),
                "edits": edits,
            },
        )

    def _do_trial_candidate(
        self, proposal, output_roles, experiment_id=None, *, _edit_receipt=None
    ):
        pending = self._reserve_trial(proposal, output_roles, experiment_id)
        candidate_id, run_id, recipe = (
            pending.candidate_id,
            pending.run_id,
            pending._recipe,
        )
        prepared = validate_proposal(proposal, self._context)
        if not prepared.can_apply:
            raise InvestigationError("Unanswered questions cannot execute.")
        assert_refinement_graph_safe(prepared.candidate, "input")
        if type(output_roles) is not list or not 1 <= len(output_roles) <= 8:
            raise InvestigationError("Bind 1..8 explicit output roles.")
        local_ids = {
            node["key"]: node_id
            for node, node_id in zip(
                proposal["nodes"], prepared.added_node_ids, strict=True
            )
        }
        trial = prepared.candidate.to_pipeline()
        roles, seen = [], set()
        for binding in output_roles:
            if type(binding) is not dict or set(binding) != {"role", "node", "port"}:
                raise InvestigationError("Each role needs exactly role/node/port.")
            role = _text(binding["role"], "Output role", 80)
            port, node_id = binding["port"], local_ids[binding["node"]]
            if (
                role in seen
                or type(port) is not int
                or not 0 <= port < len(trial.output_ports(node_id))
            ):
                raise InvestigationError(
                    "Output roles must be unique and bind existing ports."
                )
            seen.add(role)
            roles.append((role, node_id, port))
        protected_ref = None
        if self._preservation is not None:
            nominated = [
                row for row in output_roles if row["role"] == self._protected_role
            ]
            if (
                len(nominated) != 1
                or nominated[0]["node"] != PRESERVED_OUTPUT_KEY
                or nominated[0]["port"] != 0
            ):
                raise InvestigationError(
                    "Bind the protected semantic role to protected_structure port 0."
                )
            protected_ref = (local_ids[PRESERVED_OUTPUT_KEY], 0)
            preservation_mesh_creators(
                prepared.candidate, prepared.added_node_ids, protected_ref
            )
        self._guard()
        request = PipelineRunRequest(
            run_id=self.attempted_trials,
            workflow=serialize_workflow(trial, compute_request=self.compute_request),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={
                "input": SourcePayload(
                    self._source.data, image_state=self._source.state
                )
            },
            compute_request=self.compute_request,
            manual_node_ids=frozenset(trial.manual_node_ids()),
            target_node_ids=frozenset(prepared.added_node_ids),
            retain_node_ids=frozenset(prepared.added_node_ids),
            cancel_event=_CancelEvent(self._cancel_check),
        )
        try:
            result = execute_pipeline_request(request)
        except Exception as exc:
            self.stop_reason = (
                "Ordinary trial execution failed; manual review is required."
            )
            raise InvestigationStopped(self.stop_reason) from exc
        self._guard()
        if result.cancelled:
            self.stop_reason = (
                "Ordinary trial was cancelled; no output evidence exists."
            )
            raise InvestigationStopped(self.stop_reason)
        if result.error or result.failure is not None:
            return self._trial_failure(result, pending, prepared)
        if (
            result.execution_report is None
            or result.execution_report.cleanup_succeeded is not True
        ):
            self.stop_reason = (
                "Ordinary trial cleanup was failed or unknown; "
                "manual review is required."
            )
            raise InvestigationStopped(self.stop_reason)
        if result.pipeline is None:
            self.stop_reason = (
                "Ordinary trial has no completed pipeline; manual review is required."
            )
            raise InvestigationStopped(self.stop_reason)
        calculated = result.pipeline

        def presentation_exception(node_id):
            node = calculated.nodes[node_id]
            spec = NODE_LIBRARY_BY_ID[node.operation_id]
            return (
                node.operation_id == "review_images"
                and spec.presentation_only
                and spec.output_type == "none"
                and calculated.node_execution_states.get(node_id)
                in {"ready", "not_calculated"}
                and all(
                    value is None for value in calculated.node_outputs.get(node_id, ())
                )
                and calculated.outputs.get(node_id) is None
            )

        if any(
            not presentation_exception(node_id)
            and (
                calculated.node_execution_states.get(node_id) != EXECUTION_READY
                or node_id not in calculated.completed_node_ids
            )
            for node_id in prepared.added_node_ids
        ):
            self.stop_reason = "Ordinary trial did not complete every candidate node."
            raise InvestigationStopped(self.stop_reason)
        captured, evidence, role_snapshots, budget = (
            [],
            [],
            [],
            [self._source.data.nbytes],
        )
        for role, node_id, port in roles:
            data = calculated.node_outputs[node_id][port]
            state = calculated.node_output_states[node_id][port]
            if type(data) is np.ndarray:
                _validate_image(data, state)
            live_binding = _capture_binding(node_id, port, data, state, budget)
            role_snapshots.append(_Output(node_id, port, role, live_binding))
            binding = replace(live_binding, live=None)
            captured.append((role, binding))
            evidence.append(
                {
                    "role": role,
                    "output_alias": f"{candidate_id}/{role}",
                    "sha256": binding.fingerprint,
                    "diagnostics": self._summary(
                        binding, operation_id=calculated.nodes[node_id].operation_id
                    ),
                }
            )
        preservation_result = None
        if self._preservation is not None:
            actual_snapshot = capture_assessment_snapshot(
                calculated, "input", prepared.added_node_ids, run_id
            )
            # Check only the explicitly nominated and already aggregate-budgeted
            # role ports. A delivered mesh must be among them, not a hidden extra.
            actual_snapshot = replace(actual_snapshot, _outputs=tuple(role_snapshots))
            preservation_result = self._preservation.check_detached(
                self._accepted_bundle.snapshot, actual_snapshot, protected_ref
            )
        record = {
            "candidate_id": candidate_id,
            "run_id": run_id,
            "status": "completed",
            "source_sha256": self._source.fingerprint,
            "source_grid": _grid_summary(self._source.state),
            "outputs": evidence,
            "execution": {
                "cleanup_succeeded": True,
                "compute_request": self.compute_request.as_dict(),
                "presentation_exceptions": [
                    f"n{i + 1}"
                    for i, node in enumerate(prepared.added_node_ids)
                    if presentation_exception(node)
                ],
            },
            "interpretation": (
                "Completed calculation is not proof of scientific accuracy "
                "or goal achievement."
            ),
        }
        record["role_bindings"] = [
            public_role_binding(
                row["role"],
                row["node"],
                f"n{prepared.added_node_ids.index(local_ids[row['node']]) + 1}",
                row["port"],
                binding.fingerprint,
            )
            for row, (_role, binding) in zip(output_roles, captured, strict=True)
        ]
        if _edit_receipt is not None:
            record["edit_receipt"] = _edit_receipt
        status = "completed"
        if preservation_result is not None:
            record["preservation_check"] = preservation_result.public_result()
            record["eligible_for_retention"] = preservation_result.passed
            if not preservation_result.passed:
                status = record["status"] = "rejected"
                record["rejection_reason"] = (
                    "Protected foreground was lost; no pixels were repaired. "
                    "The prior retained candidate remains."
                )
        candidate = InvestigationCandidate(
            candidate_id,
            run_id,
            status,
            prepared.candidate,
            recipe,
            _json(record),
            tuple(captured),
            self._original.source_fingerprint,
            _json(output_roles),
            tuple(local_ids.items()),
        )
        self._candidates[-1] = candidate
        self._remember_candidate_context(candidate)
        return candidate.evidence

    def _trial_failure(self, result, pending, prepared):
        failure = result.failure
        if (
            type(failure) is not PipelineExecutionFailure
            or failure.cleanup_succeeded is not True
            or result.execution_report is not None
            and result.execution_report.cleanup_succeeded is not True
        ):
            self.stop_reason = (
                "Failed trial cleanup is failed or unknown; manual review is required."
            )
            raise InvestigationStopped(self.stop_reason)
        safe_failures = []
        if (
            failure.kind == "execution_error"
            and not failure.retryable
            and not failure.runtime_id
            and not failure.segment_id
            and not failure.fallback_records
            and result.pipeline is not None
        ):
            for node_id in prepared.added_node_ids:
                if result.pipeline.node_execution_states.get(node_id) != "error":
                    continue
                operation = result.pipeline.nodes[node_id].operation_id
                safe = safe_mesh_input_failure(
                    operation, failure.error_type, failure.message
                )
                if safe is None:
                    safe_failures = []
                    break
                safe_failures.append({"operation_id": operation, **safe})
        if not safe_failures:
            self.stop_reason = (
                "Ordinary trial failed outside a reviewed recoverable scientific "
                "contract; manual review is required."
            )
            raise InvestigationStopped(self.stop_reason)
        if self._calculation_repairs >= 2:
            self.stop_reason = (
                "The two-transition scientific calculation-repair limit was reached."
            )
            raise InvestigationStopped(self.stop_reason)
        self._calculation_repairs += 1
        evidence = pending.evidence | {"scientific_failures": safe_failures}
        self._candidates[-1] = replace(pending, _evidence=_json(evidence))
        return evidence

    def _do_record_observation(
        self, evidence_refs, interpretation, hypothesis, next_check
    ):
        self._validate_evidence_refs(evidence_refs)
        return {
            "evidence_refs": evidence_refs,
            "model_interpretation": _text(interpretation, "Interpretation", 800),
            "hypothesis_not_observed_fact": _text(hypothesis, "Hypothesis", 800),
            "next_check": _text(next_check, "Next check", 400),
            "meaning": (
                "Model interpretation only; not a measured fact "
                "or scientific validation."
            ),
        }

    def _validate_evidence_refs(
        self, evidence_refs, *, candidate_id=None, required_kind="numeric"
    ):
        """Validate handles only after their numeric/image payload was sent."""
        if type(evidence_refs) is not list or not 1 <= len(evidence_refs) <= 8:
            raise InvestigationError("Use one to eight issued evidence handles.")
        if required_kind not in {"numeric", "preview"}:
            raise InvestigationError("Unsupported evidence ownership kind.")
        candidate_evidence = False
        for ref in evidence_refs:
            if type(ref) is not dict:
                raise InvestigationError(
                    "Evidence references must be exact typed objects."
                )
            if ref.get("kind") == "preview" and set(ref) == {"kind", "sha256"}:
                valid = (
                    type(ref["sha256"]) is str and ref["sha256"] in self._sent_previews
                )
                if valid and candidate_id is not None and required_kind == "preview":
                    candidate_evidence |= candidate_id in self._sent_preview_owners.get(
                        ref["sha256"], ()
                    )
            elif ref.get("kind") == "numeric" and set(ref) == {
                "kind",
                "sha256",
                "alias",
            }:
                valid = (
                    type(ref["alias"]) is str
                    and type(ref["sha256"]) is str
                    and (ref["alias"], ref["sha256"]) in self._sent_numeric
                )
                if valid and candidate_id is not None and required_kind == "numeric":
                    candidate_evidence |= candidate_id in self._numeric_owners.get(
                        (ref["alias"], ref["sha256"]), ()
                    )
            else:
                valid = False
            if not valid:
                raise InvestigationError(
                    "Observation references evidence not transmitted in this session."
                )
        if candidate_id is not None and not candidate_evidence:
            raise InvestigationError(
                f"Assessment requires transmitted {required_kind} evidence from "
                "the exact experiment candidate."
            )

    def _numeric_evidence_ref(self, alias, payload, *, candidate_ids=()):
        digest = hashlib.sha256(_json(payload)).hexdigest()
        self._issued_numeric.add((alias, digest))
        self._numeric_payloads[(alias, digest)] = _json(payload)
        self._numeric_owners[(alias, digest)] = frozenset(candidate_ids)
        self._decision_numeric.pop(alias, None)
        self._decision_numeric[alias] = (alias, digest)
        return {"kind": "numeric", "alias": alias, "sha256": digest}

    def _summary(self, binding, *, operation_id=None):
        if type(binding.data) is np.ndarray:
            return _image_summary(binding.data, binding.state, self._cancel_check)
        if type(binding.data) is _MeshSnapshot:
            return _mesh_summary(binding.data, self._cancel_check)
        return _table_summary(
            binding.data, self._cancel_check, operation_id=operation_id
        )

    def _do_inspect_source(self, **options):
        return self._inspect(self._source, "input", True, **options)

    def _do_diagnostic_probe(self, input, roi, proposal, output_roles):
        self._require_output_evidence()
        self._require_assessed_experiments()
        if self.attempted_probes >= MAX_DIAGNOSTIC_PROBES:
            raise InvestigationError(
                "The six diagnostic-probe attempt limit was reached."
            )
        probe_id = f"probe{self.attempted_probes + 1}"
        record = {
            "probe_id": probe_id,
            "status": "failed",
            "input": input,
            "roi": roi,
            "proposal": proposal,
            "output_roles": output_roles,
            "source_sha256": self._source.fingerprint,
            "meaning": (
                "Disposable diagnostic only, never a retainable candidate. "
                "No completed output evidence exists unless status is completed."
            ),
        }
        # Malformed/failed requests consume this independent finite allowance;
        # they never borrow a candidate slot or create a retry loophole.
        self._probes.append(DiagnosticProbe(probe_id, "failed", _json(record)))
        if type(input) is not dict or input.get("kind") not in {"source", "candidate"}:
            raise InvestigationError(
                "Choose an exact source or completed candidate role for the probe."
            )
        if input["kind"] == "source" and set(input) == {"kind"}:
            parent = self._source
        elif input["kind"] == "candidate" and set(input) == {
            "kind",
            "candidate_id",
            "role",
        }:
            candidate = self._candidate(input["candidate_id"])
            parent = dict(candidate._outputs).get(input["role"])
            if parent is None:
                raise InvestigationError(
                    "The probe input needs an existing explicit output role."
                )
        else:
            raise InvestigationError(
                "Probe inputs cannot load a new source or omit its explicit role."
            )
        if type(parent.data) is not np.ndarray:
            raise InvestigationError(
                "A native-ROI probe needs an array input, not a mesh or table."
            )
        if (
            _data_hash(parent.data) != parent.fingerprint
            or _state_hash(parent.state) != parent.state_fingerprint
        ):
            self.stop_reason = "An immutable diagnostic input binding changed."
            raise InvestigationStopped(self.stop_reason)
        if roi is None:
            raise InvestigationError(
                "Declare every native ROI bound explicitly, including a "
                "whole-image probe."
            )
        data, state, population = _diagnostic_region(parent, roi)
        if data.size > MAX_PROBE_INPUT_VOXELS:
            raise InvestigationError(
                "A diagnostic probe supports at most 262144 native input voxels."
            )
        remaining_bytes = MAX_PROBE_SNAPSHOT_BYTES - self._probe_snapshot_bytes
        if data.nbytes > remaining_bytes:
            raise InvestigationError(
                "The session's 64 MiB diagnostic snapshot allocation is full."
            )
        owned = _capture_binding("input", 0, data, state, [0])
        try:
            captured, size = execute_diagnostic_probe(
                owned.data,
                owned.state,
                proposal,
                output_roles,
                context_id=self.proposal_context_id,
                compute_request=self.compute_request,
                run_id=self.actions_used,
                cancel_event=_CancelEvent(self._cancel_check),
                guard=self._guard,
                remaining_snapshot_bytes=remaining_bytes,
            )
        except DiagnosticProbeStopped as exc:
            self.stop_reason = str(exc)
            raise InvestigationStopped(self.stop_reason) from exc
        self._guard()
        record.update(
            {
                "status": "completed",
                "parent_sha256": parent.fingerprint,
                "parent_state_sha256": parent.state_fingerprint,
                "input_sha256": owned.fingerprint,
                "input_state_sha256": owned.state_fingerprint,
                "population": population,
                "input_grid": _grid_summary(owned.state),
                "snapshot_bytes": size,
                "outputs": [
                    {
                        "role": role,
                        "output_alias": f"{probe_id}/{role}",
                        "sha256": binding.fingerprint,
                        "state_sha256": binding.state_fingerprint,
                        "operation_id": operation,
                        "diagnostics": self._summary(binding, operation_id=operation),
                    }
                    for role, binding, operation in captured
                ],
                "execution": {
                    "cleanup_succeeded": True,
                    "compute_request": self.compute_request.as_dict(),
                },
                "meaning": (
                    "Exact computation on the explicitly declared native ROI, "
                    "not a full-volume candidate or proof of accuracy. ROI "
                    "boundaries cut components and affect filters/reconstruction. "
                    "Views address local output indices with the carried grid. "
                    "No protected-region or delivered-mesh eligibility was "
                    "inferred. Recreate a selected hypothesis in an ordinary "
                    "full checked candidate trial; "
                    "no parameter/method winner is selected automatically."
                ),
            }
        )
        record["evidence_ref"] = self._numeric_evidence_ref(probe_id, record)
        self._probe_snapshot_bytes += size
        self._probes[-1] = DiagnosticProbe(
            probe_id, "completed", _json(record), captured, size
        )
        return record

    def _do_inspect_probe(self, probe_id, role, **options):
        self._require_output_evidence()
        probe = next((p for p in self._probes if p.probe_id == probe_id), None)
        if probe is None or probe.status != "completed":
            raise InvestigationError(
                "Only a completed this-session diagnostic probe can be inspected."
            )
        output = next((row for row in probe.outputs if row[0] == role), None)
        if output is None:
            raise InvestigationError("No such explicit diagnostic output role.")
        _role, binding, operation = output
        return self._inspect(
            binding, f"{probe_id}/{role}", False, operation_id=operation, **options
        )

    def _do_inspect_candidate(self, candidate_id, role, **options):
        self._require_output_evidence()
        candidate = self._candidate(candidate_id, completed=False)
        if candidate.status not in {"completed", "rejected"}:
            raise InvestigationError(
                "Only actual completed scientific outputs can be inspected."
            )
        binding = dict(candidate._outputs).get(role)
        if binding is None:
            raise InvestigationError("No such explicit output role.")
        operation = next(
            (
                node.operation_id
                for node in candidate.graph.nodes
                if node.id == binding.node_id
            ),
            None,
        )
        return self._inspect(
            binding, f"{candidate_id}/{role}", False, operation_id=operation, **options
        )

    def _inspect(
        self,
        binding,
        alias,
        source,
        *,
        percentiles=(),
        histogram=False,
        views=(),
        roi=None,
        operation_id=None,
    ):
        if (
            _data_hash(binding.data) != binding.fingerprint
            or _state_hash(binding.state) != binding.state_fingerprint
        ):
            raise InvestigationStopped("An immutable inspection binding changed.")
        if type(binding.data) is np.ndarray:
            grid_key = _grid_summary(binding.state)
        elif type(binding.data) is _MeshSnapshot:
            grid_key = {
                "axes": "".join(
                    axis.short_label for axis in binding.state.spatial_axes
                ),
                "shape": list(binding.state.source_shape),
                "calibration": [
                    {
                        "axis": axis.short_label,
                        "scale": axis.scale,
                        "origin": axis.translation,
                        "unit": axis.unit or "unspecified",
                    }
                    for axis in binding.state.spatial_axes
                ],
            }
        else:
            grid_key = {"kind": "nonspatial"}
        cache_key = {
            "session_id": self.session_id,
            "alias": alias,
            "source": source,
            "source_sha256": self._source.fingerprint,
            "source_state_sha256": self._source.state_fingerprint,
            "output_sha256": binding.fingerprint,
            "state_sha256": binding.state_fingerprint,
            "grid": grid_key,
            "operation_id": operation_id,
            "permissions": list(self._permission_binding),
            "percentiles": list(percentiles)
            if type(percentiles) in {tuple, list}
            else percentiles,
            "histogram": histogram,
            "views": views,
            "roi": roi,
            "display_policy": {
                "renderer": "native-image-v1",
                "kind": binding.state.kind,
                "range": (
                    "source full-array extrema for source overlays; "
                    "otherwise output full-array extrema"
                ),
                "sampling": "native integer strides, no pixel-aspect rescaling",
                "foreground": "orange nonzero foreground, not semantic identity",
            }
            if type(binding.data) is np.ndarray
            else {"renderer": "mesh-v1"},
        }
        cache_id = hashlib.sha256(_json(cache_key)).hexdigest()
        if cache_id in self._inspection_cache:
            if views:
                images = self._preview_cache.lookup(cache_key)
                if images is None:
                    raise InvestigationError("The exact cached view is unavailable.")
                prior_recent = self._recent_view_ids
                self._recent_view_ids = (
                    *(item for item in self._recent_view_ids if item != cache_id),
                    cache_id,
                )[-64:]
                try:
                    self._install_inspection_board(images)
                except Exception:
                    self._recent_view_ids = prior_recent
                    raise
            self._inspection_reused = True
            cached = json.loads(self._inspection_cache[cache_id])
            self._decision_numeric.pop(alias, None)
            self._decision_numeric[alias] = (alias, cached["evidence_ref"]["sha256"])
            return cached
        if (
            type(histogram) is not bool
            or type(percentiles) not in {list, tuple}
            or len(percentiles) > 8
        ):
            raise InvestigationError("Invalid histogram/percentile request.")
        if type(binding.data) is np.ndarray:
            data, state, population = _diagnostic_region(binding, roi)
            if roi is not None:
                _validate_image(data, state, source=source)
            summary = (
                _image_summary(data, state, self._cancel_check)
                if not source or data.size <= 4_000_000
                else {
                    "kind": "scalar intensity",
                    "grid": _grid_summary(state),
                    "sampling": "exact full array",
                }
            )
            summary["population"] = population
            if roi is not None:
                summary["sampling"] = "exact every value in explicit native ROI"
            stats = exact_finite_stats(data)
            summary.update(
                element_count=int(data.size),
                finite_count=stats.count,
                minimum=stats.minimum,
                maximum=stats.maximum,
            )
            if percentiles:
                values = exact_finite_percentiles(data, tuple(percentiles))
                summary["percentiles"] = [
                    {"percent": p, "value": _percentile_value(v)}
                    for p, v in zip(percentiles, values, strict=True)
                ]
            if histogram:
                counts, value_range = exact_histogram(data)
                summary["histogram"] = {
                    "counts": counts.tolist() if counts is not None else [],
                    "range": list(value_range) if value_range else None,
                    "sampling": "exact every finite value; shared VIPP display bins",
                }
            if views:
                self._images = self._render_views(binding, alias, source, views)
        else:
            if percentiles or histogram or roi is not None:
                raise InvestigationError(
                    "Array ROIs/percentiles/histograms require an image role."
                )
            summary = self._summary(binding, operation_id=operation_id)
            if views and type(binding.data) is _MeshSnapshot:
                self._images = self._render_mesh_views(binding, alias, views)
            elif views:
                raise InvestigationError(
                    "Requested views require an image or mesh role."
                )
        result = {"alias": alias, "sha256": binding.fingerprint, "diagnostics": summary}
        if views:
            view_images = self._images
            view_id = self._preview_cache.store(cache_key, view_images)
            self._view_keys[view_id] = cache_key
            prior_recent = self._recent_view_ids
            self._recent_view_ids = (
                *(item for item in self._recent_view_ids if item != view_id),
                view_id,
            )[-64:]
            try:
                self._install_inspection_board(view_images)
            except Exception:
                self._recent_view_ids = prior_recent
                raise
            result["view_id"] = view_id
            result["view_meaning"] = (
                "Exact cached inspection PNG tuple; pin_previews can place it on the "
                "current board without rerendering. Not a numeric/PNG evidence handle."
            )
        result["evidence_ref"] = self._numeric_evidence_ref(
            alias, result, candidate_ids=() if source else (alias.split("/", 1)[0],)
        )
        encoded = _json(result)
        if (
            len(self._inspection_cache) >= 64
            or sum(map(len, self._inspection_cache.values())) + len(encoded)
            > 8 * 1024 * 1024
        ):
            raise InvestigationError("The finite inspection metadata cache is full.")
        self._inspection_cache[cache_id] = encoded
        return result

    def _do_compare_candidates(self, left, right, roles, roi=None):
        self._require_output_evidence()
        a, b = (
            self._candidate(left, completed=False),
            self._candidate(right, completed=False),
        )
        if any(
            candidate.status not in {"completed", "rejected"} for candidate in (a, b)
        ):
            raise InvestigationError(
                "Compare only actual completed scientific outputs."
            )
        if (
            type(roles) is not list
            or not 1 <= len(roles) <= 8
            or len(set(roles)) != len(roles)
            or a.source_fingerprint != b.source_fingerprint
        ):
            raise InvestigationError(
                "Comparison needs unique explicit roles and identical source revisions."
            )
        records = []
        for role in roles:
            self._guard()
            x, y = dict(a._outputs)[role], dict(b._outputs)[role]
            if (
                type(x.data) is not np.ndarray
                or type(y.data) is not np.ndarray
                or not compare_aligned_grids(
                    ImageGrid.from_image_state(x.state),
                    ImageGrid.from_image_state(y.state),
                ).compatible
            ):
                raise InvestigationError(
                    "Comparison requires aligned explicit image-role grids."
                )
            if x.data.dtype != y.data.dtype:
                # NumPy can promote mixed uint64/int64/float64 operands to a
                # float representation that collapses distinct values above
                # 2**53. Do not call that implicitly converted result exact.
                raise InvestigationError(
                    "Exact numerical comparison requires identical output dtypes; "
                    "no implicit numeric conversion is performed."
                )
            xdata, xstate, population = _diagnostic_region(x, roi)
            ydata, ystate, _ = _diagnostic_region(y, roi)
            _validate_image(xdata, xstate)
            _validate_image(ydata, ystate)
            if not np.isfinite(xdata).all() or not np.isfinite(ydata).all():
                raise InvestigationError(
                    "Non-finite comparison values cannot define foreground."
                )
            xfg, yfg = xdata != 0, ydata != 0
            records.append(
                {
                    "role": role,
                    "left_sha256": x.fingerprint,
                    "right_sha256": y.fingerprint,
                    "population": population,
                    "element_count": int(xdata.size),
                    "equal_value_count": int(np.count_nonzero(xdata == ydata)),
                    "foreground_intersection": int(np.count_nonzero(xfg & yfg)),
                    "foreground_union": int(np.count_nonzero(xfg | yfg)),
                    "left_only_foreground": int(np.count_nonzero(xfg & ~yfg)),
                    "right_only_foreground": int(np.count_nonzero(yfg & ~xfg)),
                    "foreground_change": foreground_change_summary(
                        xdata, ydata, cancelled=self._cancel_check
                    )
                    if xstate.kind in {"binary mask", "label image"}
                    and ystate.kind in {"binary mask", "label image"}
                    else None,
                    "meaning": (
                        "Exact numerical/foreground agreement, "
                        "not biological accuracy or improvement."
                    ),
                }
            )
        result = {
            "left": left,
            "right": right,
            "source_sha256": self._source.fingerprint,
            "roles": records,
        }
        result["evidence_ref"] = self._numeric_evidence_ref(
            f"{left}:{right}", result, candidate_ids=(left, right)
        )
        return result

    def _do_retain_candidate(self, candidate_id, reason):
        self._require_output_evidence()
        self._candidate(candidate_id)
        self._experiments.require_assessed(candidate_id)
        _text(reason, "Retention reason")
        self.retained_candidate_id = candidate_id
        return {"retained_candidate_id": candidate_id, "model_interpretation": reason}

    def _do_finish(self, candidate_id, summary, limitations):
        self._require_output_evidence()
        self._require_assessed_experiments()
        candidate = self._candidate(candidate_id)
        _text(summary, "Finish summary")
        if type(limitations) is not list or len(limitations) > 16:
            raise InvestigationError("Finish limitations must be a bounded list.")
        for value in limitations:
            _text(value, "Limitation")
        self._guard()
        result = {
            "retained_candidate_id": candidate_id,
            "recipe": candidate.recipe,
            "evidence": candidate.evidence,
            "model_interpretation": summary,
            "limitations": limitations,
            "publication": (
                "Not added to the user graph; separate normal checked Add is required."
            ),
        }
        if self._review_before_finish:
            self._pending_finish_review = _json(result)
            self._finish_review_context_status = None
        else:
            self.retained_candidate_id = candidate_id
            self.finished = True
        return result

    def _do_question(self, question):
        question = _text(question, "Question")
        if self._continuation_enabled:
            with self._pause_request_lock:
                self._requested_pause = (
                    "The assistant needs an answer before continuing.",
                    "question",
                    question,
                )
                self._pause_request_generation += 1
                self._pause_requested.set()
            return {"question": question, "paused": True}
        self.stop_reason = "The assistant needs an answer before continuing."
        return {"question": question, "stop_reason": self.stop_reason}

    def final_add_proposal(self) -> PreparedProposal:
        """Revalidate nominated immutable recipe against the original live graph."""
        if not self.finished or not self.retained_candidate_id:
            raise InvestigationError(
                "Finish a completed candidate before preparing Add."
            )
        self._authority()
        self._original.assert_current(self._pipeline)
        recipe = self._candidate(self.retained_candidate_id).recipe
        recipe["context_id"] = self._original.context_id
        return validate_proposal(recipe, self._original)

    def _render_views(self, binding, alias, source, views):
        if type(views) is not list or not 1 <= len(views) <= 7:
            raise InvestigationError("Request one to seven bounded views.")
        if not (self.include_input_images if source else self.include_output_images):
            raise InvestigationError("The relevant independent image consent is off.")
        result = []
        for view in views:
            if (
                type(view) is not dict
                or not {"axis", "indices"} <= view.keys()
                or not view.keys()
                <= {"axis", "indices", "crop", "overlay_source", "montage"}
            ):
                raise InvestigationError("Invalid requested view fields.")
            axis, indices = view["axis"], view["indices"]
            if (
                type(indices) is not list
                or not 1 <= len(indices) <= 7
                or any(type(i) is not int for i in indices)
            ):
                raise InvestigationError(
                    "View indices must be bounded native integers."
                )
            if (
                type(view.get("overlay_source", False)) is not bool
                or type(view.get("montage", False)) is not bool
            ):
                raise InvestigationError(
                    "Overlay and montage choices must be booleans."
                )
            overlay = view.get("overlay_source", False)
            if (
                overlay
                and binding.state.kind not in {"binary mask", "label image"}
                and binding.data.dtype.kind != "b"
            ):
                raise InvestigationError(
                    "Foreground overlays require an explicit mask or label role."
                )
            if overlay and (
                source
                or not self.include_input_images
                or not self.include_output_images
                or not compare_aligned_grids(
                    ImageGrid.from_image_state(binding.state),
                    ImageGrid.from_image_state(self._source.state),
                ).compatible
            ):
                raise InvestigationError(
                    "Source overlays need both consents and aligned image grids."
                )
            if binding.state.axis_order == "YX":
                if axis != "YX" or indices != [0]:
                    raise InvestigationError(
                        "YX data supports only axis YX, indices [0]."
                    )
                position = None
            elif axis in {"Z", "Y", "X"}:
                position = "ZYX".index(axis)
            else:
                raise InvestigationError(
                    "No guessed axes or projections are supported."
                )
            panels, descriptions = [], []
            stats = exact_finite_stats(binding.data)
            for index in indices:
                self._guard()
                if (
                    position is not None
                    and not 0 <= index < binding.data.shape[position]
                ):
                    raise InvestigationError("Requested native slice is out of bounds.")
                plane = (
                    binding.data
                    if position is None
                    else np.take(binding.data, index, axis=position)
                )
                crop = view.get("crop", [0, plane.shape[0], 0, plane.shape[1]])
                if (
                    type(crop) is not list
                    or len(crop) != 4
                    or any(type(v) is not int for v in crop)
                    or not (
                        0 <= crop[0] < crop[1] <= plane.shape[0]
                        and 0 <= crop[2] < crop[3] <= plane.shape[1]
                    )
                ):
                    raise InvestigationError(
                        "Crop bounds must be explicit native stop-exclusive bounds."
                    )
                rows, columns = slice(crop[0], crop[1]), slice(crop[2], crop[3])
                plane = plane[rows, columns]
                row_step, col_step = (
                    max(1, math.ceil(plane.shape[0] / MAX_PREVIEW_EDGE)),
                    max(1, math.ceil(plane.shape[1] / MAX_PREVIEW_EDGE)),
                )
                sampled = plane[::row_step, ::col_step]
                grey = _grey(sampled, stats.minimum, stats.maximum)
                display_stats = stats
                if overlay:
                    src = (
                        self._source.data
                        if position is None
                        else np.take(self._source.data, index, axis=position)
                    )
                    srcstats = exact_finite_stats(self._source.data)
                    display_stats = srcstats
                    grey = _grey(
                        src[rows, columns][::row_step, ::col_step],
                        srcstats.minimum,
                        srcstats.maximum,
                    )
                pixels = np.repeat(grey[..., None], 3, axis=2)
                if (
                    overlay
                    or binding.state.kind in {"binary mask", "label image"}
                    or binding.data.dtype.kind == "b"
                ):
                    foreground = sampled != 0
                    pixels[foreground] = np.rint(
                        pixels[foreground] * 0.45 + np.array([255, 80, 20]) * 0.55
                    ).astype(np.uint8)
                panels.append(pixels)
                descriptions.append(
                    f"{axis} native index {index}; crop {crop}; row/column "
                    f"strides {row_step}/{col_step}; full-array display range "
                    f"{display_stats.minimum!r}..{display_stats.maximum!r}; "
                    f"output population range {stats.minimum!r}..{stats.maximum!r}"
                )
            if view.get("montage", False):
                canvas = np.zeros(
                    (
                        max(p.shape[0] for p in panels),
                        sum(p.shape[1] for p in panels),
                        3,
                    ),
                    dtype=np.uint8,
                )
                offset = 0
                for panel in panels:
                    canvas[: panel.shape[0], offset : offset + panel.shape[1]] = panel
                    offset += panel.shape[1]
                panels = [canvas]
                descriptions = ["Montage left-to-right: " + "; ".join(descriptions)]
            for pixels, description in zip(panels, descriptions, strict=True):
                output = BytesIO()
                Image.fromarray(pixels).save(output, format="PNG")
                description += (
                    f"; grid {_grid_summary(binding.state)}; pixel aspect not "
                    f"rescaled; source overlay {overlay}; orange marks nonzero "
                    "foreground, not semantic identity. Presentation only, "
                    "no geometry/pixel changes; selected views cannot prove "
                    "full-volume accuracy."
                )
                result.append(
                    ImagePreview(
                        f"{alias} requested view {len(result) + 1}",
                        output.getvalue(),
                        description,
                    )
                )
        size = sum(len(p.png_bytes) for p in result)
        if (
            len(result) > MAX_IMAGE_PREVIEWS
            or size > MAX_PREVIEW_TOTAL_BYTES
            or self._image_bytes + size > self.limits.max_image_bytes
        ):
            raise InvestigationError(
                "Requested images exceed shared 7-image/2-MiB "
                "or session image-byte budget."
            )
        for preview in result:
            self._preview_owners[id(preview)] = (
                frozenset() if source else frozenset({alias.split("/", 1)[0]})
            )
        return tuple(result)

    def _render_mesh_views(self, binding, alias, views):
        if (
            not self.include_output_images
            or type(views) is not list
            or not 1 <= len(views) <= 2
        ):
            raise InvestigationError(
                "Mesh views need output-image consent and one or two cameras."
            )
        result = []
        mesh = binding.data
        for view in views:
            if (
                type(view) is not dict
                or set(view) != {"view_index"}
                or type(view["view_index"]) is not int
                or view["view_index"] not in {0, 1}
            ):
                raise InvestigationError("Mesh view_index must be exactly 0 or 1.")
            result.append(
                render_mesh_preview(
                    mesh.vertices,
                    mesh.faces,
                    mesh.face_object_ids,
                    mesh.object_ids,
                    mesh.colors,
                    mesh.state,
                    label=alias,
                    view_index=view["view_index"],
                    source_state=self._source.state,
                    check_cancel=self._guard,
                )
            )
        size = sum(len(p.png_bytes) for p in result)
        if (
            size > MAX_PREVIEW_TOTAL_BYTES
            or self._image_bytes + size > self.limits.max_image_bytes
        ):
            raise InvestigationError(
                "Mesh images exceed the shared preview or session byte budget."
            )
        for preview in result:
            self._preview_owners[id(preview)] = frozenset({alias.split("/", 1)[0]})
        return tuple(result)


def _diagnostic_region(binding, roi):
    """Read-only native diagnostic view; never a scientific output or source.

    The existing adapters calculate exactly the declared population. Temporary
    region metadata describes its sample coordinates only; no workflow/cache
    state, history, scientific array or presentation selection is changed.
    """
    data, state = binding.data, binding.state
    if (
        tuple(data.shape) != tuple(state.shape)
        or data.dtype != np.dtype(state.dtype)
        or state.axis_order not in {"YX", "ZYX"}
        or not state.axes_explicit
        or len(state.axes) != data.ndim
    ):
        raise InvestigationError(
            "Diagnostic arrays need matching explicit carried shape/dtype/axes."
        )
    parent_grid = _grid_summary(state)
    if roi is None:
        return data, state, {"scope": "full array", "parent_grid": parent_grid}
    if (
        type(roi) is not dict
        or set(roi) != {"axis_order", "bounds"}
        or roi["axis_order"] != state.axis_order
        or type(roi["bounds"]) is not list
        or len(roi["bounds"]) != data.ndim
    ):
        raise InvestigationError(
            "ROI needs the exact carried axis_order and one [start,stop] "
            "pair for every native axis."
        )
    bounds = roi["bounds"]
    for pair, size in zip(bounds, data.shape, strict=True):
        if (
            type(pair) is not list
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            or not 0 <= pair[0] < pair[1] <= size
        ):
            raise InvestigationError(
                "ROI bounds must be in-range native integer stop-exclusive "
                "intervals; no implicit clipping or empty ROI."
            )
    coordinates, region_axes = [], []
    for pair, axis in zip(bounds, state.axes, strict=True):
        first = axis.translation + axis.scale * pair[0]
        last = axis.translation + axis.scale * (pair[1] - 1)
        stop = axis.translation + axis.scale * pair[1]
        if not all(math.isfinite(value) for value in (first, last, stop)):
            raise InvestigationError("ROI calibrated coordinates are not finite.")
        region_axes.append(replace(axis, translation=first))
        coordinates.append(
            {
                "axis": axis.short_label,
                "first_sample": first,
                "last_sample": last,
                "stop_exclusive_coordinate": stop,
                "unit": axis.unit or "unspecified",
            }
        )
    region = data[tuple(slice(start, stop) for start, stop in bounds)]
    region_state = replace(state, shape=region.shape, axes=tuple(region_axes))
    return (
        region,
        region_state,
        {
            "scope": "explicit native ROI",
            "axis_order": state.axis_order,
            "bounds": bounds,
            "bounds_convention": "zero-based, start inclusive, stop exclusive",
            "parent_grid": parent_grid,
            "region_grid": _grid_summary(region_state),
            "sample_coordinates": coordinates,
            "meaning": (
                "Exact diagnostic region only, not a scientific output crop. "
                "Components and border contacts are cut by ROI bounds, not whole "
                "object counts/completeness. Foreground is nonzero, not wanted "
                "anatomy; selected views still address the full original binding."
            ),
        },
    )


def _grey(data, low, high):
    if not np.isfinite(data).all():
        raise InvestigationError("Non-finite preview pixels are not silently repaired.")
    if low == high:
        return np.zeros(data.shape, dtype=np.uint8)
    if data.dtype.kind in "biu":
        # Offset in exact integer space BEFORE floating conversion. Otherwise
        # adjacent uint64 IDs above 2**53 collapse into one display value.
        offsets = np.asarray(data, dtype=object) - int(low)
        values = np.asarray(offsets, dtype=np.float64) / (int(high) - int(low))
    else:
        values = np.asarray(data, dtype=np.float64)
        extent = float(high) - float(low)
        if math.isfinite(extent):
            # Direct subtraction preserves adjacent subnormal values; halving
            # first can underflow the declared finite range to zero.
            values = (values - float(low)) / extent
        else:
            # Only an overflowing opposite-sign range uses half arithmetic.
            values = (values * 0.5 - float(low) * 0.5) / (
                float(high) * 0.5 - float(low) * 0.5
            )
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


def _percentile_value(value):
    if isinstance(value, Rational) and not isinstance(value, int):
        return {
            "numerator": int(value.numerator),
            "denominator": int(value.denominator),
        }
    return value.item() if isinstance(value, np.generic) else value
