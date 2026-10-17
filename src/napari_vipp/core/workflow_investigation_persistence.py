"""Explicit, passive checkpoints for the bounded workflow assistant.

Archives are local records, not executable Python, credentials, a live GUI cache
ticket or permission to contact a provider. Restoring requires a newly inspected
source and the exact original graph/compute policy. Every restored runtime starts
paused, with fresh callbacks/locks/clock and no previous transmission grants.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from io import BytesIO
from time import monotonic
from uuid import uuid4

import numpy as np

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.grid import ImageGrid
from napari_vipp.core.meshes import MeshState
from napari_vipp.core.metadata import AxisMetadata, ImageState
from napari_vipp.core.pipeline import GraphConnection, OutputTunnel
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.workflow_agent_budget import (
    HARD_ALLOWANCE,
    MAX_CONTINUATION_GRANTS,
)
from napari_vipp.core.workflow_agent_probes import (
    MAX_DIAGNOSTIC_PROBES,
    MAX_PROBE_SNAPSHOT_BYTES,
    DiagnosticProbe,
)
from napari_vipp.core.workflow_assessment import (
    MAX_DIAGNOSTIC_ELEMENTS,
    MAX_MESH_TRIANGLES,
    MAX_SNAPSHOT_BYTES,
    AssessmentSnapshot,
    _CacheBinding,
    _data_hash,
    _MeshSnapshot,
    _Output,
    _require_owned_readonly,
    _state_hash,
    _table_bytes,
    _table_scalar,
    _validate_image,
)
from napari_vipp.core.workflow_assistant_session import (
    SavedAssistantSession,
    make_checkpoint,
)
from napari_vipp.core.workflow_investigation import (
    CompletedCandidateBundle,
    InvestigationCandidate,
    InvestigationError,
    InvestigationLimits,
    InvestigationSession,
    _binding_bytes,
    _graph_fingerprint,
    _json,
)
from napari_vipp.core.workflow_investigation_experiments import ExperimentLedger
from napari_vipp.core.workflow_investigation_notebook import (
    InvestigationNotebook,
    NotebookPreviewCache,
)
from napari_vipp.core.workflow_proposal_catalogue import build_catalogue
from napari_vipp.core.workflow_proposal_images import ImagePreview
from napari_vipp.core.workflow_proposals import ProposalDialogue, validate_proposal
from napari_vipp.core.workflow_refinement import (
    REFINEMENT_SAFE_OPERATION_IDS,
    PreservationConstraint,
    assert_refinement_graph_safe,
    capture_preservation_constraint,
)

KIND = "napari-vipp-assistant-session"
VERSION = 1
_CHECKPOINT_FIELDS = {
    "kind",
    "version",
    "session_id",
    "resume",
    "dialogue",
    "options",
    "limits",
    "usage",
    "original",
    "notebook",
    "history",
    "experiments",
    "runtime",
    "scientific",
    "ui_state",
}
_TABLE_METADATA = (
    "histogram_metadata",
    "detection_metadata",
    "observation_metadata",
    "tracking_metadata",
)
_TABLE_STATE_COUNTS = (
    "numeric_value_count",
    "nan_value_count",
    "infinite_value_count",
    "missing_value_count",
    "nonfinite_row_count",
)


class AssistantCheckpointError(ValueError):
    """An archive cannot safely represent or restore the requested runtime."""


class _UnsupportedEvidence(AssistantCheckpointError):
    """The journal can be kept, but this scientific cache has no exact codec."""


@dataclass(frozen=True, slots=True)
class InvestigationRestoreResult:
    session: InvestigationSession | None
    resumable: bool
    reason: str


def _fields(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise AssistantCheckpointError(f"Invalid saved {label} fields.")
    return value


def _integer(value, label, maximum, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AssistantCheckpointError(f"Invalid saved {label}.")
    return value


def _text(value, label, maximum=4000, *, empty=True):
    if (
        type(value) is not str
        or len(value) > maximum
        or not empty
        and not value.strip()
    ):
        raise AssistantCheckpointError(f"Invalid saved {label}.")
    return value


def _list(value, label, maximum):
    if type(value) is not list or len(value) > maximum:
        raise AssistantCheckpointError(f"Invalid saved {label}.")
    return value


def _number(value, label, maximum):
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0 <= value <= maximum
    ):
        raise AssistantCheckpointError(f"Invalid saved {label}.")
    return value


def _graph(graph):
    if graph is None:
        return None
    return {
        "nodes": [
            {
                "id": n.id,
                "operation_id": n.operation_id,
                "params": n.params,
                "execution_mode": n.execution_mode,
            }
            for n in graph.nodes
        ],
        "connections": [
            {
                "source_id": e.source_id,
                "target_id": e.target_id,
                "target_port": e.target_port,
                "source_port": e.source_port,
                "tunnel_name": e.tunnel_name,
            }
            for e in graph.connections
        ],
        "output_tunnels": [
            {"name": t.name, "source_id": t.source_id, "source_port": t.source_port}
            for t in graph.output_tunnels
        ],
    }


def _read_graph(value):
    if value is None:
        return None
    _fields(value, {"nodes", "connections", "output_tunnels"}, "graph")
    nodes, edges, tunnels = [], [], []
    for row in _list(value["nodes"], "graph nodes", 512):
        _fields(row, {"id", "operation_id", "params", "execution_mode"}, "graph node")
        if type(row["params"]) is not dict:
            raise AssistantCheckpointError("Invalid saved node parameters.")
        nodes.append(NodeSnapshot(**row))
    for row in _list(value["connections"], "graph connections", 2048):
        _fields(
            row,
            {"source_id", "target_id", "target_port", "source_port", "tunnel_name"},
            "wire",
        )
        edges.append(GraphConnection(**row))
    for row in _list(value["output_tunnels"], "graph tunnels", 512):
        _fields(row, {"name", "source_id", "source_port"}, "output tunnel")
        tunnels.append(OutputTunnel(**row))
    graph = GraphSnapshot(nodes, edges, tunnels)
    graph.to_pipeline()  # Ordinary graph/parameter validation; never executes.
    return graph


def _mesh_state(state):
    return {
        "vertex_count": state.vertex_count,
        "triangle_count": state.triangle_count,
        "spatial_axes": [a.to_dict() for a in state.spatial_axes],
        "source_shape": list(state.source_shape),
        "boundary": state.boundary,
        "source_name": state.source_name,
        "history": list(state.history),
        "kind": state.kind,
        "metadata_source": state.metadata_source,
        "object_count": state.object_count,
        "processing_history": list(state.processing_history),
    }


def _state(state):
    if type(state) is ImageState:
        return {"kind": "image", "value": state.to_dict()}
    if type(state) is MeshState:
        return {"kind": "mesh", "value": _mesh_state(state)}
    if type(state) is TableState:
        if any(getattr(state, name) is not None for name in _TABLE_METADATA):
            raise _UnsupportedEvidence(
                "Specialised table metadata has no exact session codec yet."
            )
        return {"kind": "table", "value": state.to_dict()}
    if state is None:
        return {"kind": "none", "value": None}
    raise _UnsupportedEvidence(
        "This result metadata type has no exact session codec yet."
    )


def _read_state(row):
    _fields(row, {"kind", "value"}, "result metadata")
    value = row["value"]
    if row["kind"] == "none" and value is None:
        return None
    if row["kind"] == "image":
        state = ImageState.from_dict(value)
    elif row["kind"] == "mesh":
        expected = {
            "vertex_count",
            "triangle_count",
            "spatial_axes",
            "source_shape",
            "boundary",
            "source_name",
            "history",
            "kind",
            "metadata_source",
            "object_count",
            "processing_history",
        }
        _fields(value, expected, "mesh metadata")
        _integer(value["vertex_count"], "mesh vertex count", MAX_DIAGNOSTIC_ELEMENTS)
        _integer(value["triangle_count"], "mesh triangle count", MAX_MESH_TRIANGLES)
        _integer(value["object_count"], "mesh object count", MAX_MESH_TRIANGLES)
        if (
            len(_list(value["spatial_axes"], "mesh spatial axes", 3)) != 3
            or len(_list(value["source_shape"], "mesh source shape", 3)) != 3
        ):
            raise AssistantCheckpointError(
                "Saved meshes require three explicit spatial axes."
            )
        for size in value["source_shape"]:
            _integer(size, "mesh source extent", MAX_DIAGNOSTIC_ELEMENTS, 1)
        state = MeshState(
            **(
                value
                | {
                    "spatial_axes": tuple(
                        AxisMetadata.from_dict(a) for a in value["spatial_axes"]
                    ),
                }
            )
        )
    elif row["kind"] == "table":
        required = {
            "row_count",
            "column_count",
            "columns",
            "kind",
            "table_kind",
            "metadata_source",
            "source_name",
            "history",
            "column_units",
        }
        optional = {*_TABLE_STATE_COUNTS, "nonfinite_columns"}
        if (
            type(value) is not dict
            or not required <= value.keys() <= required | optional
        ):
            raise AssistantCheckpointError("Unsupported saved table metadata.")
        if type(value["column_units"]) is not dict:
            raise AssistantCheckpointError("Invalid saved table units.")
        _integer(value["row_count"], "table row count", MAX_DIAGNOSTIC_ELEMENTS)
        _integer(value["column_count"], "table column count", 512)
        state = TableState(
            **(
                value
                | {
                    "columns": tuple(value["columns"]),
                    "history": tuple(value["history"]),
                    "column_units": tuple(value["column_units"].items()),
                    "nonfinite_columns": tuple(value.get("nonfinite_columns", ())),
                }
            )
        )
    else:
        raise AssistantCheckpointError("Unknown saved result metadata type.")
    if state is None or _json(_state(state)) != _json(row):
        raise AssistantCheckpointError(
            "Saved metadata cannot round-trip without a change."
        )
    return state


class _Writer:
    def __init__(self):
        self.blobs = {}

    def blob(self, data, folder, suffix):
        key = f"{folder}/{hashlib.sha256(data).hexdigest()}.{suffix}"
        self.blobs[key] = data
        return key

    def array(self, data):
        if (
            type(data) is not np.ndarray
            or data.dtype.kind not in "biuf"
            or data.dtype.itemsize > 8
        ):
            raise _UnsupportedEvidence("This array dtype has no exact session codec.")
        _require_owned_readonly(data)
        if data.nbytes > MAX_SNAPSHOT_BYTES or data.size > MAX_DIAGNOSTIC_ELEMENTS:
            raise AssistantCheckpointError(
                "Saved scientific arrays exceed the existing evidence bound."
            )
        output = BytesIO()
        np.save(output, data, allow_pickle=False)
        return self.blob(output.getvalue(), "arrays", "npy")

    def binding(self, binding):
        data = binding.data
        if (
            _data_hash(data) != binding.fingerprint
            or _state_hash(binding.state) != binding.state_fingerprint
        ):
            raise AssistantCheckpointError(
                "Frozen scientific evidence changed before saving."
            )
        if type(data) is np.ndarray:
            value = {"kind": "array", "array": self.array(data)}
        elif type(data) is _MeshSnapshot:
            value = {
                "kind": "mesh",
                "vertices": self.array(data.vertices),
                "faces": self.array(data.faces),
                "face_object_ids": self.array(data.face_object_ids),
                "object_ids": list(data.object_ids),
                "colors": [list(c) for c in data.colors],
                "state": _mesh_state(data.state),
            }
        elif type(data) is TableData:
            if any(getattr(data, name) is not None for name in _TABLE_METADATA):
                raise _UnsupportedEvidence(
                    "Specialised table results have no exact session codec yet."
                )
            _table_bytes(data)
            value = {
                "kind": "table",
                "columns": list(data.columns),
                "rows": [[_table_scalar(v) for v in row] for row in data.rows],
                "name": data.name,
                "table_kind": data.table_kind,
                "source_name": data.source_name,
                "column_units": [list(pair) for pair in data.column_units],
            }
        else:
            raise _UnsupportedEvidence(
                "This scientific result has no exact session codec yet."
            )
        return {
            "node_id": binding.node_id,
            "port": binding.port,
            "data": value,
            "state": _state(binding.state),
            "sha256": binding.fingerprint,
            "state_sha256": binding.state_fingerprint,
        }

    def preview(self, preview):
        return {
            "label": preview.label,
            "description": preview.description,
            "blob": self.blob(preview.png_bytes, "previews", "png"),
        }

    def snapshot(self, snap):
        return {
            "assessment_id": snap.assessment_id,
            "run_id": snap.run_id,
            "source_id": snap.source_id,
            "candidate_node_ids": list(snap.candidate_node_ids),
            "graph": _graph(snap.baseline),
            "graph_sha256": snap.baseline_fingerprint,
            "source": self.binding(snap._source),
            "outputs": [
                {
                    "node_id": o.node_id,
                    "port": o.port,
                    "role": o.role,
                    "binding": None if o.binding is None else self.binding(o.binding),
                    "unavailable": o.unavailable,
                }
                for o in snap._outputs
            ],
            "omitted": [list(row) for row in snap._omitted],
            "uncalculated": list(snap._uncalculated),
        }


class _Reader:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.arrays = {}

    def array(self, ref):
        if (
            type(ref) is not str
            or not ref.startswith("arrays/")
            or not ref.endswith(".npy")
        ):
            raise AssistantCheckpointError("Invalid saved array reference.")
        if ref not in self.arrays:
            data = np.load(
                BytesIO(self.checkpoint.blob_bytes(ref)),
                allow_pickle=False,
                max_header_size=16384,
            )
            if (
                type(data) is not np.ndarray
                or data.dtype.kind not in "biuf"
                or data.dtype.itemsize > 8
                or data.size > MAX_DIAGNOSTIC_ELEMENTS
                or data.nbytes > MAX_SNAPSHOT_BYTES
            ):
                raise AssistantCheckpointError(
                    "Saved array exceeds the scientific evidence contract."
                )
            if not data.flags.owndata or not data.flags.c_contiguous:
                # Order is a storage property; this preserves every value/dtype.
                data = np.array(data, copy=True, order="C")
            data.flags.writeable = False
            self.arrays[ref] = data
        return self.arrays[ref]

    def binding(self, row):
        _fields(
            row,
            {"node_id", "port", "data", "state", "sha256", "state_sha256"},
            "cache binding",
        )
        _text(row["node_id"], "output node", 200, empty=False)
        _integer(row["port"], "output port", 64)
        value = row["data"]
        if type(value) is not dict:
            raise AssistantCheckpointError("Invalid saved scientific result.")
        kind = value.get("kind")
        state = _read_state(row["state"])
        if kind == "array":
            _fields(value, {"kind", "array"}, "array result")
            data = self.array(value["array"])
            _validate_image(data, state)
        elif kind == "mesh":
            _fields(
                value,
                {
                    "kind",
                    "vertices",
                    "faces",
                    "face_object_ids",
                    "object_ids",
                    "colors",
                    "state",
                },
                "mesh result",
            )
            vertices, faces, ids = (
                self.array(value[name])
                for name in ("vertices", "faces", "face_object_ids")
            )
            if (
                vertices.dtype != np.dtype("float64")
                or faces.dtype != np.dtype("int64")
                or ids.dtype != np.dtype("int64")
            ):
                raise AssistantCheckpointError(
                    "Saved mesh geometry must retain its exact float64/int64 dtypes."
                )
            if (
                vertices.ndim != 2
                or vertices.shape[1:] != (3,)
                or not np.isfinite(vertices).all()
                or faces.ndim != 2
                or faces.shape[1:] != (3,)
                or len(faces) > MAX_MESH_TRIANGLES
                or ids.shape != (len(faces),)
            ):
                raise AssistantCheckpointError("Invalid saved mesh geometry.")
            if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
                raise AssistantCheckpointError(
                    "Saved mesh indices leave the vertex array."
                )
            objects = tuple(
                _integer(v, "mesh object ID", np.iinfo(np.int64).max, 1)
                for v in _list(
                    value["object_ids"], "mesh object IDs", MAX_MESH_TRIANGLES
                )
            )
            colors = tuple(
                tuple(c)
                for c in _list(value["colors"], "mesh colours", MAX_MESH_TRIANGLES)
            )
            if (
                objects != tuple(int(v) for v in np.unique(ids))
                or len(colors) != len(objects)
                or any(
                    len(c) != 4
                    or any(
                        type(v) not in {float, int}
                        or not math.isfinite(v)
                        or not 0 <= v <= 1
                        for v in c
                    )
                    for c in colors
                )
            ):
                raise AssistantCheckpointError(
                    "Invalid saved mesh object/colour binding."
                )
            mesh_state = _read_state({"kind": "mesh", "value": value["state"]})
            if (
                type(state) is not MeshState
                or _state_hash(state) != _state_hash(mesh_state)
                or state.vertex_count != len(vertices)
                or state.triangle_count != len(faces)
                or state.object_count != len(objects)
            ):
                raise AssistantCheckpointError(
                    "Saved mesh metadata and geometry disagree."
                )
            data = _MeshSnapshot(vertices, faces, ids, objects, colors, mesh_state)
        elif kind == "table":
            _fields(
                value,
                {
                    "kind",
                    "columns",
                    "rows",
                    "name",
                    "table_kind",
                    "source_name",
                    "column_units",
                },
                "table result",
            )

            def cell(v):
                if type(v) is dict:
                    _fields(v, {"nonfinite"}, "non-finite table value")
                    if v["nonfinite"] not in {"nan", "inf", "-inf"}:
                        raise AssistantCheckpointError(
                            "Unknown non-finite table value."
                        )
                    return float(v["nonfinite"])
                if v is not None and type(v) not in {str, int, bool, float}:
                    raise AssistantCheckpointError(
                        "Saved table cells must be primitive values."
                    )
                return v

            columns = tuple(
                _text(c, "table column", 400)
                for c in _list(value["columns"], "table columns", 512)
            )
            rows = tuple(
                tuple(cell(v) for v in r)
                for r in _list(value["rows"], "table rows", MAX_DIAGNOSTIC_ELEMENTS)
            )
            data = TableData(
                columns,
                rows,
                value["name"],
                value["table_kind"],
                value["source_name"],
                tuple(tuple(p) for p in value["column_units"]),
            )
            _table_bytes(data)
            if (
                type(state) is not TableState
                or state.row_count != len(rows)
                or state.column_count != len(columns)
                or state.columns != columns
            ):
                raise AssistantCheckpointError(
                    "Saved table metadata and cells disagree."
                )
        else:
            raise AssistantCheckpointError("Unsupported saved scientific result.")
        if (
            _data_hash(data) != row["sha256"]
            or _state_hash(state) != row["state_sha256"]
        ):
            raise AssistantCheckpointError(
                "Saved result bytes or metadata do not match their fingerprints."
            )
        # These live pointers refer ONLY to this archive's owned immutable copy.
        # No real workflow cache ticket/signature is recreated.
        return _CacheBinding(
            row["node_id"],
            row["port"],
            data,
            data,
            state,
            row["sha256"],
            row["state_sha256"],
        )

    def preview(self, row):
        _fields(row, {"label", "description", "blob"}, "preview")
        _text(row["label"], "preview label", 2000)
        _text(row["description"], "preview description", 16000)
        if (
            type(row["blob"]) is not str
            or not row["blob"].startswith("previews/")
            or not row["blob"].endswith(".png")
        ):
            raise AssistantCheckpointError("Invalid saved preview reference.")
        return ImagePreview(
            row["label"], self.checkpoint.blob_bytes(row["blob"]), row["description"]
        )

    def snapshot(self, row):
        expected = {
            "assessment_id",
            "run_id",
            "source_id",
            "candidate_node_ids",
            "graph",
            "graph_sha256",
            "source",
            "outputs",
            "omitted",
            "uncalculated",
        }
        _fields(row, expected, "accepted snapshot")
        graph = _read_graph(row["graph"])
        if graph is None or _graph_fingerprint(graph) != row["graph_sha256"]:
            raise AssistantCheckpointError("Saved accepted graph fingerprint changed.")
        source = self.binding(row["source"])
        outputs, total = [], _binding_bytes(source)
        for value in _list(row["outputs"], "accepted outputs", 64):
            _fields(
                value,
                {"node_id", "port", "role", "binding", "unavailable"},
                "accepted output",
            )
            binding = (
                None if value["binding"] is None else self.binding(value["binding"])
            )
            if binding is not None:
                if (binding.node_id, binding.port) != (value["node_id"], value["port"]):
                    raise AssistantCheckpointError(
                        "Saved accepted port binding changed."
                    )
                total += _binding_bytes(binding)
            outputs.append(
                _Output(
                    value["node_id"],
                    value["port"],
                    value["role"],
                    binding,
                    value["unavailable"],
                )
            )
        if total > MAX_SNAPSHOT_BYTES:
            raise AssistantCheckpointError("Saved accepted snapshot exceeds 64 MiB.")
        return AssessmentSnapshot(
            row["assessment_id"],
            row["run_id"],
            row["source_id"],
            tuple(row["candidate_node_ids"]),
            graph,
            row["graph_sha256"],
            source,
            tuple(outputs),
            tuple(tuple(v) for v in row["omitted"]),
            tuple(row["uncalculated"]),
            (),
        )


def _ui_state(value):
    """Only conversation drafts, never provider/profile/settings dictionaries."""
    defaults = {
        "version": 1,
        "goal": "",
        "chat": [],
        "replies": [],
        "clarifications": [],
        "drafts": [],
        "draft_index": -1,
        "discussion_draft": "",
        "pause_answer_draft": "",
        "initiative": 2,
        "source_id": None,
        "stage": 1,
    }
    value = {} if value is None else value
    if type(value) is not dict or not value.keys() <= defaults.keys():
        raise AssistantCheckpointError(
            "Save only explicit conversation UI state, not connection settings."
        )
    value = defaults | value
    if type(value["version"]) is not int or value["version"] != 1:
        raise AssistantCheckpointError("Unsupported saved conversation version.")
    _text(value["goal"], "goal", 4000)
    for name in ("discussion_draft", "pause_answer_draft"):
        _text(value[name], name, 262144)
    _integer(value["initiative"], "judgment choice", 2)
    _integer(value["stage"], "assistant stage", 2)
    if value["source_id"] is not None:
        _text(value["source_id"], "historical source ID", 128)
    for name in ("chat", "clarifications"):
        for row in _list(value[name], name, 2000):
            if type(row) is not list or len(row) != 2:
                raise AssistantCheckpointError("Invalid saved conversation row.")
            for item in row:
                _text(item, "conversation text", 262144)
    seen = set()
    for row in _list(value["replies"], "reply attributions", len(value["chat"])):
        _fields(row, {"index", "question", "answer", "is_question"}, "reply")
        index = _integer(row["index"], "reply index", len(value["chat"]) - 1)
        if index in seen or type(row["is_question"]) is not bool:
            raise AssistantCheckpointError("Invalid or duplicate reply attribution.")
        seen.add(index)
        _text(row["question"], "quoted question", 262144)
        _text(row["answer"], "literal answer", 262144)
    for row in _list(value["drafts"], "plan drafts", 100):
        _fields(
            row,
            {
                "summary",
                "steps",
                "assumptions",
                "questions",
                "proposal_text",
                "plan_text",
            },
            "plan draft",
        )
        for name in ("summary", "proposal_text", "plan_text"):
            _text(row[name], name, 262144)
        for name in ("assumptions", "questions"):
            for item in _list(row[name], name, 128):
                _text(item, name, 262144)
        for step in _list(row["steps"], "draft steps", 128):
            if type(step) is not list or len(step) != 2:
                raise AssistantCheckpointError("Invalid saved plan step.")
            for item in step:
                _text(item, "plan step text", 262144)
    _integer(value["draft_index"], "draft index", max(0, len(value["drafts"]) - 1), -1)
    return json.loads(_json(value))


def _contracts(operation_ids):
    available = {row["operation_id"]: row for row in build_catalogue()["operations"]}
    return {
        name: hashlib.sha256(
            _json(
                available[name]
                | {"trial_allowed": name in REFINEMENT_SAFE_OPERATION_IDS}
            )
        ).hexdigest()
        for name in sorted(operation_ids)
    }


def checkpoint_investigation(session, *, ui_state=None):
    """Detach a settled session; no model, Run, render, pause or permission change.

    Call the owner's ordinary Pause and wait for its worker first. Unsupported
    scientific codecs produce an explicitly history-only archive. A busy session
    is rejected rather than saving a possibly half-completed trial as complete.
    """
    if session is None:
        return make_checkpoint(
            {
                "kind": "napari-vipp-assistant-history",
                "version": VERSION,
                "session_id": uuid4().hex,
                "resume": {
                    "supported": False,
                    "reason": (
                        "This conversation has no paused investigation to continue. "
                        "Its saved drafts are history, not Add/Run authority."
                    ),
                },
                "ui_state": _ui_state(ui_state),
            },
            {},
        )
    if type(session) is not InvestigationSession:
        raise AssistantCheckpointError("Save one bounded InvestigationSession.")
    with session._owner_lease():
        if session._active_action or not (
            session.paused
            or session.finished
            or session.stop_reason
            or session._pending_finish_review is not None
        ):
            raise AssistantCheckpointError(
                "Pause after the current step before saving the assistant session."
            )
        now = session._now()
        session.limits.__post_init__()
        session._assert_approved_allowance()
        writer = _Writer()
        options = {
            "include_input_images": session.include_input_images,
            "include_output_images": session.include_output_images,
            "require_experiment_plan": session.require_experiment_plan,
            "review_before_finish": session.review_before_finish,
            "continuation_enabled": session.continuation_enabled,
        }
        usage = session._usage() | {
            "active_seconds_used": session._active_seconds(now),
            "paused_seconds": session._paused_seconds
            + (0 if session._pause_started is None else now - session._pause_started),
            "allowance_grants": session._allowance_grants,
            "continue_requests": session._continue_requests,
            "calculation_repairs": session._calculation_repairs,
            "control_revision": session._control_revision,
        }
        runtime = {
            "retained_candidate_id": session.retained_candidate_id,
            "finished": session.finished,
            "stop_reason": session.stop_reason,
            "pause_kind": session._pause_kind,
            "pause_reason": session._pause_reason,
            "pending_question": session._pending_question,
            "pending_finish_review": None
            if session._pending_finish_review is None
            else json.loads(session._pending_finish_review),
            "latest_finish_feedback": None
            if session._latest_finish_feedback is None
            else json.loads(session._latest_finish_feedback),
            "seed_count": session._seed_count,
            "record_refs": {
                "probes": session._probe_records,
                "ledger": session._ledger_records,
                "candidates": session._candidate_records,
                "latest_action": session._latest_action_record,
            },
        }
        original = {
            "source_id": session._source_id,
            "graph": _graph(session._original.baseline),
            "graph_sha256": session._original.baseline_fingerprint,
            "source_fingerprint": session._original.source_fingerprint,
            "source_sha256": session._source.fingerprint,
            "source_state_sha256": session._source.state_fingerprint,
            "proposal_context_id": session.proposal_context_id,
            "compute_request": session.compute_request.as_dict(),
        }
        resumable = session.continuation_enabled and not session.stop_reason
        reason = "Ready for an explicit, freshly authorised continuation."
        if session._permission_binding != (
            session.include_input_images,
            session.include_output_images,
        ):
            resumable = False
            reason = (
                "The sharing choices changed after this evidence was collected. "
                "Keep the session as history or start a newly authorised investigation."
            )
        if not resumable:
            if session.stop_reason or not session.continuation_enabled:
                reason = (
                    "This stopped or legacy session is history only; "
                    "it cannot silently restart."
                )
        try:
            source = writer.binding(session._source)
            candidates = [
                {
                    "candidate_id": c.candidate_id,
                    "run_id": c.run_id,
                    "status": c.status,
                    "graph": _graph(c.graph),
                    "recipe": c.recipe,
                    "evidence": c.evidence,
                    "outputs": [
                        {"role": role, "binding": writer.binding(b)}
                        for role, b in c._outputs
                    ],
                    "source_fingerprint": c.source_fingerprint,
                    "output_roles": c.output_roles,
                    "key_to_node": [list(v) for v in c._key_to_node],
                }
                for c in session.candidates
            ]
            probes = [
                {
                    "probe_id": p.probe_id,
                    "status": p.status,
                    "evidence": p.evidence,
                    "outputs": [
                        {"role": role, "binding": writer.binding(b), "operation_id": op}
                        for role, b, op in p.outputs
                    ],
                    "snapshot_bytes": p.snapshot_bytes,
                }
                for p in session._probes
            ]
            bundle = session._accepted_bundle
            accepted = (
                None
                if bundle is None
                else {
                    "snapshot": writer.snapshot(bundle.snapshot),
                    "recipe": bundle.recipe,
                    "roles": bundle.output_roles,
                    "key_to_node": [list(v) for v in bundle._key_to_node],
                    "binding": bundle.public_binding(),
                    "role_outputs": [
                        {
                            "node_id": o.node_id,
                            "port": o.port,
                            "role": o.role,
                            "binding": writer.binding(o.binding),
                            "unavailable": o.unavailable,
                        }
                        for o in bundle._role_outputs
                    ],
                    "protected_role": bundle.protected_role,
                    "preservation": None
                    if session._preservation is None
                    else {
                        "primary_ref": list(session._preservation.primary_ref),
                        "region_ref": None
                        if session._preservation.region_ref is None
                        else list(session._preservation.region_ref),
                        "protected": writer.array(session._preservation._protected),
                        "protected_sha256": session._preservation._protected_hash,
                        "constraint_id": session._preservation.constraint_id,
                        "public_contract": session._preservation.public_contract(),
                    },
                }
            )
            cache = []
            for view_id, key in session._view_keys.items():
                images = session._preview_cache.lookup(key)
                if images is None:
                    raise AssistantCheckpointError("An exact cached view is missing.")
                cache.append(
                    {
                        "view_id": view_id,
                        "key": key,
                        "previews": [writer.preview(p) for p in images],
                    }
                )
            previews = {
                "board": [writer.preview(p) for p in session._images],
                "initial_source": [
                    writer.preview(p) for p in session._initial_source_previews
                ],
                "cache": cache,
                "recent_view_ids": list(session._recent_view_ids),
                "pinned_view_ids": list(session._pinned_view_ids),
                "omissions": session._automatic_preview_omissions,
                "owners": [
                    {
                        "preview": writer.preview(p),
                        "candidates": sorted(session._preview_owners.get(id(p), ())),
                    }
                    for p in session._images
                    + session._initial_source_previews
                    + tuple(
                        p
                        for images in session._preview_cache._entries.values()
                        for p in images
                    )
                ],
            }
            operations = {
                n["operation_id"]
                for c in candidates
                if c["status"] in {"completed", "rejected"}
                for n in c["recipe"].get("nodes", ())
            }
            operations.update(
                op for p in session._probes for _role, _b, op in p.outputs
            )
            scientific = {
                "source": source,
                "candidates": candidates,
                "probes": probes,
                "accepted": accepted,
                "previews": previews,
                "operation_contracts": _contracts(operations),
            }
        except _UnsupportedEvidence as exc:
            scientific, writer = None, _Writer()
            resumable, reason = (
                False,
                str(exc) + " The complete journal remains available locally.",
            )
        ui = _ui_state(ui_state)
        if ui["goal"] and ui["goal"] != session.dialogue.objective:
            resumable = False
            reason = (
                "The goal draft differs from this investigation's original goal. "
                "The saved conversation remains history; start a new investigation "
                "for the changed goal."
            )
        document = {
            "kind": KIND,
            "version": VERSION,
            "session_id": session.session_id,
            "resume": {"supported": resumable, "reason": reason},
            "dialogue": {
                "objective": session.dialogue.objective,
                "clarifications": [list(v) for v in session.dialogue.clarifications],
                "deferred_questions": list(session.dialogue.deferred_questions),
            },
            "options": options,
            "limits": {name: getattr(session.limits, name) for name in HARD_ALLOWANCE},
            "usage": usage,
            "original": original,
            "notebook": session.export_notebook(),
            "history": [json.loads(v) for v in session._history],
            "experiments": list(session.experiment_ledger),
            "runtime": runtime,
            "scientific": scientific,
            "ui_state": ui,
        }
        return make_checkpoint(json.loads(_json(document)), writer.blobs)


def _restore_notebook(value, session_id):
    _fields(
        value,
        {
            "version",
            "session_id",
            "record_index",
            "records",
            "pinned_record_refs",
            "total_payload_bytes",
            "notebook_sha256",
        },
        "notebook",
    )
    if (
        value["version"] != 1
        or type(value["version"]) is not int
        or value["session_id"] != session_id
    ):
        raise AssistantCheckpointError("Saved notebook identity/version changed.")
    notebook = InvestigationNotebook(session_id)
    if (
        type(value["record_index"]) is not list
        or type(value["records"]) is not list
        or len(value["record_index"]) != len(value["records"])
    ):
        raise AssistantCheckpointError("Saved notebook index is incomplete.")
    for descriptor, record in zip(value["record_index"], value["records"], strict=True):
        _fields(record, {"record_id", "sha256", "payload"}, "notebook record")
        actual = notebook.append(
            descriptor["kind"], record["payload"], owner=descriptor["owner"]
        )
        if actual != descriptor or {k: actual[k] for k in ("record_id", "sha256")} != {
            k: record[k] for k in ("record_id", "sha256")
        }:
            raise AssistantCheckpointError(
                "Saved notebook payload/index fingerprint changed."
            )
    notebook.set_pins(value["pinned_record_refs"])
    if notebook.export() != value:
        raise AssistantCheckpointError("Saved complete notebook fingerprint changed.")
    return notebook


def _restore_ledger(rows, limits):
    ledger = ExperimentLedger(limits.max_attempts)
    for index, row in enumerate(_list(rows, "experiments", 5), 1):
        _fields(
            row,
            {"experiment_id", "state", "plan", "candidate_id", "assessment"},
            "experiment",
        )
        plan = row["plan"]
        if type(plan) is not dict or "meaning" not in plan:
            raise AssistantCheckpointError("Invalid historical experiment plan.")
        ledger.plan(
            validate_evidence_refs=lambda *_a, **_k: None,
            **{k: v for k, v in plan.items() if k != "meaning"},
        )
        identifier = f"experiment{index}"
        if row["candidate_id"] is not None:
            ledger.bind_trial(identifier, row["candidate_id"])
        assessment = row["assessment"]
        if assessment is not None:
            checks = [
                {
                    "check_id": c["check_id"],
                    "outcome": c["outcome"],
                    "evidence_refs": c["evidence_refs"],
                    "assessment": c["model_assessment"],
                }
                for c in assessment["check_results"]
            ]
            ledger.assess(
                experiment_id=identifier,
                candidate_id=row["candidate_id"],
                check_results=checks,
                unresolved_criteria=assessment["unresolved_criteria"],
                validate_evidence_refs=lambda *_a, **_k: None,
            )
        if ledger.get(identifier) != row:
            raise AssistantCheckpointError(
                "Saved historical experiment does not match its strict contract."
            )
    # Validation above is structural historical replay, NOT issuance of any
    # previous preview/numerical transmission authority to the new runtime.
    return ledger


def _restore_candidates(rows, reader, engine):
    candidates = []
    for i, row in enumerate(_list(rows, "candidates", 6)):
        _fields(
            row,
            {
                "candidate_id",
                "run_id",
                "status",
                "graph",
                "recipe",
                "evidence",
                "outputs",
                "source_fingerprint",
                "output_roles",
                "key_to_node",
            },
            "candidate",
        )
        expected = f"candidate{i if engine._seed_count else i + 1}"
        if (
            row["candidate_id"] != expected
            or row["status"] not in {"failed", "completed", "rejected"}
            or row["source_fingerprint"] != engine._original.source_fingerprint
        ):
            raise AssistantCheckpointError(
                "Saved candidate identity/status/source changed."
            )
        graph = _read_graph(row["graph"])
        outputs, total, roles = [], engine._source.data.nbytes, set()
        mapping = dict(row["key_to_node"])
        if len(mapping) != len(row["key_to_node"]):
            raise AssistantCheckpointError("Saved candidate node keys are duplicated.")
        for output in _list(row["outputs"], "candidate outputs", 8):
            _fields(output, {"role", "binding"}, "candidate output")
            role = _text(output["role"], "output role", 80, empty=False)
            binding = reader.binding(output["binding"])
            if role in roles:
                raise AssistantCheckpointError(
                    "Saved candidate output roles are duplicated."
                )
            roles.add(role)
            total += _binding_bytes(binding)
            outputs.append((role, binding))
        if total > MAX_SNAPSHOT_BYTES:
            raise AssistantCheckpointError(
                "Saved candidate source/results exceed 64 MiB."
            )
        if row["status"] in {"completed", "rejected"}:
            prepared = validate_proposal(row["recipe"], engine._context)
            if not prepared.can_apply or graph is None:
                raise AssistantCheckpointError(
                    "A saved completed candidate is not a valid ordinary proposal."
                )
            if row["candidate_id"] != "candidate0":
                assert_refinement_graph_safe(graph, "input")
            _verify_recipe_graph(
                row["recipe"],
                graph,
                mapping,
                engine._source_id if row["candidate_id"] == "candidate0" else "input",
                original=row["candidate_id"] == "candidate0",
            )
            declared = _list(row["output_roles"], "candidate roles", 8)
            if len(declared) != len(outputs):
                raise AssistantCheckpointError(
                    "Saved candidate role inventory changed."
                )
            for role_row, (role, binding) in zip(declared, outputs, strict=True):
                _fields(role_row, {"role", "node", "port"}, "candidate role")
                if role_row["role"] != role or (
                    mapping.get(role_row["node"]),
                    role_row["port"],
                ) != (binding.node_id, binding.port):
                    raise AssistantCheckpointError(
                        "Saved candidate role/key/port binding changed."
                    )
            if (
                row["evidence"].get("candidate_id") != row["candidate_id"]
                or row["evidence"].get("status") != row["status"]
                or row["evidence"].get("source_sha256") != engine._source.fingerprint
            ):
                raise AssistantCheckpointError(
                    "Saved completed candidate receipt changed."
                )
            receipt_outputs = row["evidence"].get("outputs", [])
            if [(o["role"], o["sha256"]) for o in receipt_outputs] != [
                (role, b.fingerprint) for role, b in outputs
            ]:
                raise AssistantCheckpointError(
                    "Saved candidate output receipt changed."
                )
        elif outputs:
            raise AssistantCheckpointError(
                "Failed saved candidates cannot carry completed outputs."
            )
        candidates.append(
            InvestigationCandidate(
                row["candidate_id"],
                row["run_id"],
                row["status"],
                graph,
                _json(row["recipe"]),
                _json(row["evidence"]),
                tuple(outputs),
                row["source_fingerprint"],
                _json(row["output_roles"]),
                tuple(tuple(v) for v in row["key_to_node"]),
            )
        )
    return candidates


def _verify_recipe_graph(recipe, graph, mapping, source_id, *, original=False):
    """Bind public recipe keys/wires/parameters to the actual saved graph.

    Candidate0 may include unrelated unchanged original graph nodes; only its
    explicitly selected recipe is exposed. Later candidates are ordinary isolated
    trial graphs. No graph is installed in the user's live workflow here.
    """
    nodes = {n.id: n for n in graph.nodes}
    recipe_nodes = recipe["nodes"]
    if set(mapping) != {n["key"] for n in recipe_nodes} or len(
        set(mapping.values())
    ) != len(mapping):
        raise AssistantCheckpointError("Saved recipe node identities changed.")
    if not original and (
        set(nodes) != {source_id, *mapping.values()} or graph.output_tunnels
    ):
        raise AssistantCheckpointError(
            "Saved isolated trial has extra nodes or output tunnels."
        )
    for row in recipe_nodes:
        node = nodes.get(mapping[row["key"]])
        if node is None or node.operation_id != row["operation_id"]:
            raise AssistantCheckpointError(
                "Saved recipe operations differ from the graph."
            )
        if any(
            name not in node.params or _json(node.params[name]) != _json(value)
            for name, value in row["params"].items()
        ):
            raise AssistantCheckpointError(
                "Saved recipe parameters differ from the graph."
            )

    def wire(row):
        return (
            source_id if row["source"] == "input" else mapping[row["source"]],
            row["source_port"],
            mapping[row["target"]],
            row["target_port"],
            "",
        )

    expected = sorted(wire(row) for row in recipe["connections"])
    selected = set(mapping.values())
    actual = sorted(
        (e.source_id, e.source_port, e.target_id, e.target_port, e.tunnel_name)
        for e in graph.connections
        if e.target_id in selected
    )
    if expected != actual:
        raise AssistantCheckpointError("Saved recipe wiring differs from the graph.")


def _restore_accepted(row, reader, engine):
    if row is None:
        if engine._seed_count:
            raise AssistantCheckpointError("The saved accepted baseline is missing.")
        return None
    _fields(
        row,
        {
            "snapshot",
            "recipe",
            "roles",
            "key_to_node",
            "binding",
            "role_outputs",
            "protected_role",
            "preservation",
        },
        "accepted candidate",
    )
    snapshot = reader.snapshot(row["snapshot"])
    if (
        snapshot.baseline_fingerprint != engine._original.baseline_fingerprint
        or snapshot._source.fingerprint != engine._source.fingerprint
        or snapshot._source.state_fingerprint != engine._source.state_fingerprint
    ):
        raise AssistantCheckpointError(
            "The saved accepted baseline does not match the newly bound source/graph."
        )
    candidate0 = engine._candidate("candidate0")
    if (
        row["recipe"] | {"context_id": engine.proposal_context_id} != candidate0.recipe
        or row["roles"] != candidate0.output_roles
        or row["key_to_node"] != [list(v) for v in candidate0._key_to_node]
        or snapshot.run_id != candidate0.run_id
        or snapshot.baseline != candidate0.graph
    ):
        raise AssistantCheckpointError("Saved accepted candidate inventory changed.")
    role_outputs = []
    for value in _list(row["role_outputs"], "accepted role outputs", 8):
        _fields(
            value,
            {"node_id", "port", "role", "binding", "unavailable"},
            "accepted role output",
        )
        binding = reader.binding(value["binding"])
        role_outputs.append(
            _Output(
                value["node_id"],
                value["port"],
                value["role"],
                binding,
                value["unavailable"],
            )
        )
    if [
        (o.role, o.node_id, o.port, o.binding.fingerprint, o.binding.state_fingerprint)
        for o in role_outputs
    ] != [
        (role, b.node_id, b.port, b.fingerprint, b.state_fingerprint)
        for role, b in candidate0._outputs
    ]:
        raise AssistantCheckpointError(
            "Saved accepted role outputs differ from candidate0."
        )
    preservation = None
    if row["preservation"] is not None:
        saved = _fields(
            row["preservation"],
            {
                "primary_ref",
                "region_ref",
                "protected",
                "protected_sha256",
                "constraint_id",
                "public_contract",
            },
            "protected region",
        )
        protected = reader.array(saved["protected"])
        if (
            protected.dtype != np.dtype(bool)
            or protected.shape != engine._source.data.shape
            or _data_hash(protected) != saved["protected_sha256"]
        ):
            raise AssistantCheckpointError(
                "The saved protected mask bytes/grid changed."
            )
        preservation = PreservationConstraint(
            tuple(saved["primary_ref"]),
            None if saved["region_ref"] is None else tuple(saved["region_ref"]),
            snapshot,
            protected,
            ImageGrid.from_image_state(snapshot._source.state),
            saved["protected_sha256"],
            saved["constraint_id"],
        )
        preservation.assert_snapshot(snapshot)
        recaptured = capture_preservation_constraint(
            snapshot,
            preservation.primary_ref,
            preservation.region_ref,
            output_sharing=True,
        )
        if (
            recaptured._protected_hash != preservation._protected_hash
            or recaptured.constraint_id != preservation.constraint_id
        ):
            raise AssistantCheckpointError(
                "Saved protection is not its original mask intersection."
            )
        if _json(preservation.public_contract()) != _json(saved["public_contract"]):
            raise AssistantCheckpointError("The saved preservation contract changed.")
    bundle = CompletedCandidateBundle(
        snapshot,
        _json(row["recipe"]),
        _json(row["roles"]),
        tuple(tuple(v) for v in row["key_to_node"]),
        _json(row["binding"]),
        tuple(role_outputs),
        preservation,
        row["protected_role"],
        None,
    )
    # None is deliberate: an archive NEVER recreates a GUI adoption ticket.
    engine._preservation, engine._protected_role = preservation, row["protected_role"]
    return bundle


def _restore_previews(row, reader, engine):
    _fields(
        row,
        {
            "board",
            "initial_source",
            "cache",
            "recent_view_ids",
            "pinned_view_ids",
            "omissions",
            "owners",
        },
        "preview cache",
    )
    by_manifest = {}

    def preview(value):
        p = reader.preview(value)
        return by_manifest.setdefault(_json(p.manifest()), p)

    engine._images = NotebookPreviewCache.validate_board(
        tuple(preview(v) for v in row["board"])
    )
    engine._initial_source_previews = NotebookPreviewCache.validate_board(
        tuple(preview(v) for v in row["initial_source"])
    )
    engine._preview_cache = NotebookPreviewCache(
        engine.session_id, image_limit_bytes=engine.limits.max_image_bytes
    )
    engine._view_keys = {}
    for value in _list(row["cache"], "cached views", 64):
        _fields(value, {"view_id", "key", "previews"}, "cached view")
        key = value["key"]
        if (
            type(key) is not dict
            or key.get("session_id") != engine.session_id
            or key.get("source_sha256") != engine._source.fingerprint
            or key.get("source_state_sha256") != engine._source.state_fingerprint
            or key.get("permissions") != list(engine._permission_binding)
        ):
            raise AssistantCheckpointError(
                "Saved preview source/permission binding changed."
            )
        images = tuple(preview(v) for v in value["previews"])
        view_id = engine._preview_cache.store(key, images)
        if view_id != value["view_id"] or view_id in engine._view_keys:
            raise AssistantCheckpointError(
                "Saved immutable cached view identity changed."
            )
        engine._view_keys[view_id] = key
    for name, maximum in (("recent_view_ids", 64), ("pinned_view_ids", 7)):
        values = _list(row[name], name, maximum)
        if len(values) != len(set(values)) or any(
            v not in engine._view_keys for v in values
        ):
            raise AssistantCheckpointError(
                "Saved preview pins refer to unavailable views."
            )
        setattr(engine, "_" + name, tuple(values))
    engine._automatic_preview_omissions = _list(
        row["omissions"], "preview omissions", 32
    )
    candidate_ids = {c.candidate_id for c in engine.candidates} | {
        p.probe_id for p in engine._probes
    }
    engine._preview_owners = {}
    for value in _list(row["owners"], "preview owner bindings", 1024):
        _fields(value, {"preview", "candidates"}, "preview owners")
        p = preview(value["preview"])
        owners = value["candidates"]
        if type(owners) is not list or any(o not in candidate_ids for o in owners):
            raise AssistantCheckpointError("Invalid saved preview owner binding.")
        engine._preview_owners[id(p)] = frozenset(owners)
    # Cached images/pins remain exact local assets, NOT sent evidence authority.


def _ensure_eligible(engine, candidate):
    """Recheck adoption prerequisites; old status text is not an authority grant."""
    evidence = candidate.evidence
    if (
        candidate.status != "completed"
        or evidence.get("status") != "completed"
        or evidence.get("eligible_for_retention", True) is not True
        or evidence.get("run_id") != candidate.run_id
    ):
        raise AssistantCheckpointError(
            "The saved selection is not an eligible completed candidate."
        )
    execution = evidence.get("execution", {})
    if candidate.candidate_id == "candidate0":
        if (
            execution.get("already_completed") is not True
            or execution.get("new_calculation") is not False
            or execution.get("run_id") != candidate.run_id
        ):
            raise AssistantCheckpointError("Saved accepted-run evidence is incomplete.")
    elif execution.get("cleanup_succeeded") is not True:
        raise AssistantCheckpointError(
            "Saved selected trial has no successful cleanup receipt."
        )
    if engine._preservation is not None and candidate.candidate_id != "candidate0":
        roles = [
            row
            for row in candidate.output_roles
            if row["role"] == engine._protected_role
        ]
        if len(roles) != 1:
            raise AssistantCheckpointError(
                "Saved selected trial has no protected-role binding."
            )
        role = roles[0]
        reference = (candidate.key_to_node[role["node"]], role["port"])
        outputs = tuple(
            _Output(b.node_id, b.port, name, b) for name, b in candidate._outputs
        )
        snapshot = AssessmentSnapshot(
            "restored-validation",
            candidate.run_id,
            "input",
            tuple(candidate.key_to_node.values()),
            candidate.graph,
            _graph_fingerprint(candidate.graph),
            engine._source,
            outputs,
            (),
            (),
            (),
        )
        check = engine._preservation.check_detached(
            engine._accepted_bundle.snapshot,
            snapshot,
            reference,
        )
        if not check.passed or _json(check.public_result()) != _json(
            evidence.get("preservation_check")
        ):
            raise AssistantCheckpointError(
                "The saved selected candidate no longer passes "
                "exact foreground protection."
            )
    engine._experiments.require_assessed(candidate.candidate_id)


def restore_investigation(
    checkpoint,
    pipeline,
    source_id,
    *,
    compute_request,
    assert_current,
    include_input_images=False,
    include_output_images=False,
    cancelled=None,
    clock=monotonic,
):
    """Create a new passive owner-bound runtime, or return an honest history view.

    Loading/selection never grants sharing. The caller must explicitly obtain
    fresh approval for all image/output context used by this saved session and
    separately invoke Continue/Keep/Accept. No calculation or provider is run.
    Malformed evidence raises; ordinary source/consent incompatibility returns
    ``session=None`` with an actionable history-only reason.
    """
    if type(checkpoint) is not SavedAssistantSession:
        raise AssistantCheckpointError(
            "Load a validated assistant-session archive first."
        )
    value = checkpoint.document
    if value.get("kind") == "napari-vipp-assistant-history":
        _fields(
            value,
            {"kind", "version", "session_id", "resume", "ui_state"},
            "conversation history",
        )
        if type(value["version"]) is not int or value["version"] != VERSION:
            raise AssistantCheckpointError("Unsupported conversation history version.")
        _ui_state(value["ui_state"])
        _fields(value["resume"], {"supported", "reason"}, "history status")
        if value["resume"]["supported"] is not False:
            raise AssistantCheckpointError(
                "Conversation history has no execution authority."
            )
        return InvestigationRestoreResult(None, False, value["resume"]["reason"])
    _fields(value, _CHECKPOINT_FIELDS, "assistant session")
    if (
        value["kind"] != KIND
        or type(value["version"]) is not int
        or value["version"] != VERSION
    ):
        raise AssistantCheckpointError("Unsupported saved assistant-session version.")
    session_id = _text(value["session_id"], "session ID", 128, empty=False)
    _ui_state(value["ui_state"])
    resume = _fields(value["resume"], {"supported", "reason"}, "continuation status")
    if type(resume["supported"]) is not bool:
        raise AssistantCheckpointError("Invalid saved continuation status.")
    notebook = _restore_notebook(value["notebook"], session_id)
    options = _fields(
        value["options"],
        {
            "include_input_images",
            "include_output_images",
            "require_experiment_plan",
            "review_before_finish",
            "continuation_enabled",
        },
        "options",
    )
    if (
        any(type(v) is not bool for v in options.values())
        or type(include_input_images) is not bool
        or type(include_output_images) is not bool
    ):
        raise AssistantCheckpointError("Explicit boolean sharing choices are required.")
    if not resume["supported"] or value["scientific"] is None:
        return InvestigationRestoreResult(
            None, False, _text(resume["reason"], "history-only reason", 2000)
        )
    if (
        options["include_input_images"]
        and not include_input_images
        or options["include_output_images"]
        and not include_output_images
    ):
        return InvestigationRestoreResult(
            None,
            False,
            "History only until you explicitly approve this saved session's "
            "input/output sharing choices again. Nothing was sent.",
        )
    # Do not silently change permissions of persisted matched image/evidence
    # cache keys. The user can start a new investigation with different choices.
    if (include_input_images, include_output_images) != (
        options["include_input_images"],
        options["include_output_images"],
    ):
        return InvestigationRestoreResult(
            None,
            False,
            "Use the saved sharing choices for continuation, or start a new "
            "investigation with different sharing choices.",
        )
    limits = InvestigationLimits(
        **_fields(value["limits"], HARD_ALLOWANCE, "allowance")
    )
    usage = _fields(
        value["usage"],
        {
            "attempted_trials",
            "actions_used",
            "model_calls_used",
            "image_bytes_used",
            "attempted_probes",
            "probe_snapshot_bytes_used",
            "active_seconds_used",
            "paused_seconds",
            "allowance_grants",
            "continue_requests",
            "calculation_repairs",
            "control_revision",
        },
        "usage",
    )
    for name, maximum in (
        ("attempted_trials", 5),
        ("actions_used", 64),
        ("model_calls_used", 80),
        ("image_bytes_used", 20 * 1024 * 1024),
        ("attempted_probes", MAX_DIAGNOSTIC_PROBES),
        ("probe_snapshot_bytes_used", MAX_PROBE_SNAPSHOT_BYTES),
        ("allowance_grants", MAX_CONTINUATION_GRANTS),
        ("continue_requests", 64),
        ("calculation_repairs", 5),
        ("control_revision", 1024),
    ):
        _integer(usage[name], name, maximum)
    active = _number(usage["active_seconds_used"], "active seconds", 24 * 3600)
    paused_seconds = _number(
        usage["paused_seconds"], "paused seconds", 365 * 24 * 3600 * 10
    )
    if active >= HARD_ALLOWANCE["max_seconds"]:
        return InvestigationRestoreResult(
            None,
            False,
            "The saved hard active-time budget is exhausted. "
            "Its usage cannot be reset by reopening.",
        )
    original = _fields(
        value["original"],
        {
            "source_id",
            "graph",
            "graph_sha256",
            "source_fingerprint",
            "source_sha256",
            "source_state_sha256",
            "proposal_context_id",
            "compute_request",
        },
        "original source/workflow",
    )
    original_graph = _read_graph(original["graph"])
    if (
        original_graph is None
        or _graph_fingerprint(original_graph) != original["graph_sha256"]
    ):
        raise AssistantCheckpointError("Saved original graph fingerprint changed.")
    if type(compute_request) is not ComputeRequest or not callable(assert_current):
        raise AssistantCheckpointError(
            "Fresh owner authority and compute policy are required."
        )
    if (
        compute_request.as_dict() != original["compute_request"]
        or source_id != original["source_id"]
    ):
        return InvestigationRestoreResult(
            None,
            False,
            "The current input/compute policy differs from the saved session. "
            "Reattach the original source and policy before continuing.",
        )
    dialogue = _fields(
        value["dialogue"],
        {"objective", "clarifications", "deferred_questions"},
        "dialogue",
    )
    dialogue = ProposalDialogue(
        dialogue["objective"],
        tuple(tuple(v) for v in dialogue["clarifications"]),
        tuple(dialogue["deferred_questions"]),
    )
    try:
        # False avoids even rendering a new source preview during construction.
        # Sharing choices are installed only after exact archived/source checks.
        engine = InvestigationSession(
            pipeline,
            source_id,
            dialogue,
            compute_request=compute_request,
            limits=limits,
            include_input_images=False,
            include_output_images=False,
            require_experiment_plan=options["require_experiment_plan"],
            review_before_finish=options["review_before_finish"],
            continuation_enabled=True,
            assert_current=assert_current,
            cancelled=cancelled,
            clock=clock,
        )
    except (InvestigationError, ValueError, TypeError):
        return InvestigationRestoreResult(
            None,
            False,
            "The original source is not freshly available under the current "
            "owner. Inspect or calculate that source locally before reopening "
            "continuation.",
        )
    if (
        engine._original.baseline_fingerprint != original["graph_sha256"]
        or engine._original.source_fingerprint != original["source_fingerprint"]
        or engine._source.fingerprint != original["source_sha256"]
        or engine._source.state_fingerprint != original["source_state_sha256"]
    ):
        return InvestigationRestoreResult(
            None,
            False,
            "The current image pixels, metadata/grid or workflow changed. "
            "The archive remains history only; no source or graph was replaced.",
        )
    scientific = _fields(
        value["scientific"],
        {
            "source",
            "candidates",
            "probes",
            "accepted",
            "previews",
            "operation_contracts",
        },
        "scientific evidence",
    )
    used_operations = {
        n["operation_id"]
        for c in scientific["candidates"]
        if c["status"] in {"completed", "rejected"}
        for n in c["recipe"]["nodes"]
    } | {o["operation_id"] for p in scientific["probes"] for o in p["outputs"]}
    if (
        type(scientific["operation_contracts"]) is not dict
        or set(scientific["operation_contracts"]) != used_operations
    ):
        raise AssistantCheckpointError(
            "Saved operation-contract inventory is incomplete."
        )
    try:
        current_contracts = _contracts(scientific["operation_contracts"])
    except KeyError:
        return InvestigationRestoreResult(
            None,
            False,
            "A saved operation is no longer available. Keep this archive as "
            "history rather than silently substituting a different operation.",
        )
    if current_contracts != scientific["operation_contracts"]:
        return InvestigationRestoreResult(
            None,
            False,
            "A saved scientific node contract changed. The previous evidence "
            "is history only; start a fresh investigation with current nodes.",
        )
    runtime = _fields(
        value["runtime"],
        {
            "retained_candidate_id",
            "finished",
            "stop_reason",
            "pause_kind",
            "pause_reason",
            "pending_question",
            "pending_finish_review",
            "latest_finish_feedback",
            "seed_count",
            "record_refs",
        },
        "runtime",
    )
    if (
        type(runtime["finished"]) is not bool
        or runtime["stop_reason"]
        or not options["continuation_enabled"]
    ):
        return InvestigationRestoreResult(
            None,
            False,
            "This stopped or legacy session cannot restart implicitly; "
            "its complete local history is retained.",
        )
    engine.session_id = session_id
    engine._context = replace(
        engine._context, context_id=original["proposal_context_id"]
    )
    engine.include_input_images, engine.include_output_images = (
        include_input_images,
        include_output_images,
    )
    engine._permission_binding = (include_input_images, include_output_images)
    reader = _Reader(checkpoint)
    saved_source = reader.binding(scientific["source"])
    if (
        saved_source.fingerprint != engine._source.fingerprint
        or saved_source.state_fingerprint != engine._source.state_fingerprint
    ):
        raise AssistantCheckpointError(
            "Archived source bytes disagree with its original binding."
        )
    engine._seed_count = _integer(runtime["seed_count"], "accepted baseline count", 1)
    engine._candidates = _restore_candidates(scientific["candidates"], reader, engine)
    if (
        len(engine._candidates) - engine._seed_count != usage["attempted_trials"]
        or usage["attempted_trials"] > limits.max_attempts
    ):
        raise AssistantCheckpointError("Saved trial count/allowance changed.")
    engine._probes = []
    for index, row in enumerate(
        _list(scientific["probes"], "diagnostic probes", MAX_DIAGNOSTIC_PROBES), 1
    ):
        _fields(
            row,
            {"probe_id", "status", "evidence", "outputs", "snapshot_bytes"},
            "diagnostic probe",
        )
        if row["probe_id"] != f"probe{index}" or row["status"] not in {
            "completed",
            "failed",
        }:
            raise AssistantCheckpointError(
                "Invalid saved diagnostic probe identity/status."
            )
        outputs = []
        for output in _list(row["outputs"], "probe outputs", 4):
            _fields(output, {"role", "binding", "operation_id"}, "probe output")
            outputs.append(
                (
                    output["role"],
                    reader.binding(output["binding"]),
                    output["operation_id"],
                )
            )
        size = _integer(
            row["snapshot_bytes"], "diagnostic snapshot bytes", MAX_PROBE_SNAPSHOT_BYTES
        )
        if row["status"] == "failed" and (outputs or size):
            raise AssistantCheckpointError(
                "Failed diagnostic probes cannot have completed outputs."
            )
        if sum(_binding_bytes(b) for _r, b, _o in outputs) > size:
            raise AssistantCheckpointError(
                "Saved diagnostic output allocation changed."
            )
        engine._probes.append(
            DiagnosticProbe(
                row["probe_id"],
                row["status"],
                _json(row["evidence"]),
                tuple(outputs),
                size,
            )
        )
    if (
        len(engine._probes) != usage["attempted_probes"]
        or sum(p.snapshot_bytes for p in engine._probes)
        != usage["probe_snapshot_bytes_used"]
    ):
        raise AssistantCheckpointError("Saved diagnostic-probe accounting changed.")
    engine._probe_snapshot_bytes = usage["probe_snapshot_bytes_used"]
    engine._accepted_bundle = _restore_accepted(scientific["accepted"], reader, engine)
    if bool(engine._accepted_bundle) != bool(engine._seed_count):
        raise AssistantCheckpointError("Saved warm-start evidence inventory changed.")
    engine._experiments = _restore_ledger(value["experiments"], limits)
    for entry in engine.experiment_ledger:
        if entry["candidate_id"] is not None and entry["candidate_id"] not in {
            c.candidate_id for c in engine.candidates
        }:
            raise AssistantCheckpointError(
                "A historical experiment refers to a missing trial."
            )
    engine._notebook = notebook
    engine._history = [_json(v) for v in _list(value["history"], "action history", 64)]
    engine.actions_used, engine.model_calls_used, engine._image_bytes = (
        usage["actions_used"],
        usage["model_calls_used"],
        usage["image_bytes_used"],
    )
    if (
        engine.actions_used > limits.max_actions
        or engine.model_calls_used > limits.max_model_calls
        or engine._image_bytes > limits.max_image_bytes
        or len(engine._history) > engine.actions_used
    ):
        raise AssistantCheckpointError(
            "Saved consumed counters exceed the approved absolute totals."
        )
    records = notebook.export()["records"]
    kinds = {row["record_id"]: row["kind"] for row in notebook.index()}
    requests = [
        record["payload"]
        for record in records
        if kinds[record["record_id"]] == "action_request"
    ]
    receipts = [
        record["payload"]
        for record in records
        if kinds[record["record_id"]] == "action_receipt"
    ]
    if (
        len(requests) != engine.actions_used
        or [_json(row) for row in receipts] != engine._history
        or any(row.get("session_id") != engine.session_id for row in requests)
    ):
        raise AssistantCheckpointError(
            "Saved action journal and consumed counters disagree."
        )
    engine._allowance_grants, engine._continue_requests = (
        usage["allowance_grants"],
        usage["continue_requests"],
    )
    engine._calculation_repairs = usage["calculation_repairs"]
    engine._control_revision = usage["control_revision"] + 1
    refs = _fields(
        runtime["record_refs"],
        {"probes", "ledger", "candidates", "latest_action"},
        "record references",
    )
    for group in (refs["probes"], refs["ledger"], refs["candidates"]):
        if type(group) is not dict:
            raise AssistantCheckpointError("Invalid saved notebook record references.")
        for ref in group.values():
            notebook.lookup(**ref)
    if refs["latest_action"] is not None:
        notebook.lookup(**refs["latest_action"])
    engine._probe_records, engine._ledger_records, engine._candidate_records = (
        refs["probes"],
        refs["ledger"],
        refs["candidates"],
    )
    engine._latest_action_record = refs["latest_action"]
    for name, actual in (
        ("probes", {p.probe_id: p.evidence for p in engine._probes}),
        ("candidates", {c.candidate_id: c.evidence for c in engine.candidates}),
        ("ledger", {e["experiment_id"]: e for e in engine.experiment_ledger}),
    ):
        if set(refs[name]) != set(actual) or any(
            notebook.lookup(**refs[name][key]) != payload
            for key, payload in actual.items()
        ):
            raise AssistantCheckpointError(
                "Saved current notebook records do not match their evidence."
            )
    engine.retained_candidate_id = runtime["retained_candidate_id"]
    if engine.retained_candidate_id is not None:
        _ensure_eligible(engine, engine._candidate(engine.retained_candidate_id))
    _restore_previews(scientific["previews"], reader, engine)
    engine._pending_question = runtime["pending_question"]
    if engine._pending_question is not None:
        _text(engine._pending_question, "pending question", 2000, empty=False)
    nomination = runtime["pending_finish_review"]
    if runtime["finished"]:
        nomination = next(
            (
                v["result"]
                for v in reversed(value["history"])
                if v.get("action") == "finish" and v.get("status") == "ok"
            ),
            None,
        )
        if nomination is None:
            return InvestigationRestoreResult(
                None,
                False,
                "This finished archive has no exact nomination receipt for "
                "fresh review. Its history remains available.",
            )
    if nomination is not None:
        _fields(
            nomination,
            {
                "retained_candidate_id",
                "recipe",
                "evidence",
                "model_interpretation",
                "limitations",
                "publication",
            },
            "Finish nomination",
        )
        candidate = engine._candidate(nomination["retained_candidate_id"])
        _ensure_eligible(engine, candidate)
        engine._require_assessed_experiments()
        if (
            nomination["recipe"] != candidate.recipe
            or nomination["evidence"] != candidate.evidence
            or not any(
                v.get("action") == "finish"
                and v.get("status") == "ok"
                and v.get("result") == nomination
                for v in value["history"]
            )
        ):
            raise AssistantCheckpointError(
                "Saved nomination differs from its completed candidate "
                "or exact Finish receipt."
            )
        engine._review_before_finish = True
    engine._pending_finish_review = None if nomination is None else _json(nomination)
    engine._latest_finish_feedback = (
        None
        if runtime["latest_finish_feedback"] is None
        else _json(runtime["latest_finish_feedback"])
    )
    # Rebase duration, never an old process monotonic timestamp/deadline.
    now = engine._now()
    engine._started = now - active - paused_seconds
    engine._paused_seconds = paused_seconds
    engine._pause_started = now
    engine.deadline = now + limits.max_seconds - active
    engine._pause_kind = (
        "finish_review" if nomination is not None else runtime["pause_kind"] or "owner"
    )
    engine._pause_reason = (
        "Reopened locally. Review sharing and choose Continue; nothing has been sent."
    )
    engine.finished = False
    engine.stop_reason = ""
    # Cached scientific facts are re-registered from the exact restored local
    # candidate/probe receipts. No previous provider received them in THIS owner
    # runtime: only the normal future before_call can grant sent authority.
    for candidate in engine.candidates:
        if candidate.status in {"completed", "rejected"}:
            engine._remember_candidate_context(candidate)
    for probe in engine._probes:
        if probe.status == "completed":
            payload = {k: v for k, v in probe.evidence.items() if k != "evidence_ref"}
            engine._numeric_evidence_ref(probe.probe_id, payload)
    engine._sent_previews.clear()
    engine._sent_preview_owners.clear()
    engine._sent_numeric.clear()
    engine._last_transmission = None
    engine._finish_review_context_status = None
    engine._assert_current_authority()
    return InvestigationRestoreResult(
        engine,
        True,
        "Reopened paused with exact history/evidence and consumed usage. "
        "Continue, Keep improving or review explicitly; "
        "no request or calculation has run.",
    )


__all__ = [
    "AssistantCheckpointError",
    "InvestigationRestoreResult",
    "checkpoint_investigation",
    "restore_investigation",
]
