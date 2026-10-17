"""Local, algorithm-neutral evidence support for the product acceptance driver.

This module never plans or calculates an analysis. It records product-owned
actions/results and locks a selected result before an independent comparison.
It deliberately has no reference-data loader or sample-generator dependency.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import struct
import sys
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ORIGINAL_GOAL = (
    "in this z-stack there is a wavy structure and a bright circular structure "
    "in the middle. Can you please create complete 3D masks (and ultimately "
    "meshes) of these structures independently and combine the labeled "
    "structures with different colours together, so that I can see the meshese "
    "overlaid in different colors."
)
RAW_PIXELS_SHA256 = "628b47bd3d23e70f380344e834b6d16c983ea16c34d14e56db75f9551e0ea341"
INPUT_FILE_SHA256 = "3d67f1eaa1a22166b3e9733e1af9e5f3f1c0aea4ba95a1fce313421a6662d287"


class SampleGeneratorCallGuard:
    """Interpreter-wide measurement/block of forbidden sample-module calls.

    Python 3.12 monitoring observes function entry across Python/Qt workers. The
    exact qualified filename is supplied without opening its source. Importing
    the module is permitted; invoking a function defined in it is fail-closed.
    """

    def __init__(self, filename):
        self.filename = os.path.normcase(os.path.abspath(filename))
        self.calls = []
        self.tool_id = None

    def install(self):
        if not hasattr(sys, "monitoring"):
            raise RuntimeError(
                "Sample-call measurement requires the qualified Python 3.12 runtime."
            )
        self.tool_id = next(
            (index for index in range(6) if sys.monitoring.get_tool(index) is None),
            None,
        )
        if self.tool_id is None:
            raise RuntimeError(
                "No interpreter monitoring slot is available for source isolation."
            )
        sys.monitoring.use_tool_id(self.tool_id, "vipp-blinded-source-guard")
        sys.monitoring.register_callback(
            self.tool_id, sys.monitoring.events.PY_START, self._started
        )
        sys.monitoring.set_events(self.tool_id, sys.monitoring.events.PY_START)

    def _started(self, code, _instruction_offset):
        if code.co_name == "<module>" or not code.co_filename.endswith(
            "_sample_data.py"
        ):
            return
        if os.path.normcase(os.path.abspath(code.co_filename)) == self.filename:
            self.calls.append({"function": code.co_name, "line": code.co_firstlineno})
            raise RuntimeError("A forbidden sample-module function call was blocked.")

    def receipt(self):
        return {
            "mechanism": "Python interpreter-wide PY_START monitoring",
            "filename": self.filename,
            "function_calls_observed_and_blocked": len(self.calls),
            "calls": list(self.calls),
            "sample_generator_called": bool(self.calls),
        }

    def close(self):
        if self.tool_id is not None:
            sys.monitoring.set_events(self.tool_id, 0)
            sys.monitoring.register_callback(
                self.tool_id, sys.monitoring.events.PY_START, None
            )
            sys.monitoring.free_tool_id(self.tool_id)
            self.tool_id = None


class InferenceCallGuard(SampleGeneratorCallGuard):
    """Measure/block provider inference and native RPC in a replay-only process."""

    def __init__(self, package_directory):
        super().__init__(Path(package_directory) / "_sample_data.py")
        core = Path(package_directory) / "core"
        self.targets = {
            os.path.normcase(os.path.abspath(core / filename)): names
            for filename, names in {
                "workflow_proposal_codex.py": {
                    "_call_codex",
                    "generate_codex_proposal",
                    "_Rpc.request",
                    "start_codex_login",
                    "repair_codex_profile",
                },
                "workflow_investigation_model.py": {
                    "generate_codex_investigation_action",
                    "generate_investigation_action",
                },
                "workflow_assessment_model.py": {
                    "generate_codex_assessment",
                    "generate_assessment",
                },
                "workflow_proposal_model.py": {"generate_proposal", "_request_json"},
            }.items()
        }

    def _started(self, code, _instruction_offset):
        names = self.targets.get(os.path.normcase(os.path.abspath(code.co_filename)))
        if names and (code.co_name in names or code.co_qualname in names):
            self.calls.append(
                {"function": code.co_qualname, "line": code.co_firstlineno}
            )
            raise RuntimeError(
                "Inference, RPC and authentication are forbidden during replay."
            )

    def receipt(self):
        return {
            "mechanism": "Python interpreter-wide PY_START monitoring",
            "inference_rpc_or_auth_calls_observed_and_blocked": len(self.calls),
            "calls": list(self.calls),
        }


def _graph_parts(graph):
    if isinstance(graph, dict):
        nodes, connections = graph["nodes"], graph["connections"]
        tunnels = graph.get("output_tunnels", ())
    else:
        nodes, connections = graph.nodes, graph.connections
        tunnels = getattr(graph, "output_tunnels", ())
        nodes = nodes.values() if isinstance(nodes, Mapping) else nodes
    if tunnels:
        raise ValueError("This source-local replay does not permit output tunnels.")
    records = {}
    for node in nodes:
        item = (
            node
            if isinstance(node, dict)
            else {
                "id": node.id,
                "operation_id": node.operation_id,
                "params": node.params,
                "execution_mode": getattr(node, "execution_mode", "run"),
            }
        )
        if item.get("execution_mode", "run") != "run":
            raise ValueError("A recorded replay node must execute normally.")
        if item["id"] in records:
            raise ValueError("Replay graphs cannot contain duplicate node IDs.")
        records[item["id"]] = item
    wires = []
    for edge in connections:
        if isinstance(edge, dict):
            wires.append(
                (
                    edge["source"],
                    edge["target"],
                    edge["source_port"],
                    edge["target_port"],
                )
            )
        else:
            wires.append(
                (edge.source_id, edge.target_id, edge.source_port, edge.target_port)
            )
    return records, wires


def recipe_node_mapping(recipe, graph, context_id):
    """Bind local keys using the product's deterministic context-ID contract.

    Check operations, every model parameter and all wiring. Graph iteration
    order and a consumed dialog permission tuple have no role in this mapping.
    """
    if not isinstance(context_id, str) or len(context_id) < 12:
        raise ValueError("Replay requires an explicit proposal context identity.")
    nodes, wires = _graph_parts(graph)
    by_key = {
        node["key"]: f"proposal_{context_id[:12]}_{node['key']}"
        for node in recipe["nodes"]
    }
    if not by_key or len(by_key) != len(recipe["nodes"]):
        raise ValueError("Replay requires nonempty unique recipe keys.")
    if set(nodes) != {"input", *by_key.values()}:
        raise ValueError(
            "Replay graph nodes do not match the exact source-local recipe."
        )
    for item in recipe["nodes"]:
        actual = nodes[by_key[item["key"]]]
        if actual["operation_id"] != item["operation_id"] or any(
            name not in actual["params"] or actual["params"][name] != value
            for name, value in item["params"].items()
        ):
            raise ValueError("Replay operation or declared parameters changed.")
    expected = [
        (
            "input" if edge["source"] == "input" else by_key[edge["source"]],
            by_key[edge["target"]],
            edge["source_port"],
            edge["target_port"],
        )
        for edge in recipe["connections"]
    ]
    if sorted(wires) != sorted(expected):
        raise ValueError("Replay wiring differs from the selected recipe.")
    return by_key


@dataclasses.dataclass(frozen=True)
class OwnedReplayBinding:
    added_node_ids: tuple[str, ...]
    trial_to_replay: tuple[tuple[str, str], ...]
    recipe_sha256: str

    @property
    def node_mapping(self):
        return dict(self.trial_to_replay)


def freeze_replay_binding(
    recipe, trial_graph, replay_graph, added_ids, replay_context_id
):
    trial = recipe_node_mapping(recipe, trial_graph, recipe["context_id"])
    replay = recipe_node_mapping(recipe, replay_graph, replay_context_id)
    ids = tuple(added_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(replay.values()):
        raise ValueError(
            "Freeze the exact nonempty owned Add IDs before Run consumes them."
        )
    trial_records, _ = _graph_parts(trial_graph)
    replay_records, _ = _graph_parts(replay_graph)
    for key in trial:
        if public_value(trial_records[trial[key]]["params"]) != public_value(
            replay_records[replay[key]]["params"]
        ):
            raise ValueError("Complete trial/replay parameter dictionaries differ.")
    return OwnedReplayBinding(
        ids,
        tuple((trial[key], replay[key]) for key in trial),
        sha256(json_bytes(recipe)),
    )


def verify_executed_replay_graph(expected_graph, actual_graph, binding):
    expected, wires = _graph_parts(expected_graph)
    actual, actual_wires = _graph_parts(actual_graph)
    mapping = {"input": "input", **binding.node_mapping}
    if set(expected) != set(mapping) or set(actual) != set(mapping.values()):
        raise ValueError("The completed replay graph node set changed.")
    for original, replay in binding.trial_to_replay:
        if expected[original]["operation_id"] != actual[replay][
            "operation_id"
        ] or public_value(expected[original]["params"]) != public_value(
            actual[replay]["params"]
        ):
            raise ValueError(
                "Completed replay operations or complete parameters differ."
            )
    expected_wires = [(mapping[a], mapping[b], c, d) for a, b, c, d in wires]
    if sorted(expected_wires) != sorted(actual_wires):
        raise ValueError("Completed replay wiring changed.")


def owned_completion_details(
    binding, pipeline, *, active=False, pending=False, registry=None
):
    """Never accept all([]), stale ready flags, missing nodes or an active run."""
    ids = binding.added_node_ids
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(pipeline.nodes):
        raise ValueError("The exact owned replay node set is missing or empty.")
    states = [pipeline.node_execution_states.get(node) for node in ids]
    if any(state in {"error", "blocked"} for state in states):
        return {
            "status": "failed",
            "scientific_node_ids": [],
            "presentation_exceptions": [],
        }
    scientific, exceptions = [], []
    for node_id in ids:
        operation_id = getattr(pipeline.nodes[node_id], "operation_id", None)
        if operation_id == "review_images":
            if registry is None:
                from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID

                registry = NODE_LIBRARY_BY_ID
            spec = registry.get(operation_id)
            if spec is None or not spec.presentation_only or spec.output_type != "none":
                raise ValueError(
                    "Review presentation exception is not catalogue-confirmed."
                )
            values = getattr(pipeline, "node_outputs", {}).get(node_id) or ()
            if (
                any(value is not None for value in values)
                or getattr(pipeline, "outputs", {}).get(node_id) is not None
            ):
                raise ValueError(
                    "A presentation exception cannot hide scientific cache values."
                )
            if pipeline.node_execution_states.get(node_id) in {
                "ready",
                "not_calculated",
            }:
                exceptions.append(node_id)
                continue
        scientific.append(node_id)
    if not scientific:
        raise ValueError("Owned completion needs nonempty scientific nodes.")
    details = {
        "status": "waiting",
        "scientific_node_ids": scientific,
        "presentation_exceptions": exceptions,
    }
    if active or pending:
        return details
    if all(
        pipeline.node_execution_states.get(node) == "ready" for node in scientific
    ) and set(scientific) <= set(pipeline.completed_node_ids):
        details["status"] = "ready"
    return details


def owned_replay_completion(binding, pipeline, *, active=False, pending=False):
    return owned_completion_details(binding, pipeline, active=active, pending=pending)[
        "status"
    ]


@dataclasses.dataclass(frozen=True)
class OwnedDispatchTicket:
    dispatch_serial: int
    pipeline_run_serial: int
    run_id: str
    _revision: object = dataclasses.field(repr=False)
    _initial_report: object = dataclasses.field(repr=False)


def freeze_owned_dispatch(
    widget, before_serial, before_pipeline_serial, revision, initial_report
):
    """Bind one accepted ordinary dispatch, without enabling any model service."""
    serial = widget._workflow_assistant_assessment_serial
    pipeline_serial = widget._pipeline_run_serial
    if (
        serial != before_serial + 1
        or widget._workflow_assistant_run_revision is not revision
        or pipeline_serial not in {before_pipeline_serial, before_pipeline_serial + 1}
    ):
        raise ValueError("Exactly one unchanged owned normal Run must be dispatched.")
    run_id = (
        f"background-{pipeline_serial}"
        if pipeline_serial != before_pipeline_serial
        else f"synchronous-{serial}"
    )
    return OwnedDispatchTicket(
        serial, pipeline_serial, run_id, revision, initial_report
    )


def assert_owned_dispatch(widget, ticket, *, completed=False):
    if (
        widget._workflow_assistant_assessment_serial != ticket.dispatch_serial
        or widget._pipeline_run_serial != ticket.pipeline_run_serial
        or widget._workflow_assistant_run_revision is not ticket._revision
    ):
        raise ValueError(
            "A later dispatch or changed Add superseded this owned replay."
        )
    if completed:
        report = widget._last_execution_report
        if (
            report is None
            or report is ticket._initial_report
            or report.cleanup_succeeded is not True
        ):
            raise ValueError("The owned Run has no fresh successful execution report.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


_FLOAT_TAG = "$vipp_float64_bits"
_MAP_TAG = "$vipp_mapping_items"


def recording_document(value):
    """Evidence-only reversible nonfinite floats; never change provider JSON."""
    public = public_value(value)
    changed = (
        isinstance(public, dict)
        and public.get("kind") == "napari-vipp-nonfinite-recording"
    )

    def encode(item):
        nonlocal changed
        if isinstance(item, float) and not math.isfinite(item):
            changed = True
            return {_FLOAT_TAG: struct.pack(">d", item).hex()}
        if isinstance(item, dict):
            pairs = [[key, encode(part)] for key, part in item.items()]
            if _FLOAT_TAG in item or _MAP_TAG in item:
                return {_MAP_TAG: pairs}
            return dict(pairs)
        if isinstance(item, list):
            return [encode(part) for part in item]
        return item

    encoded = encode(public)
    if not changed:
        return public
    return {
        "kind": "napari-vipp-nonfinite-recording",
        "version": 1,
        "encoding": "ieee754-binary64-hex",
        "value": encoded,
    }


def decode_recording_document(value):
    if (
        not isinstance(value, dict)
        or value.get("kind") != "napari-vipp-nonfinite-recording"
    ):
        return value
    if (
        set(value) != {"kind", "version", "encoding", "value"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["encoding"] != "ieee754-binary64-hex"
    ):
        raise ValueError("Unsupported nonfinite evidence codec.")

    def decode(item):
        if isinstance(item, list):
            return [decode(part) for part in item]
        if isinstance(item, dict):
            if set(item) == {_FLOAT_TAG}:
                bits = item[_FLOAT_TAG]
                if not isinstance(bits, str) or len(bits) != 16:
                    raise ValueError("Malformed nonfinite IEEE754 bits.")
                number = struct.unpack(">d", bytes.fromhex(bits))[0]
                if math.isfinite(number):
                    raise ValueError("Nonfinite tag cannot encode a finite value.")
                return number
            if set(item) == {_MAP_TAG}:
                pairs = item[_MAP_TAG]
                if (
                    not isinstance(pairs, list)
                    or any(
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or not isinstance(pair[0], str)
                        for pair in pairs
                    )
                    or len({pair[0] for pair in pairs}) != len(pairs)
                ):
                    raise ValueError("Malformed escaped evidence mapping.")
                return {key: decode(part) for key, part in pairs}
            return {key: decode(part) for key, part in item.items()}
        return item

    return decode(value["value"])


def read_recording_json(path):
    """Explicit local evidence reader; never use for provider contracts/replies."""
    return decode_recording_document(json.loads(Path(path).read_text("utf-8")))


def public_value(value):
    """Serialize explicit public records, never private dataclass/cache fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): public_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [public_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return public_value(value.to_dict())
    if hasattr(value, "as_dict"):
        return public_value(value.as_dict())
    if dataclasses.is_dataclass(value):
        return {
            field.name: public_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_") and field.repr
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "array-manifest-only",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_sha256": sha256(value.tobytes(order="C")),
        }
    raise TypeError(f"No approved evidence serialization for {type(value).__name__}")


def mesh_payload_manifest(mesh):
    """Canonical complete numerical mesh payload, appearance and carried state."""
    buffers = {}
    for name in ("vertices", "faces", "face_object_ids"):
        values = getattr(mesh, name)
        if type(values) is not np.ndarray:
            raise ValueError("A replay mesh must retain its exact numerical buffers.")
        buffers[name] = {
            "dtype": values.dtype.str,
            "shape": list(values.shape),
            "raw_sha256": sha256(values.tobytes(order="C")),
        }
    if hasattr(mesh, "object_ids"):
        object_ids, colors = mesh.object_ids, mesh.colors
    else:
        object_ids = tuple(item.object_id for item in mesh.objects)
        colors = tuple(item.color for item in mesh.objects)
    return {
        "buffers": buffers,
        "object_ids": public_value(object_ids),
        "colors": public_value(colors),
        "state": public_value(mesh.state),
    }


def verify_mesh_replay(expected, actual):
    if mesh_payload_manifest(expected) != mesh_payload_manifest(actual):
        raise ValueError("Final replay mesh payload differs from the selected trial.")


def verify_mesh_objects_replay(expected_objects, actual_objects):
    """Compare the full trial object manifest, including provenance/name fields."""
    if public_value(expected_objects) != public_value(actual_objects):
        raise ValueError(
            "Final replay mesh object metadata differs from the selected trial."
        )


def verify_array_replay(expected, actual):
    """Require original scientific values, representation and shape, not closeness."""
    if (
        type(expected) is not np.ndarray
        or type(actual) is not np.ndarray
        or expected.dtype.str != actual.dtype.str
        or expected.shape != actual.shape
        or expected.tobytes(order="C") != actual.tobytes(order="C")
    ):
        raise ValueError("Final replay array buffer differs from the selected trial.")


def verify_input(input_directory: Path) -> dict:
    """Read only the two explicitly allowed source files and verify identity."""
    source = input_directory / "input.npy"
    metadata = json.loads((input_directory / "metadata.json").read_text("utf-8"))
    file_hash = sha256(source.read_bytes())
    if file_hash != INPUT_FILE_SHA256:
        raise ValueError("The approved input file identity does not match.")
    pixels = np.load(source, allow_pickle=False)
    raw_hash = sha256(pixels.tobytes(order="C"))
    if raw_hash != RAW_PIXELS_SHA256 or pixels.shape != (12, 96, 128):
        raise ValueError("The approved raw pixel identity or shape does not match.")
    if pixels.dtype != np.uint8:
        raise ValueError("The approved input dtype does not match.")
    expected = {
        "axes": "ZYX",
        "shape": [12, 96, 128],
        "dtype": "uint8",
        "scale": [0.45, 0.45, 0.45],
        "unit": "micrometer",
        "origin": [0.0, 0.0, 0.0],
        "raw_pixels_sha256": RAW_PIXELS_SHA256,
        "input_file_sha256": INPUT_FILE_SHA256,
        "goal": ORIGINAL_GOAL,
        "reference_workflow_or_masks_supplied": False,
        "sample_generator_supplied": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("The approved source metadata does not match.")
    return expected


class EvidenceJournal:
    """Thread-safe append-only events and exclusive, hashed evidence artifacts."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError("Acceptance evidence must use a fresh directory.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._sequence = 0
        self._chain = "0" * 64
        self._artifacts = {}
        self._selected = False
        self._events = self.directory / "events.jsonl"
        if (
            self._events.exists()
            or (self.directory / "selected-result-lock.json").exists()
        ):
            raise FileExistsError("Acceptance evidence must use a fresh directory.")

    def _target(self, relative: str) -> Path:
        target = (self.directory / relative).resolve()
        if not target.is_relative_to(self.directory) or target == self.directory:
            raise ValueError(
                "Evidence artifacts must stay inside the evidence directory."
            )
        return target

    def artifact(self, relative: str, data: bytes) -> dict:
        with self._mutex:
            if self._selected:
                raise RuntimeError(
                    "Selection is locked; pre-comparison evidence is closed."
                )
            target = self._target(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            record = {"path": relative, "byte_count": len(data), "sha256": sha256(data)}
            self._artifacts[relative] = record
            return record.copy()

    def document(self, relative: str, value) -> dict:
        return self.artifact(relative, json_bytes(recording_document(value)))

    def event(self, kind: str, **details) -> dict:
        with self._mutex:
            if self._selected:
                raise RuntimeError(
                    "Selection is locked; pre-comparison journal is closed."
                )
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "utc": datetime.now(UTC).isoformat(),
                "kind": kind,
                "details": public_value(details),
                "previous_event_sha256": self._chain,
            }
            encoded = json_bytes(record)
            self._chain = sha256(encoded)
            with self._events.open("ab") as stream:
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False).encode()
                )
                stream.write(b"\n")
                stream.flush()
            return record

    def previews(self, prefix: str, previews) -> list[dict]:
        records = []
        for index, preview in enumerate(previews):
            artifact = self.artifact(
                f"{prefix}/preview-{index + 1:02d}.png", preview.png_bytes
            )
            records.append({**preview.manifest(), "artifact": artifact})
        self.document(f"{prefix}/manifest.json", records)
        return records

    def array(self, relative: str, pixels: np.ndarray) -> dict:
        from io import BytesIO

        buffer = BytesIO()
        np.save(buffer, pixels, allow_pickle=False)
        result = self.artifact(relative, buffer.getvalue())
        return {**result, **public_value(pixels)}

    def lock_selection(self, selection: dict) -> dict:
        """No reference loader exists; lock identity before downstream comparison."""
        with self._mutex:
            if self._selected:
                raise RuntimeError("A selected result is already locked.")
            if not selection.get("candidate_id") or not selection.get("run_id"):
                raise ValueError(
                    "Lock a completed identified candidate and run, "
                    "not the latest item."
                )
            if not selection.get("role_mapping"):
                raise ValueError(
                    "Selection must retain the product's explicit output roles."
                )
            for relative, record in self._artifacts.items():
                if sha256(self._target(relative).read_bytes()) != record["sha256"]:
                    raise ValueError(
                        "Evidence changed before the selected-result lock."
                    )
            self.event(
                "selection_locked_before_reference_comparison", selection=selection
            )
            document = {
                "kind": "napari-vipp-blinded-selected-result-lock",
                "version": 1,
                "selection": public_value(selection),
                "reference_material_read": False,
                "event_count": self._sequence,
                "last_event_sha256": self._chain,
                "events_file_sha256": sha256(self._events.read_bytes()),
                "artifacts": list(self._artifacts.values()),
            }
            encoded = json_bytes(document)
            self.artifact("selected-result-lock.json", encoded)
            self._selected = True
            return {"path": "selected-result-lock.json", "sha256": sha256(encoded)}
