"""Mechanical evidence binding for a warm-start investigation of its own result."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from replay_workflow_investigation_selection import load_array, read_json, run_inventory
from workflow_investigation_evidence import (
    RAW_PIXELS_SHA256,
    OwnedReplayBinding,
    json_bytes,
    public_value,
    read_recording_json,
    recipe_node_mapping,
    recording_document,
    sha256,
    verify_array_replay,
    verify_mesh_objects_replay,
    verify_mesh_replay,
)

BASELINE_LOCK_SHA256 = (
    "1bc02ef8e65b7965af00019bc0cded0feec1dd16b9e35c1b72cd33b17caaa138"
)
BASELINE_ROLES = (
    ("circle_mask", "refine_circle_mask", 0),
    ("band_mask", "refine_reference_mask", 0),
    ("circle_labels", "refine_circle_labels", 0),
    ("band_labels", "refine_reference_labels", 0),
    ("circle_mesh", "refine_circle_mesh", 0),
    ("band_mesh", "refine_reference_mesh", 0),
    ("combined_mesh", "refine_reference_colours", 0),
)


def recording_bytes(value):
    """Exact local-state comparison, including reversible nonfinite floats."""
    return json_bytes(recording_document(public_value(value)))


def verified_lock(root, expected_sha256):
    root = Path(root).resolve(strict=True)
    path = root / "selected-result-lock.json"
    if sha256(path.read_bytes()) != expected_sha256:
        raise ValueError("The approved baseline lock changed.")
    lock = read_json(path)
    for artifact in lock["artifacts"]:
        literal = root / artifact["path"]
        resolved = literal.resolve(strict=True)
        if (
            literal.is_symlink()
            or not resolved.is_relative_to(root)
            or sha256(resolved.read_bytes()) != artifact["sha256"]
        ):
            raise ValueError("A locked baseline artifact changed or escaped its root.")
    data = (root / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The locked baseline event chain changed.")
        previous = sha256(json_bytes(event))
    if (
        len(events) != lock["event_count"]
        or previous != lock["last_event_sha256"]
        or sha256(data) != lock["events_file_sha256"]
    ):
        raise ValueError("The baseline lock no longer binds its exact event history.")
    return root, lock


def load_completed_baseline(root):
    """Use ONLY the locked run's accepted baseline branch, never its final revision."""
    root, lock = verified_lock(root, BASELINE_LOCK_SHA256)
    recipe = read_json(root / "baseline/proposal.json")
    prepared = read_json(root / "baseline/prepared-workflow.json")
    executed = read_json(root / "baseline/completed-workflow/workflow.json")
    mapping = recipe_node_mapping(recipe, prepared, recipe["context_id"])
    if recipe_node_mapping(recipe, executed, recipe["context_id"]) != mapping:
        raise ValueError("The full accepted baseline recipe or wiring changed.")
    if len(mapping) != 33:
        raise ValueError(
            "This authorization is for the exact accepted 33-node baseline."
        )
    state = read_recording_json(root / "baseline/completed-workflow/full-state.json")
    expected = []
    buffer_count = 0
    ports = {(item["node_id"], item["port"]): item for item in state["ports"]}
    if len(ports) != len(state["ports"]):
        raise ValueError("The complete baseline contains duplicate port records.")
    for (node, port), saved in ports.items():
        if node not in {"input", *mapping.values()}:
            raise ValueError("An unrelated output entered the baseline full-state.")
        if "array" in saved:
            value = load_array(root, saved["array"])
            buffer_count += 1
        elif "mesh" in saved:
            mesh = saved["mesh"]
            value = SimpleNamespace(
                vertices=load_array(root, mesh["vertices"]),
                faces=load_array(root, mesh["faces"]),
                face_object_ids=load_array(root, mesh["face_object_ids"]),
                object_ids=tuple(item["object_id"] for item in mesh["objects"]),
                colors=tuple(item["color"] for item in mesh["objects"]),
                state=mesh["state"],
            )
            buffer_count += 3
        else:
            value = saved.get("data")
        expected.append((node, port, saved, value))
    if buffer_count != 41:
        raise ValueError(
            "The exact accepted baseline's 41 persisted buffers are missing."
        )
    source = ports[("input", 0)]
    pixels = load_array(root, source["array"])
    if (
        pixels.dtype != np.uint8
        or pixels.shape != (12, 96, 128)
        or sha256(pixels.tobytes(order="C")) != RAW_PIXELS_SHA256
    ):
        raise ValueError("The baseline source is not the approved image.")
    roles = []
    for role, key, port in BASELINE_ROLES:
        if key not in mapping or (mapping[key], port) not in ports:
            raise ValueError("An explicitly authorized accepted output role is absent.")
        roles.append({"role": role, "node": mapping[key], "port": port})
    source_params = next(
        node["params"] for node in prepared["nodes"] if node["id"] == "input"
    )
    source_path = Path(source_params["file_path"]).resolve(strict=True)
    return SimpleNamespace(
        root=root,
        before=run_inventory(root),
        lock_sha256=BASELINE_LOCK_SHA256,
        recipe=recipe,
        prepared_graph=prepared,
        executed_graph=executed,
        full_state=state,
        expected=expected,
        key_to_node=mapping,
        roles=roles,
        source_params=source_params,
        source_path=source_path,
        source_file_sha256=sha256(source_path.read_bytes()),
        buffer_count=buffer_count,
        selected_final_revision_is_not_used=True,
        original_own_baseline_lock_sha256=lock["selection"][
            "own_model_baseline_lock_sha256"
        ],
    )


def exact_baseline_replay(bundle, pipeline, node_mapping):
    """Compare all recorded scientific values and complete carried states."""
    mapping = {"input": "input", **node_mapping}
    records = []
    for old_node, port, saved, expected in bundle.expected:
        node = mapping[old_node]
        actual = pipeline.node_outputs.get(node) or [pipeline.outputs.get(node)]
        actual = actual[port]
        states = pipeline.node_output_states.get(node) or [
            pipeline.output_states.get(node)
        ]
        actual_state = states[port]
        if "array" in saved:
            verify_array_replay(expected, actual)
        elif "mesh" in saved:
            verify_mesh_replay(expected, actual)
            verify_mesh_objects_replay(saved["mesh"]["objects"], actual.objects)
        elif recording_bytes(expected) != recording_bytes(actual):
            raise ValueError(
                "The baseline table/presentation value did not replay exactly."
            )
        if recording_bytes(saved["state"]) != recording_bytes(actual_state):
            raise ValueError(
                "The complete baseline carried state did not replay exactly."
            )
        records.append(
            {
                "old_node_id": old_node,
                "node_id": node,
                "port": port,
                "value_exact": True,
                "complete_state_exact": True,
            }
        )
    return records


def runtime_baseline_roles(bundle, node_mapping):
    return [{**role, "node": node_mapping[role["node"]]} for role in bundle.roles]


def candidate_key_mapping(candidate):
    """Use product-issued explicit identity, not graph order or role guessing."""
    mapping = candidate.key_to_node
    if (
        not isinstance(mapping, dict)
        or set(mapping) != {node["key"] for node in candidate.recipe["nodes"]}
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError("The selected candidate lacks an exact product key/node map.")
    return mapping


def retained_baseline_binding(candidate, original_binding):
    """Retention uses this exact owned result, not another Add or calculation."""
    ids = tuple(candidate_key_mapping(candidate).values())
    if (
        candidate.candidate_id != "candidate0"
        or candidate.status != "completed"
        or set(ids) != set(original_binding.added_node_ids)
    ):
        raise ValueError("The retained baseline is not the exact completed normal Run.")
    return OwnedReplayBinding(
        ids,
        tuple((node, node) for node in ids),
        sha256(json_bytes(candidate.recipe)),
    )


def assert_selected_preservation(candidate, contract):
    """Require the product's exact passed constraint, never repair its pixels."""
    result = candidate.evidence.get("preservation_check")
    if (
        candidate.status != "completed"
        or not isinstance(result, dict)
        or result.get("passed") is not True
        or result.get("constraint_id") != contract["constraint_id"]
        or result.get("run_id") != candidate.run_id
        or type(result.get("protected_count")) is not int
        or result["protected_count"] != contract["protected_count"]
        or type(result.get("retained_count")) is not int
        or result.get("retained_count") != result["protected_count"]
        or type(result.get("missing_count")) is not int
        or result.get("missing_count") != 0
        or result.get("candidate_key") != "protected_structure"
        or type(result.get("candidate_port")) is not int
        or result.get("candidate_port") != 0
    ):
        raise ValueError(
            "The selected result lacks its exact passed product protection."
        )
    return result


def verify_microedit_diff(parent, candidate, action):
    """Independently verify the literal model edits, all wires and all roles."""
    args = action.arguments
    edits = args["edits"]
    receipt = candidate.evidence.get("edit_receipt")
    compact = json.dumps(
        parent.recipe, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    if receipt != {
        "parent_candidate_id": parent.candidate_id,
        "parent_recipe_sha256": sha256(compact),
        "edits": edits,
    }:
        raise ValueError("A microedit receipt does not bind its literal model action.")
    expected = deepcopy(parent.recipe)
    expected["context_id"] = candidate.recipe["context_id"]
    nodes = {node["key"]: node for node in expected["nodes"]}
    touched = set()
    if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
        raise ValueError("The microedit count is outside the finite edit contract.")
    for edit in edits:
        key = edit.get("node")
        if key not in nodes or key in touched or type(edit.get("params")) is not dict:
            raise ValueError("A microedit does not name one unique existing node.")
        touched.add(key)
        if edit.get("kind") == "set_parameters" and set(edit) == {
            "kind",
            "node",
            "params",
        }:
            if not edit["params"]:
                raise ValueError("A microedit cannot silently change no parameter.")
            nodes[key]["params"].update(deepcopy(edit["params"]))
        elif edit.get("kind") == "replace_operation" and set(edit) == {
            "kind",
            "node",
            "params",
            "operation_id",
        }:
            nodes[key]["operation_id"] = edit["operation_id"]
            nodes[key]["params"] = deepcopy(edit["params"])
        else:
            raise ValueError("A microedit changed an undeclared operation or field.")
    if (
        json_bytes(expected) != json_bytes(candidate.recipe)
        or json_bytes(parent.output_roles) != json_bytes(candidate.output_roles)
        or json_bytes(parent.recipe["nodes"]) == json_bytes(expected["nodes"])
    ):
        raise ValueError(
            "The exact microedit changed undeclared nodes, wires or roles."
        )
    return {
        "parent_candidate_id": parent.candidate_id,
        "candidate_id": candidate.candidate_id,
        "literal_model_edits": edits,
        "changed_node_keys": sorted(touched),
        "all_undeclared_parameters_and_connections_unchanged": True,
        "output_roles_unchanged": True,
    }


def bind_selected_additive_replay(
    candidate, full_prepared, actual_added, added_ids, context_id
):
    """Bind a nominated source-local candidate into a full additive user graph."""
    from run_workflow_feedback_refinement import bind_additive_graph

    from napari_vipp.core.pipeline import SPATIAL_OPERATIONS
    from napari_vipp.core.workflow import serialize_workflow

    recipe = candidate.recipe
    rebased = {**recipe, "context_id": context_id}
    _binding, new = bind_additive_graph(
        rebased, full_prepared, actual_added, added_ids, SPATIAL_OPERATIONS
    )
    old = candidate_key_mapping(candidate)
    graph = serialize_workflow(candidate.graph.to_pipeline())
    old_nodes = {node["id"]: node for node in graph["nodes"]}
    actual_nodes = {node["id"]: node for node in actual_added["nodes"]}
    if set(old_nodes) != {"input", *old.values()}:
        raise ValueError("The selected candidate graph contains an unbound branch.")

    def scientific_node(node):
        result = {key: value for key, value in node.items() if key != "id"}
        result = json.loads(json.dumps(result))
        rank = result["params"].get("resolved_spatial_ndim")
        if (
            result["operation_id"] in SPATIAL_OPERATIONS
            and type(rank) is int
            and rank in {2, 3}
        ):
            result["params"].pop("resolved_spatial_ndim")
        return result

    for key in old:
        if scientific_node(old_nodes[old[key]]) != scientific_node(
            actual_nodes[new[key]]
        ):
            raise ValueError(
                "The full selected operation/parameter record changed at Add."
            )
    remapping = {"input": "input", **{old[key]: new[key] for key in old}}
    expected_edges = sorted(
        (
            remapping[e["source"]],
            remapping[e["target"]],
            e["source_port"],
            e["target_port"],
        )
        for e in graph["connections"]
    )
    actual_edges = sorted(
        (e["source"], e["target"], e["source_port"], e["target_port"])
        for e in actual_added["connections"]
        if e["target"] in new.values()
    )
    if expected_edges != actual_edges:
        raise ValueError("The nominated candidate's exact wiring changed at Add.")
    return OwnedReplayBinding(
        tuple(added_ids),
        tuple((old[key], new[key]) for key in old),
        sha256(json_bytes(recipe)),
    )


def recorded_port_bundle(root, ports):
    """Re-open captured arrays/complete objects, without calculating anything."""
    expected = []
    for saved in ports:
        if "array" in saved:
            value = load_array(root, saved["array"])
        elif "mesh" in saved:
            mesh = saved["mesh"]
            value = SimpleNamespace(
                vertices=load_array(root, mesh["vertices"]),
                faces=load_array(root, mesh["faces"]),
                face_object_ids=load_array(root, mesh["face_object_ids"]),
                object_ids=tuple(item["object_id"] for item in mesh["objects"]),
                colors=tuple(item["color"] for item in mesh["objects"]),
                state=mesh["state"],
            )
        else:
            value = saved.get("data")
        expected.append((saved["node_id"], saved["port"], saved, value))
    return SimpleNamespace(expected=expected)


def restore_recorded_port(saved, expected):
    """Rehydrate only persisted public data; never calculate or infer metadata."""
    from napari_vipp.core.meshes import MeshData, MeshObject, MeshState
    from napari_vipp.core.metadata import AxisMetadata, ImageState
    from napari_vipp.core.tables import TableData, TableState

    state = saved["state"]
    if "array" in saved:
        value = expected.copy()
        value.setflags(write=False)
        carried = ImageState.from_dict(state)
        if carried is None:
            raise ValueError("The persisted image state cannot be restored exactly.")
    elif "mesh" in saved:
        mesh = saved["mesh"]
        params = {field.name: mesh["state"][field.name] for field in fields(MeshState)}
        params["spatial_axes"] = tuple(
            AxisMetadata.from_dict(axis) for axis in params["spatial_axes"]
        )
        carried = MeshState(**params)
        value = MeshData(
            expected.vertices,
            expected.faces,
            carried,
            expected.face_object_ids,
            tuple(MeshObject(**item) for item in mesh["objects"]),
        )
    elif "data" in saved:
        params = deepcopy(expected)
        nested = (
            "histogram_metadata",
            "detection_metadata",
            "observation_metadata",
            "tracking_metadata",
        )
        if any(
            params.get(key) is not None or state.get(key) is not None for key in nested
        ):
            raise ValueError("This cache adapter cannot infer nested table metadata.")
        for key in ("columns", "rows", "column_units"):
            params[key] = tuple(
                tuple(row) if isinstance(row, list) else row for row in params[key]
            )
        value = TableData(**params)
        params = {
            field.name: state[field.name]
            for field in fields(TableState)
            if field.name in state
        }
        params["column_units"] = tuple(state["column_units"].items())
        carried = TableState(**params)
    elif expected is None and state is None:
        return None, None
    else:
        raise ValueError("The persisted port has no supported complete state.")
    if recording_bytes(state) != recording_bytes(carried):
        raise ValueError("Restoring a port changed its exact recorded metadata.")
    if "array" in saved:
        verify_array_replay(expected, value)
    elif "mesh" in saved:
        verify_mesh_replay(expected, value)
        verify_mesh_objects_replay(saved["mesh"]["objects"], value.objects)
    elif recording_bytes(expected) != recording_bytes(value):
        raise ValueError("Restoring a table changed a recorded cell or field.")
    return value, carried


def offline_core_compatibility(bundle, goal, feedback):
    """Read-only qualified-core binding check, not native owned Run evidence."""
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import deserialize_workflow
    from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
    from napari_vipp.core.workflow_investigation import (
        InvestigationLimits,
        InvestigationSession,
        capture_completed_candidate,
    )
    from napari_vipp.core.workflow_proposals import ProposalDialogue
    from napari_vipp.core.workflow_refinement import capture_preservation_constraint

    graph = deserialize_workflow(bundle.executed_graph)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        graph["nodes"], graph["connections"], graph["output_tunnels"]
    )
    for node, port, saved, expected in bundle.expected:
        data, state = restore_recorded_port(saved, expected)
        values = pipeline.node_outputs[node]
        states = pipeline.node_output_states[node]
        if port != len(values):
            raise ValueError("Persisted baseline ports are not in exact port order.")
        values.append(data)
        states.append(state)
        if port == 0:
            pipeline.outputs[node], pipeline.output_states[node] = data, state
    pipeline.completed_node_ids = set(bundle.full_state["completed_node_ids"])
    for field in (
        "node_execution_states",
        "node_execution_messages",
        "node_compute_provenance",
        "node_cache_lineage",
    ):
        setattr(pipeline, field, deepcopy(bundle.full_state[field]))
    mapping = {node: node for node in bundle.key_to_node.values()}
    exact_baseline_replay(bundle, pipeline, mapping)
    ids = tuple(bundle.key_to_node.values())
    snapshot = capture_assessment_snapshot(
        pipeline, "input", ids, "offline-core-compatibility-not-owned"
    )
    constraint = capture_preservation_constraint(
        snapshot,
        (bundle.key_to_node["refine_reference_labels"], 0),
        (bundle.key_to_node["refine_circle_labels"], 0),
        output_sharing=True,
    )
    captured = capture_completed_candidate(
        pipeline,
        snapshot,
        bundle.roles,
        preservation_constraint=constraint,
        protected_role="band_mask",
    )
    detached = snapshot.baseline.to_pipeline()
    detached.outputs["input"] = pipeline.outputs["input"]
    detached.output_states["input"] = pipeline.output_states["input"]
    session = InvestigationSession.from_completed(
        detached,
        captured,
        ProposalDialogue(goal, (("Additional instructions 1", feedback),)),
        compute_request=ComputeRequest(),
        limits=InvestigationLimits(max_attempts=3),
        include_input_images=True,
        include_output_images=True,
        require_experiment_plan=True,
        assert_current=lambda: None,
    )
    request = session.public_request()
    if (
        session.attempted_trials
        or session.model_calls_used
        or session.actions_used
        or request["objective"] != goal
    ):
        raise ValueError(
            "A read-only baseline check attempted a trial or changed goal."
        )
    return {
        "qualified_core_compatible": True,
        "native_run_ownership_proven": False,
        "cache_only_no_execution_or_provider": True,
        "exact_restored_ports": len(bundle.expected),
        "exact_restored_buffers": bundle.buffer_count,
        "accepted_node_count": len(ids),
        "preservation": constraint.public_contract(),
        "warm_start_binding": session.warm_start_binding,
        "key_to_node": session.candidates[0].key_to_node,
        "candidate0_retained": session.retained_candidate_id == "candidate0",
    }
