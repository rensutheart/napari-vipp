"""Mechanical warm-start recording fixtures; no live model or source calculation."""

from __future__ import annotations

import importlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return SimpleNamespace(
        evidence=importlib.import_module("workflow_microedit_evidence"),
        observer=importlib.import_module("workflow_microedit_observers"),
        runner=importlib.import_module("run_workflow_microedit_investigation"),
        shared=importlib.import_module("workflow_investigation_evidence"),
    )


def _candidate(identity="candidate0"):
    recipe = {
        "context_id": "old",
        "summary": "An existing inferred candidate.",
        "nodes": [
            {
                "key": "protected_structure",
                "operation_id": "opening",
                "params": {"radius": 2, "spatial_mode": "3D ZYX"},
                "reason": "Explicit existing choice.",
            },
        ],
        "connections": [
            {
                "source": "input",
                "target": "protected_structure",
                "source_port": 0,
                "target_port": 0,
            }
        ],
    }
    return SimpleNamespace(
        candidate_id=identity,
        recipe=recipe,
        status="completed",
        run_id="owned",
        key_to_node={"protected_structure": "original-node"},
        evidence={},
        output_roles=[{"role": "band_mask", "node": "protected_structure", "port": 0}],
    )


def test_candidate0_is_exact_retention_not_a_duplicate_run(modules):
    candidate = _candidate()
    original = modules.shared.OwnedReplayBinding(
        ("original-node",), (("older", "original-node"),), "originalhash"
    )
    binding = modules.evidence.retained_baseline_binding(candidate, original)
    assert binding.added_node_ids == ("original-node",)
    assert binding.node_mapping == {"original-node": "original-node"}
    assert binding.recipe_sha256 == modules.shared.sha256(
        modules.shared.json_bytes(candidate.recipe)
    )
    for change in ("identity", "status", "node", "key"):
        wrong = deepcopy(candidate)
        if change == "identity":
            wrong.candidate_id = "candidate1"
        elif change == "status":
            wrong.status = "rejected"
        elif change == "node":
            wrong.key_to_node["protected_structure"] = "another-result"
        else:
            wrong.key_to_node = {"unknown": "original-node"}
        with pytest.raises(ValueError):
            modules.evidence.retained_baseline_binding(wrong, original)


def _edit(modules, kind="set_parameters"):
    parent, revised = _candidate(), _candidate("candidate1")
    revised.recipe["context_id"] = "new"
    edit = {"kind": kind, "node": "protected_structure", "params": {"radius": 3}}
    revised.recipe["nodes"][0]["params"]["radius"] = 3
    if kind == "replace_operation":
        edit["operation_id"] = "closing"
        revised.recipe["nodes"][0]["operation_id"] = "closing"
        revised.recipe["nodes"][0]["params"] = {"radius": 3}
    compact = json.dumps(
        parent.recipe, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    revised.evidence["edit_receipt"] = {
        "parent_candidate_id": parent.candidate_id,
        "parent_recipe_sha256": modules.shared.sha256(compact),
        "edits": [edit],
    }
    action = SimpleNamespace(arguments={"candidate_id": "candidate0", "edits": [edit]})
    return parent, revised, action


@pytest.mark.parametrize("kind", ["set_parameters", "replace_operation"])
def test_literal_model_microedits_preserve_every_undeclared_field(modules, kind):
    parent, revised, action = _edit(modules, kind)
    result = modules.evidence.verify_microedit_diff(parent, revised, action)
    assert result["changed_node_keys"] == ["protected_structure"]
    assert result["literal_model_edits"] == action.arguments["edits"]
    assert parent.recipe["nodes"][0]["params"]["radius"] == 2


@pytest.mark.parametrize("change", ["wire", "params", "reason", "roles", "hash"])
def test_undeclared_recipe_or_role_change_is_not_a_microedit(modules, change):
    parent, revised, action = _edit(modules)
    if change == "wire":
        revised.recipe["connections"][0]["source_port"] = 1
    elif change == "params":
        revised.recipe["nodes"][0]["params"]["spatial_mode"] = "2D XY"
    elif change == "reason":
        revised.recipe["nodes"][0]["reason"] = "Silently changed"
    elif change == "roles":
        revised.output_roles[0]["role"] = "another-mask"
    else:
        revised.evidence["edit_receipt"]["parent_recipe_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        modules.evidence.verify_microedit_diff(parent, revised, action)


def test_selected_candidate_requires_exact_successful_product_protection(modules):
    candidate = _candidate()
    contract = {"constraint_id": "protected", "protected_count": 4}
    result = {
        "constraint_id": "protected",
        "run_id": "owned",
        "protected_count": 4,
        "retained_count": 4,
        "missing_count": 0,
        "passed": True,
        "candidate_key": "protected_structure",
        "candidate_port": 0,
    }
    candidate.evidence = {"preservation_check": result}
    assert modules.evidence.assert_selected_preservation(candidate, contract) == result
    for key, value in (
        ("passed", False),
        ("constraint_id", "different"),
        ("run_id", "unowned"),
        ("retained_count", 3),
        ("protected_count", 5),
        ("candidate_key", "another"),
        ("candidate_port", 1),
        ("missing_count", 1),
    ):
        wrong = deepcopy(candidate)
        wrong.evidence["preservation_check"][key] = value
        with pytest.raises(ValueError, match="product protection"):
            modules.evidence.assert_selected_preservation(wrong, contract)


def _request(modules):
    return {
        "objective": modules.observer.PROTECTED_GOAL,
        "clarifications": [
            {
                "question": "Additional instructions 1",
                "answer": modules.observer.PROTECTED_FEEDBACK,
            }
        ],
        "image_permissions": {"input_images": True, "output_images": True},
        "preservation_constraint": {"constraint_id": "exact", "grid": [1, 2, 3]},
    }


def test_request_exact_goal_feedback_permissions_and_canonical_contract(modules):
    observer = modules.observer.MicroeditObservers(None)
    request = _request(modules)
    observer.preservation_contract = {"constraint_id": "exact", "grid": (1, 2, 3)}
    observer.assert_request(request)
    for key, value in (
        ("objective", "simpler"),
        ("clarifications", []),
        ("image_permissions", {"input_images": False, "output_images": True}),
        ("preservation_constraint", {"constraint_id": "exact", "grid": [True, 2, 3]}),
    ):
        with pytest.raises(ValueError):
            observer.assert_request({**request, key: value})
    request["preservation_constraint"]["grid"][0] = math.nan
    with pytest.raises(ValueError):
        observer.assert_request(request)


@pytest.fixture
def guarded(modules, monkeypatch, tmp_path):
    from napari_vipp.core import workflow_investigation as engine
    from napari_vipp.core import workflow_investigation_model as provider
    from napari_vipp.core import workflow_proposal_codex as codex

    monkeypatch.setattr(
        provider,
        "generate_codex_investigation_action",
        lambda *_a, **_k: {"safe": "synthetic"},
    )
    monkeypatch.setattr(codex, "_call_codex", lambda *_a, **_k: {"safe": True})
    observer = modules.observer.MicroeditObservers(
        modules.shared.EvidenceJournal(tmp_path / "guarded")
    )
    observer.preservation_contract = _request(modules)["preservation_constraint"]
    monkeypatch.setattr(modules.observer, "monotonic", lambda: 1)
    observer.deadline = 2
    observer.install()
    yield SimpleNamespace(
        observer=observer, engine=engine, codex=codex, provider=provider
    )
    observer.close()


def test_guard_checks_deadline_and_actual_caps_before_each_boundary(modules, guarded):
    observer = guarded.observer
    settings = guarded.codex.CodexSettings(timeout_seconds=600, max_attempts=2)
    guarded.provider.generate_codex_investigation_action(_request(modules), settings)
    assert observer.counts["logical_requests"] == 1
    observer.counts["logical_requests"] = 32
    with pytest.raises(RuntimeError, match="32-action"):
        guarded.provider.generate_codex_investigation_action(
            _request(modules), settings
        )
    observer.counts["model_calls"] = 40
    with pytest.raises(RuntimeError, match="40-native"):
        guarded.codex._call_codex()
    observer.counts["actions"] = 32
    with pytest.raises(RuntimeError, match="32-action"):
        guarded.engine.InvestigationSession.perform(None, None)
    observer.counts["trial_executions"] = 3
    with pytest.raises(RuntimeError, match="three trial"):
        guarded.engine.execute_pipeline_request(None)
    observer.deadline = 1
    with pytest.raises(RuntimeError, match="deadline"):
        guarded.codex._call_codex()
    for function in (
        guarded.codex.start_codex_login,
        guarded.codex.repair_codex_profile,
    ):
        with pytest.raises(RuntimeError, match="Authentication"):
            function()


def test_rejected_request_is_recorded_before_raise(modules, guarded):
    with pytest.raises(ValueError):
        guarded.provider.generate_codex_investigation_action(
            {**_request(modules), "clarifications": []},
            SimpleNamespace(timeout_seconds=600, max_attempts=2),
        )
    root = guarded.observer.journal.directory
    assert (root / "rejected-requests/request-001/public-request.json").exists()
    assert (root / "rejected-requests/request-001/failure.json").exists()
    assert guarded.observer.counts["model_calls"] == 0
    assert guarded.observer.counts["logical_requests"] == 0


def test_complete_recorded_mesh_and_nonfinite_table_replay(
    modules, monkeypatch, tmp_path
):
    from napari_vipp.core.meshes import MeshData, MeshObject, MeshState
    from napari_vipp.core.metadata import AxisMetadata
    from napari_vipp.core.tables import TableData, TableState

    capture = importlib.import_module("run_workflow_investigation_acceptance")
    axes = tuple(
        AxisMetadata(name=a, type="space", scale=0.45, unit="micrometer") for a in "ZYX"
    )
    state = MeshState(
        3, 1, axes, (2, 2, 2), "Close at image border", history=("all metadata",)
    )
    mesh = MeshData(
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 2]]),
        state,
        objects=(MeshObject(1, "full object", source_history=("source",)),),
    )
    table = TableData(("nan", "pos", "neg"), ((math.nan, math.inf, -math.inf),))
    table_state = TableState(1, 3, table.columns, history=("full table state",))
    pipeline = SimpleNamespace(
        nodes={"mesh": object(), "table": object(), "review": object()},
        node_outputs={"mesh": [mesh], "table": [table], "review": [None]},
        outputs={"mesh": mesh, "table": table, "review": None},
        node_output_states={"mesh": [state], "table": [table_state], "review": [None]},
        output_states={"mesh": state, "table": table_state, "review": None},
        completed_node_ids={"mesh", "table"},
        node_execution_states={
            "mesh": "ready",
            "table": "ready",
            "review": "not_calculated",
        },
        node_execution_messages={},
        node_cache_lineage={},
        node_compute_provenance={},
    )
    monkeypatch.setattr("napari_vipp.core.workflow.serialize_workflow", lambda _: {})
    journal = modules.shared.EvidenceJournal(tmp_path / "full")
    capture.capture_pipeline(journal, "current", pipeline)
    full = modules.shared.read_recording_json(
        journal.directory / "current/full-state.json"
    )
    bundle = modules.evidence.recorded_port_bundle(journal.directory, full["ports"])
    for _node, _port, saved, expected in bundle.expected:
        restored, restored_state = modules.evidence.restore_recorded_port(
            saved, expected
        )
        assert modules.evidence.recording_bytes(saved["state"]) == (
            modules.evidence.recording_bytes(restored_state)
        )
        if "data" in saved:
            assert modules.evidence.recording_bytes(expected) == (
                modules.evidence.recording_bytes(restored)
            )
    records = modules.evidence.exact_baseline_replay(
        bundle, pipeline, {"mesh": "mesh", "table": "table", "review": "review"}
    )
    assert len(records) == 3 and all(r["complete_state_exact"] for r in records)
    pipeline.node_output_states["table"] = [None]
    with pytest.raises(ValueError, match="carried state"):
        modules.evidence.exact_baseline_replay(
            bundle, pipeline, {"mesh": "mesh", "table": "table", "review": "review"}
        )


def _additive_candidate():
    source = {"id": "input", "operation_id": "input", "params": {}}
    old = {"id": "baseline", "operation_id": "opening", "params": {"radius": 4}}
    trial = {"id": "trial", "operation_id": "opening", "params": {"radius": 3}}
    new = {**trial, "id": "proposal_0123456789ab_protected_structure"}

    def edge(target):
        return {"source": "input", "source_port": 0, "target": target, "target_port": 0}

    graph = {"nodes": [source, trial], "connections": [edge("trial")]}
    full = {
        "nodes": [source, old, new],
        "connections": [edge("baseline"), edge(new["id"])],
    }
    candidate = _candidate("candidate1")
    candidate.recipe["nodes"][0]["params"] = {"radius": 3}
    candidate.key_to_node = {"protected_structure": "trial"}
    candidate.graph = SimpleNamespace(to_pipeline=lambda: graph)
    return candidate, full, new["id"]


def test_final_normal_add_binds_entire_graph_and_explicit_candidate_ids(
    modules, monkeypatch
):
    monkeypatch.setattr("napari_vipp.core.workflow.serialize_workflow", lambda p: p)
    candidate, full, new = _additive_candidate()
    binding = modules.evidence.bind_selected_additive_replay(
        candidate, full, full, [new], "0123456789abcdef"
    )
    assert binding.node_mapping == {"trial": new}
    for mutation in ("old", "wire", "parameter", "ids"):
        changed = deepcopy(full)
        ids = [new]
        if mutation == "old":
            changed["nodes"][1]["params"]["radius"] = 5
        elif mutation == "wire":
            changed["connections"][-1]["source"] = "baseline"
        elif mutation == "parameter":
            changed["nodes"][-1]["params"]["radius"] = 7
        else:
            ids = [new, new]
        with pytest.raises(ValueError):
            modules.evidence.bind_selected_additive_replay(
                candidate, full, changed, ids, "0123456789abcdef"
            )


def test_recorder_import_inventory_rejects_any_changed_frozen_helper(
    modules, monkeypatch
):
    assert len(modules.runner.source_inventory()) == 10
    monkeypatch.setitem(modules.runner.FROZEN_HELPERS, "qualify_private.py", "0" * 64)
    with pytest.raises(ValueError, match="frozen recorder"):
        modules.runner.source_inventory()


def test_rehydration_refuses_unrecorded_or_unknown_nested_metadata(modules):
    saved = {
        "data": {
            "columns": [],
            "rows": [],
            "column_units": [],
            "histogram_metadata": {"unknown": "not inferred"},
        },
        "state": {},
    }
    with pytest.raises(ValueError, match="nested table metadata"):
        modules.evidence.restore_recorded_port(saved, saved["data"])
    with pytest.raises(ValueError, match="no supported complete state"):
        modules.evidence.restore_recorded_port({"state": None}, {"unexpected": 1})
