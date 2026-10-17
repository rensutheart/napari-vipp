"""Synthetic recorder fixtures; no model, native UI, pipeline or reference data."""

from __future__ import annotations

import importlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return SimpleNamespace(
        observer=importlib.import_module("workflow_evidence_notebook_observers"),
        runner=importlib.import_module("run_workflow_evidence_notebook"),
        geometry=importlib.import_module("workflow_evidence_notebook_geometry"),
        shared=importlib.import_module("workflow_investigation_evidence"),
    )


def request(modules):
    return {
        "version": 2,
        "session_id": "synthetic",
        "objective": modules.runner.PROTECTED_GOAL,
        "clarifications": [
            {
                "question": "Additional instructions 1",
                "answer": modules.runner.PROTECTED_FEEDBACK,
            }
        ],
        "image_permissions": {"input_images": True, "output_images": True},
        "preservation_constraint": {"constraint_id": "synthetic", "grid": [1, 2, 3]},
        "evidence_notebook": {
            "version": 1,
            "session_id": "synthetic",
            "record_index": [],
            "working_records": [],
            "pinned_record_refs": [],
        },
        "closing_phase": {},
    }


def test_new_runner_pins_old_helpers_without_importing_closed_runner(modules):
    sources = modules.runner.source_inventory()
    assert len(sources) == len(modules.runner.FROZEN_HELPERS) + 4
    assert "run_workflow_microedit_investigation.py" not in sources
    assert "continue_workflow_microedit_assessment.py" not in sources
    assert "compare_microedit_terminal.py" not in sources
    assert sources["workflow_microedit_observers.py"] == (
        "f606f4def247b9a78bd62fefe83fa9dad1b311387e012c7806d28bd78d213415"
    )
    assert (
        modules.runner.NEW_EVIDENCE_ROOT.name
        == "ai-assistant-evidence-notebook-20261017"
    )


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "float-version",
        "notebook-version",
        "legacy",
        "session",
        "working",
        "closing",
        "goal",
        "sharing",
        "protection",
    ],
)
def test_notebook_wire_preserves_every_authorized_boundary(modules, change):
    observer = modules.observer.NotebookObservers(None)
    value = request(modules)
    observer.preservation_contract = deepcopy(value["preservation_constraint"])
    observer.assert_request(value)
    if change == "version":
        value["version"] = 1
    elif change == "float-version":
        value["version"] = 2.0
    elif change == "notebook-version":
        value["evidence_notebook"]["version"] = True
    elif change == "legacy":
        value["history"] = []
    elif change == "session":
        value["evidence_notebook"]["session_id"] = "other"
    elif change == "working":
        value["evidence_notebook"]["working_records"] = {}
    elif change == "closing":
        value["closing_phase"] = []
    elif change == "goal":
        value["objective"] = "Changed objective"
    elif change == "sharing":
        value["image_permissions"]["output_images"] = False
    else:
        value["preservation_constraint"]["grid"][0] = True
    with pytest.raises(ValueError):
        observer.assert_request(value)


@pytest.mark.parametrize("value", [{"value": math.nan}, {"value": b"not-public"}, []])
def test_public_notebook_never_uses_nonfinite_or_private_byte_codec(modules, value):
    with pytest.raises((ValueError, TypeError)):
        modules.observer.public_bytes(value)


class SyntheticSession:
    def __init__(self):
        self.actions_used = self.model_calls_used = self.attempted_trials = 0
        self.records, self.pins, self.transmissions = [], [], []

    def export_notebook(self):
        if getattr(self, "fail_export", False):
            raise ValueError("Synthetic notebook export failure")
        return {
            "records": self.records,
            "pins": self.pins,
            "transmissions": self.transmissions,
        }


def test_export_is_detached_and_cannot_reserve_action_or_supply_handles(modules):
    session = SyntheticSession()
    session.records.append({"record_id": "entry1", "payload": {"count": 3}})
    exported = modules.observer.assert_pure_export(session)
    exported["records"][0]["payload"]["count"] = 99
    assert session.records[0]["payload"]["count"] == 3
    assert not session.transmissions
    assert (
        session.actions_used,
        session.model_calls_used,
        session.attempted_trials,
    ) == (0, 0, 0)


def test_impure_export_fails_instead_of_advancing_unrecorded_authority(modules):
    session = SyntheticSession()

    def changes_records():
        session.records.append({"record_id": str(len(session.records))})
        return {"records": session.records}

    session.export_notebook = changes_records
    with pytest.raises(ValueError, match="export changed"):
        modules.observer.assert_pure_export(session)


@pytest.fixture
def observed(modules, monkeypatch, tmp_path):
    from napari_vipp.core import workflow_investigation as engine

    observer = modules.observer.NotebookObservers(
        modules.shared.EvidenceJournal(tmp_path / "notebook")
    )
    observer.preservation_contract = request(modules)["preservation_constraint"]
    monkeypatch.setattr(
        modules.observer.MicroeditObservers, "install", lambda _self: None
    )
    monkeypatch.setattr(observer, "_check_deadline", lambda: None)

    def perform(session, action):
        observer.counts["actions"] += 1
        session.actions_used += 1
        session.records.append(deepcopy(action))
        session.fail_export = action.get("fail_export", False)
        if action.get("fail"):
            raise ValueError("Synthetic action failed after preserving its record")
        return {"actual_product_result": action}

    def supply(session, count=1, *, request):
        if getattr(session, "fail_supply", False):
            raise ValueError("Synthetic product binding failure")
        session.model_calls_used += count
        session.transmissions.append(
            modules.shared.sha256(modules.observer.public_bytes(request))
        )

    monkeypatch.setattr(engine.InvestigationSession, "perform", perform)
    monkeypatch.setattr(engine.InvestigationSession, "record_model_call", supply)
    observer.install()
    yield SimpleNamespace(observer=observer, session=SyntheticSession(), engine=engine)
    observer.close()


@pytest.mark.parametrize(
    "action",
    [
        "read_notebook",
        "pin_notebook",
        "inspect_candidate",
        "plan_experiment",
        "assess_experiment",
        "finish",
    ],
)
def test_all_typed_actions_have_full_before_after_notebook_records(
    modules, observed, action
):
    observed.engine.InvestigationSession.perform(observed.session, {"action": action})
    root = observed.observer.journal.directory
    before = json.loads(
        (root / "notebook/actions/action-001/before.json").read_text("utf-8")
    )
    after = json.loads(
        (root / "notebook/actions/action-001/after.json").read_text("utf-8")
    )
    assert before["records"] == []
    assert after["records"] == [{"action": action}]
    assert not after["transmissions"]
    assert observed.observer.counts["trial_executions"] == 0


def mesh_candidate(*, open_band=False, empty_circle=False, duplicate_circle=False):
    import numpy as np

    from napari_vipp.core.meshes import MeshData, MeshState
    from napari_vipp.core.metadata import AxisMetadata
    from napari_vipp.core.workflow_assessment import _capture_binding

    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    roles, outputs, mapping = [], [], {}
    for role in ("band_mesh", "circle_mesh"):
        actual_faces = faces.copy()
        if role == "band_mesh" and open_band:
            actual_faces = actual_faces[:-1]
        if role == "circle_mesh" and empty_circle:
            actual_faces = actual_faces[:0]
        if role == "circle_mesh" and duplicate_circle:
            actual_faces = np.concatenate([actual_faces, actual_faces[:1]])
        state = MeshState(
            len(vertices),
            len(actual_faces),
            tuple(
                AxisMetadata(name=a, type="space", scale=0.45, unit="micrometer")
                for a in "ZYX"
            ),
            (2, 2, 2),
            "Close at image border",
            history=("Synthetic fixture only",),
        )
        data = MeshData(vertices.copy(), actual_faces, state)
        key, node = role + "_key", role + "_actual"
        roles.append({"role": role, "node": key, "port": 0})
        mapping[key] = node
        outputs.append((role, _capture_binding(node, 0, data, state, [0])))
    return SimpleNamespace(
        candidate_id="candidate1",
        run_id="synthetic-run",
        status="completed",
        recipe={"nodes": [{"key": key} for key in mapping]},
        key_to_node=mapping,
        output_roles=roles,
        _outputs=tuple(outputs),
    )


def test_closed_end_gate_uses_both_actual_role_bound_meshes(modules):
    candidate = mesh_candidate()
    result = modules.geometry.verify_closed_mesh_deliverables(candidate)
    assert result["passed"] is True
    assert [row["role"] for row in result["meshes"]] == ["band_mesh", "circle_mesh"]
    assert all(row["topology"]["boundary_edge_count"] == 0 for row in result["meshes"])
    assert all(
        row["carried_mesh_state"]["source_shape"] == [2, 2, 2]
        for row in result["meshes"]
    )
    assert not result["full_solid_manifold_or_anatomical_validity_claimed"]
    assert not result["geometry_modified_or_repaired"]


@pytest.mark.parametrize(
    "change",
    ["open", "empty", "role", "boolean-port", "binding-port", "metadata", "rejected"],
)
def test_closed_end_gate_rejects_open_empty_or_unbound_deliverables(modules, change):
    from dataclasses import replace

    candidate = mesh_candidate(
        open_band=change == "open", empty_circle=change == "empty"
    )
    if change == "role":
        candidate.output_roles[1]["role"] = "unrelated_mesh"
    elif change == "boolean-port":
        candidate.output_roles[1]["port"] = False
    elif change in {"binding-port", "metadata"}:
        role, binding = candidate._outputs[1]
        binding = (
            replace(binding, port=1)
            if change == "binding-port"
            else replace(binding, state=None)
        )
        candidate._outputs = (candidate._outputs[0], (role, binding))
    elif change == "rejected":
        candidate.status = "rejected"
    with pytest.raises(modules.geometry.ClosedMeshGeometryError) as caught:
        modules.geometry.verify_closed_mesh_deliverables(candidate)
    assert caught.value.receipt["passed"] is False
    assert caught.value.receipt["geometry_modified_or_repaired"] is False


def test_nonmanifold_duplicates_and_components_are_reported_not_hidden_extra_gates(
    modules,
):
    result = modules.geometry.verify_closed_mesh_deliverables(
        mesh_candidate(duplicate_circle=True)
    )
    assert result["passed"] is True
    circle = result["meshes"][1]
    assert circle["topology"]["boundary_edge_count"] == 0
    assert circle["topology"]["nonmanifold_edge_count"] == 3
    assert circle["topology"]["duplicate_triangle_count"] == 1
    assert circle["other_diagnostics_not_additional_gates"]


def test_action_failure_does_not_erase_complete_local_evidence(modules, observed):
    with pytest.raises(ValueError, match="Synthetic action failed"):
        observed.engine.InvestigationSession.perform(
            observed.session, {"action": "inspect_candidate", "fail": True}
        )
    path = (
        observed.observer.journal.directory / "notebook/actions/action-001/after.json"
    )
    assert json.loads(path.read_text("utf-8"))["records"][0]["fail"] is True


def test_only_actual_before_call_binds_its_exact_wire_request(modules, observed):
    value = request(modules)
    modules.observer.assert_pure_export(observed.session)
    assert not observed.session.transmissions
    observed.engine.InvestigationSession.record_model_call(
        observed.session, request=value
    )
    root = observed.observer.journal.directory / "notebook/transmissions/attempt-001"
    assert json.loads((root / "outgoing-request.json").read_text("utf-8")) == value
    after = json.loads((root / "after.json").read_text("utf-8"))
    assert after["transmissions"] == [
        modules.shared.sha256(modules.observer.public_bytes(value))
    ]
    assert observed.session.model_calls_used == 1
    assert observed.observer.counts["model_calls"] == 0  # native boundary is separate
    assert observed.observer.counts["trial_executions"] == 0
    with pytest.raises(ValueError, match="exact outgoing"):
        observed.engine.InvestigationSession.record_model_call(observed.session)
    assert observed.session.model_calls_used == 1


def test_export_failure_does_not_replace_original_action_exception(modules, observed):
    with pytest.raises(ValueError, match="Synthetic action failed"):
        observed.engine.InvestigationSession.perform(
            observed.session,
            {"action": "inspect_candidate", "fail": True, "fail_export": True},
        )
    events = (observed.observer.journal.directory / "events.jsonl").read_text("utf-8")
    assert "notebook_export_failed_during_action_failure" in events
    assert "Synthetic notebook export failure" in events


def test_rejected_actual_request_keeps_failure_receipt_without_native_call(
    modules, observed
):
    observed.session.fail_supply = True
    with pytest.raises(ValueError, match="Synthetic product binding failure"):
        observed.engine.InvestigationSession.record_model_call(
            observed.session, request=request(modules)
        )
    root = observed.observer.journal.directory / "notebook/transmissions/attempt-001"
    assert (
        json.loads((root / "binding-failure.json").read_text("utf-8"))["type"]
        == "ValueError"
    )
    assert (root / "after-failure.json").exists()
    assert observed.session.model_calls_used == 0
    assert observed.observer.counts["model_calls"] == 0  # native boundary is separate
    assert observed.observer.counts["trial_executions"] == 0
