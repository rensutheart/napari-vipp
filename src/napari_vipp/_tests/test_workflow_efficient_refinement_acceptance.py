"""Fresh hinted-driver mechanics only; no model, native UI or scientific Runs."""

from __future__ import annotations

import ast
import importlib
import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("run_workflow_efficient_refinement")


def public_request(driver):
    return {
        "version": 2,
        "session_id": "synthetic",
        "objective": driver.EFFICIENT_GOAL,
        "clarifications": [
            {
                "question": "Additional instructions 1",
                "answer": driver.PROTECTED_FEEDBACK,
            }
        ],
        "image_permissions": {"input_images": True, "output_images": True},
        "preservation_constraint": {"constraint_id": "synthetic", "grid": [1, 2]},
        "evidence_notebook": {
            "version": 1,
            "session_id": "synthetic",
            "record_index": [],
            "working_records": [],
            "pinned_record_refs": [],
        },
        "closing_phase": {},
        "decision_context": {
            "kind": "napari-vipp-investigation-decision-context",
            "version": 1,
            "session_id": "synthetic",
            "recipes": [],
            "node_contracts": [],
            "numeric_evidence": [],
            "omitted": [],
        },
    }


def test_exact_hint_is_only_appended_to_prior_goal(driver):
    assert driver.GOAL_HINT == (
        "The image background is uneven; consider whether correcting it would "
        "improve the separation and band width."
    )
    assert driver.EFFICIENT_GOAL == driver.PROTECTED_GOAL + " " + driver.GOAL_HINT
    prior = importlib.import_module("run_workflow_evidence_notebook")
    assert driver.PROTECTED_FEEDBACK == prior.PROTECTED_FEEDBACK
    assert driver.PROTECTED_SELECTION == prior.PROTECTED_SELECTION
    assert driver.DEFAULT_BASELINE == prior.DEFAULT_BASELINE
    assert driver.APPROVED_INPUT == prior.APPROVED_INPUT
    assert driver.NEW_EVIDENCE_ROOT.name == "ai-assistant-efficient-refinement-20261017"


def test_fresh_source_pins_all_old_recorders_without_mutating_them(driver):
    inventory = driver.source_inventory()
    assert len(inventory) == len(driver.FROZEN_HELPERS) + 1 == 14
    assert inventory["run_workflow_evidence_notebook.py"] == (
        "f6018dd3216046e2cb942e1713f86f97dc5b6f3a55237f54ab8f80a3506b8a40"
    )
    assert inventory["workflow_evidence_notebook_geometry.py"] == (
        "b086d98f91b83207ad91a9d738a9758a36878e60a38f8f4d7eece85be03411c5"
    )
    assert not any("compare" in name or "continue_" in name for name in inventory)


def test_observer_checks_actual_wire_without_rewriting_it(driver):
    request = public_request(driver)
    original = deepcopy(request)
    observer = driver.EfficientObservers(None)
    observer.preservation_contract = deepcopy(request["preservation_constraint"])
    observer.assert_request(request)
    assert request == original


@pytest.mark.parametrize(
    "change",
    [
        "old_goal",
        "extra_hint",
        "feedback",
        "input",
        "output",
        "protection",
        "version",
        "float_version",
        "notebook_version",
        "session",
        "working",
        "closing",
        "legacy",
        "decision_context",
    ],
)
def test_observer_rejects_changed_authority(driver, change):
    request = public_request(driver)
    observer = driver.EfficientObservers(None)
    observer.preservation_contract = deepcopy(request["preservation_constraint"])
    if change == "old_goal":
        request["objective"] = driver.PROTECTED_GOAL
    elif change == "extra_hint":
        request["objective"] += " Choose a preset method."
    elif change == "feedback":
        request["clarifications"][0]["answer"] += " Changed"
    elif change in {"input", "output"}:
        request["image_permissions"][change + "_images"] = False
    elif change == "protection":
        request["preservation_constraint"]["grid"][0] = True
    elif change == "version":
        request["version"] = True
    elif change == "float_version":
        request["version"] = 2.0
    elif change == "notebook_version":
        request["evidence_notebook"]["version"] = True
    elif change == "session":
        request["evidence_notebook"]["session_id"] = "other"
    elif change == "working":
        request["evidence_notebook"]["working_records"] = {}
    elif change == "closing":
        request["closing_phase"] = []
    elif change == "legacy":
        request["history"] = []
    else:
        request["decision_context"]["session_id"] = "other"
    with pytest.raises(ValueError):
        observer.assert_request(request)


class NormalizeFreshProtocol(ast.NodeTransformer):
    """Only new authorized wording/root/title/protocol annotations differ."""

    def visit_Name(self, node):
        node.id = {
            "EFFICIENT_GOAL": "PROTECTED_GOAL",
            "EfficientObservers": "NotebookObservers",
        }.get(node.id, node.id)
        return node

    def visit_Constant(self, node):
        if node.value == "VIPP — efficient protected refinement testing":
            node.value = "VIPP — protected evidence-notebook testing"
        return node

    def visit_Dict(self, node):
        removed = {
            "prior_protected_goal",
            "goal_hint_added",
            "prompt_changed",
            "is_fresh_authorized_efficient_refinement",
        }
        kept = [
            (key, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if not isinstance(key, ast.Constant) or key.value not in removed
        ]
        node.keys, node.values = (
            map(list, zip(*kept, strict=True)) if kept else ([], [])
        )
        return self.generic_visit(node)


def test_normal_native_replay_limits_and_geometry_are_unchanged(driver):
    prior = importlib.import_module("run_workflow_evidence_notebook")
    actual = NormalizeFreshProtocol().visit(
        ast.parse(inspect.getsource(driver.run_live))
    )
    expected = ast.parse(inspect.getsource(prior.run_live))
    assert ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def test_cache_only_preflight_uses_new_goal_and_current_preview_contract(driver):
    prior = importlib.import_module("workflow_evidence_notebook_preflight")
    actual = ast.parse(inspect.getsource(driver.offline_efficient_compatibility))
    expected = ast.parse(inspect.getsource(prior.offline_notebook_compatibility))
    actual.body[0].name = expected.body[0].name
    # Only the new passive decision-context compatibility assertion/receipt is
    # additive; the cache restore, source/grid/role binding and no-Run path stay
    # byte-equivalent at the AST boundary below.
    actual.body[0].body = [
        node
        for node in actual.body[0].body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "decision"
                for target in node.targets
            )
        )
        and not (isinstance(node, ast.If) and "decision" in ast.unparse(node.test))
    ]
    for node in ast.walk(actual):
        if isinstance(node, ast.Dict):
            pairs = [
                (key, value)
                for key, value in zip(node.keys, node.values, strict=True)
                if not (
                    isinstance(key, ast.Constant) and key.value == "decision_context"
                )
            ]
            node.keys = [key for key, _ in pairs]
            node.values = [value for _, value in pairs]
    for node in ast.walk(actual):
        if isinstance(node, ast.Name) and node.id == "EfficientObservers":
            node.id = "NotebookObservers"
    assert ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )
    source = inspect.getsource(driver.qualified_core_preflight)
    assert "bundle, EFFICIENT_GOAL, PROTECTED_FEEDBACK" in source
    assert "offline_efficient_compatibility" in source


def test_preflight_default_has_no_live_output_or_run_authority(driver, monkeypatch):
    captured = []
    bundle = SimpleNamespace(
        lock_sha256="synthetic", key_to_node={"a": "n"}, expected=(), buffer_count=0
    )
    monkeypatch.setattr(driver, "source_inventory", dict)
    monkeypatch.setattr(driver, "verify_input", lambda _: {})
    monkeypatch.setattr(driver, "load_completed_baseline", lambda _: bundle)
    monkeypatch.setattr(driver, "verify_receipt", lambda *_: {"wheel_sha256": "test"})
    monkeypatch.setattr(driver, "require_product_api", lambda *_: None)
    monkeypatch.setattr(driver, "qualified_core_preflight", lambda *_: {})
    monkeypatch.setattr(driver, "run_live", lambda *_: captured.append("live"))
    assert (
        driver.main(
            [
                "--qualified-source",
                str(Path(__file__).resolve().parent),
                "--qualification-receipt",
                "synthetic.json",
            ]
        )
        == 0
    )
    assert not captured


def test_explicit_eighth_diagnostic_role_is_preserved_not_filtered(driver):
    import numpy as np

    from napari_vipp._tests.test_workflow_evidence_notebook_acceptance import (
        mesh_candidate,
    )
    from napari_vipp.core.meshes import MeshData
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.workflow_assessment import _capture_binding

    candidate = mesh_candidate()
    for role in (
        "band_mask",
        "circle_mask",
        "band_labels",
        "circle_labels",
        "combined_mesh",
        "diagnostic_output",
    ):
        key, node = role + "_key", role + "_actual"
        if role == "combined_mesh":
            original = candidate._outputs[0][1]
            state = original.state
            data = MeshData(
                original.data.vertices.copy(), original.data.faces.copy(), state
            )
        else:
            data = np.ones((2, 2, 2), dtype=np.float32)
            state = image_state_from_array(
                data,
                axes=tuple(AxisMetadata(a, "space", "micrometer", 0.45) for a in "ZYX"),
            )
        candidate.recipe["nodes"].append({"key": key})
        candidate.key_to_node[key] = node
        candidate.output_roles.append({"role": role, "node": key, "port": 0})
        candidate._outputs += ((role, _capture_binding(node, 0, data, state, [0])),)
    original_roles = deepcopy(candidate.output_roles)
    assert len(candidate.output_roles) == len(candidate._outputs) == 8
    assert driver.verify_closed_mesh_deliverables(candidate)["passed"] is True
    assert candidate.output_roles == original_roles
    assert len(driver.candidate_key_mapping(candidate)) == 8
    source = inspect.getsource(driver.run_live)
    assert "for role, binding in selected._outputs:" in source
