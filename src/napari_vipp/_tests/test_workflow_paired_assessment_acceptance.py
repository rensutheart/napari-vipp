"""Sealed fresh-driver mechanics; no provider, native UI or scientific Runs."""

from __future__ import annotations

import ast
import importlib
import inspect
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_workflow_efficient_refinement_acceptance import (
    public_request,
)


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return importlib.import_module("run_workflow_paired_assessment")


def test_same_original_baseline_goal_feedback_and_source(driver):
    prior = importlib.import_module("run_workflow_efficient_refinement")
    for name in (
        "DEFAULT_BASELINE",
        "APPROVED_INPUT",
        "GOAL_HINT",
        "EFFICIENT_GOAL",
        "PROTECTED_GOAL",
        "PROTECTED_FEEDBACK",
        "PROTECTED_SELECTION",
    ):
        assert getattr(driver, name) == getattr(prior, name)
    assert driver.NEW_EVIDENCE_ROOT.name == "ai-assistant-paired-assessment-20261017"


def test_every_old_recorder_and_the_old_driver_are_frozen(driver):
    prior = importlib.import_module("run_workflow_efficient_refinement")
    inventory = driver.source_inventory()
    assert len(inventory) == 15
    assert inventory["run_workflow_efficient_refinement.py"] == (
        "c53d13df909487494c0ed1d2f70f60e7acd01887c4f12b4d792955dbab2389cd"
    )
    assert all(inventory[name] == value for name, value in prior.FROZEN_HELPERS.items())


class NormalizeFreshProtocol(ast.NodeTransformer):
    """Only observation, new provenance markers and the window title differ."""

    def visit_Name(self, node):
        if node.id == "PairedAssessmentObservers":
            node.id = "EfficientObservers"
        return node

    def visit_Constant(self, node):
        if node.value == "VIPP — paired protected refinement testing":
            node.value = "VIPP — efficient protected refinement testing"
        return node

    def visit_Dict(self, node):
        removed = {
            "is_fresh_authorized_paired_assessment",
            "matched_assessment_recording_only",
            "comparison_previews",
        }
        kept = [
            (key, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if not isinstance(key, ast.Constant) or key.value not in removed
        ]
        node.keys = [key for key, _ in kept]
        node.values = [value for _, value in kept]
        return self.generic_visit(node)

    def visit_If(self, node):
        if any(
            isinstance(item, ast.Constant) and item.value == "comparison_previews"
            for item in ast.walk(node.test)
        ):
            return None
        return self.generic_visit(node)


@pytest.mark.parametrize(
    "function",
    ["run_live", "qualified_core_preflight", "offline_efficient_compatibility"],
)
def test_original_normal_replay_geometry_and_preflight_flow_unchanged(driver, function):
    prior = importlib.import_module("run_workflow_efficient_refinement")
    actual = NormalizeFreshProtocol().visit(
        ast.parse(inspect.getsource(getattr(driver, function)))
    )
    expected = ast.parse(inspect.getsource(getattr(prior, function)))
    assert ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def test_original_wire_authority_checker_is_unchanged(driver):
    prior = importlib.import_module("run_workflow_efficient_refinement")
    assert ast.dump(
        ast.parse(
            dedent(inspect.getsource(driver.PairedAssessmentObservers.assert_request))
        ),
        include_attributes=False,
    ) == ast.dump(
        ast.parse(dedent(inspect.getsource(prior.EfficientObservers.assert_request))),
        include_attributes=False,
    )


def test_cache_only_preflight_adds_only_field_presence_and_literal_receipt(driver):
    source = inspect.getsource(driver.offline_efficient_compatibility)
    assert 'if type(request.get("comparison_previews")) is not list:' in source
    assert '"comparison_previews": deepcopy(request["comparison_previews"]),' in source
    assert "_request(\n        request, session.preview_images()" in source


def test_observer_does_not_modify_authorized_request(driver):
    request = public_request(driver)
    original = deepcopy(request)
    observer = driver.PairedAssessmentObservers(None)
    observer.preservation_contract = deepcopy(request["preservation_constraint"])
    observer.assert_request(request)
    assert request == original


class Preview:
    def __init__(self, sha):
        self.value = {"sha256": sha, "label": "synthetic", "byte_count": 1}

    def manifest(self):
        return deepcopy(self.value)


def observed_wire(driver, *, status="supplied"):
    request = public_request(driver)
    previews = (Preview("a" * 64), Preview("b" * 64))
    request["image_previews"] = [preview.manifest() for preview in previews]
    request["comparison_previews"] = [
        {
            "candidate_ids": ["candidate0", "candidate1"],
            "role": "band_mask",
            "selection": {
                "grid": {"shape": [2, 3, 4]},
                "roi": None,
                "view": "XY",
                "display_policy": "synthetic",
                "source_sha256": "c" * 64,
                "source_state_sha256": "d" * 64,
            },
            "views": [
                {
                    "candidate_id": "candidate0",
                    "view_id": "first",
                    "preview_sha256": "a" * 64,
                },
                {
                    "candidate_id": "candidate1",
                    "view_id": "second",
                    "preview_sha256": "b" * 64,
                },
            ]
            if status == "supplied"
            else [],
            "status": status,
            "reason": "Exact synthetic observation, not quality evidence.",
        }
    ]
    request["decision_context"]["numeric_evidence"] = [
        {
            "alias": "candidate0:candidate1/mesh_topology_review",
            "payload": {
                "kind": "mesh-topology-review",
                "roles": [{"role": "band_mesh", "before": 2, "after": 0}],
                "warnings": ["Closed edges do not certify a simple solid."],
                "limitations": ["No anatomical claim."],
            },
            "evidence_ref": {"kind": "numeric", "sha256": "e" * 64},
        },
        {"alias": "other", "payload": {"kind": "ordinary"}},
    ]
    return request, previews


@pytest.mark.parametrize("status", ["supplied", "omitted", "unavailable"])
def test_actual_binding_receipt_copies_exact_fields_without_new_authority(
    driver, status
):
    request, previews = observed_wire(driver, status=status)
    original = deepcopy(request)
    receipt = driver.paired_binding_receipt(request, previews, reserved_model_call=7)
    assert request == original
    assert receipt["request_sha256"] == driver.sha256(driver.public_bytes(request))
    assert receipt["comparison_previews"] == request["comparison_previews"]
    assert receipt["mesh_topology_reviews"] == [
        request["decision_context"]["numeric_evidence"][0]
    ]
    assert receipt["image_previews"] == request["image_previews"]
    assert receipt["native_transport_artifact_prefix"] == "transport/call-007"
    assert receipt["native_transport_recorded_separately"] is True
    assert receipt["scientific_or_quality_judgment_by_observer"] is False
    check = receipt["comparison_preview_checks"][0]
    assert check["exactly_two_current_views"] is (status == "supplied")
    receipt["comparison_previews"][0]["reason"] = "Detached recording"
    receipt["mesh_topology_reviews"][0]["payload"]["warnings"].clear()
    assert request == original


def test_no_optional_descriptor_is_not_fabricated(driver):
    request = public_request(driver)
    request["image_previews"] = []
    receipt = driver.paired_binding_receipt(request, (), reserved_model_call=1)
    assert receipt["comparison_previews_present"] is False
    assert receipt["comparison_previews"] == []
    assert receipt["mesh_topology_reviews"] == []


def test_descriptor_does_not_make_an_absent_png_present(driver):
    request, previews = observed_wire(driver)
    request["comparison_previews"][0]["views"][1]["preview_sha256"] = "f" * 64
    receipt = driver.paired_binding_receipt(request, previews, reserved_model_call=1)
    assert receipt["comparison_previews"] == request["comparison_previews"]
    check = receipt["comparison_preview_checks"][0]
    assert check["all_declared_hashes_on_current_board"] is False
    assert check["exactly_two_current_views"] is False


def test_changed_actual_png_board_fails_recording(driver):
    request, previews = observed_wire(driver)
    request["image_previews"].reverse()
    with pytest.raises(ValueError, match="different PNG board"):
        driver.paired_binding_receipt(request, previews, reserved_model_call=1)


@pytest.mark.parametrize("failed", [False, True])
def test_binding_observation_runs_only_after_original_normal_binding(
    driver, monkeypatch, failed
):
    from napari_vipp.core import workflow_investigation as engine

    calls, result = [], object()
    request, previews = observed_wire(driver)

    class Session:
        model_calls_used = 0

        def record_model_call(self, *, request):
            calls.append(("original", request))
            if failed:
                raise RuntimeError("Original binding denied")
            self.model_calls_used += 1
            return result

        def preview_images(self):
            return previews

    class Journal:
        def document(self, path, value):
            calls.append(("document", path, deepcopy(value)))

        def event(self, kind, **details):
            calls.append(("event", kind, details))

    monkeypatch.setattr(driver.LegacyNotebookObservers, "install", lambda _: None)
    monkeypatch.setattr(engine, "InvestigationSession", Session)
    observer = driver.PairedAssessmentObservers(Journal())
    observer.install()
    try:
        session = Session()
        if failed:
            with pytest.raises(RuntimeError, match="Original binding denied"):
                session.record_model_call(request=request)
            assert len(calls) == 1
            assert session.model_calls_used == 0
        else:
            assert session.record_model_call(request=request) is result
            assert [entry[0] for entry in calls] == ["original", "document", "event"]
            assert session.model_calls_used == 1
            assert calls[1][1] == (
                "paired-assessment/transmissions/attempt-001/actual-binding.json"
            )
            assert calls[0][1] is request
    finally:
        observer.close()


def test_other_baseline_is_denied_before_any_preflight(driver, monkeypatch):
    monkeypatch.setattr(
        driver, "source_inventory", lambda: pytest.fail("No baseline substitution")
    )
    with pytest.raises(SystemExit) as exc:
        driver.main(
            [
                "--baseline-run",
                str(Path(__file__).resolve().parent),
                "--qualified-source",
                "synthetic",
                "--qualification-receipt",
                "synthetic.json",
            ]
        )
    assert exc.value.code == 2


def test_default_stays_read_only_without_live_launch(driver, monkeypatch):
    launched = []
    bundle = SimpleNamespace(
        lock_sha256="synthetic", key_to_node={"a": "n"}, expected=(), buffer_count=0
    )
    monkeypatch.setattr(driver, "source_inventory", dict)
    monkeypatch.setattr(driver, "verify_input", lambda _: {})
    monkeypatch.setattr(driver, "load_completed_baseline", lambda _: bundle)
    monkeypatch.setattr(driver, "verify_receipt", lambda *_: {"wheel_sha256": "test"})
    monkeypatch.setattr(driver, "require_product_api", lambda *_: None)
    monkeypatch.setattr(driver, "qualified_core_preflight", lambda *_: {})
    monkeypatch.setattr(driver, "run_live", lambda *_: launched.append(True))
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
    assert not launched
