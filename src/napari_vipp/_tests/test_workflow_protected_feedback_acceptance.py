"""Protected-pass harness mechanics; no provider, UI or scientific calculation."""

from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    return (
        importlib.import_module("run_workflow_feedback_refinement"),
        importlib.import_module("run_workflow_protected_feedback_refinement"),
    )


def test_clarified_goal_and_latest_feedback_remain_distinct(modules):
    runner, entry = modules
    assert entry.PROTECTED_GOAL != runner.ORIGINAL_GOAL
    assert len(entry.PROTECTED_GOAL) < 4000
    request = {
        "objective": entry.PROTECTED_GOAL,
        "clarifications": [
            {
                "question": "Additional instructions 1",
                "answer": entry.PROTECTED_FEEDBACK,
            }
        ],
    }
    runner.assert_feedback_request(
        request, entry.PROTECTED_FEEDBACK, entry.PROTECTED_GOAL
    )
    with pytest.raises(ValueError, match="original goal"):
        runner.assert_feedback_request(request, entry.PROTECTED_FEEDBACK)


def test_normal_product_api_receives_only_exact_authorized_prior_refs(modules):
    runner, entry = modules
    calls = []
    contract = {"snapshot_sha256": "product-frozen", "protected_foreground_count": 3}
    constraint = SimpleNamespace(public_contract=lambda: deepcopy(contract))
    controller = SimpleNamespace(
        configure_preservation=lambda *refs: calls.append(refs) or constraint
    )
    result, receipt = runner.configure_product_preservation(
        controller,
        entry.PROTECTED_SELECTION,
        {
            "refine_reference_labels": "owned-band",
            "refine_circle_labels": "owned-region",
        },
    )
    assert result is constraint
    assert calls == [(("owned-band", 0), ("owned-region", 0))]
    assert receipt["product_contract"] == contract
    assert receipt["harness_region_or_pixels_generated"] is False
    assert receipt["configured_via_normal_product_api"] is True
    assert receipt["authorized_model_key_selection"] == entry.PROTECTED_SELECTION


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("mask_model_key", "unselected"),
        ("region_model_key", "unselected"),
        ("mask_model_key", "refine_circle_labels"),
        ("region_model_key", "refine_reference_labels"),
        ("mask_port", True),
        ("mask_port", 1),
        ("region_port", -1),
        ("policy", "repair_after_calculation"),
        ("proposal_key", "pick_best_mask"),
    ],
)
def test_changed_protection_refs_or_policy_fail_before_product(modules, key, value):
    runner, entry = modules
    calls = []
    controller = SimpleNamespace(
        configure_preservation=lambda *refs: calls.append(refs)
    )
    selection = {**entry.PROTECTED_SELECTION, key: value}
    with pytest.raises(ValueError, match="authorized|not exact"):
        runner.configure_product_preservation(
            controller,
            selection,
            {"refine_reference_labels": "band", "refine_circle_labels": "region"},
        )
    assert calls == []


def test_product_preservation_failure_is_not_repaired_or_substituted(modules):
    runner, entry = modules

    def reject(*_refs):
        raise ValueError("Product denied mismatched image grids")

    with pytest.raises(ValueError, match="mismatched image grids"):
        runner.configure_product_preservation(
            SimpleNamespace(configure_preservation=reject),
            entry.PROTECTED_SELECTION,
            {"refine_reference_labels": "band", "refine_circle_labels": "region"},
        )


def test_preflight_requires_qualified_normal_product_api(modules, tmp_path):
    runner, _entry = modules
    path = tmp_path / "package/ui/workflow_assistant_refinement.py"
    path.parent.mkdir(parents=True)
    path.write_text("class AssistantRefinementController:\n    pass\n", "utf-8")
    with pytest.raises(ValueError, match="lacks the product preservation API"):
        runner.require_product_preservation_api(
            tmp_path, {"package_directory": "package"}
        )
    path.write_text(
        "class AssistantRefinementController:\n"
        "    def configure_preservation(self, primary_ref, region_ref=None):\n"
        "        pass\n",
        "utf-8",
    )
    runner.require_product_preservation_api(tmp_path, {"package_directory": "package"})


def test_all_three_requests_carry_exact_product_constraint_and_preview_consent(modules):
    runner, _entry = modules
    contract = {"constraint_id": "frozen"}
    permissions = {"input_images": True, "output_images": True}
    first = {"preservation_constraint": contract, "image_permissions": permissions}
    revision = {
        "result_feedback": {
            "preservation_constraint": contract,
            "evidence": {"image_permissions": permissions},
        }
    }
    passed = {"passed": True, "constraint_id": "frozen"}
    final = {**first, "preservation_check": passed}
    runner.assert_preservation_request(first, "assessment", 1, contract)
    runner.assert_preservation_request(revision, "revision", 2, contract)
    runner.assert_preservation_request(final, "assessment", 3, contract, passed)
    for request, kind, index, current, result in (
        (first, "revision", 1, contract, None),
        (first, "assessment", 1, None, None),
        (first, "assessment", 1, {"constraint_id": "other"}, None),
        ({**first, "image_permissions": {}}, "assessment", 1, contract, None),
        (final, "assessment", 3, contract, None),
        (final, "assessment", 3, contract, {**passed, "passed": False}),
    ):
        with pytest.raises(ValueError, match="constraint|contract|permissions|check"):
            runner.assert_preservation_request(request, kind, index, current, result)


def test_current_passed_owned_result_required_before_success_lock(modules):
    runner, _entry = modules
    contract = {"constraint_id": "frozen"}
    constraint = SimpleNamespace(public_contract=lambda: deepcopy(contract))
    configuration = {"product_contract": contract}
    passed = {
        "constraint_id": "frozen",
        "candidate_key": "protected_structure",
        "run_id": "owned-refined-run",
        "protected_count": 3,
        "retained_count": 3,
        "missing_count": 0,
        "passed": True,
    }
    controller = SimpleNamespace(
        preservation_constraint=constraint,
        preservation_result=SimpleNamespace(
            passed=True, public_result=lambda: deepcopy(passed)
        ),
    )
    receipt = runner.preservation_outcome(
        controller, constraint, configuration, run_id="owned-refined-run"
    )
    assert receipt["product_result"] == passed
    controller.preservation_result = None
    assert (
        runner.preservation_outcome(controller, constraint, configuration)[
            "product_result"
        ]
        is None
    )
    with pytest.raises(ValueError, match="passed preservation"):
        runner.preservation_outcome(
            controller, constraint, configuration, run_id="owned-refined-run"
        )
    for key, value in (
        ("constraint_id", "other"),
        ("candidate_key", "irrelevant-copy"),
        ("run_id", "earlier-run"),
        ("retained_count", 2),
        ("missing_count", 1),
        ("passed", False),
    ):
        changed = {**passed, key: value}
        controller.preservation_result = SimpleNamespace(
            passed=changed["passed"], public_result=lambda item=changed: deepcopy(item)
        )
        with pytest.raises(ValueError, match="another constraint|passed preservation"):
            runner.preservation_outcome(
                controller, constraint, configuration, run_id="owned-refined-run"
            )
    controller.preservation_constraint = object()
    with pytest.raises(ValueError, match="snapshot changed"):
        runner.preservation_outcome(controller, constraint, configuration)


def test_actual_product_preservation_result_contract_is_recorded(modules):
    from napari_vipp.core.workflow_refinement import PreservationResult

    runner, _entry = modules
    contract = {"constraint_id": "frozen"}
    constraint = SimpleNamespace(public_contract=lambda: deepcopy(contract))
    result = PreservationResult("frozen", "owned", "actual-node", 0, 3, 3)
    controller = SimpleNamespace(
        preservation_constraint=constraint, preservation_result=result
    )
    record = runner.preservation_outcome(
        controller, constraint, {"product_contract": contract}, run_id="owned"
    )
    assert record["product_result"] == result.public_result()
    assert record["product_result"]["candidate_key"] == "protected_structure"
