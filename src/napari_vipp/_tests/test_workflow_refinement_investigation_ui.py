"""Warm refinement uses real owned Add/Run and CPU trials; model is offline."""

import json
import threading

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation_ui import (
    _experiment_assessment,
    _finish,
    _install_actions,
    _toy_proposal,
)
from napari_vipp._tests.test_workflow_investigation_ui import (
    product as product,
)


def _accepted(product, qtbot):
    d = product.dialog
    d.output_assessment_checkbox.setChecked(True)
    d.images_checkbox.setChecked(True)
    d.consent_checkbox.setChecked(True)
    request = d._ensure_context()
    d._accept_proposal(
        _toy_proposal({"proposal_context_id": request["context_id"]}, 32)
    )
    assert d.apply_button.isEnabled()
    d.apply_button.click()
    added = tuple(d._added_ids)
    d.run_button.click()
    qtbot.waitUntil(lambda: bool(d._assessment.run_id), timeout=15_000)
    d._refinement.slider.setValue(2)
    assert d._refinement.start_button.isEnabled()
    return added[0]


def test_keep_accepted_has_zero_trial_cost_and_does_not_duplicate(
    product, monkeypatch, qtbot
):
    node = _accepted(product, qtbot)
    before = product.widget.pipeline.outputs[node]
    nodes = tuple(product.widget.pipeline.nodes)
    _install_actions(product, monkeypatch, [_finish("candidate0")])
    product.dialog._refinement.start_button.click()
    controller = product.dialog._investigation
    qtbot.waitUntil(lambda: not controller.active, timeout=15_000)
    assert controller.session is not None, (
        product.errors,
        product.dialog.status_label.text(),
    )
    assert controller.session.finished, (
        product.errors,
        product.dialog.status_label.text(),
    )
    assert controller.session.attempted_trials == 0
    assert controller.session.selected_candidate.candidate_id == "candidate0"
    assert tuple(product.widget.pipeline.nodes) == nodes
    assert product.widget.pipeline.outputs[node] is before
    assert "kept the accepted" in controller.note.text()
    assert len(product.calls) == 1
    assert product.calls[0]["accepted_candidate"]


@pytest.mark.parametrize(
    "protected, edit_selected",
    [(False, False), (True, False), (True, True), (True, "stale")],
)
def test_actual_parameter_edit_is_detached_then_explicitly_promoted(
    product, monkeypatch, qtbot, protected, edit_selected
):
    node = _accepted(product, qtbot)
    d, pipeline = product.dialog, product.widget.pipeline
    original = pipeline.outputs[node]
    if protected:
        d._refinement.configure_preservation((node, 0))

    def plan(request):
        output = request["accepted_candidate"]["outputs"][0]
        return dict(
            baseline_candidate_id="candidate0",
            evidence_refs=[
                dict(
                    kind="numeric",
                    alias=output["output_alias"],
                    sha256=output["sha256"],
                )
            ],
            observed_failure="The toy mask omits values from 17 through 32.",
            hypothesis=(
                "Lowering the toy threshold should retain existing foreground "
                "and add values."
            ),
            prior_assumptions=["Toy ramp only."],
            predicted_geometric_change=(
                "More toy foreground without losing accepted pixels."
            ),
            risks=["Adding foreground is not proof of accuracy."],
            intent="cleanup",
            evaluation_checks=[
                dict(
                    check_id="foreground",
                    kind="numeric",
                    criterion="Check foreground count.",
                )
            ],
        )

    def edit(request):
        key = request["accepted_candidate"]["role_bindings"][0]["node_key"]
        return dict(
            candidate_id="candidate0",
            edits=[dict(kind="set_parameters", node=key, params={"threshold": 16.0})],
            experiment_id=request["experiment_ledger"][-1]["experiment_id"],
        )

    def assessment(request):
        # Both typed trial kinds expose the same exact supplied receipt.
        return _experiment_assessment(request)

    _install_actions(
        product,
        monkeypatch,
        [
            ("plan_experiment", plan),
            ("trial_edit", edit),
            ("assess_experiment", assessment),
            _finish("candidate1"),
        ],
    )
    d._refinement.start_button.click()
    qtbot.waitUntil(lambda: not d._investigation.active, timeout=15_000)
    session = d._investigation.session
    assert session.finished, (
        product.errors,
        d.trace_edit.toPlainText(),
        d.status_label.text(),
    )
    assert session.attempted_trials == 1
    assert pipeline.outputs[node] is original
    np.testing.assert_array_equal(original, product.data > 32)
    selected = session.selected_candidate
    np.testing.assert_array_equal(selected._outputs[0][1].data, product.data > 16)
    assert d.apply_button.isEnabled()
    if edit_selected == "stale":
        replacement = original.copy()
        pipeline.node_outputs[node][0] = replacement
        pipeline.outputs[node] = replacement
        d.apply_button.click()
        assert len(pipeline.nodes) == 2
        assert "Nothing was added" in d.status_label.text()
        assert not d.apply_button.isEnabled()
        return
    if edit_selected:
        changed = json.loads(d.json_edit.toPlainText())
        changed["nodes"][0]["params"]["threshold"] = 12.0
        d._accept_proposal(changed)
        d.apply_button.click()
        assert len(pipeline.nodes) == 2
        assert "differs from the tested" in d.status_label.text()
        np.testing.assert_array_equal(pipeline.outputs[node], product.data > 32)
        return
    d.apply_button.click()
    added = tuple(d._added_ids)
    d.run_button.click()
    qtbot.waitUntil(
        lambda: d._assessment.run_id and d._assessment.candidate_ids == added,
        timeout=15_000,
    )
    np.testing.assert_array_equal(pipeline.outputs[added[0]], product.data > 16)
    np.testing.assert_array_equal(pipeline.outputs[node], product.data > 32)
    if protected:
        assert d._refinement.preservation_result.passed


@pytest.mark.parametrize("change", ["stop", "consent", "cache", "limit"])
def test_warm_start_revokes_late_actions_without_touching_original(
    product, monkeypatch, qtbot, change
):
    node = _accepted(product, qtbot)
    original = product.widget.pipeline.outputs[node].copy()
    entered, release = threading.Event(), threading.Event()
    _install_actions(
        product, monkeypatch, [_finish("candidate0")], entered=entered, release=release
    )
    d = product.dialog
    d._refinement.start_button.click()
    qtbot.waitUntil(entered.is_set, timeout=15_000)
    try:
        if change == "stop":
            d._refinement.stop_button.click()
        elif change == "consent":
            d.output_assessment_checkbox.setChecked(False)
        elif change == "cache":
            product.widget.pipeline.outputs[node] = original.copy()
            product.widget.pipeline.node_outputs[node][0] = original.copy()
            d._investigation._tick()
        else:
            d._refinement.slider.setValue(1)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not d._investigation.active, timeout=15_000)
    qtbot.wait(30)
    assert not d._investigation.session.finished
    assert not d._busy
    assert len(product.widget.pipeline.nodes) == 2
    np.testing.assert_array_equal(product.widget.pipeline.outputs[node], original)
