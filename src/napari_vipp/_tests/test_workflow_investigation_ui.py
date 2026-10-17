"""Offline product-controller acceptance with real file I/O and CPU execution.

Provider discovery/action responses are mocked; one test injects an invalid
final UI publication to exercise the real validator's fail-closed path. CPU
execution is never mocked. This toy intensity ramp is not the blinded experiment,
a sample generator or a scientific target workflow.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID
from napari_vipp.core.workflow_investigation import InvestigationAction


@pytest.fixture
def product(qtbot, monkeypatch, tmp_path):
    from napari_vipp.core import workflow_investigation_model as provider
    from napari_vipp.core import workflow_proposal_codex as codex

    monkeypatch.setattr(
        codex,
        "discover_codex_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(
            supported=True, authenticated=True, detail="Offline provider fixture"
        ),
    )
    data = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    source = tmp_path / "toy-ramp.ome.tif"
    tifffile.imwrite(
        source,
        data,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 0.45,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.45,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": 0.45,
            "PhysicalSizeZUnit": "µm",
        },
    )
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget.pipeline.reset_empty_graph()
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(source), sample_name=""
    )
    widget._build_graph_from_pipeline()
    widget.run_pipeline(force_sync=True)
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], data)
    assert widget.pipeline.output_states["input"].axis_order == "ZYX"
    widget._history.clear()
    widget._sync_current_workflow_tab_state()
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.hide()
    dialog.goal_edit.setPlainText("Identify the above-threshold part of this toy ramp.")
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
    qtbot.waitUntil(lambda: dialog._codex_authenticated and not dialog._connection_busy)
    # Legacy automatic-Finish path.
    dialog._investigation.review_checkbox.setChecked(False)
    dialog._sync_buttons()
    calls = []
    errors = []
    dialog._investigation.signals.failed.connect(
        lambda _epoch, error: errors.append(repr(error))
    )

    def unrequested(*_args, **_kwargs):
        raise AssertionError("The test did not approve a model action.")

    monkeypatch.setattr(provider, "generate_codex_investigation_action", unrequested)
    result = SimpleNamespace(
        widget=widget,
        dialog=dialog,
        data=data,
        calls=calls,
        errors=errors,
        provider=provider,
    )
    yield result
    dialog.close()
    widget._discard_incomplete_startup_on_close = True
    widget.close()


def _toy_proposal(request, threshold):
    params = {
        parameter.name: parameter.default
        for parameter in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
    }
    params.update(foreground="Above", threshold=float(threshold))
    return {
        "kind": "napari-vipp-workflow-proposal",
        "version": 1,
        "context_id": request["proposal_context_id"],
        "summary": "Toy ramp threshold for controller acceptance only.",
        "assumptions": ["This test is not a scientific segmentation validation."],
        "questions": [],
        "nodes": [
            {
                "key": "mask",
                "operation_id": "binary_threshold",
                "params": params,
                "reason": "Exercise ordinary CPU execution on the toy ramp.",
            }
        ],
        "connections": [
            {"source": "input", "source_port": 0, "target": "mask", "target_port": 0}
        ],
    }


def _request_receipts(request):
    """Only complete action receipts actually supplied in this working view."""
    notebook = request["evidence_notebook"]
    kinds = {row["record_id"]: row["kind"] for row in notebook["record_index"]}
    return [
        row["payload"]
        for row in notebook["working_records"]
        if kinds[row["record_id"]] == "action_receipt"
    ]


def _experiment_plan(request):
    ledger = request["experiment_ledger"]
    baseline = ledger[-1]["candidate_id"] if ledger else None
    evidence = [
        {"kind": "numeric", "alias": "input", "sha256": request["source"]["sha256"]}
    ]
    if baseline is not None:
        evidence.extend(ledger[-1]["assessment"]["check_results"][0]["evidence_refs"])
    return {
        "baseline_candidate_id": baseline,
        "evidence_refs": evidence,
        "observed_failure": "The toy ramp has no selected foreground mask yet.",
        "hypothesis": "A declared threshold yields the requested toy foreground.",
        "prior_assumptions": ["This is a toy ramp, not anatomy."],
        "predicted_geometric_change": "A nonempty threshold mask on the original grid.",
        "risks": ["This fixture does not validate scientific segmentation."],
        "intent": "cleanup",
        "evaluation_checks": [
            {
                "check_id": "foreground",
                "kind": "numeric",
                "criterion": "Check foreground count.",
            }
        ],
    }


def _latest_plan(request):
    return request["experiment_ledger"][-1]


def _experiment_assessment(request):
    candidate = next(
        event["result"]
        for event in reversed(_request_receipts(request))
        if event["action"] in {"trial_candidate", "trial_edit"}
        and event["status"] == "ok"
    )
    output = candidate["outputs"][0]
    return {
        "experiment_id": _latest_plan(request)["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "check_results": [
            {
                "check_id": "foreground",
                "outcome": "supports",
                "evidence_refs": [
                    {
                        "kind": "numeric",
                        "alias": output["output_alias"],
                        "sha256": output["sha256"],
                    }
                ],
                "assessment": "The completed toy trial has nonempty foreground.",
            }
        ],
        "unresolved_criteria": ["Toy mechanics do not establish scientific accuracy."],
    }


def _install_actions(
    product, monkeypatch, actions, *, entered=None, release=None, planned_trials=None
):
    # The mocked model now emits real prospective plan and assessment actions.
    # No hidden engine calls or bypass of the product's experiment requirement.
    expanded = []
    planned = 0
    for kind, arguments in actions:
        if kind != "trial_candidate" or (
            planned_trials is not None and planned >= planned_trials
        ):
            expanded.append((kind, arguments))
            continue
        planned += 1
        expanded.append(("plan_experiment", _experiment_plan))

        def linked(request, original=arguments):
            args = original(request) if callable(original) else dict(original)
            return args | {"experiment_id": _latest_plan(request)["experiment_id"]}

        expanded.extend([(kind, linked), ("assess_experiment", _experiment_assessment)])

    def generate(request, _settings, **kwargs):
        product.calls.append(request)
        kwargs["before_call"]()
        if entered is not None and len(product.calls) == 1:
            entered.set()
            assert release.wait(8), "Offline provider fixture was not released."
        kind, arguments = expanded[len(product.calls) - 1]
        arguments = arguments(request) if callable(arguments) else arguments
        action = InvestigationAction(request["session_id"], kind, arguments)
        return product.provider.ModelInvestigationResult(
            action,
            (),
            "codex",
            "offline-provider-fixture",
            "0" * 64,
            "offline-requested-model",
            "offline-provider",
        )

    monkeypatch.setattr(
        product.provider, "generate_codex_investigation_action", generate
    )


def _trial(threshold):
    return (
        "trial_candidate",
        lambda request: {
            "proposal": _toy_proposal(request, threshold),
            "output_roles": [{"role": "toy-foreground", "node": "mask", "port": 0}],
        },
    )


def _finish(candidate_id="candidate1"):
    return (
        "finish",
        {
            "candidate_id": candidate_id,
            "summary": "Explicit toy candidate selection, not scientific success.",
            "limitations": ["Toy intensity ramp only."],
        },
    )


def _start(product, *, input_images=True):
    dialog = product.dialog
    dialog.output_assessment_checkbox.setChecked(True)
    dialog.images_checkbox.setChecked(input_images)
    dialog._investigation.checkbox.setChecked(True)
    assert dialog.prepare_button.isEnabled(), (
        dialog.source_combo.currentData(),
        dialog._provider(),
        dialog._codex_authenticated,
        dialog._connection_busy,
        dialog.goal_edit.toPlainText(),
        dialog._investigation.description.text(),
    )
    dialog.prepare_button.click()


def test_investigation_is_passive_by_default_and_consent_denial_sends_nothing(product):
    dialog = product.dialog
    controller = dialog._investigation
    assert not controller.checkbox.isChecked()
    assert controller.max_attempts.value() == 3
    controller.checkbox.setChecked(True)
    controller.max_attempts.setValue(2)
    assert controller.session is None and product.calls == []
    dialog.consent_checkbox.setChecked(False)
    controller.start()
    assert controller.session is None
    assert "approves sharing" in dialog.status_label.text()
    assert set(product.widget.pipeline.nodes) == {"input"}


def test_controller_runs_detached_trial_and_requires_explicit_final_add_and_run(
    product, monkeypatch, qtbot
):
    _install_actions(product, monkeypatch, [_trial(32), _finish()])
    _start(product)
    controller, dialog, widget = (
        product.dialog._investigation,
        product.dialog,
        product.widget,
    )
    qtbot.waitUntil(lambda: not controller.active, timeout=15_000)
    assert controller.session is not None, (dialog.status_label.text(), product.errors)
    assert controller.session.finished, (dialog.status_label.text(), product.errors)
    selected = controller.session.selected_candidate
    assert selected.candidate_id == "candidate1"
    np.testing.assert_array_equal(selected._outputs[0][1].data, product.data > 32)
    assert set(widget.pipeline.nodes) == {"input"}
    assert dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.apply_button.click()
    added = dialog._added_ids
    assert len(added) == 1
    assert widget.pipeline.outputs[added[0]] is None
    assert dialog.run_button.isEnabled()
    dialog.run_button.click()
    qtbot.waitUntil(
        lambda: widget.pipeline.outputs.get(added[0]) is not None, timeout=15_000
    )
    np.testing.assert_array_equal(widget.pipeline.outputs[added[0]], product.data > 32)
    assert widget.pipeline.output_states[added[0]] == selected._outputs[0][1].state
    assert len(product.calls) == 4
    assert len(controller.model_records) == 4
    assert controller.session.require_experiment_plan
    assert all(
        record["model"] == "offline-provider-fixture"
        and record["requested_model"] == "offline-requested-model"
        and record["model_provider"] == "offline-provider"
        and record["request_sha256"] == "0" * 64
        for record in controller.model_records
    )


def test_async_final_run_consumes_dialog_ids_but_owned_completion_waits(
    product, monkeypatch, qtbot
):
    import importlib

    from qtpy.QtCore import QRunnable

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    evidence = importlib.import_module("workflow_investigation_evidence")
    widget, dialog = product.widget, product.dialog
    widget.background_all_checkbox.setChecked(True)
    _install_actions(product, monkeypatch, [_trial(32), _finish()])
    _start(product)
    qtbot.waitUntil(lambda: not dialog._investigation.active, timeout=15_000)
    selected = dialog._investigation.session.selected_candidate
    prepared = dialog._prepared
    dialog.apply_button.click()
    binding = evidence.freeze_replay_binding(
        selected.recipe,
        selected.graph,
        widget.pipeline,
        tuple(dialog._added_ids),
        prepared._context.context_id,
    )
    entered, release = threading.Event(), threading.Event()

    class SchedulingBarrier(QRunnable):
        def run(self):
            entered.set()
            release.wait(8)

    widget._pipeline_thread_pool.start(SchedulingBarrier())
    qtbot.waitUntil(entered.is_set)
    revision = widget._workflow_assistant_run_revision
    previous = widget._workflow_assistant_assessment_serial
    previous_pipeline = widget._pipeline_run_serial
    previous_report = widget._last_execution_report
    try:
        dialog.run_button.click()
        assert dialog._added_ids == ()  # Product intentionally consumes permission.
        assert all(
            widget.pipeline.node_execution_states.get(node) == "ready"
            for node in dialog._added_ids
        )
        assert widget._active_pipeline_run_id is not None
        ticket = evidence.freeze_owned_dispatch(
            widget, previous, previous_pipeline, revision, previous_report
        )
        assert (
            evidence.owned_replay_completion(binding, widget.pipeline, active=True)
            == "waiting"
        )
        assert all(
            widget.pipeline.outputs.get(node) is None for node in binding.added_node_ids
        )
    finally:
        release.set()
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=15_000)
    assert evidence.owned_replay_completion(binding, widget.pipeline) == "ready"
    evidence.assert_owned_dispatch(widget, ticket, completed=True)
    snapshot, owner = widget._capture_workflow_assessment_snapshot(
        binding.added_node_ids, "input"
    )
    widget._assert_workflow_assessment_current(snapshot, owner)
    np.testing.assert_array_equal(
        widget.pipeline.outputs[binding.added_node_ids[0]], product.data > 32
    )
    assert len(product.calls) == 4


def test_explicit_roles_and_selection_preserve_first_not_latest_candidate(
    product, monkeypatch, qtbot
):
    actions = [
        _trial(32),
        _trial(96),
        (
            "compare_candidates",
            {"left": "candidate1", "right": "candidate2", "roles": ["toy-foreground"]},
        ),
        (
            "retain_candidate",
            {
                "candidate_id": "candidate1",
                "reason": "Explicit offline selection test.",
            },
        ),
        _finish("candidate1"),
    ]
    _install_actions(product, monkeypatch, actions)
    _start(product)
    controller = product.dialog._investigation
    qtbot.waitUntil(lambda: not controller.active, timeout=20_000)
    session = controller.session
    assert session.finished, product.dialog.status_label.text()
    assert len(session.candidates) == 2
    assert session.selected_candidate.candidate_id == "candidate1"
    assert session.selected_candidate.recipe["nodes"][0]["params"]["threshold"] == 32
    assert set(product.widget.pipeline.nodes) == {"input"}


def test_attempt_budget_cannot_be_reset_by_another_trial_action(
    product, monkeypatch, qtbot
):
    _install_actions(
        product, monkeypatch, [_trial(32), _trial(96), _finish()], planned_trials=1
    )
    product.dialog._investigation.max_attempts.setValue(1)
    _start(product)
    controller = product.dialog._investigation
    qtbot.waitUntil(lambda: controller._paused, timeout=15_000)
    session = controller.session
    assert controller.active and session.paused and not session.finished
    assert not session.stop_reason
    assert len(controller.session.candidates) == 1
    assert session.attempted_trials == 1
    # The rejected next trial pauses BEFORE executing/charging another action;
    # its actual model decision is still counted and cannot be replayed free.
    assert session.actions_used == 3 and session.model_calls_used == 4
    assert len(product.calls) == len(controller.model_records) == 4
    assert session.continuation_state["state"] == "paused"
    assert "attempt budget" in session.continuation_state["reason"].lower()
    before = (session.actions_used, session.model_calls_used, session.attempted_trials)
    controller.resume_button.click()
    assert controller.session is session and session.paused
    assert (
        session.actions_used,
        session.model_calls_used,
        session.attempted_trials,
    ) == before
    assert len(product.calls) == 4
    np.testing.assert_array_equal(
        session.candidates[0]._outputs[0][1].data, product.data > 32
    )
    assert not product.dialog.apply_button.isEnabled()
    assert not product.dialog.run_button.isEnabled()
    assert set(product.widget.pipeline.nodes) == {"input"}


def test_question_pauses_same_session_without_implicit_trial_or_add(
    product, monkeypatch, qtbot
):
    _install_actions(
        product,
        monkeypatch,
        [("question", {"question": "Which toy intensity convention is intended?"})],
    )
    _start(product)
    controller, dialog = product.dialog._investigation, product.dialog
    qtbot.waitUntil(lambda: controller._paused, timeout=10_000)
    assert controller.active and controller.session.paused
    assert not controller.session.finished
    assert not controller.session.stop_reason
    assert controller.session.candidates == ()
    assert dialog.stage_tabs.currentIndex() == 1
    assert "Which toy intensity" in controller.pause_note.text()
    assert controller.pause_answer.isVisibleTo(controller.pause_card)
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert len(product.calls) == 1


def test_input_preview_request_without_consent_is_denied_not_silently_shared(
    product, monkeypatch, qtbot
):
    _install_actions(
        product,
        monkeypatch,
        [
            _trial(32),
            (
                "inspect_source",
                {
                    "views": [{"axis": "Z", "indices": [0]}],
                },
            ),
            _finish(),
        ],
    )
    _start(product, input_images=False)
    controller = product.dialog._investigation
    qtbot.waitUntil(lambda: not controller.active, timeout=15_000)
    assert controller.session.finished, product.dialog.status_label.text()
    rejected = _request_receipts(product.calls[4])[-1]
    assert rejected["action"] == "inspect_source" and rejected["status"] == "error"
    assert rejected["image_previews"] == []
    assert all(request["image_previews"] == [] for request in product.calls)


def test_output_consent_off_blocks_start_without_a_trial_or_model_call(product):
    dialog, controller = product.dialog, product.dialog._investigation
    controller.checkbox.setChecked(True)
    assert not dialog.output_assessment_checkbox.isChecked()
    assert not dialog.prepare_button.isEnabled()
    dialog.consent_checkbox.setChecked(True)
    controller.start()
    assert controller.session is None and product.calls == []
    assert "Assess workflow outputs" in dialog.status_label.text()
    assert set(product.widget.pipeline.nodes) == {"input"}


def test_product_cannot_skip_the_prospective_experiment_plan(
    product, monkeypatch, qtbot
):
    _install_actions(
        product,
        monkeypatch,
        [
            _trial(32),
            ("question", {"question": "Stop this deliberate invalid fixture."}),
        ],
        planned_trials=0,
    )
    _start(product)
    controller = product.dialog._investigation
    qtbot.waitUntil(lambda: controller._paused, timeout=10_000)
    assert controller.active and controller.session.paused
    assert controller.session.require_experiment_plan
    assert controller.session.candidates == ()
    assert len(product.calls) == 2
    rejected = _request_receipts(product.calls[1])[-1]
    assert rejected["action"] == "trial_candidate" and rejected["status"] == "error"
    assert "experiment" in rejected["error"].lower()
    assert not product.dialog.apply_button.isEnabled()
    assert set(product.widget.pipeline.nodes) == {"input"}


def test_experiment_history_shows_judgment_not_accuracy_and_escapes_text(product):
    from napari_vipp.ui.workflow_assistant_investigation import _action_detail

    detail = _action_detail(
        {
            "action": "assess_experiment",
            "status": "ok",
            "result": {
                "assessment": {
                    "check_results": [
                        {"outcome": "supports"},
                        {"outcome": "contradicts"},
                        {"outcome": "unresolved"},
                    ]
                }
            },
        }
    )
    assert "1 check supported" in detail
    assert "1 check not supported" in detail
    assert "1 check still uncertain" in detail
    assert "not proof of accuracy" in detail
    controller = product.dialog._investigation
    controller._event("Testing <b>untrusted</b> text", "<img src='remote'>")
    assert "&lt;b&gt;" in controller.history.text()
    assert "&lt;img" in controller.history.text()


def test_rejected_final_checked_add_is_not_reported_as_investigation_success(
    product, monkeypatch, qtbot
):
    _install_actions(product, monkeypatch, [_trial(32), _finish()])
    original = product.dialog._accept_proposal

    def invalid_final_publication(proposal):
        proposal["nodes"][0]["params"]["threshold"] = "not-a-number"
        return original(proposal)

    monkeypatch.setattr(product.dialog, "_accept_proposal", invalid_final_publication)
    _start(product)
    controller, dialog = product.dialog._investigation, product.dialog
    qtbot.waitUntil(lambda: not controller.active, timeout=15_000)
    assert controller.session.finished
    assert dialog._prepared is None or not dialog._prepared.can_apply
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert "Investigation complete" not in dialog.status_label.text()
    assert "final workflow check did not pass" in controller.note.text()
    assert set(product.widget.pipeline.nodes) == {"input"}


def test_provider_failure_retains_copyable_safe_diagnostic_without_private_text(
    product, monkeypatch, qtbot
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelProposalError,
        SafeModelDiagnostic,
    )

    def failure(request, _settings, **kwargs):
        product.calls.append(request)
        kwargs["before_call"]()
        raise ModelProposalError(
            "untrusted failure C:/private/subject-data.npy",
            diagnostic=SafeModelDiagnostic(
                "codex", "request", "quota", http_status=429
            ),
        )

    monkeypatch.setattr(
        product.provider, "generate_codex_investigation_action", failure
    )
    _start(product)
    dialog = product.dialog
    qtbot.waitUntil(lambda: not dialog._investigation.active, timeout=10_000)
    assert len(product.calls) == 1
    assert dialog.copy_error_button.isEnabled()
    assert dialog._error_diagnostic.category == "quota"
    assert '"http_status": 429' in dialog.trace_edit.toPlainText()
    assert "C:/private" not in dialog.trace_edit.toPlainText()
    assert "C:/private" not in dialog.status_label.text()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()


def test_investigation_accumulates_two_explicit_question_answer_rounds(
    product, monkeypatch, qtbot
):
    question1, answer1 = "Which toy convention?", "  Preserve the first convention.  "
    question2, answer2 = "Which toy boundary?", "  Preserve the second boundary.  "
    _install_actions(
        product,
        monkeypatch,
        [
            ("question", {"question": question1}),
            ("question", {"question": question2}),
            _trial(32),
            _finish(),
        ],
    )
    _start(product)
    dialog, controller = product.dialog, product.dialog._investigation
    qtbot.waitUntil(lambda: controller._paused, timeout=10_000)
    session = controller.session
    assert controller.active and session.paused and not session.finished
    assert controller._pending_question == question1
    controller.pause_answer.setText(answer1)
    assert controller.resume_button.isEnabled()
    controller.resume_button.click()
    qtbot.waitUntil(lambda: len(product.calls) == 2 and controller._paused)
    assert controller.session is session and session.paused
    assert controller._pending_question == question2
    assert product.calls[1]["clarifications"] == [
        {"question": question1, "answer": answer1}
    ]
    controller.pause_answer.setText(answer2)
    controller.resume_button.click()
    qtbot.waitUntil(
        lambda: len(product.calls) == 6 and not controller.active, timeout=15_000
    )
    assert controller.session.finished, (dialog.status_label.text(), product.errors)
    assert controller.session is session
    assert session.actions_used == session.model_calls_used == 6
    expected = [
        {"question": question1, "answer": answer1},
        {"question": question2, "answer": answer2},
    ]
    assert product.calls[2]["clarifications"] == expected
    assert product.calls[3]["clarifications"] == expected
    assert product.calls[4]["clarifications"] == expected
    assert product.calls[5]["clarifications"] == expected
    assert session.dialogue.clarifications == (
        (question1, answer1),
        (question2, answer2),
    )
    assert dialog._clarifications == [(question1, answer1), (question2, answer2)]
    assert dialog.apply_button.isEnabled()


@pytest.mark.parametrize("change", ["goal", "source"])
def test_changed_goal_or_source_revokes_a_pending_investigation_question(
    product, monkeypatch, qtbot, change
):
    question = "Which earlier toy convention?"
    _install_actions(product, monkeypatch, [("question", {"question": question})])
    _start(product)
    dialog, controller = product.dialog, product.dialog._investigation
    qtbot.waitUntil(lambda: controller._paused, timeout=10_000)
    session = controller.session
    assert controller._pending_question == question
    if change == "goal":
        dialog.goal_edit.setPlainText(
            "A different toy objective without that question."
        )
    else:
        dialog.source_combo.setCurrentIndex(-1)
    assert controller._pending_question == ""
    assert controller.session is session and not controller.active
    assert session.stop_reason and not session.finished
    assert dialog._clarifications == []
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert len(product.calls) == 1


def test_ordinary_goal_spinner_is_not_reset_by_passive_investigation_sync(
    product, monkeypatch, qtbot
):
    from napari_vipp.core import workflow_proposal_codex as codex
    from napari_vipp.core.workflow_proposal_model import ModelCancelled

    entered, release = threading.Event(), threading.Event()

    def pending(*_args, **_kwargs):
        entered.set()
        assert release.wait(8), "Offline ordinary-planning fixture was not released."
        raise ModelCancelled("Offline ordinary-planning cancellation.")

    monkeypatch.setattr(codex, "generate_codex_proposal", pending)
    dialog = product.dialog
    dialog.show()
    assert not dialog._investigation.checkbox.isChecked()
    dialog.prepare_button.click()
    qtbot.waitUntil(entered.is_set)
    try:
        assert dialog._job_kind == "proposal" and dialog._busy
        dialog._investigation.sync()
        assert dialog.prepare_button._sending
        assert dialog.prepare_button.property("sending")
        assert dialog.prepare_button.text() == "Sending goal…"
        assert dialog.prepare_button._spinner_timer.isActive()
        assert dialog.cancel_button.isEnabled()
    finally:
        dialog.cancel_button.click()
        release.set()
    qtbot.waitUntil(lambda: not dialog._busy)


@pytest.mark.parametrize("cause", ["cancel", "graph", "sharing"])
def test_late_model_action_cannot_publish_after_authority_changes(
    product, monkeypatch, qtbot, cause
):
    entered, release = threading.Event(), threading.Event()
    _install_actions(
        product, monkeypatch, [_trial(32)], entered=entered, release=release
    )
    dialog = product.dialog
    dialog.images_checkbox.setChecked(True)
    _start(product)
    qtbot.waitUntil(entered.is_set)
    try:
        if cause == "cancel":
            dialog.cancel_button.click()
        elif cause == "graph":
            product.widget.pipeline.add_node("invert")
            dialog._investigation._tick()
        else:
            dialog.images_checkbox.setChecked(False)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not dialog._investigation.active)
    qtbot.wait(100)
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog._investigation.session.candidates == ()
    assert len(product.calls) == 1


def test_observer_driver_retains_real_trial_recipe_full_state_and_selected_arrays(
    product, monkeypatch, qtbot, tmp_path
):
    import importlib

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    driver = importlib.import_module("run_workflow_investigation_acceptance")
    evidence = importlib.import_module("workflow_investigation_evidence")
    _install_actions(product, monkeypatch, [_trial(32), _finish()])
    journal = evidence.EvidenceJournal(tmp_path / "evidence")
    observers = driver.ProductObservers(journal)
    observers.install()
    try:
        _start(product)
        controller = product.dialog._investigation
        qtbot.waitUntil(lambda: not controller.active, timeout=15_000)
        assert controller.session.finished, product.dialog.status_label.text()
        assert observers.counts == {
            "logical_requests": 4,
            "model_calls": 0,
            "actions": 4,
            "trial_executions": 1,
        }
        root = journal.directory
        assert (root / "trials/trial-001/full-state.json").is_file()
        assert (root / "trials/trial-001/workflow.json").is_file()
        assert (
            root / "actions/action-002/candidates/candidate1/proposal.json"
        ).is_file()
        selected = controller.session.selected_candidate
        saved = root / (
            f"trials/trial-001/outputs/{selected._outputs[0][1].node_id}-port-0.npy"
        )
        np.testing.assert_array_equal(np.load(saved), product.data > 32)
    finally:
        observers.close()


def test_real_file_source_widget_startup_has_no_sample_function_calls(
    monkeypatch, qtbot, tmp_path
):
    import importlib

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    support = importlib.import_module("workflow_investigation_evidence")
    package = importlib.import_module("napari_vipp")
    guard = support.SampleGeneratorCallGuard(
        Path(package.__file__).with_name("_sample_data.py")
    )
    source = tmp_path / "ordinary-file-input.ome.tif"
    data = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)
    tifffile.imwrite(
        source, data, ome=True, photometric="minisblack", metadata={"axes": "ZYX"}
    )
    widget = None
    try:
        guard.install()
        widget = VippWidget(
            ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
        )
        qtbot.addWidget(widget)
        widget.pipeline.reset_empty_graph()
        widget.pipeline.nodes["input"].params.update(
            source_mode="file path", file_path=str(source), sample_name=""
        )
        widget._build_graph_from_pipeline()
        widget.run_pipeline(force_sync=True)
        np.testing.assert_array_equal(widget.pipeline.outputs["input"], data)
        assert guard.receipt()["function_calls_observed_and_blocked"] == 0
        assert set(widget.pipeline.nodes) == {"input"}
    finally:
        guard.close()
        if widget is not None:
            widget._discard_incomplete_startup_on_close = True
            widget.close()
