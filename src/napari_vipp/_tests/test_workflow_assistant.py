from __future__ import annotations

import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import (
    ProposalError,
    reference_proposal,
    validate_proposal,
)


@pytest.fixture
def editor(qtbot, monkeypatch):
    from napari_vipp.core import workflow_proposal_codex

    monkeypatch.setattr(
        workflow_proposal_codex,
        "discover_codex_runtime",
        lambda: SimpleNamespace(
            supported=True, authenticated=False, detail="Sign in to Codex."
        ),
    )
    widget = VippWidget(ViewerModel(), defer_initial_run=True)
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    data = np.arange(24 * 32, dtype=np.float32).reshape(24, 32).copy()
    data.flags.writeable = False
    state = image_state_from_array(
        data, axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    )
    widget.pipeline.outputs["input"] = data
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.completed_node_ids.add("input")
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda **kwargs: calls.append(kwargs))
    widget._history.clear()
    return widget, calls


def prepared(editor):
    widget, _ = editor
    context, owner = widget._capture_workflow_proposal_context("input")
    return validate_proposal(reference_proposal(context), context), owner


def test_menu_opens_one_modeless_dialog_and_reference_does_not_run(editor):
    widget, calls = editor
    widget._populate_workflow_actions_menu()
    action = next(
        a
        for a in widget.workflow_actions_menu.actions()
        if a.text() == "Workflow assistant…"
    )
    action.trigger()
    dialog = widget._workflow_assistant_dialog
    widget._open_workflow_assistant()
    assert widget._workflow_assistant_dialog is dialog
    dialog.provider_combo.setCurrentIndex(2)
    dialog.prepare()
    assert dialog.apply_button.isEnabled()
    assert "Example pipeline (no AI)" in dialog.preview_edit.toPlainText()
    assert calls == []
    dialog.close()


def test_apply_preserves_existing_results_positions_and_is_one_non_running_edit(editor):
    widget, calls = editor
    proposal, owner = prepared(editor)
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    data = widget.pipeline.outputs["input"]
    source_node = widget.pipeline.nodes["input"]
    positions = widget.graph_view.node_positions()
    widget.pipeline._counters["gaussian_blur"] = 60

    ids = widget._apply_workflow_proposal(proposal, owner)

    assert len(ids) == 5
    assert len(widget._history.undo_stack) == 1
    assert widget.pipeline.outputs["input"] is data
    assert widget.pipeline.nodes["input"] is source_node
    assert widget.pipeline._counters["gaussian_blur"] >= 60
    assert all(
        widget.graph_view.node_positions()[key] == point
        for key, point in positions.items()
    )
    assert calls == []
    assert all(widget.pipeline.outputs[node_id] is None for node_id in ids)
    assert not widget._debounce_timer.isActive()
    assert any(note.attached_node == ids[-1] for note in widget._graph_notes.values())

    widget.undo()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert calls == []
    widget.redo()
    assert set(ids) <= set(widget.pipeline.nodes)
    assert calls == []


def test_stale_graph_and_changed_tab_are_rejected_without_mutation(editor, monkeypatch):
    widget, calls = editor
    proposal, owner = prepared(editor)
    widget.pipeline.nodes["gaussian"].params["sigma"] = 2.5
    changed = GraphSnapshot.from_pipeline(widget.pipeline)
    with pytest.raises(ProposalError, match="changed"):
        widget._apply_workflow_proposal(proposal, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == changed
    assert calls == []
    proposal, owner = prepared(editor)
    monkeypatch.setattr(widget, "_current_workflow_session_id", lambda: "other-tab")
    with pytest.raises(ProposalError, match="tab changed"):
        widget._apply_workflow_proposal(proposal, owner)


def test_active_or_debounced_calculation_blocks_capture(editor):
    widget, _ = editor
    widget._debounce_timer.start(30_000)
    with pytest.raises(ProposalError, match="pending parameter"):
        widget._capture_workflow_proposal_context("input")
    widget._debounce_timer.stop()
    widget._active_pipeline_run_id = 123
    try:
        with pytest.raises(ProposalError, match="current calculation"):
            widget._capture_workflow_proposal_context("input")
    finally:
        widget._active_pipeline_run_id = None


def test_failed_ui_transaction_rolls_back_graph_cache_history_without_execution(
    editor, monkeypatch
):
    widget, calls = editor
    proposal, owner = prepared(editor)
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    data = widget.pipeline.outputs["input"]
    original_add = widget.graph_view.add_node
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected graphics failure")
        return original_add(*args, **kwargs)

    monkeypatch.setattr(widget.graph_view, "add_node", fail_once)
    with pytest.raises(ProposalError, match="rolled back"):
        widget._apply_workflow_proposal(proposal, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert widget.pipeline.outputs["input"] is data
    assert not widget._history.can_undo
    assert calls == []


def test_questions_and_raw_json_edits_are_checked_automatically(editor, qtbot):
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.initiative_slider.setValue(0)  # Conservative clarification wording.
    dialog.provider_combo.setCurrentIndex(2)
    dialog.prepare()
    raw = json.loads(dialog.json_edit.toPlainText())
    raw["questions"] = ["Do touching objects need separating?"]
    dialog.json_edit.setPlainText(json.dumps(raw))
    assert not dialog.apply_button.isEnabled()
    qtbot.waitUntil(lambda: dialog.status_icon.property("tone") == "warning")
    assert not dialog.apply_button.isEnabled()
    assert "Answer the questions" in dialog.status_label.text()
    assert len(dialog._question_fields) == 1
    assert dialog.questions_layout.rowCount() == 1
    discussion = dialog.discussion_scroll.widget().layout()
    assert discussion.indexOf(dialog.questions_group) < discussion.indexOf(
        dialog.composer_frame
    )
    assert not hasattr(dialog, "validate_button")
    dialog.json_edit.setPlainText("{invalid")
    qtbot.waitUntil(lambda: dialog.status_icon.property("tone") == "error")
    assert not dialog.apply_button.isEnabled()
    dialog.close()


def test_late_application_failure_restores_undo_and_redo(editor, monkeypatch):
    widget, calls = editor
    proposal, owner = prepared(editor)
    snapshot = widget._current_history_snapshot()
    widget._history.undo_stack.append(snapshot)
    widget._history.redo_stack.append(snapshot)
    original_push = widget._push_undo_if_changed

    def fail_after_push(before):
        original_push(before)
        raise RuntimeError("injected failure after history push")

    monkeypatch.setattr(widget, "_push_undo_if_changed", fail_after_push)
    with pytest.raises(ProposalError, match="rolled back"):
        widget._apply_workflow_proposal(proposal, owner)
    assert widget._current_history_snapshot().workflow == snapshot.workflow
    assert widget._history.undo_stack == [snapshot]
    assert widget._history.redo_stack == [snapshot]
    assert widget._workflow_assistant_history_pairs == []
    assert calls == []


def test_local_worker_result_shows_attempts_without_private_baseline(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core import workflow_proposal_model

    widget, calls = editor
    widget.pipeline.nodes["input"].params["file_path"] = "C:/private/image.tif"

    def generate(context, objective, settings, *, cancelled, progress):
        assert not cancelled()
        assert settings.max_attempts == 2
        progress("Local model attempt 1")
        raw = reference_proposal(context)
        return workflow_proposal_model.ModelProposalResult(
            raw,
            validate_proposal(raw, context),
            (
                workflow_proposal_model.ModelAttempt(
                    1,
                    0.1,
                    True,
                    None,
                    json.dumps(raw),
                    100,
                    200,
                ),
            ),
            "ollama",
            "qwen-local",
            "a" * 64,
        )

    monkeypatch.setattr(workflow_proposal_model, "generate_proposal", generate)
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.provider_combo.setCurrentIndex(1)
    dialog.model_combo.setCurrentText("qwen-local")
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert "Attempt 1" in dialog.trace_edit.toPlainText()
    assert "C:/private" not in dialog.trace_edit.toPlainText()
    assert "C:/private" not in dialog.request_edit.toPlainText()
    assert calls == []
    dialog.close()


def test_local_inspect_and_plan_requires_consent_before_capture_or_send(
    editor, monkeypatch
):
    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.provider_combo.setCurrentIndex(1)
    dialog.model_combo.setCurrentText("qwen-local")
    dialog.prepare()
    assert dialog._context is None
    assert not dialog.request_edit.toPlainText()
    assert dialog.json_edit.toPlainText() == ""
    assert "approve sending" in dialog.status_label.text()
    # Shared investigation supports local image-capable models; choosing an
    # option is not permission to capture/send before an explicit Start.
    assert dialog.images_checkbox.isEnabled()
    dialog._ensure_context()
    assert "catalogue" in dialog.request_edit.toPlainText()
    dialog.consent_checkbox.setChecked(True)
    dialog.goal_edit.setPlainText("Compare another goal")
    assert not dialog.consent_checkbox.isChecked()
    assert dialog._context is None
    assert calls == []
    dialog.close()


def test_late_cancelled_response_and_stale_worker_response_cannot_apply(editor):
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.provider_combo.setCurrentIndex(2)
    dialog.prepare()
    response = SimpleNamespace(proposal=reference_proposal(dialog._context))
    old = dialog._generation
    dialog._job_kind = "proposal"
    dialog.cancel_request()
    dialog.json_edit.clear()
    dialog._ready(old, response)
    assert not dialog.apply_button.isEnabled()
    dialog._job_kind = "proposal"
    widget.pipeline.nodes["gaussian"].params["sigma"] = 3
    dialog._ready(dialog._generation, response)
    assert not dialog.apply_button.isEnabled()
    assert "changed" in dialog.status_label.text()
    dialog.close()


def test_background_model_discovery_stays_responsive_and_close_ignores_late_results(
    editor,
    qtbot,
    monkeypatch,
):
    from napari_vipp.core import workflow_proposal_model

    widget, _ = editor
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def discover(*_args):
        entered.set()
        release.wait(3)
        done.set()
        return ("qwen-local",)

    monkeypatch.setattr(workflow_proposal_model, "discover_models", discover)
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.provider_combo.setCurrentIndex(1)
    dialog._refresh_models()
    try:
        qtbot.waitUntil(entered.is_set)
        assert dialog.cancel_button.isEnabled()
        dialog.cancel_button.click()
        assert not dialog._busy
        dialog.close()
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)


def test_default_ui_is_steps_not_raw_text_and_example_is_clearly_not_ai(editor):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel

    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    assert dialog.provider_combo.currentData() == "codex"
    assert not dialog.images_checkbox.isChecked()
    assert not dialog.consent_checkbox.isChecked()
    assert dialog.advanced_group.isHidden()
    assert not hasattr(dialog, "validate_button")
    dialog.provider_combo.setCurrentIndex(2)
    assert "no AI" in dialog.provider_combo.currentText()
    dialog.prepare_button.click()
    assert dialog.status_icon.property("tone") == "success"
    assert len(dialog._step_widgets) == 5
    assert dialog.apply_button.isEnabled()
    assert "scientific assessment" in dialog.status_label.text()
    for card in dialog._step_widgets:
        assert card.objectName() == "WorkflowPlanStep"
        assert all(
            label.textFormat() == Qt.PlainText for label in card.findChildren(QLabel)
        )
    dialog.advanced_checkbox.setChecked(True)
    assert not dialog.advanced_group.isHidden()
    assert calls == []
    dialog.close()


@pytest.mark.parametrize("setting", ["goal", "model", "reasoning", "timeout", "images"])
def test_goal_and_codex_settings_revoke_prepared_plan_and_consent(editor, setting):
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog._ensure_context()
    dialog._accept_proposal(reference_proposal(dialog._context))
    assert dialog.apply_button.isEnabled()
    dialog.consent_checkbox.setChecked(True)
    if setting == "goal":
        dialog.goal_edit.setPlainText("A different objective")
    elif setting == "model":
        dialog.codex_model_edit.setText("other-model")
    elif setting == "reasoning":
        dialog.reasoning_combo.setCurrentText("medium")
    elif setting == "timeout":
        dialog.timeout_spin.setValue(60)
    else:
        dialog.images_checkbox.setChecked(True)
    assert dialog._context is None
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled()
    assert not dialog.consent_checkbox.isChecked()
    dialog.close()


def _mock_codex(monkeypatch, generate, login=None, discover=None):
    module = SimpleNamespace(
        CodexSettings=lambda **kwargs: SimpleNamespace(**kwargs),
        generate_codex_proposal=generate,
        start_codex_login=login or (lambda **_kwargs: _LoginProcess()),
        discover_codex_runtime=discover
        or (
            lambda: SimpleNamespace(
                supported=True, authenticated=True, detail="Ready to plan."
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "napari_vipp.core.workflow_proposal_codex", module)


class _LoginProcess:
    """No operating-system process is launched by a UI authentication test."""

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-login", timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_codex_preview_is_local_and_send_reuses_exact_opted_in_context(
    editor, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QDialog

    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalResult,
    )

    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    captured, sent = [], []
    original_capture = dialog._capture

    def capture(source_id, *, include_images=False):
        captured.append(include_images)
        return original_capture(source_id, include_images=include_images)

    def generate(context, objective, settings, *, cancelled, progress):
        sent.append(context)
        assert context.image_previews
        assert settings.max_attempts == 2
        assert settings.reasoning_effort == "high"
        assert not cancelled()
        progress("Repairing with validation feedback: internal JSON details")
        raw = reference_proposal(context)
        return ModelProposalResult(
            raw,
            validate_proposal(raw, context),
            (
                ModelAttempt(1, 0.1, False, "internal JSON problem", "{invalid"),
                ModelAttempt(2, 0.1, True, None, json.dumps(raw)),
            ),
            "codex",
            "configured",
            "a" * 64,
        )

    _mock_codex(monkeypatch, generate)
    monkeypatch.setattr(dialog, "_capture", capture)
    monkeypatch.setattr(QDialog, "exec", lambda _self: 0)
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.connection_refresh_button.click()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.images_checkbox.setChecked(True)
    dialog.preview_button.click()
    previewed = dialog._context
    assert previewed.image_previews
    assert sent == []
    assert captured == [True]
    assert calls == []
    assert not dialog.consent_checkbox.isChecked()
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert sent == [previewed]
    assert captured == [True]
    assert dialog.status_icon.property("tone") == "success"
    assert "Attempt 1: plan did not pass checks" in dialog.progress_label.text()
    assert "Attempt 2: plan checks passed" in dialog.progress_label.text()
    assert "internal JSON" not in dialog.progress_label.text()
    assert dialog.advanced_group.isHidden()
    assert calls == []
    dialog.close()


def test_inline_questions_send_answers_in_revision_and_never_apply_early(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    objectives = []

    def generate(context, objective, settings, *, cancelled, progress):
        objectives.append(context.public_request(objective))
        raw = reference_proposal(context)
        if len(objectives) == 1:
            raw["questions"] = ["Keep touching structures separate?"]
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "configured", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.initiative_slider.setValue(0)  # This contract requires explicit answers.
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(lambda: bool(dialog._question_fields))
    assert dialog.status_icon.property("tone") == "warning"
    assert not dialog.apply_button.isEnabled()
    assert not dialog.revise_button.isEnabled()
    dialog._question_fields[0][1].setText(
        "Yes, keep two independently labelled structures."
    )
    assert not dialog.consent_checkbox.isChecked()
    dialog.revise_button.click()
    assert len(objectives) == 1
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert len(objectives) == 2
    assert objectives[1]["objective"] == objectives[0]["objective"]
    assert objectives[1]["clarifications"] == [
        {
            "question": "Keep touching structures separate?",
            "answer": "Yes, keep two independently labelled structures.",
        }
    ]
    assert dialog.status_icon.property("tone") == "success"
    assert calls == []
    dialog.close()


def test_local_preview_choice_is_available_but_does_not_capture_or_send(editor):
    widget, calls = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.images_checkbox.setChecked(True)
    dialog.provider_combo.setCurrentIndex(1)
    assert dialog.images_checkbox.isChecked()
    assert dialog.images_checkbox.isEnabled()
    assert dialog._context is None
    assert "only when selected" in dialog.sharing_note.text()
    assert not dialog.consent_checkbox.isChecked()
    assert calls == []
    dialog.close()


def test_signin_is_user_triggered_and_does_not_capture_or_infer(
    editor, qtbot, monkeypatch
):
    calls = []

    process = _LoginProcess()

    def login(*, cancelled):
        assert not cancelled()
        calls.append("login")
        return process

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Inference not authorized"),
        login,
        lambda: _connection_runtime(),
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert calls == []
    dialog.signin_button.click()
    qtbot.waitUntil(lambda: dialog._signin_process is process)
    assert calls == ["login"]
    assert dialog._context is None
    assert not dialog.consent_checkbox.isChecked()
    assert "no image or objective" in dialog.connection_label.text()
    assert calculations == []
    dialog.close()
    qtbot.waitUntil(lambda: process.terminated == 1)


@pytest.mark.parametrize("revoke", ["consent", "goal", "answer"])
def test_revoked_request_ignores_late_codex_plan_and_stops_repair(
    editor, qtbot, monkeypatch, revoke
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    cancelled_calls = []

    def generate(context, objective, settings, *, cancelled, progress):
        entered.set()
        release.wait(3)
        cancelled_calls.append(cancelled())
        progress("Codex is recovering the connection for this same request…")
        raw = reference_proposal(context)
        done.set()
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "configured", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.connection_refresh_button.click()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    if revoke == "answer":
        dialog._ensure_context()
        raw = reference_proposal(dialog._context)
        raw["questions"] = ["Solid or hollow?"]
        dialog._accept_proposal(raw)
        dialog._question_fields[0][1].setText("Solid")
    dialog.consent_checkbox.setChecked(True)
    if revoke == "answer":
        dialog.revise_button.click()
    else:
        dialog.prepare_button.click()
    try:
        qtbot.waitUntil(entered.is_set)
        if revoke == "consent":
            dialog.consent_checkbox.setChecked(False)
        elif revoke == "goal":
            dialog.goal_edit.setPlainText("New objective")
        else:
            # UI fields are frozen, but programmatic changes also revoke ownership.
            assert not dialog._question_fields[0][1].isEnabled()
            dialog._question_fields[0][1].setText("Hollow")
        assert not dialog._busy
        status_after_revocation = dialog.status_label.text()
        release.set()
        qtbot.waitUntil(done.is_set)
        qtbot.wait(30)
        assert cancelled_calls == [True]
        assert not dialog.apply_button.isEnabled()
        assert dialog._prepared is None
        assert dialog._clarifications == []
        assert dialog._pending_clarifications is None
        assert dialog.status_label.text() == status_after_revocation
        assert "connection is recovering" not in dialog.progress_label.text()
        assert calculations == []
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)
        dialog.close()


def test_same_request_recovery_progress_is_fixed_and_not_a_plan_repair(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalResult,
    )

    recovery = "Codex is recovering the connection for this same request…"
    entered, release = threading.Event(), threading.Event()
    requests = []

    def generate(context, objective, settings, *, cancelled, progress):
        requests.append((context, settings))
        progress(recovery)
        progress(recovery)  # A continuing service recovery does not fill the timeline.
        entered.set()
        assert release.wait(2)
        assert not cancelled()
        raw = reference_proposal(context)
        return ModelProposalResult(
            raw,
            validate_proposal(raw, context),
            (ModelAttempt(1, 0.1, True, None, json.dumps(raw)),),
            "codex",
            "default",
            "a" * 64,
        )

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    try:
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(
            lambda: "connection is recovering" in dialog.status_label.text()
        )
        generation = dialog._generation
        assert len(requests) == 1
        assert not requests[0][0].image_previews
        assert requests[0][1].max_attempts == 2
        assert "same request" in dialog.status_label.text()
        assert "existing time limit" in dialog.status_label.text()
        assert (
            "Recovery does not start another plan attempt."
            in dialog.status_label.text()
        )
        assert "repairing" not in dialog.progress_label.text()
        assert dialog.progress_label.text().count("connection is recovering") == 1
        assert dialog.cancel_button.isEnabled()
        assert not dialog.apply_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert dialog._generation == generation
    assert len(requests) == 1
    assert "Attempt 1: plan checks passed" in dialog.progress_label.text()
    final_status = dialog.status_label.text()
    dialog._progress(generation, recovery)
    assert dialog.status_label.text() == final_status
    assert dialog.status_icon.property("tone") == "success"
    assert calculations == []
    dialog.close()


@pytest.mark.parametrize("stop", ["cancel", "close", "provider"])
def test_stopped_recovering_request_ignores_late_progress_and_plan(
    editor, qtbot, monkeypatch, stop
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    recovery = "Codex is recovering the connection for this same request…"
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    cancellations = []

    def generate(context, objective, settings, *, cancelled, progress):
        progress(recovery)
        entered.set()
        release.wait(2)
        cancellations.append(cancelled())
        progress(recovery)
        raw = reference_proposal(context)
        done.set()
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "default", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    try:
        qtbot.waitUntil(entered.is_set)
        qtbot.waitUntil(
            lambda: "connection is recovering" in dialog.status_label.text()
        )
        if stop == "cancel":
            dialog.cancel_button.click()
        elif stop == "close":
            dialog.close()
        else:
            dialog.provider_combo.setCurrentIndex(1)
        assert not dialog._busy
        status = dialog.status_label.text()
        release.set()
        qtbot.waitUntil(done.is_set)
        qtbot.wait(30)
        assert cancellations == [True]
        if stop != "close":
            assert dialog.status_label.text() == status
            assert dialog._prepared is None
            assert not dialog.apply_button.isEnabled()
        assert calculations == []
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)
        if stop != "close":
            dialog.close()


def test_preview_cannot_recapture_or_send_after_source_changes(
    editor, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QDialog

    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    captured = []
    original = dialog._capture

    def capture(source_id, *, include_images=False):
        captured.append(include_images)
        return original(source_id, include_images=include_images)

    monkeypatch.setattr(dialog, "_capture", capture)
    monkeypatch.setattr(QDialog, "exec", lambda _self: 0)
    _mock_codex(monkeypatch, lambda *_a, **_k: pytest.fail("Stale image sent"))
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.connection_refresh_button.click()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.images_checkbox.setChecked(True)
    dialog.preview_button.click()
    widget.pipeline.nodes["gaussian"].params["sigma"] = 8
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    assert captured == [True]
    assert "changed" in dialog.status_label.text()
    assert dialog.status_icon.property("tone") == "error"
    assert not dialog.apply_button.isEnabled()
    assert calculations == []
    dialog.close()


def test_two_rounds_of_clarification_keep_previous_user_answers(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    objectives = []

    def generate(context, objective, settings, *, cancelled, progress):
        objectives.append(context.public_request(objective))
        raw = reference_proposal(context)
        if len(objectives) < 3:
            raw["questions"] = [
                "Solid or hollow?"
                if len(objectives) == 1
                else "Separate touching objects?"
            ]
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "configured", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(lambda: bool(dialog._question_fields))
    dialog._question_fields[0][1].setText("Keep the hollow interior.")
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    qtbot.waitUntil(lambda: len(objectives) == 2 and not dialog._busy)
    dialog._question_fields[0][1].setText("Yes, retain independent identities.")
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert objectives[2]["objective"] == objectives[0]["objective"]
    assert objectives[2]["clarifications"] == [
        {"question": "Solid or hollow?", "answer": "Keep the hollow interior."},
        {
            "question": "Separate touching objects?",
            "answer": "Yes, retain independent identities.",
        },
    ]
    dialog.close()


def test_long_revision_shares_original_goal_and_each_answer_without_concatenation(
    editor, qtbot, monkeypatch
):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDialog, QLabel

    from napari_vipp.core.workflow_proposal_model import ModelProposalResult
    from napari_vipp.core.workflow_proposals import ProposalDialogue

    goal = "  Original wavy and circular structures: " + "g" * 3500 + "  "
    questions = ["Keep both structures?", "Observed data only?", "Preserve cavities?"]
    answers = ["  " + character * 1500 + "  " for character in "abc"]
    expected = [
        {"question": question, "answer": answer}
        for question, answer in zip(questions, answers, strict=True)
    ]
    requests, objectives, contexts = [], [], []

    def generate(context, objective, settings, *, cancelled, progress):
        contexts.append(context)
        objectives.append(objective)
        requests.append(context.public_request(objective))
        raw = reference_proposal(context)
        if len(requests) == 1:
            raw["questions"] = questions
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "default", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.initiative_slider.setValue(0)  # Three-question conservative response.
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.goal_edit.setPlainText(goal)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(lambda: len(dialog._question_fields) == 3 and not dialog._busy)
    assert type(objectives[0]) is str
    for (_, field), answer in zip(dialog._question_fields, answers, strict=True):
        field.setText(answer)
    assert (
        len(goal)
        + sum(len(q) + len(a) for q, a in zip(questions, answers, strict=True))
        > 4000
    )
    assert not dialog.consent_checkbox.isChecked()

    def inspect_preview(preview):
        labels = preview.findChildren(QLabel)
        original = next(
            label for label in labels if label.objectName() == "SharedOriginalObjective"
        )
        assert original.text() == "Original objective\n" + goal
        assert original.textFormat() == Qt.PlainText
        answer_labels = [
            label for label in labels if label.objectName() == "SharedClarification"
        ]
        assert [label.text() for label in answer_labels] == [
            f"Question\n{q}\n\nYour answer\n{a}"
            for q, a in zip(questions, answers, strict=True)
        ]
        assert all("ProposalDialogue(" not in label.text() for label in labels)
        assert len(requests) == 1  # Sharing preview does not submit anything.
        return 0

    monkeypatch.setattr(QDialog, "exec", inspect_preview)
    dialog.preview_button.click()
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert len(requests) == 2
    # The UI may wrap the immutable baseline with client-chosen preferences;
    # it does not recapture a different graph/source during this revision.
    assert contexts[1] == contexts[0]
    assert isinstance(objectives[1], ProposalDialogue)
    assert objectives[1].objective == goal
    assert objectives[1].clarifications == tuple(zip(questions, answers, strict=True))
    assert requests[1]["objective"] == goal
    assert requests[1]["clarifications"] == expected
    assert dialog._clarifications == list(zip(questions, answers, strict=True))
    assert dialog.goal_edit.toPlainText() == goal
    assert calculations == []
    dialog.close()


@pytest.mark.parametrize("action", ["revise", "inspect"])
@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_revision_failure_or_cancel_preserves_original_goal_and_current_answers(
    editor, qtbot, monkeypatch, action, outcome
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
        ModelProposalResult,
    )

    goal = "Original goal " + "g" * 3000
    previous = [("Earlier question?", "Earlier accepted answer.")]
    answer = "  " + "current" * 200 + "  "
    requests = []
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def generate(context, objective, settings, *, cancelled, progress):
        requests.append(context.public_request(objective))
        entered.set()
        release.wait(2)
        done.set()
        if outcome == "failed":
            raise ModelProposalError(
                "Request failed",
                (ModelAttempt(1, 0.1, False, "request failed", "", phase="request"),),
            )
        assert cancelled()
        raw = reference_proposal(context)
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "default", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.goal_edit.setPlainText(goal)
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Current question?"]
    dialog._accept_proposal(raw)
    dialog._clarifications = previous.copy()
    field = dialog._question_fields[0][1]
    field.setText(answer)
    dialog.consent_checkbox.setChecked(True)
    (dialog.revise_button if action == "revise" else dialog.prepare_button).click()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not field.isEnabled()
        if outcome == "cancelled":
            dialog.cancel_button.click()
        release.set()
        qtbot.waitUntil(done.is_set)
        qtbot.waitUntil(lambda: not dialog._busy)
        qtbot.wait(30)
        assert len(requests) == 1
        assert requests[0]["objective"] == goal
        assert requests[0]["clarifications"] == [
            {"question": q, "answer": a}
            for q, a in [*previous, ("Current question?", answer)]
        ]
        assert dialog.goal_edit.toPlainText() == goal
        assert dialog._question_fields == [("Current question?", field)]
        assert field.text() == answer
        assert field.isEnabled()
        assert dialog._clarifications == previous
        assert dialog._pending_clarifications is None
        assert not dialog.apply_button.isEnabled()
        assert dialog.revise_button.isEnabled()
        rebuilt = dialog._context.public_request(dialog._answered_objective())
        assert rebuilt["clarifications"] == requests[0]["clarifications"]
        assert calculations == []
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)
        dialog.close()


@pytest.mark.parametrize("changed", ["source", "workflow"])
def test_locally_rejected_valid_response_preserves_uncommitted_revision_answers(
    editor, qtbot, monkeypatch, changed
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    entered, release, stale = threading.Event(), threading.Event(), threading.Event()
    requests = []

    def generate(context, objective, settings, *, cancelled, progress):
        requests.append(context.public_request(objective))
        raw = reference_proposal(context)
        prepared = validate_proposal(raw, context)  # Structurally valid response.
        entered.set()
        assert release.wait(2)
        return ModelProposalResult(raw, prepared, (), "codex", "default", "a" * 64)

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    goal = "Original goal " + "g" * 3000
    dialog.goal_edit.setPlainText(goal)
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Current question?"]
    dialog._accept_proposal(raw)
    previous = [("Earlier question?", "Earlier accepted answer.")]
    dialog._clarifications = previous.copy()
    field = dialog._question_fields[0][1]
    answer = "  " + "current" * 200 + "  "
    field.setText(answer)
    original_assert = dialog._assert_current

    def assert_current(context, owner):
        if stale.is_set():
            raise ProposalError(f"The {changed} changed while the request was running.")
        return original_assert(context, owner)

    monkeypatch.setattr(dialog, "_assert_current", assert_current)
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not field.isEnabled()
        stale.set()
        release.set()
        qtbot.waitUntil(lambda: not dialog._busy)
        assert len(requests) == 1
        assert dialog.status_icon.property("tone") == "error"
        assert f"The {changed} changed" in dialog.status_label.text()
        assert dialog.goal_edit.toPlainText() == goal
        assert dialog._question_fields == [("Current question?", field)]
        assert field.text() == answer
        assert field.isEnabled()
        assert dialog._clarifications == previous
        assert dialog._pending_clarifications is None
        assert dialog._prepared is None
        assert not dialog.apply_button.isEnabled()
        assert calculations == []
        assert not widget._history.can_undo
    finally:
        release.set()
        dialog.close()


def _seed_current_answer_draft(editor, qtbot, monkeypatch, generate):
    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.goal_edit.setPlainText("  Original goal stays intact.  ")
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Current question?"]
    dialog._accept_proposal(raw)
    history = [(f"Earlier question {i}?", "Previous " + "a" * 390) for i in range(10)]
    dialog._clarifications = history.copy()
    field = dialog._question_fields[0][1]
    field.setText("  Current answer stays exact.  ")
    return widget, dialog, calculations, history, field


@pytest.mark.parametrize("choices", [(True,), (True, False), (True, False, True)])
def test_image_sharing_choices_keep_stable_answers_but_capture_a_new_request(
    editor, qtbot, monkeypatch, choices
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    sent, captured = [], []

    def generate(context, objective, settings, *, cancelled, progress):
        sent.append((context, context.public_request(objective)))
        raw = reference_proposal(context)
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "default", "a" * 64
        )

    _, dialog, calculations, history, field = _seed_current_answer_draft(
        editor, qtbot, monkeypatch, generate
    )
    old_context = dialog._context
    old_owner = dialog._owner
    original_capture = dialog._capture

    def capture(source_id, *, include_images=False):
        captured.append(include_images)
        return original_capture(source_id, include_images=include_images)

    monkeypatch.setattr(dialog, "_capture", capture)
    dialog.consent_checkbox.setChecked(True)
    for choice in choices:
        dialog.images_checkbox.setChecked(choice)
        assert dialog._context is None
        assert dialog._owner is None
        assert dialog._image_draft_guard[0] is not old_context
        assert dialog._image_draft_guard[0].image_previews == ()
        assert dialog._image_draft_guard[1] == old_owner
        assert dialog._clarifications == history
        assert dialog._question_fields == [("Current question?", field)]
        assert field.text() == "  Current answer stays exact.  "
        assert not dialog.consent_checkbox.isChecked()
        assert not dialog.apply_button.isEnabled()
        assert not dialog.json_edit.toPlainText()
        assert not dialog.request_edit.toPlainText()
        assert not dialog.trace_edit.toPlainText()
        choice_text = "included" if choice else "not included"
        assert dialog.summary_label.text() == (
            f"Input image previews are now {choice_text}. A new draft is needed."
        )
        assert "Choose Update plan" in dialog.status_label.text()
        assert dialog._recovery_kind == "update"
        assert dialog.recovery_button.text() == "Update plan"
        assert captured == []
        assert sent == []
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert captured == [choices[-1]]
    assert len(sent) == 1
    new_context, request = sent[0]
    assert new_context is not old_context
    assert new_context.context_id != old_context.context_id
    assert new_context.source_fingerprint == old_context.source_fingerprint
    assert new_context.baseline_fingerprint == old_context.baseline_fingerprint
    assert bool(new_context.image_previews) is choices[-1]
    assert request["objective"] == "  Original goal stays intact.  "
    assert request["clarifications"] == [
        {"question": q, "answer": a}
        for q, a in [*history, ("Current question?", "  Current answer stays exact.  ")]
    ]
    assert dialog._image_draft_guard is None
    assert calculations == []
    dialog.close()


def test_image_sharing_change_cancels_pending_revision_and_ignores_old_result(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalResult

    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    cancellations = []

    def generate(context, objective, settings, *, cancelled, progress):
        entered.set()
        release.wait(2)
        cancellations.append(cancelled())
        raw = reference_proposal(context)
        done.set()
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "default", "a" * 64
        )

    _, dialog, calculations, history, field = _seed_current_answer_draft(
        editor, qtbot, monkeypatch, generate
    )
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    try:
        qtbot.waitUntil(entered.is_set)
        generation = dialog._generation
        dialog.images_checkbox.setChecked(True)
        assert dialog._generation > generation
        assert not dialog._busy
        assert not dialog.consent_checkbox.isChecked()
        assert dialog._context is None
        assert field.isEnabled()
        release.set()
        qtbot.waitUntil(done.is_set)
        qtbot.wait(30)
        assert cancellations == [True]
        assert dialog._clarifications == history
        assert dialog._pending_clarifications is None
        assert dialog._question_fields == [("Current question?", field)]
        assert field.text() == "  Current answer stays exact.  "
        assert dialog._prepared is None
        assert not dialog.apply_button.isEnabled()
        assert dialog.summary_label.text() == (
            "Input image previews are now included. A new draft is needed."
        )
        assert "Choose Update plan" in dialog.status_label.text()
        assert dialog._recovery_kind == "update"
        assert dialog.recovery_button.text() == "Update plan"
        assert calculations == []
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)
        dialog.close()


@pytest.mark.parametrize("changed", ["source", "workflow", "selection"])
@pytest.mark.parametrize("when", ["before_toggle", "before_recapture"])
def test_image_sharing_cannot_retain_or_recapture_an_old_source_as_fresh(
    editor, qtbot, monkeypatch, changed, when
):
    sent, captured = [], []
    widget, dialog, calculations, _, _ = _seed_current_answer_draft(
        editor,
        qtbot,
        monkeypatch,
        lambda *_a, **_k: sent.append(True),
    )
    original_capture = dialog._capture

    def capture(source_id, *, include_images=False):
        captured.append((source_id, include_images))
        return original_capture(source_id, include_images=include_images)

    monkeypatch.setattr(dialog, "_capture", capture)
    if when == "before_recapture":
        dialog.images_checkbox.setChecked(True)
        assert dialog._image_draft_guard is not None
    if changed == "source":
        data = np.array(widget.pipeline.outputs["input"], copy=True)
        data[0, 0] += 1
        data.flags.writeable = False
        state = image_state_from_array(
            data, axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
        )
        widget.pipeline.outputs["input"] = data
        widget.pipeline.node_outputs["input"] = [data]
        widget.pipeline.output_states["input"] = state
        widget.pipeline.node_output_states["input"] = [state]
    elif changed == "workflow":
        widget.pipeline.nodes["gaussian"].params["sigma"] = 2.5
    else:
        dialog.source_combo.addItem("Another input", "another-input")
        dialog.source_combo.setCurrentIndex(1)
    if when == "before_toggle":
        dialog.images_checkbox.setChecked(True)
    else:
        dialog.preview_button.click()  # Stale guard stops before a modal or capture.
    assert dialog._image_draft_guard is None
    assert dialog._context is None
    assert dialog._clarifications == []
    assert dialog._question_fields == []
    assert not dialog.consent_checkbox.isChecked()
    assert not dialog.apply_button.isEnabled()
    assert captured == (
        [("another-input", True)]
        if changed == "selection" and when == "before_recapture"
        else []
    )  # An explicit preview may inspect the new choice, never reuse the old source.
    assert sent == []
    assert calculations == []
    dialog.close()


def test_source_change_during_image_recapture_discards_retained_answers(
    editor, qtbot, monkeypatch
):
    sent = []
    widget, dialog, calculations, _, _ = _seed_current_answer_draft(
        editor, qtbot, monkeypatch, lambda *_a, **_k: sent.append(True)
    )
    original_capture = dialog._capture

    def capture(source_id, *, include_images=False):
        result = original_capture(source_id, include_images=include_images)
        widget.pipeline.nodes["gaussian"].params["sigma"] = 2.5
        return result

    monkeypatch.setattr(dialog, "_capture", capture)
    dialog.images_checkbox.setChecked(True)
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    assert "Cannot revise" in dialog.status_label.text()
    assert dialog._image_draft_guard is None
    assert dialog._context is None
    assert dialog._clarifications == []
    assert dialog._question_fields == []
    assert not dialog.apply_button.isEnabled()
    assert not dialog.consent_checkbox.isChecked()
    assert sent == []
    assert calculations == []
    dialog.close()


def test_invalid_expert_json_still_clears_unaccepted_question_review(editor, qtbot):
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.provider_combo.setCurrentIndex(2)
    dialog.prepare()
    raw = json.loads(dialog.json_edit.toPlainText())
    raw["questions"] = ["Current question?"]
    dialog._accept_proposal(raw)
    dialog._question_fields[0][1].setText("Current answer.")
    dialog.json_edit.setPlainText("{invalid")
    qtbot.waitUntil(lambda: dialog.status_icon.property("tone") == "error")
    assert dialog._question_fields == []
    assert not dialog._step_widgets
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled()
    assert calculations == []
    dialog.close()


@pytest.mark.parametrize("budget", ["pairs", "bytes"])
def test_overbudget_clarifications_stop_before_provider_without_losing_answers(
    editor, qtbot, monkeypatch, budget
):
    requests = []

    def generate(*_args, **_kwargs):
        requests.append(True)
        raise AssertionError("An overbudget request must not reach a provider")

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    goal = "Original goal stays intact."
    dialog.goal_edit.setPlainText(goal)
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    question = "Current question?" if budget == "pairs" else "Current " + "q" * 990
    raw["questions"] = [question]
    dialog._accept_proposal(raw)
    history = (
        [(f"Earlier question {i}?", "Earlier answer.") for i in range(32)]
        if budget == "pairs"
        else [(f"Earlier {i} " + "q" * 990, "é" * 950) for i in range(22)]
    )
    dialog._clarifications = history.copy()
    answer = "Current answer." if budget == "pairs" else "é" * 950
    field = dialog._question_fields[0][1]
    field.setText(answer)
    dialog.consent_checkbox.setChecked(True)
    dialog.revise_button.click()
    assert "Cannot revise this plan" in dialog.status_label.text()
    assert dialog.status_icon.property("tone") == "error"
    assert not dialog._busy
    assert requests == []
    assert dialog.goal_edit.toPlainText() == goal
    assert dialog._clarifications == history
    assert dialog._question_fields == [(question, field)]
    assert field.text() == answer
    assert field.isEnabled()
    assert calculations == []
    dialog.close()


def test_revised_answer_wins_without_duplicate_or_trimmed_question_history(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposals import ProposalDialogue

    _mock_codex(monkeypatch, lambda *_a, **_k: pytest.fail("No request is needed"))
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.goal_edit.setPlainText("  Original goal.  ")
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Repeated question?"]
    dialog._accept_proposal(raw)
    dialog._clarifications = [(" Repeated question? ", "Old answer.")]
    dialog._question_fields[0][1].setText("  New answer.  ")
    dialogue = dialog._answered_objective()
    assert isinstance(dialogue, ProposalDialogue)
    assert dialogue.objective == "  Original goal.  "
    assert dialogue.clarifications == (("Repeated question?", "  New answer.  "),)
    assert calculations == []
    dialog.close()


def test_failed_repair_is_red_and_default_details_do_not_demand_json_edits(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
    )

    def generate(*_args, **_kwargs):
        raise ModelProposalError(
            "Unknown operation keys in generated JSON",
            (
                ModelAttempt(1, 0.1, False, "Unknown operation key", "{invalid"),
                ModelAttempt(2, 0.1, False, "Unknown operation key", "{invalid"),
            ),
        )

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(lambda: not dialog._busy)
    assert dialog.status_icon.property("tone") == "error"
    assert "two attempts (one repair)" in dialog.status_label.text()
    assert (
        "Neither the initial plan nor its automatic repair"
        in dialog.status_label.text()
    )
    assert "do not need to edit" in dialog.status_label.text()
    assert "Unknown operation key" not in dialog.status_label.text()
    assert "Unknown operation key" in dialog.trace_edit.toPlainText()
    assert dialog.advanced_group.isHidden()
    assert not dialog.apply_button.isEnabled()
    assert calculations == []
    dialog.close()


def _open_with_codex_failure(editor, qtbot, monkeypatch, error, *, limit=2):
    """Use only a mocked provider; no login, network or model request occurs."""
    requests = []

    def generate(*_args, **_kwargs):
        requests.append(True)
        raise error

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.attempts_spin.setValue(limit)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(lambda: not dialog._busy)
    assert requests == [True]
    assert calculations == []
    assert dialog.status_icon.property("tone") == "error"
    assert not dialog.apply_button.isEnabled()
    return dialog


def test_request_failure_is_not_a_plan_repair_failure(editor, qtbot, monkeypatch):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
    )

    error = ModelProposalError(
        "private-provider-error-secret",
        (
            ModelAttempt(
                1,
                0.1,
                False,
                "private-validation-secret",
                "private-response-secret",
                phase="request",
            ),
        ),
    )
    dialog = _open_with_codex_failure(editor, qtbot, monkeypatch, error)
    status = dialog.status_label.text()
    assert "could not complete the request" in status
    assert "your objective has not been assessed" in status
    assert "repair" not in status
    assert "specific" not in status
    assert (
        "Attempt 1: request failed before plan checks" in dialog.progress_label.text()
    )
    assert "Plan checks: not run" in dialog.trace_edit.toPlainText()
    assert "private-" not in status + dialog.trace_edit.toPlainText()
    assert not dialog.copy_error_button.isEnabled()
    assert dialog.advanced_group.isHidden()
    dialog.close()


@pytest.mark.parametrize(
    ("category", "reason", "expected"),
    [
        ("bad_request", "invalid_schema", "integration fix"),
        ("quota", "unknown", "usage or rate limit"),
        ("unauthorized", "unknown", "Sign in again"),
        ("bad_request", "unsupported_model", "supported model"),
        ("bad_request", "unsupported_image", "image-capable model"),
        ("unauthorized", "invalid_schema", "Sign in again"),
        ("quota", "unsupported_model", "usage or rate limit"),
    ],
)
def test_safe_request_failure_explains_cause_and_action(
    editor, qtbot, monkeypatch, category, reason, expected
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
        SafeModelDiagnostic,
    )

    error = ModelProposalError(
        "Do not show arbitrary provider text",
        (ModelAttempt(1, 0.1, False, "request error", "", phase="request"),),
        diagnostic=SafeModelDiagnostic(
            "codex",
            "turn_start",
            category,
            http_status=401
            if category == "unauthorized"
            else 429
            if category == "quota"
            else 400,
            reason=reason,
        ),
    )
    dialog = _open_with_codex_failure(editor, qtbot, monkeypatch, error)
    assert expected in dialog.status_label.text()
    assert "your objective has not been assessed" in dialog.status_label.text()
    assert "Safe error details are in Advanced" in dialog.status_label.text()
    assert "automatic repair" not in dialog.status_label.text()
    if category != "bad_request":
        assert "integration fix" not in dialog.status_label.text()
        assert "Choose a supported model" not in dialog.status_label.text()
    assert dialog.copy_error_button.isEnabled()
    assert dialog.copy_error_button.text() == "Copy error details"
    dialog.close()


@pytest.mark.parametrize(
    ("limit", "response", "expected"),
    [
        (1, "{invalid", "one attempt, so automatic repair was not attempted"),
        (2, "{invalid", "Only one proposal was checked"),
        (1, "", "one attempt, so automatic repair was not attempted"),
    ],
)
def test_one_invalid_or_blank_returned_proposal_is_still_a_plan_check(
    editor, qtbot, monkeypatch, limit, response, expected
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
    )

    error = ModelProposalError(
        "Invalid returned plan",
        (ModelAttempt(1, 0.1, False, "Invalid JSON", response),),
    )
    dialog = _open_with_codex_failure(editor, qtbot, monkeypatch, error, limit=limit)
    assert "returned plan did not pass" in dialog.status_label.text()
    assert expected in dialog.status_label.text()
    assert "two attempts" not in dialog.status_label.text()
    assert "objective has not been assessed" not in dialog.status_label.text()
    assert "plan did not pass checks" in dialog.progress_label.text()
    assert "Phase: proposal" in dialog.trace_edit.toPlainText()
    assert "Plan checks: not run" not in dialog.trace_edit.toPlainText()
    dialog.close()


def test_request_failure_after_invalid_plan_does_not_claim_failed_repair_checks(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
    )

    error = ModelProposalError(
        "Second request did not finish",
        (
            ModelAttempt(1, 0.1, False, "Invalid plan", "{invalid"),
            ModelAttempt(2, 0.1, False, "request error", "", phase="request"),
        ),
    )
    dialog = _open_with_codex_failure(editor, qtbot, monkeypatch, error)
    assert "No usable plan was received" in dialog.status_label.text()
    assert "Neither the initial plan" not in dialog.status_label.text()
    assert "Attempt 1: plan did not pass checks" in dialog.progress_label.text()
    assert (
        "Attempt 2: request failed before plan checks" in dialog.progress_label.text()
    )
    dialog.close()


def test_copy_error_details_contains_only_the_validated_six_field_capsule(
    editor, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QApplication

    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
        SafeModelDiagnostic,
    )

    diagnostic = SafeModelDiagnostic(
        "codex", "turn_start", "bad_request", -32600, 400, "invalid_schema"
    )
    error = ModelProposalError(
        "token-secret C:/private/image.tif image-bytes goal-secret",
        (
            ModelAttempt(
                1, 0.1, False, "response-secret", "source-secret", phase="request"
            ),
        ),
        diagnostic=diagnostic,
    )
    dialog = _open_with_codex_failure(editor, qtbot, monkeypatch, error)
    dialog.advanced_checkbox.setChecked(True)
    dialog.copy_error_button.click()
    copied = QApplication.clipboard().text()
    assert json.loads(copied) == diagnostic.to_dict()
    assert set(json.loads(copied)) == {
        "provider",
        "stage",
        "category",
        "code",
        "http_status",
        "reason",
    }
    assert "secret" not in copied
    assert "C:/private" not in copied
    assert "image-bytes" not in copied
    dialog.codex_model_edit.setText("another-model")
    assert not dialog.copy_error_button.isEnabled()
    assert dialog._error_diagnostic is None
    assert not dialog.trace_edit.toPlainText()
    assert not dialog.consent_checkbox.isChecked()
    dialog.close()


def test_unvalidated_details_and_request_with_no_attempts_are_not_repair(
    editor, qtbot, monkeypatch
):
    from napari_vipp.ui.workflow_assistant import WorkflowAssistantDialog

    _mock_codex(monkeypatch, lambda *_a, **_k: pytest.fail("Must not request a plan"))
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    error = SimpleNamespace(
        attempts=(), diagnostic=SimpleNamespace(reason="private-secret")
    )
    dialog._failed(dialog._generation, error)
    assert "request" in dialog.status_label.text()
    assert "repair" not in dialog.status_label.text()
    assert "private-secret" not in dialog.trace_edit.toPlainText()
    assert not dialog.copy_error_button.isEnabled()
    assert WorkflowAssistantDialog._safe_diagnostic(error) is None
    assert calculations == []
    dialog.close()


def test_new_request_clears_old_safe_diagnostics_before_response(
    editor, qtbot, monkeypatch
):
    from napari_vipp.core.workflow_proposal_model import (
        ModelAttempt,
        ModelProposalError,
        ModelProposalResult,
        SafeModelDiagnostic,
    )

    release = threading.Event()
    started = threading.Event()
    requests = []

    def generate(context, *_args, **_kwargs):
        requests.append(True)
        if len(requests) == 1:
            raise ModelProposalError(
                "Request failed",
                (ModelAttempt(1, 0.1, False, "request error", "", phase="request"),),
                diagnostic=SafeModelDiagnostic("codex", "turn", "network"),
            )
        started.set()
        assert release.wait(2)
        raw = reference_proposal(context)
        return ModelProposalResult(
            raw, validate_proposal(raw, context), (), "codex", "default", "a" * 64
        )

    _mock_codex(monkeypatch, generate)
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare_button.click()
    qtbot.waitUntil(lambda: not dialog._busy)
    assert dialog.copy_error_button.isEnabled()
    dialog.prepare_button.click()
    qtbot.waitUntil(started.is_set)
    try:
        assert not dialog.copy_error_button.isEnabled()
        assert dialog._error_diagnostic is None
        assert not dialog.trace_edit.toPlainText()
        assert not dialog.apply_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(dialog.apply_button.isEnabled)
    assert not dialog.copy_error_button.isEnabled()
    assert calculations == []
    dialog.close()


def test_sharing_preview_is_human_readable_and_raw_request_hidden_by_default(
    editor, qtbot, monkeypatch
):
    from qtpy.QtWidgets import QCheckBox, QDialog, QLabel, QPlainTextEdit, QWidget

    seen = []

    def inspect_preview(preview):
        preview.show()
        qtbot.wait(10)
        advanced = preview.findChild(QWidget, "SharedRequestAdvanced")
        raw = preview.findChild(QPlainTextEdit, "SharedRequestJSON")
        toggle = preview.findChild(QCheckBox, "SharedRequestAdvancedToggle")
        assert advanced.isHidden()
        assert not raw.isVisible()
        labels = [label.text() for label in preview.findChildren(QLabel)]
        assert any("Input: 24 × 32 · YX · float32" in text for text in labels)
        assert any("Declared calibration:" in text for text in labels)
        assert any("1 approved image previews" in text for text in labels)
        assert not any(
            '"alias":' in text or '"calibration":' in text for text in labels
        )
        toggle.setChecked(True)
        assert raw.isVisible()
        assert '"catalogue"' in raw.toPlainText()
        seen.append(True)
        preview.accept()
        return 0

    monkeypatch.setattr(QDialog, "exec", inspect_preview)
    _mock_codex(monkeypatch, lambda *_a, **_k: pytest.fail("Preview must not infer"))
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    dialog.images_checkbox.setChecked(True)
    dialog.preview_button.click()
    assert seen == [True]
    assert not dialog.consent_checkbox.isChecked()
    assert calculations == []
    dialog.close()


def _connection_runtime(authenticated=False, supported=True):
    return SimpleNamespace(
        authenticated=authenticated,
        supported=supported,
        detail="Sign in to Codex."
        if supported
        else "This Codex runtime is unsupported.",
    )


def test_already_signed_in_initial_check_is_readonly_and_does_not_relogin(
    editor, qtbot, monkeypatch
):
    checks = []

    def discover():
        checks.append("status")
        return _connection_runtime(True)

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Status check must not infer"),
        lambda **_k: pytest.fail("Existing sign-in must not relogin"),
        discover,
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert checks == ["status"]
    assert "Signed in to Codex" in dialog.connection_label.text()
    assert dialog.connection_label.property("tone") == "success"
    assert dialog.signin_button.isEnabled()
    assert dialog.signin_button.text() == "Sign in again…"
    assert dialog.connection_refresh_button.text() == "Refresh connection"
    assert dialog._context is None
    assert dialog._prepared is None
    assert not dialog.consent_checkbox.isChecked()
    assert not dialog.images_checkbox.isChecked()
    assert calculations == []
    dialog.close()


def test_browser_completion_checks_auth_once_and_goal_edits_do_not_lose_feedback(
    editor, qtbot, monkeypatch
):
    checks, logins = [], []
    process = _LoginProcess()

    def discover():
        checks.append(True)
        return _connection_runtime(len(checks) > 1)

    def login(*, cancelled):
        assert not cancelled()
        logins.append(True)
        return process

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Login must not infer"),
        login,
        discover,
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.signin_button.click()
    qtbot.waitUntil(lambda: dialog._signin_process is process)
    # Duplicate requests are rejected while our login is pending.
    dialog._signin_codex()
    for _ in range(3):
        dialog._poll_signin()
    assert checks == [True]
    assert logins == [True]
    dialog.goal_edit.setPlainText("Keep both structures independently labelled.")
    dialog.codex_model_edit.setText("test-model")
    assert dialog._signin_process is process
    assert dialog._connection_busy
    process.returncode = 0
    dialog._poll_signin()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert len(checks) == 2
    assert "Signed in to Codex" in dialog.connection_label.text()
    assert dialog.connection_label.property("tone") == "success"
    assert not dialog._signin_timer.isActive()
    assert process.terminated == 0
    assert dialog._context is None
    assert dialog._prepared is None
    assert not dialog.consent_checkbox.isChecked()
    assert not dialog.apply_button.isEnabled()
    assert calculations == []
    dialog.close()


@pytest.mark.parametrize("exit_code", [1, 0])
def test_failed_or_unconfirmed_browser_login_is_actionable_not_signed_in(
    editor, qtbot, monkeypatch, exit_code
):
    checks = []
    process = _LoginProcess()

    def discover():
        checks.append(True)
        return _connection_runtime()

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Login must not infer"),
        lambda **_k: process,
        discover,
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.signin_button.click()
    qtbot.waitUntil(lambda: dialog._signin_process is process)
    process.returncode = exit_code
    dialog._poll_signin()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert not dialog._codex_authenticated
    assert "Signed in to Codex" not in dialog.connection_label.text()
    assert dialog.signin_button.isEnabled()
    assert len(checks) == (1 if exit_code else 2)
    assert dialog._context is None
    assert calculations == []
    dialog.close()


@pytest.mark.parametrize("stop", ["cancel", "close", "provider", "timeout"])
def test_pending_login_cleanup_only_stops_its_owned_child(
    editor, qtbot, monkeypatch, stop
):
    process = _LoginProcess()
    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Login must not infer"),
        lambda **_k: process,
        lambda: _connection_runtime(),
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.signin_button.click()
    qtbot.waitUntil(lambda: dialog._signin_process is process)
    if stop == "cancel":
        dialog.cancel_button.click()
    elif stop == "close":
        dialog.close()
    elif stop == "provider":
        dialog.provider_combo.setCurrentIndex(1)
    else:
        dialog._signin_timeout_ms = 0
        dialog._poll_signin()
        assert "timed out" in dialog.connection_label.text()
    qtbot.waitUntil(lambda: process.terminated == 1)
    assert dialog._signin_process is None
    assert not dialog._connection_busy
    assert not dialog._signin_timer.isActive()
    assert dialog._context is None
    assert calculations == []
    if stop != "close":
        dialog.close()


@pytest.mark.parametrize("stop", ["cancel", "close", "provider"])
def test_late_spawned_login_process_after_cancel_is_cleaned_up(
    editor, qtbot, monkeypatch, stop
):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    process = _LoginProcess()
    tokens = []

    def login(*, cancelled):
        tokens.append(cancelled)
        entered.set()
        release.wait(3)
        done.set()
        return process  # UI also handles a provider's already-spawned-process race.

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("No inference"),
        login,
        lambda: _connection_runtime(),
    )
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.signin_button.click()
    qtbot.waitUntil(entered.is_set)
    try:
        if stop == "cancel":
            dialog.cancel_button.click()
        elif stop == "close":
            dialog.close()
        else:
            dialog.provider_combo.setCurrentIndex(1)
        assert tokens[0]()
        release.set()
        qtbot.waitUntil(done.is_set)
        qtbot.waitUntil(lambda: process.terminated == 1)
        if stop != "close":
            assert not dialog._codex_authenticated
            assert dialog._context is None
            dialog.close()
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)


def test_stale_initial_refresh_does_not_override_new_provider_or_state(
    editor, qtbot, monkeypatch
):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def discover():
        entered.set()
        release.wait(3)
        done.set()
        return _connection_runtime(True)

    _mock_codex(
        monkeypatch, lambda *_a, **_k: pytest.fail("No inference"), discover=discover
    )
    widget, _ = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(entered.is_set)
    try:
        generation = dialog._connection_generation
        dialog.provider_combo.setCurrentIndex(1)
        release.set()
        qtbot.waitUntil(done.is_set)
        dialog._connection_ready(generation, _connection_runtime(True))
        assert not dialog._codex_authenticated
        assert "Signed in to Codex" not in dialog.connection_label.text()
        assert dialog._context is None
    finally:
        release.set()
        qtbot.waitUntil(done.is_set)
        dialog.close()


def test_explicit_refresh_confirms_completed_existing_login_without_login_or_data(
    editor, qtbot, monkeypatch
):
    checks = []

    def discover():
        checks.append(True)
        return _connection_runtime(len(checks) > 1)

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("No inference"),
        lambda **_k: pytest.fail("Refresh must not login"),
        discover,
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert not dialog._codex_authenticated
    dialog.connection_refresh_button.click()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert len(checks) == 2
    assert dialog._codex_authenticated
    assert "Signed in to Codex" in dialog.connection_label.text()
    assert dialog._context is None
    assert not dialog.consent_checkbox.isChecked()
    assert calculations == []
    dialog.close()


def test_existing_signin_allows_explicit_reauthentication_without_starting_a_plan(
    editor, qtbot, monkeypatch
):
    logins = []
    process = _LoginProcess()

    def login(*, cancelled):
        assert not cancelled()
        logins.append(True)
        return process

    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Reauthentication must not infer"),
        login,
        lambda: _connection_runtime(True),
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert logins == []
    assert dialog.signin_button.text() == "Sign in again…"
    dialog.signin_button.click()
    qtbot.waitUntil(lambda: dialog._signin_process is process)
    assert logins == [True]
    assert not dialog.prepare_button.isEnabled()
    assert not dialog._codex_authenticated
    process.returncode = 0
    dialog._poll_signin()
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert "Signed in to Codex" in dialog.connection_label.text()
    assert dialog._context is None
    assert not dialog.consent_checkbox.isChecked()
    assert calculations == []
    dialog.close()


def test_codex_plan_controls_do_not_submit_while_connection_unconfirmed(
    editor, qtbot, monkeypatch
):
    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("Unconfirmed connection must not infer"),
        discover=lambda: _connection_runtime(),
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    assert not dialog.prepare_button.isEnabled()
    dialog.consent_checkbox.setChecked(True)
    dialog.prepare()
    assert dialog._context is None
    assert "No planning request" in dialog.status_label.text()
    dialog._ensure_context()  # Local preview remains available before login.
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Solid or hollow?"]
    dialog._accept_proposal(raw)
    dialog._question_fields[0][1].setText("Hollow")
    dialog.consent_checkbox.setChecked(True)
    assert not dialog.revise_button.isEnabled()
    dialog.revise_plan()
    assert not dialog._busy
    assert "Confirm the Codex connection" in dialog.status_label.text()
    assert calculations == []
    dialog.close()


def test_stubborn_owned_login_cleanup_kills_and_reaps_without_blocking_qt(
    editor, qtbot, monkeypatch
):
    from qtpy.QtCore import QTimer

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class StubbornLogin(_LoginProcess):
        killed = 0
        waits = 0

        def terminate(self):
            self.terminated += 1  # Deliberately ignores termination.

        def wait(self, timeout):
            self.waits += 1
            if self.returncode is None:
                entered.set()
                release.wait(2)
                raise subprocess.TimeoutExpired("fake-login", timeout)
            finished.set()
            return self.returncode

        def kill(self):
            self.killed += 1
            self.returncode = -9

    process = StubbornLogin()
    _mock_codex(
        monkeypatch,
        lambda *_a, **_k: pytest.fail("No inference"),
        lambda **_k: process,
        lambda: _connection_runtime(),
    )
    widget, calculations = editor
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.waitUntil(lambda: not dialog._connection_busy)
    dialog.signin_button.click()
    qtbot.waitUntil(lambda: dialog._signin_process is process)
    try:
        dialog.cancel_button.click()
        qtbot.waitUntil(entered.is_set)
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        assert process.killed == 0
        assert not dialog._connection_busy
        # A repeated stale callback must not schedule a second cleanup.
        dialog._stop_owned_login(process)
        release.set()
        qtbot.waitUntil(finished.is_set)
        assert process.terminated == 1
        assert process.killed == 1
        assert process.waits == 2
        assert calculations == []
    finally:
        release.set()
        qtbot.waitUntil(finished.is_set)
        dialog.close()


def test_owned_cleanup_thread_is_bounded_non_daemon_and_deduplicated(
    qtbot, monkeypatch
):
    from napari_vipp.ui import workflow_assistant

    real_thread = threading.Thread
    launched = []
    reaped = threading.Event()

    def thread(*args, **kwargs):
        launched.append(kwargs)
        return real_thread(*args, **kwargs)

    class CompletedLogin(_LoginProcess):
        def wait(self, timeout):
            assert timeout == 0.5
            reaped.set()
            return 0

    monkeypatch.setattr(workflow_assistant.threading, "Thread", thread)
    process = CompletedLogin(returncode=0)
    workflow_assistant.WorkflowAssistantDialog._stop_owned_login(process)
    workflow_assistant.WorkflowAssistantDialog._stop_owned_login(process)
    qtbot.waitUntil(reaped.is_set)
    assert len(launched) == 1
    assert launched[0]["daemon"] is False
    assert process.terminated == 0
