"""Recovery is explicit, bounded, and cannot reuse stale scientific authority."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QPoint

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposal_model import (
    ModelProposalError,
    SafeModelDiagnostic,
)
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui import workflow_assistant

_GOAL = "  Keep the observed goal.\nRetain its exact second line and spaces.  "
_MESSAGE = "  Unsent current message.\nPreserve this second line and spaces.  "


@pytest.fixture
def held(assistant, monkeypatch):
    queued = []
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=queued.append)),
    )
    return assistant, queued


def _local(dialog):
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-held-model")
    dialog.goal_edit.setPlainText(_GOAL)


def _plan(dialog, questions=()):
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = list(questions)
    dialog._accept_proposal(raw)
    assert dialog._prepared is not None
    return raw


def _error(category="network", reason="unknown"):
    return ModelProposalError(
        "Synthetic safe failure",
        diagnostic=SafeModelDiagnostic("local", "request", category, reason=reason),
    )


def _rect_in(widget, ancestor):
    rectangle = widget.rect()
    rectangle.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rectangle


def _payload(dialog):
    return dialog._context.public_request(dialog._request_objective)


@pytest.mark.parametrize("change", ["balanced", "best", "images"])
def test_initial_policy_change_has_no_nonexistent_previous_plan_warning(held, change):
    assistant, queued = held
    dialog = assistant.dialog
    goal, message = dialog.goal_edit.toPlainText(), dialog.discussion_edit.toPlainText()
    if change == "images":
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
        dialog.images_checkbox.setChecked(True)
    else:
        dialog.initiative_slider.setValue(1 if change == "balanced" else 2)
    assert not dialog._drafts and dialog._recovery_kind is None
    assert dialog.recovery_button.isHidden()
    assert dialog.status_icon.accessibleName() == "Information"
    assert "previous plan" not in dialog.status_label.text().lower()
    assert (
        dialog.goal_edit.toPlainText() == goal
        and dialog.discussion_edit.toPlainText() == message
    )
    assert dialog._prepared is None and not dialog.apply_button.isEnabled()
    assert assistant.captures == queued == assistant.applies == assistant.runs == []


def test_stale_policy_update_is_explicit_preserves_answers_and_defers_blank_choices(
    held, qtbot
):
    assistant, queued = held
    dialog = assistant.dialog
    _local(dialog)
    _plan(
        dialog,
        ("First choice?", "Unanswered second choice?", "Unanswered third choice?"),
    )
    dialog._question_fields[0][1].setText("  Keep the observed first boundary.  ")
    dialog._clarifications = [("Earlier question?", "  Earlier exact answer.  ")]
    dialog.discussion_edit.setPlainText(_MESSAGE)
    chat, drafts = tuple(dialog._chat), tuple(dialog._drafts)
    pixels, snapshot = (
        assistant.data.copy(),
        GraphSnapshot.from_pipeline(assistant.pipeline),
    )
    dialog.initiative_slider.setValue(2)
    assert dialog._recovery_kind == "update"
    assert (
        "initiative"
        in (dialog.summary_label.text() + dialog.status_label.text()).lower()
    )
    assert dialog._prepared is None and not dialog._draft_authorized
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert dialog._image_draft_guard[0].image_previews == ()
    assert tuple(dialog._chat) == chat and tuple(dialog._drafts) == drafts
    for size in ((760, 600), (1040, 820)):
        dialog.resize(*size)
        for stage in (0, 1, 2):
            dialog.stage_tabs.setCurrentIndex(stage)
            qtbot.wait(10)
            assert dialog.recovery_button.isVisible()
            assert dialog.rect().contains(_rect_in(dialog.recovery_button, dialog))
    assert queued == [] and assistant.captures == [("input", False)]
    dialog.recovery_button.click()
    assert len(queued) == 1 and dialog._busy and dialog._sending_action == "recovery"
    request = _payload(dialog)
    assert request["objective"] == _GOAL
    assert request["planning_preferences"]["mode"] == "use_best_judgment"
    assert request["clarifications"] == [
        {"question": "Earlier question?", "answer": "  Earlier exact answer.  "},
        {
            "question": "First choice?",
            "answer": "  Keep the observed first boundary.  ",
        },
    ]
    assert request["deferred_questions"] == [
        "Unanswered second choice?",
        "Unanswered third choice?",
    ]
    assert all(pair["answer"] != _MESSAGE for pair in request["clarifications"])
    assert dialog.discussion_edit.toPlainText() == _MESSAGE
    assert assistant.captures == [("input", False), ("input", False)]
    assert (
        dialog.recovery_button._spinner_timer.isActive()
        and not dialog.recovery_button.isEnabled()
    )
    for button in (dialog.prepare_button, dialog.send_button, dialog.revise_button):
        assert not button.property("sending") and not button._spinner_timer.isActive()
    generation = dialog._generation
    dialog.recovery_button.click()
    dialog._recover_plan_clicked()
    assert len(queued) == 1 and dialog._generation == generation
    dialog._ready(
        generation,
        SimpleNamespace(proposal=reference_proposal(dialog._context), attempts=()),
    )
    assert dialog._recovery_kind is None and not dialog._recovering
    assert (
        dialog.recovery_button.isHidden()
        and not dialog.recovery_button._spinner_timer.isActive()
    )
    assert dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert (
        dialog.goal_edit.toPlainText() == _GOAL
        and dialog.discussion_edit.toPlainText() == _MESSAGE
    )
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert assistant.applies == assistant.runs == []


def test_conservative_update_routes_to_questions_without_sending_or_capturing(held):
    assistant, queued = held
    dialog = assistant.dialog
    _local(dialog)
    _plan(dialog, ("First choice?", "Second choice?"))
    dialog.initiative_slider.setValue(1)
    dialog.stage_tabs.setCurrentIndex(2)
    assert dialog.recovery_button.text() == "Answer questions"
    dialog.recovery_button.click()
    assert dialog.stage_tabs.currentIndex() == 1
    assert queued == [] and assistant.captures == [("input", False)]
    assert len(dialog._question_fields) == 2 and not dialog.consent_checkbox.isChecked()
    for question, field in dialog._question_fields:
        field.setText(f"  Exact answer to {question}  ")
    assert dialog.recovery_button.text() == "Update plan"
    dialog.recovery_button.click()
    assert len(queued) == 1
    request = _payload(dialog)
    assert "deferred_questions" not in request
    assert request["clarifications"] == [
        {"question": question, "answer": field.text()}
        for question, field in dialog._question_fields
    ]
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    dialog.cancel_request()
    assert not dialog.recovery_button._spinner_timer.isActive()
    assert not dialog.recovery_button.property("sending")


@pytest.mark.parametrize("changed", ["source", "workflow", "selection"])
def test_update_rejects_changed_retained_source_and_review_goal_is_passive(
    held, changed
):
    assistant, queued = held
    dialog = assistant.dialog
    _local(dialog)
    _plan(dialog, ("Observed boundary?",))
    dialog._question_fields[0][1].setText("  Preserve it.  ")
    dialog.initiative_slider.setValue(2)
    if changed == "source":
        replacement = assistant.data.copy()
        replacement[0, 0] += 1
        replacement.flags.writeable = False
        assistant.pipeline.outputs["input"] = replacement
    elif changed == "workflow":
        assistant.pipeline.nodes["input"].params["sample_name"] = (
            "Changed synthetic source"
        )
    else:
        dialog.source_combo.addItem("Another synthetic source", "other")
        dialog.source_combo.blockSignals(True)
        dialog.source_combo.setCurrentIndex(dialog.source_combo.count() - 1)
        dialog.source_combo.blockSignals(False)
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.pipeline.outputs["input"].copy()
    captures = list(assistant.captures)
    dialog.stage_tabs.setCurrentIndex(2)
    dialog.recovery_button.click()
    assert queued == [] and assistant.captures == captures
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert dialog._image_draft_guard is None and not dialog._draft_authorized
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert dialog._recovery_kind == "goal"
    dialog.recovery_button.click()
    assert dialog.stage_tabs.currentIndex() == 0
    assert queued == [] and assistant.captures == captures
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.pipeline.outputs["input"], pixels)
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize(
    ("action", "edited"),
    [("goal", False), ("message", False), ("message", True), ("answers", False)],
)
def test_failed_request_retry_uses_current_literal_action_and_no_prior_invalid_response(
    held, action, edited
):
    assistant, queued = held
    dialog = assistant.dialog
    _local(dialog)
    if action != "goal":
        _plan(dialog, ("Observed boundary?",) if action == "answers" else ())
        dialog.stage_tabs.setCurrentIndex(1)
    if action == "answers":
        dialog._question_fields[0][1].setText("  Keep the exact observed boundary.  ")
        initial = dialog.revise_button
    elif action == "message":
        dialog._clarifications = [("Earlier question?", "  Earlier literal answer.  ")]
        dialog.discussion_edit.setPlainText(_MESSAGE)
        initial = dialog.send_button
    else:
        initial = dialog.prepare_button
    initial.click()
    assert len(queued) == 1 and dialog._busy
    dialog._failed(dialog._generation, _error())
    assert dialog._recovery_kind == "retry" and dialog._retry_action == action
    assert (
        not dialog._busy
        and not dialog.apply_button.isEnabled()
        and not dialog.run_button.isEnabled()
    )
    if edited:
        dialog.discussion_edit.setPlainText(
            "  Edited current message.\nSend these exact words instead.  "
        )
        assert dialog.recovery_button.text() == "Send edited message"
    else:
        assert dialog.recovery_button.text().lower() == (
            "retry plan" if action == "goal" else f"retry {action}"
        )
    message = dialog.discussion_edit.toPlainText()
    maximum = dialog.attempts_spin.value()
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    dialog.recovery_button.click()
    assert len(queued) == 2 and dialog._busy and dialog._sending_action == "recovery"
    assert dialog._recovery_submit_action == action
    request = _payload(dialog)
    assert request["objective"] == (_GOAL.strip() if action == "goal" else _GOAL)
    assert dialog.goal_edit.toPlainText() == _GOAL
    if action == "message":
        assert request["clarifications"][-1]["answer"] == message
        assert dialog.discussion_edit.toPlainText() == message
    elif action == "answers":
        assert request["clarifications"] == [
            {
                "question": "Observed boundary?",
                "answer": "  Keep the exact observed boundary.  ",
            }
        ]
    assert "previous_response" not in request and "validation_feedback" not in request
    assert dialog.attempts_spin.value() == maximum <= 2
    assert dialog.recovery_button._spinner_timer.isActive()
    generation = dialog._generation
    dialog.recovery_button.click()
    dialog._recover_plan_clicked()
    assert len(queued) == 2 and dialog._generation == generation
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert assistant.applies == assistant.runs == []
    dialog.cancel_request()
    assert not dialog._recovering
    assert not dialog.recovery_button._spinner_timer.isActive()
    assert not dialog.recovery_button.property("sending")
    dialog._ready(
        generation,
        SimpleNamespace(proposal=reference_proposal(dialog._context), attempts=()),
    )
    assert dialog._prepared is None and not dialog.apply_button.isEnabled()
    assert not dialog.recovery_button._spinner_timer.isActive()
    assert not dialog.recovery_button.property("sending")


@pytest.mark.parametrize(
    ("category", "reason", "kind"),
    [
        ("unauthorized", "unknown", "settings"),
        ("bad_request", "invalid_schema", "details"),
    ],
)
def test_settings_and_details_recovery_are_passive_not_provider_retries(
    held, monkeypatch, category, reason, kind
):
    assistant, queued = held
    dialog = assistant.dialog
    _local(dialog)
    dialog.prepare_button.click()
    dialog._failed(dialog._generation, _error(category, reason))
    assert dialog._recovery_kind == kind
    settings = []
    monkeypatch.setattr(dialog, "_open_settings", lambda: settings.append(True))
    captures = list(assistant.captures)
    request = dialog.request_edit.toPlainText()
    goal = dialog.goal_edit.toPlainText()
    dialog.recovery_button.click()
    if kind == "settings":
        assert settings == [True]
    else:
        assert (
            dialog.stage_tabs.currentIndex() == 2
            and dialog.advanced_checkbox.isChecked()
        )
        assert settings == []
    assert len(queued) == 1 and assistant.captures == captures
    assert (
        dialog.request_edit.toPlainText() == request
        and dialog.goal_edit.toPlainText() == goal
    )
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


def test_failed_model_discovery_never_offers_a_plan_retry(held, monkeypatch):
    assistant, queued = held
    dialog = assistant.dialog
    _local(dialog)
    dialog._refresh_models()
    assert len(queued) == 1 and dialog._job_kind == "models"
    dialog._failed(dialog._generation, _error())
    assert dialog._recovery_kind != "retry"
    monkeypatch.setattr(dialog, "_open_settings", lambda: None)
    if dialog.recovery_button.isVisible():
        dialog.recovery_button.click()
    assert len(queued) == 1 and assistant.captures == []
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []
