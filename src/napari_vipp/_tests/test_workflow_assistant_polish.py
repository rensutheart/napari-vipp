"""Passive preview layout and matching inline send activity, without models."""

import json
from types import SimpleNamespace

import pytest
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QCheckBox, QPlainTextEdit

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui import workflow_assistant


@pytest.mark.parametrize("action", ["goal", "message", "answers"])
@pytest.mark.parametrize("ending", ["success", "failure", "cancel", "close"])
def test_only_submitted_control_spins_and_all_terminal_paths_stop_it(
    assistant, monkeypatch, qtbot, action, ending
):
    dialog = assistant.dialog
    held = []
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=held.append)),
    )
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-held-model")
    dialog._ensure_context()
    if action != "goal":
        proposal = reference_proposal(dialog._context)
        if action == "answers":
            proposal["questions"] = ["Which boundary should be preserved?"]
        dialog._accept_proposal(proposal)
        dialog.stage_tabs.setCurrentIndex(1)
    if action == "answers":
        dialog._question_fields[0][1].setText("The observed boundary.")
        button = dialog.revise_button
    elif action == "message":
        dialog.discussion_edit.setPlainText("Keep the physical units.")
        button = dialog.send_button
    else:
        button = dialog.prepare_button
    before = GraphSnapshot.from_pipeline(assistant.pipeline)
    button.click()
    assert len(held) == 1
    assert dialog._busy and dialog._sending_action == action
    assert button.text() == f"Sending {action}…"
    assert button.property("sending") is True
    assert button._spinner_timer.isActive()
    assert button.height() >= button.fontMetrics().height() + 16
    angle = button._spinner_angle
    qtbot.waitUntil(lambda: button._spinner_angle != angle)
    for other in (dialog.prepare_button, dialog.send_button, dialog.revise_button):
        if other is not button:
            assert other.property("sending") is False
            assert not other._spinner_timer.isActive()
    generation = dialog._generation
    response = SimpleNamespace(
        proposal=reference_proposal(dialog._context), attempts=()
    )
    if ending == "success":
        dialog._ready(generation, response)
    elif ending == "failure":
        dialog._failed(generation, RuntimeError("Offline synthetic failure"))
    elif ending == "close":
        dialog.close_button.click()
    else:
        dialog.cancel_request()
    assert not dialog._busy
    for other in (dialog.prepare_button, dialog.send_button, dialog.revise_button):
        assert other.property("sending") is False
        assert not other._spinner_timer.isActive()
    if ending in {"cancel", "close"}:
        dialog._ready(generation, response)
        assert not button._spinner_timer.isActive()
        assert button.property("sending") is False
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == before
    assert assistant.applies == assistant.runs == []


def test_hidden_send_spinner_pauses_without_cancelling_request(
    assistant, monkeypatch, qtbot
):
    dialog = assistant.dialog
    held = []
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=held.append)),
    )
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-held-model")
    dialog.prepare_button.click()
    assert dialog.prepare_button._spinner_timer.isActive()
    generation = dialog._generation
    dialog.stage_tabs.setCurrentIndex(1)
    assert not dialog.prepare_button._spinner_timer.isActive()
    assert dialog._busy and dialog._generation == generation
    dialog.stage_tabs.setCurrentIndex(0)
    assert dialog.prepare_button._spinner_timer.isActive()
    dialog.cancel_request()
    assert not dialog.prepare_button._spinner_timer.isActive()
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("advanced", [False, True])
def test_sharing_preview_has_distinct_scrolling_card_and_compact_footer_gap(
    assistant, qtbot, advanced
):
    dialog = assistant.dialog
    dialog._ensure_context()
    request = dialog._context.public_request(dialog.goal_edit.toPlainText())
    preview = dialog._sharing_preview_dialog(request)
    qtbot.addWidget(preview)
    preview.show()
    if advanced:
        toggle = preview.findChild(QCheckBox, "SharedRequestAdvancedToggle")
        toggle.setChecked(True)
    qtbot.waitUntil(lambda: not preview._fit_timer.isActive())
    qtbot.wait(10)
    details = preview.findChild(QPlainTextEdit, "SharedRequestJSON")
    assert details.isReadOnly()
    assert json.loads(details.toPlainText()) == request
    assert preview.scroll.objectName() == "SharedRequestScroll"
    assert preview.scroll.viewport().palette().color(QPalette.Window) != (
        preview.palette().color(QPalette.Window)
    )
    assert "QPlainTextEdit#SharedRequestJSON" in preview.styleSheet()
    assert "border: 1px" in preview.styleSheet()
    assert preview.height() < 740
    assert preview.close_button.y() - preview.scroll.geometry().bottom() <= 16
    assert not preview.close_button.autoDefault()
    assert not dialog.consent_checkbox.isChecked()
    assert assistant.captures == [("input", False)]
    assert assistant.applies == assistant.runs == []


def test_long_shared_content_is_complete_scrollable_and_height_bounded(
    assistant, qtbot
):
    dialog = assistant.dialog
    goal = "Keep separate observed structures and calibrated units. " * 65
    dialog.goal_edit.setPlainText(goal)
    dialog._clarifications = [("Observed boundaries?", "Observed only. " * 110)]
    request = dialog._ensure_context(dialog._answered_objective())
    preview = dialog._sharing_preview_dialog(request)
    qtbot.addWidget(preview)
    preview.show()
    qtbot.waitUntil(lambda: not preview._fit_timer.isActive())
    qtbot.wait(10)
    assert preview.height() <= 740
    assert preview.scroll.verticalScrollBar().maximum() > 0
    details = preview.findChild(QPlainTextEdit, "SharedRequestJSON")
    assert json.loads(details.toPlainText()) == request
    assert request["objective"] == goal
    assert len(request["clarifications"][0]["answer"]) > 1000
    assert not dialog.consent_checkbox.isChecked()
    assert assistant.captures == [("input", False)]
    assert assistant.applies == assistant.runs == []
