"""Monotonic request activity and longer limits, without waiting or inference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy.QtCore import QElapsedTimer, QTimer

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui import workflow_assistant


@pytest.fixture
def held_clock(assistant, monkeypatch):
    """Hold every worker; advancing a fake display clock needs no Qt wait."""
    queued = []
    seconds = [0]
    dialog = assistant.dialog
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=queued.append)),
    )
    monkeypatch.setattr(dialog, "_request_elapsed_seconds", lambda: seconds[0])
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    yield SimpleNamespace(
        assistant=assistant,
        dialog=dialog,
        queued=queued,
        seconds=seconds,
        snapshot=snapshot,
    )
    dialog.cancel_request(announce=False)
    assert not dialog._request_elapsed_timer.isActive()
    assert assistant.applies == assistant.runs == []
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert not assistant.data.flags.writeable


def _local(dialog):
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-held-model")


def _send(held, action="goal"):
    dialog = held.dialog
    _local(dialog)
    if action != "goal":
        dialog._ensure_context()
        raw = reference_proposal(dialog._context)
        raw["questions"] = (
            ["Keep the observed boundary?"] if action == "answers" else []
        )
        dialog._accept_proposal(raw)
        dialog.stage_tabs.setCurrentIndex(1)
    if action == "message":
        dialog.discussion_edit.setPlainText("Keep both observed structures separate.")
        button = dialog.send_button
    elif action == "answers":
        dialog._question_fields[0][1].setText("Keep it.")
        button = dialog.revise_button
    else:
        dialog.stage_tabs.setCurrentIndex(0)
        button = dialog.prepare_button
    button.click()
    assert dialog._busy and dialog._job_kind == "proposal"
    assert len(held.queued) == 1
    return button, dialog._generation


def _tick(held, seconds):
    held.seconds[0] = seconds
    held.dialog._request_elapsed_timer.timeout.emit()


def _assert_elapsed(dialog, expected):
    assert dialog.request_elapsed_label.isVisible()
    assert expected in dialog.request_elapsed_label.text()


def _reply(dialog):
    return SimpleNamespace(proposal=reference_proposal(dialog._context), attempts=())


def test_codex_default_and_settings_allow_ten_minutes_without_a_request(
    held_clock, qtbot
):
    dialog = held_clock.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
    assert dialog.timeout_spin.value() == 600
    assert dialog.timeout_spin.maximum() >= 600
    settings = workflow_assistant._AssistantSettingsDialog(dialog)
    qtbot.addWidget(settings)
    assert settings.timeout_spin.value() == 600
    assert settings.timeout_spin.maximum() >= 600
    assert held_clock.queued == []
    assert not dialog._request_elapsed_timer.isActive()
    assert not dialog.request_elapsed_label.isVisible()


@pytest.mark.parametrize("action", ["goal", "message", "answers"])
def test_send_shows_elapsed_alongside_existing_spinner_without_running_graph(
    held_clock, action
):
    dialog = held_clock.dialog
    button, _generation = _send(held_clock, action)
    assert isinstance(dialog._request_elapsed, QElapsedTimer)
    assert dialog._request_elapsed.isValid()
    assert isinstance(dialog._request_elapsed_timer, QTimer)
    assert dialog._request_elapsed_timer.isActive()
    assert dialog.request_progress.isVisible()
    assert button.property("sending") and button._spinner_timer.isActive()
    _assert_elapsed(dialog, "0:00")
    _tick(held_clock, 65)
    _assert_elapsed(dialog, "1:05")
    assert not button.isEnabled()
    assert dialog.cancel_button.isEnabled()
    assert held_clock.assistant.applies == held_clock.assistant.runs == []


def test_check_repair_and_connection_recovery_keep_total_request_elapsed(held_clock):
    dialog = held_clock.dialog
    _button, generation = _send(held_clock)
    _tick(held_clock, 95)
    _assert_elapsed(dialog, "1:35")
    for message, seconds, expected in (
        ("Checking plan", 109, "1:49"),
        ("Repairing plan using validation feedback", 245, "4:05"),
        (
            "Codex is recovering the connection for this same request…",
            359,
            "5:59",
        ),
    ):
        dialog._progress(generation, message)
        assert dialog._request_elapsed.isValid()
        assert dialog._request_elapsed_timer.isActive()
        _tick(held_clock, seconds)
        _assert_elapsed(dialog, expected)
    assert dialog._generation == generation
    assert len(held_clock.queued) == 1


def test_elapsed_remains_visible_across_tabs_and_can_exceed_old_timeout(held_clock):
    dialog = held_clock.dialog
    _send(held_clock)
    for stage, seconds, expected in (
        (0, 181, "3:01"),
        (1, 305, "5:05"),
        (2, 599, "9:59"),
    ):
        dialog.stage_tabs.setCurrentIndex(stage)
        _tick(held_clock, seconds)
        _assert_elapsed(dialog, expected)
        assert dialog._busy and dialog._request_elapsed_timer.isActive()
    assert len(held_clock.queued) == 1


@pytest.mark.parametrize("ending", ["success", "invalid", "failure", "cancel", "close"])
def test_every_terminal_path_stops_and_hides_request_clock(held_clock, ending):
    dialog = held_clock.dialog
    _button, generation = _send(held_clock)
    _tick(held_clock, 67)
    if ending == "success":
        dialog._ready(generation, _reply(dialog))
    elif ending == "invalid":
        dialog._ready(generation, SimpleNamespace(proposal={}, attempts=()))
    elif ending == "failure":
        dialog._failed(generation, RuntimeError("Synthetic connection failure"))
    elif ending == "cancel":
        dialog.cancel_request()
    else:
        dialog.close_button.click()
        assert dialog._closed
    assert not dialog._busy
    assert not dialog._request_elapsed_timer.isActive()
    assert not dialog._request_elapsed.isValid()
    assert not dialog.request_elapsed_label.isVisible()
    assert not dialog.request_progress.isVisible()
    for button in (dialog.prepare_button, dialog.send_button, dialog.revise_button):
        assert not button.property("sending")
        assert not button._spinner_timer.isActive()
    _tick(held_clock, 600)
    assert not dialog.request_elapsed_label.isVisible()
    assert not dialog._request_elapsed_timer.isActive()


def test_fresh_retry_starts_at_zero_and_late_callbacks_cannot_stop_new_clock(
    held_clock,
):
    dialog = held_clock.dialog
    _button, old_generation = _send(held_clock)
    old_reply = _reply(dialog)
    _tick(held_clock, 179)
    dialog._failed(old_generation, RuntimeError("Synthetic retryable failure"))
    assert not dialog._request_elapsed_timer.isActive()
    held_clock.seconds[0] = 0
    dialog.recovery_button.click()
    assert len(held_clock.queued) == 2
    generation = dialog._generation
    assert generation != old_generation
    _assert_elapsed(dialog, "0:00")
    before = (
        dialog.status_label.text(),
        dialog.progress_label.text(),
        dialog.request_elapsed_label.text(),
        dialog.stage_tabs.currentIndex(),
    )
    dialog._progress(old_generation, "Repairing an abandoned plan")
    dialog._ready(old_generation, old_reply)
    dialog._failed(old_generation, RuntimeError("Stale failure"))
    assert (
        dialog.status_label.text(),
        dialog.progress_label.text(),
        dialog.request_elapsed_label.text(),
        dialog.stage_tabs.currentIndex(),
    ) == before
    assert dialog._busy and dialog._request_elapsed_timer.isActive()
    assert dialog._request_elapsed.isValid() and dialog._generation == generation
    _tick(held_clock, 32)
    _assert_elapsed(dialog, "0:32")


def test_no_ai_example_and_local_model_discovery_do_not_start_planning_clock(
    held_clock,
):
    dialog = held_clock.dialog
    dialog.prepare_button.click()
    assert dialog._prepared is not None
    assert not dialog._request_elapsed.isValid()
    assert not dialog._request_elapsed_timer.isActive()
    assert not dialog.request_elapsed_label.isVisible()
    assert held_clock.queued == []
    _local(dialog)
    dialog._refresh_models()
    assert len(held_clock.queued) == 1 and dialog._busy
    assert dialog._job_kind == "models"
    assert not dialog._request_elapsed.isValid()
    assert not dialog._request_elapsed_timer.isActive()
    assert not dialog.request_elapsed_label.isVisible()
    dialog._ready(dialog._generation, ["synthetic-model"])
    assert not dialog._busy and not dialog._request_elapsed_timer.isActive()


@pytest.mark.parametrize("action", ["signin", "status"])
def test_connection_actions_do_not_start_planning_clock_or_send_image_data(
    held_clock, monkeypatch, action
):
    from napari_vipp.core import workflow_proposal_codex

    dialog = held_clock.dialog
    monkeypatch.setattr(
        workflow_proposal_codex,
        "start_codex_login",
        lambda **_kwargs: pytest.fail("A held UI test must not perform sign-in"),
    )
    monkeypatch.setattr(
        workflow_proposal_codex,
        "discover_codex_runtime",
        lambda: pytest.fail("A held UI test must not inspect real credentials"),
    )
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
    if action == "signin":
        dialog._signin_codex()
    else:
        dialog._check_codex_connection()
    assert len(held_clock.queued) == 1 and dialog._connection_busy
    assert not dialog._busy
    assert not dialog._request_elapsed.isValid()
    assert not dialog._request_elapsed_timer.isActive()
    assert not dialog.request_elapsed_label.isVisible()
    assert dialog._context is None
    assert held_clock.assistant.captures == []
