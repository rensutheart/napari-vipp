"""Readable, staged settings and explicit connection actions without inference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFont

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.ui.workflow_assistant import _AssistantSettingsDialog
from napari_vipp.ui.workflow_assistant_style import AssistantDisclosureButton


@pytest.fixture
def settings(assistant, qtbot, monkeypatch):
    """Only synthetic state and explicit action spies; no runtime or credentials."""
    from napari_vipp.core import workflow_proposal_codex, workflow_proposal_model

    parent = assistant.dialog
    actions = []
    monkeypatch.setattr(
        workflow_proposal_model,
        "discover_models",
        lambda *_args, **_kwargs: actions.append("backend-model-list") or (),
    )
    monkeypatch.setattr(
        workflow_proposal_codex,
        "start_codex_login",
        lambda *_args, **_kwargs: pytest.fail("No real sign-in may start in UI tests"),
    )
    monkeypatch.setattr(
        workflow_proposal_codex,
        "discover_codex_runtime",
        lambda: (
            actions.append("backend-status")
            or SimpleNamespace(supported=True, authenticated=False, detail="Synthetic.")
        ),
    )
    monkeypatch.setattr(parent, "_begin_codex_signin", lambda: actions.append("signin"))
    monkeypatch.setattr(
        parent, "_check_codex_connection", lambda: actions.append("check")
    )
    monkeypatch.setattr(
        parent, "_repair_codex_connection", lambda: actions.append("repair")
    )
    monkeypatch.setattr(
        workflow_proposal_codex,
        "repair_codex_profile",
        lambda *_args, **_kwargs: pytest.fail("No real profile repair in UI tests"),
        raising=False,
    )
    window = _AssistantSettingsDialog(parent)
    qtbot.addWidget(window)
    parent._settings_dialog = window
    window.provider_combo.setCurrentIndex(window.provider_combo.findData("codex"))
    parent._codex_authenticated = False
    parent._connection_busy = False
    parent._set_connection_status("warning", "Sign in to Codex.")
    parent._sync_buttons()
    window.show()
    qtbot.waitUntil(lambda: window.isVisible())
    yield SimpleNamespace(window=window, parent=parent, actions=actions, held=assistant)
    parent._settings_dialog = None


def _rect_in(widget, ancestor):
    rectangle = widget.rect()
    rectangle.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rectangle


def _assert_passive(held, snapshot, pixels):
    assert GraphSnapshot.from_pipeline(held.pipeline) == snapshot
    np.testing.assert_array_equal(held.data, pixels)
    assert not held.data.flags.writeable
    assert held.captures == held.assertions == held.applies == held.runs == []


def test_chatgpt_signin_and_connection_check_are_distinct_obvious_actions(settings):
    window = settings.window
    assert window.signin_button.text() == "Sign in with ChatGPT…"
    assert window.connection_refresh_button.text() == "Check connection"
    assert "sign in" in window.connection_title.text().lower()
    guidance = (window.connection_hint.text() + window.browser_hint.text()).lower()
    assert "browser" in guidance
    assert "automatically" in guidance
    assert window.signin_button.isEnabled()
    assert window.connection_refresh_button.isEnabled()
    assert window.objectName() == "WorkflowAssistantSettings"
    assert window.styleSheet()
    assert not window.signin_button.autoDefault()
    assert not window.connection_refresh_button.autoDefault()
    assert settings.actions == []


def test_advanced_starts_collapsed_and_contains_technical_controls(settings, qtbot):
    window = settings.window
    toggle = window.advanced_toggle
    assert isinstance(toggle, AssistantDisclosureButton)
    assert not toggle.isChecked()
    assert window.advanced_panel.isHidden()
    assert "expand" in toggle.accessibleName().lower()
    for control in (
        window.codex_model_edit,
        window.reasoning_combo,
        window.attempts_spin,
        window.timeout_spin,
        window.connection_label,
    ):
        assert window.advanced_panel.isAncestorOf(control)
        assert not control.isVisible()
    qtbot.keyClick(toggle, Qt.Key_Return)
    assert toggle.isChecked()
    assert window.advanced_panel.isVisible()
    assert "collapse" in toggle.accessibleName().lower()
    assert settings.actions == []


def test_advanced_container_background_rules_are_scoped_and_expansion_is_passive(
    settings,
):
    snapshot = GraphSnapshot.from_pipeline(settings.held.pipeline)
    pixels = settings.held.data.copy()
    window = settings.window
    names = (
        (window.advanced_panel, "AssistantSettingsAdvanced"),
        (window.codex_options, "AssistantSettingsCodexOptions"),
        (window.connection_details, "AssistantSettingsConnectionDetails"),
    )
    for container, name in names:
        assert container.objectName() == name
        selector = f"QWidget#{name}"
        assert selector in window.styleSheet()
        rule = (
            window.styleSheet().split(selector, 1)[1].split("{", 1)[1].split("}", 1)[0]
        )
        assert "background: transparent" in rule
        assert "border: none" in rule
    window.advanced_toggle.setChecked(True)
    for control in (
        window.codex_model_edit,
        window.reasoning_combo,
        window.attempts_spin,
        window.timeout_spin,
    ):
        assert control.isVisible()
        assert control.objectName() not in {name for _container, name in names}
    assert settings.actions == []
    _assert_passive(settings.held, snapshot, pixels)


def test_open_provider_disclosure_and_resize_are_passive(settings, qtbot):
    held = settings.held
    snapshot = GraphSnapshot.from_pipeline(held.pipeline)
    pixels = held.data.copy()
    active_provider = settings.parent._provider()
    goal = settings.parent.goal_edit.toPlainText()
    for provider in ("local", "example", "codex"):
        settings.window.provider_combo.setCurrentIndex(
            settings.window.provider_combo.findData(provider)
        )
        for expanded in (True, False):
            settings.window.advanced_toggle.setChecked(expanded)
            settings.window.resize(620, 580 if not expanded else 760)
            qtbot.wait(1)
    assert settings.parent._provider() == active_provider
    assert settings.parent.goal_edit.toPlainText() == goal
    assert settings.actions == []
    _assert_passive(held, snapshot, pixels)


def test_signin_is_explicit_and_does_not_apply_settings_or_send_a_plan(settings):
    parent = settings.parent
    active_provider = parent._provider()
    settings.window.signin_button.click()
    assert settings.actions == ["signin"]
    assert parent._provider() == active_provider
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


def test_connection_check_is_explicit_and_never_starts_signin(settings):
    settings.window.connection_refresh_button.click()
    assert settings.actions == ["check"]
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


def test_completed_connection_updates_open_settings_without_reopening(settings):
    parent = settings.parent
    parent._connection_kind = "status"
    parent._connection_ready(
        parent._connection_generation,
        SimpleNamespace(supported=True, authenticated=True, detail="Synthetic check."),
    )
    window = settings.window
    assert "connected" in window.connection_title.text().lower()
    assert window.signin_button.text() == "Sign in again…"
    assert "Signed in to Codex" in window.connection_label.text()
    assert window.signin_button.isEnabled()
    assert window.accept_button.isEnabled()
    assert settings.actions == []


@pytest.mark.parametrize("kind", ["signin", "status"])
def test_busy_connection_state_disables_duplicate_actions_but_keeps_cancel(
    settings, kind
):
    parent = settings.parent
    parent._connection_busy = True
    parent._connection_kind = kind
    parent._set_connection_status("info", "Synthetic connection activity.")
    parent._sync_buttons()
    window = settings.window
    assert not window.signin_button.isEnabled()
    assert not window.connection_refresh_button.isEnabled()
    assert not window.accept_button.isEnabled()
    assert window.cancel_button.isEnabled()
    assert window.signin_button.property("sending") is (kind == "signin")
    assert window.connection_refresh_button.property("sending") is (kind == "status")
    assert (
        "sign" in window.connection_title.text().lower()
        if kind == "signin"
        else ("check" in window.connection_title.text().lower())
    )
    assert settings.actions == []
    parent._connection_busy = False
    parent._sync_buttons()


def test_modified_profile_has_setup_guidance_not_a_misleading_signin_retry(settings):
    parent = settings.parent
    diagnostic = "The VIPP Codex profile was modified; no request was sent."
    parent._connection_failed(parent._connection_generation, RuntimeError(diagnostic))
    window = settings.window
    assert parent._codex_connection_issue == "profile"
    assert window.connection_title.text() == "Connection setup needs attention"
    assert diagnostic in window.connection_label.text()
    assert "sign in again" not in parent.connection_label.text().lower()
    assert ".." not in parent.connection_label.text()
    assert not window.signin_button.isEnabled()
    assert window.connection_refresh_button.isEnabled()
    assert not parent._codex_authenticated
    assert window.advanced_panel.isHidden()
    window.advanced_toggle.setChecked(True)
    assert window.connection_label.isVisible()
    assert settings.actions == []
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


def test_generic_connection_failure_keeps_explicit_signin_available(settings):
    parent = settings.parent
    parent._connection_failed(
        parent._connection_generation, RuntimeError("Synthetic runtime check failed.")
    )
    window = settings.window
    assert parent._codex_connection_issue != "profile"
    assert not parent._codex_authenticated
    assert window.signin_button.isEnabled()
    assert window.connection_refresh_button.isEnabled()
    assert "Synthetic runtime check failed" in window.connection_label.text()
    assert ".." not in parent.connection_label.text()
    assert settings.actions == []


def _repairable_connection(settings):
    parent = settings.parent
    parent._connection_kind = "status"
    parent._connection_ready(
        parent._connection_generation,
        SimpleNamespace(
            supported=False,
            authenticated=False,
            repairable=True,
            detail="The VIPP Codex profile was modified; no request was sent.",
        ),
    )


def test_safe_setup_update_offers_repair_above_advanced_without_auto_repair(settings):
    _repairable_connection(settings)
    window = settings.window
    assert settings.parent._codex_profile_repairable
    assert window.connection_title.text() == "Connection setup needs an update"
    assert window.connection_repair_button.isVisible()
    assert window.connection_repair_button.isEnabled()
    assert window.connection_repair_button.text() == "Repair connection setup"
    assert not window.advanced_toggle.isChecked()
    assert window.signin_button.isHidden()
    assert "credentials" in window.connection_hint.text().lower()
    window.advanced_toggle.setChecked(True)
    window.advanced_toggle.setChecked(False)
    window.connection_refresh_button.click()
    assert settings.actions == ["check"]


def test_explicit_repair_button_is_a_separate_action_not_a_goal_or_signin(settings):
    _repairable_connection(settings)
    settings.window.connection_repair_button.click()
    assert settings.actions == ["repair"]
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


def test_unsafe_or_unconfirmed_setup_does_not_enable_repair(settings):
    settings.parent._connection_failed(
        settings.parent._connection_generation,
        RuntimeError("The VIPP Codex profile was modified; no request was sent."),
    )
    assert settings.window.connection_repair_button.isVisible()
    assert not settings.window.connection_repair_button.isEnabled()
    assert not settings.parent._codex_profile_repairable
    settings.window.connection_repair_button.click()
    assert settings.actions == []


def test_busy_repair_shows_only_its_spinner_and_blocks_duplicate_changes(settings):
    parent = settings.parent
    parent._connection_busy = True
    parent._connection_kind = "repair"
    parent._sync_buttons()
    window = settings.window
    assert window.connection_title.text() == "Repairing connection setup…"
    assert window.connection_repair_button.isVisible()
    assert window.connection_repair_button.property("sending")
    assert not window.signin_button.property("sending")
    assert not window.connection_refresh_button.property("sending")
    assert not window.connection_repair_button.isEnabled()
    assert not window.connection_refresh_button.isEnabled()
    assert not window.provider_combo.isEnabled()
    assert not window.accept_button.isEnabled()
    assert window.cancel_button.isEnabled()
    assert settings.actions == []
    parent._connection_busy = False
    parent._sync_buttons()


@pytest.mark.parametrize("authenticated", [False, True])
def test_repair_completion_reports_saved_signin_without_sending_goal(
    settings, authenticated
):
    _repairable_connection(settings)
    parent = settings.parent
    parent._connection_kind = "repair"
    parent._connection_busy = True
    parent._connection_ready(
        parent._connection_generation,
        SimpleNamespace(
            supported=True,
            authenticated=authenticated,
            repairable=False,
            detail="Synthetic saved sign-in status.",
        ),
    )
    assert not parent._connection_busy
    assert parent._codex_authenticated is authenticated
    assert not parent._codex_profile_repairable
    assert settings.window.connection_repair_button.isHidden()
    assert settings.window.signin_button.isVisible()
    assert settings.window.provider_combo.isEnabled()
    assert settings.window.accept_button.isEnabled()
    assert settings.actions == []
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


def test_cancelled_repair_ignores_late_success_and_keeps_connection_unconfirmed(
    settings,
):
    parent = settings.parent
    parent._connection_busy = True
    parent._connection_kind = "repair"
    generation = parent._connection_generation
    cancelled = parent._connection_cancelled
    parent._sync_buttons()
    settings.window.reject()
    assert cancelled.is_set()
    assert not parent._connection_busy
    assert not parent._codex_authenticated
    parent._connection_ready(
        generation,
        SimpleNamespace(supported=True, authenticated=True, repairable=False),
    )
    assert not parent._codex_authenticated
    assert not parent._codex_profile_repairable
    assert settings.actions == []


@pytest.mark.parametrize("route", ["result", "exception"])
@pytest.mark.parametrize(
    "diagnostic",
    [
        "This VIPP connection setup cannot be repaired automatically. "
        "Its managed configuration or profile contents need review; "
        "no credentials were removed.",
        "A previous VIPP connection repair did not finish; "
        "the connection remains blocked. No credentials were removed.",
    ],
    ids=("unrepairable", "interrupted"),
)
def test_unrepairable_and_interrupted_diagnostics_never_offer_a_signin_retry(
    settings, route, diagnostic
):
    parent = settings.parent
    parent._connection_kind = "status"
    if route == "result":
        parent._connection_ready(
            parent._connection_generation,
            SimpleNamespace(
                supported=False,
                authenticated=False,
                repairable=False,
                detail=diagnostic,
            ),
        )
    else:
        parent._connection_failed(
            parent._connection_generation, RuntimeError(diagnostic)
        )
    assert parent._codex_connection_issue == "profile"
    assert not parent._codex_profile_repairable
    assert settings.window.connection_title.text() == "Connection setup needs attention"
    assert settings.window.signin_button.isHidden()
    assert settings.window.connection_repair_button.isVisible()
    assert not settings.window.connection_repair_button.isEnabled()
    assert settings.window.connection_refresh_button.isEnabled()
    assert diagnostic in settings.window.connection_label.text()
    assert settings.actions == []


def test_real_repair_dispatch_requires_eligibility_and_passes_cancellation(
    settings, monkeypatch
):
    from napari_vipp.core import workflow_proposal_codex
    from napari_vipp.ui.workflow_assistant import WorkflowAssistantDialog

    parent = settings.parent
    jobs = []
    callbacks = []
    monkeypatch.setattr(
        parent,
        "_start_connection",
        lambda kind, call, cancelled=None: jobs.append((kind, call, cancelled)),
    )
    monkeypatch.setattr(
        workflow_proposal_codex,
        "repair_codex_profile",
        lambda *, cancelled: (
            callbacks.append(cancelled)
            or SimpleNamespace(supported=True, authenticated=False, repairable=False)
        ),
    )
    WorkflowAssistantDialog._repair_codex_connection(parent)
    assert jobs == []
    _repairable_connection(settings)
    WorkflowAssistantDialog._repair_codex_connection(parent)
    assert len(jobs) == 1
    kind, call, cancelled = jobs[0]
    assert kind == "repair"
    assert not cancelled.is_set()
    assert call().supported
    assert len(callbacks) == 1 and not callbacks[0]()
    cancelled.set()
    assert callbacks[0]()
    assert settings.actions == []
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


def test_protected_profile_details_shortcut_is_visible_and_passive(settings):
    window = settings.window
    assert window.connection_details_button.isHidden()
    settings.parent._connection_failed(
        settings.parent._connection_generation,
        RuntimeError("The VIPP Codex profile changed; no request was sent."),
    )
    assert window.connection_details_button.isVisible()
    assert not window.advanced_toggle.isChecked()
    window.connection_details_button.click()
    assert window.advanced_toggle.isChecked()
    assert window.connection_label.isVisible()
    assert not window.signin_button.isEnabled()
    assert settings.actions == []


def test_unsupported_runtime_guard_result_also_blocks_misleading_signin(settings):
    """Discovery reports expected guard failures as results, not worker errors."""
    parent = settings.parent
    parent._connection_kind = "status"
    parent._connection_ready(
        parent._connection_generation,
        SimpleNamespace(
            supported=False,
            authenticated=False,
            detail="The VIPP Codex profile was modified; no request was sent.",
        ),
    )
    assert parent._codex_connection_issue == "profile"
    assert settings.window.connection_title.text() == "Connection setup needs attention"
    assert not settings.window.signin_button.isEnabled()
    assert settings.window.connection_refresh_button.isEnabled()
    assert settings.window.connection_details_button.isVisible()
    assert not parent._codex_authenticated
    assert settings.actions == []


def test_new_confirmed_status_clears_old_setup_warning_in_settings_and_header(settings):
    parent = settings.parent
    parent.provider_combo.setCurrentIndex(parent.provider_combo.findData("codex"))
    parent._connection_kind = "status"
    parent._connection_ready(
        parent._connection_generation,
        SimpleNamespace(
            supported=False,
            authenticated=False,
            detail="The VIPP Codex profile was modified; no request was sent.",
        ),
    )
    assert parent.provider_status_label.text() == "ChatGPT · Setup needs attention"
    assert settings.window.connection_details_button.isVisible()
    parent._connection_ready(
        parent._connection_generation,
        SimpleNamespace(supported=True, authenticated=True, detail="Synthetic check."),
    )
    assert parent._codex_connection_issue == ""
    assert parent._codex_authenticated
    assert parent.provider_status_label.text() == "ChatGPT · Connected"
    assert settings.window.connection_title.text() == "ChatGPT connected"
    assert settings.window.connection_details_button.isHidden()
    assert settings.window.signin_button.isEnabled()
    assert settings.window.signin_button.text() == "Sign in again…"
    assert settings.actions == []


def test_unrelated_error_mentions_of_profile_are_not_classified_as_guard_failures(
    settings,
):
    settings.parent._connection_failed(
        settings.parent._connection_generation,
        RuntimeError("Synthetic timeout while checking an account profile."),
    )
    assert settings.parent._codex_connection_issue == ""
    assert settings.window.signin_button.isEnabled()
    assert settings.window.connection_details_button.isHidden()
    assert settings.actions == []


def test_stale_connection_failure_cannot_replace_new_confirmed_state(settings):
    parent = settings.parent
    parent._codex_authenticated = True
    parent._set_connection_status("success", "Signed in to Codex. Synthetic status.")
    parent._sync_buttons()
    before = (
        settings.window.connection_title.text(),
        settings.window.connection_label.text(),
    )
    parent._connection_failed(
        parent._connection_generation - 1,
        RuntimeError("The VIPP Codex profile was modified; no request was sent."),
    )
    assert parent._codex_authenticated
    assert before == (
        settings.window.connection_title.text(),
        settings.window.connection_label.text(),
    )
    assert settings.actions == []


@pytest.mark.parametrize("provider", ["codex", "local", "example"])
def test_provider_specific_cards_and_attempt_limits_are_staged(settings, provider):
    window = settings.window
    active_provider = settings.parent._provider()
    window.provider_combo.setCurrentIndex(window.provider_combo.findData(provider))
    assert window.codex_group.isVisible() is (provider == "codex")
    assert window.local_group.isVisible() is (provider == "local")
    assert window.example_note.isVisible() is (provider == "example")
    assert window.timeout_spin.maximum() == 600
    assert window.attempts_spin.minimum() == 1
    assert window.attempts_spin.maximum() == 2
    assert settings.parent._provider() == active_provider
    assert not window._model_busy
    assert settings.actions == []


def test_old_local_model_discovery_cannot_overwrite_new_address_or_closed_settings(
    settings,
):
    window = settings.window
    window.provider_combo.setCurrentIndex(window.provider_combo.findData("local"))
    window.model_combo.setCurrentText("chosen-synthetic-model")
    stale = window._generation
    window.address_edit.setText("http://127.0.0.1:5432")
    status = window.local_status.text()
    window._models_ready(stale, ("late-synthetic-model",))
    window._models_failed(stale, RuntimeError("Late discovery failure."))
    assert window.model_combo.currentText() == "chosen-synthetic-model"
    assert window.local_status.text() == status
    window.reject()
    window._models_ready(window._generation, ("post-close-model",))
    assert window.model_combo.currentText() == "chosen-synthetic-model"
    assert settings.actions == []


def test_local_model_listing_is_explicit_and_not_a_planning_request(settings, qtbot):
    window = settings.window
    window.provider_combo.setCurrentIndex(window.provider_combo.findData("local"))
    assert settings.actions == []
    window.refresh_button.click()
    qtbot.waitUntil(lambda: not window._model_busy)
    assert settings.actions == ["backend-model-list"]
    assert "Found 0 models" in window.local_status.text()
    assert settings.held.captures == settings.held.applies == settings.held.runs == []


@pytest.mark.parametrize("provider", ["codex", "local", "example"])
def test_narrow_large_font_advanced_settings_keep_footer_outside_scrolling_body(
    settings, qtbot, provider
):
    window = settings.window
    window.provider_combo.setCurrentIndex(window.provider_combo.findData(provider))
    font = QFont(window.font())
    font.setPointSize(16)
    window.setFont(font)
    window.advanced_toggle.setChecked(True)
    window.resize(420, 760)
    qtbot.waitUntil(lambda: not window._fit_timer.isActive())
    assert window.width() == 420
    assert window.scroll.horizontalScrollBar().maximum() == 0
    for button in (window.accept_button, window.cancel_button):
        assert not window.scroll.isAncestorOf(button)
        assert button.isVisible()
        assert window.rect().contains(_rect_in(button, window))
        assert button.width() >= button.minimumSizeHint().width()
        assert button.height() >= button.fontMetrics().height() + 10
    assert settings.actions == []


@pytest.mark.parametrize("expanded", [False, True])
def test_extremely_long_local_model_name_does_not_widen_narrow_large_font_settings(
    settings, qtbot, expanded
):
    window = settings.window
    window.provider_combo.setCurrentIndex(window.provider_combo.findData("local"))
    window.kind_combo.setCurrentIndex(1)
    font = QFont(window.font())
    font.setPointSize(16)
    window.setFont(font)
    model_name = "synthetic-local-model-with-an-extremely-long-name-" * 12
    window.model_combo.addItem(model_name)
    window.model_combo.setCurrentText(model_name)
    window.advanced_toggle.setChecked(expanded)
    window.resize(420, 760)
    qtbot.waitUntil(lambda: not window._fit_timer.isActive())
    assert window.width() == 420
    assert window.scroll.horizontalScrollBar().maximum() == 0
    assert window.model_combo.currentText() == model_name
    for control in (window.kind_combo, window.model_combo, window.refresh_button):
        rectangle = _rect_in(control, window.body)
        assert rectangle.left() >= 0
        assert rectangle.right() < window.body.width()
    assert settings.actions == []


@pytest.mark.parametrize("provider", ["codex", "local", "example"])
def test_collapsed_settings_fit_content_and_keep_footer_buttons_reachable(
    settings, qtbot, provider
):
    window = settings.window
    window.provider_combo.setCurrentIndex(window.provider_combo.findData(provider))
    qtbot.wait(10)
    assert window.height() < 650
    assert window.width() <= 760
    for button in (window.accept_button, window.cancel_button):
        assert button.isVisible()
        assert window.rect().contains(_rect_in(button, window))
        assert button.width() >= button.minimumSizeHint().width()
        assert button.height() >= button.fontMetrics().height() + 10
    assert settings.actions == []
