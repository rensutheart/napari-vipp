"""Native provider/continuation controls; no real model or paid request."""

import json
import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtCore import QPoint, QRect
from qtpy.QtWidgets import QLabel, QLineEdit

from napari_vipp._tests.test_workflow_investigation_ui import (
    _install_actions,
    _start,
)
from napari_vipp._tests.test_workflow_investigation_ui import product as product
from napari_vipp.core.workflow_agent_provider import OpenAISettings
from napari_vipp.core.workflow_investigation import InvestigationAction
from napari_vipp.core.workflow_proposal_model import ModelSettings
from napari_vipp.ui.workflow_assistant import _AssistantSettingsDialog


@pytest.mark.parametrize("provider", ["local", "openai"])
def test_native_provider_uses_same_agent_and_resumes_question(
    product, monkeypatch, qtbot, provider
):
    from napari_vipp.core import workflow_agent_provider

    d = product.dialog
    d.provider_combo.setCurrentIndex(d.provider_combo.findData(provider))
    d.model_combo.setCurrentText("offline-local-vision")
    d.api_model_edit.setText("offline-api-model")
    _install_actions(
        product,
        monkeypatch,
        [
            ("question", {"question": "Which toy foreground should be kept?"}),
            ("question", {"question": "Offline verification has finished. Stop here?"}),
        ],
    )
    settings_seen = []

    def synthetic(request, settings, **kwargs):
        settings_seen.append(settings)
        return product.provider.generate_codex_investigation_action(
            request, settings, **kwargs
        )

    monkeypatch.setattr(
        workflow_agent_provider, "generate_investigation_action", synthetic
    )
    _start(product)
    controller = d._investigation
    qtbot.waitUntil(lambda: controller._paused, timeout=15_000)
    session = controller.session
    assert controller.active and not session.stop_reason
    assert session.model_calls_used == 1
    assert session.actions_used == 1
    assert not d.request_progress.isVisible()
    assert len(product.calls[0]["image_previews"]) > 0
    assert isinstance(
        settings_seen[0], ModelSettings if provider == "local" else OpenAISettings
    )
    controller.pause_answer.setText("Keep the brighter foreground; use your judgment.")
    controller.resume_button.click()
    qtbot.waitUntil(
        lambda: session.actions_used == 2 and controller._paused, timeout=15_000
    )
    assert controller.session is session
    assert session.model_calls_used == 2
    assert d._clarifications == list(session.dialogue.clarifications)
    assert "Keep the brighter foreground" in str(product.calls[1])
    assert not session.stop_reason
    assert set(product.widget.pipeline.nodes) == {"input"}
    controller.cancel("Offline test finished.")


def test_effort_controls_passive_and_exactly_bound_to_session(
    product, monkeypatch, qtbot
):
    d = product.dialog
    controller = d._investigation
    assert controller.effort.actions.value() == 64
    controller.effort.minutes.setValue(5)
    controller.effort.actions.setValue(12)
    controller.effort.calls.setValue(15)
    controller.effort.image_mib.setValue(2)
    assert controller.session is None and product.calls == []
    _install_actions(
        product, monkeypatch, [("question", {"question": "Keep this toy choice?"})]
    )
    _start(product)
    qtbot.waitUntil(lambda: controller._paused, timeout=15_000)
    session = controller.session
    assert session.limits.max_seconds == 300
    assert session.limits.max_actions == 12
    assert session.limits.max_model_calls == 15
    assert session.limits.max_image_bytes == 2 * 1024 * 1024
    original_id = session.session_id
    original_notebook = session.export_notebook()
    with controller.agent_loop.tools.exclusive():
        session.grant_allowance(replace(session.limits, max_actions=20))
    assert session.session_id == original_id
    assert session.model_calls_used == session.actions_used == 1
    assert session.export_notebook() != original_notebook  # Owner grant is journalled.
    assert session.continuation_state["grants_used"] == 1
    controller.cancel("Offline test finished.")


def test_settings_api_is_explicit_and_keeps_only_env_name(product, qtbot, monkeypatch):
    monkeypatch.setenv("VIPP_TEST_KEY", "private-test-value-never-display")
    d = product.dialog
    settings = _AssistantSettingsDialog(d)
    qtbot.addWidget(settings)
    settings.provider_combo.setCurrentIndex(settings.provider_combo.findData("openai"))
    settings.api_key_env_edit.setText("VIPP_TEST_KEY")
    settings.api_model_edit.setText("explicit-project-model")
    assert settings.api_group.isVisibleTo(settings.body)
    fields = settings.findChildren(QLabel) + settings.findChildren(QLineEdit)
    assert all(
        "private-test-value-never-display" not in field.text() for field in fields
    )
    settings.reject()
    assert d._provider() == "codex"
    assert d.api_key_env_edit.text() == "OPENAI_API_KEY"
    assert product.calls == []


def test_api_controls_do_not_offer_unsupported_standalone_assessment(product):
    d = product.dialog
    d.provider_combo.setCurrentIndex(d.provider_combo.findData("openai"))
    d.output_assessment_checkbox.setChecked(True)
    d._sync_buttons()
    assert d._investigation.checkbox.isChecked()
    assert not d._investigation.checkbox.isEnabled()
    assert not d._assessment.assess_button.isEnabled()
    assert "Start refinement" in d._assessment.note.text()
    assert "OpenAI API" in d.output_assessment_description.text()
    assert not product.calls


def test_selecting_local_provider_preserves_ordinary_planning_choice(product):
    d = product.dialog
    assert not d._investigation.checkbox.isChecked()
    d.provider_combo.setCurrentIndex(d.provider_combo.findData("local"))
    assert not d._investigation.checkbox.isChecked()
    assert d._investigation.checkbox.isEnabled()
    assert not product.calls


def test_investigation_sharing_preview_discloses_additional_output_scope(product):
    d = product.dialog
    d._investigation.checkbox.setChecked(True)
    request = d._ensure_context(d._answered_objective())
    preview = d._sharing_preview_dialog(request)
    try:
        scope = preview.findChild(QLabel, "SharedInvestigationScope")
        assert scope is not None
        assert "not the whole investigation" in scope.text()
        assert "cached results" in scope.text()
        assert "Those output images are not shown here" in scope.text()
        assert not product.calls
    finally:
        preview.close()


def test_fresh_question_does_not_reuse_unsent_answer(product, monkeypatch, qtbot):
    _install_actions(
        product,
        monkeypatch,
        [("question", {"question": "Which toy foreground should be kept?"})],
    )
    controller = product.dialog._investigation
    controller.pause_answer.setText("An unsent answer from a previous question")
    _start(product)
    qtbot.waitUntil(lambda: controller._paused, timeout=15_000)
    assert controller.pause_answer.text() == ""
    controller.pause_answer.setText("Keep this draft while this question is shown")
    controller._show_pause()
    assert controller.pause_answer.text() == (
        "Keep this draft while this question is shown"
    )
    controller.cancel("Offline test finished.")


def test_external_start_captures_current_user_graph_and_blocks_edits(product):
    d, widget = product.dialog, product.widget
    node = widget.pipeline.add_node("binary_threshold")
    assert widget.pipeline.connect("input", node.id).success
    node.params["threshold"] = 42.0
    widget._build_graph_from_pipeline()
    widget.run_pipeline(force_sync=True)
    widget._sync_current_workflow_tab_state()
    d.output_assessment_checkbox.setChecked(True)
    d.consent_checkbox.setChecked(True)
    # An external client owns its model/authentication; VIPP needs no Codex login.
    d._codex_authenticated = False
    controller = d._investigation
    controller.start(external=True)
    assert controller.active, d.status_label.text()
    assert controller._external and not product.calls
    assert controller.session.candidates[0].candidate_id == "candidate0"
    assert (
        controller.session.candidates[0].recipe["nodes"][0]["params"]["threshold"]
        == 42.0
    )
    assert controller.session.attempted_trials == 0
    session = controller.session
    result = session.perform(
        InvestigationAction(
            session.session_id, "describe_candidate", {"candidate_id": "candidate0"}
        )
    )
    assert result.public_result()["status"] == "ok"
    assert set(widget.pipeline.nodes) == {"input", node.id}
    node.params["threshold"] = 71.0
    with pytest.raises(ValueError, match="changed"):
        controller._guard(check_budget=False)
    controller.cancel("User edit invalidated the captured graph.")


def test_external_finish_requires_real_native_review_and_keeps_original(product, qtbot):
    from napari_vipp.core.workflow_agent_live import WorkflowAgentLiveClient

    d, widget = product.dialog, product.widget
    node = widget.pipeline.add_node("binary_threshold")
    assert widget.pipeline.connect("input", node.id).success
    widget._build_graph_from_pipeline()
    widget.run_pipeline(force_sync=True)
    widget._sync_current_workflow_tab_state()
    d.output_assessment_checkbox.setChecked(True)
    d.images_checkbox.setChecked(True)
    d._live_agent.start()
    controller = d._investigation
    session = controller.session
    bridge = d._live_agent.bridge
    assert bridge is not None, d._live_agent.status_label.text()
    replies, errors = [], []

    def client_work():
        client = None
        try:
            client = WorkflowAgentLiveClient(bridge.descriptor_path)
            messages = [
                {
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "offline-native-test", "version": "1"},
                    },
                },
                {"method": "notifications/initialized"},
                {
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "vipp_session_context",
                        "arguments": {},
                    },
                },
                {
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "vipp_finish",
                        "arguments": {
                            "candidate_id": "candidate0",
                            "summary": "Keep this toy baseline.",
                            "limitations": [
                                "Mechanics test, not scientific validation."
                            ],
                        },
                    },
                },
            ]
            for message in messages:
                replies.append(
                    client.exchange(json.dumps({"jsonrpc": "2.0", **message}).encode())
                )
        except Exception as error:
            errors.append(error)
        finally:
            if client is not None:
                client.close()

    worker = threading.Thread(target=client_work, daemon=True)
    worker.start()
    qtbot.waitUntil(lambda: not worker.is_alive(), timeout=20_000)
    assert not errors
    assert all("error" not in reply for reply in replies if reply is not None)
    qtbot.waitUntil(lambda: controller._review_pending, timeout=10_000)
    assert controller.review_card.isVisibleTo(d.plans_page)
    assert not session.finished
    assert session.pending_finish_review["candidate_id"] == "candidate0"
    assert session.model_calls_used >= 1
    assert session.actions_used == 1
    assert not product.calls
    assert not d.apply_button.isEnabled()
    assert set(widget.pipeline.nodes) == {"input", node.id}
    controller.review_accept_button.click()
    assert session.finished and not controller.active
    assert session.selected_candidate.candidate_id == "candidate0"
    assert set(widget.pipeline.nodes) == {"input", node.id}
    assert d._live_agent.bridge is None


def test_pause_waits_for_paid_model_step_before_freezing_time(
    product, monkeypatch, qtbot
):
    entered, release = threading.Event(), threading.Event()
    _install_actions(
        product,
        monkeypatch,
        [
            ("question", {"question": "This decided question must not be executed."}),
        ],
        entered=entered,
        release=release,
    )
    try:
        _start(product)
        controller = product.dialog._investigation
        qtbot.waitUntil(entered.is_set, timeout=10_000)
        controller.pause_button.click()
        assert controller._pause_after_model
        assert not controller.session.paused
        release.set()
        qtbot.waitUntil(lambda: controller._paused, timeout=10_000)
        assert controller.session.actions_used == 0
        assert controller.session.model_calls_used == 1
        assert controller.session.pending_question is None
    finally:
        release.set()


def test_live_agent_controls_visual_qa(product, monkeypatch, qtbot):
    destination = os.environ.get("VIPP_AGENT_UI_EVIDENCE")
    if not destination:
        pytest.skip("No owned screenshot evidence directory requested.")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    d = product.dialog
    d.provider_combo.setCurrentIndex(d.provider_combo.findData("local"))
    d.model_combo.setCurrentText("Your local image-capable model")
    d.show()
    d.resize(1040, 850)
    d._investigation.effort.toggle.setChecked(True)
    qtbot.wait(100)
    assert d.prepare_button.isVisible()
    assert d.grab().save(str(output / "goal-local.png"))
    settings = _AssistantSettingsDialog(d)
    qtbot.addWidget(settings)
    settings.show()
    qtbot.wait(100)
    assert settings.grab().save(str(output / "settings-local.png"))
    settings.reject()
    _install_actions(
        product,
        monkeypatch,
        [("question", {"question": "Which foreground should be kept?"})],
    )
    from napari_vipp.core import workflow_agent_provider

    monkeypatch.setattr(
        workflow_agent_provider,
        "generate_investigation_action",
        product.provider.generate_codex_investigation_action,
    )
    _start(product)
    qtbot.waitUntil(lambda: d._investigation._paused, timeout=15_000)
    qtbot.wait(100)
    assert d.grab().save(str(output / "paused-question.png"))
    d.resize(760, 680)
    d._investigation._reveal_pause(d._investigation._epoch)
    qtbot.wait(100)
    assert d.close_button.isVisible() and d._investigation.resume_button.isVisible()
    viewport = d.discussion_scroll.viewport()
    button = d._investigation.resume_button
    bounds = QRect(button.mapTo(viewport, QPoint(0, 0)), button.size())
    assert viewport.rect().contains(bounds)
    assert d.grab().save(str(output / "paused-narrow.png"))
