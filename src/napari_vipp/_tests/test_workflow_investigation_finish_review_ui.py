"""User-controlled Finish review with offline providers and synthetic CPU data."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import QPoint, QRect, Qt

from napari_vipp._tests.test_workflow_investigation_ui import (
    _experiment_assessment,
    _finish,
    _install_actions,
    _start,
    _trial,
)
from napari_vipp._tests.test_workflow_investigation_ui import product as product
from napari_vipp._tests.test_workflow_refinement_investigation_ui import _accepted


def start_review(product, monkeypatch, qtbot, actions=None):
    actions = actions or [_trial(32), _finish()]
    _install_actions(product, monkeypatch, actions)
    product.dialog._investigation.review_checkbox.setChecked(True)
    _start(product)
    controller = product.dialog._investigation
    qtbot.waitUntil(
        lambda: controller._review_pending or not controller.active, timeout=15_000
    )
    assert controller._review_pending, (
        product.errors,
        product.dialog.status_label.text(),
    )
    return controller


def _freeze_review_clock(session, monkeypatch):
    """Advance controlled wall time without sleeping or changing used work."""
    ticks = [session._now()]
    monkeypatch.setattr(session, "_clock", lambda: ticks[0])
    return ticks


def test_review_setting_defaults_true_is_scoped_and_passive(product, qtbot):
    from napari_vipp.ui.workflow_assistant_investigation import (
        AssistantInvestigationController,
    )

    dialog = product.dialog
    fresh = AssistantInvestigationController(dialog)
    qtbot.addWidget(fresh.options)
    qtbot.addWidget(fresh.card)
    assert fresh.review_checkbox.isChecked()
    fresh.review_checkbox.setChecked(False)
    assert fresh.session is None and product.calls == []
    controller = dialog._investigation
    assert controller.review_checkbox.isHidden()  # Ordinary one-shot planning.
    controller.checkbox.setChecked(True)
    assert not controller.review_checkbox.isHidden()
    controller.checkbox.setChecked(False)
    dialog._refinement.slider.setValue(2)
    assert not controller.review_checkbox.isHidden()
    assert product.calls == [] and controller.session is None


def test_goal_finish_pauses_without_request_spinner_add_or_run(
    product, monkeypatch, qtbot
):
    controller = start_review(product, monkeypatch, qtbot)
    dialog, session = product.dialog, controller.session
    assert controller.active and dialog._busy and not session.finished
    assert session.pending_finish_review["candidate_id"] == "candidate1"
    assert "Trial 1 — your decision" == controller.review_title.text()
    assert "candidate1" in controller.review_title.toolTip()
    assert "Toy intensity ramp only" in controller.review_limitations.text()
    assert controller.review_limitations.textFormat() == Qt.PlainText
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert not dialog.prepare_button._sending
    assert not dialog.prepare_button._spinner_timer.isActive()
    assert not dialog._request_elapsed_timer.isActive()
    assert dialog.request_progress.isHidden() and dialog.request_activity.isHidden()
    assert (
        not dialog.source_combo.isEnabled() and not dialog.settings_button.isEnabled()
    )
    assert not controller.review_checkbox.isEnabled()
    assert len(product.calls) == 4 and set(product.widget.pipeline.nodes) == {"input"}
    assert controller.review_body_area.maximumHeight() == 220
    assert controller.review_feedback.maxLength() == 1000
    assert not controller.review_gallery_toggle.isChecked()
    assert controller.review_continue_button.property("assistantAction") == "primary"
    for button in (
        controller.review_continue_button,
        controller.review_accept_button,
        controller.review_stop_button,
    ):
        assert button.minimumHeight() == 34
        assert not button.autoDefault()
    # A timer/queued-next callback cannot start a provider or rebuild cached PNGs.
    monkeypatch.setattr(
        session, "preview_images", lambda: pytest.fail("No gallery rebuild")
    )
    freshness_checks = []
    original_assert = session.assert_finish_review_current

    def checked_once():
        freshness_checks.append(True)
        return original_assert()

    monkeypatch.setattr(session, "assert_finish_review_current", checked_once)
    controller._tick()
    assert len(freshness_checks) == 1
    controller._next_guarded()
    assert len(product.calls) == 4 and controller._review_pending


def test_good_enough_accepts_exact_choice_but_add_and_run_remain_separate(
    product, monkeypatch, qtbot
):
    controller = start_review(product, monkeypatch, qtbot)
    dialog, session = product.dialog, controller.session
    candidate = session._candidate("candidate1")
    controller.review_accept_button.click()
    assert session.finished and session.selected_candidate is candidate
    assert (
        not controller.active and not dialog._busy and controller.review_card.isHidden()
    )
    assert len(product.calls) == 4 and set(product.widget.pipeline.nodes) == {"input"}
    assert dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert "Accepted for plan review" in dialog.status_label.text()
    dialog.apply_button.click()
    added = tuple(dialog._added_ids)
    assert len(added) == 1 and product.widget.pipeline.outputs[added[0]] is None
    assert dialog.run_button.isEnabled()
    dialog.run_button.click()
    qtbot.waitUntil(
        lambda: product.widget.pipeline.outputs[added[0]] is not None, timeout=15_000
    )
    np.testing.assert_array_equal(
        product.widget.pipeline.outputs[added[0]], product.data > 32
    )


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_feedback_enter_is_not_a_decision_but_focused_button_space_is(
    product, monkeypatch, qtbot, key
):
    controller = start_review(product, monkeypatch, qtbot)
    dialog, session = product.dialog, controller.session
    dialog.show()
    controller.review_feedback.setText("This is feedback, not approval to send.")
    controller.review_feedback.setFocus()
    qtbot.keyClick(controller.review_feedback, key)
    qtbot.wait(20)
    assert controller.active and controller._review_pending and not session.finished
    assert len(product.calls) == 4 and not dialog.apply_button.isEnabled()
    assert '"user_decision"' not in json.dumps(session.export_notebook())
    controller.review_accept_button.setFocus()
    qtbot.keyClick(controller.review_accept_button, Qt.Key_Space)
    assert session.finished and not controller.active
    assert dialog.apply_button.isEnabled() and len(product.calls) == 4


def test_continue_preserves_session_counters_and_excludes_exact_review_time_once(
    product, monkeypatch, qtbot
):
    controller = start_review(
        product, monkeypatch, qtbot, [_trial(32), _finish(), _finish()]
    )
    session, deadline, session_id = (
        controller.session,
        controller.session.deadline,
        controller.session.session_id,
    )
    before = (session.actions_used, session.model_calls_used, session.attempted_trials)
    ticks = _freeze_review_clock(session, monkeypatch)
    ticks[0] += 123.5
    paused_duration = ticks[0] - session._pause_started
    expected_deadline = deadline + paused_duration
    entered, release = threading.Event(), threading.Event()
    original = product.provider.generate_codex_investigation_action

    def hold_next(request, settings, **kwargs):
        entered.set()
        assert release.wait(8), "Offline continuation was not released."
        return original(request, settings, **kwargs)

    monkeypatch.setattr(
        product.provider, "generate_codex_investigation_action", hold_next
    )
    feedback = "  Please inspect the remaining boundary uncertainty.  "
    controller.review_feedback.setText(feedback)
    controller.review_continue_button.click()
    qtbot.waitUntil(entered.is_set, timeout=10_000)
    try:
        # Duplicate while request is in flight.
        controller.review_continue_button.click()
        assert controller.session is session and session.session_id == session_id
        assert session.deadline == pytest.approx(expected_deadline)
        assert (
            session.actions_used,
            session.model_calls_used,
            session.attempted_trials,
        ) == before
        assert not controller._review_pending and len(product.calls) == 4
        assert not product.dialog.apply_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(
        lambda: controller._review_pending or not controller.active, timeout=15_000
    )
    assert controller._review_pending, product.dialog.status_label.text()
    assert controller.session is session
    assert session.deadline == pytest.approx(expected_deadline)
    assert session._paused_seconds == pytest.approx(paused_duration)
    assert (
        session.actions_used == before[0] + 1
        and session.model_calls_used == before[1] + 1
    )
    assert session.attempted_trials == before[2]
    review = product.calls[-1]["completion_review"]
    assert {key: value for key, value in review.items() if key != "reservation"} == {
        "required": True,
        "pending": False,
        "continue_requests": 1,
        "latest_feedback": {"candidate_id": "candidate1", "feedback": feedback},
    }
    assert len(product.calls) == 5 and not session.finished


def test_review_time_beyond_original_deadline_is_excluded_and_accept_stays_local(
    product, monkeypatch, qtbot
):
    controller = start_review(product, monkeypatch, qtbot)
    session = controller.session
    ticks = _freeze_review_clock(session, monkeypatch)
    usage = (session.actions_used, session.model_calls_used, session.attempted_trials)
    remaining = session.pending_finish_review["remaining_budgets"]["seconds_remaining"]
    active_seconds = session.continuation_state["active_seconds_used"]
    ticks[0] = session.deadline + 120
    controller._tick()
    assert controller.active and controller._review_pending
    assert controller.review_continue_button.isEnabled()
    assert (
        controller.review_accept_button.isEnabled()
        and controller.review_stop_button.isEnabled()
    )
    assert (
        session.pending_finish_review["remaining_budgets"]["seconds_remaining"]
        == remaining
    )
    assert session.continuation_state["active_seconds_used"] == active_seconds
    assert (
        session.actions_used,
        session.model_calls_used,
        session.attempted_trials,
    ) == usage
    assert ticks[0] > session.deadline
    controller.review_accept_button.click()
    assert session.finished and len(product.calls) == 4
    assert product.dialog.apply_button.isEnabled()


def test_exhausted_trial_budget_does_not_disable_good_enough(
    product, monkeypatch, qtbot
):
    product.dialog._investigation.max_attempts.setValue(1)
    controller = start_review(product, monkeypatch, qtbot)
    assert not controller.review_continue_button.isEnabled()
    assert "trial" in controller.review_reason.text().lower()
    assert controller.review_accept_button.isEnabled()
    controller.review_accept_button.click()
    assert controller.session.finished and controller.session.attempted_trials == 1


@pytest.mark.parametrize(
    "cause", ["stop", "output_consent", "source_revision", "close"]
)
def test_pending_stop_or_authority_withdrawal_never_promotes(
    product, monkeypatch, qtbot, cause
):
    controller = start_review(product, monkeypatch, qtbot)
    dialog, session = product.dialog, controller.session
    if cause == "stop":
        controller.review_stop_button.click()
    elif cause == "output_consent":
        dialog.output_assessment_checkbox.setChecked(False)
    elif cause == "source_revision":
        pipeline = product.widget.pipeline
        pipeline.outputs["input"] = pipeline.outputs["input"].copy()
        pipeline.node_outputs["input"][0] = pipeline.outputs["input"]
        controller._tick()
    else:
        dialog.close()
    assert not controller.active and not dialog._busy and not session.finished
    assert controller.review_card.isHidden() and controller.review_gallery_images == []
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert len(product.calls) == 4 and set(product.widget.pipeline.nodes) == {"input"}
    decisions = json.dumps(session.export_notebook())
    assert ('"decision": "stop"' in decisions) is (cause == "stop")


def test_late_duplicate_reply_cannot_bypass_pending_review(product, monkeypatch, qtbot):
    controller = start_review(product, monkeypatch, qtbot)
    epoch = controller._epoch
    controller._ready(epoch, "model", object())
    controller._ready(epoch, "action", object())
    progress = product.dialog.progress_label.text()
    controller._progress(epoch, "A late worker progress callback")
    assert product.dialog.progress_label.text() == progress
    assert controller._review_pending and not controller.session.finished
    assert len(product.calls) == 4 and not product.dialog.apply_button.isEnabled()


def test_duplicate_model_ready_before_pause_display_handles_revoked_authority(
    product, monkeypatch, qtbot
):
    controller = start_review(product, monkeypatch, qtbot)
    # Recreate the short window after core Finish but before its action UI reply.
    controller._review_pending = False
    controller._cancelled.set()
    controller._ready(controller._epoch, "model", object())
    assert not controller.active and controller.review_card.isHidden()
    assert (
        not controller.session.finished and not product.dialog.apply_button.isEnabled()
    )
    assert len(product.calls) == 4


def test_nonempty_untrusted_limitations_are_plain_text_not_a_failed_choice(
    product, monkeypatch, qtbot
):
    finish = _finish()
    finish[1]["limitations"] = [
        "Topology remains uncertain.",
        "<b>Not independently validated</b>",
    ]
    controller = start_review(product, monkeypatch, qtbot, [_trial(32), finish])
    assert "<b>Not independently validated</b>" in controller.review_limitations.text()
    assert controller.review_limitations.textFormat() == Qt.PlainText
    assert controller.review_accept_button.isEnabled()
    controller.review_accept_button.click()
    assert controller.session.finished


def test_gallery_uses_exact_cached_owners_not_the_nominated_candidate(
    product, monkeypatch, qtbot
):
    from napari_vipp.core.workflow_proposal_images import ImagePreview

    controller = start_review(product, monkeypatch, qtbot)
    session, dialog = controller.session, product.dialog
    png = session.preview_images()[0].png_bytes
    description = "Synthetic existing cached view; no scientific rendering."
    description_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
    images = tuple(
        ImagePreview(f"{owner}/toy-foreground XY slice", png, description)
        for owner in ("candidate2", "candidate0")
    )
    selection = {
        "view": "XY slice",
        "display_policy": {"render_description_sha256": description_hash},
    }
    metadata = {
        "comparison_previews": [
            {
                "status": "supplied",
                "role": "toy-foreground",
                "selection": selection,
                "views": [
                    {
                        "candidate_id": owner,
                        "preview_sha256": hashlib.sha256(png).hexdigest(),
                    }
                    for owner in ("candidate0", "candidate2")
                ],
            }
        ]
    }
    dialog.request_edit.setPlainText(json.dumps(metadata))
    monkeypatch.setattr(session, "preview_images", lambda: images)
    controller._clear_finish_review()
    controller._show_finish_review()
    captions = [caption.text() for caption, _ in controller.review_gallery_images]
    assert captions == [images[1].label, images[0].label]
    assert controller.review_title.text() == "Trial 1 — your decision"
    assert "candidate1" in controller.review_title.toolTip()
    notice = controller.review_gallery_layout.itemAt(0).widget().text()
    assert "Views may belong to other trials" in notice
    assert "Missing selected-result views are not rendered" in notice
    assert json.dumps(selection) in controller.review_gallery_images[0][0].toolTip()
    assert all(
        not picture.pixmap().isNull() for _, picture in controller.review_gallery_images
    )
    assert len(product.calls) == 4 and not session.finished


@pytest.mark.skipif(
    not os.environ.get("VIPP_FINISH_REVIEW_GRABS"),
    reason="Owned offscreen visual-QA output directory was not requested.",
)
def test_owned_offscreen_review_grabs(product, monkeypatch, qtbot):
    """Optional local widget grabs, never an existing native application window."""
    finish = _finish()
    finish[1]["summary"] = (
        "Review this detached toy trial before choosing the next step. " * 8
    )
    finish[1]["limitations"] = [
        "Synthetic layout check: boundaries remain uncertain. " * 5,
        "The accepted workflow is unchanged. " * 5,
    ]
    dialog = product.dialog
    dialog.resize(1160, 1000)
    dialog.show()
    controller = start_review(product, monkeypatch, qtbot, [_trial(32), finish])
    directory = Path(os.environ["VIPP_FINISH_REVIEW_GRABS"])
    assert directory.is_dir()
    for name, width in (("wide", 1160), ("narrow", 780)):
        target = directory / f"finish-review-{name}.png"
        assert not target.exists(), "Visual-QA artifacts must be fresh."
        dialog.resize(width, 1000)
        qtbot.wait(100)
        assert controller.review_body_area.height() <= 220
        assert controller.review_accept_button.isVisible()
        assert controller.review_stop_button.isVisible()
        viewport = dialog.review_scroll.viewport()
        for widget in (
            controller.review_title,
            controller.review_feedback,
            controller.review_continue_button,
            controller.review_accept_button,
            controller.review_stop_button,
        ):
            rectangle = QRect(widget.mapTo(viewport, QPoint()), widget.size())
            assert viewport.rect().contains(rectangle)
        assert dialog.grab().save(str(target), "PNG")
    assert len(product.calls) == 4 and not controller.session.finished


def test_warm_start_accepting_baseline_does_not_duplicate_or_run(
    product, monkeypatch, qtbot
):
    node = _accepted(product, qtbot)
    dialog, pipeline = product.dialog, product.widget.pipeline
    original, nodes = pipeline.outputs[node], tuple(pipeline.nodes)
    _install_actions(product, monkeypatch, [_finish("candidate0")])
    dialog._investigation.review_checkbox.setChecked(True)
    dialog._refinement.start_button.click()
    controller = dialog._investigation
    qtbot.waitUntil(
        lambda: controller._review_pending or not controller.active, timeout=15_000
    )
    assert controller._review_pending, (product.errors, dialog.status_label.text())
    assert "Accepted starting result" in controller.review_title.text()
    assert dialog._refinement.active and not dialog.apply_button.isEnabled()
    controller.review_accept_button.click()
    assert controller.session.finished and controller.session.attempted_trials == 0
    assert controller.session.selected_candidate.candidate_id == "candidate0"
    assert tuple(pipeline.nodes) == nodes and pipeline.outputs[node] is original
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert len(product.calls) == 1


def test_warm_protected_alternate_review_then_exact_normal_add_and_run(
    product, monkeypatch, qtbot
):
    node = _accepted(product, qtbot)
    dialog, pipeline = product.dialog, product.widget.pipeline
    original, original_nodes = pipeline.outputs[node], tuple(pipeline.nodes)
    dialog._refinement.configure_preservation((node, 0))
    dialog._investigation.review_checkbox.setChecked(True)

    def plan(request):
        output = request["accepted_candidate"]["outputs"][0]
        return {
            "baseline_candidate_id": "candidate0",
            "evidence_refs": [
                {
                    "kind": "numeric",
                    "alias": output["output_alias"],
                    "sha256": output["sha256"],
                }
            ],
            "observed_failure": "The toy mask omits values 17 through 32.",
            "hypothesis": "Lowering the toy threshold should retain accepted pixels.",
            "prior_assumptions": ["Toy ramp only."],
            "predicted_geometric_change": "More toy foreground; no accepted loss.",
            "risks": ["More foreground does not prove accuracy."],
            "intent": "cleanup",
            "evaluation_checks": [
                {
                    "check_id": "foreground",
                    "kind": "numeric",
                    "criterion": "Check foreground count.",
                }
            ],
        }

    def edit(request):
        key = request["accepted_candidate"]["role_bindings"][0]["node_key"]
        return {
            "candidate_id": "candidate0",
            "edits": [
                {"kind": "set_parameters", "node": key, "params": {"threshold": 16.0}}
            ],
            "experiment_id": request["experiment_ledger"][-1]["experiment_id"],
        }

    _install_actions(
        product,
        monkeypatch,
        [
            ("plan_experiment", plan),
            ("trial_edit", edit),
            ("assess_experiment", _experiment_assessment),
            _finish(),
        ],
    )
    dialog._refinement.start_button.click()
    controller = dialog._investigation
    qtbot.waitUntil(
        lambda: controller._review_pending or not controller.active, timeout=15_000
    )
    session = controller.session
    assert controller._review_pending, (product.errors, dialog.status_label.text())
    assert (
        tuple(pipeline.nodes) == original_nodes and pipeline.outputs[node] is original
    )
    assert not session.finished and not dialog.apply_button.isEnabled()
    expected = session._candidate("candidate1")._outputs[0][1].data.copy()
    np.testing.assert_array_equal(expected, product.data > 16)
    controller.review_accept_button.click()
    assert session.finished and session.selected_candidate.candidate_id == "candidate1"
    assert controller._selected_draft == (
        dialog._prepared._context.context_id,
        dialog._prepared.fingerprint,
    )
    assert dialog._refinement._pending_investigative_promotion is not None
    assert (
        pipeline.outputs[node] is original and tuple(pipeline.nodes) == original_nodes
    )
    assert dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    dialog.apply_button.click()
    added = tuple(dialog._added_ids)
    assert added and len(product.calls) == 4
    dialog.run_button.click()
    qtbot.waitUntil(
        lambda: dialog._assessment.run_id and dialog._assessment.candidate_ids == added,
        timeout=15_000,
    )
    np.testing.assert_array_equal(pipeline.outputs[added[0]], expected)
    np.testing.assert_array_equal(pipeline.outputs[node], product.data > 32)
    assert dialog._refinement.preservation_result.passed
    assert len(product.calls) == 4
