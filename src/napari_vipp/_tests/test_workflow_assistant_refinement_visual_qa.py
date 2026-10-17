"""Offline refinement presentation; no auth, model request or calculation."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from napari.qt import get_stylesheet
from qtpy.QtCore import QPoint

from napari_vipp._tests._qt_fonts import load_offscreen_windows_fonts
from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.core.workflow_refinement import RefinementLoop


def _rectangle(widget, ancestor):
    rectangle = widget.rect()
    rectangle.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rectangle


def _settle(dialog, qtbot):
    timers = (
        dialog._appearance_timer,
        dialog.goal_edit._height_timer,
        dialog.discussion_edit._height_timer,
        dialog.transcript_edit._fit_timer,
    )
    qtbot.waitUntil(lambda: not any(timer.isActive() for timer in timers))
    qtbot.wait(25)


def _save(dialog, theme, size, stage):
    root = os.environ.get("VIPP_REFINEMENT_UI_EVIDENCE")
    if not root:
        return
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{theme}-{size[0]}x{size[1]}-{stage}.png"
    assert not path.exists(), "Retain earlier UI evidence rather than overwrite it."
    assert dialog.grab().save(str(path))


def _assert_pinned(dialog, scroll, action_frame, buttons):
    frame = _rectangle(action_frame, dialog)
    assert dialog.rect().contains(frame)
    assert _rectangle(scroll.viewport(), dialog).bottom() < frame.top()
    close = _rectangle(dialog.close_button, dialog)
    assert dialog.rect().contains(close)
    assert close.top() > frame.bottom()
    assert dialog.close_button.isVisible()
    assert scroll.horizontalScrollBar().maximum() == 0
    for button in buttons:
        assert button.isVisible()
        assert action_frame.rect().contains(_rectangle(button, action_frame))
    return frame, close


@pytest.mark.parametrize("size", [(1040, 820), (760, 600)])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_refinement_controls_history_and_pinned_actions_offline(
    assistant, qtbot, qapp, theme, size
):
    load_offscreen_windows_fonts(qapp)
    old_style = qapp.styleSheet()
    dialog = assistant.dialog
    controller = dialog._refinement
    try:
        qapp.setStyleSheet(get_stylesheet(theme))
        dialog.resize(*size)
        # Confirmed status is synthetic fixture state, never an auth request.
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
        dialog._codex_authenticated = True
        dialog._connection_busy = False
        dialog._update_provider_status()
        dialog.goal_edit.setPlainText(
            "In this z-stack there is a wavy structure and a bright circular "
            "structure in the middle. Create independent complete 3D masks and "
            "meshes, then combine their meshes in different colours."
        )
        dialog.images_checkbox.setChecked(True)
        dialog.output_assessment_checkbox.setChecked(True)
        controller.slider.setValue(2)
        assert controller.slider.minimum() == 0 and controller.slider.maximum() == 5
        assert controller.loop is None
        dialog._refresh_appearance()
        dialog.stage_tabs.setCurrentIndex(0)
        _settle(dialog, qtbot)
        assert (dialog.width(), dialog.height()) == size
        pinned = _assert_pinned(
            dialog,
            dialog.goal_scroll,
            dialog.goal_actions_frame,
            (dialog.prepare_button, dialog.preview_button),
        )
        _save(dialog, theme, size, "goal-top")
        dialog.goal_scroll.ensureWidgetVisible(controller.options, 12, 12)
        _settle(dialog, qtbot)
        assert pinned == _assert_pinned(
            dialog,
            dialog.goal_scroll,
            dialog.goal_actions_frame,
            (dialog.prepare_button, dialog.preview_button),
        )
        assert (
            dialog.goal_scroll.viewport()
            .rect()
            .contains(_rectangle(controller.slider, dialog.goal_scroll.viewport()))
        )
        assert "Up to 2 additional run attempts" in controller.description.text()
        assert "Failed runs count toward this limit" in controller.description.text()
        _save(dialog, theme, size, "goal-refinement")
        # A detached fixed proposal is only presentation state, never a model
        # response or executor output; callbacks are spies from the fixture.
        dialog._ensure_context()
        dialog._accept_proposal(reference_proposal(dialog._context))
        for index, summary in enumerate(
            (
                "Synthetic first run: the two displayed structures remain joined.",
                "Synthetic second run: less background; part of the wave is missing.",
                "Synthetic third run: clearer separation; still needs human review.",
            ),
            start=1,
        ):
            controller.record_report(
                SimpleNamespace(
                    run_id=f"synthetic-{index}",
                    overall="partly_achieved",
                    summary=summary,
                    comparison=SimpleNamespace(
                        explanation="Synthetic comparison only; not a measured result."
                    )
                    if index > 1
                    else None,
                )
            )
        dialog._assessment.run_id = "synthetic-3"
        dialog._sync_buttons()
        controller.history_toggle.setChecked(True)
        dialog.stage_tabs.setCurrentIndex(2)
        _settle(dialog, qtbot)
        assert controller.start_button.isEnabled()
        assert controller.history.isVisible()
        assert controller.history.text().count("Synthetic") == 5
        plans_pinned = _assert_pinned(
            dialog,
            dialog.review_scroll,
            dialog.plans_actions_frame,
            (dialog.apply_button, dialog.run_button),
        )
        dialog.review_scroll.ensureWidgetVisible(controller.start_button, 12, 12)
        _settle(dialog, qtbot)
        viewport = dialog.review_scroll.viewport()
        assert viewport.rect().contains(_rectangle(controller.start_button, viewport))
        assert controller.start_button.width() >= (
            controller.start_button.fontMetrics().horizontalAdvance(
                controller.start_button.text()
            )
            + 20
        )
        _save(dialog, theme, size, "plans-start")
        dialog.review_scroll.ensureWidgetVisible(controller.history, 12, 12)
        _settle(dialog, qtbot)
        assert plans_pinned == _assert_pinned(
            dialog,
            dialog.review_scroll,
            dialog.plans_actions_frame,
            (dialog.apply_button, dialog.run_button),
        )
        _save(dialog, theme, size, "plans-history")
        # Display active controls without starting any timer/worker or advancing
        # a transition. This tests Stop's geometry, not loop behaviour.
        controller.loop = RefinementLoop.start(2, controller._binding())
        controller.note.setText(
            "Synthetic display state: assessing round 2 of up to 2…"
        )
        dialog._sync_buttons()
        dialog.review_scroll.ensureWidgetVisible(controller.stop_button, 12, 12)
        _settle(dialog, qtbot)
        assert controller.stop_button.isVisible() and controller.stop_button.isEnabled()
        assert viewport.rect().contains(_rectangle(controller.stop_button, viewport))
        assert controller.card.rect().contains(
            _rectangle(controller.start_button, controller.card)
        )
        assert controller.card.rect().contains(
            _rectangle(controller.stop_button, controller.card)
        )
        assert not controller.timer.isActive()
        _save(dialog, theme, size, "plans-stop")
        assert assistant.applies == assistant.runs == []
        assert len(assistant.captures) == 1
        assert not dialog._busy
    finally:
        controller.loop = None
        controller.timer.stop()
        qapp.setStyleSheet(old_style)
