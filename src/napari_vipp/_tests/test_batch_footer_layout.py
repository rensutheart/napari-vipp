"""Rendered geometry regressions for the single-row batch footer."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QPoint, QRect
from qtpy.QtWidgets import QApplication

from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.core.batch import BatchStatus
from napari_vipp.ui.batch import CollectionBatchDialog


def _footer_rect(dialog, widget):
    return QRect(widget.mapTo(dialog.footer, QPoint(0, 0)), widget.size())


def _assert_single_row(dialog, *, width, height, state):
    QApplication.processEvents()
    footer = dialog.footer
    assert dialog.width() == width, state
    assert footer.height() == height, state
    assert footer.minimumHeight() == footer.maximumHeight() == height, state

    buttons = [button for button in dialog._footer_buttons if button.isVisible()]
    assert buttons, state
    widgets = [dialog.batch_activity_status]
    if dialog.source_detection_progress.isVisible():
        widgets.append(dialog.source_detection_progress)
    if dialog.footer_elapsed_label.isVisible() and dialog.footer_elapsed_label.text():
        widgets.append(dialog.footer_elapsed_label)
    widgets.extend(buttons)
    rectangles = [_footer_rect(dialog, widget) for widget in widgets]
    assert all(rect.width() > 0 for rect in rectangles), state
    assert all(footer.rect().contains(rect) for rect in rectangles), (
        state,
        footer.rect(),
        rectangles,
    )
    centers = [rect.center().y() for rect in rectangles]
    assert max(centers) - min(centers) <= 2, (state, rectangles)
    assert all(
        left.right() < right.left()
        for left, right in zip(rectangles, rectangles[1:], strict=False)
    ), (state, rectangles)

    activity = _footer_rect(dialog, dialog.batch_activity_strip)
    actions = _footer_rect(dialog, dialog.footer_action_row)
    assert activity.right() < actions.left(), (state, activity, actions)
    assert abs(activity.center().y() - actions.center().y()) <= 2, state


@pytest.mark.parametrize("width", [1080, 640])
def test_footer_stays_one_row_through_batch_lifecycle(qtbot, tmp_path, width):
    plan = _preview_result(tmp_path)
    actions = replace(_actions(plan, []), check_batch=lambda *_args: None)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.resize(width, 800)
    dialog.show()
    QApplication.processEvents()
    height = dialog.footer.height()

    def check(state):
        _assert_single_row(dialog, width=width, height=height, state=state)

    check("idle")
    assert dialog.source_detection_progress.isHidden()
    assert dialog.next_button.isVisible()

    dialog.next_button.click()
    assert dialog._checking_plan
    assert dialog.source_detection_progress.isVisible()
    assert dialog.source_detection_progress.maximum() == 0
    check("checking")

    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.tabs.setCurrentIndex(1)
    assert dialog.footer_overrides_button.isVisible()
    assert dialog.next_button.isVisible()
    check("checked items")

    dialog.begin_run(3)
    dialog.update_run_progress(2, 3, plan.items[1].batch_id, "running")
    dialog._update_elapsed()
    assert dialog.cancel_run_button.isVisible()
    assert dialog.close_button.text() == "Hide window"
    assert dialog.footer_elapsed_label.text()
    assert dialog.source_detection_progress.maximum() == 3
    check("running")

    result = SimpleNamespace(
        manifest=SimpleNamespace(
            items=tuple(
                SimpleNamespace(index=item.index, status=BatchStatus.COMPLETED)
                for item in plan.items
            )
        ),
        summary={"completed": 3, "partial": 0, "skipped": 0, "failed": 0},
        saved_paths=(),
        manifest_path=tmp_path / "manifest.json",
    )
    dialog.finish_run(result)
    assert dialog.cancel_run_button.isHidden()
    assert dialog.source_detection_progress.value() == 3
    check("finished")
