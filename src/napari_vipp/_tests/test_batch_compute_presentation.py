"""Compute details stay discoverable without duplicating the toolbar summary."""

from dataclasses import replace

import pytest

from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.ui.batch import CollectionBatchDialog


@pytest.fixture
def make_batch_dialog(qtbot):
    # The real host retains its batch dialog. Mirror that ownership through
    # pytest-qt's post-call event flush: addWidget itself stores only a weakref,
    # so a test-local dialog can otherwise be collected inside a child paint.
    dialogs = []

    def create(**kwargs):
        dialog = CollectionBatchDialog(**kwargs)
        dialogs.append(dialog)
        qtbot.addWidget(dialog)
        return dialog

    yield create
    dialogs.clear()


def _assert_compute_location(dialog, *, in_toolbar):
    assert dialog.compute_summary_label.isVisible() == in_toolbar
    assert dialog.compute_icon_label.isVisible() == in_toolbar
    assert dialog.compute_info_action.isVisible() != in_toolbar
    actions = dialog.more_menu.actions()
    separator = actions[actions.index(dialog.compute_info_action) - 1]
    assert separator.isSeparator()
    assert separator.isVisible() != in_toolbar
    assert dialog.compute_info_action.text() == "Compute details…"
    assert dialog.demo_action.isVisible()


@pytest.mark.parametrize("width", [560, 679, 680, 1080])
def test_compute_details_only_appear_in_menu_without_toolbar_summary(
    qtbot, make_batch_dialog, width
):
    dialog = make_batch_dialog()
    dialog.resize(width, 800)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.width() == width)

    _assert_compute_location(dialog, in_toolbar=width >= 680)


def test_compute_menu_and_separator_follow_repeated_resize(qtbot, make_batch_dialog):
    dialog = make_batch_dialog()
    dialog.show()
    for width in (1080, 560, 680, 679, 1080):
        dialog.resize(width, 800)
        qtbot.waitUntil(lambda target_width=width: dialog.width() == target_width)
        _assert_compute_location(dialog, in_toolbar=width >= 680)
    assert dialog.more_menu.actions().count(dialog.compute_info_action) == 1


def test_compute_refresh_keeps_current_details_and_constant_menu_title(
    make_batch_dialog, tmp_path
):
    summary = [
        "Custom · saved batch",
        "Custom compute settings came from this saved batch configuration.",
    ]
    actions = replace(
        _actions(_preview_result(tmp_path), []),
        compute_summary=lambda: tuple(summary),
    )
    dialog = make_batch_dialog(actions=actions)
    dialog.resize(560, 800)
    dialog.show()

    for label, detail in (
        tuple(summary),
        ("Prefer GPU · inherited", "Uses the main toolbar's Prefer GPU settings."),
    ):
        summary[:] = [label, detail]
        dialog._refresh_compute_summary()
        _assert_compute_location(dialog, in_toolbar=False)
        assert dialog.compute_summary_label.text() == label
        assert detail in dialog.compute_summary_label.toolTip()
        assert detail in dialog.compute_icon_label.toolTip()
        assert detail in dialog.compute_info_action.toolTip()
        dialog.compute_info_action.trigger()
        assert detail in dialog.batch_activity_status.text()

    dialog.resize(1080, 800)
    dialog._refresh_compute_summary()
    _assert_compute_location(dialog, in_toolbar=True)
    assert dialog.compute_summary_label.text() == summary[0]
    assert dialog.more_menu.actions().count(dialog.compute_info_action) == 1


def test_default_compute_details_remain_available_without_host_callback(
    make_batch_dialog,
):
    dialog = make_batch_dialog()
    dialog.resize(560, 800)
    dialog.show()

    _assert_compute_location(dialog, in_toolbar=False)
    assert "main toolbar" in dialog.compute_summary_label.toolTip()
    dialog.compute_info_action.trigger()
    assert dialog.batch_activity_status.text() == dialog.compute_summary_label.toolTip()
