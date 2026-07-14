from __future__ import annotations

import pytest
from qtpy.QtCore import Qt

from napari_vipp.ui.batch_navigator import BatchNavigator


def test_batch_navigator_starts_hidden_and_presents_a_session(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)

    assert navigator.isHidden()
    assert navigator.item_count == 0
    assert navigator.current_index == 0

    navigator.set_session(
        3,
        1,
        "0002_field_b",
        {
            "Primary signal": "02_field_b.npy",
            "Reference": "beta_reference.npy",
        },
    )

    assert not navigator.isHidden()
    assert navigator.item_count == 3
    assert navigator.current_index == 1
    assert navigator.item_label.text() == "Item 2 of 3"
    assert navigator.batch_id_label.text() == "Batch ID: 0002_field_b"
    assert "Primary signal: 02_field_b.npy" in navigator.sources_label.text()
    assert "Reference: beta_reference.npy" in navigator.sources_label.text()
    assert "does not run or save the batch" in navigator.representative_label.text()
    assert navigator.slider.minimum() == 0
    assert navigator.slider.maximum() == 2
    assert navigator.slider.value() == 1
    assert not navigator.slider.hasTracking()
    assert navigator.previous_button.isEnabled()
    assert navigator.next_button.isEnabled()


def test_batch_navigator_emits_zero_based_navigation_and_workspace_requests(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)
    navigator.set_session(4, 0, "0001_a", ["a.npy", "reference-a.npy"])
    selected: list[int] = []
    workspace_requests: list[None] = []
    navigator.itemSelected.connect(selected.append)
    navigator.workspaceRequested.connect(lambda: workspace_requests.append(None))

    assert not navigator.previous_button.isEnabled()
    qtbot.mouseClick(navigator.next_button, Qt.LeftButton)
    assert selected == [1]
    assert navigator.current_index == 1
    assert navigator.slider.value() == 1
    assert navigator.item_label.text() == "Item 2 of 4"

    navigator.set_session(4, 1, "0002_b", ["b.npy", "reference-b.npy"])
    navigator.slider.setValue(3)
    assert selected == [1, 3]
    assert navigator.current_index == 3
    assert not navigator.next_button.isEnabled()

    qtbot.mouseClick(navigator.previous_button, Qt.LeftButton)
    assert selected == [1, 3, 2]
    assert navigator.current_index == 2

    qtbot.mouseClick(navigator.workspace_button, Qt.LeftButton)
    assert workspace_requests == [None]


def test_batch_navigator_clear_session_resets_and_hides_everything(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)
    navigator.set_session(1, 0, "0001_a", ["a.npy"])
    navigator.begin_batch_progress(3)

    navigator.clear_session()

    assert navigator.isHidden()
    assert navigator.item_count == 0
    assert navigator.current_index == 0
    assert navigator.item_label.text() == "Item 0 of 0"
    assert navigator.slider.maximum() == 0
    assert not navigator.previous_button.isEnabled()
    assert not navigator.next_button.isEnabled()
    assert not navigator.workspace_button.isEnabled()
    assert navigator.progress_frame.isHidden()

    navigator.set_session(0, 0, "", [])
    assert navigator.isHidden()


def test_batch_navigator_can_lock_item_changes_while_retaining_context(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)
    navigator.set_session(3, 1, "0002_b", ["b.npy"])

    navigator.set_navigation_enabled(False)

    assert navigator.item_label.text() == "Item 2 of 3"
    assert not navigator.previous_button.isEnabled()
    assert not navigator.next_button.isEnabled()
    assert not navigator.slider.isEnabled()
    assert navigator.workspace_button.isEnabled()

    navigator.set_navigation_enabled(True)
    assert navigator.previous_button.isEnabled()
    assert navigator.next_button.isEnabled()
    assert navigator.slider.isEnabled()

    navigator.set_session_stale(True)
    assert "previous plan" in navigator.representative_label.text()
    assert navigator.slider.isEnabled()

    navigator.set_session_stale(False)
    assert navigator.representative_label.text() == navigator.REPRESENTATIVE_MESSAGE
    assert navigator.slider.isEnabled()


def test_batch_navigator_validates_the_representative_index(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)

    with pytest.raises(ValueError, match="within the active session"):
        navigator.set_session(3, 3, "out-of-range", [])


def test_batch_navigator_displays_determinate_runner_progress(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)
    navigator.set_session(3, 0, "0001_a", ["a.npy"])

    navigator.begin_batch_progress(3, "Preparing three items...")
    assert not navigator.progress_frame.isHidden()
    assert navigator.progress_bar.minimum() == 0
    assert navigator.progress_bar.maximum() == 3
    assert navigator.progress_bar.value() == 0
    assert navigator.progress_label.text() == "Preparing three items..."

    navigator.update_batch_progress(2, 3, "0002_b", "running")
    assert navigator.progress_bar.value() == 1
    assert navigator.progress_label.text() == "Batch 2/3: 0002_b (running)."

    navigator.update_batch_progress(2, 3, "0002_b", "completed")
    assert navigator.progress_bar.value() == 2
    assert navigator.progress_label.text() == "Batch 2/3: 0002_b (completed)."

    navigator.finish_batch_progress("Batch finished: 3 completed.")
    assert navigator.progress_bar.value() == 3
    assert navigator.progress_label.text() == "Batch finished: 3 completed."

    navigator.begin_batch_progress(3)
    navigator.update_batch_progress(2, 3, "0002_b", "running")
    navigator.fail_batch_progress("Batch failed on item 2.")
    assert navigator.progress_bar.value() == 1
    assert navigator.progress_bar.format() == "Failed (%v / %m)"
    assert navigator.progress_label.text() == "Batch failed on item 2."

    navigator.reset_batch_progress()
    assert navigator.progress_frame.isHidden()
    assert navigator.progress_bar.value() == 0
    assert navigator.progress_label.text() == ""


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, "starting"), "at least one item"),
        ((-1, "starting"), "at least one item"),
    ],
)
def test_batch_navigator_rejects_empty_progress(qtbot, args, message):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)

    with pytest.raises(ValueError, match=message):
        navigator.begin_batch_progress(*args)


def test_batch_navigator_rejects_out_of_range_progress(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)

    with pytest.raises(ValueError, match="between 1 and total"):
        navigator.update_batch_progress(0, 3, "", "running")


def test_batch_navigator_stacks_long_details_in_a_narrow_dock(qtbot):
    navigator = BatchNavigator()
    qtbot.addWidget(navigator)
    navigator.set_session(
        30,
        12,
        "0013_extremely_long_experimental_batch_identifier_with_many_tokens",
        {
            "Primary acquisition channel with long title": (
                "2026-07-14_very_long_microscope_filename_primary_"
                "channel_field_0013.npy"
            ),
            "Reference channel": "2026-07-14_reference_channel_field_0013.npy",
        },
    )
    navigator.begin_batch_progress(
        30,
        "Processing an unusually long status message for item 13 of 30...",
    )

    navigator.resize(420, 300)
    navigator.show()
    qtbot.waitUntil(navigator.isVisible)

    assert navigator.minimumSizeHint().width() <= 420
    assert navigator.width() <= 420
    assert navigator._compact_layout is True
    assert navigator.batch_id_label.geometry().bottom() <= (
        navigator.sources_label.geometry().top()
    )
    assert navigator.progress_label.geometry().bottom() <= (
        navigator.progress_bar.geometry().top()
    )
    assert navigator.workspace_button.geometry().right() <= navigator.width()
    assert "extremely_long" in navigator.batch_id_label.toolTip()
