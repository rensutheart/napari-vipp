"""Two ordinary progress lines never shift the bars; a third may expand."""

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.core.batch import BatchPreflightProgress
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_progress import BatchProgressLabel
from napari_vipp.ui.batch_results import BatchResultsPanel


def test_progress_text_invalidates_the_outer_height_for_width_cache(qtbot):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.begin_run(14)
    panel.resize(560, 900)
    panel.show()
    qtbot.wait(5)
    group = panel.progress_group
    outer_item = panel.layout().itemAt(panel.layout().indexOf(group))
    label = panel.run_progress_label
    width = group.width()

    label.setText("First\nSecond\nThird")
    group.layout().activate()
    three_lines = outer_item.heightForWidth(width)
    label.setText("Short status")
    # No event-loop wait: the surrounding layout must not reuse the previous
    # three-line height while the inner group is already measuring two lines.
    two_lines = outer_item.heightForWidth(width)
    assert two_lines == group.layout().totalHeightForWidth(width)
    assert two_lines < three_lines


def test_elapsed_clock_contributes_one_unwrapped_line_to_intrinsic_layout(qtbot):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.begin_run(14)
    panel.elapsed_label.setText("Elapsed 123:45:56")
    panel.run_progress_label.setText("Sample")
    panel.operation_progress_label.setText("Node")
    panel.resize(560, 900)
    panel.show()
    qtbot.wait(5)

    clock = panel.elapsed_label
    group = panel.progress_group
    assert not clock.wordWrap()
    assert group.sizeHint().width() >= clock.sizeHint().width()
    assert group.sizeHint().height() == group.layout().sizeHint().height()
    assert clock.height() == clock.sizeHint().height()


def test_progress_reserve_matches_native_two_lines_after_style_polishing(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    label = BatchProgressLabel("Short status")
    label.setMargin(2)
    bar = QProgressBar()
    layout.addWidget(label)
    layout.addWidget(bar)
    layout.addStretch()
    reference = QLabel("First line\nSecond line", host)
    reference.setTextFormat(Qt.PlainText)
    reference.setWordWrap(True)
    reference.setTextInteractionFlags(Qt.TextSelectableByMouse)
    reference.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
    reference.setMargin(2)
    reference.hide()
    host.resize(500, 300)
    host.show()

    for stylesheet in (
        "QLabel { font-size: 13px; padding: 1px 4px; }",
        "QLabel { font-size: 19px; padding: 5px 7px; border: 2px solid gray; }",
        "QLabel { font-size: 15px; padding: 2px 3px; border: 1px solid gray; }",
    ):
        host.setStyleSheet(stylesheet)
        reference.ensurePolished()
        qtbot.wait(5)
        assert label.minimumHeight() == reference.sizeHint().height()
        positions = []
        for text in ("Short status", "First line\nSecond line", "Short again"):
            label.setText(text)
            qtbot.wait(5)
            positions.append(bar.y())
        assert positions[0] == positions[1] == positions[2]
        label.setText("First\nSecond\nThird")
        qtbot.wait(5)
        assert bar.y() > positions[0]
        label.setText("Short status")


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("width", [560, 1080])
@pytest.mark.parametrize("full_theme", [False, True])
def test_two_line_progress_reservation_survives_resize_and_font_change(
    qtbot, dark, width, full_theme
):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    for label in (panel.run_progress_label, panel.operation_progress_label):
        assert label.alignment() & Qt.AlignBottom
        assert label.alignment() & Qt.AlignLeft
    panel.setPalette(_palette(dark))
    if full_theme:
        from napari.qt import get_stylesheet

        panel.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    panel.begin_run(14)
    panel.resize(width, 900)
    panel.show()
    for points in (10, 14):
        panel.setFont(QFont("Segoe UI", points))
        positions = []
        for lines in ("Short status", "First line\nSecond line", "Short again"):
            panel.run_progress_label.setText(lines)
            panel.operation_progress_label.setText(lines)
            qtbot.wait(5)
            positions.append(
                (panel.run_progress_bar.y(), panel.operation_progress_bar.y())
            )
        assert positions[0] == positions[1] == positions[2]
        panel.run_progress_label.setText("First\nSecond\nThird")
        qtbot.wait(5)
        assert panel.run_progress_bar.y() > positions[0][0]
        assert (
            panel.run_progress_label.height()
            >= panel.run_progress_label.heightForWidth(panel.run_progress_label.width())
        )


def test_preparation_has_stage_byte_progress_and_cancel_without_starting_items(
    qtbot, tmp_path
):
    plan = _preview_result(tmp_path, count=14)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.begin_background_run_preparation()
    dialog.show_run_preparation_progress(
        BatchPreflightProgress(
            "checking",
            current=2,
            total=14,
            path=tmp_path / "large.czi",
            byte_current=40,
            byte_total=100,
        )
    )
    panel = dialog.results_panel
    assert dialog.tabs.currentIndex() == 3
    assert "2 of 14 files checked" in panel.operation_progress_label.text()
    assert "large.czi" in panel.operation_progress_label.text()
    assert panel.operation_progress_bar.value() == 40
    assert panel.run_progress_bar.value() == 0
    assert panel._active_index is None
    assert "Preparing" in panel.elapsed_label.text()
    assert dialog.cancel_run_button.text() == "Cancel preparation"
    assert dialog.cancel_run_button.isEnabled()
    cancelled = []
    dialog.cancelRequested.connect(lambda: cancelled.append(True))
    dialog.cancel_run_button.click()
    assert cancelled
    dialog.end_background_run_preparation()
    assert not dialog._checking_plan
    assert not panel._timer.isActive()
    assert dialog._preview_result is plan
    assert "Preparing run" not in panel.summary_label.text()


def test_processing_replaces_preparation_bar_units_and_cancel_resets_pending(qtbot):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.begin_run(2)
    panel.update_preparation(
        BatchPreflightProgress(
            "checking",
            byte_current=40,
            byte_total=100,
        )
    )
    panel.update_operation_progress(1, 2, "sample", "node", "Blur", 2, 5)
    assert panel.operation_progress_bar.text() == "2 / 5"
    panel.cancel_before_first_item()
    assert not panel._timer.isActive()
    assert not panel._running
    assert "no items started" in panel.summary_label.text()
