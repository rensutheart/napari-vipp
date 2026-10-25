"""Presentation tests use only synthetic state, never setup/hardware actions."""

from dataclasses import replace
from types import SimpleNamespace

from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication

from napari_vipp._tests.test_gpu_setup_dialog import _finish_next, _make
from napari_vipp.installer.frontend import InstallerScreen, ProgressUnit
from napari_vipp.ui import reader_setup
from napari_vipp.ui.setup_presentation import (
    SetupElapsed,
    StableSetupLog,
    measured_progress,
)


def test_measured_setup_counts_are_task_local_and_never_round_early_to_complete():
    state = SimpleNamespace(
        progress_stage="copying",
        progress_completed=999,
        progress_total=1000,
        progress_unit=ProgressUnit.FILES,
    )
    assert measured_progress(state) == "999 of 1,000 files · 99% of this task"
    state.progress_completed = 1000
    assert measured_progress(state).endswith("100% of this task")
    state.progress_completed = None
    assert "does not report" in measured_progress(state)
    state.progress_stage = "rolling_back"
    assert "has not completed" in measured_progress(state)


def test_setup_elapsed_freezes_total_without_rewriting_activity(qtbot, monkeypatch):
    from napari_vipp.ui import setup_presentation

    now = [100.0]
    monkeypatch.setattr(setup_presentation.time, "monotonic", lambda: now[0])
    label = SetupElapsed()
    qtbot.addWidget(label)
    label.update_timing(phase="packages", started=40.0)
    assert "Total elapsed: 1m 00s" in label.text()
    now[0] = 120.0
    label.refresh()
    assert "Current stage: 0m 20s" in label.text()
    label.update_timing(phase="complete", started=40.0, finished=125.0)
    now[0] = 500.0
    label.refresh()
    assert label.text() == "Total elapsed: 1m 25s"
    assert not label.timer.isActive()


def test_setup_output_preserves_manual_position_and_snapshot_is_stable(qtbot):
    output = StableSetupLog()
    qtbot.addWidget(output)
    output.resize(500, 220)
    output.show()
    output.appendPlainText("\n".join(f"Activity line {index}" for index in range(150)))
    QApplication.processEvents()
    output.verticalScrollBar().setValue(20)
    before = output.verticalScrollBar().value()
    output.appendPlainText("A new live event")
    assert output.verticalScrollBar().value() == before
    assert "A new live event" in output.toPlainText()
    output.set_snapshot(output.toPlainText())
    assert output.verticalScrollBar().value() == before
    output.follow_latest()
    assert output.verticalScrollBar().value() == output.verticalScrollBar().maximum()


def test_reader_review_keeps_exact_plan_separate_from_activity_and_requires_approval(
    qtbot, tmp_path, monkeypatch
):
    monkeypatch.setattr(reader_setup.setup, "setup_blocker", lambda: None)
    calls = []
    monkeypatch.setattr(
        reader_setup.setup, "install", lambda *_a: calls.append("install")
    )
    dialog = reader_setup.ReaderSetupDialog("tiff", tmp_path, 0, 0)
    qtbot.addWidget(dialog)
    dialog.details.appendPlainText("Original download/review activity")
    dialog._prepared(SimpleNamespace(packages=("reader==1.2", "codec==3.4")))
    assert dialog.package_plan.toPlainText() == "reader==1.2\ncodec==3.4"
    assert "2" in dialog.package_title.text()
    assert dialog.details.toPlainText() == "Original download/review activity"
    assert calls == []
    assert dialog.phase == "approve"
    dialog._action()
    assert dialog.phase == "waiting"
    assert "Step 2 of 3" in dialog.stage.text()
    assert calls == []
    dialog.timer.stop()


def test_gpu_progress_exposes_known_counts_and_keeps_console_out_of_diagnostics(
    qtbot, tmp_path
):
    dialog, backend, queue = _make(qtbot, tmp_path)
    _finish_next(queue)
    state = replace(
        dialog._controller.state,
        screen=InstallerScreen.WORKING,
        progress_stage="copying_dependencies",
        progress_unit=ProgressUnit.FILES,
        progress_completed=500,
        progress_total=2000,
        progress_fraction=0.25,
        installation_stage="installing",
        installation_started_at=1.0,
    )
    dialog._controller._state = state
    dialog._render(state)
    assert dialog.stage_label.text() == "Stage 3 of 6 · Packages"
    assert dialog.task_progress.text() == "500 of 2,000 files · 25% of this task"
    assert "Total elapsed:" in dialog.elapsed.text()
    snapshot = dialog.details.toPlainText()
    dialog._append_console("SYNTHETIC package output")
    assert dialog.details.toPlainText() == snapshot
    assert "SYNTHETIC package output" in dialog.activity.output.toPlainText()
    assert backend.applied == []
    assert backend.opened == []
    # End this synthetic busy state without executing or cancelling any worker.
    state = replace(state, screen=InstallerScreen.FAILED)
    dialog._controller._state = state
    dialog._render(state)


def test_reader_type_roles_follow_large_app_font(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(reader_setup.setup, "setup_blocker", lambda: None)
    dialog = reader_setup.ReaderSetupDialog("tiff", tmp_path, 0, 0)
    qtbot.addWidget(dialog)
    dialog.setFont(QFont("Segoe UI", 16))
    dialog.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert dialog.heading.font().pointSizeF() >= dialog.font().pointSizeF()
    assert dialog.description.font().pointSizeF() >= dialog.font().pointSizeF()
    assert dialog.action.isVisibleTo(dialog)
    assert dialog.cancel.isVisibleTo(dialog)


def test_original_input_details_keep_actions_outside_large_font_scroller(qtbot):
    from qtpy.QtCore import QPoint

    from napari_vipp.core.reproduction import ReproductionCheck, ReproductionRow
    from napari_vipp.ui.batch_reproduction_details import ReproductionCheckDetailsDialog

    check = ReproductionCheck(
        status="mismatch",
        can_run=False,
        recorded_vipp_version="old",
        current_vipp_version="new",
        version_override_used=False,
        rows=(ReproductionRow("source", 1, "changed", "Changed input"),),
        problems=("A long plain-text diagnostic. " * 15,),
    )
    dialog = ReproductionCheckDetailsDialog(check)
    qtbot.addWidget(dialog)
    dialog.setFont(QFont("Segoe UI", 16))
    dialog.resize(620, 430)
    dialog.show()
    QApplication.processEvents()
    QApplication.processEvents()
    for button in (dialog.review_button, dialog.close_button):
        position = button.mapTo(dialog, QPoint())
        assert position.y() + button.height() <= dialog.height()
    assert dialog.scroll_area.widget() is dialog.body
    assert dialog.check is check


def test_gpu_and_compute_headings_refresh_after_live_font_change(qtbot, tmp_path):
    from napari_vipp._tests.test_compute_setup_dialog import _CapturingThreadPool
    from napari_vipp.ui.compute_setup_dialog import ComputeSetupDialog

    gpu, _backend, queue = _make(qtbot, tmp_path)
    _finish_next(queue)
    compute = ComputeSetupDialog(
        thread_pool=_CapturingThreadPool(),
        host_memory_provider=lambda: None,
    )
    qtbot.addWidget(compute)
    for dialog, heading in ((gpu, gpu.heading), (compute, compute.status_heading)):
        dialog.show()
        dialog.setFont(QFont("Segoe UI", 16))
        QApplication.processEvents()
        QApplication.processEvents()
        assert heading.font().pointSizeF() >= 16
    assert not compute.checking
