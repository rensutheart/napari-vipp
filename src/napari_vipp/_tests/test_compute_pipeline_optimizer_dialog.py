from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.compute import NodeComputePreference
from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationProposal,
    PipelineOptimizationRow,
    PipelineOptimizationTimeoutReport,
    PipelineValidationWinner,
)
from napari_vipp.ui.compute_pipeline_optimizer_dialog import (
    PipelineOptimizerDialog,
    PipelineOptimizerProgress,
    PipelineOptimizerWorker,
    PipelineOptimizerWorkerOutcome,
    _candidate_timing_text,
)


def test_find_fastest_dialog_has_one_unambiguous_search_scope(qtbot):
    dialog = PipelineOptimizerDialog(locked_node_count=2)
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(dialog.analyze_requested):
        dialog.analyze_button.click()

    assert not hasattr(dialog, "override_authored_checkbox")
    assert "every scientifically eligible" in dialog.summary_label.text()
    assert dialog.summary_label.textFormat() == Qt.RichText
    assert dialog.summary_label.text().count("<li>") == 4
    for heading in ("Search:", "Locks:", "Control:", "Timing:"):
        assert f"<b>{heading}</b>" in dialog.summary_label.text()
    assert "2 explicitly locked nodes" in dialog.progress_label.text()

    singular_dialog = PipelineOptimizerDialog(locked_node_count=1)
    qtbot.addWidget(singular_dialog)
    assert "1 explicitly locked node will" in singular_dialog.progress_label.text()


def test_dialog_exposes_two_progress_channels_and_selectable_time_limit(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.overall_progress_bar is not dialog.operation_progress_bar
    assert dialog.overall_progress_label is not dialog.operation_progress_label
    assert dialog.overall_progress_bar.accessibleName() == (
        "Overall pipeline optimization progress"
    )
    assert dialog.operation_progress_bar.accessibleName() == (
        "Current operation benchmark progress"
    )
    assert dialog.time_limit_combo.isVisible()
    assert dialog.time_budget_seconds == pytest.approx(300.0)
    assert "wall-clock" in dialog.time_limit_combo.toolTip()
    assert "RAM or VRAM" in dialog.time_limit_combo.toolTip()

    dialog.time_limit_combo.setCurrentIndex(2)

    assert dialog.time_limit_combo.currentText() == "30 minutes"
    assert dialog.time_budget_seconds == pytest.approx(1_800.0)


def test_dialog_updates_overall_and_current_operation_progress_independently(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)

    dialog._on_progress(
        PipelineOptimizerProgress(
            completed=2,
            total=9,
            message="Benchmarking Subtract Background (1/3).",
            operation_completed=37,
            operation_total=171,
            operation_message="CPU: Rolling-ball YX plane (37 of 171).",
            node_id="subtract-background",
            node_title="Subtract Background",
            implementation_id="cpu-subtract_background-v1",
            measurement_phase="parity_cold",
        )
    )

    assert dialog.overall_progress_bar.minimum() == 0
    assert dialog.overall_progress_bar.maximum() == 9
    assert dialog.overall_progress_bar.value() == 2
    assert "Benchmarking Subtract Background" in dialog.overall_progress_label.text()
    assert dialog.operation_progress_bar.minimum() == 0
    assert dialog.operation_progress_bar.maximum() == 171
    assert dialog.operation_progress_bar.value() == 37
    assert "CPU: Rolling-ball YX plane (37 of 171)" in (
        dialog.operation_progress_label.text()
    )

    dialog._on_progress(
        PipelineOptimizerProgress(
            completed=3,
            total=9,
            message="Subtract Background benchmark complete.",
            operation_completed=10,
            operation_total=10,
            operation_message="Subtract Background benchmark complete.",
        )
    )

    assert dialog.overall_progress_bar.value() == 3
    assert dialog.operation_progress_bar.value() == 10


def test_worker_preserves_structured_timeout_report(qtbot):
    report = PipelineOptimizationTimeoutReport(
        stage="node-benchmark",
        stage_message="Benchmarking Subtract Background",
        elapsed_seconds=300.0,
        budget_seconds=300.0,
        overall_completed=2,
        overall_total=9,
        node_id="subtract-background",
        node_title="Subtract Background",
        node_index=1,
        node_total=3,
        operation_completed=4,
        operation_total=10,
        operation_message="CuPy warm timing round 2/5",
        completed_node_ids=("extract-channel",),
        reused_node_ids=("extract-channel",),
        baseline_completed=True,
        partial_node_discarded=True,
    )

    def time_out(_cancelled, _progress):
        raise PipelineOptimizationDeadlineExceeded(
            "Analysis timed out while benchmarking Subtract Background.",
            report=report,
        )

    worker = PipelineOptimizerWorker(time_out)
    with qtbot.waitSignal(worker.signals.finished) as blocker:
        worker.run()

    outcome = blocker.args[0]
    assert outcome.reason_code == "deadline_exceeded"
    assert outcome.timeout_report is report
    assert outcome.result is None


def test_dialog_timeout_explains_inconclusive_result_and_safe_retry(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    dialog._on_progress(
        PipelineOptimizerProgress(
            completed=2,
            total=9,
            message="Benchmarking Subtract Background (1/3).",
            operation_completed=4,
            operation_total=10,
            operation_message="CuPy warm timing round 2/5.",
        )
    )
    report = PipelineOptimizationTimeoutReport(
        stage="node-benchmark",
        stage_message="Benchmarking Subtract Background",
        elapsed_seconds=300.0,
        budget_seconds=300.0,
        overall_completed=2,
        overall_total=9,
        node_id="subtract-background",
        node_title="Subtract Background",
        node_index=1,
        node_total=3,
        operation_completed=4,
        operation_total=10,
        operation_message="CuPy warm timing round 2/5",
        completed_node_ids=("extract-channel",),
        reused_node_ids=("extract-channel",),
        baseline_completed=True,
        partial_node_discarded=True,
    )

    dialog._on_finished(
        PipelineOptimizerWorkerOutcome(
            error="Analysis timed out while benchmarking Subtract Background.",
            reason_code="deadline_exceeded",
            timeout_report=report,
        )
    )

    rendered = dialog.result_label.text().lower()
    assert dialog.result_label.textFormat() == Qt.RichText
    assert "subtract background" in rendered
    assert "no fastest assignment" in rendered
    assert "not proven fastest" in rendered
    assert "no settings changed" in rendered
    assert "retry" in rendered
    assert "longer" in rendered
    assert "partial" in rendered and "discard" in rendered
    assert dialog.overall_progress_bar.value() == 2
    assert dialog.operation_progress_bar.value() == 4
    assert dialog.time_limit_combo.isEnabled()


def test_dialog_rolls_back_running_state_when_worker_dispatch_fails(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    worker = PipelineOptimizerWorker(lambda _cancelled, _progress: object())

    class FailingPool:
        @staticmethod
        def start(_worker):
            raise RuntimeError("thread pool rejected worker")

    with pytest.raises(RuntimeError, match="rejected"):
        dialog.start(worker, FailingPool())

    assert not dialog.running
    assert dialog._worker is None
    assert dialog.analyze_button.isEnabled()
    assert dialog.close_button.isEnabled()
    assert dialog.time_limit_combo.isEnabled()
    assert dialog.cancel_button.isHidden()
    assert "could not be dispatched" in dialog.overall_progress_label.text()
    assert dialog.overall_progress_bar.minimum() == 0
    assert dialog.overall_progress_bar.maximum() == 1
    assert dialog.overall_progress_bar.value() == 0
    assert "no benchmark was started" in dialog.operation_progress_label.text()
    assert dialog.operation_progress_bar.minimum() == 0
    assert dialog.operation_progress_bar.maximum() == 1
    assert dialog.operation_progress_bar.value() == 0


def test_current_validation_winner_keeps_tested_alternative_visible(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    cpu = NodeComputePreference("cpu")
    row = PipelineOptimizationRow(
        "node-a",
        "cpu-a",
        "cpu-a",
        NodeComputePreference(),
        cpu,
        False,
        True,
    )
    proposal = PipelineOptimizationProposal(
        "identity",
        "request",
        (("node-a", "cpu-a"),),
        (row,),
        {"node-a": cpu},
        10.0,
        1.0,
        1.0,
        1.2,
        0.8,
        True,
        5,
        1.1,
        PipelineValidationWinner.CURRENT,
        (("node-a", "gpu-a"),),
    )

    dialog._render_result(SimpleNamespace(proposal=proposal, evidence={}))

    assert dialog.result_table.item(0, 2).text() == "gpu-a"
    assert dialog.result_table.item(0, 3).text() == "cpu-a"
    assert "Current won final validation" in dialog.result_table.item(0, 6).text()
    assert "current assignment won" in dialog.result_label.text().lower()


def test_modeled_timing_text_prefers_gpu_resident_series():
    evidence = SimpleNamespace(
        record=SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    implementation_id="gpu-a",
                    parity_passed=True,
                    error="",
                    warm_seconds=(1.0, 1.0, 1.0),
                    warm_resident_seconds=(0.2, 0.2, 0.2),
                ),
            )
        )
    )

    text = _candidate_timing_text(evidence)

    assert "200.0 ms" in text
    assert "GPU resident" in text
    assert "1.000 s" not in text


def test_shutdown_terminates_queued_worker_and_ignores_late_finish(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    worker = PipelineOptimizerWorker(lambda _cancelled, _progress: object())

    class CapturingPool:
        workers = []

        @classmethod
        def start(cls, queued_worker):
            cls.workers.append(queued_worker)

    dialog.start(worker, CapturingPool())
    dialog.shutdown()

    assert worker.cancel_event.is_set()
    assert not dialog.running
    assert not dialog.isVisible()
    assert dialog.windowModality() is Qt.NonModal

    worker.signals.finished.emit(
        SimpleNamespace(result=object(), error="", reason_code="")
    )

    assert dialog.outcome is None
    assert not dialog.apply_button.isEnabled()
