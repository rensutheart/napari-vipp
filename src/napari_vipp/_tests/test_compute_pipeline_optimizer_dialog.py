from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QAbstractItemView, QApplication

from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    BenchmarkCandidateResult,
    NodeComputePreference,
)
from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationProposal,
    PipelineOptimizationRow,
    PipelineOptimizationSelectionBasis,
    PipelineOptimizationTimeoutReport,
    PipelineValidationWinner,
)
from napari_vipp.ui.compute_pipeline_optimizer_dialog import (
    PipelineOptimizerDialog,
    PipelineOptimizerProgress,
    PipelineOptimizerWorker,
    PipelineOptimizerWorkerOutcome,
    _candidate_timing_display,
    _candidate_timing_text,
    _scientific_check,
    _subtle_group_brush,
)


def test_find_fastest_dialog_has_one_unambiguous_search_scope(qtbot):
    dialog = PipelineOptimizerDialog(locked_node_count=2)
    qtbot.addWidget(dialog)

    with qtbot.waitSignal(dialog.analyze_requested):
        dialog.analyze_button.click()

    assert not hasattr(dialog, "override_authored_checkbox")
    assert "every scientifically eligible" in dialog.summary_label.text()
    assert dialog.summary_label.textFormat() == Qt.RichText
    assert dialog.summary_label.text().count("<li>") == 5
    for heading in ("Search:", "Locks:", "Manual nodes:", "Control:", "Timing:"):
        assert f"<b>{heading}</b>" in dialog.summary_label.text()
    assert "live cached results are not changed" in dialog.summary_label.text()
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
    dialog = PipelineOptimizerDialog(node_titles={"node-a": "Gaussian Blur"})
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

    assert dialog.result_table.rowCount() == 2
    assert dialog.result_table.item(0, 0).text() == "Gaussian Blur"
    assert dialog.result_table.item(0, 1).text() == "CPU — built in"
    assert "Exact implementation ID: cpu-a" in (
        dialog.result_table.item(0, 1).toolTip()
    )
    assert dialog.result_table.item(0, 4).text() == "Current · kept"
    assert dialog.result_table.item(1, 1).text() == "gpu-a"
    assert dialog.result_table.item(1, 4).text() == "Tested alternative"
    assert "current settings were faster" in dialog.result_label.text().lower()


def _candidate(
    implementation_id,
    *,
    warm,
    resident=(),
    transfer=(),
    host=(),
    cold=None,
    transfers_included=False,
    peak=0,
):
    return BenchmarkCandidateResult(
        implementation_id=implementation_id,
        parity_passed=True,
        cold_seconds=cold,
        warm_seconds=tuple(warm),
        peak_memory_bytes=peak,
        synchronized=True,
        transfers_included=transfers_included,
        warm_transfer_seconds=tuple(transfer),
        warm_resident_seconds=tuple(resident),
        warm_host_materialization_seconds=tuple(host),
    )


def _inconclusive_result(*, evidence):
    current = NodeComputePreference("cpu")
    row = PipelineOptimizationRow(
        "binary",
        "cpu-binary-threshold-v1",
        "cpu-binary-threshold-v1",
        current,
        current,
        False,
        True,
    )
    proposal = PipelineOptimizationProposal(
        "identity",
        "request",
        (("binary", "cpu-binary-threshold-v1"),),
        (row,),
        {"binary": current},
        0.0824,
        0.0819,
        0.0824,
        0.0819,
        0.99,
        True,
        15,
        0.98,
        PipelineValidationWinner.INCONCLUSIVE,
        (("binary", "cupy-binary-threshold-f32-exact-v1"),),
        PipelineOptimizationSelectionBasis.PAIRED_INCONCLUSIVE_RETAINED_CURRENT,
    )
    return SimpleNamespace(
        proposal=proposal,
        evidence={"binary": evidence},
        measured_node_ids=("binary",),
        reused_node_ids=(),
    )


def test_grouped_results_are_novice_readable_selectable_and_expandable(qtbot):
    cpu = _candidate(
        "cpu-binary-threshold-v1",
        warm=(0.0018, 0.0020, 0.0019),
        cold=0.004,
        peak=1024,
    )
    gpu = _candidate(
        "cupy-binary-threshold-f32-exact-v1",
        warm=(0.0013, 0.0014, 0.0012),
        resident=(0.0007, 0.0008, 0.0006),
        transfer=(0.0004, 0.0004, 0.0004),
        host=(0.0002, 0.0002, 0.0002),
        cold=0.025,
        transfers_included=True,
        peak=2 * 1024 * 1024,
    )
    evidence = SimpleNamespace(record=SimpleNamespace(candidates=(cpu, gpu)))
    dialog = PipelineOptimizerDialog(node_titles={"binary": "Binary Threshold"})
    qtbot.addWidget(dialog)

    dialog._on_finished(
        PipelineOptimizerWorkerOutcome(result=_inconclusive_result(evidence=evidence))
    )

    table = dialog.result_table
    assert table.rowCount() == 2
    assert table.columnCount() == 10
    assert table.selectionMode() == QAbstractItemView.ExtendedSelection
    assert table.item(0, 0).text() == "Binary Threshold"
    assert table.rowSpan(0, 0) == 2
    assert table.item(0, 1).text() == "CPU — built in"
    assert table.item(1, 1).text() == "GPU — CuPy"
    assert table.item(0, 3).text() == "Matches"
    assert table.item(0, 4).text() == "Current · kept"
    assert table.item(1, 4).text() == "Tested · no clear winner"
    assert dialog.result_label.text().startswith(
        "No clear winner—current settings kept."
    )
    assert "not large and certain enough to justify a change" in (
        dialog.result_label.text()
    )
    assert "82.4 ms" in dialog.result_label.text()
    assert "81.9 ms" in dialog.result_label.text()
    assert "15 paired rounds" in dialog.result_label.text()
    assert "greater than 5% or 10 ms" in dialog.result_label.toolTip()
    assert not dialog.apply_button.isEnabled()
    assert dialog.summary_label.isHidden()
    assert dialog.overall_progress_bar.isHidden()
    assert dialog.operation_progress_bar.isHidden()

    assert table.isColumnHidden(5)
    assert dialog.details_button.text() == "Show timing details"
    dialog.details_button.click()
    assert not table.isColumnHidden(5)
    assert table.item(1, 5).text() == "700.0 µs"
    assert table.item(1, 6).text() == "600.0 µs"
    assert table.item(1, 7).text() == "25.0 ms"
    assert table.item(1, 8).text() == "2.0 MiB"
    assert table.item(1, 9).text() == "3 rounds · measured now"
    assert dialog.details_button.text() == "Hide timing details"

    QApplication.clipboard().clear()
    table.item(0, 1).setSelected(True)
    table.item(0, 2).setSelected(True)
    table.copy_selection()
    assert QApplication.clipboard().text() == "CPU — built in\t1.9 ms"


def test_scientific_parity_failure_is_not_mislabeled_as_runtime_failure():
    parity_detail = "Maximum absolute difference exceeded the accepted tolerance."
    parity_failure = BenchmarkCandidateResult(
        implementation_id="cupy-parity-failure",
        parity_passed=False,
        cold_seconds=None,
        warm_seconds=(),
        error=parity_detail,
        failure_kind=BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY,
    )
    runtime_detail = "CUDA runtime could not launch the kernel."
    runtime_failure = BenchmarkCandidateResult(
        implementation_id="cupy-runtime-failure",
        parity_passed=False,
        cold_seconds=None,
        warm_seconds=(),
        error=runtime_detail,
        failure_kind=BenchmarkCandidateFailureKind.TRANSIENT_RUNTIME,
    )

    assert _scientific_check(parity_failure) == ("Did not match", parity_detail)
    assert _scientific_check(runtime_failure) == (
        "Could not verify",
        runtime_detail,
    )


def test_later_failed_retry_clears_inconclusive_decision_tooltip(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    evidence = SimpleNamespace(
        record=SimpleNamespace(
            candidates=(
                _candidate(
                    "cpu-binary-threshold-v1",
                    warm=(0.001,),
                ),
                _candidate(
                    "cupy-binary-threshold-f32-exact-v1",
                    warm=(0.0005,),
                ),
            )
        )
    )
    dialog._on_finished(
        PipelineOptimizerWorkerOutcome(result=_inconclusive_result(evidence=evidence))
    )
    assert "greater than 5% or 10 ms" in dialog.result_label.toolTip()

    dialog._on_finished(
        PipelineOptimizerWorkerOutcome(
            error="The optimizer runtime failed.",
            reason_code="optimizer_failed",
        )
    )

    assert dialog.result_label.toolTip() == ""
    assert dialog.result_label.text() == "The optimizer runtime failed."


def test_next_analysis_restores_intro_and_progress_after_result_review(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    evidence = SimpleNamespace(
        record=SimpleNamespace(
            candidates=(
                _candidate(
                    "cpu-binary-threshold-v1",
                    warm=(0.001,),
                ),
                _candidate(
                    "cupy-binary-threshold-f32-exact-v1",
                    warm=(0.0005,),
                ),
            )
        )
    )
    dialog._render_result(_inconclusive_result(evidence=evidence))
    assert dialog.summary_label.isHidden()

    worker = PipelineOptimizerWorker(lambda _cancelled, _progress: object())

    class CapturingPool:
        workers = []

        @classmethod
        def start(cls, queued_worker):
            cls.workers.append(queued_worker)

    dialog.start(worker, CapturingPool())

    assert not dialog.summary_label.isHidden()
    assert not dialog.overall_progress_bar.isHidden()
    assert not dialog.operation_progress_bar.isHidden()
    assert dialog.result_table.isHidden()
    assert dialog.details_button.isHidden()


def test_gpu_total_is_labeled_compute_only_without_transfers():
    display = _candidate_timing_display(
        _candidate(
            "cupy-binary-threshold-f32-exact-v1",
            warm=(0.001, 0.0012, 0.0011),
            resident=(0.0008, 0.0009, 0.0007),
            transfers_included=False,
        ),
        evidence_source="measured now",
    )

    assert display.total == "1.1 ms (compute only)"
    assert "rather than an additive whole-pipeline total" in display.total_tooltip
    assert display.data_movement == "—"


def test_fixed_cpu_only_node_without_benchmark_is_truthfully_not_measured(qtbot):
    cpu = NodeComputePreference("cpu")
    row = PipelineOptimizationRow(
        "writer",
        "cpu-batch-output-v1",
        "cpu-batch-output-v1",
        cpu,
        cpu,
        False,
        False,
    )
    proposal = PipelineOptimizationProposal(
        "identity",
        "request",
        (("writer", "cpu-batch-output-v1"),),
        (row,),
        {"writer": cpu},
        0.01,
        0.01,
        0.01,
        0.01,
        1.0,
        False,
        0,
        1.0,
        PipelineValidationWinner.CURRENT,
        (("writer", "cpu-batch-output-v1"),),
        PipelineOptimizationSelectionBasis.EXACT_MODEL_RETAINED_CURRENT,
    )
    dialog = PipelineOptimizerDialog(node_titles={"writer": "Batch Output"})
    qtbot.addWidget(dialog)

    dialog._render_result(SimpleNamespace(proposal=proposal, evidence={}))

    assert dialog.result_table.rowCount() == 1
    assert dialog.result_table.item(0, 0).text() == "Batch Output"
    assert dialog.result_table.item(0, 2).text() == "—"
    assert dialog.result_table.item(0, 3).text() == "Not measured"
    assert dialog.result_table.item(0, 4).text() == "Fixed"
    assert dialog.result_table.item(0, 9).text() == "Not measured"
    assert "saved exact evidence" not in (dialog.result_table.item(0, 9).text().lower())


def test_node_group_tint_is_deliberately_subtle(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    base = dialog.result_table.palette().base().color()
    tint = _subtle_group_brush(dialog.result_table).color()
    maximum_channel_delta = max(
        abs(base.red() - tint.red()),
        abs(base.green() - tint.green()),
        abs(base.blue() - tint.blue()),
    )

    assert 0 < maximum_channel_delta <= 16


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


def test_modeled_timing_text_labels_censored_lower_bound_as_non_exact():
    evidence = SimpleNamespace(
        record=SimpleNamespace(
            candidates=(
                SimpleNamespace(
                    implementation_id="cpu-a",
                    parity_passed=True,
                    error="",
                    timing_censored=True,
                    timing_lower_bound_seconds=10.75,
                    timing_censor_incumbent_id="gpu-a",
                    timing_censor_reason=(
                        "Stopped after the confidence-adjusted bound was exceeded."
                    ),
                ),
            )
        )
    )

    text = _candidate_timing_text(evidence)

    assert "cpu-a >10.750 s" in text
    assert "censored CPU/warm lower bound" in text
    assert "versus gpu-a" in text
    assert "median" not in text


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
