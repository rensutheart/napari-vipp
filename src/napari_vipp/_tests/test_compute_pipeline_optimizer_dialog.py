from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.compute import NodeComputePreference
from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationProposal,
    PipelineOptimizationRow,
    PipelineValidationWinner,
)
from napari_vipp.ui.compute_pipeline_optimizer_dialog import (
    PipelineOptimizerDialog,
    PipelineOptimizerWorker,
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
    assert "2 explicitly locked" in dialog.progress_label.text()


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
    assert dialog.cancel_button.isHidden()
    assert "could not be dispatched" in dialog.progress_label.text()


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
