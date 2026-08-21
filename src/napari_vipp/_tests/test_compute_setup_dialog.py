from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QFormLayout

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    ExecutionReport,
    MemoryTopology,
)
from napari_vipp.core.compute_diagnostics import ComputeDoctorReport, DoctorStatus
from napari_vipp.core.compute_registry import (
    RuntimeDevice,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)
from napari_vipp.ui.compute_setup import (
    ComputeDeviceOption,
    ComputeSetupState,
    ComputeSetupTone,
    HostMemorySnapshot,
)
from napari_vipp.ui.compute_setup_dialog import (
    ComputeSetupCheckResult,
    ComputeSetupDialog,
)

GIB = 1024**3


class _CapturingThreadPool:
    def __init__(self) -> None:
        self.workers = []

    def start(self, worker) -> None:
        self.workers.append(worker)


def _report(
    *,
    status: DoctorStatus = DoctorStatus.AVAILABLE,
    summary: str = "CuPy GPU execution is available on Test RTX.",
    reason_code: str = "cuda_available",
    repair_command: str = "",
    memory_snapshot: RuntimeMemorySnapshot | None = None,
    runtime_id: str = "cuda-cupy",
    devices: tuple[RuntimeDevice, ...] | None = None,
    selected_device_id: str | None = None,
) -> ComputeDoctorReport:
    available = status is DoctorStatus.AVAILABLE
    if devices is None:
        devices = (RuntimeDevice("cuda:0", "Test RTX", 24 * GIB),) if available else ()
    if selected_device_id is None:
        selected_device_id = devices[0].device_id if available and devices else ""
    return ComputeDoctorReport(
        status=status,
        reason_code=reason_code,
        summary=summary,
        platform="win32",
        execution_mode="native",
        python="CPython 3.12 (64-bit)",
        packages=(),
        track="cuda13",
        repair_command=repair_command,
        runtime_probe=RuntimeProbeResult(
            runtime_id=runtime_id,
            available=available,
            devices=devices,
            selected_device_id=selected_device_id,
            reason_code=reason_code,
            message=summary,
        ),
        memory_snapshot=memory_snapshot,
    )


def _dialog(
    qtbot,
    *,
    doctor: Callable[..., ComputeDoctorReport],
    host_memory: HostMemorySnapshot | None = None,
    recent_execution_provider=None,
    support_writer=None,
) -> tuple[ComputeSetupDialog, _CapturingThreadPool]:
    pool = _CapturingThreadPool()
    kwargs = {}
    if support_writer is not None:
        kwargs["support_writer"] = support_writer
    dialog = ComputeSetupDialog(
        thread_pool=pool,
        doctor=doctor,
        host_memory_provider=(lambda: host_memory),
        recent_execution_provider=recent_execution_provider,
        **kwargs,
    )
    qtbot.addWidget(dialog)
    return dialog, pool


def _form_rows(form: QFormLayout) -> list[tuple[str, str, str]]:
    rows = []
    for index in range(form.rowCount()):
        label_item = form.itemAt(index, QFormLayout.LabelRole)
        field_item = form.itemAt(index, QFormLayout.FieldRole)
        assert label_item is not None
        assert field_item is not None
        label = label_item.widget()
        field = field_item.widget()
        assert label is not None
        assert field is not None
        rows.append((label.text(), field.text(), field.toolTip()))
    return rows


def _device_options(dialog: ComputeSetupDialog) -> list[ComputeDeviceOption]:
    options = []
    for index in range(dialog.device_combo.count()):
        option = dialog.device_combo.itemData(index)
        assert isinstance(option, ComputeDeviceOption)
        options.append(option)
    return options


def test_dialog_initial_state_does_not_probe_or_offer_a_command(qtbot):
    calls = []

    def doctor(**kwargs):
        calls.append(kwargs)
        return _report()

    dialog, pool = _dialog(
        qtbot,
        doctor=doctor,
        host_memory=HostMemorySnapshot(64 * GIB, 40 * GIB),
    )

    assert dialog.presentation.state is ComputeSetupState.NOT_CHECKED
    assert not dialog.checking
    assert dialog.verify_button.isEnabled()
    assert dialog.verify_button.text() == "Verify GPU setup"
    assert dialog.device_combo.count() == 1
    assert dialog.device_combo.itemText(0) == "Automatic (runtime default)"
    assert dialog.device_selection == ComputeDeviceOption(
        "", "", "Automatic (runtime default)"
    )
    assert dialog.progress.isHidden()
    assert dialog.command_edit.isHidden()
    assert dialog.copy_button.isHidden()
    assert dialog.advanced_widget.isHidden()
    assert not dialog.export_button.isEnabled()
    assert _form_rows(dialog.memory_form) == [
        (
            "System RAM",
            "40.0 GiB available of 64.0 GiB",
            "Memory available to VIPP and other host applications.",
        )
    ]
    assert calls == []
    assert pool.workers == []


def test_verified_devices_follow_report_order_and_emit_exact_user_selection(qtbot):
    devices = (
        RuntimeDevice("accelerator:4", "First accelerator", 12 * GIB),
        RuntimeDevice("accelerator:9", "Second accelerator", 48 * GIB),
    )
    report = _report(
        runtime_id="future-runtime",
        devices=devices,
        selected_device_id="accelerator:9",
    )
    dialog, pool = _dialog(qtbot, doctor=lambda **_kwargs: report)
    emitted = []
    dialog.device_selection_changed.connect(emitted.append)

    dialog.verify()
    pool.workers[0].run()

    expected = [
        ComputeDeviceOption("", "", "Automatic (runtime default)"),
        ComputeDeviceOption(
            "future-runtime",
            "accelerator:4",
            "First accelerator",
            12 * GIB,
        ),
        ComputeDeviceOption(
            "future-runtime",
            "accelerator:9",
            "Second accelerator",
            48 * GIB,
        ),
    ]
    assert _device_options(dialog) == expected
    assert dialog.presentation.default_runtime_id == "future-runtime"
    assert dialog.presentation.default_device_id == "accelerator:9"
    assert dialog.device_selection == expected[0]
    assert emitted == []

    dialog.device_combo.setCurrentIndex(2)

    assert dialog.device_selection == expected[2]
    assert emitted == [expected[2]]


def test_explicit_stale_device_is_preserved_as_unavailable_until_it_returns(qtbot):
    first = _report()
    second = _report(
        devices=(
            RuntimeDevice("cuda:0", "Primary RTX", 24 * GIB),
            RuntimeDevice("cuda:1", "Returned RTX", 16 * GIB),
        ),
        selected_device_id="cuda:0",
    )
    responses = iter((first, second))
    dialog, pool = _dialog(qtbot, doctor=lambda **_kwargs: next(responses))
    emitted = []
    dialog.device_selection_changed.connect(emitted.append)

    dialog.set_device_selection("cuda-cupy", "cuda:1", "Saved RTX")

    assert dialog.device_selection == ComputeDeviceOption(
        "cuda-cupy",
        "cuda:1",
        "Saved RTX",
        available=False,
    )
    assert dialog.device_combo.currentText().endswith("— Unavailable")
    assert emitted == []

    dialog.verify()
    assert not dialog.device_combo.isEnabled()
    assert dialog.device_selection.device_id == "cuda:1"
    pool.workers[0].run()

    assert dialog.device_selection.device_id == "cuda:1"
    assert not dialog.device_selection.available
    assert dialog.device_combo.currentText().endswith("— Unavailable")

    dialog.verify()
    assert dialog.device_selection.device_id == "cuda:1"
    pool.workers[1].run()

    assert dialog.device_selection == ComputeDeviceOption(
        "cuda-cupy",
        "cuda:1",
        "Returned RTX",
        16 * GIB,
    )
    assert "Unavailable" not in dialog.device_combo.currentText()
    assert emitted == []


def test_device_selection_editability_is_stable_and_setter_requires_exact_ids(
    qtbot,
):
    dialog, pool = _dialog(qtbot, doctor=lambda **_kwargs: _report())

    dialog.set_device_selection_editable(False)
    assert not dialog.device_combo.isEnabled()
    dialog.set_device_selection()
    assert not dialog.device_combo.isEnabled()
    dialog.verify()
    pool.workers[0].run()
    assert not dialog.device_combo.isEnabled()

    dialog.set_device_selection_editable(True)
    assert dialog.device_combo.isEnabled()
    with pytest.raises(ValueError, match="both be set or both be blank"):
        dialog.set_device_selection("cuda-cupy", "")
    with pytest.raises(TypeError, match="boolean"):
        dialog.set_device_selection_editable(1)


def test_verification_is_queued_once_and_repeated_calls_do_not_overlap(qtbot):
    calls = []

    def doctor(**kwargs):
        calls.append(kwargs)
        return _report()

    dialog, pool = _dialog(qtbot, doctor=doctor)
    dialog.track_combo.setCurrentIndex(dialog.track_combo.findData("cuda13"))

    dialog.verify()
    dialog.verify()

    assert dialog.checking
    assert dialog.presentation.state is ComputeSetupState.CHECKING
    assert dialog.progress.isVisibleTo(dialog)
    assert not dialog.verify_button.isEnabled()
    assert not dialog.track_combo.isEnabled()
    assert not dialog.device_combo.isEnabled()
    assert len(pool.workers) == 1
    assert calls == []  # starting verification never runs the doctor inline

    pool.workers[0].run()

    assert not dialog.checking
    assert calls == [{"track": "cuda13", "refresh": True}]
    assert dialog.device_combo.isEnabled()


def test_success_renders_separate_ram_and_vram_rows(qtbot):
    report = _report(
        memory_snapshot=RuntimeMemorySnapshot(
            runtime_id="cuda-cupy",
            device_id="cuda:0",
            topology=MemoryTopology.DISCRETE,
            device_total_bytes=24 * GIB,
            device_free_bytes=18 * GIB,
            runtime_live_bytes=1 * GIB,
            runtime_reserved_bytes=2 * GIB,
        )
    )
    dialog, pool = _dialog(
        qtbot,
        doctor=lambda **_kwargs: report,
        host_memory=HostMemorySnapshot(64 * GIB, 40 * GIB),
    )

    dialog.verify()
    pool.workers[0].run()

    assert dialog.presentation.state is ComputeSetupState.AVAILABLE
    assert dialog.presentation.tone is ComputeSetupTone.SUCCESS
    assert dialog.summary_label.text() == report.summary
    assert dialog.verify_button.text() == "Verify again"
    assert dialog.verify_button.isEnabled()
    assert _form_rows(dialog.memory_form) == [
        (
            "System RAM",
            "40.0 GiB available of 64.0 GiB",
            "Memory available to VIPP and other host applications.",
        ),
        (
            "GPU VRAM",
            "18.0 GiB available of 24.0 GiB",
            "Test RTX; VIPP runtime reserved 2.0 GiB; 1.0 GiB live",
        ),
    ]
    assert dialog.command_edit.isHidden()
    assert dialog.copy_button.isHidden()
    assert dialog.export_button.isEnabled()
    assert _form_rows(dialog.check_form) == [
        (
            "CUDA and GPU",
            "Ready",
            "CuPy GPU execution is available on Test RTX.",
        ),
        (
            "Optional cuCIM",
            "Not checked (optional)",
            "No cuCIM probe result was recorded.",
        ),
        (
            "VIPP GPU coverage",
            "No reviewed regions available",
            "Only reviewed combinations are offered automatically; CPU remains safe.",
        ),
    ]


def test_advanced_details_start_collapsed_and_support_export_is_explicit(
    qtbot,
    monkeypatch,
    tmp_path,
):
    report = _report()
    recent = ExecutionReport(ComputeRequest(), ComputeEnvironment())
    calls = []

    def writer(path, written_report, *, recent_execution=None):
        calls.append((Path(path), written_report, recent_execution))
        return Path(path)

    dialog, pool = _dialog(
        qtbot,
        doctor=lambda **_kwargs: report,
        recent_execution_provider=lambda: recent,
        support_writer=writer,
    )
    dialog.verify()
    pool.workers[0].run()
    target = tmp_path / "doctor-support"
    monkeypatch.setattr(
        "napari_vipp.ui.compute_setup_dialog.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), "JSON files (*.json)"),
    )

    assert dialog.advanced_widget.isHidden()
    dialog.advanced_button.click()
    assert dialog.advanced_widget.isVisibleTo(dialog)
    dialog.export_button.click()

    assert calls == [(target.with_suffix(".json"), report, recent)]
    assert "privacy-redacted" in dialog.save_status_label.text()


def test_setup_command_is_copy_only_and_never_starts_work(qtbot):
    command = (
        'python -m pip install "napari-vipp[gpu-cuda13]"; '
        "vipp-compute-doctor --track cuda13"
    )
    report = _report(
        status=DoctorStatus.UNAVAILABLE,
        summary="CuPy is not installed.",
        reason_code="cupy_missing",
        repair_command=command,
    )
    doctor_calls = []

    def doctor(**kwargs):
        doctor_calls.append(kwargs)
        return report

    dialog, pool = _dialog(qtbot, doctor=doctor)
    dialog.verify()
    pool.workers[0].run()
    worker_count = len(pool.workers)
    call_count = len(doctor_calls)
    clipboard = QApplication.clipboard()
    clipboard.clear()

    dialog.copy_button.click()

    assert dialog.command_edit.text() == command
    assert not dialog.command_edit.isHidden()
    assert not dialog.copy_button.isHidden()
    assert clipboard.text() == command
    assert len(pool.workers) == worker_count
    assert len(doctor_calls) == call_count


def test_stale_result_cannot_replace_a_newer_check(qtbot):
    first = _report(summary="First result")
    second = _report(summary="Second result")
    responses = iter((first, second))
    dialog, pool = _dialog(qtbot, doctor=lambda **_kwargs: next(responses))

    dialog.verify()
    pool.workers[0].run()
    assert dialog.summary_label.text() == "First result"

    dialog.verify()
    assert dialog.checking
    assert dialog.presentation.state is ComputeSetupState.CHECKING
    dialog._on_check_finished(ComputeSetupCheckResult(1, first))

    assert dialog.checking
    assert dialog.presentation.state is ComputeSetupState.CHECKING
    assert dialog._last_report is first

    pool.workers[1].run()
    assert not dialog.checking
    assert dialog.summary_label.text() == "Second result"
    assert dialog._last_report is second


def test_worker_failure_becomes_a_terminal_retryable_result(qtbot):
    def fail(**_kwargs):
        raise RuntimeError("driver probe exploded")

    dialog, pool = _dialog(qtbot, doctor=fail)

    dialog.verify()
    pool.workers[0].run()

    assert not dialog.checking
    assert dialog.presentation.state is ComputeSetupState.UNAVAILABLE
    assert dialog.presentation.reason_code == "diagnostic_worker_failed"
    assert "RuntimeError: driver probe exploded" in dialog.summary_label.text()
    assert dialog.verify_button.text() == "Verify GPU setup"
    assert dialog.verify_button.isEnabled()
    assert dialog._last_report is not None
    assert dialog._last_report.reason_code == "diagnostic_worker_failed"
