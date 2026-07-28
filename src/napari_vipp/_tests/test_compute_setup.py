from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from napari_vipp.core.compute import MemoryTopology
from napari_vipp.core.compute_diagnostics import ComputeDoctorReport, DoctorStatus
from napari_vipp.core.compute_registry import (
    RuntimeDevice,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)
from napari_vipp.ui.compute_setup import (
    ComputeSetupAction,
    ComputeSetupActionKind,
    ComputeSetupState,
    ComputeSetupTone,
    HostMemorySnapshot,
    compute_setup_checking,
    compute_setup_not_checked,
    present_compute_setup,
)

GIB = 1024**3


def _report(
    *,
    status: DoctorStatus = DoctorStatus.AVAILABLE,
    platform: str = "win32",
    execution_mode: str = "native",
    summary: str = "CuPy GPU execution is available on Fake RTX.",
    reason_code: str = "cuda_available",
    repair_command: str = "",
    memory_snapshot: RuntimeMemorySnapshot | None = None,
) -> ComputeDoctorReport:
    available = status is DoctorStatus.AVAILABLE
    probe = RuntimeProbeResult(
        runtime_id="cuda-cupy",
        available=available,
        devices=(RuntimeDevice("cuda:0", "Fake RTX", 24 * GIB),) if available else (),
        selected_device_id="cuda:0" if available else "",
        reason_code=reason_code,
        message=summary,
    )
    return ComputeDoctorReport(
        status=status,
        reason_code=reason_code,
        summary=summary,
        platform=platform,
        execution_mode=execution_mode,
        python="CPython 3.12 (64-bit)",
        packages=(),
        track="cuda13",
        repair_command=repair_command,
        runtime_probe=probe,
        memory_snapshot=memory_snapshot,
    )


def test_initial_and_checking_states_are_nonblocking_action_metadata():
    initial = compute_setup_not_checked(platform_name="win32", track="cuda13")
    checking = compute_setup_checking(
        platform_name="linux",
        execution_mode="wsl2",
        track="cuda13",
    )

    assert initial.state is ComputeSetupState.NOT_CHECKED
    assert initial.actions == (
        ComputeSetupAction(
            "verify_compute_setup",
            ComputeSetupActionKind.VERIFY,
            "Verify GPU setup",
            track="cuda13",
            refresh_runtime=True,
        ),
    )
    assert not initial.actions[0].automatic
    assert checking.state is ComputeSetupState.CHECKING
    assert checking.busy
    assert checking.reason_code == "diagnostic_running"
    assert checking.track == "cuda13"
    assert checking.actions[0].kind is ComputeSetupActionKind.VERIFY
    assert not checking.actions[0].enabled
    assert checking.title == "NVIDIA GPU setup · WSL 2"
    assert "responsive" in checking.details[0]


def test_available_discrete_gpu_has_separate_ram_and_vram_rows():
    report = _report(
        memory_snapshot=RuntimeMemorySnapshot(
            runtime_id="cuda-cupy",
            device_id="cuda:0",
            topology=MemoryTopology.DISCRETE,
            device_total_bytes=24 * GIB,
            device_free_bytes=18 * GIB,
            runtime_live_bytes=1 * GIB,
            runtime_reserved_bytes=2 * GIB,
            out_of_pool_bytes=512 * 1024**2,
        )
    )

    presentation = present_compute_setup(
        report,
        host_memory=HostMemorySnapshot(64 * GIB, 40 * GIB),
    )

    assert presentation.state is ComputeSetupState.AVAILABLE
    assert presentation.tone is ComputeSetupTone.SUCCESS
    assert [row.key for row in presentation.memory_rows] == [
        "system_ram",
        "gpu_vram",
    ]
    assert presentation.memory_rows[0].value == "40.0 GiB available of 64.0 GiB"
    assert presentation.memory_rows[1].value == "18.0 GiB available of 24.0 GiB"
    assert "Fake RTX" in presentation.memory_rows[1].detail
    assert "2.0 GiB" in presentation.memory_rows[1].detail
    assert presentation.actions[0].kind is ComputeSetupActionKind.VERIFY
    assert presentation.actions[0].refresh_runtime


def test_unified_memory_is_one_budget_and_never_double_counted():
    report = _report(
        memory_snapshot=RuntimeMemorySnapshot(
            runtime_id="future-apple-runtime",
            device_id="apple:0",
            topology=MemoryTopology.UNIFIED,
            device_total_bytes=32 * GIB,
            device_free_bytes=10 * GIB,
        )
    )

    presentation = present_compute_setup(
        report,
        host_memory=HostMemorySnapshot(32 * GIB, 12 * GIB),
    )

    assert len(presentation.memory_rows) == 1
    row = presentation.memory_rows[0]
    assert row.key == "shared_memory"
    assert row.label == "Shared CPU/GPU memory"
    assert row.value == "12.0 GiB available of 32.0 GiB"
    assert "must not be added together" in row.detail


def test_macos_uses_honest_apple_gpu_not_enabled_wording_without_cuda_setup():
    report = _report(
        status=DoctorStatus.UNSUPPORTED,
        platform="darwin",
        summary="CUDA acceleration is unavailable on this platform.",
        reason_code="platform_unsupported",
    )

    presentation = present_compute_setup(
        report,
        host_memory=HostMemorySnapshot(32 * GIB, 20 * GIB),
    )

    assert presentation.state is ComputeSetupState.UNSUPPORTED
    assert presentation.tone is ComputeSetupTone.INFO
    assert presentation.title == "Compute setup · macOS"
    assert "Apple GPU acceleration is not enabled" in presentation.summary
    assert "CPU processing" in presentation.summary
    assert [action.kind for action in presentation.actions] == [
        ComputeSetupActionKind.VERIFY
    ]
    assert [row.key for row in presentation.memory_rows] == ["system_ram"]
    assert not presentation.actionable


def test_missing_runtime_offers_copy_only_command_then_explicit_verify():
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

    presentation = present_compute_setup(report)

    assert presentation.state is ComputeSetupState.UNAVAILABLE
    assert presentation.actionable
    copy_action, verify_action = presentation.actions
    assert copy_action.kind is ComputeSetupActionKind.COPY_COMMAND
    assert copy_action.command == command
    assert not copy_action.automatic
    assert verify_action.kind is ComputeSetupActionKind.VERIFY
    assert verify_action.command == ""
    assert verify_action.refresh_runtime
    assert not callable(copy_action)


def test_multiline_repair_command_is_hidden_instead_of_made_copyable():
    presentation = present_compute_setup(
        _report(
            status=DoctorStatus.MISCONFIGURED,
            summary="The GPU environment is inconsistent.",
            reason_code="mixed_cupy_distributions",
            repair_command="first command\nsecond command",
        )
    )

    assert presentation.state is ComputeSetupState.MISCONFIGURED
    assert presentation.tone is ComputeSetupTone.ERROR
    assert presentation.actionable
    assert [action.kind for action in presentation.actions] == [
        ComputeSetupActionKind.VERIFY
    ]
    assert "hidden" in presentation.details[-1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_bytes": -1}, "total_bytes"),
        ({"total_bytes": 10, "available_bytes": 11}, "must not exceed"),
    ],
)
def test_host_memory_snapshot_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HostMemorySnapshot(**kwargs)


def test_compute_setup_presentation_import_is_qt_and_gpu_package_free():
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    code = (
        "import sys; import napari_vipp.ui.compute_setup; "
        "assert 'qtpy' not in sys.modules; "
        "assert not any(n == 'cupy' or n.startswith('cupyx') or "
        "n.startswith('cucim') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)
