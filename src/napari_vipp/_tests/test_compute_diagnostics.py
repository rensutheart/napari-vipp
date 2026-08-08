from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from napari_vipp.core.compute import MemoryTopology
from napari_vipp.core.compute_diagnostics import (
    ComputeDoctorReport,
    DoctorStatus,
    PackageRecord,
    _repair_command,
    collect_compute_diagnostics,
    installed_gpu_packages,
    main,
)
from napari_vipp.core.compute_registry import (
    RuntimeDevice,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)


class _FakeRuntime:
    def __init__(self, *, available: bool = True) -> None:
        self.probe_calls = 0
        self.snapshot_calls = 0
        self.closed = False
        self.result = RuntimeProbeResult(
            runtime_id="cuda-cupy",
            available=available,
            version="14.1.1",
            devices=(RuntimeDevice("cuda:0", "Fake RTX", 4 * 1024**3),)
            if available
            else (),
            selected_device_id="cuda:0" if available else "",
            reason_code="available" if available else "cupy_missing",
            message="ready" if available else "CuPy is not installed.",
        )

    def probe(self, *, refresh=False):
        self.probe_calls += 1
        return self.result

    def memory_snapshot(self, *, device_id=""):
        self.snapshot_calls += 1
        return RuntimeMemorySnapshot(
            runtime_id="cuda-cupy",
            device_id=device_id,
            topology=MemoryTopology.DISCRETE,
            device_total_bytes=4 * 1024**3,
            device_free_bytes=3 * 1024**3,
        )

    def close(self):
        self.closed = True


def _doctor(runtime, **kwargs):
    return collect_compute_diagnostics(
        runtime=runtime,
        packages=(PackageRecord("cupy-cuda13x", "14.1.1"),),
        platform_name="win32",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
        **kwargs,
    )


def test_available_report_includes_probe_memory_and_is_json_safe():
    runtime = _FakeRuntime()

    report = _doctor(runtime)
    payload = json.loads(json.dumps(report.as_dict()))

    assert report.status is DoctorStatus.AVAILABLE
    assert report.available
    assert report.repair_command == ""
    assert payload["runtime_probe"]["devices"][0]["display_name"] == "Fake RTX"
    assert payload["memory_snapshot"]["device_free_bytes"] == 3 * 1024**3
    assert runtime.probe_calls == 1
    assert runtime.snapshot_calls == 1
    assert not runtime.closed  # injected runtimes remain caller-owned


def test_mixed_cupy_distributions_refuse_to_import_or_probe():
    runtime = _FakeRuntime()

    report = collect_compute_diagnostics(
        runtime=runtime,
        packages=(
            PackageRecord("cupy-cuda12x", "13"),
            PackageRecord("cupy_cuda13x", "14"),
        ),
        platform_name="win32",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
    )

    assert report.status is DoctorStatus.MISCONFIGURED
    assert report.reason_code == "mixed_cupy_distributions"
    assert runtime.probe_calls == 0
    assert "setup_gpu_dev.ps1" in report.repair_command
    assert "--venv" in report.repair_command
    assert ".venv-gpu-cu13-repair" in report.repair_command


def test_macos_is_cpu_only_and_does_not_offer_cuda_command():
    runtime = _FakeRuntime()

    report = collect_compute_diagnostics(
        runtime=runtime,
        packages=(),
        platform_name="darwin",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
    )

    assert report.status is DoctorStatus.UNSUPPORTED
    assert report.reason_code == "platform_unsupported"
    assert report.repair_command == ""
    assert runtime.probe_calls == 0


def test_requested_track_mismatch_is_actionable_without_probe():
    runtime = _FakeRuntime()

    report = _doctor(runtime, track="cuda12")

    assert report.status is DoctorStatus.MISCONFIGURED
    assert report.reason_code == "cupy_track_mismatch"
    assert "--track cuda12" in report.repair_command
    assert runtime.probe_calls == 0


def test_unavailable_runtime_is_a_structured_non_crashing_result():
    report = _doctor(_FakeRuntime(available=False))

    assert report.status is DoctorStatus.UNAVAILABLE
    assert report.reason_code == "cupy_missing"
    assert "setup_gpu_dev.ps1" in report.repair_command
    assert report.runtime_probe is not None


def test_installed_windows_repair_command_builds_launchable_exact_environment(
    monkeypatch,
):
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.importlib.metadata.version",
        lambda name: "0.13.0a3" if name == "napari-vipp" else "",
    )

    command = _repair_command("win32", "cuda13")

    assert "py -3.12 -m venv" in command
    assert "pip install --upgrade pip" in command
    assert 'pip install --pre "napari[pyqt6]>=0.6"' in command
    assert '"napari-vipp[gpu-cuda13]==0.13.0a3"' in command
    assert "compute_diagnostics --track cuda13" in command


def test_installed_linux_repair_command_builds_launchable_exact_environment(
    monkeypatch,
):
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.importlib.metadata.version",
        lambda name: "0.13.0a3" if name == "napari-vipp" else "",
    )

    command = _repair_command("linux", "cuda13")

    assert "python3.12 -m venv" in command
    assert "pip install --upgrade pip" in command
    assert 'pip install --pre "napari[pyqt6]>=0.6"' in command
    assert '"napari-vipp[gpu-cuda13]==0.13.0a3"' in command
    assert "compute_diagnostics --track cuda13" in command


def test_owned_runtime_cleanup_failure_is_not_reported_as_available(monkeypatch):
    class CleanupFailureRuntime(_FakeRuntime):
        def close(self):
            raise RuntimeError("private allocation escaped")

    runtime = CleanupFailureRuntime()
    monkeypatch.setattr(
        "napari_vipp.core.gpu.cupy_runtime.create_runtime",
        lambda: runtime,
    )

    report = collect_compute_diagnostics(
        packages=(PackageRecord("cupy-cuda13x", "14.1.1"),),
        platform_name="win32",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
    )

    assert report.status is DoctorStatus.UNAVAILABLE
    assert report.reason_code == "runtime_cleanup_failed"
    assert "cleanup" in report.summary.lower()
    assert report.details == (
        "Runtime cleanup failed: RuntimeError: private allocation escaped",
    )


def test_cli_supports_json_and_human_output(monkeypatch, capsys):
    report = ComputeDoctorReport(
        status=DoctorStatus.UNAVAILABLE,
        reason_code="cupy_missing",
        summary="CuPy is not installed.",
        platform="win32",
        execution_mode="native",
        python="CPython 3.12 (64-bit)",
        packages=(),
        track="cuda13",
        repair_command="setup command",
    )
    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.collect_compute_diagnostics",
        lambda **_kwargs: report,
    )

    assert main(["--json"]) == 2
    assert json.loads(capsys.readouterr().out)["reason_code"] == "cupy_missing"
    assert main([]) == 2
    assert "Suggested setup command" in capsys.readouterr().out


def test_cli_converts_unexpected_diagnostic_failure_to_json(monkeypatch, capsys):
    def fail(**_kwargs):
        raise ModuleNotFoundError("optional runtime")

    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.collect_compute_diagnostics",
        fail,
    )

    assert main(["--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason_code"] == "diagnostic_failed"
    assert payload["details"][0].startswith("ModuleNotFoundError")


def test_package_inventory_uses_metadata_without_imports(monkeypatch):
    class Distribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.importlib.metadata.distributions",
        lambda: (
            Distribution("CuPy_CUDA13x", "14.1.1"),
            Distribution("nvidia-cuda-nvrtc", "13.2"),
            Distribution("nvidia-nvimgcodec-cu13", "0.8.0.22"),
            Distribution("unrelated", "1"),
        ),
    )

    assert installed_gpu_packages() == (
        PackageRecord("cupy-cuda13x", "14.1.1"),
        PackageRecord("nvidia-cuda-nvrtc", "13.2"),
        PackageRecord("nvidia-nvimgcodec-cu13", "0.8.0.22"),
    )


def test_diagnostics_module_import_does_not_load_gpu_packages():
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    code = (
        "import sys; import napari_vipp.core.compute_diagnostics; "
        "assert not any(n == 'cupy' or n.startswith('cupyx') or "
        "n.startswith('cucim') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)
