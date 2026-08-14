from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionReport,
    MemoryTopology,
    NodeComputePreference,
    NodeExecutionDecision,
)
from napari_vipp.core.compute_diagnostics import (
    ComputeDoctorReport,
    DoctorStatus,
    PackageRecord,
    _repair_command,
    build_compute_support_bundle,
    collect_compute_diagnostics,
    installed_gpu_packages,
    main,
    write_compute_support_bundle,
)
from napari_vipp.core.compute_policy import (
    PHASE1_CUCIM_BUILD_RECIPE_ID,
    PHASE1_CUCIM_SOURCE_COMMIT,
    PHASE1_CUCIM_SOURCE_TAG,
    PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
)
from napari_vipp.core.compute_registry import (
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import accelerator_compute_specs

_SCIENTIFIC_STACK = (
    ("numpy", "2.5.1"),
    ("scipy", "1.18.0"),
    ("scikit-image", "0.26.0"),
)


def _public_region_count(*library_ids: str) -> int:
    libraries = set(library_ids)
    return sum(
        spec.visible_for(allow_experimental=False)
        and spec.implementation_library_id in libraries
        for spec in accelerator_compute_specs()
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
            devices=(
                RuntimeDevice(
                    "cuda:0",
                    "Fake RTX",
                    4 * 1024**3,
                    metadata=(("compute_capability", "8.9"),),
                ),
            )
            if available
            else (),
            selected_device_id="cuda:0" if available else "",
            reason_code="available" if available else "cupy_missing",
            message="ready" if available else "CuPy is not installed.",
            environment_fingerprint="safe-test-fingerprint" if available else "",
            metadata=(
                ("driver_version", "13030"),
                ("cuda_runtime_version", "13020"),
            )
            if available
            else (),
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
    packages = kwargs.pop(
        "packages",
        (PackageRecord("cupy-cuda13x", "14.1.1"),),
    )
    scientific_stack = kwargs.pop(
        "scientific_stack_versions",
        _SCIENTIFIC_STACK,
    )
    return collect_compute_diagnostics(
        runtime=runtime,
        packages=packages,
        platform_name="win32",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
        scientific_stack_versions=scientific_stack,
        **kwargs,
    )


def _library_probes(*, cucim: bool = False, cupyx: bool = True):
    probes = [
        ImplementationLibraryProbeResult("cupy", True, version="14.1.1"),
        ImplementationLibraryProbeResult(
            "cupyx",
            cupyx,
            version="14.1.1" if cupyx else "",
            reason_code="" if cupyx else "cupyx_broken",
            message="ready" if cupyx else "CuPyX could not run its primitives.",
        ),
    ]
    if cucim:
        probes.append(
            ImplementationLibraryProbeResult(
                "cucim",
                True,
                version="26.6.0",
                metadata=(
                    ("environment_record_schema", "napari-vipp-gpu-environment"),
                    ("environment_record_schema_version", "2"),
                    ("environment_track", "cuda13"),
                    ("cupy_distribution", "cupy-cuda13x"),
                    ("cucim_distribution", "cucim-cu13"),
                    ("cucim_distribution_version", "26.6.0"),
                    (
                        "cucim_wheel_payload_sha256",
                        PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
                    ),
                    ("cucim_artifact_sha256", "a" * 64),
                    ("cucim_source_tag", PHASE1_CUCIM_SOURCE_TAG),
                    ("cucim_source_commit", PHASE1_CUCIM_SOURCE_COMMIT),
                    ("cucim_build_recipe_id", PHASE1_CUCIM_BUILD_RECIPE_ID),
                ),
            )
        )
    else:
        probes.append(
            ImplementationLibraryProbeResult(
                "cucim",
                False,
                reason_code="cucim_not_installed",
                message="The optional cuCIM add-on is not installed.",
            )
        )
    return tuple(probes)


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
    assert len(report.admission_regions) == len(accelerator_compute_specs())
    assert len(report.admitted_regions) == _public_region_count("cupy", "cupyx")
    assert report.guidance is not None
    assert report.guidance.optional


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
        lambda name: "0.13.0a4" if name == "napari-vipp" else "",
    )

    command = _repair_command("win32", "cuda13")

    assert "py -3.12 -m venv" in command
    assert "pip install --upgrade pip" in command
    assert 'pip install --pre "napari[pyqt6]>=0.6"' in command
    assert '"napari-vipp[gpu-cuda13]==0.13.0a4"' in command
    assert "compute_diagnostics --track cuda13" in command


def test_installed_linux_repair_command_builds_launchable_exact_environment(
    monkeypatch,
):
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.importlib.metadata.version",
        lambda name: "0.13.0a4" if name == "napari-vipp" else "",
    )

    command = _repair_command("linux", "cuda13")

    assert "python3.12 -m venv" in command
    assert "pip install --upgrade pip" in command
    assert 'pip install --pre "napari[pyqt6]>=0.6"' in command
    assert '"napari-vipp[gpu-cuda13]==0.13.0a4"' in command
    assert "compute_diagnostics --track cuda13" in command


def test_owned_runtime_cleanup_failure_is_not_reported_as_available(monkeypatch):
    class CleanupFailureRuntime(_FakeRuntime):
        def close(self):
            raise RuntimeError("private allocation escaped")

    runtime = CleanupFailureRuntime()
    class OwnedRegistry:
        library_descriptors = ()

        def probe_runtime(self, _runtime_id, *, refresh=False):
            return runtime.probe(refresh=refresh)

        def runtime(self, _runtime_id):
            return runtime

        def close(self):
            runtime.close()

    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.ComputeRegistry",
        OwnedRegistry,
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
    assert "clean shutdown" in report.summary.lower()
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


def test_live_catalog_reports_three_distinct_layers_and_every_public_region():
    runtime = _FakeRuntime()

    standard = _doctor(runtime, library_probes=_library_probes())
    complete = _doctor(
        runtime,
        library_probes=_library_probes(cucim=True),
        packages=(
            PackageRecord("cupy-cuda13x", "14.1.1"),
            PackageRecord("cucim-cu13", "26.6.0"),
        ),
    )

    live_ids = {
        spec.implementation_id
        for spec in accelerator_compute_specs()
        if spec.visible_for(allow_experimental=False)
    }
    assert live_ids == {
        "cucim-rolling_ball_background-v2",
        "cucim-subtract_background-v2",
        "cupyx-median-filter-v1",
        "cupyx-gaussian-blur-v1",
        "cupyx-gaussian-blur-3d-v1",
        "cupyx-convert-dtype-preserve-f32-v1",
        "cupy-binary-threshold-f32-exact-v1",
        "cupy-extract-channel-view-v1",
        "rl-cupy-f32-v1",
        "rl-tv-cupy-f32-v1",
        "cupyx-canny-edges-exact-v1",
        "cupy-otsu-threshold-exact-v1",
        "cupy-sigma-filter-v1",
        "cupyx-connected-components-v1",
        "cucim-measure-objects-basic-v1",
        "cucim-measure-objects-intensity-basic-v1",
    }
    assert standard.cuda_ready
    assert {probe.library_id for probe in standard.library_probes} == {
        "cupy",
        "cupyx",
        "cucim",
    }
    assert {region.implementation_id for region in standard.admission_regions} == (
        live_ids
    )
    assert standard.status is DoctorStatus.AVAILABLE
    assert len(standard.admitted_regions) == _public_region_count("cupy", "cupyx")
    assert len(complete.admitted_regions) == len(accelerator_compute_specs())
    assert complete.guidance is None


def test_cuda_can_start_while_library_and_public_admission_remain_degraded():
    report = _doctor(
        _FakeRuntime(),
        library_probes=_library_probes(cupyx=False),
    )

    assert report.cuda_ready
    assert report.status is DoctorStatus.DEGRADED
    assert report.reason_code == "public_cuda_degraded"
    assert len(report.admitted_regions) == _public_region_count("cupy")
    failed = {
        region.implementation_library_id
        for region in report.admission_regions
        if not region.admitted
    }
    assert failed == {"cupyx", "cucim"}


def test_empty_standard_region_catalog_never_reports_available():
    cucim_only = tuple(
        spec
        for spec in accelerator_compute_specs()
        if spec.implementation_library_id == "cucim"
    )

    report = _doctor(
        _FakeRuntime(),
        library_probes=_library_probes(cucim=True),
        implementation_specs=cucim_only,
    )

    assert report.cuda_ready
    assert len(report.admitted_regions) == 4
    assert report.status is DoctorStatus.DEGRADED


@pytest.mark.parametrize(
    "missing_evidence",
    ["fingerprint", "driver", "compute_capability", "scientific_stack"],
)
def test_runtime_ready_never_overrides_missing_public_evidence(missing_evidence):
    runtime = _FakeRuntime()
    kwargs = {}
    if missing_evidence == "fingerprint":
        runtime.result = replace(runtime.result, environment_fingerprint="")
    elif missing_evidence == "driver":
        runtime.result = replace(
            runtime.result,
            metadata=(("cuda_runtime_version", "13020"),),
        )
    elif missing_evidence == "compute_capability":
        runtime.result = replace(
            runtime.result,
            devices=(replace(runtime.result.devices[0], metadata=()),),
        )
    else:
        kwargs["scientific_stack_versions"] = ()

    report = _doctor(
        runtime,
        library_probes=_library_probes(cucim=True),
        **kwargs,
    )

    assert report.cuda_ready
    assert report.status is DoctorStatus.DEGRADED
    assert report.admitted_regions == ()


def test_cuda_starts_on_linux_but_current_public_admission_stays_closed():
    report = collect_compute_diagnostics(
        runtime=_FakeRuntime(),
        packages=(PackageRecord("cupy-cuda13x", "14.1.1"),),
        library_probes=_library_probes(cucim=True),
        platform_name="linux",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
        scientific_stack_versions=_SCIENTIFIC_STACK,
    )

    assert report.cuda_ready
    assert report.status is DoctorStatus.UNSUPPORTED
    assert report.reason_code == "public_admission_unavailable"
    assert report.admitted_regions == ()


def test_all_provider_noise_is_contained_so_json_output_stays_valid(
    monkeypatch,
    capsys,
):
    class NoisyRuntime(_FakeRuntime):
        def probe(self, *, refresh=False):
            print("runtime factory/probe stdout")
            print("runtime factory/probe stderr", file=sys.stderr)
            return super().probe(refresh=refresh)

        def memory_snapshot(self, *, device_id=""):
            print("memory provider noise")
            return super().memory_snapshot(device_id=device_id)

    runtime = NoisyRuntime()

    class NoisyRegistry:
        library_descriptors = tuple(
            SimpleNamespace(library_id=value)
            for value in ("cupy", "cupyx", "cucim")
        )

        def probe_runtime(self, _runtime_id, *, refresh=False):
            print("registry probe noise")
            return runtime.probe(refresh=refresh)

        def runtime(self, _runtime_id):
            print("runtime acquisition noise")
            return runtime

        def probe_library(self, library_id, *, refresh=False):
            del refresh
            print(f"third-party noise from {library_id}")
            return next(
                probe
                for probe in _library_probes()
                if probe.library_id == library_id
            )

        def close(self):
            print("runtime cleanup noise")

    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.ComputeRegistry",
        NoisyRegistry,
    )

    report = collect_compute_diagnostics(
        packages=(PackageRecord("cupy-cuda13x", "14.1.1"),),
        platform_name="win32",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
        scientific_stack_versions=_SCIENTIFIC_STACK,
    )

    assert report.status is DoctorStatus.AVAILABLE
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.collect_compute_diagnostics",
        lambda **_kwargs: report,
    )
    assert main(["--track", "cuda13", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "available"


def test_support_bundle_is_strictly_allowlisted_and_redacts_private_data():
    secret_node = r"C:\Users\Faten\private-node"
    execution = ExecutionReport(
        request=ComputeRequest(node_preferences={secret_node: "auto"}),
        environment=ComputeEnvironment(),
        actual_decisions=(
            NodeExecutionDecision(
                secret_node,
                "subtract_background",
                NodeComputePreference(),
                "cpu-numpy",
                "cpu",
                "cpu-subtract-background-v1",
                DecisionKind.POLICY_CPU,
                DecisionReason.EXPLICIT_CPU,
                r"Read C:\Users\Faten\samples\private.tif",
            ),
        ),
        warnings=(
            r"token=abc123 at \\lab-server\Faten\private and /home/faten/raw.tif",
            "https://faten:secret@example.test/private?token=abc123",
            "Authorization: Bearer synthetic-bearer-credential on DESKTOP-FATEN",
        ),
    )
    report = _doctor(
        _FakeRuntime(),
        library_probes=_library_probes(),
    )
    report = replace(
        report,
        details=report.details
        + (r"password=hunter2 in C:\Users\Faten\VIPP",),
    )

    bundle = build_compute_support_bundle(
        report,
        recent_execution=execution,
        generated_utc="2026-08-12T12:00:00+00:00",
    )
    serialized = json.dumps(bundle, sort_keys=True)

    assert set(bundle) == {
        "schema_id",
        "schema_version",
        "generated_utc",
        "privacy",
        "application",
        "host",
        "diagnostic",
        "recent_execution",
    }
    assert set(bundle["privacy"]) == {"policy_id", "redacted", "omitted"}
    assert set(bundle["application"]) == {"napari_vipp_version"}
    assert set(bundle["host"]) == {
        "platform",
        "execution_mode",
        "python",
        "track",
        "scientific_stack_versions",
    }
    diagnostic = bundle["diagnostic"]
    assert set(diagnostic) == {
        "status",
        "reason_code",
        "summary",
        "cuda",
        "libraries",
        "public_admission",
        "guidance",
        "details",
        "packages",
    }
    assert set(diagnostic["cuda"]) == {
        "status",
        "reason_code",
        "message",
        "version",
        "driver_version",
        "cuda_runtime_version",
        "device",
        "memory",
    }
    assert set(diagnostic["cuda"]["device"]) == {
        "device_id",
        "display_name",
        "total_memory_bytes",
        "compute_capability",
    }
    assert set(diagnostic["cuda"]["memory"]) == {
        "runtime_id",
        "device_id",
        "topology",
        "device_total_bytes",
        "device_free_bytes",
        "runtime_live_bytes",
        "runtime_reserved_bytes",
        "out_of_pool_bytes",
    }
    assert set(diagnostic["libraries"][0]) == {
        "library_id",
        "required_for_standard_cuda",
        "optional",
        "available",
        "version",
        "reason_code",
        "message",
        "metadata",
    }
    admission = diagnostic["public_admission"]
    assert set(admission) == {"admitted_count", "total_count", "regions"}
    assert set(admission["regions"][0]) == {
        "operation_id",
        "implementation_id",
        "implementation_version",
        "implementation_library_id",
        "admission_tier",
        "environment_policy_id",
        "admitted",
        "reason_code",
        "reason",
        "supported_spatial_ndims",
        "public_input_dtypes",
        "limitations",
    }
    assert set(diagnostic["guidance"]) == {
        "action_id",
        "title",
        "summary",
        "documentation_url",
        "optional",
    }
    assert set(diagnostic["packages"][0]) == {"name", "version"}
    recent_payload = bundle["recent_execution"]
    assert set(recent_payload) == {
        "requested_mode",
        "cleanup_succeeded",
        "decisions",
        "fallback_records",
        "warnings",
    }
    assert set(recent_payload["decisions"][0]) == {
        "operation_id",
        "requested_preference",
        "runtime_id",
        "implementation_library_id",
        "implementation_id",
        "implementation_version",
        "decision_kind",
        "reason_code",
        "reason",
        "fallback_used",
        "fallback_reason",
    }
    assert bundle["schema_id"] == "napari-vipp-compute-support-bundle"
    assert bundle["privacy"]["redacted"] is True
    for private_value in (
        "Faten",
        "abc123",
        "hunter2",
        "synthetic-bearer-credential",
        "DESKTOP-FATEN",
        "private.tif",
        "private-node",
        "safe-test-fingerprint",
    ):
        assert private_value not in serialized
    assert "node_id" not in serialized
    assert "environment_fingerprint" not in serialized
    assert "repair_command" not in serialized
    assert "<redacted" in serialized


def test_support_bundle_write_is_atomic_and_cli_can_export(
    tmp_path,
    monkeypatch,
    capsys,
):
    report = _doctor(_FakeRuntime(), library_probes=_library_probes())
    with pytest.raises(ValueError, match="must not be empty"):
        build_compute_support_bundle(report, generated_utc="")
    with pytest.raises(ValueError, match="UTC offset"):
        build_compute_support_bundle(
            report,
            generated_utc="2026-08-12T14:00:00+02:00",
        )
    target = tmp_path / "support.json"
    written = write_compute_support_bundle(
        target,
        report,
        generated_utc="2026-08-12T12:00:00+00:00",
    )
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == 1

    target.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(
        "napari_vipp.core.atomic_io.atomic_replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        write_compute_support_bundle(target, report)
    assert target.read_text(encoding="utf-8") == "keep me"
    assert not tuple(tmp_path.glob("*.tmp"))

    monkeypatch.undo()
    monkeypatch.setattr(
        "napari_vipp.core.compute_diagnostics.collect_compute_diagnostics",
        lambda **_kwargs: report,
    )
    cli_target = tmp_path / "cli-support.json"
    assert main(["--json", "--support-bundle", str(cli_target)]) == 0
    json.loads(capsys.readouterr().out)
    assert json.loads(cli_target.read_text(encoding="utf-8"))["privacy"][
        "redacted"
    ]


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
