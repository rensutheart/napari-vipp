from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.execution import PipelineRunResult
from napari_vipp.core.execution_telemetry import DeviceExecutionObservation
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_interactive_tuning.py"


@pytest.fixture(scope="module")
def latency_script():
    module_name = "_vipp_test_interactive_tuning_latency"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("action", ("import", "--help"))
def test_cpu_safe_surfaces_do_not_import_or_initialize_cuda(action):
    guarded = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'cupy' or name.startswith(('cupy.', 'cupyx', 'cucim')):
        raise RuntimeError('CPU-safe latency surface imported a GPU package')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
"""
    if action == "import":
        body = f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='not_main')"
    else:
        body = (
            f"sys.argv = [{str(SCRIPT_PATH)!r}, '--help']; "
            f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')"
        )
    completed = subprocess.run(
        [sys.executable, "-c", guarded + body],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "imported a GPU package" not in completed.stderr


def _fake_executor(requests):
    def execute(request):
        requests.append(request)
        restored = deserialize_workflow(request.workflow)
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        pipeline.completed_node_ids.add("gaussian_blur_1")
        accelerated = request.compute_request.mode is ComputeMode.PREFER_GPU
        decision = SimpleNamespace(
            node_id="gaussian_blur_1",
            operation_id="gaussian_blur",
            runtime_id="cuda-cupy" if accelerated else "cpu-numpy",
            implementation_library_id="cupy" if accelerated else "cpu",
            implementation_id=(
                "cupy-gaussian-blur-v1" if accelerated else "cpu-gaussian_blur-v1"
            ),
            implementation_version="1",
            decision_kind="selected" if accelerated else "policy_cpu",
            reason="selected_implementation" if accelerated else "explicit_cpu",
            reason_text="Selected by deterministic harness test.",
            fallback_used=False,
            fallback_reason="none",
        )
        environment = SimpleNamespace(
            device_id="cuda:7" if accelerated else "cpu:0",
            device_name="Fake CUDA device" if accelerated else "Host CPU",
            device_class="nvidia-cuda" if accelerated else "host",
            memory_topology="discrete" if accelerated else "host",
            runtime_ids=("cpu-numpy", "cuda-cupy") if accelerated else ("cpu-numpy",),
            implementation_libraries=("cpu", "cupy") if accelerated else ("cpu",),
            driver_version="13030" if accelerated else "",
            total_accelerator_memory_bytes=8 * 1024**3 if accelerated else 0,
            probe_status="available",
            probe_reason="",
        )
        report = SimpleNamespace(
            request=request.compute_request,
            environment=environment,
            actual_decisions=(decision,),
            plan=None,
            fallback_records=(),
            warnings=(),
            cleanup_succeeded=True,
        )
        telemetry_config = request.device_execution_telemetry
        telemetry = (
            None
            if telemetry_config is None
            else DeviceExecutionObservation(
                started_monotonic_seconds=float(len(requests)),
                elapsed_seconds=0.5,
                synchronized_device_phases=(telemetry_config.synchronize_device_phases),
            )
        )
        return PipelineRunResult(
            run_id=request.run_id,
            workflow=request.workflow,
            pipeline=pipeline,
            execution_report=report,
            device_execution_telemetry=telemetry,
        )

    return execute


def test_deterministic_harness_builds_one_cold_and_cached_warm_requests(
    latency_script,
    tmp_path,
):
    requests = []
    clock_values = iter((10.0, 10.8, 20.0, 20.3, 30.0, 30.2))
    document = latency_script.collect_latency_evidence(
        mode="prefer_gpu",
        warm_sigmas=(1.5, 1.7),
        device_id="cuda:7",
        clock=lambda: next(clock_values),
        execute=_fake_executor(requests),
    )

    assert [item["temperature"] for item in document["runs"]] == [
        "cold",
        "warm",
        "warm",
    ]
    assert [item["wall_seconds"] for item in document["runs"]] == pytest.approx(
        [0.8, 0.3, 0.2]
    )
    assert [item["sigma"] for item in document["runs"]] == [1.2, 1.5, 1.7]
    assert requests[0].dirty_node_ids is None
    assert requests[1].dirty_node_ids == frozenset({"gaussian_blur_1"})
    assert requests[2].dirty_node_ids == frozenset({"gaussian_blur_1"})
    assert requests[0].cached_outputs is None
    assert requests[1].cached_outputs is not None
    assert "gaussian_blur_1" in requests[1].cached_outputs
    assert requests[1].completed_node_ids == frozenset({"gaussian_blur_1"})
    assert all(
        request.compute_request.mode is ComputeMode.PREFER_GPU
        and request.compute_request.device_id == "cuda:7"
        and request.device_execution_telemetry is not None
        and request.device_execution_telemetry.synchronize_device_phases
        for request in requests
    )
    assert [
        next(
            node["params"]["sigma"]
            for node in request.workflow["nodes"]
            if node["id"] == "gaussian_blur_1"
        )
        for request in requests
    ] == [1.2, 1.5, 1.7]
    assert document["summary"]["warm_median_wall_seconds"] == pytest.approx(0.25)
    assert document["summary"]["actual_device_ids"] == ["cuda:7"]
    assert document["summary"]["actual_target_backends"] == [
        "cuda-cupy/cupy/cupy-gaussian-blur-v1"
    ]
    assert document["method"]["cache_contract"] == (
        "keep-all-host-results; no cross-run device residency"
    )
    for run in document["runs"]:
        assert run["core_timing"]["pipeline_run_result.device_execution_telemetry"] == {
            "started_monotonic_seconds": float(run["run_index"] + 1),
            "elapsed_seconds": 0.5,
            "spans": [],
            "synchronized_device_phases": True,
            "terminal_memory_snapshots": [],
        }

    output = latency_script._atomic_write_json(tmp_path / "latency.json", document)
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert tuple(tmp_path.iterdir()) == (output,)


def test_cli_rejects_invalid_or_empty_sigma_sequences(latency_script):
    with pytest.raises(ValueError, match="At least one warm"):
        latency_script._normalized_sigma_values(())
    with pytest.raises(ValueError, match="between 0 and 12"):
        latency_script._parse_sigma_values("1.2,13")
    with pytest.raises(ValueError, match="comma-separated"):
        latency_script._parse_sigma_values("1.2,")
    with pytest.raises(ValueError, match="valid only"):
        latency_script.collect_latency_evidence(
            mode="cpu",
            warm_sigmas=(1.3,),
            device_id="cuda:0",
        )


def test_optional_core_timing_sidecar_is_consumed_without_core_coupling(
    latency_script,
):
    @dataclass(frozen=True)
    class FakeTiming:
        h2d_seconds: float
        transfer_count: int

    @dataclass(frozen=True)
    class FakeReport:
        device_timing: FakeTiming

    @dataclass(frozen=True)
    class FakeResult:
        execution_report: FakeReport
        interaction_telemetry: dict[str, object]

    result = FakeResult(
        execution_report=FakeReport(FakeTiming(0.125, 1)),
        interaction_telemetry={"outcome": "published"},
    )

    assert latency_script._optional_core_timing(result) == {
        "pipeline_run_result.interaction_telemetry": {"outcome": "published"},
        "execution_report.device_timing": {
            "h2d_seconds": 0.125,
            "transfer_count": 1,
        },
    }


def test_real_cpu_smoke_executes_bundled_workflow_and_one_sigma_edit(
    latency_script,
):
    document = latency_script.collect_latency_evidence(
        mode="cpu",
        warm_sigmas=(1.3,),
    )

    assert len(document["runs"]) == 2
    assert all(
        item["target_backend"]["runtime_id"] == "cpu-numpy" for item in document["runs"]
    )
    assert document["runs"][0]["dirty_node_ids"] is None
    assert document["runs"][1]["dirty_node_ids"] == ["gaussian_blur_1"]
    assert document["summary"]["all_cleanup_succeeded"] is True
    assert document["summary"]["any_visible_fallback"] is False
