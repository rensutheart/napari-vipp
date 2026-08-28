from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.core import compute_history
from napari_vipp.core.compute import (
    ComputeEnvironment,
    DecisionKind,
    DecisionReason,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
)
from napari_vipp.core.compute_history import (
    PIPELINE_TIMING_HISTORY_PATH_ENV,
    JsonPipelineTimingStore,
    PipelineTimingSample,
    PipelineTimingStoreError,
)


def _environment(*, driver_version: str = "555.42") -> ComputeEnvironment:
    return ComputeEnvironment(
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy"),
        runtime_versions=(("cuda-cupy", "13.0"),),
        runtime_probe_fingerprints=(("cuda-cupy", "runtime-probe"),),
        driver_version=driver_version,
        device_id="cuda:0",
        device_name="Test GPU",
        device_class="nvidia-gpu",
        memory_topology="discrete",
        total_accelerator_memory_bytes=24 * 1024**3,
    )


def _decision(
    node_id: str,
    *,
    accelerated: bool,
) -> NodeExecutionDecision:
    operation_id = "gaussian_blur"
    if accelerated:
        return NodeExecutionDecision(
            node_id,
            operation_id,
            NodeComputePreference(),
            "cuda-cupy",
            "cupy",
            "cupy-gaussian-blur-v1",
            DecisionKind.SELECTED,
            DecisionReason.SELECTED_IMPLEMENTATION,
            "The compatible GPU implementation was selected.",
            implementation_version="1",
        )
    return NodeExecutionDecision(
        node_id,
        operation_id,
        NodeComputePreference("cpu"),
        "cpu-numpy",
        "cpu",
        "cpu-gaussian_blur-v1",
        DecisionKind.POLICY_CPU,
        DecisionReason.EXPLICIT_CPU,
        "The authoritative CPU implementation was selected.",
        implementation_version="1",
    )


def _bypass_decision(node_id: str = "crop") -> NodeExecutionDecision:
    return NodeExecutionDecision(
        node_id,
        "crop_stack",
        NodeComputePreference("best_gpu"),
        "vipp-bypass",
        "vipp-alias",
        "vipp-safe-bypass-v1",
        DecisionKind.BYPASSED,
        DecisionReason.BYPASSED,
        "The exact input was forwarded without executing an implementation.",
        implementation_version="1",
    )


def _sample(
    *,
    elapsed_seconds: float,
    accelerated: bool,
    workload_fingerprint: str = "workload-a",
    host_environment_fingerprint: str = "host-a",
    environment: ComputeEnvironment | None = None,
    created_utc: str = "2026-08-05T10:00:00+00:00",
) -> PipelineTimingSample:
    return PipelineTimingSample.completed_run(
        workload_fingerprint=workload_fingerprint,
        host_environment_fingerprint=host_environment_fingerprint,
        environment=environment or _environment(),
        decisions=(_decision("gaussian", accelerated=accelerated),),
        elapsed_seconds=elapsed_seconds,
        requested_mode="prefer_gpu" if accelerated else "cpu",
        execution_surface="planned-owned-registry-v1",
        created_utc=created_utc,
    )


def test_pytest_routes_default_history_to_current_test_temp_directory(tmp_path):
    expected = (
        tmp_path
        / ".napari-vipp-test-state"
        / "pipeline-timing-history-v2.json"
    ).resolve()

    assert compute_history.default_pipeline_timing_history_path() == expected


def test_default_history_path_honors_explicit_override(tmp_path, monkeypatch):
    override = tmp_path / "isolated" / "history.json"
    monkeypatch.setenv(PIPELINE_TIMING_HISTORY_PATH_ENV, str(override))

    assert compute_history.default_pipeline_timing_history_path() == override.resolve()


def test_default_history_path_rejects_an_empty_explicit_override(monkeypatch):
    monkeypatch.setenv(PIPELINE_TIMING_HISTORY_PATH_ENV, "   ")

    with pytest.raises(ValueError, match=PIPELINE_TIMING_HISTORY_PATH_ENV):
        compute_history.default_pipeline_timing_history_path()


def test_timing_history_override_is_inherited_by_subprocess(tmp_path, monkeypatch):
    override = tmp_path / "subprocess" / "history.json"
    monkeypatch.setenv(PIPELINE_TIMING_HISTORY_PATH_ENV, str(override))
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from napari_vipp.core.compute_history import "
                "default_pipeline_timing_history_path; "
                "print(default_pipeline_timing_history_path())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == override.resolve()


def test_one_exact_cpu_and_gpu_sample_selects_a_decisive_gpu_win(tmp_path):
    environment = _environment()
    store = JsonPipelineTimingStore(tmp_path / "history.json")
    cpu = _sample(elapsed_seconds=1.0, accelerated=False, environment=environment)
    gpu = _sample(elapsed_seconds=0.5, accelerated=True, environment=environment)
    store.append(cpu)
    store.append(gpu)

    choice = store.choose(
        workload_fingerprint="workload-a",
        host_environment_fingerprint="host-a",
        accelerator_environment_fingerprint=environment.fingerprint,
        execution_surface="planned-owned-registry-v1",
    )

    assert choice is not None
    assert choice.assignment == gpu.assignment
    assert choice.uses_accelerator
    assert choice.cpu_median_seconds == pytest.approx(1.0)
    assert choice.selected_median_seconds == pytest.approx(0.5)
    assert choice.cpu_sample_count == 1
    assert choice.selected_sample_count == 1
    assert choice.evidence_digest


def test_ambiguous_single_sample_comparison_keeps_cpu(tmp_path):
    environment = _environment()
    store = JsonPipelineTimingStore(tmp_path / "history.json")
    cpu = _sample(elapsed_seconds=1.0, accelerated=False, environment=environment)
    gpu = _sample(elapsed_seconds=0.9, accelerated=True, environment=environment)
    store.append(cpu)
    store.append(gpu)

    choice = store.choose(
        workload_fingerprint="workload-a",
        host_environment_fingerprint="host-a",
        accelerator_environment_fingerprint=environment.fingerprint,
        execution_surface="planned-owned-registry-v1",
    )

    assert choice is not None
    assert choice.assignment == cpu.assignment
    assert not choice.uses_accelerator
    assert choice.cpu_median_seconds == pytest.approx(1.0)
    assert choice.selected_median_seconds == pytest.approx(1.0)
    assert "noise margin" in choice.reason


def test_gpu_only_history_requests_one_cpu_comparison_on_the_same_surface(tmp_path):
    environment = _environment()
    store = JsonPipelineTimingStore(tmp_path / "history.json")
    store.append(
        _sample(elapsed_seconds=0.5, accelerated=True, environment=environment)
    )

    matching = store.coverage(
        workload_fingerprint="workload-a",
        host_environment_fingerprint="host-a",
        accelerator_environment_fingerprint=environment.fingerprint,
        execution_surface="planned-owned-registry-v1",
    )
    different_surface = store.coverage(
        workload_fingerprint="workload-a",
        host_environment_fingerprint="host-a",
        accelerator_environment_fingerprint=environment.fingerprint,
        execution_surface="planned-borrowed-registry-v1",
    )

    assert matching.needs_cpu_exploration
    assert matching.accelerated_sample_count == 1
    assert matching.cpu_sample_count == 0
    assert not different_surface.needs_cpu_exploration
    assert different_surface.accelerated_sample_count == 0


def test_mismatched_workload_host_or_gpu_environment_is_ignored(tmp_path):
    environment = _environment()
    store = JsonPipelineTimingStore(tmp_path / "history.json")
    store.append(
        _sample(elapsed_seconds=1.0, accelerated=False, environment=environment)
    )
    store.append(
        _sample(elapsed_seconds=0.5, accelerated=True, environment=environment)
    )

    assert (
        store.choose(
            workload_fingerprint="workload-b",
            host_environment_fingerprint="host-a",
            accelerator_environment_fingerprint=environment.fingerprint,
            execution_surface="planned-owned-registry-v1",
        )
        is None
    )
    assert (
        store.choose(
            workload_fingerprint="workload-a",
            host_environment_fingerprint="host-b",
            accelerator_environment_fingerprint=environment.fingerprint,
            execution_surface="planned-owned-registry-v1",
        )
        is None
    )
    assert (
        store.choose(
            workload_fingerprint="workload-a",
            host_environment_fingerprint="host-a",
            accelerator_environment_fingerprint=replace(
                environment,
                driver_version="556.01",
            ).fingerprint,
            execution_surface="planned-owned-registry-v1",
        )
        is None
    )


def test_fallback_free_assignment_survives_store_roundtrip(tmp_path):
    environment = _environment()
    decisions = (
        _decision("cpu-node", accelerated=False),
        _decision("gpu-node", accelerated=True),
    )
    assert all(not decision.fallback_used for decision in decisions)
    sample = PipelineTimingSample.completed_run(
        workload_fingerprint="mixed-workload",
        host_environment_fingerprint="host-a",
        environment=environment,
        decisions=decisions,
        elapsed_seconds=0.75,
        requested_mode="prefer_gpu",
        execution_surface="planned-owned-registry-v1",
        created_utc="2026-08-05T11:00:00+00:00",
    )
    path = tmp_path / "history.json"

    JsonPipelineTimingStore(path).append(sample)
    restored = JsonPipelineTimingStore(path).samples()

    assert restored == (sample,)
    assert restored[0].assignment.uses_accelerator
    assert [item.node_id for item in restored[0].assignment.decisions] == [
        "cpu-node",
        "gpu-node",
    ]


def test_timing_assignment_excludes_bypass_from_compute_and_accelerator_counts():
    sample = PipelineTimingSample.completed_run(
        workload_fingerprint="bypassed-workload",
        host_environment_fingerprint="host-a",
        environment=_environment(),
        decisions=(
            _decision("gaussian", accelerated=False),
            _bypass_decision(),
        ),
        elapsed_seconds=0.5,
        requested_mode="prefer_gpu",
        execution_surface="planned-owned-registry-v1",
    )

    assert [item.node_id for item in sample.assignment.decisions] == ["gaussian"]
    assert not sample.assignment.uses_accelerator
    assert sample.accelerator_environment_fingerprint == ""


def test_bypass_only_run_cannot_be_recorded_as_a_compute_assignment():
    with pytest.raises(ValueError, match="at least one decision"):
        PipelineTimingSample.completed_run(
            workload_fingerprint="bypass-only",
            host_environment_fingerprint="host-a",
            environment=_environment(),
            decisions=(_bypass_decision(),),
            elapsed_seconds=0.5,
            requested_mode="prefer_gpu",
            execution_surface="planned-owned-registry-v1",
        )


def test_timing_sample_rejects_a_fallback_decision():
    fallback = replace(
        _decision("gaussian", accelerated=True),
        runtime_id="cpu-numpy",
        implementation_library_id="cpu",
        implementation_id="cpu-gaussian_blur-v1",
        decision_kind=DecisionKind.FALLBACK_CPU,
        fallback_reason=FallbackReason.RUNTIME_FAILURE,
    )

    with pytest.raises(ValueError, match="fallback-free"):
        PipelineTimingSample.completed_run(
            workload_fingerprint="workload-a",
            host_environment_fingerprint="host-a",
            environment=_environment(),
            decisions=(fallback,),
            elapsed_seconds=1.0,
            requested_mode="auto",
            execution_surface="planned-owned-registry-v1",
        )


def test_store_bounds_samples_per_exact_assignment(tmp_path):
    store = JsonPipelineTimingStore(tmp_path / "history.json")
    limit = compute_history._MAX_SAMPLES_PER_ASSIGNMENT
    for index in range(limit + 3):
        store.append(
            _sample(
                elapsed_seconds=float(index + 1),
                accelerated=False,
                created_utc=f"2026-08-{index + 1:02d}T10:00:00+00:00",
            )
        )

    samples = store.samples()

    assert len(samples) == limit
    assert [item.elapsed_seconds for item in samples] == [
        float(index + 1) for index in range(3, limit + 3)
    ]


def test_two_store_instances_append_without_losing_threaded_updates(tmp_path):
    path = tmp_path / "history.json"
    stores = (JsonPipelineTimingStore(path), JsonPipelineTimingStore(path))

    def append(index: int) -> None:
        stores[index % 2].append(
            _sample(
                elapsed_seconds=float(index + 1),
                accelerated=False,
                created_utc=f"2026-08-{index + 1:02d}T12:00:00+00:00",
            )
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(append, range(8)))

    assert len(JsonPipelineTimingStore(path).samples()) == 8


def test_corrupt_history_fails_closed(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not valid JSON", encoding="utf-8")
    store = JsonPipelineTimingStore(path)

    with pytest.raises(PipelineTimingStoreError, match="Could not read"):
        store.samples()

    with pytest.raises(PipelineTimingStoreError, match="Could not read"):
        store.choose(
            workload_fingerprint="workload-a",
            host_environment_fingerprint="host-a",
            accelerator_environment_fingerprint=_environment().fingerprint,
            execution_surface="planned-owned-registry-v1",
        )
