from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.compute_benchmark_coordinator as coordinator_module
from napari_vipp._tests.test_compute_benchmark_adapter import (
    ManualClock,
    _FakeRuntime,
)
from napari_vipp.core.compute import (
    ComputeEnvironment,
    NodePreferenceKind,
)
from napari_vipp.core.compute_benchmark import (
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    JsonBenchmarkStore,
)
from napari_vipp.core.compute_benchmark_adapter import (
    build_registered_node_benchmark,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    ApplicationNodeBenchmarkCoordinator,
    NodeBenchmarkPhase,
    NodeBenchmarkUnavailable,
    stable_preference_for_benchmark_winner,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.operations import median_filter
from napari_vipp.core.pipeline import PrototypePipeline


def _environment() -> ComputeEnvironment:
    return ComputeEnvironment(
        os_name="Windows",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupyx"),
        runtime_versions=(("cuda-cupy", "14.1.1"), ("cupyx", "14.1.1")),
        runtime_probe_fingerprints=(
            ("cuda-cupy", "fake-runtime-fingerprint"),
        ),
        runtime_metadata=(
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        driver_version="13030",
        device_id="cuda:0",
        device_name="Fake RTX",
        device_class="nvidia-cuda",
        device_metadata=(("compute_capability", "12.0"),),
        memory_topology="discrete",
        total_accelerator_memory_bytes=16 * 1024**3,
        probe_status="available",
    )


def _median_pipeline(values: np.ndarray) -> tuple[PrototypePipeline, str, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    median = pipeline.add_node("median_filter")
    median.params["size"] = 3
    result = pipeline.connect(source_id, median.id)
    assert result.success
    pipeline.run(values, input_name="benchmark-input")
    return pipeline, source_id, median.id


def _coordinator_with_fake_runtime(
    tmp_path,
    monkeypatch,
    clock: ManualClock,
) -> tuple[ApplicationNodeBenchmarkCoordinator, _FakeRuntime]:
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "probe_runtime",
        lambda _runtime_id, refresh=False: runtime.probe(refresh=refresh),
    )

    def gpu(device, **kwargs):
        clock.advance(0.040)
        return runtime.allocate(median_filter(device.array, **kwargs))

    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    original_builder = build_registered_node_benchmark

    def measured_builder(call, **kwargs):
        cpu_function = call.cpu_function

        def slower_cpu(data, **cpu_kwargs):
            clock.advance(0.100)
            return cpu_function(data, **cpu_kwargs)

        return original_builder(
            replace(call, cpu_function=slower_cpu),
            **kwargs,
        )

    monkeypatch.setattr(
        coordinator_module,
        "build_registered_node_benchmark",
        measured_builder,
    )
    return (
        ApplicationNodeBenchmarkCoordinator(
            registry,
            tmp_path / "node-benchmarks.json",
            clock=clock,
        ),
        runtime,
    )


def test_selected_node_benchmark_is_detached_persisted_and_parity_gated(
    tmp_path,
    monkeypatch,
):
    clock = ManualClock()
    coordinator, runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    values = np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    pipeline, source_id, node_id = _median_pipeline(values)
    source_output = pipeline.outputs[source_id]
    source_before = np.array(source_output, copy=True)
    node_output_before = np.array(pipeline.outputs[node_id], copy=True)
    params_before = dict(pipeline.nodes[node_id].params)
    completed_before = set(pipeline.completed_node_ids)
    progress = []

    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        paired_bootstrap_samples=200,
        time_budget_seconds=10.0,
        progress=progress.append,
    )

    assert plan.node_id == node_id
    assert plan.operation_id == "median_filter"
    assert len(plan.workload_fingerprint) == 64
    assert [item.implementation_id for item in plan.eligibility] == [
        "cupyx-median-filter-v1"
    ]
    assert all(item.supported for item in plan.eligibility)
    assert not np.shares_memory(
        plan.registered.detached_call.inputs[0],
        source_output,
    )
    assert pipeline.nodes[node_id].params == params_before
    assert pipeline.completed_node_ids == completed_before
    np.testing.assert_array_equal(pipeline.outputs[source_id], source_before)
    np.testing.assert_array_equal(pipeline.outputs[node_id], node_output_before)

    result = coordinator.run(plan, progress=progress.append)

    assert result.record.accepted_implementation_id == "cupyx-median-filter-v1"
    gpu_result = next(
        item
        for item in result.record.candidates
        if item.implementation_id == "cupyx-median-filter-v1"
    )
    assert gpu_result.parity_passed
    assert not gpu_result.error
    assert result.winner_preference.kind is NodePreferenceKind.LIBRARY
    assert result.winner_preference.value == "cupyx"
    assert JsonBenchmarkStore(plan.store_path).get(result.record.key) == result.record
    assert runtime.live == {}
    assert [item.phase for item in progress] == [
        NodeBenchmarkPhase.PREPARING,
        NodeBenchmarkPhase.ELIGIBILITY,
        NodeBenchmarkPhase.READY,
        NodeBenchmarkPhase.BENCHMARKING,
        NodeBenchmarkPhase.COMPLETE,
    ]
    assert pipeline.nodes[node_id].params == params_before
    assert pipeline.completed_node_ids == completed_before
    np.testing.assert_array_equal(pipeline.outputs[source_id], source_before)
    np.testing.assert_array_equal(pipeline.outputs[node_id], node_output_before)


def test_workload_current_check_covers_parameters_and_input_content(
    tmp_path,
    monkeypatch,
):
    clock = ManualClock()
    coordinator, _runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    values = np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    pipeline, source_id, node_id = _median_pipeline(values)
    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
    )

    assert coordinator.workload_is_current(pipeline, plan)
    pipeline.nodes[node_id].params["size"] = 5
    assert not coordinator.workload_is_current(pipeline, plan)
    pipeline.nodes[node_id].params["size"] = 3
    assert coordinator.workload_is_current(pipeline, plan)
    pipeline.outputs[source_id][0, 0] += 1
    assert not coordinator.workload_is_current(pipeline, plan)


def test_static_ineligibility_rejects_before_runtime_probe(tmp_path, monkeypatch):
    values = np.arange(31 * 37, dtype=np.float32).reshape(31, 37)
    values[0, 0] = -0.0
    pipeline, _source_id, node_id = _median_pipeline(values)
    registry = ComputeRegistry()

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("statically rejected work must not probe a runtime")

    monkeypatch.setattr(registry, "probe_runtime", unexpected_probe)
    coordinator = ApplicationNodeBenchmarkCoordinator(
        registry,
        tmp_path / "benchmarks.json",
    )

    with pytest.raises(NodeBenchmarkUnavailable) as caught:
        coordinator.prepare(
            pipeline,
            node_id,
            allow_experimental=True,
        )

    assert len(caught.value.eligibility) == 1
    decision = caught.value.eligibility[0]
    assert not decision.supported
    assert decision.reason_code == "workload_unsupported"
    assert "negative zero" in decision.reason_text
    assert JsonBenchmarkStore(tmp_path / "benchmarks.json").records() == ()


def test_coordinator_import_does_not_import_optional_gpu_packages():
    script = """
import sys
import napari_vipp.core.compute_benchmark_coordinator
assert not any(
    name == 'cupy' or name.startswith('cupy.')
    or name == 'cucim' or name.startswith('cucim.')
    for name in sys.modules
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cancellation_and_preparation_budget_publish_nothing(tmp_path, monkeypatch):
    clock = ManualClock()
    coordinator, _runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    values = np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    pipeline, _source_id, node_id = _median_pipeline(values)

    with pytest.raises(BenchmarkCancelled):
        coordinator.prepare(
            pipeline,
            node_id,
            environment=_environment(),
            allow_experimental=True,
            cancelled=lambda: True,
        )
    assert coordinator.store.records() == ()


def test_preparation_consumes_the_service_time_budget(tmp_path, monkeypatch):
    clock = ManualClock()
    coordinator, _runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    pipeline, _source_id, node_id = _median_pipeline(
        np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    )
    original_workload = coordinator_module.workload_from_prepared_node_call

    def measured_workload(call, **kwargs):
        clock.advance(0.250)
        return original_workload(call, **kwargs)

    observed_budget = []
    original_builder = coordinator_module.build_registered_node_benchmark

    def capture_builder(call, **kwargs):
        observed_budget.append(kwargs["time_budget_seconds"])
        return original_builder(call, **kwargs)

    monkeypatch.setattr(
        coordinator_module,
        "workload_from_prepared_node_call",
        measured_workload,
    )
    monkeypatch.setattr(
        coordinator_module,
        "build_registered_node_benchmark",
        capture_builder,
    )

    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        time_budget_seconds=1.0,
    )

    assert observed_budget == [pytest.approx(0.750)]
    assert plan.preparation_seconds == pytest.approx(0.250)
    assert plan.registered.request.time_budget_seconds == pytest.approx(0.750)


def test_default_memory_scope_uses_current_free_vram(tmp_path, monkeypatch):
    clock = ManualClock()
    coordinator, _runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    pipeline, _source_id, node_id = _median_pipeline(
        np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    )

    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
    )

    total = 8 * 1024**3
    free = 7 * 1024**3
    reserve = max(512 * 1024**2, total // 10)
    assert plan.registered.request.safety_reserve_bytes == reserve
    assert plan.registered.request.memory_limit_bytes == min(
        total * 80 // 100,
        free - reserve,
    )


def test_eligibility_and_benchmark_share_one_immutable_capture(
    tmp_path,
    monkeypatch,
):
    clock = ManualClock()
    coordinator, _runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    pipeline, _source_id, node_id = _median_pipeline(
        np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    )
    observed_inputs = []
    original_facts = coordinator_module._complete_call_facts
    original_builder = coordinator_module.build_registered_node_benchmark

    def capture_facts(call, *args, **kwargs):
        observed_inputs.append(call.inputs[0])
        assert not call.inputs[0].flags.writeable
        return original_facts(call, *args, **kwargs)

    def capture_builder(call, **kwargs):
        observed_inputs.append(call.inputs[0])
        assert kwargs["call_is_detached"] is True
        assert not call.inputs[0].flags.writeable
        return original_builder(call, **kwargs)

    monkeypatch.setattr(coordinator_module, "_complete_call_facts", capture_facts)
    monkeypatch.setattr(
        coordinator_module,
        "build_registered_node_benchmark",
        capture_builder,
    )

    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
    )

    assert observed_inputs == [
        plan.registered.detached_call.inputs[0],
        plan.registered.detached_call.inputs[0],
    ]

    original_workload = coordinator_module.workload_from_prepared_node_call

    def slow_workload(call, **kwargs):
        clock.advance(0.200)
        return original_workload(call, **kwargs)

    monkeypatch.setattr(
        coordinator_module,
        "workload_from_prepared_node_call",
        slow_workload,
    )
    with pytest.raises(BenchmarkBudgetExceeded):
        coordinator.prepare(
            pipeline,
            node_id,
            environment=_environment(),
            allow_experimental=True,
            time_budget_seconds=0.100,
        )
    assert coordinator.store.records() == ()


def test_winner_mapping_uses_exact_pin_only_for_same_library_competition(
    tmp_path,
    monkeypatch,
):
    clock = ManualClock()
    coordinator, _runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    pipeline, _source_id, node_id = _median_pipeline(
        np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    )
    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        paired_bootstrap_samples=200,
    )
    record = coordinator.run(plan).record
    winner = plan.admitted_specs[0]
    alternate = replace(
        winner,
        implementation_id="cupyx-median-filter-alternate-v1",
        implementation_version="2",
    )

    exact = stable_preference_for_benchmark_winner(
        record,
        (winner, alternate),
        cpu_implementation_id=plan.registered.request.reference.implementation_id,
    )
    assert exact.kind is NodePreferenceKind.IMPLEMENTATION
    assert exact.value == winner.implementation_id

    cpu_id = plan.registered.request.reference.implementation_id
    cpu_record = replace(record, accepted_implementation_id=cpu_id)
    cpu = stable_preference_for_benchmark_winner(
        cpu_record,
        (winner,),
        cpu_implementation_id=cpu_id,
    )
    assert cpu.kind is NodePreferenceKind.CPU

    failed_results = tuple(
        replace(item, parity_passed=False, error="parity failed")
        if item.implementation_id == winner.implementation_id
        else item
        for item in record.candidates
    )
    failed_record = replace(record, candidates=failed_results)
    with pytest.raises(ValueError, match="scientific parity"):
        stable_preference_for_benchmark_winner(
            failed_record,
            (winner,),
            cpu_implementation_id=cpu_id,
        )
