from __future__ import annotations

import contextlib
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.compute_benchmark_adapter as adapter_module
import napari_vipp.core.compute_benchmark_coordinator as coordinator_module
from napari_vipp._tests.test_compute_benchmark_adapter import (
    ManualClock,
    _FakeRuntime,
    _two_input_rl_spec,
)
from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    NodePreferenceKind,
)
from napari_vipp.core.compute_benchmark import (
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    BenchmarkMeasurementPhase,
    BenchmarkMeasurementProgress,
    JsonBenchmarkStore,
)
from napari_vipp.core.compute_benchmark_adapter import (
    build_registered_node_benchmark,
    workload_from_prepared_node_call,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    ApplicationNodeBenchmarkCoordinator,
    NodeBenchmarkPhase,
    NodeBenchmarkUnavailable,
    benchmark_environment_fingerprint,
    stable_preference_for_benchmark_winner,
)
from napari_vipp.core.compute_planning import (
    plan_compute_decisions,
    probe_compute_environment,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.operations import (
    median_filter,
    richardson_lucy_deconvolution,
)
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload


def _environment(**updates) -> ComputeEnvironment:
    values = {
        "os_name": "Windows",
        "python_implementation": "CPython",
        "python_version": "3.12",
        "python_abi": "cpython-312",
        "runtime_ids": ("cpu-numpy", "cuda-cupy"),
        "implementation_libraries": ("cpu", "cupyx"),
        "runtime_versions": (("cuda-cupy", "14.1.1"), ("cupyx", "14.1.1")),
        "runtime_probe_fingerprints": (("cuda-cupy", "fake-runtime-fingerprint"),),
        "runtime_metadata": (
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        "driver_version": "13030",
        "device_id": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_class": "nvidia-cuda",
        "device_metadata": (("compute_capability", "12.0"),),
        "memory_topology": "discrete",
        "total_accelerator_memory_bytes": 16 * 1024**3,
        "probe_status": "available",
    }
    values.update(updates)
    return ComputeEnvironment(**values)


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


def _rl_pipeline(
    image: np.ndarray,
    psf: np.ndarray,
) -> tuple[PrototypePipeline, str, str, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    image_source_id = next(iter(pipeline.nodes))
    psf_source = pipeline.add_node("input")
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    pipeline.set_param(deconvolution.id, "iterations", 1)
    pipeline.set_param(deconvolution.id, "filter_epsilon", 1e-8)
    assert pipeline.connect(
        image_source_id,
        deconvolution.id,
        target_port=0,
    ).success
    assert pipeline.connect(
        psf_source.id,
        deconvolution.id,
        target_port=1,
    ).success
    pipeline.run(
        image,
        input_metadata={"axes": "YX"},
        source_payloads={
            psf_source.id: SourcePayload(
                psf,
                {"axes": "YX"},
                "benchmark PSF",
            )
        },
    )
    return pipeline, image_source_id, psf_source.id, deconvolution.id


def _coordinator_with_fake_rl_runtime(
    tmp_path,
    monkeypatch,
    clock: ManualClock,
):
    runtime = _FakeRuntime(clock)
    specification = _two_input_rl_spec()
    registry = ComputeRegistry()
    observed_gpu_inputs: list[tuple[tuple[int, ...], ...]] = []
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementations_for_operation",
        lambda *_args, **_kwargs: (specification,),
    )
    monkeypatch.setattr(
        registry,
        "implementation_spec",
        lambda *_args, **_kwargs: specification,
    )
    monkeypatch.setattr(
        registry,
        "probe_runtime",
        lambda _runtime_id, refresh=False: runtime.probe(refresh=refresh),
    )

    def gpu(inputs, **kwargs):
        observed_gpu_inputs.append(tuple(item.array.shape for item in inputs))
        clock.advance(0.040)
        return runtime.allocate(
            richardson_lucy_deconvolution(
                [item.array for item in inputs],
                **kwargs,
            )
        )

    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    original_builder = build_registered_node_benchmark

    def measured_builder(call, **kwargs):
        cpu_function = call.cpu_function

        def slower_cpu(inputs, **cpu_kwargs):
            clock.advance(0.100)
            return cpu_function(inputs, **cpu_kwargs)

        return original_builder(
            replace(call, cpu_function=slower_cpu),
            **kwargs,
        )

    monkeypatch.setattr(
        coordinator_module,
        "build_registered_node_benchmark",
        measured_builder,
    )
    coordinator = ApplicationNodeBenchmarkCoordinator(
        registry,
        tmp_path / "rl-node-benchmarks.json",
        clock=clock,
    )
    return coordinator, runtime, specification, observed_gpu_inputs


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
    assert [item.phase for item in progress[:4]] == [
        NodeBenchmarkPhase.PREPARING,
        NodeBenchmarkPhase.ELIGIBILITY,
        NodeBenchmarkPhase.READY,
        NodeBenchmarkPhase.BENCHMARKING,
    ]
    assert all(item.phase is NodeBenchmarkPhase.BENCHMARKING for item in progress[3:-1])
    assert progress[-1].phase is NodeBenchmarkPhase.COMPLETE
    measurements = [item for item in progress if item.measurement_total]
    assert measurements
    assert {item.measurement_phase for item in measurements} >= {
        BenchmarkMeasurementPhase.PARITY_COLD.value,
        BenchmarkMeasurementPhase.WARMUP.value,
        BenchmarkMeasurementPhase.PAIRED_WARM.value,
    }
    assert all(
        0 <= item.measurement_completed <= item.measurement_total
        and item.measurement_message
        and item.implementation_id
        and item.implementation_version
        for item in measurements
    )
    assert any(
        item.measurement_phase == BenchmarkMeasurementPhase.PAIRED_WARM.value
        and item.measurement_completed == item.measurement_total
        for item in measurements
    )
    assert pipeline.nodes[node_id].params == params_before
    assert pipeline.completed_node_ids == completed_before
    np.testing.assert_array_equal(pipeline.outputs[source_id], source_before)
    np.testing.assert_array_equal(pipeline.outputs[node_id], node_output_before)


def test_selected_node_benchmark_qualifies_secondary_nvidia_hardware(
    tmp_path,
    monkeypatch,
):
    clock = ManualClock()
    coordinator, runtime = _coordinator_with_fake_runtime(
        tmp_path,
        monkeypatch,
        clock,
    )
    pipeline, _source_id, node_id = _median_pipeline(
        np.arange(31 * 37, dtype=np.uint16).reshape(31, 37)
    )
    environment = _environment(
        device_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        device_metadata=(("compute_capability", "8.9"),),
        total_accelerator_memory_bytes=6 * 1024**3,
    )

    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=environment,
        allow_experimental=True,
        paired_bootstrap_samples=200,
        time_budget_seconds=10.0,
    )
    result = coordinator.run(plan)

    assert plan.environment == environment
    assert [spec.implementation_id for spec in plan.admitted_specs] == [
        "cupyx-median-filter-v1"
    ]
    assert result.record.accepted_implementation_id == "cupyx-median-filter-v1"
    candidate = next(
        item
        for item in result.record.candidates
        if item.implementation_id == "cupyx-median-filter-v1"
    )
    assert candidate.parity_passed
    replanned = plan_compute_decisions(
        ComputeRequest(
            mode="custom",
            node_preferences={node_id: result.winner_preference},
            allow_experimental=True,
        ),
        (workload_from_prepared_node_call(plan.registered.detached_call),),
        registry=coordinator.registry,
        environment=environment,
    )
    assert replanned.decisions[0].decision_kind is DecisionKind.SELECTED
    assert replanned.decisions[0].runtime_id == "cuda-cupy"
    assert not replanned.decisions[0].fallback_used
    assert runtime.live == {}


@pytest.mark.real_cuda
def test_real_node_benchmark_parity_on_current_compatible_cuda_device(tmp_path):
    registry = ComputeRegistry()
    try:
        values = np.arange(256 * 320, dtype=np.uint16).reshape(256, 320)
        pipeline, _source_id, node_id = _median_pipeline(values)
        specs = registry.implementations_for_operation(
            "median_filter",
            allow_experimental=True,
        )
        request = ComputeRequest(
            mode="custom",
            node_preferences={node_id: "library:cupyx"},
            allow_experimental=True,
        )
        environment, _warnings = probe_compute_environment(
            registry,
            request,
            specs,
        )
        if "cuda-cupy" not in environment.runtime_ids:
            pytest.skip(environment.probe_reason or "CUDA/CuPy is unavailable.")
        if "cupyx" not in environment.implementation_libraries:
            pytest.skip(environment.probe_reason or "CuPyX is unavailable.")

        coordinator = ApplicationNodeBenchmarkCoordinator(
            registry,
            tmp_path / "real-compatible-device-benchmarks.json",
        )
        plan = coordinator.prepare(
            pipeline,
            node_id,
            environment=environment,
            allow_experimental=True,
            time_budget_seconds=180.0,
        )
        result = coordinator.run(plan)

        assert [spec.implementation_id for spec in plan.admitted_specs] == [
            "cupyx-median-filter-v1"
        ]
        candidate = next(
            item
            for item in result.record.candidates
            if item.implementation_id == "cupyx-median-filter-v1"
        )
        assert candidate.parity_passed, candidate.error
        assert candidate.synchronized
        assert result.record.key.environment_fingerprint == (
            benchmark_environment_fingerprint(environment)
        )
    finally:
        registry.close()


def test_run_owns_transaction_lease_and_charges_wait_against_budget(
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
        warm_rounds=3,
        max_warm_rounds=3,
        paired_bootstrap_samples=200,
        time_budget_seconds=10.0,
    )
    outer_active = False
    outer_arguments = []
    nested_count = 0
    observed_service_budgets = []

    @contextlib.contextmanager
    def outer_lease(
        runtime_id,
        device_id,
        *,
        cancelled=None,
        deadline=None,
        clock=None,
    ):
        nonlocal outer_active
        assert cancelled is not None
        assert clock is not None
        assert cancelled() is False
        outer_arguments.append((runtime_id, device_id, deadline))
        clock.advance(0.250)
        assert cancelled() is False
        outer_active = True
        try:
            yield None
        finally:
            outer_active = False

    @contextlib.contextmanager
    def nested_lease(_runtime_id, _device_id, *, cancelled=None, **_kwargs):
        nonlocal nested_count
        assert outer_active
        assert cancelled is not None
        assert cancelled() is False
        nested_count += 1
        yield None

    delegate = coordinator.service

    class CapturingService:
        def benchmark(self, request, *, cancelled=None, progress=None):
            assert outer_active
            observed_service_budgets.append(request.time_budget_seconds)
            return delegate.benchmark(
                request,
                cancelled=cancelled,
                progress=progress,
            )

    coordinator.service = CapturingService()
    monkeypatch.setattr(coordinator_module, "accelerator_lease", outer_lease)
    monkeypatch.setattr(adapter_module, "accelerator_lease", nested_lease)

    coordinator.run(plan)

    assert outer_arguments == [("cuda-cupy", "cuda:0", pytest.approx(10.0))]
    assert observed_service_budgets == [pytest.approx(9.750)]
    assert nested_count > 0
    assert not outer_active


def test_transaction_lease_wait_budget_exhaustion_publishes_nothing(
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
        time_budget_seconds=0.5,
    )

    @contextlib.contextmanager
    def exhausted_lease(
        _runtime_id,
        _device_id,
        *,
        cancelled=None,
        clock=None,
        **_kwargs,
    ):
        assert cancelled is not None
        assert clock is not None
        clock.advance(0.5)
        cancelled()
        raise AssertionError("an expired lease waiter must not acquire")
        yield None

    monkeypatch.setattr(
        coordinator_module,
        "accelerator_lease",
        exhausted_lease,
    )

    with pytest.raises(BenchmarkBudgetExceeded, match="accelerator ownership"):
        coordinator.run(plan)

    assert coordinator.store.records() == ()


def test_selected_multi_input_node_uses_both_ports_and_invalidates_on_psf_change(
    tmp_path,
    monkeypatch,
):
    clock = ManualClock()
    (
        coordinator,
        runtime,
        specification,
        observed_gpu_inputs,
    ) = _coordinator_with_fake_rl_runtime(tmp_path, monkeypatch, clock)
    image = np.zeros((9, 11), dtype=np.float32)
    image[4, 5] = 1.0
    psf = np.array(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=np.float32,
    )
    pipeline, image_source_id, psf_source_id, node_id = _rl_pipeline(image, psf)
    image_source = pipeline.outputs[image_source_id]
    psf_source = pipeline.outputs[psf_source_id]

    plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        warm_rounds=3,
        max_warm_rounds=3,
        paired_bootstrap_samples=200,
    )

    assert plan.registered.detached_call.multiple_inputs
    assert plan.registered.request.workload.input_shapes == (
        image.shape,
        psf.shape,
    )
    assert plan.registered.request.workload.input_dtypes == (
        "float32",
        "float32",
    )
    assert len(plan.registered.detached_call.inputs) == 2
    assert not np.shares_memory(
        plan.registered.detached_call.inputs[0],
        image_source,
    )
    assert not np.shares_memory(
        plan.registered.detached_call.inputs[1],
        psf_source,
    )
    assert coordinator.workload_is_current(pipeline, plan)

    result = coordinator.run(plan)

    candidate = next(
        item
        for item in result.record.candidates
        if item.implementation_id == specification.implementation_id
    )
    assert candidate.parity_passed, candidate.error
    assert candidate.cold_transfer_seconds == pytest.approx(0.006)
    assert candidate.peak_runtime_reserved_bytes >= (
        image.nbytes + psf.nbytes + image.nbytes
    )
    assert observed_gpu_inputs
    assert set(observed_gpu_inputs) == {(image.shape, psf.shape)}
    assert runtime.live == {}

    psf_source[0, 1] += np.float32(0.01)
    assert not coordinator.workload_is_current(pipeline, plan)
    psf_source[0, 1] -= np.float32(0.01)
    assert coordinator.workload_is_current(pipeline, plan)
    image_source[0, 0] += np.float32(0.01)
    assert not coordinator.workload_is_current(pipeline, plan)


def test_selected_node_benchmark_refuses_writers_before_runtime_or_io(
    tmp_path,
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    writer = pipeline.add_node("save_output")
    assert pipeline.connect("input", writer.id).success
    registry = ComputeRegistry()

    def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("writer refusal must happen before runtime access")

    monkeypatch.setattr(registry, "runtime", unexpected_runtime)
    monkeypatch.setattr(registry, "probe_runtime", unexpected_runtime)
    coordinator = ApplicationNodeBenchmarkCoordinator(
        registry,
        tmp_path / "writer-benchmarks.json",
    )

    with pytest.raises(NodeBenchmarkUnavailable, match="writer operations"):
        coordinator.prepare(pipeline, writer.id, allow_experimental=True)

    assert not (tmp_path / "writer-benchmarks.json").exists()


def test_internal_operation_progress_is_forwarded_with_a_readable_backend_label(
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
        time_budget_seconds=10.0,
    )
    delegate = coordinator.service

    class ServiceWithInternalProgress:
        def benchmark(self, request, *, cancelled=None, progress=None):
            assert progress is not None
            progress(
                BenchmarkMeasurementProgress(
                    phase=BenchmarkMeasurementPhase.PARITY_COLD,
                    implementation_id=f"cpu-{plan.operation_id}-v1",
                    implementation_version="1",
                    completed=0,
                    total=1,
                    message="Measuring the CPU reference.",
                    operation_completed=37,
                    operation_total=171,
                    operation_message="Rolling-ball background",
                )
            )
            return delegate.benchmark(request, cancelled=cancelled)

    coordinator.service = ServiceWithInternalProgress()
    updates = []

    coordinator.run(plan, progress=updates.append)

    internal = next(item for item in updates if item.operation_total)
    assert (internal.operation_completed, internal.operation_total) == (37, 171)
    assert internal.operation_message == (
        "CPU: Rolling-ball background (37 of 171) — "
        "scientific parity + cold timing."
    )
    assert internal.measurement_phase == BenchmarkMeasurementPhase.PARITY_COLD.value
    assert internal.implementation_id == f"cpu-{plan.operation_id}-v1"


def test_exact_saved_record_is_reused_across_different_abort_budgets(
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
    first_plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        time_budget_seconds=10.0,
    )
    first = coordinator.run(first_plan)
    second_plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        time_budget_seconds=30.0,
    )
    before_lookup = clock.value

    reused = coordinator.cached_result(second_plan)

    assert reused is not None
    assert reused.record == first.record
    assert second_plan.key_digest == first_plan.key_digest
    assert clock.value == before_lookup


def test_exact_saved_record_reuses_completed_candidate_rejection(
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
    )
    measured = coordinator.run(plan).record
    cpu_id = plan.registered.request.reference.implementation_id
    gpu_id = plan.registered.request.candidates[0].implementation_id
    rejected = replace(
        measured,
        candidates=tuple(
            replace(
                candidate,
                parity_passed=False,
                error="scientific mismatch",
                failure_kind=(BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY),
            )
            if candidate.implementation_id == gpu_id
            else candidate
            for candidate in measured.candidates
        ),
        accepted_implementation_id=cpu_id,
    )
    coordinator.store.put(rejected)
    before_lookup = clock.value

    reused = coordinator.cached_result(plan)

    assert reused is not None
    assert reused.record == rejected
    assert reused.winner_preference.kind is NodePreferenceKind.CPU
    assert clock.value == before_lookup

    transient = replace(
        rejected,
        candidates=tuple(
            replace(
                candidate,
                error="out_of_memory: retryable",
                failure_kind=(BenchmarkCandidateFailureKind.TRANSIENT_RUNTIME),
            )
            if candidate.implementation_id == gpu_id
            else candidate
            for candidate in rejected.candidates
        ),
    )
    coordinator.store.put(transient)

    assert coordinator.cached_result(plan) is None

    censored = replace(
        measured,
        candidates=tuple(
            replace(
                candidate,
                timing_censored=True,
                timing_lower_bound_seconds=0.05,
                timing_censor_reason="CPU exceeded the decisive GPU bound.",
                timing_censor_incumbent_id=gpu_id,
            )
            if candidate.implementation_id == cpu_id
            else candidate
            for candidate in measured.candidates
        ),
        accepted_implementation_id=gpu_id,
    )
    coordinator.store.put(censored)

    assert coordinator.cached_result(plan) is None


def test_prepare_opt_in_separates_adaptive_censored_policy_identity(
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

    exact = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
    )
    adaptive = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        adaptive_candidate_stopping=True,
    )

    assert not exact.registered.request.adaptive_candidate_stopping
    assert adaptive.registered.request.adaptive_candidate_stopping
    assert exact.key_digest != adaptive.key_digest


def test_adaptive_plan_reuses_stronger_exact_uncensored_record(
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
    exact_plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
    )
    exact_record = coordinator.run(exact_plan).record
    adaptive_plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        adaptive_candidate_stopping=True,
    )
    before_lookup = clock.value

    reused = coordinator.cached_result(adaptive_plan)

    assert reused is not None
    assert reused.plan == adaptive_plan
    assert reused.record.key == adaptive_plan.registered.request.key
    assert reused.record.candidates == exact_record.candidates
    assert not any(item.timing_censored for item in reused.record.candidates)
    assert coordinator.store.get(adaptive_plan.registered.request.key) is None
    assert clock.value == before_lookup


def test_exact_profile_never_reuses_adaptive_or_censored_compatible_record(
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
    exact_plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
    )
    measured = coordinator.run(exact_plan).record
    adaptive_plan = coordinator.prepare(
        pipeline,
        node_id,
        environment=_environment(),
        allow_experimental=True,
        adaptive_candidate_stopping=True,
    )
    coordinator.store.discard(exact_plan.registered.request.key)
    coordinator.store.put(
        replace(measured, key=adaptive_plan.registered.request.key)
    )

    assert coordinator.cached_result(exact_plan) is None
    coordinator.store.discard(adaptive_plan.registered.request.key)

    cpu_id = exact_plan.registered.request.reference.implementation_id
    gpu_id = exact_plan.registered.request.candidates[0].implementation_id
    coordinator.store.put(
        replace(
            measured,
            candidates=tuple(
                replace(
                    candidate,
                    timing_censored=True,
                    timing_lower_bound_seconds=0.05,
                    timing_censor_reason="CPU exceeded the decisive GPU bound.",
                    timing_censor_incumbent_id=gpu_id,
                )
                if candidate.implementation_id == cpu_id
                else candidate
                for candidate in measured.candidates
            ),
            accepted_implementation_id=gpu_id,
        )
    )

    assert coordinator.cached_result(adaptive_plan) is None


def test_benchmark_environment_identity_includes_cpu_scientific_stack(
    monkeypatch,
):
    environment = _environment()
    original = benchmark_environment_fingerprint(environment)
    monkeypatch.setattr(
        coordinator_module.importlib.metadata,
        "version",
        lambda distribution: f"changed-{distribution}",
    )

    assert benchmark_environment_fingerprint(environment) != original


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
