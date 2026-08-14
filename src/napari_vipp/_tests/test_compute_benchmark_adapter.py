from __future__ import annotations

import contextlib
import gc
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.compute_benchmark_adapter as adapter_module
from napari_vipp.core.compute import (
    MemoryTopology,
    WorkloadDescriptor,
    canonical_digest,
)
from napari_vipp.core.compute_benchmark import (
    BenchmarkCancelled,
    BenchmarkImplementation,
    BenchmarkMeasurementPhase,
    BenchmarkProgressError,
    JsonBenchmarkStore,
    NodeBenchmarkRequest,
    NodeBenchmarkService,
    benchmark_record_staleness,
    paired_bootstrap_speedup,
)
from napari_vipp.core.compute_benchmark_adapter import (
    CUSTOM_BENCHMARK_POLICY_ID,
    PRODUCTION_BENCHMARK_POLICY_ID,
    build_registered_node_benchmark,
    detach_prepared_node_call,
    operation_parity,
    workload_from_prepared_node_call,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    RuntimeDevice,
    RuntimeExceptionInfo,
    RuntimeExceptionKind,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import accelerator_compute_specs
from napari_vipp.core.measurements import basic_measurement_layout
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.operations import (
    gaussian_blur,
    measure_objects,
    median_filter,
    richardson_lucy_deconvolution,
    subtract_background,
)
from napari_vipp.core.tables import TableData


def _identity_cpu(value, **_kwargs):
    return np.array(value, copy=True)


def _cupy_benchmark_kernel_oom_or_copy(value, *, sigma=0.0, **_kwargs):
    """Retain traceback-local scratch before a private-pool CuPy OOM."""

    import cupy

    if float(sigma) != 0.0:
        return value.copy()
    scratch = cupy.empty_like(value)
    assert scratch.shape == value.shape
    return cupy.empty((32 * 1024**2,), dtype=cupy.uint8)


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class _FakeDeviceArray:
    array: np.ndarray
    allocation: int


class _FakeOOM(RuntimeError):
    pass


class _FakeTracebackScratch:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        runtime.traceback_scratch_live += 1

    def __del__(self) -> None:
        self.runtime.traceback_scratch_live -= 1


class _NonCopyableProgress:
    def __deepcopy__(self, _memo):
        raise AssertionError("live progress must be stripped before deepcopy")


class _FakeRuntime:
    runtime_id = "cuda-cupy"
    array_domain = "cuda-cupy"

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.active = False
        self.scope_count = 0
        self.synchronize_count = 0
        self.next_allocation = 1
        self.live: dict[int, _FakeDeviceArray] = {}
        self.release_counts: Counter[int] = Counter()
        self.scope_peak_reserved = 0
        self.traceback_scratch_live = 0
        self.classified_inside_scope: list[bool] = []
        self.scope_arguments: list[dict[str, object]] = []

    @contextlib.contextmanager
    def execution_scope(self, **kwargs):
        assert not self.active
        self.active = True
        self.scope_count += 1
        self.scope_peak_reserved = 0
        self.scope_arguments.append(dict(kwargs))
        try:
            yield None
        finally:
            gc.collect()
            assert self.live == {}
            assert self.traceback_scratch_live == 0
            self.active = False

    def probe(self, *, refresh=False):
        del refresh
        return RuntimeProbeResult(
            self.runtime_id,
            True,
            version="fake-1",
            devices=(
                RuntimeDevice("cuda:0", "Fake CUDA 0", 8 * 1024**3),
                RuntimeDevice("cuda:1", "Fake CUDA 1", 8 * 1024**3),
            ),
            selected_device_id="cuda:0",
        )

    def to_device(self, value, *, device_id=""):
        del device_id
        self.clock.advance(0.002)
        return self.allocate(np.asarray(value))

    def to_host(self, value):
        self._require(value)
        self.clock.advance(0.002)
        return np.array(value.array, copy=True)

    def allocate(self, value, *, alias: _FakeDeviceArray | None = None):
        assert self.active
        if alias is None:
            allocation = self.next_allocation
            self.next_allocation += 1
        else:
            allocation = alias.allocation
        result = _FakeDeviceArray(np.array(value, copy=True), allocation)
        self.live.setdefault(allocation, result)
        self.scope_peak_reserved = max(
            self.scope_peak_reserved,
            sum(item.array.nbytes for item in self.live.values()),
        )
        return result

    def alias(self, value: _FakeDeviceArray):
        self._require(value)
        return self.allocate(value.array, alias=value)

    def is_device_value(self, value):
        return isinstance(value, _FakeDeviceArray)

    def allocation_identity(self, value):
        self._require(value)
        return value.allocation

    def release(self, value):
        self._require(value)
        allocation = value.allocation
        self.release_counts[allocation] += 1
        if self.release_counts[allocation] != 1:
            raise AssertionError("allocation released more than once")
        self.live.pop(allocation)

    def synchronize(self, *, device_id=""):
        del device_id
        assert self.active
        self.synchronize_count += 1

    def memory_snapshot(self, *, device_id=""):
        live = sum(item.array.nbytes for item in self.live.values())
        reserved = self.scope_peak_reserved if self.active else 0
        return RuntimeMemorySnapshot(
            self.runtime_id,
            device_id or "cuda:0",
            MemoryTopology.DISCRETE,
            device_total_bytes=8 * 1024**3,
            device_free_bytes=7 * 1024**3,
            runtime_live_bytes=live,
            runtime_reserved_bytes=max(live, reserved),
            out_of_pool_bytes=64 if self.active else 0,
        )

    def classify_exception(self, exc):
        self.classified_inside_scope.append(self.active)
        if isinstance(exc, _FakeOOM):
            return RuntimeExceptionInfo(
                RuntimeExceptionKind.OUT_OF_MEMORY,
                "fake_oom",
                str(exc),
                exception_type=type(exc).__name__,
                retryable=True,
            )
        return RuntimeExceptionInfo(
            RuntimeExceptionKind.KERNEL_FAILURE,
            "fake_kernel_failure",
            str(exc),
            exception_type=type(exc).__name__,
            retryable=False,
        )

    def _require(self, value):
        if not self.active or not isinstance(value, _FakeDeviceArray):
            raise TypeError("value is not owned by the active fake scope")
        if value.allocation not in self.live:
            raise RuntimeError("allocation is no longer live")


def _spec(operation_id: str):
    return next(
        spec
        for spec in accelerator_compute_specs()
        if spec.operation_id == operation_id
    )


def _two_input_rl_spec(operation_id: str = "richardson_lucy_deconvolution"):
    return _spec(operation_id)


def _fake_multi_input_registered_benchmark(
    monkeypatch,
    operation_id: str = "richardson_lucy_deconvolution",
):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    spec = _two_input_rl_spec(operation_id)
    registry = ComputeRegistry()
    calls: list[tuple[str, tuple[tuple[int, ...], ...]]] = []

    def cpu(inputs, **_kwargs):
        image, psf = inputs
        calls.append(("cpu", (image.shape, psf.shape)))
        clock.advance(0.100)
        return np.asarray(image) * np.asarray(psf).sum(dtype=np.float32)

    def gpu(inputs, **_kwargs):
        image, psf = inputs
        calls.append(("gpu", (image.array.shape, psf.array.shape)))
        clock.advance(0.040)
        return runtime.allocate(
            image.array * psf.array.sum(dtype=np.float32)
        )

    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementation_spec",
        lambda *_args, **_kwargs: spec,
    )
    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    image = np.arange(7 * 9, dtype=np.float32).reshape(7, 9) / 63
    psf = np.array(
        [[0.0, 0.125, 0.0], [0.125, 0.5, 0.125], [0.0, 0.125, 0.0]],
        dtype=np.float32,
    )
    call = PreparedNodeCall(
        "rl-node",
        operation_id,
        cpu,
        (image, psf),
        kwargs={
            "iterations": (
                10 if operation_id == "richardson_lucy_tv_deconvolution" else 3
            ),
            "resolved_spatial_ndim": 2,
            "progress": None,
        },
        multiple_inputs=True,
    )
    built = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-multi-input-environment",
        allow_experimental=True,
        clock=clock,
        warm_rounds=3,
        max_warm_rounds=3,
        paired_bootstrap_samples=200,
    )
    return clock, runtime, registry, spec, image, psf, calls, built


def _fake_registered_benchmark(
    monkeypatch,
    *,
    size: int = 3,
    alias_output: bool = False,
):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    spec = _spec("median_filter")
    registry = ComputeRegistry()

    def cpu(data, **kwargs):
        clock.advance(0.100)
        return median_filter(data, **kwargs)

    def gpu(device, **kwargs):
        clock.advance(0.040)
        if alias_output:
            assert kwargs["size"] == 1
            return runtime.alias(device)
        result = median_filter(device.array, **kwargs)
        return runtime.allocate(result)

    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    live = np.arange(17 * 19, dtype=np.float32).reshape(17, 19)
    live[0, 0] = -0.0
    live.setflags(write=False)
    call = PreparedNodeCall(
        "median-node",
        "median_filter",
        cpu,
        (live,),
        kwargs={"size": size, "channel_axis": None},
    )
    built = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-exact-environment",
        allow_experimental=True,
        clock=clock,
        paired_bootstrap_samples=200,
    )
    return clock, runtime, registry, live, built


@pytest.mark.parametrize("dtype", (">u2", ">f4"))
def test_workload_identity_preserves_non_native_byte_order(dtype: str) -> None:
    values = np.arange(25, dtype=dtype).reshape(5, 5)
    call = PreparedNodeCall(
        "sigma-node",
        "sigma_filter",
        _identity_cpu,
        (values,),
        kwargs={"radius": 2.0, "channel_axis": None},
    )

    workload = workload_from_prepared_node_call(call)

    assert workload.input_dtypes == (np.dtype(dtype).str,)


def test_workload_identity_omits_runtime_private_parameters() -> None:
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    cpu_kwargs = {}

    def cpu_reference(value, **kwargs):
        cpu_kwargs.update(kwargs)
        return np.array(value, copy=True)

    call = PreparedNodeCall(
        "coloc-node",
        "colocalized_voxels",
        cpu_reference,
        (values,),
        kwargs={
            "threshold_mode": "Costes auto",
            "_vipp_resolved_costes": {"pearson_below": float("nan")},
        },
    )

    workload = workload_from_prepared_node_call(call)

    assert dict(workload.parameters) == {"threshold_mode": "Costes auto"}
    assert adapter_module._provider_keyword_arguments(call) == {
        "threshold_mode": "Costes auto"
    }
    adapter_module._execute_cpu_reference(call)
    assert "_vipp_resolved_costes" in cpu_kwargs


def test_detached_capture_and_hash_are_read_only_and_promptly_abortable():
    values = np.arange(2_000_000, dtype=np.uint16).reshape(1000, 2000)
    call = PreparedNodeCall(
        "median-node",
        "median_filter",
        _identity_cpu,
        (values,),
        kwargs={"size": 3, "channel_axis": None},
    )
    checks = 0

    def abort_copy() -> None:
        nonlocal checks
        checks += 1
        if checks >= 4:
            raise BenchmarkCancelled("capture cancelled")

    with pytest.raises(BenchmarkCancelled, match="capture cancelled"):
        detach_prepared_node_call(call, check_abort=abort_copy)

    detached = detach_prepared_node_call(call)
    assert not detached.inputs[0].flags.writeable
    assert not np.shares_memory(detached.inputs[0], values)
    checks = 0

    def abort_hash() -> None:
        nonlocal checks
        checks += 1
        if checks >= 4:
            raise BenchmarkCancelled("hash cancelled")

    with pytest.raises(BenchmarkCancelled, match="hash cancelled"):
        workload_from_prepared_node_call(detached, check_abort=abort_hash)


def test_trusted_detached_fast_path_rejects_mutable_arrays(monkeypatch):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: _identity_cpu,
    )
    call = PreparedNodeCall(
        "median-node",
        "median_filter",
        _identity_cpu,
        (np.arange(25, dtype=np.uint16).reshape(5, 5),),
        kwargs={"size": 3, "channel_axis": None},
    )

    with pytest.raises(ValueError, match="read-only"):
        build_registered_node_benchmark(
            call,
            admitted_specs=(_spec("median_filter"),),
            registry=registry,
            environment_fingerprint="fake-exact-environment",
            allow_experimental=True,
            call_is_detached=True,
        )


def test_registered_adapter_is_transactional_synchronized_and_memory_observed(
    monkeypatch,
):
    clock, runtime, _registry, live, built = _fake_registered_benchmark(monkeypatch)
    before = live.copy()

    record = NodeBenchmarkService(clock=clock).benchmark(built.request)

    results = {result.implementation_id: result for result in record.candidates}
    candidate = results["cupyx-median-filter-v1"]
    assert record.accepted_implementation_id == "cupyx-median-filter-v1"
    assert candidate.parity_passed
    assert candidate.cold_seconds == pytest.approx(0.044)
    assert candidate.warm_seconds == pytest.approx((0.044,) * 7)
    assert candidate.cold_transfer_seconds == pytest.approx(0.004)
    assert candidate.warm_transfer_seconds == pytest.approx((0.004,) * 7)
    assert candidate.cold_resident_seconds == pytest.approx(0.040)
    assert candidate.warm_resident_seconds == pytest.approx((0.040,) * 7)
    assert candidate.timing_scope == "synchronized-end-to-end-v1"
    assert candidate.synchronized
    assert candidate.transfers_included
    assert candidate.peak_runtime_live_bytes > 0
    assert candidate.peak_runtime_reserved_bytes >= candidate.peak_runtime_live_bytes
    assert candidate.peak_out_of_pool_bytes == 64
    assert candidate.peak_memory_bytes == (
        candidate.peak_runtime_reserved_bytes + candidate.peak_out_of_pool_bytes
    )
    assert candidate.paired_speedup_median == pytest.approx(0.100 / 0.044)
    assert candidate.paired_speedup_lower_confidence_bound == pytest.approx(
        0.100 / 0.044
    )
    # cold/parity + one untimed warmup + seven paired rounds
    assert runtime.scope_count == 9
    assert len(built.observations.runs("cupyx-median-filter-v1")) == 9
    assert all(
        observation.cleanup_succeeded
        and observation.terminal_snapshot.runtime_live_bytes == 0
        and observation.terminal_snapshot.runtime_reserved_bytes == 0
        for observation in built.observations.runs("cupyx-median-filter-v1")
    )
    assert runtime.live == {}
    assert set(runtime.release_counts.values()) == {1}
    np.testing.assert_array_equal(live, before)
    assert not live.flags.writeable


def test_measurement_finalizer_runs_after_cleanup_and_reports_its_timing(
    monkeypatch,
):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    spec = _spec("measure_objects")
    labels = np.zeros((7, 9), dtype=np.int32)
    labels[1:4, 2:6] = 17
    labels[5:7, 7:9] = 9001
    kwargs = {
        "spatial_mode": "2D YX",
        "resolved_spatial_ndim": 2,
        "axis_names": ("y", "x"),
        "axis_types": ("space", "space"),
        "axis_scales": (0.25, 0.5),
        "axis_units": ("um", "um"),
        "measurement_set": "Basic morphology",
        "source_name": "benchmark labels",
        "progress": None,
    }
    expected = measure_objects(labels, **kwargs)
    layout = basic_measurement_layout(
        labels.shape,
        spatial_mode="2D YX",
        resolved_spatial_ndim=2,
        axis_names=("y", "x"),
        axis_types=("space", "space"),
        axis_scales=(0.25, 0.5),
        axis_units=("um", "um"),
    )
    indexes = {name: index for index, name in enumerate(expected.columns)}
    packed = np.asarray(
        [
            [float(row[indexes[column]]) for column in layout.packed_columns]
            for row in expected.rows
        ],
        dtype=np.float64,
    ).reshape(expected.row_count, layout.packed_width)

    def cpu(data, **parameters):
        clock.advance(0.100)
        return measure_objects(data, **parameters)

    def gpu(device, **_parameters):
        assert runtime.active
        assert device.array.dtype == np.dtype(np.int32)
        clock.advance(0.040)
        return runtime.allocate(packed)

    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementation_spec",
        lambda *_args, **_kwargs: spec,
    )
    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    call = PreparedNodeCall(
        "measurement-node",
        "measure_objects",
        cpu,
        (labels,),
        input_states=(SimpleNamespace(shape=labels.shape),),
        kwargs=kwargs,
    )
    original_apply = adapter_module.apply_host_finalizer
    finalizer_events: list[tuple[bool, int, tuple[object, ...]]] = []

    def observed_finalizer(reference, host_outputs, finalized_call):
        # The mixed-type public table must be constructed only after the
        # provider scope has released every runtime-owned allocation.
        finalizer_events.append(
            (runtime.active, len(runtime.live), finalized_call.inputs)
        )
        assert not runtime.active
        assert runtime.live == {}
        assert finalized_call.inputs == (None,)
        assert isinstance(host_outputs[0], np.ndarray)
        clock.advance(0.007)
        result = original_apply(reference, host_outputs, finalized_call)
        assert len(result) == 1
        assert isinstance(result[0], TableData)
        assert result[0] == expected
        return result

    monkeypatch.setattr(
        adapter_module,
        "apply_host_finalizer",
        observed_finalizer,
    )
    built = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-measurement-environment",
        allow_experimental=True,
        clock=clock,
        warm_rounds=3,
        max_warm_rounds=3,
        paired_bootstrap_samples=200,
    )

    record = NodeBenchmarkService(clock=clock).benchmark(built.request)

    candidate = next(
        result
        for result in record.candidates
        if result.implementation_id == spec.implementation_id
    )
    assert candidate.parity_passed, candidate.error
    assert record.accepted_implementation_id == spec.implementation_id
    assert candidate.timing_scope == "synchronized-end-to-end-host-finalized-v1"
    assert candidate.cold_seconds == pytest.approx(0.051)
    assert candidate.warm_seconds == pytest.approx((0.051,) * 3)
    assert candidate.cold_transfer_seconds == pytest.approx(0.004)
    assert candidate.warm_transfer_seconds == pytest.approx((0.004,) * 3)
    assert candidate.cold_resident_seconds == pytest.approx(0.040)
    assert candidate.warm_resident_seconds == pytest.approx((0.040,) * 3)
    assert candidate.cold_host_materialization_seconds == pytest.approx(0.007)
    assert candidate.warm_host_materialization_seconds == pytest.approx((0.007,) * 3)
    assert len(finalizer_events) == runtime.scope_count
    assert set(finalizer_events) == {(False, 0, (None,))}
    assert all(
        observation.cleanup_succeeded
        and observation.terminal_snapshot.runtime_live_bytes == 0
        and observation.terminal_snapshot.runtime_reserved_bytes == 0
        for observation in built.observations.runs(spec.implementation_id)
    )


def test_multi_input_benchmark_detaches_hashes_invokes_and_observes_every_port(
    monkeypatch,
):
    (
        clock,
        runtime,
        _registry,
        spec,
        image,
        psf,
        calls,
        built,
    ) = _fake_multi_input_registered_benchmark(monkeypatch)
    image_before = image.copy()
    psf_before = psf.copy()

    assert built.request.workload.input_shapes == (image.shape, psf.shape)
    assert built.request.workload.input_dtypes == ("float32", "float32")
    assert len(built.detached_call.inputs) == 2
    assert all(not value.flags.writeable for value in built.detached_call.inputs)
    assert not np.shares_memory(built.detached_call.inputs[0], image)
    assert not np.shares_memory(built.detached_call.inputs[1], psf)
    private_call = built.request.private_input_factory()
    assert private_call.multiple_inputs
    assert len(private_call.positional_input()) == 2
    assert all(not value.flags.writeable for value in private_call.inputs)

    record = NodeBenchmarkService(clock=clock).benchmark(built.request)

    candidate = next(
        item
        for item in record.candidates
        if item.implementation_id == spec.implementation_id
    )
    assert candidate.parity_passed, candidate.error
    assert candidate.cold_seconds == pytest.approx(0.046)
    assert candidate.warm_seconds == pytest.approx((0.046,) * 3)
    # Two H2D transfers and one D2H transfer are included in every GPU call.
    assert candidate.cold_transfer_seconds == pytest.approx(0.006)
    assert candidate.warm_transfer_seconds == pytest.approx((0.006,) * 3)
    assert candidate.cold_resident_seconds == pytest.approx(0.040)
    assert candidate.peak_runtime_reserved_bytes >= (
        image.nbytes + psf.nbytes + image.nbytes
    )
    assert {kind for kind, _shapes in calls} == {"cpu", "gpu"}
    assert {shapes for _kind, shapes in calls} == {(image.shape, psf.shape)}
    assert runtime.live == {}
    assert set(runtime.release_counts.values()) == {1}
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(psf, psf_before)


def test_rl_tv_multi_input_benchmark_uses_registered_positive_tv_parity(monkeypatch):
    clock, runtime, _registry, spec, _image, _psf, _calls, built = (
        _fake_multi_input_registered_benchmark(
            monkeypatch,
            "richardson_lucy_tv_deconvolution",
        )
    )
    record = NodeBenchmarkService(clock=clock).benchmark(built.request)

    candidate = next(
        item
        for item in record.candidates
        if item.implementation_id == spec.implementation_id
    )
    assert candidate.parity_passed, candidate.error
    assert record.accepted_implementation_id == spec.implementation_id
    assert runtime.live == {}


def test_multi_input_exact_identity_changes_when_only_psf_changes(monkeypatch):
    (
        _clock,
        _runtime,
        registry,
        spec,
        image,
        psf,
        _calls,
        first,
    ) = _fake_multi_input_registered_benchmark(monkeypatch)
    changed_psf = psf.copy()
    changed_psf[0, 1] += np.float32(0.01)
    changed = PreparedNodeCall(
        "rl-node",
        "richardson_lucy_deconvolution",
        first.detached_call.cpu_function,
        (image, changed_psf),
        kwargs={
            "iterations": 3,
            "resolved_spatial_ndim": 2,
            "progress": None,
        },
        multiple_inputs=True,
    )

    second = build_registered_node_benchmark(
        changed,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-multi-input-environment",
        allow_experimental=True,
    )

    assert (
        first.request.workload.facts_fingerprint
        != second.request.workload.facts_fingerprint
    )
    assert first.request.workload.fingerprint != second.request.workload.fingerprint
    assert first.request.key.digest != second.request.key.digest


def test_multi_input_second_transfer_failure_releases_first_input(monkeypatch):
    (
        _clock,
        runtime,
        _registry,
        _specification,
        _image,
        _psf,
        _calls,
        built,
    ) = _fake_multi_input_registered_benchmark(monkeypatch)
    original_to_device = runtime.to_device
    transfer_count = 0

    def fail_second_transfer(value, *, device_id=""):
        nonlocal transfer_count
        transfer_count += 1
        if transfer_count == 2:
            raise _FakeOOM("second input transfer failed")
        return original_to_device(value, device_id=device_id)

    monkeypatch.setattr(runtime, "to_device", fail_second_transfer)

    with pytest.raises(RuntimeError, match="second input transfer failed"):
        built.request.candidates[0].execute(
            built.request.private_input_factory()
        )

    assert transfer_count == 2
    assert runtime.live == {}
    assert runtime.release_counts == Counter({1: 1})


def test_gpu_benchmark_invocations_hold_a_cancellable_device_lease(monkeypatch):
    (
        clock,
        runtime,
        _registry,
        specification,
        _image,
        _psf,
        _calls,
        built,
    ) = _fake_multi_input_registered_benchmark(monkeypatch)
    acquired: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def observed_lease(runtime_id, device_id, *, cancelled=None, **_kwargs):
        assert cancelled is not None
        assert cancelled() is False
        acquired.append((runtime_id, device_id))
        yield None

    monkeypatch.setattr(adapter_module, "accelerator_lease", observed_lease)

    NodeBenchmarkService(clock=clock).benchmark(built.request)

    assert acquired == [
        (specification.runtime_id, "cuda:0")
    ] * len(built.observations.runs(specification.implementation_id))

    def abort_wait() -> bool:
        raise BenchmarkCancelled("cancelled while waiting for the GPU lease")

    private_call = built.request.private_input_factory()
    assert built.request.bind_operation_progress is not None
    private_call = built.request.bind_operation_progress(
        private_call,
        None,
        abort_wait,
    )
    scope_count = runtime.scope_count

    with pytest.raises(BenchmarkCancelled, match="waiting for the GPU lease"):
        built.request.candidates[0].execute(private_call)

    assert runtime.scope_count == scope_count
    assert runtime.live == {}


def test_multi_input_port_contracts_and_refused_outputs_and_writers(
    monkeypatch,
):
    (
        _clock,
        _runtime,
        registry,
        spec,
        image,
        psf,
        _calls,
        built,
    ) = _fake_multi_input_registered_benchmark(monkeypatch)
    call = built.detached_call

    with pytest.raises(ValueError, match="input contracts"):
        build_registered_node_benchmark(
            call,
            admitted_specs=(replace(spec, input_ports=spec.input_ports[:1]),),
            registry=registry,
            environment_fingerprint="multi-input-port-mismatch",
            allow_experimental=True,
            call_is_detached=True,
        )

    with pytest.raises(ValueError, match="exactly one output"):
        build_registered_node_benchmark(
            replace(call, output_port_count=2),
            admitted_specs=(spec,),
            registry=registry,
            environment_fingerprint="multi-output-refused",
            allow_experimental=True,
            call_is_detached=True,
        )

    with pytest.raises(ValueError, match="writer operations"):
        build_registered_node_benchmark(
            PreparedNodeCall(
                "writer",
                "save_output",
                _identity_cpu,
                (image, psf),
                multiple_inputs=True,
            ),
            admitted_specs=(spec,),
            registry=registry,
            environment_fingerprint="writer-refused",
            allow_experimental=True,
        )


def test_registered_background_benchmark_reports_each_completed_yx_plane(
    monkeypatch,
):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    spec = _spec("subtract_background")
    progress_contexts: list[object] = []

    def cpu(data, **kwargs):
        progress_contexts.append(kwargs["progress"])
        result = subtract_background(data, **kwargs)
        clock.advance(0.100)
        return result

    def gpu(device, **kwargs):
        progress_contexts.append(kwargs["progress"])
        result = subtract_background(device.array, **kwargs)
        clock.advance(0.040)
        return runtime.allocate(result)

    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    values = np.arange(2 * 3 * 7 * 9, dtype=np.float32).reshape(2, 3, 7, 9)
    values.setflags(write=False)
    live_progress = _NonCopyableProgress()
    call = PreparedNodeCall(
        "background-node",
        "subtract_background",
        cpu,
        (values,),
        kwargs={
            "radius": 2.0,
            "light_background": False,
            "disable_smoothing": True,
            "clip_negative": True,
            "spatial_mode": "2D YX",
            "resolved_spatial_ndim": 2,
            "progress": live_progress,
            "channel_axis": None,
        },
    )
    built = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-exact-environment",
        allow_experimental=True,
        clock=clock,
        warm_rounds=3,
        max_warm_rounds=3,
        paired_bootstrap_samples=200,
    )
    updates = []

    assert built.detached_call.kwargs["progress"] is None
    assert built.request.private_input_factory().kwargs["progress"] is None

    def collect_with_observer_delay(update):
        updates.append(update)
        if update.operation_total:
            clock.advance(5.0)

    record = NodeBenchmarkService(clock=clock).benchmark(
        built.request,
        progress=collect_with_observer_delay,
    )

    first_cpu_call = [
        update
        for update in updates
        if update.implementation_id == "cpu-subtract_background-v1"
        and update.phase is BenchmarkMeasurementPhase.PARITY_COLD
        and update.operation_total
    ]
    assert [update.operation_completed for update in first_cpu_call] == list(range(7))
    assert {update.operation_total for update in first_cpu_call} == {6}
    assert {update.operation_message for update in first_cpu_call} == {
        "Rolling-ball YX plane"
    }
    assert len(progress_contexts) == len({id(item) for item in progress_contexts})
    candidate = next(
        item
        for item in record.candidates
        if item.implementation_id == spec.implementation_id
    )
    assert candidate.cold_seconds == pytest.approx(0.044)
    assert candidate.warm_seconds == pytest.approx((0.044,) * 3)
    assert candidate.cold_resident_seconds == pytest.approx(0.040)
    assert candidate.warm_resident_seconds == pytest.approx((0.040,) * 3)

    cancel_requested = False
    aborted_updates = []
    service = NodeBenchmarkService(clock=clock)

    def cancel_during_gpu(update):
        nonlocal cancel_requested
        aborted_updates.append(update)
        if (
            update.implementation_id == spec.implementation_id
            and update.operation_completed == 1
        ):
            cancel_requested = True

    with pytest.raises(BenchmarkCancelled):
        service.benchmark(
            built.request,
            cancelled=lambda: cancel_requested,
            progress=cancel_during_gpu,
        )

    last_gpu_update = next(
        update
        for update in reversed(aborted_updates)
        if update.implementation_id == spec.implementation_id
        and update.operation_total
    )
    assert (last_gpu_update.operation_completed, last_gpu_update.operation_total) == (
        1,
        6,
    )
    assert runtime.live == {}
    assert service.quarantine.entries() == ()
    assert len(service.store) == 0

    observer_service = NodeBenchmarkService(clock=clock)

    def fail_during_gpu_progress(update):
        if (
            update.implementation_id == spec.implementation_id
            and update.operation_completed == 1
        ):
            raise RuntimeError("observer disconnected")

    with pytest.raises(BenchmarkProgressError, match="observer disconnected"):
        observer_service.benchmark(
            built.request,
            progress=fail_during_gpu_progress,
        )

    assert runtime.live == {}
    assert observer_service.quarantine.entries() == ()
    assert len(observer_service.store) == 0


def test_aliasing_input_and_output_allocation_is_released_once(monkeypatch):
    clock, runtime, _registry, _live, built = _fake_registered_benchmark(
        monkeypatch,
        size=1,
        alias_output=True,
    )

    record = NodeBenchmarkService(clock=clock).benchmark(built.request)

    candidate = next(
        result
        for result in record.candidates
        if result.implementation_id == "cupyx-median-filter-v1"
    )
    assert candidate.parity_passed
    assert runtime.scope_count == 9
    # One shared input/output allocation for each private scope.
    assert len(runtime.release_counts) == runtime.scope_count
    assert set(runtime.release_counts.values()) == {1}


def test_progress_is_removed_before_deepcopy_and_policy_label_is_exact(
    monkeypatch,
):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    spec = _spec("median_filter")
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    call = PreparedNodeCall(
        "median-progress",
        "median_filter",
        median_filter,
        (values,),
        kwargs={
            "size": 3,
            "channel_axis": None,
            "progress": _NonCopyableProgress(),
        },
    )

    production = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-progress-environment",
        allow_experimental=True,
    )
    custom = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-progress-environment",
        allow_experimental=True,
        paired_bootstrap_samples=0,
    )

    assert production.detached_call.kwargs["progress"] is None
    assert production.request.benchmark_policy_id == PRODUCTION_BENCHMARK_POLICY_ID
    assert custom.request.benchmark_policy_id == CUSTOM_BENCHMARK_POLICY_ID
    assert custom.request.key.digest != production.request.key.digest


def test_rl_benchmark_key_binds_scientific_parity_policy(monkeypatch):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    v2_spec = _two_input_rl_spec()
    v1_spec = replace(v2_spec, parity_policy_id="rl-scientific-equivalence-v1")
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    image = np.arange(7 * 9, dtype=np.float32).reshape(7, 9) / 63
    psf = np.ones((3, 3), dtype=np.float32) / 9
    call = PreparedNodeCall(
        "rl-contract-node",
        "richardson_lucy_deconvolution",
        lambda inputs, **_kwargs: np.asarray(inputs[0]),
        (image, psf),
        kwargs={
            "iterations": 3,
            "filter_epsilon": 1e-12,
            "resolved_spatial_ndim": 2,
            "progress": None,
        },
        multiple_inputs=True,
    )
    monkeypatch.setattr(
        adapter_module,
        "_validate_admitted_spec",
        lambda *_args, **_kwargs: None,
    )

    def build(spec):
        return build_registered_node_benchmark(
            call,
            admitted_specs=(spec,),
            registry=registry,
            environment_fingerprint="same-exact-environment",
            allow_experimental=True,
            clock=clock,
        ).request

    v1 = build(v1_spec)
    v2 = build(v2_spec)

    assert v1.key.implementation_ids == v2.key.implementation_ids
    assert v1.benchmark_policy_id == v2.benchmark_policy_id
    assert v1.scientific_contract_digest != v2.scientific_contract_digest
    assert v1.key.policy_id != v2.key.policy_id
    assert v1.key.digest != v2.key.digest


def test_exact_key_separates_resolved_device_and_memory_scope(monkeypatch):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    spec = _spec("median_filter")
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    call = PreparedNodeCall(
        "median-target",
        "median_filter",
        median_filter,
        (np.arange(25, dtype=np.float32).reshape(5, 5),),
        kwargs={"size": 3, "channel_axis": None},
    )

    def build(**kwargs):
        return build_registered_node_benchmark(
            call,
            admitted_specs=(spec,),
            registry=registry,
            environment_fingerprint="same-environment",
            allow_experimental=True,
            **kwargs,
        ).request

    default = build()
    other_device = build(device_id="cuda:1")
    memory_limited = build(memory_limit_bytes=64 * 1024**2)
    reserved = build(safety_reserve_bytes=16 * 1024**2)

    assert default.device_id == "cuda:0"
    assert default.key.device_id == "cuda:0"
    assert other_device.key.device_id == "cuda:1"
    assert memory_limited.key.memory_limit_bytes == 64 * 1024**2
    assert reserved.key.safety_reserve_bytes == 16 * 1024**2
    assert (
        len(
            {
                default.key.digest,
                other_device.key.digest,
                memory_limited.key.digest,
                reserved.key.digest,
            }
        )
        == 4
    )


def test_kernel_oom_traceback_is_detached_and_benchmark_runtime_reuses(
    monkeypatch,
):
    clock = ManualClock()
    runtime = _FakeRuntime(clock)
    registry = ComputeRegistry()
    spec = _spec("median_filter")
    failures_remaining = 2

    def cpu(data, **kwargs):
        clock.advance(0.100)
        return median_filter(data, **kwargs)

    def gpu(device, **kwargs):
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            scratch = _FakeTracebackScratch(runtime)
            assert scratch.runtime is runtime
            raise _FakeOOM("synthetic benchmark kernel OOM")
        clock.advance(0.040)
        return runtime.allocate(median_filter(device.array, **kwargs))

    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        registry,
        "implementation_callable",
        lambda *_args, **_kwargs: gpu,
    )
    call = PreparedNodeCall(
        "median-oom",
        "median_filter",
        cpu,
        (np.arange(49, dtype=np.float32).reshape(7, 7),),
        kwargs={"size": 3, "channel_axis": None},
    )
    built = build_registered_node_benchmark(
        call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-oom-environment",
        allow_experimental=True,
        clock=clock,
        paired_bootstrap_samples=200,
    )

    with pytest.raises(RuntimeError) as detached:
        built.request.candidates[0].execute(built.request.private_input_factory())
    assert not isinstance(detached.value, _FakeOOM)
    assert detached.value.__cause__ is None
    assert detached.value.__context__ is None
    assert "out_of_memory: fake_oom" in str(detached.value)
    assert runtime.traceback_scratch_live == 0
    assert runtime.live == {}

    failed = NodeBenchmarkService(clock=clock).benchmark(built.request)
    failed_candidate = next(
        result
        for result in failed.candidates
        if result.implementation_id == spec.implementation_id
    )

    assert not failed_candidate.parity_passed
    assert "out_of_memory: fake_oom" in failed_candidate.error
    assert "Traceback" not in failed_candidate.error
    assert runtime.classified_inside_scope == [True, True]
    assert runtime.traceback_scratch_live == 0
    assert runtime.live == {}
    failed_runs = built.observations.runs(spec.implementation_id)[:2]
    assert len(failed_runs) == 2
    assert all(
        run.cleanup_succeeded
        and run.terminal_snapshot.runtime_live_bytes == 0
        and run.terminal_snapshot.runtime_reserved_bytes == 0
        for run in failed_runs
    )

    recovered = NodeBenchmarkService(clock=clock).benchmark(built.request)
    recovered_candidate = next(
        result
        for result in recovered.candidates
        if result.implementation_id == spec.implementation_id
    )

    assert recovered_candidate.parity_passed, recovered_candidate.error
    assert runtime.traceback_scratch_live == 0
    assert runtime.live == {}
    assert all(
        run.cleanup_succeeded
        and run.terminal_snapshot.runtime_live_bytes == 0
        and run.terminal_snapshot.runtime_reserved_bytes == 0
        for run in built.observations.runs(spec.implementation_id)
    )


def test_exact_parity_checks_signed_zero_bits_and_gaussian_uses_both_gates():
    exact = np.array([0.0, -0.0, 2.0], dtype=np.float32)
    wrong_sign = np.array([0.0, 0.0, 2.0], dtype=np.float32)

    result = operation_parity("median_filter", exact, wrong_sign)

    assert not result.passed
    assert "signed_zero_mismatches=1" in result.detail

    reference_mask = np.array([[False, True], [True, False]], dtype=bool)
    wrong_mask = reference_mask.copy()
    wrong_mask[1, 1] = True
    for operation_id in ("canny_edges", "otsu_threshold"):
        assert operation_parity(
            operation_id,
            reference_mask,
            reference_mask.copy(),
        ).passed
        mask_result = operation_parity(
            operation_id,
            reference_mask,
            wrong_mask,
        )
        assert not mask_result.passed
        assert "mismatch" in mask_result.detail

    gaussian = np.linspace(-1.0, 1.0, 128, dtype=np.float32).reshape(8, 16)
    within = gaussian + np.float32(2e-7)
    outside = gaussian.copy()
    outside.flat[0] += np.float32(1e-3)

    accepted = operation_parity("gaussian_blur", gaussian, within)
    rejected = operation_parity("gaussian_blur", gaussian, outside)

    assert accepted.passed
    assert "nrmse=" in accepted.detail and "max_abs=" in accepted.detail
    assert not rejected.passed
    assert "nrmse=" in rejected.detail and "max_abs=" in rejected.detail

    background_reference = np.array([0.0, 100_000.0, 2.0], dtype=np.float32)
    background_candidate = background_reference.copy()
    background_candidate[-1] += np.float32(0.0078125)
    bounded = operation_parity(
        "subtract_background",
        background_reference,
        background_candidate,
        input_peak=100_000.0,
    )
    wrong_background_zero = background_reference.copy()
    wrong_background_zero[0] = -0.0
    zero_to_tiny = background_reference.copy()
    zero_to_tiny[0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    tiny_reference = background_reference.copy()
    tiny_reference[0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    tiny_to_zero = tiny_reference.copy()
    tiny_to_zero[0] = np.float32(0.0)

    assert bounded.passed
    assert "max_ulp=" in bounded.detail
    assert not operation_parity(
        "subtract_background",
        background_reference,
        wrong_background_zero,
        input_peak=100_000.0,
    ).passed
    for reference, candidate in (
        (background_reference, zero_to_tiny),
        (tiny_reference, tiny_to_zero),
    ):
        result = operation_parity(
            "subtract_background",
            reference,
            candidate,
            input_peak=100_000.0,
        )
        assert not result.passed
        assert result.detail == "zero masks differ"
    assert not operation_parity(
        "subtract_background",
        np.array([1, 2], dtype=np.uint16),
        np.array([1, 3], dtype=np.uint16),
        input_peak=3.0,
    ).passed

    rl_reference = np.linspace(0.0, 2.0, 256, dtype=np.float32).reshape(16, 16)
    rl_within = rl_reference + np.float32(2e-7)
    rl_scientifically_equivalent = rl_reference.copy()
    rl_scientifically_equivalent[8, 8] += np.float32(1e-3)
    rl_outside = rl_reference.copy()
    rl_outside[8, 8] += np.float32(2e-2)

    rl_accepted = operation_parity(
        "richardson_lucy_deconvolution",
        rl_reference,
        rl_within,
    )
    rl_rejected = operation_parity(
        "richardson_lucy_deconvolution",
        rl_reference,
        rl_outside,
    )
    rl_not_near_identical = operation_parity(
        "richardson_lucy_deconvolution",
        rl_reference,
        rl_scientifically_equivalent,
    )

    assert rl_accepted.passed
    assert "nrmse=" in rl_accepted.detail and "max_ulp=" in rl_accepted.detail
    assert rl_not_near_identical.passed
    assert "near_identity=false" in rl_not_near_identical.detail
    assert not rl_rejected.passed
    assert "max_abs=" in rl_rejected.detail

    tv_within = rl_reference.copy()
    tv_within[8, 8] += np.float32(1e-3)
    tv_outside = rl_reference.copy()
    tv_outside[8, 8] += np.float32(2e-2)
    tv_positive = operation_parity(
        "richardson_lucy_tv_deconvolution",
        rl_reference,
        tv_within,
        parameters={"tv_regularization": 0.002},
    )
    tv_positive_rejected = operation_parity(
        "richardson_lucy_tv_deconvolution",
        rl_reference,
        tv_outside,
        parameters={"tv_regularization": 0.002},
    )
    tv_lambda_zero = operation_parity(
        "richardson_lucy_tv_deconvolution",
        rl_reference,
        tv_within,
        parameters={"tv_regularization": 0.0},
    )
    tv_negative = rl_reference.copy()
    tv_negative[0, 0] = np.float32(-1e-6)

    assert tv_positive.passed
    assert "limit=0.005" in tv_positive.detail
    assert not tv_positive_rejected.passed
    assert tv_lambda_zero.passed
    assert "near_identity=false" in tv_lambda_zero.detail
    assert not operation_parity(
        "richardson_lucy_tv_deconvolution",
        rl_reference,
        tv_negative,
        parameters={"tv_regularization": 0.002},
    ).passed


def test_rl_parity_floor_is_independent_of_gaussian_policy(monkeypatch):
    monkeypatch.setattr(adapter_module, "GAUSSIAN_FLOAT32_ABSOLUTE_FLOOR", 1.0)
    reference = np.zeros((4, 4), dtype=np.float32)
    candidate = np.full((4, 4), 5e-7, dtype=np.float32)

    result = operation_parity(
        "richardson_lucy_deconvolution",
        reference,
        candidate,
    )

    assert not result.passed
    assert "nrmse=" in result.detail


def test_adaptive_rounds_bootstrap_and_json_store_are_exact_and_deterministic(
    tmp_path,
):
    def run():
        clock = ManualClock()

        def implementation(name: str, duration: float):
            def execute(private):
                clock.advance(duration)
                return private["value"]

            return BenchmarkImplementation(name, execute)

        request = NodeBenchmarkRequest(
            workload=WorkloadDescriptor(
                "node",
                "gaussian_blur",
                ((16, 16),),
                ("float32",),
            ),
            environment_fingerprint="exact-environment-a",
            reference=implementation("cpu", 0.100),
            candidates=(implementation("gpu", 0.095),),
            private_input_factory=lambda: {"value": 3},
            parity=lambda expected, actual: expected == actual,
            adaptive_rounds=True,
            max_warm_rounds=21,
            paired_bootstrap_samples=400,
            paired_bootstrap_seed=1234,
        )
        return request, NodeBenchmarkService(clock=clock).benchmark(request)

    request, first = run()
    _second_request, second = run()
    first_results = {item.implementation_id: item for item in first.candidates}
    second_results = {item.implementation_id: item for item in second.candidates}

    assert len(first_results["cpu"].warm_seconds) == 21
    assert len(first_results["gpu"].warm_seconds) == 21
    assert first_results["gpu"].paired_speedup_median == pytest.approx(0.100 / 0.095)
    assert (
        first_results["gpu"].paired_speedup_lower_confidence_bound
        == second_results["gpu"].paired_speedup_lower_confidence_bound
    )
    assert first.accepted_implementation_id == "cpu"

    store_path = tmp_path / "benchmarks.json"
    changed = NodeBenchmarkRequest(
        workload=request.workload,
        environment_fingerprint="exact-environment-b",
        reference=request.reference,
        candidates=request.candidates,
        private_input_factory=request.private_input_factory,
        parity=request.parity,
        adaptive_rounds=True,
        max_warm_rounds=21,
        paired_bootstrap_samples=400,
        paired_bootstrap_seed=1234,
    )
    staleness = benchmark_record_staleness(first, changed.key)
    assert staleness.stale
    assert staleness.reasons == ("environment fingerprint changed",)

    # Both instances start stale/empty. The per-path process lock and
    # reload-before-merge prevent the second writer from erasing the first.
    first_store = JsonBenchmarkStore(store_path)
    second_store = JsonBenchmarkStore(store_path)
    first_store.put(first)
    changed_record = replace(first, key=changed.key)
    second_store.put(changed_record)
    reopened = JsonBenchmarkStore(store_path)
    assert reopened.get(request.key) == first
    assert reopened.get(changed.key) == changed_record
    assert len(first_store) == 2
    assert len(second_store) == 2
    assert not tuple(tmp_path.glob(".benchmarks.json.tmp-*"))

    target_changes = (
        (replace(request, device_id="cuda:1"), "device target changed"),
        (
            replace(request, memory_limit_bytes=512 * 1024**2),
            "memory limit changed",
        ),
        (
            replace(request, safety_reserve_bytes=64 * 1024**2),
            "safety reserve changed",
        ),
    )
    for changed_request, reason in target_changes:
        target_staleness = benchmark_record_staleness(
            first,
            changed_request.key,
        )
        assert target_staleness.stale
        assert target_staleness.reasons == (reason,)

    summary = paired_bootstrap_speedup(
        (0.100,) * 7,
        (0.050,) * 7,
        sample_count=100,
        seed=99,
    )
    assert summary.median_speedup == pytest.approx(2.0)
    assert summary.lower_confidence_bound == pytest.approx(2.0)


def test_detached_input_content_participates_in_exact_workload_fingerprint(
    monkeypatch,
):
    _clock, _runtime, registry, live, first = _fake_registered_benchmark(monkeypatch)
    changed = live.copy()
    changed[4, 5] += 1
    spec = _spec("median_filter")
    second_call = PreparedNodeCall(
        "median-node",
        "median_filter",
        first.detached_call.cpu_function,
        (changed,),
        kwargs={"size": 3, "channel_axis": None},
    )
    second = build_registered_node_benchmark(
        second_call,
        admitted_specs=(spec,),
        registry=registry,
        environment_fingerprint="fake-exact-environment",
        allow_experimental=True,
    )

    assert first.request.workload.facts_fingerprint
    assert (
        first.request.workload.facts_fingerprint
        != second.request.workload.facts_fingerprint
    )
    assert first.request.key.digest != second.request.key.digest


def test_adapter_import_does_not_import_optional_gpu_packages():
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupy."):
        raise AssertionError(name)
    if name == "cupyx" or name.startswith("cupyx."):
        raise AssertionError(name)
    if name == "cucim" or name.startswith("cucim."):
        raise AssertionError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded
import napari_vipp.core.compute_benchmark_adapter as adapter
assert callable(adapter.build_registered_node_benchmark)
assert not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules)
assert not any(name == "cupyx" or name.startswith("cupyx.") for name in sys.modules)
assert not any(name == "cucim" or name.startswith("cucim.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_registered_production_benchmark_smoke_when_cuda_is_available():
    registry = ComputeRegistry()
    runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
    if not runtime_probe.available:
        pytest.skip(runtime_probe.message or "CUDA/CuPy runtime is unavailable.")

    rng = np.random.default_rng(20260727)
    background = rng.normal(17.0, 2.5, size=(31, 37)).astype(np.float32)
    background[9:14, 12:18] += np.float32(25.0)
    gaussian = rng.normal(0.0, 3.0, size=(128, 160)).astype(np.float32)
    median = rng.normal(0.5, 4.0, size=(96, 112)).astype(np.float32)
    rl_image = rng.uniform(0.0, 1.0, size=(31, 37)).astype(np.float32)
    rl_image[15, 18] += np.float32(3.0)
    rl_psf = np.array(
        [
            [0.0, 0.0, 0.0625, 0.0, 0.0],
            [0.0, 0.0625, 0.125, 0.0625, 0.0],
            [0.0625, 0.125, 0.125, 0.125, 0.0625],
            [0.0, 0.0625, 0.125, 0.0625, 0.0],
            [0.0, 0.0, 0.0625, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rl_psf /= rl_psf.sum(dtype=np.float32)
    for values in (background, gaussian, median, rl_image, rl_psf):
        values.setflags(write=False)

    cases = (
        (
            "subtract_background",
            subtract_background,
            (background,),
            {
                "radius": 5.0,
                "light_background": False,
                "disable_smoothing": False,
                "clip_negative": True,
                "spatial_mode": "2D YX",
                "resolved_spatial_ndim": 2,
                "progress": None,
                "channel_axis": None,
            },
        ),
        (
            "gaussian_blur",
            gaussian_blur,
            (gaussian,),
            {"sigma": 1.3, "channel_axis": None},
        ),
        (
            "median_filter",
            median_filter,
            (median,),
            {"size": 5, "channel_axis": None},
        ),
        (
            "richardson_lucy_deconvolution",
            richardson_lucy_deconvolution,
            (rl_image, rl_psf),
            {
                "spatial_mode": "2D YX",
                "iterations": 3,
                "normalize_psf": True,
                "clip_negative_input": True,
                "clip_output_negative": True,
                "preserve_input_scale": True,
                "filter_epsilon": 1e-8,
                "resolved_spatial_ndim": 2,
                "progress": None,
            },
        ),
    )
    try:
        for index, (operation_id, cpu_function, inputs, kwargs) in enumerate(cases):
            spec = _spec(operation_id)
            library_probe = registry.probe_library(
                spec.implementation_library_id,
                refresh=True,
            )
            if not library_probe.available:
                pytest.skip(
                    library_probe.message
                    or f"{spec.implementation_library_id} is unavailable."
                )
            environment_fingerprint = canonical_digest(
                {
                    "runtime": runtime_probe.as_dict(),
                    "library": library_probe.as_dict(),
                    "implementation_id": spec.implementation_id,
                    "implementation_version": spec.implementation_version,
                }
            )
            call = PreparedNodeCall(
                f"real-{operation_id}",
                operation_id,
                cpu_function,
                inputs,
                kwargs=kwargs,
                multiple_inputs=len(inputs) > 1,
            )
            built = build_registered_node_benchmark(
                call,
                admitted_specs=(spec,),
                registry=registry,
                environment_fingerprint=environment_fingerprint,
                allow_experimental=True,
                time_budget_seconds=180.0,
                paired_bootstrap_seed=8000 + index,
            )

            record = NodeBenchmarkService(rng=np.random.default_rng(index)).benchmark(
                built.request
            )

            candidate = next(
                result
                for result in record.candidates
                if result.implementation_id == spec.implementation_id
            )
            assert candidate.parity_passed, candidate.error
            assert len(candidate.warm_seconds) in {7, 15, 21}
            assert candidate.timing_scope == "synchronized-end-to-end-v1"
            assert candidate.synchronized and candidate.transfers_included
            assert len(candidate.warm_transfer_seconds) == len(candidate.warm_seconds)
            assert len(candidate.warm_resident_seconds) == len(candidate.warm_seconds)
            assert candidate.peak_runtime_reserved_bytes > 0
            runs = built.observations.runs(spec.implementation_id)
            assert runs
            assert all(
                run.cleanup_succeeded
                and run.terminal_snapshot.runtime_live_bytes == 0
                and run.terminal_snapshot.runtime_reserved_bytes == 0
                for run in runs
            )
    finally:
        registry.close()


def test_real_cupy_kernel_oom_is_detached_and_benchmark_runtime_reuses():
    cupy = pytest.importorskip("cupy")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.float32).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")

    base = _spec("gaussian_blur")
    implementation = replace(
        base,
        implementation_id="test-cupy-benchmark-kernel-oom-v1",
        callable_ref=f"{__name__}:_cupy_benchmark_kernel_oom_or_copy",
    )
    registry = ComputeRegistry(implementation_specs=(implementation,))
    runtime = registry.runtime(implementation.runtime_id)
    probe = runtime.probe(refresh=True)
    if not probe.available or not probe.selected_device_id:
        registry.close()
        pytest.skip(probe.message or "The CUDA runtime is unavailable.")

    data = np.arange(81, dtype=np.float32).reshape(9, 9)

    def build(*, sigma: float, environment: str):
        call = PreparedNodeCall(
            f"real-oom-{sigma:g}",
            "gaussian_blur",
            _identity_cpu,
            (data,),
            kwargs={"sigma": sigma, "channel_axis": None},
        )
        return build_registered_node_benchmark(
            call,
            admitted_specs=(implementation,),
            registry=registry,
            environment_fingerprint=environment,
            device_id=probe.selected_device_id,
            memory_limit_bytes=8 * 1024**2,
            safety_reserve_bytes=0,
            allow_experimental=True,
            time_budget_seconds=180.0,
        )

    try:
        failing = build(sigma=0.0, environment="real-cupy-oom")
        failed_record = NodeBenchmarkService().benchmark(failing.request)
        failed_candidate = next(
            result
            for result in failed_record.candidates
            if result.implementation_id == implementation.implementation_id
        )

        assert not failed_candidate.parity_passed
        assert "out_of_memory" in failed_candidate.error
        failed_runs = failing.observations.runs(implementation.implementation_id)
        assert len(failed_runs) == 1
        assert failed_runs[0].cleanup_succeeded
        first_terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert first_terminal.runtime_live_bytes == 0
        assert first_terminal.runtime_reserved_bytes == 0
        assert runtime.probe(refresh=True).available

        reusable = build(sigma=1.0, environment="real-cupy-reuse")
        recovered_record = NodeBenchmarkService().benchmark(reusable.request)
        recovered_candidate = next(
            result
            for result in recovered_record.candidates
            if result.implementation_id == implementation.implementation_id
        )

        assert recovered_candidate.parity_passed, recovered_candidate.error
        assert all(
            run.cleanup_succeeded
            and run.terminal_snapshot.runtime_live_bytes == 0
            and run.terminal_snapshot.runtime_reserved_bytes == 0
            for run in reusable.observations.runs(implementation.implementation_id)
        )
        second_terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert second_terminal.runtime_live_bytes == 0
        assert second_terminal.runtime_reserved_bytes == 0
        assert runtime.probe(refresh=True).available
    finally:
        registry.close()
