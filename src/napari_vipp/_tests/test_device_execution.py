from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.device_execution as device_execution_module
from napari_vipp.core.accelerator_lease import accelerator_lease
from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    FallbackPolicy,
    MemoryEstimate,
    NodeComputePreference,
    NodeExecutionDecision,
    OutputPortKey,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryDescriptor,
    RuntimeDescriptor,
    RuntimeDevice,
    RuntimeExceptionInfo,
    RuntimeExceptionKind,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    OperationComputeSpec,
    compute_specs_for,
)
from napari_vipp.core.device_execution import (
    DeviceExecutionCancelled,
    DeviceExecutionError,
    DeviceMemoryPreflightError,
    DevicePlanningError,
    DeviceSegmentUnit,
    HostExecutionUnit,
    execute_device_plan,
    plan_device_execution,
    preflight_device_execution,
)
from napari_vipp.core.execution_telemetry import (
    DeviceExecutionPhase,
    DeviceExecutionTelemetryConfig,
    DeviceSynchronizationPoint,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.pipeline import (
    DYNAMIC_OUTPUT_COUNT_PARAM,
    PrototypePipeline,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.tables import TableData


class _FakeOOM(RuntimeError):
    pass


class _FakeKernelFailure(RuntimeError):
    pass


class _FakeCleanupFailure(RuntimeError):
    pass


class _SteppingClock:
    def __init__(self, step: float = 0.25) -> None:
        self.current = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.current
        self.current += self.step
        return value


class _HostileClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == 1:
            return 10.0
        if self.calls == 2:
            return 9.0
        raise RuntimeError("diagnostic clock failed")


class _HostileNbytesHost:
    @property
    def nbytes(self) -> int:
        raise RuntimeError("host byte metadata failed")

    def __array__(self, dtype=None, copy=None):
        del copy
        return np.arange(25, dtype=dtype or np.float32).reshape(5, 5)


class _FakeDeviceArray:
    """Opaque test value which host/NumPy code is forbidden to coerce."""

    def __init__(self, runtime: _FakeRuntime, payload: object) -> None:
        self.runtime = runtime
        self.payload = np.array(payload, copy=True)
        self.released = False

    def __array__(self, _dtype=None, copy=None):
        del copy
        raise AssertionError("Device values must remain opaque outside the runtime.")


class _FakeTracebackScratch:
    """Track a value whose only lasting owner is a provider traceback frame."""

    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        runtime.traceback_scratch_live += 1

    def __del__(self) -> None:
        self.runtime.traceback_scratch_live -= 1


class _FakeRuntime:
    runtime_id = "fake-device"
    array_domain = "fake-array"

    def __init__(self, *, free_bytes: int = 10_000) -> None:
        self.free_bytes = free_bytes
        self.live: dict[int, _FakeDeviceArray] = {}
        self.events: list[object] = []
        self.host_to_device_count = 0
        self.device_to_host_count = 0
        self.release_count = 0
        self.operation_count = 0
        self.oom_remaining = 0
        self.registration_failure_id: int | None = None
        self.registration_checks: dict[int, int] = {}
        self.closed = False
        self.scope_active = False
        self.cleanup_fails_after_oom = False
        self.oom_was_classified = False
        self.classified_inside_scope: list[bool] = []
        self.traceback_scratch_live = 0
        self.memory_snapshot_states: list[tuple[str, int, bool]] = []

    def allocate(self, payload: object) -> _FakeDeviceArray:
        value = _FakeDeviceArray(self, payload)
        self.live[id(value)] = value
        self.events.append("allocate")
        return value

    def probe(self, *, refresh: bool = False) -> RuntimeProbeResult:
        del refresh
        return RuntimeProbeResult(
            self.runtime_id,
            True,
            version="1",
            devices=(RuntimeDevice("fake:0", "Fake device", self.free_bytes),),
            selected_device_id="fake:0",
        )

    @contextmanager
    def execution_scope(
        self,
        *,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
    ):
        self.events.append(
            ("scope-enter", device_id, memory_limit_bytes, safety_reserve_bytes)
        )
        assert not self.scope_active
        self.scope_active = True
        try:
            yield
        except _FakeOOM as exc:
            if self.cleanup_fails_after_oom:
                raise _FakeCleanupFailure(
                    "private allocation remained live after OOM"
                ) from exc
            raise
        else:
            if self.cleanup_fails_after_oom and self.oom_was_classified:
                raise _FakeCleanupFailure(
                    "private allocation remained live after detached OOM"
                )
        finally:
            self.scope_active = False
            self.events.append(("scope-exit", device_id))

    def is_device_value(self, value: object) -> bool:
        if id(value) == self.registration_failure_id:
            checks = self.registration_checks.get(id(value), 0) + 1
            self.registration_checks[id(value)] = checks
            if checks == 2:
                raise _FakeKernelFailure("synthetic output registration failure")
        return isinstance(value, _FakeDeviceArray)

    def allocation_identity(self, value: object):
        if not self.is_device_value(value):
            raise TypeError("not a fake device allocation")
        return id(value)

    def to_device(self, value: object, *, device_id: str = "") -> object:
        self.host_to_device_count += 1
        self.events.append(("to-device", device_id))
        return self.allocate(value)

    def to_host(self, value: object) -> object:
        assert isinstance(value, _FakeDeviceArray)
        assert not value.released
        self.device_to_host_count += 1
        self.events.append("to-host")
        return np.array(value.payload, copy=True)

    def release(self, value: object) -> None:
        assert self.scope_active, "Device values must be released inside their scope."
        assert isinstance(value, _FakeDeviceArray)
        assert not value.released, "A device allocation was released twice."
        value.released = True
        self.live.pop(id(value))
        self.release_count += 1
        self.events.append("release")

    def synchronize(self, *, device_id: str = "") -> None:
        self.events.append(("synchronize", device_id))

    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot:
        self.memory_snapshot_states.append(
            (device_id or "fake:0", len(self.live), self.scope_active)
        )
        private_bytes = len(self.live) * 64
        return RuntimeMemorySnapshot(
            self.runtime_id,
            device_id or "fake:0",
            "discrete",
            device_total_bytes=self.free_bytes,
            device_free_bytes=self.free_bytes,
            runtime_live_bytes=private_bytes,
            runtime_reserved_bytes=private_bytes,
        )

    def classify_exception(self, exc: BaseException) -> RuntimeExceptionInfo:
        self.classified_inside_scope.append(self.scope_active)
        if isinstance(exc, _FakeCleanupFailure):
            return RuntimeExceptionInfo(
                RuntimeExceptionKind.KERNEL_FAILURE,
                "fake_cleanup_incomplete",
                str(exc),
                exception_type=type(exc).__name__,
                retryable=False,
            )
        if isinstance(exc, _FakeOOM):
            self.oom_was_classified = True
            return RuntimeExceptionInfo(
                RuntimeExceptionKind.OUT_OF_MEMORY,
                "fake_oom",
                str(exc),
                exception_type=type(exc).__name__,
                retryable=True,
            )
        if isinstance(exc, _FakeKernelFailure):
            return RuntimeExceptionInfo(
                RuntimeExceptionKind.KERNEL_FAILURE,
                "fake_kernel_failure",
                str(exc),
                exception_type=type(exc).__name__,
            )
        return RuntimeExceptionInfo(
            RuntimeExceptionKind.UNKNOWN,
            "fake_unknown",
            str(exc),
            exception_type=type(exc).__name__,
        )

    def close(self) -> None:
        self.closed = True


class _AliasCheckingRuntime(_FakeRuntime):
    """Model a private allocator that rejects released-but-reachable arrays."""

    def __init__(self) -> None:
        super().__init__()
        self.released_references: list[weakref.ReferenceType[_FakeDeviceArray]] = []

    @contextmanager
    def execution_scope(
        self,
        *,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
    ):
        with super().execution_scope(
            device_id=device_id,
            memory_limit_bytes=memory_limit_bytes,
            safety_reserve_bytes=safety_reserve_bytes,
        ):
            try:
                yield
            finally:
                gc.collect()
                if any(
                    reference() is not None for reference in self.released_references
                ):
                    raise _FakeCleanupFailure(
                        "released device input remains reachable during cleanup"
                    )

    def release(self, value: object) -> None:
        assert isinstance(value, _FakeDeviceArray)
        self.released_references.append(weakref.ref(value))
        super().release(value)


class _SynchronizeOOMOnceRuntime(_FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.synchronize_failures_remaining = 1

    def synchronize(self, *, device_id: str = "") -> None:
        super().synchronize(device_id=device_id)
        if self.synchronize_failures_remaining:
            self.synchronize_failures_remaining -= 1
            raise _FakeOOM("synthetic synchronization OOM")


class _TerminalSnapshotFailureRuntime(_FakeRuntime):
    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot:
        if self.memory_snapshot_states:
            raise RuntimeError("synthetic terminal snapshot failure")
        return super().memory_snapshot(device_id=device_id)


class _TerminalSnapshotIdentityRuntime(_FakeRuntime):
    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot:
        if not self.memory_snapshot_states:
            return super().memory_snapshot(device_id=device_id)
        self.memory_snapshot_states.append(
            ("fake:1", len(self.live), self.scope_active)
        )
        return RuntimeMemorySnapshot(
            self.runtime_id,
            "fake:1",
            "discrete",
            device_total_bytes=self.free_bytes,
            device_free_bytes=self.free_bytes,
        )


def _device_copy(value: _FakeDeviceArray, **_kwargs) -> _FakeDeviceArray:
    assert not value.released
    value.runtime.operation_count += 1
    return value.runtime.allocate(value.payload)


def _device_alias(value: _FakeDeviceArray, **_kwargs) -> _FakeDeviceArray:
    """Model a provider view with the same underlying allocation identity."""

    assert not value.released
    value.runtime.operation_count += 1
    return value


def _device_richardson_lucy(
    values: list[_FakeDeviceArray],
    **_kwargs,
) -> _FakeDeviceArray:
    assert len(values) == 2
    image, psf = values
    assert image.runtime is psf.runtime
    image.runtime.operation_count += 1
    return image.runtime.allocate(image.payload.astype(np.float32, copy=True))


def _device_add(
    values: list[_FakeDeviceArray],
    **_kwargs,
) -> _FakeDeviceArray:
    runtime = values[0].runtime
    assert all(value.runtime is runtime and not value.released for value in values)
    runtime.operation_count += 1
    return runtime.allocate(sum(value.payload for value in values))


def _device_split(
    value: _FakeDeviceArray,
    **_kwargs,
) -> tuple[_FakeDeviceArray, _FakeDeviceArray]:
    assert not value.released
    value.runtime.operation_count += 1
    return (
        value.runtime.allocate(value.payload[0]),
        value.runtime.allocate(value.payload[1]),
    )


def _device_split_registration_failure(
    value: _FakeDeviceArray,
    **_kwargs,
) -> tuple[_FakeDeviceArray, _FakeDeviceArray]:
    assert not value.released
    value.runtime.operation_count += 1
    first = value.runtime.allocate(value.payload[0])
    second = value.runtime.allocate(value.payload[1])
    value.runtime.registration_failure_id = id(second)
    return first, second


def _device_fail(value: _FakeDeviceArray, **_kwargs) -> _FakeDeviceArray:
    assert not value.released
    value.runtime.operation_count += 1
    raise _FakeKernelFailure("synthetic kernel failure")


def _device_oom_once(value: _FakeDeviceArray, **_kwargs) -> _FakeDeviceArray:
    assert not value.released
    value.runtime.operation_count += 1
    if value.runtime.oom_remaining:
        value.runtime.oom_remaining -= 1
        raise _FakeOOM("synthetic device allocation failure")
    return value.runtime.allocate(value.payload)


def _device_oom_once_with_traceback_scratch(
    value: _FakeDeviceArray,
    **_kwargs,
) -> _FakeDeviceArray:
    assert not value.released
    value.runtime.operation_count += 1
    if value.runtime.oom_remaining:
        value.runtime.oom_remaining -= 1
        scratch = _FakeTracebackScratch(value.runtime)
        assert scratch.runtime is value.runtime
        raise _FakeOOM("synthetic kernel OOM with traceback-local scratch")
    return value.runtime.allocate(value.payload)


_HOST_FINALIZER_RUNTIME: _FakeRuntime | None = None
_HOST_FINALIZER_CALLS: list[tuple[object, ...]] = []


def _host_array_finalizer(
    host_outputs: tuple[object, ...],
    *,
    call: PreparedNodeCall,
) -> np.ndarray:
    runtime = _HOST_FINALIZER_RUNTIME
    assert runtime is not None
    assert not runtime.scope_active
    assert runtime.live == {}
    assert call.inputs == (None,) * len(call.inputs)
    _HOST_FINALIZER_CALLS.append(host_outputs)
    assert len(host_outputs) == 1
    return np.asarray(host_outputs[0]) + 10


def _escaping_table_finalizer(
    _host_outputs: tuple[object, ...],
    *,
    call: PreparedNodeCall,
) -> TableData:
    del call
    runtime = _HOST_FINALIZER_RUNTIME
    assert runtime is not None
    hidden_device_value = _FakeDeviceArray(runtime, np.asarray([1]))
    return TableData(("value",), ((hidden_device_value,),))


def _cupy_kernel_oom_or_copy(value, *, sigma=0.0, **_kwargs):
    """Create traceback-local private allocations before a real CuPy OOM."""

    import cupy

    if float(sigma) != 0.0:
        return value.copy()
    scratch = cupy.empty_like(value)
    assert scratch.shape == value.shape
    # The focused integration test installs an 8 MiB private-pool limit.  This
    # request is deliberately larger and therefore cannot consume device-wide
    # memory outside the runtime-owned allocator.
    return cupy.empty((32 * 1024**2,), dtype=cupy.uint8)


def _runtime_descriptor(
    *,
    interoperability_claims: tuple[str, ...] = (),
) -> RuntimeDescriptor:
    return RuntimeDescriptor(
        runtime_id="fake-device",
        display_name="Fake device",
        factory_ref=f"{__name__}:_unused_runtime_factory",
        array_domain="fake-array",
        device_domain="fake",
        supported_os_families=("Windows", "Linux", "macOS"),
        interoperability_claims=interoperability_claims,
    )


def _library_descriptor(
    library_id: str = "fake-library",
    *,
    interoperability_claims: tuple[str, ...] = (),
) -> ImplementationLibraryDescriptor:
    return ImplementationLibraryDescriptor(
        library_id=library_id,
        display_name="Fake implementation library",
        runtime_ids=("fake-device",),
        array_domain="fake-array",
        supported_os_families=("Windows", "Linux", "macOS"),
        interoperability_claims=interoperability_claims,
    )


def _unused_runtime_factory() -> _FakeRuntime:
    raise AssertionError("Tests inject their runtime factory directly.")


def _implementation_spec(
    operation_id: str,
    function: Callable,
) -> OperationComputeSpec:
    return replace(
        compute_specs_for(operation_id)[0],
        implementation_id=f"fake-{operation_id}-v1",
        implementation_version="1",
        runtime_id="fake-device",
        array_domain="fake-array",
        implementation_library_id="fake-library",
        callable_ref=f"{__name__}:{function.__name__}",
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        supports_device_residency=True,
    )


def _registry(
    runtime: _FakeRuntime,
    implementations: Iterable[tuple[str, Callable]],
    *,
    host_finalizer_refs: Mapping[str, str] | None = None,
) -> tuple[ComputeRegistry, dict[str, OperationComputeSpec]]:
    specs = {
        operation_id: _implementation_spec(operation_id, function)
        for operation_id, function in implementations
    }
    for operation_id, reference in (host_finalizer_refs or {}).items():
        specs[operation_id] = replace(
            specs[operation_id],
            host_finalizer_ref=reference,
        )
    return (
        ComputeRegistry(
            runtime_descriptors=(_runtime_descriptor(),),
            library_descriptors=(_library_descriptor(),),
            implementation_specs=tuple(specs.values()),
            runtime_factories={"fake-device": lambda: runtime},
        ),
        specs,
    )


def _decision(
    node_id: str,
    operation_id: str,
    spec: OperationComputeSpec,
    *,
    memory_bytes: int = 0,
) -> NodeExecutionDecision:
    return NodeExecutionDecision(
        node_id=node_id,
        operation_id=operation_id,
        requested_preference=NodeComputePreference(),
        runtime_id=spec.runtime_id,
        implementation_library_id=spec.implementation_library_id,
        implementation_id=spec.implementation_id,
        decision_kind=DecisionKind.SELECTED,
        reason=DecisionReason.SELECTED_IMPLEMENTATION,
        reason_text="Selected by the fake test policy.",
        memory_estimate=MemoryEstimate(
            total_device_peak_bytes=memory_bytes,
            model_id="fake-v1",
        ),
    )


def _decisions(
    pipeline: PrototypePipeline,
    specs: dict[str, OperationComputeSpec],
    *,
    memory_bytes: int = 0,
) -> dict[str, NodeExecutionDecision]:
    return {
        node_id: _decision(
            node_id,
            node.operation_id,
            specs[node.operation_id],
            memory_bytes=memory_bytes,
        )
        for node_id, node in pipeline.nodes.items()
        if node.operation_id in specs
    }


def _prepare_call(pipeline: PrototypePipeline):
    def prepare(node_id: str, inputs: tuple[object, ...]) -> PreparedNodeCall:
        node = pipeline.nodes[node_id]
        operation = pipeline.operation_spec(node.operation_id)
        assert operation.function is not None
        return PreparedNodeCall(
            node_id=node_id,
            operation_id=node.operation_id,
            cpu_function=operation.function,
            inputs=inputs,
            kwargs=pipeline._operation_kwargs(node),
            multiple_inputs=pipeline._node_accepts_multiple_inputs(node),
            output_port_count=len(pipeline.output_ports(node_id)),
        )

    return prepare


def _request(
    fallback: FallbackPolicy = FallbackPolicy.VISIBLE,
    *,
    memory_cap: int | None = None,
) -> ComputeRequest:
    return ComputeRequest(
        mode=ComputeMode.AUTO,
        fallback_policy=fallback,
        runtime_id="fake-device",
        device_id="fake:0",
        accelerator_memory_cap_bytes=memory_cap,
    )


def test_provider_keyword_arguments_omit_runtime_private_parameters():
    call = PreparedNodeCall(
        node_id="coloc",
        operation_id="colocalized_voxels",
        cpu_function=lambda value, **_kwargs: value,
        inputs=(np.zeros((2, 2), dtype=np.float32),),
        kwargs={
            "threshold_mode": "Costes auto",
            "_vipp_resolved_costes": {"pearson_below": float("nan")},
        },
    )

    assert device_execution_module._provider_keyword_arguments(call) == {
        "threshold_mode": "Costes auto"
    }


def test_linear_segment_keeps_intermediate_on_device_and_returns_host_only():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success

    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_copy), ("median_filter", _device_copy)),
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    assert [type(unit) for unit in plan.units] == [
        HostExecutionUnit,
        DeviceSegmentUnit,
    ]
    segment = plan.segments[0]
    assert segment.node_ids == (gaussian.id, median.id)
    assert segment.entry_ports == (OutputPortKey("input", 0),)
    assert segment.exit_ports == (OutputPortKey(median.id, 0),)

    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
    )

    np.testing.assert_array_equal(result.host_values[OutputPortKey(median.id, 0)], data)
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert runtime.operation_count == 2
    assert runtime.live == {}
    assert all(
        not runtime.is_device_value(value) for value in result.host_values.values()
    )
    assert result.telemetry is None
    assert runtime.memory_snapshot_states == [("fake:0", 0, False)]
    registry.close()


def test_device_telemetry_observes_directional_transfers_and_device_phases(
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success

    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_copy), ("median_filter", _device_copy)),
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    lease_depth = 0

    @contextmanager
    def observed_lease(_runtime_id, _device_id, **_kwargs):
        nonlocal lease_depth
        lease_depth += 1
        try:
            yield
        finally:
            lease_depth -= 1

    original_memory_snapshot = runtime.memory_snapshot

    def observed_memory_snapshot(*, device_id=""):
        if runtime.memory_snapshot_states:
            assert lease_depth == 1
        return original_memory_snapshot(device_id=device_id)

    monkeypatch.setattr(device_execution_module, "accelerator_lease", observed_lease)
    monkeypatch.setattr(runtime, "memory_snapshot", observed_memory_snapshot)

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
        telemetry=DeviceExecutionTelemetryConfig(
            clock=_SteppingClock(),
            synchronize_device_phases=True,
        ),
    )

    observation = result.telemetry
    assert observation is not None
    assert observation.synchronized_device_phases is True
    assert observation.host_to_device.count == 1
    assert observation.host_to_device.succeeded_count == 1
    assert observation.host_to_device.byte_count == data.nbytes
    assert observation.host_to_device.unknown_byte_count == 0
    assert observation.host_to_device.elapsed_seconds == pytest.approx(0.75)
    assert observation.host_to_device.all_synchronized is True
    assert observation.device_to_host.count == 1
    assert observation.device_to_host.byte_count == data.nbytes
    assert observation.device_to_host.elapsed_seconds == pytest.approx(0.75)
    assert observation.device_to_host.all_synchronized is True

    lease_spans = observation.spans_for(DeviceExecutionPhase.ACCELERATOR_LEASE_WAIT)
    lease_identities = [
        (span.runtime_id, span.device_id, span.segment_id) for span in lease_spans
    ]
    assert lease_identities == [("fake-device", "fake:0", "")]
    assert lease_spans[0].elapsed_seconds == pytest.approx(0.25)
    preflight_spans = observation.spans_for(DeviceExecutionPhase.PREFLIGHT)
    assert [span.segment_id for span in preflight_spans] == [
        plan.segments[0].segment_id
    ]
    assert preflight_spans[0].elapsed_seconds == pytest.approx(0.25)

    operation_spans = observation.spans_for(DeviceExecutionPhase.DEVICE_OPERATION)
    assert [
        (
            span.node_id,
            span.operation_id,
            span.implementation_id,
            span.segment_id,
            span.runtime_id,
            span.device_id,
        )
        for span in operation_spans
    ] == [
        (
            gaussian.id,
            "gaussian_blur",
            "fake-gaussian_blur-v1",
            plan.segments[0].segment_id,
            "fake-device",
            "fake:0",
        ),
        (
            median.id,
            "median_filter",
            "fake-median_filter-v1",
            plan.segments[0].segment_id,
            "fake-device",
            "fake:0",
        ),
    ]
    assert all(span.elapsed_seconds == pytest.approx(0.75) for span in operation_spans)
    resolution_spans = observation.spans_for(
        DeviceExecutionPhase.IMPLEMENTATION_RESOLUTION
    )
    assert [span.node_id for span in resolution_spans] == [gaussian.id, median.id]
    assert all(span.elapsed_seconds == pytest.approx(0.25) for span in resolution_spans)

    synchronization_spans = observation.spans_for(
        DeviceExecutionPhase.DEVICE_SYNCHRONIZE
    )
    assert [span.synchronization_point for span in synchronization_spans] == [
        DeviceSynchronizationPoint.AFTER_HOST_TO_DEVICE,
        DeviceSynchronizationPoint.AFTER_DEVICE_OPERATION,
        DeviceSynchronizationPoint.AFTER_DEVICE_OPERATION,
        DeviceSynchronizationPoint.AFTER_DEVICE_TO_HOST,
        DeviceSynchronizationPoint.SEGMENT_COMPLETE,
    ]
    assert all(
        span.elapsed_seconds == pytest.approx(0.25) for span in synchronization_spans
    )
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert runtime.operation_count == 2
    assert runtime.live == {}
    assert observation.terminal_memory_snapshots
    terminal = observation.terminal_memory_snapshots[0]
    assert terminal.runtime_id == "fake-device"
    assert terminal.device_id == "fake:0"
    assert terminal.private_allocations_released is True
    assert terminal.runtime_live_bytes == 0
    assert terminal.runtime_reserved_bytes == 0
    terminal_spans = observation.spans_for(
        DeviceExecutionPhase.TERMINAL_MEMORY_SNAPSHOT
    )
    assert len(terminal_spans) == 1
    assert terminal_spans[0].succeeded is True
    assert runtime.memory_snapshot_states[-1] == ("fake:0", 0, False)
    assert lease_depth == 0
    registry.close()


def test_terminal_memory_snapshot_failure_cannot_change_scientific_result():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _TerminalSnapshotFailureRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
        telemetry=DeviceExecutionTelemetryConfig(clock=_SteppingClock()),
    )

    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(gaussian.id, 0)],
        data,
    )
    assert result.telemetry is not None
    assert result.telemetry.terminal_memory_snapshots == ()
    terminal_spans = result.telemetry.spans_for(
        DeviceExecutionPhase.TERMINAL_MEMORY_SNAPSHOT
    )
    assert len(terminal_spans) == 1
    assert terminal_spans[0].succeeded is False
    assert runtime.live == {}
    registry.close()


def test_terminal_memory_snapshot_rejects_mismatched_provider_identity():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _TerminalSnapshotIdentityRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
        telemetry=DeviceExecutionTelemetryConfig(clock=_SteppingClock()),
    )

    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(gaussian.id, 0)],
        data,
    )
    assert result.telemetry is not None
    assert result.telemetry.terminal_memory_snapshots == ()
    terminal_spans = result.telemetry.spans_for(
        DeviceExecutionPhase.TERMINAL_MEMORY_SNAPSHOT
    )
    assert len(terminal_spans) == 1
    assert terminal_spans[0].succeeded is False
    registry.close()


def test_device_plan_rejects_a_different_execution_request_before_runtime_work():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    different_request = replace(request, device_id="fake:1")

    with pytest.raises(DevicePlanningError, match="does not match"):
        preflight_device_execution(plan, registry, different_request)
    with pytest.raises(DevicePlanningError, match="does not match"):
        execute_device_plan(
            plan,
            pipeline,
            registry,
            different_request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4))},
            prepare_call=_prepare_call(pipeline),
        )

    assert runtime.events == []
    assert runtime.memory_snapshot_states == []
    registry.close()


def test_device_plan_rejects_a_decision_outside_explicit_runtime_affinity():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = replace(_request(), runtime_id="different-runtime")

    with pytest.raises(DevicePlanningError, match="compute request requires"):
        plan_device_execution(
            pipeline,
            _decisions(pipeline, specs),
            registry,
            request,
        )

    assert runtime.events == []
    registry.close()


def test_explicit_missing_device_is_rejected_before_memory_or_scope_work():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = replace(_request(), device_id="fake:1")
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    with pytest.raises(DevicePlanningError, match="did not report requested device"):
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4))},
            prepare_call=_prepare_call(pipeline),
        )

    assert runtime.memory_snapshot_states == []
    assert runtime.events == []
    registry.close()


def test_telemetry_failures_cannot_change_scientific_result_or_primary_error():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): _HostileNbytesHost()},
        prepare_call=_prepare_call(pipeline),
        telemetry=DeviceExecutionTelemetryConfig(clock=_HostileClock()),
    )
    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(gaussian.id, 0)],
        np.arange(25, dtype=np.float32).reshape(5, 5),
    )
    assert runtime.live == {}
    registry.close()

    failing_runtime = _FakeRuntime()
    failing_registry, failing_specs = _registry(
        failing_runtime,
        (("gaussian_blur", _device_fail),),
    )
    failing_plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, failing_specs),
        failing_registry,
        request,
    )
    with pytest.raises(DeviceExecutionError) as raised:
        execute_device_plan(
            failing_plan,
            pipeline,
            failing_registry,
            request,
            host_values={OutputPortKey("input", 0): _HostileNbytesHost()},
            prepare_call=_prepare_call(pipeline),
            telemetry=DeviceExecutionTelemetryConfig(clock=_HostileClock()),
        )
    assert raised.value.failure.reason_code == "fake_kernel_failure"
    assert failing_runtime.live == {}
    failing_registry.close()


def test_failed_diagnostic_barrier_is_not_reported_as_synchronized():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _SynchronizeOOMOnceRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request(FallbackPolicy.VISIBLE)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
        telemetry=DeviceExecutionTelemetryConfig(
            clock=_SteppingClock(),
            synchronize_device_phases=True,
        ),
    )

    assert result.fallback_segment_ids == (plan.segments[0].segment_id,)
    assert result.telemetry is not None
    transfer_spans = result.telemetry.spans_for(DeviceExecutionPhase.HOST_TO_DEVICE)
    assert len(transfer_spans) == 1
    assert transfer_spans[0].succeeded is False
    assert transfer_spans[0].synchronized is False
    assert result.telemetry.host_to_device.all_synchronized is False
    synchronization_spans = result.telemetry.spans_for(
        DeviceExecutionPhase.DEVICE_SYNCHRONIZE
    )
    failed_barrier = next(
        span
        for span in synchronization_spans
        if span.synchronization_point is DeviceSynchronizationPoint.AFTER_HOST_TO_DEVICE
    )
    assert failed_barrier.succeeded is False
    assert failed_barrier.synchronized is False
    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(gaussian.id, 0)],
        data,
    )
    assert len(result.telemetry.terminal_memory_snapshots) == 1
    assert result.telemetry.terminal_memory_snapshots[0].private_allocations_released
    assert runtime.live == {}
    registry.close()


def test_allocation_sharing_view_stays_live_and_releases_its_allocation_once():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    extract = pipeline.add_node("extract_channel")
    assert pipeline.connect("input", extract.id).success

    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("extract_channel", _device_alias),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
    )

    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(extract.id, 0)],
        data,
    )
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert runtime.operation_count == 1
    assert runtime.release_count == 1
    assert runtime.live == {}
    registry.close()


def test_device_plan_cancels_while_waiting_for_process_accelerator_lease():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    lease_entered = threading.Event()
    release_lease = threading.Event()

    def holder() -> None:
        with accelerator_lease("fake-device", "fake:0"):
            lease_entered.set()
            assert release_lease.wait(timeout=5)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert lease_entered.wait(timeout=5)
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    try:
        with pytest.raises(DeviceExecutionCancelled, match="waiting") as raised:
            execute_device_plan(
                plan,
                pipeline,
                registry,
                request,
                host_values={
                    OutputPortKey("input", 0): np.ones((4, 4), dtype=np.float32)
                },
                prepare_call=_prepare_call(pipeline),
                cancel_callback=cancelled.is_set,
            )
        assert raised.value.cleanup_succeeded is True
        assert raised.value.fallback_records == ()
    finally:
        timer.join(timeout=5)
        release_lease.set()
        holder_thread.join(timeout=5)
    assert not holder_thread.is_alive()
    assert not any(
        isinstance(event, tuple) and event[0] == "scope-enter"
        for event in runtime.events
    )
    registry.close()


@pytest.mark.parametrize(
    ("common_claims", "expected_segments"),
    [
        ((), 2),
        (("fake-zero-copy-v1",), 1),
    ],
)
def test_cross_library_residency_requires_a_common_interoperability_contract(
    common_claims,
    expected_segments,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success

    runtime = _FakeRuntime()
    first = _implementation_spec("gaussian_blur", _device_copy)
    second = replace(
        _implementation_spec("median_filter", _device_copy),
        implementation_library_id="fake-library-b",
    )
    registry = ComputeRegistry(
        runtime_descriptors=(
            _runtime_descriptor(interoperability_claims=common_claims),
        ),
        library_descriptors=(
            _library_descriptor(interoperability_claims=common_claims),
            _library_descriptor(
                "fake-library-b",
                interoperability_claims=common_claims,
            ),
        ),
        implementation_specs=(first, second),
        runtime_factories={"fake-device": lambda: runtime},
    )
    request = _request()
    decisions = {
        gaussian.id: _decision(gaussian.id, "gaussian_blur", first),
        median.id: _decision(median.id, "median_filter", second),
    }

    plan = plan_device_execution(pipeline, decisions, registry, request)

    assert len(plan.segments) == expected_segments
    if common_claims:
        assert plan.segments[0].node_ids == (gaussian.id, median.id)
    else:
        assert tuple(segment.node_ids for segment in plan.segments) == (
            (gaussian.id,),
            (median.id,),
        )
    registry.close()


def test_branch_join_multi_output_liveness_and_retained_ports():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    split = pipeline.add_node("split_channels")
    split.params[DYNAMIC_OUTPUT_COUNT_PARAM] = 2
    first = pipeline.add_node("gaussian_blur")
    second = pipeline.add_node("gaussian_blur")
    join = pipeline.add_node("add_images")
    assert pipeline.connect("input", split.id).success
    assert pipeline.connect(split.id, first.id, source_port=0).success
    assert pipeline.connect(split.id, second.id, source_port=1).success
    assert pipeline.connect(first.id, join.id, target_port=0).success
    assert pipeline.connect(second.id, join.id, target_port=1).success

    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (
            ("split_channels", _device_split),
            ("gaussian_blur", _device_copy),
            ("add_images", _device_add),
        ),
    )
    retained = OutputPortKey(split.id, 1)
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
        retained_ports=(retained,),
    )

    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.node_ids == (split.id, first.id, second.id, join.id)
    assert segment.retained_ports == (retained,)
    assert retained in segment.exit_ports
    remaining = dict(segment.remaining_consumers)
    assert remaining == {
        OutputPortKey("input", 0): 1,
        OutputPortKey(split.id, 0): 1,
        OutputPortKey(split.id, 1): 1,
        OutputPortKey(first.id, 0): 1,
        OutputPortKey(second.id, 0): 1,
    }

    data = np.stack(
        (
            np.full((4, 4), 3, dtype=np.float32),
            np.full((4, 4), 7, dtype=np.float32),
        )
    )
    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
    )

    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(join.id, 0)],
        np.full((4, 4), 10, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        result.host_values[retained],
        np.full((4, 4), 7, dtype=np.float32),
    )
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 2
    assert runtime.live == {}
    registry.close()


def test_host_finalizer_terminates_even_an_alternate_branch_join_path():
    global _HOST_FINALIZER_RUNTIME

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    upstream = pipeline.add_node("median_filter")
    terminal = pipeline.add_node("gaussian_blur")
    join = pipeline.add_node("add_images")
    pipeline.set_param(upstream.id, "size", 1)
    pipeline.set_param(terminal.id, "sigma", 0.0)
    assert pipeline.connect("input", upstream.id).success
    assert pipeline.connect(upstream.id, terminal.id).success
    assert pipeline.connect(upstream.id, join.id, target_port=0).success
    assert pipeline.connect(terminal.id, join.id, target_port=1).success

    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (
            ("median_filter", _device_copy),
            ("gaussian_blur", _device_copy),
            ("add_images", _device_add),
        ),
        host_finalizer_refs={"gaussian_blur": f"{__name__}:_host_array_finalizer"},
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    assert tuple(segment.node_ids for segment in plan.segments) == (
        (upstream.id, terminal.id),
        (join.id,),
    )
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    observed: list[tuple[str, tuple[object, ...], str]] = []

    def observe(node_id, call, outputs, runtime_id):
        if node_id == terminal.id:
            assert not runtime.scope_active
            assert runtime.live == {}
            assert call.inputs == (None,)
            np.testing.assert_array_equal(outputs[0], data + 10)
        observed.append((node_id, outputs, runtime_id))

    _HOST_FINALIZER_CALLS.clear()
    _HOST_FINALIZER_RUNTIME = runtime
    try:
        result = execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): data},
            prepare_call=_prepare_call(pipeline),
            node_outputs_callback=observe,
            telemetry=DeviceExecutionTelemetryConfig(clock=_SteppingClock()),
        )
    finally:
        _HOST_FINALIZER_RUNTIME = None

    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(join.id, 0)],
        data * 2 + 10,
    )
    assert len(_HOST_FINALIZER_CALLS) == 1
    assert [node_id for node_id, _outputs, _runtime_id in observed] == [
        upstream.id,
        terminal.id,
        join.id,
    ]
    assert runtime.host_to_device_count == 3
    assert runtime.device_to_host_count == 3
    assert runtime.operation_count == 3
    assert runtime.live == {}
    assert result.telemetry is not None
    finalizer_spans = result.telemetry.spans_for(DeviceExecutionPhase.HOST_FINALIZER)
    assert len(finalizer_spans) == 1
    assert (
        finalizer_spans[0].node_id,
        finalizer_spans[0].operation_id,
        finalizer_spans[0].implementation_id,
        finalizer_spans[0].segment_id,
        finalizer_spans[0].runtime_id,
        finalizer_spans[0].device_id,
    ) == (
        terminal.id,
        "gaussian_blur",
        "fake-gaussian_blur-v1",
        plan.segments[0].segment_id,
        "fake-device",
        "fake:0",
    )
    assert finalizer_spans[0].elapsed_seconds == pytest.approx(0.25)
    registry.close()


def test_cancellation_after_payload_transfer_skips_finalizer_and_cleans_up():
    global _HOST_FINALIZER_RUNTIME

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_copy),),
        host_finalizer_refs={"gaussian_blur": f"{__name__}:_host_array_finalizer"},
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    callbacks: list[str] = []
    _HOST_FINALIZER_CALLS.clear()
    _HOST_FINALIZER_RUNTIME = runtime
    try:
        with pytest.raises(OperationCancelled):
            execute_device_plan(
                plan,
                pipeline,
                registry,
                request,
                host_values={
                    OutputPortKey("input", 0): np.ones((4, 4), dtype=np.float32)
                },
                prepare_call=_prepare_call(pipeline),
                cancel_callback=lambda: runtime.device_to_host_count >= 1,
                node_outputs_callback=(
                    lambda node_id, _call, _outputs, _runtime_id: callbacks.append(
                        node_id
                    )
                ),
            )
    finally:
        _HOST_FINALIZER_RUNTIME = None

    assert runtime.device_to_host_count == 1
    assert runtime.live == {}
    assert _HOST_FINALIZER_CALLS == []
    assert callbacks == []
    registry.close()


def test_cancellation_during_call_preparation_detaches_private_input_traceback():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    gaussian.params["sigma"] = 0.0
    assert pipeline.connect("input", gaussian.id).success
    runtime = _AliasCheckingRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    def cancel_prepare(_node_id: str, inputs: tuple[object, ...]):
        device_input_alias = inputs[0]
        assert runtime.is_device_value(device_input_alias)
        raise OperationCancelled("cancelled after the target node started")

    with pytest.raises(DeviceExecutionCancelled) as caught:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((8, 8), dtype=np.float32)},
            prepare_call=cancel_prepare,
        )

    assert caught.value.cleanup_succeeded is True
    assert str(caught.value) == "cancelled after the target node started"
    assert runtime.live == {}
    gc.collect()
    assert all(reference() is None for reference in runtime.released_references)

    reused = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={
            OutputPortKey("input", 0): np.arange(64, dtype=np.float32).reshape(8, 8)
        },
        prepare_call=_prepare_call(pipeline),
    )

    np.testing.assert_array_equal(
        reused.host_values[OutputPortKey(gaussian.id, 0)],
        np.arange(64, dtype=np.float32).reshape(8, 8),
    )
    assert runtime.live == {}
    gc.collect()
    assert all(reference() is None for reference in runtime.released_references)
    registry.close()


def test_visible_cpu_fallback_bypasses_device_host_finalizer():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    runtime.oom_remaining = 1
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_oom_once),),
        host_finalizer_refs={"gaussian_blur": "missing.host_finalizer.module:finalize"},
    )
    request = _request(FallbackPolicy.VISIBLE)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    observed_runtimes: list[str] = []

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
        node_outputs_callback=(
            lambda _node_id, _call, _outputs, runtime_id: observed_runtimes.append(
                runtime_id
            )
        ),
    )

    assert result.fallback_segment_ids == (plan.segments[0].segment_id,)
    np.testing.assert_array_equal(
        result.host_values[OutputPortKey(gaussian.id, 0)],
        data,
    )
    assert observed_runtimes == ["cpu-numpy"]
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    registry.close()


def test_host_finalizer_cannot_hide_a_device_value_inside_table_data():
    global _HOST_FINALIZER_RUNTIME

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_copy),),
        host_finalizer_refs={"gaussian_blur": f"{__name__}:_escaping_table_finalizer"},
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    callbacks: list[str] = []
    _HOST_FINALIZER_RUNTIME = runtime
    try:
        with pytest.raises(DeviceExecutionError) as error:
            execute_device_plan(
                plan,
                pipeline,
                registry,
                request,
                host_values={
                    OutputPortKey("input", 0): np.ones((4, 4), dtype=np.float32)
                },
                prepare_call=_prepare_call(pipeline),
                node_outputs_callback=(
                    lambda node_id, _call, _outputs, _runtime_id: callbacks.append(
                        node_id
                    )
                ),
            )
    finally:
        _HOST_FINALIZER_RUNTIME = None

    assert error.value.failure.reason_code == "fake_unknown"
    assert runtime.classified_inside_scope[-1] is False
    assert runtime.device_to_host_count == 1
    assert runtime.live == {}
    assert callbacks == []
    registry.close()


def test_source_and_writer_are_host_boundaries_and_memory_fails_before_calls():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    writer = pipeline.add_node("batch_output")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, writer.id).success

    runtime = _FakeRuntime(free_bytes=1_000)
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request(memory_cap=50)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs, memory_bytes=100),
        registry,
        request,
    )

    assert isinstance(plan.units[0], HostExecutionUnit)
    assert plan.units[0].source_boundary
    assert isinstance(plan.units[1], DeviceSegmentUnit)
    assert isinstance(plan.units[2], HostExecutionUnit)
    assert plan.units[2].writer_boundary
    prepared: list[str] = []

    def forbidden_prepare(node_id: str, _inputs: tuple[object, ...]):
        prepared.append(node_id)
        raise AssertionError("Preflight must happen before any scientific call.")

    with pytest.raises(DeviceMemoryPreflightError) as error:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4))},
            prepare_call=forbidden_prepare,
        )

    assert error.value.segment_id == plan.segments[0].segment_id
    assert error.value.required_bytes == 100
    assert error.value.available_bytes == 50
    assert prepared == []
    assert runtime.host_to_device_count == 0
    assert runtime.operation_count == 0
    registry.close()


def test_cancellation_releases_all_live_device_values_without_materializing():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_copy), ("median_filter", _device_copy)),
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    with pytest.raises(DeviceExecutionCancelled) as raised:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4))},
            prepare_call=_prepare_call(pipeline),
            cancel_callback=lambda: runtime.operation_count >= 1,
        )

    assert raised.value.cleanup_succeeded is True
    assert runtime.operation_count == 1
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    registry.close()


def test_kernel_failure_is_not_retried_and_transaction_is_cleaned():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_copy), ("median_filter", _device_fail)),
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    with pytest.raises(DeviceExecutionError) as error:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4))},
            prepare_call=_prepare_call(pipeline),
        )

    assert error.value.failure.kind is RuntimeExceptionKind.KERNEL_FAILURE
    assert runtime.operation_count == 2
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    registry.close()


def test_partial_multi_output_registration_releases_unowned_output():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    split = pipeline.add_node("split_channels")
    split.params[DYNAMIC_OUTPUT_COUNT_PARAM] = 2
    assert pipeline.connect("input", split.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (("split_channels", _device_split_registration_failure),),
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.ones((2, 4, 4), dtype=np.float32)

    with pytest.raises(DeviceExecutionError) as error:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): data},
            prepare_call=_prepare_call(pipeline),
        )

    assert error.value.failure.kind is RuntimeExceptionKind.KERNEL_FAILURE
    assert runtime.release_count == 3  # entry plus both returned outputs
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    registry.close()


@pytest.mark.parametrize(
    ("fallback_policy", "expects_fallback"),
    ((FallbackPolicy.VISIBLE, True), (FallbackPolicy.STRICT, False)),
)
def test_only_typed_retryable_oom_gets_one_visible_cpu_fallback(
    fallback_policy: FallbackPolicy,
    expects_fallback: bool,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    runtime.oom_remaining = 1
    registry, specs = _registry(runtime, (("gaussian_blur", _device_oom_once),))
    request = _request(fallback_policy)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    if not expects_fallback:
        with pytest.raises(DeviceExecutionError) as error:
            execute_device_plan(
                plan,
                pipeline,
                registry,
                request,
                host_values={OutputPortKey("input", 0): data},
                prepare_call=_prepare_call(pipeline),
            )
        assert error.value.failure.kind is RuntimeExceptionKind.OUT_OF_MEMORY
        assert error.value.cleanup_succeeded is True
    else:
        result = execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): data},
            prepare_call=_prepare_call(pipeline),
        )
        assert result.fallback_segment_ids == (plan.segments[0].segment_id,)
        assert len(result.fallback_records) == 1
        assert result.fallback_records[0].reason_code == "fake_oom"
        assert result.fallback_records[0].cpu_retry_succeeded is True
        np.testing.assert_allclose(
            result.host_values[OutputPortKey(gaussian.id, 0)],
            data,
        )
        assert all(
            not runtime.is_device_value(value) for value in result.host_values.values()
        )

    assert runtime.operation_count == 1
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    assert runtime.classified_inside_scope == [True]
    registry.close()


@pytest.mark.parametrize("cancelled", (False, True))
def test_oom_record_survives_failed_or_cancelled_cpu_retry(cancelled: bool):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    runtime.oom_remaining = 1
    registry, specs = _registry(runtime, (("gaussian_blur", _device_oom_once),))
    request = _request(FallbackPolicy.VISIBLE)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    def fail_retry(*_args):
        if cancelled:
            raise OperationCancelled("cancelled during CPU retry")
        raise ValueError("CPU retry failed")

    exception_type = DeviceExecutionCancelled if cancelled else ValueError
    with pytest.raises(exception_type) as raised:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((5, 5), dtype=np.float32)},
            prepare_call=_prepare_call(pipeline),
            node_outputs_callback=fail_retry,
        )

    records = (
        raised.value.fallback_records
        if cancelled
        else raised.value.vipp_fallback_records
    )
    assert len(records) == 1
    assert records[0].reason_code == "fake_oom"
    assert records[0].cpu_retry_count == 1
    assert records[0].cpu_retry_succeeded is False
    assert records[0].cleanup_succeeded is True
    assert runtime.live == {}
    registry.close()


def test_kernel_oom_traceback_is_detached_before_scope_cleanup_and_reuse():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    runtime.oom_remaining = 1
    registry, specs = _registry(
        runtime,
        (("gaussian_blur", _device_oom_once_with_traceback_scratch),),
    )
    request = _request(FallbackPolicy.VISIBLE)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    first = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
    )

    assert first.fallback_segment_ids == (plan.segments[0].segment_id,)
    assert runtime.classified_inside_scope == [True]
    assert runtime.traceback_scratch_live == 0
    assert runtime.live == {}

    second = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): data},
        prepare_call=_prepare_call(pipeline),
    )

    assert second.fallback_segment_ids == ()
    np.testing.assert_array_equal(
        second.host_values[OutputPortKey(gaussian.id, 0)],
        data,
    )
    assert runtime.traceback_scratch_live == 0
    assert runtime.live == {}
    assert runtime.operation_count == 2
    registry.close()


def test_real_cupy_kernel_oom_falls_back_and_runtime_remains_reusable():
    cupy = pytest.importorskip("cupy")
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.float32).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")

    base = next(
        spec
        for spec in compute_specs_for(
            "gaussian_blur",
            include_cpu=False,
            allow_experimental=True,
        )
        if spec.runtime_id == "cuda-cupy"
    )
    implementation = replace(
        base,
        implementation_id="test-cupy-kernel-oom-v1",
        callable_ref=f"{__name__}:_cupy_kernel_oom_or_copy",
    )
    registry = ComputeRegistry(implementation_specs=(implementation,))
    runtime = registry.runtime(implementation.runtime_id)
    probe = runtime.probe()
    if not probe.available or not probe.selected_device_id:
        registry.close()
        pytest.skip(probe.message or "The CUDA runtime is unavailable.")

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    request = ComputeRequest(
        mode=ComputeMode.AUTO,
        fallback_policy=FallbackPolicy.VISIBLE,
        runtime_id=implementation.runtime_id,
        device_id=probe.selected_device_id,
        accelerator_memory_cap_bytes=8 * 1024**2,
        accelerator_safety_reserve_bytes=0,
        allow_experimental=True,
    )
    plan = plan_device_execution(
        pipeline,
        {
            gaussian.id: _decision(
                gaussian.id,
                gaussian.operation_id,
                implementation,
            )
        },
        registry,
        request,
    )
    data = np.arange(81, dtype=np.float32).reshape(9, 9)

    try:
        first = execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): data},
            prepare_call=_prepare_call(pipeline),
        )

        assert first.fallback_segment_ids == (plan.segments[0].segment_id,)
        np.testing.assert_array_equal(
            first.host_values[OutputPortKey(gaussian.id, 0)],
            data,
        )
        first_terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert first_terminal.runtime_live_bytes == 0
        assert first_terminal.runtime_reserved_bytes == 0
        assert runtime.probe(refresh=True).available

        pipeline.set_param(gaussian.id, "sigma", 1.0)
        second = execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): data},
            prepare_call=_prepare_call(pipeline),
        )

        assert second.fallback_segment_ids == ()
        np.testing.assert_array_equal(
            second.host_values[OutputPortKey(gaussian.id, 0)],
            data,
        )
        second_terminal = runtime.memory_snapshot(device_id=probe.selected_device_id)
        assert second_terminal.runtime_live_bytes == 0
        assert second_terminal.runtime_reserved_bytes == 0
        assert runtime.probe(refresh=True).available
    finally:
        registry.close()


def test_cleanup_failure_overrides_oom_and_prevents_visible_cpu_fallback():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    runtime.oom_remaining = 1
    runtime.cleanup_fails_after_oom = True
    registry, specs = _registry(runtime, (("gaussian_blur", _device_oom_once),))
    request = _request(FallbackPolicy.VISIBLE)
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    with pytest.raises(DeviceExecutionError) as error:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4), dtype=np.float32)},
            prepare_call=_prepare_call(pipeline),
        )

    assert error.value.failure.kind is RuntimeExceptionKind.KERNEL_FAILURE
    assert error.value.failure.reason_code == "fake_cleanup_incomplete"
    assert not error.value.failure.retryable
    assert runtime.operation_count == 1
    assert runtime.live == {}
    registry.close()


def test_resident_callback_borrows_only_materialized_ports_after_d2h():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 1)
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(
        runtime,
        (
            ("gaussian_blur", _device_copy),
            ("median_filter", _device_copy),
        ),
    )
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )
    observed: list[tuple[OutputPortKey, str]] = []

    def observe(port, value, borrowed_runtime, device_id):
        assert runtime.scope_active
        assert borrowed_runtime is runtime
        assert isinstance(value, _FakeDeviceArray)
        assert not value.released
        assert runtime.device_to_host_count == 1
        assert runtime.events[-1] == "to-host"
        observed.append((port, device_id))

    result = execute_device_plan(
        plan,
        pipeline,
        registry,
        request,
        host_values={OutputPortKey("input", 0): np.ones((4, 4), dtype=np.float32)},
        prepare_call=_prepare_call(pipeline),
        resident_output_callback=observe,
    )

    assert observed == [(OutputPortKey(median.id, 0), "fake:0")]
    assert OutputPortKey(gaussian.id, 0) not in result.host_values
    assert runtime.device_to_host_count == 1
    assert runtime.live == {}
    registry.close()


def test_resident_callback_cancellation_propagates_after_scoped_cleanup():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _FakeRuntime()
    registry, specs = _registry(runtime, (("gaussian_blur", _device_copy),))
    request = _request()
    plan = plan_device_execution(
        pipeline,
        _decisions(pipeline, specs),
        registry,
        request,
    )

    def cancel_resident(_port, _value, _runtime, _device_id):
        raise OperationCancelled("resident presentation cancelled")

    with pytest.raises(DeviceExecutionCancelled) as error:
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4), dtype=np.float32)},
            prepare_call=_prepare_call(pipeline),
            resident_output_callback=cancel_resident,
        )

    assert error.value.cleanup_succeeded
    assert runtime.device_to_host_count == 1
    assert runtime.live == {}
    registry.close()
