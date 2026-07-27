from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

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
    DeviceExecutionError,
    DeviceMemoryPreflightError,
    DeviceSegmentUnit,
    HostExecutionUnit,
    execute_device_plan,
    plan_device_execution,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.pipeline import (
    DYNAMIC_OUTPUT_COUNT_PARAM,
    PrototypePipeline,
)
from napari_vipp.core.progress import OperationCancelled


class _FakeOOM(RuntimeError):
    pass


class _FakeKernelFailure(RuntimeError):
    pass


class _FakeDeviceArray:
    """Opaque test value which host/NumPy code is forbidden to coerce."""

    def __init__(self, runtime: _FakeRuntime, payload: object) -> None:
        self.runtime = runtime
        self.payload = np.array(payload, copy=True)
        self.released = False

    def __array__(self, _dtype=None, copy=None):
        del copy
        raise AssertionError("Device values must remain opaque outside the runtime.")


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
        try:
            yield
        finally:
            self.events.append(("scope-exit", device_id))

    def is_device_value(self, value: object) -> bool:
        if id(value) == self.registration_failure_id:
            checks = self.registration_checks.get(id(value), 0) + 1
            self.registration_checks[id(value)] = checks
            if checks == 2:
                raise _FakeKernelFailure("synthetic output registration failure")
        return isinstance(value, _FakeDeviceArray)

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
        assert isinstance(value, _FakeDeviceArray)
        assert not value.released, "A device allocation was released twice."
        value.released = True
        self.live.pop(id(value))
        self.release_count += 1
        self.events.append("release")

    def synchronize(self, *, device_id: str = "") -> None:
        self.events.append(("synchronize", device_id))

    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot:
        return RuntimeMemorySnapshot(
            self.runtime_id,
            device_id or "fake:0",
            "discrete",
            device_total_bytes=self.free_bytes,
            device_free_bytes=self.free_bytes,
        )

    def classify_exception(self, exc: BaseException) -> RuntimeExceptionInfo:
        if isinstance(exc, _FakeOOM):
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


def _device_copy(value: _FakeDeviceArray, **_kwargs) -> _FakeDeviceArray:
    assert not value.released
    value.runtime.operation_count += 1
    return value.runtime.allocate(value.payload)


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


def _runtime_descriptor() -> RuntimeDescriptor:
    return RuntimeDescriptor(
        runtime_id="fake-device",
        display_name="Fake device",
        factory_ref=f"{__name__}:_unused_runtime_factory",
        array_domain="fake-array",
        device_domain="fake",
        supported_os_families=("Windows", "Linux", "macOS"),
    )


def _library_descriptor() -> ImplementationLibraryDescriptor:
    return ImplementationLibraryDescriptor(
        library_id="fake-library",
        display_name="Fake implementation library",
        runtime_ids=("fake-device",),
        array_domain="fake-array",
        supported_os_families=("Windows", "Linux", "macOS"),
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
) -> tuple[ComputeRegistry, dict[str, OperationComputeSpec]]:
    specs = {
        operation_id: _implementation_spec(operation_id, function)
        for operation_id, function in implementations
    }
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

    with pytest.raises(OperationCancelled):
        execute_device_plan(
            plan,
            pipeline,
            registry,
            request,
            host_values={OutputPortKey("input", 0): np.ones((4, 4))},
            prepare_call=_prepare_call(pipeline),
            cancel_callback=lambda: runtime.operation_count >= 1,
        )

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
        np.testing.assert_allclose(
            result.host_values[OutputPortKey(gaussian.id, 0)],
            data,
        )
        assert all(
            not runtime.is_device_value(value)
            for value in result.host_values.values()
        )

    assert runtime.operation_count == 1
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 0
    assert runtime.live == {}
    registry.close()
