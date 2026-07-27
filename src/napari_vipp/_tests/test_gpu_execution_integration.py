from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pytest

from napari_vipp._tests.test_device_execution import (
    _device_copy,
    _device_oom_once,
    _FakeDeviceArray,
    _FakeRuntime,
    _implementation_spec,
    _library_descriptor,
    _runtime_descriptor,
)
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionPlan,
    MemoryEstimate,
    NodeComputePreference,
    NodeExecutionDecision,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import OperationComputeSpec
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.pipeline import EXECUTION_READY, PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


class _ShapeAwareDeviceArray(_FakeDeviceArray):
    @property
    def shape(self):
        return self.payload.shape

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def ndim(self):
        return self.payload.ndim


class _ShapeAwareRuntime(_FakeRuntime):
    def allocate(self, payload):
        value = _ShapeAwareDeviceArray(self, payload)
        self.live[id(value)] = value
        self.events.append("allocate")
        return value


@dataclass(frozen=True)
class _PlanningResult:
    request: ComputeRequest
    environment: ComputeEnvironment
    decisions: tuple[NodeExecutionDecision, ...]
    warnings: tuple[str, ...] = ()

    @property
    def decisions_by_node(self):
        return MappingProxyType(
            {decision.node_id: decision for decision in self.decisions}
        )

    def as_execution_plan(self, *, segments=()) -> ExecutionPlan:
        return ExecutionPlan(
            self.request.fingerprint,
            self.environment.fingerprint,
            tuple(segments),
            self.decisions,
            self.warnings,
        )


class _StaticPlanner:
    def __init__(
        self,
        request: ComputeRequest,
        decisions: tuple[NodeExecutionDecision, ...],
    ) -> None:
        self.request = request
        self.decisions = decisions
        self.workloads: tuple[WorkloadDescriptor, ...] = ()

    def __call__(self, request, workloads, **_kwargs):
        assert request is self.request
        self.workloads = tuple(workloads)
        return _PlanningResult(
            request,
            ComputeEnvironment(
                runtime_ids=("cpu-numpy", "fake-device"),
                implementation_libraries=("cpu", "fake-library"),
                device_id="fake:0",
                device_name="Fake device",
                device_class="test",
                memory_topology="discrete",
                total_accelerator_memory_bytes=10_000,
            ),
            self.decisions,
        )


def _shape_preserving_spec(spec: OperationComputeSpec) -> OperationComputeSpec:
    return replace(
        spec,
        shape_policy_id="shape-preserving-v1",
        output_ports=tuple(
            replace(
                port,
                shape_policy_id="shape-preserving-v1",
                output_dtype_policy_id="dtype-same-v1",
            )
            for port in spec.output_ports
        ),
    )


def _test_registry(
    runtime: _FakeRuntime,
    implementations: tuple[tuple[str, Callable], ...],
) -> tuple[ComputeRegistry, dict[str, OperationComputeSpec]]:
    shaped = {
        operation_id: _shape_preserving_spec(
            _implementation_spec(operation_id, function)
        )
        for operation_id, function in implementations
    }
    return (
        ComputeRegistry(
            runtime_descriptors=(_runtime_descriptor(),),
            library_descriptors=(_library_descriptor(),),
            implementation_specs=tuple(shaped.values()),
            runtime_factories={"fake-device": lambda: runtime},
        ),
        shaped,
    )


def _decision(
    node_id: str,
    spec: OperationComputeSpec,
) -> NodeExecutionDecision:
    return NodeExecutionDecision(
        node_id=node_id,
        operation_id=spec.operation_id,
        requested_preference=NodeComputePreference(),
        runtime_id=spec.runtime_id,
        implementation_library_id=spec.implementation_library_id,
        implementation_id=spec.implementation_id,
        decision_kind=DecisionKind.SELECTED,
        reason=DecisionReason.SELECTED_IMPLEMENTATION,
        reason_text="Selected by the integration-test planner.",
        memory_estimate=MemoryEstimate(model_id="fake-v1"),
    )


def _accelerated_request(
    pipeline: PrototypePipeline,
    data: np.ndarray,
    compute_request: ComputeRequest,
    *,
    retain_node_ids: frozenset[str] = frozenset(),
    prune_unretained: bool = False,
    cancel_event: threading.Event | None = None,
    manual_node_ids: frozenset[str] | None = None,
    dirty_node_ids: frozenset[str] | None = None,
    cached_pipeline: PrototypePipeline | None = None,
) -> PipelineRunRequest:
    cached = cached_pipeline
    return PipelineRunRequest(
        run_id=31,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads={},
        compute_request=compute_request,
        retain_node_ids=retain_node_ids,
        prune_unretained=prune_unretained,
        cancel_event=cancel_event,
        manual_node_ids=manual_node_ids,
        dirty_node_ids=dirty_node_ids,
        cached_outputs=None if cached is None else dict(cached.outputs),
        cached_output_states=(
            None if cached is None else dict(cached.output_states)
        ),
        cached_node_outputs=(
            None
            if cached is None
            else {
                node_id: list(outputs)
                for node_id, outputs in cached.node_outputs.items()
            }
        ),
        cached_node_output_states=(
            None
            if cached is None
            else {
                node_id: list(states)
                for node_id, states in cached.node_output_states.items()
            }
        ),
        cached_execution_states=(
            None if cached is None else dict(cached.node_execution_states)
        ),
        cached_execution_messages=(
            None if cached is None else dict(cached.node_execution_messages)
        ),
        completed_node_ids=(
            frozenset() if cached is None else frozenset(cached.completed_node_ids)
        ),
    )


def test_headless_device_chain_uses_one_transfer_and_propagates_metadata():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("subtract_background")
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(background.id, "radius", 1.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 1)
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (
            ("subtract_background", _device_copy),
            ("gaussian_blur", _device_copy),
            ("median_filter", _device_copy),
        ),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="fake-device",
        device_id="fake:0",
    )
    planner = _StaticPlanner(
        compute_request,
        tuple(
            _decision(node_id, specs[node.operation_id])
            for node_id, node in pipeline.nodes.items()
            if node.operation_id in specs
        ),
    )
    data = np.arange(63, dtype=np.float32).reshape(7, 9)
    finished = []

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            retain_node_ids=frozenset({median.id}),
            prune_unretained=True,
        ),
        node_finished_callback=finished.append,
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert runtime.operation_count == 3
    assert runtime.live == {}
    assert result.pipeline.outputs[background.id] is None
    assert result.pipeline.outputs[gaussian.id] is None
    np.testing.assert_array_equal(result.pipeline.outputs[median.id], data)
    assert result.pipeline.node_execution_states[background.id] == EXECUTION_READY
    assert result.pipeline.node_execution_states[gaussian.id] == EXECUTION_READY
    assert [item.node_id for item in finished] == [
        "input",
        background.id,
        gaussian.id,
        median.id,
    ]
    state = result.pipeline.output_states[median.id]
    assert state.axis_order == "YX"
    assert state.shape == data.shape
    assert state.dtype == "float32"
    assert state.history[-3:] == (
        "Subtract Background: radius 1 px, 2D YX",
        "Gaussian Blur",
        "Median Filter",
    )
    assert [workload.operation_id for workload in planner.workloads] == [
        "input",
        "subtract_background",
        "gaussian_blur",
        "median_filter",
    ]
    assert planner.workloads[-1].input_shapes == (data.shape,)
    registry.close()


def test_device_oom_reports_actual_cpu_fallback_decision():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    runtime = _ShapeAwareRuntime()
    runtime.oom_remaining = 1
    registry, specs = _test_registry(
        runtime,
        (("gaussian_blur", _device_oom_once),),
    )
    failing = specs["gaussian_blur"]
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="fake-device",
        device_id="fake:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(gaussian.id, failing),),
    )
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    result = execute_pipeline_request(
        _accelerated_request(pipeline, data, compute_request),
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    np.testing.assert_array_equal(result.pipeline.outputs[gaussian.id], data)
    assert result.execution_report is not None
    actual = result.execution_report.actual_decisions[0]
    assert actual.decision_kind is DecisionKind.FALLBACK_CPU
    assert actual.runtime_id == "cpu-numpy"
    assert actual.fallback_reason.value == "out_of_memory"
    assert result.execution_report.warnings
    registry.close()


def test_dirty_device_run_reuses_clean_cached_upstream_output():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 1)
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    data = np.arange(35, dtype=np.float32).reshape(5, 7)
    pipeline.run(data, input_metadata={"axes": "YX"})
    cached_gaussian = pipeline.outputs[gaussian.id]
    pipeline.set_param(median.id, "size", 3)

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (("median_filter", _device_copy),),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="fake-device",
        device_id="fake:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(median.id, specs["median_filter"]),),
    )

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            dirty_node_ids=frozenset({median.id}),
            cached_pipeline=pipeline,
        ),
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[gaussian.id] is cached_gaussian
    np.testing.assert_array_equal(result.pipeline.outputs[median.id], cached_gaussian)
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert [workload.node_id for workload in planner.workloads] == [median.id]
    registry.close()


def test_selected_manual_host_node_runs_after_a_device_predecessor():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(deconvolution.id, "iterations", 1)
    pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect("input", deconvolution.id, target_port=0).success
    assert pipeline.connect(gaussian.id, deconvolution.id, target_port=1).success

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (("gaussian_blur", _device_copy),),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="fake-device",
        device_id="fake:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(gaussian.id, specs["gaussian_blur"]),),
    )
    data = np.zeros((7, 7), dtype=np.float32)
    data[3, 3] = 1.0

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            manual_node_ids=frozenset({deconvolution.id}),
        ),
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.pipeline.outputs[deconvolution.id] is not None
    assert result.pipeline.node_execution_states[deconvolution.id] == EXECUTION_READY
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    registry.close()


def test_cancelled_device_request_does_not_publish_a_partial_pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    median = pipeline.add_node("median_filter")
    assert pipeline.connect("input", median.id).success
    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (("median_filter", _device_copy),),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="fake-device",
        device_id="fake:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(median.id, specs["median_filter"]),),
    )
    cancelled = threading.Event()
    cancelled.set()
    finished = []

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            np.arange(16, dtype=np.uint8).reshape(4, 4),
            compute_request,
            cancel_event=cancelled,
        ),
        node_finished_callback=finished.append,
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.cancelled
    assert result.pipeline is None
    assert finished == []
    assert runtime.host_to_device_count == 0
    assert runtime.device_to_host_count == 0
    registry.close()


def test_cpu_request_does_not_construct_or_call_accelerator_services():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(20, dtype=np.float32).reshape(4, 5)

    class _ForbiddenPlanner:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("CPU execution must not call compute planning.")

    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=32,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.CPU),
        ),
        compute_registry=object(),
        compute_planner=_ForbiddenPlanner(),
    )

    assert result.error == ""
    assert result.execution_report is None
    assert result.pipeline is not None
    np.testing.assert_array_equal(result.pipeline.outputs[gaussian.id], data)


def test_real_headless_background_gaussian_median_forms_one_device_segment():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    if importlib.util.find_spec("cucim") is None:
        pytest.skip("The optional cuCIM wheel is not installed.")
    try:
        import cupy

        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.float32).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("subtract_background")
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(background.id, "radius", 1.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    pipeline.set_param(median.id, "size", 3)
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    data = np.random.default_rng(42).uniform(0, 100, (11, 13)).astype(np.float32)
    pipeline.run(data, input_metadata={"axes": "YX"})
    expected = pipeline.outputs[median.id].copy()
    compute_request = ComputeRequest(
        mode=ComputeMode.SELECTIVE,
        node_preferences={
            background.id: (
                "implementation:cucim-subtract_background-v1"
            ),
            gaussian.id: "implementation:cupyx-gaussian-blur-v1",
            median.id: "implementation:cupyx-median-filter-v1",
        },
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        allow_experimental=True,
    )

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            retain_node_ids=frozenset({median.id}),
            prune_unretained=True,
        )
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    selected = {
        decision.node_id: decision
        for decision in result.execution_report.actual_decisions
        if decision.node_id in {background.id, gaussian.id, median.id}
    }
    assert all(
        decision.decision_kind is DecisionKind.SELECTED
        and decision.runtime_id == "cuda-cupy"
        for decision in selected.values()
    )
    assert len(result.execution_report.plan.segments) == 1
    assert result.execution_report.plan.segments[0].node_ids == (
        background.id,
        gaussian.id,
        median.id,
    )
    np.testing.assert_allclose(
        result.pipeline.outputs[median.id],
        expected,
        rtol=5e-6,
        atol=1e-6,
    )
