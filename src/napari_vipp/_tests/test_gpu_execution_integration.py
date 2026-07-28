from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
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
    FallbackPolicy,
    MemoryEstimate,
    NodeComputePreference,
    NodeExecutionDecision,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import OperationComputeSpec, compute_specs_for
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
    runtime_id = "cuda-cupy"
    array_domain = "cuda-cupy"

    def allocate(self, payload):
        value = _ShapeAwareDeviceArray(self, payload)
        self.live[id(value)] = value
        self.events.append("allocate")
        return value

    def probe(self, *, refresh: bool = False) -> RuntimeProbeResult:
        del refresh
        return RuntimeProbeResult(
            self.runtime_id,
            True,
            version="14.1.1",
            devices=(
                RuntimeDevice(
                    "cuda:0",
                    "Fake device",
                    self.free_bytes,
                    metadata=(("compute_capability", "12.0"),),
                ),
            ),
            selected_device_id="cuda:0",
            environment_fingerprint="fake-runtime-environment-v1",
            metadata=(
                ("cuda_runtime_version", "13020"),
                ("driver_version", "13030"),
            ),
        )


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
        self.array_facts = MappingProxyType({})

    def __call__(self, request, workloads, **_kwargs):
        assert request is self.request
        self.workloads = tuple(workloads)
        self.array_facts = MappingProxyType(dict(_kwargs.get("array_facts", {})))
        return _PlanningResult(
            request,
            ComputeEnvironment(
                runtime_ids=("cpu-numpy", "cuda-cupy"),
                implementation_libraries=("cpu", "cupyx", "cucim"),
                device_id="cuda:0",
                device_name="Fake device",
                device_class="nvidia-cuda",
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
    shaped: dict[str, OperationComputeSpec] = {}
    for operation_id, function in implementations:
        uses_cucim = operation_id in {
            "rolling_ball_background",
            "subtract_background",
        }
        shaped[operation_id] = _shape_preserving_spec(
            replace(
                _implementation_spec(operation_id, function),
                runtime_id="cuda-cupy",
                array_domain="cuda-cupy",
                implementation_library_id=("cucim" if uses_cucim else "cupyx"),
                validated_environment_policy_id=(
                    "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v2"
                    if uses_cucim
                    else "cuda-cupy-14.1.1-cpython312-windows-native-v2"
                ),
            )
        )
    library_ids = {spec.implementation_library_id for spec in shaped.values()}
    library_probes = {}
    if "cupyx" in library_ids:
        library_probes["cupyx"] = lambda: ImplementationLibraryProbeResult(
            "cupyx",
            True,
            version="14.1.1",
        )
    if "cucim" in library_ids:
        library_probes["cucim"] = lambda: ImplementationLibraryProbeResult(
            "cucim",
            True,
            version="26.06.00",
            metadata=(
                ("environment_record_schema", "napari-vipp-gpu-environment"),
                ("environment_record_schema_version", "1"),
                ("environment_track", "cuda13"),
                ("cupy_distribution", "cupy-cuda13x"),
                ("cucim_distribution", "cucim-cu13"),
                ("cucim_distribution_version", "26.6.0"),
                (
                    "cucim_artifact_sha256",
                    "586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8",
                ),
            ),
        )
    return (
        ComputeRegistry(
            runtime_descriptors=(
                replace(
                    _runtime_descriptor(),
                    runtime_id="cuda-cupy",
                    array_domain="cuda-cupy",
                    device_domain="nvidia-cuda",
                    interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
                ),
            ),
            library_descriptors=tuple(
                replace(
                    _library_descriptor(library_id),
                    runtime_ids=("cuda-cupy",),
                    array_domain="cuda-cupy",
                    interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
                )
                for library_id in sorted(library_ids)
            ),
            implementation_specs=tuple(shaped.values()),
            runtime_factories={"cuda-cupy": lambda: runtime},
            library_probes=library_probes,
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
        cached_output_states=(None if cached is None else dict(cached.output_states)),
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
        cached_compute_provenance=(
            None if cached is None else dict(cached.node_compute_provenance)
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
        runtime_id="cuda-cupy",
        device_id="cuda:0",
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
    assert set(result.pipeline.node_compute_provenance) == {median.id}
    assert result.pipeline.node_compute_provenance[
        median.id
    ].actual_implementation.implementation_id == specs[
        "median_filter"
    ].implementation_id
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


def test_cpu_extract_channel_projects_shape_through_requested_mixed_chain():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    extract = pipeline.add_node("extract_channel")
    background = pipeline.add_node("subtract_background")
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(extract.id, "channel", 1)
    pipeline.set_param(background.id, "radius", 1.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 1)
    assert pipeline.connect("input", extract.id).success
    assert pipeline.connect(extract.id, background.id).success
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
        mode=ComputeMode.SELECTIVE,
        node_preferences={
            extract.id: "cpu",
            background.id: (
                f"implementation:{specs['subtract_background'].implementation_id}"
            ),
            gaussian.id: "cpu",
            median.id: (
                f"implementation:{specs['median_filter'].implementation_id}"
            ),
        },
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        fallback_policy=FallbackPolicy.STRICT,
    )
    planned_workloads: list[WorkloadDescriptor] = []

    def planner(request, workloads, **kwargs):
        planned_workloads.extend(workloads)
        return plan_compute_decisions(request, workloads, **kwargs)

    data = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)
    request = replace(
        _accelerated_request(pipeline, data, compute_request),
        input_metadata={"axes": "CYX"},
    )

    result = execute_pipeline_request(
        request,
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    np.testing.assert_array_equal(result.pipeline.outputs[median.id], data[1])
    workloads = {workload.node_id: workload for workload in planned_workloads}
    assert workloads[extract.id].input_shapes == ((2, 5, 7),)
    assert workloads[extract.id].input_dtypes == ("uint16",)
    assert workloads[background.id].input_shapes == ((5, 7),)
    assert workloads[background.id].input_dtypes == ("uint16",)
    assert workloads[gaussian.id].input_shapes == ((5, 7),)
    assert workloads[gaussian.id].input_dtypes == ("uint16",)
    assert workloads[median.id].input_shapes == ((5, 7),)
    assert workloads[median.id].input_dtypes == ("uint16",)
    actual = {
        decision.node_id: decision
        for decision in result.execution_report.actual_decisions
    }
    assert actual[extract.id].runtime_id == "cpu-numpy"
    assert actual[extract.id].reason is DecisionReason.EXPLICIT_CPU
    assert (
        actual[background.id].implementation_id
        == specs["subtract_background"].implementation_id
    )
    assert actual[gaussian.id].runtime_id == "cpu-numpy"
    assert actual[gaussian.id].reason is DecisionReason.EXPLICIT_CPU
    assert (
        actual[median.id].implementation_id
        == specs["median_filter"].implementation_id
    )
    assert actual[median.id].decision_kind is DecisionKind.SELECTED
    assert runtime.host_to_device_count == 2
    assert runtime.device_to_host_count == 2
    registry.close()


@pytest.mark.parametrize(
    ("downstream_operation", "parameter_name", "parameter_value"),
    (
        ("median_filter", "size", 3),
        ("gaussian_blur", "sigma", 1.0),
    ),
)
def test_extreme_float_background_facts_fail_closed_for_downstream_gpu(
    downstream_operation,
    parameter_name,
    parameter_value,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("rolling_ball_background")
    downstream = pipeline.add_node(downstream_operation)
    pipeline.set_param(background.id, "radius", 2.0)
    pipeline.set_param(background.id, "light_background", True)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(downstream.id, parameter_name, parameter_value)
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, downstream.id).success

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (
            ("rolling_ball_background", _device_copy),
            (downstream_operation, _device_copy),
        ),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.SELECTIVE,
        node_preferences={
            background.id: (
                f"implementation:{specs['rolling_ball_background'].implementation_id}"
            ),
            downstream.id: (
                f"implementation:{specs[downstream_operation].implementation_id}"
            ),
        },
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    planner = _StaticPlanner(compute_request, ())
    data = np.full(
        (9, 9),
        np.finfo(np.float32).max,
        dtype=np.float32,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        result = execute_pipeline_request(
            _accelerated_request(pipeline, data, compute_request),
            compute_registry=registry,
            compute_planner=planner,
        )

    assert result.error == ""
    assert result.pipeline is not None
    assert planner.array_facts[background.id][0].all_finite is True
    predicted = planner.array_facts[downstream.id][0]
    assert predicted.completeness.value == "unknown"
    assert predicted.all_finite is None
    assert runtime.operation_count == 0
    assert np.isnan(result.pipeline.outputs[downstream.id]).all()
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
        runtime_id="cuda-cupy",
        device_id="cuda:0",
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
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(median.id, specs["median_filter"]),),
    )
    cpu_gaussian = compute_specs_for("gaussian_blur")[0]
    provenance_request = _accelerated_request(
        pipeline,
        data,
        compute_request,
    )
    source_contexts = execution_module._capture_source_scientific_contexts(
        pipeline,
        provenance_request,
        cancel_callback=None,
    )
    execution_module._publish_actual_compute_provenance(
        pipeline,
        compute_request,
        (_decision(gaussian.id, cpu_gaussian),),
        source_scientific_contexts=source_contexts,
        cancel_callback=None,
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
        runtime_id="cuda-cupy",
        device_id="cuda:0",
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
        runtime_id="cuda-cupy",
        device_id="cuda:0",
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
    registry = ComputeRegistry()
    try:
        for library_id in ("cucim", "cupyx"):
            library_probe = registry.probe_library(library_id)
            if not library_probe.available:
                pytest.skip(
                    library_probe.message
                    or f"The {library_id} implementation library is unavailable."
                )
    finally:
        registry.close()

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
            background.id: ("implementation:cucim-subtract_background-v2"),
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


@pytest.mark.parametrize(
    ("downstream_operation", "parameter_name", "parameter_value"),
    (
        ("median_filter", "size", 3),
        ("gaussian_blur", "sigma", 1.0),
    ),
)
def test_real_extreme_float_background_keeps_finite_only_nodes_on_cpu(
    downstream_operation,
    parameter_name,
    parameter_value,
):
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    if importlib.util.find_spec("cucim") is None:
        pytest.skip("The optional cuCIM wheel is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy")
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        for library_id in ("cucim", "cupyx"):
            library_probe = registry.probe_library(library_id)
            if not library_probe.available:
                pytest.skip(
                    library_probe.message
                    or f"The {library_id} implementation library is unavailable."
                )

        def implementation_for(operation_id):
            candidates = tuple(
                spec
                for spec in registry.implementations_for_operation(
                    operation_id,
                    allow_experimental=True,
                )
                if spec.runtime_id == "cuda-cupy"
            )
            if not candidates:
                pytest.skip(f"No CUDA implementation is registered for {operation_id}.")
            return max(
                candidates,
                key=lambda spec: (
                    spec.implementation_version,
                    spec.implementation_id,
                ),
            )

        background_spec = implementation_for("rolling_ball_background")
        downstream_spec = implementation_for(downstream_operation)
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        background = pipeline.add_node("rolling_ball_background")
        downstream = pipeline.add_node(downstream_operation)
        pipeline.set_param(background.id, "radius", 2.0)
        pipeline.set_param(background.id, "light_background", True)
        pipeline.set_param(background.id, "spatial_mode", "2D YX")
        pipeline.set_param(downstream.id, parameter_name, parameter_value)
        assert pipeline.connect("input", background.id).success
        assert pipeline.connect(background.id, downstream.id).success
        data = np.full(
            (9, 9),
            np.finfo(np.float32).max,
            dtype=np.float32,
        )
        compute_request = ComputeRequest(
            mode=ComputeMode.SELECTIVE,
            node_preferences={
                background.id: (f"implementation:{background_spec.implementation_id}"),
                downstream.id: (f"implementation:{downstream_spec.implementation_id}"),
            },
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            allow_experimental=True,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            result = execute_pipeline_request(
                _accelerated_request(
                    pipeline,
                    data,
                    compute_request,
                    retain_node_ids=frozenset({downstream.id}),
                    prune_unretained=True,
                ),
                compute_registry=registry,
            )

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        actual = next(
            decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id == downstream.id
        )
        assert actual.decision_kind is DecisionKind.FALLBACK_CPU
        assert actual.runtime_id == "cpu-numpy"
        assert actual.fallback_reason.value == "workload_unsupported"
        assert "finite" in actual.reason_text.lower()
        assert all(
            downstream.id not in segment.node_ids
            for segment in result.execution_report.plan.segments
        )
        assert np.isnan(result.pipeline.outputs[downstream.id]).all()
    finally:
        registry.close()
