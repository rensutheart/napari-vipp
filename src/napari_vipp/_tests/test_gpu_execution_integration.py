from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp._tests.test_device_execution import (
    _device_copy,
    _device_oom_once,
    _device_richardson_lucy,
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
from napari_vipp.core.compute_benchmark_adapter import operation_parity
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import OperationComputeSpec, compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.operations import canny_edges as cpu_canny_edges
from napari_vipp.core.operations import otsu_threshold as cpu_otsu_threshold
from napari_vipp.core.pipeline import EXECUTION_READY, PrototypePipeline, SourcePayload
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.workflow import serialize_workflow


def _assert_private_cuda_scope_clean(runtime, device_id: str) -> None:
    """Require complete VIPP-owned cleanup without claiming device-wide caches."""

    terminal = runtime.memory_snapshot(device_id=device_id)
    assert terminal.runtime_live_bytes == 0
    assert terminal.runtime_reserved_bytes == 0
    # ``out_of_pool_bytes`` is a device-wide diagnostic.  CUDA module/JIT and
    # library caches can legitimately grow while a scope runs, and VIPP does
    # not own or free them.  A clean private pool plus a healthy, reusable
    # runtime is the production cleanup contract.
    assert runtime.probe().available


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
                    "NVIDIA GeForce RTX 5090",
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


def _device_measurement_payload(
    value: _ShapeAwareDeviceArray,
    **_kwargs,
) -> _ShapeAwareDeviceArray:
    assert not value.released
    value.runtime.operation_count += 1
    return value.runtime.allocate(
        np.asarray([[1.0, float(value.payload.size)]], dtype=np.float64)
    )


def _measurement_table_finalizer(
    host_outputs: tuple[object, ...],
    *,
    call,
) -> TableData:
    assert call.inputs == (None,)
    assert len(host_outputs) == 1
    payload = np.asarray(host_outputs[0], dtype=np.float64)
    return TableData(
        columns=("label_id", "pixel_count"),
        rows=tuple((int(row[0]), int(row[1])) for row in payload),
        name="Object measurements",
        table_kind="object measurements: basic morphology",
        source_name=str(call.kwargs.get("source_name", "")),
        column_units=(("pixel_count", "px^2"),),
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
                device_name="NVIDIA GeForce RTX 5090",
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
    *,
    host_finalizer_refs: Mapping[str, str] | None = None,
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
                callable_ref=f"{function.__module__}:{function.__name__}",
                implementation_library_id=("cucim" if uses_cucim else "cupyx"),
                validated_environment_policy_id=(
                    "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3"
                    if uses_cucim
                    else "cuda-cupy-14.1.1-cpython312-windows-native-v3"
                ),
                host_finalizer_ref=((host_finalizer_refs or {}).get(operation_id, "")),
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
    assert (
        result.pipeline.node_compute_provenance[
            median.id
        ].actual_implementation.implementation_id
        == specs["median_filter"].implementation_id
    )
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


def test_headless_host_finalizer_commits_public_table_and_table_state(monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    otsu = pipeline.add_node("otsu_threshold")
    components = pipeline.add_node("label_connected_components")
    measurement = pipeline.add_node("measure_objects")
    assert pipeline.connect("input", otsu.id).success
    assert pipeline.connect(otsu.id, components.id).success
    assert pipeline.connect(components.id, measurement.id).success

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (("measure_objects", _device_measurement_payload),),
        host_finalizer_refs={
            "measure_objects": f"{__name__}:_measurement_table_finalizer"
        },
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(measurement.id, specs[measurement.operation_id]),),
    )

    def forbidden_resident_projection(*_args, **_kwargs):
        raise AssertionError(
            "A host-finalized table must use authoritative host metadata."
        )

    monkeypatch.setattr(
        execution_module,
        "_predict_device_node_states",
        forbidden_resident_projection,
    )
    labels = np.zeros((5, 7), dtype=np.int32)
    labels[1:3, 2:5] = 1

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            labels,
            compute_request,
            retain_node_ids=frozenset({measurement.id}),
            prune_unretained=True,
            manual_node_ids=frozenset({measurement.id}),
        ),
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    table = result.pipeline.outputs[measurement.id]
    assert isinstance(table, TableData)
    assert table.columns == ("label_id", "pixel_count")
    assert table.rows == ((1, 35),)
    assert table.source_name == "source"
    state = result.pipeline.output_states[measurement.id]
    assert isinstance(state, TableState)
    assert state.row_count == 1
    assert state.column_count == 2
    assert state.columns == table.columns
    assert state.source_name == "source"
    assert state.column_units == (("pixel_count", "px^2"),)
    assert state.history[-1] == "Measure Objects: measured 1 object"
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert runtime.operation_count == 1
    assert runtime.live == {}
    registry.close()


def test_headless_multi_input_rl_projects_resident_metadata_and_history():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
    pipeline.set_param(deconvolution.id, "iterations", 3)
    assert pipeline.connect("input", deconvolution.id, target_port=0).success
    assert pipeline.connect("input", deconvolution.id, target_port=1).success

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (("richardson_lucy_deconvolution", _device_richardson_lucy),),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(deconvolution.id, specs[deconvolution.operation_id]),),
    )
    data = np.ones((9, 9), dtype=np.float32)

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            retain_node_ids=frozenset({deconvolution.id}),
            prune_unretained=True,
            manual_node_ids=frozenset({deconvolution.id}),
        ),
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    assert runtime.operation_count == 1
    assert runtime.live == {}
    state = result.pipeline.output_states[deconvolution.id]
    assert state.shape == data.shape
    assert state.dtype == "float32"
    assert state.history[-1] == ("Richardson-Lucy Deconvolution: 3 iterations, 2D YX")
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
            median.id: (f"implementation:{specs['median_filter'].implementation_id}"),
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
        actual[median.id].implementation_id == specs["median_filter"].implementation_id
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


def test_real_headless_measurements_pipeline_finalizes_public_table_and_cleans(
    monkeypatch,
):
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    if importlib.util.find_spec("cucim") is None:
        pytest.skip("The optional cuCIM wheel is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        library_probe = registry.probe_library("cucim", refresh=True)
        if not library_probe.available:
            pytest.skip(library_probe.message or "cuCIM is unavailable.")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        otsu = pipeline.add_node("otsu_threshold")
        components = pipeline.add_node("label_connected_components")
        measurement = pipeline.add_node("measure_objects")
        for name, value in (
            ("spatial_mode", "2D YX"),
            ("include_shape_descriptors", False),
            ("include_axis_descriptors", False),
            ("include_2d_boundary_descriptors", False),
            ("include_derived_shape_ratios", False),
            ("include_2d_shape_moments", False),
        ):
            pipeline.set_param(measurement.id, name, value)
        assert pipeline.connect("input", otsu.id).success
        assert pipeline.connect(otsu.id, components.id).success
        assert pipeline.connect(components.id, measurement.id).success

        image = np.zeros((2, 48, 64), dtype=np.uint16)
        image[0, 3:19, 5:27] = 1_000
        image[0, 9:13, 11:16] = 0
        image[0, 29:45, 41:60] = 2_000
        image[1, 4:21, 7:30] = 1_500
        image[1, 25:43, 33:58] = 2_500

        def run_request(run_id: int, compute_request: ComputeRequest, *, registry=None):
            return execute_pipeline_request(
                replace(
                    _accelerated_request(
                        pipeline,
                        image,
                        compute_request,
                        retain_node_ids=frozenset({measurement.id}),
                        prune_unretained=True,
                        manual_node_ids=frozenset({measurement.id}),
                    ),
                    run_id=run_id,
                    input_metadata={"axes": "TYX"},
                    input_name="sparse label stack",
                ),
                compute_registry=registry,
            )

        cpu_result = run_request(371, ComputeRequest(mode=ComputeMode.CPU))
        assert cpu_result.error == ""
        assert cpu_result.pipeline is not None
        expected = cpu_result.pipeline.outputs[measurement.id]
        assert isinstance(expected, TableData)

        runtime = registry.runtime("cuda-cupy")
        transfers = {"host_to_device": 0, "device_to_host": 0}
        original_to_device = runtime.to_device
        original_to_host = runtime.to_host

        def counted_to_device(value, *, device_id=""):
            transfers["host_to_device"] += 1
            return original_to_device(value, device_id=device_id)

        def counted_to_host(value):
            transfers["device_to_host"] += 1
            return original_to_host(value)

        monkeypatch.setattr(runtime, "to_device", counted_to_device)
        monkeypatch.setattr(runtime, "to_host", counted_to_host)
        gpu_result = run_request(
            372,
            ComputeRequest(
                mode=ComputeMode.SELECTIVE,
                node_preferences={
                    otsu.id: "cpu",
                    components.id: "cpu",
                    measurement.id: "implementation:cucim-measure-objects-basic-v1",
                },
                runtime_id="cuda-cupy",
                device_id=runtime_probe.selected_device_id,
                fallback_policy=FallbackPolicy.STRICT,
            ),
            registry=registry,
        )

        assert gpu_result.error == ""
        assert gpu_result.pipeline is not None
        assert gpu_result.execution_report is not None
        assert gpu_result.execution_report.cleanup_succeeded
        actual = gpu_result.pipeline.outputs[measurement.id]
        assert isinstance(actual, TableData)
        parity = operation_parity(
            "measure_objects",
            expected,
            actual,
            input_dtypes=(np.dtype(np.int32),),
        )
        assert parity.passed, parity.detail
        assert actual == expected

        decision = next(
            item
            for item in gpu_result.execution_report.actual_decisions
            if item.node_id == measurement.id
        )
        assert decision.decision_kind is DecisionKind.SELECTED
        assert decision.runtime_id == "cuda-cupy"
        assert decision.implementation_library_id == "cucim"
        assert decision.implementation_id == "cucim-measure-objects-basic-v1"
        assert len(gpu_result.execution_report.plan.segments) == 1
        assert gpu_result.execution_report.plan.segments[0].node_ids == (
            measurement.id,
        )
        assert transfers == {"host_to_device": 1, "device_to_host": 1}

        state = gpu_result.pipeline.output_states[measurement.id]
        assert isinstance(state, TableState)
        assert state.row_count == 4
        assert state.columns == actual.columns
        assert state.source_name == "sparse label stack"
        provenance = gpu_result.pipeline.node_compute_provenance[
            measurement.id
        ].actual_implementation
        assert provenance.runtime_id == "cuda-cupy"
        assert provenance.implementation_library_id == "cucim"
        assert provenance.implementation_id == "cucim-measure-objects-basic-v1"

        _assert_private_cuda_scope_clean(
            runtime,
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()


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


def test_real_headless_rgba_canny_otsu_stays_in_one_exact_device_segment(
    monkeypatch,
):
    """Exercise luma metadata contraction and resident segmentation end to end."""

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        library_probe = registry.probe_library("cupyx", refresh=True)
        if not library_probe.available:
            pytest.skip(library_probe.message or "CuPyX is unavailable.")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        canny = pipeline.add_node("canny_edges")
        otsu = pipeline.add_node("otsu_threshold")
        for name, value in (
            ("sigma", 1.1),
            ("low_quantile", 0.1),
            ("high_quantile", 0.25),
            ("channel_axis", 1),
        ):
            pipeline.set_param(canny.id, name, value)
        pipeline.set_param(otsu.id, "threshold_scope", "Stack histogram")
        pipeline.set_param(otsu.id, "histogram_bins", 256)
        assert pipeline.connect("input", canny.id).success
        assert pipeline.connect(canny.id, otsu.id).success

        time, y, x = np.indices((2, 37, 43), dtype=np.uint32)
        rgba = np.empty((2, 4, 37, 43), dtype=np.uint16)
        rgba[:, 0] = (x * 977 + y * 131 + time * 4093) % 65_536
        rgba[:, 1] = ((x - 21) ** 2 * 53 + y * 271) % 65_536
        rgba[:, 2] = ((y - 18) ** 2 * 89 + x * 193) % 65_536
        # Alpha is deliberately unrelated; the CPU and GPU luma contracts both
        # ignore it while removing the declared channel axis.
        rgba[:, 3] = (x * 17 + y * 29 + time * 47) % 65_536

        expected_canny = cpu_canny_edges(
            rgba,
            sigma=1.1,
            low_quantile=0.1,
            high_quantile=0.25,
            channel_axis=1,
        )
        expected = cpu_otsu_threshold(
            expected_canny,
            threshold_scope="Stack histogram",
            histogram_bins=256,
        )

        compute_request = ComputeRequest(
            mode=ComputeMode.SELECTIVE,
            node_preferences={
                canny.id: "implementation:cupyx-canny-edges-exact-v1",
                otsu.id: "implementation:cupy-otsu-threshold-exact-v1",
            },
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        planned_workloads: dict[str, WorkloadDescriptor] = {}

        def planner(request, workloads, **kwargs):
            planned_workloads.update(
                (workload.node_id, workload) for workload in workloads
            )
            return plan_compute_decisions(request, workloads, **kwargs)

        runtime = registry.runtime("cuda-cupy")
        transfers = {"host_to_device": 0, "device_to_host": 0}
        original_to_device = runtime.to_device
        original_to_host = runtime.to_host

        def counted_to_device(value, *, device_id=""):
            transfers["host_to_device"] += 1
            return original_to_device(value, device_id=device_id)

        def counted_to_host(value):
            transfers["device_to_host"] += 1
            return original_to_host(value)

        monkeypatch.setattr(runtime, "to_device", counted_to_device)
        monkeypatch.setattr(runtime, "to_host", counted_to_host)
        request = replace(
            _accelerated_request(
                pipeline,
                rgba,
                compute_request,
                retain_node_ids=frozenset({otsu.id}),
                prune_unretained=True,
            ),
            input_metadata={"axes": "TCYX"},
        )

        result = execute_pipeline_request(
            request,
            compute_registry=registry,
            compute_planner=planner,
        )

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        np.testing.assert_array_equal(result.pipeline.outputs[otsu.id], expected)

        decisions = {
            decision.node_id: decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id in {canny.id, otsu.id}
        }
        assert set(decisions) == {canny.id, otsu.id}
        assert all(
            decision.decision_kind is DecisionKind.SELECTED
            and decision.runtime_id == "cuda-cupy"
            for decision in decisions.values()
        )
        assert decisions[canny.id].implementation_id == (
            "cupyx-canny-edges-exact-v1"
        )
        assert decisions[otsu.id].implementation_id == (
            "cupy-otsu-threshold-exact-v1"
        )

        assert len(result.execution_report.plan.segments) == 1
        (segment,) = result.execution_report.plan.segments
        assert segment.node_ids == (canny.id, otsu.id)
        assert len(segment.entry_ports) == 1
        assert len(segment.exit_ports) == 1
        assert transfers == {"host_to_device": 1, "device_to_host": 1}

        assert planned_workloads[canny.id].input_shapes == (rgba.shape,)
        assert planned_workloads[canny.id].input_dtypes == ("uint16",)
        assert planned_workloads[otsu.id].input_shapes == (expected_canny.shape,)
        assert planned_workloads[otsu.id].input_dtypes == ("bool",)

        state = result.pipeline.output_states[otsu.id]
        assert state.shape == expected.shape
        assert state.axis_order == "TYX"
        assert tuple(axis.name for axis in state.axes) == ("t", "y", "x")
        assert state.dtype == "bool"
        assert state.kind == "binary mask"
        assert state.channels == ()

        _assert_private_cuda_scope_clean(
            runtime,
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()


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


def test_real_headless_rl_two_source_pipeline_cleans_fft_plans_and_reuses_runtime():
    """Exercise the production multi-input RL transaction on a real CUDA device."""

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        library_probe = registry.probe_library("cupyx", refresh=True)
        if not library_probe.available:
            pytest.skip(
                library_probe.message
                or "The CuPyX implementation library is unavailable."
            )

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        image_source_id = next(iter(pipeline.nodes))
        psf_source = pipeline.add_node("input")
        deconvolution = pipeline.add_node("richardson_lucy_deconvolution")
        pipeline.set_param(deconvolution.id, "spatial_mode", "2D YX")
        pipeline.set_param(deconvolution.id, "iterations", 10)
        pipeline.set_param(deconvolution.id, "filter_epsilon", 1e-8)
        assert image_source_id != psf_source.id
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

        y, x = np.mgrid[:512, :512].astype(np.float32)
        image = np.full((512, 512), 0.002, dtype=np.float32)
        for center_y, center_x, sigma, amplitude in (
            (91, 102, 2.5, 1.0),
            (225, 391, 4.1, 0.8),
            (416, 287, 3.2, 0.6),
            (345, 79, 6.0, 0.35),
            (40, 470, 1.8, 0.25),
        ):
            image += np.float32(amplitude) * np.exp(
                -((x - np.float32(center_x)) ** 2 + (y - np.float32(center_y)) ** 2)
                / np.float32(2.0 * sigma**2)
            ).astype(np.float32)
        psf_y, psf_x = np.mgrid[-6:7, -6:7].astype(np.float32)
        psf = np.exp(-(psf_x**2 + psf_y**2) / np.float32(2.0 * 1.7**2)).astype(
            np.float32
        )
        psf /= np.float32(psf.sum(dtype=np.float64))

        workflow = serialize_workflow(pipeline)
        source_payloads = {
            psf_source.id: SourcePayload(
                psf,
                {"axes": "YX"},
                "CUDA regression PSF",
            )
        }

        def run_request(
            run_id: int,
            compute_request: ComputeRequest,
            progress_updates: list[tuple[str, int, int, str]],
            *,
            compute_registry: ComputeRegistry | None = None,
        ):
            return execute_pipeline_request(
                PipelineRunRequest(
                    run_id=run_id,
                    workflow=workflow,
                    input_data=image,
                    input_metadata={"axes": "YX"},
                    input_name="CUDA regression image",
                    source_payloads=source_payloads,
                    compute_request=compute_request,
                    manual_node_ids=frozenset({deconvolution.id}),
                    retain_node_ids=frozenset({deconvolution.id}),
                    prune_unretained=True,
                ),
                progress_callback=lambda *update: progress_updates.append(update),
                compute_registry=compute_registry,
            )

        cpu_progress: list[tuple[str, int, int, str]] = []
        cpu_result = run_request(
            401,
            ComputeRequest(mode=ComputeMode.CPU),
            cpu_progress,
        )
        assert cpu_result.error == ""
        assert cpu_result.pipeline is not None
        cpu_output = np.asarray(cpu_result.pipeline.outputs[deconvolution.id]).copy()
        expected_progress = [
            (
                deconvolution.id,
                current,
                10,
                "Richardson-Lucy deconvolution",
            )
            for current in range(11)
        ]
        assert cpu_progress == expected_progress

        compute_request = ComputeRequest(
            mode=ComputeMode.SELECTIVE,
            node_preferences={deconvolution.id: "implementation:rl-cupy-f32-v1"},
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            allow_experimental=True,
        )
        runtime = registry.runtime("cuda-cupy")

        for run_id in (402, 403):
            gpu_progress: list[tuple[str, int, int, str]] = []
            result = run_request(
                run_id,
                compute_request,
                gpu_progress,
                compute_registry=registry,
            )

            assert result.error == ""
            assert result.pipeline is not None
            assert result.execution_report is not None
            assert result.execution_report.cleanup_succeeded
            assert result.execution_report.plan is not None
            assert len(result.execution_report.plan.segments) == 1
            assert result.execution_report.plan.segments[0].node_ids == (
                deconvolution.id,
            )
            assert registry.runtime("cuda-cupy") is runtime
            decision = next(
                item
                for item in result.execution_report.actual_decisions
                if item.node_id == deconvolution.id
            )
            assert decision.decision_kind is DecisionKind.SELECTED
            assert decision.runtime_id == "cuda-cupy"
            assert decision.implementation_library_id == "cupyx"
            assert decision.implementation_id == "rl-cupy-f32-v1"
            assert decision.requested_preference == NodeComputePreference(
                "implementation",
                "rl-cupy-f32-v1",
            )

            output = result.pipeline.outputs[deconvolution.id]
            assert isinstance(output, np.ndarray)
            assert output.shape == image.shape
            assert output.dtype == np.dtype(np.float32)
            assert np.isfinite(output).all()
            parity = operation_parity(
                "richardson_lucy_deconvolution",
                cpu_output,
                output,
            )
            assert parity.passed, parity.detail
            assert gpu_progress == expected_progress

            state = result.pipeline.output_states[deconvolution.id]
            assert state.axis_order == "YX"
            assert state.shape == image.shape
            assert state.dtype == "float32"
            assert state.history[-1] == (
                "Richardson-Lucy Deconvolution: 10 iterations, 2D YX"
            )
            provenance = result.pipeline.node_compute_provenance[
                deconvolution.id
            ].actual_implementation
            assert provenance.runtime_id == "cuda-cupy"
            assert provenance.implementation_library_id == "cupyx"
            assert provenance.implementation_id == "rl-cupy-f32-v1"
            assert provenance.implementation_version == "1"

            _assert_private_cuda_scope_clean(
                runtime,
                runtime_probe.selected_device_id,
            )
    finally:
        registry.close()


def test_real_headless_rl_tv_pipeline_selects_provider_reports_and_cleans():
    """Exercise the positive-TV profile through the production planner/runtime."""

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        library_probe = registry.probe_library("cupyx", refresh=True)
        if not library_probe.available:
            pytest.skip(library_probe.message or "CuPyX is unavailable.")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        image_source_id = next(iter(pipeline.nodes))
        psf_source = pipeline.add_node("input")
        deconvolution = pipeline.add_node("richardson_lucy_tv_deconvolution")
        for name, value in (
            ("spatial_mode", "2D YX"),
            ("iterations", 10),
            ("tv_regularization", 0.002),
            ("tv_epsilon", 1e-6),
            ("filter_epsilon", 1e-12),
            ("denominator_floor", 0.05),
        ):
            pipeline.set_param(deconvolution.id, name, value)
        assert pipeline.connect(
            image_source_id,
            deconvolution.id,
            target_port=0,
        ).success
        assert pipeline.connect(psf_source.id, deconvolution.id, target_port=1).success

        rng = np.random.default_rng(20260729)
        image = rng.random((128, 129), dtype=np.float32) * np.float32(0.05)
        image[24, 31] = np.float32(1.0)
        image[83, 97] = np.float32(0.6)
        y, x = np.mgrid[-6:7, -6:7].astype(np.float32)
        psf = np.exp(-(x * x + y * y) / np.float32(2.0 * 1.7**2)).astype(
            np.float32
        )
        psf /= np.float32(psf.sum(dtype=np.float64))
        workflow = serialize_workflow(pipeline)
        source_payloads = {
            psf_source.id: SourcePayload(psf, {"axes": "YX"}, "RL-TV test PSF")
        }

        def run(run_id, compute_request, progress_updates, compute_registry=None):
            return execute_pipeline_request(
                PipelineRunRequest(
                    run_id=run_id,
                    workflow=workflow,
                    input_data=image,
                    input_metadata={"axes": "YX"},
                    input_name="RL-TV test image",
                    source_payloads=source_payloads,
                    compute_request=compute_request,
                    manual_node_ids=frozenset({deconvolution.id}),
                    retain_node_ids=frozenset({deconvolution.id}),
                    prune_unretained=True,
                ),
                progress_callback=lambda *update: progress_updates.append(update),
                compute_registry=compute_registry,
            )

        cpu_progress = []
        cpu_result = run(451, ComputeRequest(mode=ComputeMode.CPU), cpu_progress)
        assert cpu_result.error == ""
        cpu_output = np.asarray(cpu_result.pipeline.outputs[deconvolution.id]).copy()
        expected_progress = [
            (
                deconvolution.id,
                current,
                10,
                "Richardson-Lucy TV deconvolution",
            )
            for current in range(11)
        ]
        assert cpu_progress == expected_progress

        gpu_progress = []
        gpu_result = run(
            452,
            ComputeRequest(
                mode=ComputeMode.SELECTIVE,
                node_preferences={
                    deconvolution.id: "implementation:rl-tv-cupy-f32-v1"
                },
                runtime_id="cuda-cupy",
                device_id=runtime_probe.selected_device_id,
                allow_experimental=True,
            ),
            gpu_progress,
            registry,
        )

        assert gpu_result.error == ""
        assert gpu_result.execution_report.cleanup_succeeded
        assert gpu_result.execution_report.plan.segments[0].node_ids == (
            deconvolution.id,
        )
        decision = next(
            item
            for item in gpu_result.execution_report.actual_decisions
            if item.node_id == deconvolution.id
        )
        assert decision.decision_kind is DecisionKind.SELECTED
        assert decision.implementation_id == "rl-tv-cupy-f32-v1"
        assert decision.implementation_library_id == "cupyx"
        output = gpu_result.pipeline.outputs[deconvolution.id]
        parity = operation_parity(
            "richardson_lucy_tv_deconvolution",
            cpu_output,
            output,
            parameters={"tv_regularization": 0.002},
        )
        assert parity.passed, parity.detail
        assert gpu_progress == expected_progress
        provenance = gpu_result.pipeline.node_compute_provenance[
            deconvolution.id
        ].actual_implementation
        assert provenance.implementation_id == "rl-tv-cupy-f32-v1"
        assert provenance.runtime_id == "cuda-cupy"

        _assert_private_cuda_scope_clean(
            registry.runtime("cuda-cupy"),
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()
