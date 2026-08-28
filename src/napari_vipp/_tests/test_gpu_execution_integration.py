from __future__ import annotations

import importlib.util
import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import napari_vipp.core.compute_planning as planning_module
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
from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    run_batch,
    scientific_workflow_hash,
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
from napari_vipp.core.compute_history import JsonPipelineTimingStore
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import OperationComputeSpec, compute_specs_for
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.execution_telemetry import (
    DeviceExecutionPhase,
    DeviceExecutionTelemetryConfig,
    PipelinePreparationPhase,
)
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.host_memory import HostMemorySnapshot
from napari_vipp.core.operations import canny_edges as cpu_canny_edges
from napari_vipp.core.operations import gaussian_blur as cpu_gaussian_blur
from napari_vipp.core.operations import median_filter as cpu_median_filter
from napari_vipp.core.operations import otsu_threshold as cpu_otsu_threshold
from napari_vipp.core.pipeline import EXECUTION_READY, PrototypePipeline, SourcePayload
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.workflow import WORKFLOW_VERSION, serialize_workflow


@pytest.fixture
def validated_windows_compute_host(monkeypatch):
    """Keep fake-CUDA tests independent of the host running pytest."""

    base = ComputeEnvironment(
        os_name="Windows",
        execution_mode="native",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
    )
    monkeypatch.setattr(planning_module, "ComputeEnvironment", lambda: base)


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
                implementation_libraries=("cpu", "cupy", "cupyx"),
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
        uses_cupy = operation_id in {
            "rolling_ball_background",
            "subtract_background",
            "measure_objects",
            "measure_objects_intensity",
        }
        shaped[operation_id] = _shape_preserving_spec(
            replace(
                _implementation_spec(operation_id, function),
                runtime_id="cuda-cupy",
                array_domain="cuda-cupy",
                callable_ref=f"{function.__module__}:{function.__name__}",
                implementation_library_id=("cupy" if uses_cupy else "cupyx"),
                validated_environment_policy_id=(
                    "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
                    if uses_cupy
                    else "cuda-cupy-14.1.1-cpython312-windows-native-v3"
                ),
                host_finalizer_ref=((host_finalizer_refs or {}).get(operation_id, "")),
            )
        )
    library_ids = {spec.implementation_library_id for spec in shaped.values()}
    library_probes = {}
    for library_id in library_ids:
        library_probes[library_id] = (
            lambda library_id=library_id: ImplementationLibraryProbeResult(
                library_id,
                True,
                version="14.1.1",
            )
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
    performance_history_path: Path | None = None,
    device_execution_telemetry: DeviceExecutionTelemetryConfig | None = None,
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
        performance_history_path=performance_history_path,
        device_execution_telemetry=device_execution_telemetry,
    )


def test_injected_planner_cannot_return_a_different_device_request():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    compute_request = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(runtime, (("gaussian_blur", _device_copy),))

    def mismatched_planner(request, _workloads, **_kwargs):
        return _PlanningResult(
            replace(request, device_id="cuda:1"),
            ComputeEnvironment(
                runtime_ids=("cpu-numpy", "cuda-cupy"),
                implementation_libraries=("cpu", "cupy"),
                device_id="cuda:1",
                device_name="Different GPU",
                device_class="nvidia-cuda",
                memory_topology="discrete",
            ),
            (_decision(gaussian.id, specs["gaussian_blur"]),),
        )

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            np.arange(25, dtype=np.float32).reshape(5, 5),
            compute_request,
        ),
        compute_registry=registry,
        compute_planner=mismatched_planner,
    )

    assert "different runtime, device" in result.error
    assert runtime.operation_count == 0
    assert runtime.live == {}
    registry.close()


def test_prefer_gpu_then_auto_cpu_comparison_teaches_next_auto_assignment(
    tmp_path,
    validated_windows_compute_host,
):
    pipeline = PrototypePipeline()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(4_096, dtype=np.float32).reshape(64, 64)
    history_path = tmp_path / "pipeline-timings.json"

    runtime = _ShapeAwareRuntime()
    registry, _specs = _test_registry(runtime, (("gaussian_blur", _device_copy),))
    prefer_result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.PREFER_GPU),
            performance_history_path=history_path,
        ),
        compute_registry=registry,
    )
    assert not prefer_result.error

    store = JsonPipelineTimingStore(history_path)
    samples = store.samples()
    assert len(samples) == 1
    gpu_sample = next(item for item in samples if item.assignment.uses_accelerator)
    store.clear()
    store.append(replace(gpu_sample, elapsed_seconds=0.1))

    comparison_result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.AUTO),
            performance_history_path=history_path,
        ),
        compute_registry=registry,
    )

    assert not comparison_result.error
    assert comparison_result.execution_report is not None
    comparison_decision = next(
        item
        for item in comparison_result.execution_report.actual_decisions
        if item.node_id == gaussian.id
    )
    assert comparison_decision.runtime_id == "cpu-numpy"
    assert comparison_decision.reason is DecisionReason.PERFORMANCE_EXPLORATION
    assert any(
        "completed its one-time CPU comparison" in warning
        for warning in comparison_result.execution_report.warnings
    )
    samples = store.samples()
    assert len(samples) == 2
    cpu_sample = next(item for item in samples if not item.assignment.uses_accelerator)
    store.clear()
    store.append(replace(cpu_sample, elapsed_seconds=2.0))
    store.append(replace(gpu_sample, elapsed_seconds=0.1))

    auto_result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.AUTO),
            performance_history_path=history_path,
        ),
        compute_registry=registry,
    )

    assert not auto_result.error
    assert auto_result.execution_report is not None
    decision = next(
        item
        for item in auto_result.execution_report.actual_decisions
        if item.node_id == gaussian.id
    )
    assert decision.runtime_id == "cuda-cupy"
    assert decision.reason is DecisionReason.HISTORICAL_PERFORMANCE
    assert decision.performance_evidence_kind == "completed_pipeline_timing"
    assert decision.performance_evidence_digest
    assert "2.000 s for CPU" in decision.reason_text


def test_completed_run_timing_excludes_bypass_from_gpu_assignment(
    tmp_path,
    validated_windows_compute_host,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", crop.id).success
    assert pipeline.connect(crop.id, gaussian.id).success
    assert pipeline.set_node_execution_mode(crop.id, "bypass")
    data = np.arange(4_096, dtype=np.float32).reshape(64, 64)
    history_path = tmp_path / "pipeline-timings.json"
    runtime = _ShapeAwareRuntime()
    registry, _specs = _test_registry(runtime, (("gaussian_blur", _device_copy),))

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.PREFER_GPU),
            performance_history_path=history_path,
        ),
        compute_registry=registry,
    )

    assert not result.error
    assert result.execution_report is not None
    decisions = {
        decision.node_id: decision
        for decision in result.execution_report.actual_decisions
    }
    assert decisions[crop.id].decision_kind is DecisionKind.BYPASSED
    assert decisions[gaussian.id].runtime_id == "cuda-cupy"
    assert result.pipeline is not None
    assert result.pipeline.outputs[crop.id] is data
    assert (
        result.pipeline.output_states[crop.id]
        is result.pipeline.output_states["input"]
    )
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 1
    sample = JsonPipelineTimingStore(history_path).samples()[0]
    assert [decision.node_id for decision in sample.assignment.decisions] == [
        gaussian.id
    ]
    assert sample.assignment.uses_accelerator
    registry.close()


def test_retained_resident_bypass_reuses_finalized_upstream_state_identity(
    validated_windows_compute_host,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    before = pipeline.add_node("gaussian_blur")
    crop = pipeline.add_node("crop_stack")
    after = pipeline.add_node("gaussian_blur")
    pipeline.set_param(before.id, "sigma", 0.0)
    pipeline.set_param(after.id, "sigma", 0.0)
    assert pipeline.connect("input", before.id).success
    assert pipeline.connect(before.id, crop.id).success
    assert pipeline.connect(crop.id, after.id).success
    assert pipeline.set_node_execution_mode(crop.id, "bypass")
    data = np.arange(4_096, dtype=np.float32).reshape(64, 64)
    runtime = _ShapeAwareRuntime()
    registry, _specs = _test_registry(
        runtime,
        (("gaussian_blur", _device_copy),),
    )

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.PREFER_GPU),
        ),
        compute_registry=registry,
    )

    assert not result.error
    assert result.pipeline is not None
    assert result.pipeline.outputs[crop.id] is result.pipeline.outputs[before.id]
    assert (
        result.pipeline.output_states[crop.id]
        is result.pipeline.output_states[before.id]
    )
    assert runtime.host_to_device_count == 1
    assert runtime.device_to_host_count == 2
    assert runtime.operation_count == 2
    registry.close()


def test_auto_skips_optional_cpu_comparison_when_host_headroom_is_unsafe(
    tmp_path,
    monkeypatch,
    validated_windows_compute_host,
):
    pipeline = PrototypePipeline()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(4_096, dtype=np.float32).reshape(64, 64)
    history_path = tmp_path / "pipeline-timings.json"
    runtime = _ShapeAwareRuntime()
    registry, _specs = _test_registry(runtime, (("gaussian_blur", _device_copy),))
    prefer_result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.PREFER_GPU),
            performance_history_path=history_path,
        ),
        compute_registry=registry,
    )
    assert not prefer_result.error
    store = JsonPipelineTimingStore(history_path)
    gpu_sample = store.samples()[0]
    store.clear()
    store.append(replace(gpu_sample, elapsed_seconds=0.1))
    monkeypatch.setattr(
        execution_module,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="win32",
            source="windows_global_memory_status_ex",
            physical_total_bytes=1_000,
            physical_available_bytes=100,
            commit_limit_bytes=1_000,
            commit_available_bytes=100,
        ),
    )

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.AUTO),
            performance_history_path=history_path,
        ),
        compute_registry=registry,
    )

    assert not result.error
    assert result.execution_report is not None
    decision = next(
        item
        for item in result.execution_report.actual_decisions
        if item.node_id == gaussian.id
    )
    assert decision.runtime_id == "cuda-cupy"
    assert any(
        "Skipped Auto CPU timing comparison" in warning and "memory headroom" in warning
        for warning in result.execution_report.warnings
    )
    assert len(store.samples()) == 2


def test_optional_history_fingerprint_failure_does_not_fail_scientific_run(
    tmp_path,
    monkeypatch,
):
    pipeline = PrototypePipeline()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success

    def fail_fingerprint():
        raise OSError("metadata unavailable")

    monkeypatch.setattr(
        execution_module,
        "host_performance_fingerprint",
        fail_fingerprint,
    )

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            np.arange(256, dtype=np.float32).reshape(16, 16),
            ComputeRequest(mode=ComputeMode.CPU),
            performance_history_path=tmp_path / "timings.json",
        )
    )

    assert not result.error
    assert result.execution_report is not None
    assert any(
        "continue without it" in warning for warning in result.execution_report.warnings
    )
    assert not (tmp_path / "timings.json").exists()


def test_run_batch_reuses_fake_cuda_after_visible_oom_fallback(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    arrays = (
        np.arange(30, dtype=np.float32).reshape(5, 6),
        np.arange(30, dtype=np.float32).reshape(5, 6) + 10,
    )
    for index, array in enumerate(arrays, start=1):
        np.save(inputs / f"{index:02d}.npy", array)

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, output.id).success

    request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    workflow = serialize_workflow(pipeline, compute_request=request)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        compute_request=request,
    )
    runtime = _ShapeAwareRuntime()
    runtime.oom_remaining = 1
    registry, specs = _test_registry(
        runtime,
        (("gaussian_blur", _device_oom_once),),
    )
    planner = _StaticPlanner(
        request,
        (_decision(gaussian.id, specs["gaussian_blur"]),),
    )

    try:
        result = run_batch(
            workflow,
            config,
            compute_registry=registry,
            compute_planner=planner,
        )
        document = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        assert document["summary"] == {
            "completed": 2,
            "partial": 0,
            "skipped": 0,
            "cancelled": 0,
            "failed": 0,
        }
        assert document["compute"]["configured_request"] == request.as_dict()
        assert document["compute"]["effective_request"] == request.as_dict()
        assert document["compute"]["runtime_cleanup_succeeded"] is True
        assert all(
            item["execution"]["cleanup_succeeded"] is True for item in document["items"]
        )

        first, second = document["items"]
        first_gaussian = next(
            node
            for node in first["execution"]["nodes"]
            if node["node_id"] == gaussian.id
        )
        assert first_gaussian["decision_kind"] == "fallback_cpu"
        assert first_gaussian["actual_implementation"]["runtime_id"] == "cpu-numpy"
        assert first_gaussian["actual_implementation"]["implementation_id"] == (
            "cpu-gaussian_blur-v1"
        )
        assert len(first["execution"]["fallback_records"]) == 1
        fallback = first["execution"]["fallback_records"][0]
        assert fallback["reason_code"] == "fake_oom"
        assert fallback["device_attempt_count"] == 1
        assert fallback["cpu_retry_count"] == 1
        assert fallback["cpu_retry_succeeded"] is True
        assert fallback["cleanup_succeeded"] is True

        second_gaussian = next(
            node
            for node in second["execution"]["nodes"]
            if node["node_id"] == gaussian.id
        )
        assert second_gaussian["decision_kind"] == "selected"
        assert second_gaussian["actual_implementation"]["runtime_id"] == "cuda-cupy"
        assert (
            second_gaussian["actual_implementation"]["implementation_library_id"]
            == "cupyx"
        )
        assert second_gaussian["actual_implementation"]["implementation_id"] == (
            "fake-gaussian_blur-v1"
        )
        assert second["execution"]["fallback_records"] == []

        for item, expected in zip(document["items"], arrays, strict=True):
            output_record = item["outputs"][0]
            assert (
                output_record["execution_provenance_sha256"]
                == item["execution_provenance_sha256"]
            )
            np.testing.assert_array_equal(np.load(output_record["path"]), expected)

        assert runtime.host_to_device_count == 2
        assert runtime.device_to_host_count == 1
        assert runtime.operation_count == 2
        assert runtime.release_count == 3
        assert runtime.live == {}
        assert (
            sum(
                isinstance(event, tuple) and event[0] == "scope-enter"
                for event in runtime.events
            )
            == 2
        )
        assert (
            sum(
                isinstance(event, tuple) and event[0] == "scope-exit"
                for event in runtime.events
            )
            == 2
        )
    finally:
        registry.close()


@pytest.mark.real_cuda
def test_real_run_batch_gpu_provenance_cleanup_and_reuse(tmp_path):
    if os.environ.get("VIPP_RUN_REAL_CUDA_BATCH") != "1":
        pytest.skip("Set VIPP_RUN_REAL_CUDA_BATCH=1 to run the real CUDA batch smoke.")
    if importlib.util.find_spec("cupy") is None:
        pytest.fail("Real CUDA batch smoke requested, but CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not probe.available or not probe.selected_device_id:
            pytest.fail(
                "Real CUDA batch smoke requested, but CUDA is unavailable: "
                + (probe.message or "no selected device")
            )
        expected_device = os.environ.get("VIPP_EXPECT_CUDA_DEVICE", "").strip()
        selected = next(
            (
                device
                for device in probe.devices
                if device.device_id == probe.selected_device_id
            ),
            None,
        )
        if expected_device:
            assert selected is not None
            assert expected_device.casefold() in selected.display_name.casefold()
        library = registry.probe_library("cupy", refresh=True)
        if not library.available:
            pytest.fail(
                "Real CUDA batch smoke requested, but CuPy is unavailable: "
                + library.message
            )

        inputs = tmp_path / "inputs"
        inputs.mkdir()
        rng = np.random.default_rng(20260804)
        arrays = tuple(
            rng.random((256, 320), dtype=np.float32) * np.float32(100.0)
            for _ in range(2)
        )
        for index, array in enumerate(arrays, start=1):
            np.save(inputs / f"{index:02d}.npy", array)

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        gaussian = pipeline.add_node("gaussian_blur")
        median = pipeline.add_node("median_filter")
        output = pipeline.add_node("batch_output")
        pipeline.set_param(gaussian.id, "sigma", 1.25)
        pipeline.set_param(median.id, "size", 3)
        pipeline.set_param(output.id, "tag", "result")
        pipeline.set_param(output.id, "format", "npy")
        assert pipeline.connect("input", median.id).success
        assert pipeline.connect(median.id, gaussian.id).success
        assert pipeline.connect(gaussian.id, output.id).success
        request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences={
                gaussian.id: "implementation:cupy-gaussian-blur-v1",
                median.id: "implementation:cupy-median-filter-v1",
                output.id: "cpu",
            },
            runtime_id="cuda-cupy",
            device_id=probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        workflow = serialize_workflow(pipeline, compute_request=request)
        config = BatchConfig(
            workflow_file=Path("workflow.json"),
            workflow_sha256=scientific_workflow_hash(workflow),
            output_dir=tmp_path / "outputs",
            sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
            outputs=(
                BatchOutputConfig(
                    output.id,
                    "Batch Output",
                    "result",
                    "image",
                    "npy",
                    "",
                    "{source_stem}__{tag}",
                ),
            ),
            default_image_format="npy",
            save_python_script=False,
            compute_request=request,
        )
        nested_progress = []
        runtime = registry.runtime("cuda-cupy")

        result = run_batch(
            workflow,
            config,
            compute_registry=registry,
            execution_progress_callback=nested_progress.append,
        )
        document = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        assert document["summary"]["completed"] == 2
        assert document["summary"]["failed"] == 0
        assert document["compute"]["runtime_cleanup_succeeded"] is True
        assert {progress.item_index for progress in nested_progress} == {1, 2}
        assert {
            progress.node_id
            for progress in nested_progress
            if progress.message == "Node started."
        }.issuperset({gaussian.id, median.id})
        for item, source in zip(document["items"], arrays, strict=True):
            by_node = {node["node_id"]: node for node in item["execution"]["nodes"]}
            for node_id, implementation_id in (
                (gaussian.id, "cupy-gaussian-blur-v1"),
                (median.id, "cupy-median-filter-v1"),
            ):
                node = by_node[node_id]
                identity = node["actual_implementation"]
                assert node["decision_kind"] == "selected"
                assert identity["runtime_id"] == "cuda-cupy"
                assert identity["implementation_library_id"] == "cupy"
                assert identity["implementation_id"] == implementation_id
                assert identity["implementation_version"]
            segments = item["execution"]["plan"]["segments"]
            assert len(segments) == 1
            assert segments[0]["node_ids"] == [median.id, gaussian.id]
            assert item["execution"]["fallback_records"] == []
            assert item["execution"]["cleanup_succeeded"] is True
            output_record = item["outputs"][0]
            assert (
                output_record["execution_provenance_sha256"]
                == item["execution_provenance_sha256"]
            )
            actual = np.load(output_record["path"])
            parity = operation_parity(
                "gaussian_blur",
                cpu_gaussian_blur(
                    cpu_median_filter(source, size=3),
                    sigma=1.25,
                ),
                actual,
                input_dtypes=(source.dtype,),
            )
            assert parity.passed, parity.detail

        cancel_event = threading.Event()
        cancellation_progress = []

        def cancel_after_first_device_node(progress):
            cancellation_progress.append(progress)
            if (
                progress.item_index == 1
                and progress.node_id == gaussian.id
                and progress.message == "Node started."
            ):
                cancel_event.set()

        cancelled_config = replace(
            config,
            output_dir=tmp_path / "cancelled-outputs",
        )
        cancelled_result = run_batch(
            workflow,
            cancelled_config,
            compute_registry=registry,
            cancel_event=cancel_event,
            execution_progress_callback=cancel_after_first_device_node,
        )
        cancelled_document = json.loads(
            cancelled_result.manifest_path.read_text(encoding="utf-8")
        )
        assert cancelled_result.cancelled
        assert not cancelled_result.has_failures
        assert cancelled_result.saved_paths == ()
        assert cancelled_document["summary"]["cancelled"] == 1
        assert cancelled_document["summary"]["skipped"] == 1
        cancelled_item = cancelled_document["items"][0]
        assert cancelled_item["execution"]["outcome"] == "cancelled"
        assert cancelled_item["execution"]["cleanup_succeeded"] is True
        assert cancelled_item["execution"]["failure"]["kind"] == "cancelled"
        assert not list(cancelled_config.output_dir.glob("*.npy"))
        assert any(
            progress.node_id == gaussian.id and progress.message == "Node started."
            for progress in cancellation_progress
        )

        assert registry.runtime("cuda-cupy") is runtime
        _assert_private_cuda_scope_clean(runtime, probe.selected_device_id)
    finally:
        registry.close()


@pytest.mark.real_cuda
def test_real_generated_python_gpu_provenance_cancellation_and_reuse(
    tmp_path,
    monkeypatch,
):
    """Prove that exported Python preserves the shared CUDA execution contract."""

    if os.environ.get("VIPP_RUN_REAL_CUDA_BATCH") != "1":
        pytest.skip("Set VIPP_RUN_REAL_CUDA_BATCH=1 to run the real CUDA smoke.")
    if importlib.util.find_spec("cupy") is None:
        pytest.fail("Real CUDA smoke requested, but CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not probe.available or not probe.selected_device_id:
            pytest.fail(
                "Real CUDA smoke requested, but CUDA is unavailable: "
                + (probe.message or "no selected device")
            )
        expected_device = os.environ.get("VIPP_EXPECT_CUDA_DEVICE", "").strip()
        selected = next(
            (
                device
                for device in probe.devices
                if device.device_id == probe.selected_device_id
            ),
            None,
        )
        if expected_device:
            assert selected is not None
            assert expected_device.casefold() in selected.display_name.casefold()
        library = registry.probe_library("cupy", refresh=True)
        if not library.available:
            pytest.fail(
                "Real CUDA smoke requested, but CuPy is unavailable: " + library.message
            )
    finally:
        registry.close()

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    median = pipeline.add_node("median_filter")
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(median.id, "size", 3)
    pipeline.set_param(gaussian.id, "sigma", 1.25)
    assert pipeline.connect("input", median.id).success
    assert pipeline.connect(median.id, gaussian.id).success

    gpu_request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            median.id: "implementation:cupy-median-filter-v1",
            gaussian.id: "implementation:cupy-gaussian-blur-v1",
        },
        runtime_id="cuda-cupy",
        device_id=probe.selected_device_id,
        fallback_policy=FallbackPolicy.STRICT,
    )
    rng = np.random.default_rng(20260804)
    source = rng.random((256, 320), dtype=np.float32) * np.float32(100.0)

    cpu_request = ComputeRequest(
        mode=ComputeMode.CPU,
        fallback_policy=FallbackPolicy.STRICT,
    )
    cpu_result = execute_pipeline_request(
        _accelerated_request(pipeline, source, cpu_request)
    )
    assert cpu_result.error == ""
    assert cpu_result.pipeline is not None
    expected = cpu_result.pipeline.outputs[gaussian.id]

    script_path = tmp_path / "generated_cuda_pipeline.py"
    script_path.write_text(
        export_pipeline_to_python(pipeline, compute_request=gpu_request),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "vipp_generated_cuda_integration",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    embedded_request = json.loads(generated._WORKFLOW_JSON)["execution"]["compute"]
    assert embedded_request["mode"] == "custom"
    assert embedded_request["fallback_policy"] == "strict"
    assert embedded_request["node_preferences"] == {
        median.id: "implementation:cupy-median-filter-v1",
        gaussian.id: "implementation:cupy-gaussian-blur-v1",
    }

    # Generated execution owns its registry, so observe the actual provider
    # boundary without replacing the registry or CUDA runtime itself.
    from napari_vipp.core.gpu.cupy_runtime import CuPyRuntime

    transfers = {"host_to_device": 0, "device_to_host": 0}
    real_to_device = CuPyRuntime.to_device
    real_to_host = CuPyRuntime.to_host

    def counted_to_device(runtime, value, *, device_id=""):
        transfers["host_to_device"] += 1
        return real_to_device(runtime, value, device_id=device_id)

    def counted_to_host(runtime, value):
        transfers["device_to_host"] += 1
        return real_to_host(runtime, value)

    monkeypatch.setattr(CuPyRuntime, "to_device", counted_to_device)
    monkeypatch.setattr(CuPyRuntime, "to_host", counted_to_host)

    updates = []
    gpu_results = generated.run_pipeline(
        source,
        input_metadata={"axes": "YX"},
        compute_request=gpu_request,
        progress_callback=lambda *update: updates.append(update),
    )
    assert gpu_results.effective_compute_request == gpu_request
    report = gpu_results.execution_report
    assert report is not None
    assert report.cleanup_succeeded
    assert report.fallback_records == ()
    assert report.plan is not None
    assert len(report.plan.segments) == 1
    segment = report.plan.segments[0]
    assert segment.node_ids == (median.id, gaussian.id)
    assert tuple((port.node_id, port.port_index) for port in segment.entry_ports) == (
        ("input", 0),
    )
    assert transfers == {"host_to_device": 1, "device_to_host": 2}

    decisions = {decision.node_id: decision for decision in report.actual_decisions}
    for node_id, implementation_id in (
        (median.id, "cupy-median-filter-v1"),
        (gaussian.id, "cupy-gaussian-blur-v1"),
    ):
        decision = decisions[node_id]
        assert decision.decision_kind is DecisionKind.SELECTED
        assert decision.runtime_id == "cuda-cupy"
        assert decision.implementation_library_id == "cupy"
        assert decision.implementation_id == implementation_id
        exact = next(
            item
            for item in gpu_results.execution_provenance["nodes"]
            if item["node_id"] == node_id
        )
        assert exact["actual_implementation"]["runtime_id"] == "cuda-cupy"
        assert exact["actual_implementation"]["implementation_id"] == (
            implementation_id
        )
        assert exact["actual_implementation"]["implementation_version"]

    assert gpu_results.execution_provenance["fallback_records"] == []
    assert gpu_results.execution_provenance["cleanup_succeeded"] is True
    assert any(
        update[0] == median.id and update[3].startswith("Node started")
        for update in updates
    )
    parity = operation_parity(
        "gaussian_blur",
        expected,
        gpu_results[gaussian.id],
        input_dtypes=(source.dtype,),
    )
    assert parity.passed, parity.detail

    output_path = generated.save_image(
        gpu_results[gaussian.id],
        tmp_path / "generated-result.npy",
        image_state=gpu_results.image_states[gaussian.id],
        provenance=gpu_results,
        output_node_id=gaussian.id,
    )
    sidecar = json.loads(
        Path(f"{output_path}.vipp-provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["execution"]["cleanup_succeeded"] is True
    assert sidecar["execution"]["fallback_records"] == []
    assert sidecar["output"]["node_id"] == gaussian.id
    assert (
        sidecar["output"]["execution_provenance_sha256"]
        == (
            gpu_results.output_provenance[gaussian.id]["output"][
                "execution_provenance_sha256"
            ]
        )
    )

    cancel_event = threading.Event()
    cancelled_updates = []

    def cancel_on_second_node(node_id, current, total, message):
        cancelled_updates.append((node_id, current, total, message))
        if node_id == gaussian.id and message.startswith("Node started"):
            cancel_event.set()

    with pytest.raises(generated.OperationCancelled) as caught:
        generated.run_pipeline(
            source,
            input_metadata={"axes": "YX"},
            compute_request=gpu_request,
            progress_callback=cancel_on_second_node,
            cancel_event=cancel_event,
        )
    cancelled_execution = caught.value.provenance["execution"]
    assert cancelled_execution["outcome"] == "cancelled"
    assert cancelled_execution["failure"]["kind"] == "cancelled"
    assert cancelled_execution["failure"]["reason_code"] == ("operation_cancelled")
    assert cancelled_execution["cleanup_succeeded"] is True
    assert any(update[0] == gaussian.id for update in cancelled_updates)

    # A successful third call proves that generated cancellation did not leave
    # CUDA state unusable.  The exported API owns and closes each registry.
    reusable = generated.run_pipeline(
        source,
        input_metadata={"axes": "YX"},
        compute_request=gpu_request,
    )
    assert reusable.execution_report.cleanup_succeeded
    assert reusable.execution_provenance["fallback_records"] == []
    retry_parity = operation_parity(
        "gaussian_blur",
        expected,
        reusable[gaussian.id],
        input_dtypes=(source.dtype,),
    )
    assert retry_parity.passed, retry_parity.detail


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
    assert result.device_execution_telemetry is None
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


def test_headless_device_chain_propagates_opt_in_execution_telemetry():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success

    runtime = _ShapeAwareRuntime()
    registry, specs = _test_registry(
        runtime,
        (("gaussian_blur", _device_copy),),
    )
    compute_request = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        device_id="cuda:0",
    )
    planner = _StaticPlanner(
        compute_request,
        (_decision(gaussian.id, specs[gaussian.operation_id]),),
    )
    data = np.arange(63, dtype=np.float32).reshape(7, 9)

    result = execute_pipeline_request(
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            device_execution_telemetry=DeviceExecutionTelemetryConfig(
                synchronize_device_phases=True,
            ),
        ),
        compute_registry=registry,
        compute_planner=planner,
    )

    assert result.error == ""
    assert result.pipeline is not None
    preparation = result.pre_device_execution_telemetry
    assert preparation is not None
    assert preparation.completed is True
    assert {span.phase for span in preparation.spans} == set(PipelinePreparationPhase)
    observation = result.device_execution_telemetry
    assert observation is not None
    assert (
        preparation.started_monotonic_seconds + preparation.elapsed_seconds
        <= observation.started_monotonic_seconds
    )
    assert observation.synchronized_device_phases is True
    assert observation.host_to_device.count == 1
    assert observation.host_to_device.byte_count == data.nbytes
    assert observation.host_to_device.all_synchronized is True
    assert observation.device_to_host.count == 1
    assert observation.device_to_host.byte_count == data.nbytes
    assert observation.device_to_host.all_synchronized is True
    operation_spans = observation.spans_for(DeviceExecutionPhase.DEVICE_OPERATION)
    assert [(span.node_id, span.operation_id) for span in operation_spans] == [
        (gaussian.id, gaussian.operation_id)
    ]
    resolution_spans = observation.spans_for(
        DeviceExecutionPhase.IMPLEMENTATION_RESOLUTION
    )
    assert [span.implementation_id for span in resolution_spans] == [
        specs[gaussian.operation_id].implementation_id
    ]
    assert runtime.live == {}
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


def test_cpu_extract_channel_projects_shape_through_requested_mixed_chain(
    validated_windows_compute_host,
):
    del validated_windows_compute_host
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
        mode=ComputeMode.CUSTOM,
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


def test_prefer_gpu_non_native_extract_is_selected_for_cpu_before_execution(
    validated_windows_compute_host,
):
    del validated_windows_compute_host
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    extract = pipeline.add_node("extract_channel")
    pipeline.set_param(extract.id, "channel", 1)
    assert pipeline.connect("input", extract.id).success

    data = (
        np.arange(3 * 5 * 7, dtype=np.uint16)
        .reshape(3, 5, 7)
        .astype(np.dtype(np.uint16).newbyteorder("S"))
    )
    request = replace(
        _accelerated_request(
            pipeline,
            data,
            ComputeRequest(mode=ComputeMode.PREFER_GPU),
        ),
        input_metadata={"axes": "CYX"},
    )

    result = execute_pipeline_request(request)

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    output = np.asarray(result.pipeline.outputs[extract.id])
    np.testing.assert_array_equal(output, data[1], strict=True)
    assert not output.dtype.isnative
    decision = next(
        item
        for item in result.execution_report.actual_decisions
        if item.node_id == extract.id
    )
    assert decision.implementation_id == "cpu-extract_channel-v1"
    assert decision.reason is DecisionReason.WORKLOAD_UNSUPPORTED
    assert not decision.fallback_used
    assert "native-endian" in decision.reason_text


@pytest.mark.parametrize(
    ("operation_id", "source_dtype", "parameters"),
    (
        (
            "binary_threshold",
            np.dtype(np.float32).newbyteorder("S"),
            {"threshold": 0.5},
        ),
        (
            "convert_dtype",
            np.dtype(np.uint16).newbyteorder("S"),
            {"output_dtype": "float32", "scaling": "preserve"},
        ),
    ),
)
def test_prefer_gpu_non_native_inputs_select_cpu_before_device_upload(
    validated_windows_compute_host,
    operation_id,
    source_dtype,
    parameters,
):
    del validated_windows_compute_host
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    for name, value in parameters.items():
        pipeline.set_param(node.id, name, value)
    assert pipeline.connect("input", node.id).success
    data = np.arange(5 * 7).reshape(5, 7).astype(source_dtype)
    request = _accelerated_request(
        pipeline,
        data,
        ComputeRequest(mode=ComputeMode.PREFER_GPU),
    )

    result = execute_pipeline_request(request)

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    decision = next(
        item
        for item in result.execution_report.actual_decisions
        if item.node_id == node.id
    )
    assert decision.runtime_id == "cpu-numpy"
    assert decision.reason is DecisionReason.WORKLOAD_UNSUPPORTED
    assert "native-endian" in decision.reason_text


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
    validated_windows_compute_host,
):
    del validated_windows_compute_host
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
        mode=ComputeMode.CUSTOM,
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
        _accelerated_request(
            pipeline,
            data,
            compute_request,
            device_execution_telemetry=DeviceExecutionTelemetryConfig(
                synchronize_device_phases=True,
            ),
        ),
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
    assert len(result.execution_report.fallback_records) == 1
    fallback = result.execution_report.fallback_records[0]
    assert fallback.segment_id
    assert fallback.runtime_id == "cuda-cupy"
    assert fallback.node_ids == (gaussian.id,)
    assert fallback.reason.value == "out_of_memory"
    assert fallback.reason_code == "fake_oom"
    assert fallback.exception_type == "_FakeOOM"
    assert fallback.retryable
    assert fallback.device_attempt_count == 1
    assert fallback.cpu_retry_count == 1
    assert fallback.cleanup_succeeded
    assert fallback.memory_topology.value == "discrete"
    assert fallback.device_total_bytes == runtime.free_bytes
    assert fallback.device_free_bytes == runtime.free_bytes
    serialized = result.execution_report.as_dict()["fallback_records"][0]
    assert serialized["reason_code"] == "fake_oom"
    assert serialized["memory_topology"] == "discrete"
    assert result.execution_report.warnings
    observation = result.device_execution_telemetry
    assert observation is not None
    assert observation.host_to_device.count == 1
    assert observation.device_to_host.count == 0
    (failed_operation,) = observation.spans_for(DeviceExecutionPhase.DEVICE_OPERATION)
    assert failed_operation.node_id == gaussian.id
    assert failed_operation.succeeded is False
    assert runtime.live == {}
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
    captured_source_contexts = execution_module._capture_source_scientific_contexts(
        pipeline,
        provenance_request,
        cancel_callback=None,
    )
    source_contexts = {
        node_id: captured.scientific_context_fingerprint
        for node_id, captured in captured_source_contexts.items()
    }
    source_reuse_envelopes = {
        node_id: captured.source_reuse_envelope_fingerprint
        for node_id, captured in captured_source_contexts.items()
    }
    execution_module._publish_actual_compute_provenance(
        pipeline,
        compute_request,
        (_decision(gaussian.id, cpu_gaussian),),
        source_scientific_contexts=source_contexts,
        source_reuse_envelope_fingerprints=source_reuse_envelopes,
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
    assert result.execution_report is not None
    assert result.execution_report.request.mode is ComputeMode.CPU
    assert result.execution_report.actual_decisions
    assert all(
        decision.runtime_id == "cpu-numpy"
        for decision in result.execution_report.actual_decisions
    )
    assert result.pipeline is not None
    np.testing.assert_array_equal(result.pipeline.outputs[gaussian.id], data)


def test_real_headless_measurements_pipeline_finalizes_public_table_and_cleans(
    monkeypatch,
):
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        library_probe = registry.probe_library("cupy", refresh=True)
        if not library_probe.available:
            pytest.skip(library_probe.message or "CuPy is unavailable.")

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
                mode=ComputeMode.CUSTOM,
                node_preferences={
                    otsu.id: "cpu",
                    components.id: "cpu",
                    measurement.id: "implementation:cupy-measure-objects-basic-v1",
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
        assert decision.implementation_library_id == "cupy"
        assert decision.implementation_id == "cupy-measure-objects-basic-v1"
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
        assert provenance.implementation_library_id == "cupy"
        assert provenance.implementation_id == "cupy-measure-objects-basic-v1"

        _assert_private_cuda_scope_clean(
            runtime,
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()


def test_real_headless_background_gaussian_median_forms_one_device_segment():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    try:
        import cupy

        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.float32).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")
    registry = ComputeRegistry()
    try:
        for library_id in ("cupy", "cupyx"):
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
        mode=ComputeMode.CUSTOM,
        node_preferences={
            background.id: ("implementation:cupy-subtract-background-v1"),
            gaussian.id: "implementation:cupy-gaussian-blur-v1",
            median.id: "implementation:cupy-median-filter-v1",
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


def test_real_convert_dtype_gaussian_corridor_uses_one_device_round_trip(
    monkeypatch,
):
    """Prove the visible repair remains resident through downstream Gaussian."""

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
        conversion = pipeline.add_node("convert_dtype")
        gaussian = pipeline.add_node("gaussian_blur")
        pipeline.set_param(conversion.id, "output_dtype", "float32")
        pipeline.set_param(conversion.id, "scaling", "preserve")
        pipeline.set_param(gaussian.id, "sigma", 1.2)
        assert pipeline.connect("input", conversion.id).success
        assert pipeline.connect(conversion.id, gaussian.id).success

        data = np.arange(128 * 160, dtype=np.uint16).reshape(128, 160)
        pipeline.run(data, input_metadata={"axes": "YX"})
        expected = pipeline.outputs[gaussian.id].copy()

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
        compute_request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences={
                conversion.id: ("implementation:cupyx-convert-dtype-preserve-f32-v1"),
                gaussian.id: "implementation:cupy-gaussian-blur-v1",
            },
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        result = execute_pipeline_request(
            _accelerated_request(
                pipeline,
                data,
                compute_request,
                retain_node_ids=frozenset({gaussian.id}),
                prune_unretained=True,
            ),
            compute_registry=registry,
        )

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        assert [
            (segment.runtime_id, segment.node_ids)
            for segment in result.execution_report.plan.segments
        ] == [("cuda-cupy", (conversion.id, gaussian.id))]
        decisions = {
            decision.node_id: decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id in {conversion.id, gaussian.id}
        }
        assert decisions[conversion.id].implementation_id == (
            "cupyx-convert-dtype-preserve-f32-v1"
        )
        assert decisions[gaussian.id].implementation_id == ("cupy-gaussian-blur-v1")
        assert all(
            decision.runtime_id == "cuda-cupy"
            and decision.decision_kind is DecisionKind.SELECTED
            for decision in decisions.values()
        )
        assert transfers == {"host_to_device": 1, "device_to_host": 1}
        parity = operation_parity(
            "gaussian_blur",
            expected,
            result.pipeline.outputs[gaussian.id],
            input_dtypes=(np.dtype(np.float32),),
        )
        assert parity.passed, parity.detail
        _assert_private_cuda_scope_clean(
            runtime,
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("include_gaussian", "retain_cleanup"),
    ((False, False), (True, False), (False, True)),
)
def test_real_segmentation_cleanup_corridor_is_one_cuda_segment(
    monkeypatch,
    include_gaussian,
    retain_cleanup,
):
    """Prove the complete reviewed segmentation corridor stays resident."""

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        for library_id in ("cupy", "cupyx"):
            library_probe = registry.probe_library(library_id, refresh=True)
            if not library_probe.available:
                pytest.skip(library_probe.message or f"{library_id} is unavailable.")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        extract = pipeline.add_node("extract_channel")
        conversion = pipeline.add_node("convert_dtype")
        gaussian = pipeline.add_node("gaussian_blur") if include_gaussian else None
        threshold = pipeline.add_node("binary_threshold")
        remove_small = pipeline.add_node("remove_small_objects")
        fill = pipeline.add_node("fill_holes")
        components = pipeline.add_node("label_connected_components")
        pipeline.set_param(extract.id, "channel", 1)
        pipeline.set_param(conversion.id, "output_dtype", "float32")
        pipeline.set_param(conversion.id, "scaling", "preserve")
        if gaussian is not None:
            pipeline.set_param(gaussian.id, "sigma", 1.25)
        pipeline.set_param(
            threshold.id,
            "threshold",
            777.125 if include_gaussian else 1000.0,
        )
        pipeline.set_param(remove_small.id, "min_size", 8)
        pipeline.set_param(remove_small.id, "spatial_mode", "2D YX")
        pipeline.set_param(remove_small.id, "connectivity", "Face connected")
        pipeline.set_param(fill.id, "max_hole_size", 0)
        pipeline.set_param(fill.id, "spatial_mode", "2D YX")
        pipeline.set_param(fill.id, "connectivity", "Face connected")
        pipeline.set_param(components.id, "spatial_mode", "2D YX")
        pipeline.set_param(components.id, "connectivity", "Full connectivity")
        assert pipeline.connect("input", extract.id).success
        assert pipeline.connect(extract.id, conversion.id).success
        previous = conversion.id
        if gaussian is not None:
            assert pipeline.connect(previous, gaussian.id).success
            previous = gaussian.id
        assert pipeline.connect(previous, threshold.id).success
        assert pipeline.connect(threshold.id, remove_small.id).success
        assert pipeline.connect(remove_small.id, fill.id).success
        assert pipeline.connect(fill.id, components.id).success

        data = np.zeros((3, 64, 80), dtype=np.uint16)
        data[1, 8:24, 10:28] = 4000
        data[1, 36:57, 45:71] = 3000
        data[1, 4, 4] = 4000
        if not include_gaussian:
            data[1, 14, 18] = 0
        before = data.copy()
        pipeline.run(data, input_metadata={"axes": "CYX"})
        expected_fill = pipeline.outputs[fill.id].copy()
        expected = pipeline.outputs[components.id].copy()
        if gaussian is not None:
            gaussian_output = np.asarray(pipeline.outputs[gaussian.id])
            margin = float(
                np.min(
                    np.abs(
                        gaussian_output
                        - float(pipeline.nodes[threshold.id].params["threshold"])
                    )
                )
            )
            assert margin > 1.0

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
        preferences = {
            extract.id: "implementation:cupy-extract-channel-view-v1",
            conversion.id: ("implementation:cupyx-convert-dtype-preserve-f32-v1"),
            threshold.id: "implementation:cupy-binary-threshold-f32-exact-v1",
            remove_small.id: ("implementation:cupyx-remove-small-objects-bool-v1"),
            fill.id: "implementation:cupyx-fill-holes-all-v1",
            components.id: "implementation:cupyx-connected-components-v1",
        }
        if gaussian is not None:
            preferences[gaussian.id] = "implementation:cupy-gaussian-blur-v1"
        compute_request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences=preferences,
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        request = replace(
            _accelerated_request(
                pipeline,
                data,
                compute_request,
                retain_node_ids=frozenset(
                    {components.id, *([fill.id] if retain_cleanup else [])}
                ),
                prune_unretained=True,
            ),
            input_metadata={"axes": "CYX"},
        )

        result = execute_pipeline_request(request, compute_registry=registry)

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        expected_node_ids = [extract.id, conversion.id]
        if gaussian is not None:
            expected_node_ids.append(gaussian.id)
        expected_node_ids.extend(
            (threshold.id, remove_small.id, fill.id, components.id)
        )
        assert [
            (segment.runtime_id, segment.node_ids)
            for segment in result.execution_report.plan.segments
        ] == [("cuda-cupy", tuple(expected_node_ids))]
        (segment,) = result.execution_report.plan.segments
        assert {(port.node_id, port.port_index) for port in segment.exit_ports} == {
            (components.id, 0),
            *({(fill.id, 0)} if retain_cleanup else set()),
        }
        decisions = {
            decision.node_id: decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id in expected_node_ids
        }
        assert {
            node_id: decision.implementation_id
            for node_id, decision in decisions.items()
        } == {
            extract.id: "cupy-extract-channel-view-v1",
            conversion.id: "cupyx-convert-dtype-preserve-f32-v1",
            **({gaussian.id: "cupy-gaussian-blur-v1"} if gaussian is not None else {}),
            threshold.id: "cupy-binary-threshold-f32-exact-v1",
            remove_small.id: "cupyx-remove-small-objects-bool-v1",
            fill.id: "cupyx-fill-holes-all-v1",
            components.id: "cupyx-connected-components-v1",
        }
        assert all(
            decision.runtime_id == "cuda-cupy"
            and decision.decision_kind is DecisionKind.SELECTED
            and not decision.fallback_used
            for decision in decisions.values()
        )
        assert transfers == {
            "host_to_device": 1,
            "device_to_host": 2 if retain_cleanup else 1,
        }
        retained_ids = {components.id, *([fill.id] if retain_cleanup else [])}
        assert all(
            result.pipeline.outputs[node_id] is None
            for node_id in set(expected_node_ids) - retained_ids
        )
        if retain_cleanup:
            np.testing.assert_array_equal(
                result.pipeline.outputs[fill.id],
                expected_fill,
                strict=True,
            )
            fill_state = result.pipeline.output_states[fill.id]
            assert fill_state.dtype == "bool"
            assert fill_state.kind == "binary mask"
            assert fill_state.history[-2:] == (
                "Remove Small Objects",
                "Fill Holes",
            )
        np.testing.assert_array_equal(
            result.pipeline.outputs[components.id],
            expected,
            strict=True,
        )
        np.testing.assert_array_equal(data, before, strict=True)
        state = result.pipeline.output_states[components.id]
        assert state.shape == data.shape[1:]
        assert state.dtype == "int32"
        assert tuple(axis.name for axis in state.axes) == ("y", "x")
        assert state.channels == ()
        assert state.kind == "label image"
        assert state.history[-3:] == (
            "Remove Small Objects",
            "Fill Holes",
            "Label Connected Components",
        )
        # Pruned intermediates remain represented by exact execution-report
        # decisions without forcing extra D2H transfers. The retained terminal
        # additionally carries its committed node provenance.
        provenance = result.pipeline.node_compute_provenance[components.id]
        assert provenance.actual_implementation.implementation_id == (
            decisions[components.id].implementation_id
        )
        assert set(result.pipeline.node_compute_provenance) == retained_ids
        if retain_cleanup:
            assert result.pipeline.node_compute_provenance[
                fill.id
            ].actual_implementation.implementation_id == ("cupyx-fill-holes-all-v1")
        _assert_private_cuda_scope_clean(
            runtime,
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()


def test_real_student_gaussian_otsu_remove_corridor_attests_exact_assignments():
    """Reproduce the #32 host/GPU/host/GPU planning corridor without writers."""

    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        for library_id in ("cupy", "cupyx"):
            library_probe = registry.probe_library(library_id, refresh=True)
            if not library_probe.available:
                pytest.skip(library_probe.message or f"{library_id} is unavailable.")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        subtract = pipeline.add_node("subtract_background")
        rescale = pipeline.add_node("rescale_intensity")
        conversion = pipeline.add_node("convert_dtype")
        gaussian = pipeline.add_node("gaussian_blur")
        unsharp = pipeline.add_node("unsharp_mask")
        otsu = pipeline.add_node("otsu_threshold")
        remove_small = pipeline.add_node("remove_small_objects")
        pipeline.set_param(subtract.id, "radius", 11.0)
        pipeline.set_param(subtract.id, "light_background", False)
        pipeline.set_param(subtract.id, "disable_smoothing", False)
        pipeline.set_param(subtract.id, "clip_negative", True)
        pipeline.set_param(subtract.id, "spatial_mode", "2D YX")
        pipeline.set_param(rescale.id, "cutoff_mode", "Values")
        pipeline.set_param(rescale.id, "in_low_value", 0.0)
        pipeline.set_param(rescale.id, "in_high_value", 135.689)
        pipeline.set_param(rescale.id, "out_min", 0.0)
        pipeline.set_param(rescale.id, "out_max", 65535.0)
        pipeline.set_param(conversion.id, "output_dtype", "float32")
        pipeline.set_param(conversion.id, "scaling", "preserve")
        pipeline.set_param(gaussian.id, "sigma", 0.8)
        pipeline.set_param(unsharp.id, "radius", 2.0)
        pipeline.set_param(unsharp.id, "amount", 1.5)
        pipeline.set_param(otsu.id, "threshold_scope", "Stack histogram")
        pipeline.set_param(otsu.id, "histogram_bins", 256)
        pipeline.set_param(remove_small.id, "min_size", 27)
        pipeline.set_param(remove_small.id, "spatial_mode", "Auto from axes")
        pipeline.set_param(remove_small.id, "connectivity", "Face connected")
        previous = "input"
        for node in (
            subtract,
            rescale,
            conversion,
            gaussian,
            unsharp,
            otsu,
            remove_small,
        ):
            assert pipeline.connect(previous, node.id).success
            previous = node.id

        data = np.zeros((4, 48, 64), dtype=np.uint16)
        data[:, 8:30, 10:32] = 90
        data[1:4, 27:44, 39:58] = 125
        data[0, 3, 3] = 135
        data[2, 42:45, 5:8] = 75
        before = data.copy()
        pipeline.run(data, input_metadata={"axes": "ZYX"})
        expected = {
            node.id: np.asarray(pipeline.outputs[node.id]).copy()
            for node in (gaussian, otsu, remove_small)
        }

        implementation_ids = {
            gaussian.id: "cupy-gaussian-blur-v1",
            otsu.id: "cupy-otsu-threshold-exact-v1",
            remove_small.id: "cupyx-remove-small-objects-bool-v1",
        }
        preferences = {
            node_id: f"implementation:{implementation_id}"
            for node_id, implementation_id in implementation_ids.items()
        }
        compute_request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences=preferences,
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        request = replace(
            _accelerated_request(
                pipeline,
                data,
                compute_request,
                retain_node_ids=frozenset(implementation_ids),
                prune_unretained=True,
            ),
            input_metadata={"axes": "ZYX"},
        )

        result = execute_pipeline_request(request, compute_registry=registry)

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        assert result.execution_report.plan is not None
        assert [
            (
                segment.runtime_id,
                tuple(
                    node_id
                    for node_id in segment.node_ids
                    if node_id in implementation_ids
                ),
            )
            for segment in result.execution_report.plan.segments
            if set(segment.node_ids) & set(implementation_ids)
        ] == [
            ("cuda-cupy", (gaussian.id,)),
            ("cuda-cupy", (otsu.id, remove_small.id)),
        ]
        planned = {
            decision.node_id: decision
            for decision in result.execution_report.plan.decisions
            if decision.node_id in implementation_ids
        }
        actual = {
            decision.node_id: decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id in implementation_ids
        }
        assert set(planned) == set(actual) == set(implementation_ids)
        for node_id, implementation_id in implementation_ids.items():
            expected_library = "cupy" if node_id == gaussian.id else "cupyx"
            for decision in (planned[node_id], actual[node_id]):
                assert decision.implementation_id == implementation_id
                assert decision.runtime_id == "cuda-cupy"
                assert decision.implementation_library_id == expected_library
                assert not decision.fallback_used
            assert actual[node_id].decision_kind is DecisionKind.SELECTED

        gaussian_parity = operation_parity(
            "gaussian_blur",
            expected[gaussian.id],
            result.pipeline.outputs[gaussian.id],
            input_dtypes=(np.dtype(np.float32),),
        )
        assert gaussian_parity.passed, gaussian_parity.detail
        for node in (otsu, remove_small):
            parity = operation_parity(
                node.operation_id,
                expected[node.id],
                result.pipeline.outputs[node.id],
            )
            assert parity.passed, parity.detail
        np.testing.assert_array_equal(data, before, strict=True)
        _assert_private_cuda_scope_clean(
            registry.runtime("cuda-cupy"),
            runtime_probe.selected_device_id,
        )
    finally:
        registry.close()


def test_real_generated_cleanup_runner_preserves_v4_intent_and_provenance(
    tmp_path,
):
    """Generated workflow-v4 Python must execute both exact cleanup IDs."""

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
    finally:
        registry.close()

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    remove_small = pipeline.add_node("remove_small_objects")
    fill = pipeline.add_node("fill_holes")
    components = pipeline.add_node("label_connected_components")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(remove_small.id, "min_size", 3)
    pipeline.set_param(remove_small.id, "spatial_mode", "2D YX")
    pipeline.set_param(remove_small.id, "connectivity", "Face connected")
    pipeline.set_param(fill.id, "max_hole_size", 0)
    pipeline.set_param(fill.id, "spatial_mode", "2D YX")
    pipeline.set_param(fill.id, "connectivity", "Face connected")
    pipeline.set_param(components.id, "spatial_mode", "2D YX")
    pipeline.set_param(components.id, "connectivity", "Full connectivity")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, remove_small.id).success
    assert pipeline.connect(remove_small.id, fill.id).success
    assert pipeline.connect(fill.id, components.id).success

    source = np.zeros((31, 37), dtype=np.float32)
    source[4:15, 5:17] = 1.0
    source[9, 10] = 0.0
    source[22:28, 24:34] = 2.0
    source[2, 32] = 1.0
    before = source.copy()
    pipeline.run(source, input_metadata={"axes": "YX"})
    expected = pipeline.outputs[components.id].copy()

    preferences = {
        threshold.id: "implementation:cupy-binary-threshold-f32-exact-v1",
        remove_small.id: "implementation:cupyx-remove-small-objects-bool-v1",
        fill.id: "implementation:cupyx-fill-holes-all-v1",
        components.id: "implementation:cupyx-connected-components-v1",
    }
    compute_request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences=preferences,
        runtime_id="cuda-cupy",
        device_id=runtime_probe.selected_device_id,
        fallback_policy=FallbackPolicy.STRICT,
    )
    script_path = tmp_path / "generated-segmentation-cleanup.py"
    script_path.write_text(
        export_pipeline_to_python(pipeline, compute_request=compute_request),
        encoding="utf-8",
    )
    module_spec = importlib.util.spec_from_file_location(
        "vipp_generated_cleanup_integration",
        script_path,
    )
    assert module_spec is not None and module_spec.loader is not None
    generated = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(generated)
    embedded = json.loads(generated._WORKFLOW_JSON)

    assert embedded["version"] == WORKFLOW_VERSION
    assert embedded["execution"]["compute"]["mode"] == "custom"
    assert embedded["execution"]["compute"]["node_preferences"] == preferences

    results = generated.run_pipeline(
        source,
        input_metadata={"axes": "YX"},
    )

    report = results.execution_report
    assert report is not None
    assert report.cleanup_succeeded
    assert report.fallback_records == ()
    assert len(report.plan.segments) == 1
    assert report.plan.segments[0].node_ids == (
        threshold.id,
        remove_small.id,
        fill.id,
        components.id,
    )
    np.testing.assert_array_equal(results[components.id], expected, strict=True)
    np.testing.assert_array_equal(source, before, strict=True)
    assert results.output_states[remove_small.id].dtype == "bool"
    assert results.output_states[remove_small.id].kind == "binary mask"
    assert results.output_states[fill.id].dtype == "bool"
    assert results.output_states[fill.id].kind == "binary mask"
    assert results.output_states[components.id].dtype == "int32"
    assert results.output_states[components.id].kind == "label image"

    by_node = {item["node_id"]: item for item in results.execution_provenance["nodes"]}
    for node_id, implementation_id in (
        (remove_small.id, "cupyx-remove-small-objects-bool-v1"),
        (fill.id, "cupyx-fill-holes-all-v1"),
    ):
        identity = by_node[node_id]["actual_implementation"]
        assert by_node[node_id]["decision_kind"] == "selected"
        assert identity["runtime_id"] == "cuda-cupy"
        assert identity["implementation_library_id"] == "cupyx"
        assert identity["implementation_id"] == implementation_id
        assert identity["implementation_version"] == "1"
        assert (
            results.node_compute_provenance[
                node_id
            ].actual_implementation.implementation_id
            == implementation_id
        )
    assert results.execution_provenance["fallback_records"] == []
    assert results.execution_provenance["cleanup_succeeded"] is True


@pytest.mark.parametrize("mode", (ComputeMode.AUTO, ComputeMode.PREFER_GPU))
def test_real_gpu_modes_run_every_eligible_node_without_benchmark_evidence(mode):
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed.")
    try:
        import cupy

        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("No CUDA device is available.")
        cupy.zeros(1, dtype=cupy.float32).sum().item()
    except Exception as exc:
        pytest.skip(f"A working CUDA device is unavailable: {exc}")
    registry = ComputeRegistry()
    try:
        library_probes = tuple(
            registry.probe_library(library_id, refresh=True)
            for library_id in ("cupy", "cupyx")
        )
    finally:
        registry.close()
    for library_probe in library_probes:
        if not library_probe.available:
            pytest.skip(
                library_probe.message
                or f"{library_probe.library_id} is not policy-admitted."
            )

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    extract = pipeline.add_node("extract_channel")
    background = pipeline.add_node("subtract_background")
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(extract.id, "channel", 1)
    pipeline.set_param(background.id, "radius", 2.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(gaussian.id, "sigma", 1.2)
    pipeline.set_param(median.id, "size", 3)
    assert pipeline.connect("input", extract.id).success
    assert pipeline.connect(extract.id, background.id).success
    assert pipeline.connect(background.id, gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success

    data = np.random.default_rng(84).integers(
        0,
        4096,
        size=(2, 31, 37),
        dtype=np.uint16,
    )
    pipeline.run(data, input_metadata={"axes": "CYX"})
    expected = pipeline.outputs[median.id].copy()
    compute_request = ComputeRequest(mode=mode)
    request = replace(
        _accelerated_request(pipeline, data, compute_request),
        input_metadata={"axes": "CYX"},
    )

    result = execute_pipeline_request(request)

    assert result.error == ""
    assert result.pipeline is not None
    assert result.execution_report is not None
    assert result.execution_report.request.mode is mode
    assert result.execution_report.cleanup_succeeded
    decisions = {
        decision.node_id: decision
        for decision in result.execution_report.actual_decisions
    }
    assert decisions[extract.id].implementation_id == "cpu-extract_channel-v1"
    assert decisions[background.id].implementation_id == ("cupy-subtract-background-v1")
    assert decisions[gaussian.id].implementation_id == "cpu-gaussian_blur-v1"
    assert decisions[median.id].implementation_id == "cupy-median-filter-v1"
    assert all(not decision.fallback_used for decision in decisions.values())
    assert not result.execution_report.fallback_records
    np.testing.assert_array_equal(result.pipeline.outputs[median.id], expected)
    assert tuple(
        axis.name for axis in result.pipeline.output_states[median.id].axes
    ) == ("y", "x")
    for node_id in (extract.id, background.id, gaussian.id, median.id):
        decision = decisions[node_id]
        provenance = result.pipeline.node_compute_provenance[node_id]
        assert provenance.actual_implementation.implementation_id == (
            decision.implementation_id
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
            mode=ComputeMode.CUSTOM,
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
        assert decisions[canny.id].implementation_id == ("cupyx-canny-edges-exact-v1")
        assert decisions[otsu.id].implementation_id == ("cupy-otsu-threshold-exact-v1")

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

    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy")
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "The CUDA runtime is unavailable.")
        for library_id in ("cupy", "cupyx"):
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
            mode=ComputeMode.CUSTOM,
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
            mode=ComputeMode.CUSTOM,
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
        psf = np.exp(-(x * x + y * y) / np.float32(2.0 * 1.7**2)).astype(np.float32)
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
                mode=ComputeMode.CUSTOM,
                node_preferences={deconvolution.id: "implementation:rl-tv-cupy-f32-v1"},
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
