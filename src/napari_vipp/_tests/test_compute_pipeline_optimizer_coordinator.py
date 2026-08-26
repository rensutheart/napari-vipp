from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.compute_benchmark_adapter as adapter_module
from napari_vipp.core.compute import (
    BenchmarkCandidateResult,
    BenchmarkRecord,
    BenchmarkRecordKey,
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionPlan,
    ExecutionReport,
    ExecutionSegment,
    MemoryEstimate,
    MemoryTopology,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    canonical_digest,
)
from napari_vipp.core.compute_benchmark import BenchmarkBudgetExceeded
from napari_vipp.core.compute_benchmark_adapter import (
    workload_from_prepared_node_call,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    NodeBenchmarkPhase,
    NodeBenchmarkProgress,
    NodeBenchmarkUnavailable,
)
from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationCancelled,
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationEvidenceIncomplete,
    PipelineOptimizationSelectionBasis,
    PipelineValidationRequest,
    PipelineValidationWinner,
)
from napari_vipp.core.compute_pipeline_optimizer_coordinator import (
    ApplicationPipelineOptimizerCoordinator,
    PipelineOptimizerPhase,
    _actionable_repair_refusals,
    _adaptive_cpu_stop_is_safe_for_current_assignment,
    _build_optimizer_graph,
    _detach_source_payloads,
    _optimizer_validation_node_ids,
    _pipeline_input_peak,
    _pipeline_output_parity,
    _reviewable_pipeline_deviation,
    discover_pipeline_compute_repairs,
    fingerprint_pipeline_optimizer_sources,
    probe_pipeline_optimizer_environment,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    RuntimeMemorySnapshot,
)
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunResult
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.pipeline import (
    MANUAL_RUN_SKIP,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class _DeviceValue:
    value: np.ndarray


class _TransferRuntime:
    runtime_id = "cuda-cupy"
    array_domain = "cuda-cupy"

    def __init__(self, clock: _ManualClock) -> None:
        self.clock = clock
        self.released = 0
        self.released_values = []

    @contextmanager
    def execution_scope(self, **_kwargs):
        yield
        assert all(reference() is None for reference in self.released_values)

    def to_device(self, value, *, device_id=""):
        del device_id
        self.clock.advance(0.002)
        return _DeviceValue(np.array(value, copy=True))

    def to_host(self, value):
        self.clock.advance(0.003)
        return np.array(value.value, copy=True)

    def synchronize(self, *, device_id="") -> None:
        del device_id

    def release(self, value) -> None:
        self.released += 1
        self.released_values.append(weakref.ref(value))

    def memory_snapshot(self, *, device_id="") -> RuntimeMemorySnapshot:
        return RuntimeMemorySnapshot(
            self.runtime_id,
            device_id,
            MemoryTopology.DISCRETE,
            device_total_bytes=16 * 1024**3,
            device_free_bytes=15 * 1024**3,
        )


def _environment() -> ComputeEnvironment:
    return ComputeEnvironment(
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupyx"),
        runtime_versions=(("cuda-cupy", "test"),),
        runtime_probe_fingerprints=(("cuda-cupy", "test-runtime"),),
        device_id="cuda:0",
        device_name="Test GPU",
        device_class="nvidia-cuda",
        memory_topology=MemoryTopology.DISCRETE,
        total_accelerator_memory_bytes=16 * 1024**3,
    )


def _execution_plan(request, decisions, *, repair_suggestions=()):
    decisions = tuple(decisions)
    segments = tuple(
        ExecutionSegment(
            f"test-segment-{index}",
            decision.runtime_id,
            (decision.node_id,),
        )
        for index, decision in enumerate(decisions)
        if decision.runtime_id != "cpu-numpy"
    )
    return ExecutionPlan(
        request.compute_request.fingerprint,
        _environment().fingerprint,
        segments,
        decisions,
        repair_suggestions=tuple(repair_suggestions),
    )


class _NodeBenchmarker:
    def __init__(self, environment: ComputeEnvironment, registry: ComputeRegistry):
        self.environment = environment
        self.registry = registry
        self.observed_budgets: list[float] = []
        self.adaptive_stop_values: list[bool] = []
        self.allow_exact_workload_test_values: list[bool] = []

    def prepare(self, pipeline, node_id, **kwargs):
        self.observed_budgets.append(float(kwargs["time_budget_seconds"]))
        self.adaptive_stop_values.append(
            bool(kwargs.get("adaptive_candidate_stopping", False))
        )
        self.allow_exact_workload_test_values.append(
            bool(kwargs.get("allow_exact_workload_test", False))
        )
        progress = kwargs.get("progress")
        if progress is not None:
            progress(
                NodeBenchmarkProgress(
                    NodeBenchmarkPhase.PREPARING,
                    1,
                    4,
                    "Detached the exact node workload for testing.",
                )
            )
        call = pipeline.prepare_node_call(node_id)
        assert call is not None
        workload = workload_from_prepared_node_call(call)
        admitted = self.registry.implementations_for_operation(
            call.operation_id,
            allow_experimental=True,
        )
        assert len(admitted) == 1
        cpu_spec = compute_specs_for(call.operation_id)[0]
        gpu_spec = admitted[0]
        key = BenchmarkRecordKey(
            workload.fingerprint,
            self.environment.fingerprint,
            (cpu_spec.implementation_id, gpu_spec.implementation_id),
            "pipeline-test-v1",
            device_id="cuda:0",
        )
        return SimpleNamespace(
            environment=self.environment,
            admitted_specs=admitted,
            workload_fingerprint=workload.fingerprint,
            key=key,
        )

    def run(self, plan, **kwargs):
        cpu_id, gpu_id = plan.key.implementation_ids
        progress = kwargs.get("progress")
        if progress is not None:
            gpu_spec = plan.admitted_specs[0]
            progress(
                NodeBenchmarkProgress(
                    NodeBenchmarkPhase.BENCHMARKING,
                    3,
                    4,
                    "Running parity checks and paired benchmark rounds.",
                    measurement_completed=2,
                    measurement_total=3,
                    measurement_message=(
                        "Measuring paired warm round 3 of 3 for the GPU."
                    ),
                    implementation_id=gpu_spec.implementation_id,
                    implementation_version=gpu_spec.implementation_version,
                    measurement_phase="paired_warm",
                    operation_completed=37,
                    operation_total=171,
                    operation_message=(
                        "CuPy GPU: Rolling-ball background (37 of 171)."
                    ),
                )
            )
        record = BenchmarkRecord(
            plan.key,
            (
                BenchmarkCandidateResult(
                    cpu_id,
                    True,
                    0.1,
                    (0.1, 0.1, 0.1),
                ),
                BenchmarkCandidateResult(
                    gpu_id,
                    True,
                    0.025,
                    (0.025, 0.025, 0.025),
                    timing_scope="end-to-end-and-resident",
                    synchronized=True,
                    transfers_included=True,
                    warm_transfer_seconds=(0.005, 0.005, 0.005),
                    warm_resident_seconds=(0.02, 0.02, 0.02),
                ),
            ),
            "2026-07-28T00:00:00Z",
            "pipeline-test-v1",
            accepted_implementation_id=gpu_id,
        )
        return SimpleNamespace(plan=plan, record=record)


class _TimeoutNodeBenchmarker(_NodeBenchmarker):
    def __init__(
        self,
        environment: ComputeEnvironment,
        registry: ComputeRegistry,
        clock: _ManualClock,
    ) -> None:
        super().__init__(environment, registry)
        self.clock = clock

    def prepare(self, pipeline, node_id, **kwargs):
        plan = super().prepare(pipeline, node_id, **kwargs)
        progress = kwargs.get("progress")
        if progress is not None:
            gpu_spec = plan.admitted_specs[0]
            progress(
                NodeBenchmarkProgress(
                    NodeBenchmarkPhase.BENCHMARKING,
                    3,
                    4,
                    "Running parity checks and paired benchmark rounds.",
                    measurement_completed=1,
                    measurement_total=3,
                    measurement_message=(
                        "Measuring paired warm round 2 of 3 for the GPU."
                    ),
                    implementation_id=gpu_spec.implementation_id,
                    implementation_version=gpu_spec.implementation_version,
                    measurement_phase="paired_warm",
                )
            )
        self.clock.advance(float(kwargs["time_budget_seconds"]))
        raise BenchmarkBudgetExceeded("node benchmark time budget exhausted")


class _UnavailableNodeBenchmarker:
    def prepare(self, _pipeline, _node_id, **_kwargs):
        raise NodeBenchmarkUnavailable(
            "The uint16 image input requires an exact float32 conversion."
        )


class _PrivateExecutor:
    def __init__(
        self,
        clock: _ManualClock,
        environment: ComputeEnvironment,
        operation_node_id: str,
        writer_node_id: str,
        gpu_implementation_id: str,
        *,
        cpu_seconds: float = 0.1,
        gpu_seconds: float = 0.02,
    ) -> None:
        self.clock = clock
        self.environment = environment
        self.operation_node_id = operation_node_id
        self.writer_node_id = writer_node_id
        self.gpu_implementation_id = gpu_implementation_id
        self.cpu_seconds = cpu_seconds
        self.gpu_seconds = gpu_seconds
        self.target_sets: list[frozenset[str]] = []
        self.retained_sets: list[frozenset[str]] = []
        self.detached_source_arrays: list[np.ndarray] = []
        self.compute_requests: list[ComputeRequest] = []
        self.writer_execution_count = 0

    def __call__(self, request, **_kwargs):
        self.compute_requests.append(request.compute_request)
        restored = deserialize_workflow(request.workflow)
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        targets = frozenset(request.target_node_ids or ())
        retained = frozenset(request.retain_node_ids or ())
        self.target_sets.append(targets)
        self.retained_sets.append(retained)
        if self.writer_node_id:
            assert self.writer_node_id not in targets
            assert self.writer_node_id not in retained
        payload = next(iter(request.source_payloads.values()))
        self.detached_source_arrays.append(payload.data)
        assert isinstance(payload.data, np.ndarray)
        assert not payload.data.flags.writeable
        pipeline.run(
            payload.data,
            input_metadata=payload.metadata,
            input_name=payload.name,
            source_payloads=request.source_payloads,
            dirty_node_ids=request.dirty_node_ids,
            target_node_ids=request.target_node_ids,
            retain_node_ids=request.retain_node_ids,
            prune_unretained=request.prune_unretained,
        )
        if self.writer_node_id and self.writer_node_id in pipeline.completed_node_ids:
            self.writer_execution_count += 1
        preference = request.compute_request.preference_for(self.operation_node_id)
        gpu = preference.kind is NodePreferenceKind.IMPLEMENTATION
        operation_id = pipeline.nodes[self.operation_node_id].operation_id
        cpu_spec = compute_specs_for(operation_id)[0]
        gpu_spec = next(
            item
            for item in ComputeRegistry().implementations_for_operation(
                operation_id,
                allow_experimental=True,
            )
            if item.implementation_id == self.gpu_implementation_id
        )
        selected = gpu_spec if gpu else cpu_spec
        decision = NodeExecutionDecision(
            self.operation_node_id,
            operation_id,
            preference,
            selected.runtime_id,
            selected.implementation_library_id,
            selected.implementation_id,
            DecisionKind.SELECTED if gpu else DecisionKind.POLICY_CPU,
            (
                DecisionReason.SELECTED_IMPLEMENTATION
                if gpu
                else DecisionReason.AUTO_CPU
            ),
            "Private test assignment.",
        )
        self.clock.advance(self.gpu_seconds if gpu else self.cpu_seconds)
        report = ExecutionReport(
            request.compute_request,
            self.environment,
            plan=_execution_plan(request, (decision,)),
            actual_decisions=(decision,),
        )
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=pipeline,
            execution_report=report,
        )


class _RepairBaselineExecutor(_PrivateExecutor):
    def __call__(self, request, **kwargs):
        result = super().__call__(request, **kwargs)
        assert result.pipeline is not None
        assert result.execution_report is not None
        repairs = discover_pipeline_compute_repairs(
            ComputeRegistry(),
            result.pipeline,
            request.compute_request,
        )
        report = ExecutionReport(
            request.compute_request,
            self.environment,
            plan=_execution_plan(
                request,
                result.execution_report.actual_decisions,
                repair_suggestions=repairs,
            ),
            actual_decisions=result.execution_report.actual_decisions,
        )
        return PipelineRunResult(
            result.run_id,
            result.workflow,
            pipeline=result.pipeline,
            execution_report=report,
        )


def _writer_workflow(
    *,
    writer_operation_id: str = "save_output",
    connected: bool = True,
    output_path: str = "",
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    median = pipeline.add_node("median_filter")
    median.params["size"] = 3
    writer = pipeline.add_node(writer_operation_id)
    if writer_operation_id == "save_output" and output_path:
        writer.params.update(
            enabled="on",
            path=output_path,
            format="npy",
            overwrite="yes",
        )
    elif writer_operation_id == "batch_output":
        writer.params.update(
            tag="segmented cells",
            format="npy",
            subfolder="images/masks",
            filename_template="{source_stem}__{tag}",
            overwrite="yes",
        )
    assert pipeline.connect(source_id, median.id).success
    if connected:
        assert pipeline.connect(median.id, writer.id).success
    return pipeline, source_id, median.id, writer.id


def _pipeline_array_cache_snapshot(pipeline: PrototypePipeline):
    def value_snapshot(value):
        if value is None:
            return None
        array = np.asarray(value)
        return array.dtype.str, tuple(array.shape), array.tobytes()

    return (
        {node_id: value_snapshot(value) for node_id, value in pipeline.outputs.items()},
        {
            node_id: tuple(value_snapshot(value) for value in values)
            for node_id, values in pipeline.node_outputs.items()
        },
        frozenset(pipeline.completed_node_ids),
    )


def test_manual_rl_branches_expose_one_shared_dtype_repair_without_running_them():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    image_id = next(iter(pipeline.nodes))
    psf = pipeline.add_node("input")
    rl = pipeline.add_node("richardson_lucy_deconvolution")
    rl_tv = pipeline.add_node("richardson_lucy_tv_deconvolution")
    for node in (rl, rl_tv):
        node.params.update(
            spatial_mode="2D YX",
            iterations=25,
            filter_epsilon=1e-12,
        )
        assert pipeline.connect(image_id, node.id, target_port=0).success
        assert pipeline.connect(psf.id, node.id, target_port=1).success
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    psf_data = np.ones((3, 3), dtype=np.float32)
    psf_data /= psf_data.sum()
    pipeline.run(
        image,
        source_payloads={
            image_id: SourcePayload(image),
            psf.id: SourcePayload(psf_data),
        },
        manual_mode=MANUAL_RUN_SKIP,
    )
    authored = {
        node.id: (node.params["filter_epsilon"], node.params["iterations"])
        for node in (rl, rl_tv)
    }

    repairs = discover_pipeline_compute_repairs(
        ComputeRegistry(),
        pipeline,
        ComputeRequest("custom"),
        (rl.id, rl_tv.id),
    )

    assert {item.node_id for item in repairs} == {rl.id, rl_tv.id}
    assert all(item.input_port_index == 0 for item in repairs)
    assert all(item.current_dtype == "uint16" for item in repairs)
    assert all(item.target_dtype == "float32" for item in repairs)
    assert all(item.scaling == "preserve" and item.exact for item in repairs)
    assert all(pipeline.node_outputs[item.node_id] == [] for item in repairs)
    assert {
        node.id: (node.params["filter_epsilon"], node.params["iterations"])
        for node in (rl, rl_tv)
    } == authored

    refusals = _actionable_repair_refusals(pipeline, repairs)
    assert len(refusals) == 1
    assert refusals[0].code == "shared_dtype_conversion_available"
    assert "one visible Convert Dtype" in refusals[0].message
    assert "all 2 branches" in refusals[0].message
    assert "parameters will not be changed" in refusals[0].message


def test_repair_discovery_does_not_hash_pipeline_input_bytes(monkeypatch) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    image_id = next(iter(pipeline.nodes))
    rl = pipeline.add_node("richardson_lucy_deconvolution")
    psf = pipeline.add_node("input")
    assert pipeline.connect(image_id, rl.id, target_port=0).success
    assert pipeline.connect(psf.id, rl.id, target_port=1).success
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    psf_data = np.ones((3, 3), dtype=np.float32)
    psf_data /= psf_data.sum()
    pipeline.run(
        image,
        source_payloads={
            image_id: SourcePayload(image),
            psf.id: SourcePayload(psf_data),
        },
        manual_mode=MANUAL_RUN_SKIP,
    )

    def reject_hash(*_args, **_kwargs):
        raise AssertionError("repair discovery must not hash input bytes")

    monkeypatch.setattr(adapter_module, "_call_facts_fingerprint", reject_hash)
    repairs = discover_pipeline_compute_repairs(
        ComputeRegistry(),
        pipeline,
        ComputeRequest("custom"),
        (rl.id,),
    )

    assert len(repairs) == 1
    assert repairs[0].node_id == rl.id
    assert repairs[0].current_dtype == "uint16"
    assert repairs[0].target_dtype == "float32"


def test_native_uint16_otsu_does_not_offer_an_unnecessary_conversion():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    otsu = pipeline.add_node("otsu_threshold")
    assert pipeline.connect(source_id, otsu.id).success
    data = np.arange(256, dtype=np.uint16).reshape(16, 16)
    pipeline.run(data)

    repairs = discover_pipeline_compute_repairs(
        ComputeRegistry(),
        pipeline,
        ComputeRequest("custom"),
        (otsu.id,),
    )

    assert repairs == ()


def test_optimizer_returns_actionable_dtype_repair_instead_of_generic_refusal(
    tmp_path,
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect(source_id, gaussian.id).success
    document = serialize_workflow(
        pipeline,
        compute_request=ComputeRequest("custom"),
    )
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation("gaussian_blur")[0]
    executor = _RepairBaselineExecutor(
        clock,
        environment,
        gaussian.id,
        "",
        gpu_spec.implementation_id,
    )
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_UnavailableNodeBenchmarker(),
    )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as caught:
        coordinator.optimize(
            document,
            {source_id: SourcePayload(np.arange(64, dtype=np.uint16).reshape(8, 8))},
            ComputeRequest("custom"),
            time_budget_seconds=20.0,
        )

    assert len(caught.value.reasons) == 1
    reason = caught.value.reasons[0]
    assert reason.code == "dtype_conversion_available"
    assert "Add conversion" in reason.message
    assert "run Find fastest again" in reason.message
    assert "parameter will be changed" in reason.message
    assert "No unlocked node" not in str(caught.value)


def test_application_optimizer_is_private_writer_free_and_evidence_gated(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64).T
    original = values.copy()
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator.probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    executor = _PrivateExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
    )
    node_benchmarker = _NodeBenchmarker(environment, registry)
    progress = []
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=node_benchmarker,
    )

    baseline_request = ComputeRequest("prefer_gpu", allow_experimental=True)
    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(values, name="private-source")},
        ComputeRequest("custom", allow_experimental=True),
        baseline_compute_request=baseline_request,
        time_budget_seconds=20.0,
        progress=progress.append,
    )

    assert result.proposal.identity_digest == result.identity.digest
    assert dict(result.proposal.baseline_assignment)[median_id].startswith("cpu-")
    proposed = next(row for row in result.proposal.rows if row.node_id == median_id)
    assert proposed.changed
    assert proposed.proposed_implementation_id == gpu_spec.implementation_id
    assert result.proposal.validation_measurement_rounds == 5
    assert len(executor.target_sets) == 13
    assert writer_id not in {row.node_id for row in result.proposal.rows}
    assert all(writer_id not in targets for targets in executor.target_sets)
    assert runtime.released == 3
    assert progress[0].phase is PipelineOptimizerPhase.PREPARING
    assert progress[-1].phase is PipelineOptimizerPhase.COMPLETE
    assert node_benchmarker.observed_budgets == [pytest.approx(19.9)]
    assert node_benchmarker.adaptive_stop_values == [False]
    assert node_benchmarker.allow_exact_workload_test_values == [True]
    assert executor.compute_requests[0].fingerprint == baseline_request.fingerprint
    assert all(
        request.mode.value == "custom" for request in executor.compute_requests[1:]
    )
    operation = next(
        item for item in progress if item.measurement_phase == "paired_warm"
    )
    assert operation.phase is PipelineOptimizerPhase.BENCHMARKING
    assert (operation.completed, operation.total) == (2, 7)
    assert (operation.operation_completed, operation.operation_total) == (37, 171)
    assert operation.node_id == median_id
    assert operation.node_title == pipeline.nodes[median_id].title
    assert operation.implementation_id == gpu_spec.implementation_id
    assert operation.operation_message == (
        "CuPy GPU: Rolling-ball background (37 of 171)."
    )
    np.testing.assert_array_equal(values, original)
    assert all(
        not np.shares_memory(item, values) for item in executor.detached_source_arrays
    )
    assert (
        fingerprint_pipeline_optimizer_sources(
            document,
            {source_id: SourcePayload(values, name="private-source")},
        )
        == result.identity.source_fingerprint
    )
    values[0, 0] += 1
    assert (
        fingerprint_pipeline_optimizer_sources(
            document,
            {source_id: SourcePayload(values, name="private-source")},
        )
        != result.identity.source_fingerprint
    )


def test_optimizer_source_identity_includes_and_preserves_source_item(tmp_path):
    pipeline, source_id, _median_id, _writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    source_path = tmp_path / "source.npy"
    np.save(source_path, np.arange(64, dtype=np.uint16).reshape(8, 8))
    payload = load_frozen_file_source_snapshot(source_path).payload
    assert payload.source_item is not None
    changed_item = replace(
        payload.source_item,
        reader=replace(
            payload.source_item.reader,
            version=f"{payload.source_item.reader.version}-changed",
        ),
    )
    changed_payload = replace(payload, source_item=changed_item)

    original_fingerprint = fingerprint_pipeline_optimizer_sources(
        document,
        {source_id: payload},
    )
    changed_fingerprint = fingerprint_pipeline_optimizer_sources(
        document,
        {source_id: changed_payload},
    )
    detached = _detach_source_payloads(
        pipeline,
        {source_id: payload},
        frozenset({source_id}),
        check_abort=lambda: None,
    )

    assert changed_fingerprint != original_fingerprint
    assert detached.payloads[source_id].source_item == payload.source_item


def test_unavailable_proposed_backend_is_rejected_and_optimizer_continues(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    cpu_spec = compute_specs_for("median_filter")[0]

    class PlanningMismatchExecutor(_PrivateExecutor):
        def __call__(self, request, **kwargs):
            result = super().__call__(request, **kwargs)
            if (
                request.compute_request.preference_for(median_id).kind
                is not NodePreferenceKind.IMPLEMENTATION
            ):
                return result
            assert result.pipeline is not None
            decision = NodeExecutionDecision(
                median_id,
                "median_filter",
                request.compute_request.preference_for(median_id),
                cpu_spec.runtime_id,
                cpu_spec.implementation_library_id,
                cpu_spec.implementation_id,
                DecisionKind.POLICY_CPU,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                "An upstream output descriptor was unresolved.",
            )
            return PipelineRunResult(
                result.run_id,
                result.workflow,
                pipeline=result.pipeline,
                execution_report=ExecutionReport(
                    request.compute_request,
                    environment,
                    plan=_execution_plan(request, (decision,)),
                    actual_decisions=(decision,),
                ),
            )

    executor = PlanningMismatchExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
    )
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_NodeBenchmarker(environment, registry),
    )

    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(values)},
        ComputeRequest("custom", allow_experimental=True),
        time_budget_seconds=20.0,
    )

    median_row = next(row for row in result.proposal.rows if row.node_id == median_id)
    assert not median_row.changed
    assert median_row.proposed_implementation_id == cpu_spec.implementation_id
    assert len(result.candidate_refusals) == 1
    refusal = result.candidate_refusals[0]
    assert refusal.node_id == median_id
    assert refusal.code == "proposed_parity_planning_assignment_mismatch"
    assert "unresolved" in refusal.message
    assert all(
        (item.node_id, item.implementation_id)
        != (median_id, gpu_spec.implementation_id)
        for item in result.exact_workload_qualifications
    )
    assert executor.writer_execution_count == 0


def test_rejected_candidate_does_not_block_unrelated_gpu_improvement(
    tmp_path,
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    unavailable = pipeline.add_node("median_filter")
    surviving = pipeline.add_node("median_filter")
    unavailable.title = "Unavailable Median"
    surviving.title = "Surviving Median"
    unavailable.params["size"] = 3
    surviving.params["size"] = 5
    assert pipeline.connect(source_id, unavailable.id).success
    assert pipeline.connect(source_id, surviving.id).success
    request = ComputeRequest("custom", allow_experimental=True)
    document = serialize_workflow(pipeline, compute_request=request)
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    cpu_spec = compute_specs_for("median_filter")[0]
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]

    def executor(run_request, **_kwargs):
        restored = deserialize_workflow(run_request.workflow)
        result_pipeline = PrototypePipeline()
        result_pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        payload = run_request.source_payloads[source_id]
        result_pipeline.run(
            payload.data,
            input_metadata=payload.metadata,
            input_name=payload.name,
            source_payloads=run_request.source_payloads,
            dirty_node_ids=run_request.dirty_node_ids,
            target_node_ids=run_request.target_node_ids,
            retain_node_ids=run_request.retain_node_ids,
            prune_unretained=run_request.prune_unretained,
        )
        decisions = []
        gpu_count = 0
        for node in (unavailable, surviving):
            preference = run_request.compute_request.preference_for(node.id)
            requested_gpu = (
                preference.kind is NodePreferenceKind.IMPLEMENTATION
                and preference.value == gpu_spec.implementation_id
            )
            selected = (
                cpu_spec
                if node.id == unavailable.id and requested_gpu
                else gpu_spec
                if requested_gpu
                else cpu_spec
            )
            selected_gpu = selected.runtime_id != "cpu-numpy"
            gpu_count += int(selected_gpu)
            decisions.append(
                NodeExecutionDecision(
                    node.id,
                    node.operation_id,
                    preference,
                    selected.runtime_id,
                    selected.implementation_library_id,
                    selected.implementation_id,
                    (
                        DecisionKind.SELECTED
                        if selected_gpu
                        else DecisionKind.POLICY_CPU
                    ),
                    (
                        DecisionReason.SELECTED_IMPLEMENTATION
                        if selected_gpu
                        else DecisionReason.WORKLOAD_UNSUPPORTED
                        if requested_gpu
                        else DecisionReason.EXPLICIT_CPU
                    ),
                    (
                        "An upstream output descriptor was unresolved."
                        if requested_gpu and not selected_gpu
                        else "Exact parallel test assignment."
                    ),
                )
            )
        clock.advance(0.02 if gpu_count else 0.1)
        return PipelineRunResult(
            run_request.run_id,
            run_request.workflow,
            pipeline=result_pipeline,
            execution_report=ExecutionReport(
                run_request.compute_request,
                environment,
                plan=_execution_plan(run_request, decisions),
                actual_decisions=tuple(decisions),
            ),
        )

    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_NodeBenchmarker(environment, registry),
    )

    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(values)},
        request,
        time_budget_seconds=30.0,
    )

    rows = {row.node_id: row for row in result.proposal.rows}
    assert not rows[unavailable.id].changed
    assert rows[unavailable.id].proposed_implementation_id == cpu_spec.implementation_id
    assert rows[surviving.id].changed
    assert rows[surviving.id].proposed_implementation_id == gpu_spec.implementation_id
    assert [(item.node_id, item.code) for item in result.candidate_refusals] == [
        (unavailable.id, "proposed_parity_planning_assignment_mismatch")
    ]


@pytest.mark.parametrize("writer_operation_id", ["batch_output", "save_output"])
@pytest.mark.parametrize("connected", [True, False], ids=["connected", "disconnected"])
@pytest.mark.parametrize("retention_mode", ["keep-all", "smart", "low"])
def test_retained_writers_are_identity_only_and_never_run(
    tmp_path,
    monkeypatch,
    writer_operation_id,
    connected,
    retention_mode,
):
    artifact = tmp_path / "writer-must-not-run.npy"
    pipeline, source_id, median_id, writer_id = _writer_workflow(
        writer_operation_id=writer_operation_id,
        connected=connected,
        output_path=str(artifact),
    )
    values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    if writer_operation_id == "save_output":
        pipeline.nodes[writer_id].params["enabled"] = "off"
    pipeline.run(values)
    if writer_operation_id == "save_output":
        pipeline.nodes[writer_id].params["enabled"] = "on"
    assert not artifact.exists()

    request = ComputeRequest("custom", allow_experimental=True)
    document = serialize_workflow(pipeline, compute_request=request)
    live_cache_before = _pipeline_array_cache_snapshot(pipeline)
    requested_retained = (
        frozenset(pipeline.nodes)
        if retention_mode == "keep-all"
        else frozenset({writer_id})
    )

    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    executor = _PrivateExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
    )
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_NodeBenchmarker(environment, registry),
    )

    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(values)},
        request,
        retain_node_ids=requested_retained,
        time_budget_seconds=20.0,
    )

    assert result.identity.cache_retention_fingerprint == canonical_digest(
        sorted(requested_retained)
    )
    assert median_id in result.evidence
    assert median_id in result.benchmarked_node_ids
    assert writer_id not in {row.node_id for row in result.proposal.rows}
    assert executor.writer_execution_count == 0
    assert executor.target_sets
    assert executor.retained_sets
    assert all(median_id in targets for targets in executor.target_sets)
    assert all(writer_id not in targets for targets in executor.target_sets)
    assert all(writer_id not in retained for retained in executor.retained_sets)
    assert not artifact.exists()
    assert serialize_workflow(pipeline, compute_request=request) == document
    assert _pipeline_array_cache_snapshot(pipeline) == live_cache_before


def test_retained_writer_cancellation_keeps_writer_and_live_state_untouched(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "cancelled-writer-must-not-run.npy"
    pipeline, source_id, median_id, writer_id = _writer_workflow(
        output_path=str(artifact)
    )
    request = ComputeRequest("custom", allow_experimental=True)
    document = serialize_workflow(pipeline, compute_request=request)
    values = np.arange(64, dtype=np.uint16).reshape(8, 8)
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    executor = _PrivateExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
    )
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_NodeBenchmarker(environment, registry),
    )

    with pytest.raises(PipelineOptimizationCancelled):
        coordinator.optimize(
            document,
            {source_id: SourcePayload(values)},
            request,
            retain_node_ids=frozenset(pipeline.nodes),
            time_budget_seconds=20.0,
            cancelled=lambda: bool(executor.target_sets),
        )

    assert len(executor.target_sets) == 1
    assert executor.writer_execution_count == 0
    assert writer_id not in executor.target_sets[0]
    assert writer_id not in executor.retained_sets[0]
    assert not artifact.exists()
    assert serialize_workflow(pipeline, compute_request=request) == document


def test_node_benchmark_timeout_reports_stage_progress_and_no_optimality(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "timeout-writer-must-not-run.npy"
    pipeline, source_id, median_id, writer_id = _writer_workflow(
        output_path=str(artifact)
    )
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    executor = _PrivateExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
    )
    node_benchmarker = _TimeoutNodeBenchmarker(environment, registry, clock)
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=node_benchmarker,
    )
    request = ComputeRequest("custom", allow_experimental=True)
    progress = []

    with pytest.raises(PipelineOptimizationDeadlineExceeded) as caught:
        coordinator.optimize(
            document,
            {
                source_id: SourcePayload(
                    np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
                )
            },
            request,
            retain_node_ids=frozenset(pipeline.nodes),
            time_budget_seconds=20.0,
            progress=progress.append,
        )

    error = caught.value
    report = error.report
    assert report is not None
    assert report.stage == "node-benchmark"
    assert report.stage_message == ("Measuring paired warm round 2 of 3 for the GPU.")
    assert report.elapsed_seconds == pytest.approx(20.0)
    assert report.budget_seconds == pytest.approx(20.0)
    assert (report.overall_completed, report.overall_total) == (2, 7)
    assert report.node_id == median_id
    assert report.node_title == pipeline.nodes[median_id].title
    assert (report.node_index, report.node_total) == (1, 1)
    assert (report.operation_completed, report.operation_total) == (1, 3)
    assert report.operation_message == report.stage_message
    assert report.completed_node_ids == ()
    assert report.reused_node_ids == ()
    assert report.baseline_completed
    assert not report.validation_started
    assert not report.validation_completed
    assert report.partial_node_discarded
    message = str(error).lower()
    assert "no fastest assignment was determined" in message
    assert "current pipeline was not proven fastest" in message
    assert "no settings changed" in message
    assert "partial timings" in message
    assert request.preference_for(median_id).kind is NodePreferenceKind.AUTO
    assert node_benchmarker.observed_budgets == [pytest.approx(19.9)]
    nested = next(item for item in progress if item.measurement_phase == "paired_warm")
    assert (nested.operation_completed, nested.operation_total) == (1, 3)
    assert nested.node_id == median_id
    assert executor.writer_execution_count == 0
    assert all(writer_id not in targets for targets in executor.target_sets)
    assert all(writer_id not in retained for retained in executor.retained_sets)
    assert not artifact.exists()


def test_close_pipeline_validation_uses_full_fifteen_rounds(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    executor = _PrivateExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
        cpu_seconds=0.1,
        gpu_seconds=0.096,
    )
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_NodeBenchmarker(environment, registry),
    )

    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))},
        ComputeRequest("custom", allow_experimental=True),
        time_budget_seconds=20.0,
    )

    proposal = result.proposal
    median_row = next(row for row in proposal.rows if row.node_id == median_id)
    assert proposal.validation_winner is PipelineValidationWinner.INCONCLUSIVE
    assert (
        proposal.selection_basis
        is PipelineOptimizationSelectionBasis.PAIRED_INCONCLUSIVE_RETAINED_CURRENT
    )
    assert proposal.validation_measurement_rounds == 15
    assert median_row.current_implementation_id == median_row.proposed_implementation_id
    assert median_row.current_preference == median_row.proposed_preference
    assert dict(proposal.tested_assignment)[median_id] == gpu_spec.implementation_id
    assert median_id in result.evidence
    assert median_id in result.measured_node_ids
    assert len(executor.target_sets) == 33


def test_decisive_current_pipeline_win_stops_after_five_rounds(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    environment = _environment()
    clock = _ManualClock()
    registry = ComputeRegistry()
    runtime = _TransferRuntime(clock)
    monkeypatch.setattr(registry, "runtime", lambda _runtime_id: runtime)
    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator."
        "probe_compute_environment",
        lambda *_args, **_kwargs: (environment, ()),
    )
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    executor = _PrivateExecutor(
        clock,
        environment,
        median_id,
        writer_id,
        gpu_spec.implementation_id,
        cpu_seconds=0.02,
        gpu_seconds=0.10,
    )
    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
        node_benchmarker=_NodeBenchmarker(environment, registry),
    )

    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64))},
        ComputeRequest("custom", allow_experimental=True),
        retain_node_ids=frozenset(pipeline.nodes),
        time_budget_seconds=20.0,
    )

    proposal = result.proposal
    median_row = next(row for row in proposal.rows if row.node_id == median_id)
    assert proposal.validation_winner is PipelineValidationWinner.CURRENT
    assert proposal.validation_measurement_rounds == 5
    assert proposal.validated_current_speedup_lower_confidence_bound > 1.0
    assert dict(proposal.tested_assignment)[median_id] == gpu_spec.implementation_id
    assert median_row.current_implementation_id == median_row.proposed_implementation_id
    assert len(executor.target_sets) == 13
    assert executor.writer_execution_count == 0
    assert all(writer_id not in retained for retained in executor.retained_sets)


def test_locked_gpu_graph_node_needs_no_comparative_benchmark_record():
    pipeline, _source_id, median_id, _writer_id = _writer_workflow()
    pipeline.run(np.arange(64, dtype=np.uint16).reshape(8, 8))
    registry = ComputeRegistry()
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    request = ComputeRequest(
        "custom",
        {
            median_id: NodeComputePreference(
                "implementation",
                gpu_spec.implementation_id,
            )
        },
        allow_experimental=True,
    )

    nodes, _edges, _workloads = _build_optimizer_graph(
        registry,
        pipeline,
        frozenset({next(iter(pipeline.nodes)), median_id}),
        frozenset(),
        request,
        {
            median_id: SimpleNamespace(
                implementation_id=gpu_spec.implementation_id,
                memory_estimate=MemoryEstimate(
                    runtime_managed_peak_bytes=1_024,
                    total_device_peak_bytes=2_048,
                    uncertainty_bytes=512,
                    model_id="locked-test-v1",
                ),
            )
        },
        {},
        {},
        frozenset({median_id}),
        check_abort=lambda: None,
    )

    median = next(node for node in nodes if node.node_id == median_id)
    assert median.optimizer_locked
    assert median.current_implementation_id == gpu_spec.implementation_id
    assert [candidate.implementation_id for candidate in median.candidates] == [
        gpu_spec.implementation_id
    ]
    assert median.candidates[0].minimum_workspace_bytes == 2_560


def test_application_optimizer_refuses_non_custom_and_missing_sources(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    median = pipeline.add_node("median_filter")
    assert pipeline.connect(source_id, median.id).success
    document = serialize_workflow(pipeline)
    coordinator = ApplicationPipelineOptimizerCoordinator(
        ComputeRegistry(),
        tmp_path / "benchmarks.json",
    )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as non_custom:
        coordinator.optimize(document, {}, ComputeRequest("cpu"))
    assert non_custom.value.reasons[0].code == "custom_required"

    with pytest.raises(TypeError, match="baseline_compute_request"):
        coordinator.optimize(
            document,
            {},
            ComputeRequest("custom"),
            baseline_compute_request=object(),
        )

    with pytest.raises(ValueError, match="incompatible field.*device_id"):
        coordinator.optimize(
            document,
            {},
            ComputeRequest("custom"),
            baseline_compute_request=ComputeRequest("auto", device_id="cuda:1"),
        )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as missing:
        coordinator.optimize(document, {}, ComputeRequest("custom"))
    assert missing.value.reasons[0].code == "source_identity_incomplete"


def test_writer_filtering_does_not_hide_unknown_retention_ids(tmp_path):
    pipeline, _source_id, _median_id, writer_id = _writer_workflow(
        writer_operation_id="batch_output",
        connected=False,
    )
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("custom"))
    coordinator = ApplicationPipelineOptimizerCoordinator(
        ComputeRegistry(),
        tmp_path / "benchmarks.json",
    )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as caught:
        coordinator.optimize(
            document,
            {},
            ComputeRequest("custom"),
            retain_node_ids=(writer_id, "missing_output"),
        )

    assert len(caught.value.reasons) == 1
    reason = caught.value.reasons[0]
    assert reason.code == "retain_identity_invalid"
    assert "missing_output" in reason.message


def test_adaptive_cpu_stop_requires_a_retained_gpu_current_assignment():
    assert not _adaptive_cpu_stop_is_safe_for_current_assignment(None)
    assert not _adaptive_cpu_stop_is_safe_for_current_assignment(
        SimpleNamespace(runtime_id="cpu-numpy")
    )
    assert _adaptive_cpu_stop_is_safe_for_current_assignment(
        SimpleNamespace(runtime_id="cuda-cupy")
    )


def test_environment_recheck_probes_exact_candidates_without_execution(monkeypatch):
    pipeline, _source_id, _median_id, _writer_id = _writer_workflow()
    document = serialize_workflow(pipeline)
    environment = _environment()
    registry = ComputeRegistry()
    calls = []

    def probe(_registry, request, specs):
        calls.append((request, tuple(specs)))
        return environment, ()

    monkeypatch.setattr(
        "napari_vipp.core.compute_pipeline_optimizer_coordinator.probe_compute_environment",
        probe,
    )
    request = ComputeRequest("custom", allow_experimental=True)

    actual = probe_pipeline_optimizer_environment(registry, document, request)

    assert actual.fingerprint == environment.fingerprint
    assert len(calls) == 1
    assert {item.operation_id for item in calls[0][1]} == {"median_filter"}

    public = probe_pipeline_optimizer_environment(
        registry,
        document,
        ComputeRequest("custom", allow_experimental=False),
    )

    assert public.fingerprint == environment.fingerprint
    assert len(calls) == 2
    assert {item.operation_id for item in calls[1][1]} == {"median_filter"}


def test_pipeline_validation_checks_unchanged_observable_downstream_output(
    tmp_path,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    gaussian = pipeline.add_node("gaussian_blur")
    threshold = pipeline.add_node("binary_threshold")
    assert pipeline.connect(source_id, gaussian.id).success
    assert pipeline.connect(gaussian.id, threshold.id).success
    document = serialize_workflow(pipeline)
    registry = ComputeRegistry()
    gpu_spec = registry.implementations_for_operation(
        "gaussian_blur",
        allow_experimental=True,
    )[0]
    retained_sets = []

    def executor(request, **_kwargs):
        restored = deserialize_workflow(request.workflow)
        result_pipeline = PrototypePipeline()
        result_pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        retained_sets.append(frozenset(request.retain_node_ids))
        gpu = (
            request.compute_request.preference_for(gaussian.id).value
            == gpu_spec.implementation_id
        )
        source = np.zeros((4, 4), dtype=np.float32)
        gaussian_output = np.full(
            (4, 4),
            1.0 + (1e-7 if gpu else -1e-7),
            dtype=np.float32,
        )
        threshold_output = gaussian_output > 1.0
        for node_id, value in (
            (source_id, source),
            (gaussian.id, gaussian_output),
            (threshold.id, threshold_output),
        ):
            result_pipeline.outputs[node_id] = value
            result_pipeline.node_outputs[node_id] = [value]
            result_pipeline.completed_node_ids.add(node_id)
        gaussian_spec = gpu_spec if gpu else compute_specs_for("gaussian_blur")[0]
        threshold_spec = compute_specs_for("binary_threshold")[0]
        decisions = (
            NodeExecutionDecision(
                gaussian.id,
                "gaussian_blur",
                request.compute_request.preference_for(gaussian.id),
                gaussian_spec.runtime_id,
                gaussian_spec.implementation_library_id,
                gaussian_spec.implementation_id,
                DecisionKind.SELECTED if gpu else DecisionKind.POLICY_CPU,
                (
                    DecisionReason.SELECTED_IMPLEMENTATION
                    if gpu
                    else DecisionReason.EXPLICIT_CPU
                ),
                "Exact private validation decision.",
            ),
            NodeExecutionDecision(
                threshold.id,
                "binary_threshold",
                request.compute_request.preference_for(threshold.id),
                threshold_spec.runtime_id,
                threshold_spec.implementation_library_id,
                threshold_spec.implementation_id,
                DecisionKind.POLICY_CPU,
                DecisionReason.EXPLICIT_CPU,
                "Exact private validation decision.",
            ),
        )
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=result_pipeline,
            execution_report=ExecutionReport(
                request.compute_request,
                _environment(),
                plan=_execution_plan(request, decisions),
                actual_decisions=decisions,
            ),
        )

    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        executor=executor,
    )
    current = (
        (source_id, compute_specs_for("input")[0].implementation_id),
        (gaussian.id, compute_specs_for("gaussian_blur")[0].implementation_id),
        (
            threshold.id,
            compute_specs_for("binary_threshold")[0].implementation_id,
        ),
    )
    proposed = tuple(
        (node_id, gpu_spec.implementation_id if node_id == gaussian.id else impl)
        for node_id, impl in current
    )

    validation = coordinator._validate_assignments(
        document,
        {source_id: SourcePayload(np.zeros((4, 4), dtype=np.float32))},
        ComputeRequest("custom", allow_experimental=True),
        _environment(),
        pipeline,
        frozenset(pipeline.nodes),
        frozenset(),
        PipelineValidationRequest("identity", current, proposed),
        deadline=coordinator.clock() + 30.0,
        cancelled=None,
    )

    assert not validation.parity_passed
    assert "Threshold" in validation.detail
    assert len(retained_sets) == 2
    assert all(threshold.id in node_ids for node_ids in retained_sets)


def test_pipeline_validation_quantifies_small_mask_difference_then_times(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    otsu = pipeline.add_node("otsu_threshold")
    assert pipeline.connect(source_id, otsu.id).success
    document = serialize_workflow(pipeline)
    registry = ComputeRegistry()
    gpu_spec = registry.implementations_for_operation(
        "otsu_threshold",
        allow_experimental=True,
    )[0]
    cpu_spec = compute_specs_for("otsu_threshold")[0]
    clock = _ManualClock()
    execution_count = 0

    def executor(request, **_kwargs):
        nonlocal execution_count
        execution_count += 1
        restored = deserialize_workflow(request.workflow)
        result_pipeline = PrototypePipeline()
        result_pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        gpu = (
            request.compute_request.preference_for(otsu.id).value
            == gpu_spec.implementation_id
        )
        source = np.zeros((1000,), dtype=np.float32)
        output = np.zeros((1000,), dtype=bool)
        if gpu:
            output[17] = True
        for node_id, value in ((source_id, source), (otsu.id, output)):
            result_pipeline.outputs[node_id] = value
            result_pipeline.node_outputs[node_id] = [value]
            result_pipeline.completed_node_ids.add(node_id)
        selected = gpu_spec if gpu else cpu_spec
        decision = NodeExecutionDecision(
            otsu.id,
            "otsu_threshold",
            request.compute_request.preference_for(otsu.id),
            selected.runtime_id,
            selected.implementation_library_id,
            selected.implementation_id,
            DecisionKind.SELECTED if gpu else DecisionKind.POLICY_CPU,
            (
                DecisionReason.SELECTED_IMPLEMENTATION
                if gpu
                else DecisionReason.EXPLICIT_CPU
            ),
            "Exact private validation decision.",
        )
        clock.advance(0.01 if gpu else 0.1)
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=result_pipeline,
            execution_report=ExecutionReport(
                request.compute_request,
                _environment(),
                plan=_execution_plan(request, (decision,)),
                actual_decisions=(decision,),
            ),
        )

    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
    )
    current = (
        (source_id, compute_specs_for("input")[0].implementation_id),
        (otsu.id, cpu_spec.implementation_id),
    )
    proposed = (
        current[0],
        (otsu.id, gpu_spec.implementation_id),
    )

    validation = coordinator._validate_assignments(
        document,
        {source_id: SourcePayload(np.zeros((1000,), dtype=np.float32))},
        ComputeRequest("custom", allow_experimental=True),
        _environment(),
        pipeline,
        frozenset(pipeline.nodes),
        frozenset(),
        PipelineValidationRequest("identity", current, proposed),
        deadline=clock() + 30.0,
        cancelled=None,
    )

    assert not validation.parity_passed
    assert validation.synchronized
    assert validation.current_seconds == pytest.approx(0.1)
    assert validation.proposed_seconds == pytest.approx(0.01)
    assert execution_count >= 12
    assert len(validation.reviewable_deviations) == 1
    deviation = validation.reviewable_deviations[0]
    assert deviation.node_id == otsu.id
    assert deviation.differing_values == 1
    assert deviation.total_values == 1000
    assert deviation.differing_fraction == pytest.approx(0.001)


def test_pipeline_validation_stops_at_skipped_manual_barrier(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    median = pipeline.add_node("median_filter")
    threshold = pipeline.add_node("binary_threshold")
    metrics = pipeline.add_node("masked_colocalization_metrics")
    assert pipeline.connect(source_id, median.id).success
    assert pipeline.connect(source_id, threshold.id).success
    assert pipeline.connect(median.id, metrics.id, target_port=0).success
    assert pipeline.connect(source_id, metrics.id, target_port=1).success
    assert pipeline.connect(threshold.id, metrics.id, target_port=2).success
    document = serialize_workflow(pipeline)
    registry = ComputeRegistry()
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    clock = _ManualClock()
    decision_sets = []
    retained_sets = []

    def executor(request, **_kwargs):
        restored = deserialize_workflow(request.workflow)
        result_pipeline = PrototypePipeline()
        result_pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        schedule = result_pipeline.plan_execution(
            request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            target_node_ids=request.target_node_ids,
        )
        payload = request.source_payloads[source_id]
        result_pipeline.run(
            payload.data,
            input_metadata=payload.metadata,
            input_name=payload.name,
            source_payloads=request.source_payloads,
            dirty_node_ids=request.dirty_node_ids,
            manual_mode=MANUAL_RUN_SKIP,
            target_node_ids=request.target_node_ids,
            retain_node_ids=request.retain_node_ids,
            prune_unretained=request.prune_unretained,
        )
        decisions = []
        requested_gpu = (
            request.compute_request.preference_for(median.id).kind
            is NodePreferenceKind.IMPLEMENTATION
        )
        for node_id in result_pipeline.topological_order():
            node = result_pipeline.nodes[node_id]
            if node_id not in schedule.runnable_node_ids or not node.has_input:
                continue
            cpu_spec = compute_specs_for(node.operation_id)[0]
            selected = gpu_spec if node_id == median.id and requested_gpu else cpu_spec
            decisions.append(
                NodeExecutionDecision(
                    node_id,
                    node.operation_id,
                    request.compute_request.preference_for(node_id),
                    selected.runtime_id,
                    selected.implementation_library_id,
                    selected.implementation_id,
                    (
                        DecisionKind.SELECTED
                        if selected is gpu_spec
                        else DecisionKind.POLICY_CPU
                    ),
                    (
                        DecisionReason.SELECTED_IMPLEMENTATION
                        if selected is gpu_spec
                        else DecisionReason.EXPLICIT_CPU
                    ),
                    "Private validation executed only the runnable subgraph.",
                )
            )
        decision_sets.append(frozenset(item.node_id for item in decisions))
        retained_sets.append(frozenset(request.retain_node_ids))
        clock.advance(0.01 if requested_gpu else 0.1)
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=result_pipeline,
            execution_report=ExecutionReport(
                request.compute_request,
                _environment(),
                plan=_execution_plan(request, decisions),
                actual_decisions=tuple(decisions),
            ),
        )

    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        clock=clock,
        executor=executor,
    )
    current = tuple(
        (
            node_id,
            compute_specs_for(node.operation_id)[0].implementation_id,
        )
        for node_id, node in pipeline.nodes.items()
    )
    proposed = tuple(
        (
            node_id,
            gpu_spec.implementation_id if node_id == median.id else implementation_id,
        )
        for node_id, implementation_id in current
    )

    validation = coordinator._validate_assignments(
        document,
        {source_id: SourcePayload(np.arange(64, dtype=np.uint16).reshape(8, 8))},
        ComputeRequest("custom", allow_experimental=True),
        _environment(),
        pipeline,
        frozenset(pipeline.nodes),
        frozenset(),
        PipelineValidationRequest("identity", current, proposed),
        deadline=clock() + 30.0,
        cancelled=None,
    )

    assert validation.parity_passed
    assert decision_sets
    assert all(metrics.id not in node_ids for node_ids in decision_sets)
    assert all(metrics.id not in node_ids for node_ids in retained_sets)
    assert all(median.id in node_ids for node_ids in retained_sets)


def test_optimizer_private_runs_can_explicitly_include_manual_measurements():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    assert pipeline.connect(source_id, threshold.id).success
    assert pipeline.connect(threshold.id, labels.id).success
    assert pipeline.connect(labels.id, measurements.id).success
    safe_ids = frozenset(pipeline.nodes)

    skipped = _optimizer_validation_node_ids(pipeline, safe_ids)
    selected = _optimizer_validation_node_ids(
        pipeline,
        safe_ids,
        manual_node_ids=frozenset({measurements.id}),
    )

    assert measurements.id not in skipped
    assert {source_id, threshold.id, labels.id, measurements.id} <= selected


@pytest.mark.parametrize(
    ("failure_mode", "expected_code"),
    [
        ("planning", "proposed_parity_planning_assignment_mismatch"),
        ("device-plan", "proposed_parity_device_segment_mismatch"),
        ("actual", "proposed_parity_actual_assignment_mismatch"),
        ("cleanup", "current_parity_cleanup_failed"),
    ],
)
def test_pipeline_validation_rejects_untrustworthy_execution_report(
    tmp_path,
    failure_mode,
    expected_code,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    median = pipeline.add_node("median_filter")
    assert pipeline.connect(source_id, median.id).success
    document = serialize_workflow(pipeline)
    registry = ComputeRegistry()
    cpu_spec = compute_specs_for("median_filter")[0]
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]

    def executor(request, **_kwargs):
        restored = deserialize_workflow(request.workflow)
        result_pipeline = PrototypePipeline()
        result_pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        value = np.zeros((8, 8), dtype=np.uint16)
        for node_id in (source_id, median.id):
            result_pipeline.outputs[node_id] = value
            result_pipeline.node_outputs[node_id] = [value]
            result_pipeline.completed_node_ids.add(node_id)
        requested_gpu = (
            request.compute_request.preference_for(median.id).value
            == gpu_spec.implementation_id
        )
        requested_spec = gpu_spec if requested_gpu else cpu_spec
        planned_spec = (
            cpu_spec if failure_mode == "planning" and requested_gpu else requested_spec
        )
        actual_spec = (
            cpu_spec if failure_mode == "actual" and requested_gpu else planned_spec
        )

        def make_decision(spec):
            selected_gpu = spec.runtime_id != "cpu-numpy"
            return NodeExecutionDecision(
                median.id,
                "median_filter",
                request.compute_request.preference_for(median.id),
                spec.runtime_id,
                spec.implementation_library_id,
                spec.implementation_id,
                DecisionKind.SELECTED if selected_gpu else DecisionKind.POLICY_CPU,
                (
                    DecisionReason.SELECTED_IMPLEMENTATION
                    if selected_gpu
                    else DecisionReason.EXPLICIT_CPU
                ),
                "Test executor returned staged validation provenance.",
            )

        planned = make_decision(planned_spec)
        actual = make_decision(actual_spec)
        plan = _execution_plan(request, (planned,))
        if failure_mode == "device-plan" and requested_gpu:
            plan = ExecutionPlan(
                request.compute_request.fingerprint,
                _environment().fingerprint,
                (),
                (planned,),
            )
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=result_pipeline,
            execution_report=ExecutionReport(
                request.compute_request,
                _environment(),
                plan=plan,
                actual_decisions=(actual,),
                cleanup_succeeded=failure_mode != "cleanup",
            ),
        )

    coordinator = ApplicationPipelineOptimizerCoordinator(
        registry,
        tmp_path / "benchmarks.json",
        executor=executor,
    )
    current = (
        (source_id, compute_specs_for("input")[0].implementation_id),
        (median.id, cpu_spec.implementation_id),
    )
    proposed = (
        current[0],
        (median.id, gpu_spec.implementation_id),
    )

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as error:
        coordinator._validate_assignments(
            document,
            {source_id: SourcePayload(np.zeros((8, 8), dtype=np.uint16))},
            ComputeRequest("custom", allow_experimental=True),
            _environment(),
            pipeline,
            frozenset(pipeline.nodes),
            frozenset(),
            PipelineValidationRequest("identity", current, proposed),
            deadline=coordinator.clock() + 30.0,
            cancelled=None,
        )

    assert error.value.reasons[0].code == expected_code


def test_pipeline_exact_shortcut_preserves_signed_zero_parity_policy():
    reference = np.array([0.0, -0.0, 2.0], dtype=np.float32)
    candidate = np.array([-0.0, -0.0, 2.0], dtype=np.float32)

    passed, detail = _pipeline_output_parity(
        "median_filter",
        reference,
        candidate,
    )

    assert not passed
    assert "signed_zero_mismatches=1" in detail
    assert (
        _reviewable_pipeline_deviation(
            "median_1",
            "median_filter",
            0,
            reference,
            candidate,
            input_peak=None,
            parity_detail=detail,
        )
        is None
    )


def test_small_discrete_difference_is_quantified_for_explicit_review():
    reference = np.zeros((1000,), dtype=bool)
    candidate = reference.copy()
    candidate[17] = True

    deviation = _reviewable_pipeline_deviation(
        "otsu_1",
        "otsu_threshold",
        0,
        reference,
        candidate,
        input_peak=None,
        parity_detail="bitwise mask mismatch",
    )

    assert deviation is not None
    assert deviation.metric.value == "differing-value-fraction"
    assert deviation.differing_values == 1
    assert deviation.total_values == 1000
    assert deviation.differing_fraction == pytest.approx(0.001)
    assert deviation.measured_difference == pytest.approx(0.001)


def test_small_float_difference_is_symmetric_and_bounded_for_review():
    reference = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    candidate = np.nextafter(reference, np.float32(np.inf))

    deviation = _reviewable_pipeline_deviation(
        "gaussian_1",
        "gaussian_blur",
        0,
        reference,
        candidate,
        input_peak=1.0,
        parity_detail="registered Gaussian tolerance did not pass",
    )

    assert deviation is not None
    assert deviation.metric.value == "normalized-rmse"
    assert deviation.differing_values > 900
    assert deviation.normalized_root_mean_square_error < 1e-6
    assert deviation.normalized_maximum_absolute_error < 1e-6


def test_reviewable_difference_uses_bounded_chunks_for_strided_outputs(
    monkeypatch,
):
    reference_storage = np.zeros((1024, 1024), dtype=bool)
    candidate_storage = reference_storage.copy()
    candidate_storage[17, 34] = True
    reference = reference_storage[:, ::2]
    candidate = candidate_storage[:, ::2]
    observed_staging_sizes: list[int] = []
    original_ascontiguousarray = np.ascontiguousarray

    def observed_ascontiguousarray(values, *args, **kwargs):
        observed_staging_sizes.append(int(np.asarray(values).size))
        return original_ascontiguousarray(values, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", observed_ascontiguousarray)

    deviation = _reviewable_pipeline_deviation(
        "otsu_1",
        "otsu_threshold",
        0,
        reference,
        candidate,
        input_peak=None,
        parity_detail="bitwise mask mismatch",
    )

    assert deviation is not None
    assert deviation.differing_values == 1
    assert observed_staging_sizes
    assert max(observed_staging_sizes) < reference.size


def test_exact_parity_shortcut_uses_bounded_chunks_for_strided_outputs(
    monkeypatch,
):
    storage = np.arange(1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    reference = storage[:, ::2]
    candidate = storage.copy()[:, ::2]
    observed_staging_sizes: list[int] = []
    original_ascontiguousarray = np.ascontiguousarray

    def observed_ascontiguousarray(values, *args, **kwargs):
        observed_staging_sizes.append(int(np.asarray(values).size))
        return original_ascontiguousarray(values, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", observed_ascontiguousarray)

    passed, detail = _pipeline_output_parity(
        "gaussian_blur",
        reference,
        candidate,
    )

    assert passed, detail
    assert observed_staging_sizes
    assert max(observed_staging_sizes) < reference.size


def test_pipeline_input_peak_uses_bounded_chunks(monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    storage = np.linspace(-3.0, 5.0, 1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    values = storage[:, ::2]
    expected_peak = float(np.max(np.abs(values.astype(np.float64))))
    pipeline.outputs["input"] = values
    observed_staging_sizes: list[int] = []
    original_asarray = np.asarray

    def observed_asarray(value, *args, **kwargs):
        converted = original_asarray(value, *args, **kwargs)
        if converted is not values:
            observed_staging_sizes.append(int(converted.size))
        return converted

    monkeypatch.setattr(np, "asarray", observed_asarray)

    peak = _pipeline_input_peak(pipeline, gaussian.id)

    assert peak == expected_peak
    assert not observed_staging_sizes or max(observed_staging_sizes) < values.size


def test_large_or_structural_difference_is_not_reviewable():
    reference = np.zeros((1000,), dtype=bool)
    candidate = reference.copy()
    candidate[:2] = True
    assert (
        _reviewable_pipeline_deviation(
            "remove_1",
            "remove_small_objects",
            0,
            reference,
            candidate,
            input_peak=None,
            parity_detail="mask mismatch",
        )
        is None
    )
    assert (
        _reviewable_pipeline_deviation(
            "remove_1",
            "remove_small_objects",
            0,
            reference.reshape(10, 100),
            candidate,
            input_peak=None,
            parity_detail="shape mismatch",
        )
        is None
    )


def test_pipeline_background_parity_uses_benchmark_input_scale():
    reference = np.full((8, 8), 1_000.0, dtype=np.float32)
    candidate = reference + np.float32(0.001)

    without_input_scale, _detail = _pipeline_output_parity(
        "subtract_background",
        reference,
        candidate,
    )
    with_input_scale, detail = _pipeline_output_parity(
        "subtract_background",
        reference,
        candidate,
        input_peak=65_535.0,
    )

    assert not without_input_scale
    assert with_input_scale
    assert "max_abs=" in detail
