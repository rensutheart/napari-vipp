from __future__ import annotations

import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import (
    BenchmarkCandidateResult,
    BenchmarkRecord,
    BenchmarkRecordKey,
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionReport,
    MemoryEstimate,
    MemoryTopology,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
)
from napari_vipp.core.compute_benchmark import BenchmarkBudgetExceeded
from napari_vipp.core.compute_benchmark_adapter import (
    workload_from_prepared_node_call,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    NodeBenchmarkPhase,
    NodeBenchmarkProgress,
)
from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationEvidenceIncomplete,
    PipelineOptimizationNotBeneficial,
    PipelineValidationRequest,
    PipelineValidationWinner,
)
from napari_vipp.core.compute_pipeline_optimizer_coordinator import (
    ApplicationPipelineOptimizerCoordinator,
    PipelineOptimizerPhase,
    _build_optimizer_graph,
    _pipeline_output_parity,
    fingerprint_pipeline_optimizer_sources,
    probe_pipeline_optimizer_environment,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    RuntimeMemorySnapshot,
)
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import PipelineRunResult
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
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


class _NodeBenchmarker:
    def __init__(self, environment: ComputeEnvironment, registry: ComputeRegistry):
        self.environment = environment
        self.registry = registry
        self.observed_budgets: list[float] = []

    def prepare(self, pipeline, node_id, **kwargs):
        self.observed_budgets.append(float(kwargs["time_budget_seconds"]))
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
        self.detached_source_arrays: list[np.ndarray] = []

    def __call__(self, request, **_kwargs):
        restored = deserialize_workflow(request.workflow)
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        targets = frozenset(request.target_node_ids or ())
        self.target_sets.append(targets)
        assert self.writer_node_id not in targets
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
            actual_decisions=(decision,),
        )
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=pipeline,
            execution_report=report,
        )


def _writer_workflow():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source_id = next(iter(pipeline.nodes))
    median = pipeline.add_node("median_filter")
    median.params["size"] = 3
    writer = pipeline.add_node("save_output")
    assert pipeline.connect(source_id, median.id).success
    assert pipeline.connect(median.id, writer.id).success
    return pipeline, source_id, median.id, writer.id


def test_application_optimizer_is_private_writer_free_and_evidence_gated(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("selective"))
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

    result = coordinator.optimize(
        document,
        {source_id: SourcePayload(values, name="private-source")},
        ComputeRequest("selective", allow_experimental=True),
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
    operation = next(
        item for item in progress if item.measurement_phase == "paired_warm"
    )
    assert operation.phase is PipelineOptimizerPhase.BENCHMARKING
    assert (operation.completed, operation.total) == (2, 7)
    assert (operation.operation_completed, operation.operation_total) == (2, 3)
    assert operation.node_id == median_id
    assert operation.node_title == pipeline.nodes[median_id].title
    assert operation.implementation_id == gpu_spec.implementation_id
    assert operation.operation_message == (
        "Measuring paired warm round 3 of 3 for the GPU."
    )
    np.testing.assert_array_equal(values, original)
    assert all(
        not np.shares_memory(item, values)
        for item in executor.detached_source_arrays
    )
    assert fingerprint_pipeline_optimizer_sources(
        document,
        {source_id: SourcePayload(values, name="private-source")},
    ) == result.identity.source_fingerprint
    values[0, 0] += 1
    assert fingerprint_pipeline_optimizer_sources(
        document,
        {source_id: SourcePayload(values, name="private-source")},
    ) != result.identity.source_fingerprint


def test_node_benchmark_timeout_reports_stage_progress_and_no_optimality(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("selective"))
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
    request = ComputeRequest("selective", allow_experimental=True)
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


def test_close_pipeline_validation_uses_full_fifteen_rounds(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("selective"))
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

    with pytest.raises(PipelineOptimizationNotBeneficial):
        coordinator.optimize(
            document,
            {
                source_id: SourcePayload(
                    np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
                )
            },
            ComputeRequest("selective", allow_experimental=True),
            time_budget_seconds=20.0,
        )

    assert len(executor.target_sets) == 33


def test_decisive_current_pipeline_win_stops_after_five_rounds(
    tmp_path,
    monkeypatch,
):
    pipeline, source_id, median_id, writer_id = _writer_workflow()
    document = serialize_workflow(pipeline, compute_request=ComputeRequest("selective"))
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
        {
            source_id: SourcePayload(
                np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
            )
        },
        ComputeRequest("selective", allow_experimental=True),
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


def test_locked_gpu_graph_node_needs_no_comparative_benchmark_record():
    pipeline, _source_id, median_id, _writer_id = _writer_workflow()
    pipeline.run(np.arange(64, dtype=np.uint16).reshape(8, 8))
    registry = ComputeRegistry()
    gpu_spec = registry.implementations_for_operation(
        "median_filter",
        allow_experimental=True,
    )[0]
    request = ComputeRequest(
        "selective",
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


def test_application_optimizer_refuses_non_selective_and_missing_sources(tmp_path):
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

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as non_selective:
        coordinator.optimize(document, {}, ComputeRequest("cpu"))
    assert non_selective.value.reasons[0].code == "selective_required"

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as missing:
        coordinator.optimize(document, {}, ComputeRequest("selective"))
    assert missing.value.reasons[0].code == "source_identity_incomplete"


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
    request = ComputeRequest("selective", allow_experimental=True)

    actual = probe_pipeline_optimizer_environment(registry, document, request)

    assert actual.fingerprint == environment.fingerprint
    assert len(calls) == 1
    assert {item.operation_id for item in calls[0][1]} == {"median_filter"}

    with pytest.raises(PipelineOptimizationEvidenceIncomplete) as unavailable:
        probe_pipeline_optimizer_environment(
            registry,
            document,
            ComputeRequest("selective", allow_experimental=False),
        )
    assert unavailable.value.reasons[0].code == "no_gpu_candidates"


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
        gaussian_spec = (
            gpu_spec if gpu else compute_specs_for("gaussian_blur")[0]
        )
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
        ComputeRequest("selective", allow_experimental=True),
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


@pytest.mark.parametrize(
    ("failure_mode", "expected_code"),
    [
        ("assignment", "proposed_parity_assignment_mismatch"),
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
        actual_spec = (
            cpu_spec
            if failure_mode == "assignment" or not requested_gpu
            else gpu_spec
        )
        ignored = NodeExecutionDecision(
            median.id,
            "median_filter",
            request.compute_request.preference_for(median.id),
            actual_spec.runtime_id,
            actual_spec.implementation_library_id,
            actual_spec.implementation_id,
            DecisionKind.SELECTED if requested_gpu else DecisionKind.POLICY_CPU,
            (
                DecisionReason.SELECTED_IMPLEMENTATION
                if requested_gpu
                else DecisionReason.EXPLICIT_CPU
            ),
            "Test executor returned untrustworthy validation provenance.",
        )
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=result_pipeline,
            execution_report=ExecutionReport(
                request.compute_request,
                _environment(),
                actual_decisions=(ignored,),
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
            ComputeRequest("selective", allow_experimental=True),
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
