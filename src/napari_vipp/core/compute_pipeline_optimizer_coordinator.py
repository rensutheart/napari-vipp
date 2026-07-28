"""Qt-free application transaction for whole-pipeline compute optimization.

The lower-level :mod:`compute_pipeline_optimizer` module deliberately accepts
only already-qualified evidence.  This module builds that evidence from one
detached workflow and immutable source snapshot.  It executes only pure/source
nodes, keeps benchmark and validation work private, and publishes a proposal
only after operation-specific parity and paired end-to-end timing succeed.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

import numpy as np

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    FallbackPolicy,
    NodeComputePreference,
    NodePreferenceKind,
    canonical_digest,
)
from napari_vipp.core.compute_benchmark import (
    ADAPTIVE_WARM_ROUNDS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    MINIMUM_WARM_ROUNDS,
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    paired_bootstrap_speedup,
)
from napari_vipp.core.compute_benchmark_adapter import (
    operation_parity,
    workload_from_prepared_node_call,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    ApplicationNodeBenchmarkCoordinator,
    NodeBenchmarkUnavailable,
)
from napari_vipp.core.compute_pipeline_optimizer import (
    DirectionalTransferProfile,
    EvidenceRefusal,
    PipelineAssignmentValidation,
    PipelineNodeBenchmarkEvidence,
    PipelineOptimizationCancelled,
    PipelineOptimizationCandidate,
    PipelineOptimizationCoordinator,
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationEdge,
    PipelineOptimizationEvidenceIncomplete,
    PipelineOptimizationIdentity,
    PipelineOptimizationNode,
    PipelineOptimizationProposal,
    PipelineValidationRequest,
)
from napari_vipp.core.compute_planning import (
    plan_compute_decisions,
    probe_compute_environment,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import compute_specs_for
from napari_vipp.core.execution import (
    PipelineRunRequest,
    PipelineRunResult,
    execute_pipeline_request,
)
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.workflow import deserialize_workflow

ProgressCallback = Callable[["PipelineOptimizerProgress"], None]
CancelCallback = Callable[[], bool]
PipelineExecutor = Callable[..., PipelineRunResult]

_CPU_RUNTIME_ID = "cpu-numpy"
_TRANSFER_ROUNDS = 3
_VALIDATION_ROUNDS = 7
_TRANSFER_SMALL_BYTES = 256 * 1024
_TRANSFER_LARGE_BYTES = 16 * 1024**2
_SOURCE_HASH_CHUNK_BYTES = 8 * 1024**2
_DEFAULT_ACCELERATOR_RESERVE_BYTES = 512 * 1024**2


class PipelineOptimizerPhase(StrEnum):
    PREPARING = "preparing"
    BASELINE = "baseline"
    BENCHMARKING = "benchmarking"
    TRANSFERS = "transfers"
    SOLVING = "solving"
    VALIDATING = "validating"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class PipelineOptimizerProgress:
    phase: PipelineOptimizerPhase | str
    completed: int
    total: int
    message: str

    def __post_init__(self) -> None:
        phase = (
            self.phase
            if isinstance(self.phase, PipelineOptimizerPhase)
            else PipelineOptimizerPhase(str(self.phase).strip().lower())
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.completed, self.total)
        ):
            raise ValueError("optimizer progress values must be non-negative")
        if self.total < 1 or self.completed > self.total:
            raise ValueError("optimizer progress must fit inside its total")
        message = str(self.message).strip()
        if not message:
            raise ValueError("optimizer progress message must not be empty")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class ApplicationPipelineOptimizationResult:
    """Proposal plus the exact machine-local evidence used to produce it."""

    proposal: PipelineOptimizationProposal
    identity: PipelineOptimizationIdentity
    environment: ComputeEnvironment
    transfer_profile: DirectionalTransferProfile
    evidence: Mapping[str, PipelineNodeBenchmarkEvidence]
    benchmarked_node_ids: tuple[str, ...]
    cpu_only_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.proposal.identity_digest != self.identity.digest:
            raise ValueError("proposal and application identity do not match")
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(sorted(self.evidence.items()))),
        )
        object.__setattr__(
            self, "benchmarked_node_ids", tuple(self.benchmarked_node_ids)
        )
        object.__setattr__(self, "cpu_only_node_ids", tuple(self.cpu_only_node_ids))


@dataclass(frozen=True, slots=True)
class _DetachedSources:
    payloads: Mapping[str, SourcePayload]
    fingerprints: Mapping[str, str]


class _CallbackCancelEvent:
    """The execution service only needs the ``Event.is_set`` surface."""

    def __init__(self, callback: Callable[[], bool]) -> None:
        self._callback = callback

    def is_set(self) -> bool:
        return bool(self._callback())


class ApplicationPipelineOptimizerCoordinator:
    """Acquire, solve, and validate one exact Selective pipeline assignment."""

    def __init__(
        self,
        registry: ComputeRegistry,
        benchmark_store_path: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
        executor: PipelineExecutor = execute_pipeline_request,
        optimizer: PipelineOptimizationCoordinator | None = None,
        node_benchmarker: ApplicationNodeBenchmarkCoordinator | None = None,
    ) -> None:
        if not isinstance(registry, ComputeRegistry):
            raise TypeError("registry must be a ComputeRegistry")
        if not callable(clock) or not callable(executor):
            raise TypeError("clock and executor must be callable")
        self.registry = registry
        self.benchmark_store_path = (
            Path(benchmark_store_path).expanduser().resolve(strict=False)
        )
        self.clock = clock
        self.executor = executor
        self.optimizer = optimizer or PipelineOptimizationCoordinator(clock=clock)
        self.node_benchmarker = node_benchmarker or ApplicationNodeBenchmarkCoordinator(
            registry,
            self.benchmark_store_path,
            clock=clock,
        )
        self._run_ids = iter(range(1, 2**63))

    def optimize(
        self,
        workflow: Mapping[str, object],
        source_payloads: Mapping[str, SourcePayload],
        compute_request: ComputeRequest,
        retain_node_ids: Sequence[str] | frozenset[str] = (),
        *,
        time_budget_seconds: float = 120.0,
        max_assignments: int = 100_000,
        override_authored: bool = False,
        cancelled: CancelCallback | None = None,
        progress: ProgressCallback | None = None,
    ) -> ApplicationPipelineOptimizationResult:
        """Return a reviewed proposal or raise one typed, non-mutating refusal."""

        if not isinstance(compute_request, ComputeRequest):
            raise TypeError("compute_request must be a ComputeRequest")
        if compute_request.mode is not ComputeMode.SELECTIVE:
            _refuse("selective_required", "Choose Selective compute policy.")
        if compute_request.accelerator_memory_cap_bytes == 0:
            _refuse(
                "accelerator_memory_cap_invalid",
                "Pipeline optimization requires a positive accelerator memory cap.",
            )
        if not isinstance(workflow, Mapping):
            raise TypeError("workflow must be a detached workflow mapping")
        if not isinstance(source_payloads, Mapping):
            raise TypeError("source_payloads must be a mapping")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        budget = _positive_duration(time_budget_seconds, "time_budget_seconds")
        if isinstance(max_assignments, bool) or not isinstance(max_assignments, int):
            raise ValueError("max_assignments must be a positive integer")
        if max_assignments < 1:
            raise ValueError("max_assignments must be a positive integer")
        started = _read_clock(self.clock)
        deadline = started + budget

        def check_abort() -> None:
            _check_abort(self.clock, deadline, cancelled)

        def is_cancelled_or_expired() -> bool:
            return bool(
                (cancelled is not None and cancelled())
                or _read_clock(self.clock) >= deadline
            )

        _emit(
            progress,
            PipelineOptimizerPhase.PREPARING,
            0,
            6,
            "Detaching the workflow and source data.",
        )
        check_abort()
        document = deepcopy(dict(workflow))
        restored = deserialize_workflow(document)
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            restored["nodes"],
            restored["connections"],
            restored.get("output_tunnels", ()),
        )
        safe_ids, unsafe_ids = _writer_free_node_ids(pipeline)
        retained = _normalized_node_ids(retain_node_ids)
        unknown_retained = retained - set(pipeline.nodes)
        if unknown_retained:
            _refuse(
                "retain_identity_invalid",
                "Retained outputs reference unknown nodes: "
                + ", ".join(sorted(unknown_retained)),
            )
        blocked_retained = retained & unsafe_ids
        if blocked_retained:
            _refuse(
                "writer_retention_unsafe",
                "Private optimization cannot execute or retain writer/side-effect "
                "nodes: " + ", ".join(sorted(blocked_retained)),
            )
        sources = _detach_source_payloads(
            pipeline,
            source_payloads,
            safe_ids,
            check_abort=check_abort,
        )
        check_abort()

        environment, accelerator_runtime_id = (
            _probe_optimizer_environment_for_pipeline(
                self.registry,
                pipeline,
                safe_ids,
                compute_request,
            )
        )
        check_abort()

        _emit(
            progress,
            PipelineOptimizerPhase.BASELINE,
            1,
            6,
            "Running a private writer-free baseline.",
        )
        baseline = self._execute(
            document,
            sources.payloads,
            compute_request,
            environment,
            target_node_ids=frozenset(safe_ids),
            retain_node_ids=frozenset(safe_ids),
            prune_unretained=False,
            cancel_callback=is_cancelled_or_expired,
        )
        check_abort()
        baseline_pipeline = _successful_pipeline(baseline, "baseline")
        decisions = {
            item.node_id: item
            for item in (
                baseline.execution_report.actual_decisions
                if baseline.execution_report is not None
                else ()
            )
        }
        if baseline.execution_report is None:
            _refuse(
                "baseline_provenance_missing",
                "The private baseline did not produce exact compute provenance.",
            )
        if not baseline.execution_report.cleanup_succeeded:
            _refuse(
                "baseline_cleanup_failed",
                "The private baseline did not release its accelerator resources; "
                "analysis cannot continue with a potentially poisoned VRAM state.",
            )
        if baseline.execution_report.environment.fingerprint != environment.fingerprint:
            _refuse(
                "environment_identity_changed",
                "The runtime environment changed during the private baseline.",
            )

        eligible_ids = tuple(
            node_id
            for node_id in baseline_pipeline.topological_order()
            if node_id in safe_ids
            and baseline_pipeline.nodes[node_id].has_input
            and self.registry.implementations_for_operation(
                baseline_pipeline.nodes[node_id].operation_id,
                allow_experimental=compute_request.allow_experimental,
            )
        )
        total_steps = len(eligible_ids) + 6
        evidence_records: dict[str, object] = {}
        benchmark_plans: dict[str, object] = {}
        cpu_only: set[str] = set(safe_ids) - set(eligible_ids)
        for index, node_id in enumerate(eligible_ids, start=1):
            check_abort()
            _emit(
                progress,
                PipelineOptimizerPhase.BENCHMARKING,
                1 + index,
                total_steps,
                f"Benchmarking {baseline_pipeline.nodes[node_id].title} "
                f"({index}/{len(eligible_ids)}).",
            )
            remaining = deadline - _read_clock(self.clock)
            # Keep a bounded share for transfer measurement, solving, and paired
            # validation.  The selected-node service enforces this share itself.
            node_budget = remaining / max(2, len(eligible_ids) - index + 2)
            if node_budget <= 0:
                check_abort()
            try:
                plan = self.node_benchmarker.prepare(
                    baseline_pipeline,
                    node_id,
                    environment=environment,
                    device_id=(compute_request.device_id or environment.device_id),
                    memory_limit_bytes=compute_request.accelerator_memory_cap_bytes,
                    safety_reserve_bytes=(
                        compute_request.accelerator_safety_reserve_bytes
                    ),
                    warm_rounds=MINIMUM_WARM_ROUNDS,
                    max_warm_rounds=ADAPTIVE_WARM_ROUNDS[0],
                    time_budget_seconds=node_budget,
                    allow_experimental=compute_request.allow_experimental,
                    paired_bootstrap_samples=DEFAULT_BOOTSTRAP_SAMPLES,
                    paired_bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
                    cancelled=is_cancelled_or_expired,
                )
                result = self.node_benchmarker.run(
                    plan,
                    cancelled=is_cancelled_or_expired,
                )
            except NodeBenchmarkUnavailable:
                cpu_only.add(node_id)
                continue
            except BenchmarkCancelled as exc:
                check_abort()
                raise PipelineOptimizationCancelled(str(exc)) from exc
            except BenchmarkBudgetExceeded as exc:
                raise PipelineOptimizationDeadlineExceeded(
                    "Pipeline optimization exhausted its node benchmark budget."
                ) from exc
            if result.plan.environment.fingerprint != environment.fingerprint:
                _refuse(
                    "benchmark_environment_changed",
                    "Node benchmark environment identity changed during analysis.",
                    node_id,
                )
            evidence_records[node_id] = result.record
            benchmark_plans[node_id] = result.plan

        check_abort()
        graph_nodes, graph_edges, workload_fingerprints = _build_optimizer_graph(
            baseline_pipeline,
            safe_ids,
            retained,
            compute_request,
            decisions,
            evidence_records,
            benchmark_plans,
            check_abort=check_abort,
        )
        if not evidence_records:
            _refuse(
                "no_gpu_workload_eligible",
                "No node has a parity-testable GPU implementation for the "
                "current data.",
            )
        identity = PipelineOptimizationIdentity(
            pipeline_fingerprint=canonical_digest(document),
            source_fingerprint=canonical_digest(dict(sources.fingerprints)),
            topology_fingerprint=_topology_fingerprint(pipeline),
            cache_retention_fingerprint=canonical_digest(sorted(retained)),
            environment_fingerprint=environment.fingerprint,
            workload_fingerprints=workload_fingerprints,
        )
        evidence = {
            node_id: PipelineNodeBenchmarkEvidence(
                node_id,
                identity.digest,
                record,
            )
            for node_id, record in evidence_records.items()
        }

        _emit(
            progress,
            PipelineOptimizerPhase.TRANSFERS,
            total_steps - 3,
            total_steps,
            "Measuring synchronized host-to-GPU and GPU-to-host transfers.",
        )
        transfer_profile = _measure_directional_transfers(
            self.registry,
            accelerator_runtime_id,
            compute_request.device_id or environment.device_id,
            environment,
            identity,
            baseline_pipeline,
            tuple(evidence),
            compute_request,
            clock=self.clock,
            check_abort=check_abort,
        )

        _emit(
            progress,
            PipelineOptimizerPhase.SOLVING,
            total_steps - 2,
            total_steps,
            "Solving the bounded whole-pipeline assignment.",
        )

        def validate(
            validation_request: PipelineValidationRequest,
        ) -> PipelineAssignmentValidation:
            _emit(
                progress,
                PipelineOptimizerPhase.VALIDATING,
                total_steps - 1,
                total_steps,
                "Validating parity, synchronization, and paired pipeline timing.",
            )
            return self._validate_assignments(
                document,
                sources.payloads,
                compute_request,
                environment,
                baseline_pipeline,
                safe_ids,
                retained,
                validation_request,
                deadline=deadline,
                cancelled=cancelled,
            )

        proposal = self.optimizer.optimize(
            compute_request,
            identity,
            graph_nodes,
            graph_edges,
            evidence,
            transfer_profile,
            validate,
            deadline=deadline,
            max_assignments=max_assignments,
            override_authored=override_authored,
            cancelled=cancelled,
        )
        check_abort()
        _emit(
            progress,
            PipelineOptimizerPhase.COMPLETE,
            total_steps,
            total_steps,
            "Pipeline optimization completed; review the proposal before applying it.",
        )
        return ApplicationPipelineOptimizationResult(
            proposal,
            identity,
            environment,
            transfer_profile,
            evidence,
            tuple(evidence),
            tuple(sorted(cpu_only)),
        )

    def _execute(
        self,
        workflow: dict,
        source_payloads: Mapping[str, SourcePayload],
        request: ComputeRequest,
        environment: ComputeEnvironment,
        *,
        target_node_ids: frozenset[str],
        retain_node_ids: frozenset[str],
        prune_unretained: bool,
        cancel_callback: Callable[[], bool],
    ) -> PipelineRunResult:
        first = next(iter(source_payloads.values()))

        def planner(planning_request, workloads, **kwargs):
            return plan_compute_decisions(
                planning_request,
                workloads,
                registry=kwargs.get("registry") or self.registry,
                environment=environment,
                array_facts=kwargs.get("array_facts"),
                performance_evidence=kwargs.get("performance_evidence"),
            )

        run_request = PipelineRunRequest(
            run_id=next(self._run_ids),
            workflow=deepcopy(workflow),
            input_data=first.data,
            input_metadata=first.metadata,
            input_name=first.name,
            source_payloads=dict(source_payloads),
            compute_request=request,
            dirty_node_ids=frozenset(target_node_ids),
            target_node_ids=target_node_ids,
            retain_node_ids=retain_node_ids,
            prune_unretained=prune_unretained,
            cancel_event=_CallbackCancelEvent(cancel_callback),
        )
        return self.executor(
            run_request,
            compute_registry=self.registry,
            compute_planner=planner,
        )

    def _validate_assignments(
        self,
        workflow: dict,
        source_payloads: Mapping[str, SourcePayload],
        base_request: ComputeRequest,
        environment: ComputeEnvironment,
        baseline_pipeline: PrototypePipeline,
        safe_ids: frozenset[str],
        retained: frozenset[str],
        validation: PipelineValidationRequest,
        *,
        deadline: float,
        cancelled: CancelCallback | None,
    ) -> PipelineAssignmentValidation:
        def check_abort() -> None:
            _check_abort(self.clock, deadline, cancelled)

        def cancel_or_expired() -> bool:
            return bool(
                (cancelled is not None and cancelled())
                or _read_clock(self.clock) >= deadline
            )

        current_map = dict(validation.current_assignment)
        proposed_map = dict(validation.proposed_assignment)
        changed = tuple(
            node_id
            for node_id in safe_ids
            if current_map.get(node_id) != proposed_map.get(node_id)
            and baseline_pipeline.nodes[node_id].has_input
        )
        if not changed:
            _refuse(
                "validation_assignment_unchanged",
                "The proposed exact assignment does not change an executable node.",
            )
        observable_boundaries = _observable_pipeline_boundaries(
            baseline_pipeline,
            safe_ids,
            retained,
        )
        affected = baseline_pipeline.descendants_inclusive(changed)
        parity_targets = frozenset(
            set(changed) | (set(observable_boundaries) & affected)
        )
        parity_retain = frozenset(
            set(observable_boundaries) | set(parity_targets)
        )
        current_request = _exact_assignment_request(
            base_request, baseline_pipeline, current_map
        )
        proposed_request = _exact_assignment_request(
            base_request,
            baseline_pipeline,
            proposed_map,
        )
        current_parity = self._execute(
            workflow,
            source_payloads,
            current_request,
            environment,
            target_node_ids=safe_ids,
            retain_node_ids=parity_retain,
            prune_unretained=True,
            cancel_callback=cancel_or_expired,
        )
        check_abort()
        proposed_parity = self._execute(
            workflow,
            source_payloads,
            proposed_request,
            environment,
            target_node_ids=safe_ids,
            retain_node_ids=parity_retain,
            prune_unretained=True,
            cancel_callback=cancel_or_expired,
        )
        check_abort()
        current_pipeline = _successful_exact_pipeline(
            current_parity,
            "current parity",
            expected_request=current_request,
            expected_assignment=current_map,
            reference_pipeline=baseline_pipeline,
            node_ids=safe_ids,
            environment=environment,
        )
        proposed_pipeline = _successful_exact_pipeline(
            proposed_parity,
            "proposed parity",
            expected_request=proposed_request,
            expected_assignment=proposed_map,
            reference_pipeline=baseline_pipeline,
            node_ids=safe_ids,
            environment=environment,
        )
        for node_id in baseline_pipeline.topological_order():
            if node_id not in parity_targets:
                continue
            operation_id = baseline_pipeline.nodes[node_id].operation_id
            input_peak = _pipeline_input_peak(baseline_pipeline, node_id)
            current_outputs = tuple(current_pipeline.node_outputs.get(node_id, ()))
            proposed_outputs = tuple(proposed_pipeline.node_outputs.get(node_id, ()))
            if not current_outputs or len(current_outputs) != len(proposed_outputs):
                _refuse(
                    "pipeline_parity_output_missing",
                    "Whole-pipeline parity requires every retained output port.",
                    node_id,
                )
            for port_index, (current_output, proposed_output) in enumerate(
                zip(current_outputs, proposed_outputs, strict=True)
            ):
                passed, detail = _pipeline_output_parity(
                    operation_id,
                    current_output,
                    proposed_output,
                    input_peak=input_peak,
                )
                if not passed:
                    return PipelineAssignmentValidation(
                        validation.identity_digest,
                        validation.current_assignment,
                        validation.proposed_assignment,
                        False,
                        True,
                        0.0,
                        0.0,
                        0.0,
                        (
                            f"{baseline_pipeline.nodes[node_id].title} output "
                            f"{port_index + 1} failed whole-pipeline parity: "
                            f"{detail}"
                        ),
                    )

        # Parity is deliberately complete before any timings are accepted.
        current_times: list[float] = []
        proposed_times: list[float] = []
        timing_retain = observable_boundaries
        for round_index in range(_VALIDATION_ROUNDS):
            check_abort()
            order = (
                (("current", current_request), ("proposed", proposed_request))
                if round_index % 2 == 0
                else (("proposed", proposed_request), ("current", current_request))
            )
            measured: dict[str, float] = {}
            for label, request in order:
                check_abort()
                began = _read_clock(self.clock)
                result = self._execute(
                    workflow,
                    source_payloads,
                    request,
                    environment,
                    target_node_ids=safe_ids,
                    retain_node_ids=timing_retain,
                    prune_unretained=True,
                    cancel_callback=cancel_or_expired,
                )
                check_abort()
                _successful_exact_pipeline(
                    result,
                    f"{label} timing",
                    expected_request=request,
                    expected_assignment=(
                        current_map if label == "current" else proposed_map
                    ),
                    reference_pipeline=baseline_pipeline,
                    node_ids=safe_ids,
                    environment=environment,
                )
                ended = _read_clock(self.clock)
                measured[label] = _nonnegative_elapsed(began, ended)
            current_times.append(measured["current"])
            proposed_times.append(measured["proposed"])
        paired = paired_bootstrap_speedup(
            current_times,
            proposed_times,
            sample_count=DEFAULT_BOOTSTRAP_SAMPLES,
            seed=DEFAULT_BOOTSTRAP_SEED,
        )
        return PipelineAssignmentValidation(
            validation.identity_digest,
            validation.current_assignment,
            validation.proposed_assignment,
            True,
            True,
            float(statistics.median(current_times)),
            float(statistics.median(proposed_times)),
            paired.lower_confidence_bound,
            "Changed-node and observable-boundary parity passed before paired "
            "synchronized timing.",
        )


def _observable_pipeline_boundaries(
    pipeline: PrototypePipeline,
    safe_ids: frozenset[str],
    retained: frozenset[str],
) -> frozenset[str]:
    """Return host-visible outputs whose final scientific value must agree."""

    safe_successors = {node_id: set() for node_id in safe_ids}
    for connection in pipeline.connections:
        if connection.source_id in safe_ids and connection.target_id in safe_ids:
            safe_successors[connection.source_id].add(connection.target_id)
    terminals = {
        node_id for node_id, successors in safe_successors.items() if not successors
    }
    tunnels = {
        tunnel.source_id
        for tunnel in pipeline.output_tunnel_list()
        if tunnel.source_id in safe_ids
    }
    return frozenset((set(retained) & set(safe_ids)) | terminals | tunnels)


def _pipeline_output_parity(
    operation_id: str,
    reference: object,
    candidate: object,
    *,
    input_peak: float | None = None,
) -> tuple[bool, str]:
    """Check exact boundary equality, then a registered numeric tolerance."""

    if _scientific_values_equal(reference, candidate):
        return True, "Exact output equality passed."
    try:
        parity = operation_parity(
            operation_id,
            reference,
            candidate,
            input_peak=input_peak,
        )
    except (TypeError, ValueError) as exc:
        return (
            False,
            "outputs differ and no applicable production parity policy accepted "
            f"the boundary ({exc})",
        )
    return bool(parity.passed), parity.detail


def _pipeline_input_peak(
    pipeline: PrototypePipeline,
    node_id: str,
) -> float | None:
    """Match the node benchmark's first-input scale for parity validation."""

    values = tuple(pipeline.input_data_by_port_for_node(node_id).values())
    if not values or values[0] is None:
        return None
    try:
        array = np.asarray(values[0])
    except Exception:
        return None
    if not np.issubdtype(array.dtype, np.number) or not array.size:
        return 0.0
    finite = np.isfinite(array)
    if not bool(np.any(finite)):
        return 0.0
    return float(np.max(np.abs(array[finite].astype(np.float64))))


def _scientific_values_equal(reference: object, candidate: object) -> bool:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        return set(reference) == set(candidate) and all(
            _scientific_values_equal(reference[key], candidate[key])
            for key in reference
        )
    if isinstance(reference, (tuple, list)) and isinstance(candidate, type(reference)):
        return len(reference) == len(candidate) and all(
            _scientific_values_equal(left, right)
            for left, right in zip(reference, candidate, strict=True)
        )
    equals = getattr(reference, "equals", None)
    if callable(equals) and type(reference) is type(candidate):
        try:
            return bool(equals(candidate))
        except Exception:
            return False
    try:
        left = np.asarray(reference)
        right = np.asarray(candidate)
    except Exception:
        try:
            return bool(reference == candidate)
        except Exception:
            return False
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if not left.dtype.hasobject:
        try:
            left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(-1)
            right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(-1)
            return bool(np.array_equal(left_bytes, right_bytes))
        except (TypeError, ValueError):
            return False
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except (TypeError, ValueError):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False


def probe_pipeline_optimizer_environment(
    registry: ComputeRegistry,
    workflow: Mapping[str, object],
    compute_request: ComputeRequest,
) -> ComputeEnvironment:
    """Re-probe the exact optimizer candidate environment without executing data.

    Application surfaces use this immediately before applying a proposal and
    compare ``fingerprint`` with the captured optimization identity.  The helper
    performs descriptor/provider probes only; it never copies a source, runs a
    node, reads benchmark records, or mutates the workflow.
    """

    if not isinstance(registry, ComputeRegistry):
        raise TypeError("registry must be a ComputeRegistry")
    if not isinstance(workflow, Mapping):
        raise TypeError("workflow must be a detached workflow mapping")
    if not isinstance(compute_request, ComputeRequest):
        raise TypeError("compute_request must be a ComputeRequest")
    if compute_request.mode is not ComputeMode.SELECTIVE:
        _refuse("selective_required", "Choose Selective compute policy.")
    restored = deserialize_workflow(deepcopy(dict(workflow)))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    safe_ids, _unsafe_ids = _writer_free_node_ids(pipeline)
    environment, _runtime_id = _probe_optimizer_environment_for_pipeline(
        registry,
        pipeline,
        safe_ids,
        compute_request,
    )
    return environment


def fingerprint_pipeline_optimizer_sources(
    workflow: Mapping[str, object],
    source_payloads: Mapping[str, SourcePayload],
    *,
    time_budget_seconds: float = 120.0,
    cancelled: CancelCallback | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> str:
    """Reconstruct the exact detached source identity without running a node."""

    if not isinstance(workflow, Mapping):
        raise TypeError("workflow must be a detached workflow mapping")
    if not isinstance(source_payloads, Mapping):
        raise TypeError("source_payloads must be a mapping")
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None")
    if not callable(clock):
        raise TypeError("clock must be callable")
    deadline = _read_clock(clock) + _positive_duration(
        time_budget_seconds,
        "time_budget_seconds",
    )

    def check_abort() -> None:
        _check_abort(clock, deadline, cancelled)

    restored = deserialize_workflow(deepcopy(dict(workflow)))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored.get("output_tunnels", ()),
    )
    safe_ids, _unsafe_ids = _writer_free_node_ids(pipeline)
    sources = _detach_source_payloads(
        pipeline,
        source_payloads,
        safe_ids,
        check_abort=check_abort,
    )
    check_abort()
    return canonical_digest(dict(sources.fingerprints))


def _probe_optimizer_environment_for_pipeline(
    registry: ComputeRegistry,
    pipeline: PrototypePipeline,
    safe_ids: frozenset[str],
    compute_request: ComputeRequest,
) -> tuple[ComputeEnvironment, str]:
    operation_ids = {pipeline.nodes[item].operation_id for item in safe_ids}
    candidate_specs = tuple(
        spec
        for spec in registry.implementation_specs
        if spec.operation_id in operation_ids
        and spec.visible_for(allow_experimental=compute_request.allow_experimental)
        and (
            not compute_request.runtime_id
            or spec.runtime_id == compute_request.runtime_id
        )
    )
    runtime_ids = {spec.runtime_id for spec in candidate_specs}
    if not candidate_specs:
        _refuse(
            "no_gpu_candidates",
            "This pipeline has no declared GPU implementation to optimize.",
        )
    if len(runtime_ids) != 1:
        _refuse(
            "multiple_accelerator_runtimes",
            "This optimizer version requires GPU candidates to share one "
            "array runtime.",
        )
    accelerator_runtime_id = next(iter(runtime_ids))
    environment, _warnings = probe_compute_environment(
        registry,
        compute_request,
        candidate_specs,
    )
    if accelerator_runtime_id not in environment.runtime_ids:
        _refuse(
            "accelerator_unavailable",
            environment.probe_reason
            or f"Runtime {accelerator_runtime_id!r} is unavailable.",
        )
    return environment, accelerator_runtime_id


def _writer_free_node_ids(
    pipeline: PrototypePipeline,
) -> tuple[frozenset[str], frozenset[str]]:
    direct_unsafe = {
        node_id
        for node_id, node in pipeline.nodes.items()
        if compute_specs_for(node.operation_id)[0].side_effect_policy_id
        == "host-writer-v1"
    }
    unsafe = set(direct_unsafe)
    changed = True
    while changed:
        changed = False
        for connection in pipeline.connections:
            if connection.source_id in unsafe and connection.target_id not in unsafe:
                unsafe.add(connection.target_id)
                changed = True
    safe = frozenset(set(pipeline.nodes) - unsafe)
    if not safe:
        _refuse(
            "writer_only_pipeline",
            "The pipeline has no writer-free scientific subgraph to optimize.",
        )
    return safe, frozenset(unsafe)


def _detach_source_payloads(
    pipeline: PrototypePipeline,
    source_payloads: Mapping[str, SourcePayload],
    safe_ids: frozenset[str],
    *,
    check_abort: Callable[[], None],
) -> _DetachedSources:
    source_ids = {
        node_id for node_id in safe_ids if not pipeline.nodes[node_id].has_input
    }
    normalized = {str(key).strip(): value for key, value in source_payloads.items()}
    if any(not key for key in normalized):
        _refuse("source_identity_invalid", "Source payload IDs must not be empty.")
    missing = source_ids - set(normalized)
    unknown = set(normalized) - set(pipeline.nodes)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        _refuse(
            "source_identity_incomplete",
            "Exact source payload mapping is " + "; ".join(details) + ".",
        )
    payloads: dict[str, SourcePayload] = {}
    fingerprints: dict[str, str] = {}
    for node_id in sorted(source_ids):
        check_abort()
        payload = normalized[node_id]
        if not isinstance(payload, SourcePayload):
            raise TypeError("source_payloads values must be SourcePayload instances")
        array = np.asarray(payload.data)
        if array.dtype.hasobject:
            _refuse(
                "source_dtype_unsupported",
                "Object-valued source arrays cannot receive an exact byte identity.",
                node_id,
            )
        detached = _copy_array_with_abort(array, check_abort=check_abort)
        detached.setflags(write=False)
        data_digest = _array_digest(detached, check_abort=check_abort)
        if _array_digest(array, check_abort=check_abort) != data_digest:
            _refuse(
                "source_changed_during_capture",
                "Source data changed while its private optimizer snapshot was "
                "being captured.",
                node_id,
            )
        metadata = deepcopy(payload.metadata)
        image_state = deepcopy(payload.image_state)
        fingerprint = canonical_digest(
            {
                "data": data_digest,
                "shape": list(detached.shape),
                "dtype": detached.dtype.str,
                "strides": list(detached.strides),
                "metadata": _identity_value(metadata),
                "name": payload.name,
                "image_state": _identity_value(image_state),
                "revision_token": _identity_value(payload.revision_token),
            }
        )
        payloads[node_id] = SourcePayload(
            detached,
            metadata,
            str(payload.name),
            image_state,
            fingerprint,
        )
        fingerprints[node_id] = fingerprint
    return _DetachedSources(
        MappingProxyType(payloads),
        MappingProxyType(fingerprints),
    )


def _array_digest(
    array: np.ndarray,
    *,
    check_abort: Callable[[], None],
) -> str:
    digest = sha256()
    values_per_chunk = max(1, _SOURCE_HASH_CHUNK_BYTES // max(1, array.itemsize))
    iterator = np.nditer(
        array,
        flags=["buffered", "external_loop", "zerosize_ok"],
        op_flags=[["readonly"]],
        order="C",
        buffersize=values_per_chunk,
    )
    for raw_chunk in iterator:
        check_abort()
        chunk = np.ascontiguousarray(np.asarray(raw_chunk))
        digest.update(memoryview(chunk).cast("B"))
    check_abort()
    return digest.hexdigest()


def _copy_array_with_abort(
    array: np.ndarray,
    *,
    check_abort: Callable[[], None],
) -> np.ndarray:
    detached = np.empty_like(array, order="K", subok=False)
    values_per_chunk = max(1, _SOURCE_HASH_CHUNK_BYTES // max(1, array.itemsize))
    iterator = np.nditer(
        (array, detached),
        flags=["buffered", "external_loop", "zerosize_ok"],
        op_flags=[["readonly"], ["writeonly"]],
        order="K",
        buffersize=values_per_chunk,
    )
    for source_chunk, target_chunk in iterator:
        check_abort()
        target_chunk[...] = source_chunk
    check_abort()
    return detached


def _identity_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"nonfinite_float": "nan"}
        return {"nonfinite_float": "+inf" if value > 0 else "-inf"}
    if isinstance(value, np.generic):
        return _identity_value(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _identity_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("identity arrays must not have object dtype")
        return {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "digest": sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _build_optimizer_graph(
    pipeline: PrototypePipeline,
    safe_ids: frozenset[str],
    retained: frozenset[str],
    request: ComputeRequest,
    decisions: Mapping[str, object],
    records: Mapping[str, object],
    plans: Mapping[str, object],
    *,
    check_abort: Callable[[], None],
) -> tuple[
    tuple[PipelineOptimizationNode, ...],
    tuple[PipelineOptimizationEdge, ...],
    Mapping[str, str],
]:
    connections = tuple(
        item
        for item in pipeline.connections
        if item.source_id in safe_ids and item.target_id in safe_ids
    )
    edge_pairs = tuple((item.source_id, item.target_id) for item in connections)
    if len(set(edge_pairs)) != len(edge_pairs):
        _refuse(
            "parallel_edges_unsupported",
            "Whole-pipeline optimization cannot yet model multiple ports between "
            "the same pair of nodes without over-counting transfers.",
        )
    successors = {node_id: set() for node_id in safe_ids}
    for item in connections:
        successors[item.source_id].add(item.target_id)
    tunnel_sources = {
        item.source_id
        for item in pipeline.output_tunnel_list()
        if item.source_id in safe_ids
    }
    nodes: list[PipelineOptimizationNode] = []
    workloads: dict[str, str] = {}
    for node_id in pipeline.topological_order():
        if node_id not in safe_ids:
            continue
        check_abort()
        node = pipeline.nodes[node_id]
        cpu_spec = compute_specs_for(node.operation_id)[0]
        record = records.get(node_id)
        plan = plans.get(node_id)
        if record is None:
            candidates = (
                PipelineOptimizationCandidate(
                    cpu_spec.implementation_id,
                    cpu_spec.implementation_library_id,
                    cpu_spec.runtime_id,
                ),
            )
        else:
            gpu_specs = {spec.implementation_id: spec for spec in plan.admitted_specs}
            candidates_list = []
            for result in record.candidates:
                spec = (
                    cpu_spec
                    if result.implementation_id == cpu_spec.implementation_id
                    else gpu_specs.get(result.implementation_id)
                )
                if spec is None:
                    _refuse(
                        "benchmark_candidate_identity_missing",
                        "A benchmark result has no exact implementation declaration.",
                        node_id,
                    )
                candidates_list.append(
                    PipelineOptimizationCandidate(
                        spec.implementation_id,
                        spec.implementation_library_id,
                        spec.runtime_id,
                    )
                )
            candidates = tuple(candidates_list)
        decision = decisions.get(node_id)
        current_id = (
            getattr(decision, "implementation_id", "")
            if node.has_input
            else cpu_spec.implementation_id
        )
        if not current_id:
            current_id = cpu_spec.implementation_id
        if current_id not in {item.implementation_id for item in candidates}:
            _refuse(
                "current_assignment_unbenchmarked",
                "The private baseline implementation is absent from exact "
                "benchmark evidence.",
                node_id,
            )
        if plan is not None:
            workload_fingerprint = plan.workload_fingerprint
        elif node.has_input:
            call = pipeline.prepare_node_call(node_id)
            if call is None:
                _refuse(
                    "workload_identity_unresolved",
                    "Current node inputs could not be prepared for identity capture.",
                    node_id,
                )
            workload_fingerprint = workload_from_prepared_node_call(
                call,
                check_abort=check_abort,
            ).fingerprint
        else:
            workload_fingerprint = canonical_digest(
                {
                    "node_id": node_id,
                    "operation_id": node.operation_id,
                    "params": node.params,
                    "outputs": _output_byte_count(pipeline, node_id),
                }
            )
        workloads[node_id] = workload_fingerprint
        requires_host = (
            node_id in retained or node_id in tunnel_sources or not successors[node_id]
        )
        nodes.append(
            PipelineOptimizationNode(
                node_id,
                node.operation_id,
                candidates,
                current_id,
                authored_preference=request.preference_for(node_id),
                output_bytes=_output_byte_count(pipeline, node_id),
                requires_host_output=requires_host,
                cache_retained=node_id in retained,
            )
        )
    edges = tuple(
        PipelineOptimizationEdge(item.source_id, item.target_id) for item in connections
    )
    return tuple(nodes), edges, MappingProxyType(workloads)


def _output_byte_count(pipeline: PrototypePipeline, node_id: str) -> int:
    values = tuple(pipeline.node_outputs.get(node_id, ()))
    total = 0
    seen: set[int] = set()
    for value in values:
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if array.dtype.hasobject:
            continue
        total += int(array.nbytes)
    return total


def _measure_directional_transfers(
    registry: ComputeRegistry,
    runtime_id: str,
    device_id: str,
    environment: ComputeEnvironment,
    identity: PipelineOptimizationIdentity,
    pipeline: PrototypePipeline,
    benchmarked_node_ids: Sequence[str],
    request: ComputeRequest,
    *,
    clock: Callable[[], float],
    check_abort: Callable[[], None],
) -> DirectionalTransferProfile:
    runtime = registry.runtime(runtime_id)
    dtype, maximum_bytes = _representative_transfer_dtype(
        pipeline,
        benchmarked_node_ids,
    )
    snapshot = runtime.memory_snapshot(device_id=device_id)
    total = int(snapshot.device_total_bytes or 0)
    free = int(snapshot.device_free_bytes or 0)
    if total <= 0 or free <= 0:
        _refuse(
            "transfer_memory_unknown",
            "Current accelerator total/free memory is unavailable.",
        )
    reserve = (
        max(_DEFAULT_ACCELERATOR_RESERVE_BYTES, total // 10)
        if request.accelerator_safety_reserve_bytes is None
        else request.accelerator_safety_reserve_bytes
    )
    requested_limit = (
        total * 80 // 100
        if request.accelerator_memory_cap_bytes is None
        else request.accelerator_memory_cap_bytes
    )
    memory_limit = min(requested_limit, free - reserve)
    if memory_limit <= 0:
        _refuse(
            "transfer_memory_unavailable",
            "No accelerator memory remains after the active safety reserve.",
        )
    sample_sizes = _transfer_sample_sizes(maximum_bytes, np.dtype(dtype).itemsize)
    observations: dict[int, tuple[float, float]] = {}
    with runtime.execution_scope(
        device_id=device_id,
        memory_limit_bytes=memory_limit,
        safety_reserve_bytes=0,
    ):
        for sample_bytes in sample_sizes:
            count = max(1, sample_bytes // np.dtype(dtype).itemsize)
            host = np.zeros(count, dtype=dtype)
            actual_bytes = int(host.nbytes)
            h2d: list[float] = []
            d2h: list[float] = []
            for _round in range(_TRANSFER_ROUNDS):
                check_abort()
                began = _read_clock(clock)
                device = runtime.to_device(host, device_id=device_id)
                try:
                    runtime.synchronize(device_id=device_id)
                    ended = _read_clock(clock)
                    h2d.append(_nonnegative_elapsed(began, ended))
                    began = _read_clock(clock)
                    returned = runtime.to_host(device)
                    runtime.synchronize(device_id=device_id)
                    ended = _read_clock(clock)
                    d2h.append(_nonnegative_elapsed(began, ended))
                    if np.asarray(returned).shape != host.shape:
                        _refuse(
                            "transfer_roundtrip_invalid",
                            "Runtime transfer round-trip changed the sample shape.",
                        )
                finally:
                    runtime.release(device)
                    # CuPy ownership ends only when the final Python alias dies;
                    # ``release`` validates ownership but deliberately cannot
                    # invalidate aliases.  Do not retain the last transfer sample
                    # until the private execution scope performs its leak check.
                    del device
            observations[actual_bytes] = (
                float(statistics.median(h2d)),
                float(statistics.median(d2h)),
            )
    sizes = tuple(sorted(observations))
    h_fixed, h_slope = _linear_transfer_cost(
        sizes,
        tuple(observations[size][0] for size in sizes),
    )
    d_fixed, d_slope = _linear_transfer_cost(
        sizes,
        tuple(observations[size][1] for size in sizes),
    )
    return DirectionalTransferProfile(
        identity.digest,
        environment.fingerprint,
        runtime_id,
        h_fixed,
        h_slope,
        d_fixed,
        d_slope,
        memory_limit,
        max(sizes),
        synchronized=True,
    )


def _representative_transfer_dtype(
    pipeline: PrototypePipeline,
    node_ids: Sequence[str],
) -> tuple[np.dtype, int]:
    selected: tuple[np.dtype, int] | None = None
    for node_id in node_ids:
        for value in pipeline.input_data_by_port_for_node(node_id).values():
            try:
                array = np.asarray(value)
            except Exception:
                continue
            if array.dtype.hasobject or array.nbytes <= 0:
                continue
            candidate = (array.dtype, int(array.nbytes))
            if selected is None or candidate[1] > selected[1]:
                selected = candidate
    if selected is None:
        _refuse(
            "transfer_sample_missing",
            "No numeric GPU-node input is available for transfer measurement.",
        )
    return selected


def _transfer_sample_sizes(maximum_bytes: int, itemsize: int) -> tuple[int, ...]:
    maximum = max(itemsize, min(int(maximum_bytes), _TRANSFER_LARGE_BYTES))
    small = max(itemsize, min(maximum, _TRANSFER_SMALL_BYTES))
    if maximum == small:
        return (maximum,)
    return (small, maximum)


def _linear_transfer_cost(
    sizes: Sequence[int],
    seconds: Sequence[float],
) -> tuple[float, float]:
    if len(sizes) != len(seconds) or not sizes:
        raise ValueError("transfer model requires aligned observations")
    if len(sizes) == 1:
        return 0.0, float(seconds[0]) / float(sizes[0])
    delta_bytes = float(sizes[-1] - sizes[0])
    slope = max(0.0, (float(seconds[-1]) - float(seconds[0])) / delta_bytes)
    fixed = max(0.0, float(seconds[0]) - slope * float(sizes[0]))
    return fixed, slope


def _exact_assignment_request(
    request: ComputeRequest,
    pipeline: PrototypePipeline,
    assignment: Mapping[str, str],
) -> ComputeRequest:
    preferences: dict[str, NodeComputePreference] = {}
    for node_id, implementation_id in assignment.items():
        node = pipeline.nodes.get(node_id)
        if node is None or not node.has_input:
            continue
        cpu_id = compute_specs_for(node.operation_id)[0].implementation_id
        preferences[node_id] = (
            NodeComputePreference(NodePreferenceKind.CPU)
            if implementation_id == cpu_id
            else NodeComputePreference(
                NodePreferenceKind.IMPLEMENTATION,
                implementation_id,
            )
        )
    return replace(
        request,
        mode=ComputeMode.SELECTIVE,
        node_preferences=preferences,
        fallback_policy=FallbackPolicy.STRICT,
    )


def _successful_pipeline(result: PipelineRunResult, label: str) -> PrototypePipeline:
    if result.cancelled:
        raise PipelineOptimizationCancelled(
            result.error or f"Private {label} execution was cancelled."
        )
    if result.error or result.pipeline is None:
        _refuse(
            f"{label.replace(' ', '_')}_failed",
            result.error or f"Private {label} execution returned no pipeline.",
        )
    return result.pipeline


def _successful_exact_pipeline(
    result: PipelineRunResult,
    label: str,
    *,
    expected_request: ComputeRequest,
    expected_assignment: Mapping[str, str],
    reference_pipeline: PrototypePipeline,
    node_ids: frozenset[str],
    environment: ComputeEnvironment,
) -> PrototypePipeline:
    """Require private validation to execute the assignment it claims to time."""

    pipeline = _successful_pipeline(result, label)
    report = result.execution_report
    code_label = label.replace(" ", "_")
    if report is None:
        _refuse(
            f"{code_label}_provenance_missing",
            f"Private {label} execution returned no exact compute provenance.",
        )
    if report.request.fingerprint != expected_request.fingerprint:
        _refuse(
            f"{code_label}_request_mismatch",
            f"Private {label} execution did not echo the exact compute request.",
        )
    if report.environment.fingerprint != environment.fingerprint:
        _refuse(
            f"{code_label}_environment_mismatch",
            f"Private {label} execution used a different runtime environment.",
        )
    if not report.cleanup_succeeded:
        _refuse(
            f"{code_label}_cleanup_failed",
            f"Private {label} execution did not release its accelerator resources.",
        )
    decision_ids = tuple(item.node_id for item in report.actual_decisions)
    if len(set(decision_ids)) != len(decision_ids):
        _refuse(
            f"{code_label}_decision_ambiguous",
            f"Private {label} execution reported duplicate node decisions.",
        )
    decisions = {item.node_id: item for item in report.actual_decisions}
    expected_processing_ids = {
        node_id
        for node_id in node_ids
        if reference_pipeline.nodes[node_id].has_input
    }
    unexpected_processing_ids = {
        item.node_id
        for item in report.actual_decisions
        if item.node_id not in reference_pipeline.nodes
        or (
            reference_pipeline.nodes[item.node_id].has_input
            and item.node_id not in expected_processing_ids
        )
    }
    if unexpected_processing_ids:
        _refuse(
            f"{code_label}_decision_scope_mismatch",
            f"Private {label} reported decisions outside the safe processing "
            "subgraph: " + ", ".join(sorted(unexpected_processing_ids)),
        )
    for node_id in reference_pipeline.topological_order():
        if node_id not in node_ids:
            continue
        node = reference_pipeline.nodes[node_id]
        if not node.has_input:
            continue
        expected_implementation = expected_assignment.get(node_id)
        decision = decisions.get(node_id)
        if (
            not expected_implementation
            or decision is None
            or decision.operation_id != node.operation_id
            or decision.implementation_id != expected_implementation
            or decision.fallback_used
        ):
            actual = "missing" if decision is None else decision.implementation_id
            _refuse(
                f"{code_label}_assignment_mismatch",
                f"Private {label} requested {expected_implementation!r} but "
                f"reported {actual!r}; exact validation cannot continue.",
                node_id,
            )
    return pipeline


def _topology_fingerprint(pipeline: PrototypePipeline) -> str:
    return canonical_digest(
        {
            "nodes": [
                (node_id, pipeline.nodes[node_id].operation_id)
                for node_id in pipeline.topological_order()
            ],
            "connections": [
                (
                    item.source_id,
                    item.source_port,
                    item.target_id,
                    item.target_port,
                    item.tunnel_name,
                )
                for item in pipeline.connections
            ],
            "tunnels": [
                (item.name, item.source_id, item.source_port)
                for item in pipeline.output_tunnel_list()
            ],
        }
    )


def _normalized_node_ids(values: Sequence[str] | frozenset[str]) -> frozenset[str]:
    normalized = frozenset(str(value).strip() for value in values)
    if "" in normalized:
        raise ValueError("node IDs must not be empty")
    return normalized


def _positive_duration(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _read_clock(clock: Callable[[], float]) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("optimizer clock must return a finite value")
    return float(value)


def _nonnegative_elapsed(started: float, ended: float) -> float:
    elapsed = float(ended) - float(started)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("optimizer clock must be monotonic")
    return elapsed


def _check_abort(
    clock: Callable[[], float],
    deadline: float,
    cancelled: CancelCallback | None,
) -> None:
    if cancelled is not None and cancelled():
        raise PipelineOptimizationCancelled("pipeline optimization cancelled")
    if _read_clock(clock) >= deadline:
        raise PipelineOptimizationDeadlineExceeded(
            "Pipeline optimization exceeded its end-to-end deadline."
        )


def _emit(
    callback: ProgressCallback | None,
    phase: PipelineOptimizerPhase,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(PipelineOptimizerProgress(phase, completed, total, message))


def _refuse(code: str, message: str, node_id: str = "") -> None:
    raise PipelineOptimizationEvidenceIncomplete(
        (EvidenceRefusal(code, message, node_id),)
    )


__all__ = [
    "ApplicationPipelineOptimizationResult",
    "ApplicationPipelineOptimizerCoordinator",
    "PipelineOptimizerPhase",
    "PipelineOptimizerProgress",
    "fingerprint_pipeline_optimizer_sources",
    "probe_pipeline_optimizer_environment",
]
