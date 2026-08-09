"""Qt-free application coordinator for selected-node benchmarking.

The lower-level benchmark service intentionally accepts prepared callables.  This
module bridges that contract to a live :class:`PrototypePipeline` without running
or mutating it: graph structure is cloned, current inputs are captured by the
production adapter, candidates pass the normal support/memory gates, and only a
complete parity-qualified record is published to a machine-local JSON store.
"""

from __future__ import annotations

import importlib.metadata
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from napari_vipp.core.accelerator_lease import accelerator_lease
from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    BenchmarkRecord,
    ComputeEnvironment,
    ComputeRequest,
    NodeComputePreference,
    NodePreferenceKind,
    canonical_digest,
)
from napari_vipp.core.compute_benchmark import (
    ADAPTIVE_WARM_ROUNDS,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    MINIMUM_WARM_ROUNDS,
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
    BenchmarkMeasurementPhase,
    BenchmarkMeasurementProgress,
    BenchmarkRejected,
    JsonBenchmarkStore,
    NodeBenchmarkService,
)
from napari_vipp.core.compute_benchmark_adapter import (
    BENCHMARK_WRITER_OPERATION_IDS,
    RegisteredNodeBenchmark,
    build_registered_node_benchmark,
    detach_prepared_node_call,
    workload_from_prepared_node_call,
)
from napari_vipp.core.compute_planning import probe_compute_environment
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    FactCompleteness,
    estimate_candidate_memory,
    evaluate_candidate_support,
    evaluate_candidate_workload_support,
    evaluate_memory_support,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import OperationComputeSpec
from napari_vipp.core.node_execution import PreparedNodeCall

if TYPE_CHECKING:
    from napari_vipp.core.pipeline import PrototypePipeline

ProgressCallback = Callable[["NodeBenchmarkProgress"], None]
CancelCallback = Callable[[], bool]
_DEFAULT_ACCELERATOR_RESERVE_BYTES = 512 * 1024**2
_BENCHMARK_ENVIRONMENT_POLICY_ID = "node-benchmark-environment-v2"
_SCIENTIFIC_STACK_DISTRIBUTIONS = (
    "napari-vipp",
    "numpy",
    "scipy",
    "scikit-image",
)


def benchmark_environment_fingerprint(environment: ComputeEnvironment) -> str:
    """Bind reusable timings to the exact CPU/GPU scientific software stack."""

    if not isinstance(environment, ComputeEnvironment):
        raise TypeError("environment must be a ComputeEnvironment.")
    versions: dict[str, str] = {}
    for distribution in _SCIENTIFIC_STACK_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return canonical_digest(
        {
            "policy": _BENCHMARK_ENVIRONMENT_POLICY_ID,
            "compute_environment": environment.fingerprint,
            "scientific_stack": versions,
        }
    )


class NodeBenchmarkPhase(StrEnum):
    """Coarse phases suitable for an indeterminate application worker."""

    PREPARING = "preparing"
    ELIGIBILITY = "eligibility"
    READY = "ready"
    BENCHMARKING = "benchmarking"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class NodeBenchmarkProgress:
    phase: NodeBenchmarkPhase | str
    completed: int
    total: int
    message: str
    measurement_completed: int = 0
    measurement_total: int = 0
    measurement_message: str = ""
    implementation_id: str = ""
    implementation_version: str = ""
    measurement_phase: str = ""
    operation_completed: int = 0
    operation_total: int = 0
    operation_message: str = ""

    def __post_init__(self) -> None:
        phase = (
            self.phase
            if isinstance(self.phase, NodeBenchmarkPhase)
            else NodeBenchmarkPhase(str(self.phase).strip().lower())
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.completed, self.total)
        ):
            raise ValueError("benchmark progress values must be non-negative integers.")
        if self.total < 1 or self.completed > self.total:
            raise ValueError("benchmark progress must fit inside its declared total.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.measurement_completed, self.measurement_total)
        ):
            raise ValueError("measurement progress values must be non-negative.")
        if self.measurement_completed > self.measurement_total:
            raise ValueError("measurement progress must fit inside its declared total.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.operation_completed, self.operation_total)
        ):
            raise ValueError("operation progress values must be non-negative.")
        if self.operation_completed > self.operation_total:
            raise ValueError("operation progress must fit inside its declared total.")
        message = str(self.message).strip()
        if not message:
            raise ValueError("benchmark progress message must not be empty.")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "message", message)
        for name in (
            "measurement_message",
            "implementation_id",
            "implementation_version",
            "measurement_phase",
            "operation_message",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if self.measurement_total and not self.measurement_message:
            raise ValueError("measurement progress requires a message.")
        if self.operation_total and not self.operation_message:
            raise ValueError("operation progress requires a message.")


@dataclass(frozen=True, slots=True)
class NodeBenchmarkCandidateEligibility:
    """One declared GPU candidate and its exact current-workload decision."""

    implementation_id: str
    implementation_library_id: str
    supported: bool
    reason_code: str
    reason_text: str

    def __post_init__(self) -> None:
        for name in (
            "implementation_id",
            "implementation_library_id",
            "reason_code",
            "reason_text",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        if not isinstance(self.supported, bool):
            raise TypeError("supported must be a boolean.")


class NodeBenchmarkUnavailable(BenchmarkRejected):
    """The selected node has no scientifically eligible GPU candidate."""

    def __init__(
        self,
        message: str,
        eligibility: Sequence[NodeBenchmarkCandidateEligibility] = (),
    ) -> None:
        self.eligibility = tuple(eligibility)
        super().__init__(str(message).strip() or "Node benchmark is unavailable.")


@dataclass(frozen=True, slots=True)
class ApplicationNodeBenchmarkPlan:
    """Detached, immutable-enough input to an application benchmark worker."""

    registered: RegisteredNodeBenchmark
    environment: ComputeEnvironment
    admitted_specs: tuple[OperationComputeSpec, ...]
    eligibility: tuple[NodeBenchmarkCandidateEligibility, ...]
    store_path: Path
    preparation_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.registered, RegisteredNodeBenchmark):
            raise TypeError("registered must be a RegisteredNodeBenchmark.")
        if not isinstance(self.environment, ComputeEnvironment):
            raise TypeError("environment must be a ComputeEnvironment.")
        specs = tuple(self.admitted_specs)
        if not specs or any(
            not isinstance(item, OperationComputeSpec) for item in specs
        ):
            raise ValueError("admitted_specs must contain GPU implementation specs.")
        if any(
            item.operation_id != self.registered.request.workload.operation_id
            for item in specs
        ):
            raise ValueError("admitted specs must match the benchmark operation.")
        eligibility = tuple(self.eligibility)
        if any(
            not isinstance(item, NodeBenchmarkCandidateEligibility)
            for item in eligibility
        ):
            raise TypeError("eligibility contains an invalid entry.")
        path = Path(self.store_path).expanduser().resolve(strict=False)
        seconds = float(self.preparation_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("preparation_seconds must be finite and non-negative.")
        object.__setattr__(self, "admitted_specs", specs)
        object.__setattr__(self, "eligibility", eligibility)
        object.__setattr__(self, "store_path", path)
        object.__setattr__(self, "preparation_seconds", seconds)

    @property
    def node_id(self) -> str:
        return self.registered.request.workload.node_id

    @property
    def operation_id(self) -> str:
        return self.registered.request.workload.operation_id

    @property
    def workload_fingerprint(self) -> str:
        return self.registered.request.workload.fingerprint

    @property
    def key_digest(self) -> str:
        return self.registered.request.key.digest

    def preference_for(self, record: BenchmarkRecord) -> NodeComputePreference:
        if record.key != self.registered.request.key:
            raise ValueError("benchmark record does not match this exact plan.")
        return stable_preference_for_benchmark_winner(
            record,
            self.admitted_specs,
            cpu_implementation_id=(self.registered.request.reference.implementation_id),
        )


@dataclass(frozen=True, slots=True)
class ApplicationNodeBenchmarkResult:
    plan: ApplicationNodeBenchmarkPlan
    record: BenchmarkRecord
    winner_preference: NodeComputePreference

    def __post_init__(self) -> None:
        if self.record.key != self.plan.registered.request.key:
            raise ValueError("benchmark result does not match its prepared plan.")
        expected = self.plan.preference_for(self.record)
        if self.winner_preference != expected:
            raise ValueError("winner_preference does not match the benchmark record.")


def _record_is_complete_for_plan(
    record: BenchmarkRecord,
    plan: ApplicationNodeBenchmarkPlan,
) -> bool:
    """Distinguish a completed rejection from incomplete/corrupt evidence."""

    request = plan.registered.request
    if record.key != request.key:
        return False
    expected_ids = {
        request.reference.implementation_id,
        *(candidate.implementation_id for candidate in request.candidates),
    }
    by_id = {candidate.implementation_id: candidate for candidate in record.candidates}
    if len(by_id) != len(record.candidates) or set(by_id) != expected_ids:
        return False
    # A censored lower bound is useful to the active whole-pipeline search, but
    # it is intentionally not an exact reusable median.  A later run may reuse
    # only a record in which every admitted timing completed.
    if any(candidate.timing_censored for candidate in record.candidates):
        return False

    def has_complete_timings(implementation_id: str) -> bool:
        candidate = by_id[implementation_id]
        return (
            candidate.parity_passed
            and not candidate.error
            and candidate.cold_seconds is not None
            and len(candidate.warm_seconds) >= request.warm_rounds
        )

    reference_id = request.reference.implementation_id
    accepted_id = str(record.accepted_implementation_id).strip()
    if (
        accepted_id not in by_id
        or not has_complete_timings(reference_id)
        or not has_complete_timings(accepted_id)
    ):
        return False

    # Qualified alternatives require complete timings.  Only a typed scientific
    # parity mismatch is durable rejection evidence.  Runtime, OOM, cleanup, or
    # timing failures remain retryable and invalidate reuse of the whole record.
    return all(
        candidate.failure_kind is BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY
        or has_complete_timings(candidate.implementation_id)
        for candidate in record.candidates
    )


class ApplicationNodeBenchmarkCoordinator:
    """Prepare and run exact selected-node benchmarks without GUI imports."""

    _PROGRESS_TOTAL = 4

    def __init__(
        self,
        registry: ComputeRegistry,
        store_path: str | Path,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(registry, ComputeRegistry):
            raise TypeError("registry must be a ComputeRegistry.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.registry = registry
        self.store = JsonBenchmarkStore(store_path)
        self.clock = clock
        self.service = NodeBenchmarkService(store=self.store, clock=clock)

    def prepare(
        self,
        pipeline: PrototypePipeline,
        node_id: str,
        *,
        environment: ComputeEnvironment | None = None,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
        warm_rounds: int = MINIMUM_WARM_ROUNDS,
        max_warm_rounds: int = ADAPTIVE_WARM_ROUNDS[-1],
        time_budget_seconds: float | None = None,
        allow_experimental: bool = False,
        paired_bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
        paired_bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
        paired_confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        adaptive_candidate_stopping: bool = False,
        cancelled: CancelCallback | None = None,
        progress: ProgressCallback | None = None,
    ) -> ApplicationNodeBenchmarkPlan:
        """Capture current inputs and build an admitted detached benchmark."""

        _validate_callbacks(cancelled, progress)
        if not isinstance(adaptive_candidate_stopping, bool):
            raise TypeError("adaptive_candidate_stopping must be a boolean.")
        budget = _validated_budget(time_budget_seconds)
        started = _read_clock(self.clock)

        def check_abort() -> None:
            _check_cancelled(cancelled)
            _check_preparation_budget(budget, started, self.clock)

        check_abort()
        _emit_progress(
            progress,
            NodeBenchmarkPhase.PREPARING,
            0,
            self._PROGRESS_TOTAL,
            "Capturing the selected node and its current inputs.",
        )
        call = _prepared_call_from_detached_pipeline(
            pipeline,
            node_id,
            check_abort=check_abort,
        )
        workload = workload_from_prepared_node_call(
            call,
            check_abort=check_abort,
        )

        _emit_progress(
            progress,
            NodeBenchmarkPhase.ELIGIBILITY,
            1,
            self._PROGRESS_TOTAL,
            "Checking scientific, environment, and memory eligibility.",
        )
        facts = _complete_call_facts(
            call,
            workload.fingerprint,
            check_abort=check_abort,
        )
        all_specs = self.registry.implementations_for_operation(
            call.operation_id,
            allow_experimental=allow_experimental,
        )
        static_supported: list[OperationComputeSpec] = []
        decisions: dict[str, tuple[bool, str, str]] = {}
        for spec in all_specs:
            check_abort()
            support = evaluate_candidate_workload_support(
                spec,
                workload,
                array_facts=facts,
            )
            decisions[spec.implementation_id] = (
                support.supported,
                support.reason.value,
                support.reason_text,
            )
            if support.supported:
                static_supported.append(spec)

        selected_environment = environment
        if selected_environment is not None and not isinstance(
            selected_environment,
            ComputeEnvironment,
        ):
            raise TypeError("environment must be a ComputeEnvironment or None.")
        if selected_environment is None and static_supported:
            selected_environment, _warnings = probe_compute_environment(
                self.registry,
                ComputeRequest(
                    mode="custom",
                    device_id=str(device_id).strip(),
                    allow_experimental=allow_experimental,
                ),
                static_supported,
            )
            check_abort()
        if selected_environment is None:
            selected_environment = ComputeEnvironment()

        resolved_device = str(device_id).strip()
        if not resolved_device and selected_environment.device_id.startswith("cuda:"):
            resolved_device = selected_environment.device_id
        (
            effective_memory_limit,
            effective_safety_reserve,
            runtime_total_memory,
        ) = _resolved_runtime_memory_limits(
            self.registry,
            tuple(static_supported),
            device_id=resolved_device,
            memory_limit_bytes=memory_limit_bytes,
            safety_reserve_bytes=safety_reserve_bytes,
            check_abort=check_abort,
        )

        admitted: list[OperationComputeSpec] = []
        for spec in static_supported:
            check_abort()
            support = evaluate_candidate_support(
                spec,
                workload,
                selected_environment,
                allow_experimental=allow_experimental,
                array_facts=facts,
            )
            if support.supported:
                estimate = estimate_candidate_memory(spec, workload)
                support = evaluate_memory_support(
                    estimate,
                    memory_cap_bytes=effective_memory_limit,
                    total_device_bytes=runtime_total_memory,
                    safety_reserve_bytes=0,
                )
            decisions[spec.implementation_id] = (
                support.supported,
                support.reason.value,
                support.reason_text,
            )
            if support.supported:
                admitted.append(spec)

        eligibility = tuple(
            NodeBenchmarkCandidateEligibility(
                spec.implementation_id,
                spec.implementation_library_id,
                *decisions[spec.implementation_id],
            )
            for spec in all_specs
        )
        if not admitted:
            reasons = "; ".join(
                item.reason_text for item in eligibility if not item.supported
            )
            raise NodeBenchmarkUnavailable(
                reasons or "The selected node has no declared GPU benchmark candidate.",
                eligibility,
            )

        check_abort()
        registered = build_registered_node_benchmark(
            call,
            admitted_specs=tuple(admitted),
            registry=self.registry,
            environment_fingerprint=benchmark_environment_fingerprint(
                selected_environment
            ),
            device_id=resolved_device,
            memory_limit_bytes=effective_memory_limit,
            safety_reserve_bytes=effective_safety_reserve,
            warm_rounds=warm_rounds,
            max_warm_rounds=max_warm_rounds,
            time_budget_seconds=_remaining_budget(
                budget,
                started,
                self.clock,
            ),
            allow_experimental=allow_experimental,
            clock=self.clock,
            paired_bootstrap_samples=paired_bootstrap_samples,
            paired_bootstrap_seed=paired_bootstrap_seed,
            paired_confidence_level=paired_confidence_level,
            check_abort=check_abort,
            call_is_detached=True,
        )
        registered = replace(
            registered,
            request=replace(
                registered.request,
                adaptive_candidate_stopping=adaptive_candidate_stopping,
            ),
        )
        check_abort()
        preparation_seconds = _elapsed(self.clock, started)
        if budget is not None:
            remaining = _remaining_budget(budget, started, self.clock)
            registered = replace(
                registered,
                request=replace(
                    registered.request,
                    time_budget_seconds=remaining,
                ),
            )
        _emit_progress(
            progress,
            NodeBenchmarkPhase.READY,
            2,
            self._PROGRESS_TOTAL,
            "Benchmark input is detached and ready.",
        )
        return ApplicationNodeBenchmarkPlan(
            registered,
            selected_environment,
            tuple(admitted),
            eligibility,
            self.store.path,
            preparation_seconds,
        )

    def run(
        self,
        plan: ApplicationNodeBenchmarkPlan,
        *,
        cancelled: CancelCallback | None = None,
        progress: ProgressCallback | None = None,
    ) -> ApplicationNodeBenchmarkResult:
        """Run one prepared transaction and publish only its complete record."""

        if not isinstance(plan, ApplicationNodeBenchmarkPlan):
            raise TypeError("plan must be an ApplicationNodeBenchmarkPlan.")
        if plan.store_path != self.store.path:
            raise ValueError("plan belongs to a different local benchmark store.")
        _validate_callbacks(cancelled, progress)
        _check_cancelled(cancelled)
        _emit_progress(
            progress,
            NodeBenchmarkPhase.BENCHMARKING,
            3,
            self._PROGRESS_TOTAL,
            "Running parity checks and paired benchmark rounds.",
        )

        def forward_measurement(update: BenchmarkMeasurementProgress) -> None:
            phase = str(getattr(update.phase, "value", update.phase))
            operation_completed = int(getattr(update, "operation_completed", 0) or 0)
            operation_total = int(getattr(update, "operation_total", 0) or 0)
            operation_message = ""
            if operation_total:
                operation_message = _operation_progress_message(update)
            _emit_progress(
                progress,
                NodeBenchmarkPhase.BENCHMARKING,
                3,
                self._PROGRESS_TOTAL,
                "Running parity checks and paired benchmark rounds.",
                measurement_completed=update.completed,
                measurement_total=update.total,
                measurement_message=update.message,
                implementation_id=update.implementation_id,
                implementation_version=update.implementation_version,
                measurement_phase=phase,
                operation_completed=operation_completed,
                operation_total=operation_total,
                operation_message=operation_message,
            )

        request = plan.registered.request
        budget = _validated_budget(request.time_budget_seconds)
        lease_started = _read_clock(self.clock)
        deadline = None if budget is None else lease_started + budget

        def check_lease_abort() -> bool:
            _check_cancelled(cancelled)
            if deadline is not None and _read_clock(self.clock) >= deadline:
                raise BenchmarkBudgetExceeded(
                    "node benchmark budget was exhausted while waiting for "
                    "accelerator ownership"
                )
            return False

        runtime_ids = {spec.runtime_id for spec in plan.admitted_specs}
        if len(runtime_ids) != 1:
            raise ValueError("one node benchmark requires one accelerator runtime.")
        with accelerator_lease(
            next(iter(runtime_ids)),
            request.device_id,
            cancelled=check_lease_abort,
            deadline=deadline,
            clock=self.clock,
        ):
            remaining = _remaining_run_budget(deadline, self.clock)
            if remaining is not None:
                request = replace(request, time_budget_seconds=remaining)
            record = self.service.benchmark(
                request,
                cancelled=cancelled,
                progress=forward_measurement,
            )
        preference = plan.preference_for(record)
        _emit_progress(
            progress,
            NodeBenchmarkPhase.COMPLETE,
            4,
            self._PROGRESS_TOTAL,
            "Benchmark completed and local evidence was saved.",
        )
        return ApplicationNodeBenchmarkResult(plan, record, preference)

    def cached_result(
        self,
        plan: ApplicationNodeBenchmarkPlan,
    ) -> ApplicationNodeBenchmarkResult | None:
        """Return complete exact evidence, including rejected alternatives.

        A typed deterministic scientific-parity failure is a completed result,
        not a partial benchmark transaction. Keeping that result reusable lets
        the pipeline optimizer exclude only the rejected implementation without
        repeatedly timing the CPU and every other qualified alternative. Runtime,
        OOM, cleanup, and timing failures remain retryable cache misses.
        """

        if not isinstance(plan, ApplicationNodeBenchmarkPlan):
            raise TypeError("plan must be an ApplicationNodeBenchmarkPlan.")
        if plan.store_path != self.store.path:
            raise ValueError("plan belongs to a different local benchmark store.")
        request = plan.registered.request
        record = self.store.get(request.key)
        if record is None or not _record_is_complete_for_plan(record, plan):
            record = self._compatible_exact_record(plan)
            if record is None:
                return None
        try:
            preference = plan.preference_for(record)
        except (TypeError, ValueError):
            return None
        return ApplicationNodeBenchmarkResult(plan, record, preference)

    def _compatible_exact_record(
        self,
        plan: ApplicationNodeBenchmarkPlan,
    ) -> BenchmarkRecord | None:
        """Adapt strictly stronger legacy evidence to an adaptive plan.

        The compatibility direction is deliberately one-way.  An adaptive
        request may reuse a complete, uncensored record produced by the exact
        profile, while an exact request never looks at an adaptive record.  The
        returned record is an in-memory view keyed to the prepared plan; the
        durable source record remains under its original exact identity.
        """

        request = plan.registered.request
        if not request.adaptive_candidate_stopping:
            return None
        exact_key = request.exact_timing_compatible_key
        if exact_key == request.key:
            return None
        exact_request = replace(
            request,
            adaptive_candidate_stopping=False,
        )
        exact_plan = replace(
            plan,
            registered=replace(
                plan.registered,
                request=exact_request,
            ),
        )
        record = self.store.get(exact_key)
        if record is None or not _record_is_complete_for_plan(record, exact_plan):
            return None
        return replace(record, key=request.key)

    def benchmark(
        self,
        pipeline: PrototypePipeline,
        node_id: str,
        **kwargs,
    ) -> ApplicationNodeBenchmarkResult:
        """Convenience method for non-Qt callers that do not split worker phases."""

        cancelled = kwargs.get("cancelled")
        progress = kwargs.get("progress")
        plan = self.prepare(pipeline, node_id, **kwargs)
        return self.run(plan, cancelled=cancelled, progress=progress)

    def workload_is_current(
        self,
        pipeline: PrototypePipeline,
        plan: ApplicationNodeBenchmarkPlan,
        *,
        cancelled: CancelCallback | None = None,
    ) -> bool:
        """Return whether current data/parameters still match a completed plan."""

        if not isinstance(plan, ApplicationNodeBenchmarkPlan):
            raise TypeError("plan must be an ApplicationNodeBenchmarkPlan.")
        _validate_callbacks(cancelled, None)

        def check_abort() -> None:
            _check_cancelled(cancelled)

        check_abort()
        try:
            call = _prepared_call_from_detached_pipeline(
                pipeline,
                plan.node_id,
                check_abort=check_abort,
            )
        except NodeBenchmarkUnavailable:
            return False
        current = workload_from_prepared_node_call(
            call,
            check_abort=check_abort,
        )
        check_abort()
        return current.fingerprint == plan.workload_fingerprint


def stable_preference_for_benchmark_winner(
    record: BenchmarkRecord,
    admitted_specs: Sequence[OperationComputeSpec],
    *,
    cpu_implementation_id: str,
) -> NodeComputePreference:
    """Map a qualified winner to the least brittle portable preference."""

    if not isinstance(record, BenchmarkRecord):
        raise TypeError("record must be a BenchmarkRecord.")
    cpu_id = str(cpu_implementation_id).strip()
    if not cpu_id:
        raise ValueError("cpu_implementation_id must not be empty.")
    winner = str(record.accepted_implementation_id).strip()
    by_result = {item.implementation_id: item for item in record.candidates}
    result = by_result.get(winner)
    if result is None:
        raise ValueError("accepted implementation is absent from benchmark results.")
    if not result.parity_passed or result.error:
        raise ValueError("accepted implementation did not pass scientific parity.")
    if winner == cpu_id:
        return NodeComputePreference(NodePreferenceKind.CPU)

    specs = tuple(admitted_specs)
    matches = tuple(spec for spec in specs if spec.implementation_id == winner)
    if len(matches) != 1:
        raise ValueError("GPU benchmark winner has no unique admitted declaration.")
    selected = matches[0]
    same_library = tuple(
        spec
        for spec in specs
        if spec.implementation_library_id == selected.implementation_library_id
    )
    if len(same_library) == 1:
        return NodeComputePreference(
            NodePreferenceKind.LIBRARY,
            selected.implementation_library_id,
        )
    return NodeComputePreference(
        NodePreferenceKind.IMPLEMENTATION,
        selected.implementation_id,
    )


def _prepared_call_from_detached_pipeline(
    pipeline: PrototypePipeline,
    node_id: str,
    *,
    check_abort: Callable[[], None] | None = None,
) -> PreparedNodeCall:
    # Importing the operation catalog can cause optional scientific packages to
    # inspect installed array backends. Keep it behind the explicit benchmark
    # action so importing this coordinator remains accelerator-package safe.
    from napari_vipp.core.pipeline import PrototypePipeline

    if not isinstance(pipeline, PrototypePipeline):
        raise TypeError("pipeline must be a PrototypePipeline.")
    if check_abort is not None and not callable(check_abort):
        raise TypeError("check_abort must be callable or None.")
    if check_abort is not None:
        check_abort()
    selected = str(node_id).strip()
    if selected not in pipeline.nodes:
        raise NodeBenchmarkUnavailable(f"Unknown selected node {selected!r}.")
    if pipeline.nodes[selected].operation_id in BENCHMARK_WRITER_OPERATION_IDS:
        raise NodeBenchmarkUnavailable(
            "Selected-node benchmarking refuses writer operations."
        )
    detached = PrototypePipeline()
    detached.restore_graph(
        tuple(pipeline.nodes.values()),
        tuple(pipeline.connections),
        pipeline.output_tunnel_list(),
    )
    if check_abort is not None:
        check_abort()
    values_by_port = pipeline.input_data_by_port_for_node(selected)
    states_by_port = pipeline.input_states_by_port_for_node(selected)
    ports = tuple(sorted(values_by_port))
    if not ports or any(values_by_port[port] is None for port in ports):
        raise NodeBenchmarkUnavailable(
            "The selected node does not have resolved current inputs."
        )
    call = detached.prepare_node_call(
        selected,
        tuple(values_by_port[port] for port in ports),
        tuple(states_by_port.get(port) for port in ports),
    )
    if call is None:
        raise NodeBenchmarkUnavailable(
            "The selected node cannot be prepared from its current inputs."
        )
    if call.output_port_count != 1:
        raise NodeBenchmarkUnavailable(
            "Selected-node benchmarking requires exactly one output; "
            "multiple ordered inputs are supported."
        )
    return detach_prepared_node_call(call, check_abort=check_abort)


def _complete_call_facts(
    call: PreparedNodeCall,
    workload_fingerprint: str,
    *,
    check_abort: Callable[[], None],
) -> tuple[ArrayFacts, ...]:
    return tuple(
        _complete_array_facts(
            value,
            revision_fingerprint=f"{workload_fingerprint}:{index}",
            check_abort=check_abort,
        )
        for index, value in enumerate(call.inputs)
    )


def _complete_array_facts(
    value: object,
    *,
    revision_fingerprint: str,
    check_abort: Callable[[], None],
) -> ArrayFacts:
    array = np.asarray(value)
    check_abort()
    finite_count: int | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    guarantees: list[str] = []
    completeness = FactCompleteness.UNKNOWN
    numeric = (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or array.dtype == np.dtype(bool)
    )
    started = time.perf_counter()
    if numeric:
        completeness = FactCompleteness.COMPLETE
        finite_count = 0
        negative_zero = False
        iterator = np.nditer(
            array,
            flags=["buffered", "external_loop", "zerosize_ok"],
            op_flags=[["readonly"]],
            order="K",
            buffersize=262_144,
        )
        for raw_chunk in iterator:
            check_abort()
            chunk = np.asarray(raw_chunk)
            if np.issubdtype(array.dtype, np.floating):
                finite = np.isfinite(chunk)
                count = int(np.count_nonzero(finite))
                values = chunk if count == chunk.size else chunk[finite]
            else:
                count = int(chunk.size)
                values = chunk
            finite_count += count
            if count:
                low = values.min().item()
                high = values.max().item()
                if array.dtype == np.dtype(bool):
                    low, high = int(low), int(high)
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                if np.issubdtype(array.dtype, np.floating):
                    negative_zero = negative_zero or bool(
                        np.any((values == 0) & np.signbit(values))
                    )
        if not negative_zero:
            guarantees.append("no-negative-zero")
        if minimum is not None and minimum >= 0:
            guarantees.append("nonnegative")
    check_abort()
    return ArrayFacts(
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.name,
        element_count=int(array.size),
        revision_fingerprint=revision_fingerprint,
        completeness=completeness,
        finite_count=finite_count,
        minimum=minimum,
        maximum=maximum,
        strides=tuple(int(stride) for stride in array.strides),
        contiguous=bool(array.flags.c_contiguous),
        guarantees=tuple(guarantees),
        scan_seconds=max(0.0, time.perf_counter() - started),
    )


def _resolved_runtime_memory_limits(
    registry: ComputeRegistry,
    specs: tuple[OperationComputeSpec, ...],
    *,
    device_id: str,
    memory_limit_bytes: int | None,
    safety_reserve_bytes: int | None,
    check_abort: Callable[[], None],
) -> tuple[int | None, int | None, int]:
    """Mirror the CUDA runtime's default cap against current free memory."""
    if memory_limit_bytes is not None and (
        isinstance(memory_limit_bytes, bool)
        or not isinstance(memory_limit_bytes, int)
        or memory_limit_bytes <= 0
    ):
        raise ValueError("memory_limit_bytes must be positive or None.")
    if safety_reserve_bytes is not None and (
        isinstance(safety_reserve_bytes, bool)
        or not isinstance(safety_reserve_bytes, int)
        or safety_reserve_bytes < 0
    ):
        raise ValueError("safety_reserve_bytes must be non-negative or None.")
    if not specs:
        return memory_limit_bytes, safety_reserve_bytes, 0
    runtime_ids = {spec.runtime_id for spec in specs}
    if len(runtime_ids) != 1:
        raise NodeBenchmarkUnavailable(
            "Selected-node benchmarking requires candidates to share one runtime."
        )
    check_abort()
    runtime = registry.runtime(next(iter(runtime_ids)))
    try:
        snapshot = runtime.memory_snapshot(device_id=device_id)
    except Exception as exc:
        raise NodeBenchmarkUnavailable(
            f"Current accelerator memory could not be inspected: {exc}"
        ) from exc
    check_abort()
    total = int(snapshot.device_total_bytes)
    free = int(snapshot.device_free_bytes)
    if total <= 0 or free <= 0:
        raise NodeBenchmarkUnavailable(
            "Current accelerator free/total memory is unavailable."
        )
    reserve = (
        max(_DEFAULT_ACCELERATOR_RESERVE_BYTES, total // 10)
        if safety_reserve_bytes is None
        else safety_reserve_bytes
    )
    if free <= reserve:
        raise NodeBenchmarkUnavailable(
            "Accelerator free memory does not exceed the safety reserve."
        )
    requested_limit = (
        total * 80 // 100 if memory_limit_bytes is None else memory_limit_bytes
    )
    effective_limit = min(requested_limit, free - reserve)
    if effective_limit <= 0:
        raise NodeBenchmarkUnavailable(
            "No accelerator memory remains after the safety reserve."
        )
    return effective_limit, reserve, total


def _validate_callbacks(
    cancelled: CancelCallback | None,
    progress: ProgressCallback | None,
) -> None:
    if cancelled is not None and not callable(cancelled):
        raise TypeError("cancelled must be callable or None.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None.")


def _check_cancelled(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise BenchmarkCancelled("node benchmark cancelled")


def _operation_progress_message(update: BenchmarkMeasurementProgress) -> str:
    """Describe internal work together with the invocation that is repeating it."""

    implementation_id = str(update.implementation_id).strip()
    if implementation_id.startswith("cpu-"):
        implementation = "CPU"
    elif "cucim" in implementation_id.lower():
        implementation = "cuCIM GPU"
    elif "cupy" in implementation_id.lower():
        implementation = "CuPy GPU"
    else:
        implementation = implementation_id or "Current implementation"
    detail = str(getattr(update, "operation_message", "")).strip()
    completed = int(getattr(update, "operation_completed", 0) or 0)
    total = int(getattr(update, "operation_total", 0) or 0)
    measurement_completed = int(getattr(update, "completed", 0) or 0)
    measurement_total = int(getattr(update, "total", 0) or 0)
    raw_phase = getattr(update.phase, "value", update.phase)
    phase = str(raw_phase).strip()
    if phase == BenchmarkMeasurementPhase.PARITY_COLD.value:
        invocation = "scientific parity + cold timing"
    elif phase == BenchmarkMeasurementPhase.PARITY.value:
        invocation = "scientific parity"
    elif phase == BenchmarkMeasurementPhase.COLD.value:
        invocation = "cold timing"
    elif phase == BenchmarkMeasurementPhase.WARMUP.value:
        invocation = (
            f"warmup call {min(measurement_completed + 1, measurement_total)} "
            f"of {measurement_total}"
        )
    elif phase == BenchmarkMeasurementPhase.PAIRED_WARM.value:
        invocation = (
            "paired warm timing "
            f"{min(measurement_completed + 1, measurement_total)} "
            f"of {measurement_total}"
        )
    else:
        invocation = str(update.message).strip().rstrip(".")
    return (
        f"{implementation}: {detail} ({completed} of {total}) — "
        f"{invocation}."
    )


def _emit_progress(
    callback: ProgressCallback | None,
    phase: NodeBenchmarkPhase,
    completed: int,
    total: int,
    message: str,
    *,
    measurement_completed: int = 0,
    measurement_total: int = 0,
    measurement_message: str = "",
    implementation_id: str = "",
    implementation_version: str = "",
    measurement_phase: str = "",
    operation_completed: int = 0,
    operation_total: int = 0,
    operation_message: str = "",
) -> None:
    if callback is not None:
        callback(
            NodeBenchmarkProgress(
                phase=phase,
                completed=completed,
                total=total,
                message=message,
                measurement_completed=measurement_completed,
                measurement_total=measurement_total,
                measurement_message=measurement_message,
                implementation_id=implementation_id,
                implementation_version=implementation_version,
                measurement_phase=measurement_phase,
                operation_completed=operation_completed,
                operation_total=operation_total,
                operation_message=operation_message,
            )
        )


def _validated_budget(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("time_budget_seconds must be finite and positive.")
    return float(value)


def _check_preparation_budget(
    budget: float | None,
    started: float,
    clock: Callable[[], float],
) -> None:
    if budget is None:
        return
    if _elapsed(clock, started) >= budget:
        raise BenchmarkBudgetExceeded(
            "node benchmark budget was exhausted during preparation"
        )


def _remaining_budget(
    budget: float | None,
    started: float,
    clock: Callable[[], float],
) -> float | None:
    if budget is None:
        return None
    remaining = budget - _elapsed(clock, started)
    if remaining <= 0:
        raise BenchmarkBudgetExceeded(
            "node benchmark budget was exhausted during preparation"
        )
    return remaining


def _remaining_run_budget(
    deadline: float | None,
    clock: Callable[[], float],
) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - _read_clock(clock)
    if remaining <= 0:
        raise BenchmarkBudgetExceeded(
            "node benchmark budget was exhausted while waiting for "
            "accelerator ownership"
        )
    return remaining


def _read_clock(clock: Callable[[], float]) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("benchmark coordinator clock must return a finite value.")
    return float(value)


def _elapsed(clock: Callable[[], float], started: float) -> float:
    elapsed = _read_clock(clock) - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("benchmark coordinator clock must be monotonic.")
    return elapsed


__all__ = [
    "ApplicationNodeBenchmarkCoordinator",
    "ApplicationNodeBenchmarkPlan",
    "ApplicationNodeBenchmarkResult",
    "NodeBenchmarkCandidateEligibility",
    "NodeBenchmarkPhase",
    "NodeBenchmarkProgress",
    "NodeBenchmarkUnavailable",
    "benchmark_environment_fingerprint",
    "stable_preference_for_benchmark_winner",
]
