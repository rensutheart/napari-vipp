"""Qt-free compute admission and per-node decision planning.

This module turns prepared workload descriptors into typed CPU/GPU decisions.
It deliberately does not import the pipeline or any optional accelerator.  A
CPU request returns before a default registry is even constructed; non-CPU
requests probe only runtimes and implementation libraries relevant to their
visible candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionPlan,
    ExecutionSegment,
    FallbackPolicy,
    FallbackReason,
    MemoryEstimate,
    MemoryTopology,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    PerformanceEvidence,
    SupportDecision,
    estimate_candidate_memory,
    evaluate_auto_performance,
    evaluate_candidate_support,
    evaluate_memory_support,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import OperationComputeSpec

CPU_RUNTIME_ID = "cpu-numpy"
CPU_LIBRARY_ID = "cpu"


@dataclass(frozen=True, slots=True)
class ComputePreflightFailure:
    """One node-local planning failure collected before execution."""

    node_id: str
    operation_id: str
    preference: NodeComputePreference
    reason: DecisionReason
    reason_text: str


class ComputePreflightError(RuntimeError):
    """Raised after collecting invalid or unhonored strict preferences."""

    def __init__(self, failures: Sequence[ComputePreflightFailure]) -> None:
        self.failures = tuple(failures)
        if not self.failures:
            raise ValueError("ComputePreflightError requires at least one failure.")
        summary = "; ".join(
            f"{item.node_id}: {item.reason_text}" for item in self.failures
        )
        super().__init__(f"Compute preflight failed: {summary}")


@dataclass(frozen=True, slots=True)
class ComputePlanningResult:
    """Environment, ordered node decisions, and visible planning warnings."""

    request: ComputeRequest
    environment: ComputeEnvironment
    decisions: tuple[NodeExecutionDecision, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        node_ids = tuple(decision.node_id for decision in decisions)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Planning decisions must have unique node IDs.")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "warnings",
            tuple(str(value).strip() for value in self.warnings if str(value).strip()),
        )

    @property
    def decisions_by_node(self) -> Mapping[str, NodeExecutionDecision]:
        return MappingProxyType(
            {decision.node_id: decision for decision in self.decisions}
        )

    def as_execution_plan(
        self,
        segments: Sequence[ExecutionSegment] = (),
    ) -> ExecutionPlan:
        """Build the public plan shell after graph partitioning supplies segments."""

        return ExecutionPlan(
            self.request.fingerprint,
            self.environment.fingerprint,
            tuple(segments),
            self.decisions,
            self.warnings,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    spec: OperationComputeSpec
    memory_estimate: MemoryEstimate
    evidence: PerformanceEvidence | None


def plan_compute_decisions(
    request: ComputeRequest,
    workloads: Sequence[WorkloadDescriptor],
    *,
    registry: ComputeRegistry | None = None,
    environment: ComputeEnvironment | None = None,
    array_facts: Mapping[str, tuple[ArrayFacts, ...]] | None = None,
    performance_evidence: Mapping[
        tuple[str, str], PerformanceEvidence
    ] | None = None,
) -> ComputePlanningResult:
    """Resolve CPU/Auto/Selective intent for prepared node workloads.

    ``performance_evidence`` is keyed by ``(node_id, implementation_id)``.
    Selective library/exact pins bypass the Auto speed threshold, but never the
    scientific, environment, or memory gates.  A strict forced preference is
    collected into :class:`ComputePreflightError`; visible fallback returns an
    explicit typed CPU decision and warning.
    """

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    prepared = tuple(workloads)
    if any(not isinstance(item, WorkloadDescriptor) for item in prepared):
        raise TypeError("workloads must contain WorkloadDescriptor values.")
    node_ids = tuple(item.node_id for item in prepared)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("workload node IDs must be unique.")
    facts_by_node = dict(array_facts or {})
    evidence_by_candidate = dict(performance_evidence or {})
    unknown_facts = set(facts_by_node) - set(node_ids)
    if unknown_facts:
        names = ", ".join(sorted(unknown_facts))
        raise ValueError(f"Array facts reference unscheduled node(s): {names}.")

    # The CPU fast path intentionally precedes default registry construction,
    # descriptor lookup, runtime probing, and optional-library discovery.
    if request.mode is ComputeMode.CPU:
        resolved_environment = environment or ComputeEnvironment()
        return ComputePlanningResult(
            request,
            resolved_environment,
            tuple(
                _cpu_decision(
                    workload,
                    request.preference_for(workload.node_id),
                    reason=DecisionReason.EXPLICIT_CPU,
                    reason_text="The global compute mode requires authoritative CPU.",
                )
                for workload in prepared
            ),
        )

    selected_registry = registry or ComputeRegistry()
    owns_registry = registry is None
    try:
        potential = _potential_specs(selected_registry, request, prepared)
        if environment is None:
            resolved_environment, probe_warnings = _probe_environment(
                selected_registry,
                request,
                potential,
            )
        else:
            resolved_environment = environment
            probe_warnings = ()

        warnings = list(probe_warnings)
        decisions: list[NodeExecutionDecision] = []
        failures: list[ComputePreflightFailure] = []
        for workload in prepared:
            preference = (
                request.preference_for(workload.node_id)
                if request.mode is ComputeMode.SELECTIVE
                else NodeComputePreference(NodePreferenceKind.AUTO)
            )
            if (
                request.mode is ComputeMode.SELECTIVE
                and preference.kind is NodePreferenceKind.CPU
            ):
                decisions.append(
                    _cpu_decision(
                        workload,
                        preference,
                        reason=DecisionReason.EXPLICIT_CPU,
                        reason_text="The node preference requires authoritative CPU.",
                    )
                )
                continue

            specs = _specs_for_preference(
                selected_registry,
                request,
                workload.operation_id,
                preference,
            )
            candidates, rejections = _admit_candidates(
                request,
                workload,
                specs,
                resolved_environment,
                facts_by_node.get(workload.node_id, ()),
                evidence_by_candidate,
            )
            selected, selection_rejection = _select_candidate(
                request,
                preference,
                candidates,
            )
            if selected is not None:
                decisions.append(
                    _selected_decision(workload, preference, selected, request.mode)
                )
                continue

            rejection = selection_rejection or _preferred_rejection(
                rejections,
                request=request,
                preference=preference,
            )
            forced = _is_forced_gpu_preference(request.mode, preference)
            failure = ComputePreflightFailure(
                workload.node_id,
                workload.operation_id,
                preference,
                rejection.reason,
                rejection.reason_text,
            )
            if not rejection.fallback_allowed or (
                forced and request.fallback_policy is FallbackPolicy.STRICT
            ):
                failures.append(failure)
                continue
            if forced:
                fallback_reason = _fallback_reason(rejection.reason)
                warning = (
                    f"Node {workload.node_id!r} used visible CPU fallback: "
                    f"{rejection.reason_text}"
                )
                warnings.append(warning)
                decisions.append(
                    _cpu_decision(
                        workload,
                        preference,
                        reason=DecisionReason.VISIBLE_FALLBACK,
                        reason_text=warning,
                        decision_kind=DecisionKind.FALLBACK_CPU,
                        fallback_reason=fallback_reason,
                    )
                )
            else:
                decisions.append(
                    _cpu_decision(
                        workload,
                        preference,
                        reason=rejection.reason,
                        reason_text=rejection.reason_text,
                    )
                )
        if failures:
            raise ComputePreflightError(failures)
        return ComputePlanningResult(
            request,
            resolved_environment,
            tuple(decisions),
            tuple(warnings),
        )
    finally:
        if owns_registry:
            selected_registry.close()


def actual_cpu_fallback_decision(
    decision: NodeExecutionDecision,
    fallback_reason: FallbackReason,
    *,
    reason_text: str,
) -> NodeExecutionDecision:
    """Convert one planned GPU decision into its actual CPU fallback record."""

    if not isinstance(decision, NodeExecutionDecision):
        raise TypeError("decision must be a NodeExecutionDecision.")
    reason = {
        FallbackReason.ENVIRONMENT_UNSUPPORTED: DecisionReason.ENVIRONMENT_UNSUPPORTED,
        FallbackReason.DEPENDENCY_UNAVAILABLE: DecisionReason.DEPENDENCY_UNAVAILABLE,
        FallbackReason.WORKLOAD_UNSUPPORTED: DecisionReason.WORKLOAD_UNSUPPORTED,
        FallbackReason.MEMORY_LIMIT: DecisionReason.MEMORY_LIMIT,
        FallbackReason.OUT_OF_MEMORY: DecisionReason.OUT_OF_MEMORY_FALLBACK,
        FallbackReason.RUNTIME_FAILURE: DecisionReason.VISIBLE_FALLBACK,
    }.get(fallback_reason)
    if reason is None or fallback_reason is FallbackReason.NONE:
        raise ValueError("fallback_reason must describe an actual CPU fallback.")
    text = str(reason_text).strip()
    if not text:
        raise ValueError("reason_text must not be empty.")
    return NodeExecutionDecision(
        decision.node_id,
        decision.operation_id,
        decision.requested_preference,
        CPU_RUNTIME_ID,
        CPU_LIBRARY_ID,
        _cpu_implementation_id(decision.operation_id),
        DecisionKind.FALLBACK_CPU,
        reason,
        text,
        fallback_reason=fallback_reason,
        benchmark_record_digest=decision.benchmark_record_digest,
    )


def _potential_specs(
    registry: ComputeRegistry,
    request: ComputeRequest,
    workloads: tuple[WorkloadDescriptor, ...],
) -> tuple[OperationComputeSpec, ...]:
    selected: list[OperationComputeSpec] = []
    identities: set[tuple[str, str]] = set()
    for workload in workloads:
        preference = (
            request.preference_for(workload.node_id)
            if request.mode is ComputeMode.SELECTIVE
            else NodeComputePreference(NodePreferenceKind.AUTO)
        )
        if preference.kind is NodePreferenceKind.CPU:
            continue
        for spec in _specs_for_preference(
            registry,
            request,
            workload.operation_id,
            preference,
        ):
            identity = (spec.implementation_id, spec.implementation_version)
            if identity not in identities:
                selected.append(spec)
                identities.add(identity)
    return tuple(selected)


def _specs_for_preference(
    registry: ComputeRegistry,
    request: ComputeRequest,
    operation_id: str,
    preference: NodeComputePreference,
) -> tuple[OperationComputeSpec, ...]:
    specs = registry.implementations_for_operation(
        operation_id,
        allow_experimental=request.allow_experimental,
    )
    if request.runtime_id:
        specs = tuple(spec for spec in specs if spec.runtime_id == request.runtime_id)
    if preference.kind is NodePreferenceKind.LIBRARY:
        specs = tuple(
            spec
            for spec in specs
            if spec.implementation_library_id == preference.value
        )
    elif preference.kind is NodePreferenceKind.IMPLEMENTATION:
        specs = tuple(
            spec for spec in specs if spec.implementation_id == preference.value
        )
    return specs


def _probe_environment(
    registry: ComputeRegistry,
    request: ComputeRequest,
    specs: tuple[OperationComputeSpec, ...],
) -> tuple[ComputeEnvironment, tuple[str, ...]]:
    base = ComputeEnvironment()
    runtime_ids: list[str] = [CPU_RUNTIME_ID]
    library_ids: list[str] = [CPU_LIBRARY_ID]
    versions: list[tuple[str, str]] = []
    warnings: list[str] = []
    selected_device = None

    relevant_runtimes = tuple(dict.fromkeys(spec.runtime_id for spec in specs))
    relevant_libraries = tuple(
        dict.fromkeys(spec.implementation_library_id for spec in specs)
    )
    for runtime_id in relevant_runtimes:
        descriptor = registry.runtime_descriptor(runtime_id)
        if base.os_name not in descriptor.supported_os_families:
            warnings.append(
                f"Runtime {runtime_id!r} is not validated on {base.os_name}."
            )
            continue
        probe = registry.probe_runtime(runtime_id)
        if not probe.available:
            warnings.append(
                probe.message
                or f"Runtime {runtime_id!r} is unavailable ({probe.reason_code})."
            )
            continue
        device = None
        if request.device_id:
            device = next(
                (item for item in probe.devices if item.device_id == request.device_id),
                None,
            )
            if device is None:
                warnings.append(
                    f"Runtime {runtime_id!r} did not report requested device "
                    f"{request.device_id!r}."
                )
                continue
        elif probe.selected_device_id:
            device = next(
                (
                    item
                    for item in probe.devices
                    if item.device_id == probe.selected_device_id
                ),
                None,
            )
        elif probe.devices:
            device = probe.devices[0]
        runtime_ids.append(runtime_id)
        if probe.version:
            versions.append((runtime_id, probe.version))
        if selected_device is None:
            selected_device = device

    for library_id in relevant_libraries:
        descriptor = registry.library_descriptor(library_id)
        if base.os_name not in descriptor.supported_os_families:
            warnings.append(
                f"Implementation library {library_id!r} is not validated on "
                f"{base.os_name}."
            )
            continue
        if not any(runtime_id in runtime_ids for runtime_id in descriptor.runtime_ids):
            continue
        probe = registry.probe_library(library_id)
        if not probe.available:
            warnings.append(
                probe.message
                or f"Implementation library {library_id!r} is unavailable "
                f"({probe.reason_code})."
            )
            continue
        library_ids.append(library_id)
        if probe.version:
            versions.append((library_id, probe.version))

    available = len(runtime_ids) > 1
    return (
        replace(
            base,
            runtime_ids=tuple(runtime_ids),
            implementation_libraries=tuple(library_ids),
            runtime_versions=tuple(versions),
            driver_version="",
            device_id=(
                selected_device.device_id if selected_device is not None else "cpu:0"
            ),
            device_name=(
                selected_device.display_name
                if selected_device is not None
                else "Host CPU"
            ),
            device_class=("nvidia-cuda" if available else "host"),
            memory_topology=(
                MemoryTopology.DISCRETE if available else MemoryTopology.HOST
            ),
            total_accelerator_memory_bytes=(
                selected_device.total_memory_bytes or 0
                if selected_device is not None
                else 0
            ),
            probe_status="available" if available else "unavailable",
            probe_reason="; ".join(warnings),
        ),
        tuple(warnings),
    )


def _admit_candidates(
    request: ComputeRequest,
    workload: WorkloadDescriptor,
    specs: tuple[OperationComputeSpec, ...],
    environment: ComputeEnvironment,
    facts: tuple[ArrayFacts, ...],
    evidence: Mapping[tuple[str, str], PerformanceEvidence],
) -> tuple[tuple[_Candidate, ...], tuple[SupportDecision, ...]]:
    candidates: list[_Candidate] = []
    rejections: list[SupportDecision] = []
    for spec in specs:
        if spec.runtime_id not in environment.runtime_ids:
            rejections.append(
                SupportDecision(
                    False,
                    DecisionReason.ENVIRONMENT_UNSUPPORTED,
                    f"Runtime {spec.runtime_id!r} is unavailable in this environment.",
                )
            )
            continue
        if spec.implementation_library_id not in environment.implementation_libraries:
            rejections.append(
                SupportDecision(
                    False,
                    DecisionReason.DEPENDENCY_UNAVAILABLE,
                    f"Implementation library {spec.implementation_library_id!r} "
                    "is unavailable in this environment.",
                )
            )
            continue
        support = evaluate_candidate_support(
            spec,
            workload,
            environment,
            allow_experimental=request.allow_experimental,
            array_facts=facts,
        )
        if not support.supported:
            rejections.append(support)
            continue
        try:
            memory = estimate_candidate_memory(spec, workload)
        except (TypeError, ValueError) as exc:
            rejections.append(
                SupportDecision(
                    False,
                    DecisionReason.MEMORY_LIMIT,
                    f"No safe memory bound is available: {exc}",
                )
            )
            continue
        memory_support = evaluate_memory_support(
            memory,
            memory_cap_bytes=request.accelerator_memory_cap_bytes,
            total_device_bytes=environment.total_accelerator_memory_bytes,
            safety_reserve_bytes=request.accelerator_safety_reserve_bytes or 0,
        )
        if not memory_support.supported:
            rejections.append(memory_support)
            continue
        candidates.append(
            _Candidate(
                spec,
                memory,
                evidence.get((workload.node_id, spec.implementation_id)),
            )
        )
    return tuple(candidates), tuple(rejections)


def _select_candidate(
    request: ComputeRequest,
    preference: NodeComputePreference,
    candidates: tuple[_Candidate, ...],
) -> tuple[_Candidate | None, SupportDecision | None]:
    if not candidates:
        return None, None
    forced = _is_forced_gpu_preference(request.mode, preference)
    if forced:
        if len(candidates) == 1:
            return candidates[0], None
        measured = tuple(candidate for candidate in candidates if candidate.evidence)
        if len(measured) != len(candidates):
            return None, SupportDecision(
                False,
                DecisionReason.PERFORMANCE_GATE,
                "Best-GPU/library selection has multiple validated candidates but "
                "does not yet have comparable evidence for all of them.",
            )
        return min(
            measured,
            key=lambda candidate: (
                candidate.evidence.end_to_end_candidate_seconds,
                candidate.spec.implementation_id,
            ),
        ), None

    accepted = []
    for candidate in candidates:
        if candidate.evidence is None:
            continue
        performance = evaluate_auto_performance(candidate.evidence)
        if performance.select_candidate:
            accepted.append(candidate)
    if not accepted:
        return None, SupportDecision(
            False,
            DecisionReason.PERFORMANCE_GATE,
            "No validated candidate clears the conservative Auto performance gate.",
        )
    return min(
        accepted,
        key=lambda candidate: (
            candidate.evidence.end_to_end_candidate_seconds,
            candidate.spec.implementation_id,
        ),
    ), None


def _preferred_rejection(
    rejections: tuple[SupportDecision, ...],
    *,
    request: ComputeRequest,
    preference: NodeComputePreference,
) -> SupportDecision:
    if rejections:
        priority = {
            DecisionReason.WORKLOAD_UNSUPPORTED: 0,
            DecisionReason.MEMORY_LIMIT: 1,
            DecisionReason.DEPENDENCY_UNAVAILABLE: 2,
            DecisionReason.ENVIRONMENT_UNSUPPORTED: 3,
        }
        return min(
            rejections,
            key=lambda rejection: (
                0 if not rejection.fallback_allowed else 1,
                priority.get(rejection.reason, 10),
            ),
        )
    if request.runtime_id:
        return SupportDecision(
            False,
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            f"No visible implementation uses requested runtime {request.runtime_id!r}.",
        )
    if preference.kind is NodePreferenceKind.LIBRARY:
        return SupportDecision(
            False,
            DecisionReason.DEPENDENCY_UNAVAILABLE,
            "No visible implementation is available from library "
            f"{preference.value!r}.",
        )
    if preference.kind is NodePreferenceKind.IMPLEMENTATION:
        return SupportDecision(
            False,
            DecisionReason.DEPENDENCY_UNAVAILABLE,
            f"Exact implementation {preference.value!r} is unavailable.",
        )
    return SupportDecision(
        False,
        DecisionReason.NO_VALIDATED_IMPLEMENTATION,
        "No validated accelerator implementation is visible for this operation.",
    )


def _selected_decision(
    workload: WorkloadDescriptor,
    preference: NodeComputePreference,
    candidate: _Candidate,
    mode: ComputeMode,
) -> NodeExecutionDecision:
    if mode is ComputeMode.SELECTIVE and preference.kind not in {
        NodePreferenceKind.AUTO,
        NodePreferenceKind.CPU,
    }:
        reason_text = (
            f"The validated {preference.kind.value} preference selected "
            f"{candidate.spec.implementation_id!r}."
        )
    else:
        reason_text = (
            f"{candidate.spec.implementation_id!r} clears scientific, memory, "
            "and Auto performance policy."
        )
    return NodeExecutionDecision(
        workload.node_id,
        workload.operation_id,
        preference,
        candidate.spec.runtime_id,
        candidate.spec.implementation_library_id,
        candidate.spec.implementation_id,
        DecisionKind.SELECTED,
        DecisionReason.SELECTED_IMPLEMENTATION,
        reason_text,
        memory_estimate=candidate.memory_estimate,
    )


def _cpu_decision(
    workload: WorkloadDescriptor,
    preference: NodeComputePreference,
    *,
    reason: DecisionReason,
    reason_text: str,
    decision_kind: DecisionKind = DecisionKind.POLICY_CPU,
    fallback_reason: FallbackReason = FallbackReason.NONE,
) -> NodeExecutionDecision:
    return NodeExecutionDecision(
        workload.node_id,
        workload.operation_id,
        preference,
        CPU_RUNTIME_ID,
        CPU_LIBRARY_ID,
        _cpu_implementation_id(workload.operation_id),
        decision_kind,
        reason,
        reason_text,
        fallback_reason=fallback_reason,
    )


def _cpu_implementation_id(operation_id: str) -> str:
    return f"cpu-{operation_id}-v1"


def _is_forced_gpu_preference(
    mode: ComputeMode,
    preference: NodeComputePreference,
) -> bool:
    return mode is ComputeMode.SELECTIVE and preference.kind in {
        NodePreferenceKind.BEST_GPU,
        NodePreferenceKind.LIBRARY,
        NodePreferenceKind.IMPLEMENTATION,
    }


def _fallback_reason(reason: DecisionReason) -> FallbackReason:
    return {
        DecisionReason.ENVIRONMENT_UNSUPPORTED: (
            FallbackReason.ENVIRONMENT_UNSUPPORTED
        ),
        DecisionReason.DEPENDENCY_UNAVAILABLE: FallbackReason.DEPENDENCY_UNAVAILABLE,
        DecisionReason.MEMORY_LIMIT: FallbackReason.MEMORY_LIMIT,
        DecisionReason.OUT_OF_MEMORY_FALLBACK: FallbackReason.OUT_OF_MEMORY,
    }.get(reason, FallbackReason.WORKLOAD_UNSUPPORTED)


__all__ = [
    "CPU_LIBRARY_ID",
    "CPU_RUNTIME_ID",
    "ComputePlanningResult",
    "ComputePreflightError",
    "ComputePreflightFailure",
    "actual_cpu_fallback_decision",
    "plan_compute_decisions",
]
