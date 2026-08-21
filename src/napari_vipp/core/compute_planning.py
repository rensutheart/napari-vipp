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
    ComputeRepairSuggestion,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExactWorkloadCandidateQualification,
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
    exact_workload_identity_digest,
)
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    PerformanceEvidence,
    SupportDecision,
    _evaluate_phase1_cuda_host_environment,
    estimate_candidate_memory,
    evaluate_auto_performance,
    evaluate_candidate_exact_workload_test_support,
    evaluate_candidate_support,
    evaluate_candidate_workload_support,
    evaluate_memory_support,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_repairs import suggest_compute_repairs
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
    repair_suggestions: tuple[ComputeRepairSuggestion, ...] = ()

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
        suggestions = tuple(self.repair_suggestions)
        if any(
            not isinstance(suggestion, ComputeRepairSuggestion)
            for suggestion in suggestions
        ):
            raise TypeError(
                "repair_suggestions must contain ComputeRepairSuggestion values."
            )
        object.__setattr__(self, "repair_suggestions", suggestions)

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
            self.repair_suggestions,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    spec: OperationComputeSpec
    memory_estimate: MemoryEstimate
    evidence: PerformanceEvidence | None
    exact_workload_qualification: ExactWorkloadCandidateQualification | None = None


def plan_compute_decisions(
    request: ComputeRequest,
    workloads: Sequence[WorkloadDescriptor],
    *,
    registry: ComputeRegistry | None = None,
    environment: ComputeEnvironment | None = None,
    array_facts: Mapping[str, tuple[ArrayFacts, ...]] | None = None,
    performance_evidence: Mapping[tuple[str, str], PerformanceEvidence] | None = None,
    exact_workload_qualifications: frozenset[
        ExactWorkloadCandidateQualification
    ] = frozenset(),
    exact_workload_qualification_scope_digest: str = "",
) -> ComputePlanningResult:
    """Resolve CPU/Auto/Prefer-GPU/Custom intent for prepared workloads.

    ``performance_evidence`` is keyed by ``(node_id, implementation_id)``.
    Custom library/exact pins bypass the Auto speed threshold, but never the
    scientific, environment, or memory gates. Prefer GPU applies the same
    safety gates to every visible GPU candidate while deliberately ignoring the
    CPU-versus-GPU speed threshold. A strict forced preference is collected
    into :class:`ComputePreflightError`; visible fallback returns an explicit
    typed CPU decision and warning.
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
    qualifications = frozenset(exact_workload_qualifications)
    if any(
        not isinstance(item, ExactWorkloadCandidateQualification)
        for item in qualifications
    ):
        raise TypeError(
            "exact_workload_qualifications must contain "
            "ExactWorkloadCandidateQualification values."
        )
    qualification_scope = str(exact_workload_qualification_scope_digest).strip()
    if qualifications and not qualification_scope:
        raise ValueError(
            "exact_workload_qualification_scope_digest is required when exact "
            "workload qualifications are supplied."
        )
    if any(
        item.qualification_scope_digest != qualification_scope
        for item in qualifications
    ):
        raise ValueError(
            "exact workload qualifications must match the active scope digest."
        )
    qualification_keys = tuple(item.candidate_key for item in qualifications)
    if len(set(qualification_keys)) != len(qualification_keys):
        raise ValueError(
            "exact_workload_qualifications contain duplicate candidate identities."
        )
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

    if (
        environment is not None
        and request.runtime_id
        and request.device_id
        and request.runtime_id in environment.runtime_ids
        and environment.device_id != request.device_id
    ):
        raise ValueError(
            "The supplied compute environment describes device "
            f"{environment.device_id!r}, but the request requires "
            f"{request.device_id!r} for runtime {request.runtime_id!r}."
        )

    selected_registry = registry or ComputeRegistry()
    owns_registry = registry is None
    try:
        potential = _potential_specs(selected_registry, request, prepared)
        if environment is None:
            resolved_environment, probe_warnings = probe_compute_environment(
                selected_registry,
                request,
                potential,
            )
        else:
            resolved_environment = environment
            probe_warnings = (
                (resolved_environment.probe_reason,)
                if resolved_environment.probe_reason
                else ()
            )

        warnings = list(probe_warnings)
        decisions: list[NodeExecutionDecision] = []
        failures: list[ComputePreflightFailure] = []
        for workload in prepared:
            preference = (
                request.preference_for(workload.node_id)
                if request.mode is ComputeMode.CUSTOM
                else NodeComputePreference(NodePreferenceKind.AUTO)
            )
            if (
                request.mode is ComputeMode.CUSTOM
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

            if not workload.inputs_resolved:
                decisions.append(
                    _cpu_decision(
                        workload,
                        preference,
                        reason=DecisionReason.WORKLOAD_UNSUPPORTED,
                        reason_text=(
                            "Accelerator planning was deferred because an upstream "
                            "output could not be resolved before execution. The "
                            "authoritative CPU path will report any operation error."
                        ),
                    )
                )
                continue

            specs = _specs_for_preference(
                selected_registry,
                request,
                workload.operation_id,
                preference,
            )
            specs, residency_rejections = _residency_aware_specs(
                selected_registry,
                request,
                preference,
                workload,
                specs,
                decisions,
                facts_by_node.get(workload.node_id, ()),
            )
            candidates, rejections = _admit_candidates(
                request,
                preference,
                workload,
                specs,
                resolved_environment,
                facts_by_node.get(workload.node_id, ()),
                evidence_by_candidate,
                qualifications,
                qualification_scope,
            )
            rejections = (*residency_rejections, *rejections)
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
        decisions_by_node = {decision.node_id: decision for decision in decisions}
        repair_suggestions = tuple(
            suggestion
            for workload in prepared
            if decisions_by_node[workload.node_id].runtime_id == CPU_RUNTIME_ID
            for suggestion in suggest_compute_repairs(
                request,
                workload,
                selected_registry,
                resolved_environment,
                array_facts=facts_by_node.get(workload.node_id, ()),
                performance_evidence=evidence_by_candidate,
            )
        )
        return ComputePlanningResult(
            request,
            resolved_environment,
            tuple(decisions),
            tuple(warnings),
            repair_suggestions,
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
        performance_evidence_kind=decision.performance_evidence_kind,
        performance_evidence_digest=decision.performance_evidence_digest,
        implementation_version="1",
    )


def _potential_specs(
    registry: ComputeRegistry,
    request: ComputeRequest,
    workloads: tuple[WorkloadDescriptor, ...],
) -> tuple[OperationComputeSpec, ...]:
    selected: list[OperationComputeSpec] = []
    identities: set[tuple[str, str]] = set()
    for workload in workloads:
        if not workload.inputs_resolved:
            continue
        preference = (
            request.preference_for(workload.node_id)
            if request.mode is ComputeMode.CUSTOM
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
    automatic_intent = request.mode is ComputeMode.AUTO or (
        request.mode is ComputeMode.CUSTOM
        and preference.kind is NodePreferenceKind.AUTO
    )
    if automatic_intent:
        specs = tuple(
            spec
            for spec in specs
            if spec.eligible_for_auto(
                allow_experimental=request.allow_experimental,
            )
        )
    if request.runtime_id:
        specs = tuple(spec for spec in specs if spec.runtime_id == request.runtime_id)
    if preference.kind is NodePreferenceKind.LIBRARY:
        specs = tuple(
            spec for spec in specs if spec.implementation_library_id == preference.value
        )
    elif preference.kind is NodePreferenceKind.IMPLEMENTATION:
        specs = tuple(
            spec for spec in specs if spec.implementation_id == preference.value
        )
    return specs


def probe_compute_environment(
    registry: ComputeRegistry,
    request: ComputeRequest,
    specs: Sequence[OperationComputeSpec],
) -> tuple[ComputeEnvironment, tuple[str, ...]]:
    """Probe only candidate providers and preserve their exact provenance.

    The returned environment can be passed back to
    :func:`plan_compute_decisions` to avoid a second probe.  No provider is
    imported until the caller explicitly invokes this function.
    """

    if not isinstance(registry, ComputeRegistry):
        raise TypeError("registry must be a ComputeRegistry.")
    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    selected_specs = tuple(specs)
    if any(not isinstance(spec, OperationComputeSpec) for spec in selected_specs):
        raise TypeError("specs must contain OperationComputeSpec values.")

    base = ComputeEnvironment()
    if not selected_specs:
        return base, ()

    warnings: list[str] = []
    probe_specs: list[OperationComputeSpec] = []
    for spec in selected_specs:
        host_decision = _evaluate_phase1_cuda_host_environment(spec, base)
        if host_decision is None:
            probe_specs.append(spec)
            continue
        if host_decision.reason_text not in warnings:
            warnings.append(host_decision.reason_text)
    if not probe_specs:
        return replace(base, probe_reason="; ".join(warnings)), tuple(warnings)

    runtime_ids: list[str] = [CPU_RUNTIME_ID]
    library_ids: list[str] = [CPU_LIBRARY_ID]
    versions: list[tuple[str, str]] = []
    runtime_fingerprints: list[tuple[str, str]] = []
    runtime_metadata: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    library_metadata: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    selected_device = None
    selected_runtime_metadata: tuple[tuple[str, str], ...] = ()

    relevant_runtimes = tuple(dict.fromkeys(spec.runtime_id for spec in probe_specs))
    relevant_libraries = tuple(
        dict.fromkeys(spec.implementation_library_id for spec in probe_specs)
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
        if probe.environment_fingerprint:
            runtime_fingerprints.append((runtime_id, probe.environment_fingerprint))
        runtime_metadata.append((runtime_id, probe.metadata))
        if selected_device is None:
            selected_device = device
            selected_runtime_metadata = probe.metadata

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
        library_metadata.append((library_id, probe.metadata))

    available = len(runtime_ids) > 1
    selected_runtime_values = dict(selected_runtime_metadata)
    return (
        replace(
            base,
            runtime_ids=tuple(runtime_ids),
            implementation_libraries=tuple(library_ids),
            runtime_versions=tuple(versions),
            runtime_probe_fingerprints=tuple(runtime_fingerprints),
            runtime_metadata=tuple(runtime_metadata),
            implementation_library_metadata=tuple(library_metadata),
            driver_version=selected_runtime_values.get("driver_version", ""),
            device_id=(
                selected_device.device_id if selected_device is not None else "cpu:0"
            ),
            device_name=(
                selected_device.display_name
                if selected_device is not None
                else "Host CPU"
            ),
            device_class=("nvidia-cuda" if available else "host"),
            device_metadata=(
                selected_device.metadata if selected_device is not None else ()
            ),
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
    preference: NodeComputePreference,
    workload: WorkloadDescriptor,
    specs: tuple[OperationComputeSpec, ...],
    environment: ComputeEnvironment,
    facts: tuple[ArrayFacts, ...],
    evidence: Mapping[tuple[str, str], PerformanceEvidence],
    exact_workload_qualifications: frozenset[ExactWorkloadCandidateQualification],
    exact_workload_qualification_scope_digest: str,
) -> tuple[tuple[_Candidate, ...], tuple[SupportDecision, ...]]:
    candidates: list[_Candidate] = []
    rejections: list[SupportDecision] = []
    forced = _is_forced_gpu_preference(request.mode, preference)
    prefer_gpu = request.mode is ComputeMode.PREFER_GPU
    for spec in specs:
        static_support = evaluate_candidate_workload_support(
            spec,
            workload,
            array_facts=(),
        )
        if not static_support.supported and not static_support.requires_complete_facts:
            rejections.append(static_support)
            continue
        host_support = _evaluate_phase1_cuda_host_environment(spec, environment)
        if host_support is not None:
            rejections.append(host_support)
            continue
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
        candidate_evidence = evidence.get((workload.node_id, spec.implementation_id))
        if not forced and not prefer_gpu:
            if candidate_evidence is not None:
                performance = evaluate_auto_performance(candidate_evidence)
                if not performance.select_candidate:
                    rejections.append(
                        SupportDecision(
                            False,
                            performance.reason,
                            performance.reason_text,
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
        qualification = _matching_exact_workload_qualification(
            exact_workload_qualifications,
            workload,
            spec,
            environment,
            exact_workload_qualification_scope_digest,
        )
        if (
            not support.supported
            and support.exact_workload_test_allowed
            and qualification is not None
        ):
            support = evaluate_candidate_exact_workload_test_support(
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
                candidate_evidence,
                qualification,
            )
        )
    return tuple(candidates), tuple(rejections)


def _matching_exact_workload_qualification(
    qualifications: frozenset[ExactWorkloadCandidateQualification],
    workload: WorkloadDescriptor,
    spec: OperationComputeSpec,
    environment: ComputeEnvironment,
    scope_digest: str,
) -> ExactWorkloadCandidateQualification | None:
    """Return the one exact proof matching every planner-visible identity."""

    if not qualifications or not scope_digest:
        return None
    workload_identity = exact_workload_identity_digest(workload)
    return next(
        (
            item
            for item in qualifications
            if item.node_id == workload.node_id
            and item.operation_id == workload.operation_id
            and item.implementation_id == spec.implementation_id
            and item.implementation_version == spec.implementation_version
            and item.workload_identity_digest == workload_identity
            and item.compute_environment_fingerprint == environment.fingerprint
            and item.parity_policy_id == spec.parity_policy_id
            and item.qualification_scope_digest == scope_digest
        ),
        None,
    )


def _residency_aware_specs(
    registry: ComputeRegistry,
    request: ComputeRequest,
    preference: NodeComputePreference,
    workload: WorkloadDescriptor,
    specs: tuple[OperationComputeSpec, ...],
    decisions: Sequence[NodeExecutionDecision],
    array_facts: tuple[ArrayFacts, ...],
) -> tuple[tuple[OperationComputeSpec, ...], tuple[SupportDecision, ...]]:
    """Keep a cheap channel slice on host unless it already receives CUDA data.

    Extract Channel is an allocation-sharing device view.  At a host source,
    selecting it merely to satisfy Prefer GPU would upload and retain every
    channel before discarding all but one.  Keeping that first slice on CPU
    uploads only the selected channel into the following device segment.  A
    Custom GPU pin remains an explicit opt-in to whole-input residency.
    """

    automatic_or_prefer_intent = request.mode in {
        ComputeMode.AUTO,
        ComputeMode.PREFER_GPU,
    } or (
        request.mode is ComputeMode.CUSTOM
        and preference.kind is NodePreferenceKind.AUTO
    )
    if (
        workload.operation_id != "extract_channel"
        or not automatic_or_prefer_intent
        or not specs
    ):
        return specs, ()

    support_by_spec = {
        spec: evaluate_candidate_workload_support(
            spec,
            workload,
            array_facts=array_facts,
        )
        for spec in specs
    }
    placement_eligible = tuple(
        spec for spec, support in support_by_spec.items() if support.supported
    )
    if not placement_eligible:
        # Preserve the more useful scientific/dtype/parameter rejection.  A
        # transfer-placement explanation applies only to an otherwise usable
        # device view.
        return specs, ()

    decisions_by_node = {decision.node_id: decision for decision in decisions}
    predecessors = tuple(
        decisions_by_node[node_id]
        for node_id in workload.resident_predecessors
        if node_id in decisions_by_node
    )

    def receives_compatible_device_array(target: OperationComputeSpec) -> bool:
        for decision in predecessors:
            if decision.runtime_id == CPU_RUNTIME_ID:
                continue
            try:
                source = registry.implementation_spec(
                    decision.implementation_id,
                    decision.implementation_version,
                    allow_experimental=request.allow_experimental,
                )
            except KeyError:
                continue
            if (
                not source.supports_device_residency
                or source.host_boundary
                or source.host_finalizer_ref
                or source.runtime_id != target.runtime_id
                or source.array_domain != target.array_domain
            ):
                continue
            if source.implementation_library_id == target.implementation_library_id:
                return True
            if registry.interoperability_contract(
                target.runtime_id,
                (
                    source.implementation_library_id,
                    target.implementation_library_id,
                ),
            ):
                return True
        return False

    rejected = tuple(
        spec for spec, support in support_by_spec.items() if not support.supported
    )
    retained = tuple(
        spec for spec in placement_eligible if receives_compatible_device_array(spec)
    )
    if retained:
        return (*rejected, *retained), ()
    return rejected, (
        SupportDecision(
            False,
            DecisionReason.PERFORMANCE_GATE,
            "Extract Channel stayed on CPU to avoid uploading the complete "
            "multichannel image merely to select one channel. If a following "
            "GPU segment is selected, VIPP uploads only that selected channel. "
            "A Custom GPU choice can explicitly retain the complete input on "
            "the device.",
        ),
    )


def _select_candidate(
    request: ComputeRequest,
    preference: NodeComputePreference,
    candidates: tuple[_Candidate, ...],
) -> tuple[_Candidate | None, SupportDecision | None]:
    if not candidates:
        return None, None
    forced = _is_forced_gpu_preference(request.mode, preference)
    if request.mode is ComputeMode.PREFER_GPU:
        measured = tuple(candidate for candidate in candidates if candidate.evidence)
        if len(measured) == len(candidates):
            return min(
                measured,
                key=lambda candidate: (
                    candidate.evidence.end_to_end_candidate_seconds,
                    candidate.spec.implementation_id,
                ),
            ), None
        return min(
            candidates,
            key=lambda candidate: candidate.spec.implementation_id,
        ), None
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

    measured = tuple(candidate for candidate in candidates if candidate.evidence)
    if measured:
        return min(
            measured,
            key=lambda candidate: (
                candidate.evidence.end_to_end_candidate_seconds,
                candidate.spec.implementation_id,
            ),
        ), None
    # PUBLIC_AUTO_CANDIDATE is itself a reviewed release-policy default.  An
    # absence of exact local timing is unknown performance, not evidence that
    # CPU is faster. Exact compatible evidence above can still reject or rank
    # candidates, and all scientific/environment/memory gates remain active.
    return min(
        candidates,
        key=lambda candidate: candidate.spec.implementation_id,
    ), None


def _preferred_rejection(
    rejections: tuple[SupportDecision, ...],
    *,
    request: ComputeRequest,
    preference: NodeComputePreference,
) -> SupportDecision:
    if rejections:
        priority = {
            DecisionReason.PERFORMANCE_GATE: -1,
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
    if candidate.exact_workload_qualification is not None:
        reason_text = (
            f"{candidate.spec.implementation_id!r} passed exact CPU/GPU "
            "scientific parity for this workload and optimizer identity."
        )
    elif mode is ComputeMode.PREFER_GPU:
        reason_text = (
            f"{candidate.spec.implementation_id!r} is scientifically eligible, "
            "available, and within the admitted memory bound. Prefer GPU selected "
            "it without applying the CPU-versus-GPU speed threshold."
        )
    elif mode is ComputeMode.CUSTOM and preference.kind not in {
        NodePreferenceKind.AUTO,
        NodePreferenceKind.CPU,
    }:
        reason_text = (
            f"The validated {preference.kind.value} preference selected "
            f"{candidate.spec.implementation_id!r}."
        )
    elif candidate.evidence is not None:
        reason_text = (
            f"{candidate.spec.implementation_id!r} clears scientific, memory, "
            "and Auto performance policy."
        )
    else:
        reason_text = (
            f"{candidate.spec.implementation_id!r} is a reviewed Auto default "
            "and clears the current scientific, environment, and memory gates; "
            "no exact compatible local speed comparison was available."
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
        benchmark_record_digest=(
            candidate.exact_workload_qualification.benchmark_record_digest
            if candidate.exact_workload_qualification is not None
            else ""
        ),
        memory_estimate=candidate.memory_estimate,
        implementation_version=candidate.spec.implementation_version,
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
        implementation_version="1",
    )


def _cpu_implementation_id(operation_id: str) -> str:
    return f"cpu-{operation_id}-v1"


def _is_forced_gpu_preference(
    mode: ComputeMode,
    preference: NodeComputePreference,
) -> bool:
    return mode is ComputeMode.CUSTOM and preference.kind in {
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
    "probe_compute_environment",
]
