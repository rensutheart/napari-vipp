"""Qt-free, counterfactual suggestions for transparent compute graph repairs.

The evaluator deliberately consumes typed declarations and policy decisions.
Human-readable ``reason_text`` values are never inspected, so wording changes
cannot create or suppress a graph-edit suggestion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRepairAction,
    ComputeRepairCandidate,
    ComputeRepairSuggestion,
    ComputeRequest,
    NodeComputePreference,
    NodePreferenceKind,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    FactCompleteness,
    PerformanceEvidence,
    estimate_candidate_memory,
    evaluate_auto_performance,
    evaluate_candidate_environment_support,
    evaluate_candidate_workload_support,
    evaluate_memory_support,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import OperationComputeSpec

_SAFE_INTEGER_SOURCES = frozenset({"uint8", "uint16"})
_REPAIR_TARGET_DTYPE = "float32"
_REPAIR_SCALING = "preserve"


@dataclass(frozen=True, slots=True)
class _AdmittedCounterfactual:
    spec: OperationComputeSpec
    evidence: PerformanceEvidence | None


def potential_compute_repair_specs(
    request: ComputeRequest,
    workload: WorkloadDescriptor,
    registry: ComputeRegistry,
) -> tuple[OperationComputeSpec, ...]:
    """Return candidates worth probing for a possible exact dtype repair.

    This provider-free pre-probe helper applies request visibility, input-port,
    shape, and parameter policy. A facts-gated counterfactual remains potential;
    :func:`suggest_compute_repairs` makes the final decision only after the real
    environment and any required complete input facts are available.
    """

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    if not isinstance(workload, WorkloadDescriptor):
        raise TypeError("workload must be a WorkloadDescriptor.")
    if not isinstance(registry, ComputeRegistry):
        raise TypeError("registry must be a ComputeRegistry.")
    if request.mode is ComputeMode.CPU or not workload.inputs_resolved:
        return ()
    preference = (
        request.preference_for(workload.node_id)
        if request.mode is ComputeMode.CUSTOM
        else NodeComputePreference(NodePreferenceKind.AUTO)
    )
    if preference.kind is NodePreferenceKind.CPU:
        return ()
    selected: list[OperationComputeSpec] = []
    for spec in _specs_for_request(registry, request, workload, preference):
        for port_index, raw_dtype in enumerate(workload.input_dtypes):
            current_dtype = _dtype_name(raw_dtype)
            if current_dtype not in _SAFE_INTEGER_SOURCES:
                continue
            if not _port_contract_proves_dtype_blocker(
                spec,
                port_index,
                current_dtype,
            ):
                continue
            counterfactual = _counterfactual_workload(workload, port_index)
            support = evaluate_candidate_workload_support(
                spec,
                counterfactual,
                array_facts=(),
            )
            if support.supported or support.requires_complete_facts:
                selected.append(spec)
                break
    return tuple(selected)


def suggest_compute_repairs(
    request: ComputeRequest,
    workload: WorkloadDescriptor,
    registry: ComputeRegistry,
    environment: ComputeEnvironment,
    *,
    array_facts: tuple[ArrayFacts, ...] = (),
    performance_evidence: Mapping[
        tuple[str, str],
        PerformanceEvidence,
    ]
    | None = None,
) -> tuple[ComputeRepairSuggestion, ...]:
    """Return safe one-click repairs for one prepared node workload.

    The initial repair region is intentionally narrow: one ``uint8`` or
    ``uint16`` input may be converted to ``float32`` with ``preserve`` scaling.
    Both integer domains are represented exactly by float32. A suggestion is
    returned only when an eligible candidate is available in the *current*
    environment and the counterfactual workload clears every remaining
    scientific, shape, parameter, input-fact, performance, and memory gate.

    The caller can resolve the source connection from ``node_id`` and
    ``input_port_index`` and insert the visible Convert Dtype node using the
    suggestion's :attr:`~ComputeRepairSuggestion.conversion_parameters`.
    """

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    if not isinstance(workload, WorkloadDescriptor):
        raise TypeError("workload must be a WorkloadDescriptor.")
    if not isinstance(registry, ComputeRegistry):
        raise TypeError("registry must be a ComputeRegistry.")
    if not isinstance(environment, ComputeEnvironment):
        raise TypeError("environment must be a ComputeEnvironment.")
    facts = tuple(array_facts)
    if any(not isinstance(item, ArrayFacts) for item in facts):
        raise TypeError("array_facts must contain ArrayFacts values.")
    evidence = dict(performance_evidence or {})
    if any(not isinstance(item, PerformanceEvidence) for item in evidence.values()):
        raise TypeError("performance_evidence values must be PerformanceEvidence.")
    if request.mode is ComputeMode.CPU or not workload.inputs_resolved:
        return ()
    if facts and not _facts_match_workload(facts, workload):
        # Stale or partial facts are another blocker. Fail closed rather than
        # manufacturing a suggestion from a mismatched scientific snapshot.
        return ()

    preference = (
        request.preference_for(workload.node_id)
        if request.mode is ComputeMode.CUSTOM
        else NodeComputePreference(NodePreferenceKind.AUTO)
    )
    if preference.kind is NodePreferenceKind.CPU:
        return ()
    specs = _specs_for_request(registry, request, workload, preference)
    if not specs:
        return ()

    # If any candidate already clears its real workload gates, dtype is not the
    # sole accelerator blocker and adding a conversion would be misleading.
    if any(
        _admit_counterfactual(
            spec,
            request,
            workload,
            environment,
            facts,
            evidence,
            preference,
        )
        is not None
        for spec in specs
    ):
        return ()

    suggestions: list[ComputeRepairSuggestion] = []
    considered_ports: set[int] = set()
    for port_index, raw_dtype in enumerate(workload.input_dtypes):
        current_dtype = _dtype_name(raw_dtype)
        if current_dtype not in _SAFE_INTEGER_SOURCES:
            continue
        if port_index in considered_ports:
            continue
        counterfactual = _counterfactual_workload(workload, port_index)
        counterfactual_facts = _counterfactual_array_facts(
            workload,
            facts,
            port_index,
        )
        admitted = tuple(
            item
            for spec in specs
            if (
                item := _admit_counterfactual(
                    spec,
                    request,
                    counterfactual,
                    environment,
                    counterfactual_facts,
                    evidence,
                    preference,
                )
            )
            is not None
            and _port_contract_proves_dtype_blocker(
                spec,
                port_index,
                current_dtype,
            )
            and not evaluate_candidate_workload_support(
                spec,
                workload,
                array_facts=facts,
            ).supported
        )
        selected = _select_counterfactual(
            admitted,
            request=request,
            preference=preference,
        )
        if selected is None:
            continue
        port = selected.spec.input_ports[port_index]
        memory_factor = (
            np.dtype(np.float32).itemsize // np.dtype(current_dtype).itemsize
        )
        suggestions.append(
            ComputeRepairSuggestion(
                action=ComputeRepairAction.INSERT_CONVERT_DTYPE,
                node_id=workload.node_id,
                operation_id=workload.operation_id,
                input_port_index=port_index,
                input_port_name=port.port_name,
                current_dtype=current_dtype,
                target_dtype=_REPAIR_TARGET_DTYPE,
                scaling=_REPAIR_SCALING,
                exact=True,
                message=(
                    "This node could become eligible for GPU use if its "
                    f"{port.port_name} input is converted from {current_dtype} to "
                    "float32. The conversion will preserve every pixel value "
                    f"exactly, while that converted input uses {memory_factor}× "
                    "as much memory."
                ),
                candidate=ComputeRepairCandidate(
                    implementation_id=selected.spec.implementation_id,
                    implementation_version=selected.spec.implementation_version,
                    runtime_id=selected.spec.runtime_id,
                    implementation_library_id=(
                        selected.spec.implementation_library_id
                    ),
                ),
            )
        )
        considered_ports.add(port_index)
    return tuple(suggestions)


def _specs_for_request(
    registry: ComputeRegistry,
    request: ComputeRequest,
    workload: WorkloadDescriptor,
    preference: NodeComputePreference,
) -> tuple[OperationComputeSpec, ...]:
    specs = registry.implementations_for_operation(
        workload.operation_id,
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
            spec
            for spec in specs
            if spec.implementation_library_id == preference.value
        )
    elif preference.kind is NodePreferenceKind.IMPLEMENTATION:
        specs = tuple(
            spec for spec in specs if spec.implementation_id == preference.value
        )
    return tuple(
        sorted(
            (spec for spec in specs if spec.is_gpu),
            key=lambda spec: (spec.implementation_id, spec.implementation_version),
        )
    )


def _admit_counterfactual(
    spec: OperationComputeSpec,
    request: ComputeRequest,
    workload: WorkloadDescriptor,
    environment: ComputeEnvironment,
    facts: tuple[ArrayFacts, ...],
    evidence: Mapping[tuple[str, str], PerformanceEvidence],
    preference: NodeComputePreference,
) -> _AdmittedCounterfactual | None:
    environment_support = evaluate_candidate_environment_support(
        spec,
        environment,
        allow_experimental=request.allow_experimental,
    )
    if not environment_support.supported:
        return None
    workload_support = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=facts,
    )
    if not workload_support.supported:
        return None
    try:
        memory = estimate_candidate_memory(spec, workload)
    except (TypeError, ValueError):
        return None
    memory_support = evaluate_memory_support(
        memory,
        memory_cap_bytes=request.accelerator_memory_cap_bytes,
        total_device_bytes=environment.total_accelerator_memory_bytes,
        safety_reserve_bytes=request.accelerator_safety_reserve_bytes or 0,
    )
    if not memory_support.supported:
        return None
    candidate_evidence = evidence.get((workload.node_id, spec.implementation_id))
    forced = request.mode is ComputeMode.CUSTOM and preference.kind in {
        NodePreferenceKind.BEST_GPU,
        NodePreferenceKind.LIBRARY,
        NodePreferenceKind.IMPLEMENTATION,
    }
    if (
        not forced
        and request.mode is not ComputeMode.PREFER_GPU
        and candidate_evidence is not None
        and not evaluate_auto_performance(candidate_evidence).select_candidate
    ):
        return None
    return _AdmittedCounterfactual(spec, candidate_evidence)


def _select_counterfactual(
    candidates: tuple[_AdmittedCounterfactual, ...],
    *,
    request: ComputeRequest,
    preference: NodeComputePreference,
) -> _AdmittedCounterfactual | None:
    if not candidates:
        return None
    forced = request.mode is ComputeMode.CUSTOM and preference.kind in {
        NodePreferenceKind.BEST_GPU,
        NodePreferenceKind.LIBRARY,
        NodePreferenceKind.IMPLEMENTATION,
    }
    measured = tuple(item for item in candidates if item.evidence is not None)
    if request.mode is ComputeMode.PREFER_GPU:
        if len(measured) == len(candidates):
            return min(measured, key=_candidate_performance_key)
        return min(candidates, key=_candidate_identity_key)
    if forced:
        if len(candidates) == 1:
            return candidates[0]
        if len(measured) != len(candidates):
            # Best-GPU/library selection still has a comparison blocker after
            # the dtype change, so dtype is not the only missing condition.
            return None
        return min(measured, key=_candidate_performance_key)
    if measured:
        return min(measured, key=_candidate_performance_key)
    return min(candidates, key=_candidate_identity_key)


def _candidate_performance_key(
    item: _AdmittedCounterfactual,
) -> tuple[float, str, str]:
    assert item.evidence is not None
    return (
        item.evidence.end_to_end_candidate_seconds,
        item.spec.implementation_id,
        item.spec.implementation_version,
    )


def _candidate_identity_key(
    item: _AdmittedCounterfactual,
) -> tuple[str, str]:
    return item.spec.implementation_id, item.spec.implementation_version


def _port_contract_proves_dtype_blocker(
    spec: OperationComputeSpec,
    port_index: int,
    current_dtype: str,
) -> bool:
    if port_index >= len(spec.input_ports):
        return False
    public_dtypes = tuple(
        _dtype_name(value)
        for value in spec.input_ports[port_index].public_dtypes
    )
    return (
        "*" not in spec.input_ports[port_index].public_dtypes
        and current_dtype not in public_dtypes
        and _REPAIR_TARGET_DTYPE in public_dtypes
    )


def _counterfactual_workload(
    workload: WorkloadDescriptor,
    port_index: int,
) -> WorkloadDescriptor:
    dtypes = list(workload.input_dtypes)
    dtypes[port_index] = _REPAIR_TARGET_DTYPE
    return replace(
        workload,
        input_dtypes=tuple(dtypes),
        facts_fingerprint="",
    )


def _counterfactual_array_facts(
    workload: WorkloadDescriptor,
    facts: tuple[ArrayFacts, ...],
    port_index: int,
) -> tuple[ArrayFacts, ...]:
    if facts:
        projected = list(facts)
        source = facts[port_index]
    else:
        projected = [
            _unknown_facts(shape, dtype, index)
            for index, (shape, dtype) in enumerate(
                zip(
                    workload.input_shapes,
                    workload.input_dtypes,
                    strict=True,
                )
            )
        ]
        source = None
    shape = workload.input_shapes[port_index]
    element_count = int(math.prod(shape))
    guarantees = set(source.guarantees if source is not None else ())
    guarantees.update(("nonnegative", "no-negative-zero"))
    projected[port_index] = ArrayFacts(
        shape=shape,
        dtype=_REPAIR_TARGET_DTYPE,
        element_count=element_count,
        revision_fingerprint=(
            f"{source.revision_fingerprint}>convert_dtype:float32:preserve"
            if source is not None
            else f"counterfactual:{workload.node_id}:{port_index}:float32:preserve"
        ),
        completeness=FactCompleteness.COMPLETE,
        finite_count=element_count,
        minimum=source.minimum if source is not None else None,
        maximum=source.maximum if source is not None else None,
        guarantees=tuple(sorted(guarantees)),
        scan_seconds=source.scan_seconds if source is not None else 0.0,
    )
    return tuple(projected)


def _unknown_facts(
    shape: tuple[int, ...],
    dtype: str,
    port_index: int,
) -> ArrayFacts:
    return ArrayFacts(
        shape=shape,
        dtype=dtype,
        element_count=int(math.prod(shape)),
        revision_fingerprint=f"counterfactual:unscanned:{port_index}",
        completeness=FactCompleteness.UNKNOWN,
    )


def _facts_match_workload(
    facts: tuple[ArrayFacts, ...],
    workload: WorkloadDescriptor,
) -> bool:
    return len(facts) == len(workload.input_shapes) and all(
        item.shape == shape and _dtype_name(item.dtype) == _dtype_name(dtype)
        for item, shape, dtype in zip(
            facts,
            workload.input_shapes,
            workload.input_dtypes,
            strict=True,
        )
    )


def _dtype_name(value: object) -> str:
    try:
        return np.dtype(value).name
    except (TypeError, ValueError):
        return str(value).strip()


__all__ = ["potential_compute_repair_specs", "suggest_compute_repairs"]
