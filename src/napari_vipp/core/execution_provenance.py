"""Provider-free serialization of actual pipeline compute provenance.

The execution service deliberately returns provider-neutral decisions, while
the pipeline cache owns exact scientific implementation identities.  Durable
surfaces need both: decisions explain *why* a backend ran, and identities name
the exact versioned implementation that produced a result.  This module merges
those records without importing an optional accelerator provider.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionReport,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
    canonical_digest,
)
from napari_vipp.core.compute_specs import (
    OperationComputeSpec,
    compute_specs_for,
)

if TYPE_CHECKING:
    from napari_vipp.core.pipeline import PrototypePipeline


EXECUTION_PROVENANCE_TYPE = "napari-vipp-compute-execution"
EXECUTION_PROVENANCE_VERSION = 1


def serialize_execution_provenance(
    request: ComputeRequest,
    pipeline: PrototypePipeline | None,
    execution_report: ExecutionReport | None,
    *,
    completed_node_ids: Sequence[str] = (),
    implementation_specs: Sequence[OperationComputeSpec] = (),
    failure: object | None = None,
) -> dict[str, object]:
    """Return JSON-safe actual compute provenance for one pipeline run.

    ``completed_node_ids`` should be supplied by callers that prune pipeline
    caches.  It preserves records for completed intermediates whose host values
    and cache provenance were intentionally released before this function is
    called.  When omitted, report decisions and retained cache provenance are
    used as the best available completed-node evidence.
    """

    if not isinstance(request, ComputeRequest):
        raise TypeError("request must be a ComputeRequest.")
    if execution_report is not None and not isinstance(
        execution_report,
        ExecutionReport,
    ):
        raise TypeError("execution_report must be an ExecutionReport or None.")

    failure_document = _failure_document(failure)
    decisions = (
        () if execution_report is None else execution_report.actual_decisions
    )
    decisions_by_node = {decision.node_id: decision for decision in decisions}
    cached_by_node = (
        {} if pipeline is None else dict(pipeline.node_compute_provenance)
    )
    completed = (
        ()
        if pipeline is None
        else _ordered_completed_nodes(
            pipeline,
            completed_node_ids,
            decision_node_ids=tuple(decisions_by_node),
            cached_node_ids=tuple(cached_by_node),
        )
    )
    specs = (
        {}
        if pipeline is None
        else _implementation_specs_by_identity(
            pipeline,
            completed,
            implementation_specs,
            allow_experimental=request.allow_experimental,
        )
    )

    node_records: list[dict[str, object]] = []
    for node_id in completed:
        assert pipeline is not None
        node = pipeline.nodes.get(node_id)
        if node is None:
            continue
        operation_id = str(node.operation_id)
        if not pipeline.operation_spec(operation_id).has_input:
            # Source boundaries are inputs, not computed scientific results.
            # Durable callers record their own source identity separately.
            continue
        cached = cached_by_node.get(node_id)
        decision = decisions_by_node.get(node_id)
        if decision is None:
            decision = (
                _bypass_decision(
                    request,
                    node_id=node_id,
                    operation_id=operation_id,
                )
                if pipeline.node_is_bypassed(node_id)
                else _cpu_decision(
                    request,
                    node_id=node_id,
                    operation_id=operation_id,
                )
            )
        identity = _actual_identity(
            decision,
            cached,
            specs,
        )
        record = {
            "node_id": node_id,
            "operation_id": operation_id,
            "execution_mode": node.execution_mode,
            "bypassed": decision.decision_kind is DecisionKind.BYPASSED,
            "requested_preference": decision.requested_preference.as_dict(),
            "actual_implementation": identity,
            "decision_kind": decision.decision_kind.value,
            "reason": decision.reason.value,
            "reason_text": decision.reason_text,
            "fallback_used": decision.fallback_used,
            "fallback_reason": decision.fallback_reason.value,
            "benchmark_record_digest": decision.benchmark_record_digest,
            "performance_evidence_kind": decision.performance_evidence_kind,
            "performance_evidence_digest": decision.performance_evidence_digest,
            "memory_estimate": _json_safe(decision.memory_estimate),
        }
        node_records.append(record)

    fallbacks = [
        {
            "node_id": record["node_id"],
            "operation_id": record["operation_id"],
            "fallback_reason": record["fallback_reason"],
            "requested_preference": record["requested_preference"],
            "actual_implementation": record["actual_implementation"],
            "out_of_memory": (
                record["fallback_reason"] == FallbackReason.OUT_OF_MEMORY.value
            ),
        }
        for record in node_records
        if bool(record["fallback_used"])
    ]
    environment = (
        execution_report.environment
        if execution_report is not None
        else ComputeEnvironment()
        if request.mode is ComputeMode.CPU
        else None
    )
    failure_cleanup = (
        None
        if failure_document is None
        else failure_document.get("cleanup_succeeded")
    )
    cleanup_succeeded: bool | None
    if execution_report is not None:
        cleanup_succeeded = bool(execution_report.cleanup_succeeded)
    elif failure_cleanup is not None:
        cleanup_succeeded = bool(failure_cleanup)
    elif failure_document is None:
        cleanup_succeeded = True
    else:
        # Absence of cleanup evidence on a failed accelerated request is not a
        # successful proof. Durable publishers must fail closed.
        cleanup_succeeded = (
            True if request.mode is ComputeMode.CPU else None
        )
    payload: dict[str, object] = {
        "type": EXECUTION_PROVENANCE_TYPE,
        "version": EXECUTION_PROVENANCE_VERSION,
        "request": request.as_dict(),
        "request_fingerprint": request.fingerprint,
        "environment": None if environment is None else environment.as_dict(),
        "environment_fingerprint": (
            "" if environment is None else environment.fingerprint
        ),
        "plan": (
            None
            if execution_report is None or execution_report.plan is None
            else _json_safe(execution_report.plan)
        ),
        "nodes": node_records,
        "fallbacks": fallbacks,
        "fallback_records": (
            list(failure_document.get("fallback_records", []))
            if execution_report is None and failure_document is not None
            else [
                _json_safe(record)
                for record in (
                    ()
                    if execution_report is None
                    else execution_report.fallback_records
                )
            ]
        ),
        "outcome": (
            "completed"
            if failure_document is None
            else "cancelled"
            if failure_document.get("kind") == "cancelled"
            else "failed"
        ),
        "failure": failure_document,
        "warnings": (
            []
            if execution_report is None
            else list(execution_report.warnings)
        ),
        "cleanup_succeeded": cleanup_succeeded,
    }
    return payload


def execution_provenance_digest(payload: Mapping[str, object]) -> str:
    """Return the stable digest used to link outputs to execution evidence."""

    if not isinstance(payload, Mapping):
        raise TypeError("execution provenance must be a mapping.")
    return canonical_digest(dict(payload))


def _failure_document(failure: object | None) -> dict[str, object] | None:
    if failure is None:
        return None
    if isinstance(failure, Mapping):
        return {str(key): _json_safe(value) for key, value in failure.items()}
    as_dict = getattr(failure, "as_dict", None)
    if not callable(as_dict):
        raise TypeError("failure must expose a callable as_dict() method.")
    document = as_dict()
    if not isinstance(document, Mapping):
        raise TypeError("failure as_dict() must return a mapping.")
    return {str(key): _json_safe(value) for key, value in document.items()}


def _ordered_completed_nodes(
    pipeline: PrototypePipeline,
    completed_node_ids: Sequence[str],
    *,
    decision_node_ids: Sequence[str],
    cached_node_ids: Sequence[str],
) -> tuple[str, ...]:
    requested = {
        str(node_id).strip()
        for node_id in (
            *tuple(completed_node_ids),
            *tuple(decision_node_ids),
            *tuple(cached_node_ids),
        )
        if str(node_id).strip()
    }
    order = tuple(pipeline.topological_order())
    return tuple(node_id for node_id in order if node_id in requested)


def _implementation_specs_by_identity(
    pipeline: PrototypePipeline,
    node_ids: Sequence[str],
    supplied: Sequence[OperationComputeSpec],
    *,
    allow_experimental: bool,
) -> dict[tuple[str, str, str, str], OperationComputeSpec]:
    specs: list[OperationComputeSpec] = list(supplied)
    operation_ids = {
        pipeline.nodes[node_id].operation_id
        for node_id in node_ids
        if node_id in pipeline.nodes
    }
    for operation_id in sorted(operation_ids):
        try:
            specs.extend(
                compute_specs_for(
                    operation_id,
                    allow_experimental=allow_experimental,
                )
            )
        except (KeyError, TypeError, ValueError):
            # A third-party operation may be represented only by supplied
            # declarations or retained cache provenance.
            continue
    return {
        (
            spec.operation_id,
            spec.runtime_id,
            spec.implementation_library_id,
            spec.implementation_id,
        ): spec
        for spec in specs
    }


def _actual_identity(
    decision: NodeExecutionDecision,
    cached: object,
    specs: Mapping[tuple[str, str, str, str], OperationComputeSpec],
) -> dict[str, object]:
    cached_identity = getattr(cached, "actual_implementation", None)
    if (
        cached_identity is not None
        and str(getattr(cached_identity, "operation_id", ""))
        == decision.operation_id
        and str(getattr(cached_identity, "runtime_id", ""))
        == decision.runtime_id
        and str(getattr(cached_identity, "implementation_library_id", ""))
        == decision.implementation_library_id
        and str(getattr(cached_identity, "implementation_id", ""))
        == decision.implementation_id
    ):
        return {
            "identity_complete": True,
            "runtime_id": cached_identity.runtime_id,
            "array_domain": cached_identity.array_domain,
            "implementation_library_id": (
                cached_identity.implementation_library_id
            ),
            "implementation_id": cached_identity.implementation_id,
            "implementation_version": cached_identity.implementation_version,
            "parity_policy_id": cached_identity.parity_policy_id,
            "cache_equivalence_group": cached_identity.cache_equivalence_group,
        }

    if decision.decision_kind is DecisionKind.BYPASSED:
        return {
            "identity_complete": True,
            "runtime_id": "vipp-bypass",
            "array_domain": "resident-alias",
            "implementation_library_id": "vipp-alias",
            "implementation_id": "vipp-safe-bypass-v1",
            "implementation_version": "1",
            "parity_policy_id": "exact-unary-alias-v1",
            "cache_equivalence_group": "",
        }

    spec = specs.get(
        (
            decision.operation_id,
            decision.runtime_id,
            decision.implementation_library_id,
            decision.implementation_id,
        )
    )
    if spec is None:
        # The decision remains useful and honest, but an external/custom
        # planner must supply its declaration to claim an exact version.
        return {
            "identity_complete": False,
            "runtime_id": decision.runtime_id,
            "array_domain": "",
            "implementation_library_id": decision.implementation_library_id,
            "implementation_id": decision.implementation_id,
            "implementation_version": decision.implementation_version,
            "parity_policy_id": "",
            "cache_equivalence_group": "",
        }
    return {
        "identity_complete": True,
        "runtime_id": spec.runtime_id,
        "array_domain": spec.array_domain,
        "implementation_library_id": spec.implementation_library_id,
        "implementation_id": spec.implementation_id,
        "implementation_version": spec.implementation_version,
        "parity_policy_id": spec.parity_policy_id,
        "cache_equivalence_group": spec.cache_equivalence_group,
    }


def _cpu_decision(
    request: ComputeRequest,
    *,
    node_id: str,
    operation_id: str,
) -> NodeExecutionDecision:
    preference = (
        request.preference_for(node_id)
        if request.mode in {ComputeMode.CPU, ComputeMode.CUSTOM}
        else NodeComputePreference()
    )
    reason = (
        DecisionReason.EXPLICIT_CPU
        if request.mode is ComputeMode.CPU
        else DecisionReason.AUTO_CPU
    )
    return NodeExecutionDecision(
        node_id=node_id,
        operation_id=operation_id,
        requested_preference=preference,
        runtime_id="cpu-numpy",
        implementation_library_id="cpu",
        implementation_id=f"cpu-{operation_id}-v1",
        decision_kind=DecisionKind.POLICY_CPU,
        reason=reason,
        reason_text=(
            "The authoritative host implementation completed this node."
        ),
        fallback_reason=FallbackReason.NONE,
        implementation_version="1",
    )


def _bypass_decision(
    request: ComputeRequest,
    *,
    node_id: str,
    operation_id: str,
) -> NodeExecutionDecision:
    return NodeExecutionDecision(
        node_id=node_id,
        operation_id=operation_id,
        requested_preference=request.preference_for(node_id),
        runtime_id="vipp-bypass",
        implementation_library_id="vipp-alias",
        implementation_id="vipp-safe-bypass-v1",
        implementation_version="1",
        decision_kind=DecisionKind.BYPASSED,
        reason=DecisionReason.BYPASSED,
        reason_text=(
            "Safe Node Bypass forwarded the exact input without invoking the "
            "operation."
        ),
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


__all__ = [
    "EXECUTION_PROVENANCE_TYPE",
    "EXECUTION_PROVENANCE_VERSION",
    "execution_provenance_digest",
    "serialize_execution_provenance",
]
