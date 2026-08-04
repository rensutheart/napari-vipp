"""Evidence-gated whole-pipeline implementation assignment.

The optimizer is intentionally provider- and GUI-free.  It consumes exact local
benchmark records, a measured directional transfer profile, and an injected
whole-pipeline validation callback.  It never executes a node or mutates a
workflow itself.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType

from napari_vipp.core.compute import (
    BenchmarkCandidateFailureKind,
    BenchmarkCandidateResult,
    BenchmarkRecord,
    ComputeMode,
    ComputeRequest,
    NodeComputePreference,
    NodePreferenceKind,
    canonical_digest,
)
from napari_vipp.core.compute_benchmark import (
    HOST_RUNTIME_ID,
    GraphCostEdge,
    GraphCostNode,
    GraphImplementationCost,
    GraphOptimizationCancelled,
    GraphOptimizationError,
    GraphOptimizationProblem,
    NoFeasibleGraphAssignment,
    RuntimeTransitionCost,
    optimize_graph_assignment,
)

_MINIMUM_RELATIVE_IMPROVEMENT = 0.05
_MINIMUM_ABSOLUTE_IMPROVEMENT_SECONDS = 0.010


class PipelineOptimizationError(RuntimeError):
    """Base class for safe pipeline-optimizer failures and refusals."""


class PipelineOptimizationCancelled(PipelineOptimizationError):
    """Cooperative cancellation stopped analysis without publishing a proposal."""


@dataclass(frozen=True, slots=True)
class PipelineOptimizationTimeoutReport:
    """Structured context explaining where an optimization deadline expired."""

    stage: str = "unknown"
    stage_message: str = ""
    elapsed_seconds: float | None = None
    budget_seconds: float | None = None
    overall_completed: int = 0
    overall_total: int = 1
    node_id: str = ""
    node_title: str = ""
    node_index: int = 0
    node_total: int = 0
    operation_completed: int = 0
    operation_total: int = 0
    operation_message: str = ""
    completed_node_ids: tuple[str, ...] = ()
    reused_node_ids: tuple[str, ...] = ()
    baseline_completed: bool = False
    validation_started: bool = False
    validation_completed: bool = False
    partial_node_discarded: bool = False

    def __post_init__(self) -> None:
        stage = str(self.stage).strip().lower() or "unknown"
        object.__setattr__(self, "stage", stage)
        for name in ("stage_message", "node_id", "node_title", "operation_message"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        for name in ("elapsed_seconds", "budget_seconds"):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "overall_completed",
            "overall_total",
            "node_index",
            "node_total",
            "operation_completed",
            "operation_total",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.overall_total < 1 or self.overall_completed > self.overall_total:
            raise ValueError("overall timeout progress is out of range")
        if self.node_index > self.node_total:
            raise ValueError("node timeout progress is out of range")
        if self.operation_completed > self.operation_total:
            raise ValueError("operation timeout progress is out of range")
        for name in (
            "baseline_completed",
            "validation_started",
            "validation_completed",
            "partial_node_discarded",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "completed_node_ids",
            tuple(
                str(item).strip()
                for item in self.completed_node_ids
                if str(item).strip()
            ),
        )
        object.__setattr__(
            self,
            "reused_node_ids",
            tuple(
                str(item).strip()
                for item in self.reused_node_ids
                if str(item).strip()
            ),
        )


class PipelineOptimizationDeadlineExceeded(PipelineOptimizationError):
    """The absolute optimizer deadline expired."""

    def __init__(
        self,
        message: str = "Pipeline optimization exceeded its deadline.",
        *,
        report: PipelineOptimizationTimeoutReport | None = None,
    ) -> None:
        if report is not None and not isinstance(
            report,
            PipelineOptimizationTimeoutReport,
        ):
            raise TypeError("report must be a PipelineOptimizationTimeoutReport")
        self.report = report
        super().__init__(str(message).strip() or "Pipeline optimization timed out.")


class PipelineOptimizationNotBeneficial(PipelineOptimizationError):
    """The measured proposal did not clear the benefit/noise gate."""


class PipelineOptimizationStale(PipelineOptimizationError):
    """Evidence or validation belongs to a different immutable identity."""


class PipelineValidationWinner(StrEnum):
    """Assignment selected by the final whole-pipeline evidence gate."""

    NOT_RUN = "not-run"
    CURRENT = "current"
    PROPOSED = "proposed"


@dataclass(frozen=True, slots=True)
class EvidenceRefusal:
    code: str
    message: str
    node_id: str = ""

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        message = str(self.message).strip()
        if not code or not message:
            raise ValueError("evidence refusal requires a code and message")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "node_id", str(self.node_id).strip())


class PipelineOptimizationEvidenceIncomplete(PipelineOptimizationError):
    """Exact evidence is absent, stale, unsynchronized, or scientifically unsafe."""

    def __init__(self, reasons: Sequence[EvidenceRefusal]) -> None:
        self.reasons = tuple(reasons)
        if not self.reasons:
            raise ValueError("evidence-incomplete refusal requires at least one reason")
        detail = "; ".join(
            f"{item.node_id}: {item.message}" if item.node_id else item.message
            for item in self.reasons
        )
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class PipelineOptimizationIdentity:
    """Every live-state component that can stale a pipeline proposal."""

    pipeline_fingerprint: str
    source_fingerprint: str
    topology_fingerprint: str
    cache_retention_fingerprint: str
    environment_fingerprint: str
    workload_fingerprints: Mapping[str, str] = field(default_factory=dict)
    identity_policy_id: str = "pipeline-optimizer-identity-v1"
    benchmark_environment_fingerprint: str = ""
    optimizer_locked_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "pipeline_fingerprint",
            "source_fingerprint",
            "topology_fingerprint",
            "cache_retention_fingerprint",
            "environment_fingerprint",
            "identity_policy_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        benchmark_environment = str(
            self.benchmark_environment_fingerprint
        ).strip()
        if not benchmark_environment:
            benchmark_environment = self.environment_fingerprint
        locked = tuple(
            str(node_id).strip() for node_id in self.optimizer_locked_node_ids
        )
        if any(not node_id for node_id in locked) or len(set(locked)) != len(locked):
            raise ValueError("optimizer lock IDs must be unique and non-empty")
        object.__setattr__(
            self,
            "benchmark_environment_fingerprint",
            benchmark_environment,
        )
        object.__setattr__(self, "optimizer_locked_node_ids", tuple(sorted(locked)))
        normalized: dict[str, str] = {}
        for raw_node_id, raw_fingerprint in self.workload_fingerprints.items():
            node_id = str(raw_node_id).strip()
            fingerprint = str(raw_fingerprint).strip()
            if not node_id or not fingerprint:
                raise ValueError("workload fingerprints require non-empty IDs")
            normalized[node_id] = fingerprint
        object.__setattr__(
            self,
            "workload_fingerprints",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "pipeline": self.pipeline_fingerprint,
                "source": self.source_fingerprint,
                "topology": self.topology_fingerprint,
                "cache_retention": self.cache_retention_fingerprint,
                "environment": self.environment_fingerprint,
                "benchmark_environment": self.benchmark_environment_fingerprint,
                "optimizer_locks": list(self.optimizer_locked_node_ids),
                "workloads": dict(self.workload_fingerprints),
                "policy": self.identity_policy_id,
            }
        )


@dataclass(frozen=True, slots=True)
class PipelineOptimizationCandidate:
    implementation_id: str
    implementation_library_id: str
    runtime_id: str
    available: bool = True
    minimum_workspace_bytes: int = 0
    host_output_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "implementation_id",
            "implementation_library_id",
            "runtime_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean")
        if (
            isinstance(self.minimum_workspace_bytes, bool)
            or not isinstance(self.minimum_workspace_bytes, int)
            or self.minimum_workspace_bytes < 0
        ):
            raise ValueError("minimum_workspace_bytes must be non-negative")
        if not isinstance(self.host_output_only, bool):
            raise TypeError("host_output_only must be a boolean")


@dataclass(frozen=True, slots=True)
class PipelineOptimizationNode:
    node_id: str
    operation_id: str
    candidates: tuple[PipelineOptimizationCandidate, ...]
    current_implementation_id: str
    authored_preference: NodeComputePreference = NodeComputePreference()
    output_bytes: int = 0
    host_input_bytes: int = 0
    requires_host_output: bool = False
    is_writer: bool = False
    has_side_effects: bool = False
    cache_retained: bool = False
    optimizer_locked: bool = False

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        operation_id = str(self.operation_id).strip()
        current = str(self.current_implementation_id).strip()
        candidates = tuple(self.candidates)
        if not node_id or not operation_id or not current or not candidates:
            raise ValueError("optimizer nodes require IDs, current choice, candidates")
        identifiers = tuple(item.implementation_id for item in candidates)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"node {node_id!r} has duplicate candidate IDs")
        if current not in identifiers:
            raise ValueError(f"node {node_id!r} current implementation is undeclared")
        for name in ("output_bytes", "host_input_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "requires_host_output",
            "is_writer",
            "has_side_effects",
            "cache_retained",
            "optimizer_locked",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        preference = NodeComputePreference.parse(self.authored_preference)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "current_implementation_id", current)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "authored_preference", preference)


@dataclass(frozen=True, slots=True)
class PipelineOptimizationEdge:
    source_node_id: str
    target_node_id: str

    def __post_init__(self) -> None:
        for name in ("source_node_id", "target_node_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class PipelineNodeBenchmarkEvidence:
    node_id: str
    identity_digest: str
    record: BenchmarkRecord

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        identity = str(self.identity_digest).strip()
        if not node_id or not identity:
            raise ValueError("benchmark evidence requires node and identity IDs")
        if not isinstance(self.record, BenchmarkRecord):
            raise TypeError("record must be a BenchmarkRecord")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "identity_digest", identity)


@dataclass(frozen=True, slots=True)
class DirectionalTransferProfile:
    """Measured host/device transfer model for one exact environment."""

    identity_digest: str
    environment_fingerprint: str
    accelerator_runtime_id: str
    host_to_accelerator_fixed_seconds: float
    host_to_accelerator_seconds_per_byte: float
    accelerator_to_host_fixed_seconds: float
    accelerator_to_host_seconds_per_byte: float
    accelerator_memory_limit_bytes: int
    sample_bytes: int
    synchronized: bool = True
    host_runtime_id: str = HOST_RUNTIME_ID

    def __post_init__(self) -> None:
        for name in (
            "identity_digest",
            "environment_fingerprint",
            "accelerator_runtime_id",
            "host_runtime_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        for name in (
            "host_to_accelerator_fixed_seconds",
            "host_to_accelerator_seconds_per_byte",
            "accelerator_to_host_fixed_seconds",
            "accelerator_to_host_seconds_per_byte",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        for name in ("accelerator_memory_limit_bytes", "sample_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.synchronized, bool):
            raise TypeError("synchronized must be a boolean")
        if self.accelerator_runtime_id == self.host_runtime_id:
            raise ValueError("accelerator and host runtime IDs must be distinct")


@dataclass(frozen=True, slots=True)
class PipelineValidationRequest:
    identity_digest: str
    current_assignment: tuple[tuple[str, str], ...]
    proposed_assignment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        identity = str(self.identity_digest).strip()
        if not identity:
            raise ValueError("validation request identity must not be empty")
        object.__setattr__(self, "identity_digest", identity)
        object.__setattr__(
            self,
            "current_assignment",
            _validated_assignment(self.current_assignment, "current_assignment"),
        )
        object.__setattr__(
            self,
            "proposed_assignment",
            _validated_assignment(self.proposed_assignment, "proposed_assignment"),
        )


@dataclass(frozen=True, slots=True)
class PipelineAssignmentValidation:
    identity_digest: str
    current_assignment: tuple[tuple[str, str], ...]
    proposed_assignment: tuple[tuple[str, str], ...]
    parity_passed: bool
    synchronized: bool
    current_seconds: float
    proposed_seconds: float
    speedup_lower_confidence_bound: float
    detail: str = ""
    measurement_rounds: int = 0
    current_speedup_lower_confidence_bound: float = 0.0

    def __post_init__(self) -> None:
        identity = str(self.identity_digest).strip()
        if not identity:
            raise ValueError("validation identity must not be empty")
        for name in (
            "current_seconds",
            "proposed_seconds",
            "speedup_lower_confidence_bound",
            "current_speedup_lower_confidence_bound",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.parity_passed, bool) or not isinstance(
            self.synchronized, bool
        ):
            raise TypeError("validation parity/synchronization must be booleans")
        object.__setattr__(self, "identity_digest", identity)
        object.__setattr__(
            self,
            "current_assignment",
            _validated_assignment(self.current_assignment, "current_assignment"),
        )
        object.__setattr__(
            self,
            "proposed_assignment",
            _validated_assignment(self.proposed_assignment, "proposed_assignment"),
        )
        object.__setattr__(self, "detail", str(self.detail).strip())
        if (
            isinstance(self.measurement_rounds, bool)
            or not isinstance(self.measurement_rounds, int)
            or self.measurement_rounds < 0
        ):
            raise ValueError("measurement_rounds must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PipelineOptimizationRow:
    node_id: str
    current_implementation_id: str
    proposed_implementation_id: str
    current_preference: NodeComputePreference
    proposed_preference: NodeComputePreference
    changed: bool
    eligible: bool
    locked: bool = False


@dataclass(frozen=True, slots=True)
class PipelineOptimizationProposal:
    identity_digest: str
    request_fingerprint: str
    baseline_assignment: tuple[tuple[str, str], ...]
    rows: tuple[PipelineOptimizationRow, ...]
    preference_mapping: Mapping[str, NodeComputePreference]
    estimated_current_seconds: float
    estimated_proposed_seconds: float
    validated_current_seconds: float
    validated_proposed_seconds: float
    validated_speedup_lower_confidence_bound: float
    pipeline_validation_performed: bool = True
    validation_measurement_rounds: int = 0
    validated_current_speedup_lower_confidence_bound: float = 0.0
    validation_winner: PipelineValidationWinner | str = (
        PipelineValidationWinner.NOT_RUN
    )
    tested_assignment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_validation_performed, bool):
            raise TypeError("pipeline_validation_performed must be a boolean")
        if (
            isinstance(self.validation_measurement_rounds, bool)
            or not isinstance(self.validation_measurement_rounds, int)
            or self.validation_measurement_rounds < 0
        ):
            raise ValueError(
                "validation_measurement_rounds must be a non-negative integer"
            )
        reverse_bound = self.validated_current_speedup_lower_confidence_bound
        if (
            isinstance(reverse_bound, bool)
            or not isinstance(reverse_bound, (int, float))
            or not math.isfinite(float(reverse_bound))
            or reverse_bound < 0
        ):
            raise ValueError(
                "validated_current_speedup_lower_confidence_bound must be "
                "finite and non-negative"
            )
        winner = (
            self.validation_winner
            if isinstance(self.validation_winner, PipelineValidationWinner)
            else PipelineValidationWinner(str(self.validation_winner).strip())
        )
        if (
            self.pipeline_validation_performed
            and winner is PipelineValidationWinner.NOT_RUN
        ):
            raise ValueError("performed validation must identify its winner")
        if (
            not self.pipeline_validation_performed
            and winner is PipelineValidationWinner.PROPOSED
        ):
            raise ValueError("an unvalidated proposal cannot select an alternative")
        object.__setattr__(
            self,
            "validated_current_speedup_lower_confidence_bound",
            float(reverse_bound),
        )
        object.__setattr__(self, "validation_winner", winner)
        tested_assignment = self.tested_assignment or tuple(
            (row.node_id, row.proposed_implementation_id) for row in self.rows
        )
        object.__setattr__(
            self,
            "tested_assignment",
            _validated_assignment(tested_assignment, "tested_assignment"),
        )
        object.__setattr__(
            self,
            "preference_mapping",
            MappingProxyType(dict(sorted(self.preference_mapping.items()))),
        )

    def is_current(
        self,
        identity: PipelineOptimizationIdentity,
        request: ComputeRequest | None = None,
        assignments: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    ) -> bool:
        if not isinstance(identity, PipelineOptimizationIdentity):
            return False
        if identity.digest != self.identity_digest:
            return False
        if request is not None and request.fingerprint != self.request_fingerprint:
            return False
        if assignments is not None:
            normalized = tuple(sorted(dict(assignments).items()))
            if normalized != tuple(sorted(self.baseline_assignment)):
                return False
        return True

    def updated_request(self, request: ComputeRequest) -> ComputeRequest:
        """Return authored intent with this reviewed proposal applied atomically."""

        if request.fingerprint != self.request_fingerprint:
            raise PipelineOptimizationStale("compute request changed after analysis")
        preferences = dict(request.node_preferences)
        for row in self.rows:
            preference = self.preference_mapping.get(
                row.node_id,
                row.current_preference,
            )
            if preference.kind is NodePreferenceKind.AUTO:
                preferences.pop(row.node_id, None)
            else:
                preferences[row.node_id] = preference
        return replace(request, node_preferences=preferences)


class PipelineOptimizationCoordinator:
    """Solve one exact graph and accept only a validated, material improvement."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock

    def optimize(
        self,
        request: ComputeRequest,
        identity: PipelineOptimizationIdentity,
        nodes: Sequence[PipelineOptimizationNode],
        edges: Sequence[PipelineOptimizationEdge],
        evidence: Mapping[str, PipelineNodeBenchmarkEvidence],
        transfer_profile: DirectionalTransferProfile,
        validate: Callable[[PipelineValidationRequest], PipelineAssignmentValidation],
        *,
        deadline: float,
        max_assignments: int = 100_000,
        cancelled: Callable[[], bool] | None = None,
    ) -> PipelineOptimizationProposal:
        if not isinstance(request, ComputeRequest):
            raise TypeError("request must be a ComputeRequest")
        if request.mode is not ComputeMode.SELECTIVE:
            raise PipelineOptimizationEvidenceIncomplete(
                (
                    EvidenceRefusal(
                        "selective_required",
                        "Choose Selective compute policy.",
                    ),
                )
            )
        if not isinstance(identity, PipelineOptimizationIdentity):
            raise TypeError("identity must be a PipelineOptimizationIdentity")
        if not callable(validate):
            raise TypeError("validate must be callable")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None")
        self._check_abort(deadline, cancelled)
        node_values = tuple(nodes)
        edge_values = tuple(edges)
        node_ids = tuple(node.node_id for node in node_values)
        if not node_values or len(set(node_ids)) != len(node_ids):
            raise ValueError("optimizer graph requires unique non-empty nodes")
        if set(request.node_preferences) - set(node_ids) or any(
            request.preference_for(node.node_id) != node.authored_preference
            for node in node_values
        ):
            raise PipelineOptimizationStale(
                "Captured node preferences do not match the compute request."
            )
        captured_locks = tuple(
            sorted(node.node_id for node in node_values if node.optimizer_locked)
        )
        if captured_locks != identity.optimizer_locked_node_ids:
            raise PipelineOptimizationStale(
                "Captured optimizer locks do not match the optimization identity."
            )
        if set(identity.workload_fingerprints) != set(node_ids):
            raise PipelineOptimizationEvidenceIncomplete(
                (
                    EvidenceRefusal(
                        "workload_identity_incomplete",
                        "Exact workload identity is not available for every node.",
                    ),
                )
            )
        if (
            transfer_profile.identity_digest != identity.digest
            or transfer_profile.environment_fingerprint
            != identity.environment_fingerprint
            or not transfer_profile.synchronized
        ):
            raise PipelineOptimizationEvidenceIncomplete(
                (
                    EvidenceRefusal(
                        "transfer_profile_stale",
                        "Directional transfer evidence is stale or unsynchronized.",
                    ),
                )
            )
        by_evidence = dict(evidence)
        graph_nodes: list[GraphCostNode] = []
        refusals: list[EvidenceRefusal] = []
        allowed_by_node: dict[str, tuple[PipelineOptimizationCandidate, ...]] = {}
        for node in node_values:
            self._check_abort(deadline, cancelled)
            allowed = _allowed_candidates(
                node,
            )
            allowed_by_node[node.node_id] = allowed
            allowed_ids = {item.implementation_id for item in allowed}
            if node.current_implementation_id not in allowed_ids:
                refusals.append(
                    EvidenceRefusal(
                        "current_assignment_outside_constraints",
                        "The current implementation is unavailable or outside the "
                        "captured authored constraint.",
                        node.node_id,
                    )
                )
                continue
            unsupported_runtimes = sorted(
                {
                    item.runtime_id
                    for item in allowed
                    if item.runtime_id
                    not in {
                        transfer_profile.host_runtime_id,
                        transfer_profile.accelerator_runtime_id,
                    }
                }
            )
            if unsupported_runtimes:
                refusals.append(
                    EvidenceRefusal(
                        "transfer_runtime_unsupported",
                        "Directional transfer evidence does not cover runtime(s): "
                        + ", ".join(unsupported_runtimes),
                        node.node_id,
                    )
                )
                continue
            costs: list[GraphImplementationCost] = []
            variable = len(allowed) > 1
            node_evidence = by_evidence.get(node.node_id)
            results = {}
            if variable:
                reason = _evidence_refusal(node, node_evidence, identity)
                if reason is not None:
                    refusals.append(reason)
                    continue
                assert node_evidence is not None
                results = {
                    item.implementation_id: item
                    for item in node_evidence.record.candidates
                }
            for candidate in allowed:
                if variable:
                    result = results.get(candidate.implementation_id)
                    cost, refusal = _candidate_cost(
                        node,
                        candidate,
                        result,
                        host_runtime_id=transfer_profile.host_runtime_id,
                    )
                    if refusal is not None:
                        if refusal.code == "candidate_parity_failed":
                            continue
                        refusals.append(refusal)
                        continue
                else:
                    cost = 0.0
                    result = None
                workspace = candidate.minimum_workspace_bytes
                if result is not None:
                    workspace = max(workspace, int(result.peak_memory_bytes))
                costs.append(
                    GraphImplementationCost(
                        candidate.implementation_id,
                        candidate.runtime_id,
                        cost,
                        workspace_bytes=workspace,
                        host_materialization_seconds=(
                            _candidate_host_materialization_cost(
                                candidate,
                                result,
                                host_runtime_id=transfer_profile.host_runtime_id,
                            )
                        ),
                        host_output_only=candidate.host_output_only,
                        available=candidate.available,
                    )
                )
            qualified_ids = {item.implementation_id for item in costs}
            allowed_by_node[node.node_id] = tuple(
                item for item in allowed if item.implementation_id in qualified_ids
            )
            if node.current_implementation_id not in qualified_ids:
                refusals.append(
                    EvidenceRefusal(
                        "current_candidate_unqualified",
                        "The captured current implementation did not produce "
                        "complete parity-qualified timing evidence.",
                        node.node_id,
                    )
                )
                continue
            if not costs:
                refusals.append(
                    EvidenceRefusal(
                        "no_qualified_candidate",
                        "No candidate has complete parity-qualified timing evidence.",
                        node.node_id,
                    )
                )
                continue
            graph_nodes.append(
                GraphCostNode(
                    node.node_id,
                    tuple(costs),
                    output_bytes=node.output_bytes,
                    host_input_bytes=node.host_input_bytes,
                    requires_host_output=(
                        node.requires_host_output or node.cache_retained
                    ),
                    forced_implementation_id=(
                        node.current_implementation_id
                        if node.is_writer
                        or node.has_side_effects
                        or node.optimizer_locked
                        else ""
                    ),
                )
            )
        if refusals:
            raise PipelineOptimizationEvidenceIncomplete(tuple(refusals))

        transitions = (
            RuntimeTransitionCost(
                transfer_profile.host_runtime_id,
                transfer_profile.accelerator_runtime_id,
                transfer_profile.host_to_accelerator_fixed_seconds,
                transfer_profile.host_to_accelerator_seconds_per_byte,
            ),
            RuntimeTransitionCost(
                transfer_profile.accelerator_runtime_id,
                transfer_profile.host_runtime_id,
                transfer_profile.accelerator_to_host_fixed_seconds,
                transfer_profile.accelerator_to_host_seconds_per_byte,
            ),
        )
        graph_edges = tuple(
            GraphCostEdge(edge.source_node_id, edge.target_node_id)
            for edge in edge_values
        )
        base_problem = GraphOptimizationProblem(
            tuple(graph_nodes),
            graph_edges,
            transitions,
            runtime_memory_limits={
                transfer_profile.accelerator_runtime_id: (
                    transfer_profile.accelerator_memory_limit_bytes
                )
            },
            host_runtime_id=transfer_profile.host_runtime_id,
            max_assignments=max_assignments,
        )
        try:
            proposed = optimize_graph_assignment(
                base_problem,
                cancelled=lambda: self._cancelled_or_expired(deadline, cancelled),
            )
            current_problem = replace(
                base_problem,
                nodes=tuple(
                    replace(
                        graph_node,
                        forced_implementation_id=next(
                            node.current_implementation_id
                            for node in node_values
                            if node.node_id == graph_node.node_id
                        ),
                    )
                    for graph_node in base_problem.nodes
                ),
            )
            current = optimize_graph_assignment(
                current_problem,
                cancelled=lambda: self._cancelled_or_expired(deadline, cancelled),
            )
        except GraphOptimizationCancelled as exc:
            self._check_abort(deadline, cancelled)
            raise PipelineOptimizationCancelled(str(exc)) from exc
        except (GraphOptimizationError, NoFeasibleGraphAssignment) as exc:
            raise PipelineOptimizationEvidenceIncomplete(
                (EvidenceRefusal("no_feasible_assignment", str(exc)),)
            ) from exc
        self._check_abort(deadline, cancelled)
        current_assignment = tuple(current.assignments)
        tested_assignment = tuple(proposed.assignments)
        proposed_assignment = tested_assignment
        validation_performed = tested_assignment != current_assignment
        if validation_performed:
            validation_request = PipelineValidationRequest(
                identity.digest,
                current_assignment,
                tested_assignment,
            )
            validation = validate(validation_request)
            self._check_abort(deadline, cancelled)
            if not isinstance(validation, PipelineAssignmentValidation):
                raise TypeError("validate must return PipelineAssignmentValidation")
            if (
                validation.identity_digest != identity.digest
                or validation.current_assignment != current_assignment
                or validation.proposed_assignment != tested_assignment
            ):
                raise PipelineOptimizationStale(
                    "Whole-pipeline validation did not echo the exact proposal "
                    "identity."
                )
            if not validation.parity_passed or not validation.synchronized:
                raise PipelineOptimizationEvidenceIncomplete(
                    (
                        EvidenceRefusal(
                            "pipeline_validation_failed",
                            validation.detail
                            or "The proposed pipeline failed parity or "
                            "synchronization.",
                        ),
                    )
                )
            savings = validation.current_seconds - validation.proposed_seconds
            proposed_required = max(
                _MINIMUM_ABSOLUTE_IMPROVEMENT_SECONDS,
                _MINIMUM_RELATIVE_IMPROVEMENT * validation.current_seconds,
            )
            proposed_wins = (
                savings > proposed_required
                and validation.speedup_lower_confidence_bound > 1.0
            )
            current_required = max(
                _MINIMUM_ABSOLUTE_IMPROVEMENT_SECONDS,
                _MINIMUM_RELATIVE_IMPROVEMENT * validation.proposed_seconds,
            )
            current_wins = (
                -savings > current_required
                and validation.current_speedup_lower_confidence_bound > 1.0
            )
            if current_wins:
                proposed_assignment = current_assignment
                validation_winner = PipelineValidationWinner.CURRENT
            elif proposed_wins:
                validation_winner = PipelineValidationWinner.PROPOSED
            else:
                raise PipelineOptimizationNotBeneficial(
                    "Neither assignment decisively exceeded the other by the "
                    "greater of 5% or 10 ms with a paired lower confidence "
                    "bound above 1.0."
                )
        else:
            validation = PipelineAssignmentValidation(
                identity.digest,
                current_assignment,
                proposed_assignment,
                True,
                True,
                current.total_seconds,
                proposed.total_seconds,
                1.0,
                "The measured model retained the current exact assignment.",
            )
            validation_winner = PipelineValidationWinner.CURRENT

        current_map = dict(current_assignment)
        proposed_map = dict(proposed_assignment)
        rows: list[PipelineOptimizationRow] = []
        preferences: dict[str, NodeComputePreference] = {}
        for node in node_values:
            allowed = allowed_by_node[node.node_id]
            proposed_preference = _proposed_preference(
                node,
                proposed_map[node.node_id],
                allowed,
                host_runtime_id=transfer_profile.host_runtime_id,
            )
            row = PipelineOptimizationRow(
                node.node_id,
                current_map[node.node_id],
                proposed_map[node.node_id],
                node.authored_preference,
                proposed_preference,
                current_map[node.node_id] != proposed_map[node.node_id],
                len(allowed) > 1 and not node.is_writer and not node.has_side_effects,
                node.optimizer_locked,
            )
            rows.append(row)
            preferences[node.node_id] = proposed_preference
        return PipelineOptimizationProposal(
            identity.digest,
            request.fingerprint,
            current_assignment,
            tuple(rows),
            preferences,
            current.total_seconds,
            proposed.total_seconds,
            validation.current_seconds,
            validation.proposed_seconds,
            validation.speedup_lower_confidence_bound,
            validation_performed,
            validation.measurement_rounds,
            validation.current_speedup_lower_confidence_bound,
            validation_winner,
            tested_assignment,
        )

    def _cancelled_or_expired(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        return bool((cancelled is not None and cancelled()) or self.clock() >= deadline)

    def _check_abort(
        self,
        deadline: float,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        if cancelled is not None and cancelled():
            raise PipelineOptimizationCancelled("pipeline optimization cancelled")
        value = float(self.clock())
        if not math.isfinite(value):
            raise PipelineOptimizationError(
                "optimizer clock returned a non-finite value"
            )
        if value >= float(deadline):
            raise PipelineOptimizationDeadlineExceeded(
                "Pipeline optimization exceeded its end-to-end deadline."
            )


def _allowed_candidates(
    node: PipelineOptimizationNode,
) -> tuple[PipelineOptimizationCandidate, ...]:
    available = tuple(item for item in node.candidates if item.available)
    if node.is_writer or node.has_side_effects or node.optimizer_locked:
        return tuple(
            item
            for item in available
            if item.implementation_id == node.current_implementation_id
        )
    return available


def _evidence_refusal(
    node: PipelineOptimizationNode,
    evidence: PipelineNodeBenchmarkEvidence | None,
    identity: PipelineOptimizationIdentity,
) -> EvidenceRefusal | None:
    if evidence is None:
        return EvidenceRefusal(
            "benchmark_missing",
            "Exact node benchmark evidence is missing.",
            node.node_id,
        )
    if evidence.node_id != node.node_id or evidence.identity_digest != identity.digest:
        return EvidenceRefusal(
            "benchmark_identity_mismatch",
            "Node benchmark evidence belongs to a different pipeline identity.",
            node.node_id,
        )
    record = evidence.record
    if (
        record.key.workload_fingerprint
        != identity.workload_fingerprints[node.node_id]
        or record.key.environment_fingerprint
        != identity.benchmark_environment_fingerprint
    ):
        return EvidenceRefusal(
            "benchmark_key_mismatch",
            "Node benchmark workload or environment is stale.",
            node.node_id,
        )
    expected = {item.implementation_id for item in node.candidates}
    observed_ids = tuple(item.implementation_id for item in record.candidates)
    observed = set(observed_ids)
    if len(observed) != len(observed_ids) or expected != observed:
        return EvidenceRefusal(
            "benchmark_candidate_mismatch",
            "Node benchmark candidate set is incomplete or changed.",
            node.node_id,
        )
    return None


def _candidate_cost(
    node: PipelineOptimizationNode,
    candidate: PipelineOptimizationCandidate,
    result: BenchmarkCandidateResult | None,
    *,
    host_runtime_id: str,
) -> tuple[float, EvidenceRefusal | None]:
    if result is None:
        return 0.0, EvidenceRefusal(
            "candidate_timing_missing",
            f"Timing for {candidate.implementation_id!r} is missing.",
            node.node_id,
        )
    if (
        result.failure_kind
        is BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY
    ):
        return 0.0, EvidenceRefusal(
            "candidate_parity_failed",
            f"{candidate.implementation_id!r} did not pass scientific parity.",
            node.node_id,
        )
    if not result.parity_passed or result.error:
        return 0.0, EvidenceRefusal(
            "candidate_runtime_failed",
            f"{candidate.implementation_id!r} had a retryable benchmark "
            "runtime/timing failure.",
            node.node_id,
        )
    if candidate.runtime_id == host_runtime_id:
        if not result.warm_seconds:
            return 0.0, EvidenceRefusal(
                "cpu_timing_incomplete",
                "CPU warm timing evidence is incomplete.",
                node.node_id,
            )
        return float(statistics.median(result.warm_seconds)), None
    if (
        not result.synchronized
        or not result.warm_resident_seconds
        or len(result.warm_resident_seconds) != len(result.warm_seconds)
    ):
        return 0.0, EvidenceRefusal(
            "gpu_resident_timing_incomplete",
            f"{candidate.implementation_id!r} lacks synchronized resident timings.",
            node.node_id,
        )
    if candidate.host_output_only and (
        not result.warm_host_materialization_seconds
        or len(result.warm_host_materialization_seconds)
        != len(result.warm_seconds)
    ):
        return 0.0, EvidenceRefusal(
            "gpu_host_finalization_timing_incomplete",
            f"{candidate.implementation_id!r} lacks typed host-finalization "
            "timings.",
            node.node_id,
        )
    return float(statistics.median(result.warm_resident_seconds)), None


def _candidate_host_materialization_cost(
    candidate: PipelineOptimizationCandidate,
    result: BenchmarkCandidateResult | None,
    *,
    host_runtime_id: str,
) -> float:
    if (
        result is None
        or candidate.runtime_id == host_runtime_id
        or not candidate.host_output_only
        or not result.warm_host_materialization_seconds
    ):
        return 0.0
    return float(statistics.median(result.warm_host_materialization_seconds))


def _proposed_preference(
    node: PipelineOptimizationNode,
    implementation_id: str,
    allowed: Sequence[PipelineOptimizationCandidate],
    *,
    host_runtime_id: str,
) -> NodeComputePreference:
    if (
        implementation_id == node.current_implementation_id
        and (
            len(allowed) <= 1
            or node.is_writer
            or node.has_side_effects
            or node.optimizer_locked
            or node.authored_preference.kind
            is NodePreferenceKind.IMPLEMENTATION
        )
    ):
        # Fixed/excluded rows had no measured alternative and must not acquire a
        # new CPU pin.  An exact authored pin already preserves the assignment.
        # Broader eligible Auto/Best GPU/library rows are still narrowed below.
        return node.authored_preference
    selected = next(
        item for item in node.candidates if item.implementation_id == implementation_id
    )
    if selected.runtime_id == host_runtime_id:
        return NodeComputePreference(NodePreferenceKind.CPU)
    same_library = tuple(
        item
        for item in node.candidates
        if item.implementation_library_id == selected.implementation_library_id
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


def _validated_assignment(
    assignment: Sequence[tuple[str, str]],
    name: str,
) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for raw_item in assignment:
        try:
            raw_node_id, raw_implementation_id = raw_item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} entries must be node/implementation pairs"
            ) from exc
        node_id = str(raw_node_id).strip()
        implementation_id = str(raw_implementation_id).strip()
        if not node_id or not implementation_id:
            raise ValueError(f"{name} IDs must not be empty")
        values.append((node_id, implementation_id))
    node_ids = tuple(node_id for node_id, _implementation_id in values)
    if not values or len(set(node_ids)) != len(node_ids):
        raise ValueError(f"{name} must contain unique, non-empty nodes")
    return tuple(values)


__all__ = [
    "DirectionalTransferProfile",
    "EvidenceRefusal",
    "PipelineAssignmentValidation",
    "PipelineNodeBenchmarkEvidence",
    "PipelineOptimizationCancelled",
    "PipelineOptimizationCandidate",
    "PipelineOptimizationCoordinator",
    "PipelineOptimizationDeadlineExceeded",
    "PipelineOptimizationEdge",
    "PipelineOptimizationError",
    "PipelineOptimizationEvidenceIncomplete",
    "PipelineOptimizationIdentity",
    "PipelineOptimizationNode",
    "PipelineOptimizationNotBeneficial",
    "PipelineOptimizationProposal",
    "PipelineOptimizationRow",
    "PipelineOptimizationStale",
    "PipelineOptimizationTimeoutReport",
    "PipelineValidationRequest",
    "PipelineValidationWinner",
]
