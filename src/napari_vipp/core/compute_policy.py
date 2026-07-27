"""Pure policy identifiers and declaration validation.

Policy records are versioned data.  This initial catalog validates references
without making a GPU performance decision; Commit C extends it with workload
support and benefit evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from napari_vipp.core.compute import (
    ComputeEnvironment,
    DecisionReason,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_specs import OperationComputeSpec


class PolicyKind(StrEnum):
    ENVIRONMENT = "environment"
    PARAMETER = "parameter"
    WORKLOAD = "workload"
    PARITY = "parity"
    MEMORY = "memory"
    SHAPE = "shape"
    OUTPUT_DTYPE = "output_dtype"
    CONVERSION = "conversion"
    NONFINITE = "nonfinite"
    ROUNDING = "rounding"
    OVERFLOW = "overflow"
    BOUNDARY = "boundary"
    PRECISION = "precision"
    PROGRESS = "progress"
    CANCELLATION = "cancellation"
    SIDE_EFFECT = "side_effect"
    DYNAMIC_OUTPUT = "dynamic_output"


class FactCompleteness(StrEnum):
    COMPLETE = "complete"
    SAMPLED = "sampled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ArrayFacts:
    """Revision-keyed facts used by support and performance policy."""

    shape: tuple[int, ...]
    dtype: str
    element_count: int
    revision_fingerprint: str
    completeness: FactCompleteness | str = FactCompleteness.UNKNOWN
    finite_count: int | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    label_maximum: int | None = None
    label_count: int | None = None
    foreground_density: float | None = None
    strides: tuple[int, ...] | None = None
    contiguous: bool | None = None
    guarantees: tuple[str, ...] = ()
    scan_seconds: float = 0.0

    def __post_init__(self) -> None:
        completeness = (
            self.completeness
            if isinstance(self.completeness, FactCompleteness)
            else FactCompleteness(str(self.completeness).strip().lower())
        )
        shape = tuple(self.shape)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise ValueError("shape requires non-negative integer dimensions.")
        if (
            isinstance(self.element_count, bool)
            or not isinstance(self.element_count, int)
            or self.element_count < 0
        ):
            raise ValueError("element_count must be a non-negative integer.")
        if math.prod(shape) != self.element_count:
            raise ValueError("element_count must equal the product of shape.")
        dtype = str(self.dtype).strip()
        if not dtype:
            raise ValueError("dtype must not be empty.")
        for name in ("finite_count", "label_maximum", "label_count"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if self.finite_count is not None and self.finite_count > self.element_count:
            raise ValueError("finite_count must not exceed element_count.")
        for name in ("minimum", "maximum", "foreground_density"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite or None.")
        if self.foreground_density is not None and not (
            0.0 <= self.foreground_density <= 1.0
        ):
            raise ValueError("foreground_density must be between zero and one.")
        if self.contiguous is not None and not isinstance(self.contiguous, bool):
            raise TypeError("contiguous must be a boolean or None.")
        strides = None if self.strides is None else tuple(self.strides)
        if strides is not None and (
            len(strides) != len(shape)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in strides
            )
        ):
            raise ValueError("strides must contain one integer per shape dimension.")
        if (
            isinstance(self.scan_seconds, bool)
            or not isinstance(self.scan_seconds, (int, float))
            or not math.isfinite(float(self.scan_seconds))
            or self.scan_seconds < 0
        ):
            raise ValueError("scan_seconds must be finite and non-negative.")
        revision = str(self.revision_fingerprint).strip()
        if not revision:
            raise ValueError("revision_fingerprint must not be empty.")
        guarantees = tuple(
            dict.fromkeys(
                str(value).strip() for value in self.guarantees if str(value).strip()
            )
        )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "completeness", completeness)
        object.__setattr__(self, "revision_fingerprint", revision)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "guarantees", guarantees)
        object.__setattr__(self, "scan_seconds", float(self.scan_seconds))

    @property
    def all_finite(self) -> bool | None:
        if self.finite_count is None:
            return None
        return self.finite_count == self.element_count


@dataclass(frozen=True, slots=True)
class ValueDescriptor:
    shape: tuple[int, ...]
    dtype: str
    schema_id: str = "array-v1"
    guarantees: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in shape
        ):
            raise ValueError("shape requires non-negative integer dimensions.")
        dtype = str(self.dtype).strip()
        schema_id = str(self.schema_id).strip()
        if not dtype or not schema_id:
            raise ValueError("dtype and schema_id must not be empty.")
        guarantees = tuple(
            dict.fromkeys(
                str(value).strip() for value in self.guarantees if str(value).strip()
            )
        )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "schema_id", schema_id)
        object.__setattr__(self, "guarantees", guarantees)


@dataclass(frozen=True, slots=True)
class ArrayFactsKey:
    """Identity for facts that become stale when a source revision changes."""

    output_port: OutputPortKey
    revision_fingerprint: str
    fact_policy_id: str = "array-facts-v1"

    def __post_init__(self) -> None:
        for name in ("revision_fingerprint", "fact_policy_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)


class ArrayFactsCache:
    """Small revision-aware cache; sampled and complete records never alias."""

    def __init__(self) -> None:
        self._records: dict[ArrayFactsKey, ArrayFacts] = {}

    def get(self, key: ArrayFactsKey) -> ArrayFacts | None:
        return self._records.get(key)

    def put(self, key: ArrayFactsKey, facts: ArrayFacts) -> None:
        if key.revision_fingerprint != facts.revision_fingerprint:
            raise ValueError("facts revision does not match its cache key.")
        self.invalidate_port(key.output_port, keep_revision=key.revision_fingerprint)
        self._records[key] = facts

    def invalidate_port(
        self,
        output_port: OutputPortKey,
        *,
        keep_revision: str = "",
    ) -> int:
        stale = tuple(
            key
            for key in self._records
            if key.output_port == output_port
            and (not keep_revision or key.revision_fingerprint != keep_revision)
        )
        for key in stale:
            del self._records[key]
        return len(stale)

    def clear(self) -> None:
        self._records.clear()


@dataclass(frozen=True, slots=True)
class SupportDecision:
    supported: bool
    reason: DecisionReason | str
    reason_text: str
    requires_complete_facts: bool = False

    def __post_init__(self) -> None:
        reason = (
            self.reason
            if isinstance(self.reason, DecisionReason)
            else DecisionReason(str(self.reason).strip().lower())
        )
        reason_text = str(self.reason_text).strip()
        if not reason_text:
            raise ValueError("reason_text must not be empty.")
        if not isinstance(self.supported, bool):
            raise TypeError("supported must be a boolean.")
        if not isinstance(self.requires_complete_facts, bool):
            raise TypeError("requires_complete_facts must be a boolean.")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "reason_text", reason_text)


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    cpu_seconds: float
    candidate_seconds: float
    transfer_seconds: float = 0.0
    fact_scan_seconds: float = 0.0
    lower_confidence_speedup: float | None = None
    local_benchmark: bool = False

    def __post_init__(self) -> None:
        for name in (
            "cpu_seconds",
            "candidate_seconds",
            "transfer_seconds",
            "fact_scan_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.lower_confidence_speedup is not None and (
            not math.isfinite(float(self.lower_confidence_speedup))
            or self.lower_confidence_speedup < 0
        ):
            raise ValueError("lower_confidence_speedup must be non-negative.")

    @property
    def end_to_end_candidate_seconds(self) -> float:
        return self.candidate_seconds + self.transfer_seconds + self.fact_scan_seconds


@dataclass(frozen=True, slots=True)
class PerformanceDecision:
    select_candidate: bool
    reason: DecisionReason
    reason_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.select_candidate, bool):
            raise TypeError("select_candidate must be a boolean.")
        reason = (
            self.reason
            if isinstance(self.reason, DecisionReason)
            else DecisionReason(str(self.reason).strip().lower())
        )
        reason_text = str(self.reason_text).strip()
        if not reason_text:
            raise ValueError("reason_text must not be empty.")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "reason_text", reason_text)


CUDA_ENVIRONMENT_POLICIES = {
    "cuda-cupy-py312-windows-linux-v1",
    "cuda-cupy-cucim-py312-windows-linux-v1",
}


def evaluate_candidate_support(
    spec: OperationComputeSpec,
    workload: WorkloadDescriptor,
    environment: ComputeEnvironment,
    *,
    allow_experimental: bool,
    array_facts: tuple[ArrayFacts, ...] = (),
) -> SupportDecision:
    """Evaluate declared scientific/environment support without timing."""

    if not spec.visible_for(allow_experimental=allow_experimental):
        return SupportDecision(
            False,
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            "The implementation is not exposed in this admission tier.",
        )
    if spec.runtime_id == "cpu-numpy":
        return SupportDecision(True, DecisionReason.EXPLICIT_CPU, "CPU is supported.")
    if spec.validated_environment_policy_id not in CUDA_ENVIRONMENT_POLICIES:
        return SupportDecision(
            False,
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            "No executable environment policy is registered.",
        )
    if (
        environment.os_name not in {"Windows", "Linux"}
        or environment.python_implementation != "CPython"
        or environment.python_version != "3.12"
        or not environment.python_abi.startswith("cpython-312")
        or spec.runtime_id not in environment.runtime_ids
        or spec.implementation_library_id not in environment.implementation_libraries
        or environment.probe_status != "available"
    ):
        return SupportDecision(
            False,
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            "The current OS/Python/runtime/library probe is outside the "
            "validated matrix.",
        )
    if (
        workload.resolved_spatial_ndim is not None
        and workload.resolved_spatial_ndim not in spec.supported_spatial_ndims
    ):
        return SupportDecision(
            False,
            DecisionReason.WORKLOAD_UNSUPPORTED,
            "The resolved spatial dimensionality is unsupported.",
        )
    if len(workload.input_dtypes) != len(spec.input_ports):
        return SupportDecision(
            False,
            DecisionReason.WORKLOAD_UNSUPPORTED,
            "The resolved input-port count does not match the declaration.",
        )
    for dtype, port in zip(workload.input_dtypes, spec.input_ports, strict=True):
        if "*" not in port.public_dtypes and dtype not in port.public_dtypes:
            return SupportDecision(
                False,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                f"Input dtype {dtype!r} is outside the declared port region.",
            )
    requires_finite = "finite-only" in spec.limitations or any(
        port.nonfinite_policy_id == "finite-only-v1" for port in spec.input_ports
    )
    if requires_finite:
        if len(array_facts) != len(spec.input_ports):
            return SupportDecision(
                False,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                "Complete finite-value facts are required.",
                requires_complete_facts=True,
            )
        if any(
            facts.completeness is not FactCompleteness.COMPLETE
            or facts.all_finite is not True
            for facts in array_facts
        ):
            return SupportDecision(
                False,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                "This implementation is admitted only for completely finite inputs.",
                requires_complete_facts=True,
            )
    return SupportDecision(
        True,
        DecisionReason.SELECTED_IMPLEMENTATION,
        "The candidate is inside its declared scientific and environment region.",
    )


def propagate_output_descriptors(
    spec: OperationComputeSpec,
    inputs: tuple[ValueDescriptor, ...],
) -> tuple[ValueDescriptor, ...] | None:
    """Apply pure shape/dtype rules or return ``None`` for conservative CPU."""

    if not inputs:
        return None
    if spec.shape_policy_id not in {"shape-preserving-v1", "cpu-reference-v1"}:
        return None
    if spec.shape_policy_id == "cpu-reference-v1" and spec.is_gpu:
        return None
    primary = inputs[0]
    outputs = []
    for port in spec.output_ports:
        dtype = primary.dtype
        if port.output_dtype_policy_id.startswith("fixed:"):
            dtype = port.output_dtype_policy_id.partition(":")[2]
        elif port.output_dtype_policy_id not in {"dtype-same-v1"}:
            return None
        outputs.append(
            ValueDescriptor(primary.shape, dtype, port.schema_id, primary.guarantees)
        )
    return tuple(outputs)


def evaluate_auto_performance(
    evidence: PerformanceEvidence,
    *,
    minimum_lower_confidence_speedup: float = 1.20,
    minimum_saving_seconds: float = 0.020,
    local_noise_fraction: float = 0.05,
    local_noise_seconds: float = 0.010,
) -> PerformanceDecision:
    """Apply the conservative non-benchmarked or local Auto gate."""

    thresholds = (
        minimum_lower_confidence_speedup,
        minimum_saving_seconds,
        local_noise_fraction,
        local_noise_seconds,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in thresholds
    ):
        raise ValueError("performance thresholds must be finite and non-negative.")
    candidate = evidence.end_to_end_candidate_seconds
    saving = evidence.cpu_seconds - candidate
    if evidence.local_benchmark:
        noise = max(local_noise_seconds, local_noise_fraction * evidence.cpu_seconds)
        accepted = saving > noise
        text = (
            "Local paired evidence shows a clear end-to-end win."
            if accepted
            else "The measured difference is within the versioned local noise floor."
        )
    else:
        lower_bound = evidence.lower_confidence_speedup
        accepted = (
            lower_bound is not None
            and lower_bound >= minimum_lower_confidence_speedup
            and saving >= minimum_saving_seconds
        )
        text = (
            "The validated lower-confidence prediction clears the Auto gate."
            if accepted
            else "The candidate does not clear the conservative Auto performance gate."
        )
    return PerformanceDecision(
        accepted,
        (
            DecisionReason.SELECTED_IMPLEMENTATION
            if accepted
            else DecisionReason.PERFORMANCE_GATE
        ),
        text,
    )


@dataclass(frozen=True, slots=True)
class PolicyCatalog:
    """Immutable known-policy IDs grouped by semantic role."""

    policies: Mapping[PolicyKind | str, Iterable[str]]

    def __post_init__(self) -> None:
        normalized: dict[PolicyKind, frozenset[str]] = {}
        for raw_kind, raw_ids in self.policies.items():
            kind = (
                raw_kind
                if isinstance(raw_kind, PolicyKind)
                else PolicyKind(str(raw_kind).strip().lower())
            )
            identifiers = frozenset(
                str(identifier).strip()
                for identifier in raw_ids
                if str(identifier).strip()
            )
            if not identifiers:
                raise ValueError(f"{kind.value} must declare at least one policy ID.")
            normalized[kind] = identifiers
        missing = set(PolicyKind) - set(normalized)
        if missing:
            names = ", ".join(sorted(kind.value for kind in missing))
            raise ValueError(f"Policy catalog is missing kind(s): {names}.")
        object.__setattr__(self, "policies", MappingProxyType(normalized))

    def contains(self, kind: PolicyKind, policy_id: str) -> bool:
        return str(policy_id) in self.policies[kind]


DEFAULT_POLICY_CATALOG = PolicyCatalog(
    {
        PolicyKind.ENVIRONMENT: {"vipp-cpu-supported-v1"},
        PolicyKind.PARAMETER: {"cpu-reference-parameters-v1"},
        PolicyKind.WORKLOAD: {"cpu-reference-v1", "vipp-best-available-v1"},
        PolicyKind.PARITY: {"authoritative-cpu-v1"},
        PolicyKind.MEMORY: {"host-reference-v1"},
        PolicyKind.SHAPE: {
            "cpu-reference-v1",
            "cpu-dynamic-output-v1",
            "shape-unknown-v1",
        },
        PolicyKind.OUTPUT_DTYPE: {
            "dtype-same-v1",
            "cpu-dynamic-output-v1",
        },
        PolicyKind.CONVERSION: {"identity-v1"},
        PolicyKind.NONFINITE: {"cpu-reference-v1"},
        PolicyKind.ROUNDING: {"cpu-reference-v1"},
        PolicyKind.OVERFLOW: {"cpu-reference-v1"},
        PolicyKind.BOUNDARY: {"cpu-reference-v1"},
        PolicyKind.PRECISION: {"scientific-default-v1"},
        PolicyKind.PROGRESS: {"cpu-reference-v1"},
        PolicyKind.CANCELLATION: {"cpu-reference-v1"},
        PolicyKind.SIDE_EFFECT: {"pure-or-source-v1", "host-writer-v1"},
        PolicyKind.DYNAMIC_OUTPUT: {"static-v1", "cpu-dynamic-output-v1"},
    }
)


def validate_spec_policy_references(
    spec: OperationComputeSpec,
    *,
    catalog: PolicyCatalog = DEFAULT_POLICY_CATALOG,
) -> None:
    """Fail when a declaration references an unknown policy record."""

    references = (
        (PolicyKind.ENVIRONMENT, spec.validated_environment_policy_id),
        (PolicyKind.PARAMETER, spec.parameter_policy_id),
        (PolicyKind.WORKLOAD, spec.workload_policy_id),
        (PolicyKind.PARITY, spec.parity_policy_id),
        (PolicyKind.MEMORY, spec.memory_model_id),
        (PolicyKind.SHAPE, spec.shape_policy_id),
        (PolicyKind.BOUNDARY, spec.boundary_policy_id),
        (PolicyKind.PRECISION, spec.precision_policy_id),
        (PolicyKind.PROGRESS, spec.progress_policy_id),
        (PolicyKind.CANCELLATION, spec.cancellation_policy_id),
        (PolicyKind.SIDE_EFFECT, spec.side_effect_policy_id),
        (PolicyKind.DYNAMIC_OUTPUT, spec.dynamic_output_policy_id),
    )
    for port in (*spec.input_ports, *spec.output_ports):
        references += (
            (PolicyKind.SHAPE, port.shape_policy_id),
            (PolicyKind.OUTPUT_DTYPE, port.output_dtype_policy_id),
            (PolicyKind.CONVERSION, port.conversion_policy_id),
            (PolicyKind.NONFINITE, port.nonfinite_policy_id),
            (PolicyKind.ROUNDING, port.rounding_policy_id),
            (PolicyKind.OVERFLOW, port.overflow_policy_id),
            (PolicyKind.BOUNDARY, port.boundary_policy_id),
            (PolicyKind.PRECISION, port.precision_policy_id),
        )
    for kind, policy_id in references:
        if not catalog.contains(kind, policy_id):
            raise ValueError(
                f"Implementation {spec.implementation_id!r} references unknown "
                f"{kind.value} policy {policy_id!r}."
            )


__all__ = [
    "ArrayFacts",
    "ArrayFactsCache",
    "ArrayFactsKey",
    "CUDA_ENVIRONMENT_POLICIES",
    "DEFAULT_POLICY_CATALOG",
    "FactCompleteness",
    "PerformanceDecision",
    "PerformanceEvidence",
    "PolicyCatalog",
    "PolicyKind",
    "SupportDecision",
    "ValueDescriptor",
    "evaluate_auto_performance",
    "evaluate_candidate_support",
    "propagate_output_descriptors",
    "validate_spec_policy_references",
]
