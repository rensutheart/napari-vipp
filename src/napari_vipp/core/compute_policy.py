"""Pure policy identifiers and declaration validation.

Policy records are versioned data.  This initial catalog validates references
without making a GPU performance decision; Commit C extends it with workload
support and benefit evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

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
    "DEFAULT_POLICY_CATALOG",
    "PolicyCatalog",
    "PolicyKind",
    "validate_spec_policy_references",
]
