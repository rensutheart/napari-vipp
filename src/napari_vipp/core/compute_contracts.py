"""Foundational immutable compute-contract value types.

This module contains only provider-free declaration types.  Operation-owned
compute contracts may import them without depending on the shared built-in
registration table in :mod:`napari_vipp.core.compute_specs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AdmissionTier(StrEnum):
    """Visibility of an implementation candidate."""

    DEVELOPER_HIDDEN = "developer_hidden"
    PUBLIC_SELECTIVE = "public_selective"
    PUBLIC_AUTO_CANDIDATE = "public_auto_candidate"


class ValueKind(StrEnum):
    ARRAY = "array"
    IMAGE = "image"
    LABELS = "labels"
    MASK = "mask"
    TABLE = "table"
    SCALAR = "scalar"
    ANY = "any"


@dataclass(frozen=True, slots=True)
class ComputePortContract:
    """Scientific and conversion contract for one input or output port."""

    port_index: int
    value_kind: ValueKind | str
    port_name: str = ""
    public_dtypes: tuple[str, ...] = ("*",)
    internal_dtypes: tuple[str, ...] = ("same",)
    accumulation_dtype: str = "same"
    value_domain: str = "any"
    shape_policy_id: str = "shape-unknown-v1"
    output_dtype_policy_id: str = "dtype-same-v1"
    conversion_policy_id: str = "identity-v1"
    nonfinite_policy_id: str = "cpu-reference-v1"
    rounding_policy_id: str = "cpu-reference-v1"
    overflow_policy_id: str = "cpu-reference-v1"
    boundary_policy_id: str = "cpu-reference-v1"
    precision_policy_id: str = "scientific-default-v1"
    schema_id: str = "array-v1"

    def __post_init__(self) -> None:
        if self.port_index < 0:
            raise ValueError("port_index must not be negative.")
        kind = (
            self.value_kind
            if isinstance(self.value_kind, ValueKind)
            else ValueKind(str(self.value_kind).strip().lower())
        )
        public = _normalized_nonempty(self.public_dtypes, "public_dtypes")
        internal = _normalized_nonempty(self.internal_dtypes, "internal_dtypes")
        port_name = str(self.port_name).strip() or f"port_{self.port_index}"
        required = (
            "accumulation_dtype",
            "value_domain",
            "shape_policy_id",
            "output_dtype_policy_id",
            "conversion_policy_id",
            "nonfinite_policy_id",
            "rounding_policy_id",
            "overflow_policy_id",
            "boundary_policy_id",
            "precision_policy_id",
            "schema_id",
        )
        for name in required:
            normalized = str(getattr(self, name)).strip()
            if not normalized:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, normalized)
        object.__setattr__(self, "value_kind", kind)
        object.__setattr__(self, "port_name", port_name)
        object.__setattr__(self, "public_dtypes", public)
        object.__setattr__(self, "internal_dtypes", internal)


@dataclass(frozen=True, slots=True)
class OperationComputeSpec:
    """One versioned implementation of one VIPP operation."""

    operation_id: str
    implementation_id: str
    implementation_version: str
    runtime_id: str
    array_domain: str
    implementation_library_id: str
    callable_ref: str
    host_boundary: bool
    admission_tier: AdmissionTier | str
    validated_environment_policy_id: str
    input_ports: tuple[ComputePortContract, ...]
    output_ports: tuple[ComputePortContract, ...]
    parameter_policy_id: str
    workload_policy_id: str
    parity_policy_id: str
    memory_model_id: str
    shape_policy_id: str
    boundary_policy_id: str
    precision_policy_id: str
    progress_policy_id: str
    cancellation_policy_id: str
    side_effect_policy_id: str
    dynamic_output_policy_id: str = "static-v1"
    supported_spatial_ndims: tuple[int, ...] = (2, 3)
    supports_device_residency: bool = False
    limitations: tuple[str, ...] = ()
    cache_equivalence_group: str = ""
    host_finalizer_ref: str = ""

    def __post_init__(self) -> None:
        tier = (
            self.admission_tier
            if isinstance(self.admission_tier, AdmissionTier)
            else AdmissionTier(str(self.admission_tier).strip().lower())
        )
        required = (
            "operation_id",
            "implementation_id",
            "implementation_version",
            "runtime_id",
            "array_domain",
            "implementation_library_id",
            "validated_environment_policy_id",
            "parameter_policy_id",
            "workload_policy_id",
            "parity_policy_id",
            "memory_model_id",
            "shape_policy_id",
            "boundary_policy_id",
            "precision_policy_id",
            "progress_policy_id",
            "cancellation_policy_id",
            "side_effect_policy_id",
            "dynamic_output_policy_id",
        )
        for name in required:
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        callable_ref = str(self.callable_ref).strip()
        if not callable_ref and not self.host_boundary:
            raise ValueError("non-boundary implementations require callable_ref.")
        if callable_ref and ":" not in callable_ref:
            raise ValueError("callable_ref must use 'module:attribute' syntax.")
        host_finalizer_ref = str(self.host_finalizer_ref).strip()
        if host_finalizer_ref:
            finalizer_module, separator, finalizer_attribute = (
                host_finalizer_ref.partition(":")
            )
            if (
                not separator
                or not finalizer_module.strip()
                or not finalizer_attribute.strip()
            ):
                raise ValueError(
                    "host_finalizer_ref must use 'module:attribute' syntax."
                )
        if host_finalizer_ref and (
            self.host_boundary or not self.supports_device_residency
        ):
            raise ValueError(
                "host finalizers require a resident, non-boundary implementation."
            )
        if not self.output_ports:
            raise ValueError("an implementation must declare at least one output port.")
        _validate_port_indexes(self.input_ports, "input")
        _validate_port_indexes(self.output_ports, "output")
        spatial_dims = tuple(sorted(set(self.supported_spatial_ndims)))
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in {1, 2, 3}
            for value in spatial_dims
        ):
            raise ValueError("supported_spatial_ndims may contain only 1, 2, and 3.")
        object.__setattr__(self, "callable_ref", callable_ref)
        object.__setattr__(self, "host_finalizer_ref", host_finalizer_ref)
        object.__setattr__(self, "supported_spatial_ndims", spatial_dims)
        object.__setattr__(
            self,
            "limitations",
            tuple(
                str(value).strip() for value in self.limitations if str(value).strip()
            ),
        )
        object.__setattr__(self, "admission_tier", tier)

    @property
    def is_gpu(self) -> bool:
        return self.runtime_id != "cpu-numpy"

    def visible_for(self, *, allow_experimental: bool) -> bool:
        return (
            self.admission_tier is not AdmissionTier.DEVELOPER_HIDDEN
            or allow_experimental
        )

    def eligible_for_auto(self, *, allow_experimental: bool) -> bool:
        """Return whether automatic policy may consider this implementation."""

        return self.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE or (
            self.admission_tier is AdmissionTier.DEVELOPER_HIDDEN and allow_experimental
        )


def _normalized_nonempty(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value.")
    return normalized


def _validate_port_indexes(
    ports: tuple[ComputePortContract, ...],
    description: str,
) -> None:
    indexes = tuple(port.port_index for port in ports)
    if indexes != tuple(range(len(ports))):
        raise ValueError(
            f"{description} port indexes must be contiguous and zero-based."
        )


__all__ = [
    "AdmissionTier",
    "ComputePortContract",
    "OperationComputeSpec",
    "ValueKind",
]
