"""Immutable operation implementation declarations.

Declarations are intentionally separate from :mod:`napari_vipp.core.pipeline`.
They contain stable identifiers and import paths, never imported optional
callables.  The runtime registry resolves a callable only after planning selects
an admitted implementation.
"""

from __future__ import annotations

from collections.abc import Collection
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


_MICROSCOPY_DTYPES = ("uint8", "uint16", "float32")


def _gpu_image_port(
    port_index: int,
    *,
    name: str,
    public_dtypes: tuple[str, ...],
    internal_dtypes: tuple[str, ...],
    conversion_policy_id: str,
    nonfinite_policy_id: str,
    rounding_policy_id: str,
    overflow_policy_id: str,
    boundary_policy_id: str,
    precision_policy_id: str,
    output: bool = False,
) -> ComputePortContract:
    return ComputePortContract(
        port_index,
        ValueKind.IMAGE if output else ValueKind.ARRAY,
        port_name=name,
        public_dtypes=public_dtypes,
        internal_dtypes=internal_dtypes,
        accumulation_dtype=internal_dtypes[-1],
        value_domain="microscopy-intensity-v1",
        shape_policy_id="shape-preserving-v1",
        output_dtype_policy_id="dtype-same-v1",
        conversion_policy_id=conversion_policy_id,
        nonfinite_policy_id=nonfinite_policy_id,
        rounding_policy_id=rounding_policy_id,
        overflow_policy_id=overflow_policy_id,
        boundary_policy_id=boundary_policy_id,
        precision_policy_id=precision_policy_id,
    )


def _background_spec(operation_id: str) -> OperationComputeSpec:
    port_values = {
        "public_dtypes": _MICROSCOPY_DTYPES,
        "internal_dtypes": ("float32",),
        "conversion_policy_id": "background-float-workspace-restore-v1",
        "nonfinite_policy_id": "background-cpu-parity-v1",
        "rounding_policy_id": "background-bankers-round-clip-v1",
        "overflow_policy_id": "background-clip-public-dtype-v1",
        "boundary_policy_id": "background-nearest-rolling-ball-v1",
        "precision_policy_id": "background-public-dtype-v1",
    }
    return OperationComputeSpec(
        operation_id=operation_id,
        implementation_id=f"cucim-{operation_id}-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cucim",
        callable_ref=(
            "napari_vipp.core.gpu.cucim_background:"
            f"{operation_id}"
        ),
        host_boundary=False,
        admission_tier=AdmissionTier.DEVELOPER_HIDDEN,
        validated_environment_policy_id=(
            "cuda-cupy-cucim-py312-windows-linux-v1"
        ),
        input_ports=(
            _gpu_image_port(0, name="image", **port_values),
        ),
        output_ports=(
            _gpu_image_port(0, name="image", output=True, **port_values),
        ),
        parameter_policy_id="background-parameters-v1",
        workload_policy_id="background-u8-u16-f32-v1",
        parity_policy_id="background-production-exact-v1",
        memory_model_id="cucim-background-memory-v1",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="background-nearest-rolling-ball-v1",
        precision_policy_id="background-public-dtype-v1",
        progress_policy_id="background-block-progress-v1",
        cancellation_policy_id="background-block-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2, 3),
        supports_device_residency=True,
        limitations=("experimental-cucim-wheel-v1",),
    )


def _median_spec() -> OperationComputeSpec:
    port_values = {
        "public_dtypes": _MICROSCOPY_DTYPES,
        "internal_dtypes": ("same",),
        "conversion_policy_id": "cupyx-median-identity-v1",
        "nonfinite_policy_id": "finite-no-negative-zero-v1",
        "rounding_policy_id": "median-bitwise-v1",
        "overflow_policy_id": "preserve-public-dtype-v1",
        "boundary_policy_id": "scipy-reflect-v1",
        "precision_policy_id": "median-bitwise-v1",
    }
    return OperationComputeSpec(
        operation_id="median_filter",
        implementation_id="cupyx-median-filter-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref="napari_vipp.core.gpu.cupy_median:median_filter",
        host_boundary=False,
        admission_tier=AdmissionTier.DEVELOPER_HIDDEN,
        validated_environment_policy_id="cuda-cupy-py312-windows-linux-v1",
        input_ports=(
            _gpu_image_port(0, name="image", **port_values),
        ),
        output_ports=(
            _gpu_image_port(0, name="image", output=True, **port_values),
        ),
        parameter_policy_id="median-parameters-v1",
        workload_policy_id="median-exact-u8-u16-f32-v1",
        parity_policy_id="median-production-bitwise-v1",
        memory_model_id="cupyx-median-memory-v1",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="scipy-reflect-v1",
        precision_policy_id="median-bitwise-v1",
        progress_policy_id="monolithic-sync-progress-v1",
        cancellation_policy_id="monolithic-boundary-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2,),
        supports_device_residency=True,
        limitations=(
            "float32-requires-complete-finite-no-negative-zero-v1",
        ),
    )


def _gaussian_spec(*, three_dimensional: bool) -> OperationComputeSpec:
    operation_id = "gaussian_blur_3d" if three_dimensional else "gaussian_blur"
    implementation_id = (
        "cupyx-gaussian-blur-3d-v1"
        if three_dimensional
        else "cupyx-gaussian-blur-v1"
    )
    port_values = {
        # Integer execution is deliberately not advertised.  The reviewed RTX
        # matrix found content-dependent one-unit disagreements, so uint8 and
        # uint16 remain explicit, first-class CPU regions in policy.
        "public_dtypes": ("float32",),
        "internal_dtypes": ("float32",),
        "conversion_policy_id": "cupyx-gaussian-float32-v1",
        "nonfinite_policy_id": "finite-only-v1",
        "rounding_policy_id": "gaussian-float32-tolerance-v1",
        "overflow_policy_id": "preserve-public-dtype-v1",
        "boundary_policy_id": "scipy-reflect-v1",
        "precision_policy_id": "gaussian-float32-v1",
    }
    return OperationComputeSpec(
        operation_id=operation_id,
        implementation_id=implementation_id,
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref=(
            "napari_vipp.core.gpu.cupy_gaussian:"
            f"{operation_id}"
        ),
        host_boundary=False,
        admission_tier=AdmissionTier.DEVELOPER_HIDDEN,
        validated_environment_policy_id="cuda-cupy-py312-windows-linux-v1",
        input_ports=(
            _gpu_image_port(0, name="image", **port_values),
        ),
        output_ports=(
            _gpu_image_port(0, name="image", output=True, **port_values),
        ),
        parameter_policy_id=(
            "gaussian-3d-parameters-v1"
            if three_dimensional
            else "gaussian-2d-parameters-v1"
        ),
        workload_policy_id="gaussian-finite-f32-v1",
        parity_policy_id="gaussian-float32-tolerance-v1",
        memory_model_id=(
            "cupyx-gaussian-3d-memory-v1"
            if three_dimensional
            else "cupyx-gaussian-2d-memory-v1"
        ),
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="scipy-reflect-v1",
        precision_policy_id="gaussian-float32-v1",
        progress_policy_id="monolithic-sync-progress-v1",
        cancellation_policy_id="monolithic-boundary-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=((3,) if three_dimensional else (2,)),
        supports_device_residency=True,
        limitations=(
            "finite-only",
            "uint8-uint16-evaluated-cpu-v1",
            "float64-unvalidated-v1",
        ),
    )


_BUILTIN_ACCELERATOR_SPECS: tuple[OperationComputeSpec, ...] = (
    _background_spec("rolling_ball_background"),
    _background_spec("subtract_background"),
    _median_spec(),
    _gaussian_spec(three_dimensional=False),
    _gaussian_spec(three_dimensional=True),
)


def accelerator_compute_specs() -> tuple[OperationComputeSpec, ...]:
    """Return built-in accelerator declarations without importing providers."""

    return _BUILTIN_ACCELERATOR_SPECS


def compute_specs_for(
    operation_id: str,
    *,
    include_cpu: bool = True,
    allow_experimental: bool = False,
) -> tuple[OperationComputeSpec, ...]:
    """Return declared implementations for ``operation_id``.

    The CPU declaration is synthesized lazily from the authoritative operation
    library so this module does not create a pipeline import cycle.
    """

    operation_id = str(operation_id).strip()
    if not operation_id:
        raise ValueError("operation_id must not be empty.")
    selected = tuple(
        spec
        for spec in _BUILTIN_ACCELERATOR_SPECS
        if spec.operation_id == operation_id
        and spec.visible_for(allow_experimental=allow_experimental)
    )
    if include_cpu:
        return (_cpu_compute_spec(operation_id), *selected)
    return selected


def validate_compute_specs(
    specs: tuple[OperationComputeSpec, ...] | None = None,
    *,
    known_operation_ids: Collection[str] | None = None,
) -> None:
    """Validate declaration uniqueness without importing the pipeline.

    A caller that already owns the authoritative operation catalog may supply
    ``known_operation_ids`` for the optional cross-check.  Registry construction
    deliberately omits that check: importing the full pipeline there can make
    storage plugins discover optional accelerator packages before a GPU request.
    """

    declarations = _BUILTIN_ACCELERATOR_SPECS if specs is None else tuple(specs)
    implementation_ids: set[str] = set()
    identities: set[tuple[str, str]] = set()
    known = (
        None
        if known_operation_ids is None
        else frozenset(str(value).strip() for value in known_operation_ids)
    )

    for spec in declarations:
        if known is not None and spec.operation_id not in known:
            raise ValueError(
                f"Compute spec {spec.implementation_id!r} references unknown "
                f"operation {spec.operation_id!r}."
            )
        if spec.implementation_id in implementation_ids:
            raise ValueError(f"Duplicate implementation ID {spec.implementation_id!r}.")
        identity = (spec.operation_id, spec.implementation_id)
        if identity in identities:
            raise ValueError(f"Duplicate operation implementation {identity!r}.")
        implementation_ids.add(spec.implementation_id)
        identities.add(identity)


def _cpu_compute_spec(operation_id: str) -> OperationComputeSpec:
    from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID

    operation = NODE_LIBRARY_BY_ID.get(operation_id)
    if operation is None:
        raise KeyError(f"Unknown operation {operation_id!r}.")
    host_boundary = operation.function is None or operation_id in {
        "save_output",
        "batch_output",
    }
    if operation.function is None:
        callable_ref = ""
    else:
        callable_ref = (
            f"{operation.function.__module__}:{operation.function.__qualname__}"
        )
    inputs = operation.input_ports
    input_contracts = tuple(
        ComputePortContract(
            index,
            _value_kind(item.input_type),
            port_name=item.name,
            shape_policy_id="cpu-reference-v1",
        )
        for index, item in enumerate(inputs)
    )
    if operation.output_factory is not None:
        output_contracts = (
            ComputePortContract(
                0,
                ValueKind.ANY,
                port_name="dynamic",
                shape_policy_id="cpu-dynamic-output-v1",
                output_dtype_policy_id="cpu-dynamic-output-v1",
                schema_id="dynamic-ports-v1",
            ),
        )
    else:
        output_contracts = tuple(
            ComputePortContract(
                index,
                _value_kind(item.output_type),
                port_name=item.name,
                shape_policy_id="cpu-reference-v1",
            )
            for index, item in enumerate(operation.output_ports)
        )
    return OperationComputeSpec(
        operation_id=operation_id,
        implementation_id=f"cpu-{operation_id}-v1",
        implementation_version="1",
        runtime_id="cpu-numpy",
        array_domain="host-numpy",
        implementation_library_id="cpu",
        callable_ref=callable_ref,
        host_boundary=host_boundary,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id="vipp-cpu-supported-v1",
        input_ports=input_contracts,
        output_ports=output_contracts,
        parameter_policy_id="cpu-reference-parameters-v1",
        workload_policy_id="cpu-reference-v1",
        parity_policy_id="authoritative-cpu-v1",
        memory_model_id="host-reference-v1",
        shape_policy_id="cpu-reference-v1",
        boundary_policy_id="cpu-reference-v1",
        precision_policy_id="scientific-default-v1",
        progress_policy_id="cpu-reference-v1",
        cancellation_policy_id="cpu-reference-v1",
        side_effect_policy_id=(
            "host-writer-v1"
            if operation_id in {"save_output", "batch_output"}
            else "pure-or-source-v1"
        ),
        dynamic_output_policy_id=(
            "cpu-dynamic-output-v1"
            if operation.output_factory is not None
            else "static-v1"
        ),
        supported_spatial_ndims=(1, 2, 3),
        supports_device_residency=False,
    )


def _value_kind(value: str) -> ValueKind:
    normalized = str(value).strip().lower()
    aliases = {
        "array": ValueKind.ARRAY,
        "image": ValueKind.IMAGE,
        "labels": ValueKind.LABELS,
        "mask": ValueKind.MASK,
        "table": ValueKind.TABLE,
        "scalar": ValueKind.SCALAR,
    }
    return aliases.get(normalized, ValueKind.ANY)


__all__ = [
    "AdmissionTier",
    "ComputePortContract",
    "OperationComputeSpec",
    "ValueKind",
    "accelerator_compute_specs",
    "compute_specs_for",
    "validate_compute_specs",
]
