"""Immutable operation implementation declarations.

Declarations are intentionally separate from :mod:`napari_vipp.core.pipeline`.
They contain stable identifiers and import paths, never imported optional
callables.  The runtime registry resolves a callable only after planning selects
an admitted implementation.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
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

    def eligible_for_auto(self, *, allow_experimental: bool) -> bool:
        """Return whether automatic policy may consider this implementation."""

        return self.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE or (
            self.admission_tier is AdmissionTier.DEVELOPER_HIDDEN
            and allow_experimental
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
        "precision_policy_id": "background-public-dtype-v2",
    }
    return OperationComputeSpec(
        operation_id=operation_id,
        implementation_id=f"cucim-{operation_id}-v2",
        implementation_version="2",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cucim",
        callable_ref=(f"napari_vipp.core.gpu.cucim_background:{operation_id}"),
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3"
        ),
        input_ports=(_gpu_image_port(0, name="image", **port_values),),
        output_ports=(_gpu_image_port(0, name="image", output=True, **port_values),),
        parameter_policy_id="background-parameters-v1",
        workload_policy_id="background-u8-u16-f32-v2",
        parity_policy_id="background-dtype-parity-v2",
        memory_model_id="cucim-background-memory-v1",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="background-nearest-rolling-ball-v1",
        precision_policy_id="background-public-dtype-v2",
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
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(_gpu_image_port(0, name="image", **port_values),),
        output_ports=(_gpu_image_port(0, name="image", output=True, **port_values),),
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
        limitations=("float32-requires-complete-finite-no-negative-zero-v1",),
    )


def _gaussian_spec(*, three_dimensional: bool) -> OperationComputeSpec:
    operation_id = "gaussian_blur_3d" if three_dimensional else "gaussian_blur"
    implementation_id = (
        "cupyx-gaussian-blur-3d-v1" if three_dimensional else "cupyx-gaussian-blur-v1"
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
        callable_ref=(f"napari_vipp.core.gpu.cupy_gaussian:{operation_id}"),
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(_gpu_image_port(0, name="image", **port_values),),
        output_ports=(_gpu_image_port(0, name="image", output=True, **port_values),),
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


def _richardson_lucy_spec() -> OperationComputeSpec:
    """Return the first production-faithful CuPy Richardson-Lucy contract.

    The initial admitted region is intentionally narrow: both the observed
    image and PSF must already be finite float32 arrays.  VIPP therefore never
    hides an implicit dtype conversion inside accelerator selection, and the
    fixed-float32 output contract remains explicit for future typed planning.
    """

    input_values = {
        "public_dtypes": ("float32",),
        "internal_dtypes": ("float32",),
        "conversion_policy_id": "cupyx-rl-float32-identity-v1",
        "nonfinite_policy_id": "finite-only-v1",
        "rounding_policy_id": "rl-float32-tolerance-v1",
        "overflow_policy_id": "finite-float32-cleanup-v1",
        "boundary_policy_id": "scipy-signal-zero-fill-same-v1",
        "precision_policy_id": "rl-float32-v1",
    }
    image_input = _gpu_image_port(0, name="image", **input_values)
    psf_input = ComputePortContract(
        1,
        ValueKind.ARRAY,
        port_name="psf",
        public_dtypes=("float32",),
        internal_dtypes=("float32",),
        accumulation_dtype="float64",
        value_domain="nonnegative-psf-kernel-v1",
        shape_policy_id="psf-spatial-kernel-v1",
        output_dtype_policy_id="dtype-same-v1",
        conversion_policy_id="cupyx-rl-float32-identity-v1",
        nonfinite_policy_id="finite-only-v1",
        rounding_policy_id="rl-float32-tolerance-v1",
        overflow_policy_id="finite-float32-cleanup-v1",
        boundary_policy_id="scipy-signal-zero-fill-same-v1",
        precision_policy_id="rl-float32-v1",
    )
    image_output = ComputePortContract(
        0,
        ValueKind.IMAGE,
        port_name="image",
        public_dtypes=("float32",),
        internal_dtypes=("float32",),
        accumulation_dtype="float32",
        value_domain="microscopy-intensity-v1",
        shape_policy_id="shape-preserving-v1",
        output_dtype_policy_id="fixed:float32",
        conversion_policy_id="cupyx-rl-float32-identity-v1",
        nonfinite_policy_id="finite-output-v1",
        rounding_policy_id="rl-float32-tolerance-v1",
        overflow_policy_id="finite-float32-cleanup-v1",
        boundary_policy_id="scipy-signal-zero-fill-same-v1",
        precision_policy_id="rl-float32-v1",
    )
    return OperationComputeSpec(
        operation_id="richardson_lucy_deconvolution",
        implementation_id="rl-cupy-f32-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref=("napari_vipp.core.gpu.cupy_rl:richardson_lucy_deconvolution"),
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(image_input, psf_input),
        output_ports=(image_output,),
        parameter_policy_id="rl-parameters-v1",
        workload_policy_id="rl-finite-f32-v1",
        parity_policy_id="rl-float32-tolerance-v1",
        memory_model_id="cupyx-richardson-lucy-fft-memory-v2",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="scipy-signal-zero-fill-same-v1",
        precision_policy_id="rl-float32-v1",
        progress_policy_id="deconvolution-block-iteration-progress-v1",
        cancellation_policy_id="deconvolution-iteration-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2, 3),
        supports_device_residency=True,
        limitations=(
            "finite-only",
            "float32-only-v1",
            "filter-epsilon-exactly-1e-8-v1",
            "iterations-at-most-25-v1",
            "odd-psf-extents-v1",
            "default-safe-options-v1",
        ),
    )


def _richardson_lucy_tv_spec() -> OperationComputeSpec:
    """Return the initial production-faithful CuPy RL-TV contract."""

    ordinary = _richardson_lucy_spec()
    boundary_policy_id = "rl-tv-zero-fill-same-central-gradient-edge1-v1"

    def tv_port(port: ComputePortContract) -> ComputePortContract:
        return replace(
            port,
            rounding_policy_id="rl-tv-float32-tolerance-v1",
            boundary_policy_id=boundary_policy_id,
            precision_policy_id="rl-tv-float32-v1",
        )

    return replace(
        ordinary,
        operation_id="richardson_lucy_tv_deconvolution",
        implementation_id="rl-tv-cupy-f32-v1",
        callable_ref=(
            "napari_vipp.core.gpu.cupy_rl_tv:richardson_lucy_tv_deconvolution"
        ),
        input_ports=tuple(tv_port(port) for port in ordinary.input_ports),
        output_ports=tuple(tv_port(port) for port in ordinary.output_ports),
        parameter_policy_id="rl-tv-parameters-v1",
        workload_policy_id="rl-tv-finite-f32-v1",
        parity_policy_id="rl-tv-float32-tolerance-v1",
        memory_model_id="cupyx-richardson-lucy-tv-fft-memory-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="rl-tv-float32-v1",
        limitations=(
            "finite-only",
            "float32-only-v1",
            "validated-rl-tv-profiles-v1",
            "lambda-zero-iterations-at-most-25-v1",
            "positive-tv-iterations-10-or-25-v1",
            "odd-psf-extents-v1",
            "default-safe-options-v1",
        ),
    )


def _mask_port(
    port_index: int,
    *,
    name: str,
    public_dtypes: tuple[str, ...],
    internal_dtypes: tuple[str, ...],
    conversion_policy_id: str,
    nonfinite_policy_id: str,
    overflow_policy_id: str,
    boundary_policy_id: str,
    precision_policy_id: str,
    output: bool = False,
) -> ComputePortContract:
    return ComputePortContract(
        port_index,
        ValueKind.MASK if output else ValueKind.ARRAY,
        port_name=name,
        public_dtypes=public_dtypes,
        internal_dtypes=internal_dtypes,
        accumulation_dtype=internal_dtypes[-1],
        value_domain=("binary-mask-v1" if output else "real-image-v1"),
        shape_policy_id="scalar-plane-luma-mask-v1",
        output_dtype_policy_id=("fixed:bool" if output else "dtype-same-v1"),
        conversion_policy_id=conversion_policy_id,
        nonfinite_policy_id=nonfinite_policy_id,
        rounding_policy_id="mask-bitwise-v1",
        overflow_policy_id=overflow_policy_id,
        boundary_policy_id=boundary_policy_id,
        precision_policy_id=precision_policy_id,
    )


def _canny_spec() -> OperationComputeSpec:
    boundary_policy_id = "skimage-canny-constant-zero-v1"
    input_port = _mask_port(
        0,
        name="image",
        public_dtypes=("bool", "uint8", "uint16"),
        internal_dtypes=("float32",),
        conversion_policy_id="canny-plane-float32-or-luma-v1",
        nonfinite_policy_id="finite-only-v1",
        overflow_policy_id="finite-float32-workspace-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="canny-exact-mask-v1",
    )
    output_port = _mask_port(
        0,
        name="mask",
        public_dtypes=("bool",),
        internal_dtypes=("bool",),
        conversion_policy_id="canny-plane-float32-or-luma-v1",
        nonfinite_policy_id="finite-output-v1",
        overflow_policy_id="binary-mask-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="canny-exact-mask-v1",
        output=True,
    )
    return OperationComputeSpec(
        operation_id="canny_edges",
        implementation_id="cupyx-canny-edges-exact-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref="napari_vipp.core.gpu.cupy_canny:canny_edges",
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(input_port,),
        output_ports=(output_port,),
        parameter_policy_id="canny-parameters-v1",
        workload_policy_id="canny-exact-bool-u8-u16-v2",
        parity_policy_id="mask-bitwise-v1",
        memory_model_id="cupyx-canny-exact-memory-v1",
        shape_policy_id="scalar-plane-luma-mask-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="canny-exact-mask-v1",
        progress_policy_id="scalar-plane-sync-progress-v1",
        cancellation_policy_id="scalar-plane-boundary-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2,),
        supports_device_residency=True,
        limitations=(
            "finite-only",
            "bool-uint8-uint16-public-v2",
            "float32-subnormal-intermediates-cpu-v1",
            "sigma-zero-through-twelve-v1",
            "quantile-thresholds-only-v1",
            "exact-mask-v1",
        ),
    )


def _otsu_spec() -> OperationComputeSpec:
    boundary_policy_id = "otsu-strict-greater-finite-mask-v1"
    real_dtypes = (
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    )
    input_port = _mask_port(
        0,
        name="image",
        public_dtypes=real_dtypes,
        internal_dtypes=("same", "float64"),
        conversion_policy_id="otsu-native-or-luma-v1",
        nonfinite_policy_id="otsu-finite-histogram-v1",
        overflow_policy_id="otsu-native-span-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="otsu-exact-mask-v1",
    )
    output_port = _mask_port(
        0,
        name="mask",
        public_dtypes=("bool",),
        internal_dtypes=("bool",),
        conversion_policy_id="otsu-native-or-luma-v1",
        nonfinite_policy_id="finite-output-v1",
        overflow_policy_id="binary-mask-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="otsu-exact-mask-v1",
        output=True,
    )
    return OperationComputeSpec(
        operation_id="otsu_threshold",
        implementation_id="cupy-otsu-threshold-exact-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref="napari_vipp.core.gpu.cupy_otsu:otsu_threshold",
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(input_port,),
        output_ports=(output_port,),
        parameter_policy_id="otsu-parameters-v1",
        workload_policy_id="otsu-real-exact-v1",
        parity_policy_id="mask-bitwise-v1",
        memory_model_id="cupy-otsu-histogram-memory-v1",
        shape_policy_id="scalar-plane-luma-mask-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="otsu-exact-mask-v1",
        progress_policy_id="histogram-scope-sync-progress-v1",
        cancellation_policy_id="scalar-plane-boundary-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2,),
        supports_device_residency=True,
        limitations=(
            "real-dtypes-only-v1",
            "integer-span-at-most-65536-requires-complete-facts-for-wide-dtypes-v2",
            "float-histogram-bins-2-through-65536-v1",
            "exact-mask-v1",
        ),
    )


def _sigma_filter_spec() -> OperationComputeSpec:
    """Return the clean-room VIPP CuPy RawKernel Sigma Filter contract."""

    port_values = {
        "public_dtypes": _MICROSCOPY_DTYPES,
        "internal_dtypes": ("float32", "float64"),
        "conversion_policy_id": "sigma-float32-workspace-restore-v1",
        "nonfinite_policy_id": "sigma-finite-only-v1",
        "rounding_policy_id": "sigma-half-up-u8-u16-f32-identity-v1",
        "overflow_policy_id": "sigma-float32-square-safe-v1",
        "boundary_policy_id": "sigma-nearest-circular-footprint-v1",
        "precision_policy_id": "sigma-ordered-f32-square-f64-accum-v1",
    }
    return OperationComputeSpec(
        operation_id="sigma_filter",
        implementation_id="cupy-sigma-filter-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupy",
        callable_ref="napari_vipp.core.gpu.cupy_sigma:sigma_filter",
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
        ),
        input_ports=(_gpu_image_port(0, name="image", **port_values),),
        output_ports=(_gpu_image_port(0, name="image", output=True, **port_values),),
        parameter_policy_id="sigma-filter-parameters-v1",
        workload_policy_id="sigma-u8-u16-finite-f32-v1",
        parity_policy_id="sigma-dtype-parity-v1",
        memory_model_id="cupy-sigma-filter-memory-v1",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id="sigma-nearest-circular-footprint-v1",
        precision_policy_id="sigma-ordered-f32-square-f64-accum-v1",
        progress_policy_id="sigma-row-tile-sync-progress-v1",
        cancellation_policy_id="sigma-row-tile-boundary-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2,),
        supports_device_residency=True,
        limitations=(
            "finite-only-v1",
            "uint8-uint16-float32-only-v1",
            "native-endian-only-v1",
            "float32-requires-complete-magnitude-facts-v1",
            "radius-half-through-ten-v1",
            "no-roi-mask-input-v1",
        ),
    )


def _connected_components_spec() -> OperationComputeSpec:
    """Return the exact CuPyX connected-components contract."""

    boundary_policy_id = "scipy-binary-connectivity-v1"
    precision_policy_id = "connected-components-exact-label-order-v1"
    input_port = ComputePortContract(
        0,
        ValueKind.MASK,
        port_name="mask",
        public_dtypes=("bool",),
        internal_dtypes=("bool",),
        accumulation_dtype="bool",
        value_domain="binary-mask-v1",
        shape_policy_id="shape-preserving-v1",
        output_dtype_policy_id="dtype-same-v1",
        conversion_policy_id="identity-v1",
        nonfinite_policy_id="finite-only-v1",
        rounding_policy_id="mask-bitwise-v1",
        overflow_policy_id="binary-mask-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id=precision_policy_id,
    )
    output_port = ComputePortContract(
        0,
        ValueKind.LABELS,
        port_name="labels",
        public_dtypes=("int32",),
        internal_dtypes=("int32",),
        accumulation_dtype="int32",
        value_domain="nonnegative-labels-v1",
        shape_policy_id="shape-preserving-v1",
        output_dtype_policy_id="fixed:int32",
        conversion_policy_id="binary-mask-to-int32-labels-v1",
        nonfinite_policy_id="finite-output-v1",
        rounding_policy_id="labels-bitwise-int32-v1",
        overflow_policy_id="connected-components-int32-safe-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id=precision_policy_id,
    )
    return OperationComputeSpec(
        operation_id="label_connected_components",
        implementation_id="cupyx-connected-components-v1",
        implementation_version="1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref=(
            "napari_vipp.core.gpu.cupy_connected_components:"
            "label_connected_components"
        ),
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(input_port,),
        output_ports=(output_port,),
        parameter_policy_id="connected-components-parameters-v1",
        workload_policy_id="connected-components-bool-2d-3d-v1",
        parity_policy_id="labels-bitwise-int32-v1",
        memory_model_id="cupyx-connected-components-memory-v1",
        shape_policy_id="shape-preserving-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id=precision_policy_id,
        progress_policy_id="spatial-block-sync-progress-v1",
        cancellation_policy_id="spatial-block-boundary-cancel-v1",
        side_effect_policy_id="pure-v1",
        supported_spatial_ndims=(2, 3),
        supports_device_residency=True,
        limitations=(
            "bool-mask-public-v1",
            "exact-scipy-label-ids-v1",
            "int32-spatial-block-under-2pow31-minus-2-v1",
            "leading-block-label-ids-restart-v1",
        ),
    )


_BUILTIN_ACCELERATOR_SPECS: tuple[OperationComputeSpec, ...] = (
    _background_spec("rolling_ball_background"),
    _background_spec("subtract_background"),
    _median_spec(),
    _gaussian_spec(three_dimensional=False),
    _gaussian_spec(three_dimensional=True),
    _richardson_lucy_spec(),
    _richardson_lucy_tv_spec(),
    _canny_spec(),
    _otsu_spec(),
    _sigma_filter_spec(),
    _connected_components_spec(),
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
