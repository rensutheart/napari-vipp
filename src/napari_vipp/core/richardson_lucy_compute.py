"""Operation-owned compute contracts for Richardson--Lucy variants.

The shared compute registries delegate RL/RL-TV declaration, admission, and
memory behavior to this provider-free module.  Consequently, adding an
unrelated operation no longer changes the scientific source boundary used by
the committed RL evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from napari_vipp.core.compute import MemoryEstimate, WorkloadDescriptor
from napari_vipp.core.compute_contracts import (
    AdmissionTier,
    ComputePortContract,
    OperationComputeSpec,
    ValueKind,
)

RICHARDSON_LUCY_MINIMUM_FILTER_EPSILON = 1e-12
RICHARDSON_LUCY_MAXIMUM_FILTER_EPSILON = 1e-6
RICHARDSON_LUCY_FILTER_EPSILON = RICHARDSON_LUCY_MINIMUM_FILTER_EPSILON
RICHARDSON_LUCY_MAXIMUM_ITERATIONS = 500
RICHARDSON_LUCY_TV_FILTER_EPSILON = 1e-12
RICHARDSON_LUCY_TV_MAXIMUM_ITERATIONS = 100
RICHARDSON_LUCY_TV_POSITIVE_ITERATIONS = frozenset({10, 25})
RICHARDSON_LUCY_TV_REGULARIZATION = 0.002
RICHARDSON_LUCY_TV_EPSILON = 1e-6
RICHARDSON_LUCY_TV_DENOMINATOR_FLOOR = 0.05

RICHARDSON_LUCY_MEMORY_MODEL_IDS = frozenset(
    {
        "cupyx-richardson-lucy-fft-memory-v2",
        "cupyx-richardson-lucy-tv-fft-memory-v1",
    }
)
RICHARDSON_LUCY_POLICY_IDS = MappingProxyType(
    {
        "parameter": frozenset({"rl-parameters-v3", "rl-tv-parameters-v3"}),
        "workload": frozenset({"rl-finite-f32-v3", "rl-tv-finite-f32-v3"}),
        "parity": frozenset(
            {
                "rl-scientific-equivalence-v2",
                "rl-tv-scientific-equivalence-v2",
            }
        ),
        "memory": RICHARDSON_LUCY_MEMORY_MODEL_IDS,
        "conversion": frozenset({"cupyx-rl-float32-identity-v1"}),
        "rounding": frozenset(
            {
                "rl-scientific-equivalence-v2",
                "rl-tv-scientific-equivalence-v2",
            }
        ),
        "overflow": frozenset({"finite-float32-cleanup-v1"}),
        "boundary": frozenset(
            {
                "scipy-signal-zero-fill-same-v1",
                "rl-tv-zero-fill-same-central-gradient-edge1-v1",
            }
        ),
        "precision": frozenset({"rl-float32-v1", "rl-tv-float32-v1"}),
        "progress": frozenset({"deconvolution-block-iteration-progress-v1"}),
        "cancellation": frozenset({"deconvolution-iteration-cancel-v1"}),
    }
)

_FLOAT32_BYTES = 4
_COMPLEX64_BYTES = 8
_LIVE_BLOCK_BUFFERS = 6
_TV_EXTRA_LIVE_BLOCK_BUFFERS_PER_AXIS = 3
_TV_EXTRA_LIVE_BLOCK_BUFFERS = 4
_FFT_PLAN_WORKSPACE_MULTIPLIER = 4
_FIRST_USE_ALLOWANCE_BYTES = 32 * 1024**2


@dataclass(frozen=True, slots=True)
class RegionRejection:
    """Operation-owned rejection details adapted by the shared policy layer."""

    reason_text: str
    fallback_allowed: bool = True
    exact_workload_test_allowed: bool = False


def richardson_lucy_compute_specs() -> tuple[OperationComputeSpec, ...]:
    """Return the exact public CuPy declarations for ordinary RL and RL-TV."""

    ordinary = _richardson_lucy_spec()
    return ordinary, _richardson_lucy_tv_spec(ordinary)


def _richardson_lucy_spec() -> OperationComputeSpec:
    input_values = {
        "public_dtypes": ("float32",),
        "internal_dtypes": ("float32",),
        "conversion_policy_id": "cupyx-rl-float32-identity-v1",
        "nonfinite_policy_id": "finite-only-v1",
        "rounding_policy_id": "rl-scientific-equivalence-v2",
        "overflow_policy_id": "finite-float32-cleanup-v1",
        "boundary_policy_id": "scipy-signal-zero-fill-same-v1",
        "precision_policy_id": "rl-float32-v1",
    }
    image_input = _image_port(0, name="image", **input_values)
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
        rounding_policy_id="rl-scientific-equivalence-v2",
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
        rounding_policy_id="rl-scientific-equivalence-v2",
        overflow_policy_id="finite-float32-cleanup-v1",
        boundary_policy_id="scipy-signal-zero-fill-same-v1",
        precision_policy_id="rl-float32-v1",
    )
    return OperationComputeSpec(
        operation_id="richardson_lucy_deconvolution",
        implementation_id="rl-cupy-f32-v1",
        implementation_version="2",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        callable_ref="napari_vipp.core.gpu.cupy_rl:richardson_lucy_deconvolution",
        host_boundary=False,
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id=(
            "cuda-cupy-14.1.1-cpython312-windows-native-v3"
        ),
        input_ports=(image_input, psf_input),
        output_ports=(image_output,),
        parameter_policy_id="rl-parameters-v3",
        workload_policy_id="rl-finite-f32-v3",
        parity_policy_id="rl-scientific-equivalence-v2",
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
            "ui-valid-rl-parameters-v3",
            "cpu-gpu-parity-advisory-v3",
        ),
    )


def _richardson_lucy_tv_spec(
    ordinary: OperationComputeSpec,
) -> OperationComputeSpec:
    boundary_policy_id = "rl-tv-zero-fill-same-central-gradient-edge1-v1"

    def tv_port(port: ComputePortContract) -> ComputePortContract:
        return replace(
            port,
            rounding_policy_id="rl-tv-scientific-equivalence-v2",
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
        parameter_policy_id="rl-tv-parameters-v3",
        workload_policy_id="rl-tv-finite-f32-v3",
        parity_policy_id="rl-tv-scientific-equivalence-v2",
        memory_model_id="cupyx-richardson-lucy-tv-fft-memory-v1",
        boundary_policy_id=boundary_policy_id,
        precision_policy_id="rl-tv-float32-v1",
        limitations=(
            "finite-only",
            "float32-only-v1",
            "ui-valid-rl-tv-parameters-v3",
            "positive-tv-gradient-axis-at-least-two-v1",
            "cpu-gpu-parity-advisory-v3",
        ),
    )


def _image_port(
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
) -> ComputePortContract:
    return ComputePortContract(
        port_index,
        ValueKind.ARRAY,
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


def evaluate_richardson_lucy_region(
    workload: WorkloadDescriptor,
    array_facts: tuple[object, ...],
) -> RegionRejection | None:
    return _deconvolution_region_policy(
        workload,
        array_facts,
        operation_name="Richardson-Lucy",
        maximum_iterations=RICHARDSON_LUCY_MAXIMUM_ITERATIONS,
        maximum_filter_epsilon=1.0,
    )


def evaluate_richardson_lucy_tv_region(
    workload: WorkloadDescriptor,
    array_facts: tuple[object, ...],
) -> RegionRejection | None:
    parameters = dict(workload.parameters)
    regularization = _finite_number(
        parameters.get("tv_regularization", RICHARDSON_LUCY_TV_REGULARIZATION)
    )
    if regularization is None or not 0.0 <= regularization <= 0.1:
        return _reject(
            "Richardson-Lucy TV regularization must be finite and between 0 and 0.1.",
            fallback_allowed=False,
        )
    lambda_zero = regularization == 0.0
    common = _deconvolution_region_policy(
        workload,
        array_facts,
        operation_name="Richardson-Lucy TV",
        maximum_iterations=RICHARDSON_LUCY_TV_MAXIMUM_ITERATIONS,
        maximum_filter_epsilon=1e-3,
    )
    if common is not None:
        return common

    for parameter_name, default, minimum, maximum, display_name in (
        ("tv_epsilon", RICHARDSON_LUCY_TV_EPSILON, 1e-12, 1e-2, "TV epsilon"),
        (
            "denominator_floor",
            RICHARDSON_LUCY_TV_DENOMINATOR_FLOOR,
            1e-6,
            1.0,
            "denominator floor",
        ),
    ):
        value = _finite_number(parameters.get(parameter_name, default))
        if value is None or not minimum <= value <= maximum:
            return _reject(
                f"Richardson-Lucy TV {display_name} must be finite and between "
                f"{minimum:g} and {maximum:g}.",
                fallback_allowed=False,
            )
    if lambda_zero:
        return None

    image_shape = workload.input_shapes[0]
    spatial_ndim = workload.resolved_spatial_ndim
    if spatial_ndim in {2, 3} and any(
        extent < 2 for extent in image_shape[-spatial_ndim:]
    ):
        return _reject(
            "Richardson-Lucy TV requires at least two samples along every active "
            "spatial axis for its central-gradient stencil.",
            fallback_allowed=False,
        )
    return None


def _deconvolution_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[object, ...],
    *,
    operation_name: str,
    maximum_iterations: int,
    maximum_filter_epsilon: float,
) -> RegionRejection | None:
    if len(workload.input_shapes) != 2 or len(workload.input_dtypes) != 2:
        return _reject(
            f"{operation_name} GPU execution requires ordered Image and PSF inputs.",
            fallback_allowed=False,
        )
    if any(_dtype_name(dtype) != "float32" for dtype in workload.input_dtypes):
        return _reject(
            f"{operation_name} GPU execution initially requires explicit float32 "
            "Image and PSF inputs; add Convert Dtype when appropriate."
        )
    spatial_ndim = workload.resolved_spatial_ndim
    if spatial_ndim not in {2, 3}:
        return _reject(
            f"{operation_name} Auto mode requires a resolved 2D or 3D spatial rank.",
            fallback_allowed=False,
        )
    image_shape, psf_shape = workload.input_shapes
    if len(image_shape) < spatial_ndim or len(psf_shape) != spatial_ndim:
        return _reject(
            "The PSF rank must match the resolved Richardson-Lucy spatial rank.",
            fallback_allowed=False,
        )
    if any(size <= 0 for size in (*image_shape, *psf_shape)):
        return _reject(
            f"{operation_name} inputs must not contain empty dimensions.",
            fallback_allowed=False,
        )
    parameters = dict(workload.parameters)
    mode = str(parameters.get("spatial_mode", "Auto from axes")).strip().casefold()
    declared_rank = {
        "auto from axes": spatial_ndim,
        "2d yx": 2,
        "2d per xy slice (advanced)": 2,
        "3d zyx": 3,
        "3d zyx volume": 3,
    }.get(mode)
    if declared_rank is None or declared_rank != spatial_ndim:
        return _reject(
            f"{operation_name} spatial parameters disagree with the resolved rank.",
            fallback_allowed=False,
        )
    iterations = parameters.get("iterations", 25)
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        return _reject(
            f"{operation_name} iterations must be an integer.",
            fallback_allowed=False,
        )
    if not 1 <= iterations <= maximum_iterations:
        return _reject(
            f"{operation_name} iterations must be between 1 and {maximum_iterations}.",
            fallback_allowed=False,
        )
    safety_flags = (
        "normalize_psf",
        "clip_negative_input",
        "clip_output_negative",
        "preserve_input_scale",
    )
    for name in safety_flags:
        if name in parameters and not isinstance(parameters[name], bool):
            return _reject(
                f"{operation_name} parameter {name!r} must be boolean.",
                fallback_allowed=False,
            )
    filter_epsilon = _finite_number(parameters.get("filter_epsilon", 1e-12))
    if filter_epsilon is None or not 0.0 <= filter_epsilon <= maximum_filter_epsilon:
        return _reject(
            f"{operation_name} filter epsilon must be finite and between 0 and "
            f"{maximum_filter_epsilon:g}.",
            fallback_allowed=False,
        )
    if len(array_facts) == 2:
        psf_facts = array_facts[1]
        completeness = getattr(psf_facts, "completeness", None)
        completeness_value = getattr(completeness, "value", completeness)
        maximum = getattr(psf_facts, "maximum", None)
        if (
            completeness_value == "complete"
            and maximum is not None
            and (float(maximum) <= 1e-12)
        ):
            return _reject(
                "The finite PSF has no positive mass above the validation floor.",
                fallback_allowed=False,
            )
    return None


def richardson_lucy_parity_warnings(
    workload: WorkloadDescriptor,
) -> tuple[str, ...]:
    """Describe numerical uncertainty without rejecting executable workloads.

    The former v2 prequalification envelope determines when to show an
    advisory. This does not claim CPU equivalence for the broader v3 region.
    """
    parameters = dict(workload.parameters)
    warnings: list[str] = []
    image_shape, psf_shape = workload.input_shapes
    if any(extent % 2 == 0 for extent in psf_shape):
        warnings.append(
            "Even-sized PSFs can give different convolution centering on CPU "
            "and GPU. GPU execution remains enabled."
        )
    if any(
        kernel > image
        for kernel, image in zip(psf_shape, image_shape[-len(psf_shape) :], strict=True)
    ):
        warnings.append(
            "The PSF is larger than the image on at least one axis. CPU/GPU "
            "agreement is not prequalified for this geometry."
        )
    default_options = all(
        parameters.get(option, True) is True
        for option in (
            "normalize_psf",
            "clip_negative_input",
            "clip_output_negative",
            "preserve_input_scale",
        )
    )
    iterations = parameters.get("iterations", 25)
    epsilon = _finite_number(parameters.get("filter_epsilon", 1e-12))
    reviewed = (
        iterations <= 100
        and epsilon is not None
        and RICHARDSON_LUCY_MINIMUM_FILTER_EPSILON
        <= epsilon
        <= RICHARDSON_LUCY_MAXIMUM_FILTER_EPSILON
        and default_options
    )
    if workload.operation_id == "richardson_lucy_tv_deconvolution":
        regularization = _finite_number(
            parameters.get("tv_regularization", RICHARDSON_LUCY_TV_REGULARIZATION)
        )
        if regularization != 0:
            reviewed = (
                default_options
                and iterations in RICHARDSON_LUCY_TV_POSITIVE_ITERATIONS
                and regularization == RICHARDSON_LUCY_TV_REGULARIZATION
                and epsilon == RICHARDSON_LUCY_TV_FILTER_EPSILON
                and _finite_number(parameters.get("tv_epsilon", 1e-6)) == 1e-6
                and _finite_number(parameters.get("denominator_floor", 0.05)) == 0.05
            )
    if not reviewed:
        warnings.append(
            "These settings extend beyond the prequalified CPU/GPU comparison "
            "range. GPU execution remains enabled; reconstruction values may "
            "differ. No CPU comparison is required to run this node."
        )
    return tuple(warnings)


def estimate_richardson_lucy_memory(
    spec: OperationComputeSpec,
    workload: WorkloadDescriptor,
    *,
    input_bytes: int,
    output_bytes: int,
) -> MemoryEstimate:
    """Return the versioned FFT peak model for an RL-family candidate."""

    if spec.memory_model_id not in RICHARDSON_LUCY_MEMORY_MODEL_IDS:
        raise ValueError(f"Unknown RL memory model {spec.memory_model_id!r}.")
    spatial_ndim = workload.resolved_spatial_ndim
    if spatial_ndim not in {2, 3}:
        raise ValueError(
            "Richardson-Lucy memory estimation requires a resolved 2D or 3D "
            "spatial rank."
        )
    image_shape = workload.input_shapes[0]
    block_elements = math.prod(image_shape[-spatial_ndim:])
    psf_shape = workload.input_shapes[1]
    psf_elements = math.prod(psf_shape)
    fft_shape = tuple(
        _next_235_smooth_length(image_extent + psf_extent - 1)
        for image_extent, psf_extent in zip(
            image_shape[-spatial_ndim:],
            psf_shape,
            strict=True,
        )
    )
    fft_real_elements = math.prod(fft_shape)
    fft_complex_elements = math.prod(fft_shape[:-1]) * (fft_shape[-1] // 2 + 1)
    fft_real_bytes = fft_real_elements * _FLOAT32_BYTES
    fft_complex_bytes = fft_complex_elements * _COMPLEX64_BYTES

    live_block_buffers = _LIVE_BLOCK_BUFFERS
    tv_regularization = _finite_number(
        dict(workload.parameters).get(
            "tv_regularization",
            RICHARDSON_LUCY_TV_REGULARIZATION,
        )
    )
    if (
        spec.memory_model_id == "cupyx-richardson-lucy-tv-fft-memory-v1"
        and tv_regularization != 0.0
    ):
        live_block_buffers += (
            _TV_EXTRA_LIVE_BLOCK_BUFFERS_PER_AXIS * spatial_ndim
            + _TV_EXTRA_LIVE_BLOCK_BUFFERS
        )
    logical_block_workspace = block_elements * _FLOAT32_BYTES * live_block_buffers
    fft_array_workspace = fft_real_bytes + 3 * fft_complex_bytes
    fft_plan_workspace = _FFT_PLAN_WORKSPACE_MULTIPLIER * (
        fft_real_bytes + fft_complex_bytes
    )
    psf_workspace = psf_elements * _FLOAT32_BYTES * 4
    workspace = (
        logical_block_workspace
        + fft_array_workspace
        + fft_plan_workspace
        + psf_workspace
    )
    runtime_peak = input_bytes + output_bytes + workspace
    uncertainty = max(_FIRST_USE_ALLOWANCE_BYTES, runtime_peak // 4)
    return MemoryEstimate(
        runtime_managed_peak_bytes=runtime_peak,
        total_device_peak_bytes=runtime_peak,
        host_materialization_peak_bytes=output_bytes,
        uncertainty_bytes=uncertainty,
        model_id=spec.memory_model_id,
    )


def canonical_richardson_lucy_spec_digest(spec: OperationComputeSpec) -> str:
    """Fingerprint the stable RL declaration, ignoring empty generic extensions."""

    payload = asdict(spec)
    for name in tuple(payload):
        value = payload[name]
        if name not in _STABLE_SPEC_FIELDS and (
            value is None
            or value is False
            or (isinstance(value, str) and not value)
            or (isinstance(value, tuple) and not value)
        ):
            payload.pop(name)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


_STABLE_SPEC_FIELDS = frozenset(
    {
        "operation_id",
        "implementation_id",
        "implementation_version",
        "runtime_id",
        "array_domain",
        "implementation_library_id",
        "callable_ref",
        "host_boundary",
        "admission_tier",
        "validated_environment_policy_id",
        "input_ports",
        "output_ports",
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
        "supported_spatial_ndims",
        "supports_device_residency",
        "limitations",
        "cache_equivalence_group",
    }
)


def _next_235_smooth_length(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("FFT extents must be positive integers.")
    best = 1 << (value - 1).bit_length()
    power_two = 1
    while power_two <= best:
        power_three = power_two
        while power_three <= best:
            candidate = power_three
            while candidate < value:
                candidate *= 5
            if candidate < best:
                best = candidate
            power_three *= 3
        power_two *= 2
    return best


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _dtype_name(value: object) -> str:
    if str(value).strip() == "*":
        return "*"
    try:
        return np.dtype(value).name
    except (TypeError, ValueError):
        return str(value).strip()


def _reject(
    reason_text: str,
    *,
    fallback_allowed: bool = True,
    exact_workload_test_allowed: bool = False,
) -> RegionRejection:
    return RegionRejection(
        reason_text,
        fallback_allowed=fallback_allowed,
        exact_workload_test_allowed=exact_workload_test_allowed,
    )


__all__ = [
    "RICHARDSON_LUCY_FILTER_EPSILON",
    "RICHARDSON_LUCY_MAXIMUM_FILTER_EPSILON",
    "RICHARDSON_LUCY_MAXIMUM_ITERATIONS",
    "RICHARDSON_LUCY_MINIMUM_FILTER_EPSILON",
    "RICHARDSON_LUCY_MEMORY_MODEL_IDS",
    "RICHARDSON_LUCY_POLICY_IDS",
    "RICHARDSON_LUCY_TV_DENOMINATOR_FLOOR",
    "RICHARDSON_LUCY_TV_EPSILON",
    "RICHARDSON_LUCY_TV_FILTER_EPSILON",
    "RICHARDSON_LUCY_TV_MAXIMUM_ITERATIONS",
    "RICHARDSON_LUCY_TV_POSITIVE_ITERATIONS",
    "RICHARDSON_LUCY_TV_REGULARIZATION",
    "RegionRejection",
    "canonical_richardson_lucy_spec_digest",
    "estimate_richardson_lucy_memory",
    "evaluate_richardson_lucy_region",
    "evaluate_richardson_lucy_tv_region",
    "richardson_lucy_compute_specs",
    "richardson_lucy_parity_warnings",
]
