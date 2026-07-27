"""Provider-free scientific support, memory, and performance policy.

Executable admission remains authoritative here.  The packaged Phase 1 policy
artifact mirrors the reviewed regions and thresholds for auditability without
becoming a second execution engine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from napari_vipp.core.compute import (
    ComputeEnvironment,
    DecisionReason,
    MemoryEstimate,
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
    fallback_allowed: bool = True

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
        if not isinstance(self.fallback_allowed, bool):
            raise TypeError("fallback_allowed must be a boolean.")
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
    if workload.operation_id != spec.operation_id:
        return SupportDecision(
            False,
            DecisionReason.WORKLOAD_UNSUPPORTED,
            "The workload operation does not match this implementation.",
            fallback_allowed=False,
        )
    if len(workload.input_dtypes) != len(spec.input_ports):
        return SupportDecision(
            False,
            DecisionReason.WORKLOAD_UNSUPPORTED,
            "The resolved input-port count does not match the declaration.",
            fallback_allowed=False,
        )
    facts_decision = _validate_supplied_facts(workload, array_facts)
    if facts_decision is not None:
        return facts_decision
    operation_decision = _evaluate_operation_region(
        spec,
        workload,
        array_facts=array_facts,
    )
    if operation_decision is not None:
        return operation_decision
    for dtype, port in zip(workload.input_dtypes, spec.input_ports, strict=True):
        normalized_dtype = _dtype_name(dtype)
        public_dtypes = tuple(_dtype_name(item) for item in port.public_dtypes)
        if "*" not in public_dtypes and normalized_dtype not in public_dtypes:
            return SupportDecision(
                False,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                f"Input dtype {normalized_dtype!r} is outside the declared "
                "port region.",
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


def estimate_candidate_memory(
    spec: OperationComputeSpec,
    workload: WorkloadDescriptor,
) -> MemoryEstimate:
    """Return a conservative, versioned incremental peak for one candidate.

    These models intentionally include public input/output storage and temporary
    array upper bounds.  Runtime context and library-handle allocations remain
    represented by ``uncertainty_bytes`` until production measurements replace
    the initial conservative floor.
    """

    if workload.operation_id != spec.operation_id:
        raise ValueError("workload operation does not match the implementation.")
    if len(workload.input_shapes) != len(workload.input_dtypes):
        raise ValueError("workload shapes and dtypes must have equal length.")
    input_bytes = sum(
        math.prod(shape) * _dtype_itemsize(dtype)
        for shape, dtype in zip(
            workload.input_shapes,
            workload.input_dtypes,
            strict=True,
        )
    )
    primary_elements = (
        math.prod(workload.input_shapes[0]) if workload.input_shapes else 0
    )
    primary_itemsize = (
        _dtype_itemsize(workload.input_dtypes[0]) if workload.input_dtypes else 0
    )
    output_bytes = primary_elements * primary_itemsize * len(spec.output_ports)

    if spec.memory_model_id == "host-reference-v1":
        return MemoryEstimate(
            host_materialization_peak_bytes=output_bytes,
            model_id=spec.memory_model_id,
        )
    if spec.memory_model_id == "cucim-background-memory-v1":
        workspace_itemsize = max(primary_itemsize, 4)
        parameters = dict(workload.parameters)
        requested_radius = _finite_number(parameters.get("radius", 50.0))
        radius = max(1, int(math.ceil(requested_radius or 1.0)))
        spatial_ndim = _background_spatial_ndim(workload, parameters)
        footprint_elements = (2 * radius + 1) ** spatial_ndim
        image_workspace = primary_elements * workspace_itemsize * 8
        footprint_workspace = footprint_elements * workspace_itemsize * 3
        workspace = image_workspace + footprint_workspace
    elif spec.memory_model_id == "cupyx-median-memory-v1":
        # CuPyX median may allocate an output, rank-sized footprint metadata,
        # and an implementation workspace.  The 4x upper bound is calibrated
        # conservatively until the reviewed real-device matrix replaces it.
        requested_size = _finite_number(dict(workload.parameters).get("size", 5))
        canonical_size = max(1, int(round(requested_size or 1.0)))
        if canonical_size % 2 == 0:
            canonical_size += 1
        footprint_bytes = canonical_size**2 * max(primary_itemsize, 4) * 2
        workspace = max(input_bytes * 4, primary_elements * 4) + footprint_bytes
    elif spec.memory_model_id == "cupyx-gaussian-2d-memory-v1":
        sigma = _finite_number(dict(workload.parameters).get("sigma", 1.2)) or 0.0
        kernel_bytes = (2 * math.ceil(4 * sigma) + 1) * 8 * 2
        workspace = primary_elements * max(primary_itemsize, 4) * 4 + kernel_bytes
    elif spec.memory_model_id == "cupyx-gaussian-3d-memory-v1":
        parameters = dict(workload.parameters)
        sigmas = tuple(
            _finite_number(parameters.get(name, 2.0)) or 0.0
            for name in ("sigma_z", "sigma_y", "sigma_x")
        )
        kernel_bytes = sum(2 * math.ceil(4 * sigma) + 1 for sigma in sigmas) * 8
        workspace = primary_elements * max(primary_itemsize, 4) * 5 + kernel_bytes
    else:
        raise ValueError(
            f"No executable memory model is registered for {spec.memory_model_id!r}."
        )
    runtime_peak = input_bytes + output_bytes + workspace
    uncertainty = max(8 * 1024**2, runtime_peak // 4)
    return MemoryEstimate(
        runtime_managed_peak_bytes=runtime_peak,
        total_device_peak_bytes=runtime_peak,
        host_materialization_peak_bytes=output_bytes,
        uncertainty_bytes=uncertainty,
        model_id=spec.memory_model_id,
    )


def evaluate_memory_support(
    estimate: MemoryEstimate,
    *,
    memory_cap_bytes: int | None,
    total_device_bytes: int = 0,
    safety_reserve_bytes: int = 0,
) -> SupportDecision:
    """Apply declared request/device limits to a candidate memory estimate."""

    for name, value in (
        ("memory_cap_bytes", memory_cap_bytes),
        ("total_device_bytes", total_device_bytes),
        ("safety_reserve_bytes", safety_reserve_bytes),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer or None.")
    available = []
    if memory_cap_bytes is not None:
        available.append(memory_cap_bytes)
    if total_device_bytes:
        available.append(max(0, total_device_bytes - safety_reserve_bytes))
    required = estimate.total_device_peak_bytes + estimate.uncertainty_bytes
    if available and required > min(available):
        return SupportDecision(
            False,
            DecisionReason.MEMORY_LIMIT,
            f"The candidate requires approximately {required} device bytes, "
            f"but only {min(available)} bytes are admitted.",
        )
    return SupportDecision(
        True,
        DecisionReason.SELECTED_IMPLEMENTATION,
        "The candidate memory estimate fits the active limits.",
    )


def _validate_supplied_facts(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    if not array_facts:
        return None
    if len(array_facts) != len(workload.input_shapes):
        return SupportDecision(
            False,
            DecisionReason.WORKLOAD_UNSUPPORTED,
            "Array facts must describe every input port.",
            requires_complete_facts=True,
            fallback_allowed=False,
        )
    for index, (facts, shape, dtype) in enumerate(
        zip(
            array_facts,
            workload.input_shapes,
            workload.input_dtypes,
            strict=True,
        )
    ):
        if facts.shape != shape or _dtype_name(facts.dtype) != _dtype_name(dtype):
            return SupportDecision(
                False,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                f"Array facts for input port {index} do not match the workload.",
                requires_complete_facts=True,
                fallback_allowed=False,
            )
    return None


def _evaluate_operation_region(
    spec: OperationComputeSpec,
    workload: WorkloadDescriptor,
    *,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    policy_id = spec.parameter_policy_id
    if policy_id == "background-parameters-v1":
        return _evaluate_background_region(workload)
    if policy_id == "median-parameters-v1":
        return _evaluate_median_region(workload, array_facts=array_facts)
    if policy_id in {"gaussian-2d-parameters-v1", "gaussian-3d-parameters-v1"}:
        return _evaluate_gaussian_region(
            workload,
            three_dimensional=policy_id == "gaussian-3d-parameters-v1",
        )
    return None


def _evaluate_background_region(
    workload: WorkloadDescriptor,
) -> SupportDecision | None:
    dtype = _dtype_name(workload.input_dtypes[0])
    if dtype not in {"uint8", "uint16", "float32"}:
        return _workload_rejection(
            f"Background GPU execution has no promoted {dtype!r} region; "
            "CPU is required."
        )
    parameters = dict(workload.parameters)
    radius = _finite_number(parameters.get("radius", 50.0))
    if radius is None:
        return _workload_rejection(
            "Background radius must be finite.",
            fallback_allowed=False,
        )
    for name in ("light_background", "disable_smoothing"):
        if name in parameters and not isinstance(parameters[name], bool):
            return _workload_rejection(
                f"Background parameter {name!r} must be boolean.",
                fallback_allowed=False,
            )
    if workload.operation_id == "subtract_background":
        if "clip_negative" in parameters and not isinstance(
            parameters["clip_negative"], bool
        ):
            return _workload_rejection(
                "Background parameter 'clip_negative' must be boolean.",
                fallback_allowed=False,
            )
    shape = workload.input_shapes[0]
    active_axes = _active_axes(shape, parameters.get("channel_axis"))
    if active_axes is None:
        return _workload_rejection(
            "The declared channel axis is invalid.",
            fallback_allowed=False,
        )
    spatial_mode = str(parameters.get("spatial_mode", "2D YX")).strip().casefold()
    mode_dimensions = {
        "auto from axes": None,
        "2d yx": 2,
        "2d per xy slice (advanced)": 2,
        "3d zyx": 3,
        "3d zyx volume": 3,
    }
    if spatial_mode not in mode_dimensions:
        return _workload_rejection(
            "The background spatial mode is invalid.",
            fallback_allowed=False,
        )
    spatial_ndim = mode_dimensions[spatial_mode]
    if spatial_ndim is None:
        spatial_ndim = parameters.get(
            "resolved_spatial_ndim",
            workload.resolved_spatial_ndim,
        )
    if spatial_ndim not in {2, 3}:
        return _workload_rejection(
            "Background Auto mode requires a resolved 2D or 3D spatial rank.",
            fallback_allowed=False,
        )
    if workload.resolved_spatial_ndim not in {None, spatial_ndim}:
        return _workload_rejection(
            "Background parameters disagree with the resolved spatial rank.",
            fallback_allowed=False,
        )
    if len(active_axes) < spatial_ndim:
        return _workload_rejection(
            "The input does not contain enough non-channel axes for this "
            "background mode.",
            fallback_allowed=False,
        )
    maximum_radius = 50.0 if spatial_ndim == 3 else 500.0
    if not 1.0 <= radius <= maximum_radius:
        return _workload_rejection(
            f"{spatial_ndim}D background GPU execution is initially validated "
            f"for radii in the 1..{int(maximum_radius)} range."
        )
    return None


def _evaluate_median_region(
    workload: WorkloadDescriptor,
    *,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    dtype = _dtype_name(workload.input_dtypes[0])
    if dtype not in {"uint8", "uint16", "float32"}:
        return _workload_rejection(
            f"Median GPU execution has no promoted {dtype!r} region; CPU is required."
        )
    parameters = dict(workload.parameters)
    shape = workload.input_shapes[0]
    active_axes = _active_axes(shape, parameters.get("channel_axis"))
    if active_axes is None or len(active_axes) < 2:
        return _workload_rejection(
            "Median GPU execution requires two valid non-channel XY axes.",
            fallback_allowed=False,
        )
    requested_size = _finite_number(parameters.get("size", 5))
    if requested_size is None:
        return _workload_rejection(
            "Median kernel size must be finite.",
            fallback_allowed=False,
        )
    canonical_size = max(int(round(requested_size)), 1)
    if canonical_size % 2 == 0:
        canonical_size += 1
    if canonical_size > 51:
        return _workload_rejection(
            "Median GPU execution is validated only through a 51-pixel footprint."
        )
    if any(shape[index] < canonical_size for index in active_axes[-2:]):
        return _workload_rejection(
            "The median footprint exceeds an active XY extent outside the exact matrix."
        )
    if dtype == "float32":
        if not array_facts:
            return _complete_facts_rejection(
                "Float32 median requires complete finite and signed-zero facts."
            )
        facts = array_facts[0]
        if (
            facts.completeness is not FactCompleteness.COMPLETE
            or facts.all_finite is not True
            or "no-negative-zero" not in facts.guarantees
        ):
            return _complete_facts_rejection(
                "Float32 median is admitted only when complete facts prove finite "
                "values and no negative zero."
            )
    return None


def _evaluate_gaussian_region(
    workload: WorkloadDescriptor,
    *,
    three_dimensional: bool,
) -> SupportDecision | None:
    dtype = _dtype_name(workload.input_dtypes[0])
    if dtype in {"uint8", "uint16"}:
        return _workload_rejection(
            "Integer Gaussian remains an evaluated CPU region because reviewed "
            "fixtures found content-dependent one-unit GPU disagreements."
        )
    if dtype != "float32":
        return _workload_rejection(
            f"Gaussian GPU execution has no separately proven {dtype!r} region."
        )
    parameters = dict(workload.parameters)
    shape = workload.input_shapes[0]
    active_axes = _active_axes(shape, parameters.get("channel_axis"))
    required_axes = 3 if three_dimensional else 2
    if active_axes is None or len(active_axes) < required_axes:
        return _workload_rejection(
            f"Gaussian GPU execution requires {required_axes} valid non-channel axes.",
            fallback_allowed=False,
        )
    names = ("sigma_z", "sigma_y", "sigma_x") if three_dimensional else ("sigma",)
    defaults = (2.0, 2.0, 2.0) if three_dimensional else (1.2,)
    for name, default in zip(names, defaults, strict=True):
        value = _finite_number(parameters.get(name, default))
        if value is None:
            return _workload_rejection(
                "Gaussian sigma values must be finite numbers.",
                fallback_allowed=False,
            )
        if not 0.0 <= value <= 12.0:
            return _workload_rejection(
                "Gaussian float32 execution is initially validated for finite "
                "sigma values in the public 0..12 range."
            )
    return None


def _active_axes(
    shape: tuple[int, ...],
    raw_channel_axis: object,
) -> tuple[int, ...] | None:
    axes = list(range(len(shape)))
    if raw_channel_axis is None:
        return tuple(axes)
    if isinstance(raw_channel_axis, bool) or not isinstance(raw_channel_axis, int):
        return None
    if (
        len(shape) < 3
        or raw_channel_axis < -len(shape)
        or raw_channel_axis >= len(shape)
    ):
        return None
    axes.remove(raw_channel_axis % len(shape))
    return tuple(axes)


def _background_spatial_ndim(
    workload: WorkloadDescriptor,
    parameters: Mapping[str, object],
) -> int:
    mode = str(parameters.get("spatial_mode", "2D YX")).strip().casefold()
    if mode in {"3d zyx", "3d zyx volume"}:
        return 3
    if mode in {"2d yx", "2d per xy slice (advanced)"}:
        return 2
    resolved = parameters.get(
        "resolved_spatial_ndim",
        workload.resolved_spatial_ndim,
    )
    return int(resolved) if resolved in {2, 3} else 2


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


def _dtype_itemsize(value: object) -> int:
    try:
        return int(np.dtype(value).itemsize)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported workload dtype {value!r}.") from exc


def _workload_rejection(
    reason_text: str,
    *,
    fallback_allowed: bool = True,
) -> SupportDecision:
    return SupportDecision(
        False,
        DecisionReason.WORKLOAD_UNSUPPORTED,
        reason_text,
        fallback_allowed=fallback_allowed,
    )


def _complete_facts_rejection(reason_text: str) -> SupportDecision:
    return SupportDecision(
        False,
        DecisionReason.WORKLOAD_UNSUPPORTED,
        reason_text,
        requires_complete_facts=True,
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
        PolicyKind.ENVIRONMENT: {
            "vipp-cpu-supported-v1",
            *CUDA_ENVIRONMENT_POLICIES,
        },
        PolicyKind.PARAMETER: {
            "cpu-reference-parameters-v1",
            "background-parameters-v1",
            "median-parameters-v1",
            "gaussian-2d-parameters-v1",
            "gaussian-3d-parameters-v1",
        },
        PolicyKind.WORKLOAD: {
            "cpu-reference-v1",
            "vipp-best-available-v1",
            "background-u8-u16-f32-v1",
            "median-exact-u8-u16-f32-v1",
            "gaussian-finite-f32-v1",
        },
        PolicyKind.PARITY: {
            "authoritative-cpu-v1",
            "background-production-exact-v1",
            "median-production-bitwise-v1",
            "gaussian-float32-tolerance-v1",
        },
        PolicyKind.MEMORY: {
            "host-reference-v1",
            "cucim-background-memory-v1",
            "cupyx-median-memory-v1",
            "cupyx-gaussian-2d-memory-v1",
            "cupyx-gaussian-3d-memory-v1",
        },
        PolicyKind.SHAPE: {
            "cpu-reference-v1",
            "cpu-dynamic-output-v1",
            "shape-unknown-v1",
            "shape-preserving-v1",
        },
        PolicyKind.OUTPUT_DTYPE: {
            "dtype-same-v1",
            "cpu-dynamic-output-v1",
        },
        PolicyKind.CONVERSION: {
            "identity-v1",
            "background-float-workspace-restore-v1",
            "cupyx-median-identity-v1",
            "cupyx-gaussian-float32-v1",
        },
        PolicyKind.NONFINITE: {
            "cpu-reference-v1",
            "background-cpu-parity-v1",
            "finite-no-negative-zero-v1",
            "finite-only-v1",
        },
        PolicyKind.ROUNDING: {
            "cpu-reference-v1",
            "background-bankers-round-clip-v1",
            "median-bitwise-v1",
            "gaussian-float32-tolerance-v1",
        },
        PolicyKind.OVERFLOW: {
            "cpu-reference-v1",
            "background-clip-public-dtype-v1",
            "preserve-public-dtype-v1",
        },
        PolicyKind.BOUNDARY: {
            "cpu-reference-v1",
            "background-nearest-rolling-ball-v1",
            "scipy-reflect-v1",
        },
        PolicyKind.PRECISION: {
            "scientific-default-v1",
            "background-public-dtype-v1",
            "median-bitwise-v1",
            "gaussian-float32-v1",
        },
        PolicyKind.PROGRESS: {
            "cpu-reference-v1",
            "background-block-progress-v1",
            "monolithic-sync-progress-v1",
        },
        PolicyKind.CANCELLATION: {
            "cpu-reference-v1",
            "background-block-cancel-v1",
            "monolithic-boundary-cancel-v1",
        },
        PolicyKind.SIDE_EFFECT: {
            "pure-or-source-v1",
            "host-writer-v1",
            "pure-v1",
        },
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
    "evaluate_memory_support",
    "estimate_candidate_memory",
    "propagate_output_descriptors",
    "validate_spec_policy_references",
]
