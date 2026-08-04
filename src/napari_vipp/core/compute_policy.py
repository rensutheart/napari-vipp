"""Provider-free scientific support, memory, and performance policy.

Executable admission remains authoritative here.  The packaged Phase 1 policy
artifact mirrors the reviewed regions and thresholds for auditability without
becoming a second execution engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
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
from napari_vipp.core.measurements import basic_measurement_layout
from napari_vipp.core.richardson_lucy_compute import (
    RICHARDSON_LUCY_FILTER_EPSILON,
    RICHARDSON_LUCY_MAXIMUM_ITERATIONS,
    RICHARDSON_LUCY_MEMORY_MODEL_IDS,
    RICHARDSON_LUCY_POLICY_IDS,
    RICHARDSON_LUCY_TV_DENOMINATOR_FLOOR,
    RICHARDSON_LUCY_TV_EPSILON,
    RICHARDSON_LUCY_TV_FILTER_EPSILON,
    RICHARDSON_LUCY_TV_MAXIMUM_ITERATIONS,
    RICHARDSON_LUCY_TV_POSITIVE_ITERATIONS,
    RICHARDSON_LUCY_TV_REGULARIZATION,
    RegionRejection,
    estimate_richardson_lucy_memory,
    evaluate_richardson_lucy_region,
    evaluate_richardson_lucy_tv_region,
)

OTSU_DEFAULT_HISTOGRAM_BINS = 256
OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS = 65_536
CONNECTED_COMPONENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
MEASUREMENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
SIGMA_FILTER_FLOAT32_SQUARE_LIMIT = float(
    np.float32(math.sqrt(float(np.finfo(np.float32).max)))
)


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


CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID = (
    "cuda-cupy-14.1.1-cpython312-windows-native-v3"
)
CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID = (
    "cuda-cupy-14.1.1-cucim-26.6.0-cpython312-windows-native-v3"
)
CUDA_CUPY_RAWKERNEL_WINDOWS_ENVIRONMENT_POLICY_ID = (
    "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
)
CUDA_ENVIRONMENT_POLICIES = {
    CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID,
    CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID,
    CUDA_CUPY_RAWKERNEL_WINDOWS_ENVIRONMENT_POLICY_ID,
}

_PHASE1_CUDA_POLICY_PROVIDER = {
    CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID: ("cuda-cupy", "cupyx"),
    CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID: ("cuda-cupy", "cucim"),
    CUDA_CUPY_RAWKERNEL_WINDOWS_ENVIRONMENT_POLICY_ID: ("cuda-cupy", "cupy"),
}

_PHASE1_CUPY_VERSION = "14.1.1"
_PHASE1_CUDA_RUNTIME_VERSION = "13020"
_PHASE1_CUDA_DRIVER_VERSION = "13030"
_PHASE1_CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 5090"
_PHASE1_CUDA_COMPUTE_CAPABILITY = "12.0"
_PHASE1_CPU_SCIENTIFIC_STACK = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-image": "0.26.0",
}
_PHASE1_CUCIM_VERSIONS = frozenset({"26.6.0", "26.06.00"})
_PHASE1_CUCIM_ARTIFACT_SHA256 = (
    "586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8"
)


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
    environment_decision = _evaluate_phase1_cuda_environment(spec, environment)
    if environment_decision is not None:
        return environment_decision
    return evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=array_facts,
    )


def evaluate_candidate_workload_support(
    spec: OperationComputeSpec,
    workload: WorkloadDescriptor,
    *,
    array_facts: tuple[ArrayFacts, ...] = (),
) -> SupportDecision:
    """Evaluate provider-free scientific/workload gates before any probe.

    Visibility and environment admission intentionally remain the caller's
    responsibility.  This split lets execution discard statically unsupported
    candidates without importing or probing an optional GPU provider.
    """

    if not isinstance(spec, OperationComputeSpec):
        raise TypeError("spec must be an OperationComputeSpec.")
    if not isinstance(workload, WorkloadDescriptor):
        raise TypeError("workload must be a WorkloadDescriptor.")
    facts = tuple(array_facts)
    if any(not isinstance(item, ArrayFacts) for item in facts):
        raise TypeError("array_facts must contain ArrayFacts values.")
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
    facts_decision = _validate_supplied_facts(workload, facts)
    if facts_decision is not None:
        return facts_decision
    operation_decision = _evaluate_operation_region(
        spec,
        workload,
        array_facts=facts,
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
    finite_fact_indices = tuple(
        index
        for index, (dtype, port) in enumerate(
            zip(workload.input_dtypes, spec.input_ports, strict=True)
        )
        if _dtype_can_contain_nonfinite(dtype)
        and (
            "finite-only" in spec.limitations
            or port.nonfinite_policy_id == "finite-only-v1"
        )
    )
    if requires_finite and finite_fact_indices:
        if len(facts) != len(spec.input_ports):
            return SupportDecision(
                False,
                DecisionReason.WORKLOAD_UNSUPPORTED,
                "Complete finite-value facts are required.",
                requires_complete_facts=True,
            )
        if any(
            facts[index].completeness is not FactCompleteness.COMPLETE
            or facts[index].all_finite is not True
            for index in finite_fact_indices
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


def _evaluate_phase1_cuda_host_environment(
    spec: OperationComputeSpec,
    environment: ComputeEnvironment,
) -> SupportDecision | None:
    """Reject an invalid provider/policy binding or host without probing CUDA."""

    def rejected(reason: str) -> SupportDecision:
        return SupportDecision(
            False,
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            reason,
        )

    expected_provider = _PHASE1_CUDA_POLICY_PROVIDER.get(
        spec.validated_environment_policy_id
    )
    actual_provider = (spec.runtime_id, spec.implementation_library_id)
    if expected_provider is None:
        return rejected("No executable environment policy is registered.")
    if actual_provider != expected_provider:
        return rejected(
            f"Environment policy {spec.validated_environment_policy_id!r} is bound "
            f"to runtime/library {expected_provider!r}, not {actual_provider!r}."
        )
    if environment.os_name == "Darwin":
        return rejected(
            "Phase-1 CUDA admission is unavailable on macOS; Apple GPU provider "
            "investigation remains pending and CPU is authoritative."
        )
    if environment.os_name == "Linux":
        return rejected(
            "Native Linux Phase-1 CUDA admission is pending clean-host "
            "validation evidence and therefore fails closed."
        )
    if environment.os_name != "Windows" or environment.execution_mode != "native":
        return rejected(
            "Phase-1 GPU admission requires an exactly validated native Windows "
            "environment."
        )
    if (
        environment.python_implementation != "CPython"
        or environment.python_version != "3.12"
        or environment.python_abi != "cpython-312"
    ):
        return rejected(
            "Phase-1 GPU admission requires the exact CPython 3.12 cpython-312 ABI."
        )
    scientific_versions = dict(environment.scientific_stack_versions)
    missing_scientific_versions = tuple(
        distribution
        for distribution in _PHASE1_CPU_SCIENTIFIC_STACK
        if not scientific_versions.get(distribution)
        or scientific_versions[distribution] == "not-installed"
    )
    if missing_scientific_versions:
        missing = ", ".join(missing_scientific_versions)
        return rejected(
            "Phase-1 GPU admission requires version metadata for the validated "
            "authoritative CPU scientific stack (NumPy 2.5.1, SciPy 1.18.0, "
            "scikit-image 0.26.0); missing: "
            f"{missing}. CPU remains authoritative."
        )
    mismatched_scientific_versions = tuple(
        (distribution, scientific_versions[distribution], required)
        for distribution, required in _PHASE1_CPU_SCIENTIFIC_STACK.items()
        if scientific_versions[distribution] != required
    )
    if mismatched_scientific_versions:
        mismatch = ", ".join(
            f"{distribution} {actual} (validated {required})"
            for distribution, actual, required in mismatched_scientific_versions
        )
        return rejected(
            "Phase-1 GPU admission requires the validated authoritative CPU "
            "scientific stack (NumPy 2.5.1, SciPy 1.18.0, scikit-image 0.26.0); "
            f"found {mismatch}. CPU remains authoritative."
        )
    return None


def _evaluate_phase1_cuda_environment(
    spec: OperationComputeSpec,
    environment: ComputeEnvironment,
) -> SupportDecision | None:
    def rejected(reason: str) -> SupportDecision:
        return SupportDecision(
            False,
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            reason,
        )

    host_decision = _evaluate_phase1_cuda_host_environment(spec, environment)
    if host_decision is not None:
        return host_decision
    if environment.probe_status != "available":
        return rejected("The accelerator runtime probe did not report availability.")
    if spec.runtime_id not in environment.runtime_ids:
        return rejected(f"Runtime {spec.runtime_id!r} was not admitted by the probe.")
    if spec.implementation_library_id not in environment.implementation_libraries:
        return rejected(
            f"Implementation library {spec.implementation_library_id!r} was not "
            "admitted by its probe."
        )

    versions = dict(environment.runtime_versions)
    if versions.get("cuda-cupy") != _PHASE1_CUPY_VERSION:
        return rejected("Phase-1 CUDA admission requires exact CuPy 14.1.1 provenance.")
    if spec.implementation_library_id in {"cupy", "cupyx"} and (
        versions.get(spec.implementation_library_id) != _PHASE1_CUPY_VERSION
    ):
        library_name = "CuPy" if spec.implementation_library_id == "cupy" else "CuPyX"
        return rejected(
            f"Phase-1 {library_name} admission requires exact {library_name} "
            "14.1.1 provenance."
        )

    fingerprints = dict(environment.runtime_probe_fingerprints)
    if not fingerprints.get("cuda-cupy"):
        return rejected(
            "The CUDA runtime probe did not supply a nonempty environment fingerprint."
        )
    runtime_metadata = _metadata_for_scope(
        environment.runtime_metadata,
        "cuda-cupy",
    )
    cuda_runtime = runtime_metadata.get("cuda_runtime_version", "")
    if not cuda_runtime.isascii() or not cuda_runtime.isdecimal():
        return rejected("The CUDA runtime version metadata must be numeric.")
    if cuda_runtime != _PHASE1_CUDA_RUNTIME_VERSION:
        return rejected(
            "Public GPU admission requires the validated CUDA runtime API 13.2 "
            f"({_PHASE1_CUDA_RUNTIME_VERSION}); found {cuda_runtime}. CUDA 12 "
            "and other runtime versions remain developer qualification tracks, "
            "so CPU remains authoritative."
        )
    metadata_driver = runtime_metadata.get("driver_version", "")
    if (
        not metadata_driver.isascii()
        or not metadata_driver.isdecimal()
        or int(metadata_driver) <= 0
        or not environment.driver_version
        or environment.driver_version != metadata_driver
    ):
        return rejected(
            "The CUDA probe must preserve matching numeric driver-version metadata."
        )
    if metadata_driver != _PHASE1_CUDA_DRIVER_VERSION:
        return rejected(
            "Public GPU admission requires the validated CUDA driver API 13.3 "
            f"({_PHASE1_CUDA_DRIVER_VERSION}); found {metadata_driver}. Other "
            "driver versions require renewed evidence, so CPU remains authoritative."
        )
    if (
        not environment.device_id.startswith("cuda:")
        or environment.device_class != "nvidia-cuda"
        or not environment.device_name
    ):
        return rejected("Phase-1 admission requires a selected NVIDIA CUDA device.")
    if environment.device_name != _PHASE1_CUDA_DEVICE_NAME:
        return rejected(
            "Public GPU admission is currently validated only for "
            f"{_PHASE1_CUDA_DEVICE_NAME}; found "
            f"{environment.device_name or 'an unnamed device'}. Secondary NVIDIA "
            "hardware remains a qualification track, so CPU remains authoritative."
        )
    compute_capability = dict(environment.device_metadata).get(
        "compute_capability",
        "",
    )
    if not _valid_compute_capability(compute_capability):
        return rejected(
            "The selected NVIDIA device is missing numeric compute-capability metadata."
        )
    if compute_capability != _PHASE1_CUDA_COMPUTE_CAPABILITY:
        return rejected(
            "Public GPU admission requires the validated NVIDIA compute capability "
            f"{_PHASE1_CUDA_COMPUTE_CAPABILITY}; found {compute_capability}. "
            "Secondary hardware requires its own evidence, so CPU remains "
            "authoritative."
        )

    if spec.validated_environment_policy_id == (
        CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID
    ):
        cucim_version = versions.get("cucim", "")
        if cucim_version not in _PHASE1_CUCIM_VERSIONS:
            return rejected(
                "Phase-1 cuCIM admission requires exact cuCIM 26.6.0/26.06.00 "
                "provenance."
            )
        library_metadata = _metadata_for_scope(
            environment.implementation_library_metadata,
            "cucim",
        )
        expected_metadata = {
            "environment_record_schema": "napari-vipp-gpu-environment",
            "environment_record_schema_version": "1",
            "environment_track": "cuda13",
            "cupy_distribution": "cupy-cuda13x",
            "cucim_distribution": "cucim-cu13",
            "cucim_artifact_sha256": _PHASE1_CUCIM_ARTIFACT_SHA256,
        }
        for key, expected in expected_metadata.items():
            if library_metadata.get(key) != expected:
                return rejected(
                    "The cuCIM environment record is missing or has unapproved "
                    f"{key!r} provenance."
                )
        if library_metadata.get("cucim_distribution_version") not in (
            _PHASE1_CUCIM_VERSIONS
        ):
            return rejected(
                "The installed cuCIM distribution version is outside the exact "
                "Phase-1 matrix."
            )
        if int(cuda_runtime) // 1000 != 13:
            return rejected(
                "The approved Phase-1 cuCIM artifact is specific to the CUDA 13 "
                "environment track."
            )
    return None


def _metadata_for_scope(
    values: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    scope: str,
) -> dict[str, str]:
    return dict(dict(values).get(scope, ()))


def _valid_compute_capability(value: str) -> bool:
    major, separator, minor = value.partition(".")
    return bool(
        separator
        and major.isascii()
        and major.isdecimal()
        and minor.isascii()
        and minor.isdecimal()
        and int(major) > 0
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
    output_bytes = sum(
        primary_elements
        * _output_port_itemsize(
            port.output_dtype_policy_id,
            primary_itemsize,
        )
        for port in spec.output_ports
    )
    if spec.memory_model_id == "cucim-basic-measurements-memory-v1":
        measurement_layout = _basic_measurement_layout_for_workload(workload)
        # At most one positive object can be represented by each authored
        # label element.  The provider's private output is therefore bounded
        # by N packed float64 rows, including across leading blocks.
        output_bytes = (
            primary_elements
            * measurement_layout.packed_width
            * np.dtype(np.float64).itemsize
        )
    if spec.memory_model_id in RICHARDSON_LUCY_MEMORY_MODEL_IDS:
        return estimate_richardson_lucy_memory(
            spec,
            workload,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )
    host_workspace = 0
    uncertainty_floor = 8 * 1024**2

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
    elif spec.memory_model_id == "cupy-sigma-filter-memory-v1":
        parameters = dict(workload.parameters)
        shape = workload.input_shapes[0]
        active_axes = _active_axes(shape, parameters.get("channel_axis"))
        if active_axes is None or len(active_axes) < 2:
            raise ValueError(
                "Sigma Filter memory estimation requires two valid non-channel YX axes."
            )
        # The provider scans one row tile at a time but explicitly materializes
        # the complete authored array in canonical contiguous float32 order.
        # A non-identity axis restoration can retain one typed intermediate
        # alongside the final contiguous output counted below.  Reserve that
        # worst case plus the bounded radius-10 offset table and validation
        # status scalar; no image-sized neighbourhood tensor is constructed.
        full_contiguous_copy = primary_elements * np.dtype(np.float32).itemsize
        typed_axis_staging = primary_elements * primary_itemsize
        offset_and_status_bytes = (325 * 2 * np.dtype(np.int32).itemsize) + 4
        workspace = full_contiguous_copy + typed_axis_staging + offset_and_status_bytes
    elif spec.memory_model_id == "cupyx-canny-exact-memory-v1":
        parameters = dict(workload.parameters)
        scalar_shape, luma_itemsize = _scalar_luma_shape_and_itemsize(
            workload.input_shapes[0],
            workload.input_dtypes[0],
            parameters.get("channel_axis"),
        )
        scalar_elements = math.prod(scalar_shape)
        plane_elements = (
            math.prod(scalar_shape[-2:]) if len(scalar_shape) > 2 else scalar_elements
        )
        uses_luma = parameters.get("channel_axis") is not None
        raw_dtype = np.dtype(workload.input_dtypes[0])
        luma_dtype = np.result_type(raw_dtype, np.float32)
        # RGB/RGBA reduction retains the coefficient product (3N) and scalar
        # luma result (N).  Inputs whose dtype differs from the luma work dtype
        # additionally retain the explicit three-channel cast (3N).  A
        # same-dtype RGB view aliases the already-counted source allocation.
        luma_buffers = 4 + (3 if raw_dtype != luma_dtype else 0)
        luma_workspace = (
            scalar_elements * luma_itemsize * luma_buffers if uses_luma else 0
        )
        # The exact provider retains float32 mask correction, smoothed image,
        # Sobel components/magnitude and NMS arrays while its exact correlation
        # kernels and CuPyX labeling/percentile calls own transient workspaces.
        # Leading planes execute sequentially, so reserve a deliberately
        # conservative 24 float32-equivalent plane buffers rather than scaling
        # by stack depth.
        plane_workspace = plane_elements * np.dtype(np.float32).itemsize * 24
        workspace = luma_workspace + plane_workspace
    elif spec.memory_model_id == "cupy-otsu-histogram-memory-v1":
        parameters = dict(workload.parameters)
        scalar_shape, luma_itemsize = _scalar_luma_shape_and_itemsize(
            workload.input_shapes[0],
            workload.input_dtypes[0],
            parameters.get("channel_axis"),
        )
        scalar_elements = math.prod(scalar_shape)
        scope = (
            str(parameters.get("threshold_scope", "Stack histogram")).strip().casefold()
        )
        histogram_elements = (
            math.prod(scalar_shape[-2:])
            if scope == "slice histogram" and len(scalar_shape) > 2
            else scalar_elements
        )
        raw_dtype = np.dtype(workload.input_dtypes[0])
        uses_luma = parameters.get("channel_axis") is not None
        effective_dtype = (
            np.dtype(f"float{luma_itemsize * 8}") if uses_luma else raw_dtype
        )
        # RGB/RGBA reduction can simultaneously retain the three-channel work
        # cast, its coefficient product, and the scalar luma result.  Count all
        # 3N + 3N + N elements; the source allocation is already included in
        # ``input_bytes`` below.
        luma_workspace = scalar_elements * luma_itemsize * 7 if uses_luma else 0
        if effective_dtype == np.dtype(bool):
            histogram_workspace = 0
            histogram_bins = 0
        elif np.issubdtype(effective_dtype, np.integer):
            # The exact integer path owns uint64 relative levels and an int64
            # bincount input concurrently. Admission proves a span no larger
            # than the authoritative 65,536-level ceiling.
            histogram_bins = OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS
            histogram_workspace = histogram_elements * 16
        else:
            requested_bins = _histogram_bin_count(parameters.get("histogram_bins", 256))
            histogram_bins = requested_bins or OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS
            # Finite mask + compact effective values + float64 histogram input
            # + int64 comparison-derived bin indices.  The latter avoids
            # CuPy's non-exact uniform-range histogram shortcut.
            histogram_workspace = histogram_elements * (
                1
                + effective_dtype.itemsize
                + np.dtype(np.float64).itemsize
                + np.dtype(np.int64).itemsize
            )
        bounded_histogram_bytes = histogram_bins * np.dtype(np.intp).itemsize
        edge_bytes = (histogram_bins + 1) * max(effective_dtype.itemsize, 1)
        workspace = (
            luma_workspace + histogram_workspace + bounded_histogram_bytes + edge_bytes
        )
        # Otsu deliberately finalizes only bounded histogram metadata on the
        # host to retain NumPy's exact cumulative arithmetic and first-argmax
        # tie break.  Reserve counts plus both authored edges and centers.
        host_workspace = bounded_histogram_bytes + 2 * edge_bytes
    elif spec.memory_model_id == "cupyx-connected-components-memory-v1":
        spatial_ndim = workload.resolved_spatial_ndim
        if spatial_ndim not in {2, 3}:
            raise ValueError(
                "Connected-components memory estimation requires a resolved "
                "2D or 3D spatial rank."
            )
        shape = workload.input_shapes[0]
        if len(shape) < spatial_ndim:
            raise ValueError(
                "Connected-components spatial rank exceeds the input rank."
            )
        block_elements = math.prod(shape[-spatial_ndim:])
        # CuPyX processes leading blocks sequentially. Real-device high-label
        # checkerboards retained about 11.1 bytes per spatial element including
        # the one-byte mask and four-byte result. The generic input/output terms
        # below already count those five bytes for the complete authored array;
        # reserve seven additional bytes for the largest active block to cover
        # union-find roots, sorting, and a non-contiguous block copy.
        workspace = block_elements * 7
    elif spec.memory_model_id == "cucim-basic-measurements-memory-v1":
        layout = _basic_measurement_layout_for_workload(workload)
        block_elements = math.prod(layout.spatial_shape)
        include_intensity = len(workload.input_shapes) == 2
        # The complete authored arrays may be retained in canonical axis order.
        # Per active block, reserve compaction/search/sort arrays, cuCIM's
        # region-property workspace, grouped float64 reductions, and packing.
        # Packed rows retained across blocks are already represented by
        # ``output_bytes`` above.
        working_copies = input_bytes
        per_block_workspace = block_elements * (224 if include_intensity else 128)
        # The provider retains every per-block packed matrix and then allocates
        # the final concatenated matrix. At assembly peak both complete packed
        # representations are live, including for a single block because CuPy
        # concatenate materializes a new allocation.
        workspace = working_copies + per_block_workspace + output_bytes
        # D2H first materializes the packed matrix.  Typed Python rows are then
        # constructed transactionally while that matrix is still live.  Python
        # scalar/container overhead is implementation-dependent, so reserve a
        # deliberately conservative multiple rather than claiming byte-exact
        # host accounting.
        host_workspace = output_bytes * 4
        uncertainty_floor = 64 * 1024**2
    else:
        raise ValueError(
            f"No executable memory model is registered for {spec.memory_model_id!r}."
        )
    runtime_peak = input_bytes + output_bytes + workspace
    uncertainty = max(uncertainty_floor, runtime_peak // 4)
    return MemoryEstimate(
        runtime_managed_peak_bytes=runtime_peak,
        total_device_peak_bytes=runtime_peak,
        host_materialization_peak_bytes=output_bytes + host_workspace,
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
    evaluator = _OPERATION_REGION_EVALUATORS.get(spec.parameter_policy_id)
    if evaluator is None:
        return None
    return evaluator(workload, array_facts)


def _background_region_policy(
    workload: WorkloadDescriptor,
    _array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _evaluate_background_region(workload)


def _median_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _evaluate_median_region(workload, array_facts=array_facts)


def _sigma_filter_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _evaluate_sigma_filter_region(workload, array_facts=array_facts)


def _gaussian_2d_region_policy(
    workload: WorkloadDescriptor,
    _array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _evaluate_gaussian_region(workload, three_dimensional=False)


def _gaussian_3d_region_policy(
    workload: WorkloadDescriptor,
    _array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _evaluate_gaussian_region(workload, three_dimensional=True)


def _richardson_lucy_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _adapt_richardson_lucy_region_rejection(
        evaluate_richardson_lucy_region(workload, array_facts)
    )


def _richardson_lucy_tv_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    return _adapt_richardson_lucy_region_rejection(
        evaluate_richardson_lucy_tv_region(workload, array_facts)
    )


def _adapt_richardson_lucy_region_rejection(
    rejection: RegionRejection | None,
) -> SupportDecision | None:
    if rejection is None:
        return None
    return _workload_rejection(
        rejection.reason_text,
        fallback_allowed=rejection.fallback_allowed,
    )


def _canny_region_policy(
    workload: WorkloadDescriptor,
    _array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    parameters = dict(workload.parameters)
    shape = workload.input_shapes[0]
    if not shape or any(extent == 0 for extent in shape):
        return _workload_rejection(
            "Canny requires non-empty image data.",
            fallback_allowed=False,
        )
    _channel_axis, channel_error = _validated_luma_axis(
        shape,
        parameters.get("channel_axis"),
        operation="Canny",
    )
    if channel_error is not None:
        return _workload_rejection(channel_error, fallback_allowed=False)
    scalar_rank = len(shape) - (_channel_axis is not None)
    if scalar_rank < 2:
        return _workload_rejection(
            "Canny requires trailing two-dimensional scalar image planes.",
            fallback_allowed=False,
        )

    low = _finite_number(parameters.get("low_quantile", 0.1))
    high = _finite_number(parameters.get("high_quantile", 0.2))
    if low is None or high is None:
        return _workload_rejection(
            "Canny low and high quantiles must be finite numbers.",
            fallback_allowed=False,
        )
    if not 0.0 <= low <= 1.0 or not 0.0 <= high <= 1.0:
        return _workload_rejection(
            "Canny low and high quantiles must be between 0 and 1.",
            fallback_allowed=False,
        )
    if high < low:
        return _workload_rejection(
            "Canny low quantile must not exceed the high quantile.",
            fallback_allowed=False,
        )

    sigma = _finite_number(parameters.get("sigma", 1.0))
    if sigma is None:
        return _workload_rejection(
            "Canny sigma must be a finite number.",
            fallback_allowed=False,
        )
    canonical_sigma = max(sigma, 0.0)
    if canonical_sigma > 12.0:
        return _workload_rejection(
            "Canny GPU execution is validated for canonical sigma values in "
            "the 0..12 range; this authored value remains on CPU."
        )

    dtype = _dtype_name(workload.input_dtypes[0])
    if dtype == "float32":
        return _workload_rejection(
            "Canny float32 GPU execution remains on CPU because finite float32 "
            "inputs can produce subnormal Gaussian, gradient, magnitude, or "
            "interpolation intermediates. CUDA flush-to-zero behavior can then "
            "change an exact edge-mask bit; bool, uint8, and uint16 retain the "
            "validated GPU path."
        )
    if dtype not in {"bool", "uint8", "uint16"}:
        return _workload_rejection(
            f"Canny GPU execution has no promoted {dtype!r} input region; CPU "
            "remains authoritative."
        )
    return None


def _otsu_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    parameters = dict(workload.parameters)
    shape = workload.input_shapes[0]
    if any(extent == 0 for extent in shape):
        return _workload_rejection(
            "Otsu threshold requires non-empty image data.",
            fallback_allowed=False,
        )
    channel_axis, channel_error = _validated_luma_axis(
        shape,
        parameters.get("channel_axis"),
        operation="Otsu threshold",
    )
    if channel_error is not None:
        return _workload_rejection(channel_error, fallback_allowed=False)

    scope = str(parameters.get("threshold_scope", "Stack histogram")).strip().casefold()
    if scope not in {"stack histogram", "slice histogram"}:
        return _workload_rejection(
            "Threshold scope must be 'Stack histogram' or 'Slice histogram'.",
            fallback_allowed=False,
        )

    dtype = np.dtype(workload.input_dtypes[0])
    if dtype.kind not in "biuf":
        return _workload_rejection(
            "Automatic histogram thresholds require boolean, integer, or "
            "floating-point image data.",
            fallback_allowed=False,
        )
    effective_float = channel_axis is not None or np.issubdtype(dtype, np.floating)
    if effective_float:
        if (
            _histogram_bin_count(
                parameters.get("histogram_bins", OTSU_DEFAULT_HISTOGRAM_BINS)
            )
            is None
        ):
            return _workload_rejection(
                "Float histogram bins must be an integer from 2 to 65,536.",
                fallback_allowed=False,
            )
        if (
            channel_axis is None
            and array_facts
            and array_facts[0].completeness is FactCompleteness.COMPLETE
            and array_facts[0].finite_count == 0
        ):
            return _workload_rejection(
                "Automatic thresholding requires at least one finite input value.",
                fallback_allowed=False,
            )
        return None

    if dtype == np.dtype(bool):
        # The CPU contract treats a scalar boolean image as an identity and
        # deliberately does not inspect histogram_bins.
        return None
    if np.issubdtype(dtype, np.integer):
        dtype_limits = np.iinfo(dtype)
        type_span = int(dtype_limits.max) - int(dtype_limits.min) + 1
        if type_span <= OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS:
            # Every possible value of <=16-bit integer dtypes fits the exact
            # native-level histogram, so an image-sized extrema scan cannot
            # change admission and is intentionally skipped.
            return None
    if not array_facts:
        return _complete_facts_rejection(
            "Exact wide-integer Otsu admission requires complete native extrema facts."
        )
    facts = array_facts[0]
    if (
        facts.completeness is not FactCompleteness.COMPLETE
        or not isinstance(facts.minimum, int)
        or not isinstance(facts.maximum, int)
    ):
        return _complete_facts_rejection(
            "Exact wide-integer Otsu admission requires complete native integer "
            "minimum and maximum facts."
        )
    span = facts.maximum - facts.minimum + 1
    if span < 1:
        return _workload_rejection(
            "Integer Otsu facts contain an invalid native intensity range.",
            fallback_allowed=False,
        )
    if span > OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS:
        scalar_shape = shape
        if channel_axis is not None:
            scalar_shape = shape[:channel_axis] + shape[channel_axis + 1 :]
        if (
            scope == "slice histogram"
            and len(scalar_shape) > 2
            and math.prod(scalar_shape[:-2]) > 1
        ):
            return _workload_rejection(
                "Exact integer Otsu slice admission currently has only "
                "whole-stack extrema. The stack spans "
                f"{span:,} levels, so GPU safety cannot be proved per plane; "
                "CPU remains authoritative for this run. Individual planes "
                "may still be valid.",
            )
        return _workload_rejection(
            f"Integer intensity span contains {span:,} levels; automatic "
            "thresholding supports at most 65,536 exact integer levels. Convert "
            "or rescale the image explicitly.",
            fallback_allowed=False,
        )
    return None


def _connected_components_region_policy(
    workload: WorkloadDescriptor,
    _array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    if len(workload.input_shapes) != 1 or len(workload.input_dtypes) != 1:
        return _workload_rejection(
            "Connected components requires exactly one mask input.",
            fallback_allowed=False,
        )
    if _dtype_name(workload.input_dtypes[0]) != "bool":
        return _workload_rejection(
            "The promoted connected-components GPU region requires a boolean "
            "mask. Numeric nonzero coercion remains authoritative on CPU."
        )
    spatial_ndim = workload.resolved_spatial_ndim
    if spatial_ndim not in {2, 3}:
        return _workload_rejection(
            "Connected-components GPU execution requires a resolved 2D or 3D "
            "spatial rank.",
            fallback_allowed=False,
        )
    shape = workload.input_shapes[0]
    if len(shape) < spatial_ndim:
        return _workload_rejection(
            "Connected-components spatial rank exceeds the input rank.",
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
        return _workload_rejection(
            "Connected-components spatial parameters disagree with the resolved rank.",
            fallback_allowed=False,
        )
    connectivity = (
        str(parameters.get("connectivity", "Full connectivity")).strip().casefold()
    )
    if connectivity not in {"face connected", "full connectivity"}:
        return _workload_rejection(
            "Connectivity must be 'Face connected' or 'Full connectivity'.",
            fallback_allowed=False,
        )

    block_elements = math.prod(shape[-spatial_ndim:])
    if block_elements >= CONNECTED_COMPONENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        return _workload_rejection(
            "Each connected-components GPU spatial block must contain fewer than "
            f"{CONNECTED_COMPONENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements "
            "so the exact CuPyX int32 label path remains valid."
        )
    return None


def _basic_measurements_region_policy(
    workload: WorkloadDescriptor,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    include_intensity = workload.operation_id == "measure_objects_intensity"
    expected_inputs = 2 if include_intensity else 1
    if len(workload.input_shapes) != expected_inputs:
        return _workload_rejection(
            f"{workload.operation_id} requires exactly {expected_inputs} ordered "
            "input port(s).",
            fallback_allowed=False,
        )
    if include_intensity and workload.input_shapes[1] != workload.input_shapes[0]:
        return _workload_rejection(
            "Intensity-aware measurements require labels and intensity image "
            "with the same shape.",
            fallback_allowed=False,
        )

    try:
        labels_dtype = np.dtype(workload.input_dtypes[0])
    except (TypeError, ValueError):
        return _workload_rejection(
            "Object measurements require a valid non-boolean integer label dtype.",
            fallback_allowed=False,
        )
    if labels_dtype == np.dtype(bool) or not np.issubdtype(
        labels_dtype,
        np.integer,
    ):
        return _workload_rejection(
            "Object measurements require a non-boolean integer label image; "
            f"{labels_dtype} is invalid for both CPU and GPU execution.",
            fallback_allowed=False,
        )
    if labels_dtype != np.dtype(np.int32) or not labels_dtype.isnative:
        return _workload_rejection(
            "The promoted basic-measurement GPU region requires native int32 "
            "labels. Other non-negative integer label dtypes remain on CPU."
        )

    spatial_ndim = workload.resolved_spatial_ndim
    if spatial_ndim not in {2, 3}:
        return _workload_rejection(
            "Basic GPU measurements require a resolved 2D or 3D spatial rank.",
            fallback_allowed=False,
        )
    parameters = dict(workload.parameters)
    extended_options = (
        "include_shape_descriptors",
        "include_axis_descriptors",
        "include_2d_boundary_descriptors",
        "include_derived_shape_ratios",
        "include_2d_shape_moments",
    )
    for name in extended_options:
        value = parameters.get(name, False)
        if not isinstance(value, bool):
            return _workload_rejection(
                f"Measurement option {name!r} must be boolean.",
                fallback_allowed=False,
            )
        if value:
            return _workload_rejection(
                "Extended measurement columns are not yet in the promoted GPU "
                f"region ({name} is enabled); the complete CPU schema remains "
                "authoritative."
            )
    try:
        layout = _basic_measurement_layout_for_workload(workload)
    except (TypeError, ValueError) as exc:
        return _workload_rejection(
            f"The authored measurement axis layout is invalid: {exc}",
            fallback_allowed=False,
        )
    spatial_elements = math.prod(layout.spatial_shape)
    if spatial_elements == 0:
        return _workload_rejection(
            "Empty spatial measurement blocks remain on CPU."
        )
    if spatial_elements >= MEASUREMENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        return _workload_rejection(
            "Each GPU measurement spatial block must contain fewer than "
            f"{MEASUREMENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements so "
            "private compact labels remain int32-safe."
        )

    if not array_facts:
        return _complete_facts_rejection(
            "Basic GPU measurements require complete label facts proving that "
            "all native int32 labels are non-negative."
        )
    label_facts = array_facts[0]
    if label_facts.completeness is not FactCompleteness.COMPLETE:
        return _complete_facts_rejection(
            "Basic GPU measurements require complete non-negative label facts."
        )
    if "nonnegative" not in label_facts.guarantees:
        if label_facts.minimum is not None and label_facts.minimum < 0:
            return _workload_rejection(
                "Object labels must contain only non-negative integers.",
                fallback_allowed=False,
            )
        return _complete_facts_rejection(
            "Complete label facts did not prove the non-negative label region."
        )

    if not include_intensity:
        return None
    try:
        intensity_dtype = np.dtype(workload.input_dtypes[1])
    except (TypeError, ValueError):
        return _workload_rejection(
            "Basic GPU intensity measurements require a supported native dtype."
        )
    supported_intensity_dtypes = {
        np.dtype(bool),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    }
    if (
        intensity_dtype not in supported_intensity_dtypes
        or not intensity_dtype.isnative
    ):
        return _workload_rejection(
            "The promoted intensity-measurement GPU region supports native bool, "
            "uint8, uint16, and finite float32 data; this dtype remains on CPU."
        )
    if intensity_dtype == np.dtype(np.float32):
        if len(array_facts) != 2:
            return _complete_facts_rejection(
                "GPU float32 intensity measurements require complete finite-value "
                "facts for the intensity input."
            )
        intensity_facts = array_facts[1]
        if (
            intensity_facts.completeness is not FactCompleteness.COMPLETE
            or intensity_facts.all_finite is not True
        ):
            return _complete_facts_rejection(
                "GPU float32 intensity measurements require completely finite "
                "intensity data."
            )
    return None


_OPERATION_REGION_EVALUATORS: Mapping[
    str,
    Callable[
        [WorkloadDescriptor, tuple[ArrayFacts, ...]],
        SupportDecision | None,
    ],
] = MappingProxyType(
    {
        "background-parameters-v1": _background_region_policy,
        "median-parameters-v1": _median_region_policy,
        "sigma-filter-parameters-v1": _sigma_filter_region_policy,
        "gaussian-2d-parameters-v1": _gaussian_2d_region_policy,
        "gaussian-3d-parameters-v1": _gaussian_3d_region_policy,
        "rl-parameters-v1": _richardson_lucy_region_policy,
        "rl-tv-parameters-v1": _richardson_lucy_tv_region_policy,
        "canny-parameters-v1": _canny_region_policy,
        "otsu-parameters-v1": _otsu_region_policy,
        "connected-components-parameters-v1": (_connected_components_region_policy),
        "basic-measurements-parameters-v1": _basic_measurements_region_policy,
    }
)


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


def _evaluate_sigma_filter_region(
    workload: WorkloadDescriptor,
    *,
    array_facts: tuple[ArrayFacts, ...],
) -> SupportDecision | None:
    try:
        input_dtype = np.dtype(workload.input_dtypes[0])
    except (TypeError, ValueError):
        return _workload_rejection(
            "Sigma Filter requires a concrete NumPy-compatible input dtype."
        )
    if not input_dtype.isnative:
        return _workload_rejection(
            "Sigma Filter GPU execution requires native-endian input so its "
            "dtype contract matches the authoritative CPU operation."
        )
    dtype = input_dtype.name
    if dtype not in {"uint8", "uint16", "float32"}:
        return _workload_rejection(
            f"Sigma Filter GPU execution supports uint8, uint16, and float32, "
            f"not {dtype!r}; CPU is required."
        )
    shape = workload.input_shapes[0]
    if not shape or any(size <= 0 for size in shape):
        return _workload_rejection(
            "Sigma Filter requires non-empty image data.",
            fallback_allowed=False,
        )
    parameters = dict(workload.parameters)
    active_axes = _active_axes(shape, parameters.get("channel_axis"))
    if active_axes is None or len(active_axes) < 2:
        return _workload_rejection(
            "Sigma Filter requires two valid non-channel YX axes.",
            fallback_allowed=False,
        )

    radius = _finite_number(parameters.get("radius", 2.0))
    if radius is None or not 0.5 <= radius <= 10.0:
        return _workload_rejection(
            "Sigma Filter radius must be finite and between 0.5 and 10 inclusive.",
            fallback_allowed=False,
        )
    sigma_width = _finite_number(parameters.get("sigma_width", 2.0))
    if sigma_width is None or sigma_width < 0.0:
        return _workload_rejection(
            "Sigma Filter sigma_width must be finite and non-negative.",
            fallback_allowed=False,
        )
    minimum_fraction = _finite_number(parameters.get("minimum_pixel_fraction", 0.2))
    if minimum_fraction is None or not 0.0 <= minimum_fraction <= 1.0:
        return _workload_rejection(
            "Sigma Filter minimum_pixel_fraction must be finite and between "
            "zero and one.",
            fallback_allowed=False,
        )
    if not isinstance(parameters.get("outlier_aware", True), (bool, np.bool_)):
        return _workload_rejection(
            "Sigma Filter outlier_aware must be boolean.",
            fallback_allowed=False,
        )

    if dtype == "float32":
        if not array_facts:
            return _complete_facts_rejection(
                "Float32 Sigma Filter requires complete finite magnitude facts."
            )
        facts = array_facts[0]
        if (
            facts.completeness is not FactCompleteness.COMPLETE
            or facts.all_finite is not True
            or facts.minimum is None
            or facts.maximum is None
        ):
            return _complete_facts_rejection(
                "Float32 Sigma Filter requires complete finite extrema facts."
            )
        maximum_magnitude = max(
            abs(float(facts.minimum)),
            abs(float(facts.maximum)),
        )
        if maximum_magnitude > SIGMA_FILTER_FLOAT32_SQUARE_LIMIT:
            return _workload_rejection(
                "Float32 Sigma Filter input magnitude would overflow the "
                "Fiji-compatible float32 square workspace."
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
    if isinstance(raw_channel_axis, (bool, np.bool_)) or not isinstance(
        raw_channel_axis,
        Integral,
    ):
        return None
    if (
        len(shape) < 3
        or raw_channel_axis < -len(shape)
        or raw_channel_axis >= len(shape)
    ):
        return None
    axes.remove(raw_channel_axis % len(shape))
    return tuple(axes)


def _validated_luma_axis(
    shape: tuple[int, ...],
    raw_channel_axis: object,
    *,
    operation: str,
) -> tuple[int | None, str | None]:
    """Resolve an explicit RGB/RGBA axis without touching array data."""

    if raw_channel_axis is None:
        return None, None
    if isinstance(raw_channel_axis, (bool, np.bool_)) or not isinstance(
        raw_channel_axis,
        Integral,
    ):
        return None, f"{operation} channel_axis must be an integer or None."
    if len(shape) < 3:
        return (
            None,
            f"{operation} requires at least two spatial dimensions when "
            "channel_axis is set.",
        )
    if raw_channel_axis < -len(shape) or raw_channel_axis >= len(shape):
        return (
            None,
            f"{operation} channel_axis {raw_channel_axis} is out of range for "
            f"{len(shape)}D input.",
        )
    axis = int(raw_channel_axis % len(shape))
    channel_count = shape[axis]
    if channel_count not in {3, 4}:
        return (
            None,
            f"{operation} channel_axis must contain exactly 3 RGB or 4 RGBA "
            f"channels, not {channel_count}.",
        )
    return axis, None


def _scalar_luma_shape_and_itemsize(
    shape: tuple[int, ...],
    dtype: object,
    raw_channel_axis: object,
) -> tuple[tuple[int, ...], int]:
    """Return post-luma scalar shape and work itemsize for memory policy."""

    axis, error = _validated_luma_axis(
        shape,
        raw_channel_axis,
        operation="GPU operation",
    )
    if error is not None or axis is None:
        return shape, _dtype_itemsize(dtype)
    scalar_shape = shape[:axis] + shape[axis + 1 :]
    work_dtype = np.result_type(np.dtype(dtype), np.float32)
    return scalar_shape, int(work_dtype.itemsize)


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


def _histogram_bin_count(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        return None
    return count if 2 <= count <= OTSU_MAXIMUM_NATIVE_INTEGER_LEVELS else None


def _dtype_can_contain_nonfinite(value: object) -> bool:
    try:
        return bool(np.issubdtype(np.dtype(value), np.inexact))
    except (TypeError, ValueError):
        return True


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


def _basic_measurement_layout_for_workload(workload: WorkloadDescriptor):
    if workload.operation_id not in {
        "measure_objects",
        "measure_objects_intensity",
    }:
        raise ValueError("workload is not a basic measurement operation.")
    expected_inputs = 2 if workload.operation_id == "measure_objects_intensity" else 1
    if len(workload.input_shapes) != expected_inputs:
        raise ValueError(
            f"{workload.operation_id} requires {expected_inputs} input shape(s)."
        )
    parameters = dict(workload.parameters)
    return basic_measurement_layout(
        workload.input_shapes[0],
        spatial_mode=str(parameters.get("spatial_mode", "Auto from axes")),
        resolved_spatial_ndim=workload.resolved_spatial_ndim,
        axis_names=parameters.get("axis_names"),
        axis_types=parameters.get("axis_types"),
        axis_scales=parameters.get("axis_scales"),
        axis_units=parameters.get("axis_units"),
        include_intensity=(workload.operation_id == "measure_objects_intensity"),
    )


def _output_port_itemsize(policy_id: str, primary_itemsize: int) -> int:
    """Resolve static output storage without importing an implementation.

    Fixed dtype policies are intentionally data-driven so future boolean,
    label, and conversion providers do not need another memory-model branch.
    """

    normalized = str(policy_id).strip()
    if normalized == "dtype-same-v1":
        return primary_itemsize
    prefix = "fixed:"
    if normalized.startswith(prefix):
        return _dtype_itemsize(normalized[len(prefix) :])
    raise ValueError(f"Unsupported output dtype policy {normalized!r}.")


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
            "sigma-filter-parameters-v1",
            "gaussian-2d-parameters-v1",
            "gaussian-3d-parameters-v1",
            *RICHARDSON_LUCY_POLICY_IDS["parameter"],
            "canny-parameters-v1",
            "otsu-parameters-v1",
            "connected-components-parameters-v1",
            "basic-measurements-parameters-v1",
        },
        PolicyKind.WORKLOAD: {
            "cpu-reference-v1",
            "vipp-best-available-v1",
            "background-u8-u16-f32-v2",
            "median-exact-u8-u16-f32-v1",
            "sigma-u8-u16-finite-f32-v1",
            "gaussian-finite-f32-v1",
            *RICHARDSON_LUCY_POLICY_IDS["workload"],
            "canny-exact-bool-u8-u16-v2",
            "otsu-real-exact-v1",
            "connected-components-bool-2d-3d-v1",
            "measurements-int32-basic-2d-3d-v1",
            "measurements-int32-bool-u8-u16-finite-f32-basic-2d-3d-v1",
        },
        PolicyKind.PARITY: {
            "authoritative-cpu-v1",
            "background-dtype-parity-v2",
            "median-production-bitwise-v1",
            "sigma-dtype-parity-v1",
            "gaussian-float32-tolerance-v1",
            *RICHARDSON_LUCY_POLICY_IDS["parity"],
            "mask-bitwise-v1",
            "labels-bitwise-int32-v1",
            "basic-measurement-table-v1",
        },
        PolicyKind.MEMORY: {
            "host-reference-v1",
            "cucim-background-memory-v1",
            "cupyx-median-memory-v1",
            "cupy-sigma-filter-memory-v1",
            "cupyx-gaussian-2d-memory-v1",
            "cupyx-gaussian-3d-memory-v1",
            *RICHARDSON_LUCY_POLICY_IDS["memory"],
            "cupyx-canny-exact-memory-v1",
            "cupy-otsu-histogram-memory-v1",
            "cupyx-connected-components-memory-v1",
            "cucim-basic-measurements-memory-v1",
        },
        PolicyKind.SHAPE: {
            "cpu-reference-v1",
            "cpu-dynamic-output-v1",
            "shape-unknown-v1",
            "shape-preserving-v1",
            "psf-spatial-kernel-v1",
            "scalar-plane-luma-mask-v1",
            "measurement-input-shape-v1",
            "measurement-packed-rows-v1",
        },
        PolicyKind.OUTPUT_DTYPE: {
            "dtype-same-v1",
            "fixed:float32",
            "fixed:bool",
            "fixed:int32",
            "fixed:float64",
            "cpu-dynamic-output-v1",
        },
        PolicyKind.CONVERSION: {
            "identity-v1",
            "background-float-workspace-restore-v1",
            "cupyx-median-identity-v1",
            "sigma-float32-workspace-restore-v1",
            "cupyx-gaussian-float32-v1",
            *RICHARDSON_LUCY_POLICY_IDS["conversion"],
            "canny-plane-float32-or-luma-v1",
            "otsu-native-or-luma-v1",
            "binary-mask-to-int32-labels-v1",
            "measurement-native-labels-v1",
            "measurement-intensity-float64-reductions-v1",
            "packed-float64-to-typed-table-v1",
        },
        PolicyKind.NONFINITE: {
            "cpu-reference-v1",
            "background-cpu-parity-v1",
            "finite-no-negative-zero-v1",
            "finite-only-v1",
            "finite-output-v1",
            "sigma-finite-only-v1",
            "otsu-finite-histogram-v1",
            "integer-nonnegative-v1",
            "measurement-packed-finite-v1",
        },
        PolicyKind.ROUNDING: {
            "cpu-reference-v1",
            "background-bankers-round-clip-v1",
            "median-bitwise-v1",
            "sigma-half-up-u8-u16-f32-identity-v1",
            "gaussian-float32-tolerance-v1",
            *RICHARDSON_LUCY_POLICY_IDS["rounding"],
            "mask-bitwise-v1",
            "labels-bitwise-int32-v1",
            "measurement-two-pass-reductions-v1",
            "measurement-typed-fields-v1",
        },
        PolicyKind.OVERFLOW: {
            "cpu-reference-v1",
            "background-clip-public-dtype-v1",
            "preserve-public-dtype-v1",
            "sigma-float32-square-safe-v1",
            *RICHARDSON_LUCY_POLICY_IDS["overflow"],
            "finite-float32-workspace-v1",
            "binary-mask-v1",
            "otsu-native-span-v1",
            "connected-components-int32-safe-v1",
            "measurements-int32-label-compact-v1",
            "measurement-float64-reductions-v1",
            "measurement-exact-integer-fields-v1",
        },
        PolicyKind.BOUNDARY: {
            "cpu-reference-v1",
            "background-nearest-rolling-ball-v1",
            "scipy-reflect-v1",
            "sigma-nearest-circular-footprint-v1",
            *RICHARDSON_LUCY_POLICY_IDS["boundary"],
            "skimage-canny-constant-zero-v1",
            "otsu-strict-greater-finite-mask-v1",
            "scipy-binary-connectivity-v1",
            "measurement-leading-spatial-blocks-v1",
        },
        PolicyKind.PRECISION: {
            "scientific-default-v1",
            "background-public-dtype-v2",
            "median-bitwise-v1",
            "sigma-ordered-f32-square-f64-accum-v1",
            "gaussian-float32-v1",
            *RICHARDSON_LUCY_POLICY_IDS["precision"],
            "canny-exact-mask-v1",
            "otsu-exact-mask-v1",
            "connected-components-exact-label-order-v1",
            "basic-measurement-table-v1",
        },
        PolicyKind.PROGRESS: {
            "cpu-reference-v1",
            "background-block-progress-v1",
            "monolithic-sync-progress-v1",
            "sigma-row-tile-sync-progress-v1",
            *RICHARDSON_LUCY_POLICY_IDS["progress"],
            "scalar-plane-sync-progress-v1",
            "histogram-scope-sync-progress-v1",
            "spatial-block-sync-progress-v1",
            "measurement-block-stage-progress-v1",
        },
        PolicyKind.CANCELLATION: {
            "cpu-reference-v1",
            "background-block-cancel-v1",
            "monolithic-boundary-cancel-v1",
            "sigma-row-tile-boundary-cancel-v1",
            *RICHARDSON_LUCY_POLICY_IDS["cancellation"],
            "scalar-plane-boundary-cancel-v1",
            "spatial-block-boundary-cancel-v1",
            "measurement-block-stage-cancel-v1",
        },
        PolicyKind.SIDE_EFFECT: {
            "pure-or-source-v1",
            "host-writer-v1",
            "pure-v1",
        },
        PolicyKind.DYNAMIC_OUTPUT: {
            "static-v1",
            "cpu-dynamic-output-v1",
            "typed-host-table-finalizer-v1",
        },
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
    "CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID",
    "CUDA_CUPY_RAWKERNEL_WINDOWS_ENVIRONMENT_POLICY_ID",
    "CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID",
    "CUDA_ENVIRONMENT_POLICIES",
    "CONNECTED_COMPONENTS_MAXIMUM_SPATIAL_BLOCK_ELEMENTS",
    "DEFAULT_POLICY_CATALOG",
    "FactCompleteness",
    "PerformanceDecision",
    "PerformanceEvidence",
    "PolicyCatalog",
    "PolicyKind",
    "RICHARDSON_LUCY_MAXIMUM_ITERATIONS",
    "RICHARDSON_LUCY_FILTER_EPSILON",
    "RICHARDSON_LUCY_TV_DENOMINATOR_FLOOR",
    "RICHARDSON_LUCY_TV_EPSILON",
    "RICHARDSON_LUCY_TV_FILTER_EPSILON",
    "RICHARDSON_LUCY_TV_MAXIMUM_ITERATIONS",
    "RICHARDSON_LUCY_TV_POSITIVE_ITERATIONS",
    "RICHARDSON_LUCY_TV_REGULARIZATION",
    "SIGMA_FILTER_FLOAT32_SQUARE_LIMIT",
    "SupportDecision",
    "ValueDescriptor",
    "evaluate_auto_performance",
    "evaluate_candidate_support",
    "evaluate_candidate_workload_support",
    "evaluate_memory_support",
    "estimate_candidate_memory",
    "propagate_output_descriptors",
    "validate_spec_policy_references",
]
