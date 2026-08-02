"""Strict loader for packaged, provider-independent compute policy records.

The resource is an auditable derivative of executable scientific policy.  It
does not replace the admission checks in :mod:`napari_vipp.core.compute_policy`;
numeric support summaries are explicitly labelled as mirrors so drift can be
detected without making a JSON file a second execution engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any

POLICY_SCHEMA_ID = "napari-vipp-compute-policy-artifact-v1"
PHASE1_POLICY_ID = "phase1-gpu-public-v4"
PHASE1_POLICY_RESOURCE = "phase1-gpu-public-v4.json"
PHASE1_POLICY_SHA256 = (
    "0a9565111f29cc44f35250f6fbb8ef78b4cdf0551d5d19ee3fc1107b767cb473"
)
_POLICY_PACKAGE = "napari_vipp.compute_policies"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


class ComputePolicyArtifactError(ValueError):
    """A packaged compute policy is invalid or unsupported."""


class ComputePolicyDigestError(ComputePolicyArtifactError):
    """A packaged compute policy does not match its declared digest."""


@dataclass(frozen=True, slots=True)
class AutoSelectionPolicy:
    broad_calibration_enabled: bool
    evidence_scope: str
    non_local_lower_confidence_speedup: float
    non_local_minimum_saving_ms: float
    local_noise_relative_fraction: float
    local_noise_absolute_ms: float
    tie_breaker: str
    pending_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalBenchmarkPolicy:
    initial_warm_rounds: int
    adaptive_warm_rounds: tuple[int, ...]
    confidence_level: float
    confidence_method: str
    bootstrap_resamples: int
    bootstrap_seed: int
    outlier_policy: str
    round_order_policy: str
    adaptive_near_threshold_relative_fraction: float
    adaptive_near_threshold_absolute_ms: float
    adaptive_speedup_mad_fraction: float


@dataclass(frozen=True, slots=True)
class PlatformAdmissionPolicy:
    operating_systems: tuple[str, ...]
    execution_modes: tuple[str, ...]
    python_implementation: str
    python_minor_versions: tuple[str, ...]
    python_abis: tuple[str, ...]
    numpy_versions: tuple[str, ...]
    scipy_versions: tuple[str, ...]
    scikit_image_versions: tuple[str, ...]
    cuda_major_versions: tuple[int, ...]
    cuda_runtime_versions: tuple[str, ...]
    driver_versions: tuple[str, ...]
    cupy_versions: tuple[str, ...]
    cupyx_versions: tuple[str, ...]
    runtime_probe_fingerprint_required: bool
    driver_version_metadata_required: bool
    nvidia_compute_capability_required: bool
    nvidia_device_names: tuple[str, ...]
    nvidia_compute_capabilities: tuple[str, ...]
    cucim_versions: tuple[str, ...]
    cucim_environment_record_schema: str
    cucim_environment_record_schema_version: int
    cucim_environment_track: str
    cupy_distribution: str
    cucim_distribution: str
    cucim_artifact_sha256: str
    validated_environment_policy_ids: tuple[str, ...]
    linux_policy: str
    macos_policy: str
    public_advertisement_enabled: bool


@dataclass(frozen=True, slots=True)
class ParameterBound:
    parameter: str
    minimum: float
    maximum: float
    scope: str
    canonicalization_policy_id: str


@dataclass(frozen=True, slots=True)
class OperationSupportSummary:
    authority: str
    public_dtypes: tuple[str, ...]
    spatial_semantics_id: str
    required_facts: tuple[str, ...]
    parameter_bounds: tuple[ParameterBound, ...]
    explicit_cpu_regions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PackagedOperationPolicy:
    operation_id: str
    implementation_id: str
    implementation_version: str
    runtime_id: str
    implementation_library_id: str
    admission_tier: str
    environment_policy_id: str
    parameter_policy_id: str
    workload_policy_id: str
    parity_policy_id: str
    memory_model_id: str
    supported_spatial_ndims: tuple[int, ...]
    supports_device_residency: bool
    support_summary: OperationSupportSummary


@dataclass(frozen=True, slots=True)
class ComputePolicyExposure:
    global_default: str
    public_controls_enabled: bool
    developer_enablement_required: bool


@dataclass(frozen=True, slots=True)
class Phase1ComputePolicyArtifact:
    schema_id: str
    policy_id: str
    policy_version: int
    content_sha256: str
    phase: str
    status: str
    exposure: ComputePolicyExposure
    auto_selection: AutoSelectionPolicy
    local_benchmark: LocalBenchmarkPolicy
    platform_admission: PlatformAdmissionPolicy
    operations: tuple[PackagedOperationPolicy, ...]

    def operation(self, operation_id: str) -> PackagedOperationPolicy:
        """Return the unique policy for ``operation_id``."""

        matches = tuple(
            record for record in self.operations if record.operation_id == operation_id
        )
        if len(matches) != 1:
            raise KeyError(operation_id)
        return matches[0]


def canonical_artifact_digest(document: Mapping[str, object]) -> str:
    """Return the stable SHA-256 over an artifact excluding its digest field."""

    payload = dict(document)
    payload.pop("content_sha256", None)
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ComputePolicyArtifactError(
            "Compute policy content is not canonical JSON."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def load_phase1_compute_policy() -> Phase1ComputePolicyArtifact:
    """Load Phase 1 policy through package resources, including from a wheel."""

    resource = resources.files(_POLICY_PACKAGE).joinpath(PHASE1_POLICY_RESOURCE)
    return parse_compute_policy_artifact(resource.read_bytes())


def parse_compute_policy_artifact(
    raw: bytes | bytearray | memoryview | str,
) -> Phase1ComputePolicyArtifact:
    """Verify and strictly parse the supported Phase 1 policy document."""

    try:
        document = json.loads(
            bytes(raw).decode("utf-8") if not isinstance(raw, str) else raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComputePolicyArtifactError(
            "Compute policy is not valid UTF-8 JSON."
        ) from exc
    root = _object(
        document,
        "$",
        {"schema_id", "policy_id", "policy_version", "content_sha256", "policy"},
    )
    declared_digest = _string(root["content_sha256"], "$.content_sha256")
    if not _SHA256_RE.fullmatch(declared_digest):
        raise ComputePolicyArtifactError(
            "$.content_sha256 must be a lowercase SHA-256 digest."
        )
    actual_digest = canonical_artifact_digest(root)
    if actual_digest != declared_digest:
        raise ComputePolicyDigestError(
            "Compute policy digest mismatch: packaged content may be damaged "
            "or tampered."
        )

    schema_id = _string(root["schema_id"], "$.schema_id")
    if schema_id != POLICY_SCHEMA_ID:
        raise ComputePolicyArtifactError(
            f"Unsupported compute policy schema {schema_id!r}."
        )
    policy_id = _identifier(root["policy_id"], "$.policy_id")
    if policy_id != PHASE1_POLICY_ID:
        raise ComputePolicyArtifactError(f"Unsupported Phase 1 policy {policy_id!r}.")
    policy_version = _integer(root["policy_version"], "$.policy_version", minimum=1)
    if policy_version != 4:
        raise ComputePolicyArtifactError(
            f"Unsupported Phase 1 policy version {policy_version}."
        )

    policy = _object(
        root["policy"],
        "$.policy",
        {
            "phase",
            "status",
            "exposure",
            "auto_selection",
            "local_benchmark",
            "platform_admission",
            "operations",
        },
    )
    operations_raw = _array(policy["operations"], "$.policy.operations", nonempty=True)
    operations = tuple(
        _parse_operation(value, f"$.policy.operations[{index}]")
        for index, value in enumerate(operations_raw)
    )
    _require_unique(
        (operation.operation_id for operation in operations),
        "$.policy.operations operation_id",
    )
    _require_unique(
        (operation.implementation_id for operation in operations),
        "$.policy.operations implementation_id",
    )
    result = Phase1ComputePolicyArtifact(
        schema_id=schema_id,
        policy_id=policy_id,
        policy_version=policy_version,
        content_sha256=declared_digest,
        phase=_literal(policy["phase"], "$.policy.phase", {"phase1"}),
        status=_literal(policy["status"], "$.policy.status", {"public-validated"}),
        exposure=_parse_exposure(policy["exposure"]),
        auto_selection=_parse_auto_selection(policy["auto_selection"]),
        local_benchmark=_parse_benchmark(policy["local_benchmark"]),
        platform_admission=_parse_platform(policy["platform_admission"]),
        operations=operations,
    )
    if result.content_sha256 != PHASE1_POLICY_SHA256:
        raise ComputePolicyDigestError(
            "Compute policy content is valid but is not the immutable Phase 1 record."
        )
    return result


def _parse_exposure(value: object) -> ComputePolicyExposure:
    path = "$.policy.exposure"
    record = _object(
        value,
        path,
        {"global_default", "public_controls_enabled", "developer_enablement_required"},
    )
    return ComputePolicyExposure(
        global_default=_literal(
            record["global_default"], f"{path}.global_default", {"auto"}
        ),
        public_controls_enabled=_boolean(
            record["public_controls_enabled"], f"{path}.public_controls_enabled"
        ),
        developer_enablement_required=_boolean(
            record["developer_enablement_required"],
            f"{path}.developer_enablement_required",
        ),
    )


def _parse_auto_selection(value: object) -> AutoSelectionPolicy:
    path = "$.policy.auto_selection"
    record = _object(
        value,
        path,
        {
            "broad_calibration_enabled",
            "evidence_scope",
            "non_local_lower_confidence_speedup",
            "non_local_minimum_saving_ms",
            "local_noise_relative_fraction",
            "local_noise_absolute_ms",
            "tie_breaker",
            "pending_evidence",
        },
    )
    lower_speedup = _number(
        record["non_local_lower_confidence_speedup"],
        f"{path}.non_local_lower_confidence_speedup",
        minimum=1.0,
    )
    relative_noise = _number(
        record["local_noise_relative_fraction"],
        f"{path}.local_noise_relative_fraction",
        minimum=0.0,
    )
    if relative_noise >= 1.0:
        raise ComputePolicyArtifactError(
            f"{path}.local_noise_relative_fraction must be less than one."
        )
    return AutoSelectionPolicy(
        broad_calibration_enabled=_boolean(
            record["broad_calibration_enabled"], f"{path}.broad_calibration_enabled"
        ),
        evidence_scope=_identifier(record["evidence_scope"], f"{path}.evidence_scope"),
        non_local_lower_confidence_speedup=lower_speedup,
        non_local_minimum_saving_ms=_number(
            record["non_local_minimum_saving_ms"],
            f"{path}.non_local_minimum_saving_ms",
            minimum=0.0,
        ),
        local_noise_relative_fraction=relative_noise,
        local_noise_absolute_ms=_number(
            record["local_noise_absolute_ms"],
            f"{path}.local_noise_absolute_ms",
            minimum=0.0,
        ),
        tie_breaker=_identifier(record["tie_breaker"], f"{path}.tie_breaker"),
        pending_evidence=_identifier_tuple(
            record["pending_evidence"], f"{path}.pending_evidence", nonempty=True
        ),
    )


def _parse_benchmark(value: object) -> LocalBenchmarkPolicy:
    path = "$.policy.local_benchmark"
    record = _object(
        value,
        path,
        {
            "initial_warm_rounds",
            "adaptive_warm_rounds",
            "confidence_level",
            "confidence_method",
            "bootstrap_resamples",
            "bootstrap_seed",
            "outlier_policy",
            "round_order_policy",
            "adaptive_near_threshold_relative_fraction",
            "adaptive_near_threshold_absolute_ms",
            "adaptive_speedup_mad_fraction",
        },
    )
    initial = _integer(
        record["initial_warm_rounds"], f"{path}.initial_warm_rounds", minimum=1
    )
    adaptive = _integer_tuple(
        record["adaptive_warm_rounds"], f"{path}.adaptive_warm_rounds", minimum=1
    )
    if tuple(sorted(set(adaptive))) != adaptive or any(
        value <= initial for value in adaptive
    ):
        raise ComputePolicyArtifactError(
            f"{path}.adaptive_warm_rounds must be unique, increasing, and above "
            "the initial count."
        )
    confidence = _number(
        record["confidence_level"], f"{path}.confidence_level", minimum=0.0
    )
    if not 0.0 < confidence < 1.0:
        raise ComputePolicyArtifactError(
            f"{path}.confidence_level must be between zero and one."
        )
    for name in (
        "adaptive_near_threshold_relative_fraction",
        "adaptive_speedup_mad_fraction",
    ):
        fraction = _number(record[name], f"{path}.{name}", minimum=0.0)
        if fraction >= 1.0:
            raise ComputePolicyArtifactError(f"{path}.{name} must be less than one.")
    return LocalBenchmarkPolicy(
        initial_warm_rounds=initial,
        adaptive_warm_rounds=adaptive,
        confidence_level=confidence,
        confidence_method=_identifier(
            record["confidence_method"], f"{path}.confidence_method"
        ),
        bootstrap_resamples=_integer(
            record["bootstrap_resamples"], f"{path}.bootstrap_resamples", minimum=1
        ),
        bootstrap_seed=_integer(
            record["bootstrap_seed"], f"{path}.bootstrap_seed", minimum=0
        ),
        outlier_policy=_identifier(record["outlier_policy"], f"{path}.outlier_policy"),
        round_order_policy=_identifier(
            record["round_order_policy"], f"{path}.round_order_policy"
        ),
        adaptive_near_threshold_relative_fraction=float(
            record["adaptive_near_threshold_relative_fraction"]
        ),
        adaptive_near_threshold_absolute_ms=_number(
            record["adaptive_near_threshold_absolute_ms"],
            f"{path}.adaptive_near_threshold_absolute_ms",
            minimum=0.0,
        ),
        adaptive_speedup_mad_fraction=float(record["adaptive_speedup_mad_fraction"]),
    )


def _parse_platform(value: object) -> PlatformAdmissionPolicy:
    path = "$.policy.platform_admission"
    record = _object(
        value,
        path,
        {
            "operating_systems",
            "execution_modes",
            "python_implementation",
            "python_minor_versions",
            "python_abis",
            "numpy_versions",
            "scipy_versions",
            "scikit_image_versions",
            "cuda_major_versions",
            "cuda_runtime_versions",
            "driver_versions",
            "cupy_versions",
            "cupyx_versions",
            "runtime_probe_fingerprint_required",
            "driver_version_metadata_required",
            "nvidia_compute_capability_required",
            "nvidia_device_names",
            "nvidia_compute_capabilities",
            "cucim_versions",
            "cucim_environment_record_schema",
            "cucim_environment_record_schema_version",
            "cucim_environment_track",
            "cupy_distribution",
            "cucim_distribution",
            "cucim_artifact_sha256",
            "validated_environment_policy_ids",
            "linux_policy",
            "macos_policy",
            "public_advertisement_enabled",
        },
    )
    cucim_artifact_sha256 = _string(
        record["cucim_artifact_sha256"],
        f"{path}.cucim_artifact_sha256",
    )
    if not _SHA256_RE.fullmatch(cucim_artifact_sha256):
        raise ComputePolicyArtifactError(
            f"{path}.cucim_artifact_sha256 must be a lowercase SHA-256 digest."
        )
    return PlatformAdmissionPolicy(
        operating_systems=_string_tuple(
            record["operating_systems"], f"{path}.operating_systems", nonempty=True
        ),
        execution_modes=_identifier_tuple(
            record["execution_modes"], f"{path}.execution_modes", nonempty=True
        ),
        python_implementation=_string(
            record["python_implementation"], f"{path}.python_implementation"
        ),
        python_minor_versions=_string_tuple(
            record["python_minor_versions"],
            f"{path}.python_minor_versions",
            nonempty=True,
        ),
        python_abis=_identifier_tuple(
            record["python_abis"], f"{path}.python_abis", nonempty=True
        ),
        numpy_versions=_string_tuple(
            record["numpy_versions"], f"{path}.numpy_versions", nonempty=True
        ),
        scipy_versions=_string_tuple(
            record["scipy_versions"], f"{path}.scipy_versions", nonempty=True
        ),
        scikit_image_versions=_string_tuple(
            record["scikit_image_versions"],
            f"{path}.scikit_image_versions",
            nonempty=True,
        ),
        cuda_major_versions=_integer_tuple(
            record["cuda_major_versions"], f"{path}.cuda_major_versions", minimum=1
        ),
        cuda_runtime_versions=_string_tuple(
            record["cuda_runtime_versions"],
            f"{path}.cuda_runtime_versions",
            nonempty=True,
        ),
        driver_versions=_string_tuple(
            record["driver_versions"], f"{path}.driver_versions", nonempty=True
        ),
        cupy_versions=_string_tuple(
            record["cupy_versions"], f"{path}.cupy_versions", nonempty=True
        ),
        cupyx_versions=_string_tuple(
            record["cupyx_versions"], f"{path}.cupyx_versions", nonempty=True
        ),
        runtime_probe_fingerprint_required=_boolean(
            record["runtime_probe_fingerprint_required"],
            f"{path}.runtime_probe_fingerprint_required",
        ),
        driver_version_metadata_required=_boolean(
            record["driver_version_metadata_required"],
            f"{path}.driver_version_metadata_required",
        ),
        nvidia_compute_capability_required=_boolean(
            record["nvidia_compute_capability_required"],
            f"{path}.nvidia_compute_capability_required",
        ),
        nvidia_device_names=_string_tuple(
            record["nvidia_device_names"],
            f"{path}.nvidia_device_names",
            nonempty=True,
        ),
        nvidia_compute_capabilities=_string_tuple(
            record["nvidia_compute_capabilities"],
            f"{path}.nvidia_compute_capabilities",
            nonempty=True,
        ),
        cucim_versions=_string_tuple(
            record["cucim_versions"], f"{path}.cucim_versions", nonempty=True
        ),
        cucim_environment_record_schema=_identifier(
            record["cucim_environment_record_schema"],
            f"{path}.cucim_environment_record_schema",
        ),
        cucim_environment_record_schema_version=_integer(
            record["cucim_environment_record_schema_version"],
            f"{path}.cucim_environment_record_schema_version",
            minimum=1,
        ),
        cucim_environment_track=_identifier(
            record["cucim_environment_track"],
            f"{path}.cucim_environment_track",
        ),
        cupy_distribution=_identifier(
            record["cupy_distribution"], f"{path}.cupy_distribution"
        ),
        cucim_distribution=_identifier(
            record["cucim_distribution"], f"{path}.cucim_distribution"
        ),
        cucim_artifact_sha256=cucim_artifact_sha256,
        validated_environment_policy_ids=_identifier_tuple(
            record["validated_environment_policy_ids"],
            f"{path}.validated_environment_policy_ids",
            nonempty=True,
        ),
        linux_policy=_identifier(record["linux_policy"], f"{path}.linux_policy"),
        macos_policy=_identifier(record["macos_policy"], f"{path}.macos_policy"),
        public_advertisement_enabled=_boolean(
            record["public_advertisement_enabled"],
            f"{path}.public_advertisement_enabled",
        ),
    )


def _parse_operation(value: object, path: str) -> PackagedOperationPolicy:
    record = _object(
        value,
        path,
        {
            "operation_id",
            "implementation_id",
            "implementation_version",
            "runtime_id",
            "implementation_library_id",
            "admission_tier",
            "environment_policy_id",
            "parameter_policy_id",
            "workload_policy_id",
            "parity_policy_id",
            "memory_model_id",
            "supported_spatial_ndims",
            "supports_device_residency",
            "support_summary",
        },
    )
    return PackagedOperationPolicy(
        operation_id=_identifier(record["operation_id"], f"{path}.operation_id"),
        implementation_id=_identifier(
            record["implementation_id"], f"{path}.implementation_id"
        ),
        implementation_version=_string(
            record["implementation_version"], f"{path}.implementation_version"
        ),
        runtime_id=_identifier(record["runtime_id"], f"{path}.runtime_id"),
        implementation_library_id=_identifier(
            record["implementation_library_id"], f"{path}.implementation_library_id"
        ),
        admission_tier=_literal(
            record["admission_tier"],
            f"{path}.admission_tier",
            {
                "developer_hidden",
                "public_selective",
                "public_auto_candidate",
            },
        ),
        environment_policy_id=_identifier(
            record["environment_policy_id"], f"{path}.environment_policy_id"
        ),
        parameter_policy_id=_identifier(
            record["parameter_policy_id"], f"{path}.parameter_policy_id"
        ),
        workload_policy_id=_identifier(
            record["workload_policy_id"], f"{path}.workload_policy_id"
        ),
        parity_policy_id=_identifier(
            record["parity_policy_id"], f"{path}.parity_policy_id"
        ),
        memory_model_id=_identifier(
            record["memory_model_id"], f"{path}.memory_model_id"
        ),
        supported_spatial_ndims=_integer_tuple(
            record["supported_spatial_ndims"],
            f"{path}.supported_spatial_ndims",
            minimum=1,
        ),
        supports_device_residency=_boolean(
            record["supports_device_residency"], f"{path}.supports_device_residency"
        ),
        support_summary=_parse_support_summary(record["support_summary"], path),
    )


def _parse_support_summary(
    value: object, operation_path: str
) -> OperationSupportSummary:
    path = f"{operation_path}.support_summary"
    record = _object(
        value,
        path,
        {
            "authority",
            "public_dtypes",
            "spatial_semantics_id",
            "required_facts",
            "parameter_bounds",
            "explicit_cpu_regions",
        },
    )
    bounds_raw = _array(
        record["parameter_bounds"], f"{path}.parameter_bounds", nonempty=True
    )
    bounds = tuple(
        _parse_parameter_bound(item, f"{path}.parameter_bounds[{index}]")
        for index, item in enumerate(bounds_raw)
    )
    _require_unique(
        ((bound.parameter, bound.scope) for bound in bounds),
        f"{path}.parameter_bounds parameter/scope",
    )
    return OperationSupportSummary(
        authority=_literal(
            record["authority"],
            f"{path}.authority",
            {"audit-mirror-of-executable-policy-v1"},
        ),
        public_dtypes=_string_tuple(
            record["public_dtypes"], f"{path}.public_dtypes", nonempty=True
        ),
        spatial_semantics_id=_identifier(
            record["spatial_semantics_id"], f"{path}.spatial_semantics_id"
        ),
        required_facts=_identifier_tuple(
            record["required_facts"], f"{path}.required_facts"
        ),
        parameter_bounds=bounds,
        explicit_cpu_regions=_identifier_tuple(
            record["explicit_cpu_regions"],
            f"{path}.explicit_cpu_regions",
            nonempty=True,
        ),
    )


def _parse_parameter_bound(value: object, path: str) -> ParameterBound:
    record = _object(
        value,
        path,
        {"parameter", "minimum", "maximum", "scope", "canonicalization_policy_id"},
    )
    minimum = _number(record["minimum"], f"{path}.minimum")
    maximum = _number(record["maximum"], f"{path}.maximum")
    if minimum > maximum:
        raise ComputePolicyArtifactError(f"{path}.minimum must not exceed maximum.")
    return ParameterBound(
        parameter=_identifier(record["parameter"], f"{path}.parameter"),
        minimum=minimum,
        maximum=maximum,
        scope=_identifier(record["scope"], f"{path}.scope"),
        canonicalization_policy_id=_identifier(
            record["canonicalization_policy_id"], f"{path}.canonicalization_policy_id"
        ),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComputePolicyArtifactError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ComputePolicyArtifactError(
        f"Non-finite JSON constant {value!r} is forbidden."
    )


def _object(value: object, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComputePolicyArtifactError(f"{path} must be an object.")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ComputePolicyArtifactError(
            f"{path} has invalid fields; missing={missing}, extra={extra}."
        )
    return value


def _array(value: object, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise ComputePolicyArtifactError(f"{path} must be an array.")
    if nonempty and not value:
        raise ComputePolicyArtifactError(f"{path} must not be empty.")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ComputePolicyArtifactError(f"{path} must be a non-empty trimmed string.")
    return value


def _identifier(value: object, path: str) -> str:
    result = _string(value, path)
    if not _IDENTIFIER_RE.fullmatch(result):
        raise ComputePolicyArtifactError(
            f"{path} must be a lowercase policy identifier."
        )
    return result


def _literal(value: object, path: str, allowed: set[str]) -> str:
    result = _string(value, path)
    if result not in allowed:
        raise ComputePolicyArtifactError(
            f"{path} must be one of {sorted(allowed)!r}; got {result!r}."
        )
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ComputePolicyArtifactError(f"{path} must be a boolean.")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComputePolicyArtifactError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise ComputePolicyArtifactError(f"{path} must be at least {minimum}.")
    return value


def _number(value: object, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComputePolicyArtifactError(f"{path} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ComputePolicyArtifactError(f"{path} must be finite.")
    if minimum is not None and result < minimum:
        raise ComputePolicyArtifactError(f"{path} must be at least {minimum}.")
    return result


def _string_tuple(
    value: object, path: str, *, nonempty: bool = False
) -> tuple[str, ...]:
    items = _array(value, path, nonempty=nonempty)
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(items)
    )
    _require_unique(result, path)
    return result


def _identifier_tuple(
    value: object, path: str, *, nonempty: bool = False
) -> tuple[str, ...]:
    items = _array(value, path, nonempty=nonempty)
    result = tuple(
        _identifier(item, f"{path}[{index}]") for index, item in enumerate(items)
    )
    _require_unique(result, path)
    return result


def _integer_tuple(value: object, path: str, *, minimum: int) -> tuple[int, ...]:
    items = _array(value, path, nonempty=True)
    result = tuple(
        _integer(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(items)
    )
    _require_unique(result, path)
    return result


def _require_unique(values: object, path: str) -> None:
    materialized = tuple(values)  # type: ignore[arg-type]
    if len(materialized) != len(set(materialized)):
        raise ComputePolicyArtifactError(f"{path} must contain unique values.")


__all__ = [
    "AutoSelectionPolicy",
    "ComputePolicyArtifactError",
    "ComputePolicyDigestError",
    "ComputePolicyExposure",
    "LocalBenchmarkPolicy",
    "OperationSupportSummary",
    "PHASE1_POLICY_ID",
    "PHASE1_POLICY_RESOURCE",
    "PHASE1_POLICY_SHA256",
    "POLICY_SCHEMA_ID",
    "PackagedOperationPolicy",
    "ParameterBound",
    "Phase1ComputePolicyArtifact",
    "PlatformAdmissionPolicy",
    "canonical_artifact_digest",
    "load_phase1_compute_policy",
    "parse_compute_policy_artifact",
]
