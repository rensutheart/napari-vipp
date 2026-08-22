"""Qt-free compute intent, environment, decision, and provenance contracts.

The production contract distinguishes a user request (CPU, Auto, Prefer GPU,
or Custom) from an array runtime (NumPy, CuPy, or a future Metal runtime)
and from the implementation library used by one node (CPU, CuPyX, cuCIM,
...).  This module contains data and validation only: importing it must never
import an optional accelerator package or initialize a device.

The older :class:`ComputeBackend` helpers remain as a compatibility shell for
the original capability spike.  New execution code uses :class:`ComputeMode`,
:class:`ComputeRequest`, and per-node preferences.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256
from numbers import Integral
from types import MappingProxyType
from typing import Any


class ComputeMode(StrEnum):
    """Portable global compute intent."""

    CPU = "cpu"
    AUTO = "auto"
    PREFER_GPU = "prefer_gpu"
    CUSTOM = "custom"
    # Compatibility for unreleased development workflow/config snapshots.
    SELECTIVE = "custom"

    @classmethod
    def parse(cls, value: ComputeMode | str) -> ComputeMode:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        if normalized == "selective":
            normalized = cls.CUSTOM.value
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported compute mode {value!r}; expected one of {choices}."
            ) from exc


class FallbackPolicy(StrEnum):
    """Behavior when an explicitly required accelerator choice is unavailable."""

    VISIBLE = "visible"
    STRICT = "strict"

    @classmethod
    def parse(cls, value: FallbackPolicy | str) -> FallbackPolicy:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported fallback policy {value!r}; expected {choices}."
            ) from exc


class NodePreferenceKind(StrEnum):
    """Strength of one authored per-node choice."""

    AUTO = "auto"
    CPU = "cpu"
    BEST_GPU = "best_gpu"
    LIBRARY = "library"
    IMPLEMENTATION = "implementation"


_LEGACY_IMPLEMENTATION_PREFERENCE_ALIASES = {
    "cucim-measure-objects-basic-v1": "cupy-measure-objects-basic-v1",
    "cucim-measure-objects-intensity-basic-v1": (
        "cupy-measure-objects-intensity-basic-v1"
    ),
}


@dataclass(frozen=True, slots=True)
class NodeComputePreference:
    """Portable authored preference for one node.

    Library and implementation choices use stable identifiers rather than
    environment-specific objects.  Availability is decided during preflight.
    """

    kind: NodePreferenceKind | str = NodePreferenceKind.AUTO
    value: str = ""

    def __post_init__(self) -> None:
        try:
            kind = (
                self.kind
                if isinstance(self.kind, NodePreferenceKind)
                else NodePreferenceKind(str(self.kind).strip().lower())
            )
        except ValueError as exc:
            choices = ", ".join(member.value for member in NodePreferenceKind)
            raise ValueError(
                f"Unsupported node compute preference {self.kind!r}; "
                f"expected one of {choices}."
            ) from exc
        value = str(self.value).strip()
        if kind is NodePreferenceKind.IMPLEMENTATION:
            value = _LEGACY_IMPLEMENTATION_PREFERENCE_ALIASES.get(value, value)
        needs_value = kind in {
            NodePreferenceKind.LIBRARY,
            NodePreferenceKind.IMPLEMENTATION,
        }
        if needs_value and not value:
            raise ValueError(f"{kind.value} preference requires a stable value.")
        if not needs_value and value:
            raise ValueError(f"{kind.value} preference must not include a value.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)

    @classmethod
    def parse(
        cls,
        value: NodeComputePreference | str | Mapping[str, object],
    ) -> NodeComputePreference:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - {"kind", "value"}
            if unknown:
                names = ", ".join(sorted(map(str, unknown)))
                raise ValueError(f"Unknown node preference field(s): {names}.")
            return cls(value.get("kind", "auto"), str(value.get("value", "")))
        normalized = str(value).strip()
        if ":" in normalized:
            kind, selected = normalized.split(":", 1)
            return cls(kind, selected)
        return cls(normalized)

    def as_dict(self) -> dict[str, str]:
        payload = {"kind": self.kind.value}
        if self.value:
            payload["value"] = self.value
        return payload


@dataclass(frozen=True, slots=True)
class ComputeRequest:
    """Immutable JSON-safe compute intent captured for one run."""

    mode: ComputeMode | str = ComputeMode.CPU
    node_preferences: Mapping[str, NodeComputePreference | str | Mapping] = field(
        default_factory=dict
    )
    fallback_policy: FallbackPolicy | str = FallbackPolicy.VISIBLE
    runtime_id: str = ""
    device_id: str = ""
    precision_policy_id: str = "scientific-default-v1"
    workload_policy_id: str = "vipp-best-available-v1"
    accelerator_memory_cap_bytes: int | None = None
    accelerator_safety_reserve_bytes: int | None = None
    allow_experimental: bool = False

    def __post_init__(self) -> None:
        mode = ComputeMode.parse(self.mode)
        fallback = FallbackPolicy.parse(self.fallback_policy)
        if (
            mode is ComputeMode.PREFER_GPU
            and fallback is not FallbackPolicy.VISIBLE
        ):
            raise ValueError(
                "Prefer GPU requires visible CPU fallback so it can use GPU "
                "wherever possible and CPU everywhere else."
            )
        normalized: dict[str, NodeComputePreference] = {}
        if not isinstance(self.node_preferences, Mapping):
            raise TypeError("node_preferences must be a mapping keyed by node ID.")
        for raw_node_id, raw_preference in self.node_preferences.items():
            node_id = str(raw_node_id).strip()
            if not node_id:
                raise ValueError("node preference IDs must not be empty.")
            normalized[node_id] = NodeComputePreference.parse(raw_preference)
        for field_name in (
            "accelerator_memory_cap_bytes",
            "accelerator_safety_reserve_bytes",
        ):
            raw_value = getattr(self, field_name)
            if raw_value is not None and (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise ValueError(
                    f"{field_name} must be a non-negative integer or None."
                )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "fallback_policy", fallback)
        object.__setattr__(
            self,
            "node_preferences",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        for field_name in ("runtime_id", "device_id"):
            object.__setattr__(self, field_name, str(getattr(self, field_name)).strip())
        for field_name in ("precision_policy_id", "workload_policy_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty.")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.allow_experimental, bool):
            raise TypeError("allow_experimental must be a boolean.")

    def preference_for(self, node_id: str) -> NodeComputePreference:
        return self.node_preferences.get(str(node_id), NodeComputePreference())

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "node_preferences": {
                node_id: preference.as_dict()
                for node_id, preference in self.node_preferences.items()
            },
            "fallback_policy": self.fallback_policy.value,
            "runtime_id": self.runtime_id,
            "device_id": self.device_id,
            "precision_policy_id": self.precision_policy_id,
            "workload_policy_id": self.workload_policy_id,
            "accelerator_memory_cap_bytes": self.accelerator_memory_cap_bytes,
            "accelerator_safety_reserve_bytes": (self.accelerator_safety_reserve_bytes),
            "allow_experimental": bool(self.allow_experimental),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ComputeRequest:
        if not isinstance(payload, Mapping):
            raise TypeError("compute request must be an object.")
        fields = {
            "mode",
            "node_preferences",
            "fallback_policy",
            "runtime_id",
            "device_id",
            "precision_policy_id",
            "workload_policy_id",
            "accelerator_memory_cap_bytes",
            "accelerator_safety_reserve_bytes",
            "allow_experimental",
        }
        unknown = set(payload) - fields
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"Unknown compute request field(s): {names}.")
        return cls(**dict(payload))

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.as_dict())


class MemoryTopology(StrEnum):
    HOST = "host"
    DISCRETE = "discrete"
    UNIFIED = "unified"


_SCIENTIFIC_STACK_DISTRIBUTIONS = ("numpy", "scipy", "scikit-image")


def _installed_scientific_stack_versions() -> tuple[tuple[str, str], ...]:
    """Read CPU-reference versions without importing scientific/GPU packages."""

    versions: list[tuple[str, str]] = []
    for distribution in _SCIENTIFIC_STACK_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        versions.append((distribution, version))
    return tuple(versions)


@dataclass(frozen=True, slots=True)
class ComputeEnvironment:
    """One immutable capability/provenance snapshot."""

    os_name: str = field(default_factory=platform.system)
    os_release: str = field(default_factory=platform.release)
    execution_mode: str = "native"
    python_implementation: str = field(default_factory=platform.python_implementation)
    python_version: str = field(
        default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    python_abi: str = field(default_factory=lambda: sys.implementation.cache_tag or "")
    runtime_ids: tuple[str, ...] = ("cpu-numpy",)
    implementation_libraries: tuple[str, ...] = ("cpu",)
    runtime_versions: tuple[tuple[str, str], ...] = ()
    scientific_stack_versions: tuple[tuple[str, str], ...] = field(
        default_factory=_installed_scientific_stack_versions
    )
    runtime_probe_fingerprints: tuple[tuple[str, str], ...] = ()
    runtime_metadata: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    implementation_library_metadata: tuple[
        tuple[str, tuple[tuple[str, str], ...]], ...
    ] = ()
    driver_version: str = ""
    device_id: str = "cpu:0"
    device_name: str = "Host CPU"
    device_class: str = "host"
    device_metadata: tuple[tuple[str, str], ...] = ()
    memory_topology: MemoryTopology | str = MemoryTopology.HOST
    total_accelerator_memory_bytes: int = 0
    probe_status: str = "available"
    probe_reason: str = ""

    def __post_init__(self) -> None:
        topology = (
            self.memory_topology
            if isinstance(self.memory_topology, MemoryTopology)
            else MemoryTopology(str(self.memory_topology).strip().lower())
        )
        versions = _normalized_unique_pairs(
            self.runtime_versions,
            "runtime_versions",
            strip_values=True,
        )
        scientific_stack_versions = _normalized_unique_pairs(
            self.scientific_stack_versions,
            "scientific_stack_versions",
            strip_values=True,
        )
        runtime_fingerprints = _normalized_unique_pairs(
            self.runtime_probe_fingerprints,
            "runtime_probe_fingerprints",
            strip_values=True,
        )
        runtime_metadata = _normalized_scoped_metadata(
            self.runtime_metadata,
            "runtime_metadata",
        )
        library_metadata = _normalized_scoped_metadata(
            self.implementation_library_metadata,
            "implementation_library_metadata",
        )
        device_metadata = _normalized_unique_pairs(
            self.device_metadata,
            "device_metadata",
        )
        if (
            isinstance(self.total_accelerator_memory_bytes, bool)
            or not isinstance(self.total_accelerator_memory_bytes, int)
            or self.total_accelerator_memory_bytes < 0
        ):
            raise ValueError(
                "total_accelerator_memory_bytes must be a non-negative integer."
            )
        object.__setattr__(self, "memory_topology", topology)
        object.__setattr__(
            self, "runtime_ids", _normalized_operation_ids(self.runtime_ids)
        )
        object.__setattr__(
            self,
            "implementation_libraries",
            _normalized_operation_ids(self.implementation_libraries),
        )
        object.__setattr__(self, "runtime_versions", versions)
        object.__setattr__(
            self,
            "scientific_stack_versions",
            scientific_stack_versions,
        )
        object.__setattr__(
            self,
            "runtime_probe_fingerprints",
            runtime_fingerprints,
        )
        object.__setattr__(self, "runtime_metadata", runtime_metadata)
        object.__setattr__(
            self,
            "implementation_library_metadata",
            library_metadata,
        )
        object.__setattr__(self, "device_metadata", device_metadata)
        object.__setattr__(self, "driver_version", str(self.driver_version).strip())

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["memory_topology"] = self.memory_topology.value
        payload["runtime_ids"] = list(self.runtime_ids)
        payload["implementation_libraries"] = list(self.implementation_libraries)
        payload["runtime_versions"] = dict(self.runtime_versions)
        payload["scientific_stack_versions"] = dict(
            self.scientific_stack_versions
        )
        payload["runtime_probe_fingerprints"] = dict(self.runtime_probe_fingerprints)
        payload["runtime_metadata"] = {
            scope: dict(metadata) for scope, metadata in self.runtime_metadata
        }
        payload["implementation_library_metadata"] = {
            scope: dict(metadata)
            for scope, metadata in self.implementation_library_metadata
        }
        payload["device_metadata"] = dict(self.device_metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ComputeEnvironment:
        if not isinstance(payload, Mapping):
            raise TypeError("compute environment must be an object.")
        values = dict(payload)
        versions = values.get("runtime_versions", ())
        if isinstance(versions, Mapping):
            values["runtime_versions"] = tuple(versions.items())
        scientific_stack_versions = values.get("scientific_stack_versions", ())
        if isinstance(scientific_stack_versions, Mapping):
            values["scientific_stack_versions"] = tuple(
                scientific_stack_versions.items()
            )
        fingerprints = values.get("runtime_probe_fingerprints", ())
        if isinstance(fingerprints, Mapping):
            values["runtime_probe_fingerprints"] = tuple(fingerprints.items())
        for field_name in (
            "runtime_metadata",
            "implementation_library_metadata",
        ):
            raw_scopes = values.get(field_name, ())
            if isinstance(raw_scopes, Mapping):
                values[field_name] = tuple(
                    (
                        scope,
                        tuple(metadata.items())
                        if isinstance(metadata, Mapping)
                        else metadata,
                    )
                    for scope, metadata in raw_scopes.items()
                )
        device_metadata = values.get("device_metadata", ())
        if isinstance(device_metadata, Mapping):
            values["device_metadata"] = tuple(device_metadata.items())
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(self.as_dict())


class DecisionReason(StrEnum):
    """Why a supported implementation was or was not selected."""

    EXPLICIT_CPU = "explicit_cpu"
    AUTO_CPU = "auto_cpu"
    SELECTED_IMPLEMENTATION = "selected_implementation"
    NO_VALIDATED_IMPLEMENTATION = "no_validated_implementation"
    ENVIRONMENT_UNSUPPORTED = "environment_unsupported"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    WORKLOAD_UNSUPPORTED = "workload_unsupported"
    PERFORMANCE_GATE = "performance_gate"
    PERFORMANCE_EXPLORATION = "performance_exploration"
    HISTORICAL_PERFORMANCE = "historical_performance"
    MEMORY_LIMIT = "memory_limit"
    VISIBLE_FALLBACK = "visible_fallback"
    OUT_OF_MEMORY_FALLBACK = "out_of_memory_fallback"


class DecisionKind(StrEnum):
    """High-level result of planning one node."""

    POLICY_CPU = "policy_cpu"
    SELECTED = "selected"
    FALLBACK_CPU = "fallback_cpu"


class ComputeRepairAction(StrEnum):
    """Stable graph-edit action offered by compute planning."""

    INSERT_CONVERT_DTYPE = "insert_convert_dtype"


class FallbackReason(StrEnum):
    """Typed cause when and only when CPU fallback occurred."""

    NONE = "none"
    ENVIRONMENT_UNSUPPORTED = "environment_unsupported"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    WORKLOAD_UNSUPPORTED = "workload_unsupported"
    MEMORY_LIMIT = "memory_limit"
    OUT_OF_MEMORY = "out_of_memory"
    RUNTIME_FAILURE = "runtime_failure"


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Incremental peak bytes for one candidate segment."""

    runtime_managed_peak_bytes: int = 0
    total_device_peak_bytes: int = 0
    host_materialization_peak_bytes: int = 0
    uncertainty_bytes: int = 0
    model_id: str = "none"

    def __post_init__(self) -> None:
        for name in (
            "runtime_managed_peak_bytes",
            "total_device_peak_bytes",
            "host_materialization_peak_bytes",
            "uncertainty_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class ComputeRepairCandidate:
    """Stable identity of the candidate proven by a counterfactual repair."""

    implementation_id: str
    implementation_version: str
    runtime_id: str
    implementation_library_id: str

    def __post_init__(self) -> None:
        for name in (
            "implementation_id",
            "implementation_version",
            "runtime_id",
            "implementation_library_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, str]:
        return {
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "runtime_id": self.runtime_id,
            "implementation_library_id": self.implementation_library_id,
        }


@dataclass(frozen=True, slots=True)
class ComputeRepairSuggestion:
    """One transparent graph edit that can remove a proven compute blocker."""

    action: ComputeRepairAction | str
    node_id: str
    operation_id: str
    input_port_index: int
    input_port_name: str
    current_dtype: str
    target_dtype: str
    scaling: str
    exact: bool
    message: str
    candidate: ComputeRepairCandidate

    def __post_init__(self) -> None:
        action = (
            self.action
            if isinstance(self.action, ComputeRepairAction)
            else ComputeRepairAction(str(self.action).strip().lower())
        )
        for name in (
            "node_id",
            "operation_id",
            "input_port_name",
            "current_dtype",
            "target_dtype",
            "scaling",
            "message",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.input_port_index, bool)
            or not isinstance(self.input_port_index, int)
            or self.input_port_index < 0
        ):
            raise ValueError("input_port_index must be a non-negative integer.")
        if not isinstance(self.exact, bool):
            raise TypeError("exact must be a boolean.")
        if not isinstance(self.candidate, ComputeRepairCandidate):
            raise TypeError("candidate must be a ComputeRepairCandidate.")
        if action is ComputeRepairAction.INSERT_CONVERT_DTYPE:
            if self.current_dtype not in {"uint8", "uint16"}:
                raise ValueError(
                    "insert_convert_dtype currently requires a uint8 or uint16 "
                    "source dtype."
                )
            if self.target_dtype != "float32":
                raise ValueError(
                    "insert_convert_dtype currently requires a float32 target."
                )
            if self.scaling.casefold() != "preserve" or not self.exact:
                raise ValueError(
                    "insert_convert_dtype must be an exact Preserve conversion."
                )
        object.__setattr__(self, "action", action)

    @property
    def conversion_parameters(self) -> dict[str, str]:
        """Parameters for the visible Convert Dtype node to insert."""

        return {"output_dtype": self.target_dtype, "scaling": self.scaling}

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "node_id": self.node_id,
            "operation_id": self.operation_id,
            "input_port_index": self.input_port_index,
            "input_port_name": self.input_port_name,
            "current_dtype": self.current_dtype,
            "target_dtype": self.target_dtype,
            "scaling": self.scaling,
            "exact": self.exact,
            "message": self.message,
            "candidate": self.candidate.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class WorkloadDescriptor:
    """Canonical policy/benchmark description for one prepared node."""

    node_id: str
    operation_id: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    parameters: tuple[tuple[str, object], ...] = ()
    resolved_spatial_ndim: int | None = None
    resident_predecessors: tuple[str, ...] = ()
    resident_successors: tuple[str, ...] = ()
    required_host_boundaries: int = 0
    facts_fingerprint: str = ""
    inputs_resolved: bool = True

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        operation_id = str(self.operation_id).strip()
        if not node_id or not operation_id:
            raise ValueError("node_id and operation_id must not be empty.")
        shapes: list[tuple[int, ...]] = []
        for shape in self.input_shapes:
            normalized_shape: list[int] = []
            for size in shape:
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError("input shapes require non-negative integers.")
                normalized_shape.append(size)
            shapes.append(tuple(normalized_shape))
        dtypes = tuple(str(dtype).strip() for dtype in self.input_dtypes)
        if len(dtypes) != len(shapes) or any(not dtype for dtype in dtypes):
            raise ValueError("input_dtypes must name every input shape.")
        parameters = tuple(
            sorted(
                (str(name).strip(), _freeze_json_value(value))
                for name, value in self.parameters
            )
        )
        if any(not name for name, _value in parameters):
            raise ValueError("parameter names must not be empty.")
        if self.resolved_spatial_ndim not in {None, 1, 2, 3}:
            raise ValueError("resolved_spatial_ndim must be None or 1, 2, or 3.")
        if (
            isinstance(self.required_host_boundaries, bool)
            or not isinstance(self.required_host_boundaries, int)
            or self.required_host_boundaries < 0
        ):
            raise ValueError("required_host_boundaries must be non-negative.")
        if not isinstance(self.inputs_resolved, bool):
            raise TypeError("inputs_resolved must be a boolean.")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "operation_id", operation_id)
        object.__setattr__(self, "input_shapes", tuple(shapes))
        object.__setattr__(self, "input_dtypes", dtypes)
        object.__setattr__(self, "parameters", parameters)

    @property
    def fingerprint(self) -> str:
        return canonical_digest(asdict(self))


def exact_workload_identity_digest(workload: WorkloadDescriptor) -> str:
    """Return the planner-compatible scientific identity of one node workload.

    Benchmark descriptors bind their ``facts_fingerprint`` to every input byte,
    while execution descriptors intentionally bind that field only to the facts
    needed for support policy.  Those fingerprints cannot be compared directly.
    This narrower digest covers the fields both paths construct identically;
    exact source bytes and graph identity remain bound by the qualification's
    caller-authoritative scope digest.
    """

    if not isinstance(workload, WorkloadDescriptor):
        raise TypeError("workload must be a WorkloadDescriptor.")
    return canonical_digest(
        {
            "policy": "exact-workload-planner-identity-v1",
            "node_id": workload.node_id,
            "operation_id": workload.operation_id,
            "input_shapes": workload.input_shapes,
            "input_dtypes": workload.input_dtypes,
            "parameters": workload.parameters,
            "resolved_spatial_ndim": workload.resolved_spatial_ndim,
            "inputs_resolved": workload.inputs_resolved,
        }
    )


@dataclass(frozen=True, slots=True)
class ExactWorkloadCandidateQualification:
    """Private proof that one soft-prequalified candidate passed node parity.

    The record is deliberately more specific than a node/implementation pair.
    Planning must match its exact parameters and input geometry, implementation
    version, parity policy, compute environment, and caller-owned optimizer
    identity scope.  The benchmark workload and record digests retain the exact
    byte-bound evidence identity for provenance and stale-result invalidation.
    """

    node_id: str
    operation_id: str
    implementation_id: str
    implementation_version: str
    workload_identity_digest: str
    benchmark_workload_fingerprint: str
    compute_environment_fingerprint: str
    benchmark_environment_fingerprint: str
    parity_policy_id: str
    benchmark_record_digest: str
    qualification_scope_digest: str
    identity_policy_id: str = "exact-workload-candidate-qualification-v1"

    def __post_init__(self) -> None:
        for name in (
            "node_id",
            "operation_id",
            "implementation_id",
            "implementation_version",
            "workload_identity_digest",
            "benchmark_workload_fingerprint",
            "compute_environment_fingerprint",
            "benchmark_environment_fingerprint",
            "parity_policy_id",
            "benchmark_record_digest",
            "qualification_scope_digest",
            "identity_policy_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

    @property
    def candidate_key(self) -> tuple[str, str]:
        return self.node_id, self.implementation_id


@dataclass(frozen=True, slots=True)
class BenchmarkRecordKey:
    workload_fingerprint: str
    environment_fingerprint: str
    implementation_ids: tuple[str, ...]
    policy_id: str
    device_id: str = ""
    memory_limit_bytes: int | None = None
    safety_reserve_bytes: int | None = None

    def __post_init__(self) -> None:
        device_id = str(self.device_id).strip()
        memory_limit = self.memory_limit_bytes
        safety_reserve = self.safety_reserve_bytes
        if memory_limit is not None and (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or memory_limit <= 0
        ):
            raise ValueError("memory_limit_bytes must be positive or None.")
        if safety_reserve is not None and (
            isinstance(safety_reserve, bool)
            or not isinstance(safety_reserve, int)
            or safety_reserve < 0
        ):
            raise ValueError("safety_reserve_bytes must be non-negative or None.")
        object.__setattr__(self, "device_id", device_id)

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


class BenchmarkCandidateFailureKind(StrEnum):
    """Typed provenance for a benchmark candidate that was not qualified."""

    NONE = ""
    SCIENTIFIC_PARITY = "scientific-parity"
    TRANSIENT_RUNTIME = "transient-runtime"


@dataclass(frozen=True, slots=True)
class BenchmarkCandidateResult:
    implementation_id: str
    parity_passed: bool
    cold_seconds: float | None
    warm_seconds: tuple[float, ...]
    peak_memory_bytes: int = 0
    error: str = ""
    timing_scope: str = "implementation-only"
    synchronized: bool = False
    transfers_included: bool = False
    cold_transfer_seconds: float | None = None
    warm_transfer_seconds: tuple[float, ...] = ()
    cold_resident_seconds: float | None = None
    warm_resident_seconds: tuple[float, ...] = ()
    cold_host_materialization_seconds: float | None = None
    warm_host_materialization_seconds: tuple[float, ...] = ()
    peak_runtime_live_bytes: int = 0
    peak_runtime_reserved_bytes: int = 0
    peak_out_of_pool_bytes: int = 0
    paired_speedup_median: float | None = None
    paired_speedup_lower_confidence_bound: float | None = None
    paired_bootstrap_samples: int = 0
    paired_bootstrap_seed: int = 0
    failure_kind: BenchmarkCandidateFailureKind | str = (
        BenchmarkCandidateFailureKind.NONE
    )
    timing_censored: bool = False
    timing_lower_bound_seconds: float | None = None
    timing_censor_reason: str = ""
    timing_censor_incumbent_id: str = ""

    def __post_init__(self) -> None:
        implementation_id = str(self.implementation_id).strip()
        if not implementation_id:
            raise ValueError("implementation_id must not be empty.")
        times = (
            (() if self.cold_seconds is None else (self.cold_seconds,))
            + tuple(self.warm_seconds)
            + (
                ()
                if self.cold_transfer_seconds is None
                else (self.cold_transfer_seconds,)
            )
            + tuple(self.warm_transfer_seconds)
            + (
                ()
                if self.cold_resident_seconds is None
                else (self.cold_resident_seconds,)
            )
            + tuple(self.warm_resident_seconds)
            + (
                ()
                if self.cold_host_materialization_seconds is None
                else (self.cold_host_materialization_seconds,)
            )
            + tuple(self.warm_host_materialization_seconds)
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in times
        ):
            raise ValueError("benchmark times must be finite and non-negative.")
        for name in (
            "peak_memory_bytes",
            "peak_runtime_live_bytes",
            "peak_runtime_reserved_bytes",
            "peak_out_of_pool_bytes",
            "paired_bootstrap_samples",
            "paired_bootstrap_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in (
            "paired_speedup_median",
            "paired_speedup_lower_confidence_bound",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative or None.")
        if not isinstance(self.timing_censored, bool):
            raise TypeError("timing_censored must be a boolean.")
        lower_bound = self.timing_lower_bound_seconds
        if lower_bound is not None and (
            isinstance(lower_bound, bool)
            or not isinstance(lower_bound, (int, float))
            or not math.isfinite(float(lower_bound))
            or lower_bound <= 0
        ):
            raise ValueError(
                "timing_lower_bound_seconds must be finite and positive or None."
            )
        censor_reason = str(self.timing_censor_reason).strip()
        censor_incumbent = str(self.timing_censor_incumbent_id).strip()
        if self.timing_censored:
            if (
                lower_bound is None
                or not censor_reason
                or not censor_incumbent
                or not self.parity_passed
                or str(self.error).strip()
            ):
                raise ValueError(
                    "censored timing evidence requires passed parity, a positive "
                    "lower bound, an incumbent ID, a reason, and no runtime error."
                )
        elif lower_bound is not None or censor_reason or censor_incumbent:
            raise ValueError(
                "timing censor metadata requires timing_censored=True."
            )
        timing_scope = str(self.timing_scope).strip()
        if not timing_scope:
            raise ValueError("timing_scope must not be empty.")
        failure_kind = (
            self.failure_kind
            if isinstance(self.failure_kind, BenchmarkCandidateFailureKind)
            else BenchmarkCandidateFailureKind(str(self.failure_kind).strip())
        )
        if (
            failure_kind is BenchmarkCandidateFailureKind.SCIENTIFIC_PARITY
            and (self.parity_passed or not str(self.error).strip())
        ):
            raise ValueError(
                "scientific-parity failure provenance requires a failed parity "
                "result with an error detail."
            )
        if (
            failure_kind is BenchmarkCandidateFailureKind.TRANSIENT_RUNTIME
            and not str(self.error).strip()
        ):
            raise ValueError(
                "transient-runtime failure provenance requires an error detail."
            )
        if not isinstance(self.synchronized, bool):
            raise TypeError("synchronized must be a boolean.")
        if not isinstance(self.transfers_included, bool):
            raise TypeError("transfers_included must be a boolean.")
        for name in (
            "warm_transfer_seconds",
            "warm_resident_seconds",
            "warm_host_materialization_seconds",
        ):
            values = tuple(getattr(self, name))
            if values and len(values) != len(self.warm_seconds):
                raise ValueError(f"{name} must be empty or describe every warm round.")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        object.__setattr__(self, "implementation_id", implementation_id)
        object.__setattr__(self, "timing_scope", timing_scope)
        object.__setattr__(self, "failure_kind", failure_kind)
        object.__setattr__(self, "timing_censor_reason", censor_reason)
        object.__setattr__(
            self,
            "timing_censor_incumbent_id",
            censor_incumbent,
        )
        if lower_bound is not None:
            object.__setattr__(
                self,
                "timing_lower_bound_seconds",
                float(lower_bound),
            )
        object.__setattr__(
            self, "warm_seconds", tuple(float(value) for value in self.warm_seconds)
        )
        if self.cold_seconds is not None:
            object.__setattr__(self, "cold_seconds", float(self.cold_seconds))
        for name in (
            "cold_transfer_seconds",
            "cold_resident_seconds",
            "cold_host_materialization_seconds",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        for name in (
            "paired_speedup_median",
            "paired_speedup_lower_confidence_bound",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    key: BenchmarkRecordKey
    candidates: tuple[BenchmarkCandidateResult, ...]
    created_utc: str
    benchmark_policy_id: str
    accepted_implementation_id: str = ""
    paired_confidence_level: float | None = None
    paired_bootstrap_samples: int = 0
    paired_bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if self.paired_confidence_level is not None and (
            isinstance(self.paired_confidence_level, bool)
            or not isinstance(self.paired_confidence_level, (int, float))
            or not 0 < float(self.paired_confidence_level) < 1
        ):
            raise ValueError("paired_confidence_level must be between zero and one.")
        for name in ("paired_bootstrap_samples", "paired_bootstrap_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.paired_confidence_level is not None:
            object.__setattr__(
                self,
                "paired_confidence_level",
                float(self.paired_confidence_level),
            )


@dataclass(frozen=True, slots=True)
class NodeExecutionDecision:
    node_id: str
    operation_id: str
    requested_preference: NodeComputePreference
    runtime_id: str
    implementation_library_id: str
    implementation_id: str
    decision_kind: DecisionKind | str
    reason: DecisionReason | str
    reason_text: str
    fallback_reason: FallbackReason | str = FallbackReason.NONE
    benchmark_record_digest: str = ""
    memory_estimate: MemoryEstimate = MemoryEstimate()
    performance_evidence_kind: str = ""
    performance_evidence_digest: str = ""
    implementation_version: str = ""

    def __post_init__(self) -> None:
        reason = (
            self.reason
            if isinstance(self.reason, DecisionReason)
            else DecisionReason(str(self.reason).strip().lower())
        )
        decision_kind = (
            self.decision_kind
            if isinstance(self.decision_kind, DecisionKind)
            else DecisionKind(str(self.decision_kind).strip().lower())
        )
        fallback_reason = (
            self.fallback_reason
            if isinstance(self.fallback_reason, FallbackReason)
            else FallbackReason(str(self.fallback_reason).strip().lower())
        )
        if decision_kind is DecisionKind.FALLBACK_CPU:
            if fallback_reason is FallbackReason.NONE:
                raise ValueError("fallback_cpu requires a typed fallback_reason.")
        elif fallback_reason is not FallbackReason.NONE:
            raise ValueError(
                "fallback_reason is valid only for a fallback_cpu decision."
            )
        object.__setattr__(self, "decision_kind", decision_kind)
        object.__setattr__(self, "reason", reason)
        if not isinstance(self.memory_estimate, MemoryEstimate):
            raise TypeError("memory_estimate must be a MemoryEstimate.")
        for name in (
            "performance_evidence_kind",
            "performance_evidence_digest",
            "implementation_version",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if bool(self.performance_evidence_kind) != bool(
            self.performance_evidence_digest
        ):
            raise ValueError(
                "performance evidence kind and digest must be supplied together."
            )
        object.__setattr__(self, "fallback_reason", fallback_reason)

    @property
    def fallback_used(self) -> bool:
        return self.decision_kind is DecisionKind.FALLBACK_CPU


@dataclass(frozen=True, slots=True)
class OutputPortKey:
    """Stable identity for one node output used by liveness and caching."""

    node_id: str
    port_index: int = 0

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("node_id must not be empty.")
        if (
            isinstance(self.port_index, bool)
            or not isinstance(self.port_index, int)
            or self.port_index < 0
        ):
            raise ValueError("port_index must be a non-negative integer.")
        object.__setattr__(self, "node_id", node_id)


@dataclass(frozen=True, slots=True)
class ExecutionSegment:
    segment_id: str
    runtime_id: str
    node_ids: tuple[str, ...]
    entry_ports: tuple[OutputPortKey, ...] = ()
    exit_ports: tuple[OutputPortKey, ...] = ()
    retained_ports: tuple[OutputPortKey, ...] = ()
    remaining_consumers: tuple[tuple[OutputPortKey, int], ...] = ()
    memory_estimate: MemoryEstimate = MemoryEstimate()

    def __post_init__(self) -> None:
        for _port, count in self.remaining_consumers:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("remaining consumer counts must be non-negative.")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    request_fingerprint: str
    environment_fingerprint: str
    segments: tuple[ExecutionSegment, ...]
    decisions: tuple[NodeExecutionDecision, ...]
    warnings: tuple[str, ...] = ()
    repair_suggestions: tuple[ComputeRepairSuggestion, ...] = ()

    def __post_init__(self) -> None:
        suggestions = tuple(self.repair_suggestions)
        if any(
            not isinstance(suggestion, ComputeRepairSuggestion)
            for suggestion in suggestions
        ):
            raise TypeError(
                "repair_suggestions must contain ComputeRepairSuggestion values."
            )
        object.__setattr__(self, "repair_suggestions", suggestions)


@dataclass(frozen=True, slots=True)
class ExecutionFallbackRecord:
    """One classified device failure followed by an allowed CPU retry."""

    segment_id: str
    runtime_id: str
    node_ids: tuple[str, ...]
    reason: FallbackReason | str
    reason_code: str
    exception_type: str = ""
    message: str = ""
    retryable: bool = True
    device_attempt_count: int = 1
    cpu_retry_count: int = 1
    cpu_retry_succeeded: bool | None = None
    cleanup_succeeded: bool = True
    memory_estimate: MemoryEstimate = MemoryEstimate()
    memory_topology: MemoryTopology | str | None = None
    device_total_bytes: int | None = None
    device_free_bytes: int | None = None
    runtime_live_bytes: int = 0
    runtime_reserved_bytes: int = 0
    out_of_pool_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("segment_id", "runtime_id", "reason_code"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        node_ids = tuple(str(node_id).strip() for node_id in self.node_ids)
        if not node_ids or any(not node_id for node_id in node_ids):
            raise ValueError("node_ids must contain at least one non-empty node ID.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must not contain duplicates.")
        object.__setattr__(self, "node_ids", node_ids)
        reason = (
            self.reason
            if isinstance(self.reason, FallbackReason)
            else FallbackReason(str(self.reason).strip().lower())
        )
        if reason is FallbackReason.NONE:
            raise ValueError("A fallback record requires a typed fallback reason.")
        object.__setattr__(self, "reason", reason)
        topology = self.memory_topology
        if topology is not None and not isinstance(topology, MemoryTopology):
            topology = MemoryTopology(str(topology).strip().lower())
        object.__setattr__(self, "memory_topology", topology)
        for name in ("exception_type", "message"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        for name in ("retryable", "cleanup_succeeded"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.cpu_retry_succeeded is not None and not isinstance(
            self.cpu_retry_succeeded,
            bool,
        ):
            raise TypeError("cpu_retry_succeeded must be a boolean or None.")
        for name in ("device_attempt_count", "cpu_retry_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "device_total_bytes",
            "device_free_bytes",
            "runtime_live_bytes",
            "runtime_reserved_bytes",
            "out_of_pool_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")

    def as_dict(self) -> dict[str, object]:
        return _json_safe(self)


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    request: ComputeRequest
    environment: ComputeEnvironment
    plan: ExecutionPlan | None = None
    actual_decisions: tuple[NodeExecutionDecision, ...] = ()
    fallback_records: tuple[ExecutionFallbackRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    cleanup_succeeded: bool = True

    def __post_init__(self) -> None:
        records = tuple(self.fallback_records)
        if any(not isinstance(record, ExecutionFallbackRecord) for record in records):
            raise TypeError(
                "fallback_records must contain ExecutionFallbackRecord values."
            )
        object.__setattr__(self, "fallback_records", records)

    def as_dict(self) -> dict[str, object]:
        return _json_safe(self)


@dataclass(frozen=True, slots=True)
class ScientificResultKey:
    operation_id: str
    output_port_index: int
    output_contract_id: str
    parameter_fingerprint: str
    upstream_fingerprints: tuple[str, ...]
    implementation_id: str
    implementation_version: str
    dependency_fingerprint: str
    result_contract_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.output_port_index, bool)
            or not isinstance(self.output_port_index, int)
            or self.output_port_index < 0
        ):
            raise ValueError("output_port_index must be a non-negative integer.")
        for name in (
            "operation_id",
            "output_contract_id",
            "parameter_fingerprint",
            "implementation_id",
            "implementation_version",
            "dependency_fingerprint",
            "result_contract_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class CacheAdmissibility:
    admissible: bool
    reason: str
    required_implementation_id: str = ""


def canonical_digest(value: object) -> str:
    """Return a deterministic digest for a JSON-safe contract payload."""

    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _freeze_json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract values must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        items = []
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("contract mapping keys must not be empty.")
            items.append((key, _freeze_json_value(item)))
        return tuple(sorted(items))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(
        f"contract values must be JSON-safe primitives, not {type(value).__name__}."
    )


def _json_safe(value: object) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_safe(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


class ComputeBackend(StrEnum):
    """User-facing compute-backend request."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"

    @classmethod
    def parse(cls, value: ComputeBackend | str) -> ComputeBackend:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported compute backend {value!r}; expected one of {choices}."
            ) from exc


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """Availability and operation coverage for one compute backend."""

    backend: ComputeBackend
    available: bool
    provider: str
    version: str
    device_name: str
    supported_operation_ids: tuple[str, ...]
    reason: str = ""

    def supports(self, operation_id: str) -> bool:
        return self.available and str(operation_id) in self.supported_operation_ids


@dataclass(frozen=True, slots=True)
class ComputeCapabilityReport:
    """CPU/GPU capabilities without importing Qt or changing runtime state."""

    cpu: BackendCapability
    gpu: BackendCapability

    def for_backend(self, backend: ComputeBackend | str) -> BackendCapability:
        parsed = ComputeBackend.parse(backend)
        if parsed is ComputeBackend.AUTO:
            raise ValueError("Auto is a selection policy, not a concrete capability.")
        return self.cpu if parsed is ComputeBackend.CPU else self.gpu

    def as_dict(self) -> dict[str, object]:
        return {
            "cpu": _capability_dict(self.cpu),
            "gpu": _capability_dict(self.gpu),
        }


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """Resolved backend and the reason for that decision."""

    requested: ComputeBackend
    resolved: ComputeBackend
    operation_id: str
    fell_back: bool
    reason: str


class ComputeBackendUnavailable(RuntimeError):
    """Raised when an explicit GPU request cannot be fulfilled safely."""


def detect_compute_capabilities(
    *,
    supported_gpu_operation_ids: Iterable[str] = (),
    supported_cpu_operation_ids: Iterable[str] = (),
) -> ComputeCapabilityReport:
    """Return current CPU/CuPy capability without a module-level GPU import.

    ``supported_gpu_operation_ids`` must describe GPU implementations that VIPP
    actually exposes, not operations merely offered by CuPy or cuCIM.  It is
    intentionally empty in the spike so the UI cannot claim support before an
    implementation passes parity, speed, memory, progress, and cancellation
    gates.
    """

    cpu = BackendCapability(
        backend=ComputeBackend.CPU,
        available=True,
        provider="NumPy/SciPy/scikit-image",
        version="; ".join(
            (
                f"numpy {_distribution_version('numpy')}",
                f"scipy {_distribution_version('scipy')}",
                f"scikit-image {_distribution_version('scikit-image')}",
            )
        ),
        device_name=platform.processor() or "Host CPU",
        supported_operation_ids=_normalized_operation_ids(supported_cpu_operation_ids),
    )
    gpu = _detect_cupy_capability(
        _normalized_operation_ids(supported_gpu_operation_ids)
    )
    return ComputeCapabilityReport(cpu=cpu, gpu=gpu)


def select_compute_backend(
    requested: ComputeBackend | str,
    operation_id: str,
    *,
    capabilities: ComputeCapabilityReport | None = None,
    allow_explicit_gpu_fallback: bool = False,
) -> BackendSelection:
    """Resolve Auto/CPU/GPU without silently weakening an explicit GPU request."""

    requested_backend = ComputeBackend.parse(requested)
    operation_id = str(operation_id).strip()
    if not operation_id:
        raise ValueError("operation_id must not be empty.")
    if requested_backend is ComputeBackend.CPU:
        return BackendSelection(
            requested=requested_backend,
            resolved=ComputeBackend.CPU,
            operation_id=operation_id,
            fell_back=False,
            reason="CPU was explicitly requested.",
        )

    report = capabilities or detect_compute_capabilities()

    if report.gpu.supports(operation_id):
        reason = (
            f"{report.gpu.provider} supports {operation_id!r} on "
            f"{report.gpu.device_name}."
        )
        return BackendSelection(
            requested=requested_backend,
            resolved=ComputeBackend.GPU,
            operation_id=operation_id,
            fell_back=False,
            reason=reason,
        )

    unavailable_reason = report.gpu.reason or (
        f"{operation_id!r} has no validated GPU implementation."
    )
    if requested_backend is ComputeBackend.AUTO:
        return BackendSelection(
            requested=requested_backend,
            resolved=ComputeBackend.CPU,
            operation_id=operation_id,
            fell_back=False,
            reason=f"Auto selected CPU: {unavailable_reason}",
        )
    if allow_explicit_gpu_fallback:
        return BackendSelection(
            requested=requested_backend,
            resolved=ComputeBackend.CPU,
            operation_id=operation_id,
            fell_back=True,
            reason=f"Explicit GPU request fell back to CPU: {unavailable_reason}",
        )
    raise ComputeBackendUnavailable(
        f"Cannot run {operation_id!r} on GPU: {unavailable_reason}"
    )


def _detect_cupy_capability(
    supported_operation_ids: tuple[str, ...],
) -> BackendCapability:
    try:
        cupy = importlib.import_module("cupy")
    except Exception as exc:
        return BackendCapability(
            backend=ComputeBackend.GPU,
            available=False,
            provider="CuPy",
            version="",
            device_name="",
            supported_operation_ids=(),
            reason=_provider_error("CuPy is not importable", exc),
        )

    version = str(getattr(cupy, "__version__", "unknown"))
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA devices were reported")
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        raw_name = properties.get("name", "CUDA device 0")
        device_name = (
            raw_name.decode(errors="replace")
            if isinstance(raw_name, bytes)
            else str(raw_name)
        )
        # Force context creation and one real device operation.  Import success
        # alone is insufficient when the wheel and driver/runtime are mismatched.
        probe = cupy.arange(1, dtype=cupy.float32)
        cupy.cuda.get_current_stream().synchronize()
        del probe
    except Exception as exc:
        return BackendCapability(
            backend=ComputeBackend.GPU,
            available=False,
            provider="CuPy",
            version=version,
            device_name="",
            supported_operation_ids=(),
            reason=_provider_error("CuPy could not execute a CUDA probe", exc),
        )
    return BackendCapability(
        backend=ComputeBackend.GPU,
        available=True,
        provider="CuPy",
        version=version,
        device_name=device_name,
        supported_operation_ids=supported_operation_ids,
        reason=(
            ""
            if supported_operation_ids
            else "No VIPP GPU operations have passed promotion gates yet."
        ),
    )


def _normalized_operation_ids(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _normalized_unique_pairs(
    values: Iterable[tuple[object, object]],
    field_name: str,
    *,
    strip_values: bool = False,
) -> tuple[tuple[str, str], ...]:
    normalized = tuple(
        (
            str(raw_key).strip(),
            str(raw_value).strip() if strip_values else str(raw_value),
        )
        for raw_key, raw_value in values
    )
    if any(not key for key, _value in normalized):
        raise ValueError(f"{field_name} keys must not be empty.")
    if len({key for key, _value in normalized}) != len(normalized):
        raise ValueError(f"{field_name} keys must be unique.")
    return tuple(sorted(normalized))


def _normalized_scoped_metadata(
    values: Iterable[tuple[object, Iterable[tuple[object, object]]]],
    field_name: str,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    normalized = tuple(
        (
            str(raw_scope).strip(),
            _normalized_unique_pairs(metadata, f"{field_name}[{raw_scope!r}]"),
        )
        for raw_scope, metadata in values
    )
    if any(not scope for scope, _metadata in normalized):
        raise ValueError(f"{field_name} scope IDs must not be empty.")
    if len({scope for scope, _metadata in normalized}) != len(normalized):
        raise ValueError(f"{field_name} scope IDs must be unique.")
    return tuple(sorted(normalized))


def _provider_error(prefix: str, exc: Exception) -> str:
    message = str(exc).strip()
    suffix = f": {type(exc).__name__}"
    if message:
        suffix += f": {message}"
    return prefix + suffix


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _capability_dict(capability: BackendCapability) -> dict[str, object]:
    payload = asdict(capability)
    payload["backend"] = capability.backend.value
    payload["supported_operation_ids"] = list(capability.supported_operation_ids)
    return payload


__all__ = [
    "BackendCapability",
    "BenchmarkCandidateResult",
    "BenchmarkRecord",
    "BenchmarkRecordKey",
    "BackendSelection",
    "CacheAdmissibility",
    "ComputeBackend",
    "ComputeBackendUnavailable",
    "ComputeCapabilityReport",
    "ComputeEnvironment",
    "ComputeMode",
    "ComputeRepairAction",
    "ComputeRepairCandidate",
    "ComputeRepairSuggestion",
    "ComputeRequest",
    "DecisionKind",
    "DecisionReason",
    "ExecutionPlan",
    "ExecutionFallbackRecord",
    "ExactWorkloadCandidateQualification",
    "ExecutionReport",
    "ExecutionSegment",
    "FallbackReason",
    "FallbackPolicy",
    "MemoryEstimate",
    "MemoryTopology",
    "NodeComputePreference",
    "NodeExecutionDecision",
    "NodePreferenceKind",
    "OutputPortKey",
    "ScientificResultKey",
    "WorkloadDescriptor",
    "canonical_digest",
    "detect_compute_capabilities",
    "exact_workload_identity_digest",
    "select_compute_backend",
]
