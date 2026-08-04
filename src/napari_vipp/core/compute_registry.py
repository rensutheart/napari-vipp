"""Lazy runtime and implementation registry for optional compute providers.

The registry stores immutable descriptors and import strings.  Constructing or
validating it never imports a provider module, creates a device context, or
discovers third-party entry points.  A runtime factory or implementation
callable is resolved only after an execution request explicitly asks for it.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import re
import sys
import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from napari_vipp.core.accelerator_lease import accelerator_lease
from napari_vipp.core.compute import MemoryTopology
from napari_vipp.core.compute_policy import validate_spec_policy_references
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    OperationComputeSpec,
    accelerator_compute_specs,
    validate_compute_specs,
)

_CUCIM_MEASUREMENT_CACHE_LOCK = threading.RLock()
_CUCIM_MEASUREMENT_CACHE_POOLS: dict[
    tuple[int, str],
    tuple[object, object, object, object],
] = {}
_CUCIM_MEASUREMENT_CACHE_MAX_BYTES = 1 * 1024**2


def _required_text(value: object, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _validated_ref(value: object, field_name: str) -> str:
    reference = _required_text(value, field_name)
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise ValueError(f"{field_name} must use 'module:attribute' syntax.")
    return reference


def _normalized_strings(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _normalized_nonempty(values: Sequence[object], field_name: str) -> tuple[str, ...]:
    normalized = _normalized_strings(values)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value.")
    return normalized


def _normalized_metadata(
    values: Sequence[tuple[object, object]],
) -> tuple[tuple[str, str], ...]:
    normalized = tuple((str(key).strip(), str(value)) for key, value in values)
    if any(not key for key, _value in normalized):
        raise ValueError("metadata keys must not be empty.")
    if len({key for key, _value in normalized}) != len(normalized):
        raise ValueError("metadata keys must be unique.")
    return normalized


def _validate_optional_bytes(
    value: object,
    field_name: str,
    *,
    optional: bool = True,
) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or None" if optional else ""
        raise ValueError(f"{field_name} must be a non-negative integer{suffix}.")


class RuntimeExceptionKind(StrEnum):
    """Provider-neutral classification of a runtime failure."""

    OUT_OF_MEMORY = "out_of_memory"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    INVALID_DEVICE = "invalid_device"
    KERNEL_FAILURE = "kernel_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeDevice:
    """One device reported by a runtime probe."""

    device_id: str
    display_name: str
    total_memory_bytes: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device_id", _required_text(self.device_id, "device_id")
        )
        object.__setattr__(
            self,
            "display_name",
            _required_text(self.display_name, "display_name"),
        )
        _validate_optional_bytes(self.total_memory_bytes, "total_memory_bytes")
        object.__setattr__(self, "metadata", _normalized_metadata(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "display_name": self.display_name,
            "total_memory_bytes": self.total_memory_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeProbeResult:
    """JSON-safe result of an explicitly requested runtime probe."""

    runtime_id: str
    available: bool
    version: str = ""
    devices: tuple[RuntimeDevice, ...] = ()
    selected_device_id: str = ""
    reason_code: str = ""
    message: str = ""
    environment_fingerprint: str = ""
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_id",
            _required_text(self.runtime_id, "runtime_id"),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean.")
        devices = tuple(self.devices)
        if any(not isinstance(device, RuntimeDevice) for device in devices):
            raise TypeError("devices must contain RuntimeDevice values.")
        device_ids = tuple(device.device_id for device in devices)
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("runtime probe device IDs must be unique.")
        selected = str(self.selected_device_id).strip()
        if selected and selected not in device_ids:
            raise ValueError("selected_device_id must reference a reported device.")
        object.__setattr__(self, "version", str(self.version).strip())
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "selected_device_id", selected)
        for name in ("reason_code", "message", "environment_fingerprint"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        object.__setattr__(self, "metadata", _normalized_metadata(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime_id": self.runtime_id,
            "available": self.available,
            "version": self.version,
            "devices": [device.as_dict() for device in self.devices],
            "selected_device_id": self.selected_device_id,
            "reason_code": self.reason_code,
            "message": self.message,
            "environment_fingerprint": self.environment_fingerprint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ImplementationLibraryProbeResult:
    """JSON-safe result of an explicitly requested implementation probe."""

    library_id: str
    available: bool
    version: str = ""
    reason_code: str = ""
    message: str = ""
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "library_id",
            _required_text(self.library_id, "library_id"),
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean.")
        for name in ("version", "reason_code", "message"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        object.__setattr__(self, "metadata", _normalized_metadata(self.metadata))

    def as_dict(self) -> dict[str, object]:
        return {
            "library_id": self.library_id,
            "available": self.available,
            "version": self.version,
            "reason_code": self.reason_code,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RuntimeMemorySnapshot:
    """Device-wide and runtime-owned memory observed at one checkpoint."""

    runtime_id: str
    device_id: str
    topology: MemoryTopology | str
    device_total_bytes: int | None = None
    device_free_bytes: int | None = None
    runtime_live_bytes: int = 0
    runtime_reserved_bytes: int = 0
    out_of_pool_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_id",
            _required_text(self.runtime_id, "runtime_id"),
        )
        object.__setattr__(self, "device_id", str(self.device_id).strip())
        topology = (
            self.topology
            if isinstance(self.topology, MemoryTopology)
            else MemoryTopology(str(self.topology).strip().lower())
        )
        object.__setattr__(self, "topology", topology)
        for name in (
            "device_total_bytes",
            "device_free_bytes",
            "runtime_live_bytes",
            "runtime_reserved_bytes",
            "out_of_pool_bytes",
        ):
            _validate_optional_bytes(
                getattr(self, name), name, optional=name.startswith("device_")
            )
        if (
            self.device_total_bytes is not None
            and self.device_free_bytes is not None
            and self.device_free_bytes > self.device_total_bytes
        ):
            raise ValueError("device_free_bytes must not exceed device_total_bytes.")
        if self.runtime_live_bytes > self.runtime_reserved_bytes:
            raise ValueError(
                "runtime_live_bytes must not exceed runtime_reserved_bytes."
            )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["topology"] = self.topology.value
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeExceptionInfo:
    """Sanitized provider-neutral information about one exception."""

    kind: RuntimeExceptionKind | str
    reason_code: str
    message: str
    exception_type: str = ""
    retryable: bool = False
    cleanup_required: bool = True

    def __post_init__(self) -> None:
        kind = (
            self.kind
            if isinstance(self.kind, RuntimeExceptionKind)
            else RuntimeExceptionKind(str(self.kind).strip().lower())
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "reason_code",
            _required_text(self.reason_code, "reason_code"),
        )
        object.__setattr__(self, "message", str(self.message).strip())
        object.__setattr__(self, "exception_type", str(self.exception_type).strip())
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean.")
        if not isinstance(self.cleanup_required, bool):
            raise TypeError("cleanup_required must be a boolean.")

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


@runtime_checkable
class RuntimeProtocol(Protocol):
    """Array-runtime boundary used by device execution.

    Device values are deliberately typed as ``object``.  Callers must not
    inspect or coerce them outside the owning runtime.
    """

    runtime_id: str
    array_domain: str

    def probe(self, *, refresh: bool = False) -> RuntimeProbeResult: ...

    def execution_scope(
        self,
        *,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
    ) -> AbstractContextManager[None]: ...

    def is_device_value(self, value: object) -> bool: ...

    def allocation_identity(self, value: object) -> Hashable:
        """Return the runtime-owned allocation backing ``value``.

        Distinct array objects (for example a base array and a view) must
        return the same identity when they share one allocation.  The method
        also serves as the ownership check for the active execution scope.
        """
        ...

    def to_device(self, value: object, *, device_id: str = "") -> object: ...

    def to_host(self, value: object) -> object: ...

    def release(self, value: object) -> None:
        """Relinquish runtime ownership; callers must then drop all aliases."""
        ...

    def synchronize(self, *, device_id: str = "") -> None: ...

    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot: ...

    def classify_exception(self, exc: BaseException) -> RuntimeExceptionInfo: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[], RuntimeProtocol]
ImplementationLibraryProbe = Callable[[], ImplementationLibraryProbeResult]


@dataclass(frozen=True, slots=True)
class RuntimeDescriptor:
    """Import-free declaration of one array/device runtime."""

    runtime_id: str
    display_name: str
    factory_ref: str
    array_domain: str
    device_domain: str
    supported_os_families: tuple[str, ...]
    interoperability_claims: tuple[str, ...] = ()
    origin: str = "builtin"

    def __post_init__(self) -> None:
        for name in (
            "runtime_id",
            "display_name",
            "array_domain",
            "device_domain",
            "origin",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self, "factory_ref", _validated_ref(self.factory_ref, "factory_ref")
        )
        object.__setattr__(
            self,
            "supported_os_families",
            _normalized_nonempty(self.supported_os_families, "supported_os_families"),
        )
        object.__setattr__(
            self,
            "interoperability_claims",
            _normalized_strings(self.interoperability_claims),
        )


@dataclass(frozen=True, slots=True)
class ImplementationLibraryDescriptor:
    """Import-free declaration of a library layered on array runtimes."""

    library_id: str
    display_name: str
    runtime_ids: tuple[str, ...]
    array_domain: str
    supported_os_families: tuple[str, ...]
    probe_ref: str = ""
    interoperability_claims: tuple[str, ...] = ()
    origin: str = "builtin"

    def __post_init__(self) -> None:
        for name in ("library_id", "display_name", "array_domain", "origin"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "runtime_ids",
            _normalized_nonempty(self.runtime_ids, "runtime_ids"),
        )
        object.__setattr__(
            self,
            "supported_os_families",
            _normalized_nonempty(self.supported_os_families, "supported_os_families"),
        )
        probe_ref = str(self.probe_ref).strip()
        object.__setattr__(
            self,
            "probe_ref",
            _validated_ref(probe_ref, "probe_ref") if probe_ref else "",
        )
        object.__setattr__(
            self,
            "interoperability_claims",
            _normalized_strings(self.interoperability_claims),
        )


CUDA_CUPY_RUNTIME = RuntimeDescriptor(
    runtime_id="cuda-cupy",
    display_name="CUDA via CuPy",
    factory_ref="napari_vipp.core.gpu.cupy_runtime:create_runtime",
    array_domain="cuda-cupy",
    device_domain="nvidia-cuda",
    supported_os_families=("Windows", "Linux"),
    interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
)

CUPY_LIBRARY = ImplementationLibraryDescriptor(
    library_id="cupy",
    display_name="CuPy custom kernels",
    runtime_ids=("cuda-cupy",),
    array_domain="cuda-cupy",
    supported_os_families=("Windows", "Linux"),
    probe_ref="napari_vipp.core.compute_registry:_probe_cupy_library",
    interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
)

CUPYX_LIBRARY = ImplementationLibraryDescriptor(
    library_id="cupyx",
    display_name="CuPyX SciPy-compatible ndimage",
    runtime_ids=("cuda-cupy",),
    array_domain="cuda-cupy",
    supported_os_families=("Windows", "Linux"),
    probe_ref="napari_vipp.core.compute_registry:_probe_cupyx_library",
    interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
)

CUCIM_SKIMAGE_LIBRARY = ImplementationLibraryDescriptor(
    library_id="cucim",
    display_name="cuCIM scikit-image",
    runtime_ids=("cuda-cupy",),
    array_domain="cuda-cupy",
    supported_os_families=("Windows", "Linux"),
    probe_ref="napari_vipp.core.compute_registry:_probe_cucim_skimage_library",
    interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
)

DEFAULT_RUNTIME_DESCRIPTORS = (CUDA_CUPY_RUNTIME,)
DEFAULT_LIBRARY_DESCRIPTORS = (
    CUPY_LIBRARY,
    CUPYX_LIBRARY,
    CUCIM_SKIMAGE_LIBRARY,
)


class ComputeRegistryError(RuntimeError):
    """Base class for lazy registry failures."""


class ComputeRegistryClosed(ComputeRegistryError):
    """Raised when a terminally closed registry is reused."""


class ComputeRegistryLoadError(ComputeRegistryError):
    """Raised when a lazy factory or implementation cannot be loaded."""


def validate_registry(
    runtime_descriptors: Sequence[RuntimeDescriptor],
    library_descriptors: Sequence[ImplementationLibraryDescriptor],
    implementation_specs: Sequence[OperationComputeSpec],
) -> None:
    """Validate registry references without loading any descriptor import ref."""

    runtimes = _unique_by_id(runtime_descriptors, "runtime", "runtime_id")
    libraries = _unique_by_id(library_descriptors, "library", "library_id")
    specs = tuple(implementation_specs)
    if specs:
        validate_compute_specs(specs)
    identities: set[tuple[str, str, str]] = set()
    for spec in specs:
        validate_spec_policy_references(spec)
        identity = (
            spec.runtime_id,
            spec.implementation_id,
            spec.implementation_version,
        )
        if identity in identities:
            raise ValueError(f"Duplicate runtime implementation identity {identity!r}.")
        identities.add(identity)
        runtime = runtimes.get(spec.runtime_id)
        if runtime is None:
            raise ValueError(
                f"Implementation {spec.implementation_id!r} references unknown "
                f"runtime {spec.runtime_id!r}."
            )
        library = libraries.get(spec.implementation_library_id)
        if library is None:
            raise ValueError(
                f"Implementation {spec.implementation_id!r} references unknown "
                f"library {spec.implementation_library_id!r}."
            )
        if spec.runtime_id not in library.runtime_ids:
            raise ValueError(
                f"Library {library.library_id!r} does not support runtime "
                f"{spec.runtime_id!r}."
            )
        if spec.array_domain != runtime.array_domain:
            raise ValueError(
                f"Implementation {spec.implementation_id!r} array domain "
                f"{spec.array_domain!r} does not match runtime domain "
                f"{runtime.array_domain!r}."
            )
        if spec.array_domain != library.array_domain:
            raise ValueError(
                f"Implementation {spec.implementation_id!r} array domain "
                f"{spec.array_domain!r} does not match library domain "
                f"{library.array_domain!r}."
            )
    for library in libraries.values():
        missing = set(library.runtime_ids) - set(runtimes)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"Library {library.library_id!r} references unknown runtime(s): "
                f"{names}."
            )


class ComputeRegistry:
    """Thread-safe lazy runtime and implementation registry."""

    def __init__(
        self,
        *,
        runtime_descriptors: Sequence[RuntimeDescriptor] = DEFAULT_RUNTIME_DESCRIPTORS,
        library_descriptors: Sequence[ImplementationLibraryDescriptor] = (
            DEFAULT_LIBRARY_DESCRIPTORS
        ),
        implementation_specs: Sequence[OperationComputeSpec] | None = None,
        runtime_factories: Mapping[str, RuntimeFactory] | None = None,
        library_probes: Mapping[str, ImplementationLibraryProbe] | None = None,
    ) -> None:
        runtimes = tuple(runtime_descriptors)
        libraries = tuple(library_descriptors)
        specs = tuple(
            accelerator_compute_specs()
            if implementation_specs is None
            else implementation_specs
        )
        validate_registry(runtimes, libraries, specs)
        runtime_map = _unique_by_id(runtimes, "runtime", "runtime_id")
        library_map = _unique_by_id(libraries, "library", "library_id")
        factories = dict(runtime_factories or {})
        unknown_factories = set(factories) - set(runtime_map)
        if unknown_factories:
            names = ", ".join(sorted(unknown_factories))
            raise ValueError(
                f"Runtime factories reference unknown runtime(s): {names}."
            )
        if any(not callable(factory) for factory in factories.values()):
            raise TypeError("runtime_factories values must be callable.")
        probes = dict(library_probes or {})
        unknown_probes = set(probes) - set(library_map)
        if unknown_probes:
            names = ", ".join(sorted(unknown_probes))
            raise ValueError(
                f"Library probes reference unknown implementation libraries: {names}."
            )
        if any(not callable(probe) for probe in probes.values()):
            raise TypeError("library_probes values must be callable.")

        self._runtime_descriptors = MappingProxyType(runtime_map)
        self._library_descriptors = MappingProxyType(library_map)
        self._implementation_specs = specs
        self._runtime_factories = MappingProxyType(factories)
        self._library_probes = MappingProxyType(probes)
        self._runtime_instances: dict[str, RuntimeProtocol] = {}
        self._probe_results: dict[str, RuntimeProbeResult] = {}
        self._library_probe_results: dict[str, ImplementationLibraryProbeResult] = {}
        self._implementation_callables: dict[tuple[str, str, str], Callable] = {}
        self._lock = threading.RLock()
        self._closed = False

    @property
    def runtime_descriptors(self) -> tuple[RuntimeDescriptor, ...]:
        return tuple(self._runtime_descriptors.values())

    @property
    def library_descriptors(self) -> tuple[ImplementationLibraryDescriptor, ...]:
        return tuple(self._library_descriptors.values())

    @property
    def implementation_specs(self) -> tuple[OperationComputeSpec, ...]:
        return self._implementation_specs

    def runtime_descriptor(self, runtime_id: str) -> RuntimeDescriptor:
        return self._lookup(self._runtime_descriptors, runtime_id, "runtime")

    def library_descriptor(self, library_id: str) -> ImplementationLibraryDescriptor:
        return self._lookup(self._library_descriptors, library_id, "library")

    def implementations_for_operation(
        self,
        operation_id: str,
        *,
        allow_experimental: bool = False,
    ) -> tuple[OperationComputeSpec, ...]:
        operation_id = str(operation_id).strip()
        return tuple(
            spec
            for spec in self._implementation_specs
            if spec.operation_id == operation_id
            and spec.visible_for(allow_experimental=allow_experimental)
        )

    def implementation_spec(
        self,
        implementation_id: str,
        implementation_version: str | None = None,
        *,
        allow_experimental: bool = False,
    ) -> OperationComputeSpec:
        implementation_id = str(implementation_id).strip()
        version = (
            None
            if implementation_version is None
            else str(implementation_version).strip()
        )
        matches = tuple(
            spec
            for spec in self._implementation_specs
            if spec.implementation_id == implementation_id
            and (version is None or spec.implementation_version == version)
        )
        if not matches:
            raise KeyError(f"Unknown implementation {implementation_id!r}.")
        if len(matches) > 1:
            raise KeyError(
                f"Implementation {implementation_id!r} has multiple versions; "
                "an exact version is required."
            )
        spec = matches[0]
        if (
            spec.admission_tier is AdmissionTier.DEVELOPER_HIDDEN
            and not allow_experimental
        ):
            raise KeyError(f"Implementation {implementation_id!r} is developer-hidden.")
        return spec

    def runtime(self, runtime_id: str) -> RuntimeProtocol:
        """Return one process-lifetime runtime, constructing it on first use."""

        runtime_id = str(runtime_id).strip()
        with self._lock:
            self._ensure_open()
            descriptor = self.runtime_descriptor(runtime_id)
            existing = self._runtime_instances.get(runtime_id)
            if existing is not None:
                return existing
            try:
                factory = self._runtime_factories.get(runtime_id)
                if factory is None:
                    factory = _load_ref(descriptor.factory_ref)
                if not callable(factory):
                    raise TypeError(
                        f"Runtime factory {descriptor.factory_ref!r} is not callable."
                    )
                instance = factory()
                _validate_runtime_instance(instance, descriptor)
            except Exception as exc:
                _close_quietly(locals().get("instance"))
                raise ComputeRegistryLoadError(
                    f"Could not load runtime {runtime_id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._runtime_instances[runtime_id] = instance
            return instance

    def probe_runtime(
        self,
        runtime_id: str,
        *,
        refresh: bool = False,
    ) -> RuntimeProbeResult:
        runtime_id = str(runtime_id).strip()
        with self._lock:
            self._ensure_open()
            if not refresh and runtime_id in self._probe_results:
                return self._probe_results[runtime_id]
            try:
                runtime = self.runtime(runtime_id)
            except ComputeRegistryLoadError as exc:
                result = RuntimeProbeResult(
                    runtime_id,
                    False,
                    reason_code="runtime_load_failed",
                    message=str(exc),
                )
                self._probe_results[runtime_id] = result
                return result

        # Provider probes may wait for the process accelerator lease.  Never
        # retain the registry lock across that call: execution owns the lease
        # before resolving implementation callables through this registry.
        try:
            result = runtime.probe(refresh=refresh)
        except Exception as exc:
            try:
                failure = runtime.classify_exception(exc)
            except Exception:
                failure = RuntimeExceptionInfo(
                    RuntimeExceptionKind.UNKNOWN,
                    "runtime_probe_failed",
                    str(exc),
                    exception_type=type(exc).__name__,
                )
            result = RuntimeProbeResult(
                runtime_id,
                False,
                reason_code=failure.reason_code,
                message=failure.message,
                metadata=(
                    ("exception_kind", failure.kind.value),
                    ("exception_type", failure.exception_type),
                ),
            )
        if not isinstance(result, RuntimeProbeResult):
            raise TypeError("Runtime probe must return RuntimeProbeResult.")
        if result.runtime_id != runtime_id:
            raise ValueError("Runtime probe result belongs to a different runtime.")
        with self._lock:
            self._ensure_open()
            if not refresh and runtime_id in self._probe_results:
                return self._probe_results[runtime_id]
            if self._runtime_instances.get(runtime_id) is runtime:
                self._probe_results[runtime_id] = result
            return result

    def probe_library(
        self,
        library_id: str,
        *,
        refresh: bool = False,
    ) -> ImplementationLibraryProbeResult:
        """Probe one optional implementation library only when requested."""

        library_id = str(library_id).strip()
        with self._lock:
            self._ensure_open()
            descriptor = self.library_descriptor(library_id)
            if not refresh and library_id in self._library_probe_results:
                return self._library_probe_results[library_id]
            probe = self._library_probes.get(library_id)
            if probe is None and descriptor.probe_ref:
                try:
                    probe = _load_ref(descriptor.probe_ref)
                except Exception as exc:
                    result = ImplementationLibraryProbeResult(
                        library_id,
                        False,
                        reason_code="library_probe_load_failed",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                    self._library_probe_results[library_id] = result
                    return result
            if probe is None:
                result = ImplementationLibraryProbeResult(
                    library_id,
                    False,
                    reason_code="library_probe_missing",
                    message="No implementation-library probe is declared.",
                )
                self._library_probe_results[library_id] = result
                return result
            if not callable(probe):
                result = ImplementationLibraryProbeResult(
                    library_id,
                    False,
                    reason_code="library_probe_invalid",
                    message="The implementation-library probe is not callable.",
                )
                self._library_probe_results[library_id] = result
                return result

        try:
            result = probe()
        except Exception as exc:
            result = ImplementationLibraryProbeResult(
                library_id,
                False,
                reason_code="library_probe_failed",
                message=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(result, ImplementationLibraryProbeResult):
            raise TypeError(
                "Implementation-library probes must return "
                "ImplementationLibraryProbeResult."
            )
        if result.library_id != library_id:
            raise ValueError(
                "Implementation-library probe result belongs to a different "
                "library."
            )
        with self._lock:
            self._ensure_open()
            if not refresh and library_id in self._library_probe_results:
                return self._library_probe_results[library_id]
            self._library_probe_results[library_id] = result
            return result

    def interoperability_contract(
        self,
        runtime_id: str,
        library_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return common zero-copy claims without importing any provider."""

        runtime = self.runtime_descriptor(runtime_id)
        claims = set(runtime.interoperability_claims)
        normalized_ids = _normalized_nonempty(library_ids, "library_ids")
        for library_id in normalized_ids:
            library = self.library_descriptor(library_id)
            if (
                runtime.runtime_id not in library.runtime_ids
                or runtime.array_domain != library.array_domain
            ):
                return ()
            claims.intersection_update(library.interoperability_claims)
        return tuple(sorted(claims))

    def implementation_callable(
        self,
        implementation: OperationComputeSpec | str,
        implementation_version: str | None = None,
        *,
        allow_experimental: bool = False,
    ) -> Callable:
        """Resolve one implementation import string only when explicitly used."""

        with self._lock:
            self._ensure_open()
            if isinstance(implementation, OperationComputeSpec):
                spec = self.implementation_spec(
                    implementation.implementation_id,
                    implementation.implementation_version,
                    allow_experimental=allow_experimental,
                )
                if spec != implementation:
                    raise KeyError(
                        f"Implementation {implementation.implementation_id!r} does "
                        "not match its registered declaration."
                    )
            else:
                spec = self.implementation_spec(
                    implementation,
                    implementation_version,
                    allow_experimental=allow_experimental,
                )
            if (
                spec.admission_tier is AdmissionTier.DEVELOPER_HIDDEN
                and not allow_experimental
            ):
                raise KeyError(
                    f"Implementation {spec.implementation_id!r} is developer-hidden."
                )
            if not spec.callable_ref:
                raise ComputeRegistryLoadError(
                    f"Implementation {spec.implementation_id!r} is a host boundary "
                    "without a callable."
                )
            identity = (
                spec.runtime_id,
                spec.implementation_id,
                spec.implementation_version,
            )
            cached = self._implementation_callables.get(identity)
            if cached is not None:
                return cached
            try:
                candidate = _load_ref(spec.callable_ref)
            except Exception as exc:
                raise ComputeRegistryLoadError(
                    f"Could not load implementation {spec.implementation_id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not callable(candidate):
                raise ComputeRegistryLoadError(
                    f"Implementation {spec.callable_ref!r} is not callable."
                )
            self._implementation_callables[identity] = candidate
            return candidate

    def release_runtime(self, runtime_id: str) -> bool:
        """Close and evict one initialized runtime without closing the registry."""

        runtime_id = str(runtime_id).strip()
        with self._lock:
            self._ensure_open()
            instance = self._runtime_instances.pop(runtime_id, None)
            self._probe_results.pop(runtime_id, None)
            for library in self._library_descriptors.values():
                if runtime_id in library.runtime_ids:
                    self._library_probe_results.pop(library.library_id, None)
        if instance is None:
            return False
        instance.close()
        return True

    def close(self) -> None:
        """Close every initialized runtime.  Closing is idempotent and terminal."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            instances = tuple(reversed(tuple(self._runtime_instances.values())))
            self._runtime_instances.clear()
            self._probe_results.clear()
            self._library_probe_results.clear()
            self._implementation_callables.clear()
        failures = []
        for instance in instances:
            try:
                instance.close()
            except Exception as exc:  # pragma: no cover - exercised via aggregation
                failures.append(f"{type(exc).__name__}: {exc}")
        if failures:
            raise ComputeRegistryError(
                "One or more runtimes failed to close: " + "; ".join(failures)
            )

    def __enter__(self) -> ComputeRegistry:
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ComputeRegistryClosed("Compute registry is closed.")

    @staticmethod
    def _lookup(mapping: Mapping[str, object], value: str, description: str):
        identifier = str(value).strip()
        try:
            return mapping[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown {description} {identifier!r}.") from exc


def _probe_cupy_library() -> ImplementationLibraryProbeResult:
    """Import CuPy lazily and exercise compilation of a custom RawKernel."""

    cupy = importlib.import_module("cupy")
    if not callable(getattr(cupy, "RawKernel", None)):
        return ImplementationLibraryProbeResult(
            "cupy",
            False,
            version=str(getattr(cupy, "__version__", "")),
            reason_code="cupy_rawkernel_missing",
            message="CuPy does not expose the RawKernel API required by VIPP.",
        )
    device_id = _cupy_current_device_id(cupy)
    with accelerator_lease("cuda-cupy", device_id):
        return _exercise_cupy_library(cupy)


def _exercise_cupy_library(cupy: object) -> ImplementationLibraryProbeResult:
    pool = cupy.cuda.MemoryPool()
    values = output = None
    probe_error: BaseException | None = None
    try:
        kernel = cupy.RawKernel(
            r"""
            extern "C" __global__
            void vipp_cupy_probe(
                const float* values,
                float* output,
                const unsigned long long size)
            {
                const unsigned long long index =
                    (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (index < size) {
                    output[index] = values[index] + 1.0f;
                }
            }
            """,
            "vipp_cupy_probe",
            options=("--std=c++11", "--fmad=false"),
        )
        with cupy.cuda.using_allocator(pool.malloc):
            values = cupy.arange(32, dtype=cupy.float32)
            output = cupy.empty_like(values)
            kernel((1,), (32,), (values, output, values.size))
            if float(output.sum().item()) != 528.0:
                raise RuntimeError("CuPy RawKernel probe produced an invalid result.")
            cupy.cuda.get_current_stream().synchronize()
        return ImplementationLibraryProbeResult(
            "cupy",
            True,
            version=str(getattr(cupy, "__version__", "")),
            message="CuPy compiled and synchronized a VIPP custom RawKernel probe.",
        )
    except BaseException as exc:
        probe_error = exc
        raise
    finally:
        values = output = None
        _drain_private_probe_pool(
            cupy,
            pool,
            library_id="cupy",
            suppress_errors=probe_error is not None,
        )


def _probe_cupyx_library() -> ImplementationLibraryProbeResult:
    """Import and exercise required CuPyX ndimage and signal primitives."""

    cupy = importlib.import_module("cupy")
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    signal = importlib.import_module("cupyx.scipy.signal")
    if (
        not callable(getattr(ndimage, "gaussian_filter", None))
        or not callable(getattr(ndimage, "median_filter", None))
        or not callable(getattr(ndimage, "label", None))
        or not callable(getattr(signal, "convolve", None))
    ):
        return ImplementationLibraryProbeResult(
            "cupyx",
            False,
            version=str(getattr(cupy, "__version__", "")),
            reason_code="cupyx_ndimage_incomplete",
            message=(
                "CuPyX does not expose the required Gaussian, median, "
                "connected-components, and signal-convolution functions."
            ),
        )
    device_id = _cupy_current_device_id(cupy)
    with accelerator_lease("cuda-cupy", device_id):
        return _exercise_cupyx_library(cupy, ndimage, signal)


def _exercise_cupyx_library(
    cupy: object,
    ndimage: object,
    signal: object,
) -> ImplementationLibraryProbeResult:
    pool = cupy.cuda.MemoryPool()
    values = gaussian = median = convolved = None
    label_source = labeled = expected_labels = None
    probe_error: BaseException | None = None
    try:
        with cupy.cuda.using_allocator(pool.malloc):
            values = cupy.arange(49, dtype=cupy.float32).reshape(7, 7)
            gaussian = ndimage.gaussian_filter(
                values,
                sigma=0.8,
                mode="reflect",
            )
            median = ndimage.median_filter(values, size=3, mode="reflect")
            convolved = signal.convolve(
                values,
                cupy.asarray([[0.0, 1.0, 0.0]], dtype=cupy.float32),
                mode="same",
                method="fft",
            )
            label_source = cupy.asarray(
                [[True, False], [False, True]],
                dtype=np.bool_,
            )
            labeled, label_count = ndimage.label(label_source)
            expected_labels = cupy.asarray(
                [[1, 0], [0, 2]],
                dtype=np.int32,
            )
            exact_labels = bool((labeled == expected_labels).all().item())
            float((gaussian + median + convolved).sum().item())
            cupy.cuda.get_current_stream().synchronize()
            if (
                label_count != 2
                or labeled.dtype != np.dtype(np.int32)
                or not exact_labels
            ):
                return ImplementationLibraryProbeResult(
                    "cupyx",
                    False,
                    version=str(getattr(cupy, "__version__", "")),
                    reason_code="cupyx_label_semantics_mismatch",
                    message=(
                        "CuPyX connected-components probing did not return the "
                        "required exact int32 labels."
                    ),
                )
        return ImplementationLibraryProbeResult(
            "cupyx",
            True,
            version=str(getattr(cupy, "__version__", "")),
            message=(
                "CuPyX completed synchronized Gaussian, median, connected-"
                "components, and signal-convolution probes."
            ),
        )
    except BaseException as exc:
        probe_error = exc
        raise
    finally:
        values = gaussian = median = convolved = None
        label_source = labeled = expected_labels = None
        _drain_private_probe_pool(
            cupy,
            pool,
            library_id="cupyx",
            suppress_errors=probe_error is not None,
        )


def _probe_cucim_skimage_library(
    *,
    record_path: Path | None = None,
) -> ImplementationLibraryProbeResult:
    """Import the checksum-installed cuCIM restoration layer lazily."""

    path = record_path or _gpu_environment_record_path()
    try:
        provenance = _read_cucim_environment_provenance(path)
    except FileNotFoundError:
        return ImplementationLibraryProbeResult(
            "cucim",
            False,
            reason_code="cucim_provenance_missing",
            message=(
                f"The verified GPU environment record is missing at {path}. "
                "Re-run scripts/setup_gpu_dev.py with --cucim-wheel and "
                "--cucim-sha256."
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return ImplementationLibraryProbeResult(
            "cucim",
            False,
            reason_code="cucim_provenance_invalid",
            message=(
                f"The GPU environment record at {path} is invalid: {exc}. "
                "Re-run scripts/setup_gpu_dev.py with a verified cuCIM wheel."
            ),
        )
    if provenance is None:
        return ImplementationLibraryProbeResult(
            "cucim",
            False,
            reason_code="cucim_provenance_unverified",
            message=(
                "The completed GPU setup did not approve a checksum-verified "
                "cuCIM wheel. Re-run scripts/setup_gpu_dev.py with "
                "--cucim-wheel and --cucim-sha256."
            ),
        )

    try:
        distribution_version = _verify_installed_cucim_provenance(provenance)
    except _InstalledProvenanceError as exc:
        return ImplementationLibraryProbeResult(
            "cucim",
            False,
            reason_code=exc.reason_code,
            message=(
                f"{exc} Re-run scripts/setup_gpu_dev.py with the verified cuCIM wheel."
            ),
        )

    metadata = (
        ("environment_record_schema", _GPU_ENVIRONMENT_RECORD_SCHEMA),
        (
            "environment_record_schema_version",
            str(_GPU_ENVIRONMENT_RECORD_SCHEMA_VERSION),
        ),
        ("environment_track", provenance.track),
        ("cupy_distribution", provenance.cupy_distribution),
        ("cucim_distribution", provenance.distribution),
        ("cucim_distribution_version", distribution_version),
        ("cucim_artifact_sha256", provenance.wheel_sha256),
    )

    cucim = importlib.import_module("cucim")
    restoration = importlib.import_module("cucim.skimage.restoration")
    is_available = getattr(cucim, "is_available", None)
    if callable(is_available) and not bool(is_available("skimage")):
        return ImplementationLibraryProbeResult(
            "cucim",
            False,
            version=str(getattr(cucim, "__version__", "")),
            reason_code="cucim_skimage_unavailable",
            message="cuCIM reports that its skimage component is unavailable.",
            metadata=metadata,
        )
    if not callable(getattr(restoration, "rolling_ball", None)):
        return ImplementationLibraryProbeResult(
            "cucim",
            False,
            version=str(getattr(cucim, "__version__", "")),
            reason_code="cucim_rolling_ball_missing",
            message="cuCIM restoration does not expose rolling_ball.",
            metadata=metadata,
        )
    cupy = importlib.import_module("cupy")
    device_id = _cupy_current_device_id(cupy)
    with accelerator_lease("cuda-cupy", device_id):
        try:
            measure, regionprops_euler = _load_and_warm_cucim_measurement_caches(
                cupy,
                device_id=device_id,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            return ImplementationLibraryProbeResult(
                "cucim",
                False,
                version=str(getattr(cucim, "__version__", "")),
                reason_code="cucim_measurements_api_missing",
                message=(
                    "cuCIM could not initialize the reviewed regionprops_table "
                    "and private robust Euler APIs required by GPU Measurements: "
                    f"{exc}"
                ),
                metadata=metadata,
            )
        return _exercise_cucim_skimage_library(
            cupy,
            cucim,
            restoration,
            measure,
            regionprops_euler,
            metadata,
        )


def _load_and_warm_cucim_measurement_caches(
    cupy: object,
    *,
    device_id: str,
) -> tuple[object, object]:
    """Prime cuCIM's process-lifetime 2D/3D measurement device caches.

    cuCIM 26.06.00 retains tiny region-property and robust-Euler lookup arrays
    from the allocator active on first use.  If first use occurs inside a VIPP
    transaction, that private pool can never prove zero live allocations.  A
    separate retained pool makes the library-owned lifetime explicit without
    touching CuPy's external default pool; subsequent transaction pools drain
    completely.  The bounded cache is keyed by exact modules, API functions,
    and device.
    """

    key = (id(cupy), str(device_id))
    with _CUCIM_MEASUREMENT_CACHE_LOCK:
        existing = _CUCIM_MEASUREMENT_CACHE_POOLS.get(key)
        if existing is not None:
            existing_cupy, measure, regionprops_euler, _pool = existing
            if existing_cupy is cupy:
                return measure, regionprops_euler
        pool = cupy.cuda.MemoryPool()
        labels = properties = euler = None
        try:
            with cupy.cuda.using_allocator(pool.malloc):
                measure = importlib.import_module("cucim.skimage.measure")
                measurement_kernels = importlib.import_module(
                    "cucim.skimage.measure._regionprops_gpu_misc_kernels"
                )
                regionprops_table = getattr(measure, "regionprops_table", None)
                regionprops_euler = getattr(
                    measurement_kernels,
                    "regionprops_euler",
                    None,
                )
                if not callable(regionprops_table) or not callable(
                    regionprops_euler
                ):
                    raise AttributeError(
                        "required measurement callables are unavailable"
                    )
                for host_labels in (
                    _cucim_probe_labels_2d(),
                    _cucim_probe_labels_3d(),
                ):
                    labels = cupy.asarray(host_labels)
                    properties = regionprops_table(
                        labels,
                        properties=("label", "num_pixels", "bbox", "centroid"),
                        batch_processing=True,
                    )
                    euler = regionprops_euler(
                        labels,
                        connectivity=None,
                        max_label=2,
                        robust=True,
                    )
                    float(cupy.asarray(properties["num_pixels"]).sum().item())
                    float(cupy.asarray(euler).sum().item())
                    cupy.cuda.get_current_stream().synchronize()
                    labels = properties = euler = None
            gc.collect()
            pool.free_all_blocks()
            used = int(pool.used_bytes())
            reserved = int(pool.total_bytes())
            if used > _CUCIM_MEASUREMENT_CACHE_MAX_BYTES or reserved > (
                _CUCIM_MEASUREMENT_CACHE_MAX_BYTES
            ):
                raise RuntimeError(
                    "cuCIM retained an unexpectedly large measurement cache "
                    f"(used={used}, reserved={reserved})."
                )
            _CUCIM_MEASUREMENT_CACHE_POOLS[key] = (
                cupy,
                measure,
                regionprops_euler,
                pool,
            )
            return measure, regionprops_euler
        except BaseException:
            labels = properties = euler = None
            gc.collect()
            try:
                pool.free_all_blocks()
            except BaseException:
                pass
            raise


def _cucim_probe_labels_2d() -> np.ndarray:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[5:7, 4:7] = 2
    return labels


def _cucim_probe_labels_3d() -> np.ndarray:
    labels = np.zeros((5, 8, 8), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1
    labels[2:5, 5:7, 4:7] = 2
    return labels


def _exercise_cucim_skimage_library(
    cupy: object,
    cucim: object,
    restoration: object,
    measure: object,
    regionprops_euler: object,
    metadata: tuple[tuple[str, str], ...],
) -> ImplementationLibraryProbeResult:
    pool = cupy.cuda.MemoryPool()
    values = background = labels = properties = euler = None
    probe_error: BaseException | None = None
    try:
        with cupy.cuda.using_allocator(pool.malloc):
            values = cupy.arange(81, dtype=cupy.float32).reshape(9, 9)
            background = restoration.rolling_ball(values, radius=2)
            float(background.sum().item())
            labels = cupy.asarray(
                np.asarray(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0],
                        [0, 1, 1, 0, 2],
                        [0, 0, 0, 0, 2],
                        [0, 0, 0, 0, 0],
                    ],
                    dtype=np.int32,
                )
            )
            properties = measure.regionprops_table(
                labels,
                properties=("label", "num_pixels", "bbox", "centroid"),
                batch_processing=True,
            )
            required = {
                "label",
                "num_pixels",
                "bbox-0",
                "bbox-1",
                "bbox-2",
                "bbox-3",
                "centroid-0",
                "centroid-1",
            }
            if not required.issubset(properties):
                raise RuntimeError(
                    "cuCIM measurement probe omitted required region properties."
                )
            euler = regionprops_euler(
                labels,
                connectivity=None,
                max_label=2,
                robust=True,
            )
            if tuple(int(size) for size in cupy.asarray(euler).shape) != (2,):
                raise RuntimeError(
                    "cuCIM measurement probe returned malformed Euler values."
                )
            float(cupy.asarray(properties["num_pixels"]).sum().item())
            float(cupy.asarray(euler).sum().item())
            cupy.cuda.get_current_stream().synchronize()
        return ImplementationLibraryProbeResult(
            "cucim",
            True,
            version=str(getattr(cucim, "__version__", "")),
            message=(
                "cuCIM completed synchronized rolling-ball, region-properties, "
                "and robust-Euler probes."
            ),
            metadata=metadata,
        )
    except BaseException as exc:
        probe_error = exc
        raise
    finally:
        values = background = labels = properties = euler = None
        _drain_private_probe_pool(
            cupy,
            pool,
            library_id="cucim",
            suppress_errors=probe_error is not None,
        )


def _cupy_current_device_id(cupy: object) -> str:
    cuda = getattr(cupy, "cuda", None)
    runtime = getattr(cuda, "runtime", None)
    get_device = getattr(runtime, "getDevice", None)
    if callable(get_device):
        index = int(get_device())
    else:
        device_factory = getattr(cuda, "Device", None)
        if callable(device_factory):
            try:
                index = int(device_factory().id)
            except TypeError:
                index = 0
        else:
            index = 0
    if index < 0:
        raise ValueError(f"CUDA returned an invalid current device index: {index}.")
    return f"cuda:{index}"


_GPU_ENVIRONMENT_RECORD_SCHEMA = "napari-vipp-gpu-environment"
_GPU_ENVIRONMENT_RECORD_SCHEMA_VERSION = 1
_GPU_ENVIRONMENT_RECORD_RELATIVE_PATH = (
    Path("share") / "napari-vipp" / "gpu-environment.json"
)
_GPU_ENVIRONMENT_RECORD_KEYS = frozenset(
    {"schema", "schema_version", "track", "cupy_distribution", "cucim"}
)
_CUCIM_RECORD_KEYS = frozenset({"distribution", "wheel_sha256"})
_TRACK_DISTRIBUTIONS = {
    "cuda12": "cupy-cuda12x",
    "cuda13": "cupy-cuda13x",
}


@dataclass(frozen=True, slots=True)
class _CucimEnvironmentProvenance:
    track: str
    cupy_distribution: str
    distribution: str
    wheel_sha256: str


class _InstalledProvenanceError(ValueError):
    """An actionable mismatch between a setup marker and installed packages."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _gpu_environment_record_path(*, prefix: Path | None = None) -> Path:
    """Return the record location for an interpreter, with test injection."""

    root = Path(sys.prefix) if prefix is None else Path(prefix)
    return root / _GPU_ENVIRONMENT_RECORD_RELATIVE_PATH


def _read_cucim_environment_provenance(
    path: Path,
) -> _CucimEnvironmentProvenance | None:
    """Read a strict setup marker and return its optional cuCIM approval."""

    if path.stat().st_size > 64 * 1024:
        raise ValueError("record exceeds the 64 KiB size limit")
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream, object_pairs_hook=_unique_json_object)
    if not isinstance(document, dict):
        raise ValueError("record root must be a JSON object")
    if set(document) != _GPU_ENVIRONMENT_RECORD_KEYS:
        raise ValueError("record fields do not match schema version 1")
    if document["schema"] != _GPU_ENVIRONMENT_RECORD_SCHEMA:
        raise ValueError("record schema identifier is not supported")
    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("record schema_version must be integer 1")
    track = document["track"]
    cupy_distribution = document["cupy_distribution"]
    if not isinstance(track, str) or track not in _TRACK_DISTRIBUTIONS:
        raise ValueError("record track must be cuda12 or cuda13")
    if cupy_distribution != _TRACK_DISTRIBUTIONS[track]:
        raise ValueError("record CuPy distribution does not match its track")

    cucim = document["cucim"]
    if cucim is None:
        return None
    if track != "cuda13":
        raise ValueError("verified cuCIM provenance is valid only for cuda13")
    if not isinstance(cucim, dict) or set(cucim) != _CUCIM_RECORD_KEYS:
        raise ValueError("record cucim fields do not match schema version 1")
    distribution = cucim["distribution"]
    if distribution != "cucim-cu13":
        raise ValueError("record cuCIM distribution must be cucim-cu13")
    digest = cucim["wheel_sha256"]
    if (
        not isinstance(digest, str)
        or re.fullmatch(
            r"[0-9a-fA-F]{64}",
            digest,
        )
        is None
    ):
        raise ValueError("record cuCIM wheel_sha256 must be 64 hexadecimal digits")
    return _CucimEnvironmentProvenance(
        track=track,
        cupy_distribution=cupy_distribution,
        distribution=distribution,
        wheel_sha256=digest.lower(),
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _verify_installed_cucim_provenance(
    provenance: _CucimEnvironmentProvenance,
) -> str:
    """Verify that installed packages still match the completed setup marker."""

    try:
        cupy_distributions = _installed_cupy_distribution_names()
    except Exception as exc:
        raise _InstalledProvenanceError(
            "cucim_provenance_stale",
            f"Could not verify the installed CuPy distribution: {exc}.",
        ) from exc
    expected_cupy = _canonical_distribution_name(provenance.cupy_distribution)
    if cupy_distributions != (expected_cupy,):
        rendered = ", ".join(cupy_distributions) if cupy_distributions else "none"
        raise _InstalledProvenanceError(
            "cucim_provenance_stale",
            "The GPU environment record approves "
            f"{expected_cupy}, but the installed CuPy distributions are {rendered}.",
        )

    try:
        distribution = importlib.metadata.distribution(provenance.distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise _InstalledProvenanceError(
            "cucim_provenance_stale",
            "The GPU environment record approves "
            f"{provenance.distribution}, but that distribution is not installed.",
        ) from exc
    except Exception as exc:
        raise _InstalledProvenanceError(
            "cucim_provenance_stale",
            f"Could not inspect the installed {provenance.distribution}: {exc}.",
        ) from exc

    installed_name = _canonical_distribution_name(distribution.metadata.get("Name", ""))
    if installed_name != _canonical_distribution_name(provenance.distribution):
        raise _InstalledProvenanceError(
            "cucim_provenance_stale",
            "The installed cuCIM distribution metadata does not match the "
            "approved name.",
        )
    version = str(distribution.version).strip()
    if not version:
        raise _InstalledProvenanceError(
            "cucim_provenance_stale",
            "The installed cuCIM distribution does not report a version.",
        )

    try:
        direct_url = distribution.read_text("direct_url.json")
    except (OSError, UnicodeError) as exc:
        raise _InstalledProvenanceError(
            "cucim_artifact_unverified",
            f"Could not read installed cuCIM PEP 610 provenance: {exc}.",
        ) from exc
    if not direct_url:
        raise _InstalledProvenanceError(
            "cucim_artifact_unverified",
            "The installed cuCIM distribution has no PEP 610 archive provenance.",
        )
    try:
        installed_digest = _pep610_archive_sha256(direct_url)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _InstalledProvenanceError(
            "cucim_artifact_unverified",
            f"Installed cuCIM PEP 610 provenance is invalid: {exc}.",
        ) from exc
    if installed_digest != provenance.wheel_sha256:
        raise _InstalledProvenanceError(
            "cucim_artifact_mismatch",
            "The installed cuCIM archive SHA-256 does not match the approved wheel "
            f"(expected {provenance.wheel_sha256}, found {installed_digest}).",
        )
    return version


def _installed_cupy_distribution_names() -> tuple[str, ...]:
    names: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = _canonical_distribution_name(distribution.metadata.get("Name", ""))
        if (
            name == "cupy"
            or name == "amd-cupy"
            or name.startswith("cupy-cuda")
            or name.startswith("cupy-rocm")
        ):
            names.add(name)
    return tuple(sorted(names))


def _canonical_distribution_name(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value)).lower()


def _pep610_archive_sha256(document_text: str) -> str:
    document = json.loads(document_text, object_pairs_hook=_unique_json_object)
    if not isinstance(document, dict):
        raise ValueError("direct_url.json root must be a JSON object")
    archive_info = document.get("archive_info")
    if not isinstance(archive_info, dict):
        raise ValueError("direct_url.json has no archive_info object")

    candidates: list[str] = []
    hashes = archive_info.get("hashes")
    if hashes is not None:
        if not isinstance(hashes, dict):
            raise ValueError("archive_info.hashes must be a JSON object")
        candidate = hashes.get("sha256")
        if candidate is not None:
            candidates.append(str(candidate).lower())
    legacy_hash = archive_info.get("hash")
    if legacy_hash is not None:
        match = re.fullmatch(r"sha256=([0-9a-fA-F]{64})", str(legacy_hash))
        if match is None:
            raise ValueError("archive_info.hash must contain a SHA-256 digest")
        candidates.append(match.group(1).lower())
    if not candidates or any(
        re.fullmatch(r"[0-9a-f]{64}", candidate) is None for candidate in candidates
    ):
        raise ValueError("archive_info does not contain a valid SHA-256 digest")
    if len(set(candidates)) != 1:
        raise ValueError("archive_info SHA-256 fields disagree")
    return candidates[0]


def _drain_private_probe_pool(
    cupy,
    pool,
    *,
    library_id: str,
    suppress_errors: bool = False,
) -> None:
    """Synchronize and release only the allocation pool owned by a probe."""

    cleanup_errors: list[BaseException] = []
    try:
        cupy.cuda.get_current_stream().synchronize()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        pool.free_all_blocks()
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        used = int(pool.used_bytes())
        reserved = int(pool.total_bytes())
        if used or reserved:
            cleanup_errors.append(
                RuntimeError(
                    f"{library_id} probe private memory pool did not drain "
                    f"(used={used}, reserved={reserved})."
                )
            )
    except BaseException as exc:
        cleanup_errors.append(exc)
    if cleanup_errors and not suppress_errors:
        primary = cleanup_errors[0]
        for additional in cleanup_errors[1:]:
            primary.add_note(
                "Additional private-pool cleanup failure: "
                f"{type(additional).__name__}: {additional}"
            )
        raise primary


def _load_ref(reference: str):
    module_name, attribute_path = reference.split(":", 1)
    value = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    return value


def _validate_runtime_instance(
    instance: object,
    descriptor: RuntimeDescriptor,
) -> None:
    if not isinstance(instance, RuntimeProtocol):
        raise TypeError(
            "Runtime factory returned an object that violates RuntimeProtocol."
        )
    if str(instance.runtime_id) != descriptor.runtime_id:
        raise ValueError("Runtime instance ID does not match its descriptor.")
    if str(instance.array_domain) != descriptor.array_domain:
        raise ValueError("Runtime instance array domain does not match its descriptor.")


def _unique_by_id(values: Sequence, description: str, field_name: str) -> dict:
    result = {}
    for value in values:
        if not hasattr(value, field_name):
            raise TypeError(f"{description} descriptors have an invalid type.")
        identifier = getattr(value, field_name)
        if identifier in result:
            raise ValueError(f"Duplicate {description} ID {identifier!r}.")
        result[identifier] = value
    return result


def _close_quietly(instance: object) -> None:
    close = getattr(instance, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


__all__ = [
    "CUCIM_SKIMAGE_LIBRARY",
    "CUDA_CUPY_RUNTIME",
    "CUPY_LIBRARY",
    "CUPYX_LIBRARY",
    "ComputeRegistry",
    "ComputeRegistryClosed",
    "ComputeRegistryError",
    "ComputeRegistryLoadError",
    "DEFAULT_LIBRARY_DESCRIPTORS",
    "DEFAULT_RUNTIME_DESCRIPTORS",
    "ImplementationLibraryDescriptor",
    "ImplementationLibraryProbe",
    "ImplementationLibraryProbeResult",
    "RuntimeDescriptor",
    "RuntimeDevice",
    "RuntimeExceptionInfo",
    "RuntimeExceptionKind",
    "RuntimeFactory",
    "RuntimeMemorySnapshot",
    "RuntimeProbeResult",
    "RuntimeProtocol",
    "validate_registry",
]
