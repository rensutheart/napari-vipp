"""Qt-free compute-backend capability and selection contracts.

This module deliberately does not change pipeline execution.  It provides the
small, dependency-free contract needed to discover an optional GPU provider and
to decide whether a future operation-level implementation may use it.  Optional
GPU packages are imported only when :func:`detect_compute_capabilities` runs.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum


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
        supported_operation_ids=_normalized_operation_ids(
            supported_cpu_operation_ids
        ),
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
    report = capabilities or detect_compute_capabilities()

    if requested_backend is ComputeBackend.CPU:
        return BackendSelection(
            requested=requested_backend,
            resolved=ComputeBackend.CPU,
            operation_id=operation_id,
            fell_back=False,
            reason="CPU was explicitly requested.",
        )

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
            fell_back=True,
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
    "BackendSelection",
    "ComputeBackend",
    "ComputeBackendUnavailable",
    "ComputeCapabilityReport",
    "detect_compute_capabilities",
    "select_compute_backend",
]
