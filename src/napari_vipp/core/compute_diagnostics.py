"""Headless diagnostics for VIPP's optional GPU compute environment.

Run with ``python -m napari_vipp.core.compute_diagnostics``.  Importing this
module never imports CuPy, cuCIM, Qt, or napari.  An explicit diagnostic run
separately answers three questions:

* can the CUDA/CuPy runtime start;
* are the standard and optional implementation libraries usable; and
* which current VIPP GPU operation regions are publicly admitted here.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import platform
import re
import struct
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from napari_vipp.core.atomic_io import atomic_write_json
from napari_vipp.core.compute import ComputeEnvironment, ExecutionReport, MemoryTopology
from napari_vipp.core.compute_policy import evaluate_candidate_environment_support
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
    RuntimeProtocol,
)
from napari_vipp.core.compute_specs import (
    OperationComputeSpec,
    accelerator_compute_specs,
)

_CUPY_NAMES = ("cupy", "amd-cupy")
_STANDARD_LIBRARY_IDS = frozenset({"cupy", "cupyx"})
_OPTIONAL_LIBRARY_ID = "cucim"
_SUPPORT_SCHEMA = "napari-vipp-compute-support-bundle"
_SUPPORT_SCHEMA_VERSION = 1
_SUPPORT_PRIVACY_POLICY = "napari-vipp-compute-support-redaction-v1"
_CUCIM_GUIDE_URL = (
    "https://github.com/rensutheart/napari-vipp/blob/main/"
    "scripts/README-cucim-windows-installer.md"
)
_GPU_GUIDE_URL = (
    "https://github.com/rensutheart/napari-vipp/blob/main/docs/gpu-guide.md"
)


class DoctorStatus(StrEnum):
    """High-level result of a compute environment inspection."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    MISCONFIGURED = "misconfigured"


@dataclass(frozen=True, slots=True)
class PackageRecord:
    """One relevant installed distribution without importing its package."""

    name: str
    version: str

    def __post_init__(self) -> None:
        name = _canonical_name(self.name)
        if not name:
            raise ValueError("Package name must not be empty.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", str(self.version).strip())

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class RepairGuidance:
    """One prioritized, plain-language next action."""

    action_id: str
    title: str
    summary: str
    command: str = ""
    documentation_url: str = ""
    optional: bool = False

    def __post_init__(self) -> None:
        for name in ("action_id", "title", "summary"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        for name in ("command", "documentation_url"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if not isinstance(self.optional, bool):
            raise TypeError("optional must be a boolean.")

    def as_dict(self, *, include_command: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "action_id": self.action_id,
            "title": self.title,
            "summary": self.summary,
            "documentation_url": self.documentation_url,
            "optional": self.optional,
        }
        if include_command:
            payload["command"] = self.command
        return payload


@dataclass(frozen=True, slots=True)
class PublicAdmissionRegion:
    """Environment admission for one current public GPU declaration."""

    operation_id: str
    implementation_id: str
    implementation_version: str
    implementation_library_id: str
    admission_tier: str
    environment_policy_id: str
    admitted: bool
    reason_code: str
    reason: str
    supported_spatial_ndims: tuple[int, ...]
    public_input_dtypes: tuple[tuple[str, ...], ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "implementation_id",
            "implementation_version",
            "implementation_library_id",
            "admission_tier",
            "environment_policy_id",
            "reason_code",
            "reason",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        if not isinstance(self.admitted, bool):
            raise TypeError("admitted must be a boolean.")
        ndims = tuple(self.supported_spatial_ndims)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in ndims
        ):
            raise ValueError("supported_spatial_ndims must contain positive integers.")
        object.__setattr__(self, "supported_spatial_ndims", ndims)
        dtypes = tuple(
            tuple(str(value).strip() for value in port)
            for port in self.public_input_dtypes
        )
        if any(not port or any(not value for value in port) for port in dtypes):
            raise ValueError(
                "public_input_dtypes must contain non-empty dtype regions."
            )
        object.__setattr__(self, "public_input_dtypes", dtypes)
        object.__setattr__(
            self,
            "limitations",
            tuple(
                str(value).strip()
                for value in self.limitations
                if str(value).strip()
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "implementation_library_id": self.implementation_library_id,
            "admission_tier": self.admission_tier,
            "environment_policy_id": self.environment_policy_id,
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "supported_spatial_ndims": list(self.supported_spatial_ndims),
            "public_input_dtypes": [list(port) for port in self.public_input_dtypes],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ComputeDoctorReport:
    """Stable JSON-safe compute doctor result."""

    status: DoctorStatus
    reason_code: str
    summary: str
    platform: str
    execution_mode: str
    python: str
    packages: tuple[PackageRecord, ...]
    track: str
    repair_command: str = ""
    runtime_probe: RuntimeProbeResult | None = None
    memory_snapshot: RuntimeMemorySnapshot | None = None
    details: tuple[str, ...] = ()
    library_probes: tuple[ImplementationLibraryProbeResult, ...] = ()
    admission_regions: tuple[PublicAdmissionRegion, ...] = ()
    guidance: RepairGuidance | None = None
    environment: ComputeEnvironment | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, DoctorStatus)
            else DoctorStatus(str(self.status).strip().lower())
        )
        object.__setattr__(self, "status", status)
        for name in (
            "reason_code",
            "summary",
            "platform",
            "execution_mode",
            "python",
            "track",
            "repair_command",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        packages = tuple(self.packages)
        if any(not isinstance(package, PackageRecord) for package in packages):
            raise TypeError("packages must contain PackageRecord values.")
        object.__setattr__(self, "packages", packages)
        object.__setattr__(
            self,
            "details",
            tuple(
                str(detail).strip()
                for detail in self.details
                if str(detail).strip()
            ),
        )
        probes = tuple(self.library_probes)
        if any(
            not isinstance(probe, ImplementationLibraryProbeResult)
            for probe in probes
        ):
            raise TypeError(
                "library_probes must contain ImplementationLibraryProbeResult values."
            )
        if len({probe.library_id for probe in probes}) != len(probes):
            raise ValueError("library probe IDs must be unique.")
        object.__setattr__(self, "library_probes", probes)
        regions = tuple(self.admission_regions)
        if any(not isinstance(region, PublicAdmissionRegion) for region in regions):
            raise TypeError(
                "admission_regions must contain PublicAdmissionRegion values."
            )
        identities = tuple(region.implementation_id for region in regions)
        if len(set(identities)) != len(identities):
            raise ValueError("admission implementation IDs must be unique.")
        object.__setattr__(self, "admission_regions", regions)
        if self.guidance is not None and not isinstance(self.guidance, RepairGuidance):
            raise TypeError("guidance must be RepairGuidance or None.")
        if self.environment is not None and not isinstance(
            self.environment, ComputeEnvironment
        ):
            raise TypeError("environment must be ComputeEnvironment or None.")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("schema_version must be a positive integer.")

    @property
    def available(self) -> bool:
        """Whether the standard public CUDA route passed, excluding optional cuCIM."""

        return self.status is DoctorStatus.AVAILABLE

    @property
    def cuda_ready(self) -> bool:
        return bool(self.runtime_probe is not None and self.runtime_probe.available)

    @property
    def cucim_probe(self) -> ImplementationLibraryProbeResult | None:
        return next(
            (
                probe
                for probe in self.library_probes
                if probe.library_id == _OPTIONAL_LIBRARY_ID
            ),
            None,
        )

    @property
    def admitted_regions(self) -> tuple[PublicAdmissionRegion, ...]:
        return tuple(region for region in self.admission_regions if region.admitted)

    def as_dict(self) -> dict[str, object]:
        """Return the local diagnostic contract.

        This compatibility-oriented representation can contain a local repair
        command.  Use :func:`build_compute_support_bundle` for a shareable,
        privacy-redacted document.
        """

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "available": self.available,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "platform": self.platform,
            "execution_mode": self.execution_mode,
            "python": self.python,
            "packages": [package.as_dict() for package in self.packages],
            "track": self.track,
            "repair_command": self.repair_command,
            "runtime_probe": (
                self.runtime_probe.as_dict() if self.runtime_probe else None
            ),
            "memory_snapshot": (
                self.memory_snapshot.as_dict() if self.memory_snapshot else None
            ),
            "library_probes": [probe.as_dict() for probe in self.library_probes],
            "public_admission": {
                "admitted_count": len(self.admitted_regions),
                "total_count": len(self.admission_regions),
                "regions": [region.as_dict() for region in self.admission_regions],
            },
            "guidance": self.guidance.as_dict() if self.guidance else None,
            "details": list(self.details),
        }


def collect_compute_diagnostics(
    *,
    track: str = "auto",
    runtime: RuntimeProtocol | None = None,
    registry: ComputeRegistry | None = None,
    packages: Sequence[PackageRecord | tuple[str, str]] | None = None,
    library_probes: Sequence[ImplementationLibraryProbeResult] | None = None,
    implementation_specs: Sequence[OperationComputeSpec] | None = None,
    scientific_stack_versions: Sequence[tuple[str, str]] | None = None,
    platform_name: str | None = None,
    python_implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
    python_abi: str | None = None,
    pointer_bits: int | None = None,
    refresh: bool = False,
) -> ComputeDoctorReport:
    """Inspect CUDA, implementation libraries, and public VIPP admission.

    Injected runtimes and registries remain caller-owned.  ``library_probes``
    and ``implementation_specs`` are deterministic test/embedding seams; normal
    callers should let the doctor use the built-in registry and live catalog.
    """

    if runtime is not None and registry is not None:
        raise ValueError("runtime and registry cannot both be supplied.")
    requested_track = str(track).strip().lower()
    if requested_track not in {"auto", "cuda12", "cuda13"}:
        raise ValueError("track must be 'auto', 'cuda12', or 'cuda13'.")
    current_platform = sys.platform if platform_name is None else str(platform_name)
    implementation = (
        platform.python_implementation()
        if python_implementation is None
        else str(python_implementation)
    )
    version = (
        tuple(sys.version_info[:2])
        if python_version is None
        else tuple(python_version)
    )
    bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    package_records = _normalize_packages(
        installed_gpu_packages() if packages is None else packages
    )
    python_label = f"{implementation} {version[0]}.{version[1]} ({bits}-bit)"
    execution_mode = _execution_mode(current_platform)
    cupy_packages = tuple(
        package for package in package_records if _is_cupy_distribution(package.name)
    )
    detected_tracks = {
        detected
        for package in cupy_packages
        if (detected := _distribution_track(package.name)) is not None
    }
    selected_track = (
        requested_track
        if requested_track != "auto"
        else (next(iter(detected_tracks)) if len(detected_tracks) == 1 else "cuda13")
    )
    repair = _repair_command(current_platform, selected_track)

    common = {
        "platform": current_platform,
        "execution_mode": execution_mode,
        "python": python_label,
        "packages": package_records,
        "track": selected_track,
    }
    if len(cupy_packages) > 1:
        names = ", ".join(package.name for package in cupy_packages)
        guidance = RepairGuidance(
            "rebuild_standard_cuda_environment",
            "Create a clean VIPP GPU environment",
            (
                "More than one CuPy package is installed. Keep using CPU until "
                "a clean, single-track environment is ready."
            ),
            command=repair,
            documentation_url=_GPU_GUIDE_URL,
        )
        return ComputeDoctorReport(
            status=DoctorStatus.MISCONFIGURED,
            reason_code="mixed_cupy_distributions",
            summary=f"Multiple CuPy distributions are installed: {names}.",
            repair_command=repair,
            details=("Create a fresh dedicated VIPP GPU environment.",),
            guidance=guidance,
            **common,
        )

    if current_platform != "win32" and not current_platform.startswith("linux"):
        return ComputeDoctorReport(
            status=DoctorStatus.UNSUPPORTED,
            reason_code="platform_unsupported",
            summary=(
                "NVIDIA CUDA acceleration is not available in this VIPP build "
                "on this platform."
            ),
            details=("VIPP remains fully usable with CPU processing.",),
            guidance=RepairGuidance(
                "use_cpu_on_this_platform",
                "Continue with CPU processing",
                (
                    "No CUDA repair is needed. VIPP will use its portable CPU "
                    "implementations."
                ),
                documentation_url=_GPU_GUIDE_URL,
            ),
            **common,
        )

    if implementation != "CPython" or version != (3, 12) or bits != 64:
        return ComputeDoctorReport(
            status=DoctorStatus.UNSUPPORTED,
            reason_code="python_unsupported",
            summary="The public GPU environment requires 64-bit CPython 3.12.",
            repair_command=repair,
            guidance=RepairGuidance(
                "install_supported_python_environment",
                "Use the supported VIPP environment",
                (
                    "Keep using CPU here, or create the dedicated 64-bit "
                    "CPython 3.12 GPU environment."
                ),
                command=repair,
                documentation_url=_GPU_GUIDE_URL,
            ),
            **common,
        )

    if requested_track != "auto" and cupy_packages:
        installed_track = _distribution_track(cupy_packages[0].name)
        if installed_track != requested_track:
            return ComputeDoctorReport(
                status=DoctorStatus.MISCONFIGURED,
                reason_code="cupy_track_mismatch",
                summary=(
                    f"{cupy_packages[0].name} does not match the requested "
                    f"{requested_track} track."
                ),
                repair_command=repair,
                details=("Use a separate environment for each CUDA-major track.",),
                guidance=RepairGuidance(
                    "rebuild_requested_cuda_track",
                    "Create the requested CUDA-track environment",
                    "Do not mix CUDA-major packages in one VIPP environment.",
                    command=repair,
                    documentation_url=_GPU_GUIDE_URL,
                ),
                **common,
            )

    specs = tuple(
        accelerator_compute_specs()
        if implementation_specs is None
        else implementation_specs
    )
    if any(not isinstance(spec, OperationComputeSpec) for spec in specs):
        raise TypeError(
            "implementation_specs must contain OperationComputeSpec values."
        )
    public_specs = tuple(
        spec for spec in specs if spec.visible_for(allow_experimental=False)
    )
    owned_registry = registry is None and runtime is None
    diagnostic_registry = ComputeRegistry() if owned_registry else registry
    details: list[str] = []
    probe: RuntimeProbeResult
    snapshot: RuntimeMemorySnapshot | None = None
    cleanup_failure = ""
    try:
        if runtime is not None:
            probe = _probe_runtime(runtime, refresh=refresh)
            if probe.available:
                snapshot = _memory_snapshot(runtime, probe, details)
        else:
            assert diagnostic_registry is not None
            probe = _probe_registry_runtime(diagnostic_registry, refresh=refresh)
            if probe.available:
                try:
                    with _provider_output_guard():
                        runtime_instance = diagnostic_registry.runtime("cuda-cupy")
                    snapshot = _memory_snapshot(runtime_instance, probe, details)
                except Exception as exc:
                    details.append("Memory snapshot failed: " + _exception_summary(exc))

        if library_probes is not None:
            resolved_library_probes = tuple(library_probes)
        elif diagnostic_registry is not None:
            resolved_library_probes = _probe_libraries(
                diagnostic_registry,
                runtime_available=probe.available,
                refresh=refresh,
            )
        else:
            resolved_library_probes = _legacy_library_probes(
                probe,
                package_records,
            )

        environment = _diagnostic_environment(
            probe,
            resolved_library_probes,
            platform_name=current_platform,
            execution_mode=execution_mode,
            python_implementation=implementation,
            python_version=version,
            python_abi=python_abi,
            scientific_stack_versions=scientific_stack_versions,
        )
        regions = _public_admission_regions(public_specs, environment)
    finally:
        if owned_registry and diagnostic_registry is not None:
            try:
                with _provider_output_guard():
                    diagnostic_registry.close()
            except Exception as exc:
                cleanup_failure = _exception_summary(exc)
                details.append("Runtime cleanup failed: " + cleanup_failure)

    status, reason_code, summary = _overall_result(
        probe,
        resolved_library_probes,
        regions,
        current_platform=current_platform,
        selected_track=selected_track,
        cleanup_failure=cleanup_failure,
    )
    guidance = _repair_guidance(
        status,
        probe,
        resolved_library_probes,
        regions,
        package_records,
        repair_command=repair,
    )
    return ComputeDoctorReport(
        status=status,
        reason_code=reason_code,
        summary=summary,
        repair_command=(
            guidance.command if guidance is not None and guidance.command else ""
        ),
        runtime_probe=probe,
        memory_snapshot=snapshot,
        details=tuple(details),
        library_probes=resolved_library_probes,
        admission_regions=regions,
        guidance=guidance,
        environment=environment,
        **common,
    )


def installed_gpu_packages() -> tuple[PackageRecord, ...]:
    """List relevant distributions using metadata only, never package imports."""

    records: list[PackageRecord] = []
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name", "")
        name = _canonical_name(raw_name)
        if _is_gpu_distribution(name):
            records.append(PackageRecord(name, str(distribution.version)))
    return _normalize_packages(records)


def build_compute_support_bundle(
    report: ComputeDoctorReport,
    *,
    recent_execution: ExecutionReport | None = None,
    generated_utc: str | None = None,
) -> dict[str, object]:
    """Build an allowlisted support document without private local identity."""

    if not isinstance(report, ComputeDoctorReport):
        raise TypeError("report must be a ComputeDoctorReport.")
    if recent_execution is not None and not isinstance(
        recent_execution, ExecutionReport
    ):
        raise TypeError("recent_execution must be ExecutionReport or None.")
    generated = (
        datetime.now(UTC).isoformat()
        if generated_utc is None
        else str(generated_utc).strip()
    )
    if not generated:
        raise ValueError("generated_utc must not be empty.")
    try:
        generated_datetime = datetime.fromisoformat(generated)
    except ValueError as exc:
        raise ValueError("generated_utc must be an ISO-8601 timestamp.") from exc
    if (
        generated_datetime.tzinfo is None
        or generated_datetime.utcoffset() != timedelta(0)
    ):
        raise ValueError("generated_utc must use a UTC offset.")

    probe = report.runtime_probe
    selected_device = _selected_device(probe)
    runtime_payload: dict[str, object] = {
        "status": "ready" if report.cuda_ready else "unavailable",
        "reason_code": probe.reason_code if probe is not None else "not_checked",
        "message": _redact_private_text(probe.message) if probe is not None else "",
        "version": probe.version if probe is not None else "",
        "driver_version": "",
        "cuda_runtime_version": "",
        "device": None,
        "memory": (
            {
                "runtime_id": report.memory_snapshot.runtime_id,
                "device_id": report.memory_snapshot.device_id,
                "topology": report.memory_snapshot.topology.value,
                "device_total_bytes": report.memory_snapshot.device_total_bytes,
                "device_free_bytes": report.memory_snapshot.device_free_bytes,
                "runtime_live_bytes": report.memory_snapshot.runtime_live_bytes,
                "runtime_reserved_bytes": (
                    report.memory_snapshot.runtime_reserved_bytes
                ),
                "out_of_pool_bytes": report.memory_snapshot.out_of_pool_bytes,
            }
            if report.memory_snapshot
            else None
        ),
    }
    if probe is not None:
        runtime_metadata = dict(probe.metadata)
        runtime_payload["driver_version"] = runtime_metadata.get("driver_version", "")
        runtime_payload["cuda_runtime_version"] = runtime_metadata.get(
            "cuda_runtime_version", ""
        )
    if selected_device is not None:
        device_metadata = dict(selected_device.metadata)
        runtime_payload["device"] = {
            "device_id": selected_device.device_id,
            "display_name": selected_device.display_name,
            "total_memory_bytes": selected_device.total_memory_bytes,
            "compute_capability": device_metadata.get("compute_capability", ""),
        }

    libraries = []
    for library in report.library_probes:
        metadata = _support_library_metadata(library)
        libraries.append(
            {
                "library_id": library.library_id,
                "required_for_standard_cuda": (
                    library.library_id in _STANDARD_LIBRARY_IDS
                ),
                "optional": library.library_id == _OPTIONAL_LIBRARY_ID,
                "available": library.available,
                "version": library.version,
                "reason_code": library.reason_code,
                "message": _redact_private_text(library.message),
                "metadata": metadata,
            }
        )

    regions = []
    for region in report.admission_regions:
        regions.append(
            {
                "operation_id": region.operation_id,
                "implementation_id": region.implementation_id,
                "implementation_version": region.implementation_version,
                "implementation_library_id": region.implementation_library_id,
                "admission_tier": region.admission_tier,
                "environment_policy_id": region.environment_policy_id,
                "admitted": region.admitted,
                "reason_code": region.reason_code,
                "reason": region.reason,
                "supported_spatial_ndims": list(
                    region.supported_spatial_ndims
                ),
                "public_input_dtypes": [
                    list(port) for port in region.public_input_dtypes
                ],
                "limitations": list(region.limitations),
            }
        )
    environment = report.environment
    host = {
        "platform": report.platform,
        "execution_mode": report.execution_mode,
        "python": report.python,
        "track": report.track,
        "scientific_stack_versions": (
            {
                name: version
                for name, version in environment.scientific_stack_versions
                if name in {"numpy", "scipy", "scikit-image"}
            }
            if environment
            else {}
        ),
    }
    document = {
        "schema_id": _SUPPORT_SCHEMA,
        "schema_version": _SUPPORT_SCHEMA_VERSION,
        "generated_utc": generated,
        "privacy": {
            "policy_id": _SUPPORT_PRIVACY_POLICY,
            "redacted": True,
            "omitted": [
                "user and host names",
                "local filesystem and source paths",
                "environment variables and credentials",
                "workflow names, source names, and node IDs",
                "raw runtime and workload fingerprints",
                "local repair commands",
            ],
        },
        "application": {"napari_vipp_version": _installed_vipp_version()},
        "host": host,
        "diagnostic": {
            "status": report.status.value,
            "reason_code": report.reason_code,
            "summary": _redact_private_text(report.summary),
            "cuda": runtime_payload,
            "libraries": libraries,
            "public_admission": {
                "admitted_count": len(report.admitted_regions),
                "total_count": len(report.admission_regions),
                "regions": regions,
            },
            "guidance": (
                _redacted_guidance(report.guidance) if report.guidance else None
            ),
            "details": [_redact_private_text(detail) for detail in report.details],
            "packages": [package.as_dict() for package in report.packages],
        },
        "recent_execution": _support_recent_execution(recent_execution),
    }
    return _privacy_redact_document(document)


def write_compute_support_bundle(
    path: str | Path,
    report: ComputeDoctorReport,
    *,
    recent_execution: ExecutionReport | None = None,
    generated_utc: str | None = None,
) -> Path:
    """Atomically save one strict privacy-redacted support JSON document."""

    document = build_compute_support_bundle(
        report,
        recent_execution=recent_execution,
        generated_utc=generated_utc,
    )
    return atomic_write_json(path, document)


def main(argv: list[str] | None = None) -> int:
    """Run the headless compute doctor; unavailable CUDA is a normal result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("auto", "cuda12", "cuda13"), default="auto")
    parser.add_argument("--json", action="store_true", help="Print local JSON output.")
    parser.add_argument(
        "--support-bundle",
        type=Path,
        help="Atomically write privacy-redacted support JSON to this path.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached runtime probes."
    )
    args = parser.parse_args(argv)
    try:
        report = collect_compute_diagnostics(track=args.track, refresh=args.refresh)
    except Exception as exc:
        report = ComputeDoctorReport(
            status=DoctorStatus.UNAVAILABLE,
            reason_code="diagnostic_failed",
            summary="Compute diagnostics could not complete.",
            platform=sys.platform,
            execution_mode=_execution_mode(sys.platform),
            python=(
                f"{platform.python_implementation()} "
                f"{sys.version_info.major}.{sys.version_info.minor} "
                f"({struct.calcsize('P') * 8}-bit)"
            ),
            packages=(),
            track=args.track,
            details=(_exception_summary(exc),),
            guidance=RepairGuidance(
                "export_and_request_support",
                "Try the check again",
                (
                    "If it fails again, save the privacy-redacted support report "
                    "and share it with VIPP support."
                ),
                documentation_url=_GPU_GUIDE_URL,
            ),
        )

    support_error: Exception | None = None
    if args.support_bundle is not None:
        try:
            written = write_compute_support_bundle(args.support_bundle, report)
        except Exception as exc:
            support_error = exc
        else:
            if not args.json:
                print(f"Privacy-redacted support report saved to {written}.")

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    if support_error is not None:
        print(
            "Support report could not be saved: " + _exception_summary(support_error),
            file=sys.stderr,
        )
        return 3
    return 0 if report.available else 2


def _probe_runtime(runtime: RuntimeProtocol, *, refresh: bool) -> RuntimeProbeResult:
    try:
        with _provider_output_guard():
            result = runtime.probe(refresh=refresh)
    except Exception as exc:
        return RuntimeProbeResult(
            runtime_id="cuda-cupy",
            available=False,
            reason_code="diagnostic_probe_failed",
            message=_exception_summary(exc),
        )
    if not isinstance(result, RuntimeProbeResult):
        raise TypeError("Runtime probe must return RuntimeProbeResult.")
    return result


def _probe_registry_runtime(
    registry: ComputeRegistry,
    *,
    refresh: bool,
) -> RuntimeProbeResult:
    try:
        with _provider_output_guard():
            return registry.probe_runtime("cuda-cupy", refresh=refresh)
    except Exception as exc:
        return RuntimeProbeResult(
            runtime_id="cuda-cupy",
            available=False,
            reason_code="diagnostic_probe_failed",
            message=_exception_summary(exc),
        )


def _memory_snapshot(
    runtime: RuntimeProtocol,
    probe: RuntimeProbeResult,
    details: list[str],
) -> RuntimeMemorySnapshot | None:
    try:
        with _provider_output_guard():
            return runtime.memory_snapshot(device_id=probe.selected_device_id)
    except Exception as exc:
        details.append("Memory snapshot failed: " + _exception_summary(exc))
        return None


def _probe_libraries(
    registry: ComputeRegistry,
    *,
    runtime_available: bool,
    refresh: bool,
) -> tuple[ImplementationLibraryProbeResult, ...]:
    results = []
    for descriptor in registry.library_descriptors:
        if not runtime_available:
            results.append(
                ImplementationLibraryProbeResult(
                    descriptor.library_id,
                    False,
                    reason_code="runtime_unavailable",
                    message="The library was not probed because CUDA could not start.",
                )
            )
            continue
        # Some optional third-party imports print advisory text directly.  A
        # structured doctor must keep stdout valid, especially with ``--json``.
        with _provider_output_guard():
            result = registry.probe_library(descriptor.library_id, refresh=refresh)
        results.append(result)
    return tuple(results)


@contextlib.contextmanager
def _provider_output_guard():
    """Keep third-party advisory output out of CLI JSON and the application."""

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        yield


def _legacy_library_probes(
    probe: RuntimeProbeResult,
    packages: tuple[PackageRecord, ...],
) -> tuple[ImplementationLibraryProbeResult, ...]:
    """Preserve the old injected-runtime testing seam without claiming cuCIM."""

    version = probe.version
    standard_available = probe.available
    results = [
        ImplementationLibraryProbeResult(
            library_id,
            standard_available,
            version=version if standard_available else "",
            reason_code="" if standard_available else "runtime_unavailable",
            message=(
                "Available through the injected runtime diagnostic seam."
                if standard_available
                else "The library was not probed because CUDA could not start."
            ),
        )
        for library_id in ("cupy", "cupyx")
    ]
    cucim_installed = any(package.name.startswith("cucim") for package in packages)
    results.append(
        ImplementationLibraryProbeResult(
            "cucim",
            False,
            reason_code=(
                "cucim_not_probed" if cucim_installed else "cucim_not_installed"
            ),
            message=(
                "cuCIM was installed but not probed through the injected runtime seam."
                if cucim_installed
                else "The optional cuCIM add-on is not installed."
            ),
        )
    )
    return tuple(results)


def _diagnostic_environment(
    probe: RuntimeProbeResult,
    library_probes: tuple[ImplementationLibraryProbeResult, ...],
    *,
    platform_name: str,
    execution_mode: str,
    python_implementation: str,
    python_version: tuple[int, int],
    python_abi: str | None,
    scientific_stack_versions: Sequence[tuple[str, str]] | None,
) -> ComputeEnvironment:
    base = ComputeEnvironment(
        os_name=_os_family(platform_name),
        execution_mode=execution_mode,
        python_implementation=python_implementation,
        python_version=f"{python_version[0]}.{python_version[1]}",
        python_abi=(
            str(python_abi).strip()
            if python_abi is not None
            else _python_abi(python_implementation, python_version)
        ),
        scientific_stack_versions=(
            tuple(scientific_stack_versions)
            if scientific_stack_versions is not None
            else ComputeEnvironment().scientific_stack_versions
        ),
    )
    if not probe.available:
        return replace(base, probe_status="unavailable", probe_reason=probe.message)
    device = _selected_device(probe)
    available_libraries = tuple(item for item in library_probes if item.available)
    versions = [(probe.runtime_id, probe.version)] if probe.version else []
    versions.extend(
        (item.library_id, item.version) for item in available_libraries if item.version
    )
    runtime_metadata = dict(probe.metadata)
    return replace(
        base,
        runtime_ids=("cpu-numpy", probe.runtime_id),
        implementation_libraries=(
            "cpu",
            *(item.library_id for item in available_libraries),
        ),
        runtime_versions=tuple(versions),
        runtime_probe_fingerprints=(
            ((probe.runtime_id, probe.environment_fingerprint),)
            if probe.environment_fingerprint
            else ()
        ),
        runtime_metadata=((probe.runtime_id, probe.metadata),),
        implementation_library_metadata=tuple(
            (item.library_id, item.metadata) for item in available_libraries
        ),
        driver_version=runtime_metadata.get("driver_version", ""),
        device_id=device.device_id if device is not None else "cpu:0",
        device_name=device.display_name if device is not None else "Host CPU",
        device_class="nvidia-cuda" if device is not None else "host",
        device_metadata=device.metadata if device is not None else (),
        memory_topology=(
            MemoryTopology.DISCRETE
            if device is not None
            else MemoryTopology.HOST
        ),
        total_accelerator_memory_bytes=(
            device.total_memory_bytes or 0 if device is not None else 0
        ),
        probe_status="available",
        probe_reason="",
    )


def _public_admission_regions(
    specs: tuple[OperationComputeSpec, ...],
    environment: ComputeEnvironment,
) -> tuple[PublicAdmissionRegion, ...]:
    regions = []
    for spec in specs:
        decision = evaluate_candidate_environment_support(
            spec,
            environment,
            allow_experimental=False,
        )
        regions.append(
            PublicAdmissionRegion(
                operation_id=spec.operation_id,
                implementation_id=spec.implementation_id,
                implementation_version=spec.implementation_version,
                implementation_library_id=spec.implementation_library_id,
                admission_tier=spec.admission_tier.value,
                environment_policy_id=spec.validated_environment_policy_id,
                admitted=decision.supported,
                reason_code=decision.reason.value,
                reason=decision.reason_text,
                supported_spatial_ndims=spec.supported_spatial_ndims,
                public_input_dtypes=tuple(
                    tuple(port.public_dtypes) for port in spec.input_ports
                ),
                limitations=spec.limitations,
            )
        )
    return tuple(regions)


def _overall_result(
    probe: RuntimeProbeResult,
    library_probes: tuple[ImplementationLibraryProbeResult, ...],
    regions: tuple[PublicAdmissionRegion, ...],
    *,
    current_platform: str,
    selected_track: str,
    cleanup_failure: str,
) -> tuple[DoctorStatus, str, str]:
    if cleanup_failure:
        return (
            DoctorStatus.UNAVAILABLE,
            "runtime_cleanup_failed",
            "The CUDA checks completed, but VIPP could not prove clean shutdown.",
        )
    if not probe.available:
        return (
            DoctorStatus.UNAVAILABLE,
            probe.reason_code or "cuda_unavailable",
            probe.message or "CUDA could not start; VIPP will use CPU.",
        )
    device = _selected_device(probe)
    device_name = device.display_name if device is not None else "the CUDA device"
    by_library = {item.library_id: item for item in library_probes}
    standard_probes_ready = all(
        by_library.get(library_id) is not None
        and bool(by_library[library_id].available)
        for library_id in _STANDARD_LIBRARY_IDS
    )
    standard_regions = tuple(
        region
        for region in regions
        if region.implementation_library_id in _STANDARD_LIBRARY_IDS
    )
    standard_admitted = tuple(region for region in standard_regions if region.admitted)
    admitted_count = sum(region.admitted for region in regions)
    if (
        standard_regions
        and standard_probes_ready
        and len(standard_admitted) == len(standard_regions)
    ):
        return (
            DoctorStatus.AVAILABLE,
            "public_cuda_available",
            (
                f"CUDA is ready on {device_name}; {admitted_count} reviewed "
                "VIPP GPU operation regions are available."
            ),
        )
    if current_platform != "win32" or selected_track != "cuda13":
        return (
            DoctorStatus.UNSUPPORTED,
            "public_admission_unavailable",
            (
                f"CUDA starts on {device_name}, but this environment is outside "
                "VIPP's current public GPU support policy. VIPP will use CPU."
            ),
        )
    return (
        DoctorStatus.DEGRADED,
        "public_cuda_degraded",
        (
            f"CUDA starts on {device_name}, but part of VIPP's standard GPU "
            "support needs attention."
        ),
    )


def _repair_guidance(
    status: DoctorStatus,
    probe: RuntimeProbeResult,
    library_probes: tuple[ImplementationLibraryProbeResult, ...],
    regions: tuple[PublicAdmissionRegion, ...],
    packages: tuple[PackageRecord, ...],
    *,
    repair_command: str,
) -> RepairGuidance | None:
    if status in {
        DoctorStatus.UNAVAILABLE,
        DoctorStatus.DEGRADED,
        DoctorStatus.MISCONFIGURED,
    }:
        return RepairGuidance(
            "repair_standard_cuda_environment",
            "Repair the standard CUDA setup",
            (
                "VIPP will keep using CPU safely. Recreate or repair the "
                "dedicated CUDA environment, then verify again."
            ),
            command=repair_command,
            documentation_url=_GPU_GUIDE_URL,
        )
    if status is DoctorStatus.UNSUPPORTED:
        return RepairGuidance(
            "use_cpu_outside_public_gpu_policy",
            "Continue with CPU processing",
            (
                "CUDA may start, but no local setting can promote an environment "
                "that has not passed VIPP's public qualification policy."
            ),
            documentation_url=_GPU_GUIDE_URL,
        )
    cucim = next(
        (item for item in library_probes if item.library_id == _OPTIONAL_LIBRARY_ID),
        None,
    )
    cucim_regions = tuple(
        region
        for region in regions
        if region.implementation_library_id == _OPTIONAL_LIBRARY_ID
    )
    if cucim is not None and (
        not cucim.available or not all(region.admitted for region in cucim_regions)
    ):
        installed = any(package.name.startswith("cucim") for package in packages)
        return RepairGuidance(
            "repair_optional_cucim" if installed else "install_optional_cucim",
            (
                "Repair the optional cuCIM add-on"
                if installed
                else "Optional: add cuCIM support"
            ),
            (
                "Standard CUDA is ready. Repair cuCIM only if you want its "
                "reviewed GPU measurement regions."
                if installed
                else (
                    "Standard CUDA is ready. Build cuCIM locally only if you "
                    "want its reviewed GPU measurement regions."
                )
            ),
            documentation_url=_CUCIM_GUIDE_URL,
            optional=True,
        )
    return None


def _support_library_metadata(
    probe: ImplementationLibraryProbeResult,
) -> dict[str, str]:
    allowlist = {
        "environment_record_schema",
        "environment_record_schema_version",
        "environment_track",
        "cupy_distribution",
        "cucim_distribution",
        "cucim_distribution_version",
        "cucim_wheel_payload_sha256",
        "cucim_source_tag",
        "cucim_source_commit",
        "cucim_build_recipe_id",
    }
    return {
        key: value
        for key, value in probe.metadata
        if key in allowlist
    }


def _support_recent_execution(report: ExecutionReport | None) -> object:
    if report is None:
        return None
    decisions = [
        {
            "operation_id": decision.operation_id,
            "requested_preference": decision.requested_preference.as_dict(),
            "runtime_id": decision.runtime_id,
            "implementation_library_id": decision.implementation_library_id,
            "implementation_id": decision.implementation_id,
            "implementation_version": decision.implementation_version,
            "decision_kind": decision.decision_kind.value,
            "reason_code": decision.reason.value,
            "reason": _redact_private_text(decision.reason_text),
            "fallback_used": decision.fallback_used,
            "fallback_reason": decision.fallback_reason.value,
        }
        for decision in report.actual_decisions
    ]
    fallback_records = [
        {
            "runtime_id": record.runtime_id,
            "reason": record.reason.value,
            "reason_code": record.reason_code,
            "exception_type": record.exception_type,
            "retryable": record.retryable,
            "device_attempt_count": record.device_attempt_count,
            "cpu_retry_count": record.cpu_retry_count,
            "cpu_retry_succeeded": record.cpu_retry_succeeded,
            "cleanup_succeeded": record.cleanup_succeeded,
        }
        for record in report.fallback_records
    ]
    return {
        "requested_mode": report.request.mode.value,
        "cleanup_succeeded": report.cleanup_succeeded,
        "decisions": decisions,
        "fallback_records": fallback_records,
        "warnings": [_redact_private_text(value) for value in report.warnings],
    }


def _redacted_guidance(guidance: RepairGuidance) -> dict[str, object]:
    payload = guidance.as_dict(include_command=False)
    payload["title"] = _redact_private_text(str(payload["title"]))
    payload["summary"] = _redact_private_text(str(payload["summary"]))
    return payload


def _privacy_redact_document(value: object) -> object:
    """Recursively redact every string value in the fixed support schema."""

    if isinstance(value, dict):
        return {
            str(key): _privacy_redact_document(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_privacy_redact_document(item) for item in value]
    if isinstance(value, tuple):
        return [_privacy_redact_document(item) for item in value]
    if isinstance(value, str):
        return _redact_private_text(value)
    return value


def _redact_private_text(value: object) -> str:
    text = str(value)
    for private_root in (Path.home(), Path(sys.prefix), Path.cwd()):
        root = str(private_root).strip()
        if root:
            text = re.sub(re.escape(root), "<redacted-path>", text, flags=re.IGNORECASE)
    private_names = {Path.home().name.strip(), platform.node().strip()}
    for private_name in private_names:
        if len(private_name) >= 3 and private_name.lower() not in {
            "home",
            "user",
            "users",
        }:
            text = re.sub(
                rf"(?<![\w-]){re.escape(private_name)}(?![\w-])",
                "<redacted-identity>",
                text,
                flags=re.IGNORECASE,
            )
    text = re.sub(
        r"https?://[^\s\"'<>]+",
        lambda match: _redact_url(match.group(0)),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?i)\bauthorization\s*[:=]\s*[^\r\n,;]+",
        "Authorization: <redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]+",
        "<redacted-credential>",
        text,
    )
    text = re.sub(
        r"(?i)\b(token|password|secret|api[_ -]?key)(\s*[:=]\s*)[^\s,;]+",
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:gh[pousr]_[a-z0-9_]+|sk-[a-z0-9_-]{12,})\b",
        "<redacted-credential>",
        text,
    )
    text = re.sub(
        r"(?i)(?:[a-z]:[\\/]|\\\\)[^\r\n\"']+",
        "<redacted-path>",
        text,
    )
    text = re.sub(
        r"(?<![\w:])/(?:home|Users|tmp|var|opt|private|mnt)/[^\r\n\"']+",
        "<redacted-path>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?i)\b(?:desktop|laptop)-[a-z0-9-]+\b",
        "<redacted-host>",
        text,
    )
    text = re.sub(
        r"(?i)\b[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
        "<redacted-email>",
        text,
    )
    return text.strip()


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or "redacted-host"
    try:
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        port = ""
    return urlunsplit((parsed.scheme, hostname + port, "/<redacted>", "", ""))


def _print_human_report(report: ComputeDoctorReport) -> None:
    print(f"VIPP compute doctor: {report.status.value}")
    print(report.summary)
    print(f"Platform: {report.platform} ({report.execution_mode})")
    print(f"Python: {report.python}")
    print(f"GPU track: {report.track}")
    print("CUDA: " + ("Ready" if report.cuda_ready else "Could not start"))
    cucim = report.cucim_probe
    if cucim is None:
        cucim_text = "Not checked"
    elif cucim.available:
        cucim_text = "Ready"
    elif cucim.reason_code in {"cucim_not_installed", "cucim_provenance_missing"}:
        cucim_text = "Not installed (optional)"
    else:
        cucim_text = "Needs attention (optional)"
    print(f"Optional cuCIM: {cucim_text}")
    print(
        "VIPP public GPU regions: "
        f"{len(report.admitted_regions)} of {len(report.admission_regions)} ready"
    )
    if report.guidance is not None:
        print(f"Next step: {report.guidance.title}")
        print(report.guidance.summary)
    if report.repair_command:
        print("Suggested setup command:")
        print(report.repair_command)
    for detail in report.details:
        print(f"Detail: {detail}")


def _selected_device(probe: RuntimeProbeResult | None):
    if probe is None or not probe.devices:
        return None
    if probe.selected_device_id:
        selected = next(
            (
                item
                for item in probe.devices
                if item.device_id == probe.selected_device_id
            ),
            None,
        )
        if selected is not None:
            return selected
    return probe.devices[0]


def _normalize_packages(
    packages: Iterable[PackageRecord | tuple[str, str]],
) -> tuple[PackageRecord, ...]:
    unique: dict[str, PackageRecord] = {}
    for package in packages:
        record = (
            package
            if isinstance(package, PackageRecord)
            else PackageRecord(package[0], package[1])
        )
        unique[record.name] = record
    return tuple(unique[name] for name in sorted(unique))


def _canonical_name(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value).strip()).lower()


def _is_cupy_distribution(name: str) -> bool:
    return (
        name in _CUPY_NAMES
        or name.startswith("cupy-cuda")
        or name.startswith("cupy-rocm")
    )


def _is_gpu_distribution(name: str) -> bool:
    return (
        _is_cupy_distribution(name)
        or name.startswith("cucim")
        or name in {"cuda-pathfinder", "cuda-toolkit"}
        or name.startswith("nvidia-cuda-")
        or name.startswith("nvidia-cublas")
        or name.startswith("nvidia-cufft")
        or name.startswith("nvidia-curand")
        or name.startswith("nvidia-cusolver")
        or name.startswith("nvidia-cusparse")
        or name.startswith("nvidia-nvjitlink")
        or name.startswith("nvidia-nvimgcodec")
    )


def _distribution_track(name: str) -> str | None:
    if name.startswith("cupy-cuda12"):
        return "cuda12"
    if name.startswith("cupy-cuda13"):
        return "cuda13"
    return None


def _repair_command(platform_name: str, track: str) -> str:
    project_root = Path(__file__).resolve().parents[3]
    track_suffix = track.removeprefix("cuda")
    repair_venv = project_root / f".venv-gpu-cu{track_suffix}-repair"
    package_requirement = _installed_vipp_requirement(track)
    if platform_name == "win32":
        script = project_root / "scripts" / "setup_gpu_dev.ps1"
        if script.is_file():
            return (
                "powershell -ExecutionPolicy Bypass -File "
                f'"{script}" --track {track} --venv "{repair_venv}"'
            )
        venv = f".venv-vipp-gpu-cu{track_suffix}"
        python = f".\\{venv}\\Scripts\\python.exe"
        return (
            f'py -3.12 -m venv "{venv}"; '
            f'& "{python}" -m pip install --upgrade pip; '
            f'& "{python}" -m pip install --pre "napari[pyqt6]>=0.6" '
            f'"{package_requirement}"; '
            f'& "{python}" -m napari_vipp.core.compute_diagnostics --track {track}'
        )
    if platform_name.startswith("linux"):
        script = project_root / "scripts" / "setup_gpu_dev.sh"
        if script.is_file():
            return f'bash "{script}" --track {track} --venv "{repair_venv}"'
        venv = f".venv-vipp-gpu-cu{track_suffix}"
        return (
            f'python3.12 -m venv "{venv}" && '
            f'"./{venv}/bin/python" -m pip install --upgrade pip && '
            f'"./{venv}/bin/python" -m pip install '
            f'--pre "napari[pyqt6]>=0.6" "{package_requirement}" && '
            f'"./{venv}/bin/python" -m '
            f"napari_vipp.core.compute_diagnostics --track {track}"
        )
    return ""


def _installed_vipp_requirement(track: str) -> str:
    requirement = f"napari-vipp[gpu-{track}]"
    try:
        installed_version = importlib.metadata.version("napari-vipp").strip()
    except importlib.metadata.PackageNotFoundError:
        return requirement
    if not installed_version or installed_version == "0.0.0":
        return requirement
    return f"{requirement}=={installed_version}"


def _installed_vipp_version() -> str:
    try:
        return importlib.metadata.version("napari-vipp").strip() or "unknown"
    except importlib.metadata.PackageNotFoundError:
        return "source-checkout"


def _execution_mode(platform_name: str) -> str:
    if platform_name.startswith("linux") and "microsoft" in platform.release().lower():
        return "wsl2"
    return "native"


def _os_family(platform_name: str) -> str:
    normalized = str(platform_name).strip().lower()
    if normalized.startswith("win"):
        return "Windows"
    if normalized.startswith("linux"):
        return "Linux"
    if normalized in {"darwin", "macos"}:
        return "Darwin"
    return str(platform_name)


def _python_abi(implementation: str, version: tuple[int, int]) -> str:
    if implementation == "CPython":
        return f"cpython-{version[0]}{version[1]}"
    return ""


def _exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = [
    "ComputeDoctorReport",
    "DoctorStatus",
    "PackageRecord",
    "PublicAdmissionRegion",
    "RepairGuidance",
    "build_compute_support_bundle",
    "collect_compute_diagnostics",
    "installed_gpu_packages",
    "main",
    "write_compute_support_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
