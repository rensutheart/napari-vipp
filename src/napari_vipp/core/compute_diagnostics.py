"""Headless diagnostics for VIPP's optional GPU compute environment.

Run with ``python -m napari_vipp.core.compute_diagnostics``.  The module is
safe to import and run when CuPy, CUDA, Qt, and napari are all absent.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import struct
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from napari_vipp.core.compute_registry import (
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
    RuntimeProtocol,
)

_CUPY_NAMES = ("cupy", "amd-cupy")


class DoctorStatus(StrEnum):
    """High-level result of a compute environment inspection."""

    AVAILABLE = "available"
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
    schema_version: int = 1

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
                str(detail).strip() for detail in self.details if str(detail).strip()
            ),
        )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("schema_version must be a positive integer.")

    @property
    def available(self) -> bool:
        return self.status is DoctorStatus.AVAILABLE

    def as_dict(self) -> dict[str, object]:
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
            "details": list(self.details),
        }


def collect_compute_diagnostics(
    *,
    track: str = "auto",
    runtime: RuntimeProtocol | None = None,
    packages: Sequence[PackageRecord | tuple[str, str]] | None = None,
    platform_name: str | None = None,
    python_implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
    pointer_bits: int | None = None,
    refresh: bool = False,
) -> ComputeDoctorReport:
    """Inspect packaging and the CUDA runtime, returning errors as data."""

    requested_track = str(track).strip().lower()
    if requested_track not in {"auto", "cuda12", "cuda13"}:
        raise ValueError("track must be 'auto', 'cuda12', or 'cuda13'.")
    current_platform = sys.platform if platform_name is None else platform_name
    implementation = (
        platform.python_implementation()
        if python_implementation is None
        else python_implementation
    )
    version = tuple(sys.version_info[:2]) if python_version is None else python_version
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

    if len(cupy_packages) > 1:
        names = ", ".join(package.name for package in cupy_packages)
        return ComputeDoctorReport(
            status=DoctorStatus.MISCONFIGURED,
            reason_code="mixed_cupy_distributions",
            summary=f"Multiple CuPy distributions are installed: {names}.",
            platform=current_platform,
            execution_mode=execution_mode,
            python=python_label,
            packages=package_records,
            track=selected_track,
            repair_command=repair,
            details=("Create a fresh dedicated VIPP GPU environment.",),
        )

    if current_platform != "win32" and not current_platform.startswith("linux"):
        return ComputeDoctorReport(
            status=DoctorStatus.UNSUPPORTED,
            reason_code="platform_unsupported",
            summary="CUDA acceleration is unavailable on this platform.",
            platform=current_platform,
            execution_mode=execution_mode,
            python=python_label,
            packages=package_records,
            track=selected_track,
            details=(
                "VIPP remains usable on CPU; Apple GPU support requires a future "
                "non-CUDA runtime.",
            ),
        )

    if implementation != "CPython" or version != (3, 12) or bits != 64:
        return ComputeDoctorReport(
            status=DoctorStatus.UNSUPPORTED,
            reason_code="python_unsupported",
            summary="The Phase-1 GPU environment requires 64-bit CPython 3.12.",
            platform=current_platform,
            execution_mode=execution_mode,
            python=python_label,
            packages=package_records,
            track=selected_track,
            repair_command=repair,
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
                platform=current_platform,
                execution_mode=execution_mode,
                python=python_label,
                packages=package_records,
                track=selected_track,
                repair_command=repair,
                details=("Use a separate environment for each CUDA-major track.",),
            )

    owned_runtime = runtime is None
    details: list[str] = []
    try:
        if runtime is None:
            from napari_vipp.core.gpu.cupy_runtime import create_runtime

            runtime = create_runtime()
        probe = runtime.probe(refresh=refresh)
    except Exception as exc:
        probe = RuntimeProbeResult(
            runtime_id="cuda-cupy",
            available=False,
            reason_code="diagnostic_probe_failed",
            message=_exception_summary(exc),
        )

    snapshot = None
    if probe.available:
        try:
            snapshot = runtime.memory_snapshot(device_id=probe.selected_device_id)
        except Exception as exc:
            details.append("Memory snapshot failed: " + _exception_summary(exc))

    cleanup_failure = ""
    if owned_runtime and runtime is not None:
        try:
            runtime.close()
        except Exception as exc:
            cleanup_failure = _exception_summary(exc)
            details.append("Runtime cleanup failed: " + cleanup_failure)

    if cleanup_failure:
        status = DoctorStatus.UNAVAILABLE
        reason_code = "runtime_cleanup_failed"
        summary = "The CUDA runtime probe succeeded but cleanup did not."
    elif probe.available:
        device = probe.devices[0].display_name if probe.devices else "CUDA device"
        status = DoctorStatus.AVAILABLE
        reason_code = "cuda_available"
        summary = f"CuPy GPU execution is available on {device}."
        repair = ""
    else:
        status = DoctorStatus.UNAVAILABLE
        reason_code = probe.reason_code or "cuda_unavailable"
        summary = probe.message or "CuPy GPU execution is unavailable."

    return ComputeDoctorReport(
        status=status,
        reason_code=reason_code,
        summary=summary,
        platform=current_platform,
        execution_mode=execution_mode,
        python=python_label,
        packages=package_records,
        track=selected_track,
        repair_command=repair,
        runtime_probe=probe,
        memory_snapshot=snapshot,
        details=tuple(details),
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


def main(argv: list[str] | None = None) -> int:
    """Run the headless compute doctor; unavailable CUDA is a normal result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("auto", "cuda12", "cuda13"), default="auto")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore a cached runtime probe."
    )
    args = parser.parse_args(argv)
    try:
        report = collect_compute_diagnostics(
            track=args.track,
            refresh=args.refresh,
        )
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
        )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"VIPP compute doctor: {report.status.value}")
        print(report.summary)
        print(f"Platform: {report.platform} ({report.execution_mode})")
        print(f"Python: {report.python}")
        print(f"GPU track: {report.track}")
        if report.runtime_probe is not None:
            print(
                "Runtime: "
                f"{report.runtime_probe.runtime_id} "
                f"({report.runtime_probe.reason_code or 'unknown'})"
            )
        if report.repair_command:
            print("Suggested setup command:")
            print(report.repair_command)
        for detail in report.details:
            print(f"Detail: {detail}")
    return 0 if report.available else 2


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
    if platform_name == "win32":
        script = project_root / "scripts" / "setup_gpu_dev.ps1"
        if script.is_file():
            return (
                'powershell -ExecutionPolicy Bypass -File '
                f'"{script}" --track {track} --venv "{repair_venv}"'
            )
        venv = f".venv-vipp-gpu-cu{track_suffix}"
        python = f".\\{venv}\\Scripts\\python.exe"
        return (
            f'py -3.12 -m venv "{venv}"; '
            f'& "{python}" -m pip install "napari-vipp[gpu-{track}]"; '
            f'& "{python}" -m napari_vipp.core.compute_diagnostics '
            f"--track {track}"
        )
    if platform_name.startswith("linux"):
        script = project_root / "scripts" / "setup_gpu_dev.sh"
        if script.is_file():
            return (
                f'bash "{script}" --track {track} '
                f'--venv "{repair_venv}"'
            )
        venv = f".venv-vipp-gpu-cu{track_suffix}"
        return (
            f'python3.12 -m venv "{venv}" && '
            f'"./{venv}/bin/python" -m pip install '
            f'"napari-vipp[gpu-{track}]" && '
            f'"./{venv}/bin/python" -m '
            f"napari_vipp.core.compute_diagnostics --track {track}"
        )
    return ""


def _execution_mode(platform_name: str) -> str:
    if platform_name.startswith("linux") and "microsoft" in platform.release().lower():
        return "wsl2"
    return "native"


def _exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = [
    "ComputeDoctorReport",
    "DoctorStatus",
    "PackageRecord",
    "collect_compute_diagnostics",
    "installed_gpu_packages",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
