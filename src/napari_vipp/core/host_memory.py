"""Cross-platform host-memory diagnostics and conservative allocation gates.

The values called ``ullTotalPageFile`` and ``ullAvailPageFile`` by Windows'
``GlobalMemoryStatusEx`` describe the system commit limit and remaining commit
headroom, respectively.  They are not merely the configured page-file size.
Keeping those values separate from physical RAM is important: a large CPU
allocation can fail when commit is exhausted even while some physical memory
is still reported as available.

This module is Qt-free and has no optional dependencies.  Applications can use
the immutable snapshot for presentation and use :func:`preflight_host_allocation`
to fail closed before an optional, memory-intensive comparison.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

KIB = 1024
GIB = 1024**3

DEFAULT_HOST_MEMORY_SAFETY_RESERVE_BYTES = GIB
DEFAULT_HOST_MEMORY_SAFETY_RESERVE_FRACTION = 0.05


class HostMemorySource(StrEnum):
    """Native source used to collect a host-memory snapshot."""

    WINDOWS_GLOBAL_MEMORY_STATUS_EX = "windows_global_memory_status_ex"
    LINUX_PROC_MEMINFO = "linux_proc_meminfo"
    MACOS_HOST_STATISTICS = "macos_host_statistics"
    POSIX_SYSCONF = "posix_sysconf"
    UNAVAILABLE = "unavailable"


class HostMemoryStatus(StrEnum):
    """Completeness of a host-memory snapshot."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class HostMemoryConstraint(StrEnum):
    """Resource that determined an allocation-preflight result."""

    NONE = "none"
    PHYSICAL = "physical"
    COMMIT = "commit"
    SNAPSHOT = "snapshot"


class HostMemoryPreflightReason(StrEnum):
    """Stable reason codes for allocation-preflight decisions."""

    ADMITTED = "admitted"
    SNAPSHOT_UNAVAILABLE = "snapshot_unavailable"
    COMMIT_HEADROOM_UNAVAILABLE = "commit_headroom_unavailable"
    INSUFFICIENT_COMMIT_HEADROOM = "insufficient_commit_headroom"
    INSUFFICIENT_PHYSICAL_HEADROOM = "insufficient_physical_headroom"


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """One immutable native host-memory observation.

    ``commit_limit_bytes`` and ``commit_available_bytes`` are populated on
    Windows and when Linux exposes the corresponding ``/proc/meminfo`` rows.
    macOS intentionally reports unified physical memory without pretending it
    has Windows-style commit accounting.
    """

    platform: str
    source: HostMemorySource | str
    physical_total_bytes: int | None = None
    physical_available_bytes: int | None = None
    commit_limit_bytes: int | None = None
    commit_available_bytes: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        platform_name = str(self.platform).strip().lower()
        if not platform_name:
            raise ValueError("platform must not be empty.")
        object.__setattr__(self, "platform", platform_name)
        source = (
            self.source
            if isinstance(self.source, HostMemorySource)
            else HostMemorySource(str(self.source).strip().lower())
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "detail", str(self.detail).strip())
        for name in (
            "physical_total_bytes",
            "physical_available_bytes",
            "commit_limit_bytes",
            "commit_available_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        _validate_available_not_above_total(
            self.physical_available_bytes,
            self.physical_total_bytes,
            "physical_available_bytes",
            "physical_total_bytes",
        )
        _validate_available_not_above_total(
            self.commit_available_bytes,
            self.commit_limit_bytes,
            "commit_available_bytes",
            "commit_limit_bytes",
        )

    @classmethod
    def unavailable(cls, platform: str, detail: str) -> HostMemorySnapshot:
        """Build a typed, non-raising snapshot for a failed native probe."""

        return cls(
            platform=platform or "unknown",
            source=HostMemorySource.UNAVAILABLE,
            detail=detail,
        )

    @property
    def status(self) -> HostMemoryStatus:
        physical_complete = (
            self.physical_total_bytes is not None
            and self.physical_available_bytes is not None
        )
        commit_complete = (
            self.commit_limit_bytes is not None
            and self.commit_available_bytes is not None
        )
        if self.source is HostMemorySource.UNAVAILABLE or not physical_complete:
            if not physical_complete and not commit_complete:
                return HostMemoryStatus.UNAVAILABLE
            return HostMemoryStatus.PARTIAL
        if self.platform.startswith("win") and not commit_complete:
            return HostMemoryStatus.PARTIAL
        return HostMemoryStatus.AVAILABLE

    @property
    def has_commit_accounting(self) -> bool:
        return (
            self.commit_limit_bytes is not None
            and self.commit_available_bytes is not None
        )

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-safe diagnostics without changing units."""

        return {
            "platform": self.platform,
            "source": self.source.value,
            "status": self.status.value,
            "physical_total_bytes": self.physical_total_bytes,
            "physical_available_bytes": self.physical_available_bytes,
            "commit_limit_bytes": self.commit_limit_bytes,
            "commit_available_bytes": self.commit_available_bytes,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class HostMemoryPreflight:
    """Typed decision for one prospective additional host allocation."""

    allowed: bool
    reason_code: HostMemoryPreflightReason | str
    reason: str
    limiting_resource: HostMemoryConstraint | str
    required_bytes: int
    physical_reserve_bytes: int | None
    commit_reserve_bytes: int | None
    physical_headroom_after_bytes: int | None
    commit_headroom_after_bytes: int | None

    def __post_init__(self) -> None:
        reason_code = (
            self.reason_code
            if isinstance(self.reason_code, HostMemoryPreflightReason)
            else HostMemoryPreflightReason(str(self.reason_code).strip().lower())
        )
        constraint = (
            self.limiting_resource
            if isinstance(self.limiting_resource, HostMemoryConstraint)
            else HostMemoryConstraint(str(self.limiting_resource).strip().lower())
        )
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "limiting_resource", constraint)
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("reason must not be empty.")
        object.__setattr__(self, "reason", reason)
        _validate_nonnegative_int(self.required_bytes, "required_bytes")
        for name in ("physical_reserve_bytes", "commit_reserve_bytes"):
            value = getattr(self, name)
            if value is not None:
                _validate_nonnegative_int(value, name)
        for name in (
            "physical_headroom_after_bytes",
            "commit_headroom_after_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"{name} must be an integer or None.")

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-safe decision record for provenance."""

        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "limiting_resource": self.limiting_resource.value,
            "required_bytes": self.required_bytes,
            "physical_reserve_bytes": self.physical_reserve_bytes,
            "commit_reserve_bytes": self.commit_reserve_bytes,
            "physical_headroom_after_bytes": self.physical_headroom_after_bytes,
            "commit_headroom_after_bytes": self.commit_headroom_after_bytes,
        }


class _GlobalMemoryStatusAPI(Protocol):
    def GlobalMemoryStatusEx(self, status_pointer: object) -> int: ...


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _MacOSVMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


def capture_host_memory(*, platform_name: str | None = None) -> HostMemorySnapshot:
    """Capture host memory with a typed safe fallback on every platform."""

    resolved_platform = str(platform_name or sys.platform).strip().lower()
    if resolved_platform == "win32":
        return _capture_windows_memory(platform_name=resolved_platform)
    if resolved_platform == "darwin":
        return _capture_macos_memory(platform_name=resolved_platform)
    if resolved_platform.startswith("linux"):
        proc_snapshot = _capture_linux_proc_memory(platform_name=resolved_platform)
        if proc_snapshot.status is not HostMemoryStatus.UNAVAILABLE:
            return proc_snapshot
    return _capture_posix_sysconf_memory(platform_name=resolved_platform)


def preflight_host_allocation(
    snapshot: HostMemorySnapshot,
    *,
    required_bytes: int,
    purpose: str = "host allocation",
    safety_reserve_bytes: int = DEFAULT_HOST_MEMORY_SAFETY_RESERVE_BYTES,
    safety_reserve_fraction: float = DEFAULT_HOST_MEMORY_SAFETY_RESERVE_FRACTION,
) -> HostMemoryPreflight:
    """Decide whether an additional allocation has conservative headroom.

    The caller supplies the estimated *additional peak* bytes rather than the
    total process footprint.  A snapshot with no usable physical observation
    fails closed.  Windows also fails closed when commit accounting is absent.
    On platforms that expose commit data, both commit and physical headroom
    must pass.  The returned reason is suitable for a message strip or an
    optimization report and never claims an exact timing for a skipped test.
    """

    if not isinstance(snapshot, HostMemorySnapshot):
        raise TypeError("snapshot must be a HostMemorySnapshot.")
    _validate_nonnegative_int(required_bytes, "required_bytes")
    _validate_nonnegative_int(safety_reserve_bytes, "safety_reserve_bytes")
    if (
        isinstance(safety_reserve_fraction, bool)
        or not isinstance(safety_reserve_fraction, (int, float))
        or not 0.0 <= float(safety_reserve_fraction) <= 1.0
    ):
        raise ValueError("safety_reserve_fraction must be between 0 and 1.")
    purpose_text = str(purpose).strip()
    if not purpose_text:
        raise ValueError("purpose must not be empty.")

    physical_reserve = _effective_reserve(
        snapshot.physical_total_bytes,
        safety_reserve_bytes,
        float(safety_reserve_fraction),
    )
    commit_reserve = _effective_reserve(
        snapshot.commit_limit_bytes,
        safety_reserve_bytes,
        float(safety_reserve_fraction),
    )
    physical_after = _remaining_after(
        snapshot.physical_available_bytes,
        required_bytes,
        physical_reserve,
    )
    commit_after = _remaining_after(
        snapshot.commit_available_bytes,
        required_bytes,
        commit_reserve,
    )

    if snapshot.physical_available_bytes is None or physical_reserve is None:
        return _preflight_result(
            allowed=False,
            reason_code=HostMemoryPreflightReason.SNAPSHOT_UNAVAILABLE,
            reason=(
                f"Skipped {purpose_text}: available physical memory could not "
                "be measured safely."
            ),
            limiting_resource=HostMemoryConstraint.SNAPSHOT,
            required_bytes=required_bytes,
            physical_reserve=physical_reserve,
            commit_reserve=commit_reserve,
            physical_after=physical_after,
            commit_after=commit_after,
        )

    windows_commit_required = snapshot.platform.startswith("win")
    if windows_commit_required and not snapshot.has_commit_accounting:
        return _preflight_result(
            allowed=False,
            reason_code=HostMemoryPreflightReason.COMMIT_HEADROOM_UNAVAILABLE,
            reason=(
                f"Skipped {purpose_text}: Windows commit headroom could not be "
                "measured safely."
            ),
            limiting_resource=HostMemoryConstraint.COMMIT,
            required_bytes=required_bytes,
            physical_reserve=physical_reserve,
            commit_reserve=commit_reserve,
            physical_after=physical_after,
            commit_after=commit_after,
        )

    if snapshot.has_commit_accounting and (commit_after is None or commit_after < 0):
        assert snapshot.commit_available_bytes is not None
        assert commit_reserve is not None
        return _preflight_result(
            allowed=False,
            reason_code=HostMemoryPreflightReason.INSUFFICIENT_COMMIT_HEADROOM,
            reason=_insufficient_reason(
                purpose_text,
                resource="commit",
                available_bytes=snapshot.commit_available_bytes,
                required_bytes=required_bytes,
                reserve_bytes=commit_reserve,
            ),
            limiting_resource=HostMemoryConstraint.COMMIT,
            required_bytes=required_bytes,
            physical_reserve=physical_reserve,
            commit_reserve=commit_reserve,
            physical_after=physical_after,
            commit_after=commit_after,
        )

    if physical_after is None or physical_after < 0:
        return _preflight_result(
            allowed=False,
            reason_code=HostMemoryPreflightReason.INSUFFICIENT_PHYSICAL_HEADROOM,
            reason=_insufficient_reason(
                purpose_text,
                resource="physical-memory",
                available_bytes=snapshot.physical_available_bytes,
                required_bytes=required_bytes,
                reserve_bytes=physical_reserve,
            ),
            limiting_resource=HostMemoryConstraint.PHYSICAL,
            required_bytes=required_bytes,
            physical_reserve=physical_reserve,
            commit_reserve=commit_reserve,
            physical_after=physical_after,
            commit_after=commit_after,
        )

    display_purpose = purpose_text[:1].upper() + purpose_text[1:]
    detail = (
        f"{display_purpose} admitted: estimated additional peak "
        f"{_format_bytes(required_bytes)} preserves "
        f"{_format_bytes(physical_reserve)} of physical-memory reserve"
    )
    if snapshot.has_commit_accounting and commit_reserve is not None:
        detail += f" and {_format_bytes(commit_reserve)} of commit reserve"
    detail += "."
    return _preflight_result(
        allowed=True,
        reason_code=HostMemoryPreflightReason.ADMITTED,
        reason=detail,
        limiting_resource=HostMemoryConstraint.NONE,
        required_bytes=required_bytes,
        physical_reserve=physical_reserve,
        commit_reserve=commit_reserve,
        physical_after=physical_after,
        commit_after=commit_after,
    )


def _capture_windows_memory(
    *,
    platform_name: str,
    kernel32: _GlobalMemoryStatusAPI | None = None,
) -> HostMemorySnapshot:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        native_api = kernel32
        if native_api is None:
            native_api = ctypes.windll.kernel32
        ok = native_api.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return HostMemorySnapshot.unavailable(
            platform_name,
            f"GlobalMemoryStatusEx unavailable: {exc}",
        )
    if not ok:
        return HostMemorySnapshot.unavailable(
            platform_name,
            "GlobalMemoryStatusEx returned failure.",
        )
    return HostMemorySnapshot(
        platform=platform_name,
        source=HostMemorySource.WINDOWS_GLOBAL_MEMORY_STATUS_EX,
        physical_total_bytes=int(status.ullTotalPhys),
        physical_available_bytes=int(status.ullAvailPhys),
        commit_limit_bytes=int(status.ullTotalPageFile),
        commit_available_bytes=int(status.ullAvailPageFile),
    )


def _capture_linux_proc_memory(
    *,
    platform_name: str,
    path: Path = Path("/proc/meminfo"),
) -> HostMemorySnapshot:
    try:
        rows = _parse_linux_meminfo(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        return HostMemorySnapshot.unavailable(
            platform_name,
            f"/proc/meminfo unavailable: {exc}",
        )
    total = rows.get("MemTotal")
    available = rows.get("MemAvailable", rows.get("MemFree"))
    commit_limit = rows.get("CommitLimit")
    committed = rows.get("Committed_AS")
    commit_available = None
    if commit_limit is not None and committed is not None:
        commit_available = max(commit_limit - committed, 0)
    if total is None or available is None:
        return HostMemorySnapshot.unavailable(
            platform_name,
            "/proc/meminfo did not report MemTotal and available memory.",
        )
    return HostMemorySnapshot(
        platform=platform_name,
        source=HostMemorySource.LINUX_PROC_MEMINFO,
        physical_total_bytes=total,
        physical_available_bytes=min(available, total),
        commit_limit_bytes=commit_limit,
        commit_available_bytes=commit_available,
    )


def _capture_macos_memory(*, platform_name: str) -> HostMemorySnapshot:
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return HostMemorySnapshot.unavailable(platform_name, "os.sysconf unavailable.")
    try:
        page_size = int(sysconf("SC_PAGE_SIZE"))
        total_pages = int(sysconf("SC_PHYS_PAGES"))
        lib_system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib_system.mach_host_self.restype = ctypes.c_uint32
        host = lib_system.mach_host_self()
        statistics = _MacOSVMStatistics64()
        count = ctypes.c_uint32(
            ctypes.sizeof(statistics) // ctypes.sizeof(ctypes.c_int)
        )
        result = lib_system.host_statistics64(
            host,
            4,  # HOST_VM_INFO64
            ctypes.byref(statistics),
            ctypes.byref(count),
        )
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return HostMemorySnapshot.unavailable(
            platform_name,
            f"macOS host statistics unavailable: {exc}",
        )
    if result != 0:
        return HostMemorySnapshot.unavailable(
            platform_name,
            f"host_statistics64 returned {result}.",
        )
    available_pages = int(statistics.free_count) + int(statistics.inactive_count)
    total_bytes = total_pages * page_size
    return HostMemorySnapshot(
        platform=platform_name,
        source=HostMemorySource.MACOS_HOST_STATISTICS,
        physical_total_bytes=total_bytes,
        physical_available_bytes=min(available_pages * page_size, total_bytes),
        detail="Unified-memory physical availability; no Windows-style commit value.",
    )


def _capture_posix_sysconf_memory(*, platform_name: str) -> HostMemorySnapshot:
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return HostMemorySnapshot.unavailable(platform_name, "os.sysconf unavailable.")
    try:
        page_size = int(sysconf("SC_PAGE_SIZE"))
        total_pages = int(sysconf("SC_PHYS_PAGES"))
        available_pages = int(sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return HostMemorySnapshot.unavailable(
            platform_name,
            f"POSIX memory statistics unavailable: {exc}",
        )
    total_bytes = total_pages * page_size
    return HostMemorySnapshot(
        platform=platform_name,
        source=HostMemorySource.POSIX_SYSCONF,
        physical_total_bytes=total_bytes,
        physical_available_bytes=min(available_pages * page_size, total_bytes),
    )


def _parse_linux_meminfo(text: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for raw_line in text.splitlines():
        key, separator, raw_value = raw_line.partition(":")
        if not separator:
            continue
        fields = raw_value.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if value < 0:
            continue
        multiplier = KIB if len(fields) > 1 and fields[1].lower() == "kb" else 1
        rows[key.strip()] = value * multiplier
    return rows


def _preflight_result(
    *,
    allowed: bool,
    reason_code: HostMemoryPreflightReason,
    reason: str,
    limiting_resource: HostMemoryConstraint,
    required_bytes: int,
    physical_reserve: int | None,
    commit_reserve: int | None,
    physical_after: int | None,
    commit_after: int | None,
) -> HostMemoryPreflight:
    return HostMemoryPreflight(
        allowed=allowed,
        reason_code=reason_code,
        reason=reason,
        limiting_resource=limiting_resource,
        required_bytes=required_bytes,
        physical_reserve_bytes=physical_reserve,
        commit_reserve_bytes=commit_reserve,
        physical_headroom_after_bytes=physical_after,
        commit_headroom_after_bytes=commit_after,
    )


def _effective_reserve(
    total_bytes: int | None,
    fixed_bytes: int,
    fraction: float,
) -> int | None:
    if total_bytes is None:
        return None
    return max(fixed_bytes, int(total_bytes * fraction))


def _remaining_after(
    available_bytes: int | None,
    required_bytes: int,
    reserve_bytes: int | None,
) -> int | None:
    if available_bytes is None or reserve_bytes is None:
        return None
    return available_bytes - required_bytes - reserve_bytes


def _insufficient_reason(
    purpose: str,
    *,
    resource: str,
    available_bytes: int,
    required_bytes: int,
    reserve_bytes: int,
) -> str:
    return (
        f"Skipped {purpose}: {_format_bytes(available_bytes)} of {resource} "
        f"headroom is available, but the candidate needs an estimated "
        f"{_format_bytes(required_bytes)} plus a {_format_bytes(reserve_bytes)} "
        "safety reserve."
    )


def _format_bytes(value: int) -> str:
    if value < KIB:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / KIB:.1f} KiB"
    if value < GIB:
        return f"{value / 1024**2:.1f} MiB"
    return f"{value / GIB:.1f} GiB"


def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_available_not_above_total(
    available: int | None,
    total: int | None,
    available_name: str,
    total_name: str,
) -> None:
    if available is not None and total is not None and available > total:
        raise ValueError(f"{available_name} must not exceed {total_name}.")


__all__ = [
    "DEFAULT_HOST_MEMORY_SAFETY_RESERVE_BYTES",
    "DEFAULT_HOST_MEMORY_SAFETY_RESERVE_FRACTION",
    "HostMemoryConstraint",
    "HostMemoryPreflight",
    "HostMemoryPreflightReason",
    "HostMemorySnapshot",
    "HostMemorySource",
    "HostMemoryStatus",
    "capture_host_memory",
    "preflight_host_allocation",
]
