from __future__ import annotations

from pathlib import Path

import pytest

from napari_vipp.core.host_memory import (
    GIB,
    HostMemoryConstraint,
    HostMemoryPreflightReason,
    HostMemorySnapshot,
    HostMemorySource,
    HostMemoryStatus,
    _capture_linux_proc_memory,
    _capture_windows_memory,
    capture_host_memory,
    preflight_host_allocation,
)


class _FakeKernel32:
    def __init__(
        self,
        *,
        total_physical: int,
        available_physical: int,
        commit_limit: int,
        commit_available: int,
        succeeds: bool = True,
    ) -> None:
        self.total_physical = total_physical
        self.available_physical = available_physical
        self.commit_limit = commit_limit
        self.commit_available = commit_available
        self.succeeds = succeeds
        self.observed_length = 0

    def GlobalMemoryStatusEx(self, status_pointer) -> int:
        status = status_pointer._obj
        self.observed_length = int(status.dwLength)
        status.ullTotalPhys = self.total_physical
        status.ullAvailPhys = self.available_physical
        status.ullTotalPageFile = self.commit_limit
        status.ullAvailPageFile = self.commit_available
        return int(self.succeeds)


def _snapshot(
    *,
    platform: str = "win32",
    physical_total: int = 64 * GIB,
    physical_available: int = 32 * GIB,
    commit_limit: int | None = 96 * GIB,
    commit_available: int | None = 40 * GIB,
) -> HostMemorySnapshot:
    return HostMemorySnapshot(
        platform=platform,
        source=(
            HostMemorySource.WINDOWS_GLOBAL_MEMORY_STATUS_EX
            if platform == "win32"
            else HostMemorySource.POSIX_SYSCONF
        ),
        physical_total_bytes=physical_total,
        physical_available_bytes=physical_available,
        commit_limit_bytes=commit_limit,
        commit_available_bytes=commit_available,
    )


def test_windows_snapshot_distinguishes_physical_and_commit_headroom() -> None:
    kernel32 = _FakeKernel32(
        total_physical=64 * GIB,
        available_physical=21 * GIB,
        commit_limit=80 * GIB,
        commit_available=7 * GIB,
    )

    snapshot = _capture_windows_memory(
        platform_name="win32",
        kernel32=kernel32,
    )

    assert kernel32.observed_length > 0
    assert snapshot.status is HostMemoryStatus.AVAILABLE
    assert snapshot.physical_total_bytes == 64 * GIB
    assert snapshot.physical_available_bytes == 21 * GIB
    assert snapshot.commit_limit_bytes == 80 * GIB
    assert snapshot.commit_available_bytes == 7 * GIB
    assert snapshot.as_dict()["source"] == "windows_global_memory_status_ex"


def test_windows_probe_failure_returns_safe_typed_snapshot() -> None:
    snapshot = _capture_windows_memory(
        platform_name="win32",
        kernel32=_FakeKernel32(
            total_physical=0,
            available_physical=0,
            commit_limit=0,
            commit_available=0,
            succeeds=False,
        ),
    )

    assert snapshot.status is HostMemoryStatus.UNAVAILABLE
    assert snapshot.source is HostMemorySource.UNAVAILABLE
    assert snapshot.physical_available_bytes is None
    assert "returned failure" in snapshot.detail


def test_linux_proc_snapshot_reports_available_and_remaining_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meminfo"
    path.write_text(
        "\n".join(
            (
                "MemTotal:       65536 kB",
                "MemFree:         4096 kB",
                "MemAvailable:   32768 kB",
                "CommitLimit:    98304 kB",
                "Committed_AS:   73728 kB",
            )
        ),
        encoding="utf-8",
    )

    snapshot = _capture_linux_proc_memory(platform_name="linux", path=path)

    assert snapshot.source is HostMemorySource.LINUX_PROC_MEMINFO
    assert snapshot.physical_total_bytes == 65536 * 1024
    assert snapshot.physical_available_bytes == 32768 * 1024
    assert snapshot.commit_limit_bytes == 98304 * 1024
    assert snapshot.commit_available_bytes == (98304 - 73728) * 1024


def test_capture_host_memory_falls_back_when_linux_proc_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = HostMemorySnapshot.unavailable("linux", "missing proc")
    expected = HostMemorySnapshot(
        platform="linux",
        source=HostMemorySource.POSIX_SYSCONF,
        physical_total_bytes=8 * GIB,
        physical_available_bytes=4 * GIB,
    )
    monkeypatch.setattr(
        "napari_vipp.core.host_memory._capture_linux_proc_memory",
        lambda **_kwargs: unavailable,
    )
    monkeypatch.setattr(
        "napari_vipp.core.host_memory._capture_posix_sysconf_memory",
        lambda **_kwargs: expected,
    )

    assert capture_host_memory(platform_name="linux") == expected


def test_preflight_rejects_commit_exhaustion_even_with_free_physical_ram() -> None:
    snapshot = _snapshot(
        physical_available=30 * GIB,
        commit_available=5 * GIB,
    )

    decision = preflight_host_allocation(
        snapshot,
        required_bytes=4 * GIB,
        purpose="optional CPU evidence comparison",
        safety_reserve_bytes=2 * GIB,
        safety_reserve_fraction=0.0,
    )

    assert not decision.allowed
    assert (
        decision.reason_code is HostMemoryPreflightReason.INSUFFICIENT_COMMIT_HEADROOM
    )
    assert decision.limiting_resource is HostMemoryConstraint.COMMIT
    assert decision.commit_headroom_after_bytes == -GIB
    assert "5.0 GiB of commit headroom" in decision.reason
    assert "4.0 GiB plus a 2.0 GiB safety reserve" in decision.reason


def test_preflight_rejects_physical_pressure_after_commit_passes() -> None:
    decision = preflight_host_allocation(
        _snapshot(physical_available=5 * GIB, commit_available=30 * GIB),
        required_bytes=4 * GIB,
        safety_reserve_bytes=2 * GIB,
        safety_reserve_fraction=0.0,
    )

    assert not decision.allowed
    assert (
        decision.reason_code is HostMemoryPreflightReason.INSUFFICIENT_PHYSICAL_HEADROOM
    )
    assert decision.limiting_resource is HostMemoryConstraint.PHYSICAL
    assert decision.physical_headroom_after_bytes == -GIB


def test_windows_preflight_fails_closed_without_commit_accounting() -> None:
    decision = preflight_host_allocation(
        _snapshot(commit_limit=None, commit_available=None),
        required_bytes=GIB,
        safety_reserve_fraction=0.0,
    )

    assert not decision.allowed
    assert decision.reason_code is HostMemoryPreflightReason.COMMIT_HEADROOM_UNAVAILABLE
    assert "Windows commit headroom could not be measured" in decision.reason


def test_preflight_fails_closed_when_snapshot_is_unavailable() -> None:
    decision = preflight_host_allocation(
        HostMemorySnapshot.unavailable("linux", "native probe failed"),
        required_bytes=GIB,
    )

    assert not decision.allowed
    assert decision.reason_code is HostMemoryPreflightReason.SNAPSHOT_UNAVAILABLE
    assert decision.limiting_resource is HostMemoryConstraint.SNAPSHOT


def test_physical_only_platform_can_admit_with_conservative_reserve() -> None:
    decision = preflight_host_allocation(
        _snapshot(
            platform="darwin",
            physical_total=64 * GIB,
            physical_available=20 * GIB,
            commit_limit=None,
            commit_available=None,
        ),
        required_bytes=8 * GIB,
        purpose="CPU benchmark",
        safety_reserve_bytes=GIB,
        safety_reserve_fraction=0.1,
    )

    assert decision.allowed
    assert decision.reason_code is HostMemoryPreflightReason.ADMITTED
    assert decision.physical_reserve_bytes == pytest.approx(6.4 * GIB, abs=1)
    assert decision.commit_reserve_bytes is None
    assert decision.commit_headroom_after_bytes is None
    assert "CPU benchmark admitted" in decision.reason


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"required_bytes": -1}, "required_bytes"),
        ({"required_bytes": True}, "required_bytes"),
        ({"required_bytes": 1, "safety_reserve_fraction": 1.1}, "fraction"),
        ({"required_bytes": 1, "purpose": ""}, "purpose"),
    ),
)
def test_preflight_rejects_invalid_requests(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        preflight_host_allocation(_snapshot(), **kwargs)


def test_snapshot_validation_rejects_impossible_available_values() -> None:
    with pytest.raises(ValueError, match="physical_available_bytes"):
        HostMemorySnapshot(
            platform="linux",
            source="posix_sysconf",
            physical_total_bytes=100,
            physical_available_bytes=101,
        )
