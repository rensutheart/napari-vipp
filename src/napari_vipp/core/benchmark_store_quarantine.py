"""Fail-closed quarantine for durable benchmark evidence stores."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from napari_vipp.core.compute_benchmark import (
    BenchmarkStoreError,
    _benchmark_store_lock,
    _benchmark_store_process_lock,
)


@dataclass(frozen=True, slots=True)
class BenchmarkStoreQuarantineResult:
    """Outcome of moving a suspect timing store away from its active path."""

    active_path: Path
    marker_path: Path
    quarantined_path: Path | None = None
    safe_for_restart: bool = False
    marker_present: bool = False
    error: str = ""


class BenchmarkStoreQuarantinedError(BenchmarkStoreError):
    """Raised when a durable poison marker cannot be resolved safely."""


def benchmark_store_quarantine_marker(path: str | os.PathLike[str]) -> Path:
    store_path = Path(path).expanduser().resolve(strict=False)
    return store_path.with_name(f"{store_path.name}.quarantine-required.json")


def ensure_benchmark_store_ready(path: str | os.PathLike[str]) -> Path:
    """Resolve a durable poison marker before a process reads timing evidence."""

    store_path = Path(path).expanduser().resolve(strict=False)
    marker_path = benchmark_store_quarantine_marker(store_path)
    with _benchmark_store_lock(store_path):
        with _benchmark_store_process_lock(store_path):
            try:
                marker_present = _strict_path_exists(marker_path)
            except OSError as exc:
                raise BenchmarkStoreQuarantinedError(
                    "Benchmark evidence safety could not be checked at "
                    f"{marker_path}: {type(exc).__name__}: {exc}."
                ) from exc
            if not marker_present:
                return store_path
            quarantined_path, error = _move_store_aside_locked(store_path)
            if error:
                raise BenchmarkStoreQuarantinedError(
                    f"Benchmark evidence remains quarantined at {store_path}: "
                    f"{error}. Rename or delete the active store and marker "
                    f"({marker_path}) before benchmarking."
                )
            try:
                marker_path.unlink(missing_ok=True)
                _fsync_parent_directory(marker_path)
            except OSError as exc:
                raise BenchmarkStoreQuarantinedError(
                    "The suspect benchmark store was moved aside"
                    + (
                        f" to {quarantined_path}"
                        if quarantined_path is not None
                        else ""
                    )
                    + ", but its durable quarantine marker could not be removed: "
                    f"{type(exc).__name__}: {exc}. Remove {marker_path} before "
                    "benchmarking."
                ) from exc
    return store_path


def quarantine_benchmark_store(
    path: str | os.PathLike[str],
    *,
    reason: str,
) -> BenchmarkStoreQuarantineResult:
    """Poison first, then move the store under its process mutation lock."""

    try:
        store_path = Path(path).expanduser().resolve(strict=False)
        marker_path = benchmark_store_quarantine_marker(store_path)
    except Exception as exc:
        fallback = Path(os.fspath(path))
        return BenchmarkStoreQuarantineResult(
            fallback,
            fallback.with_name(f"{fallback.name}.quarantine-required.json"),
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        with _benchmark_store_lock(store_path):
            with _benchmark_store_process_lock(store_path):
                marker_error = _write_quarantine_marker_locked(
                    marker_path,
                    reason=reason,
                )
                if marker_error:
                    return BenchmarkStoreQuarantineResult(
                        store_path,
                        marker_path,
                        marker_present=_path_exists(marker_path),
                        error=marker_error,
                    )
                quarantined_path, move_error = _move_store_aside_locked(store_path)
                if move_error:
                    return BenchmarkStoreQuarantineResult(
                        store_path,
                        marker_path,
                        marker_present=True,
                        error=move_error,
                    )
                # Keep the poison marker after a successful move. The next
                # process must observe it, confirm/move any recreated active
                # path under the same lock, and only then clear the marker.
                return BenchmarkStoreQuarantineResult(
                    store_path,
                    marker_path,
                    quarantined_path=quarantined_path,
                    safe_for_restart=True,
                    marker_present=True,
                )
    except Exception as exc:
        return BenchmarkStoreQuarantineResult(
            store_path,
            marker_path,
            marker_present=_path_exists(marker_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_quarantine_marker_locked(marker_path: Path, *, reason: str) -> str:
    temporary_path = marker_path.with_name(
        f"{marker_path.name}.tmp-{os.getpid()}-{threading.get_ident()}-"
        f"{uuid.uuid4().hex}"
    )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "reason": str(reason).strip() or "benchmark evidence cleanup failed",
    }
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, marker_path)
        _fsync_parent_directory(marker_path)
    except Exception as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        return f"could not write durable quarantine marker: {type(exc).__name__}: {exc}"
    return ""


def _move_store_aside_locked(store_path: Path) -> tuple[Path | None, str]:
    try:
        store_present = _strict_path_exists(store_path)
    except OSError as exc:
        return None, f"could not inspect active store: {type(exc).__name__}: {exc}"
    if not store_present:
        return None, ""
    for index in range(1_000):
        candidate = store_path.with_name(
            f"{store_path.name}.unsafe-{os.getpid()}-{index}"
        )
        if candidate.exists():
            continue
        try:
            store_path.rename(candidate)
            _fsync_parent_directory(store_path)
        except FileExistsError:
            continue
        except OSError as exc:
            return None, f"could not move active store: {type(exc).__name__}: {exc}"
        return candidate, ""
    return None, "no unique quarantine filename was available"


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _strict_path_exists(path: Path) -> bool:
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def _fsync_parent_directory(path: Path) -> None:
    """Persist directory-entry changes where the platform exposes directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BenchmarkStoreQuarantineResult",
    "BenchmarkStoreQuarantinedError",
    "benchmark_store_quarantine_marker",
    "ensure_benchmark_store_ready",
    "quarantine_benchmark_store",
]
