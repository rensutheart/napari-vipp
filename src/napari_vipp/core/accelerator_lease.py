"""Process-wide, provider-neutral accelerator ownership.

Scientific execution, benchmarks, and health diagnostics may be initiated by
different application services and may even use different runtime instances.
This module gives all of them one small Qt-free coordination primitive.  A
lease is scoped by runtime and physical device, is fair between waiting
threads, and is re-entrant only for its owning thread.

CPU work deliberately bypasses the registry.  The lease protects VIPP work in
this Python process; it is not an inter-process GPU scheduler.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from napari_vipp.core.progress import OperationCancelled

LeaseCancelCallback = Callable[[], bool]
LeaseClock = Callable[[], float]

_WAIT_POLL_SECONDS = 0.05
_CPU_RUNTIME_ID = "cpu-numpy"


class AcceleratorLeaseCancelled(OperationCancelled):
    """Raised when accelerator ownership is cancelled while waiting."""


class AcceleratorLeaseDeadlineExceeded(TimeoutError):
    """Raised when accelerator ownership cannot be obtained by its deadline."""


@dataclass(frozen=True, slots=True)
class AcceleratorLeaseKey:
    """Stable identity for one runtime's access to one physical device."""

    runtime_id: str
    device_id: str

    def __post_init__(self) -> None:
        runtime_id = str(self.runtime_id).strip()
        device_id = str(self.device_id).strip()
        if not runtime_id:
            raise ValueError("runtime_id must not be empty.")
        if not device_id:
            raise ValueError("device_id must not be empty.")
        object.__setattr__(self, "runtime_id", runtime_id)
        object.__setattr__(self, "device_id", device_id)


@dataclass(slots=True)
class _LeaseState:
    owner_thread_id: int | None = None
    depth: int = 0
    waiters: deque[object] = field(default_factory=deque)


class AcceleratorLeaseManager:
    """Serialize accelerator work without importing an accelerator package."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._states: dict[AcceleratorLeaseKey, _LeaseState] = {}

    @contextmanager
    def acquire(
        self,
        runtime_id: str,
        device_id: str,
        *,
        cancelled: LeaseCancelCallback | None = None,
        deadline: float | None = None,
        clock: LeaseClock = time.monotonic,
    ) -> Iterator[AcceleratorLeaseKey | None]:
        """Own ``runtime_id``/``device_id`` until the context exits.

        ``deadline`` is an absolute value from ``clock``.  A cancellation
        callback may return true or raise a caller-owned typed exception; a
        raised exception is intentionally propagated unchanged.
        """

        if not callable(clock):
            raise TypeError("clock must be callable.")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable or None.")
        normalized_runtime = str(runtime_id).strip()
        normalized_device = str(device_id).strip()
        if _is_cpu_identity(normalized_runtime, normalized_device):
            yield None
            return
        key = AcceleratorLeaseKey(normalized_runtime, normalized_device)
        normalized_deadline = _normalized_deadline(deadline)
        self._acquire_key(
            key,
            cancelled=cancelled,
            deadline=normalized_deadline,
            clock=clock,
        )
        try:
            yield key
        finally:
            self._release_key(key)

    def _acquire_key(
        self,
        key: AcceleratorLeaseKey,
        *,
        cancelled: LeaseCancelCallback | None,
        deadline: float | None,
        clock: LeaseClock,
    ) -> None:
        thread_id = threading.get_ident()
        token = object()
        queued = False
        try:
            while True:
                _check_wait_abort(key, cancelled, deadline, clock)
                with self._condition:
                    state = self._states.setdefault(key, _LeaseState())
                    if state.owner_thread_id == thread_id:
                        state.depth += 1
                        return
                    if not queued:
                        state.waiters.append(token)
                        queued = True
                    if (
                        state.owner_thread_id is None
                        and state.waiters
                        and state.waiters[0] is token
                    ):
                        state.waiters.popleft()
                        state.owner_thread_id = thread_id
                        state.depth = 1
                        return
                    timeout = _wait_timeout(deadline, clock)
                    self._condition.wait(timeout=timeout)
        except BaseException:
            if queued:
                with self._condition:
                    state = self._states.get(key)
                    if state is not None:
                        try:
                            state.waiters.remove(token)
                        except ValueError:
                            pass
                        self._discard_idle_state(key, state)
                        self._condition.notify_all()
            raise

    def _release_key(self, key: AcceleratorLeaseKey) -> None:
        thread_id = threading.get_ident()
        with self._condition:
            state = self._states.get(key)
            if state is None or state.owner_thread_id != thread_id:
                raise RuntimeError(
                    "Accelerator leases must be released by their owning thread."
                )
            state.depth -= 1
            if state.depth < 0:
                raise RuntimeError("Accelerator lease depth became invalid.")
            if state.depth == 0:
                state.owner_thread_id = None
                self._discard_idle_state(key, state)
                self._condition.notify_all()

    def _discard_idle_state(
        self,
        key: AcceleratorLeaseKey,
        state: _LeaseState,
    ) -> None:
        if state.owner_thread_id is None and not state.waiters:
            self._states.pop(key, None)


PROCESS_ACCELERATOR_LEASES = AcceleratorLeaseManager()


@contextmanager
def accelerator_lease(
    runtime_id: str,
    device_id: str,
    *,
    cancelled: LeaseCancelCallback | None = None,
    deadline: float | None = None,
    clock: LeaseClock = time.monotonic,
) -> Iterator[AcceleratorLeaseKey | None]:
    """Acquire the process-wide lease for one runtime/device pair."""

    with PROCESS_ACCELERATOR_LEASES.acquire(
        runtime_id,
        device_id,
        cancelled=cancelled,
        deadline=deadline,
        clock=clock,
    ) as key:
        yield key


def _is_cpu_identity(runtime_id: str, device_id: str) -> bool:
    return runtime_id == _CPU_RUNTIME_ID or device_id.lower().startswith("cpu:")


def _normalized_deadline(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise TypeError("deadline must be a finite number or None.")
    value = float(deadline)
    if not math.isfinite(value):
        raise ValueError("deadline must be finite.")
    return value


def _check_wait_abort(
    key: AcceleratorLeaseKey,
    cancelled: LeaseCancelCallback | None,
    deadline: float | None,
    clock: LeaseClock,
) -> None:
    if cancelled is not None and bool(cancelled()):
        raise AcceleratorLeaseCancelled(
            "Accelerator execution was cancelled while waiting for "
            f"{key.runtime_id} on {key.device_id}."
        )
    if deadline is not None and _read_clock(clock) >= deadline:
        raise AcceleratorLeaseDeadlineExceeded(
            "Timed out while waiting for accelerator ownership of "
            f"{key.runtime_id} on {key.device_id}."
        )


def _wait_timeout(deadline: float | None, clock: LeaseClock) -> float:
    if deadline is None:
        return _WAIT_POLL_SECONDS
    return max(0.0, min(_WAIT_POLL_SECONDS, deadline - _read_clock(clock)))


def _read_clock(clock: LeaseClock) -> float:
    value = float(clock())
    if not math.isfinite(value):
        raise ValueError("clock must return a finite value.")
    return value


__all__ = [
    "AcceleratorLeaseCancelled",
    "AcceleratorLeaseDeadlineExceeded",
    "AcceleratorLeaseKey",
    "AcceleratorLeaseManager",
    "PROCESS_ACCELERATOR_LEASES",
    "accelerator_lease",
]
