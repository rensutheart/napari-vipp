from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from napari_vipp.core.accelerator_lease import (
    AcceleratorLeaseCancelled,
    AcceleratorLeaseDeadlineExceeded,
    AcceleratorLeaseManager,
)


@contextmanager
def _held_by_thread(manager: AcceleratorLeaseManager) -> Iterator[None]:
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with manager.acquire("cuda-cupy", "cuda:0"):
            entered.set()
            assert release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        yield
    finally:
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_same_runtime_and_device_are_serialized_and_released() -> None:
    manager = AcceleratorLeaseManager()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with manager.acquire("cuda-cupy", "cuda:0"):
            first_entered.set()
            assert release_first.wait(timeout=5)

    def second() -> None:
        second_started.set()
        with manager.acquire("cuda-cupy", "cuda:0"):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread.start()
    assert second_started.wait(timeout=5)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    assert second_entered.wait(timeout=5)
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()


def test_different_devices_can_proceed_concurrently() -> None:
    manager = AcceleratorLeaseManager()
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with manager.acquire("cuda-cupy", "cuda:0"):
            first_entered.set()
            assert release_first.wait(timeout=5)

    def second() -> None:
        with manager.acquire("cuda-cupy", "cuda:1"):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread.start()
    assert second_entered.wait(timeout=5)
    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)


def test_wait_is_cooperatively_cancellable_without_stealing_ownership() -> None:
    manager = AcceleratorLeaseManager()
    cancel = threading.Event()
    callback_observed = threading.Event()

    def cancelled() -> bool:
        callback_observed.set()
        return cancel.is_set()

    with manager.acquire("cuda-cupy", "cuda:0"):
        outcome: list[BaseException] = []

        def waiter() -> None:
            try:
                with manager.acquire(
                    "cuda-cupy",
                    "cuda:0",
                    cancelled=cancelled,
                ):
                    raise AssertionError("cancelled waiter acquired the lease")
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=waiter)
        thread.start()
        assert callback_observed.wait(timeout=5)
        cancel.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], AcceleratorLeaseCancelled)

    with manager.acquire("cuda-cupy", "cuda:0"):
        pass


def test_wait_honors_absolute_deadline_and_cleans_up_waiter() -> None:
    manager = AcceleratorLeaseManager()
    with _held_by_thread(manager):
        with pytest.raises(AcceleratorLeaseDeadlineExceeded, match="cuda:0"):
            with manager.acquire(
                "cuda-cupy",
                "cuda:0",
                deadline=time.monotonic() + 0.05,
            ):
                raise AssertionError("expired waiter acquired the lease")

    with manager.acquire("cuda-cupy", "cuda:0"):
        pass


def test_caller_owned_abort_exception_is_preserved() -> None:
    manager = AcceleratorLeaseManager()

    class CallerDeadline(RuntimeError):
        pass

    def abort() -> bool:
        raise CallerDeadline("whole transaction expired")

    with _held_by_thread(manager):
        with pytest.raises(CallerDeadline, match="whole transaction"):
            with manager.acquire(
                "cuda-cupy",
                "cuda:0",
                cancelled=abort,
            ):
                raise AssertionError("aborted waiter acquired the lease")


def test_reentrant_owner_and_exceptional_body_release_exactly_once() -> None:
    manager = AcceleratorLeaseManager()
    with pytest.raises(RuntimeError, match="body failed"):
        with manager.acquire("cuda-cupy", "cuda:0"):
            with manager.acquire("cuda-cupy", "cuda:0"):
                pass
            raise RuntimeError("body failed")

    acquired_by_other_thread = threading.Event()

    def acquire_after_failure() -> None:
        with manager.acquire("cuda-cupy", "cuda:0"):
            acquired_by_other_thread.set()

    thread = threading.Thread(target=acquire_after_failure)
    thread.start()
    assert acquired_by_other_thread.wait(timeout=5)
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_cpu_identity_never_enters_accelerator_registry() -> None:
    manager = AcceleratorLeaseManager()
    with manager.acquire("cpu-numpy", "") as key:
        assert key is None
    with manager.acquire("custom-runtime", "cpu:0") as key:
        assert key is None
