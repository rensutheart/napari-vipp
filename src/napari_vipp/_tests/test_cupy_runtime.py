from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.gpu.cupy_runtime as cupy_runtime_module
from napari_vipp.core.accelerator_lease import AcceleratorLeaseManager
from napari_vipp.core.compute_registry import (
    RuntimeExceptionKind,
    RuntimeProtocol,
)
from napari_vipp.core.gpu.cupy_runtime import CUDACleanupError, CuPyRuntime


class _FakeOutOfMemoryError(MemoryError):
    pass


class _FakeCUDARuntimeError(RuntimeError):
    def __init__(self, message: str, *, status: int = 999) -> None:
        super().__init__(message)
        self.status = status


class _FakeCompileException(RuntimeError):
    pass


class _FakeMemory:
    def __init__(self, pool: _FakeMemoryPool, size: int) -> None:
        self.pool = weakref.ref(pool)
        self.device_id = pool.device_id
        self._size = size
        self._returned = False

    def free(self) -> None:
        pool = self.pool()
        if pool is not None:
            pool.forced_free_calls += 1
        raise AssertionError("Runtime code must never force-free pooled memory.")

    def __del__(self) -> None:
        if self._returned:
            return
        pool = self.pool()
        if pool is not None:
            pool.used -= self._size
        self._returned = True


class _FakeMemoryPool:
    def __init__(self, owner: _FakeCuPy) -> None:
        self.owner = owner
        self.device_id = owner.device_index
        self.used = 0
        self.reserved = 0
        self.limit = None
        self.forced_free_calls = 0
        self.free_devices: list[int] = []

    def set_limit(self, *, size: int) -> None:
        self.limit = size

    def malloc(self, size: int) -> _FakeMemory:
        if self.limit is not None and self.used + size > self.limit:
            raise _FakeOutOfMemoryError("private pool limit exceeded")
        self.used += size
        self.reserved = max(self.reserved, self.used)
        return _FakeMemory(self, size)

    def used_bytes(self) -> int:
        return self.used

    def total_bytes(self) -> int:
        return self.reserved

    def free_all_blocks(self) -> None:
        self.free_devices.append(self.owner.device_index)
        if self.owner.device_index != self.device_id:
            raise AssertionError("A memory pool was cleaned on the wrong device.")
        self.reserved = self.used


class _FakeArray:
    def __init__(
        self,
        owner: _FakeCuPy,
        value: np.ndarray,
        *,
        memory: _FakeMemory | None = None,
    ) -> None:
        self._owner = owner
        self._value = np.asarray(value)
        allocator = owner.allocator or owner.default_pool.malloc
        if memory is None:
            memory = allocator(int(self._value.nbytes))
        self.data = SimpleNamespace(mem=memory)

    @property
    def shape(self):
        return self._value.shape

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def nbytes(self):
        return self._value.nbytes

    @property
    def device(self):
        return SimpleNamespace(id=self.data.mem.device_id)

    @property
    def flags(self):
        return self._value.flags

    def reshape(self, shape):
        return _FakeArray(
            self._owner,
            self._value.reshape(shape),
            memory=self.data.mem,
        )

    def __getitem__(self, key):
        return _FakeArray(
            self._owner,
            self._value[key],
            memory=self.data.mem,
        )

    def set(self, value) -> None:
        self._value[...] = value

    def get(self, *, out=None, blocking=True):
        assert blocking
        if out is None:
            return self._value.copy()
        out[...] = self._value
        return out


class _FakeDevice:
    def __init__(self, owner: _FakeCuPy, index: int) -> None:
        self.owner = owner
        self.id = index
        self.compute_capability = "12.0"
        self._previous = 0

    def __enter__(self):
        self._previous = self.owner.device_index
        self.owner.device_index = self.id
        return self

    def __exit__(self, *_args):
        self.owner.device_index = self._previous


class _FakeStream:
    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1


class _FakeRuntimeAPI:
    cudaErrorInvalidDevice = 10
    CUDARuntimeError = _FakeCUDARuntimeError

    def __init__(self, owner: _FakeCuPy) -> None:
        self.owner = owner

    def getDeviceCount(self) -> int:
        return 1

    def getDeviceProperties(self, index: int):
        assert index == 0
        return {"name": b"Fake RTX", "major": 12, "minor": 0}

    def memGetInfo(self):
        reserved = sum(pool.reserved for pool in self.owner.pools)
        return self.owner.total_memory - reserved, self.owner.total_memory

    def driverGetVersion(self) -> int:
        return 13020

    def runtimeGetVersion(self) -> int:
        return 13020


class _FakeFFTPlan:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeFFTPlanNode:
    def __init__(self, plan: _FakeFFTPlan) -> None:
        self.plan = plan


class _FakeFFTPlanCache:
    """Small MRU-first model of CuPy 14's per-thread/device PlanCache."""

    def __init__(self) -> None:
        self.maximum_size = 16
        self.maximum_memsize = -1
        self.entries: list[tuple[object, _FakeFFTPlan]] = []
        self.events: list[tuple[str, object]] = []

    def __iter__(self):
        return iter(tuple((key, _FakeFFTPlanNode(plan)) for key, plan in self.entries))

    def __delitem__(self, key: object) -> None:
        self.events.append(("delete", key))
        for index, (candidate, _plan) in enumerate(self.entries):
            if candidate == key:
                del self.entries[index]
                return
        raise KeyError(key)

    def __setitem__(self, key: object, plan: _FakeFFTPlan) -> None:
        self.events.append(("insert", key))
        self.entries = [
            (candidate, candidate_plan)
            for candidate, candidate_plan in self.entries
            if candidate != key
        ]
        if self.maximum_size == 0:
            return
        self.entries.insert(0, (key, plan))
        self._trim()

    def get_size(self) -> int:
        return self.maximum_size

    def get_memsize(self) -> int:
        return self.maximum_memsize

    def get_curr_size(self) -> int:
        return len(self.entries)

    def get_curr_memsize(self) -> int:
        return 0

    def set_size(self, size: int) -> None:
        self.events.append(("set_size", size))
        self.maximum_size = size
        self._trim()

    def set_memsize(self, memsize: int) -> None:
        self.events.append(("set_memsize", memsize))
        self.maximum_memsize = memsize

    def _trim(self) -> None:
        if self.maximum_size >= 0:
            del self.entries[self.maximum_size :]


class _FakeFFTConfig:
    def __init__(self, owner: _FakeCuPy) -> None:
        self.owner = owner
        self.caches: dict[tuple[int, int], _FakeFFTPlanCache] = {}

    def get_plan_cache(self) -> _FakeFFTPlanCache:
        key = (threading.get_ident(), self.owner.device_index)
        return self.caches.setdefault(key, _FakeFFTPlanCache())


class _FakeFFT:
    def __init__(self, owner: _FakeCuPy) -> None:
        self.owner = owner
        self.config = _FakeFFTConfig(owner)

    def rfftn(self, value: _FakeArray) -> _FakeArray:
        cache = self.config.get_plan_cache()
        cache[("rfftn", value.shape)] = _FakeFFTPlan("rfftn")
        return _FakeArray(self.owner, np.fft.rfftn(value._value))

    def irfftn(self, value: _FakeArray, *, s) -> _FakeArray:
        cache = self.config.get_plan_cache()
        cache[("irfftn", tuple(s))] = _FakeFFTPlan("irfftn")
        axes = tuple(range(len(s)))
        return _FakeArray(self.owner, np.fft.irfftn(value._value, s=s, axes=axes))


class _FakeCuPy:
    __version__ = "14.1.1"
    float32 = np.float32
    ndarray = _FakeArray

    def __init__(self) -> None:
        self.total_memory = 4 * 1024**3
        self.device_index = 0
        self.allocator = None
        self.pools: list[_FakeMemoryPool] = []
        self.default_pool = _FakeMemoryPool(self)
        self.pools.append(self.default_pool)
        self.stream = _FakeStream()
        self.fft = _FakeFFT(self)
        runtime = _FakeRuntimeAPI(self)

        def memory_pool_factory():
            pool = _FakeMemoryPool(self)
            self.pools.append(pool)
            return pool

        self.cuda = SimpleNamespace(
            runtime=runtime,
            memory=SimpleNamespace(OutOfMemoryError=_FakeOutOfMemoryError),
            compiler=SimpleNamespace(CompileException=_FakeCompileException),
            Device=lambda index: _FakeDevice(self, index),
            MemoryPool=memory_pool_factory,
            using_allocator=self._using_allocator,
            get_current_stream=lambda: self.stream,
        )

    @contextmanager
    def _using_allocator(self, allocator):
        previous = self.allocator
        self.allocator = allocator
        try:
            yield
        finally:
            self.allocator = previous

    def arange(self, size, *, dtype):
        return _FakeArray(self, np.arange(size, dtype=dtype))

    def empty(self, shape, *, dtype):
        return _FakeArray(self, np.empty(shape, dtype=dtype))

    def ascontiguousarray(self, value):
        return _FakeArray(self, np.ascontiguousarray(value._value))


class _FakeNdimage:
    def __init__(self, cupy: _FakeCuPy) -> None:
        self.cupy = cupy

    def gaussian_filter(self, value, *, sigma):
        assert sigma == 1.0
        return _FakeArray(self.cupy, value._value.copy())

    def median_filter(self, value, *, size):
        assert size == 3
        return _FakeArray(self.cupy, value._value.copy())


def _fake_runtime():
    cupy = _FakeCuPy()
    ndimage = _FakeNdimage(cupy)
    imports: list[str] = []

    def load(name: str):
        imports.append(name)
        if name == "cupy":
            return cupy
        if name == "cupyx.scipy.ndimage":
            return ndimage
        raise ModuleNotFoundError(name)

    runtime = CuPyRuntime(
        module_loader=load,
        platform_name="win32",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
    )
    return runtime, cupy, imports


def _fake_plan_entries(
    cache: _FakeFFTPlanCache,
) -> tuple[tuple[object, _FakeFFTPlan], ...]:
    return tuple((key, node.plan) for key, node in cache)


def test_runtime_modules_are_import_safe_without_optional_dependencies():
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    code = (
        "import sys; "
        "import napari_vipp.core.gpu; "
        "import napari_vipp.core.gpu.cupy_runtime; "
        "assert not any(n == 'cupy' or n.startswith('cupyx') or "
        "n.startswith('cucim') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_successful_probe_is_real_cached_and_json_safe():
    runtime, cupy, imports = _fake_runtime()

    result = runtime.probe()

    assert result.available
    assert result.selected_device_id == "cuda:0"
    assert result.devices[0].display_name == "Fake RTX"
    assert result.devices[0].total_memory_bytes == cupy.total_memory
    assert dict(result.devices[0].metadata)["compute_capability"] == "12.0"
    assert imports == ["cupy", "cupyx.scipy.ndimage"]
    assert json.loads(json.dumps(result.as_dict()))["available"] is True
    assert runtime.probe() is result
    assert isinstance(runtime, RuntimeProtocol)


def test_private_scope_transfers_accounts_and_releases():
    runtime, cupy, _imports = _fake_runtime()
    host = np.arange(12, dtype=np.float32).reshape(3, 4)

    with runtime.execution_scope(
        memory_limit_bytes=128 * 1024**2,
        safety_reserve_bytes=64 * 1024**2,
    ):
        device = runtime.to_device(host)
        np.testing.assert_array_equal(runtime.to_host(device), host)
        snapshot = runtime.memory_snapshot()
        assert snapshot.runtime_live_bytes == host.nbytes
        assert snapshot.runtime_reserved_bytes >= snapshot.runtime_live_bytes
        assert snapshot.out_of_pool_bytes == 0
        runtime.release(device)
        del device
        assert runtime.memory_snapshot().runtime_live_bytes == 0

    execution_pool = cupy.pools[-1]
    assert execution_pool.limit == 128 * 1024**2
    assert execution_pool.used_bytes() == 0
    assert execution_pool.total_bytes() == 0
    with pytest.raises(RuntimeError, match="active execution scope"):
        runtime.to_device(host)


@pytest.mark.parametrize("outcome", ["success", "body_failure", "cleanup_failure"])
def test_scope_restores_external_fft_plan_cache_for_every_exit(outcome):
    runtime, cupy, _imports = _fake_runtime()
    assert runtime.probe().available
    cache = cupy.fft.config.get_plan_cache()
    cache.set_size(7)
    cache.set_memsize(123_456)
    oldest = _FakeFFTPlan("oldest")
    newest = _FakeFFTPlan("newest")
    cache[("external", 1)] = oldest
    cache[("external", 2)] = newest
    before = _fake_plan_entries(cache)
    cache.events.clear()
    escaped = None

    def exercise_scope() -> None:
        nonlocal escaped
        with runtime.execution_scope(safety_reserve_bytes=0):
            assert cache.get_size() == 0
            assert cache.get_memsize() == 123_456
            assert cache.get_curr_size() == 0
            # This models a CuPy FFT attempting to cache a plan in the VIPP
            # scope.  A zero size limit must keep the cache empty.
            cache[("vipp", 1)] = _FakeFFTPlan("temporary")
            assert cache.get_curr_size() == 0
            if outcome == "body_failure":
                raise ValueError("synthetic body failure")
            if outcome == "cleanup_failure":
                escaped = runtime.to_device(np.ones(32, dtype=np.float32))

    if outcome == "body_failure":
        with pytest.raises(ValueError, match="synthetic body failure"):
            exercise_scope()
    elif outcome == "cleanup_failure":
        with pytest.raises(CUDACleanupError, match="live private allocation"):
            exercise_scope()
    else:
        exercise_scope()

    after = _fake_plan_entries(cache)
    assert cache.get_size() == 7
    assert cache.get_memsize() == 123_456
    assert tuple(key for key, _plan in after) == tuple(key for key, _plan in before)
    assert all(
        restored_plan is original_plan
        for (_key, restored_plan), (_original_key, original_plan) in zip(
            after, before, strict=True
        )
    )
    assert ("set_size", 0) in cache.events
    assert ("set_memsize", 123_456) in cache.events
    assert cache.events.count(("delete", ("external", 1))) == 1
    assert cache.events.count(("delete", ("external", 2))) == 1
    del escaped


def test_nested_scope_is_rejected_without_destroying_outer_scope():
    runtime, _cupy, _imports = _fake_runtime()
    with runtime.execution_scope(safety_reserve_bytes=0):
        with pytest.raises(RuntimeError, match="Nested"):
            with runtime.execution_scope(safety_reserve_bytes=0):
                pass
        value = runtime.to_device(np.ones(2, dtype=np.float32))
        runtime.release(value)
        del value


def test_separate_runtime_instances_serialize_the_same_physical_device():
    first_runtime, _first_cupy, _first_imports = _fake_runtime()
    second_runtime, _second_cupy, _second_imports = _fake_runtime()
    assert first_runtime.probe().available
    assert second_runtime.probe().available
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with first_runtime.execution_scope(safety_reserve_bytes=0):
            first_entered.set()
            assert release_first.wait(timeout=5)

    def second() -> None:
        second_started.set()
        with second_runtime.execution_scope(safety_reserve_bytes=0):
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


def test_snapshot_and_outer_execution_lease_have_one_lock_order(monkeypatch):
    """An outer process owner and diagnostic snapshot must never deadlock."""

    runtime, _cupy, _imports = _fake_runtime()
    assert runtime.probe().available
    manager = AcceleratorLeaseManager()
    monkeypatch.setattr(cupy_runtime_module, "accelerator_lease", manager.acquire)
    snapshot_has_instance_lock = threading.Event()
    execution_is_attempting_instance_lock = threading.Event()
    lease_held = threading.Event()
    execution_finished = threading.Event()
    snapshot_finished = threading.Event()
    failures: list[BaseException] = []
    underlying_lock = runtime._lock

    class ObservedRLock:
        def __init__(self) -> None:
            self.reported_snapshot_acquisition = False

        def acquire(self, *args, **kwargs):
            acquired = underlying_lock.acquire(*args, **kwargs)
            if (
                acquired
                and threading.current_thread().name == "snapshot-thread"
                and not self.reported_snapshot_acquisition
            ):
                self.reported_snapshot_acquisition = True
                snapshot_has_instance_lock.set()
                if not execution_is_attempting_instance_lock.wait(timeout=5):
                    raise AssertionError("execution thread did not contend")
            return acquired

        def release(self) -> None:
            underlying_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args) -> None:
            self.release()

    runtime._lock = ObservedRLock()

    def execute_with_outer_lease() -> None:
        try:
            with manager.acquire("cuda-cupy", "cuda:0"):
                lease_held.set()
                if not snapshot_has_instance_lock.wait(timeout=5):
                    raise AssertionError("snapshot did not acquire the instance lock")
                execution_is_attempting_instance_lock.set()
                with runtime.execution_scope(safety_reserve_bytes=0):
                    pass
            execution_finished.set()
        except BaseException as exc:
            failures.append(exc)

    def take_snapshot() -> None:
        try:
            if not lease_held.wait(timeout=5):
                raise AssertionError("execution did not acquire the process lease")
            runtime.memory_snapshot(device_id="cuda:0")
            snapshot_finished.set()
        except BaseException as exc:
            failures.append(exc)

    execution_thread = threading.Thread(
        target=execute_with_outer_lease,
        name="execution-thread",
        daemon=True,
    )
    snapshot_thread = threading.Thread(
        target=take_snapshot,
        name="snapshot-thread",
        daemon=True,
    )
    execution_thread.start()
    assert lease_held.wait(timeout=5)
    snapshot_thread.start()
    execution_thread.join(timeout=5)
    snapshot_thread.join(timeout=5)

    assert execution_finished.is_set(), "execution deadlocked on the instance lock"
    assert snapshot_finished.is_set(), "snapshot deadlocked on the process lease"
    assert not execution_thread.is_alive()
    assert not snapshot_thread.is_alive()
    assert failures == []


def test_release_tracks_shared_allocation_without_force_freeing_aliases():
    runtime, cupy, _imports = _fake_runtime()
    host = np.arange(16, dtype=np.float32)

    with runtime.execution_scope(safety_reserve_bytes=0):
        base = runtime.to_device(host)
        alias = base[4:8]
        assert runtime.allocation_identity(base) == runtime.allocation_identity(alias)

        runtime.release(base)
        del base
        replacement = runtime.to_device(np.full(16, 777, dtype=np.float32))

        np.testing.assert_array_equal(runtime.to_host(alias), host[4:8])
        runtime.release(alias)
        runtime.release(replacement)
        del alias, replacement

    assert sum(pool.forced_free_calls for pool in cupy.pools) == 0


def test_external_pool_array_is_rejected_and_left_untouched():
    runtime, cupy, _imports = _fake_runtime()
    external = cupy.empty((8,), dtype=np.float32)
    external.set(np.arange(8, dtype=np.float32))
    external_alias = external[2:6]

    with runtime.execution_scope(safety_reserve_bytes=0):
        assert runtime.is_device_value(external)
        with pytest.raises(TypeError, match="private memory pool"):
            runtime.release(external)
        private = runtime.to_device(np.full(8, 999, dtype=np.float32))
        np.testing.assert_array_equal(external_alias._value, np.arange(2, 6))
        runtime.release(private)
        del private

    assert cupy.default_pool.forced_free_calls == 0
    del external_alias, external
    cupy.default_pool.free_all_blocks()


def test_scope_fails_closed_and_retains_terminal_live_allocation_truth():
    runtime, cupy, _imports = _fake_runtime()
    escaped = None

    with pytest.raises(RuntimeError, match="live private allocation"):
        with runtime.execution_scope(safety_reserve_bytes=0):
            escaped = runtime.to_device(np.ones(32, dtype=np.float32))

    snapshot = runtime.memory_snapshot()
    assert snapshot.runtime_live_bytes == 32 * np.dtype(np.float32).itemsize
    assert snapshot.runtime_reserved_bytes >= snapshot.runtime_live_bytes
    assert runtime.probe().reason_code == "runtime_unhealthy"
    assert sum(pool.forced_free_calls for pool in cupy.pools) == 0

    del escaped
    # The terminal checkpoint remains diagnostic truth even after a late
    # external reference finally disappears.
    assert runtime.memory_snapshot().runtime_live_bytes == snapshot.runtime_live_bytes


def test_oom_with_live_private_residue_becomes_nonretryable_cleanup_failure():
    runtime, _cupy, _imports = _fake_runtime()
    escaped = None
    original = _FakeOutOfMemoryError("synthetic operation OOM")

    with pytest.raises(CUDACleanupError) as caught:
        with runtime.execution_scope(safety_reserve_bytes=0):
            escaped = runtime.to_device(np.ones(32, dtype=np.float32))
            raise original

    assert caught.value.__cause__ is original
    failure = runtime.classify_exception(caught.value)
    assert failure.kind is RuntimeExceptionKind.KERNEL_FAILURE
    assert failure.reason_code == "cuda_cleanup_incomplete"
    assert not failure.retryable
    assert runtime.probe().reason_code == "runtime_unhealthy"
    del escaped


def test_external_cache_delta_is_reported_without_poisoning_runtime():
    runtime, cupy, _imports = _fake_runtime()

    with runtime.execution_scope(safety_reserve_bytes=0):
        with cupy._using_allocator(cupy.default_pool.malloc):
            cache_seed = cupy.empty((64,), dtype=np.float32)
        del cache_seed

    snapshot = runtime.memory_snapshot()
    assert snapshot.runtime_live_bytes == 0
    assert snapshot.runtime_reserved_bytes == 0
    assert snapshot.out_of_pool_bytes == 64 * np.dtype(np.float32).itemsize
    assert runtime.probe().available
    cupy.default_pool.free_all_blocks()


def test_scope_cleanup_uses_the_selected_device_context():
    runtime, cupy, _imports = _fake_runtime()
    cupy.device_index = 7

    with runtime.execution_scope(device_id="cuda:0", safety_reserve_bytes=0):
        value = runtime.to_device(np.ones(4, dtype=np.float32))
        runtime.release(value)
        del value

    execution_pool = cupy.pools[-1]
    assert execution_pool.device_id == 0
    assert execution_pool.free_devices
    assert set(execution_pool.free_devices) == {0}
    assert cupy.device_index == 7


def test_close_finalizes_an_active_pool_on_its_selected_device():
    runtime, cupy, _imports = _fake_runtime()
    cupy.device_index = 7

    with runtime.execution_scope(device_id="cuda:0", safety_reserve_bytes=0):
        value = runtime.to_device(np.ones(4, dtype=np.float32))
        runtime.release(value)
        del value
        runtime.close()

    execution_pool = cupy.pools[-1]
    assert execution_pool.free_devices
    assert set(execution_pool.free_devices) == {0}
    assert cupy.device_index == 7
    assert runtime.probe().reason_code == "runtime_closed"


def test_exception_classification_uses_types_not_oom_text():
    runtime, _cupy, _imports = _fake_runtime()
    runtime.probe()

    typed = runtime.classify_exception(_FakeOutOfMemoryError("full"))
    generic = runtime.classify_exception(RuntimeError("out of memory"))
    invalid = runtime.classify_exception(_FakeCUDARuntimeError("bad device", status=10))

    assert typed.kind is RuntimeExceptionKind.OUT_OF_MEMORY
    assert typed.retryable
    assert generic.kind is RuntimeExceptionKind.UNKNOWN
    assert invalid.kind is RuntimeExceptionKind.INVALID_DEVICE


@pytest.mark.parametrize(
    ("failure_name", "error", "expected_reason"),
    [
        ("cupy", ModuleNotFoundError("cupy"), "cupy_missing"),
        (
            "cupyx.scipy.ndimage",
            ModuleNotFoundError("cupyx.scipy.ndimage"),
            "cupyx_ndimage_missing",
        ),
        (
            "cupyx.scipy.ndimage",
            OSError("missing DLL"),
            "cupyx_ndimage_import_failed",
        ),
    ],
)
def test_missing_or_broken_components_are_structured(
    failure_name, error, expected_reason
):
    cupy = _FakeCuPy()

    def load(name: str):
        if name == failure_name:
            raise error
        if name == "cupy":
            return cupy
        return _FakeNdimage(cupy)

    runtime = CuPyRuntime(
        module_loader=load,
        platform_name="linux",
        python_implementation="CPython",
        python_version=(3, 12),
        pointer_bits=64,
    )

    result = runtime.probe()

    assert not result.available
    assert result.reason_code == expected_reason


def test_close_before_probe_does_not_import_cupy_and_is_idempotent():
    imports: list[str] = []
    runtime = CuPyRuntime(module_loader=lambda name: imports.append(name))

    runtime.close()
    runtime.close()

    assert imports == []
    assert runtime.probe().reason_code == "runtime_closed"


def test_real_fft_work_areas_leave_no_private_residue_and_runtime_reuses():
    cupy = pytest.importorskip("cupy")
    signal = pytest.importorskip("cupyx.scipy.signal")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)
    selected = probe.selected_device_id
    device_index = int(selected.partition(":")[2])
    with cupy.cuda.Device(device_index):
        cache = cupy.fft.config.get_plan_cache()
        original_size = cache.get_size()
        original_memsize = cache.get_memsize()
        original_entries = tuple((key, node.plan) for key, node in cache)

    host_image = np.ones((512, 512), dtype=np.float32)
    host_psf = np.full((31, 31), 1.0 / (31 * 31), dtype=np.float32)
    for _attempt in range(2):
        with runtime.execution_scope(
            device_id=selected,
            memory_limit_bytes=512 * 1024**2,
            safety_reserve_bytes=0,
        ):
            image = runtime.to_device(host_image, device_id=selected)
            psf = runtime.to_device(host_psf, device_id=selected)
            result = signal.fftconvolve(image, psf, mode="same")
            runtime.allocation_identity(result)
            runtime.synchronize(device_id=selected)
            runtime.release(result)
            runtime.release(psf)
            runtime.release(image)
            del result, psf, image

        snapshot = runtime.memory_snapshot(device_id=selected)
        assert snapshot.runtime_live_bytes == 0
        assert snapshot.runtime_reserved_bytes == 0
        assert snapshot.out_of_pool_bytes == 0

    with cupy.cuda.Device(device_index):
        restored_entries = tuple((key, node.plan) for key, node in cache)
        assert cache.get_size() == original_size
        assert cache.get_memsize() == original_memsize
        assert len(restored_entries) == len(original_entries)
        assert all(
            restored_key == original_key and restored_plan is original_plan
            for (restored_key, restored_plan), (original_key, original_plan) in zip(
                restored_entries, original_entries, strict=True
            )
        )
    runtime.close()


def test_real_runtime_scope_releases_its_private_pool_when_cuda_is_available():
    pytest.importorskip("cupy")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)

    host = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    with runtime.execution_scope(
        device_id=probe.selected_device_id,
        memory_limit_bytes=128 * 1024**2,
        safety_reserve_bytes=0,
    ):
        device = runtime.to_device(host, device_id=probe.selected_device_id)
        np.testing.assert_array_equal(runtime.to_host(device), host)
        assert (
            runtime.memory_snapshot(
                device_id=probe.selected_device_id
            ).runtime_live_bytes
            >= host.nbytes
        )
        runtime.release(device)
        del device

    snapshot = runtime.memory_snapshot(device_id=probe.selected_device_id)
    assert snapshot.runtime_live_bytes == 0
    assert snapshot.runtime_reserved_bytes == 0
    runtime.close()


def test_real_runtime_classifies_oom_cleans_and_recovers_on_next_scope():
    pytest.importorskip("cupy")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)

    oversized = np.ones(2 * 1024**2 // np.dtype(np.float32).itemsize, dtype=np.float32)
    with pytest.raises(Exception) as raised:
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=1024**2,
            safety_reserve_bytes=0,
        ):
            runtime.to_device(oversized, device_id=probe.selected_device_id)

    failure = runtime.classify_exception(raised.value)
    assert failure.kind is RuntimeExceptionKind.OUT_OF_MEMORY
    assert failure.retryable
    snapshot = runtime.memory_snapshot(device_id=probe.selected_device_id)
    assert snapshot.runtime_live_bytes == 0
    assert snapshot.runtime_reserved_bytes == 0
    assert runtime.probe().available

    recovery = np.arange(4096, dtype=np.float32)
    with runtime.execution_scope(
        device_id=probe.selected_device_id,
        memory_limit_bytes=8 * 1024**2,
        safety_reserve_bytes=0,
    ):
        device = runtime.to_device(recovery, device_id=probe.selected_device_id)
        np.testing.assert_array_equal(runtime.to_host(device), recovery)
        runtime.release(device)
        del device

    recovered = runtime.memory_snapshot(device_id=probe.selected_device_id)
    assert recovered.runtime_live_bytes == 0
    assert recovered.runtime_reserved_bytes == 0
    runtime.close()


def test_real_runtime_keeps_a_view_valid_after_releasing_its_base():
    cupy = pytest.importorskip("cupy")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)

    host = np.arange(1024, dtype=np.float32)
    with runtime.execution_scope(
        device_id=probe.selected_device_id,
        memory_limit_bytes=128 * 1024**2,
        safety_reserve_bytes=0,
    ):
        base = runtime.to_device(host, device_id=probe.selected_device_id)
        alias = base[100:110]
        assert runtime.allocation_identity(base) == runtime.allocation_identity(alias)
        runtime.release(base)
        del base

        replacement = cupy.full(1024, 777, dtype=cupy.float32)
        runtime.allocation_identity(replacement)
        np.testing.assert_array_equal(runtime.to_host(alias), host[100:110])
        runtime.release(alias)
        runtime.release(replacement)
        del alias, replacement

    assert runtime.memory_snapshot().runtime_live_bytes == 0
    runtime.close()


def test_real_runtime_rejects_an_external_pool_array_without_corruption():
    cupy = pytest.importorskip("cupy")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)

    device_index = int(probe.selected_device_id.partition(":")[2])
    with cupy.cuda.Device(device_index):
        external = cupy.arange(32, dtype=cupy.float32)
        external_alias = external[8:16]
    with runtime.execution_scope(
        device_id=probe.selected_device_id,
        memory_limit_bytes=128 * 1024**2,
        safety_reserve_bytes=0,
    ):
        with pytest.raises(TypeError, match="private memory pool"):
            runtime.release(external)
        private = runtime.to_device(
            np.full(32, 999, dtype=np.float32),
            device_id=probe.selected_device_id,
        )
        np.testing.assert_array_equal(
            cupy.asnumpy(external_alias),
            np.arange(8, 16, dtype=np.float32),
        )
        runtime.release(private)
        del private

    del external_alias, external
    with cupy.cuda.Device(device_index):
        cupy.get_default_memory_pool().free_all_blocks()
    runtime.close()


def test_real_runtime_reports_an_escaped_private_allocation_after_scope():
    pytest.importorskip("cupy")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)

    escaped = None
    with pytest.raises(RuntimeError, match="live private allocation"):
        with runtime.execution_scope(
            device_id=probe.selected_device_id,
            memory_limit_bytes=128 * 1024**2,
            safety_reserve_bytes=0,
        ):
            escaped = runtime.to_device(
                np.ones(1024 * 1024, dtype=np.float32),
                device_id=probe.selected_device_id,
            )

    snapshot = runtime.memory_snapshot(device_id=probe.selected_device_id)
    assert snapshot.runtime_live_bytes >= 4 * 1024**2
    assert snapshot.runtime_reserved_bytes >= snapshot.runtime_live_bytes
    assert runtime.probe().reason_code == "runtime_unhealthy"
    del escaped
    runtime.close()


def test_real_runtime_reports_external_cache_without_claiming_ownership():
    cupy = pytest.importorskip("cupy")
    runtime = CuPyRuntime()
    probe = runtime.probe()
    if not probe.available:
        pytest.skip(probe.message)

    device_index = int(probe.selected_device_id.partition(":")[2])
    with cupy.cuda.Device(device_index):
        default_pool = cupy.get_default_memory_pool()
        default_pool.free_all_blocks()
    with runtime.execution_scope(
        device_id=probe.selected_device_id,
        memory_limit_bytes=128 * 1024**2,
        safety_reserve_bytes=0,
    ):
        with cupy.cuda.using_allocator(default_pool.malloc):
            cache_seed = cupy.empty(4 * 1024**2, dtype=cupy.uint8)
        del cache_seed

    snapshot = runtime.memory_snapshot(device_id=probe.selected_device_id)
    assert snapshot.runtime_live_bytes == 0
    assert snapshot.runtime_reserved_bytes == 0
    assert snapshot.out_of_pool_bytes >= 4 * 1024**2
    assert runtime.probe().available
    with cupy.cuda.Device(device_index):
        default_pool.free_all_blocks()
    runtime.close()
