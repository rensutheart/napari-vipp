"""Lazy, resource-bounded CuPy runtime for VIPP CUDA execution."""

from __future__ import annotations

import importlib
import os
import platform
import struct
import sys
import tempfile
import threading
import traceback as traceback_module
from collections.abc import Callable, Hashable, Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType

import numpy as np

from napari_vipp.core.accelerator_lease import accelerator_lease
from napari_vipp.core.compute import MemoryTopology, canonical_digest
from napari_vipp.core.compute_registry import (
    RuntimeDevice,
    RuntimeExceptionInfo,
    RuntimeExceptionKind,
    RuntimeMemorySnapshot,
    RuntimeProbeResult,
)

_MIB = 1024 * 1024
_DEFAULT_RESERVE_BYTES = 512 * _MIB
_CUPY_CACHE_POLICY_LOCK = threading.RLock()
_CUPY_CACHE_FINGERPRINT_KEYS = (
    "cupy_cache_in_memory",
    "cupy_cache_reason",
    "cupy_cache_non_ascii_path_kinds",
)


class InvalidCUDADeviceError(ValueError):
    """Raised when a requested device is not part of the probe result."""


class CUDAAdmissionError(MemoryError):
    """Raised before execution when the requested memory budget is unsafe."""


class CUDACleanupError(RuntimeError):
    """Raised when a private CUDA scope cannot prove complete cleanup."""


@dataclass(slots=True)
class _FFTPlanCachePolicy:
    """Temporarily disable one thread/device cache without losing its plans."""

    cache: object
    maximum_size: int
    maximum_memsize: int
    entries: tuple[tuple[object, object], ...]
    restored: bool = False

    @classmethod
    def disable(cls, cupy: object) -> _FFTPlanCachePolicy:
        cache = _fft_plan_cache(cupy)
        maximum_size = _cache_integer(cache, "get_size")
        maximum_memsize = _cache_integer(cache, "get_memsize")
        entries = _fft_plan_cache_entries(cache)
        policy = cls(cache, maximum_size, maximum_memsize, entries)
        try:
            for key, _plan in entries:
                del cache[key]
            # CuPy 14.1.1 evicts existing entries when this is called.  The
            # plans above remain alive in ``entries`` and are restored later.
            cache.set_size(0)
            if _cache_integer(cache, "get_curr_size") != 0:
                raise RuntimeError(
                    "CuPy's FFT plan cache remained populated after disabling it."
                )
        except BaseException as setup_error:
            try:
                policy.restore()
            except BaseException as restore_error:
                raise CUDACleanupError(
                    "Could not restore the external CuPy FFT plan cache after "
                    "scope setup failed."
                ) from restore_error
            raise setup_error
        return policy

    def restore(self) -> None:
        if self.restored:
            return
        cache = self.cache
        original_by_key = {key: plan for key, plan in self.entries}
        missing = object()
        current = _fft_plan_cache_entries(cache)
        for key, plan in current:
            original = original_by_key.get(key, missing)
            if original is not missing and original is not plan:
                raise RuntimeError(
                    "The CuPy FFT plan cache replaced an externally owned key "
                    "while VIPP held the device lease."
                )
            # Any non-baseline entry was created inside this exclusive VIPP
            # scope.  Remove that attributable entry, never an unknown cache.
            del cache[key]
        cache.set_memsize(self.maximum_memsize)
        cache.set_size(self.maximum_size)
        # Iteration is MRU-first while insertion makes an entry MRU.  Reverse
        # insertion therefore restores the exact external LRU order.
        for key, plan in reversed(self.entries):
            cache[key] = plan
        if _cache_integer(cache, "get_size") != self.maximum_size:
            raise RuntimeError("CuPy FFT plan-cache size limit was not restored.")
        if _cache_integer(cache, "get_memsize") != self.maximum_memsize:
            raise RuntimeError("CuPy FFT plan-cache memory limit was not restored.")
        restored_entries = _fft_plan_cache_entries(cache)
        if len(restored_entries) != len(self.entries) or any(
            restored_key != expected_key or restored_plan is not expected_plan
            for (restored_key, restored_plan), (expected_key, expected_plan) in zip(
                restored_entries,
                self.entries,
                strict=True,
            )
        ):
            raise RuntimeError("CuPy FFT plan-cache entries were not restored exactly.")
        self.restored = True
        self.entries = ()


class CuPyRuntime:
    """A lazy CuPy adapter with a private memory pool per execution scope.

    The instance lock protects private-pool bookkeeping.  A process-wide
    runtime/device lease additionally coordinates separate registries and
    runtime instances, while leaving distinct devices independently schedulable.
    """

    runtime_id = "cuda-cupy"
    array_domain = "cuda-cupy"

    def __init__(
        self,
        *,
        module_loader: Callable[[str], ModuleType | object] | None = None,
        platform_name: str | None = None,
        python_implementation: str | None = None,
        python_version: tuple[int, int] | None = None,
        pointer_bits: int | None = None,
    ) -> None:
        self._module_loader = module_loader or importlib.import_module
        self._platform_name = sys.platform if platform_name is None else platform_name
        self._python_implementation = (
            platform.python_implementation()
            if python_implementation is None
            else python_implementation
        )
        self._python_version = (
            tuple(sys.version_info[:2]) if python_version is None else python_version
        )
        self._pointer_bits = (
            struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
        )
        self._cupy: object | None = None
        self._cupy_cache_metadata: tuple[tuple[str, str], ...] = ()
        self._ndimage: object | None = None
        self._probe_result: RuntimeProbeResult | None = None
        self._lock = threading.RLock()
        self._scope_active = False
        self._scope_thread_id: int | None = None
        self._active_device_id = ""
        self._active_pool: object | None = None
        self._active_pool_owner: object | None = None
        self._baseline_device_used_bytes = 0
        self._scope_limit_bytes = 0
        self._scope_reserve_bytes = 0
        self._closed = False
        self._unhealthy_message = ""
        self._terminal_snapshots: dict[str, RuntimeMemorySnapshot] = {}
        self._terminal_device_id = ""

    def probe(self, *, refresh: bool = False) -> RuntimeProbeResult:
        """Exercise CuPy and its two Phase-1 filters without leaking errors."""

        with self._lock:
            if self._closed:
                return self._remember_unavailable(
                    "runtime_closed", "This runtime instance has been closed."
                )
            if self._unhealthy_message:
                message = self._unhealthy_message
                if self._cupy is not None:
                    try:
                        self._refresh_cupy_cache_policy_unlocked(self._cupy)
                    except Exception as exc:
                        # The runtime is already unusable. Preserve that
                        # fail-closed result, but never attach a cache-policy
                        # claim which could no longer be revalidated.
                        self._cupy_cache_metadata = ()
                        message += (
                            " CuPy cache-policy revalidation also failed: "
                            + _exception_summary(exc)
                        )
                return self._remember_unavailable("runtime_unhealthy", message)
            # A cached failure before CuPy loaded has no process-wide cache
            # policy to revalidate.  Once CuPy has loaded, every probe must
            # pass through ``_load_cupy`` so an external environment mutation
            # cannot leave cached availability or metadata stale.
            if self._probe_result is not None and not refresh and self._cupy is None:
                return self._probe_result
            compatibility = self._compatibility_failure()
            if compatibility is not None:
                return self._remember_unavailable(*compatibility)

        # Optional imports and device enumeration do not allocate scientific
        # buffers.  They happen without retaining the instance lock so the
        # process lease can always be acquired before ``self._lock`` below.
        try:
            cupy = self._load_cupy()
        except Exception as exc:
            reason = (
                "cupy_missing"
                if isinstance(exc, ModuleNotFoundError)
                else "cupy_import_failed"
            )
            with self._lock:
                return self._remember_unavailable(
                    reason,
                    f"CuPy could not be loaded: {_exception_summary(exc)}",
                )
        with self._lock:
            if self._probe_result is not None and not refresh:
                return self._probe_result
        try:
            ndimage = self._load_ndimage()
        except Exception as exc:
            reason = (
                "cupyx_ndimage_missing"
                if isinstance(exc, ModuleNotFoundError)
                else "cupyx_ndimage_import_failed"
            )
            with self._lock:
                return self._remember_unavailable(
                    reason,
                    "CuPy's ndimage provider could not be loaded: "
                    + _exception_summary(exc),
                    version=str(getattr(cupy, "__version__", "unknown")),
                )

        version = str(getattr(cupy, "__version__", "unknown"))
        try:
            runtime = cupy.cuda.runtime
            device_count = int(runtime.getDeviceCount())
        except Exception as exc:
            info = self.classify_exception(exc)
            with self._lock:
                return self._remember_unavailable(
                    info.reason_code,
                    f"CuPy loaded but its CUDA probe failed: {_exception_summary(exc)}",
                    version=version,
                )
        if device_count < 1:
            with self._lock:
                return self._remember_unavailable(
                    "no_cuda_device",
                    "CuPy loaded, but CUDA reported no devices.",
                    version=version,
                )

        # Probe one device at a time, always taking its process lease before
        # re-entering instance state.  Avoiding simultaneous multi-device
        # leases also lets independent devices make progress without creating
        # a second lock-order hierarchy between device IDs.
        devices_list: list[RuntimeDevice] = []
        driver_version = ""
        runtime_version = ""
        for device_index in range(device_count):
            with accelerator_lease(self.runtime_id, f"cuda:{device_index}"):
                with self._lock:
                    if self._closed:
                        return self._remember_unavailable(
                            "runtime_closed",
                            "This runtime instance has been closed.",
                        )
                    if self._unhealthy_message:
                        return self._remember_unavailable(
                            "runtime_unhealthy", self._unhealthy_message
                        )
                    if self._probe_result is not None and not refresh:
                        return self._probe_result
                    try:
                        devices_list.append(self._probe_device(cupy, device_index))
                        self._exercise_runtime(
                            cupy,
                            ndimage,
                            device_index=device_index,
                        )
                        if device_index == 0:
                            driver_version = _optional_runtime_version(
                                runtime, "driverGetVersion"
                            )
                            runtime_version = _optional_runtime_version(
                                runtime, "runtimeGetVersion"
                            )
                    except Exception as exc:
                        info = self.classify_exception(exc)
                        return self._remember_unavailable(
                            info.reason_code,
                            "CuPy loaded but its CUDA probe failed: "
                            + _exception_summary(exc),
                            version=version,
                        )

        devices = tuple(devices_list)
        with self._lock:
            if self._closed:
                return self._remember_unavailable(
                    "runtime_closed", "This runtime instance has been closed."
                )
            if self._unhealthy_message:
                return self._remember_unavailable(
                    "runtime_unhealthy", self._unhealthy_message
                )
            if self._probe_result is not None and not refresh:
                return self._probe_result
            fingerprint_payload = {
                "runtime_id": self.runtime_id,
                "cupy": version,
                "driver": driver_version,
                "cuda_runtime": runtime_version,
                "devices": [
                    {
                        "name": device.display_name,
                        "memory": device.total_memory_bytes,
                        "metadata": dict(device.metadata),
                    }
                    for device in devices
                ],
            }
            cache_policy = _cupy_cache_fingerprint_metadata(self._cupy_cache_metadata)
            if cache_policy:
                fingerprint_payload["cupy_cache_policy"] = cache_policy
            self._probe_result = RuntimeProbeResult(
                runtime_id=self.runtime_id,
                available=True,
                version=version,
                devices=devices,
                selected_device_id=devices[0].device_id,
                reason_code="available",
                message=(
                    "CuPy completed allocation, Gaussian, median, fixed-size "
                    "uniform smoothing, and FFT convolution probes."
                ),
                environment_fingerprint=canonical_digest(fingerprint_payload),
                metadata=(
                    ("driver_version", driver_version),
                    ("cuda_runtime_version", runtime_version),
                    *self._cupy_cache_metadata,
                ),
            )
            return self._probe_result

    @contextmanager
    def execution_scope(
        self,
        *,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
    ) -> Iterator[None]:
        """Own one process-wide runtime/device lease and private allocator."""

        result = self.probe()
        selected = device_id.strip() or result.selected_device_id
        if not result.available or not selected:
            with self._execution_scope_unleased(
                probe_result=result,
                device_id=device_id,
                memory_limit_bytes=memory_limit_bytes,
                safety_reserve_bytes=safety_reserve_bytes,
            ):
                yield None
            return
        with accelerator_lease(self.runtime_id, selected):
            with self._execution_scope_unleased(
                probe_result=result,
                device_id=selected,
                memory_limit_bytes=memory_limit_bytes,
                safety_reserve_bytes=safety_reserve_bytes,
            ):
                yield None

    @contextmanager
    def _execution_scope_unleased(
        self,
        *,
        probe_result: RuntimeProbeResult,
        device_id: str = "",
        memory_limit_bytes: int | None = None,
        safety_reserve_bytes: int | None = None,
    ) -> Iterator[None]:
        """Own a CUDA device and private allocator for one execution segment."""

        self._lock.acquire()
        pool = None
        cupy = None
        device_index: int | None = None
        fft_cache_policy: _FFTPlanCachePolicy | None = None
        cleanup_cause: BaseException | None = None
        cleanup_failed = False
        owns_scope = False
        try:
            if self._scope_active:
                raise RuntimeError("Nested CuPy execution scopes are not supported.")
            result = self._probe_result or probe_result
            if self._closed:
                raise RuntimeError("This CuPy runtime instance has been closed.")
            if self._unhealthy_message:
                raise RuntimeError(self._unhealthy_message)
            if not result.available:
                raise RuntimeError(
                    f"CuPy runtime unavailable ({result.reason_code}): {result.message}"
                )
            selected = device_id.strip() or result.selected_device_id
            device_index = _device_index(selected, result)
            cupy = self._load_cupy()
            with cupy.cuda.Device(device_index):
                free_bytes, total_bytes = _memory_info(cupy)
                reserve = (
                    max(_DEFAULT_RESERVE_BYTES, total_bytes // 10)
                    if safety_reserve_bytes is None
                    else _nonnegative_bytes(
                        safety_reserve_bytes, "safety_reserve_bytes"
                    )
                )
                if free_bytes <= reserve:
                    raise CUDAAdmissionError(
                        "CUDA free memory does not exceed the configured safety "
                        f"reserve ({free_bytes} <= {reserve} bytes)."
                    )
                requested_limit = (
                    total_bytes * 80 // 100
                    if memory_limit_bytes is None
                    else _positive_bytes(memory_limit_bytes, "memory_limit_bytes")
                )
                limit = min(requested_limit, free_bytes - reserve)
                if limit < 1:
                    raise CUDAAdmissionError(
                        "No CUDA memory remains after applying the safety reserve."
                    )
                pool = cupy.cuda.MemoryPool()
                pool.set_limit(size=limit)
                pool_owner = _private_pool_owner(pool)
                try:
                    fft_cache_policy = _FFTPlanCachePolicy.disable(cupy)
                except CUDACleanupError as exc:
                    self._mark_unhealthy(
                        "CuPy FFT plan-cache setup could not restore external "
                        f"state: {_exception_summary(exc)}"
                    )
                    raise
                self._scope_active = True
                owns_scope = True
                self._scope_thread_id = threading.get_ident()
                self._active_device_id = selected
                self._active_pool = pool
                self._active_pool_owner = pool_owner
                self._baseline_device_used_bytes = total_bytes - free_bytes
                self._scope_limit_bytes = limit
                self._scope_reserve_bytes = reserve
                with cupy.cuda.using_allocator(pool.malloc):
                    yield None
                    # ``close`` may be called by the owning thread while it is
                    # inside the scope.  In that case close has already
                    # synchronized and finalized this pool.
                    if self._scope_active:
                        self.synchronize(device_id=selected)
        finally:
            body_error = sys.exc_info()[1]
            if (
                owns_scope
                and self._scope_active
                and self._scope_thread_id == threading.get_ident()
            ):
                try:
                    if cupy is None or pool is None:
                        raise RuntimeError(
                            "The active CUDA scope lost its private allocator."
                        )
                    cleanup_failed, cleanup_cause = self._finish_scope(
                        cupy,
                        pool,
                        device_index=_raw_device_index(self._active_device_id),
                    )
                except Exception as exc:  # cleanup must still reset ownership
                    cleanup_failed = True
                    cleanup_cause = exc
                    self._mark_unhealthy(
                        "CUDA cleanup failed before its terminal memory state "
                        f"could be verified: {_exception_summary(exc)}"
                    )
                finally:
                    self._clear_scope_state()
            if fft_cache_policy is not None:
                try:
                    if cupy is None or device_index is None:
                        raise RuntimeError(
                            "The CUDA scope lost the device needed to restore "
                            "its FFT plan cache."
                        )
                    with cupy.cuda.Device(device_index):
                        fft_cache_policy.restore()
                except BaseException as exc:
                    cleanup_failed = True
                    if cleanup_cause is None:
                        cleanup_cause = exc
                    self._mark_unhealthy(
                        "CuPy FFT plan-cache restoration failed: "
                        + _exception_summary(exc)
                    )
            self._lock.release()
            if cleanup_failed:
                error = CUDACleanupError(self._unhealthy_message)
                # Cleanup integrity has precedence over a retryable body
                # failure such as OOM.  Chaining retains the scientific
                # failure for diagnostics while classification remains a
                # non-retryable cleanup failure.
                cause = body_error or cleanup_cause
                if cause is not None:
                    raise error from cause
                raise error

    def is_device_value(self, value: object) -> bool:
        try:
            cupy = self._load_cupy()
        except Exception:
            return False
        ndarray_type = getattr(cupy, "ndarray", None)
        return isinstance(ndarray_type, type) and isinstance(value, ndarray_type)

    def allocation_identity(self, value: object) -> Hashable:
        """Return a scope-private allocation identity shared by array views."""

        self._require_active_scope("")
        if not self.is_device_value(value):
            raise TypeError("The value is not a CuPy array.")
        allocation = _array_allocation(value)
        nbytes = int(getattr(value, "nbytes", 0))
        active_index = _raw_device_index(self._active_device_id)
        value_index = _allocation_device_index(value, allocation)
        if value_index is not None and value_index != active_index:
            raise InvalidCUDADeviceError(
                f"The CuPy value belongs to cuda:{value_index}, not "
                f"{self._active_device_id}."
            )
        if allocation is None:
            if nbytes == 0:
                return (self.runtime_id, self._active_device_id, "empty", id(value))
            raise TypeError("The CuPy value does not expose a device allocation.")
        if nbytes == 0 and int(getattr(allocation, "size", 0)) == 0:
            return (self.runtime_id, self._active_device_id, "empty", id(value))
        owner = _allocation_pool_owner(allocation)
        if owner is None or owner is not self._active_pool_owner:
            raise TypeError(
                "The CuPy value was not allocated by this execution scope's "
                "private memory pool."
            )
        return (self.runtime_id, self._active_device_id, id(allocation))

    def to_device(self, value: object, *, device_id: str = "") -> object:
        self._require_active_scope(device_id)
        cupy = self._load_cupy()
        host = np.ascontiguousarray(np.asarray(value))
        with cupy.cuda.Device(_raw_device_index(self._active_device_id)):
            result = cupy.empty(host.shape, dtype=host.dtype)
            result.set(host)
            self.synchronize(device_id=self._active_device_id)
            self.allocation_identity(result)
        return result

    def to_host(self, value: object) -> object:
        self._require_active_scope("")
        if not self.is_device_value(value):
            raise TypeError("to_host requires an array owned by this CuPy runtime.")
        self.allocation_identity(value)
        cupy = self._load_cupy()
        contiguous = value
        temporary = None
        if not bool(getattr(getattr(value, "flags", None), "c_contiguous", False)):
            temporary = cupy.ascontiguousarray(value)
            contiguous = temporary
        host = np.empty(tuple(contiguous.shape), dtype=contiguous.dtype)
        contiguous.get(out=host, blocking=True)
        self.synchronize(device_id=self._active_device_id)
        if temporary is not None:
            self.release(temporary)
        return host

    def release(self, value: object) -> None:
        """Relinquish runtime ownership without invalidating Python aliases.

        CuPy returns pooled memory automatically when the last array/view which
        references an allocation dies.  Calling ``PooledMemory.free`` here
        would bypass that alias lifetime and can silently corrupt a live view.
        """

        self._require_active_scope("")
        self.allocation_identity(value)

    def synchronize(self, *, device_id: str = "") -> None:
        if device_id:
            index = _raw_device_index(device_id)
            cupy = self._load_cupy()
            with cupy.cuda.Device(index):
                cupy.cuda.get_current_stream().synchronize()
            return
        cupy = self._load_cupy()
        cupy.cuda.get_current_stream().synchronize()

    def memory_snapshot(self, *, device_id: str = "") -> RuntimeMemorySnapshot:
        requested = device_id.strip()
        with self._lock:
            selected_terminal = requested or self._terminal_device_id
            terminal = (
                None
                if self._scope_active
                else self._terminal_snapshots.get(selected_terminal)
            )
        if terminal is not None:
            with accelerator_lease(self.runtime_id, terminal.device_id):
                with self._lock:
                    current_terminal = self._terminal_snapshots.get(terminal.device_id)
                    if not self._scope_active and current_terminal is not None:
                        return self._refreshed_terminal_snapshot_unleased(
                            current_terminal
                        )

        result = self.probe()
        if not result.available:
            raise RuntimeError(
                f"CuPy runtime unavailable ({result.reason_code}): {result.message}"
            )
        with self._lock:
            selected = requested or self._active_device_id or result.selected_device_id
            _device_index(selected, result)
            if self._scope_active and selected != self._active_device_id:
                raise InvalidCUDADeviceError(
                    "Memory snapshots during execution must use the active device."
                )

        # Never retain the instance lock while waiting for process ownership.
        # Once ownership is held, re-read all mutable scope/terminal state.
        with accelerator_lease(self.runtime_id, selected):
            with self._lock:
                current_result = self._probe_result or result
                if self._closed:
                    raise RuntimeError("This CuPy runtime instance has been closed.")
                if self._unhealthy_message:
                    raise RuntimeError(self._unhealthy_message)
                if not current_result.available:
                    raise RuntimeError(
                        "CuPy runtime unavailable "
                        f"({current_result.reason_code}): {current_result.message}"
                    )
                selected = (
                    requested
                    or self._active_device_id
                    or current_result.selected_device_id
                )
                _device_index(selected, current_result)
                if self._scope_active and selected != self._active_device_id:
                    raise InvalidCUDADeviceError(
                        "Memory snapshots during execution must use the active device."
                    )
                if not self._scope_active:
                    terminal = self._terminal_snapshots.get(selected)
                    if terminal is not None:
                        return self._refreshed_terminal_snapshot_unleased(terminal)
                cupy = self._load_cupy()
                with cupy.cuda.Device(_raw_device_index(selected)):
                    free_bytes, total_bytes = _memory_info(cupy)
                    pool = self._active_pool if self._scope_active else None
                    live = int(pool.used_bytes()) if pool is not None else 0
                    reserved = int(pool.total_bytes()) if pool is not None else 0
                    device_used_delta = max(
                        0,
                        total_bytes - free_bytes - self._baseline_device_used_bytes,
                    )
                    out_of_pool = (
                        max(0, device_used_delta - reserved) if pool is not None else 0
                    )
                    return RuntimeMemorySnapshot(
                        runtime_id=self.runtime_id,
                        device_id=selected,
                        topology=MemoryTopology.DISCRETE,
                        device_total_bytes=total_bytes,
                        device_free_bytes=free_bytes,
                        runtime_live_bytes=live,
                        runtime_reserved_bytes=reserved,
                        out_of_pool_bytes=out_of_pool,
                    )

    def classify_exception(self, exc: BaseException) -> RuntimeExceptionInfo:
        """Map typed CuPy/Python failures without brittle message matching."""

        chain = tuple(_exception_chain(exc))
        exception_type = type(exc).__name__
        if any(isinstance(item, CUDACleanupError) for item in chain):
            return _exception_info(
                RuntimeExceptionKind.KERNEL_FAILURE,
                "cuda_cleanup_incomplete",
                exc,
                retryable=False,
            )
        if any(isinstance(item, InvalidCUDADeviceError) for item in chain):
            return _exception_info(
                RuntimeExceptionKind.INVALID_DEVICE,
                "invalid_device",
                exc,
                retryable=False,
            )
        if any(isinstance(item, CUDAAdmissionError) for item in chain):
            return _exception_info(
                RuntimeExceptionKind.OUT_OF_MEMORY,
                "insufficient_device_memory",
                exc,
                retryable=True,
            )
        if any(isinstance(item, ModuleNotFoundError) for item in chain):
            return _exception_info(
                RuntimeExceptionKind.RUNTIME_UNAVAILABLE,
                "dependency_missing",
                exc,
                retryable=False,
            )
        if any(isinstance(item, (FileNotFoundError, OSError)) for item in chain):
            return _exception_info(
                RuntimeExceptionKind.RUNTIME_UNAVAILABLE,
                "runtime_component_missing",
                exc,
                retryable=False,
            )

        cupy = self._cupy
        if cupy is not None:
            oom_type = _nested_type(cupy, "cuda", "memory", "OutOfMemoryError")
            if oom_type is not None and any(
                isinstance(item, oom_type) for item in chain
            ):
                return _exception_info(
                    RuntimeExceptionKind.OUT_OF_MEMORY,
                    "cuda_out_of_memory",
                    exc,
                    retryable=True,
                )
            runtime_error = _nested_type(cupy, "cuda", "runtime", "CUDARuntimeError")
            if runtime_error is not None and any(
                isinstance(item, runtime_error) for item in chain
            ):
                invalid_status = getattr(
                    cupy.cuda.runtime, "cudaErrorInvalidDevice", 10
                )
                if any(
                    getattr(item, "status", None) == invalid_status
                    for item in chain
                    if isinstance(item, runtime_error)
                ):
                    return _exception_info(
                        RuntimeExceptionKind.INVALID_DEVICE,
                        "invalid_device",
                        exc,
                        retryable=False,
                    )
                return _exception_info(
                    RuntimeExceptionKind.KERNEL_FAILURE,
                    "cuda_runtime_failure",
                    exc,
                    retryable=False,
                )
            compile_type = _nested_type(cupy, "cuda", "compiler", "CompileException")
            if compile_type is not None and any(
                isinstance(item, compile_type) for item in chain
            ):
                return _exception_info(
                    RuntimeExceptionKind.KERNEL_FAILURE,
                    "cuda_kernel_compile_failure",
                    exc,
                    retryable=False,
                )
        return RuntimeExceptionInfo(
            kind=RuntimeExceptionKind.UNKNOWN,
            reason_code="unknown_runtime_error",
            message=_exception_summary(exc),
            exception_type=exception_type,
            retryable=False,
            cleanup_required=True,
        )

    def close(self) -> None:
        """Release only resources owned by this instance; never global pools."""

        with self._lock:
            if self._closed:
                return
            cleanup_failed = False
            cleanup_cause: BaseException | None = None
            if self._scope_active:
                try:
                    if self._cupy is None or self._active_pool is None:
                        raise RuntimeError(
                            "The active CUDA scope lost its private allocator."
                        )
                    cleanup_failed, cleanup_cause = self._finish_scope(
                        self._cupy,
                        self._active_pool,
                        device_index=_raw_device_index(self._active_device_id),
                    )
                except Exception as exc:
                    cleanup_failed = True
                    cleanup_cause = exc
                    self._mark_unhealthy(
                        "CUDA cleanup failed while closing the runtime: "
                        + _exception_summary(exc)
                    )
                finally:
                    self._clear_scope_state()
            self._probe_result = None
            self._ndimage = None
            self._cupy = None
            # Once this instance drops CuPy it can no longer revalidate a
            # process-wide environment setting, so a later closed probe must
            # not claim the last observed effective cache policy.
            self._cupy_cache_metadata = ()
            self._closed = True
            if cleanup_failed:
                error = CUDACleanupError(self._unhealthy_message)
                if cleanup_cause is not None:
                    raise error from cleanup_cause
                raise error

    def _finish_scope(
        self,
        cupy: object,
        pool: object,
        *,
        device_index: int,
    ) -> tuple[bool, BaseException | None]:
        """Clean a private pool on its device and retain terminal diagnostics."""

        errors: list[BaseException] = []
        pre_live = 0
        pre_reserved = 0
        post_live: int | None = None
        post_reserved: int | None = None
        free_bytes: int | None = None
        total_bytes: int | None = None
        selected = self._active_device_id

        with cupy.cuda.Device(device_index):
            try:
                cupy.cuda.get_current_stream().synchronize()
            except Exception as exc:
                errors.append(exc)
            try:
                pre_live = max(0, int(pool.used_bytes()))
                pre_reserved = max(pre_live, int(pool.total_bytes()))
            except Exception as exc:
                errors.append(exc)
            try:
                pool.free_all_blocks()
            except Exception as exc:
                errors.append(exc)
            try:
                cupy.cuda.get_current_stream().synchronize()
            except Exception as exc:
                errors.append(exc)
            try:
                post_live = max(0, int(pool.used_bytes()))
                post_reserved = max(post_live, int(pool.total_bytes()))
            except Exception as exc:
                errors.append(exc)
            try:
                free_bytes, total_bytes = _memory_info(cupy)
            except Exception as exc:
                errors.append(exc)

        live = pre_live if post_live is None else post_live
        reserved = pre_reserved if post_reserved is None else post_reserved
        reserved = max(live, reserved)
        out_of_pool = 0
        if free_bytes is not None and total_bytes is not None:
            device_used_delta = max(
                0,
                total_bytes - free_bytes - self._baseline_device_used_bytes,
            )
            out_of_pool = max(0, device_used_delta - reserved)
        terminal = RuntimeMemorySnapshot(
            runtime_id=self.runtime_id,
            device_id=selected,
            topology=MemoryTopology.DISCRETE,
            device_total_bytes=total_bytes,
            device_free_bytes=free_bytes,
            runtime_live_bytes=live,
            runtime_reserved_bytes=reserved,
            out_of_pool_bytes=out_of_pool,
        )
        self._terminal_snapshots[selected] = terminal
        self._terminal_device_id = selected

        issues: list[str] = []
        if errors:
            issues.append("one or more CUDA cleanup operations failed")
        if live:
            issues.append(f"{live} bytes remain in a live private allocation")
        if reserved:
            issues.append(f"{reserved} private-pool bytes remain reserved")
        # ``memGetInfo`` is device-wide.  This delta can include legitimate
        # CUDA module/JIT/cuCIM caches and allocations by other WDDM clients,
        # none of which this runtime owns or may free.  Preserve it in the
        # terminal snapshot for diagnostics, but never poison the runtime from
        # this instantaneous observation alone.
        if issues:
            self._mark_unhealthy("; ".join(issues))
            return True, errors[0] if errors else None
        return False, None

    def _mark_unhealthy(self, detail: str) -> None:
        self._unhealthy_message = (
            "CUDA execution cleanup was incomplete: "
            + detail.rstrip(". ")
            + ". Create a new runtime before retrying."
        )
        self._probe_result = None

    def _refreshed_terminal_snapshot_unleased(
        self,
        terminal: RuntimeMemorySnapshot,
    ) -> RuntimeMemorySnapshot:
        """Refresh totals while the caller owns device then instance locks."""

        cupy = self._cupy
        if cupy is None:
            return terminal
        try:
            with cupy.cuda.Device(_raw_device_index(terminal.device_id)):
                free_bytes, total_bytes = _memory_info(cupy)
        except Exception:
            return terminal
        return RuntimeMemorySnapshot(
            runtime_id=terminal.runtime_id,
            device_id=terminal.device_id,
            topology=terminal.topology,
            device_total_bytes=total_bytes,
            device_free_bytes=free_bytes,
            runtime_live_bytes=terminal.runtime_live_bytes,
            runtime_reserved_bytes=terminal.runtime_reserved_bytes,
            out_of_pool_bytes=terminal.out_of_pool_bytes,
        )

    def _load_cupy(self) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError("This CuPy runtime instance has been closed.")
            if self._cupy is None:
                cupy = self._module_loader("cupy")
                self._cupy = cupy
            else:
                cupy = self._cupy
            self._refresh_cupy_cache_policy_unlocked(cupy)
            return self._cupy

    def _refresh_cupy_cache_policy_unlocked(self, cupy: object) -> None:
        """Reassert process policy and invalidate stale probe provenance."""

        metadata = _configure_windows_unicode_safe_cupy_cache(
            cupy,
            platform_name=self._platform_name,
            environment=os.environ,
            temp_paths=_effective_temp_paths(),
        )
        previous = dict(self._cupy_cache_metadata)
        current = dict(metadata)
        if (
            current.get("cupy_cache_reason") == "windows_non_ascii_runtime_path"
            and previous.get("cupy_cache_explicit_setting_overridden") == "true"
            and "cupy_cache_explicit_setting_overridden" not in current
        ):
            metadata = (
                *metadata,
                ("cupy_cache_explicit_setting_overridden", "true"),
            )
        if metadata != self._cupy_cache_metadata:
            self._cupy_cache_metadata = metadata
            self._probe_result = None

    def _load_ndimage(self) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError("This CuPy runtime instance has been closed.")
            if self._ndimage is None:
                self._ndimage = self._module_loader("cupyx.scipy.ndimage")
            return self._ndimage

    def _compatibility_failure(self) -> tuple[str, str] | None:
        if not (
            self._platform_name == "win32" or self._platform_name.startswith("linux")
        ):
            return (
                "platform_unsupported",
                "The CUDA runtime is supported only on native Windows and Linux.",
            )
        if (
            self._python_implementation != "CPython"
            or self._python_version != (3, 12)
            or self._pointer_bits != 64
        ):
            return (
                "python_unsupported",
                "The Phase-1 CUDA runtime requires 64-bit CPython 3.12.",
            )
        return None

    def _probe_device(self, cupy: object, index: int) -> RuntimeDevice:
        with cupy.cuda.Device(index) as device:
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            name = _property(properties, "name", f"CUDA device {index}")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            _free, total = _memory_info(cupy)
            capability = getattr(device, "compute_capability", "")
            if isinstance(capability, bytes):
                capability = capability.decode(errors="replace")
            capability = _display_compute_capability(capability)
            return RuntimeDevice(
                device_id=f"cuda:{index}",
                display_name=str(name),
                total_memory_bytes=total,
                metadata=(("compute_capability", str(capability)),),
            )

    def _exercise_runtime(
        self, cupy: object, ndimage: object, *, device_index: int
    ) -> None:
        with cupy.cuda.Device(device_index):
            pool = cupy.cuda.MemoryPool()
            fft_cache_policy = _FFTPlanCachePolicy.disable(cupy)
            try:
                with cupy.cuda.using_allocator(pool.malloc):
                    try:
                        _run_probe_operations(cupy, ndimage)
                    except BaseException as exc:
                        # A CuPy compiler/kernel exception can retain its input
                        # arrays through traceback frames.  Those probe-owned
                        # references must be released before judging the private
                        # pool; otherwise the cleanup error masks the real CUDA
                        # failure as a false leak.  Preserve the exception chain
                        # and type for classification, but detach its frames.
                        body_error = _detach_exception_tracebacks(exc)
                    else:
                        body_error = None
                    try:
                        cupy.cuda.get_current_stream().synchronize()
                        pool.free_all_blocks()
                        cupy.cuda.get_current_stream().synchronize()
                        live = int(pool.used_bytes())
                        reserved = int(pool.total_bytes())
                        if live or reserved:
                            raise RuntimeError(
                                "CuPy probe leaked its private memory pool "
                                f"(live={live}, reserved={reserved} bytes)."
                            )
                    except BaseException as cleanup_error:
                        if body_error is not None:
                            raise cleanup_error from body_error
                        raise
                    if body_error is not None:
                        raise body_error
            finally:
                fft_cache_policy.restore()

    def _remember_unavailable(
        self, reason_code: str, message: str, *, version: str = ""
    ) -> RuntimeProbeResult:
        fingerprint_payload: dict[str, object] = {
            "runtime_id": self.runtime_id,
            "platform": self._platform_name,
            "python": self._python_version,
            "reason": reason_code,
            "version": version,
        }
        cache_policy = _cupy_cache_fingerprint_metadata(self._cupy_cache_metadata)
        if cache_policy:
            fingerprint_payload["cupy_cache_policy"] = cache_policy
        self._probe_result = RuntimeProbeResult(
            runtime_id=self.runtime_id,
            available=False,
            version=version,
            reason_code=reason_code,
            message=message,
            environment_fingerprint=canonical_digest(fingerprint_payload),
            metadata=self._cupy_cache_metadata,
        )
        return self._probe_result

    def _require_active_scope(self, device_id: str) -> None:
        if not self._scope_active or self._scope_thread_id != threading.get_ident():
            raise RuntimeError("CuPy transfers require an active execution scope.")
        if device_id and device_id.strip() != self._active_device_id:
            raise InvalidCUDADeviceError(
                f"Execution scope owns {self._active_device_id}, not {device_id}."
            )

    def _clear_scope_state(self) -> None:
        self._scope_active = False
        self._scope_thread_id = None
        self._active_device_id = ""
        self._active_pool = None
        self._active_pool_owner = None
        self._baseline_device_used_bytes = 0
        self._scope_limit_bytes = 0
        self._scope_reserve_bytes = 0


def create_runtime() -> CuPyRuntime:
    """Create the built-in lazy CuPy runtime."""

    return CuPyRuntime()


def _run_probe_operations(cupy: object, ndimage: object) -> None:
    """Exercise the runtime in a frame that ends before pool verification."""

    source = cupy.arange(64, dtype=cupy.float32).reshape((8, 8))
    gaussian = ndimage.gaussian_filter(source, sigma=1.0)
    median = ndimage.median_filter(source, size=3)
    smoothed = ndimage.uniform_filter1d(
        source,
        size=3,
        axis=0,
        output=cupy.float64,
        mode="nearest",
    )
    frequency = cupy.fft.rfftn(source)
    restored = cupy.fft.irfftn(frequency, s=source.shape)
    cupy.cuda.get_current_stream().synchronize()
    # Keep every result alive through synchronization.  Returning then drops
    # all probe-owned arrays before the caller frees and verifies its pool.
    _ = gaussian, median, smoothed, restored


def _detach_exception_tracebacks(exc: BaseException) -> BaseException:
    """Release failed-probe frame locals while preserving exception identity."""

    for chained in _exception_chain(exc):
        traceback = chained.__traceback__
        if traceback is not None:
            traceback_module.clear_frames(traceback)
            chained.__traceback__ = None
    return exc


def _configure_windows_unicode_safe_cupy_cache(
    cupy: object,
    *,
    platform_name: str,
    environment: MutableMapping[str, str],
    temp_paths: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Atomically derive and enforce the process-wide CuPy cache policy."""

    with _CUPY_CACHE_POLICY_LOCK:
        return _configure_windows_unicode_safe_cupy_cache_unlocked(
            cupy,
            platform_name=platform_name,
            environment=environment,
            temp_paths=temp_paths,
        )


def _configure_windows_unicode_safe_cupy_cache_unlocked(
    cupy: object,
    *,
    platform_name: str,
    environment: MutableMapping[str, str],
    temp_paths: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Keep NVRTC temporary source filenames off non-ASCII Windows paths.

    CuPy 14.1.1's Windows NVRTC bridge can encode a Unicode source filename
    as mojibake before the compiler opens it.  In-memory caching prevents that
    temporary source-file round trip.  This is a process-wide safety setting:
    when required, an explicit ``CUPY_CACHE_IN_MEMORY=0`` is deliberately
    overridden rather than allowing a later, misleading kernel failure.

    In-memory caching cannot repair a non-ASCII compiler include path by
    itself.  Such a path still receives this setting, but the normal runtime
    probe remains fail-closed if NVRTC cannot open an installed header.
    """

    previous = environment.get("CUPY_CACHE_IN_MEMORY")
    if not platform_name.startswith("win32"):
        if not _cupy_cache_in_memory_enabled(previous):
            return ()
        return (
            ("cupy_cache_in_memory", "1"),
            ("cupy_cache_reason", "process_in_memory_setting"),
        )
    candidates: list[tuple[str, object]] = [
        *(("temp", path) for path in temp_paths),
        ("cupy_module", getattr(cupy, "__file__", "")),
    ]
    # CuPy's internal include tree is below its package root, covered by
    # ``__file__`` / ``__path__``.  ``sys.prefix``, the user's home directory,
    # and ``CUPY_CACHE_DIR`` are intentionally not compiler-path candidates:
    # they need not be on an NVRTC path, and CuPy accesses its cubin cache with
    # Unicode-safe Python file I/O.
    module_paths = getattr(cupy, "__path__", ())
    try:
        candidates.extend(("cupy_module", path) for path in module_paths)
    except TypeError:
        candidates.append(("cupy_module", module_paths))
    affected = tuple(
        dict.fromkeys(
            label for label, path in candidates if _path_contains_non_ascii(path)
        )
    )
    if not affected:
        if not _cupy_cache_in_memory_enabled(previous):
            return ()
        return (
            ("cupy_cache_in_memory", "1"),
            ("cupy_cache_reason", "process_in_memory_setting"),
        )
    environment["CUPY_CACHE_IN_MEMORY"] = "1"
    metadata = [
        ("cupy_cache_in_memory", "1"),
        ("cupy_cache_reason", "windows_non_ascii_runtime_path"),
        ("cupy_cache_non_ascii_path_kinds", ",".join(affected)),
    ]
    if previous is not None and not _cupy_cache_in_memory_enabled(previous):
        metadata.append(("cupy_cache_explicit_setting_overridden", "true"))
    return tuple(metadata)


def _cupy_cache_fingerprint_metadata(
    metadata: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    """Return only effective cache state which can change runtime behavior."""

    values = dict(metadata)
    return {key: values[key] for key in _CUPY_CACHE_FINGERPRINT_KEYS if key in values}


def _cupy_cache_in_memory_enabled(value: str | None) -> bool:
    """Match CuPy 14's integer-valued cache environment parsing."""

    if value is None or not value:
        return False
    try:
        return int(value) == 1
    except ValueError:
        return False


def _effective_temp_paths() -> tuple[str, ...]:
    """Return the source-file root Python tempfile will actually select.

    CuPy's NVRTC bridge uses ``TemporaryDirectory()`` without an explicit
    directory.  ``tempfile.gettempdir()`` is therefore the effective path;
    other TEMP/TMP variables may be present but inactive and must not force a
    process-wide in-memory cache policy.
    """

    return (tempfile.gettempdir(),)


def _path_contains_non_ascii(path: object) -> bool:
    try:
        return not os.fspath(path).isascii()
    except TypeError:
        return False


def _fft_plan_cache(cupy: object) -> object:
    """Return CuPy's cache for the selected thread and CUDA device."""

    fft = getattr(cupy, "fft", None)
    config = getattr(fft, "config", None)
    getter = getattr(config, "get_plan_cache", None)
    if not callable(getter):
        raise RuntimeError("CuPy does not expose its FFT plan-cache API.")
    cache = getter()
    required = (
        "get_size",
        "get_memsize",
        "get_curr_size",
        "set_size",
        "set_memsize",
    )
    missing = [name for name in required if not callable(getattr(cache, name, None))]
    if missing:
        raise RuntimeError(
            "CuPy's FFT plan cache is missing required operations: "
            + ", ".join(missing)
            + "."
        )
    return cache


def _cache_integer(cache: object, method_name: str) -> int:
    method = getattr(cache, method_name, None)
    if not callable(method):
        raise RuntimeError(f"CuPy's FFT plan cache does not expose {method_name}().")
    value = method()
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"CuPy's FFT plan cache returned an invalid {method_name}() value."
        )
    minimum = -1 if method_name == "get_memsize" else 0
    if value < minimum:
        raise RuntimeError(
            f"CuPy's FFT plan cache returned an invalid {method_name}() value."
        )
    return value


def _fft_plan_cache_entries(cache: object) -> tuple[tuple[object, object], ...]:
    """Snapshot plan identities in the cache's MRU-to-LRU iteration order."""

    try:
        raw_entries = tuple(cache)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CuPy's FFT plan cache is not inspectable.") from exc
    entries: list[tuple[object, object]] = []
    keys: set[object] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, tuple) or len(raw_entry) != 2:
            raise RuntimeError("CuPy's FFT plan cache returned an unrecognized entry.")
        key, node = raw_entry
        try:
            if key in keys:
                raise RuntimeError("CuPy's FFT plan cache returned a duplicate key.")
            keys.add(key)
        except TypeError as exc:
            raise RuntimeError(
                "CuPy's FFT plan cache returned an unhashable key."
            ) from exc
        missing = object()
        plan = getattr(node, "plan", missing)
        if plan is missing:
            raise RuntimeError("CuPy's FFT plan cache entry does not expose its plan.")
        entries.append((key, plan))
    return tuple(entries)


def _raw_device_index(device_id: str) -> int:
    prefix, separator, raw_index = str(device_id).partition(":")
    if prefix != "cuda" or not separator:
        raise InvalidCUDADeviceError(f"Invalid CUDA device ID: {device_id!r}.")
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise InvalidCUDADeviceError(f"Invalid CUDA device ID: {device_id!r}.") from exc
    if index < 0:
        raise InvalidCUDADeviceError(f"Invalid CUDA device ID: {device_id!r}.")
    return index


def _device_index(device_id: str, result: RuntimeProbeResult) -> int:
    index = _raw_device_index(device_id)
    if device_id not in {device.device_id for device in result.devices}:
        raise InvalidCUDADeviceError(
            f"CUDA device {device_id!r} was not reported by the runtime probe."
        )
    return index


def _positive_bytes(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer number of bytes.")
    return value


def _nonnegative_bytes(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer number of bytes.")
    return value


def _private_pool_owner(pool: object) -> object:
    """Discover the internal pool token without retaining an allocation."""

    pointer = None
    allocation = None
    try:
        pointer = pool.malloc(1)
        allocation = getattr(pointer, "mem", pointer)
        owner = _allocation_pool_owner(allocation)
        if owner is None:
            raise RuntimeError(
                "CuPy's private memory pool did not expose allocation ownership."
            )
        return owner
    finally:
        # Dropping MemoryPointer/PooledMemory references lets CuPy's normal
        # alias-aware lifetime return the sentinel block to this pool.
        allocation = None
        pointer = None
        pool.free_all_blocks()


def _array_allocation(value: object) -> object | None:
    return getattr(getattr(value, "data", None), "mem", None)


def _allocation_pool_owner(allocation: object) -> object | None:
    pool_reference = getattr(allocation, "pool", None)
    if callable(pool_reference):
        try:
            return pool_reference()
        except TypeError:
            return None
    return pool_reference


def _allocation_device_index(
    value: object,
    allocation: object | None,
) -> int | None:
    candidates = (
        getattr(allocation, "device_id", None),
        getattr(getattr(value, "device", None), "id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
    return None


def _memory_info(cupy: object) -> tuple[int, int]:
    free, total = cupy.cuda.runtime.memGetInfo()
    return int(free), int(total)


def _property(properties: object, name: str, default: object) -> object:
    if not hasattr(properties, "get"):
        return default
    value = properties.get(name)
    return properties.get(name.encode(), default) if value is None else value


def _optional_runtime_version(runtime: object, name: str) -> str:
    function = getattr(runtime, name, None)
    if not callable(function):
        return "unknown"
    try:
        return str(int(function()))
    except Exception:
        return "unknown"


def _display_compute_capability(value: object) -> str:
    rendered = str(value).strip()
    if rendered.isdigit() and len(rendered) >= 2:
        return f"{rendered[:-1]}.{rendered[-1]}"
    return rendered


def _nested_type(root: object, *names: str) -> type[BaseException] | None:
    value = root
    for name in names:
        value = getattr(value, name, None)
        if value is None:
            return None
    return (
        value if isinstance(value, type) and issubclass(value, BaseException) else None
    )


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _exception_info(
    kind: RuntimeExceptionKind,
    reason_code: str,
    exc: BaseException,
    *,
    retryable: bool,
) -> RuntimeExceptionInfo:
    return RuntimeExceptionInfo(
        kind=kind,
        reason_code=reason_code,
        message=_exception_summary(exc),
        exception_type=type(exc).__name__,
        retryable=retryable,
        cleanup_required=True,
    )


__all__ = [
    "CUDAAdmissionError",
    "CUDACleanupError",
    "CuPyRuntime",
    "InvalidCUDADeviceError",
    "create_runtime",
]
