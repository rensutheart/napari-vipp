"""Bounded background counts for actual immutable filter input/output pairs."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from napari_vipp.core.object_filter_counts import (
    ObjectFilterCounts,
    object_filter_counts,
)
from napari_vipp.core.progress import OperationCancelled


@dataclass(frozen=True, eq=False)
class ObjectFilterDiagnosticKey:
    """One borrowed scientific pair; equality is explicitly identity-based."""

    input_data: object
    output_data: object
    spatial_ndim: int
    connectivity: str
    spatial_axes: tuple[int, ...] | None = None

    @property
    def cache_key(self) -> tuple:
        return (
            id(self.input_data), id(self.output_data),
            self.spatial_ndim, self.connectivity, self.spatial_axes,
        )

    def matches(
        self, input_data, output_data, spatial_ndim, connectivity="Face connected",
        *, spatial_axes=None,
    ) -> bool:
        return (
            self.input_data is input_data
            and self.output_data is output_data
            and self.spatial_ndim == spatial_ndim
            and self.connectivity == connectivity
            and self.spatial_axes == spatial_axes
        )


@dataclass(frozen=True)
class _CountResult:
    generation: int
    key: ObjectFilterDiagnosticKey
    counts: ObjectFilterCounts | None = None
    error: str = ""
    cancelled: bool = False


@dataclass(frozen=True)
class _CacheEntry:
    input_identity: weakref.ReferenceType
    output_identity: weakref.ReferenceType
    counts: ObjectFilterCounts | None
    error: str


class _CountSignals(QObject):
    finished = Signal(object)


class _CountWorker(QRunnable):
    def __init__(self, key: ObjectFilterDiagnosticKey, generation: int) -> None:
        super().__init__()
        self.key = key
        self.generation = generation
        self.cancel_event = threading.Event()
        self.signals = _CountSignals()

    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                raise OperationCancelled("Object counts cancelled.")
            input_data, output_data = self.key.input_data, self.key.output_data
            if self.key.spatial_axes is not None:
                if len(self.key.spatial_axes) != self.key.spatial_ndim:
                    raise ValueError("Spatial axes must match the processing rank.")
                destination = tuple(range(-self.key.spatial_ndim, 0))
                input_data = np.moveaxis(
                    input_data, self.key.spatial_axes, destination
                )
                output_data = np.moveaxis(
                    output_data, self.key.spatial_axes, destination
                )
            counts = object_filter_counts(
                input_data,
                output_data,
                spatial_ndim=self.key.spatial_ndim,
                connectivity=self.key.connectivity,
                cancel_callback=self.cancel_event.is_set,
            )
            if self.cancel_event.is_set():
                raise OperationCancelled("Object counts cancelled.")
            result = _CountResult(self.generation, self.key, counts=counts)
        except OperationCancelled:
            result = _CountResult(self.generation, self.key, cancelled=True)
        except Exception as exc:
            result = _CountResult(
                self.generation, self.key, error=f"{type(exc).__name__}: {exc}"
            )
        try:
            self.signals.finished.emit(result)
        except RuntimeError:
            # The owning Qt object may already be gone during application exit.
            pass


class ObjectFilterDiagnostics(QObject):
    """Count off the GUI thread, retaining at most one active and one next pair.

    Scientific arrays must remain stable while borrowed. The widget separately
    verifies that both payloads and node state are still current before showing
    ``ready`` results. Cache entries retain only weak identities and four counts;
    these diagnostics never modify pipeline outputs or scientific cache keys.
    Call ``close()`` before owning-widget teardown; no destroyed-signal Python
    callback accesses the child thread pool during native destruction.
    """

    ready = Signal(object)

    def __init__(self, parent=None, *, max_entries: int = 32) -> None:
        super().__init__(parent)
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Object diagnostic cache size must be positive.")
        self._max_entries = max_entries
        self._cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self._active: _CountWorker | None = None
        self._pending: ObjectFilterDiagnosticKey | None = None
        self._generation = 0
        self._closed = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def _entry(self, key: ObjectFilterDiagnosticKey) -> _CacheEntry | None:
        entry = self._cache.get(key.cache_key)
        if entry is not None:
            if (
                entry.input_identity() is key.input_data
                and entry.output_identity() is key.output_data
            ):
                self._cache.move_to_end(key.cache_key)
                return entry
            del self._cache[key.cache_key]
        return None

    def cached(
        self, input_data, output_data, spatial_ndim, connectivity="Face connected",
        *, spatial_axes=None,
    ) -> ObjectFilterCounts | None:
        entry = self._entry(ObjectFilterDiagnosticKey(
            input_data, output_data, spatial_ndim, connectivity,
            None if spatial_axes is None else tuple(spatial_axes),
        ))
        return None if entry is None else entry.counts

    def error(
        self, input_data, output_data, spatial_ndim, connectivity="Face connected",
        *, spatial_axes=None,
    ) -> str:
        entry = self._entry(ObjectFilterDiagnosticKey(
            input_data, output_data, spatial_ndim, connectivity,
            None if spatial_axes is None else tuple(spatial_axes),
        ))
        return "" if entry is None else entry.error

    def request(
        self, input_data, output_data, spatial_ndim, connectivity="Face connected",
        *, spatial_axes=None,
    ) -> None:
        if self._closed:
            return
        # Validate the weak identity mechanism without copying any image pixels.
        weakref.ref(input_data)
        weakref.ref(output_data)
        key = ObjectFilterDiagnosticKey(
            input_data, output_data, spatial_ndim, connectivity,
            None if spatial_axes is None else tuple(spatial_axes),
        )
        if self._entry(key) is not None:
            return
        if self._active is not None:
            if (
                not self._active.cancel_event.is_set()
                and self._active.key.matches(
                    input_data, output_data, spatial_ndim, connectivity,
                    spatial_axes=key.spatial_axes,
                )
            ):
                return
            self._active.cancel_event.set()
            # New selection/edits replace, rather than append to, pending work.
            self._pending = key
            return
        self._start(key)

    def _start(self, key: ObjectFilterDiagnosticKey) -> None:
        self._prune_cache()
        worker = _CountWorker(key, self._generation)
        worker.signals.finished.connect(self._finished)
        self._active = worker
        self._pool.start(worker)

    @Slot(object)
    def _finished(self, result: _CountResult) -> None:
        active = self._active
        if active is None or active.key is not result.key:
            return
        self._active = None
        if (
            not self._closed
            and result.generation == self._generation
            and not result.cancelled
            and not active.cancel_event.is_set()
        ):
            self._cache[result.key.cache_key] = _CacheEntry(
                weakref.ref(result.key.input_data),
                weakref.ref(result.key.output_data),
                result.counts,
                result.error,
            )
            self._prune_cache()
            self.ready.emit(result.key)
        pending, self._pending = self._pending, None
        if not self._closed and pending is not None and self._active is None:
            self._start(pending)

    def _prune_cache(self) -> None:
        for key, entry in tuple(self._cache.items()):
            if entry.input_identity() is None or entry.output_identity() is None:
                del self._cache[key]
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Cancel work and reject even completions already queued to the GUI."""
        self._generation += 1
        if self._active is not None:
            self._active.cancel_event.set()
        self._pending = None
        self._cache.clear()

    def close(self) -> None:
        self._closed = True
        self.clear()

    def wait(self, timeout_ms: int = 1000) -> bool:
        """Bounded joining for shutdown/tests, never a normal UI refresh."""
        return self._pool.waitForDone(max(0, int(timeout_ms)))


__all__ = ["ObjectFilterDiagnosticKey", "ObjectFilterDiagnostics"]
