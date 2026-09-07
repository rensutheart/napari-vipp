"""Identity-cached, presentation-only measurements of immutable mesh inputs."""

from __future__ import annotations

import sys
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from napari_vipp.core.meshes import MeshData, measure_mesh_geometry
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import TableData


@dataclass(frozen=True)
class _MeasurementResult:
    generation: int
    mesh: MeshData
    table: TableData | None = None
    error: str = ""
    nbytes: int = 0
    cancelled: bool = False


@dataclass(frozen=True)
class _CacheEntry:
    identity: weakref.ReferenceType[MeshData]
    table: TableData | None
    error: str
    nbytes: int


class _MeasurementSignals(QObject):
    finished = Signal(object)


def _table_retained_bytes(table: TableData, progress: ProgressContext) -> int:
    """Conservatively count retained Python table values, off the GUI thread."""
    total = sys.getsizeof(table) + sys.getsizeof(table.rows)
    for value in (
        table.name,
        table.table_kind,
        table.source_name,
        table.columns,
        table.column_units,
    ):
        total += sys.getsizeof(value)
    total += sum(sys.getsizeof(column) for column in table.columns)
    for unit in table.column_units:
        total += sys.getsizeof(unit) + sum(sys.getsizeof(value) for value in unit)
    for index, row in enumerate(table.rows):
        if index % 256 == 0:
            progress.check_cancelled()
        # Counting shared scalar/string references repeatedly is conservative.
        total += sys.getsizeof(row) + sum(sys.getsizeof(value) for value in row)
    return total


class _MeasurementWorker(QRunnable):
    def __init__(self, mesh: MeshData, generation: int, max_bytes: int) -> None:
        super().__init__()
        self.mesh = mesh
        self.generation = generation
        self.max_bytes = max_bytes
        self.cancel_event = threading.Event()
        self.signals = _MeasurementSignals()

    def run(self) -> None:
        progress = ProgressContext(cancelled=self.cancel_event.is_set)
        try:
            progress.check_cancelled()
            # This is the same measurement contract used by mesh-object filters.
            table = measure_mesh_geometry(
                self.mesh, include_convex_hull_metrics=False, progress=progress
            )
            progress.check_cancelled()
            nbytes = _table_retained_bytes(table, progress)
            progress.check_cancelled()
            if nbytes > self.max_bytes:
                result = _MeasurementResult(
                    self.generation,
                    self.mesh,
                    error="Mesh histogram measurements exceed the diagnostic cache "
                    "memory limit; scientific mesh filtering is still available.",
                )
            else:
                result = _MeasurementResult(
                    self.generation, self.mesh, table=table, nbytes=nbytes
                )
        except OperationCancelled:
            result = _MeasurementResult(self.generation, self.mesh, cancelled=True)
        except Exception as exc:
            result = _MeasurementResult(
                self.generation, self.mesh, error=f"{type(exc).__name__}: {exc}"
            )
        try:
            self.signals.finished.emit(result)
        except RuntimeError:
            # Qt may already have destroyed signal objects during application exit.
            pass


class MeshMeasurementDiagnostics(QObject):
    """Measure each resident immutable mesh once for any input metric plot.

    Call from the owning GUI thread. ``ready`` emits the actual mesh object on
    success or failure; consumers must check that it is still their input.
    Tables retain every measurement row, including unavailable/non-finite
    values and units. Property, bounds, and logarithmic display choices are
    deliberately absent from the cache key and from scientific pipeline state.
    The owning widget must call ``close()`` before QObject teardown to cancel
    active work; no Python callbacks run from Qt's ``destroyed`` signal.
    """

    ready = Signal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        max_entries: int = 4,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        super().__init__(parent)
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("Mesh diagnostic cache limits must be positive.")
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._cache: OrderedDict[int, _CacheEntry] = OrderedDict()
        self._pending: dict[int, _MeasurementWorker] = {}
        self._generation = 0
        self._closed = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        # Explicit close() owns cancellation. Running Python destroyed-signal
        # callbacks while Qt recursively tears down pools/runnables is unsafe
        # on Windows/PyQt6, even if the callback only clears Python references.
        # Qt's child-pool destructor joins remaining work without such a hook.

    def _entry(self, mesh: MeshData) -> _CacheEntry | None:
        entry = self._cache.get(id(mesh))
        if entry is not None and entry.identity() is mesh:
            self._cache.move_to_end(id(mesh))
            return entry
        if entry is not None:
            del self._cache[id(mesh)]
        return None

    def cached(self, mesh: MeshData) -> TableData | None:
        """Return the shared immutable diagnostic table, if already measured."""
        entry = self._entry(mesh)
        return None if entry is None else entry.table

    def error(self, mesh: MeshData) -> str:
        """Return a cached measurement error, or an empty string."""
        entry = self._entry(mesh)
        return "" if entry is None else entry.error

    def request(self, mesh: MeshData) -> None:
        """Schedule one measurement without blocking or restarting pending work."""
        if self._closed:
            return
        if not isinstance(mesh, MeshData):
            raise TypeError("Mesh diagnostics require immutable MeshData.")
        if self._entry(mesh) is not None or id(mesh) in self._pending:
            return
        self._prune_cache()
        worker = _MeasurementWorker(mesh, self._generation, self._max_bytes)
        worker.signals.finished.connect(self._finished)
        self._pending[id(mesh)] = worker
        self._pool.start(worker)

    @Slot(object)
    def _finished(self, result: _MeasurementResult) -> None:
        if self._closed or result.generation != self._generation:
            return
        pending = self._pending.pop(id(result.mesh), None)
        if pending is None or pending.mesh is not result.mesh or result.cancelled:
            return
        self._cache[id(result.mesh)] = _CacheEntry(
            weakref.ref(result.mesh), result.table, result.error, result.nbytes
        )
        self._cache.move_to_end(id(result.mesh))
        self._prune_cache()
        self.ready.emit(result.mesh)

    def _prune_cache(self) -> None:
        for key, entry in tuple(self._cache.items()):
            if entry.identity() is None:
                del self._cache[key]
        retained_bytes = sum(entry.nbytes for entry in self._cache.values())
        while self._cache and (
            len(self._cache) > self._max_entries or retained_bytes > self._max_bytes
        ):
            _key, entry = self._cache.popitem(last=False)
            retained_bytes -= entry.nbytes

    def clear(self) -> None:
        """Cancel pending requests and invalidate even already-queued results."""
        self._generation += 1
        for worker in tuple(self._pending.values()):
            worker.cancel_event.set()
        self._pending.clear()
        self._pool.clear()
        self._cache.clear()

    def close(self) -> None:
        """Permanently stop requests; running work cancels cooperatively."""
        self._closed = True
        self.clear()

    def wait(self, timeout_ms: int = 1000) -> bool:
        """Join workers for bounded shutdown/testing, not ordinary UI refresh."""
        return self._pool.waitForDone(max(0, int(timeout_ms)))


__all__ = ["MeshMeasurementDiagnostics"]
