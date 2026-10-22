"""Optional, import-only warming of required reader libraries after UI startup.

This module itself uses only the standard library. It never opens a source,
constructs a reader, requests napari plugin discovery, or initializes Qt or an
accelerator.
Normal reader imports remain authoritative, including after a warm-up failure.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

READER_WARMUP_MODULES = ("ome_types", "ome_zarr.reader", "ome_zarr.writer")
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReaderWarmupResult:
    module: str
    seconds: float
    error: str = ""


class ReaderLibraryWarmup:
    """Import one small, explicit module list serially on a dedicated thread.

    Python's normal import locks share completed modules with file operations;
    the UI never joins this worker or waits for a readiness flag. A non-daemon
    thread lets the current native import finish safely on interpreter exit.
    Cancellation skips the remaining imports; an import in progress cannot be
    interrupted safely. Results and logs are diagnostics, not reader capability
    or scientific qualification evidence.
    """

    def __init__(
        self,
        *,
        modules: tuple[str, ...] = READER_WARMUP_MODULES,
        loader: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self._modules = modules
        self._loader = loader
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._results: list[ReaderWarmupResult] = []

    @property
    def results(self) -> tuple[ReaderWarmupResult, ...]:
        with self._lock:
            return tuple(self._results)

    def pending(self, module: str) -> bool:
        """Whether a scheduled import could still hold its normal module lock."""
        with self._lock:
            return (
                module in self._modules
                and not self._cancelled.is_set()
                and not self._finished.is_set()
                and not any(result.module == module for result in self._results)
            )

    def start(self) -> bool:
        """Start at most once; return immediately without importing a library."""
        with self._lock:
            if self._thread is not None or self._cancelled.is_set():
                return False
            thread = threading.Thread(
                target=self._run, name="VIPP reader warm-up", daemon=False
            )
            self._thread = thread
            thread.start()
        return True

    def cancel(self) -> None:
        """Skip imports that have not started, without waiting for the worker."""
        self._cancelled.set()

    def wait(self, timeout: float) -> bool:
        """Diagnostic/test wait only; never called by the interface."""
        return self._finished.wait(timeout)

    def _run(self) -> None:
        try:
            for name in self._modules:
                if self._cancelled.is_set():
                    break
                started = time.perf_counter()
                error = ""
                try:
                    self._loader(name)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    _LOGGER.warning("Reader warm-up failed for %s: %s", name, error)
                result = ReaderWarmupResult(name, time.perf_counter() - started, error)
                with self._lock:
                    self._results.append(result)
                _LOGGER.info("Reader warm-up: %s in %.3f s", name, result.seconds)
        finally:
            self._finished.set()
