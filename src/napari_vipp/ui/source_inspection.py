"""Qt worker adapter for verified, metadata-only source inspection."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.io.errors import (
    ImageSourceError,
    ImageSourceErrorCode,
    as_image_source_error,
)
from napari_vipp.core.metadata import AxisDeclaration
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import LocalSourceIdentity
from napari_vipp.core.source_inspection import (
    ResolvedSourceInspection,
    SourceInspectionPhase,
    SourceInspectionProgress,
    SourceInspectionProgressUnit,
    inspect_local_source_item,
)
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.source_resolution import SourceItemResolutionError


class StaleSourceInspectionGeneration(OperationCancelled):
    """Raised when newer UI state supersedes a metadata inspection."""


@dataclass(frozen=True, slots=True)
class SourceInspectionWorkerSpec:
    """One generation-qualified metadata-only source inspection request."""

    generation: int
    path: str
    node_id: str = ""
    series_index_hint: int = 0
    item_key: str = ""
    expected_identity: LocalSourceIdentity | None = None
    expected_source_item: SourceItem | None = None
    axis_declaration: AxisDeclaration | None = None

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("A source inspection generation must be non-negative.")
        path = str(self.path).strip()
        if not path:
            raise ValueError("A source inspection requires a local source path.")
        if isinstance(self.series_index_hint, bool) or int(
            self.series_index_hint
        ) < 0:
            raise ValueError(
                "A source inspection series index hint must be non-negative."
            )
        if self.expected_identity is not None and not isinstance(
            self.expected_identity,
            LocalSourceIdentity,
        ):
            raise TypeError(
                "expected_identity must be a LocalSourceIdentity or None."
            )
        if self.expected_source_item is not None and not isinstance(
            self.expected_source_item,
            SourceItem,
        ):
            raise TypeError("expected_source_item must be a SourceItem or None.")
        declaration = AxisDeclaration.from_value(self.axis_declaration)
        item_key = str(self.item_key).strip()
        if self.expected_source_item is not None:
            expected_key = self.expected_source_item.selector.key
            if item_key and item_key != expected_key:
                raise ValueError(
                    "The inspection item key must match the saved SourceItem key."
                )
            item_key = expected_key
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "node_id", str(self.node_id))
        object.__setattr__(self, "series_index_hint", int(self.series_index_hint))
        object.__setattr__(self, "item_key", item_key)
        object.__setattr__(self, "axis_declaration", declaration)


@dataclass(frozen=True, slots=True)
class SourceInspectionWorkerProgress:
    """A generation-qualified metadata-inspection progress publication."""

    generation: int
    path: str
    node_id: str
    phase: SourceInspectionPhase
    current: int
    total: int
    unit: SourceInspectionProgressUnit
    message: str = ""

    @property
    def determinate(self) -> bool:
        return (
            self.total > 0
            and self.unit is not SourceInspectionProgressUnit.INDETERMINATE
        )


@dataclass(frozen=True, slots=True)
class SourceInspectionWorkerResult:
    """The sole terminal result from one metadata-inspection worker."""

    generation: int
    path: str
    node_id: str = ""
    resolved: ResolvedSourceInspection | None = None
    source_error: ImageSourceError | None = None
    cancelled: bool = False
    stale: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("A source inspection generation must be non-negative.")
        if (self.resolved is None) == (self.source_error is None):
            raise ValueError(
                "A source inspection result must contain exactly one of resolved "
                "or source_error."
            )
        if self.resolved is not None and (self.cancelled or self.stale):
            raise ValueError(
                "A successful source inspection cannot be cancelled or stale."
            )
        if self.stale and not self.cancelled:
            raise ValueError("A stale source inspection must also be cancelled.")
        object.__setattr__(self, "generation", int(self.generation))

    @property
    def error(self) -> str:
        return "" if self.source_error is None else self.source_error.display_text

    @property
    def succeeded(self) -> bool:
        return self.resolved is not None


class SourceInspectionWorkerSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class SourceInspectionWorker(QRunnable):
    """Resolve a source header and SourceItem away from Qt's GUI thread."""

    def __init__(
        self,
        spec: SourceInspectionWorkerSpec,
        *,
        current_generation: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(spec, SourceInspectionWorkerSpec):
            raise TypeError("spec must be a SourceInspectionWorkerSpec instance.")
        if current_generation is not None and not callable(current_generation):
            raise TypeError("current_generation must be callable or None.")
        self.spec = spec
        self.signals = SourceInspectionWorkerSignals()
        self._current_generation = current_generation
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._started = False
        self._terminal = False

    def cancel(self) -> None:
        """Request cancellation at the next inspection checkpoint."""

        with self._state_lock:
            if not self._terminal:
                self._cancel_event.set()

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        with self._state_lock:
            if self._started or self._terminal:
                return
            self._started = True

        spec = self.spec
        try:
            self._check_active()
            resolved = inspect_local_source_item(
                spec.path,
                spec.series_index_hint,
                item_key=(spec.item_key or None),
                expected_identity=spec.expected_identity,
                expected_source_item=spec.expected_source_item,
                axis_declaration=spec.axis_declaration,
                cancel_callback=self._cancel_checkpoint,
                progress_callback=self._emit_progress,
            )
            self._check_active()
        except Exception as exc:
            stale = isinstance(exc, StaleSourceInspectionGeneration)
            source_error = _inspection_source_error(exc, spec)
            self._emit_finished(
                SourceInspectionWorkerResult(
                    generation=spec.generation,
                    path=spec.path,
                    node_id=spec.node_id,
                    source_error=source_error,
                    cancelled=(
                        stale
                        or source_error.code is ImageSourceErrorCode.CANCELLED
                    ),
                    stale=stale,
                )
            )
            return

        self._emit_finished(
            SourceInspectionWorkerResult(
                generation=spec.generation,
                path=spec.path,
                node_id=spec.node_id,
                resolved=resolved,
            )
        )

    def _cancel_checkpoint(self) -> bool:
        self._check_active()
        return False

    def _check_active(self) -> None:
        if self.cancellation_requested:
            raise OperationCancelled(
                f"Source inspection generation {self.spec.generation} was "
                "cancelled."
            )
        if self._current_generation is not None:
            current = int(self._current_generation())
            if current != self.spec.generation:
                raise StaleSourceInspectionGeneration(
                    f"Source inspection generation {self.spec.generation} was "
                    f"superseded by generation {current}."
                )

    def _emit_progress(self, progress: SourceInspectionProgress) -> None:
        self._check_active()
        self.signals.progress.emit(
            SourceInspectionWorkerProgress(
                generation=self.spec.generation,
                path=self.spec.path,
                node_id=self.spec.node_id,
                phase=progress.phase,
                current=progress.current,
                total=progress.total,
                unit=progress.unit,
                message=progress.message,
            )
        )
        self._check_active()

    def _emit_finished(self, result: SourceInspectionWorkerResult) -> None:
        with self._state_lock:
            if self._terminal:
                return
            self._terminal = True
        self.signals.finished.emit(result)


def _inspection_source_error(
    error: Exception,
    spec: SourceInspectionWorkerSpec,
) -> ImageSourceError:
    if isinstance(error, SourceItemResolutionError):
        return ImageSourceError(
            ImageSourceErrorCode.CONTRACT_MISMATCH,
            str(error),
            stage="metadata-inspection",
            path=spec.path,
            item=spec.item_key or spec.series_index_hint,
            remediation="Refresh the source and explicitly review its item binding.",
        )
    return as_image_source_error(
        error,
        stage="metadata-inspection",
        path=spec.path,
        item=spec.item_key or spec.series_index_hint,
    )


__all__ = [
    "SourceInspectionWorker",
    "SourceInspectionWorkerProgress",
    "SourceInspectionWorkerResult",
    "SourceInspectionWorkerSignals",
    "SourceInspectionWorkerSpec",
    "StaleSourceInspectionGeneration",
]
