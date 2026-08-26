"""Qt worker boundary for presentation-only OME-Zarr source previews."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.io.errors import (
    ImageSourceError,
    ImageSourceErrorCode,
    as_image_source_error,
)
from napari_vipp.core.io.ome_zarr import (
    inspect_ome_zarr,
    read_ome_zarr_presentation_preview,
)
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    verify_local_source_identity,
)
from napari_vipp.core.source_preview import (
    SourcePreviewControl,
    SourcePreviewProgress,
    SourcePreviewRequest,
    SourcePreviewResult,
    StaleSourcePreviewGeneration,
)
from napari_vipp.core.source_resolution import (
    SourceItemResolutionError,
    select_inspected_item,
)


@dataclass(frozen=True, slots=True)
class SourcePreviewSeriesSelection:
    """A stable logical OME-Zarr item plus its non-authoritative index hint."""

    item_key: str
    series_index_hint: int = 0

    def __post_init__(self) -> None:
        item_key = str(self.item_key).strip()
        if not item_key:
            raise ValueError("A source preview requires a stable item key.")
        if isinstance(self.series_index_hint, bool) or int(self.series_index_hint) < 0:
            raise ValueError("A source preview series index hint must be non-negative.")
        object.__setattr__(self, "item_key", item_key)
        object.__setattr__(self, "series_index_hint", int(self.series_index_hint))


@dataclass(frozen=True, slots=True)
class SourcePreviewWorkerSpec:
    """One generation-qualified request for a local presentation preview."""

    generation: int
    path: str
    selection: SourcePreviewSeriesSelection
    request: SourcePreviewRequest
    node_id: str = ""
    expected_identity: LocalSourceIdentity | None = None

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("A source preview generation must be non-negative.")
        path = str(self.path).strip()
        if not path:
            raise ValueError("A source preview requires a local OME-Zarr path.")
        if not isinstance(self.selection, SourcePreviewSeriesSelection):
            raise TypeError(
                "selection must be a SourcePreviewSeriesSelection instance."
            )
        if not isinstance(self.request, SourcePreviewRequest):
            raise TypeError("request must be a SourcePreviewRequest instance.")
        if self.expected_identity is not None and not isinstance(
            self.expected_identity,
            LocalSourceIdentity,
        ):
            raise TypeError("expected_identity must be a LocalSourceIdentity or None.")
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "node_id", str(self.node_id))


@dataclass(frozen=True, slots=True)
class SourcePreviewWorkerResult:
    """The sole terminal publication from one source-preview worker."""

    generation: int
    path: str
    item_key: str
    node_id: str = ""
    preview: SourcePreviewResult | None = None
    source_error: ImageSourceError | None = None
    cancelled: bool = False
    stale: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("A source preview generation must be non-negative.")
        if (self.preview is None) == (self.source_error is None):
            raise ValueError(
                "A source preview terminal result must contain exactly one of "
                "preview or source_error."
            )
        if self.preview is not None:
            if self.preview.generation != int(self.generation):
                raise ValueError(
                    "A source preview result must belong to the worker generation."
                )
            if not self.preview.presentation_only:
                raise ValueError("A source preview must remain presentation-only.")
            if self.cancelled or self.stale:
                raise ValueError(
                    "A successful source preview cannot be cancelled or stale."
                )
        if self.stale and not self.cancelled:
            raise ValueError("A stale source preview must also be cancelled.")
        object.__setattr__(self, "generation", int(self.generation))

    @property
    def error(self) -> str:
        """Concise text safe to display without losing structured details."""

        return "" if self.source_error is None else self.source_error.display_text

    @property
    def succeeded(self) -> bool:
        return self.preview is not None


class SourcePreviewWorkerSignals(QObject):
    """Queued Qt publications from :class:`SourcePreviewWorker`."""

    progress = Signal(object)
    finished = Signal(object)


class SourcePreviewWorker(QRunnable):
    """Read one bounded OME-Zarr preview away from Qt's GUI thread.

    Consumers must still compare every progress/result ``generation`` with
    their active generation.  ``current_generation`` additionally stops stale
    work cooperatively before it can publish a successful terminal result.
    """

    def __init__(
        self,
        spec: SourcePreviewWorkerSpec,
        *,
        current_generation: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(spec, SourcePreviewWorkerSpec):
            raise TypeError("spec must be a SourcePreviewWorkerSpec instance.")
        if current_generation is not None and not callable(current_generation):
            raise TypeError("current_generation must be callable or None.")
        self.spec = spec
        self.signals = SourcePreviewWorkerSignals()
        self._current_generation = current_generation
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._started = False
        self._terminal = False

    def cancel(self) -> None:
        """Request cancellation at the next preview-reader checkpoint."""

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
        control = SourcePreviewControl(
            generation=spec.generation,
            cancelled=self._cancel_event.is_set,
            current_generation=self._current_generation,
            reporter=self._emit_progress,
        )
        try:
            control.check_active()
            self._verify_source_revision(control)
            selected = self._resolve_selection()
            control.check_active()
            preview = read_ome_zarr_presentation_preview(
                Path(spec.path),
                selected.index,
                request=spec.request,
                control=control,
            )
            control.check_active()
            self._verify_selection_unchanged(selected.index)
            self._verify_source_revision(control)
            control.check_active()
            if preview.generation != spec.generation:
                raise SourceItemResolutionError(
                    "The preview reader returned a different publication generation."
                )
            if not preview.presentation_only:
                raise SourceItemResolutionError(
                    "The preview reader returned data without the presentation-only "
                    "contract."
                )
        except Exception as exc:
            stale = isinstance(exc, StaleSourcePreviewGeneration)
            source_error = _preview_source_error(exc, spec)
            self._emit_finished(
                SourcePreviewWorkerResult(
                    generation=spec.generation,
                    path=spec.path,
                    item_key=spec.selection.item_key,
                    node_id=spec.node_id,
                    source_error=source_error,
                    cancelled=(
                        stale or source_error.code is ImageSourceErrorCode.CANCELLED
                    ),
                    stale=stale,
                )
            )
            return

        self._emit_finished(
            SourcePreviewWorkerResult(
                generation=spec.generation,
                path=spec.path,
                item_key=spec.selection.item_key,
                node_id=spec.node_id,
                preview=preview,
            )
        )

    def _verify_source_revision(self, control: SourcePreviewControl) -> None:
        expected = self.spec.expected_identity
        if expected is None:
            return

        def check_active() -> bool:
            control.check_active()
            return False

        verify_local_source_identity(
            Path(self.spec.path),
            expected,
            cancel_callback=check_active,
        )
        control.check_active()

    def _resolve_selection(self):
        inspection = inspect_ome_zarr(Path(self.spec.path))
        return select_inspected_item(
            inspection,
            item_key=self.spec.selection.item_key,
            series_index=self.spec.selection.series_index_hint,
        )

    def _verify_selection_unchanged(self, decoded_index: int) -> None:
        selected = self._resolve_selection()
        if selected.index != int(decoded_index):
            raise SourceItemResolutionError(
                f"OME-Zarr item {self.spec.selection.item_key!r} changed position "
                "while its preview was being read; the result was discarded."
            )

    def _emit_progress(self, progress: SourcePreviewProgress) -> None:
        if self.cancellation_requested:
            return
        if progress.generation != self.spec.generation:
            return
        self.signals.progress.emit(progress)

    def _emit_finished(self, result: SourcePreviewWorkerResult) -> None:
        with self._state_lock:
            if self._terminal:
                return
            self._terminal = True
        self.signals.finished.emit(result)


def _preview_source_error(
    error: Exception,
    spec: SourcePreviewWorkerSpec,
) -> ImageSourceError:
    if isinstance(error, SourceItemResolutionError):
        return ImageSourceError(
            ImageSourceErrorCode.CONTRACT_MISMATCH,
            str(error),
            stage="preview",
            path=spec.path,
            format="ome-zarr",
            item=spec.selection.item_key,
            remediation="Refresh the source and explicitly select the item again.",
        )
    return as_image_source_error(
        error,
        stage="preview",
        path=spec.path,
        format="ome-zarr",
        item=spec.selection.item_key,
    )


__all__ = [
    "SourcePreviewSeriesSelection",
    "SourcePreviewWorker",
    "SourcePreviewWorkerResult",
    "SourcePreviewWorkerSignals",
    "SourcePreviewWorkerSpec",
]
