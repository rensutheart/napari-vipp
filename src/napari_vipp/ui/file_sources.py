"""Qt scheduling adapter for verified local file-source snapshots."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.file_sources import (
    ImageReader,
    SourceFileSnapshot,
    load_frozen_file_source_snapshot,
)
from napari_vipp.core.io.errors import (
    ImageSourceError,
    ImageSourceErrorCode,
    as_image_source_error,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import LocalSourceIdentity
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.source_window import SourceWindowRequest


class SourceLoadPhase(StrEnum):
    """Stable source-I/O phases shared by workers and presentation surfaces."""

    INSPECT = "inspect"
    DOWNLOAD = "download"
    READ = "read"
    DECODE = "decode"
    NORMALIZE = "normalize"
    PREVIEW = "preview"
    VERIFY = "verify"


class SourceLoadProgressUnit(StrEnum):
    """Unit carried by a source progress update."""

    INDETERMINATE = "indeterminate"
    BYTES = "bytes"
    ITEMS = "items"
    STEPS = "steps"


@dataclass(frozen=True, slots=True)
class SourceLoadProgress:
    """One generation-owned source-load progress update.

    ``run_id`` is the publication generation. Consumers must ignore an update
    whose run does not match their active generation; this keeps queued Qt
    signals from an obsolete worker from replacing newer UI state.
    """

    run_id: int
    phase: SourceLoadPhase
    node_id: str = ""
    item_index: int = 0
    item_total: int = 0
    current: int = 0
    total: int = 0
    unit: SourceLoadProgressUnit = SourceLoadProgressUnit.INDETERMINATE
    message: str = ""

    def __post_init__(self) -> None:
        if int(self.run_id) < 0:
            raise ValueError("Source-load run IDs must be non-negative.")
        if int(self.item_index) < 0 or int(self.item_total) < 0:
            raise ValueError("Source-load item progress must be non-negative.")
        if self.item_total and self.item_index > self.item_total:
            raise ValueError("Source-load item progress cannot exceed its total.")
        if int(self.current) < 0 or int(self.total) < 0:
            raise ValueError("Source-load progress must be non-negative.")
        if self.total and self.current > self.total:
            raise ValueError("Source-load progress cannot exceed its total.")

    @property
    def generation_id(self) -> int:
        """Alias documenting that ``run_id`` owns result publication."""

        return self.run_id

    @property
    def determinate(self) -> bool:
        return self.total > 0 and self.unit is not SourceLoadProgressUnit.INDETERMINATE


@dataclass(frozen=True, slots=True)
class SourceFileLoadSpec:
    """One path and series requested by an Image Source graph node."""

    node_id: str
    path: str
    series_index: int
    cache_key: tuple[object, ...]
    item_key: str = ""
    expected_identity: LocalSourceIdentity | None = None
    expected_source_item: SourceItem | None = None
    exact_window_request: SourceWindowRequest | None = None


@dataclass(frozen=True, slots=True)
class SourceFileLoadResult:
    """All snapshots from one atomic UI load attempt, or one explicit error."""

    run_id: int
    snapshots: dict[tuple[object, ...], SourceFileSnapshot]
    error: str = ""
    node_id: str = ""
    source_error: ImageSourceError | None = None
    cancelled: bool = False
    last_phase: SourceLoadPhase | None = None


class SourceFileLoadSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class SourceFileLoadWorker(QRunnable):
    """Materialize a group of source series away from Qt's GUI thread."""

    def __init__(
        self,
        run_id: int,
        specs: tuple[SourceFileLoadSpec, ...],
        *,
        reader: ImageReader,
    ) -> None:
        super().__init__()
        self.run_id = int(run_id)
        self.specs = tuple(specs)
        self.reader = reader
        self.signals = SourceFileLoadSignals()
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._started = False
        self._terminal = False
        self._last_phase: SourceLoadPhase | None = None

    def cancel(self) -> None:
        """Request cancellation at the next source-I/O checkpoint."""

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
        snapshots: dict[tuple[object, ...], SourceFileSnapshot] = {}
        loaded_identities: dict[str, LocalSourceIdentity] = {}
        current_node_id = ""
        current_spec: SourceFileLoadSpec | None = None
        try:
            self._check_cancelled("before source inspection")
            for item_index, spec in enumerate(self.specs, start=1):
                current_spec = spec
                current_node_id = spec.node_id
                self._check_cancelled("before source inspection")
                self._emit_progress(
                    SourceLoadPhase.INSPECT,
                    spec=spec,
                    item_index=item_index,
                    current=0,
                    total=0,
                    unit=SourceLoadProgressUnit.INDETERMINATE,
                    message="Inspecting and verifying the selected source.",
                )
                expected_identity = spec.expected_identity or loaded_identities.get(
                    spec.path
                )
                read_phase_emitted = False

                def report_core_progress(
                    current: int,
                    total: int,
                    message: str,
                    *,
                    _spec: SourceFileLoadSpec = spec,
                    _item_index: int = item_index,
                ) -> None:
                    nonlocal read_phase_emitted
                    if self.cancellation_requested:
                        return
                    phase = _phase_from_core_message(message)
                    unit = (
                        SourceLoadProgressUnit.BYTES
                        if int(total) > 0
                        else SourceLoadProgressUnit.INDETERMINATE
                    )
                    self._emit_progress(
                        phase,
                        spec=_spec,
                        item_index=_item_index,
                        current=current,
                        total=total,
                        unit=unit,
                        message=message,
                    )
                    if (
                        phase is SourceLoadPhase.INSPECT
                        and int(total) > 0
                        and int(current) >= int(total)
                        and not read_phase_emitted
                    ):
                        read_phase_emitted = True
                        self._emit_progress(
                            SourceLoadPhase.READ,
                            spec=_spec,
                            item_index=_item_index,
                            current=0,
                            total=0,
                            unit=SourceLoadProgressUnit.INDETERMINATE,
                            message="Opening the selected image item.",
                        )

                snapshot = load_frozen_file_source_snapshot(
                    spec.path,
                    spec.series_index,
                    item_key=spec.item_key,
                    expected_identity=expected_identity,
                    expected_source_item=spec.expected_source_item,
                    exact_window_request=spec.exact_window_request,
                    reader=self.reader,
                    cancel_callback=self._cancel_event.is_set,
                    progress_callback=report_core_progress,
                )
                self._check_cancelled("after decoding image data")
                self._emit_progress(
                    SourceLoadPhase.NORMALIZE,
                    spec=spec,
                    item_index=item_index,
                    current=1,
                    total=1,
                    unit=SourceLoadProgressUnit.STEPS,
                    message="Image metadata and axes normalized.",
                )
                self._check_cancelled("after normalizing image metadata")
                snapshots[spec.cache_key] = snapshot
                loaded_identities[spec.path] = snapshot.identity
        except Exception as exc:
            fallback_spec = current_spec or (self.specs[0] if self.specs else None)
            source_error = as_image_source_error(
                exc,
                path=("" if fallback_spec is None else fallback_spec.path),
                item=("" if fallback_spec is None else fallback_spec.series_index),
            )
            self._emit_finished(
                SourceFileLoadResult(
                    self.run_id,
                    {},
                    source_error.display_text,
                    current_node_id,
                    source_error,
                    cancelled=(
                        self.cancellation_requested
                        or source_error.code is ImageSourceErrorCode.CANCELLED
                    ),
                    last_phase=self._last_phase,
                )
            )
            return
        self._emit_finished(
            SourceFileLoadResult(
                self.run_id,
                snapshots,
                last_phase=self._last_phase,
            )
        )

    def _emit_progress(
        self,
        phase: SourceLoadPhase,
        *,
        spec: SourceFileLoadSpec,
        item_index: int,
        current: int,
        total: int,
        unit: SourceLoadProgressUnit,
        message: str,
    ) -> None:
        if self.cancellation_requested:
            return
        safe_total = max(int(total), 0)
        safe_current = max(int(current), 0)
        if safe_total:
            safe_current = min(safe_current, safe_total)
        self._last_phase = phase
        self.signals.progress.emit(
            SourceLoadProgress(
                run_id=self.run_id,
                phase=phase,
                node_id=spec.node_id,
                item_index=int(item_index),
                item_total=len(self.specs),
                current=safe_current,
                total=safe_total,
                unit=unit,
                message=str(message),
            )
        )

    def _emit_finished(self, result: SourceFileLoadResult) -> None:
        with self._state_lock:
            if self._terminal:
                return
            self._terminal = True
        self.signals.finished.emit(result)

    def _check_cancelled(self, stage: str) -> None:
        if self.cancellation_requested:
            raise OperationCancelled(f"Source loading cancelled {stage}.")


def _phase_from_core_message(message: str) -> SourceLoadPhase:
    """Map the Qt-free loader's progress text onto the public typed phases."""

    normalized = str(message).casefold()
    if "reverification" in normalized or "re-verification" in normalized:
        return SourceLoadPhase.VERIFY
    if "validation" in normalized or "hash" in normalized:
        return SourceLoadPhase.INSPECT
    if "materializ" in normalized or "decod" in normalized:
        return SourceLoadPhase.DECODE
    if "download" in normalized:
        return SourceLoadPhase.DOWNLOAD
    if "preview" in normalized:
        return SourceLoadPhase.PREVIEW
    if "metadata" in normalized or "axes" in normalized:
        return SourceLoadPhase.NORMALIZE
    return SourceLoadPhase.READ


__all__ = [
    "SourceFileLoadResult",
    "SourceFileLoadSignals",
    "SourceFileLoadSpec",
    "SourceFileLoadWorker",
    "SourceLoadPhase",
    "SourceLoadProgress",
    "SourceLoadProgressUnit",
]
