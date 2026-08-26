"""Verified, metadata-only inspection and SourceItem resolution services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from napari_vipp.core.file_sources import VerifiedSourceInspection
from napari_vipp.core.io import inspect_image_source, inspect_image_state
from napari_vipp.core.io.errors import annotate_image_source_exception
from napari_vipp.core.io.model import ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import (
    AxisDeclaration,
    ImageState,
    apply_axis_declaration,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_bundle,
    local_source_identity_from_bundle,
    verify_local_source_identity,
)
from napari_vipp.core.source_items import SourceItem
from napari_vipp.core.source_resolution import (
    SourceItemResolutionError,
    resolve_source_item,
    select_inspected_item,
    verify_saved_source_item,
)


class SourceInspectionPhase(StrEnum):
    """Stable metadata-inspection phases for presentation progress."""

    IDENTITY = "identity"
    HEADER = "header"
    NORMALIZE = "normalize"
    VERIFY = "verify"


class SourceInspectionProgressUnit(StrEnum):
    """Unit carried by a metadata-inspection progress update."""

    INDETERMINATE = "indeterminate"
    BYTES = "bytes"
    STEPS = "steps"


@dataclass(frozen=True, slots=True)
class SourceInspectionProgress:
    """One Qt-free progress update from metadata-only source inspection."""

    phase: SourceInspectionPhase
    current: int = 0
    total: int = 0
    unit: SourceInspectionProgressUnit = (
        SourceInspectionProgressUnit.INDETERMINATE
    )
    message: str = ""

    def __post_init__(self) -> None:
        phase = SourceInspectionPhase(self.phase)
        unit = SourceInspectionProgressUnit(self.unit)
        current = int(self.current)
        total = int(self.total)
        if current < 0 or total < 0:
            raise ValueError("Source inspection progress must be non-negative.")
        if total and current > total:
            raise ValueError(
                "Source inspection progress cannot exceed its declared total."
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "message", str(self.message))

    @property
    def determinate(self) -> bool:
        return (
            self.total > 0
            and self.unit is not SourceInspectionProgressUnit.INDETERMINATE
        )


@dataclass(frozen=True, slots=True)
class ResolvedSourceInspection:
    """One exact header inspection and its canonical logical item binding."""

    verified: VerifiedSourceInspection
    selected_series: ImageSeriesInfo
    raw_image_state: ImageState
    effective_image_state: ImageState
    source_item: SourceItem

    def __post_init__(self) -> None:
        if self.selected_series.key != self.source_item.selector.key:
            raise ValueError(
                "Resolved inspection series and SourceItem keys must agree."
            )
        if self.effective_image_state.shape != self.source_item.resolved.shape:
            raise ValueError(
                "Resolved inspection metadata and SourceItem shapes must agree."
            )

    @property
    def inspection(self) -> SourceInspection:
        return self.verified.inspection

    @property
    def identity(self) -> LocalSourceIdentity:
        return self.verified.identity


InspectionCancelCallback = Callable[[], bool]
InspectionProgressCallback = Callable[[SourceInspectionProgress], None]


def inspect_local_source_item(
    path: str | Path,
    series_index: int = 0,
    *,
    item_key: str | None = None,
    expected_identity: LocalSourceIdentity | None = None,
    expected_source_item: SourceItem | None = None,
    axis_declaration: AxisDeclaration | str | dict[str, object] | None = None,
    inspector: Callable[[str | Path], SourceInspection] | None = None,
    state_inspector: Callable[..., ImageState] | None = None,
    cancel_callback: InspectionCancelCallback | None = None,
    progress_callback: InspectionProgressCallback | None = None,
) -> ResolvedSourceInspection:
    """Inspect and resolve one exact local source revision without a payload read.

    The source is hashed before header inspection and reverified afterwards.
    A saved item key is authoritative; ``series_index`` is used only when no
    canonical key exists.  Third-party header readers may not offer internal
    cancellation points, so cancellation is checked immediately before and
    after each reader boundary.
    """

    source = Path(path).expanduser().resolve(strict=False)
    index_hint = int(series_index)
    if isinstance(series_index, bool) or index_hint < 0:
        raise ValueError("Source inspection series index must be non-negative.")
    requested_key = str(item_key or "").strip()
    if expected_source_item is not None:
        expected_key = expected_source_item.selector.key
        if requested_key and requested_key != expected_key:
            raise SourceItemResolutionError(
                "The requested item key does not match the saved SourceItem key."
            )
        requested_key = expected_key
    declaration = AxisDeclaration.from_value(axis_declaration)
    _check_cancelled(cancel_callback, "before source identity capture")

    try:
        bundle = capture_local_source_bundle(
            source,
            cancel_callback=cancel_callback,
            progress_callback=_byte_progress_reporter(
                progress_callback,
                SourceInspectionPhase.IDENTITY,
            ),
        )
        identity = local_source_identity_from_bundle(bundle)
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="metadata-identity",
            path=source,
            item=requested_key or index_hint,
        )
        raise

    if expected_identity is not None and identity != expected_identity:
        error = SourceChangedError(
            "Local scientific source no longer matches its pinned revision: "
            f"{source}. Press Refresh to inspect the new revision."
        )
        annotate_image_source_exception(
            error,
            stage="metadata-identity",
            path=source,
            item=requested_key or index_hint,
        )
        raise error
    if (
        expected_source_item is not None
        and bundle.revision != expected_source_item.container.revision
    ):
        error = SourceChangedError(
            "Local scientific source changed after its SourceItem was saved. "
            "Press Refresh and explicitly review the new revision."
        )
        annotate_image_source_exception(
            error,
            stage="metadata-identity",
            path=source,
            item=requested_key,
        )
        raise error

    _check_cancelled(cancel_callback, "before opening source metadata")
    _report_step(
        progress_callback,
        SourceInspectionPhase.HEADER,
        0,
        "Opening source metadata headers.",
    )
    selected_inspector = inspect_image_source if inspector is None else inspector
    try:
        inspection = selected_inspector(source)
        selected = select_inspected_item(
            inspection,
            series_index=(None if requested_key else index_hint),
            item_key=(requested_key or None),
        )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="metadata-header",
            path=source,
            item=requested_key or index_hint,
        )
        raise
    _check_cancelled(cancel_callback, "after opening source metadata")
    _report_step(
        progress_callback,
        SourceInspectionPhase.HEADER,
        1,
        f"Inspected source item {selected.key!r}.",
    )

    _report_step(
        progress_callback,
        SourceInspectionPhase.NORMALIZE,
        0,
        "Normalizing axes and metadata evidence.",
    )
    selected_state_inspector = (
        inspect_image_state if state_inspector is None else state_inspector
    )
    try:
        raw_state = selected_state_inspector(
            source,
            inspection=inspection,
            series_index=selected.index,
        )
        effective_state = (
            raw_state
            if declaration is None
            else apply_axis_declaration(
                raw_state,
                declaration,
                declaration_source="Image Source",
            )
        )
        if expected_source_item is None:
            source_item = resolve_source_item(
                bundle,
                inspection,
                item_key=selected.key,
                image_state=effective_state,
                axis_declaration=declaration,
            )
        else:
            source_item = verify_saved_source_item(
                expected_source_item,
                bundle,
                inspection,
                image_state=effective_state,
            )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="source-item-resolution",
            path=source,
            format=inspection.format,
            backend=selected.reader_key,
            item=selected.key,
        )
        raise
    _check_cancelled(cancel_callback, "after normalizing source metadata")
    _report_step(
        progress_callback,
        SourceInspectionPhase.NORMALIZE,
        1,
        "SourceItem metadata contract resolved.",
    )

    _check_cancelled(cancel_callback, "before source identity reverification")
    try:
        verify_local_source_identity(
            source,
            identity,
            cancel_callback=cancel_callback,
            progress_callback=_byte_progress_reporter(
                progress_callback,
                SourceInspectionPhase.VERIFY,
            ),
        )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="metadata-reverification",
            path=source,
            format=inspection.format,
            backend=selected.reader_key,
            item=selected.key,
        )
        raise
    _check_cancelled(cancel_callback, "after source identity reverification")

    verified = VerifiedSourceInspection(inspection, identity, bundle)
    return ResolvedSourceInspection(
        verified=verified,
        selected_series=selected,
        raw_image_state=raw_state,
        effective_image_state=effective_state,
        source_item=source_item,
    )


def _byte_progress_reporter(
    callback: InspectionProgressCallback | None,
    phase: SourceInspectionPhase,
) -> Callable[[int, int, str], None] | None:
    if callback is None:
        return None

    def report(current: int, total: int, message: str) -> None:
        callback(
            SourceInspectionProgress(
                phase=phase,
                current=max(int(current), 0),
                total=max(int(total), 0),
                unit=(
                    SourceInspectionProgressUnit.BYTES
                    if int(total) > 0
                    else SourceInspectionProgressUnit.INDETERMINATE
                ),
                message=message,
            )
        )

    return report


def _report_step(
    callback: InspectionProgressCallback | None,
    phase: SourceInspectionPhase,
    current: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(
        SourceInspectionProgress(
            phase=phase,
            current=current,
            total=1,
            unit=SourceInspectionProgressUnit.STEPS,
            message=message,
        )
    )


def _check_cancelled(
    callback: InspectionCancelCallback | None,
    stage: str,
) -> None:
    if callback is not None and callback():
        raise OperationCancelled(f"Source inspection cancelled {stage}.")


__all__ = [
    "ResolvedSourceInspection",
    "SourceInspectionPhase",
    "SourceInspectionProgress",
    "SourceInspectionProgressUnit",
    "inspect_local_source_item",
]
