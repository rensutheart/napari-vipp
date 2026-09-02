"""Verified, detached snapshots for local interactive image sources.

This module owns the scientific boundary between a mutable local file/store and
the pipeline. It deliberately contains no Qt behavior; scheduling and cache
policy remain responsibilities of the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from pathlib import Path
from typing import Protocol

import numpy as np

from napari_vipp.core.host_memory import (
    capture_host_memory,
    preflight_host_allocation,
)
from napari_vipp.core.io import (
    ImageDataset,
    SourceInspection,
    inspect_image_source,
    inspect_image_state,
    normalize_local_image_source_path,
    read_image,
    read_image_exact_window,
    validate_local_image_source_path,
)
from napari_vipp.core.io.errors import annotate_image_source_exception
from napari_vipp.core.metadata import (
    AxisDeclaration,
    apply_axis_declaration,
    image_state_from_array,
)
from napari_vipp.core.pipeline import SourcePayload
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_bundle,
    local_source_identity_from_bundle,
    verify_local_source_identity,
)
from napari_vipp.core.source_items import SourceContainerBundle, SourceItem
from napari_vipp.core.source_resolution import (
    resolve_source_item,
    select_inspected_item,
    verify_saved_source_item,
)
from napari_vipp.core.source_window import (
    ExactSourceWindowData,
    SourceWindowControl,
    SourceWindowReadEstimate,
    SourceWindowRequest,
)

FILE_SOURCE_SNAPSHOT_POLICY = "pinned until Refresh"
_MATERIALIZE_CHUNK_BYTES = 16 * 1024 * 1024
_MEMORY_PREFLIGHT_MIN_BYTES = 256 * 1024 * 1024

CancelCallback = Callable[[], bool]
SourceLoadProgressCallback = Callable[[int, int, str], None]


class ImageReader(Protocol):
    """Callable shape accepted by the frozen-source loader."""

    def __call__(
        self,
        path: str | Path,
        *,
        series_index: int = 0,
    ) -> ImageDataset: ...


class ImageInspector(Protocol):
    def __call__(self, path: str | Path) -> SourceInspection: ...


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """One exact local source revision detached for pipeline ownership."""

    payload: SourcePayload
    inspection: SourceInspection
    identity: LocalSourceIdentity
    source_item: SourceItem


@dataclass(frozen=True, slots=True)
class VerifiedSourceInspection:
    """Source structure paired with the exact revision that was inspected."""

    inspection: SourceInspection
    identity: LocalSourceIdentity
    bundle: SourceContainerBundle | None = None


def load_frozen_file_source_snapshot(
    path: str | Path,
    series_index: int = 0,
    *,
    item_key: str | None = None,
    axis_declaration: AxisDeclaration | str | dict[str, object] | None = None,
    expected_identity: LocalSourceIdentity | None = None,
    expected_source_item: SourceItem | None = None,
    reader: ImageReader | None = None,
    inspector: ImageInspector | None = None,
    exact_window_request: SourceWindowRequest | None = None,
    cancel_callback: CancelCallback | None = None,
    progress_callback: SourceLoadProgressCallback | None = None,
) -> SourceFileSnapshot:
    """Read one exact local revision into an owned, read-only NumPy array.

    The complete file or directory tree is hashed before the reader opens it
    and verified again only after lazy data has been fully materialized. A
    caller may inject its reader; this keeps UI monkeypatching and alternate
    reader dispatch outside the core scientific boundary.
    """
    requested_declaration = AxisDeclaration.from_value(axis_declaration)
    saved_declaration = None
    if expected_source_item is not None and expected_source_item.selector.source_axes:
        saved_declaration = AxisDeclaration(
            ",".join(expected_source_item.selector.source_axes),
            ",".join(expected_source_item.selector.effective_axes),
        )
    if (
        requested_declaration is not None
        and saved_declaration is not None
        and requested_declaration != saved_declaration
    ):
        raise ValueError(
            "The requested source-axis declaration conflicts with the saved "
            "canonical SourceItem."
        )
    effective_declaration = saved_declaration or requested_declaration
    source = normalize_local_image_source_path(path)
    _check_boundary_cancelled(
        cancel_callback,
        "before validating the source",
        stage="source-validation",
        source=source,
        item=series_index,
    )
    try:
        source = validate_local_image_source_path(source)
        container = capture_local_source_bundle(
            source,
            cancel_callback=cancel_callback,
            progress_callback=_phase_progress_callback(
                progress_callback,
                "Source validation 1/3",
            ),
        )
        identity = local_source_identity_from_bundle(container)
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="source-validation",
            path=source,
            item=series_index,
        )
        raise
    if expected_identity is not None and identity != expected_identity:
        error = SourceChangedError(
            "Local scientific source changed after its interactive snapshot "
            f"was pinned: {source}. Press Refresh to load the new revision."
        )
        annotate_image_source_exception(
            error,
            stage="source-validation",
            path=source,
            item=series_index,
        )
        raise error

    if (
        expected_source_item is not None
        and container.revision != expected_source_item.container.revision
    ):
        error = SourceChangedError(
            "Local scientific source changed after its SourceItem was saved: "
            f"{source}. Press Refresh and explicitly review the new revision."
        )
        annotate_image_source_exception(
            error,
            stage="source-validation",
            path=source,
            item=expected_source_item.selector.key,
        )
        raise error

    requested_key = (
        expected_source_item.selector.key
        if expected_source_item is not None
        else str(item_key or "").strip()
    )
    effective_series_index = int(series_index)
    preinspected: SourceInspection | None = None
    preinspected_item = None
    default_reader_selects_saved_key = bool(
        expected_source_item is not None
        and source.suffix.casefold() == ".czi"
        and expected_source_item.reader.implementation == "czifile"
    )
    if (
        inspector is not None
        or exact_window_request is not None
        or (reader is None and not default_reader_selects_saved_key)
    ):
        selected_inspector = inspect_image_source if inspector is None else inspector
        try:
            preinspected = selected_inspector(source)
            preinspected_item = select_inspected_item(
                preinspected,
                series_index=(None if requested_key else effective_series_index),
                item_key=(requested_key or None),
            )
        except Exception as exc:
            annotate_image_source_exception(
                exc,
                stage="item-selection",
                path=source,
                item=requested_key,
            )
            raise
        effective_series_index = preinspected_item.index
        if exact_window_request is None:
            _preflight_source_memory(
                preinspected_item,
                source=source,
                progress_callback=progress_callback,
            )
    elif expected_source_item is not None:
        # A UI worker may inject the reader to preserve one deterministic read
        # boundary.  The already verified SourceItem still carries the exact
        # decoded-size and lazy/eager contract needed for a pre-decode memory
        # decision, so do not silently skip preflight in that route.
        _preflight_source_memory(
            expected_source_item,
            source=source,
            progress_callback=progress_callback,
        )

    if exact_window_request is not None:
        if preinspected is None or preinspected_item is None:
            raise RuntimeError(
                "Exact source-window loading requires a verified source inspection."
            )
        return _load_exact_window_snapshot(
            source,
            effective_series_index,
            container=container,
            identity=identity,
            inspection=preinspected,
            selected=preinspected_item,
            request=exact_window_request,
            effective_declaration=effective_declaration,
            saved_declaration=saved_declaration,
            expected_source_item=expected_source_item,
            cancel_callback=cancel_callback,
            progress_callback=progress_callback,
        )

    selected_reader = read_image if reader is None else reader
    _check_boundary_cancelled(
        cancel_callback,
        "before opening the source",
        stage="open",
        source=source,
        item=series_index,
    )
    try:
        if (
            requested_key
            and default_reader_selects_saved_key
            and selected_reader is read_image
        ):
            dataset = selected_reader(
                source,
                series_index=effective_series_index,
                item_key=requested_key,
            )
        else:
            dataset = selected_reader(source, series_index=effective_series_index)
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="open",
            path=source,
            item=series_index,
        )
        raise
    if requested_key and dataset.selected_series.key != requested_key:
        error = RuntimeError(
            "Reader contract mismatch: requested item key "
            f"{requested_key!r} but the reader returned "
            f"{dataset.selected_series.key!r}."
        )
        annotate_image_source_exception(
            error,
            stage="item-selection",
            path=source,
            format=dataset.inspection.format,
            backend=_dataset_reader_backend(dataset),
            item=requested_key,
        )
        raise error
    if (
        preinspected_item is not None
        and dataset.selected_series.key != preinspected_item.key
    ):
        error = RuntimeError(
            "Reader contract mismatch: inspection selected item key "
            f"{preinspected_item.key!r} but read returned "
            f"{dataset.selected_series.key!r}."
        )
        annotate_image_source_exception(
            error,
            stage="item-selection",
            path=source,
            format=dataset.inspection.format,
            backend=_dataset_reader_backend(dataset),
            item=preinspected_item.key,
        )
        raise error
    # A canonical item key may resolve to a different ordinal after a reader
    # upgrade.  Persist the ordinal actually returned by the verified reader;
    # the stable key remains authoritative for future loads.
    effective_series_index = int(dataset.selected_series.index)
    backend = _dataset_reader_backend(dataset)
    _check_boundary_cancelled(
        cancel_callback,
        "before materializing image data",
        stage="materialization",
        source=source,
        format=dataset.inspection.format,
        backend=backend,
        item=dataset.selected_series.key,
    )
    try:
        data = _materialize_owned_array(
            dataset.data,
            source=source,
            cancel_callback=cancel_callback,
            progress_callback=_phase_progress_callback(
                progress_callback,
                "Source materialization 2/3",
            ),
        )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="materialization",
            path=source,
            format=dataset.inspection.format,
            backend=backend,
            item=dataset.selected_series.key,
        )
        raise
    data.setflags(write=False)
    _check_boundary_cancelled(
        cancel_callback,
        "before reverifying the source",
        stage="source-reverification",
        source=source,
        format=dataset.inspection.format,
        backend=backend,
        item=dataset.selected_series.key,
    )
    try:
        verify_local_source_identity(
            source,
            identity,
            cancel_callback=cancel_callback,
            progress_callback=_phase_progress_callback(
                progress_callback,
                "Source reverification 3/3",
            ),
        )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="source-reverification",
            path=source,
            format=dataset.inspection.format,
            backend=backend,
            item=dataset.selected_series.key,
        )
        raise
    _check_boundary_cancelled(
        cancel_callback,
        "after reverifying the source",
        stage="source-reverification",
        source=source,
        format=dataset.inspection.format,
        backend=backend,
        item=dataset.selected_series.key,
    )

    source_state = dataset.image_state
    try:
        snapshot_state = image_state_from_array(
            data,
            axes=source_state.axes,
            metadata_source=source_state.metadata_source,
            source_name=source_state.source_name,
            history=source_state.history,
            channels=source_state.channels,
            acquisition=source_state.acquisition,
            source=source_state.source,
        )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="metadata-normalization",
            path=source,
            format=dataset.inspection.format,
            backend=backend,
            item=dataset.selected_series.key,
        )
        raise
    if snapshot_state is not None:
        snapshot_state = replace(snapshot_state, kind=source_state.kind)

    axis_semantics_resolved = effective_declaration is not None
    if effective_declaration is not None:
        snapshot_state = apply_axis_declaration(
            snapshot_state,
            effective_declaration,
            declaration_source=(
                "saved SourceItem"
                if saved_declaration is not None
                else "Image Source"
            ),
        )

    try:
        if expected_source_item is None:
            source_item = resolve_source_item(
                container,
                dataset.inspection,
                item_key=dataset.selected_series.key,
                image_state=snapshot_state,
                axis_declaration=effective_declaration,
            )
        else:
            source_item = verify_saved_source_item(
                expected_source_item,
                container,
                dataset.inspection,
                image_state=snapshot_state,
            )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="source-item-resolution",
            path=source,
            format=dataset.inspection.format,
            backend=backend,
            item=dataset.selected_series.key,
        )
        raise

    payload = SourcePayload(
        data,
        {
            "vipp_source_path": str(source),
            "vipp_source_identity": identity.to_dict(),
            "vipp_source_series_index": effective_series_index,
            "vipp_source_item_key": source_item.selector.key,
            "vipp_source_snapshot_policy": FILE_SOURCE_SNAPSHOT_POLICY,
            "vipp_source_item_digest": source_item.digest,
            "vipp_source_item": source_item.to_public_dict(),
        },
        dataset.selected_series.name,
        snapshot_state,
        identity,
        axis_semantics_resolved,
        source_item=source_item,
    )
    return SourceFileSnapshot(payload, dataset.inspection, identity, source_item)


def _load_exact_window_snapshot(
    source: Path,
    series_index: int,
    *,
    container: SourceContainerBundle,
    identity: LocalSourceIdentity,
    inspection: SourceInspection,
    selected,
    request: SourceWindowRequest,
    effective_declaration: AxisDeclaration | None,
    saved_declaration: AxisDeclaration | None,
    expected_source_item: SourceItem | None,
    cancel_callback: CancelCallback | None,
    progress_callback: SourceLoadProgressCallback | None,
) -> SourceFileSnapshot:
    """Resolve a complete SourceItem, then materialize only one exact window."""

    if request.axis_declaration != effective_declaration:
        raise ValueError(
            "The exact source-window axis declaration is stale or does not "
            "match the Image Source declaration. Recalculate the workflow."
        )
    raw_state = inspect_image_state(
        source,
        inspection=inspection,
        series_index=series_index,
    )
    full_state = raw_state
    if effective_declaration is not None:
        full_state = apply_axis_declaration(
            raw_state,
            effective_declaration,
            declaration_source=(
                "saved SourceItem"
                if saved_declaration is not None
                else "Image Source"
            ),
        )
    try:
        if expected_source_item is None:
            source_item = resolve_source_item(
                container,
                inspection,
                item_key=selected.key,
                image_state=full_state,
                axis_declaration=effective_declaration,
            )
        else:
            source_item = verify_saved_source_item(
                expected_source_item,
                container,
                inspection,
                image_state=full_state,
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

    revision = source_item.container.revision.sha256
    if request.source_revision and request.source_revision != revision:
        raise SourceChangedError(
            "The exact source-window plan was made for a different source "
            "revision. Refresh and review the source again."
        )
    if request.source_item_digest and request.source_item_digest != source_item.digest:
        raise SourceChangedError(
            "The exact source-window plan was made for a different SourceItem. "
            "Refresh and review the selected image again."
        )
    bound_request = replace(
        request,
        source_revision=revision,
        source_item_digest=source_item.digest,
    )
    _check_boundary_cancelled(
        cancel_callback,
        "before reading the exact source window",
        stage="materialization",
        source=source,
        format=inspection.format,
        backend=selected.reader_key,
        item=selected.key,
    )
    control = SourceWindowControl(
        cancelled=cancel_callback,
        reporter=_phase_progress_callback(
            progress_callback,
            "Source window materialization 2/3",
        ),
        preflight=lambda estimate: _preflight_source_window_memory(
            selected,
            bound_request,
            estimate,
            source=source,
            progress_callback=progress_callback,
        ),
    )
    try:
        result = read_image_exact_window(
            source,
            series_index=series_index,
            request=bound_request,
            control=control,
        )
    except Exception as exc:
        annotate_image_source_exception(
            exc,
            stage="materialization",
            path=source,
            format=inspection.format,
            backend=selected.reader_key,
            item=selected.key,
        )
        raise
    if (
        result.identity.source_shape != tuple(source_item.resolved.shape)
        or result.identity.source_dtype != np.dtype(source_item.resolved.dtype).name
        or result.identity.axis_names
        != tuple(axis.name.casefold() for axis in full_state.axes)
        or result.identity.item_key != source_item.selector.key
        or result.identity.source_revision != revision
        or result.identity.source_item_digest != source_item.digest
    ):
        raise RuntimeError(
            "Exact source-window reader evidence does not match the verified "
            "complete SourceItem. No pixels were published."
        )

    _check_boundary_cancelled(
        cancel_callback,
        "before reverifying the source",
        stage="source-reverification",
        source=source,
        format=inspection.format,
        backend=selected.reader_key,
        item=selected.key,
    )
    verify_local_source_identity(
        source,
        identity,
        cancel_callback=cancel_callback,
        progress_callback=_phase_progress_callback(
            progress_callback,
            "Source reverification 3/3",
        ),
    )
    wrapped = ExactSourceWindowData(
        result.data,
        result.image_state,
        result.identity,
    )
    metadata = {
        "vipp_source_path": str(source),
        "vipp_source_identity": identity.to_dict(),
        "vipp_source_series_index": int(series_index),
        "vipp_source_item_key": source_item.selector.key,
        "vipp_source_snapshot_policy": FILE_SOURCE_SNAPSHOT_POLICY,
        "vipp_source_item_digest": source_item.digest,
        "vipp_source_item": source_item.to_public_dict(),
        "vipp_source_window": result.identity.to_dict(include_source_uri=False),
        "vipp_source_window_digest": result.identity.digest,
        "vipp_source_window_read_estimate": (
            None
            if result.identity.read_estimate is None
            else result.identity.read_estimate.to_dict()
        ),
        "vipp_source_read_strategy": "exact-level-0-window",
    }
    payload = SourcePayload(
        wrapped,
        metadata,
        selected.name,
        full_state,
        identity,
        effective_declaration is not None,
        source_item=source_item,
    )
    return SourceFileSnapshot(payload, inspection, identity, source_item)


def _materialize_owned_array(
    value,
    *,
    source: Path,
    cancel_callback: CancelCallback | None,
    progress_callback: SourceLoadProgressCallback | None,
) -> np.ndarray:
    """Copy an array-like source with bounded cooperative checkpoints."""
    shape = _declared_array_shape(value)
    dtype = _declared_array_dtype(value)
    if shape is None or dtype is None or not hasattr(value, "__getitem__"):
        _report_progress(
            progress_callback,
            0,
            0,
            f"Materializing image data: {source}",
        )
        _check_cancelled(cancel_callback, "while materializing image data")
        result = np.array(
            np.asarray(value),
            copy=True,
            order="K",
            subok=False,
        )
        _check_cancelled(cancel_callback, "while materializing image data")
        _report_progress(
            progress_callback,
            int(result.nbytes),
            int(result.nbytes),
            f"Image data materialized: {source}",
        )
        return result

    order = "F" if _is_fortran_only_array(value) else "C"
    result = np.empty(shape, dtype=dtype, order=order)
    total_bytes = int(result.nbytes)
    _report_progress(
        progress_callback,
        0,
        total_bytes,
        f"Materializing image data: {source}",
    )
    if not shape or total_bytes == 0:
        _check_cancelled(cancel_callback, "while materializing image data")
        if not shape:
            result[...] = np.asarray(value, dtype=dtype)
        _check_cancelled(cancel_callback, "while materializing image data")
        _report_progress(
            progress_callback,
            total_bytes,
            total_bytes,
            f"Image data materialized: {source}",
        )
        return result

    axis, axis_step = _materialization_axis_and_step(shape, dtype.itemsize)
    suffix_items = prod(shape[axis + 1 :])
    bytes_per_axis_item = max(1, suffix_items * int(dtype.itemsize))
    processed_bytes = 0
    prefix_ranges = tuple(range(size) for size in shape[:axis])
    prefix_values = product(*prefix_ranges) if prefix_ranges else ((),)
    for prefix in prefix_values:
        for start in range(0, shape[axis], axis_step):
            stop = min(shape[axis], start + axis_step)
            selector = (
                *prefix,
                slice(start, stop),
                *(slice(None) for _ in shape[axis + 1 :]),
            )
            _check_cancelled(cancel_callback, "while materializing image data")
            block = np.asarray(value[selector], dtype=dtype)
            _check_cancelled(cancel_callback, "while materializing image data")
            result[selector] = block
            processed_bytes += (stop - start) * bytes_per_axis_item
            _report_progress(
                progress_callback,
                min(processed_bytes, total_bytes),
                total_bytes,
                f"Materializing image data: {source}",
            )
            _check_cancelled(cancel_callback, "while materializing image data")
    _report_progress(
        progress_callback,
        total_bytes,
        total_bytes,
        f"Image data materialized: {source}",
    )
    return result


def _preflight_source_memory(
    selected,
    *,
    source: Path,
    progress_callback: SourceLoadProgressCallback | None,
) -> None:
    if isinstance(selected, SourceItem):
        estimated = selected.resolved.estimated_decoded_bytes
        lazy_data = selected.capabilities.lazy_data
        name = selected.resolved.name
        key = selected.selector.key
        backend = selected.reader.implementation
    else:
        estimated = selected.estimated_decoded_bytes
        lazy_data = "lazy_data" in set(selected.capabilities)
        name = selected.name
        key = selected.key
        backend = selected.reader_key
    if estimated is None or estimated < _MEMORY_PREFLIGHT_MIN_BYTES:
        return
    required = int(estimated) * (1 if lazy_data else 2)
    decision = preflight_host_allocation(
        capture_host_memory(),
        required_bytes=required,
        purpose=f"loading {name or key}",
    )
    _report_progress(
        progress_callback,
        0,
        required,
        "Source memory preflight: " + decision.reason,
    )
    if decision.allowed:
        return
    error = MemoryError("Source memory preflight refused the load. " + decision.reason)
    annotate_image_source_exception(
        error,
        stage="memory-preflight",
        path=source,
        format="",
        backend=backend,
        item=key,
    )
    raise error


def _preflight_source_window_memory(
    selected,
    request: SourceWindowRequest,
    estimate: SourceWindowReadEstimate,
    *,
    source: Path,
    progress_callback: SourceLoadProgressCallback | None,
) -> None:
    """Preflight the ROI plus every decoded chunk it intersects."""

    shape = tuple(int(size) for size in selected.shape)
    selection = request.normalized_selection(shape)
    output_shape = tuple(
        int(selector.stop) - int(selector.start) for selector in selection
    )
    output_bytes = prod(output_shape) * np.dtype(selected.dtype).itemsize
    if not isinstance(estimate, SourceWindowReadEstimate):
        raise TypeError("Source-window memory preflight requires a read estimate.")
    if estimate.requested_decoded_bytes != int(output_bytes):
        raise RuntimeError(
            "Source-window read estimate does not match the verified ROI shape."
        )
    required = int(estimate.estimated_peak_bytes)
    if required < _MEMORY_PREFLIGHT_MIN_BYTES:
        return
    decision = preflight_host_allocation(
        capture_host_memory(),
        required_bytes=required,
        purpose=(
            "decoding touched chunks for the retained source window from "
            f"{selected.name or selected.key}"
        ),
    )
    _report_progress(
        progress_callback,
        0,
        required,
        "Source-window memory preflight: "
        f"{estimate.estimated_touched_chunk_count} chunks, "
        f"{estimate.estimated_touched_chunk_decoded_bytes} decoded chunk bytes; "
        + decision.reason,
    )
    if decision.allowed:
        return
    error = MemoryError(
        "Source-window memory preflight refused the retained crop because its "
        "intersecting decoded chunks and working buffers do not fit. "
        + decision.reason
    )
    annotate_image_source_exception(
        error,
        stage="memory-preflight",
        path=source,
        format="",
        backend=selected.reader_key,
        item=selected.key,
    )
    raise error


def _declared_array_shape(value) -> tuple[int, ...] | None:
    try:
        shape = tuple(int(size) for size in value.shape)
    except (AttributeError, TypeError, ValueError):
        return None
    if any(size < 0 for size in shape):
        return None
    return shape


def _declared_array_dtype(value) -> np.dtype | None:
    try:
        return np.dtype(value.dtype)
    except (AttributeError, TypeError, ValueError):
        return None


def _is_fortran_only_array(value) -> bool:
    flags = getattr(value, "flags", None)
    return bool(
        flags is not None
        and getattr(flags, "f_contiguous", False)
        and not getattr(flags, "c_contiguous", False)
    )


def _materialization_axis_and_step(
    shape: tuple[int, ...],
    itemsize: int,
) -> tuple[int, int]:
    for axis in range(len(shape)):
        suffix_items = prod(shape[axis + 1 :])
        bytes_per_axis_item = max(1, suffix_items * int(itemsize))
        if bytes_per_axis_item <= _MATERIALIZE_CHUNK_BYTES:
            return axis, max(1, _MATERIALIZE_CHUNK_BYTES // bytes_per_axis_item)
    return len(shape) - 1, 1


def _phase_progress_callback(
    callback: SourceLoadProgressCallback | None,
    phase: str,
) -> SourceLoadProgressCallback | None:
    if callback is None:
        return None

    def report(current: int, total: int, message: str) -> None:
        _report_progress(
            callback,
            current,
            total,
            f"{phase}: {message}",
        )

    return report


def _check_cancelled(
    callback: CancelCallback | None,
    stage: str,
) -> None:
    if callback is not None and callback():
        raise OperationCancelled(f"Source loading cancelled {stage}.")


def _check_boundary_cancelled(
    callback: CancelCallback | None,
    message_stage: str,
    *,
    stage: str,
    source: Path,
    format: str = "",
    backend: str = "",
    item: str | int = "",
) -> None:
    try:
        _check_cancelled(callback, message_stage)
    except OperationCancelled as exc:
        annotate_image_source_exception(
            exc,
            stage=stage,
            path=source,
            format=format,
            backend=backend,
            item=item,
        )
        raise


def _dataset_reader_backend(dataset: ImageDataset) -> str:
    selected_backend = str(getattr(dataset.selected_series, "reader_key", "") or "")
    if selected_backend:
        return selected_backend
    provenance = dataset.provenance
    if not isinstance(provenance, dict):
        return ""
    return str(provenance.get("reader", "") or "")


def _report_progress(
    callback: SourceLoadProgressCallback | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(int(current), int(total), str(message))
    except Exception:
        # Presentation hooks never invalidate the verified source snapshot.
        return


__all__ = [
    "FILE_SOURCE_SNAPSHOT_POLICY",
    "ImageReader",
    "ImageInspector",
    "SourceFileSnapshot",
    "VerifiedSourceInspection",
    "load_frozen_file_source_snapshot",
]
