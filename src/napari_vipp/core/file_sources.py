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

from napari_vipp.core.io import (
    ImageDataset,
    SourceInspection,
    read_image,
)
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import SourcePayload
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_identity,
    verify_local_source_identity,
)

FILE_SOURCE_SNAPSHOT_POLICY = "pinned until Refresh"
_MATERIALIZE_CHUNK_BYTES = 16 * 1024 * 1024

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


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """One exact local source revision detached for pipeline ownership."""

    payload: SourcePayload
    inspection: SourceInspection
    identity: LocalSourceIdentity


@dataclass(frozen=True, slots=True)
class VerifiedSourceInspection:
    """Source structure paired with the exact revision that was inspected."""

    inspection: SourceInspection
    identity: LocalSourceIdentity


def load_frozen_file_source_snapshot(
    path: str | Path,
    series_index: int,
    *,
    expected_identity: LocalSourceIdentity | None = None,
    reader: ImageReader | None = None,
    cancel_callback: CancelCallback | None = None,
    progress_callback: SourceLoadProgressCallback | None = None,
) -> SourceFileSnapshot:
    """Read one exact local revision into an owned, read-only NumPy array.

    The complete file or directory tree is hashed before the reader opens it
    and verified again only after lazy data has been fully materialized. A
    caller may inject its reader; this keeps UI monkeypatching and alternate
    reader dispatch outside the core scientific boundary.
    """
    source = Path(path).expanduser().resolve(strict=False)
    _check_cancelled(cancel_callback, "before validating the source")
    identity = capture_local_source_identity(
        source,
        cancel_callback=cancel_callback,
        progress_callback=_phase_progress_callback(
            progress_callback,
            "Source validation 1/3",
        ),
    )
    if expected_identity is not None and identity != expected_identity:
        raise SourceChangedError(
            "Local scientific source changed after its interactive snapshot "
            f"was pinned: {source}. Press Refresh to load the new revision."
        )

    selected_reader = read_image if reader is None else reader
    _check_cancelled(cancel_callback, "before opening the source")
    dataset = selected_reader(source, series_index=int(series_index))
    _check_cancelled(cancel_callback, "before materializing image data")
    data = _materialize_owned_array(
        dataset.data,
        source=source,
        cancel_callback=cancel_callback,
        progress_callback=_phase_progress_callback(
            progress_callback,
            "Source materialization 2/3",
        ),
    )
    data.setflags(write=False)
    _check_cancelled(cancel_callback, "before reverifying the source")
    verify_local_source_identity(
        source,
        identity,
        cancel_callback=cancel_callback,
        progress_callback=_phase_progress_callback(
            progress_callback,
            "Source reverification 3/3",
        ),
    )
    _check_cancelled(cancel_callback, "after reverifying the source")

    source_state = dataset.image_state
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
    if snapshot_state is not None:
        snapshot_state = replace(snapshot_state, kind=source_state.kind)

    payload = SourcePayload(
        data,
        {
            "vipp_source_path": str(source),
            "vipp_source_identity": identity.to_dict(),
            "vipp_source_series_index": int(series_index),
            "vipp_source_snapshot_policy": FILE_SOURCE_SNAPSHOT_POLICY,
        },
        dataset.selected_series.name,
        snapshot_state,
        identity,
    )
    return SourceFileSnapshot(payload, dataset.inspection, identity)


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
    "SourceFileSnapshot",
    "VerifiedSourceInspection",
    "load_frozen_file_source_snapshot",
]
