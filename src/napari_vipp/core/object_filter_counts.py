"""Exact, presentation-only counts for paired object-filter results.

The caller supplies an input and its corresponding completed output on the same
grid, with semantic spatial axes already placed last. No pixels are modified or
sampled. Leading dimensions are independent: label 7 at two time points counts
as two objects, not one globally shared object.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import prod
from numbers import Integral

import numpy as np
from scipy import ndimage as ndi

from napari_vipp.core.host_memory import (
    capture_host_memory,
    preflight_host_allocation,
)
from napari_vipp.core.progress import OperationCancelled

OBJECT_COUNT_CHUNK_ELEMENTS = 1_048_576


@dataclass(frozen=True, slots=True)
class ObjectFilterCounts:
    """Object totals aggregated across every independent spatial block."""

    input_count: int
    kept_count: int
    removed_count: int
    block_count: int


def object_filter_counts(
    input_data,
    output_data,
    *,
    spatial_ndim: int,
    connectivity: str = "Face connected",
    cancel_callback: Callable[[], bool] | None = None,
) -> ObjectFilterCounts:
    """Count input, retained, and removed objects without changing either array.

    Integer labels use positive distinct IDs, irrespective of connectivity or
    disconnected pieces sharing an ID. An ID is retained if any of its original
    pixels remains; every nonzero output pixel must preserve the input ID at
    that position. Negative labels and relabelled/new foreground are rejected.

    Boolean masks use connected components with the same Face/Full choices as
    ``diagnostics.object_sizes``. Every input component must be kept completely
    or removed completely; partial removal cannot be called an object-filter
    result and raises ``ValueError``. Clear Border's Boolean implementation uses
    full connectivity, so its caller must explicitly request that connectivity.

    Cancellation is checked before materialization, per block, and per bounded
    pixel chunk. SciPy's component-labelling call itself is not interruptible;
    cancellation is checked immediately after it returns.
    """
    _check_cancelled(cancel_callback)
    before = np.asarray(input_data)
    after = np.asarray(output_data)
    if before.shape != after.shape:
        raise ValueError("Object-filter counts require identical input/output shapes.")
    if (
        isinstance(spatial_ndim, bool)
        or not isinstance(spatial_ndim, Integral)
        or not 1 <= spatial_ndim <= before.ndim
    ):
        raise ValueError(
            "Spatial dimensionality must be an integer between 1 and the array rank."
        )
    spatial_ndim = int(spatial_ndim)
    for values in (before, after):
        if values.dtype.kind not in {"b", "i", "u"}:
            raise TypeError("Object-filter counts require Boolean or integer data.")
    is_mask = before.dtype.kind == "b"
    if is_mask != (after.dtype.kind == "b"):
        raise TypeError(
            "Input and output must both be Boolean masks or integer labels."
        )

    structure = None
    if is_mask:
        normalized = str(connectivity).strip().casefold()
        if normalized == "face connected":
            rank = 1
        elif normalized in {"full connectivity", "fully connected"}:
            rank = spatial_ndim
        else:
            raise ValueError(
                "Connectivity must be 'Face connected' or 'Full connectivity'."
            )
        structure = ndi.generate_binary_structure(spatial_ndim, rank)

    leading_shape = before.shape[:-spatial_ndim]
    block_count = prod(leading_shape) if leading_shape else 1
    input_count = kept_count = 0
    for index in np.ndindex(leading_shape):
        _check_cancelled(cancel_callback)
        before_block = before[index]
        after_block = after[index]
        if is_mask:
            counts = _mask_counts(before_block, after_block, structure, cancel_callback)
        else:
            counts = _label_counts(before_block, after_block, cancel_callback)
        input_count += counts[0]
        kept_count += counts[1]
        _check_cancelled(cancel_callback)
    return ObjectFilterCounts(
        input_count=input_count,
        kept_count=kept_count,
        removed_count=input_count - kept_count,
        block_count=block_count,
    )


def _paired_chunks(
    before: np.ndarray, after: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    # nditer preserves matched C-order positions even for transposed/reversed
    # views, without flattening a complete spatial block into another buffer.
    with np.nditer(
        [before, after],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        order="C",
        buffersize=OBJECT_COUNT_CHUNK_ELEMENTS,
    ) as iterator:
        yield from iterator


def _label_counts(before, after, cancel_callback):
    if not before.size:
        return 0, 0
    _guard_working_memory(min(before.size, OBJECT_COUNT_CHUNK_ELEMENTS) * 64)
    input_ids: set[int] = set()
    kept_ids: set[int] = set()
    for input_chunk, output_chunk in _paired_chunks(before, after):
        _check_cancelled(cancel_callback)
        if np.any(input_chunk < 0) or np.any(output_chunk < 0):
            raise ValueError(
                "Object-filter counts require non-negative integer labels."
            )
        # Comparing mixed uint64/int64 directly can promote both to float64 and
        # merge different IDs above 2**53. Nonnegative integers fit uint64 exactly.
        input_values = input_chunk.astype(np.uint64, copy=False)
        output_values = output_chunk.astype(np.uint64, copy=False)
        if np.any((output_values != 0) & (output_values != input_values)):
            raise ValueError(
                "Output contains new foreground or changed label IDs; "
                "object-filter counts require unchanged retained IDs."
            )
        unique_input = np.unique(input_values[input_values > 0])
        unique_output = np.unique(output_values[output_values > 0])
        # Python int/set storage is data dependent. Check a conservative bound
        # before adding this chunk (including existing-set resize workspace).
        _guard_working_memory(
            (len(input_ids) + len(kept_ids)) * 64
            + (unique_input.size + unique_output.size) * 192
        )
        input_ids.update(int(value) for value in unique_input)
        kept_ids.update(int(value) for value in unique_output)
    return len(input_ids), len(kept_ids)


def _mask_counts(before, after, structure, cancel_callback):
    if not before.size:
        return 0, 0
    # Component labels, the labeller's equivalence workspace, membership flags,
    # and bounded per-chunk temporaries. Inputs are already resident/read-only.
    _guard_working_memory(
        before.size * 32 + min(before.size, OBJECT_COUNT_CHUNK_ELEMENTS) * 32
    )
    component_labels, component_count = ndi.label(before, structure=structure)
    _check_cancelled(cancel_callback)
    kept = np.zeros(component_count + 1, dtype=bool)
    removed = np.zeros(component_count + 1, dtype=bool)
    for labels, output_mask in _paired_chunks(component_labels, after):
        _check_cancelled(cancel_callback)
        if np.any(output_mask & (labels == 0)):
            raise ValueError(
                "Output contains new foreground; object-filter counts require "
                "a subset of the input mask."
            )
        kept[labels[output_mask]] = True
        removed[labels[~output_mask]] = True
    if np.any(kept[1:] & removed[1:]):
        raise ValueError(
            "Output partially removes an input component; exact kept/removed "
            "object counts require whole-component filtering."
        )
    return int(component_count), int(np.count_nonzero(kept[1:]))


def _guard_working_memory(required_bytes: int):
    decision = preflight_host_allocation(
        capture_host_memory(),
        required_bytes=int(required_bytes),
        purpose="exact object-filter counts",
    )
    if not decision.allowed:
        raise MemoryError(decision.reason)


def _check_cancelled(callback: Callable[[], bool] | None):
    if callback is not None and callback():
        raise OperationCancelled("Object-filter counts cancelled.")


__all__ = ["ObjectFilterCounts", "object_filter_counts"]
