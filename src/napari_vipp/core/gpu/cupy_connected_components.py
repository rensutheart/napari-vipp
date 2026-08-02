"""Exact device-resident CuPyX connected-component labeling.

CuPy and CuPyX are imported only for an explicit provider call.  The public
implementation labels each resolved spatial block independently into a
preallocated ``int32`` output view, preserving SciPy's deterministic label IDs
and restarting those IDs at one for every leading block.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.connected_components import (
    connectivity_structure,
    resolve_spatial_ndim,
)

_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
_PROGRESS_MESSAGE = "Connected-component blocks"


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this accelerator provider is selected."""

    return importlib.import_module("cupy")


@cache
def _cupyx_ndimage_module() -> ModuleType:
    """Load the public CuPyX ndimage API only for an explicit GPU call."""

    return importlib.import_module("cupyx.scipy.ndimage")


def label_connected_components(
    data,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Full connectivity",
    resolved_spatial_ndim: int | None = None,
    progress=None,
):
    """Label resident foreground blocks with exact SciPy-compatible IDs.

    Numeric inputs are converted to VIPP's nonzero foreground mask on the
    device.  The promoted compute region supplies boolean masks, for which no
    conversion copy is needed.  CuPyX's public ``label`` primitive writes
    directly into each contiguous spatial view of the final device output.

    A CuPyX call is an indivisible lifecycle boundary.  Cancellation is
    therefore checked between complete leading blocks, and progress is only
    reported after the current CUDA stream proves that a block is finished.
    """

    cupy = _cupy_module()
    cupyx_ndimage = _cupyx_ndimage_module()
    array = cupy.asarray(data)
    mask = array if np.dtype(array.dtype) == np.dtype(bool) else array != 0
    spatial_ndim = resolve_spatial_ndim(
        int(mask.ndim),
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim not in {2, 3}:
        raise ValueError(
            "Connected-components GPU execution requires a resolved 2D or "
            "3D spatial rank."
        )
    connectivity_name = str(connectivity).strip().casefold()
    if connectivity_name not in {"face connected", "full connectivity"}:
        raise ValueError(
            "Connectivity must be 'Face connected' or 'Full connectivity'."
        )

    spatial_shape = tuple(int(size) for size in mask.shape[-spatial_ndim:])
    block_elements = math.prod(spatial_shape)
    if block_elements >= _MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        raise ValueError(
            "Each connected-components GPU spatial block must contain fewer "
            f"than {_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements so the exact "
            "CuPyX int32 label path remains valid."
        )

    structure = connectivity_structure(spatial_ndim, connectivity)
    output = cupy.empty(mask.shape, dtype=cupy.int32)
    leading_shape = tuple(int(size) for size in mask.shape[:-spatial_ndim])
    indexes = (None,) if not leading_shape else np.ndindex(leading_shape)
    block_count = 1 if not leading_shape else math.prod(leading_shape)
    total = max(block_count, 1)

    if progress is not None:
        progress.report(0, total, _PROGRESS_MESSAGE)
    for completed, index in enumerate(indexes, start=1):
        if progress is not None:
            progress.check_cancelled()
        source_block = mask if index is None else mask[index]
        target_block = output if index is None else output[index]
        cupyx_ndimage.label(
            source_block,
            structure=structure,
            output=target_block,
        )
        if progress is not None:
            cupy.cuda.get_current_stream().synchronize()
            progress.report(completed, total, _PROGRESS_MESSAGE)
    if block_count == 0 and progress is not None:
        progress.report(1, 1, _PROGRESS_MESSAGE)
    return output


__all__ = ["label_connected_components"]
