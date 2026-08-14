"""Exact device-resident CuPyX filling of complete mask holes.

CuPy and CuPyX are imported only after this provider is selected.  The first
public implementation deliberately covers only ``max_hole_size=0``: every
enclosed background component is filled independently in each resolved
spatial block, with the authoritative SciPy connectivity semantics.
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

_MAXIMUM_PADDED_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
_PROGRESS_MESSAGE = "Fill-hole blocks"


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this accelerator provider is selected."""

    return importlib.import_module("cupy")


@cache
def _cupyx_ndimage_module() -> ModuleType:
    """Load the public CuPyX ndimage API only for an explicit GPU call."""

    return importlib.import_module("cupyx.scipy.ndimage")


def fill_holes(
    data,
    max_hole_size: int = 0,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Face connected",
    resolved_spatial_ndim: int | None = None,
    progress=None,
):
    """Fill every enclosed hole in resident boolean spatial blocks.

    This provider is intentionally narrower than the authoritative CPU
    operation.  Size-limited filling and numeric nonzero-mask coercion remain
    CPU fallbacks until their separate GPU contracts have durable evidence.

    One CuPyX fill is an indivisible lifecycle boundary.  Cancellation is
    therefore checked between complete leading blocks, and progress is only
    reported after the current CUDA stream proves that a block has finished.
    """

    if (
        isinstance(max_hole_size, (bool, np.bool_))
        or not isinstance(max_hole_size, (int, np.integer))
        or int(max_hole_size) != 0
    ):
        raise ValueError(
            "Fill Holes GPU execution currently requires max_hole_size=0 "
            "to fill every enclosed hole."
        )

    cupy = _cupy_module()
    cupyx_ndimage = _cupyx_ndimage_module()
    mask = cupy.asarray(data)
    if np.dtype(mask.dtype) != np.dtype(bool):
        raise ValueError(
            "Fill Holes GPU execution requires a boolean mask; numeric "
            "nonzero conversion remains on CPU."
        )

    spatial_ndim = resolve_spatial_ndim(
        int(mask.ndim),
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim not in {2, 3}:
        raise ValueError(
            "Fill Holes GPU execution requires a resolved 2D or 3D spatial rank."
        )

    connectivity_name = str(connectivity).strip().casefold()
    if connectivity_name not in {"face connected", "full connectivity"}:
        raise ValueError(
            "Connectivity must be 'Face connected' or 'Full connectivity'."
        )

    spatial_shape = tuple(int(size) for size in mask.shape[-spatial_ndim:])
    padded_block_elements = math.prod(size + 2 for size in spatial_shape)
    if padded_block_elements >= _MAXIMUM_PADDED_SPATIAL_BLOCK_ELEMENTS:
        raise ValueError(
            "Each Fill Holes GPU spatial block, including CuPyX's one-pixel "
            "boundary padding, must contain fewer than "
            f"{_MAXIMUM_PADDED_SPATIAL_BLOCK_ELEMENTS:,} elements so the "
            "reviewed int32 labeling path remains valid."
        )

    structure = connectivity_structure(spatial_ndim, connectivity)
    output = cupy.empty(mask.shape, dtype=bool)
    leading_shape = tuple(int(size) for size in mask.shape[:-spatial_ndim])
    indexes = (None,) if not leading_shape else np.ndindex(leading_shape)
    block_count = 1 if not leading_shape else math.prod(leading_shape)
    total = max(block_count, 1)

    if progress is not None:
        progress.check_cancelled()
        progress.report(0, total, _PROGRESS_MESSAGE)
    for completed, index in enumerate(indexes, start=1):
        if progress is not None:
            progress.check_cancelled()
        source_block = mask if index is None else mask[index]
        target_block = output if index is None else output[index]
        cupyx_ndimage.binary_fill_holes(
            source_block,
            structure=structure,
            output=target_block,
        )
        if progress is not None:
            cupy.cuda.get_current_stream().synchronize()
            progress.check_cancelled()
            progress.report(completed, total, _PROGRESS_MESSAGE)
    if block_count == 0 and progress is not None:
        progress.check_cancelled()
        progress.report(1, 1, _PROGRESS_MESSAGE)
    return output


__all__ = ["fill_holes"]
