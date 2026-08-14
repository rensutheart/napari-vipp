"""Exact device-resident small-component removal for boolean masks.

CuPy and CuPyX remain optional and are imported only when this provider is
selected.  The implementation mirrors VIPP's authoritative CPU semantics for
the promoted boolean-mask region: connectivity is evaluated independently in
each leading non-spatial block and components whose size is below ``min_size``
are cleared without changing pixel order or dtype.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np

from napari_vipp.core.connected_components import (
    connectivity_structure,
    resolve_spatial_ndim,
)

_MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2
_PROGRESS_MESSAGE = "Small-object blocks"


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this accelerator provider is selected."""

    return importlib.import_module("cupy")


@cache
def _cupyx_ndimage_module() -> ModuleType:
    """Load the public CuPyX ndimage API only for an explicit GPU call."""

    return importlib.import_module("cupyx.scipy.ndimage")


def remove_small_objects(
    data,
    min_size: int = 10,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Face connected",
    resolved_spatial_ndim: int | None = None,
    progress=None,
):
    """Remove undersized resident boolean components per spatial block.

    Integer label images deliberately remain outside this provider's contract:
    their IDs, rather than their connected pieces, define objects in VIPP's
    authoritative CPU implementation.  A CuPyX label call is an indivisible
    lifecycle boundary, so cancellation and progress are observed only after
    the complete label/count/gather operation for a block has synchronized.
    """

    cupy = _cupy_module()
    cupyx_ndimage = _cupyx_ndimage_module()
    array = cupy.asarray(data)
    if np.dtype(array.dtype) != np.dtype(bool):
        raise ValueError(
            "Remove Small Objects GPU execution requires a boolean mask; "
            "integer label images remain authoritative on CPU."
        )
    if isinstance(min_size, (bool, np.bool_)) or not isinstance(
        min_size,
        (Integral, np.integer),
    ):
        raise ValueError("Minimum object size must be an integer.")
    minimum = max(int(min_size), 0)

    spatial_ndim = resolve_spatial_ndim(
        int(array.ndim),
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim not in {2, 3}:
        raise ValueError(
            "Remove Small Objects GPU execution requires a resolved 2D or "
            "3D spatial rank."
        )
    connectivity_name = str(connectivity).strip().casefold()
    if connectivity_name not in {"face connected", "full connectivity"}:
        raise ValueError(
            "Connectivity must be 'Face connected' or 'Full connectivity'."
        )

    spatial_shape = tuple(int(size) for size in array.shape[-spatial_ndim:])
    block_elements = math.prod(spatial_shape)
    if block_elements >= _MAXIMUM_SPATIAL_BLOCK_ELEMENTS:
        raise ValueError(
            "Each Remove Small Objects GPU spatial block must contain fewer "
            f"than {_MAXIMUM_SPATIAL_BLOCK_ELEMENTS:,} elements so the exact "
            "CuPyX int32 component-label path remains valid."
        )

    leading_shape = tuple(int(size) for size in array.shape[:-spatial_ndim])
    block_count = 1 if not leading_shape else math.prod(leading_shape)
    removes_every_component = minimum > block_elements
    output = cupy.empty(array.shape, dtype=bool)
    labels = (
        cupy.empty(spatial_shape, dtype=cupy.int32)
        if (
            minimum > 1
            and not removes_every_component
            and block_elements > 0
            and block_count > 0
        )
        else None
    )
    structure = connectivity_structure(spatial_ndim, connectivity)
    indexes = (None,) if not leading_shape else np.ndindex(leading_shape)
    total = max(block_count, 1)

    if progress is not None:
        progress.check_cancelled()
        progress.report(0, total, _PROGRESS_MESSAGE)
    for completed, index in enumerate(indexes, start=1):
        if progress is not None:
            progress.check_cancelled()
        source_block = array if index is None else array[index]
        target_block = output if index is None else output[index]
        if removes_every_component:
            # No connected component can contain more elements than its
            # complete spatial block.  Besides avoiding needless work, this
            # keeps arbitrarily large authored Python integers away from
            # CuPy's fixed-width scalar comparison conversion.
            target_block[...] = False
        elif labels is None:
            target_block[...] = source_block
        else:
            cupyx_ndimage.label(
                source_block,
                structure=structure,
                output=labels,
            )
            sizes = cupy.bincount(labels.ravel())
            keep = sizes >= minimum
            keep[0] = False
            target_block[...] = keep[labels]
        if progress is not None:
            cupy.cuda.get_current_stream().synchronize()
            progress.check_cancelled()
            progress.report(completed, total, _PROGRESS_MESSAGE)
    if block_count == 0 and progress is not None:
        progress.check_cancelled()
        progress.report(1, 1, _PROGRESS_MESSAGE)
    return output


__all__ = ["remove_small_objects"]
