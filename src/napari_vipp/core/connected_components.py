"""Authoritative connected-components contract shared by CPU and GPU paths.

The CPU implementation defines VIPP's scientific semantics: nonzero values are
foreground, connectivity follows SciPy's generated binary structures, labels
are assigned independently in every leading non-spatial block, and each block
restarts deterministic ``int32`` IDs at one.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy import ndimage as ndi


def label_connected_components(
    data,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Full connectivity",
    resolved_spatial_ndim: int | None = None,
    progress=None,
) -> np.ndarray:
    """Assign deterministic SciPy-compatible IDs to foreground components.

    Leading dimensions are batches, not connected dimensions.  Consequently,
    component IDs restart at one in every independent spatial block.  Optional
    progress is reported only after a complete block has finished.
    """

    array = np.asarray(data)
    mask = array if array.dtype == bool else array != 0
    spatial_ndim = resolve_spatial_ndim(
        mask.ndim,
        spatial_mode,
        resolved_spatial_ndim,
    )
    structure = connectivity_structure(spatial_ndim, connectivity)
    output = np.empty(mask.shape, dtype=np.int32)
    leading_shape = mask.shape[: max(mask.ndim - spatial_ndim, 0)]
    indexes = (None,) if not leading_shape else np.ndindex(leading_shape)
    block_count = (
        1 if not leading_shape else int(np.prod(leading_shape, dtype=np.int64))
    )
    total = max(block_count, 1)
    message = "Connected-component blocks"

    if progress is not None:
        progress.report(0, total, message)
    for completed, index in enumerate(indexes, start=1):
        if progress is not None:
            progress.check_cancelled()
        source_block = mask if index is None else mask[index]
        # Request the public dtype from SciPy itself. SciPy then raises instead
        # of allowing a silent narrowing wrap if a block ever contains more
        # components than int32 can represent.
        labels, _count = ndi.label(
            source_block,
            structure=structure,
            output=np.int32,
        )
        target_block = output if index is None else output[index]
        target_block[...] = labels.astype(np.int32, copy=False)
        if progress is not None:
            progress.report(completed, total, message)
    if block_count == 0 and progress is not None:
        progress.report(1, 1, message)
    return output


def resolve_spatial_ndim(
    array_ndim: int,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
) -> int:
    """Resolve VIPP's authored spatial mode without inspecting array values."""

    ndim = int(array_ndim)
    if ndim < 0:
        raise ValueError("Array rank must not be negative.")
    mode = str(spatial_mode).strip().casefold()
    dimensions = {
        "auto from axes": None,
        "2d yx": 2,
        "2d per xy slice (advanced)": 2,
        "3d zyx": 3,
        "3d zyx volume": 3,
    }
    if mode not in dimensions:
        raise ValueError(
            "Spatial mode must be Auto from axes, 2D YX, "
            "2D per XY slice (advanced), 3D ZYX, or 3D ZYX volume."
        )
    requested = dimensions[mode]
    if requested is None and resolved_spatial_ndim is not None:
        if isinstance(resolved_spatial_ndim, (bool, np.bool_)) or not isinstance(
            resolved_spatial_ndim,
            (Integral, np.integer),
        ):
            raise ValueError("resolved_spatial_ndim must be an integer from 1 to 3.")
        requested = int(resolved_spatial_ndim)
        if requested not in {1, 2, 3}:
            raise ValueError("resolved_spatial_ndim must be an integer from 1 to 3.")
    if requested is None:
        if ndim > 2:
            raise ValueError(
                "Auto from axes requires explicit axis semantics. Supply "
                "resolved_spatial_ndim or select an explicit 2D/3D mode."
            )
        requested = max(ndim, 1)
    if requested > max(ndim, 1):
        raise ValueError(
            f"{requested}D spatial processing cannot be applied to a {ndim}D array."
        )
    return requested


def connectivity_structure(spatial_ndim: int, connectivity: str) -> np.ndarray:
    """Return the exact SciPy structure used by the authoritative CPU path."""

    rank = 1 if str(connectivity).lower().startswith("face") else spatial_ndim
    return ndi.generate_binary_structure(spatial_ndim, rank)


__all__ = [
    "connectivity_structure",
    "label_connected_components",
    "resolve_spatial_ndim",
]
