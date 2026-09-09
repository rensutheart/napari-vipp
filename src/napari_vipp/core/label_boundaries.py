"""Exact, CPU label-boundary masks on independent 2D/3D spatial blocks.

The reference algorithm is ``skimage.segmentation.find_boundaries`` with
background fixed at zero and modes ``inner``, ``outer``, and ``thick``. Only
observed label transitions are outlined: the image is not padded with an
invented exterior background. In particular, a spatial block filled by one
label has no boundary. Results are voxel masks, not meshes or subpixel contours;
physical scale does not affect this discrete neighborhood operation.

Reference: https://scikit-image.org/docs/stable/api/skimage.segmentation.html
    #skimage.segmentation.find_boundaries
"""

from __future__ import annotations

import math

import numpy as np
from skimage import segmentation

from napari_vipp.core.connected_components import resolve_spatial_ndim

_BOUNDARY_MODES = {
    "Inside objects": "inner",
    "Outside objects": "outer",
    "Both sides": "thick",
}
_CONNECTIVITY_RANKS = {"Face connected": 1, "Full connectivity": None}
_MAX_EXACT_BACKEND_LABEL = 2**53 - 1


def find_label_boundaries(
    data,
    boundary_placement: str = "Inside objects",
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Face connected",
    resolved_spatial_ndim: int | None = None,
    progress=None,
) -> np.ndarray:
    """Return a new same-shape Boolean mask of transitions between label IDs.

    Accept Boolean masks or nonnegative integer IDs, including full-range
    uint64 values. Zero is background; distinct positive IDs remain distinct,
    even when touching. ``Outside objects`` follows scikit-image's ``outer``
    semantics, including marking interfaces between touching positive labels.

    Spatial blocks occupy trailing YX or ZYX positions; every leading position
    is an independent batch. Higher-rank Auto mode requires the caller to pass
    an axis-resolved spatial rank. No axis inference from dimension lengths,
    exterior padding, resampling, or physical-distance calculation is performed.
    Read-only/non-contiguous inputs are supported and never modified. Progress
    and cooperative cancellation are checked between complete spatial blocks,
    including immediately after each blocking reference call.
    """
    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("Find Label Boundaries requires at least a 2D array.")
    if array.dtype.kind not in "bui":
        raise ValueError(
            "Find Label Boundaries requires a Boolean mask or nonnegative "
            "integer label IDs; floating-point and other dtypes are not supported."
        )
    if (
        not isinstance(boundary_placement, str)
        or boundary_placement not in _BOUNDARY_MODES
    ):
        raise ValueError(
            "Boundary placement must be Inside objects, Outside objects, or Both sides."
        )
    if not isinstance(connectivity, str) or connectivity not in _CONNECTIVITY_RANKS:
        raise ValueError("Connectivity must be Face connected or Full connectivity.")
    spatial_ndim = resolve_spatial_ndim(array.ndim, spatial_mode, resolved_spatial_ndim)
    if spatial_ndim not in {2, 3}:
        raise ValueError("Find Label Boundaries requires 2D or 3D spatial processing.")
    rank = _CONNECTIVITY_RANKS[connectivity] or spatial_ndim
    leading_shape = array.shape[:-spatial_ndim]
    block_count = math.prod(leading_shape)
    total = max(block_count, 1)
    message = "Label-boundary blocks"
    if progress is not None:
        progress.check_cancelled()
        progress.report(0, total, message)

    output = np.empty(array.shape, dtype=bool)
    indexes = (None,) if not leading_shape else np.ndindex(leading_shape)
    for completed, index in enumerate(indexes, start=1):
        if progress is not None:
            progress.check_cancelled()
        source = array if index is None else array[index]
        target = output if index is None else output[index]
        if source.size:
            if source.dtype.kind == "i" and source.min() < 0:
                raise ValueError("Label IDs must be nonnegative; zero is background.")
            backend_labels = _exact_backend_labels(source)
            if progress is not None:
                progress.check_cancelled()
            target[...] = segmentation.find_boundaries(
                backend_labels,
                connectivity=rank,
                mode=_BOUNDARY_MODES[boundary_placement],
                background=0,
            )
        # Empty spatial blocks have no observed transitions and require no
        # backend call. Their empty target already has the public shape/dtype.
        if progress is not None:
            progress.check_cancelled()
            progress.report(completed, total, message)
            progress.check_cancelled()
    if block_count == 0 and progress is not None:
        progress.check_cancelled()
        progress.report(1, 1, message)
        progress.check_cancelled()
    return output


def _exact_backend_labels(source: np.ndarray) -> np.ndarray:
    """Protect wide IDs from SciPy morphology's intermediate double values.

    Sorting/unique comparisons operate on the original integer dtype. This
    one-to-one recoding changes neither ID equality nor zero's background role;
    it is private to the boundary calculation and never replaces input IDs.
    """
    if source.dtype.itemsize <= 4:
        return source
    values, inverse = np.unique(source, return_inverse=True)
    has_background = values[0] == 0
    largest_dense_id = values.size - int(has_background)
    if largest_dense_id > _MAX_EXACT_BACKEND_LABEL:
        raise OverflowError(
            "Too many distinct labels in one spatial block for exact boundary "
            "calculation."
        )
    if not has_background:
        inverse += 1
    dtype = np.uint32 if largest_dense_id <= np.iinfo(np.uint32).max else np.uint64
    return inverse.reshape(source.shape).astype(dtype, copy=False)


__all__ = ["find_label_boundaries"]
