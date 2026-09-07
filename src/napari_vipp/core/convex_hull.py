"""Binary convex hulls with bounded rasterization and degenerate 3D support.

The hull uses the same half-pixel, axis-aligned offsets as scikit-image's
``convex_hull_image(offset_coordinates=True, include_borders=True)``. These
are edge midpoints (a diamond/octahedron), not the corners of a pixel cube.
Unlike its preliminary 3D Qhull pass, flat, line and point inputs are valid.
No random perturbation (QJ), downsampling or approximate hull is used.

Reference: https://scikit-image.org/docs/stable/api/skimage.morphology.html
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from napari_vipp.core.progress import ProgressContext

_RASTER_CHUNK_SIZE = 65_536
_HULL_TOLERANCE = 1e-10


def convex_hull_block(
    mask: np.ndarray, *, progress: ProgressContext | None = None
) -> np.ndarray:
    """Return a new hull mask for one Boolean YX plane or ZYX volume."""
    if progress is not None:
        progress.report(0, 100, "Finding convex hull boundary")
    result = np.zeros(mask.shape, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        if progress is not None:
            progress.report(100, 100, "Empty foreground; hull is empty")
        return result

    # Only the first and last foreground voxel in each X row can be extreme:
    # every other foreground voxel is on the segment joining these endpoints.
    # This avoids a coordinate array proportional to the entire foreground.
    rows = np.nonzero(np.any(mask, axis=-1))
    first = np.argmax(mask, axis=-1)[rows]
    last = mask.shape[-1] - 1 - np.argmax(mask[..., ::-1], axis=-1)[rows]
    starts = np.column_stack((*rows, first))
    ends = np.column_stack((*rows, last))
    coords = np.concatenate((starts, ends[first != last])).astype(np.float64)
    lower = coords.min(axis=0).astype(np.intp)
    upper = coords.max(axis=0).astype(np.intp) + 1
    # Local coordinates keep Qhull and its inequality tolerance independent
    # of the object's location within a much larger source image.
    coords -= lower
    if progress is not None:
        progress.report(5, 100, "Building convex hull")
    try:
        vertices = _extreme_vertices(coords)
        ndim = mask.ndim
        offsets = np.concatenate((np.eye(ndim), -np.eye(ndim))) * 0.5
        points = np.unique((vertices[:, None, :] + offsets).reshape(-1, ndim), axis=0)
        equations = ConvexHull(points).equations
    except QhullError as exc:
        # Never report a failed geometric calculation as an empty segmentation.
        raise ValueError(
            "Convex Hull could not resolve this mask's geometry. "
            "No hull was produced; inspect the input mask."
        ) from exc
    if progress is not None:
        progress.report(10, 100, "Filling convex hull")

    shape = tuple(upper - lower)
    count = int(np.prod(shape, dtype=np.int64))
    for start in range(0, count, _RASTER_CHUNK_SIZE):
        stop = min(start + _RASTER_CHUNK_SIZE, count)
        indices = np.unravel_index(np.arange(start, stop), shape)
        grid = np.asarray(indices, dtype=np.float64)
        inside = np.ones(stop - start, dtype=bool)
        # One facet at a time: never allocate a voxels-by-facets matrix or
        # a full-volume coordinate grid.
        for index, equation in enumerate(equations):
            if progress is not None and index % 32 == 0:
                progress.check_cancelled()
            inside &= equation[:-1] @ grid + equation[-1] < _HULL_TOLERANCE
        result[
            tuple(axis + offset for axis, offset in zip(indices, lower, strict=True))
        ] = inside
        if progress is not None:
            progress.report(10 + 90 * stop // count, 100, "Filling convex hull")
    # A hull must be extensive. Catch a numerical failure rather than silently
    # returning a plausible-looking segmentation with missing source voxels.
    if np.any(mask & ~result):
        raise ValueError("Convex Hull lost foreground voxels due to numerical error.")
    return result


def _extreme_vertices(coords: np.ndarray) -> np.ndarray:
    """Reduce boundary points before adding extent, including lower-rank sets."""
    if len(coords) == 1:
        return coords
    try:
        return coords[ConvexHull(coords).vertices]
    except QhullError:
        # Qhull requires full-dimensional points. Project only to the actual
        # affine span to find its extreme vertices, then retain their original
        # integer-grid coordinates. Offsetting these gives a full-rank hull.
        centered = coords - coords[0]
        _, singular, basis = np.linalg.svd(centered, full_matrices=False)
        tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        if rank == coords.shape[1]:
            raise
        projected = centered @ basis[:rank].T
        if rank == 1:
            return coords[[projected[:, 0].argmin(), projected[:, 0].argmax()]]
        return coords[ConvexHull(projected).vertices]
