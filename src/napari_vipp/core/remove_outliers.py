"""ImageJ-compatible one-polarity outlier cleanup for binary masks.

The footprint construction follows ``RankFilters.makeLineRadii`` from ImageJ
1.x, including its two historical radius-compatibility intervals.  ImageJ's
general Remove Outliers command replaces a selected bright or dark pixel with
the neighbourhood median when the difference exceeds a threshold.  On a
canonical binary mask, choosing any threshold below the full binary contrast
reduces exactly to the one-polarity majority operation implemented here.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy import ndimage as ndi

from napari_vipp.core.progress import ProgressContext

FOREGROUND_OUTLIERS = "Foreground (remove)"
BACKGROUND_OUTLIERS = "Background (fill)"
OUTLIER_CHOICES = (FOREGROUND_OUTLIERS, BACKGROUND_OUTLIERS)

_MINIMUM_RADIUS = 0.5
_MAXIMUM_RADIUS = 100.0
_IMAGEJ_SQRT_EPSILON = 1e-10
_PROGRESS_MESSAGE = "Remove Outliers (Binary) YX planes"
_VALIDATION_CHUNK_VALUES = 1_048_576
_DENSE_CORRELATE_MAX_POINTS = 225


def imagej_remove_outliers_footprint(radius: float = 2.0) -> np.ndarray:
    """Return ImageJ's circular RankFilters footprint for ``radius``.

    This is deliberately not a conventional Euclidean disk.  ImageJ first
    truncates ``radius**2``, adds one, and retains two historical radius
    adjustments so older RankFilters versions produce the same masks.
    """

    resolved_radius = _validated_radius(radius)
    if 1.5 <= resolved_radius < 1.75:
        resolved_radius = 1.75
    elif 2.5 <= resolved_radius < 2.85:
        resolved_radius = 2.85

    radius_squared = int(resolved_radius * resolved_radius) + 1
    kernel_radius = int(math.sqrt(radius_squared + _IMAGEJ_SQRT_EPSILON))
    footprint = np.zeros(
        (2 * kernel_radius + 1, 2 * kernel_radius + 1),
        dtype=bool,
    )
    center = kernel_radius
    footprint[center, :] = True
    for y_offset in range(1, kernel_radius + 1):
        x_radius = int(
            math.sqrt(radius_squared - y_offset * y_offset + _IMAGEJ_SQRT_EPSILON)
        )
        footprint[center - y_offset, center - x_radius : center + x_radius + 1] = True
        footprint[center + y_offset, center - x_radius : center + x_radius + 1] = True
    return footprint


def remove_binary_outliers(
    data,
    radius: float = 2.0,
    which_outliers: str = FOREGROUND_OUTLIERS,
    progress: ProgressContext | None = None,
) -> np.ndarray:
    """Remove foreground specks or fill background notches in YX mask planes.

    The last two axes are processed as ImageJ YX planes.  Every leading index
    is independent, neighbourhoods use nearest-edge extension, and all pixels
    in a plane are decided from the unchanged input plane.  Inputs must be
    boolean or canonical uint8 binary data encoded as 0/1 or 0/255.  The result
    is always a newly allocated boolean array.
    """

    if progress is not None:
        progress.check_cancelled()
    mask = _validated_binary_mask(data, progress=progress)
    if mask.ndim < 2:
        raise ValueError(
            "Remove Outliers (Binary) requires at least two dimensions with "
            "trailing YX axes."
        )
    polarity = _validated_outlier_choice(which_outliers)
    footprint = imagej_remove_outliers_footprint(radius)
    footprint_point_count = int(np.count_nonzero(footprint))
    majority_threshold = footprint_point_count // 2

    leading_shape = mask.shape[:-2]
    plane_count = int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1
    result = np.empty(mask.shape, dtype=bool)
    if progress is not None:
        progress.report(0, plane_count, _PROGRESS_MESSAGE)

    remove_foreground = polarity == FOREGROUND_OUTLIERS
    leading_indexes = np.ndindex(leading_shape) if leading_shape else ((),)
    for plane_index, leading_index in enumerate(leading_indexes):
        if progress is not None:
            progress.check_cancelled()
        plane = mask[leading_index]
        foreground_counts = _foreground_counts(
            plane,
            footprint,
            footprint_point_count=footprint_point_count,
            progress=progress,
        )
        local_median = foreground_counts > majority_threshold
        if remove_foreground:
            np.logical_and(plane, local_median, out=result[leading_index])
        else:
            np.logical_or(plane, local_median, out=result[leading_index])
        del foreground_counts, local_median
        if progress is not None:
            progress.report(plane_index + 1, plane_count, _PROGRESS_MESSAGE)
    return result


def _foreground_counts(
    plane: np.ndarray,
    footprint: np.ndarray,
    *,
    footprint_point_count: int,
    progress: ProgressContext | None,
) -> np.ndarray:
    """Count foreground under one exact footprint with nearest-edge extension."""

    if footprint_point_count <= _DENSE_CORRELATE_MAX_POINTS:
        counts = ndi.correlate(
            plane,
            footprint.astype(np.int32, copy=False),
            output=np.int32,
            mode="nearest",
            origin=0,
        )
        if progress is not None:
            progress.check_cancelled()
        return counts
    return _row_span_foreground_counts(plane, footprint, progress=progress)


def _row_span_foreground_counts(
    plane: np.ndarray,
    footprint: np.ndarray,
    *,
    progress: ProgressContext | None,
) -> np.ndarray:
    """Count a large ImageJ footprint without a dense correlation.

    Every ImageJ footprint row is one contiguous horizontal span. Rows with
    the same half-width share a horizontal prefix-sum calculation; their
    vertical offsets are then accumulated with nearest-edge replication.
    Integer counts make this path bitwise identical to dense correlation.
    """

    if progress is not None:
        progress.check_cancelled()
    height, width = plane.shape
    prefix = np.empty((height, width + 1), dtype=np.int32)
    prefix[:, 0] = 0
    np.cumsum(plane, axis=1, dtype=np.int32, out=prefix[:, 1:])
    counts = np.zeros((height, width), dtype=np.int32)
    horizontal = np.empty_like(counts)
    positions = np.arange(width, dtype=np.int64)

    for x_radius, y_offsets in _grouped_row_spans(footprint):
        if progress is not None:
            progress.check_cancelled()
        left = np.maximum(positions - x_radius, 0)
        right = np.minimum(positions + x_radius, width - 1)
        np.take(prefix, right + 1, axis=1, out=horizontal)
        horizontal -= np.take(prefix, left, axis=1)

        left_edge_width = min(x_radius, width)
        if left_edge_width:
            left_repeats = (x_radius - positions[:left_edge_width]).astype(
                np.int32, copy=False
            )
            horizontal[:, :left_edge_width] += plane[:, :1] * left_repeats
        right_edge_start = max(width - x_radius, 0)
        if right_edge_start < width:
            right_repeats = (
                positions[right_edge_start:] + x_radius - (width - 1)
            ).astype(np.int32, copy=False)
            horizontal[:, right_edge_start:] += plane[:, -1:] * right_repeats

        for y_offset in y_offsets:
            if progress is not None:
                progress.check_cancelled()
            _add_nearest_y_shift(counts, horizontal, y_offset)
    if progress is not None:
        progress.check_cancelled()
    return counts


def _grouped_row_spans(
    footprint: np.ndarray,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Group ImageJ footprint Y offsets by their horizontal half-width."""

    center_y = footprint.shape[0] // 2
    grouped: dict[int, list[int]] = {}
    for row_index, row in enumerate(footprint):
        columns = np.flatnonzero(row)
        x_radius = int((columns[-1] - columns[0]) // 2)
        grouped.setdefault(x_radius, []).append(row_index - center_y)
    return tuple(
        (x_radius, tuple(y_offsets)) for x_radius, y_offsets in sorted(grouped.items())
    )


def _add_nearest_y_shift(
    counts: np.ndarray,
    horizontal: np.ndarray,
    y_offset: int,
) -> None:
    """Accumulate one vertically shifted row sum using nearest-edge extension."""

    height = counts.shape[0]
    if y_offset == 0:
        counts += horizontal
        return
    if y_offset > 0:
        if y_offset >= height:
            counts += horizontal[-1]
            return
        counts[:-y_offset] += horizontal[y_offset:]
        counts[-y_offset:] += horizontal[-1]
        return

    magnitude = -y_offset
    if magnitude >= height:
        counts += horizontal[0]
        return
    counts[magnitude:] += horizontal[:-magnitude]
    counts[:magnitude] += horizontal[0]


def _validated_radius(radius: float) -> float:
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real):
        raise ValueError("Remove Outliers (Binary) radius must be a finite number.")
    resolved = float(radius)
    if not math.isfinite(resolved):
        raise ValueError("Remove Outliers (Binary) radius must be finite.")
    if not _MINIMUM_RADIUS <= resolved <= _MAXIMUM_RADIUS:
        raise ValueError(
            "Remove Outliers (Binary) radius must be between 0.5 and 100 pixels."
        )
    return resolved


def _validated_outlier_choice(which_outliers: str) -> str:
    choice = str(which_outliers).strip()
    if choice not in OUTLIER_CHOICES:
        expected = " or ".join(repr(value) for value in OUTLIER_CHOICES)
        raise ValueError(f"Remove Outliers (Binary) outlier type must be {expected}.")
    return choice


def _validated_binary_mask(
    data,
    *,
    progress: ProgressContext | None = None,
) -> np.ndarray:
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("Remove Outliers (Binary) does not accept empty masks.")
    if arr.dtype == np.dtype(bool):
        return arr
    if arr.dtype != np.dtype(np.uint8):
        raise ValueError(
            "Remove Outliers (Binary) requires bool or canonical uint8 mask "
            "data; grayscale images, floating-point arrays, and integer labels "
            "must be converted to a binary mask first."
        )
    populated_levels: set[int] = set()
    chunks = np.nditer(
        arr,
        flags=("external_loop", "buffered"),
        op_flags=("readonly",),
        order="K",
        buffersize=_VALIDATION_CHUNK_VALUES,
    )
    for raw_chunk in chunks:
        if progress is not None:
            progress.check_cancelled()
        chunk = np.asarray(raw_chunk)
        populated_levels.update(
            np.flatnonzero(np.bincount(chunk, minlength=256)).tolist()
        )
        if not (populated_levels <= {0, 1} or populated_levels <= {0, 255}):
            raise ValueError(
                "Remove Outliers (Binary) uint8 input must use one canonical "
                "binary encoding: 0/1 or 0/255, with no intermediate levels."
            )
    mask = np.empty(arr.shape, dtype=bool)
    np.not_equal(arr, 0, out=mask)
    return mask


__all__ = [
    "BACKGROUND_OUTLIERS",
    "FOREGROUND_OUTLIERS",
    "OUTLIER_CHOICES",
    "imagej_remove_outliers_footprint",
    "remove_binary_outliers",
]
