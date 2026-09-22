"""Same-grid, full-population image QC without registration or normalization.

SSIM uses scikit-image's uniform-window/sample-covariance definition. Only
centres with a complete valid window count. Slabs bound temporary SSIM arrays.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from scipy.ndimage import minimum_filter
from skimage.metrics import structural_similarity

from napari_vipp.core.grid import validate_aligned_image_states
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.tables import TableData

_SSIM_SLAB_VALUES = 1048576


def _moments(a, b, valid, progress):
    count = 0
    mean_a = mean_b = m2_a = m2_b = cross = squared = 0.0
    # nditer buffers preserve arbitrary layouts without copying full volumes.
    iterator = np.nditer(
        [a, b, valid],
        flags=["external_loop", "buffered"],
        op_flags=[["readonly"]] * 3,
        buffersize=262144,
        order="C",
    )
    for raw_a, raw_b, selected in iterator:
        if progress is not None:
            progress.check_cancelled()
        av, bv = raw_a[selected].astype(np.float64), raw_b[selected].astype(np.float64)
        n = len(av)
        if not n:
            continue
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError(
                "Compare Images requires finite values inside the selected coverage."
            )
        for raw in (raw_a[selected], raw_b[selected]):
            if raw.dtype.kind in "iu" and (
                int(raw.min()) < -(2**53) or int(raw.max()) > 2**53
            ):
                raise ValueError(
                    "Compare Images cannot exactly represent these wide integer "
                    "intensities. Explicitly convert the comparison images first."
                )
        am, bm = float(av.mean()), float(bv.mean())
        da, db = am - mean_a, bm - mean_b
        factor = count * n / (count + n)
        centered_a, centered_b = av - am, bv - bm
        m2_a += float(centered_a @ centered_a) + da * da * factor
        m2_b += float(centered_b @ centered_b) + db * db * factor
        cross += float(centered_a @ centered_b) + da * db * factor
        squared += float((av - bv) @ (av - bv))
        mean_a += da * n / (count + n)
        mean_b += db * n / (count + n)
        count += n
    if not count:
        raise ValueError("Compare Images has no valid overlapping pixels/voxels.")
    if not np.isfinite([squared, m2_a, m2_b, cross]).all():
        raise ValueError(
            "Comparison overflowed: use explicitly scaled finite input images."
        )
    correlation = (
        float(np.clip(cross / (math.sqrt(m2_a) * math.sqrt(m2_b)), -1, 1))
        if m2_a > 0 and m2_b > 0
        else None
    )
    return count, math.sqrt(squared / count), correlation


def _ssim(a, b, valid, data_range, window, progress):
    if min(a.shape) < window:
        return None, 0
    radius = window // 2
    centers = minimum_filter(
        valid.astype(np.uint8), size=window, mode="constant", cval=0
    ).astype(bool)
    count = int(centers.sum())
    if not count:
        return None, 0
    slab = max(window, _SSIM_SLAB_VALUES // math.prod(a.shape[1:]))
    total = 0.0
    for start in range(0, a.shape[0], slab):
        if progress is not None:
            progress.check_cancelled()
        end = min(start + slab, a.shape[0])
        selected = centers[start:end]
        # A final short slab can contain only excluded image-border centres.
        # Avoid invoking SSIM on a slab smaller than its window in that case.
        if not np.any(selected):
            continue
        lo, hi = max(0, start - radius), min(a.shape[0], end + radius)
        # Outside coverage cannot enter any accepted window. Replace non-finite
        # excluded values only in these temporary buffers to avoid NaN spreading.
        first = np.where(valid[lo:hi], a[lo:hi], 0).astype(np.float64)
        second = np.where(valid[lo:hi], b[lo:hi], 0).astype(np.float64)
        _mean, score = structural_similarity(
            first, second, data_range=data_range, win_size=window, full=True
        )
        scores = score[start - lo : end - lo][selected]
        if not np.isfinite(scores).all():
            raise ValueError(
                "SSIM overflowed: use explicitly scaled comparison images "
                "and intensity range."
            )
        total += float(np.sum(scores, dtype=np.float64))
    return total / count, count


def compare_images(
    inputs,
    *,
    input_states,
    data_range=1.0,
    window_size=7,
    use_mask=False,
    progress=None,
):
    """Return one QC row per explicit non-spatial block (T/C, never mixed)."""
    if not isinstance(use_mask, bool) or len(inputs) != (3 if use_mask else 2):
        raise ValueError(
            "Connect Reference and Comparison, plus Valid coverage when enabled."
        )
    if (
        isinstance(data_range, bool)
        or not isinstance(data_range, Real)
        or not np.isfinite(data_range)
        or data_range <= 0
    ):
        raise ValueError(
            "Comparison intensity range must be explicitly positive and finite."
        )
    if (
        isinstance(window_size, bool)
        or not isinstance(window_size, Integral)
        or window_size < 3
        or window_size > 101
        or window_size % 2 != 1
    ):
        raise ValueError("SSIM window size must be odd, between 3 and 101.")
    arrays = [np.asarray(value) for value in inputs]
    for array, state in zip(arrays, input_states, strict=True):
        if (
            not isinstance(state, ImageState)
            or tuple(array.shape) != state.shape
            or not state.axes_explicit
            or array.dtype.name != state.dtype
        ):
            raise ValueError(
                "Compare Images requires explicit axes and matching carried "
                "shapes/dtypes for all inputs."
            )
        if array.dtype.kind not in "biuf" or not array.size:
            raise ValueError("Compare Images requires nonempty real numeric inputs.")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError(
            "Compare Images requires equal shapes; "
            "resample explicitly with Apply Transform first."
        )
    validate_aligned_image_states(tuple(input_states), operation_title="Compare Images")
    state = input_states[0]
    names = tuple(axis.name.lower() for axis in state.axes)
    if len(set(names)) != len(names):
        raise ValueError("Compare Images requires unique axis names.")
    for axis in state.axes:
        expected_type = (
            "time"
            if axis.name.lower() == "t"
            else "channel"
            if axis.name.lower() == "c"
            else "space"
        )
        if axis.type != expected_type:
            raise ValueError("Compare Images axis names and semantic types disagree.")
    spatial = tuple(i for i, axis in enumerate(state.axes) if axis.type == "space")
    if len(spatial) not in (2, 3) or {
        state.axes[i].name.lower() for i in spatial
    } not in ({"y", "x"}, {"z", "y", "x"}):
        raise ValueError("Compare Images requires explicit YX or ZYX spatial axes.")
    leading = tuple(i for i in range(len(state.axes)) if i not in spatial)
    if any(state.axes[i].name.lower() not in {"t", "c"} for i in leading):
        raise ValueError("Compare Images accepts only T/C non-spatial axes.")
    ordered = [array.transpose(leading + spatial) for array in arrays]
    if use_mask and arrays[2].dtype != np.bool_:
        raise ValueError(
            "Valid coverage must be a Boolean mask, not label IDs or intensities."
        )
    leading_shape = ordered[0].shape[: len(leading)]
    rows = []
    for index in np.ndindex(leading_shape):
        if progress is not None:
            progress.report(
                len(rows), math.prod(leading_shape), "Comparing aligned spatial volumes"
            )
        a, b = ordered[0][index], ordered[1][index]
        valid = ordered[2][index] if use_mask else np.ones(a.shape, dtype=bool)
        count, rmse, correlation = _moments(a, b, valid, progress)
        ssim, ssim_count = _ssim(
            a, b, valid, float(data_range), int(window_size), progress
        )
        psnr = 20 * (math.log10(data_range) - math.log10(rmse)) if rmse else None
        rows.append(
            (
                *index,
                count,
                rmse,
                correlation,
                ssim,
                ssim_count,
                psnr,
                "infinite (exact match)" if rmse == 0 else "finite",
                "defined" if correlation is not None else "undefined (constant image)",
                "defined" if ssim is not None else "no complete valid SSIM windows",
            )
        )
    return TableData(
        columns=tuple(state.axes[i].name.lower() + "_index" for i in leading)
        + (
            "valid_count",
            "rmse",
            "pearson_r",
            "ssim",
            "ssim_window_count",
            "psnr_db",
            "psnr_status",
            "correlation_status",
            "ssim_status",
        ),
        rows=tuple(rows),
        name="Image comparison",
        table_kind="Image comparison QC",
        source_name=state.source_name,
    )
