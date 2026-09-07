"""Narrow, device-resident CuPy provider for fixed binary thresholding.

The optional CUDA dependency is imported only when this provider is called.
Its deliberately small production region is a scalar ``float32`` image and a
finite authored threshold; RGB/RGBA reduction remains on the CPU path.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.threshold_range import validate_threshold_range

_PROGRESS_MESSAGE = "Applying binary threshold"
_THRESHOLD_ERROR = "Binary Threshold threshold must be a finite number."


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after this accelerator implementation is selected."""

    return importlib.import_module("cupy")


def binary_threshold(
    data,
    threshold: float = 0.5,
    channel_axis: int | None = None,
    progress=None,
    foreground: str = "Above",
    low_threshold: float = 0.25,
    high_threshold: float = 0.75,
):
    """Return an exact resident Above/Below or In/Outside-range boolean mask.

    This mirrors the scalar region of
    :func:`napari_vipp.core.operations.binary_threshold`: every pixel,
    including NaN, infinities, and either signed zero, is compared with the
    finite Python ``float`` threshold using CuPy's NumPy-compatible scalar
    promotion.  The input is never modified and no image-sized host value or
    host transfer is created. Range modes need one additional temporary boolean
    array; neither mode modifies the input.

    If progress is supplied, completion is reported only after the current
    stream has synchronized.  Cancellation before or after that boundary
    prevents a result from being returned.
    """

    if foreground not in ("Above", "Below", "In range", "Outside range"):
        raise ValueError(
            "Binary Threshold foreground must be 'Above', 'Below', "
            "'In range' or 'Outside range'."
        )
    range_mode = foreground in ("In range", "Outside range")
    if range_mode:
        low, high = validate_threshold_range(low_threshold, high_threshold)
    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None and not np.dtype(source_dtype).isnative:
        raise ValueError(
            "Binary Threshold GPU provider requires native-endian float32 image data."
        )
    cupy = _cupy_module()
    array = cupy.asarray(data)
    if array.size == 0:
        raise ValueError("Binary threshold requires non-empty image data.")
    if np.dtype(array.dtype) != np.dtype(np.float32):
        raise ValueError("Binary Threshold GPU provider requires float32 image data.")
    if channel_axis is not None:
        raise ValueError(
            "Binary Threshold GPU provider supports scalar images only; "
            "channel_axis must be None."
        )
    if not range_mode:
        threshold_value = _validated_threshold(threshold)

    _progress_start(progress)
    if foreground == "In range":
        result = array >= low
        result &= array <= high
    elif foreground == "Outside range":
        result = array < low
        result |= array > high
    else:
        result = (
            array < threshold_value
            if foreground == "Below"
            else array > threshold_value
        )
    _progress_finish(progress, cupy=cupy)
    return result


def _validated_threshold(threshold) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_THRESHOLD_ERROR) from exc
    if not np.isfinite(value):
        raise ValueError(_THRESHOLD_ERROR)
    return value


def _progress_start(progress) -> None:
    if progress is None:
        return
    progress.check_cancelled()
    progress.report(0, 1, _PROGRESS_MESSAGE)
    progress.check_cancelled()


def _progress_finish(progress, *, cupy: ModuleType) -> None:
    if progress is None:
        return
    cupy.cuda.get_current_stream().synchronize()
    progress.check_cancelled()
    progress.report(1, 1, _PROGRESS_MESSAGE)


__all__ = ["binary_threshold"]
