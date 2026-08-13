"""Lazy, device-resident CuPy adapter for VIPP dtype conversion.

The optional CUDA stack is imported only when this adapter is executed.  The
implementation mirrors :func:`napari_vipp.core.operations.convert_dtype` for
every public dtype and scaling mode, while the separately declared production
admission region may remain deliberately narrower.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType

import numpy as np

_OUTPUT_DTYPES = frozenset({"bool", "uint8", "uint16", "float32"})
_SCALING_MODES = frozenset({"rescale", "clip", "preserve"})
_PROGRESS_MESSAGE = "Converting dtype"


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after an explicit accelerator implementation is used."""

    return importlib.import_module("cupy")


def convert_dtype(
    data,
    output_dtype: str = "uint8",
    scaling: str = "rescale",
    progress=None,
):
    """Convert a resident array with the authoritative VIPP scaling rules.

    No image-sized host value is created.  If a progress context is supplied,
    the single device operation is synchronized before completion is reported,
    providing truthful cancellation boundaries without penalizing ordinary
    resident execution where no operation-level reporter is attached.
    """

    cupy = _cupy_module()
    array = cupy.asarray(data)
    output_dtype = str(output_dtype).lower()
    scaling = str(scaling).lower()
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(
            "Convert Dtype output_dtype must be bool, uint8, uint16, or float32."
        )
    if scaling not in _SCALING_MODES:
        raise ValueError("Convert Dtype scaling must be rescale, clip, or preserve.")

    _progress_start(progress)
    if output_dtype == "bool":
        result = array.copy() if np.dtype(array.dtype) == np.dtype(bool) else array != 0
    else:
        dtype = np.dtype(output_dtype)
        if np.issubdtype(dtype, np.floating):
            result = _convert_to_float(array, dtype, scaling, cupy=cupy)
        elif np.issubdtype(dtype, np.integer):
            result = _convert_to_integer(array, dtype, scaling, cupy=cupy)
        else:  # pragma: no cover - guarded by the closed public dtype set
            result = array.astype(dtype, copy=True)
    _progress_finish(progress, cupy=cupy)
    return result


def _convert_to_float(array, dtype: np.dtype, scaling: str, *, cupy: ModuleType):
    if scaling == "preserve":
        return array.astype(dtype, copy=True)
    values = array.astype(np.float32, copy=False)
    if scaling == "clip":
        return cupy.nan_to_num(
            cupy.clip(values, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).astype(dtype)
    rescaled, retains_nonfinite_values = _rescale_values(
        values,
        0.0,
        1.0,
        cupy=cupy,
    )
    result = rescaled.astype(dtype)
    if retains_nonfinite_values:
        return _restore_float32_nan_payloads(result, values, cupy=cupy)
    return result


def _convert_to_integer(array, dtype: np.dtype, scaling: str, *, cupy: ModuleType):
    info = np.iinfo(dtype)
    values = array.astype(np.float64, copy=False)
    if scaling == "preserve":
        if not _device_scalar_bool(cupy.all(cupy.isfinite(values))):
            raise ValueError(
                "Convert Dtype preserve cannot represent non-finite values "
                "in an integer output."
            )
        outside_range = cupy.any((values < info.min) | (values > info.max))
        if _device_scalar_bool(outside_range):
            raise ValueError(
                "Convert Dtype preserve input values exceed the output dtype range; "
                "choose clip or rescale explicitly."
            )
        return values.astype(dtype)
    if scaling == "clip":
        scaled = cupy.clip(values, info.min, info.max)
    else:
        scaled, _retains_nonfinite_values = _rescale_values(
            values,
            float(info.min),
            float(info.max),
            cupy=cupy,
        )
    scaled = cupy.nan_to_num(
        scaled,
        nan=0.0,
        posinf=float(info.max),
        neginf=float(info.min),
    )
    return cupy.clip(scaled, info.min, info.max).astype(dtype)


def _rescale_values(values, target_min: float, target_max: float, *, cupy: ModuleType):
    """Reproduce the CPU finite-extrema and constant-array rescale contract."""

    finite = values[cupy.isfinite(values)]
    if finite.size == 0:
        return cupy.zeros_like(values, dtype=np.float64), False
    source_min = finite.min()
    source_max = finite.max()
    if _device_scalar_bool(source_max == source_min):
        fill = target_max if _device_scalar_bool(source_max > 0) else target_min
        return cupy.full_like(values, fill, dtype=np.float64), False
    normalized = (values.astype(np.float64) - source_min) / (
        source_max - source_min
    )
    scaled = normalized * (target_max - target_min) + target_min
    return scaled, True


def _device_scalar_bool(value) -> bool:
    """Resolve one control scalar without materializing an image buffer."""

    return bool(value.item() if hasattr(value, "item") else value)


def _restore_float32_nan_payloads(result, values, *, cupy: ModuleType):
    """Match NumPy's float32-rescale NaN sign, payload, and quieting bits."""

    if np.dtype(values.dtype) != np.dtype(np.float32):
        return result
    result_bits = result.view(np.uint32)
    source_bits = values.view(np.uint32)
    quiet_source_bits = source_bits | np.uint32(0x0040_0000)
    result_bits[...] = cupy.where(
        cupy.isnan(values),
        quiet_source_bits,
        result_bits,
    )
    return result


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


__all__ = ["convert_dtype"]
