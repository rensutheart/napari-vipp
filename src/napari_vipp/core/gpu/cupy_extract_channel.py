"""Lazy, device-resident CuPy bridge for semantic channel extraction.

The authoritative CPU operation selects the first axis explicitly described as
a channel axis, preferring axis types over axis names.  This adapter preserves
that contract but returns a CuPy view instead of allocating a second device
array.  The shared device executor owns aliases by their underlying allocation,
so the input allocation remains live for exactly as long as the selected view.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np

_CHANNEL_AXIS_NAMES = frozenset({"c", "channel", "rgb", "rgba"})
_SUPPORTED_DTYPES = frozenset(
    {np.dtype(bool), np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)}
)
_PROGRESS_MESSAGE = "Extracting channel"


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after an admitted accelerator implementation is used."""

    return importlib.import_module("cupy")


def extract_channel(
    data,
    channel: int = 0,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    progress=None,
):
    """Return one semantic channel as a device-resident, allocation-sharing view.

    This function intentionally performs no host conversion and launches no
    device kernel.  Cancellation therefore has boundaries immediately before
    and after the constant-time view construction; no synchronization is
    required before reporting completion.
    """

    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None:
        source_dtype = np.dtype(source_dtype)
        if not source_dtype.isnative:
            raise ValueError(
                "Extract Channel GPU execution requires native-endian input "
                "so its dtype remains unchanged."
            )
        if source_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                "Extract Channel GPU execution supports only bool, uint8, "
                f"uint16, and float32 input; received {source_dtype}."
            )

    cupy = _cupy_module()
    array = cupy.asarray(data)
    if np.dtype(array.dtype) not in _SUPPORTED_DTYPES:
        raise ValueError(
            "Extract Channel GPU execution supports only bool, uint8, uint16, "
            f"and float32 input; received {array.dtype}."
        )
    channel_axis = _strict_channel_axis(
        array.ndim,
        axis_names=axis_names,
        axis_types=axis_types,
    )
    if channel_axis is None:
        raise ValueError(
            "Extract Channel requires an explicitly declared channel axis."
        )
    if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, Integral):
        raise ValueError("Extract Channel channel index must be an integer.")

    selected_channel = int(channel)
    channel_count = int(array.shape[channel_axis])
    if selected_channel < 0:
        selected_channel += channel_count
    if selected_channel < 0 or selected_channel >= channel_count:
        raise ValueError(
            f"Extract Channel channel index {selected_channel!r} is out of range "
            f"for {channel_count} channels."
        )

    _progress_start(progress)
    selection = [slice(None)] * array.ndim
    selection[channel_axis] = selected_channel
    result = array[tuple(selection)]
    _progress_finish(progress)
    return result


def _strict_channel_axis(
    ndim: int,
    *,
    axis_names: Sequence[str],
    axis_types: Sequence[str],
) -> int | None:
    for index, axis_type in enumerate(axis_types[:ndim]):
        if str(axis_type).strip().lower() == "channel":
            return index
    for index, axis_name in enumerate(axis_names[:ndim]):
        if str(axis_name).strip().lower() in _CHANNEL_AXIS_NAMES:
            return index
    return None


def _progress_start(progress) -> None:
    if progress is None:
        return
    progress.check_cancelled()
    progress.report(0, 1, _PROGRESS_MESSAGE)
    progress.check_cancelled()


def _progress_finish(progress) -> None:
    if progress is None:
        return
    progress.check_cancelled()
    progress.report(1, 1, _PROGRESS_MESSAGE)


__all__ = ["extract_channel"]
