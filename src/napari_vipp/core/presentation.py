"""Presentation-only views that never enter scientific pipeline state."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from napari_vipp.core import operations as _operations


def crop_stack_presentation_view(
    data,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    channel_axis: int | None = None,
    axis_names: Sequence[str] = (),
    z_start: int = 0,
    z_end: int = 0,
    z_axis_explicit: bool | None = None,
) -> np.ndarray:
    """Resolve Crop Stack bounds as a read-only, zero-copy card preview.

    The scientific operation retains its established contiguous-copy contract.
    This presentation-only path shares the same validation helpers but returns
    a view which is never published as pipeline data, cache, or provenance.
    """

    arr = np.asarray(data)
    channel_axis = _operations._validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Crop stack",
    )
    if arr.ndim < 2:
        if any(
            value != 0
            for value in (top, bottom, left, right, z_start, z_end)
        ):
            raise ValueError("Crop stack requires at least two spatial axes.")
        result = arr.view()
        result.setflags(write=False)
        return result

    names = tuple(str(name).strip().casefold() for name in axis_names)
    if names and len(names) != arr.ndim:
        raise ValueError("Declared axis names must match the input array rank.")
    y_axis, x_axis = _operations._xy_axes(
        arr,
        channel_axis=channel_axis,
        axis_names=names,
    )
    slices = [slice(None)] * arr.ndim
    top, bottom = _operations._crop_pair(top, bottom, arr.shape[y_axis])
    left, right = _operations._crop_pair(left, right, arr.shape[x_axis])
    slices[y_axis] = slice(top, arr.shape[y_axis] - bottom)
    slices[x_axis] = slice(left, arr.shape[x_axis] - right)
    z_start, z_end = _operations._crop_margin_values(z_start, z_end)
    if z_start != 0 or z_end != 0:
        if names.count("z") != 1 or z_axis_explicit is False:
            raise ValueError(
                "Z cropping requires exactly one explicitly declared Z axis. "
                "If a generic leading axis is depth, record an exact mapping "
                "such as QYX -> ZYX first."
            )
        z_axis = names.index("z")
        if channel_axis == z_axis:
            raise ValueError(
                "The declared channel axis cannot also be the Z spatial axis."
            )
        z_start, z_end = _operations._crop_pair(
            z_start,
            z_end,
            arr.shape[z_axis],
        )
        slices[z_axis] = slice(z_start, arr.shape[z_axis] - z_end)
    result = arr[tuple(slices)].view()
    result.setflags(write=False)
    return result


__all__ = ["crop_stack_presentation_view"]
