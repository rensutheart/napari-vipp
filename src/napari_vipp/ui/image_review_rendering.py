"""Read-only native mask display adapters, never scientific mask processing.

Three-dimensional interpolation and surface lighting are display approximations
on the original grid. No voxel smoothing, resampling or label inference occurs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from napari.components import ViewerModel
from napari.layers import Image
from napari.utils.colormaps import Colormap

from napari_vipp.core.host_memory import (
    capture_host_memory,
    preflight_host_allocation,
)
from napari_vipp.core.review_images import ReviewImageInput

_STORAGE_CHUNK_SIZE = 1_048_576


def _canonical_boolean_storage(data: np.ndarray) -> bool:
    """Check actual Boolean bytes without allocating a whole-volume mask.

    NumPy permits noncanonical True bytes (for example 255). Reinterpreting
    those as scalar occupancy would shift a linearly interpolated surface, so
    they require a separate canonical presentation buffer instead.
    """
    values = data.view(np.uint8)
    with np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=_STORAGE_CHUNK_SIZE,
    ) as chunks:
        return all(not np.any(chunk > 1) for chunk in chunks)


def prepare_mask_review_data(image: ReviewImageInput) -> np.ndarray:
    """Prepare a private window-owned occupancy view/buffer once per input.

    Ordinary 0/1 mask volumes use a zero-copy uint8 view. The full uint8 contrast
    range is necessary because native GL normalizes that texture by 255 when
    mapping the occupancy isovalue. Contrast limits still remain exactly 0/1.
    Noncanonical Boolean storage requires a guarded one-byte-per-voxel display
    conversion, never treating byte magnitudes as occupancy or object identity.
    """
    if image.kind != "mask" or image.data.dtype != np.dtype(bool):
        raise ValueError("Mask review rendering requires an explicit Boolean mask.")
    values = image.data.view(np.uint8)
    if not _canonical_boolean_storage(image.data):
        decision = preflight_host_allocation(
            capture_host_memory(),
            required_bytes=image.data.size,
            purpose="one-byte-per-voxel mask presentation conversion",
        )
        guidance = (
            "Not enough memory for this mask's one-byte-per-voxel display "
            "conversion. Free memory or review a smaller crop. "
            "The scientific mask has not been changed."
        )
        if not decision.allowed:
            raise ValueError(f"{guidance} {decision.reason}")
        try:
            values = np.empty(image.data.shape, dtype=np.uint8)
            np.not_equal(image.data.view(np.uint8), 0, out=values)
        except MemoryError as exc:
            raise ValueError(guidance) from exc
    values.setflags(write=False)
    return values


def add_mask_review_layer(
    viewer: ViewerModel,
    image: ReviewImageInput,
    *,
    display: int,
    prepared_data: np.ndarray | None = None,
    **common: Any,
) -> Image:
    """Borrow prepared occupancy into an isolated native display layer.

    A window should prepare volume occupancy before constructing its Qt viewers,
    then reuse it for both panes and subsequent volume/slice mode switches.
    Two-dimensional display retains the original Boolean view.
    """
    if image.kind != "mask" or image.data.dtype != np.dtype(bool):
        raise ValueError("Mask review rendering requires an explicit Boolean mask.")
    if display == 3:
        values = (
            prepare_mask_review_data(image)
            if prepared_data is None
            else prepared_data
        )
        if (
            not isinstance(values, np.ndarray)
            or values.shape != image.data.shape
            or values.dtype != np.dtype(np.uint8)
            or values.flags.writeable
        ):
            raise ValueError(
                "Prepared mask review data must be a read-only uint8 grid."
            )
    else:
        values = image.data.view()
        values.setflags(write=False)
    layer = viewer.add_image(
        values,
        rgb=False,
        contrast_limits=(0, 1),
        rendering="iso" if display == 3 else "mip",
        iso_threshold=0.5,
        interpolation2d="nearest",
        interpolation3d="linear",
        **common,
    )
    if display == 3:
        layer.contrast_limits_range = (0, 255)
    return layer


def apply_mask_review_style(
    layer: Image,
    *,
    color: str,
    opacity: float,
    display: int,
) -> None:
    """Apply foreground colour/opacity only to the private layer wrapper."""
    rgba = tuple(int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)) + (
        1.0,
    )
    layer.opacity = opacity
    layer.blending = "translucent"
    if display == 3:
        # The native iso shader applies lighting before colour mapping. A
        # binary step map erases that lighting; this opaque ramp retains it.
        # The iso shader itself discards zero/background fragments, so there
        # is no opaque black bounding cube.
        layer.colormap = Colormap([(0, 0, 0, 1), rgba], name="review-mask-surface")
    else:
        layer.colormap = Colormap(
            [(0, 0, 0, 0), rgba],
            controls=[0, 0.5, 1],
            interpolation="zero",
            name="review-mask",
        )
