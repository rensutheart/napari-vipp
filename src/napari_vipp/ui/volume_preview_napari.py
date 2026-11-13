"""Bound the texture request for an individual VIPP-owned napari layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from weakref import ref

import numpy as np

from napari_vipp.ui.volume_preview import (
    plan_volume_preview,
    preview_data,
    renderer_volume_limits,
)

VOLUME_PREVIEW_METADATA = "vipp_volume_preview_3d"
_KNOWN_NONSPATIAL_LABELS = frozenset({"c", "channel", "t", "time", "rgb", "rgba"})
_REQUEST_FIELDS = frozenset(
    {
        "id",
        "slice_input",
        "data_at_data_level",
        "data_at_thumbnail_level",
        "rgb",
    }
)
_RESPONSE_FIELDS = frozenset(
    {"image", "slice_input", "tile_to_data", "request_id", "empty"}
)


@dataclass(frozen=True)
class _VolumeRequest:
    """Keep napari's request ID and avoid capturing any live Qt/layer object."""

    id: int
    resolve: Callable

    def __call__(self):
        return self.resolve()


def _known_texture_bytes(dtype) -> int | None:
    dtype = np.dtype(dtype)
    if dtype in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
        return dtype.itemsize
    return None


def _label_texture_bytes(layer) -> int | None:
    try:
        # Napari reports the current colormap's native upload dtype by
        # encoding a zero-length sample. This does not scan the label volume.
        return _known_texture_bytes(layer._slice_dtype())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def default_label_texture_bytes(dtype: np.dtype) -> int | None:
    """Determine the upload size for Review's default, unmodified label map."""
    try:
        from napari.layers import Labels

        # Probe a model, not a viewer, using one synthetic pixel. Reading its
        # display dtype also works before napari exposed the slice dtype hook.
        sample = Labels(np.zeros((1, 1), dtype=dtype))
        return _known_texture_bytes(sample._data_view.dtype)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return None


def install_volume_preview(layer) -> bool:
    """Install only on the recognized sync/async scalar request boundary.

    Both the stock synchronous slicer and async request producer reach this
    function before the array can be sent to the VisPy visual. In contrast, a
    third-party ndisplay event observer runs after napari's core observer.
    Older or unfamiliar napari internals retain native behavior, without any
    class monkeypatch or changes to another viewer/layer.
    """
    try:
        from napari.layers import Labels
        from napari.layers._scalar_field._slice import _ScalarFieldSliceResponse
        from napari.layers._scalar_field.scalar_field import ScalarFieldSlicingState
        from napari.utils.transforms import Affine
    except ImportError:
        return False

    state = getattr(layer, "_slicing_state", None)
    current = getattr(state, "_make_slice_request_internal", None)
    original = getattr(ScalarFieldSlicingState, "_make_slice_request_internal", None)
    if getattr(current, "_vipp_volume_preview", False):
        return True
    if not (
        isinstance(state, ScalarFieldSlicingState)
        and callable(original)
        and getattr(current, "__func__", None) is original
        and hasattr(state._slice_request_class, "__dataclass_fields__")
        and _REQUEST_FIELDS
        <= {field.name for field in fields(state._slice_request_class)}
        and _RESPONSE_FIELDS
        <= {field.name for field in fields(_ScalarFieldSliceResponse)}
    ):
        return False

    state_ref = ref(state)

    def make_request(*args, **kwargs):
        instance = state_ref()
        if instance is None:
            raise RuntimeError("The napari slicing state has been released.")
        request = original(instance, *args, **kwargs)
        layer = instance.layer
        slice_input = request.slice_input
        data = request.data_at_data_level
        axes = tuple(int(axis) for axis in slice_input.displayed)
        if (
            slice_input.ndisplay != 3
            or len(axes) != 3
            or not isinstance(data, np.ndarray)
            or request.rgb
        ):
            layer.metadata.pop(VOLUME_PREVIEW_METADATA, None)
            return request

        labels = isinstance(layer, Labels)
        binary = data.dtype.kind == "b" or layer.metadata.get("data_kind") == "mask"
        names = tuple(str(name).strip().casefold() for name in layer.axis_labels)
        # An existing dims transition can initially expose C/Y/X on Z/C/Y/X.
        # Never coarsen C or T while waiting for the corrected spatial order.
        allow_reduction = (
            not labels
            and not any(names[axis] in _KNOWN_NONSPATIAL_LABELS for axis in axes)
            and (data.ndim == 3 or {names[axis] for axis in axes} == {"x", "y", "z"})
        )
        limits = renderer_volume_limits()
        plan = plan_volume_preview(
            data,
            axes,
            is_binary_mask=binary,
            allow_reduction=allow_reduction,
            texture_bytes_per_voxel=_label_texture_bytes(layer) if labels else None,
            max_texture_axis=limits.max_texture_axis,
            max_texture_bytes=limits.max_texture_bytes,
            admit=limits.admit,
        )
        if plan.can_display_3d and not plan.is_reduced:
            layer.metadata.pop(VOLUME_PREVIEW_METADATA, None)
            return request

        layer.metadata[VOLUME_PREVIEW_METADATA] = {
            "mode": "reduced" if plan.can_display_3d else "unavailable",
            "reason": plan.reason,
            "shape": plan.shape,
            "strides": plan.strides,
        }
        if not plan.can_display_3d:
            # A stock empty slice is not an alternate segmentation; napari
            # hides its visual. The full-resolution layer remains intact.
            dtype = layer._slice_dtype()
            request_id = request.id
            return _VolumeRequest(
                request_id,
                lambda: _ScalarFieldSliceResponse.make_empty(
                    slice_input=slice_input,
                    rgb=False,
                    request_id=request_id,
                    dtype=dtype,
                ),
            )

        sampled = preview_data(data, plan)
        sampled_request = replace(
            request,
            data_at_data_level=sampled,
            data_at_thumbnail_level=(
                sampled
                if request.data_at_thumbnail_level is data
                else request.data_at_thumbnail_level
            ),
        )

        def resolve():
            response = sampled_request()
            if response.empty:
                return response
            # The response image was transposed into the viewer's displayed
            # order, but tile2data still maps *logical* layer axes. Compose
            # in that order; retain the original data-to-world calibration.
            original_tile = response.tile_to_data
            transform = Affine(
                affine_matrix=original_tile.affine_matrix @ np.diag((*plan.strides, 1)),
                name="tile2data",
            )
            return replace(response, tile_to_data=transform)

        return _VolumeRequest(request.id, resolve)

    make_request._vipp_volume_preview = True
    state._make_slice_request_internal = make_request
    return True


def add_managed_scalar_layer(viewer, data, *, labels: bool = False, **kwargs):
    """Prepare the guard before the first slice/visual created by insertion."""
    axis_labels = kwargs.pop("_vipp_axis_labels", None)
    native = viewer.add_labels if labels else viewer.add_image
    if not isinstance(data, np.ndarray) or data.ndim < 3 or kwargs.get("rgb"):
        return native(data, **kwargs)
    try:
        from napari.components import ViewerModel
        from napari.layers import Image, Labels
        from napari.layers._scalar_field.scalar_field import ScalarFieldSlicingState
    except ImportError:
        return native(data, **kwargs)

    if not isinstance(viewer, ViewerModel) or not callable(
        getattr(ScalarFieldSlicingState, "_make_slice_request_internal", None)
    ):
        # Keep the documented viewer interface for test/non-napari viewers.
        return native(data, **kwargs)
    constructor_kwargs = dict(kwargs)
    if axis_labels is not None:
        constructor_kwargs["axis_labels"] = axis_labels
    if not labels:
        constructor_kwargs.setdefault("rgb", False)
        constructor_kwargs.setdefault("blending", "translucent_no_depth")
    layer = (Labels if labels else Image)(data, **constructor_kwargs)
    if not install_volume_preview(layer):
        return native(data, **kwargs)
    return viewer.add_layer(layer)
