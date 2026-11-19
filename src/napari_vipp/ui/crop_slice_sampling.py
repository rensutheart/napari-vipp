"""Does the displayed napari slice or slab contain any retained samples?"""

from __future__ import annotations

import numpy as np


def crop_intersects_slice(layer, dims, hidden_intervals) -> bool | None:
    """Intersect local, half-open voxel ranges; never read the image or project it.

    Derive the request from today's dims whenever napari exposes a slice-input
    factory: a point, thickness or axis-order callback can arrive before the
    layer has finished drawing its next slice. Both factory APIs use napari's
    calibrated world-to-data transform; only older APIs fall back to the cache.
    """
    if not hidden_intervals:
        return True
    try:
        slicing = getattr(layer, "_slicing_state", None)
        if hasattr(slicing, "_slice_indices") and hasattr(slicing, "make_slice_input"):
            sample = slicing._slice_indices(slicing.make_slice_input(dims), dims)
        elif callable(make_slice_input := getattr(layer, "_make_slice_input", None)):
            sample = make_slice_input(dims).data_slice(layer._data_to_world.inverse)
        else:
            sample = layer._data_slice
        project = str(getattr(layer, "projection_mode", "none")).lower() != "none"
        for i, start, stop in hidden_intervals:
            centre = sample.point[i]
            if not project:
                lo = round(centre)
                hi = lo + 1
            else:
                # Match napari's _data_slice_to_slices (exclusive stop, except
                # an exact upper sample centre is included; minimum one slice).
                left = centre - sample.margin_left[i]
                right = centre + sample.margin_right[i]
                if round(right) < 0 or round(left) >= layer.data.shape[i]:
                    return False
                lo = max(0, round(left))
                hi = max(0, round(right))
                if np.isclose(hi, right) or lo == hi:
                    hi += 1
            if max(lo, start, 0) >= min(hi, stop, layer.data.shape[i]):
                return False
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        # A canvas may be between old and new dimensions. Do not present an
        # unknown plane as confidently retained (or confidently excluded).
        return None
    return True
