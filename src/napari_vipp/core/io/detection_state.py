"""Lossless optional template-response metadata shared by image file formats."""

import math
from dataclasses import replace

import numpy as np

from napari_vipp.core.grid import _unit_dimension_and_factor
from napari_vipp.core.metadata import ImageState

TEMPLATE_STATE_KEY = "template_match_state"


def template_state_payload(state, data):
    """Return tagged state only after validating its unchanged storage layout."""
    if state is None or state.template_match_metadata is None:
        return None
    _validate_layout(state, data)
    return state.to_dict()


def restore_template_state(
    payload, state, data, *, native_sampling=False, native_origin=False
):
    """Restore explicit scientific state without casting/reordering file pixels.

    This is a metadata-only seam, including for lazy source inspection. Exact
    paired data revisions are still checked by Find Peaks before consumption.
    """
    if payload is None:
        return state
    carried = ImageState.from_dict(payload) if isinstance(payload, dict) else None
    if carried is None or carried.template_match_metadata is None:
        raise ValueError(
            "Invalid saved Template Match state; provenance was not restored."
        )
    _validate_layout(carried, data)
    if tuple(axis.name.lower() for axis in state.axes) != tuple(
        axis.name.lower() for axis in carried.axes
    ):
        raise ValueError("Saved Template Match axes differ from the file axes.")
    if native_sampling:
        for stored, native in zip(carried.axes, state.axes, strict=True):
            su, sf = _unit_dimension_and_factor(stored.unit)
            nu, nf = _unit_dimension_and_factor(native.unit)
            if (
                su != nu
                or not math.isclose(
                    stored.scale * sf, native.scale * nf, rel_tol=1e-9, abs_tol=0
                )
                or (
                    native_origin
                    and not math.isclose(
                        stored.translation * sf,
                        native.translation * nf,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                )
            ):
                raise ValueError(
                    "Saved Template Match calibration conflicts with the file grid."
                )
    return replace(
        state,
        axes=carried.axes,
        kind=carried.kind,
        history=carried.history,
        channels=carried.channels,
        acquisition=carried.acquisition,
        template_match_metadata=carried.template_match_metadata,
    )


def _validate_layout(state, data):
    tag = state.template_match_metadata
    dtype = np.dtype(data.dtype)
    if (
        tuple(data.shape) != state.shape
        or not state.axes_explicit
        or any(axis.type != "space" for axis in state.axes)
        or state.shape != tag.score_shape
        or dtype.name != state.dtype
        or (dtype == np.dtype(bool) and state.kind != "binary mask")
        or (
            dtype == np.dtype(np.float64) and state.kind != "template match score image"
        )
        or dtype not in (np.dtype(bool), np.dtype(np.float64))
        or tuple(axis.name.lower() for axis in state.axes) != tag.axes
    ):
        raise ValueError(
            "Template Match files must retain their exact score-grid shape, axes, "
            "float64 scores or Boolean validity dtype and semantic kind."
        )
    for i, axis in enumerate(state.axes):
        unit, factor = _unit_dimension_and_factor(axis.unit)
        original_unit, original_factor = _unit_dimension_and_factor(tag.search_units[i])
        expected_origin = (
            tag.search_origin[i] + tag.center_offset[i] * tag.search_scale[i]
        )
        if (
            unit != original_unit
            or not math.isclose(
                axis.scale * factor,
                tag.search_scale[i] * original_factor,
                rel_tol=1e-12,
                abs_tol=0,
            )
            or not math.isclose(
                axis.translation * factor,
                expected_origin * original_factor,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Template Match output calibration changed before persistence."
            )
