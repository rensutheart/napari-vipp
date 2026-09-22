"""Metadata-only registration contracts; never estimate or invent a transform.

Planning objects deliberately are not ``TransformData`` and have no matrices,
serialization, or numerical execution interface. They describe only facts that
are knowable before pixels are read. Acceptance of an estimated alignment,
finite-value checks, and coverage are still decided by actual CPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from napari_vipp.core.metadata import AxisMetadata, ImageState
from napari_vipp.core.registration import _fraction, _integer, _same_sampling
from napari_vipp.core.transforms import (
    RegistrationGrid,
    TransformData,
    TransformState,
    apply_transform_output_state,
    registration_grid,
)

REGISTRATION_PLANNING_OPERATIONS = frozenset(
    {"estimate_registration", "apply_transform"}
)


@dataclass(frozen=True, slots=True)
class TransformPlan:
    """A grid/time promise, not a computed or accepted registration result."""

    moving_grid: RegistrationGrid
    reference_grid: RegistrationGrid
    state: TransformState
    time_axis: AxisMetadata | None = None
    reference_time: int | None = None

    @property
    def is_time_series(self) -> bool:
        return self.time_axis is not None

    @property
    def transform_count(self) -> int:
        return self.state.transform_count


def _image_contract(value, state):
    """Inspect descriptors only; even ``np.asarray(value)`` is forbidden here."""
    if not isinstance(state, ImageState):
        raise ValueError("Registration planning requires carried image metadata.")
    dtype = np.dtype(state.dtype)
    if (
        tuple(getattr(value, "shape", ())) != state.shape
        or getattr(value, "dtype", None) is None
        or np.dtype(value.dtype) != dtype
    ):
        raise ValueError("Registration data and carried shape/dtype metadata disagree.")
    if dtype.kind not in "biuf" or not state.shape or min(state.shape) < 1:
        raise ValueError("Registration requires a nonempty real numeric image.")
    return registration_grid(state)


def _channel_contract(state, channel):
    names = tuple(axis.name.lower() for axis in state.axes)
    if "c" in names:
        if channel >= state.shape[names.index("c")]:
            raise ValueError(
                "The selected registration channel is outside the input image."
            )
    elif channel != 0:
        raise ValueError("A single-channel image only has channel index 0.")


def _estimate_plan(call):
    params = call.kwargs
    mode = params.get("mode", "Two images")
    model = params.get("model", "Translation")
    if mode not in ("Two images", "Time series") or model not in (
        "Translation",
        "Rigid",
        "Affine",
    ):
        raise ValueError(
            "Choose Two images/Time series and Translation/Rigid/Affine registration."
        )
    if params.get("metric", "Correlation") not in ("Correlation", "Mutual information"):
        raise ValueError(
            "Choose Correlation or Mutual information as the registration metric."
        )
    channel = _integer(params.get("channel", 0), "Channel", 0, 100000)
    reference_channel = _integer(
        params.get("reference_channel", 0), "Reference channel", 0, 100000
    )
    reference_time = _integer(
        params.get("reference_time", 0), "Reference time", 0, 1000000
    )
    _integer(params.get("precision", 10), "Subpixel precision", 1, 1000)
    _integer(params.get("iterations", 200), "Iterations", 1, 10000)
    _fraction(
        params.get("max_shift", 0.25), "Maximum displacement fraction", 0.001, 1.0
    )
    _fraction(
        params.get("minimum_overlap", 0.25), "Minimum overlap fraction", 0.001, 1.0
    )
    expected_inputs = 1 if mode == "Time series" else 2
    if len(call.inputs) != expected_inputs or len(call.input_states) != expected_inputs:
        raise ValueError(
            "Time series needs one image; Two images needs Moving and Reference inputs."
        )
    moving_state = call.input_states[0]
    moving_grid = _image_contract(call.inputs[0], moving_state)
    _channel_contract(moving_state, channel)
    names = tuple(axis.name.lower() for axis in moving_state.axes)
    if mode == "Time series":
        if "t" not in names:
            raise ValueError("Time series registration requires an explicit T axis.")
        count = moving_state.shape[names.index("t")]
        if count < 2 or reference_time >= count:
            raise ValueError(
                "Select a valid reference time from an image "
                "with at least two time points."
            )
        time_axis = moving_state.axes[names.index("t")]
        reference_grid, reference_state = moving_grid, moving_state
    else:
        reference_state = call.input_states[1]
        reference_grid = _image_contract(call.inputs[1], reference_state)
        _channel_contract(reference_state, reference_channel)
        if "t" in names or any(
            axis.name.lower() == "t" for axis in reference_state.axes
        ):
            raise ValueError(
                "Two images mode does not consume T. "
                "Select a time point or use Time series mode."
            )
        count, time_axis = 1, None
    if min((*moving_grid.shape, *reference_grid.shape)) < 4:
        raise ValueError(
            "Registration needs at least four pixels/voxels along every spatial axis."
        )
    if (
        moving_grid.axes != reference_grid.axes
        or moving_grid.unit != reference_grid.unit
    ):
        raise ValueError(
            "Moving and reference images need the same spatial rank "
            "and compatible coordinate units."
        )
    if model == "Translation" and not _same_sampling(moving_grid, reference_grid):
        raise ValueError(
            "Translation needs equal spatial shape and sampling. "
            "Use Rigid/Affine for different grids, or explicitly resample first."
        )
    state = TransformState(
        model,
        moving_grid.axes,
        count,
        moving_grid.unit,
        moving_state.source_name,
        reference_state.source_name,
        ("Metadata-only registration plan; no alignment has been estimated",),
    )
    plan = TransformPlan(
        moving_grid,
        reference_grid,
        state,
        time_axis=time_axis,
        reference_time=reference_time if time_axis else None,
    )
    # Diagnostic values and quality acceptance require actual image pixels.
    return ((plan, state), (None, None))


def _apply_plan(pipeline, call):
    if len(call.inputs) != 2 or len(call.input_states) != 2:
        raise ValueError("Apply Transform needs an image and a registration transform.")
    transform = call.inputs[1]
    if not isinstance(transform, (TransformPlan, TransformData)):
        raise TypeError(
            "Apply Transform needs a registration transform, not an image/table."
        )
    image_state = call.input_states[0]
    grid = _image_contract(call.inputs[0], image_state)
    expected = transform.moving_grid
    if grid.frame_id != expected.frame_id:
        raise ValueError(
            "The image belongs to a different coordinate frame. Use the original "
            "moving image or a sibling channel/mask retaining its source identity."
        )
    if not _same_sampling(grid, expected) or not np.allclose(
        grid.origin, expected.origin, rtol=1e-9, atol=1e-12
    ):
        raise ValueError(
            "The image grid differs from the transform's moving grid. "
            "Re-estimate after cropping or changing calibration."
        )
    names = tuple(axis.name.lower() for axis in image_state.axes)
    if transform.is_time_series:
        if "t" not in names or (
            image_state.shape[names.index("t")] != transform.state.transform_count
        ):
            raise ValueError(
                "Transform series requires the same time-point count "
                "as its moving image."
            )
        axis = image_state.axes[names.index("t")]
        anchor = transform.time_axis
        if (axis.unit, axis.scale, axis.translation) != (
            anchor.unit,
            anchor.scale,
            anchor.translation,
        ):
            raise ValueError("Time calibration does not match the transform series.")
    elif "t" in names:
        raise ValueError(
            "A pairwise transform cannot be silently broadcast over time; "
            "estimate a Time series transform."
        )
    interpolation = call.kwargs.get("interpolation", "Automatic")
    if interpolation not in ("Automatic", "Nearest neighbour", "Nearest", "Linear"):
        raise ValueError("Choose Automatic, Nearest neighbour or Linear interpolation.")
    dtype = np.dtype(image_state.dtype)
    labels = (
        dtype.kind == "b"
        or "label" in image_state.kind.lower()
        or "mask" in image_state.kind.lower()
    )
    nearest = interpolation in ("Nearest", "Nearest neighbour") or (
        interpolation == "Automatic" and labels
    )
    if labels and not nearest:
        raise ValueError(
            "Masks and labels require Nearest neighbour interpolation to preserve IDs."
        )
    output_dtype = dtype if nearest else np.dtype(np.float64)
    outside = call.kwargs.get("outside_value", 0.0)
    try:
        if not np.isfinite(outside):
            raise ValueError
        if nearest and dtype.kind in "biu":
            low, high = (
                (0, 1)
                if dtype.kind == "b"
                else (np.iinfo(dtype).min, np.iinfo(dtype).max)
            )
            if outside != int(outside) or not low <= outside <= high:
                raise ValueError
        if not np.isfinite(np.asarray(outside, dtype=output_dtype).item()):
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            "Outside value must be finite and representable in the output dtype "
            "(integer for nearest integer data)."
        ) from exc
    shape = tuple(
        transform.reference_grid.shape[grid.axes.index(name)]
        if name in grid.axes
        else image_state.shape[index]
        for index, name in enumerate(names)
    )
    aligned = pipeline._axis_contract_proxy(shape, output_dtype)
    coverage = pipeline._axis_contract_proxy(shape, np.dtype(bool))
    return (
        (aligned, apply_transform_output_state(image_state, transform, aligned)),
        (
            coverage,
            apply_transform_output_state(image_state, transform, coverage, True),
        ),
    )


def project_registration_outputs(pipeline, call):
    """Return exact typed port descriptors, or None for non-registration nodes.

    Invalid metadata raises the same actionable boundary errors as execution.
    A genuinely unknown upstream image contract stops propagation conservatively.
    """
    if call is None or call.operation_id not in REGISTRATION_PLANNING_OPERATIONS:
        return None
    if not call.input_states or any(state is None for state in call.input_states):
        return ((None, None),) * call.output_port_count
    if call.operation_id == "estimate_registration":
        return _estimate_plan(call)
    return _apply_plan(pipeline, call)
