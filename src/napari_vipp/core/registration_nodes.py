"""Thin node adapters for the registration API (no UI or backend imports)."""

from __future__ import annotations

from types import MappingProxyType

# Scientific input metadata is supplied by node preparation, never authored as
# a UI setting. Keep the injection seam and callable-contract checks in sync.
REGISTRATION_RUNTIME_KEYWORDS = MappingProxyType(
    {
        operation_id: frozenset({"input_states"})
        for operation_id in (
            "estimate_registration",
            "apply_transform",
            "compare_images",
        )
    }
)


def registration_source_frame(state, data, node_id, cancelled=None):
    """Give anonymous graph sources a revision-bound, reproducible frame ID.

    Derived channels/labels retain the source UUID. Separate imported siblings
    must explicitly carry a shared source frame; equal shape/name is not proof.
    """
    import json
    from dataclasses import replace
    from hashlib import sha256

    import numpy as np

    from napari_vipp.core.progress import OperationCancelled

    if state.source.source_uuid or state.source.uri:
        return state
    digest = sha256(
        json.dumps(
            {
                "node": node_id,
                "shape": state.shape,
                "dtype": state.dtype,
                "axes": [axis.to_dict() for axis in state.axes],
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    array = np.asarray(data)
    if array.dtype.hasobject:
        raise ValueError("Registration source frames require numeric image data.")
    for chunk in np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=262144,
    ):
        if cancelled is not None and cancelled():
            raise OperationCancelled("Registration source identification cancelled.")
        digest.update(np.ascontiguousarray(chunk).tobytes())
    return replace(
        state,
        source=replace(state.source, source_uuid="vipp-frame-" + digest.hexdigest()),
    )


def _require_source_frame(state):
    if state is None or not (state.source.source_uuid or state.source.uri):
        raise ValueError(
            "This cached image has no registration source identity. "
            "Refresh the source image, then recalculate registration."
        )


def estimate_registration_node(inputs, *, input_states, progress=None, **params):
    from napari_vipp.core.registration import estimate_registration

    for state in input_states:
        _require_source_frame(state)
    return estimate_registration(
        inputs[0],
        inputs[1] if len(inputs) == 2 else None,
        moving_state=input_states[0],
        reference_state=input_states[1] if len(inputs) == 2 else None,
        progress_context=progress,
        **params,
    )


def apply_transform_node(inputs, *, input_states, progress=None, **params):
    from napari_vipp.core.registration import apply_transform

    _require_source_frame(input_states[0])
    return apply_transform(
        inputs[0],
        inputs[1],
        image_state=input_states[0],
        progress_context=progress,
        **params,
    )


def registration_node_specs():
    from napari_vipp.core.image_comparison import compare_images

    # Called only after the pipeline's declaration types have been defined.
    from napari_vipp.core.pipeline import (
        IMAGE_DATA_CATEGORY,
        PARAMETER_VISIBILITY_PARAMETER_IN,
        InputSpec,
        OperationSpec,
        OutputSpec,
        ParameterSpec,
    )

    def choice(name, title, default, values, tooltip, **visibility):
        return ParameterSpec(
            name,
            title,
            "choice",
            default,
            0,
            0,
            1,
            choices=values,
            tooltip=tooltip,
            **visibility,
        )

    def visible(parameter, *values):
        return dict(
            visibility=PARAMETER_VISIBILITY_PARAMETER_IN,
            visibility_parameter=parameter,
            visibility_values=values,
        )

    return (
        OperationSpec(
            "compare_images",
            "Compare Images",
            IMAGE_DATA_CATEGORY,
            "array",
            "table",
            (
                ParameterSpec(
                    "use_mask",
                    "Use valid-coverage mask",
                    "bool",
                    False,
                    0,
                    1,
                    1,
                    tooltip=(
                        "Connect the same Boolean coverage mask for before/after "
                        "comparisons so scores use the same pixels. "
                        "SSIM requires a complete valid window."
                    ),
                ),
                ParameterSpec(
                    "data_range",
                    "Intensity range for SSIM / PSNR",
                    "float",
                    1.0,
                    1e-12,
                    1e12,
                    1.0,
                    12,
                    tooltip=(
                        "Explicit expected intensity span: e.g. 1 for 0–1 data, "
                        "255 for 8-bit. This does not normalize either image. "
                        "Use the same range before and after."
                    ),
                ),
                ParameterSpec(
                    "window_size",
                    "SSIM window (pixels per spatial axis)",
                    "int",
                    7,
                    3,
                    101,
                    2,
                    tooltip=(
                        "Odd square/cubic uniform window. Only centres whose entire "
                        "window fits the valid coverage are included. "
                        "T and C are compared separately."
                    ),
                ),
            ),
            compare_images,
            max_inputs=3,
            inputs=(
                InputSpec("reference", "array", "Reference"),
                InputSpec("comparison", "array", "Comparison"),
                InputSpec("coverage", "mask", "Valid coverage"),
            ),
            subcategory="Registration",
            execution_policy="manual",
            stack_processing_note=(
                "Same-grid quality control: RMSE, Pearson correlation, SSIM and PSNR "
                "per spatial volume. No implicit alignment or normalization. "
                "Improved similarity is not proof of biological correctness."
            ),
        ),
        OperationSpec(
            "estimate_registration",
            "Estimate Registration",
            IMAGE_DATA_CATEGORY,
            "array",
            "transform",
            (
                choice(
                    "mode",
                    "Register",
                    "Two images",
                    ("Two images", "Time series"),
                    "Align a moving image to a reference, or each complete spatial "
                    "volume in a time series to one reference time.",
                ),
                choice(
                    "model",
                    "Motion model",
                    "Translation",
                    ("Translation", "Rigid", "Affine"),
                    "Translation corrects drift. Rigid also rotates without changing "
                    "shape. Advanced affine also scales and shears; "
                    "it can change measured object shapes.",
                ),
                ParameterSpec(
                    "channel",
                    "Estimation channel (zero-based)",
                    "int",
                    0,
                    0,
                    9999,
                    1,
                    tooltip=(
                        "Use this channel to estimate motion; Apply Transform reuses "
                        "that motion for all channels. "
                        "Use 0 for a single-channel image."
                    ),
                ),
                ParameterSpec(
                    "reference_channel",
                    "Reference channel (zero-based)",
                    "int",
                    0,
                    0,
                    9999,
                    1,
                    tooltip="Channel in the reference image used for estimation.",
                    **visible("mode", "Two images"),
                ),
                ParameterSpec(
                    "reference_time",
                    "Reference time (zero-based)",
                    "int",
                    0,
                    0,
                    1000000,
                    1,
                    tooltip=(
                        "Align every time point directly to this time. One XYZ volume "
                        "moves as a unit; Z slices are never registered separately."
                    ),
                    **visible("mode", "Time series"),
                ),
                ParameterSpec(
                    "precision",
                    "Translation subpixel factor",
                    "int",
                    10,
                    1,
                    100,
                    1,
                    tooltip=(
                        "10 estimates translations in steps of one tenth of a pixel; "
                        "larger values cost more computation."
                    ),
                ),
                ParameterSpec(
                    "max_shift",
                    "Maximum translation (fraction of image)",
                    "float",
                    0.25,
                    0.001,
                    0.49,
                    0.01,
                    3,
                    tooltip=(
                        "Reject estimated translation larger than this fraction of "
                        "the spatial image extent. "
                        "A limit helps reject implausible matches."
                    ),
                ),
                ParameterSpec(
                    "minimum_overlap",
                    "Minimum valid overlap (fraction)",
                    "float",
                    0.25,
                    0.01,
                    1.0,
                    0.05,
                    2,
                    tooltip=(
                        "Reject results with less than this fraction of the reference "
                        "covered by the moving image."
                    ),
                ),
                choice(
                    "metric",
                    "Similarity for optimization",
                    "Correlation",
                    ("Correlation", "Mutual information"),
                    "Correlation suits similar contrast. Mutual information supports "
                    "differing contrasts. Neither score proves "
                    "biological correspondence.",
                    **visible("model", "Rigid", "Affine"),
                ),
                ParameterSpec(
                    "iterations",
                    "Maximum optimization iterations",
                    "int",
                    200,
                    1,
                    5000,
                    10,
                    tooltip=(
                        "Maximum iterations at each resolution level. "
                        "Inspect the diagnostics and overlay after registration."
                    ),
                    **visible("model", "Rigid", "Affine"),
                ),
            ),
            estimate_registration_node,
            max_inputs=2,
            inputs=(
                InputSpec("moving", "array", "Moving image / time series"),
                InputSpec("reference", "array", "Reference image"),
            ),
            outputs=(
                OutputSpec("transform", "transform", "Transform"),
                OutputSpec("diagnostics", "table", "Diagnostics"),
            ),
            subcategory="Registration",
            execution_policy="manual",
            stack_processing_note=(
                "CPU registration. Requires explicit YX or ZYX spatial axes; "
                "T and C are not spatial. Motion is estimated only—connect "
                "Apply Transform to resample original data once."
            ),
        ),
        OperationSpec(
            "apply_transform",
            "Apply Transform",
            IMAGE_DATA_CATEGORY,
            "array",
            "image",
            (
                choice(
                    "interpolation",
                    "Interpolation",
                    "Automatic",
                    ("Automatic", "Nearest", "Linear"),
                    "Automatic uses nearest neighbour for masks/labels "
                    "(preserving IDs) and linear interpolation with "
                    "floating-point output for intensities.",
                ),
                ParameterSpec(
                    "outside_value",
                    "Outside-image fill",
                    "float",
                    0.0,
                    -1e12,
                    1e12,
                    1.0,
                    3,
                    tooltip=(
                        "Value where the moving image does not cover the reference "
                        "grid. Use the valid-coverage output to exclude these "
                        "regions from measurements."
                    ),
                ),
            ),
            apply_transform_node,
            max_inputs=2,
            inputs=(
                InputSpec("image", "array", "Image / masks / labels"),
                InputSpec("transform", "transform", "Transform"),
            ),
            outputs=(
                OutputSpec("aligned", "image", "Aligned image"),
                OutputSpec("coverage", "mask", "Valid coverage"),
            ),
            subcategory="Registration",
            execution_policy="manual",
            stack_processing_note=(
                "Resamples onto the saved reference grid. Reuse motion only for data "
                "sharing the moving coordinate frame and matching time points. "
                "All channels share each volume's transform."
            ),
        ),
    )
