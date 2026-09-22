from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.registration import apply_transform, estimate_registration
from napari_vipp.core.registration_samples import (
    affine_pair,
    affine_volume_pair,
    drift_series,
    landmark_errors,
    rigid_image_pair,
    rigid_volume_pair,
    translation_pair,
)
from napari_vipp.core.transforms import (
    TransformData,
    TransformState,
    apply_transform_output_state,
    registration_grid,
)


def state(array, names, *, spacing=None, origin=None, frame="moving", kind=None):
    spatial = [name for name in names.lower() if name in "zyx"]
    spacing = spacing or (1.0,) * len(spatial)
    origin = origin or (0.0,) * len(spatial)
    axes = []
    for name in names.lower():
        if name in "zyx":
            j = spatial.index(name)
            axes.append(AxisMetadata(name, "space", "um", spacing[j], origin[j]))
        else:
            axes.append(
                AxisMetadata(
                    name,
                    "time" if name == "t" else "channel",
                    "s" if name == "t" else None,
                )
            )
    result = image_state_from_array(
        array,
        axes=tuple(axes),
        source_name=frame,
        source=SourceMetadata(source_uuid=frame),
    )
    return replace(result, kind=kind) if kind else result


def pair_states(phantom):
    return (
        state(
            phantom.moving, phantom.axes, spacing=phantom.spacing, origin=phantom.origin
        ),
        state(
            phantom.reference,
            phantom.axes,
            spacing=phantom.spacing,
            origin=phantom.origin,
            frame="reference",
        ),
    )


@pytest.mark.parametrize("noisy", [False, True])
def test_translation_independent_physical_landmarks_and_immutable_inputs(noisy):
    phantom = translation_pair(noisy=noisy)
    moving_state, reference_state = pair_states(phantom)
    before = phantom.moving.copy()
    transform, diagnostics = estimate_registration(
        phantom.moving,
        phantom.reference,
        moving_state=moving_state,
        reference_state=reference_state,
        precision=20,
    )
    assert max(landmark_errors(transform.matrices[0], phantom)) < 0.2
    assert diagnostics.row_count == 1
    assert diagnostics.records()[0]["correlation"] > 0.99
    np.testing.assert_array_equal(phantom.moving, before)
    assert not phantom.moving.flags.writeable
    aligned, valid = apply_transform(
        phantom.moving, transform, image_state=moving_state
    )
    assert aligned.dtype == np.float64
    assert valid.dtype == bool and valid.shape == aligned.shape
    assert np.corrcoef(aligned[valid], phantom.reference[valid])[0, 1] > 0.99


def test_translation_accounts_for_different_origins():
    phantom = translation_pair(noisy=False)
    moving_state, reference_state = pair_states(phantom)
    reference_state = replace(
        reference_state,
        axes=tuple(
            replace(a, translation=a.translation + 12) for a in reference_state.axes
        ),
    )
    transform, _ = estimate_registration(
        phantom.moving,
        phantom.reference,
        moving_state=moving_state,
        reference_state=reference_state,
    )
    expected = np.asarray(phantom.moving_to_reference[0]).copy()
    expected[:-1, -1] += 12
    np.testing.assert_allclose(transform.matrices[0], expected, atol=0.12)
    aligned, valid = apply_transform(
        phantom.moving, transform, image_state=moving_state
    )
    assert np.corrcoef(aligned[valid], phantom.reference[valid])[0, 1] > 0.99
    output_state = apply_transform_output_state(moving_state, transform, aligned)
    assert (
        tuple(a.translation for a in output_state.axes)
        == transform.reference_grid.origin
    )


def test_whole_volume_drift_all_channels_and_sibling_labels():
    phantom = drift_series()
    image_state = state(
        phantom.moving, phantom.axes, spacing=phantom.spacing, origin=phantom.origin
    )
    transform, diagnostics = estimate_registration(
        phantom.moving, moving_state=image_state, mode="Time series", precision=10
    )
    assert (
        transform.is_time_series and len(transform.matrices) == phantom.moving.shape[0]
    )
    assert diagnostics.row_count == phantom.moving.shape[0]
    for t, matrix in enumerate(transform.matrices):
        assert max(landmark_errors(matrix, phantom, time_index=t)) < 0.15
    aligned, valid = apply_transform(phantom.moving, transform, image_state=image_state)
    for t in range(len(transform.matrices)):
        for c in range(2):
            assert (
                np.corrcoef(aligned[t, c][valid[t, c]], aligned[0, c][valid[t, c]])[
                    0, 1
                ]
                > 0.998
            )
    label_state = state(
        phantom.labels,
        "TZYX",
        spacing=phantom.spacing,
        origin=phantom.origin,
        kind="label image",
    )
    labels, labels_valid = apply_transform(
        phantom.labels, transform, image_state=label_state
    )
    assert labels.dtype == phantom.labels.dtype
    assert set(np.unique(labels)) <= set(np.unique(phantom.labels))
    assert labels_valid.shape == labels.shape


@pytest.mark.parametrize(
    "model,factory,tolerance",
    [
        ("Rigid", rigid_volume_pair, 0.4),
        ("Rigid", rigid_image_pair, 0.4),
        ("Affine", affine_pair, 0.4),
        ("Affine", affine_volume_pair, 0.4),
    ],
)
def test_itk_independent_landmark_oracle(model, factory, tolerance):
    pytest.importorskip("SimpleITK")
    phantom = factory()
    moving_state, reference_state = pair_states(phantom)
    transform, diagnostics = estimate_registration(
        phantom.moving,
        phantom.reference,
        moving_state=moving_state,
        reference_state=reference_state,
        model=model,
        iterations=150,
    )
    assert max(landmark_errors(transform.matrices[0], phantom)) < tolerance
    assert diagnostics.records()[0]["correlation"] > 0.99


def manual_transform(array, image_state, matrix):
    grid = registration_grid(image_state)
    return TransformData(
        (matrix,), grid, grid, TransformState("Translation", grid.axes, 1, grid.unit)
    )


@pytest.mark.parametrize(
    "dtype,values",
    [(np.uint64, (2**63 + 1, 2**64 - 1)), (np.int64, (-(2**63) + 1, 2**63 - 1))],
)
def test_nearest_wide_labels_are_exact(dtype, values):
    labels = np.zeros((7, 9), dtype=dtype)
    labels[2, 3], labels[4, 5] = values
    labels.setflags(write=False)
    image_state = state(labels, "YX", kind="label image")
    matrix = np.eye(3)
    matrix[:2, 2] = (1, -2)
    transform = manual_transform(labels, image_state, matrix)
    result, valid = apply_transform(labels, transform, image_state=image_state)
    assert result.dtype == labels.dtype
    assert result[3, 1] == values[0] and result[5, 3] == values[1]
    assert not valid[0].any() and not valid[:, -2:].any()
    assert set(np.unique(result)) == set(np.unique(labels))
    with pytest.raises(ValueError, match="Nearest neighbour"):
        apply_transform(
            labels, transform, image_state=image_state, interpolation="Linear"
        )


@pytest.mark.parametrize("dtype", [np.float64, np.uint64, np.int64])
def test_exact_physical_grid_translation_keeps_every_boundary_pixel(dtype):
    array = np.arange(64, dtype=dtype).reshape(8, 8) + 1
    if dtype == np.uint64:
        array += np.uint64(2**63)
    elif dtype == np.int64:
        array += np.int64(-(2**63))
    moving_state = state(
        array,
        "YX",
        spacing=(0.3, 0.4),
        origin=(11, -7),
        kind="label image" if dtype != np.float64 else None,
    )
    reference_state = state(
        array, "YX", spacing=(0.3, 0.4), origin=(1, 2), frame="reference"
    )
    moving_grid, reference_grid = (
        registration_grid(moving_state),
        registration_grid(reference_state),
    )
    matrix = np.eye(3)
    matrix[:2, -1] = (-10, 9)
    transform = TransformData(
        (matrix,),
        moving_grid,
        reference_grid,
        TransformState("Translation", moving_grid.axes, 1, moving_grid.unit),
    )
    aligned, coverage = apply_transform(array, transform, image_state=moving_state)
    assert coverage.all()
    if dtype == np.float64:
        np.testing.assert_allclose(aligned, array, rtol=0, atol=1e-12)
    else:
        np.testing.assert_array_equal(aligned, array)
        assert aligned.dtype == array.dtype


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize(
    "interpolation,displacement", [("Linear", 1e-9), ("Nearest", 0.5001)]
)
def test_roundoff_boundary_snap_does_not_admit_real_outside_points(
    direction, interpolation, displacement
):
    array = np.arange(64, dtype=np.float64).reshape(8, 8) + 1
    moving_state = state(array, "YX", spacing=(0.3, 0.4), origin=(11, -7))
    reference_state = state(
        array, "YX", spacing=(0.3, 0.4), origin=(1, 2), frame="reference"
    )
    moving_grid = registration_grid(moving_state)
    matrix = np.eye(3)
    matrix[:2, -1] = (-10 + direction * displacement * 0.3, 9)
    transform = TransformData(
        (matrix,),
        moving_grid,
        registration_grid(reference_state),
        TransformState("Translation", moving_grid.axes, 1, moving_grid.unit),
    )
    aligned, coverage = apply_transform(
        array,
        transform,
        image_state=moving_state,
        interpolation=interpolation,
        outside_value=-99,
    )
    excluded_row = 0 if direction > 0 else -1
    assert not coverage[excluded_row].any()
    assert np.all(aligned[excluded_row] == -99)
    assert coverage.sum() == 56


def test_exact_alignment_at_different_physical_origins_passes_full_overlap_gate():
    array = translation_pair(noisy=False).reference
    moving_state = state(array, "YX", spacing=(0.3, 0.4), origin=(11, -7))
    reference_state = state(
        array, "YX", spacing=(0.3, 0.4), origin=(1, 2), frame="reference"
    )
    transform, diagnostics = estimate_registration(
        array,
        array,
        moving_state=moving_state,
        reference_state=reference_state,
        minimum_overlap=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(transform.matrices[0])[:2, -1], (-10, 9), rtol=0, atol=1e-12
    )
    assert diagnostics.records()[0]["valid_overlap_fraction"] == 1
    aligned, coverage = apply_transform(array, transform, image_state=moving_state)
    assert coverage.all()
    np.testing.assert_allclose(aligned, array, rtol=0, atol=1e-12)


def test_semantic_axis_reordering_is_explicit_and_reversible():
    phantom = translation_pair(noisy=False)
    moving, reference = phantom.moving.T, phantom.reference.T
    moving_state = state(
        moving, "XY", spacing=phantom.spacing[::-1], origin=phantom.origin[::-1]
    )
    reference_state = state(
        reference,
        "XY",
        spacing=phantom.spacing[::-1],
        origin=phantom.origin[::-1],
        frame="reference",
    )
    transform, _ = estimate_registration(
        moving, reference, moving_state=moving_state, reference_state=reference_state
    )
    aligned, valid = apply_transform(moving, transform, image_state=moving_state)
    assert aligned.shape == moving.shape
    assert np.corrcoef(aligned[valid], reference[valid])[0, 1] > 0.99


@pytest.mark.parametrize(
    "problem",
    [
        "constant",
        "nonfinite",
        "ambiguous",
        "different sampling",
        "wrong frame",
        "overlap",
        "shift",
    ],
)
def test_actionable_failures(problem):
    phantom = translation_pair(noisy=False)
    moving, reference = phantom.moving.copy(), phantom.reference.copy()
    moving_state, reference_state = pair_states(phantom)
    kwargs = {}
    if problem == "constant":
        moving[:] = 1
    elif problem == "nonfinite":
        moving[0, 0] = np.nan
    elif problem == "ambiguous":
        moving_state = replace(
            moving_state,
            axes=tuple(
                replace(a, confidence="shape-inferred") for a in moving_state.axes
            ),
        )
    elif problem == "different sampling":
        reference_state = replace(
            reference_state,
            axes=tuple(replace(a, scale=a.scale * 2) for a in reference_state.axes),
        )
    elif problem == "overlap":
        kwargs["minimum_overlap"] = 1.0
    elif problem == "shift":
        kwargs["max_shift"] = 0.001
    if problem == "wrong frame":
        transform, _ = estimate_registration(
            moving,
            reference,
            moving_state=moving_state,
            reference_state=reference_state,
        )
        wrong = replace(moving_state, source=SourceMetadata(source_uuid="other"))
        with pytest.raises(ValueError, match="coordinate frame"):
            apply_transform(moving, transform, image_state=wrong)
    else:
        with pytest.raises(ValueError):
            estimate_registration(
                moving,
                reference,
                moving_state=moving_state,
                reference_state=reference_state,
                **kwargs,
            )


def test_cancel_estimation_and_apply_no_partial_result():
    phantom = translation_pair(noisy=False)
    moving_state, reference_state = pair_states(phantom)
    context = ProgressContext(cancelled=lambda: True)
    with pytest.raises(OperationCancelled):
        estimate_registration(
            phantom.moving,
            phantom.reference,
            moving_state=moving_state,
            reference_state=reference_state,
            progress_context=context,
        )
    transform = manual_transform(phantom.moving, moving_state, np.eye(3))
    with pytest.raises(OperationCancelled):
        apply_transform(
            phantom.moving,
            transform,
            image_state=moving_state,
            progress_context=context,
        )


def test_time_series_rejects_missing_or_changed_time_calibration():
    phantom = drift_series()
    image_state = state(
        phantom.moving, phantom.axes, spacing=phantom.spacing, origin=phantom.origin
    )
    transform, _ = estimate_registration(
        phantom.moving, moving_state=image_state, mode="Time series"
    )
    changed = replace(
        image_state, axes=(replace(image_state.axes[0], scale=2), *image_state.axes[1:])
    )
    with pytest.raises(ValueError, match="Time calibration"):
        apply_transform(phantom.moving, transform, image_state=changed)


def test_nearest_and_linear_coverage_use_interpolation_support():
    array = np.arange(30, dtype=np.float64).reshape(5, 6)
    image_state = state(array, "YX")
    matrix = np.eye(3)
    matrix[:2, 2] = (0.25, 0.25)
    transform = manual_transform(array, image_state, matrix)
    nearest, valid_nearest = apply_transform(
        array, transform, image_state=image_state, interpolation="Nearest neighbour"
    )
    linear, valid_linear = apply_transform(
        array, transform, image_state=image_state, interpolation="Linear"
    )
    np.testing.assert_array_equal(nearest, array)
    assert valid_nearest.all()
    assert not valid_linear[0].any() and not valid_linear[:, 0].any()
    assert np.all(linear[~valid_linear] == 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel": -1},
        {"precision": 0},
        {"reference_time": True},
        {"iterations": 0},
        {"minimum_overlap": np.nan},
        {"max_shift": 2},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    phantom = translation_pair(noisy=False)
    moving_state, reference_state = pair_states(phantom)
    with pytest.raises(ValueError):
        estimate_registration(
            phantom.moving,
            phantom.reference,
            moving_state=moving_state,
            reference_state=reference_state,
            **kwargs,
        )


def test_repeated_structure_rejects_ambiguous_integer_peaks():
    yy, xx = np.indices((64, 64))
    image = ((yy % 8 < 3) & (xx % 8 < 3)).astype(np.float32)
    image_state = state(image, "YX")
    with pytest.raises(ValueError, match="ambiguous"):
        estimate_registration(
            image, image, moving_state=image_state, reference_state=image_state
        )


def test_transform_records_exact_revision_but_allows_same_frame_sibling_content():
    phantom = translation_pair(noisy=False)
    moving_state, reference_state = pair_states(phantom)
    transform, _ = estimate_registration(
        phantom.moving,
        phantom.reference,
        moving_state=moving_state,
        reference_state=reference_state,
    )
    settings = dict(transform.settings)
    assert len(settings["moving_content_sha256"]) == 64
    assert settings["moving_content_sha256"] != settings["reference_content_sha256"]
    # A same-frame binary mask is intentionally different content.
    mask = phantom.moving > 0.1
    mask_state = replace(moving_state, dtype="bool", kind="binary mask")
    result, _ = apply_transform(mask, transform, image_state=mask_state)
    assert result.dtype == bool


def test_frame_reference_time_can_be_nonzero_and_spatial_order_interleaved():
    phantom = drift_series()
    # T Z C Y X is legal because semantic axis mappings are explicit.
    reordered = phantom.moving.transpose(0, 2, 1, 3, 4)
    image_state = state(
        reordered, "TZCYX", spacing=phantom.spacing, origin=phantom.origin
    )
    transform, _ = estimate_registration(
        reordered,
        moving_state=image_state,
        mode="Time series",
        reference_time=3,
        precision=20,
    )
    reference_matrix = np.asarray(phantom.moving_to_reference[3])
    for t, matrix in enumerate(transform.matrices):
        expected = np.linalg.inv(reference_matrix) @ phantom.moving_to_reference[t]
        np.testing.assert_allclose(matrix, expected, atol=0.08)
    aligned, valid = apply_transform(reordered, transform, image_state=image_state)
    assert aligned.shape == reordered.shape
    np.testing.assert_allclose(aligned[3], reordered[3], atol=1e-10)
    assert valid[3].all()
