"""Ground-truth fixture tests do not depend on the estimator under test."""

import json

import numpy as np
import pytest

from napari_vipp._sample_data import make_registration_sample_data, make_sample_data
from napari_vipp.core.metadata import DEFERRED_VALUE_RANGE, image_state_from_array
from napari_vipp.core.registration_samples import (
    affine_pair,
    affine_volume_pair,
    drift_series,
    landmark_errors,
    rigid_image_pair,
    rigid_volume_pair,
    translation_pair,
)


@pytest.mark.parametrize(
    "factory",
    [
        translation_pair,
        rigid_volume_pair,
        affine_pair,
        drift_series,
        rigid_image_pair,
        affine_volume_pair,
    ],
)
def test_phantoms_are_finite_read_only_reproducible_and_independently_calibrated(
    factory,
):
    phantom = factory()
    repeat = factory()
    for name in ("reference", "moving", "labels"):
        values = getattr(phantom, name)
        assert not values.flags.writeable
        assert np.all(np.isfinite(values))
        np.testing.assert_array_equal(values, getattr(repeat, name))
        with pytest.raises(ValueError):
            values.flat[0] = 123
    assert set(np.unique(phantom.labels)) == set(range(8))
    assert len(set(phantom.spacing)) > 1
    assert any(phantom.origin)
    for time_index, matrix in enumerate(phantom.moving_to_reference):
        np.testing.assert_allclose(
            landmark_errors(matrix, phantom, time_index=time_index), 0, atol=1e-12
        )


def test_translation_ground_truth_has_correct_direction_and_physical_units():
    phantom = translation_pair(noisy=False)
    matrix = np.asarray(phantom.moving_to_reference[0])
    np.testing.assert_allclose(matrix[:2, -1], (-1.7, 1.95))
    delta = np.asarray(phantom.moving_landmarks[0]) - phantom.reference_landmarks
    np.testing.assert_allclose(
        delta / phantom.spacing, np.tile((4.25, -6.5), (7, 1)), atol=1e-12
    )
    assert np.mean((phantom.reference - phantom.moving) ** 2) > 0.005
    assert not np.array_equal(translation_pair().moving, phantom.moving)


def test_true_3d_rotation_is_rigid_in_physical_not_index_coordinates():
    phantom = rigid_volume_pair()
    matrix = np.asarray(phantom.moving_to_reference[0])
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.count_nonzero(np.abs(rotation - np.diag(np.diag(rotation))) > 0.01) == 6
    moving = np.asarray(phantom.moving_landmarks[0])
    reference = np.asarray(phantom.reference_landmarks)
    np.testing.assert_allclose(
        np.linalg.norm(moving[1:] - moving[0], axis=1),
        np.linalg.norm(reference[1:] - reference[0], axis=1),
    )
    index_rotation = (
        np.diag(1 / np.asarray(phantom.spacing)) @ rotation @ np.diag(phantom.spacing)
    )
    assert not np.allclose(index_rotation @ index_rotation.T, np.eye(3))


def test_time_series_moves_xyz_as_one_volume_and_shares_motion_across_channels():
    phantom = drift_series()
    assert phantom.moving.shape == (6, 2, 24, 64, 80)
    assert phantom.labels.shape == (6, 24, 64, 80)
    assert phantom.axes == "TCZYX"
    np.testing.assert_array_equal(phantom.reference, phantom.moving[0, 0])
    assert not np.array_equal(phantom.moving[:, 0], phantom.moving[:, 1])
    np.testing.assert_array_equal(phantom.moving_to_reference[0], np.eye(4))
    expected = (
        (0, 0, 0),
        (0.6, -1.25, 2.5),
        (1.2, -2.5, 4.5),
        (-0.4, -0.75, 2),
        (0.8, 1.5, -2),
        (1.5, 2.25, -3.5),
    )
    for matrix, displacement in zip(phantom.moving_to_reference, expected, strict=True):
        np.testing.assert_allclose(
            -np.asarray(matrix)[:3, 3] / phantom.spacing, displacement, atol=1e-12
        )


@pytest.mark.parametrize("factory", (make_registration_sample_data, make_sample_data))
def test_sample_catalogue_defers_metadata_statistics_until_execution(
    monkeypatch, factory,
):
    def unexpected_statistics(*_args, **_kwargs):
        raise AssertionError("Listing bundled samples must not scan value statistics")

    with monkeypatch.context() as blocked:
        blocked.setattr(
            "napari_vipp.core.metadata._value_range_label", unexpected_statistics
        )
        blocked.setattr(
            "napari_vipp.core.metadata._value_pattern_label", unexpected_statistics
        )
        samples = factory()

    registration_samples = [
        item for item in samples if item[1]["name"].startswith("VIPP registration ")
    ]
    assert len(registration_samples) == 6
    for data, kwargs, kind in registration_samples:
        metadata = kwargs["metadata"]
        carried = metadata["vipp_image_state"]
        assert carried["value_range"] == DEFERRED_VALUE_RANGE
        assert carried["value_pattern"] == ""
        calculated = image_state_from_array(data, layer_metadata=metadata)
        assert calculated.value_range != DEFERRED_VALUE_RANGE
        assert calculated.source.source_uuid == carried["source"]["source_uuid"]
        assert calculated.kind == carried["kind"]
        if kind == "labels":
            assert calculated.kind == "label image"
        assert not data.flags.writeable


def test_sample_metadata_preserves_explicit_axes_origin_and_shared_label_frame():
    samples = make_registration_sample_data()
    assert len(samples) == 6
    states = []
    for data, kwargs, _kind in samples:
        metadata = kwargs["metadata"]
        state = image_state_from_array(
            data, layer_metadata=metadata, source_name=kwargs["name"]
        )
        assert state.spatial_axes_explicit
        assert state.source.source_uuid
        assert any(axis.translation for axis in state.axes if axis.type == "space")
        assert all(
            axis.unit == "micrometer" for axis in state.axes if axis.type == "space"
        )
        assert (
            metadata["registration_ground_truth"]["matrix_direction"]
            == "moving to reference"
        )
        json.dumps(metadata, allow_nan=False)
        states.append(state)
    assert states[-1].source.source_uuid == states[-2].source.source_uuid
    assert states[-2].axis_order == "TCZYX"
    assert states[-1].axis_order == "TZYX"
    assert states[-1].kind == "label image"
    assert states[0].source.source_uuid != states[1].source.source_uuid
