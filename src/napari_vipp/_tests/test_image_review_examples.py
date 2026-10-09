"""Deterministic, aligned review fixtures; no scientific index is implied."""

from pathlib import Path

import numpy as np
import pytest

from napari_vipp._sample_data import make_image_review_sample_data
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.workflow import load_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path

ROOT = Path(__file__).resolve().parents[3]
REVIEW_EXAMPLES = (
    "review-channels-2d",
    "review-mask-3d",
    "review-labels-time-series",
    "review-rgb-index-3d",
)


def test_review_samples_are_repeatable_explicit_and_nonpreferred():
    first = make_image_review_sample_data()
    second = make_image_review_sample_data()
    assert len(first) == 8
    for (data, kwargs, layer_kind), (again, again_kwargs, again_kind) in zip(
        first, second, strict=True
    ):
        np.testing.assert_array_equal(data, again)
        assert kwargs == again_kwargs
        assert layer_kind == again_kind
        metadata = kwargs["metadata"]
        state = ImageState.from_dict(metadata["vipp_image_state"])
        assert state.axes_explicit
        assert state.shape == data.shape
        assert not kwargs["visible"]
        assert not metadata["napari_vipp_preferred_input"]
        assert metadata["image_review_ground_truth"]["racc_computed"] is False


def test_review_sample_pairs_share_calibrated_grid_without_guessed_rgb():
    samples = make_image_review_sample_data()
    expected = (
        ((80, 104), "intensity image", "intensity image"),
        ((48, 64, 80), "intensity image", "binary mask"),
        ((4, 12, 40, 48), "intensity image", "label image"),
        ((48, 64, 80), "RGB image", "intensity image"),
    )
    for index, (shape, kind_a, kind_b) in enumerate(expected):
        a, b = samples[2 * index : 2 * index + 2]
        state_a = ImageState.from_dict(a[1]["metadata"]["vipp_image_state"])
        state_b = ImageState.from_dict(b[1]["metadata"]["vipp_image_state"])
        assert state_a.kind == kind_a
        assert state_b.kind == kind_b
        axes_a = tuple(axis for axis in state_a.axes if axis.type != "channel")
        axes_b = tuple(axis for axis in state_b.axes if axis.type != "channel")
        assert axes_a == axes_b
        assert b[0].shape == shape
        assert a[0].shape == shape + ((3,) if kind_a == "RGB image" else ())
    rgb = samples[6]
    assert rgb[1]["rgb"] is True
    assert rgb[0].dtype == np.uint8
    assert (
        ImageState.from_dict(rgb[1]["metadata"]["vipp_image_state"]).axes[-1].name
        == "rgb"
    )
    assert set(np.unique(samples[5][0])) == {0, 7, 42}
    assert samples[3][0].dtype == bool
    index = samples[7][0]
    assert index.dtype == np.float32
    assert 0 < index.max() <= 1
    assert np.all(index[~samples[3][0]] == 0)


def test_mask_review_has_analytical_physical_spheres_without_smoothing():
    samples = make_image_review_sample_data()
    intensity, intensity_kwargs, _ = samples[2]
    mask, mask_kwargs, _ = samples[3]
    expected_spacing = (0.6, 0.5, 0.4)
    assert intensity_kwargs["scale"] == mask_kwargs["scale"] == expected_spacing
    assert intensity_kwargs["translate"] == mask_kwargs["translate"] == (-2, 4, 8)
    truth = mask_kwargs["metadata"]["image_review_ground_truth"]
    assert truth["construction_geometry"] == "analytical physical spheres"
    assert truth["coordinate_tick_um"] == 0.1
    assert truth["centres_zyx_um"] == [[7.0, 14.5, 17.2], [17.2, 26.0, 30.8]]
    assert truth["radii_um"] == [5.4, 5.0]
    assert (
        min(
            2 * radius / spacing
            for radius in (5.4, 5.0)
            for spacing in expected_spacing
        )
        >= 16
    )
    coordinates = np.indices((48, 64, 80), dtype=np.int64)
    expected_mask = np.zeros(mask.shape, dtype=bool)
    expected_intensity = np.zeros(intensity.shape, dtype=np.float64)
    for centre, radius_ticks, amplitude in (
        ((15, 21, 23), 54, 1.0),
        ((32, 44, 57), 50, 0.75),
    ):
        distance_squared_ticks = sum(
            ((coordinates[axis] - centre[axis]) * step) ** 2
            for axis, step in enumerate((6, 5, 4))
        )
        expected_mask |= distance_squared_ticks <= radius_ticks**2
        expected_intensity += amplitude * np.exp(
            -distance_squared_ticks / radius_ticks**2
        )
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(intensity, expected_intensity.astype(np.float32))
    assert mask[15, 21, 23] and mask[32, 44, 57]
    assert mask[32, 54, 57]  # Exactly on the 5.0-micrometre sphere boundary.
    assert not mask[32, 55, 57]


@pytest.mark.parametrize("example_id", REVIEW_EXAMPLES)
def test_review_example_has_presentation_sink_and_mirrored_saved_recipe(example_id):
    spec = _example_workflow_by_id(example_id)
    assert spec is not None
    path = _example_workflow_path(spec)
    assert path.read_bytes() == (ROOT / "examples" / spec.filename).read_bytes()
    document = load_workflow(path)
    nodes = {node.id: node for node in document["nodes"]}
    assert nodes["review"].operation_id == "review_images"
    assert nodes["review"].params == {}
    assert {node.operation_id for node in nodes.values()} == {
        "input",
        "review_images",
    }
    sample_names = {kwargs["name"] for _, kwargs, _ in make_image_review_sample_data()}
    assert set(spec.samples) <= sample_names
    assert {nodes[key].params["sample_name"] for key in ("input", "image_b")} == set(
        spec.samples
    )
    assert document["metadata"]["vipp"]["inspector"]["selected_node_id"] == "review"
    expected_names = {
        "review-channels-2d": ("Red channel", "Green channel"),
        "review-mask-3d": ("Synthetic intensity", "Segmentation"),
        "review-labels-time-series": ("Synthetic intensity", "Object labels"),
        "review-rgb-index-3d": ("Original RGB", "Scalar index"),
    }
    name_a, name_b = expected_names[example_id]
    assert document["metadata"]["vipp"]["node_names"] == {
        "input": name_a,
        "image_b": name_b,
        "review": "Review Images",
    }
    recipe = document["metadata"]["vipp"]["inspector"]["image_reviews"]["review"]
    assert recipe["mode"] == "side-by-side"
    assert recipe["link_navigation"] is True
    if example_id == "review-rgb-index-3d":
        assert recipe["ndisplay"] == 2
        assert recipe["b"]["contrast_limits"] == [0, 1]
        assert recipe["b"]["lock_contrast"] is True
        assert recipe["b"]["threshold"] == 0.05
        assert "not a computed RACC" in " ".join(
            note["text"] for note in document["notes"]
        )
