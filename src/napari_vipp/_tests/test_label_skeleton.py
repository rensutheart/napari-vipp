from __future__ import annotations

import math

import numpy as np
import pytest
from skimage.morphology import skeletonize

import napari_vipp.core.label_skeleton as label_skeleton
from napari_vipp.core.label_skeleton import (
    analyze_skeleton_per_label,
    skeletonize_labels,
)
from napari_vipp.core.operations import analyze_skeleton
from napari_vipp.core.progress import OperationCancelled, ProgressContext


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint64])
@pytest.mark.parametrize("ndim", [2, 3])
def test_each_label_matches_independent_full_image_thinning(dtype, ndim):
    shape = (16, 19) if ndim == 2 else (9, 16, 19)
    labels = np.zeros(shape, dtype=dtype)
    # Touching interfaces, image-border objects and disconnected equal IDs.
    labels[(slice(None),) * (ndim - 2) + (slice(0, 9), slice(0, 6))] = 7
    labels[(slice(None),) * (ndim - 2) + (slice(0, 9), slice(6, 12))] = 12
    labels[(slice(None),) * (ndim - 2) + (slice(12, 14), slice(16, 19))] = 7
    labels[(0,) * ndim] = 23
    before = labels.copy()
    labels.flags.writeable = False
    actual = skeletonize_labels(
        labels, spatial_mode=f"{ndim}D {'YX' if ndim == 2 else 'ZYX'}"
    )
    expected = np.zeros_like(labels)
    method = "zhang" if ndim == 2 else "lee"
    for label_id in (7, 12, 23):
        expected[skeletonize(labels == label_id, method=method)] = label_id
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(labels, before)
    assert actual.dtype == dtype
    assert not np.shares_memory(actual, labels)
    assert actual.flags.owndata


def test_touching_skeletons_are_measured_independently():
    labels = np.zeros((6, 8), dtype=np.uint16)
    labels[2, 1:6] = 20
    labels[3, 1:6] = 40
    summary, components = analyze_skeleton_per_label(labels, skeleton=labels)
    assert [row["label_id"] for row in summary.records()] == [20, 40]
    for row in summary.records():
        assert row["skeleton_component_count"] == 1
        assert row["skeleton_voxel_count"] == 5
        assert row["endpoint_voxel_count"] == 2
        assert row["junction_voxel_count"] == 0
        assert row["skeleton_length_pixels"] == 4
    assert components.row_count == 2


@pytest.mark.parametrize("method", ["Zhang 2D", "unused method"])
def test_supplied_skeleton_ignores_thinning_method(method):
    labels = np.zeros((5, 6, 7), dtype=np.uint16)
    labels[2, 3, 1:6] = 8
    expected = analyze_skeleton_per_label(
        labels, skeleton=labels, spatial_mode="3D ZYX"
    )
    actual = analyze_skeleton_per_label(
        labels, skeleton=labels, spatial_mode="3D ZYX", method=method
    )
    assert actual == expected
    with pytest.raises(ValueError, match="Skeletonize|skeletonization"):
        analyze_skeleton_per_label(labels, spatial_mode="3D ZYX", method=method)


def test_fragments_and_isolates_retain_original_label_and_summary_totals():
    labels = np.zeros((12, 13), dtype=np.int32)
    labels[2, 1:6] = 42
    labels[8, 9] = 42
    summary, components = analyze_skeleton_per_label(labels, skeleton=labels)
    row = summary.records()[0]
    assert row["label_id"] == 42
    assert row["skeleton_component_count"] == 2
    assert row["isolated_node_count"] == 1
    assert row["skeleton_voxel_count"] == 6
    assert row["skeleton_length_pixels"] == 4
    assert [r["label_id"] for r in components.records()] == [42, 42]
    assert [r["component_id"] for r in components.records()] == [1, 2]
    assert [r["component_count_in_label"] for r in components.records()] == [2, 2]
    assert [r["component_voxel_fraction_in_label"] for r in components.records()] == [
        5 / 6,
        1 / 6,
    ]
    for name in label_skeleton._COUNT_COLUMNS:
        assert row[name] == sum(r[name] for r in components.records())


def test_sparse_uint64_ids_remain_exact_in_images_and_both_tables():
    labels = np.zeros((8, 12), dtype=np.uint64)
    ids = (1, 2**63 + 1, 2**64 - 1)
    for index, label_id in enumerate(ids):
        labels[2 + 2 * index, 1:10] = label_id
    actual = skeletonize_labels(labels)
    np.testing.assert_array_equal(actual, labels)
    summary, components = analyze_skeleton_per_label(labels, skeleton=actual)
    for table in (summary, components):
        assert tuple(r["label_id"] for r in table.records()) == ids
        assert all(isinstance(r["label_id"], int) for r in table.records())


def test_provided_skeleton_can_leave_an_original_object_empty():
    labels = np.zeros((5, 10), dtype=np.uint32)
    labels[1:4, 1:4] = 8
    labels[1:4, 6:9] = 12
    skeleton = np.zeros_like(labels)
    skeleton[2, 1:4] = 8
    labels.flags.writeable = False
    skeleton.flags.writeable = False
    summary, components = analyze_skeleton_per_label(labels, skeleton=skeleton)
    empty = summary.records()[1]
    assert empty["label_id"] == 12
    assert empty["skeleton_status"] == "empty"
    assert empty["skeleton_component_count"] == 0
    assert empty["skeleton_length_pixels"] == 0
    assert all(empty[name] == 0 for name in label_skeleton._COUNT_COLUMNS)
    assert components.row_count == 1
    assert components.records()[0]["label_id"] == 8


def test_lee_vanished_object_still_has_summary_row():
    labels = np.zeros((8, 8, 8), dtype=np.int16)
    labels[2:6, 2:6, 2:6] = 8
    expected_skeleton = skeletonize(labels == 8, method="lee")
    summary, components = analyze_skeleton_per_label(labels, spatial_mode="3D ZYX")
    assert summary.row_count == 1
    assert summary.records()[0]["label_id"] == 8
    assert summary.records()[0]["skeleton_voxel_count"] == np.count_nonzero(
        expected_skeleton
    )
    assert summary.records()[0]["skeleton_status"] == (
        "ok" if expected_skeleton.any() else "empty"
    )
    if not expected_skeleton.any():
        assert components.row_count == 0


def test_anisotropic_diagonal_length_uses_spacing_and_compatible_units():
    labels = np.zeros((5, 5, 5), dtype=np.uint16)
    labels[1, 1, 1] = 3
    labels[2, 2, 2] = 3
    summary, components = analyze_skeleton_per_label(
        labels,
        skeleton=labels,
        axis_names=("z", "y", "x"),
        axis_scales=(2000, 3, 4),
        axis_units=("nm", "µm", "um"),
    )
    for table in (summary, components):
        row = table.records()[0]
        assert row["skeleton_length_voxels"] == pytest.approx(math.sqrt(3))
        assert row["skeleton_length_physical"] == pytest.approx(math.sqrt(29))
        assert row["physical_unit"] == "um"
        assert table.unit_for("skeleton_length_physical") == "um"


@pytest.mark.parametrize("axis_scales", [None, (2, 3), (1, 1)])
def test_missing_units_never_invent_physical_calibration(axis_scales):
    labels = np.ones((2, 4), dtype=np.uint8)
    summary, components = analyze_skeleton_per_label(labels, axis_scales=axis_scales)
    for table in (summary, components):
        assert "skeleton_length_pixels" in table.columns
        assert "skeleton_length_physical" not in table.columns
        assert "physical_unit" not in table.columns


@pytest.mark.parametrize("unit", ["voxel", "voxels", "pixel", "pixels"])
def test_index_units_never_invent_physical_calibration(unit):
    labels = np.zeros((5, 6, 7), dtype=np.uint8)
    labels[2, 3, 1:6] = 8
    summary, components = analyze_skeleton_per_label(
        labels,
        skeleton=labels,
        spatial_mode="3D ZYX",
        axis_scales=(1, 1, 1),
        axis_units=(unit,) * 3,
    )
    for table in (summary, components):
        assert "skeleton_length_physical" not in table.columns
        assert table.records()[0]["skeleton_length_voxels"] == 4


@pytest.mark.parametrize(
    "scales,units",
    [
        ((1, -1), ("um", "um")),
        ((1, 0), ("um", "um")),
        ((1, np.nan), ("um", "um")),
        ((1, np.inf), ("um", "um")),
        ((1, 1), ("um", None)),
        ((1, 1), ("um", "s")),
        ((1, 1), ("arbitrary", "arbitrary")),
        (None, ("um", "um")),
        ((1,), ("um", "um")),
        ((1, 1), ("um",)),
    ],
)
def test_invalid_or_incomplete_physical_calibration_fails(scales, units):
    with pytest.raises(ValueError):
        analyze_skeleton_per_label(
            np.ones((3, 3), dtype=np.int32), axis_scales=scales, axis_units=units
        )


@pytest.mark.parametrize("permutation", [(0, 1, 2, 3), (2, 0, 3, 1)])
def test_tzyx_and_nontrailing_spatial_axes_preserve_block_identity(permutation):
    original = np.zeros((2, 7, 9, 11), dtype=np.uint64)
    original[0, 2:5, 3:6, 1:8] = 101
    original[1, 1:4, 4:7, 2:9] = 101
    labels = np.transpose(original, permutation)
    names = tuple(("t", "z", "y", "x")[index] for index in permutation)
    types = tuple(("time", "space", "space", "space")[index] for index in permutation)
    scales = tuple((1, 2, 3, 4)[index] for index in permutation)
    units = tuple(("s", "um", "um", "um")[index] for index in permutation)
    skeleton = skeletonize_labels(labels, axis_names=names, axis_types=types)
    expected = np.zeros_like(original)
    for time in range(2):
        expected[time][skeletonize(original[time] == 101, method="lee")] = 101
    np.testing.assert_array_equal(skeleton, np.transpose(expected, permutation))
    direct = analyze_skeleton_per_label(
        labels,
        axis_names=names,
        axis_types=types,
        axis_scales=scales,
        axis_units=units,
    )
    provided = analyze_skeleton_per_label(
        labels,
        skeleton=skeleton,
        axis_names=names,
        axis_types=types,
        axis_scales=scales,
        axis_units=units,
    )
    assert direct == provided
    assert [r["t_index"] for r in direct[0].records()] == [0, 1]
    assert [r["label_id"] for r in direct[0].records()] == [101, 101]


def test_explicit_2d_processes_z_slices_independently():
    labels = np.zeros((2, 5, 8), dtype=np.uint16)
    labels[:, 2, 1:6] = 9
    summary, components = analyze_skeleton_per_label(
        labels, skeleton=labels, spatial_mode="2D YX", axis_names=("z", "y", "x")
    )
    assert [r["z_index"] for r in summary.records()] == [0, 1]
    assert components.row_count == 2
    assert all(r["skeleton_length_pixels"] == 4 for r in summary.records())


@pytest.mark.parametrize(
    "data",
    [
        np.ones((3, 3), dtype=bool),
        np.ones((3, 3), dtype=float),
        np.full((3, 3), -1),
        np.array([[np.nan]]),
        np.ones(3, dtype=int),
    ],
)
@pytest.mark.parametrize("function", [skeletonize_labels, analyze_skeleton_per_label])
def test_invalid_label_types_or_values_fail_without_mutation(data, function):
    original = data.copy()
    data.flags.writeable = False
    with pytest.raises(ValueError):
        function(data)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "defect", ["shape", "binary", "different_id", "outside", "wide_rounding"]
)
def test_supplied_skeleton_must_preserve_exact_labels_on_same_grid(defect):
    labels = np.zeros((5, 6), dtype=np.uint64)
    labels[2, 1:5] = 2**63 + 1 if defect == "wide_rounding" else 3
    skeleton = labels.copy()
    if defect == "shape":
        skeleton = skeleton[:3]
    elif defect == "binary":
        skeleton = skeleton.astype(bool)
    elif defect == "different_id":
        skeleton[2, 2] = 4
    elif defect == "outside":
        skeleton[0, 0] = 3
    else:
        skeleton = np.zeros((5, 6), dtype=np.int64)
        skeleton[2, 2] = 2**63 - 1
    with pytest.raises(ValueError):
        analyze_skeleton_per_label(labels, skeleton=skeleton)


@pytest.mark.parametrize(
    "shape,mode", [((0, 5, 6), "2D YX"), ((5, 6), "2D YX"), ((2, 3, 4), "3D ZYX")]
)
def test_empty_label_populations_have_stable_schemas_and_complete_progress(shape, mode):
    labels = np.zeros(shape, dtype=np.uint64)
    updates = []
    summary, components = analyze_skeleton_per_label(
        labels,
        spatial_mode=mode,
        source_name="test image",
        progress=ProgressContext(reporter=updates.append),
    )
    assert summary.row_count == components.row_count == 0
    assert "label_id" in summary.columns
    assert "label_id" in components.columns
    assert summary.source_name == components.source_name == "test image"
    assert updates[0].current == 0
    assert updates[-1].current == updates[-1].total
    np.testing.assert_array_equal(skeletonize_labels(labels, spatial_mode=mode), labels)


@pytest.mark.parametrize(
    "function,message",
    [
        (skeletonize_labels, "skeletonized label"),
        (analyze_skeleton_per_label, "analyzed label"),
    ],
)
def test_cancellation_between_labels_does_not_mutate_input(function, message):
    labels = np.zeros((9, 11), dtype=np.int32)
    labels[2, 1:8] = 1
    labels[6, 1:8] = 2
    before = labels.copy()
    cancelled = False
    visited = []

    def report(update):
        nonlocal cancelled
        if message in update.message:
            visited.append(update.message)
            cancelled = True

    with pytest.raises(OperationCancelled):
        function(
            labels,
            progress=ProgressContext(cancelled=lambda: cancelled, reporter=report),
        )
    assert len(visited) == 1
    np.testing.assert_array_equal(labels, before)


def test_bounds_scan_is_chunked_and_independent_of_largest_id(monkeypatch):
    monkeypatch.setattr(label_skeleton, "_SCAN_CHUNK_SIZE", 7)
    labels = np.zeros((9, 11), dtype=np.uint64)
    labels[1:8, 1:4] = 2**64 - 1
    labels[2:6, 4:9] = 3
    skeleton = skeletonize_labels(labels)
    expected = np.zeros_like(labels)
    for label_id in (3, 2**64 - 1):
        expected[skeletonize(labels == label_id)] = label_id
    np.testing.assert_array_equal(skeleton, expected)


def test_existing_graph_semantics_match_for_a_loop_and_branched_fragment():
    labels = np.zeros((13, 14), dtype=np.uint32)
    labels[2, 2:6] = labels[5, 2:6] = 30
    labels[2:6, 2] = labels[2:6, 5] = 30
    labels[9, 8:13] = 30
    labels[8:12, 10] = 30
    legacy = analyze_skeleton(labels > 0).records()
    summary, components = analyze_skeleton_per_label(labels, skeleton=labels)
    assert len(legacy) == components.row_count
    for original, actual in zip(legacy, components.records(), strict=True):
        for name in (*label_skeleton._COUNT_COLUMNS, "skeleton_length_pixels"):
            assert original[name] == actual[name]
    assert summary.records()[0]["cycle_count"] >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis_names": ("z", "x")},
        {"axis_names": ("x",)},
        {"axis_types": ("space",)},
        {"spatial_mode": "3D ZYX"},
        {"axis_names": ("y", "x"), "axis_types": ("time", "space")},
        {"resolved_spatial_ndim": 1},
        {"method": "not a method"},
    ],
)
def test_invalid_spatial_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        skeletonize_labels(np.zeros((6, 7), dtype=np.int16), **kwargs)


def test_auto_without_spatial_metadata_rejects_ambiguous_stack():
    with pytest.raises(ValueError, match="explicit spatial axis"):
        skeletonize_labels(np.zeros((2, 6, 7), dtype=np.uint16))
