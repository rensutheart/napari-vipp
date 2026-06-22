from __future__ import annotations

import numpy as np
import tifffile

from napari_vipp.core.metadata import (
    image_state_from_array,
    transform_image_state,
    transform_multi_input_image_state,
)
from napari_vipp.core.operations import (
    adaptive_gaussian_threshold,
    adaptive_mean_threshold,
    add_images,
    add_metadata_columns,
    analyze_skeleton,
    assign_channel_colors,
    average_blur,
    bilateral_filter,
    binary_threshold,
    black_hat,
    calculate_weighted_image,
    canny_edges,
    clear_border_objects,
    clip_intensity,
    closing,
    combine_channels,
    composite_to_rgb,
    convert_dtype,
    crop_stack,
    difference_of_gaussians_filter,
    dilate,
    erode,
    extract_channel,
    fill_holes,
    filter_labels_by_property,
    filter_labels_by_volume,
    gamma_correction,
    gaussian_blur,
    gaussian_blur_3d,
    hysteresis_threshold,
    isodata_threshold,
    label_connected_components,
    laplace_filter,
    li_threshold,
    linear_scale_offset,
    logical_and,
    logical_or,
    logical_xor,
    mask_image,
    measure_objects,
    measure_objects_with_intensity,
    median_filter,
    merge_tables,
    minimum_threshold,
    morphological_gradient,
    niblack_threshold,
    non_local_means_filter,
    normalize_image,
    opening,
    orthogonal_projection,
    otsu_threshold,
    project_image,
    ratio_image,
    relabel_sequential,
    remove_small_objects,
    reorder_axes,
    rescale_axes,
    rescale_intensity,
    sauvola_threshold,
    save_array_output,
    save_output,
    select_axis_slice,
    select_table_columns,
    set_pixel_size,
    skeletonize_mask,
    sobel_filter,
    split_channels,
    subtract_images,
    top_hat,
    triangle_threshold,
    unsharp_mask_filter,
    yen_threshold,
)
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.tables import save_table_output, table_from_columns


def test_vipp_operation_nodes_are_registered():
    expected = {
        "crop_stack",
        "linear_scale_offset",
        "gamma_correction",
        "average_blur",
        "gaussian_blur",
        "gaussian_blur_3d",
        "median_filter",
        "bilateral_filter",
        "non_local_means_filter",
        "difference_of_gaussians",
        "unsharp_mask",
        "sobel_filter",
        "laplace_filter",
        "binary_threshold",
        "hysteresis_threshold",
        "canny_edges",
        "adaptive_mean_threshold",
        "adaptive_gaussian_threshold",
        "sauvola_threshold",
        "niblack_threshold",
        "otsu_threshold",
        "triangle_threshold",
        "li_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
        "dilate",
        "erode",
        "opening",
        "closing",
        "top_hat",
        "black_hat",
        "morphological_gradient",
        "fill_holes",
        "remove_small_objects",
        "clear_border_objects",
        "label_connected_components",
        "filter_labels_by_volume",
        "filter_labels_by_property",
        "relabel_sequential",
        "extract_channel",
        "combine_channels",
        "calculate_weighted_image",
        "add_images",
        "subtract_images",
        "ratio_image",
        "mask_image",
        "measure_objects",
        "measure_objects_intensity",
        "skeletonize",
        "analyze_skeleton",
        "assign_channel_colors",
        "merge_tables",
        "add_metadata_columns",
        "select_table_columns",
        "logical_and",
        "logical_or",
        "logical_xor",
        "composite_to_rgb",
        "split_channels",
        "rescale_intensity",
        "normalize_image",
        "clip_intensity",
        "convert_dtype",
        "select_axis_slice",
        "reorder_axes",
        "set_pixel_size",
        "rescale_axes",
        "project_image",
        "orthogonal_projection",
        "save_output",
    }

    assert expected <= set(NODE_LIBRARY_BY_ID)


def test_pipeline_runs_mask_to_labels_to_label_volume_filter_in_3d():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[:, 1:4, 1:4] = 10
    data[1, 7, 7] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    filtered = pipeline.add_node("filter_labels_by_volume")
    relabeled = pipeline.add_node("relabel_sequential")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(filtered.id, "min_volume", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, filtered.id)
    pipeline.connect(filtered.id, relabeled.id)

    outputs = pipeline.run(
        data,
        input_metadata={"axes": "ZYX"},
        input_name="3D nuclei",
    )

    assert outputs[labels.id].dtype == np.int32
    assert set(np.unique(outputs[labels.id])) == {0, 1, 2}
    assert set(np.unique(outputs[filtered.id])) == {0, 1}
    assert set(np.unique(outputs[relabeled.id])) == {0, 1}
    assert pipeline.output_states[labels.id].kind == "label image"
    assert pipeline.nodes[labels.id].params["resolved_spatial_ndim"] == 3


def test_measure_objects_reports_3d_objects_with_physical_volume():
    labels = np.zeros((2, 3, 6, 7), dtype=np.int32)
    labels[0, :, 1:3, 2:5] = 1
    labels[1, 1, 3:5, 1:3] = 2

    table = measure_objects(
        labels,
        resolved_spatial_ndim=3,
        axis_names=("t", "z", "y", "x"),
        axis_types=("time", "space", "space", "space"),
        axis_scales=(1.0, 0.5, 0.2, 0.2),
        axis_units=("second", "micrometer", "micrometer", "micrometer"),
    )

    assert table.columns[:5] == (
        "t_index",
        "label_id",
        "volume_voxels",
        "volume_physical",
        "physical_unit",
    )
    assert table.row_count == 2
    records = table.records()
    assert records[0]["t_index"] == 0
    assert records[0]["label_id"] == 1
    assert records[0]["volume_voxels"] == 18
    assert np.isclose(records[0]["volume_physical"], 0.36)
    assert records[0]["physical_unit"] == "micrometer^3"
    assert records[1]["t_index"] == 1
    assert records[1]["volume_voxels"] == 4


def test_measure_objects_reports_selected_extended_2d_properties():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:8, 3:9] = 1

    table = measure_objects(
        labels,
        resolved_spatial_ndim=2,
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_2d_boundary_descriptors=True,
    )
    record = table.records()[0]

    assert table.row_count == 1
    assert record["bbox_area_pixels"] == 36
    assert record["filled_area_pixels"] == 36
    assert record["convex_area_pixels"] == 36
    assert record["solidity"] == 1.0
    assert record["major_axis_length_pixels"] > 0
    assert record["minor_axis_length_pixels"] > 0
    assert record["inertia_tensor_eigval_0"] >= 0
    assert record["inertia_tensor_eigval_1"] >= 0
    assert "eccentricity" in table.columns
    assert "orientation_radians" in table.columns
    assert record["perimeter_pixels"] > 0
    assert record["perimeter_crofton_pixels"] > 0
    assert table.unit_for("orientation_radians") == "radians"
    assert "shape descriptors" in table.table_kind


def test_measure_objects_omits_2d_only_properties_for_3d_measurements():
    labels = np.zeros((4, 12, 12), dtype=np.int32)
    labels[1:3, 2:8, 3:9] = 1

    table = measure_objects(
        labels,
        resolved_spatial_ndim=3,
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_2d_boundary_descriptors=True,
    )
    record = table.records()[0]

    assert table.row_count == 1
    assert record["bbox_volume_voxels"] == 72
    assert record["filled_volume_voxels"] == 72
    assert record["major_axis_length_voxels"] > 0
    assert record["minor_axis_length_voxels"] > 0
    assert "inertia_tensor_eigval_2" in table.columns
    assert "perimeter_pixels" not in table.columns
    assert "orientation_radians" not in table.columns
    assert "convex_volume_voxels" not in table.columns
    assert table.unit_for("inertia_tensor_eigval_2") == "voxels^2"


def test_measure_objects_with_intensity_reports_per_label_values():
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:3, 1:4] = 1
    labels[3:5, 4:6] = 2
    intensity = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)

    table = measure_objects_with_intensity(
        [labels, intensity],
        resolved_spatial_ndim=2,
    )
    records = table.records()

    assert table.row_count == 2
    assert "intensity_mean" in table.columns
    assert "intensity_sum" in table.columns
    assert records[0]["label_id"] == 1
    assert np.isclose(records[0]["intensity_mean"], intensity[labels == 1].mean())
    assert np.isclose(records[0]["intensity_sum"], intensity[labels == 1].sum())
    assert records[1]["label_id"] == 2
    assert records[1]["intensity_min"] == float(intensity[labels == 2].min())
    assert records[1]["intensity_max"] == float(intensity[labels == 2].max())


def test_measure_objects_with_intensity_can_include_extended_properties():
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:3, 1:4] = 1
    intensity = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)

    table = measure_objects_with_intensity(
        [labels, intensity],
        resolved_spatial_ndim=2,
        include_axis_descriptors=True,
    )
    record = table.records()[0]

    assert "intensity_mean" in table.columns
    assert "major_axis_length_pixels" in table.columns
    assert record["major_axis_length_pixels"] > 0


def test_filter_labels_by_property_filters_using_measurement_table():
    labels = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 2, 2],
            [0, 3, 3, 3],
        ],
        dtype=np.int32,
    )
    table = table_from_columns(
        {
            "label_id": [1, 2, 3],
            "area_pixels": [3, 3, 3],
            "intensity_mean": [5.0, 20.0, 40.0],
        }
    )

    filtered = filter_labels_by_property(
        [labels, table],
        property_column="intensity_mean",
        min_value=10,
        max_value=30,
    )

    np.testing.assert_array_equal(
        filtered,
        np.where(labels == 2, labels, 0),
    )


def test_filter_labels_by_property_matches_leading_axis_indices():
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 1:3, 1:3] = 1
    table = table_from_columns(
        {
            "t_index": [0, 1],
            "label_id": [1, 1],
            "intensity_mean": [5.0, 20.0],
        }
    )

    filtered = filter_labels_by_property(
        [labels, table],
        property_column="intensity_mean",
        min_value=10,
        resolved_spatial_ndim=2,
        axis_names=("t", "y", "x"),
        axis_types=("time", "space", "space"),
    )

    assert not filtered[0].any()
    np.testing.assert_array_equal(filtered[1], labels[1])


def test_merge_tables_joins_on_identity_columns_and_suffixes_duplicates():
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:3, 1:4] = 1
    labels[3:5, 4:6] = 2
    intensity = np.arange(labels.size, dtype=np.float32).reshape(labels.shape)
    morphology = measure_objects(labels, resolved_spatial_ndim=2)
    intensity_table = measure_objects_with_intensity(
        [labels, intensity],
        resolved_spatial_ndim=2,
    )

    merged = merge_tables(
        [morphology, intensity_table],
        input_count=2,
        join_mode="Left join",
        join_keys="auto",
    )
    records = merged.records()

    assert merged.row_count == 2
    assert merged.columns[:2] == ("label_id", "area_pixels")
    assert "area_pixels_table2" in merged.columns
    assert "intensity_mean" in merged.columns
    assert records[0]["label_id"] == 1
    assert records[0]["intensity_sum"] == float(intensity[labels == 1].sum())
    assert records[1]["label_id"] == 2


def test_merge_tables_can_join_equal_length_tables_by_row_position():
    first = table_from_columns(
        {"volume": [10, 20]},
        table_kind="synthetic measurements",
    )
    second = table_from_columns(
        {"condition": ["control", "treated"]},
        table_kind="sample metadata",
    )

    merged = merge_tables(
        [first, second],
        input_count=2,
        join_mode="Inner join",
        join_keys="",
    )

    assert merged.row_count == first.row_count
    assert merged.records()[0]["condition"] == "control"


def test_add_metadata_columns_appends_constant_values_and_blocks_collisions():
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:3, 1:4] = 1
    table = measure_objects(labels, resolved_spatial_ndim=2)

    annotated = add_metadata_columns(
        table,
        metadata_columns="condition=control, replicate=1",
    )
    records = annotated.records()

    assert annotated.columns[-2:] == ("condition", "replicate")
    assert records[0]["condition"] == "control"
    assert records[0]["replicate"] == "1"
    try:
        add_metadata_columns(annotated, metadata_columns="condition=drug")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected duplicate metadata column to be rejected")


def test_select_table_columns_keeps_drops_and_reorders_columns():
    table = table_from_columns(
        {
            "label_id": [1, 2],
            "area_pixels": [6, 4],
            "intensity_mean": [10.0, 20.0],
            "condition": ["control", "treated"],
        },
        column_units={"area_pixels": "pixels", "intensity_mean": "intensity"},
    )

    kept = select_table_columns(
        table,
        columns="intensity_mean, label_id",
    )
    dropped = select_table_columns(
        table,
        columns="condition",
        selection_mode="Drop listed columns",
    )
    reordered = select_table_columns(
        table,
        columns="condition, label_id",
        append_unlisted="yes",
    )

    assert kept.columns == ("intensity_mean", "label_id")
    assert kept.rows[0] == (10.0, 1)
    assert kept.unit_for("intensity_mean") == "intensity"
    assert dropped.columns == ("label_id", "area_pixels", "intensity_mean")
    assert reordered.columns == (
        "condition",
        "label_id",
        "area_pixels",
        "intensity_mean",
    )


def test_pipeline_measure_objects_creates_table_state():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[:, 1:4, 1:4] = 10
    data[1, 7, 7] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)

    outputs = pipeline.run(data, input_metadata={"axes": "ZYX"})
    table = outputs[measurements.id]
    state = pipeline.output_states[measurements.id]

    assert table.row_count == 2
    assert state.kind == "measurement table"
    assert state.row_count == 2
    assert "volume_voxels" in state.columns
    assert state.history[-1] == "Measure Objects: measured 2 objects"


def test_pipeline_measure_objects_with_intensity_uses_named_input_ports():
    data = np.zeros((7, 7), dtype=np.float32)
    data[1:3, 1:4] = 10
    data[4:6, 4:6] = 20
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects_intensity")
    pipeline.set_param(threshold.id, "threshold", 5)

    assert [port.name for port in pipeline.input_ports(measurements.id)] == [
        "labels",
        "intensity",
    ]
    assert [port.input_type for port in pipeline.input_ports(measurements.id)] == [
        "labels",
        "image",
    ]
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    label_result = pipeline.connect(labels.id, measurements.id)
    intensity_result = pipeline.connect("input", measurements.id)
    bad_result = pipeline.connect("input", measurements.id, target_port=0)

    outputs = pipeline.run(data, input_metadata={"axes": "YX"})
    table = outputs[measurements.id]
    records = table.records()

    assert label_result.success
    assert label_result.connection.target_port == 0
    assert intensity_result.success
    assert intensity_result.connection.target_port == 1
    assert not bad_result.success
    assert table.row_count == 2
    assert records[0]["intensity_mean"] == 10.0
    assert records[1]["intensity_mean"] == 20.0
    assert pipeline.output_states[measurements.id].history[-1] == (
        "Measure Objects + Intensity: measured 2 objects"
    )


def test_pipeline_filter_labels_by_property_uses_named_input_ports():
    data = np.zeros((8, 8), dtype=np.float32)
    data[1:4, 1:4] = 10
    data[6, 6] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    filtered = pipeline.add_node("filter_labels_by_property")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(filtered.id, "property_column", "area_pixels")
    pipeline.set_param(filtered.id, "min_value", 5)

    assert [port.name for port in pipeline.input_ports(filtered.id)] == [
        "labels",
        "table",
    ]
    assert [port.input_type for port in pipeline.input_ports(filtered.id)] == [
        "labels",
        "table",
    ]
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    label_result = pipeline.connect(labels.id, filtered.id, target_port=0)
    table_result = pipeline.connect(measurements.id, filtered.id, target_port=1)
    bad_result = pipeline.connect("input", filtered.id, target_port=1)

    outputs = pipeline.run(data, input_metadata={"axes": "YX"})

    assert label_result.success
    assert table_result.success
    assert not bad_result.success
    assert set(np.unique(outputs[labels.id])) == {0, 1, 2}
    assert set(np.unique(outputs[filtered.id])) == {0, 1}
    assert pipeline.output_states[filtered.id].kind == "label image"
    assert pipeline.output_states[filtered.id].history[-1] == (
        "Filter Labels By Property: filtered by area_pixels"
    )


def test_pipeline_mask_image_uses_named_image_and_mask_ports():
    data = np.zeros((5, 6), dtype=np.float32)
    data[1:4, 2:5] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(masked.id, "outside_value", -1)

    input_ports = pipeline.input_ports(masked.id)
    assert [port.name for port in input_ports] == ["image", "mask"]
    assert [port.label for port in input_ports] == ["Image", "Mask"]
    assert [port.input_type for port in input_ports] == ["image", "mask_or_labels"]
    assert "input_count" not in pipeline.nodes[masked.id].params

    pipeline.connect("input", threshold.id)
    image_result = pipeline.connect("input", masked.id, target_port=0)
    mask_result = pipeline.connect(threshold.id, masked.id, target_port=1)
    bad_result = pipeline.connect("input", masked.id, target_port=1)

    outputs = pipeline.run(data, input_metadata={"axes": "YX"})
    output = outputs[masked.id]

    assert image_result.success
    assert mask_result.success
    assert not bad_result.success
    assert output.dtype == data.dtype
    assert output[0, 0] == -1
    assert output[2, 3] == 10
    assert pipeline.output_states[masked.id].history[-1] == "Mask Image: applied mask"


def test_pipeline_merges_and_annotates_measurement_tables():
    data = np.zeros((7, 7), dtype=np.float32)
    data[1:3, 1:4] = 10
    data[4:6, 4:6] = 20
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    morphology = pipeline.add_node("measure_objects")
    intensity = pipeline.add_node("measure_objects_intensity")
    merged = pipeline.add_node("merge_tables")
    annotated = pipeline.add_node("add_metadata_columns")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(annotated.id, "metadata_columns", "condition=demo")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, morphology.id)
    pipeline.connect(labels.id, intensity.id, target_port=0)
    pipeline.connect("input", intensity.id, target_port=1)
    pipeline.connect(morphology.id, merged.id, target_port=0)
    pipeline.connect(intensity.id, merged.id, target_port=1)
    pipeline.connect(merged.id, annotated.id)

    outputs = pipeline.run(data, input_metadata={"axes": "YX"})
    table = outputs[annotated.id]
    state = pipeline.output_states[annotated.id]

    assert table.row_count == 2
    assert "intensity_mean" in table.columns
    assert table.records()[0]["condition"] == "demo"
    assert state.history[-2] == "Merge Tables: merged 2 rows"
    assert state.history[-1] == "Add Metadata Columns: annotated 2 rows"


def test_pipeline_select_table_columns_creates_table_state():
    data = np.zeros((7, 7), dtype=np.float32)
    data[1:3, 1:4] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    selected = pipeline.add_node("select_table_columns")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(selected.id, "columns", "label_id,area_pixels")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect(measurements.id, selected.id)

    outputs = pipeline.run(data, input_metadata={"axes": "YX"})
    table = outputs[selected.id]
    state = pipeline.output_states[selected.id]

    assert table.columns == ("label_id", "area_pixels")
    assert state.columns == ("label_id", "area_pixels")
    assert state.history[-1] == "Select Table Columns: selected 2 columns"


def test_save_table_output_writes_csv(tmp_path):
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:3, 2:5] = 1
    table = measure_objects(labels, resolved_spatial_ndim=2)
    path = tmp_path / "measurements.csv"

    saved = save_table_output(table, path)

    assert saved == path
    text = path.read_text(encoding="utf-8")
    assert text.startswith("label_id,area_pixels")
    assert "\n1,6," in text


def test_skeletonize_mask_reduces_binary_objects_to_skeleton():
    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 1:6] = True

    skeleton = skeletonize_mask(mask, resolved_spatial_ndim=2)

    assert skeleton.dtype == bool
    assert skeleton.sum() < mask.sum()
    assert skeleton.any()


def test_analyze_skeleton_reports_plus_shape_topology():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True

    table = analyze_skeleton(skeleton, resolved_spatial_ndim=2)
    record = table.records()[0]

    assert table.row_count == 1
    assert record["component_count_in_block"] == 1
    assert record["component_voxel_fraction"] == 1.0
    assert record["skeleton_voxel_count"] == 9
    assert record["endpoint_voxel_count"] == 4
    assert record["junction_voxel_count"] == 1
    assert record["isolated_node_count"] == 0
    assert record["branch_count"] == 4
    assert record["graph_node_count"] == 5
    assert record["graph_edge_count"] == 4
    assert record["voxel_graph_edge_count"] == 8
    assert record["cycle_count"] == 0
    assert record["skeleton_length_pixels"] == 8.0


def test_analyze_skeleton_reports_isolated_components():
    skeleton = np.zeros((5, 5), dtype=bool)
    skeleton[1, 1] = True
    skeleton[3, 3] = True

    table = analyze_skeleton(skeleton, resolved_spatial_ndim=2)
    records = table.records()

    assert table.row_count == 2
    assert {record["component_id"] for record in records} == {1, 2}
    assert all(record["component_count_in_block"] == 2 for record in records)
    assert all(record["component_voxel_fraction"] == 0.5 for record in records)
    assert all(record["skeleton_voxel_count"] == 1 for record in records)
    assert all(record["isolated_node_count"] == 1 for record in records)
    assert all(record["graph_node_count"] == 1 for record in records)
    assert all(record["graph_edge_count"] == 0 for record in records)
    assert all(record["voxel_graph_edge_count"] == 0 for record in records)
    assert all(record["cycle_count"] == 0 for record in records)


def test_analyze_skeleton_reports_3d_network_graph_edges():
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[1:4, 2, 2] = True

    table = analyze_skeleton(skeleton, resolved_spatial_ndim=3)
    record = table.records()[0]

    assert table.row_count == 1
    assert record["skeleton_voxel_count"] == 3
    assert record["endpoint_voxel_count"] == 2
    assert record["junction_voxel_count"] == 0
    assert record["isolated_node_count"] == 0
    assert record["graph_node_count"] == 2
    assert record["graph_edge_count"] == 1
    assert record["voxel_graph_edge_count"] == 2
    assert record["cycle_count"] == 0
    assert record["skeleton_length_voxels"] == 2.0


def test_analyze_skeleton_uses_axis_scale_for_physical_length():
    skeleton = np.zeros((1, 5, 5), dtype=bool)
    skeleton[0, 1:4, 2] = True

    table = analyze_skeleton(
        skeleton,
        resolved_spatial_ndim=2,
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(2.0, 0.5, 0.5),
        axis_units=("micrometer", "micrometer", "micrometer"),
    )
    record = table.records()[0]

    assert record["z_index"] == 0
    assert record["skeleton_length_pixels"] == 2.0
    assert record["skeleton_length_physical"] == 1.0
    assert record["physical_unit"] == "micrometer"


def test_pipeline_skeleton_analysis_creates_table_state():
    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    measurements = pipeline.add_node("analyze_skeleton")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, measurements.id)

    outputs = pipeline.run(image, input_metadata={"axes": "YX"})
    table = outputs[measurements.id]
    state = pipeline.output_states[measurements.id]

    assert table.row_count == 1
    assert state.kind == "measurement table"
    assert "branch_count" in state.columns
    assert state.history[-1] == "Analyze Skeleton: analyzed 1 component"


def test_save_output_writes_npy_when_enabled(tmp_path):
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)
    path = tmp_path / "node-output.npy"

    result = save_output(
        data,
        enabled="on",
        path=str(path),
        format="npy",
        overwrite="yes",
    )

    assert path.exists()
    np.testing.assert_array_equal(np.load(path), data)
    np.testing.assert_array_equal(result, data)


def test_save_array_output_respects_overwrite(tmp_path):
    data = np.zeros((2, 2), dtype=np.uint8)
    path = tmp_path / "existing.npy"
    save_array_output(data, path)

    try:
        save_array_output(data + 1, path, overwrite=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected overwrite=False to reject existing output")


def test_save_array_output_rejects_blank_path():
    try:
        save_array_output(np.zeros((2, 2), dtype=np.uint8), "")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected blank save path to be rejected")


def test_save_array_output_writes_imagej_hyperstack_and_mask_values(tmp_path):
    data = np.zeros((5, 3, 12, 8, 9), dtype=bool)
    data[:, 1, 4:7, 2:5, 3:6] = True
    state = image_state_from_array(data, layer_metadata={"axes": "TCZYX"})
    path = tmp_path / "otsu-threshold.tif"

    save_array_output(data, path, format="imagej-tiff", image_state=state)

    with tifffile.TiffFile(path) as tif:
        metadata = tif.imagej_metadata
        series = tif.series[0]
        saved = series.asarray()

    assert metadata["hyperstack"] is True
    assert metadata["frames"] == 5
    assert metadata["slices"] == 12
    assert metadata["channels"] == 3
    assert series.axes == "TZCYX"
    assert saved.shape == (5, 12, 3, 8, 9)
    assert set(np.unique(saved)) == {0, 255}
    expected = np.transpose(data.astype(np.uint8) * 255, (0, 2, 1, 3, 4))
    np.testing.assert_array_equal(saved, expected)


def test_slice_wise_filters_preserve_z_independence():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[1, 4, 4] = 1.0

    blurred = gaussian_blur(data, sigma=1.0)
    averaged = average_blur(data, size=3)
    medianed = median_filter(data, size=3)

    assert blurred.shape == data.shape
    assert averaged.shape == data.shape
    assert medianed.shape == data.shape
    assert np.allclose(blurred[0], 0)
    assert np.allclose(averaged[0], 0)
    assert np.allclose(medianed[0], 0)


def test_gaussian_3d_spreads_across_z_axis():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[1, 4, 4] = 1.0

    blurred = gaussian_blur_3d(data, sigma_z=1.0, sigma_y=0.0, sigma_x=0.0)

    assert blurred.shape == data.shape
    assert blurred[0, 4, 4] > 0
    assert blurred[2, 4, 4] > 0


def test_bilateral_filter_preserves_shape():
    data = np.zeros((2, 8, 8), dtype=np.uint8)
    data[:, 2:6, 2:6] = 200

    result = bilateral_filter(data, diameter=3, sigma_color=5, sigma_space=2)

    assert result.shape == data.shape
    assert result.dtype == data.dtype


def test_additional_filter_nodes_preserve_shape():
    data = np.zeros((2, 16, 16), dtype=np.float32)
    data[:, 5:11, 5:11] = 1.0

    dog = difference_of_gaussians_filter(data, low_sigma=1.0, high_sigma=3.0)
    unsharp = unsharp_mask_filter(data, radius=1.0, amount=1.0)
    nlm = non_local_means_filter(
        data,
        patch_size=3,
        patch_distance=2,
        h=0.05,
    )
    sobel = sobel_filter(data)
    laplace = laplace_filter(data, kernel_size=3)

    for result in (dog, unsharp, nlm, sobel, laplace):
        assert result.shape == data.shape
        assert result.dtype == np.float32

    assert dog.max() > 0
    assert sobel.max() > 0
    assert not np.allclose(laplace, 0)


def test_safe_filter_nodes_preserve_integer_dtype():
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    data[:, 5:11, 5:11] = 40000

    bilateral = bilateral_filter(data, diameter=3, sigma_color=5, sigma_space=2)
    unsharp = unsharp_mask_filter(data, radius=1.0, amount=1.0)
    nlm = non_local_means_filter(
        data,
        patch_size=3,
        patch_distance=2,
        h=0.05,
    )
    sobel = sobel_filter(data)
    dog = difference_of_gaussians_filter(data, low_sigma=1.0, high_sigma=3.0)
    laplace = laplace_filter(data, kernel_size=3)

    for result in (bilateral, unsharp, nlm, sobel):
        assert result.shape == data.shape
        assert result.dtype == data.dtype

    assert sobel.max() > 0
    assert dog.dtype == np.float32
    assert laplace.dtype == np.float32


def test_additional_global_thresholds_return_masks():
    data = np.zeros((2, 16, 16), dtype=np.float32)
    data[:, 3:8, 3:8] = 0.4
    data[:, 9:14, 9:14] = 1.0

    masks = [
        li_threshold(data),
        yen_threshold(data),
        isodata_threshold(data),
        minimum_threshold(data),
    ]

    for mask in masks:
        assert mask.shape == data.shape
        assert mask.dtype == bool


def test_additional_local_thresholds_return_masks():
    y = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    data = np.tile(y[:, None], (1, 16))
    data[5:11, 5:11] += 0.6
    stack = np.stack([data, data * 0.8])

    sauvola = sauvola_threshold(stack, window_size=5, k=0.2)
    niblack = niblack_threshold(stack, window_size=5, k=0.2)

    for mask in (sauvola, niblack):
        assert mask.shape == stack.shape
        assert mask.dtype == bool
        assert mask.any()


def test_edge_segmentation_operations_return_masks():
    data = np.zeros((3, 32, 32), dtype=np.float32)
    data[:, 8:24, 8:24] = 1.0
    data[:, 12:20, 12:20] = 0.5

    canny = canny_edges(data, sigma=1.0, low_quantile=0.05, high_quantile=0.2)
    hysteresis = hysteresis_threshold(
        data,
        low_threshold=0.4,
        high_threshold=0.8,
        spatial_mode="3D ZYX",
    )

    for mask in (canny, hysteresis):
        assert mask.shape == data.shape
        assert mask.dtype == bool
        assert mask.any()

    assert not canny[:, 16, 16].any()
    assert hysteresis[:, 16, 16].all()


def test_linear_scale_gamma_crop_and_extract_channel():
    data = np.zeros((2, 6, 7, 3), dtype=np.uint8)
    data[..., 1] = 64
    data[:, 2:5, 1:6, 2] = 128

    stretched = linear_scale_offset(data, alpha=2, beta=1)
    gamma = gamma_correction(data, gamma=0.5)
    cropped = crop_stack(data, top=1, bottom=2, left=1, right=3)
    channel = extract_channel(data, channel=2)

    assert stretched.dtype == np.uint8
    assert stretched[..., 1].max() == 129
    assert gamma.dtype == np.uint8
    assert cropped.shape == (2, 3, 3, 3)
    assert channel.shape == (2, 6, 7)
    assert channel.max() == 128

    uint16 = np.array([[0, 20000, 40000]], dtype=np.uint16)
    stretched16 = linear_scale_offset(uint16, alpha=2, beta=1)
    gamma16 = gamma_correction(uint16, gamma=0.5)

    assert stretched16.dtype == uint16.dtype
    assert gamma16.dtype == uint16.dtype


def test_reorder_axes_accepts_named_and_numeric_orders():
    data = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)

    named = reorder_axes(data, order="TZYXC", axis_names=("t", "c", "z", "y", "x"))
    numeric = reorder_axes(data, order="0,2,3,4,1")
    invalid = reorder_axes(data, order="ZYX")

    assert named.shape == (2, 4, 5, 6, 3)
    assert numeric.shape == (2, 4, 5, 6, 3)
    assert invalid.shape == data.shape


def test_project_image_uses_canonical_axes_and_projection_methods():
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

    projected = project_image(
        data,
        axes="axis:0",
        method="Mean",
        axis_names=("z", "y", "x"),
    )
    reduced = project_image(data, axes=(0, 1), method="Maximum")
    named_projected = project_image(
        data,
        axes="name:z",
        method="Maximum",
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
    )
    auto_projected = project_image(
        data,
        axes="auto",
        method="Maximum",
        axis_names=("z", "y", "x"),
    )

    assert projected.shape == (3, 4)
    assert projected.dtype == np.float32
    np.testing.assert_allclose(projected, data.mean(axis=0))
    np.testing.assert_array_equal(named_projected, data.max(axis=0))
    np.testing.assert_array_equal(auto_projected, data.max(axis=0))
    assert reduced.shape == (4,)
    assert reduced.dtype == data.dtype
    np.testing.assert_array_equal(reduced, data.max(axis=(0, 1)))


def test_project_image_auto_detects_z_axis_from_common_czyx_shape():
    data = np.zeros((3, 2, 4, 5), dtype=np.uint16)
    data[:, 1] = 7

    projected = project_image(data, axes="auto", method="Maximum")

    assert projected.shape == (3, 4, 5)
    assert projected.max() == 7


def test_orthogonal_projection_builds_xy_xz_yz_montage():
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

    projected = orthogonal_projection(
        data,
        method="Maximum",
        axis_names=("z", "y", "x"),
    )

    assert projected.shape == (5, 6)
    np.testing.assert_array_equal(projected[:3, :4], data.max(axis=0))
    np.testing.assert_array_equal(projected[3:, :4], data.max(axis=1))
    np.testing.assert_array_equal(projected[:3, 4:], data.max(axis=2).T)


def test_orthogonal_projection_uses_physical_z_spacing():
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

    scaled = orthogonal_projection(
        data,
        method="Maximum",
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(2.0, 1.0, 1.0),
        axis_units=("micrometer", "micrometer", "micrometer"),
    )
    unscaled = orthogonal_projection(
        data,
        method="Maximum",
        use_physical_scale=False,
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(2.0, 1.0, 1.0),
        axis_units=("micrometer", "micrometer", "micrometer"),
    )

    assert scaled.shape == (7, 8)
    assert unscaled.shape == (5, 6)


def test_projection_metadata_updates_axes():
    data = np.zeros((2, 3, 4), dtype=np.uint16)
    state = image_state_from_array(data, layer_metadata={"axes": "ZYX"})

    projected_state = transform_image_state(
        project_image(data, axes="axis:0", axis_names=("z", "y", "x")),
        state,
        operation_id="project_image",
        operation_title="Project Image",
        params={"axes": "axis:0", "method": "Maximum"},
    )
    orthogonal_state = transform_image_state(
        orthogonal_projection(data, axis_names=("z", "y", "x")),
        state,
        operation_id="orthogonal_projection",
        operation_title="Orthogonal Projection",
        params={"method": "Maximum"},
    )

    assert [axis.name for axis in projected_state.axes] == ["y", "x"]
    assert [axis.name for axis in orthogonal_state.axes] == ["y", "x"]
    assert orthogonal_state.shape == (5, 6)


def test_set_pixel_size_updates_spatial_axis_metadata_and_pipeline_projection():
    data = np.zeros((2, 3, 5), dtype=np.uint16)
    state = image_state_from_array(data)
    calibrated = set_pixel_size(
        data,
        x_size=0.5,
        y_size=0.5,
        z_size=2.0,
        unit="micrometer",
    )
    calibrated_state = transform_image_state(
        calibrated,
        state,
        operation_id="set_pixel_size",
        operation_title="Set Pixel Size / Units",
        params={
            "x_size": 0.5,
            "y_size": 0.5,
            "z_size": 2.0,
            "unit": "micrometer",
        },
    )

    assert [axis.name for axis in calibrated_state.axes] == ["z", "y", "x"]
    assert [axis.scale for axis in calibrated_state.axes] == [2.0, 0.5, 0.5]
    assert [axis.unit for axis in calibrated_state.axes] == [
        "micrometer",
        "micrometer",
        "micrometer",
    ]

    pipeline = PrototypePipeline()
    pixel_size = pipeline.add_node("set_pixel_size")
    projection = pipeline.add_node("orthogonal_projection")
    pipeline.set_param(pixel_size.id, "x_size", 0.5)
    pipeline.set_param(pixel_size.id, "y_size", 0.5)
    pipeline.set_param(pixel_size.id, "z_size", 2.0)
    pipeline.connect("input", pixel_size.id)
    pipeline.connect(pixel_size.id, projection.id)
    pipeline.run(data)

    assert pipeline.outputs[projection.id].shape == (11, 13)
    assert [axis.scale for axis in pipeline.output_states[projection.id].axes] == [
        0.5,
        0.5,
    ]


def test_rescale_axes_updates_shape_dtype_and_physical_scale_metadata():
    data = np.arange(2 * 3 * 5, dtype=np.uint16).reshape(2, 3, 5)
    state = image_state_from_array(data)

    scaled = rescale_axes(
        data,
        x_scale=2.0,
        y_scale=0.5,
        z_scale=1.5,
        lock_xy=False,
        interpolation="Nearest neighbor",
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
    )
    scaled_state = transform_image_state(
        scaled,
        state,
        operation_id="rescale_axes",
        operation_title="Rescale Axes",
        params={
            "x_scale": 2.0,
            "y_scale": 0.5,
            "z_scale": 1.5,
            "lock_xy": False,
            "interpolation": "Nearest neighbor",
        },
    )

    assert scaled.shape == (3, 2, 10)
    assert scaled.dtype == data.dtype
    assert [axis.name for axis in scaled_state.axes] == ["z", "y", "x"]
    assert [axis.scale for axis in scaled_state.axes] == [
        1.0 / 1.5,
        2.0,
        0.5,
    ]


def test_rescale_axes_z_scale_only_changes_semantic_z_dimension_in_pipeline():
    data = np.zeros((3, 12, 96, 128), dtype=np.uint16)
    pipeline = PrototypePipeline()
    rescale = pipeline.add_node("rescale_axes")
    pipeline.connect("input", rescale.id)
    pipeline.set_param(rescale.id, "x_scale", 1.0)
    pipeline.set_param(rescale.id, "y_scale", 1.0)
    pipeline.set_param(rescale.id, "z_scale", 2.0)
    pipeline.set_param(rescale.id, "lock_xy", True)

    pipeline.run(data, input_metadata={"axes": "CZYX"})
    state = pipeline.output_states[rescale.id]

    assert pipeline.outputs[rescale.id].shape == (3, 24, 96, 128)
    assert state is not None
    assert state.shape == (3, 24, 96, 128)
    assert state.axis_order == "CZYX"
    assert {axis.name: axis.scale for axis in state.axes} == {
        "c": 1.0,
        "z": 0.5,
        "y": 1.0,
        "x": 1.0,
    }


def test_rescale_axes_uses_current_reordered_spatial_semantics():
    data = np.zeros((3, 12, 96, 128), dtype=np.uint16)
    pipeline = PrototypePipeline()
    reorder = pipeline.add_node("reorder_axes")
    rescale = pipeline.add_node("rescale_axes")
    pipeline.connect("input", reorder.id)
    pipeline.connect(reorder.id, rescale.id)
    pipeline.set_param(reorder.id, "order", "CYZX")
    pipeline.set_param(rescale.id, "x_scale", 1.0)
    pipeline.set_param(rescale.id, "y_scale", 1.0)
    pipeline.set_param(rescale.id, "z_scale", 2.0)
    pipeline.set_param(rescale.id, "lock_xy", True)

    pipeline.run(data, input_metadata={"axes": "CZYX"})
    reorder_state = pipeline.output_states[reorder.id]
    rescale_state = pipeline.output_states[rescale.id]

    assert pipeline.outputs[reorder.id].shape == (3, 96, 12, 128)
    assert reorder_state is not None
    assert reorder_state.axis_order == "CZYX"
    assert [axis.name for axis in reorder_state.axes] == ["c", "z", "y", "x"]
    assert pipeline.outputs[rescale.id].shape == (3, 192, 12, 128)
    assert rescale_state is not None
    assert rescale_state.axis_order == "CZYX"


def test_rescale_axes_auto_uses_nearest_for_masks_and_labels():
    mask = np.array([[False, True], [True, False]], dtype=bool)
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    scaled_mask = rescale_axes(mask, x_scale=2.0, y_scale=2.0)
    scaled_labels = rescale_axes(
        labels,
        x_scale=2.0,
        y_scale=2.0,
        input_kind="label image",
    )

    assert scaled_mask.dtype == bool
    assert scaled_mask.shape == (4, 4)
    assert scaled_labels.dtype == labels.dtype
    assert set(np.unique(scaled_labels)) == {0, 1, 2}


def test_extract_channel_supports_czyx_stacks():
    data = np.zeros((3, 2, 5, 6), dtype=np.uint16)
    data[2] = 42

    channel = extract_channel(data, channel=2)

    assert channel.shape == (2, 5, 6)
    assert channel.max() == 42


def test_combine_channels_stacks_multiple_inputs_as_channels():
    first = np.full((2, 5, 6), 10, dtype=np.uint16)
    second = np.full((2, 5, 6), 20, dtype=np.uint16)

    composite = combine_channels([first, second], input_count=2, channel_axis=1)

    assert composite.shape == (2, 2, 5, 6)
    assert composite.dtype == np.uint16
    np.testing.assert_array_equal(composite[:, 0], first)
    np.testing.assert_array_equal(composite[:, 1], second)


def test_calculate_weighted_image_sums_inputs_with_offset():
    first = np.full((2, 3), 10, dtype=np.uint16)
    second = np.full((2, 3), 2, dtype=np.uint16)

    calculated = calculate_weighted_image(
        [first, second],
        input_count=2,
        weights="0.5,-2",
        offset=3,
    )

    assert calculated.dtype == np.float32
    np.testing.assert_array_equal(calculated, np.full((2, 3), 4, dtype=np.float32))


def test_intensity_rescale_normalize_and_clip():
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)

    rescaled = rescale_intensity(data, out_min=0, out_max=255)
    normalized = normalize_image(data, method="min-max")
    clipped = clip_intensity(data, minimum=2, maximum=4)

    assert rescaled.dtype == data.dtype
    assert rescaled.min() == 0
    assert rescaled.max() == 255
    value_rescaled = rescale_intensity(
        data,
        in_low_value=2,
        in_high_value=4,
        out_min=0,
        out_max=10,
    )
    np.testing.assert_array_equal(
        value_rescaled,
        np.array([[0, 0, 0], [5, 10, 10]], dtype=np.uint16),
    )
    assert rescale_intensity(data.astype(np.float32)).dtype == np.float32
    assert normalized.dtype == np.float32
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalize_image(data.astype(np.float64)).dtype == np.float64
    assert clipped.dtype == data.dtype
    np.testing.assert_array_equal(clipped, np.array([[2, 2, 2], [3, 4, 4]]))


def test_image_math_nodes_add_subtract_ratio_and_mask():
    first = np.full((2, 3), 10, dtype=np.uint16)
    second = np.full((2, 3), 2, dtype=np.uint16)
    mask = np.array([[True, False, True], [False, True, False]])

    added = add_images([first, second])
    subtracted = subtract_images([first, second])
    ratio = ratio_image([first, second], epsilon=0)
    masked = mask_image([first, mask], outside_value=99)

    np.testing.assert_array_equal(added, np.full((2, 3), 12, dtype=np.float32))
    np.testing.assert_array_equal(subtracted, np.full((2, 3), 8, dtype=np.float32))
    np.testing.assert_array_equal(ratio, np.full((2, 3), 5, dtype=np.float32))
    np.testing.assert_array_equal(
        masked,
        np.array([[10, 99, 10], [99, 10, 99]], dtype=np.uint16),
    )


def test_mask_image_broadcasts_spatial_mask_over_rgb_channels():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:] = np.array([10, 20, 30], dtype=np.uint8)
    mask = np.array([[True, False, True], [False, True, False]])

    masked = mask_image([image, mask], outside_value=0)

    assert masked.dtype == image.dtype
    assert masked.shape == image.shape
    np.testing.assert_array_equal(masked[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(masked[0, 1], [0, 0, 0])
    np.testing.assert_array_equal(masked[1, 1], [10, 20, 30])


def test_mask_image_broadcasts_stack_mask_over_channel_first_image():
    image = np.ones((3, 2, 4, 5), dtype=np.float32)
    mask = np.zeros((2, 4, 5), dtype=bool)
    mask[:, 1:3, 2:4] = True

    masked = mask_image([image, mask], outside_value=-5)

    assert masked.shape == image.shape
    assert masked.dtype == image.dtype
    assert np.all(masked[:, :, 0, 0] == -5)
    assert np.all(masked[:, :, 1, 2] == 1)


def test_mask_image_rejects_non_broadcastable_mask_shape():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)

    try:
        mask_image([image, mask])
    except ValueError as exc:
        assert "broadcastable" in str(exc)
    else:
        raise AssertionError("Mask Image should reject incompatible mask shapes.")


def test_logical_nodes_combine_masks():
    first = np.array([[True, False], [True, False]])
    second = np.array([[True, True], [False, False]])

    np.testing.assert_array_equal(
        logical_and([first, second]),
        np.array([[True, False], [False, False]]),
    )
    np.testing.assert_array_equal(
        logical_or([first, second]),
        np.array([[True, True], [True, False]]),
    )
    np.testing.assert_array_equal(
        logical_xor([first, second]),
        np.array([[False, True], [True, False]]),
    )


def test_composite_to_rgb_maps_three_channels():
    data = np.zeros((3, 8, 8), dtype=np.float32)
    data[0, 0, 0] = 1.0
    data[1, 1, 1] = 1.0
    data[2, 2, 2] = 1.0

    rgb = composite_to_rgb(data)

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.float32
    # Fluorescence stacks match thumbnail order: channel 0 blue, 1 green, 2 red.
    assert rgb[2, 2, 0] == 1.0
    assert rgb[1, 1, 1] == 1.0
    assert rgb[0, 0, 2] == 1.0
    assert rgb[0, 0, 0] == 0.0


def test_composite_to_rgb_constant_nonzero_channels_are_visible():
    data = np.zeros((3, 4, 5), dtype=np.uint16)
    data[0] = 1000
    data[1] = 2000
    data[2] = 3000

    rgb = composite_to_rgb(data, channel_axis=0, red_channel=0, green_channel=1)

    assert rgb.shape == (4, 5, 3)
    assert rgb[..., 0].max() == 1.0
    assert rgb[..., 1].max() == 1.0
    assert rgb[..., 2].max() == 1.0


def test_composite_to_rgb_single_channel_is_white():
    data = np.zeros((8, 8), dtype=np.float32)
    data[3, 3] = 1.0

    rgb = composite_to_rgb(data)

    assert rgb.shape == (8, 8, 3)
    np.testing.assert_allclose(rgb[..., 0], rgb[..., 1])
    np.testing.assert_allclose(rgb[..., 1], rgb[..., 2])


def test_composite_to_rgb_two_channels_uses_fluorescence_order():
    data = np.zeros((2, 8, 8), dtype=np.float32)
    data[0, 0, 0] = 1.0
    data[1, 1, 1] = 1.0

    rgb = composite_to_rgb(data)

    assert rgb.shape == (8, 8, 3)
    assert rgb[0, 0, 2] == 1.0
    assert rgb[1, 1, 1] == 1.0
    assert rgb[..., 0].max() == 0.0


def test_composite_to_rgb_accepts_channel_last_rgb():
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    data[0, 0, 0] = 255
    data[1, 1, 1] = 255
    data[2, 2, 2] = 255

    rgb = composite_to_rgb(data)

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.float32
    assert rgb[0, 0, 0] == 1.0
    assert rgb[1, 1, 1] == 1.0
    assert rgb[2, 2, 2] == 1.0


def test_composite_to_rgb_channel_last_c_axis_can_use_fluorescence_order():
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    data[0, 0, 0] = 255
    data[1, 1, 1] = 255
    data[2, 2, 2] = 255

    rgb = composite_to_rgb(data, channel_axis=2, channel_axis_semantics="c")

    assert rgb[2, 2, 0] == 1.0
    assert rgb[1, 1, 1] == 1.0
    assert rgb[0, 0, 2] == 1.0


def test_composite_to_rgb_auto_blends_named_channel_colours():
    data = np.zeros((2, 8, 8), dtype=np.float32)
    data[0, 2, 2] = 1.0
    data[1, 5, 5] = 1.0

    rgb = composite_to_rgb(data, channel_colors="Yellow,Cyan")

    assert rgb[2, 2, 0] == 1.0
    assert rgb[2, 2, 1] == 1.0
    assert rgb[2, 2, 2] == 0.0
    assert rgb[5, 5, 0] == 0.0
    assert rgb[5, 5, 1] == 1.0
    assert rgb[5, 5, 2] == 1.0


def test_assign_channel_colors_passes_data_through():
    data = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    output = assign_channel_colors(data, channel_colors="Yellow,Cyan,Magenta")

    np.testing.assert_array_equal(output, data)


def test_assign_channel_colors_updates_carried_metadata():
    data = np.zeros((2, 4, 5), dtype=np.uint16)
    input_state = image_state_from_array(data, layer_metadata={"axes": "CYX"})
    output = assign_channel_colors(data, channel_colors="Yellow,Cyan")

    state = transform_image_state(
        output,
        input_state,
        operation_id="assign_channel_colors",
        operation_title="Assign Channel Colors",
        params={"channel_colors": "Yellow,Cyan"},
    )

    assert state.channels[0].color == 0xFFFF00
    assert state.channels[1].color == 0x00FFFF


def test_combine_channels_colours_become_carried_metadata():
    first = np.ones((4, 5), dtype=np.uint16)
    second = np.ones((4, 5), dtype=np.uint16) * 2
    first_state = image_state_from_array(first, layer_metadata={"axes": "YX"})
    second_state = image_state_from_array(second, layer_metadata={"axes": "YX"})
    output = combine_channels([first, second], input_count=2, channel_axis=0)

    state = transform_multi_input_image_state(
        output,
        [first_state, second_state],
        operation_id="combine_channels",
        operation_title="Combine Channels",
        params={"channel_axis": 0, "channel_colors": "Yellow,Cyan"},
    )

    assert state.channels[0].color == 0xFFFF00
    assert state.channels[1].color == 0x00FFFF


def test_split_channels_returns_all_channels_losslessly():
    data = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)

    channels = split_channels(data)

    assert len(channels) == 3
    for channel in channels:
        assert channel.shape == (8, 8)
        assert channel.dtype == np.uint8
    np.testing.assert_array_equal(channels[0], data[..., 0])
    np.testing.assert_array_equal(channels[1], data[..., 1])
    np.testing.assert_array_equal(channels[2], data[..., 2])


def test_split_channels_preview_channel_does_not_change_outputs():
    data = np.zeros((3, 4, 5), dtype=np.uint8)
    data[0] = 10
    data[1] = 20
    data[2] = 30

    channels = split_channels(data, preview_channel=2)

    assert len(channels) == 3
    assert int(channels[0].max()) == 10
    assert int(channels[1].max()) == 20
    assert int(channels[2].max()) == 30


def test_split_channels_returns_true_channel_count():
    data = np.zeros((2, 8, 8), dtype=np.uint8)
    data[0] = 10
    data[1] = 20

    channels = split_channels(data)

    assert len(channels) == 2
    assert int(channels[0].max()) == 10
    assert int(channels[1].max()) == 20


def test_split_channels_grayscale_returns_single_output():
    data = np.full((5, 5), 7, dtype=np.uint8)

    channels = split_channels(data)

    assert len(channels) == 1
    np.testing.assert_array_equal(channels[0], data)


def test_select_axis_slice_removes_requested_axis():
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)

    selected = select_axis_slice(data, axis=1, index=2)

    assert selected.shape == (2, 4)
    np.testing.assert_array_equal(selected, data[:, 2, :])


def test_select_axis_slice_can_remove_multiple_axes():
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    selected = select_axis_slice(data, axes="0,2", indices="1,3")

    assert selected.shape == (3, 5)
    np.testing.assert_array_equal(selected, data[1, :, 3, :])


def test_select_axis_slice_can_retain_axis_ranges():
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    selected = select_axis_slice(
        data,
        ranges="0:1:1;2:1:3",
        range_mode=True,
    )

    assert selected.shape == (1, 3, 3, 5)
    np.testing.assert_array_equal(selected, data[1:2, :, 1:4, :])


def test_select_axis_slice_can_retain_ranges_and_remove_axes():
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    selected = select_axis_slice(
        data,
        ranges="2:1:3",
        range_mode=True,
        remove_axes="1",
        remove_indices="2",
    )

    assert selected.shape == (2, 3, 5)
    np.testing.assert_array_equal(selected, data[:, 2, 1:4, :])


def test_linear_scale_offset_uses_linear_offset_without_abs():
    data = np.array([0, 10, 20], dtype=np.uint8)

    stretched = linear_scale_offset(data, alpha=10, beta=-50)

    assert stretched.tolist() == [0, 50, 150]


def test_convert_dtype_rescales_and_preserves_shape():
    data = np.array([[0, 1000], [500, 250]], dtype=np.uint16)

    converted = convert_dtype(data, output_dtype="uint8", scaling="rescale")

    assert converted.dtype == np.uint8
    assert converted.shape == data.shape
    assert converted.min() == 0
    assert converted.max() == 255


def test_convert_dtype_to_bool_and_float():
    data = np.array([[0, 2], [4, 8]], dtype=np.uint16)

    mask = convert_dtype(data, output_dtype="bool", scaling="rescale")
    floated = convert_dtype(data, output_dtype="float32", scaling="rescale")

    assert mask.dtype == bool
    assert mask.tolist() == [[False, True], [True, True]]
    assert floated.dtype == np.float32
    np.testing.assert_allclose(floated.min(), 0.0)
    np.testing.assert_allclose(floated.max(), 1.0)


def test_thresholding_operations_return_masks():
    data = np.tile(np.arange(12, dtype=np.uint8), (12, 1))
    data = np.stack([data, data + 20])

    masks = [
        binary_threshold(data, threshold=6),
        adaptive_mean_threshold(data, block_size=5, c=0),
        adaptive_gaussian_threshold(data, block_size=5, c=0),
        otsu_threshold(data),
        triangle_threshold(data),
    ]

    for mask in masks:
        assert mask.dtype == bool
        assert mask.shape == data.shape
        assert mask.any()


def test_global_threshold_scope_can_use_stack_or_slice_histogram():
    data = np.zeros((2, 10, 10), dtype=np.float32)
    data[0, :, 5:] = 10.0
    data[1, :, :5] = 100.0
    data[1, :, 5:] = 110.0

    stack_mask = otsu_threshold(data, threshold_scope="Stack histogram")
    slice_mask = otsu_threshold(data, threshold_scope="Slice histogram")

    assert stack_mask.shape == data.shape
    assert slice_mask.shape == data.shape
    assert stack_mask.dtype == bool
    assert slice_mask.dtype == bool
    assert not np.array_equal(stack_mask, slice_mask)


def test_morphology_and_small_object_operations_return_masks():
    mask = np.zeros((3, 9, 9), dtype=bool)
    mask[1, 4, 4] = True
    mask[1, 2:7, 2:7] = True
    mask[1, 4, 4] = False
    mask[0, 0, 0] = True

    closed_cavity = np.ones((3, 5, 5), dtype=bool)
    closed_cavity[1, 2, 2] = False
    filled = fill_holes(closed_cavity)
    filtered = remove_small_objects(
        mask,
        min_size=5,
        spatial_mode="3D ZYX",
    )

    assert dilate(mask, size=3, iterations=1).sum() > mask.sum()
    assert erode(filled, size=3, iterations=1).sum() < filled.sum()
    assert opening(mask, size=2).dtype == bool
    assert closing(mask, size=2).dtype == bool
    assert top_hat(mask, size=2).dtype == bool
    assert black_hat(mask, size=2).dtype == bool
    assert morphological_gradient(mask, size=2).dtype == bool
    assert filled[1, 2, 2]
    assert not filtered[0, 0, 0]
    assert filtered[1].any()


def test_fill_holes_supports_size_limited_2d_filling():
    mask = np.ones((9, 9), dtype=bool)
    mask[2, 2] = False
    mask[5:7, 5:7] = False

    limited = fill_holes(
        mask,
        max_hole_size=1,
        spatial_mode="2D per XY slice (advanced)",
    )
    filled_all = fill_holes(
        mask,
        max_hole_size=0,
        spatial_mode="2D per XY slice (advanced)",
    )

    assert limited[2, 2]
    assert not limited[5:7, 5:7].any()
    assert filled_all.all()


def test_fill_holes_distinguishes_2d_slices_from_3d_volume():
    mask = np.ones((3, 7, 7), dtype=bool)
    mask[0, 3, 3] = False

    slice_wise = fill_holes(
        mask,
        spatial_mode="2D per XY slice (advanced)",
    )
    volumetric = fill_holes(
        mask,
        spatial_mode="3D ZYX volume",
    )

    assert slice_wise[0, 3, 3]
    assert not volumetric[0, 3, 3]


def test_fill_holes_processes_leading_frames_independently():
    mask = np.ones((2, 3, 7, 7), dtype=bool)
    mask[0, 1, 3, 3] = False
    mask[1, 0, 3, 3] = False

    filled = fill_holes(mask, spatial_mode="3D ZYX volume")

    assert filled[0, 1, 3, 3]
    assert not filled[1, 0, 3, 3]


def test_fill_holes_node_uses_metadata_for_auto_spatial_mode():
    mask = np.ones((2, 3, 7, 7), dtype=bool)
    mask[:, 1, 3, 3] = False
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    filled = pipeline.add_node("fill_holes")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, filled.id)

    outputs = pipeline.run(
        mask.astype(np.float32),
        input_metadata={"axes": "TZYX"},
    )

    assert outputs[filled.id].all()
    assert pipeline.nodes[filled.id].params["resolved_spatial_ndim"] == 3


def test_label_connected_components_respects_2d_connectivity():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True

    face = label_connected_components(
        mask,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    full = label_connected_components(
        mask,
        spatial_mode="2D YX",
        connectivity="Full connectivity",
    )

    assert face.dtype == np.int32
    assert int(face.max()) == 2
    assert int(full.max()) == 1


def test_clear_border_objects_preserves_label_ids_and_supports_buffer():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[0:2, 1:3] = 5
    labels[1:3, 4:6] = 9
    labels[3:5, 3:6] = 20

    cleared = clear_border_objects(
        labels,
        resolved_spatial_ndim=2,
    )
    buffered = clear_border_objects(
        labels,
        border_buffer=1,
        resolved_spatial_ndim=2,
    )

    assert cleared.dtype == labels.dtype
    assert set(np.unique(cleared)) == {0, 9, 20}
    assert set(np.unique(buffered)) == {0, 20}


def test_clear_border_objects_supports_all_or_lateral_3d_boundaries():
    labels = np.zeros((3, 8, 8), dtype=np.int32)
    labels[0, 3:5, 3:5] = 5
    labels[1, 0:2, 3:5] = 9
    labels[1, 3:5, 3:5] = 20

    all_borders = clear_border_objects(
        labels,
        boundary_mode="All spatial borders",
        resolved_spatial_ndim=3,
    )
    lateral = clear_border_objects(
        labels,
        boundary_mode="Lateral borders only (YX)",
        resolved_spatial_ndim=3,
    )

    assert set(np.unique(all_borders)) == {0, 20}
    assert set(np.unique(lateral)) == {0, 5, 20}


def test_clear_border_objects_rejects_intensity_images():
    data = np.ones((5, 5), dtype=np.float32)

    try:
        clear_border_objects(data)
    except ValueError as exc:
        assert "binary mask or integer label image" in str(exc)
    else:
        raise AssertionError("Expected intensity input to be rejected")


def test_clear_border_node_accepts_masks_and_labels_but_not_images():
    pipeline = PrototypePipeline()
    clear_mask = pipeline.add_node("clear_border_objects")
    labels = pipeline.add_node("label_connected_components")
    clear_labels = pipeline.add_node("clear_border_objects")

    image_result = pipeline.connect("input", clear_mask.id)
    mask_result = pipeline.connect("threshold", clear_mask.id)
    labels_result = pipeline.connect("threshold", labels.id)
    clear_labels_result = pipeline.connect(labels.id, clear_labels.id)

    assert not image_result.success
    assert mask_result.success
    assert labels_result.success
    assert clear_labels_result.success
    assert pipeline.output_ports(clear_mask.id)[0].output_type == "mask"
    assert pipeline.output_ports(clear_labels.id)[0].output_type == "labels"


def test_clear_border_node_preserves_label_metadata_and_resolved_spatial_mode():
    data = np.zeros((3, 9, 9), dtype=np.float32)
    data[:, 0:3, 0:3] = 10
    data[1, 4:7, 4:7] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    cleared = pipeline.add_node("clear_border_objects")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, cleared.id)

    outputs = pipeline.run(
        data,
        input_metadata={"axes": "ZYX"},
        input_name="3D nuclei",
    )

    assert set(np.unique(outputs[cleared.id])) == {0, 2}
    assert pipeline.output_states[cleared.id].kind == "label image"
    assert pipeline.nodes[cleared.id].params["resolved_spatial_ndim"] == 3


def test_remove_small_objects_preserves_labels_and_processes_frames():
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 0:2, 0:2] = 5
    labels[0, 4, 4] = 9
    labels[1, 0, 0] = 5
    labels[1, 3:5, 2:5] = 20

    filtered = remove_small_objects(
        labels,
        min_size=2,
        spatial_mode="2D YX",
    )

    assert filtered.dtype == labels.dtype
    assert set(np.unique(filtered[0])) == {0, 5}
    assert set(np.unique(filtered[1])) == {0, 20}


def test_remove_small_objects_node_accepts_masks_and_labels_not_images():
    pipeline = PrototypePipeline()
    mask_filter = pipeline.add_node("remove_small_objects")
    labels = pipeline.add_node("label_connected_components")
    label_filter = pipeline.add_node("remove_small_objects")

    image_result = pipeline.connect("input", mask_filter.id)
    mask_result = pipeline.connect("threshold", mask_filter.id)
    labels_result = pipeline.connect("threshold", labels.id)
    label_result = pipeline.connect(labels.id, label_filter.id)

    assert not image_result.success
    assert mask_result.success
    assert labels_result.success
    assert label_result.success
    assert pipeline.output_ports(mask_filter.id)[0].output_type == "mask"
    assert pipeline.output_ports(label_filter.id)[0].output_type == "labels"


def test_label_connected_components_processes_true_3d_and_frames_independently():
    mask = np.zeros((2, 3, 7, 7), dtype=bool)
    mask[0, :, 1:3, 1:3] = True
    mask[0, 1, 5, 5] = True
    mask[1, :, 3:5, 3:5] = True

    labels = label_connected_components(
        mask,
        spatial_mode="3D ZYX",
        connectivity="Full connectivity",
    )

    assert labels.shape == mask.shape
    assert set(np.unique(labels[0])) == {0, 1, 2}
    assert set(np.unique(labels[1])) == {0, 1}
    assert np.all(labels[0, :, 1:3, 1:3] == 1)


def test_filter_labels_by_volume_preserves_ids_and_filters_per_frame():
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 0:2, 0:2] = 5
    labels[0, 4, 4] = 9
    labels[1, 0, 0] = 5
    labels[1, 3:5, 2:5] = 20

    filtered = filter_labels_by_volume(
        labels,
        min_volume=2,
        max_volume=5,
        spatial_mode="2D YX",
    )

    assert set(np.unique(filtered[0])) == {0, 5}
    assert set(np.unique(filtered[1])) == {0}


def test_relabel_sequential_compacts_ids_per_frame():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 0:2, 0:2] = 5
    labels[0, 3:5, 3:5] = 20
    labels[1, 1:4, 1:4] = 20

    relabeled = relabel_sequential(labels, spatial_mode="2D YX")

    assert set(np.unique(relabeled[0])) == {0, 1, 2}
    assert set(np.unique(relabeled[1])) == {0, 1}
    assert np.all(relabeled[0, 0:2, 0:2] == 1)
    assert np.all(relabeled[0, 3:5, 3:5] == 2)


def test_label_operations_reject_non_integer_label_images():
    data = np.array([[0.0, 1.0]], dtype=np.float32)

    try:
        filter_labels_by_volume(data)
    except ValueError as exc:
        assert "integer label image" in str(exc)
    else:
        raise AssertionError("Expected float labels to be rejected")


def test_save_array_output_preserves_int32_label_ids(tmp_path):
    labels = np.array([[0, 70_000], [2, 3]], dtype=np.int32)
    input_state = image_state_from_array(labels, layer_metadata={"axes": "YX"})
    state = transform_image_state(
        labels,
        input_state,
        operation_id="label_connected_components",
        operation_title="Label Connected Components",
        params={},
    )
    path = tmp_path / "labels.tif"

    save_array_output(labels, path, image_state=state)

    with tifffile.TiffFile(path) as tif:
        saved = tif.asarray()
        assert not tif.is_imagej

    assert saved.dtype == np.int32
    np.testing.assert_array_equal(saved, labels)
