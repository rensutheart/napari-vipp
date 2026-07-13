from __future__ import annotations

import inspect
from fractions import Fraction

import numpy as np
import pytest
import tifffile
from scipy import signal
from skimage import exposure

import napari_vipp.core.operations as operations
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
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
    auto_watershed_from_mask,
    average_blur,
    bilateral_filter,
    binary_threshold,
    black_hat,
    born_wolf_psf,
    born_wolf_psf_outputs,
    calculate_weighted_image,
    canny_edges,
    clear_border_objects,
    clip_intensity,
    closing,
    colocalization_metrics,
    colocalization_scatter_plot,
    colocalization_threshold_values,
    colocalized_voxels,
    combine_channels,
    composite_to_rgb,
    convert_dtype,
    crop_stack,
    difference_of_gaussians_filter,
    dilate,
    erode,
    euclidean_distance_transform,
    event_localization,
    expand_labels_image,
    extract_channel,
    fill_holes,
    filter_labels_by_property,
    filter_labels_by_volume,
    gamma_correction,
    gaussian_blur,
    gaussian_blur_3d,
    h_maxima_markers,
    hysteresis_threshold,
    isodata_threshold,
    label_connected_components,
    label_overlap_association,
    label_skeleton_branches,
    label_skeleton_components,
    laplace_filter,
    li_threshold,
    linear_scale_offset,
    logical_and,
    logical_or,
    logical_xor,
    marker_controlled_watershed,
    mask_image,
    measure_3d_mesh_morphology,
    measure_objects,
    measure_objects_with_intensity,
    measure_overall_skeleton_network,
    measure_skeleton_branches,
    median_filter,
    merge_tables,
    minimum_threshold,
    morphological_gradient,
    nearest_object_distance,
    niblack_threshold,
    non_local_means_filter,
    normalize_image,
    object_colocalization_metrics,
    opening,
    orthogonal_projection,
    otsu_threshold,
    prepare_validate_psf,
    project_image,
    prune_skeleton_branches,
    racc_index,
    ratio_image,
    relabel_sequential,
    remove_small_objects,
    reorder_axes,
    rescale_axes,
    rescale_intensity,
    richardson_lucy_deconvolution,
    richardson_lucy_tv_deconvolution,
    rolling_ball_background,
    sauvola_threshold,
    save_array_output,
    save_output,
    select_axis_slice,
    select_table_columns,
    set_pixel_size,
    skeleton_graph_overlay,
    skeleton_graph_tables,
    skeleton_keypoints,
    skeletonize_mask,
    sobel_filter,
    split_axis,
    split_channels,
    subtract_background,
    subtract_images,
    summarize_measurements,
    summarize_skeleton_branches,
    top_hat,
    triangle_threshold,
    unsharp_mask_filter,
    yen_threshold,
)
from napari_vipp.core.pipeline import (
    EXECUTION_NOT_CALCULATED,
    EXECUTION_POLICIES,
    EXECUTION_READY,
    EXECUTION_STALE,
    MANUAL_RUN_SKIP,
    NODE_LIBRARY,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import save_table_output, table_from_columns
from napari_vipp.core.workflow import serialize_workflow


def _compact_psf_2d() -> np.ndarray:
    psf = np.array(
        [
            [0.0, 0.05, 0.0],
            [0.05, 0.8, 0.05],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float32,
    )
    return psf / psf.sum()


def _compact_psf_3d() -> np.ndarray:
    psf = np.zeros((3, 3, 3), dtype=np.float32)
    psf[1, 1, 1] = 0.7
    psf[0, 1, 1] = 0.05
    psf[2, 1, 1] = 0.05
    psf[1, 0, 1] = 0.05
    psf[1, 2, 1] = 0.05
    psf[1, 1, 0] = 0.05
    psf[1, 1, 2] = 0.05
    return psf / psf.sum()


def _total_variation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(sum(np.abs(np.diff(arr, axis=axis)).sum() for axis in range(arr.ndim)))


def test_vipp_operation_nodes_are_registered():
    expected = {
        "crop_stack",
        "linear_scale_offset",
        "gamma_correction",
        "average_blur",
        "gaussian_blur",
        "gaussian_blur_3d",
        "prepare_validate_psf",
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
        "median_filter",
        "bilateral_filter",
        "non_local_means_filter",
        "rolling_ball_background",
        "subtract_background",
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
        "euclidean_distance_transform",
        "auto_watershed_from_mask",
        "h_maxima_markers",
        "marker_controlled_watershed",
        "expand_labels",
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
        "measure_skeleton_branches",
        "skeleton_graph_tables",
        "measure_overall_skeleton_network",
        "summarize_skeleton_branches",
        "skeleton_keypoints",
        "skeleton_graph_overlay",
        "label_skeleton_components",
        "label_skeleton_branches",
        "prune_skeleton_branches",
        "assign_channel_colors",
        "colocalization_metrics",
        "masked_colocalization_metrics",
        "colocalized_voxels",
        "masked_colocalized_voxels",
        "racc_index",
        "masked_racc_index",
        "object_colocalization_metrics",
        "label_overlap_association",
        "nearest_object_distance",
        "event_localization",
        "merge_tables",
        "add_metadata_columns",
        "select_table_columns",
        "summarize_measurements",
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


def test_pipeline_resolves_deferred_source_statistics_during_execution():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    deferred = image_state_from_array(data, defer_statistics=True)
    pipeline = PrototypePipeline()

    pipeline.run(
        data,
        source_payloads={
            "input": SourcePayload(data, image_state=deferred),
        },
    )

    assert pipeline.output_states["input"].value_range == "0 to 15"


@pytest.mark.parametrize(
    "operation_id",
    [
        "otsu_threshold",
        "triangle_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
    ],
)
def test_histogram_threshold_nodes_persist_float_bin_count(operation_id):
    spec = NODE_LIBRARY_BY_ID[operation_id]
    bins = next(param for param in spec.parameters if param.name == "histogram_bins")

    assert bins.label == "Float histogram bins"
    assert bins.default == 256
    assert bins.minimum == 2
    assert bins.maximum == 65_536


def test_li_threshold_does_not_expose_histogram_bins():
    names = {param.name for param in NODE_LIBRARY_BY_ID["li_threshold"].parameters}

    assert "histogram_bins" not in names


def test_registered_operation_specs_match_callable_and_ui_contracts():
    operation_ids = [spec.id for spec in NODE_LIBRARY]
    assert len(operation_ids) == len(set(operation_ids))

    for spec in NODE_LIBRARY:
        assert spec.execution_policy in EXECUTION_POLICIES, spec.id
        if spec.inputs:
            assert spec.max_inputs == len(spec.inputs), spec.id
        assert (spec.function is not None) == spec.has_input, spec.id
        if spec.function is not None:
            signature_params = tuple(
                inspect.signature(spec.function).parameters.values()
            )
            declared = {param.name for param in spec.parameters}
            accepted = {param.name for param in signature_params}
            required = {
                param.name
                for param in signature_params[1:]
                if param.default is inspect.Parameter.empty
                and param.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            }
            assert declared <= accepted, spec.id
            assert required <= declared, spec.id

        output_names = [port.name for port in spec.output_ports]
        assert len(output_names) == len(set(output_names)), spec.id
        for param in spec.parameters:
            assert param.minimum <= param.maximum, (spec.id, param.name)
            if param.choices:
                assert param.default in param.choices, (spec.id, param.name)
            elif isinstance(param.default, (int, float)) and not isinstance(
                param.default,
                bool,
            ):
                assert param.minimum <= param.default <= param.maximum, (
                    spec.id,
                    param.name,
                )
            if param.choice_labels:
                assert len(param.choice_labels) == len(param.choices), (
                    spec.id,
                    param.name,
                )


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


def test_touching_object_separation_splits_two_touching_disks():
    yy, xx = np.mgrid[:48, :64]
    mask = ((yy - 24) ** 2 + (xx - 22) ** 2 <= 13**2) | (
        (yy - 24) ** 2 + (xx - 42) ** 2 <= 13**2
    )

    distance = euclidean_distance_transform(mask, resolved_spatial_ndim=2)
    markers = h_maxima_markers(distance, h=1.0, resolved_spatial_ndim=2)
    labels = marker_controlled_watershed(
        [distance, markers, mask],
        resolved_spatial_ndim=2,
    )

    assert distance.dtype == np.float32
    assert markers.dtype == np.int32
    assert labels.dtype == np.int32
    assert int(markers.max()) == 2
    assert int(labels.max()) == 2
    assert np.all(labels[mask] > 0)


def test_auto_watershed_from_mask_splits_two_touching_disks():
    yy, xx = np.mgrid[:48, :64]
    mask = ((yy - 24) ** 2 + (xx - 22) ** 2 <= 13**2) | (
        (yy - 24) ** 2 + (xx - 42) ** 2 <= 13**2
    )

    labels = auto_watershed_from_mask(mask, resolved_spatial_ndim=2)

    assert labels.dtype == np.int32
    assert int(labels.max()) == 2
    assert np.all(labels[mask] > 0)


def test_expand_labels_can_run_in_3d_or_slice_wise():
    labels = np.zeros((3, 7, 7), dtype=np.int32)
    labels[0, 3, 3] = 1

    expanded_3d = expand_labels_image(
        labels,
        distance=1.1,
        spatial_mode="3D ZYX",
    )
    expanded_2d = expand_labels_image(
        labels,
        distance=1.1,
        spatial_mode="2D YX",
    )

    assert expanded_3d[1, 3, 3] == 1
    assert expanded_2d[1, 3, 3] == 0


def test_touching_object_pipeline_defaults_to_3d_for_z_stacks():
    data = np.zeros((3, 16, 16), dtype=bool)
    data[:, 5:11, 5:11] = True
    pipeline = PrototypePipeline()
    distance = pipeline.add_node("euclidean_distance_transform")
    markers = pipeline.add_node("h_maxima_markers")
    watershed = pipeline.add_node("marker_controlled_watershed")
    pipeline.connect("input", distance.id)
    pipeline.connect(distance.id, markers.id)
    pipeline.connect(distance.id, watershed.id, target_port=0)
    pipeline.connect(markers.id, watershed.id, target_port=1)
    pipeline.connect("input", watershed.id, target_port=2)

    outputs = pipeline.run(data, input_metadata={"axes": "ZYX"})

    assert outputs[distance.id].dtype == np.float32
    assert outputs[markers.id].dtype == np.int32
    assert outputs[watershed.id].dtype == np.int32
    assert pipeline.nodes[distance.id].params["resolved_spatial_ndim"] == 3
    assert pipeline.nodes[markers.id].params["resolved_spatial_ndim"] == 3
    assert pipeline.nodes[watershed.id].params["resolved_spatial_ndim"] == 3
    assert pipeline.output_states[watershed.id].kind == "label image"


def test_auto_watershed_pipeline_defaults_to_3d_for_z_stacks():
    data = np.zeros((3, 16, 16), dtype=bool)
    data[:, 5:11, 5:11] = True
    pipeline = PrototypePipeline()
    watershed = pipeline.add_node("auto_watershed_from_mask")
    pipeline.connect("input", watershed.id)

    outputs = pipeline.run(data, input_metadata={"axes": "ZYX"})

    assert outputs[watershed.id].dtype == np.int32
    assert pipeline.nodes[watershed.id].params["resolved_spatial_ndim"] == 3
    assert pipeline.output_states[watershed.id].kind == "label image"


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
        include_derived_shape_ratios=True,
        include_2d_shape_moments=True,
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
    assert record["axis_ratio_major_minor"] >= 1.0
    assert record["bbox_axis_0_length_pixels"] == 6.0
    assert record["bbox_axis_1_length_pixels"] == 6.0
    assert record["bbox_axis_ratio_0_1"] == 1.0
    assert record["bbox_fill_fraction"] == 1.0
    assert record["inertia_eigval_ratio_0_1"] >= 1.0
    assert 0.0 < record["circularity"] <= 1.0
    assert record["perimeter_area_ratio"] > 0.0
    for index in range(7):
        assert f"hu_moment_{index}" in table.columns
    assert table.unit_for("orientation_radians") == "radians"
    assert table.unit_for("axis_ratio_major_minor") == "ratio"
    assert table.unit_for("circularity") == "ratio"
    assert "shape descriptors" in table.table_kind
    assert "derived shape ratios" in table.table_kind
    assert "2D shape moments" in table.table_kind


def test_measure_objects_reports_calibrated_extended_2d_properties():
    labels = np.zeros((4, 6), dtype=np.int32)
    labels[1:3, 2:5] = 1

    table = measure_objects(
        labels,
        resolved_spatial_ndim=2,
        axis_names=("y", "x"),
        axis_types=("space", "space"),
        axis_scales=(2.0, 2.0),
        axis_units=("micrometer", "micrometer"),
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_2d_boundary_descriptors=True,
        include_derived_shape_ratios=True,
        include_2d_shape_moments=True,
    )
    record = table.records()[0]

    assert np.isclose(record["area_physical"], 24.0)
    assert record["physical_unit"] == "micrometer^2"
    assert np.isclose(record["equivalent_diameter_physical"], np.sqrt(96.0 / np.pi))
    assert np.isclose(record["centroid_y_physical"], 3.0)
    assert np.isclose(record["centroid_x_physical"], 6.0)
    assert np.isclose(record["bbox_y_min_physical"], 2.0)
    assert np.isclose(record["bbox_x_max_physical"], 10.0)
    assert np.isclose(record["bbox_area_physical"], 24.0)
    assert np.isclose(record["filled_area_physical"], 24.0)
    assert np.isclose(record["convex_area_physical"], 24.0)
    assert np.isclose(
        record["feret_diameter_max_physical"],
        record["feret_diameter_max_pixels"] * 2.0,
    )
    assert np.isclose(
        record["major_axis_length_physical"],
        record["major_axis_length_pixels"] * 2.0,
    )
    assert np.isclose(record["bbox_axis_0_length_physical"], 4.0)
    assert np.isclose(record["bbox_axis_1_length_physical"], 6.0)
    assert np.isclose(record["perimeter_physical"], 12.0)
    assert np.isclose(record["perimeter_area_ratio_physical"], 0.5)
    assert table.unit_for("equivalent_diameter_physical") == "micrometer"
    assert table.unit_for("bbox_area_physical") == "micrometer^2"
    assert table.unit_for("major_axis_length_physical") == "micrometer"
    assert table.unit_for("perimeter_area_ratio_physical") == "1/micrometer"


def test_measure_objects_marks_anisotropic_2d_perimeter_physical_as_nan():
    labels = np.zeros((4, 6), dtype=np.int32)
    labels[1:3, 2:5] = 1

    table = measure_objects(
        labels,
        resolved_spatial_ndim=2,
        axis_names=("y", "x"),
        axis_types=("space", "space"),
        axis_scales=(0.5, 2.0),
        axis_units=("micrometer", "micrometer"),
        include_2d_boundary_descriptors=True,
        include_2d_shape_moments=True,
    )
    record = table.records()[0]

    assert np.isclose(record["area_physical"], 6.0)
    assert np.isclose(record["centroid_y_physical"], 0.75)
    assert np.isclose(record["centroid_x_physical"], 6.0)
    assert np.isnan(record["perimeter_physical"])
    assert np.isnan(record["perimeter_crofton_physical"])
    assert np.isnan(record["perimeter_area_ratio_physical"])


def test_measure_objects_omits_2d_only_properties_for_3d_measurements():
    labels = np.zeros((4, 12, 12), dtype=np.int32)
    labels[1:3, 2:8, 3:9] = 1

    table = measure_objects(
        labels,
        resolved_spatial_ndim=3,
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_2d_boundary_descriptors=True,
        include_derived_shape_ratios=True,
        include_2d_shape_moments=True,
    )
    record = table.records()[0]

    assert table.row_count == 1
    assert record["bbox_volume_voxels"] == 72
    assert record["filled_volume_voxels"] == 72
    assert record["major_axis_length_voxels"] > 0
    assert record["minor_axis_length_voxels"] > 0
    assert record["bbox_axis_0_length_voxels"] == 2.0
    assert record["bbox_axis_1_length_voxels"] == 6.0
    assert record["bbox_axis_2_length_voxels"] == 6.0
    assert np.isclose(record["bbox_axis_ratio_0_1"], 1.0 / 3.0)
    assert record["bbox_fill_fraction"] == 1.0
    assert "axis_ratio_major_minor" in table.columns
    assert "inertia_tensor_eigval_2" in table.columns
    assert "inertia_eigval_ratio_0_2" in table.columns
    assert "perimeter_pixels" not in table.columns
    assert "orientation_radians" not in table.columns
    assert "convex_volume_voxels" not in table.columns
    assert "circularity" not in table.columns
    assert "hu_moment_0" not in table.columns
    assert table.unit_for("inertia_tensor_eigval_2") == "voxels^2"
    assert table.unit_for("bbox_axis_0_length_voxels") == "voxels"


def test_measure_objects_reports_calibrated_extended_3d_properties():
    labels = np.zeros((4, 12, 12), dtype=np.int32)
    labels[1:3, 2:8, 3:9] = 1

    table = measure_objects(
        labels,
        resolved_spatial_ndim=3,
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(2.0, 0.5, 0.25),
        axis_units=("micrometer", "micrometer", "micrometer"),
        include_shape_descriptors=True,
        include_axis_descriptors=True,
        include_derived_shape_ratios=True,
    )
    record = table.records()[0]

    assert record["volume_voxels"] == 72
    assert np.isclose(record["volume_physical"], 18.0)
    assert record["physical_unit"] == "micrometer^3"
    expected_equivalent = 2.0 * ((3.0 * 18.0) / (4.0 * np.pi)) ** (1.0 / 3.0)
    assert np.isclose(record["equivalent_diameter_physical"], expected_equivalent)
    assert np.isclose(record["bbox_volume_physical"], 18.0)
    assert np.isclose(record["filled_volume_physical"], 18.0)
    assert np.isclose(record["bbox_axis_0_length_physical"], 4.0)
    assert np.isclose(record["bbox_axis_1_length_physical"], 3.0)
    assert np.isclose(record["bbox_axis_2_length_physical"], 1.5)
    assert record["major_axis_length_physical"] > 0.0
    assert record["minor_axis_length_physical"] > 0.0
    assert record["inertia_tensor_eigval_0_physical"] >= 0.0
    assert table.unit_for("equivalent_diameter_physical") == "micrometer"
    assert table.unit_for("bbox_volume_physical") == "micrometer^3"
    assert table.unit_for("bbox_axis_0_length_physical") == "micrometer"
    assert table.unit_for("inertia_tensor_eigval_0_physical") == "micrometer^2"


def test_measure_3d_mesh_morphology_reports_surface_and_failure_status():
    labels = np.zeros((8, 14, 14), dtype=np.int32)
    labels[1:6, 1:6, 1:6] = 1
    labels[6, 11, 11:13] = 2

    table = measure_3d_mesh_morphology(
        labels,
        resolved_spatial_ndim=3,
        minimum_voxel_count=4,
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(2.0, 0.5, 0.5),
        axis_units=("micrometer", "micrometer", "micrometer"),
    )
    records = table.records()

    assert table.row_count == 2
    assert records[0]["label_id"] == 1
    assert records[0]["mesh_status"] == "ok"
    assert records[0]["voxel_count"] == 125
    assert np.isclose(records[0]["voxel_volume_physical"], 62.5)
    assert records[0]["mesh_surface_area_physical"] > 0
    assert records[0]["mesh_volume_physical"] > 0
    assert records[0]["sphericity"] > 0
    assert records[0]["solidity_3d"] > 0
    assert records[0]["physical_unit"] == "micrometer"
    assert table.unit_for("mesh_surface_area_physical") == "micrometer^2"
    assert table.unit_for("mesh_volume_physical") == "micrometer^3"
    assert records[1]["label_id"] == 2
    assert records[1]["mesh_status"] == "skipped_too_few_voxels"
    assert np.isnan(records[1]["mesh_volume_physical"])


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
        include_derived_shape_ratios=True,
        include_2d_shape_moments=True,
    )
    record = table.records()[0]

    assert "intensity_mean" in table.columns
    assert "major_axis_length_pixels" in table.columns
    assert "axis_ratio_major_minor" in table.columns
    assert "circularity" in table.columns
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


def test_merge_tables_auto_uses_branch_id_for_skeleton_branch_tables():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True
    branches = measure_skeleton_branches(skeleton, resolved_spatial_ndim=2)

    merged = merge_tables(
        [branches, branches],
        input_count=2,
        join_mode="Left join",
        join_keys="auto",
    )

    assert merged.row_count == branches.row_count
    assert "branch_length_pixels_table2" in merged.columns
    assert [record["branch_id"] for record in merged.records()] == [1, 2, 3, 4]


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
    empty = select_table_columns(table, columns="__none__")

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
    assert empty.columns == ()
    assert empty.rows == ((), ())


def test_colocalization_metrics_overlay_scatter_and_racc_outputs():
    channel_1 = np.zeros((5, 6), dtype=np.uint16)
    channel_2 = np.zeros((5, 6), dtype=np.uint16)
    channel_1[1:3, 1:3] = np.asarray([[80, 100], [120, 140]], dtype=np.uint16)
    channel_2[1:3, 1:3] = np.asarray([[90, 110], [130, 150]], dtype=np.uint16)
    channel_1[3, 1] = 150
    channel_2[3, 4] = 150

    table = colocalization_metrics(
        [channel_1, channel_2],
        channel_1_threshold=20,
        channel_2_threshold=20,
    )
    record = table.records()[0]

    assert table.table_kind == "colocalization metrics"
    assert record["threshold_mode"] == "Manual"
    assert record["colocalized_voxels"] == 4
    assert record["channel_1_positive_voxels"] == 5
    assert record["channel_2_positive_voxels"] == 5
    assert np.isclose(record["colocalized_fraction"], 4 / 30)
    assert record["manders_m1"] > 0
    assert record["manders_m2"] > 0

    roi = np.zeros_like(channel_1, dtype=bool)
    roi[1:4, 1:4] = True
    masked_table = colocalization_metrics(
        [channel_1, channel_2, roi],
        channel_1_threshold=20,
        channel_2_threshold=20,
    )
    masked_record = masked_table.records()[0]
    assert masked_record["mask_restricted"] is True
    assert masked_record["total_voxels"] == int(np.count_nonzero(roi))
    assert masked_record["colocalized_voxels"] == 4

    costes = colocalization_metrics(
        [channel_1, channel_2],
        threshold_mode="Costes auto",
    )
    costes_record = costes.records()[0]
    assert 0 <= costes_record["channel_1_threshold"] <= 255
    assert 0 <= costes_record["channel_2_threshold"] <= 255
    assert costes_record["costes_iterations"] >= 0
    threshold_1, threshold_2 = colocalization_threshold_values(
        [channel_1, channel_2],
        threshold_mode="Costes auto",
    )
    assert np.isclose(costes_record["channel_1_threshold"], threshold_1)
    assert np.isclose(costes_record["channel_2_threshold"], threshold_2)

    overlay = colocalized_voxels(
        [channel_1, channel_2],
        channel_1_threshold=20,
        channel_2_threshold=20,
        display_mode="White on black",
    )
    assert overlay.shape == channel_1.shape + (3,)
    assert overlay.dtype == np.float32
    assert np.allclose(overlay[1, 1], (1.0, 1.0, 1.0))
    assert np.allclose(overlay[3, 1], (0.0, 0.0, 0.0))
    masked_overlay = colocalized_voxels(
        [channel_1, channel_2, roi],
        channel_1_threshold=20,
        channel_2_threshold=20,
        display_mode="Channel colors only",
    )
    assert masked_overlay.shape == channel_1.shape + (3,)
    assert np.allclose(masked_overlay[0, 0], (0.0, 0.0, 0.0))
    assert float(masked_overlay[1, 1].max()) > 0.0

    scatter = colocalization_scatter_plot(
        [channel_1, channel_2],
        channel_1_threshold=20,
        channel_2_threshold=20,
        bins=64,
    )
    assert scatter.shape == (512, 512, 3)
    assert scatter.dtype == np.float32
    assert float(scatter.max()) <= 1.0

    index = racc_index(
        [channel_1, channel_2],
        channel_1_threshold=20,
        channel_2_threshold=20,
    )
    assert index.shape == channel_1.shape
    assert index.dtype == np.float32
    assert float(index.max()) > 0.0
    assert index[3, 1] == 0.0
    masked_index = racc_index(
        [channel_1, channel_2, roi],
        channel_1_threshold=20,
        channel_2_threshold=20,
    )
    assert masked_index.shape == channel_1.shape
    assert masked_index[0, 0] == 0.0
    assert masked_index[3, 4] == 0.0


def test_object_colocalization_and_association_tables():
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 2
    channel_1 = np.zeros_like(labels, dtype=np.uint16)
    channel_2 = np.zeros_like(labels, dtype=np.uint16)
    channel_1[labels == 1] = 100
    channel_2[labels == 1] = 80
    channel_1[labels == 2] = 100

    stacked_labels = np.stack([labels, labels])
    stacked_channel_1 = np.stack([channel_1, channel_1])
    stacked_channel_2 = np.stack([channel_2, channel_2])
    object_table = object_colocalization_metrics(
        [stacked_labels, stacked_channel_1, stacked_channel_2],
        channel_1_threshold=50,
        channel_2_threshold=50,
        spatial_mode="2D YX",
        axis_names=("t", "y", "x"),
        axis_types=("time", "space", "space"),
    )
    object_records = object_table.records()
    label_1 = next(
        record
        for record in object_records
        if record["t_index"] == 0 and record["label_id"] == 1
    )
    label_2 = next(
        record
        for record in object_records
        if record["t_index"] == 0 and record["label_id"] == 2
    )

    assert object_table.table_kind == "per-object colocalization metrics"
    assert object_table.columns[:2] == ("t_index", "label_id")
    assert object_table.row_count == 4
    assert label_1["object_voxels"] == 4
    assert label_1["colocalized_voxels"] == 4
    assert np.isclose(label_1["manders_m1"], 1.0)
    assert np.isclose(label_1["manders_m2"], 1.0)
    assert label_2["channel_1_positive_voxels"] == 4
    assert label_2["channel_2_positive_voxels"] == 0
    assert label_2["colocalized_voxels"] == 0

    target = np.zeros_like(labels)
    target[1:3, 1:2] = 5
    target[3:5, 4:6] = 6
    overlap_table = label_overlap_association(
        [labels, target],
        spatial_mode="2D YX",
    )
    overlap_records = {
        (record["label_id"], record["target_label_id"]): record
        for record in overlap_table.records()
    }
    assert overlap_table.table_kind == "label overlap association"
    assert overlap_records[(1, 5)]["overlap_voxels"] == 2
    assert np.isclose(
        overlap_records[(1, 5)]["reference_overlap_fraction"],
        0.5,
    )
    assert np.isclose(overlap_records[(2, 6)]["intersection_over_union"], 2 / 6)

    distance_table = nearest_object_distance(
        [labels, target],
        spatial_mode="2D YX",
        axis_names=("y", "x"),
        axis_types=("space", "space"),
        axis_scales=(0.5, 2.0),
        axis_units=("micrometer", "micrometer"),
    )
    distance_records = {
        record["label_id"]: record for record in distance_table.records()
    }
    assert distance_table.table_kind == "nearest object distance"
    assert distance_records[1]["nearest_label_id"] == 5
    assert np.isclose(distance_records[1]["centroid_distance_pixels"], 0.5)
    assert np.isclose(distance_records[1]["centroid_distance_physical"], 1.0)
    assert distance_records[1]["physical_unit"] == "micrometer"

    events = np.zeros_like(labels)
    events[1, 1] = 1
    events[0, 0] = 2
    localization_table = event_localization(
        [events, labels],
        spatial_mode="2D YX",
    )
    localization_records = {
        record["event_id"]: record for record in localization_table.records()
    }
    assert localization_table.table_kind == "event localization"
    assert localization_records[1]["region_label_id"] == 1
    assert localization_records[1]["in_region"] is True
    assert localization_records[2]["region_label_id"] == 0
    assert localization_records[2]["in_region"] is False


def test_summarize_measurements_groups_by_metadata_and_units():
    table = table_from_columns(
        {
            "condition": ["control", "control", "drug"],
            "t_index": [0, 0, 0],
            "label_id": [1, 2, 1],
            "area_pixels": [24, 30, 40],
            "intensity_mean": [10.0, 20.0, 30.0],
        },
        column_units={"area_pixels": "pixel"},
    )

    summarized = summarize_measurements(
        table,
        group_by="condition,t_index",
        value_columns="area_pixels",
        statistics="count,mean,median,std,min,max,q25,q75",
    )
    records = summarized.records()

    assert summarized.columns == (
        "condition",
        "t_index",
        "row_count",
        "area_pixels_count",
        "area_pixels_mean",
        "area_pixels_median",
        "area_pixels_std",
        "area_pixels_min",
        "area_pixels_max",
        "area_pixels_q25",
        "area_pixels_q75",
    )
    assert summarized.unit_for("area_pixels_mean") == "pixel"
    assert summarized.unit_for("area_pixels_count") == ""
    assert records[0]["condition"] == "control"
    assert records[0]["row_count"] == 2
    assert records[0]["area_pixels_count"] == 2
    assert records[0]["area_pixels_mean"] == 27.0
    assert np.isclose(records[0]["area_pixels_std"], np.sqrt(18.0))
    assert records[0]["area_pixels_q25"] == 25.5
    assert records[0]["area_pixels_q75"] == 28.5
    assert records[1]["condition"] == "drug"
    assert records[1]["row_count"] == 1
    assert records[1]["area_pixels_std"] == 0.0


def test_summarize_measurements_auto_groups_by_time_index():
    table = table_from_columns(
        {
            "t_index": [0, 0, 1],
            "label_id": [1, 2, 1],
            "area_pixels": [24, 30, 40],
        }
    )

    summarized = summarize_measurements(
        table,
        value_columns="area_pixels",
        statistics="mean,min,max",
    )

    assert summarized.columns == (
        "t_index",
        "row_count",
        "area_pixels_mean",
        "area_pixels_min",
        "area_pixels_max",
    )
    assert summarized.records()[0]["area_pixels_mean"] == 27.0
    assert summarized.records()[1]["area_pixels_mean"] == 40.0


def test_summarize_skeleton_branches_groups_branch_distributions():
    table = table_from_columns(
        {
            "t_index": [0, 0, 0, 1],
            "component_id": [1, 1, 2, 1],
            "branch_id": [1, 2, 3, 1],
            "branch_type": [
                "endpoint_to_junction",
                "endpoint_to_junction",
                "isolated",
                "junction_to_junction",
            ],
            "branch_length_pixels": [2.0, 4.0, 0.0, 6.0],
            "branch_tortuosity": [1.0, 1.2, 0.0, 1.5],
        },
        table_kind="Skeleton branches",
        column_units={"branch_length_pixels": "pixels"},
    )

    summarized = summarize_skeleton_branches(
        table,
        group_by="t_index",
        statistics="mean,median,std,min,max,q25,q75",
    )
    records = summarized.records()

    assert summarized.table_kind == "Skeleton branch summary"
    assert summarized.unit_for("branch_length_pixels_total") == "pixels"
    assert summarized.unit_for("branch_tortuosity_mean") == "ratio"
    assert summarized.columns[:5] == (
        "t_index",
        "branch_count",
        "component_count",
        "branch_length_pixels_total",
        "branch_length_pixels_mean",
    )
    assert records[0]["t_index"] == 0
    assert records[0]["branch_count"] == 3
    assert records[0]["component_count"] == 2
    assert records[0]["branch_length_pixels_total"] == 6.0
    assert records[0]["branch_length_pixels_mean"] == 2.0
    assert np.isclose(records[0]["branch_length_pixels_std"], 2.0)
    assert records[0]["branch_type_endpoint_to_junction_count"] == 2
    assert np.isclose(
        records[0]["branch_type_endpoint_to_junction_fraction"],
        2.0 / 3.0,
    )
    assert records[0]["branch_type_isolated_count"] == 1
    assert records[1]["t_index"] == 1
    assert records[1]["branch_count"] == 1
    assert records[1]["branch_type_junction_to_junction_count"] == 1


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


def test_pipeline_manual_measurement_nodes_skip_calculate_and_stale_cache():
    data = np.zeros((9, 9), dtype=np.float32)
    data[1:4, 1:4] = 10
    data[6, 6] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)

    outputs = pipeline.run(
        data,
        input_metadata={"axes": "YX"},
        manual_mode=MANUAL_RUN_SKIP,
    )

    assert outputs[measurements.id] is None
    assert pipeline.output_states[measurements.id] is None
    assert (
        pipeline.node_execution_states[measurements.id]
        == EXECUTION_NOT_CALCULATED
    )

    outputs = pipeline.run(
        data,
        input_metadata={"axes": "YX"},
        dirty_node_ids={measurements.id},
        manual_mode=MANUAL_RUN_SKIP,
        manual_node_ids={measurements.id},
    )
    table = outputs[measurements.id]

    assert table.row_count == 2
    assert pipeline.node_execution_states[measurements.id] == EXECUTION_READY

    pipeline.set_param(threshold.id, "threshold", 15)
    outputs = pipeline.run(
        data,
        input_metadata={"axes": "YX"},
        dirty_node_ids={threshold.id},
        manual_mode=MANUAL_RUN_SKIP,
    )

    assert outputs[measurements.id] is table
    assert pipeline.node_execution_states[measurements.id] == EXECUTION_STALE
    workflow = serialize_workflow(pipeline)
    workflow_text = repr(workflow)
    assert "node_execution_states" not in workflow_text
    assert "node_outputs" not in workflow_text


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


def test_pipeline_summarizes_measurement_table_by_time_index():
    data = np.zeros((2, 12, 12), dtype=np.float32)
    data[0, 1:4, 1:5] = 10
    data[0, 7:10, 7:11] = 10
    data[1, 2:7, 2:6] = 10
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    summarized = pipeline.add_node("summarize_measurements")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(summarized.id, "value_columns", "area_pixels")
    pipeline.set_param(summarized.id, "statistics", "count,mean,min,max")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect(measurements.id, summarized.id)

    outputs = pipeline.run(data, input_metadata={"axes": "TYX"})
    table = outputs[summarized.id]
    state = pipeline.output_states[summarized.id]
    records = table.records()

    assert table.row_count == 2
    assert records[0]["t_index"] == 0
    assert records[0]["row_count"] == 2
    assert records[0]["area_pixels_count"] == 2
    assert records[0]["area_pixels_mean"] == 12.0
    assert records[1]["t_index"] == 1
    assert records[1]["row_count"] == 1
    assert records[1]["area_pixels_mean"] == 20.0
    assert state.history[-1] == "Summarize Measurements: summarized 2 groups"


def test_pipeline_summarizes_skeleton_branch_table_by_time_index():
    data = np.zeros((2, 7, 7), dtype=np.float32)
    data[:, 1:6, 3] = 1
    data[0, 3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    branches = pipeline.add_node("measure_skeleton_branches")
    summarized = pipeline.add_node("summarize_skeleton_branches")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, branches.id)
    pipeline.connect(branches.id, summarized.id)

    outputs = pipeline.run(data, input_metadata={"axes": "TYX"})
    table = outputs[summarized.id]
    state = pipeline.output_states[summarized.id]
    records = table.records()

    assert table.row_count == 2
    assert records[0]["t_index"] == 0
    assert records[0]["branch_count"] == 4
    assert records[0]["branch_type_endpoint_to_junction_count"] == 4
    assert records[1]["t_index"] == 1
    assert records[1]["branch_count"] == 1
    assert records[1]["branch_type_endpoint_to_endpoint_count"] == 1
    assert "branch_length_pixels_total" in state.columns
    assert state.history[-1] == "Summarize Skeleton Branches: summarized 2 groups"


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


def test_skeleton_keypoints_identifies_endpoints_junctions_and_isolates():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True
    skeleton[0, 0] = True

    endpoints, junctions, isolated = skeleton_keypoints(
        skeleton,
        resolved_spatial_ndim=2,
    )

    assert endpoints.sum() == 4
    assert junctions.sum() == 1
    assert isolated.sum() == 1
    assert junctions[3, 3]
    assert isolated[0, 0]


def test_label_skeleton_components_labels_connected_skeleton_blocks():
    skeleton = np.zeros((2, 7, 7), dtype=bool)
    skeleton[0, 1:6, 3] = True
    skeleton[0, 3, 1:6] = True
    skeleton[0, 0, 0] = True
    skeleton[1, 2:5, 2] = True

    labels = label_skeleton_components(skeleton, resolved_spatial_ndim=2)

    assert labels.dtype == np.int32
    assert labels[0].max() == 2
    assert labels[1].max() == 1
    assert labels[0, 3, 3] == labels[0, 1, 3]
    assert labels[0, 0, 0] != labels[0, 3, 3]


def test_label_skeleton_branches_labels_paths_around_junctions():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True

    labels = label_skeleton_branches(skeleton, resolved_spatial_ndim=2)

    assert labels.dtype == np.int32
    assert labels.max() == 4
    assert labels[3, 3] == 0
    assert labels[1, 3] != 0
    assert labels[5, 3] != 0
    assert labels[3, 1] != 0
    assert labels[3, 5] != 0


def test_skeleton_graph_overlay_supports_edge_and_node_modes():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True

    colored = skeleton_graph_overlay(
        skeleton,
        display_mode="Colored edges",
        resolved_spatial_ndim=2,
    )
    node_overlay = skeleton_graph_overlay(
        skeleton,
        display_mode="White edges + colored nodes",
        resolved_spatial_ndim=2,
    )

    assert colored.shape == (7, 7, 3)
    assert colored.dtype == np.float32
    assert np.any(colored[2, 3] != colored[4, 3])
    np.testing.assert_allclose(node_overlay[2, 3], (1.0, 1.0, 1.0))
    np.testing.assert_allclose(node_overlay[1, 3], (0.0, 1.0, 0.0))
    np.testing.assert_allclose(node_overlay[3, 3], (1.0, 0.0, 1.0))


def test_pipeline_skeleton_graph_overlay_is_rgb_image_state():
    image = np.zeros((3, 7, 7), dtype=np.float32)
    image[:, 1:6, 3] = 1
    image[:, 3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    overlay = pipeline.add_node("skeleton_graph_overlay")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, overlay.id)

    outputs = pipeline.run(image, input_metadata={"axes": "ZYX"})
    state = pipeline.output_states[overlay.id]

    assert outputs[overlay.id].shape == (3, 7, 7, 3)
    assert state.kind == "RGB image"
    assert tuple(axis.name for axis in state.axes) == ("z", "y", "x", "rgb")
    assert (
        state.history[-1]
        == "Skeleton Graph Overlay: Colored edges + colored nodes"
    )


def test_measure_skeleton_branches_reports_branch_lengths_and_tortuosity():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True

    table = measure_skeleton_branches(skeleton, resolved_spatial_ndim=2)
    records = table.records()

    assert table.row_count == 4
    assert table.table_kind == "Skeleton branches"
    assert {record["branch_type"] for record in records} == {
        "endpoint_to_junction"
    }
    assert all(record["branch_length_pixels"] == 2.0 for record in records)
    assert all(record["branch_edge_count"] == 2 for record in records)
    assert all(record["branch_tortuosity"] == 1.0 for record in records)
    assert {"start_y", "start_x", "end_y", "end_x"} <= set(table.columns)


def test_skeleton_graph_tables_export_nodes_and_edges():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True

    node_table, edge_table = skeleton_graph_tables(
        skeleton,
        resolved_spatial_ndim=2,
    )
    node_records = node_table.records()
    edge_records = edge_table.records()

    assert node_table.table_kind == "Skeleton graph nodes"
    assert edge_table.table_kind == "Skeleton graph edges"
    assert node_table.row_count == 5
    assert edge_table.row_count == 4
    assert {record["node_type"] for record in node_records} == {
        "endpoint",
        "junction",
    }
    assert sum(record["node_type"] == "junction" for record in node_records) == 1
    assert {record["branch_type"] for record in edge_records} == {
        "endpoint_to_junction"
    }
    assert all(record["start_node_id"] > 0 for record in edge_records)
    assert all(record["end_node_id"] > 0 for record in edge_records)
    assert all(record["branch_length_pixels"] == 2.0 for record in edge_records)
    assert {"y_coord", "x_coord"} <= set(node_table.columns)


def test_measure_overall_skeleton_network_reports_block_metrics():
    skeleton = np.zeros((7, 7), dtype=bool)
    skeleton[1:6, 3] = True
    skeleton[3, 1:6] = True

    table = measure_overall_skeleton_network(skeleton, resolved_spatial_ndim=2)
    record = table.records()[0]

    assert table.row_count == 1
    assert table.table_kind == "Overall skeleton network"
    assert record["component_count"] == 1
    assert record["skeleton_voxel_count"] == 9
    assert record["largest_component_voxel_fraction"] == 1.0
    assert record["endpoint_voxel_count"] == 4
    assert record["junction_voxel_count"] == 1
    assert record["branch_count"] == 4
    assert record["graph_edge_count"] == 4
    assert record["skeleton_length_pixels"] == 8.0
    assert record["mean_branch_length_pixels"] == 2.0
    assert record["median_branch_length_pixels"] == 2.0
    assert record["max_branch_length_pixels"] == 2.0
    assert record["network_connectedness_fraction"] == 1.0
    assert record["branches_per_component"] == 4.0
    assert record["endpoints_per_component"] == 4.0
    assert record["junctions_per_component"] == 1.0
    assert record["branches_per_skeleton_length"] == 0.5
    assert record["endpoints_per_skeleton_length"] == 0.5
    assert record["junctions_per_skeleton_length"] == 0.125
    assert record["isolated_component_fraction"] == 0.0


def test_prune_skeleton_branches_removes_short_terminal_spurs():
    skeleton = np.zeros((7, 9), dtype=bool)
    skeleton[3, 1:8] = True
    skeleton[1:4, 4] = True
    skeleton[4, 4] = True

    pruned = prune_skeleton_branches(
        skeleton,
        min_branch_length=2,
        resolved_spatial_ndim=2,
    )

    assert not pruned[4, 4]
    assert pruned[1, 4]
    assert pruned[3, 1]
    assert pruned[3, 7]
    assert pruned.sum() == skeleton.sum() - 1


def test_prune_skeleton_branches_can_threshold_in_physical_units():
    skeleton = np.zeros((7, 9), dtype=bool)
    skeleton[3, 1:8] = True
    skeleton[1:4, 4] = True
    skeleton[4, 4] = True

    pixel_pruned = prune_skeleton_branches(
        skeleton,
        min_branch_length=0.75,
        length_units="Pixels/voxels",
        resolved_spatial_ndim=2,
        axis_scales=(0.5, 0.5),
    )
    physical_pruned = prune_skeleton_branches(
        skeleton,
        min_branch_length=0.75,
        length_units="Physical units",
        resolved_spatial_ndim=2,
        axis_scales=(0.5, 0.5),
    )

    assert pixel_pruned[4, 4]
    assert not physical_pruned[4, 4]
    assert physical_pruned.sum() == skeleton.sum() - 1


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


def test_pipeline_skeleton_branch_measurement_creates_table_state():
    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    measurements = pipeline.add_node("measure_skeleton_branches")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, measurements.id)

    outputs = pipeline.run(image, input_metadata={"axes": "YX"})
    table = outputs[measurements.id]
    state = pipeline.output_states[measurements.id]

    assert table.row_count == 4
    assert state.kind == "measurement table"
    assert state.table_kind == "Skeleton branches"
    assert "branch_length_pixels" in state.columns
    assert state.history[-1] == "Measure Skeleton Branches: measured 4 branches"


def test_pipeline_skeleton_graph_tables_create_two_table_states():
    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    graph_tables = pipeline.add_node("skeleton_graph_tables")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, graph_tables.id)

    pipeline.run(image, input_metadata={"axes": "YX"})
    outputs = pipeline.node_outputs[graph_tables.id]
    states = pipeline.node_output_states[graph_tables.id]

    assert [table.row_count for table in outputs] == [5, 4]
    assert [state.table_kind for state in states] == [
        "Skeleton graph nodes",
        "Skeleton graph edges",
    ]
    assert states[0].history[-1] == (
        "Skeleton Graph Tables (Graph nodes): exported 5 nodes"
    )
    assert states[1].history[-1] == (
        "Skeleton Graph Tables (Graph edges): exported 4 edges"
    )


def test_pipeline_measure_overall_skeleton_network_creates_table_state():
    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    summary = pipeline.add_node("measure_overall_skeleton_network")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, summary.id)

    outputs = pipeline.run(image, input_metadata={"axes": "YX"})
    table = outputs[summary.id]
    state = pipeline.output_states[summary.id]

    assert table.row_count == 1
    assert state.table_kind == "Overall skeleton network"
    assert "network_connectedness_fraction" in state.columns
    assert state.history[-1] == "Measure Overall Skeleton Network: measured 1 block"


def test_pipeline_skeleton_keypoints_creates_three_mask_outputs():
    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    keypoints = pipeline.add_node("skeleton_keypoints")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, keypoints.id)

    pipeline.run(image, input_metadata={"axes": "YX"})
    outputs = pipeline.node_outputs[keypoints.id]
    states = pipeline.node_output_states[keypoints.id]

    assert [int(output.sum()) for output in outputs] == [4, 1, 0]
    assert [state.kind for state in states] == [
        "binary mask",
        "binary mask",
        "binary mask",
    ]
    assert (
        states[0].history[-1]
        == "Skeleton Keypoints (Endpoints): extracted Endpoints"
    )
    assert (
        states[1].history[-1]
        == "Skeleton Keypoints (Junctions): extracted Junctions"
    )
    assert (
        states[2].history[-1]
        == "Skeleton Keypoints (Isolated nodes): extracted Isolated nodes"
    )


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


def test_save_table_output_rejects_blank_path():
    table = table_from_columns({"label_id": [1]})

    with pytest.raises(ValueError, match="blank"):
        save_table_output(table, "")


def test_image_math_requires_configured_input_count():
    image = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="needs 2 connected input"):
        add_images([image], input_count=2)


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


def test_born_wolf_psf_generates_normalized_3d_metadata_sized_kernel():
    reference = np.zeros((4, 24, 24), dtype=np.uint16)

    psf = born_wolf_psf(
        reference,
        spatial_mode="3D ZYX",
        xy_size=17,
        z_size=7,
        pupil_samples=64,
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(0.3, 0.1, 0.1),
        axis_units=("micrometer", "micrometer", "micrometer"),
        channel_emission_wavelengths=(610.0,),
        channel_emission_wavelength_units=("nanometer",),
        numerical_aperture=1.2,
        refractive_index=1.33,
        pixel_size_xy_um=0.1,
        z_step_um=0.3,
    )

    assert psf.shape == (7, 17, 17)
    assert psf.dtype == np.float32
    assert np.isclose(float(psf.sum()), 1.0, rtol=1e-5)
    assert psf[3, 8, 8] == psf.max()


def test_born_wolf_psf_can_generate_2d_kernel():
    reference = np.zeros((24, 24), dtype=np.float32)

    psf = born_wolf_psf(
        reference,
        spatial_mode="2D YX",
        xy_size=15,
        z_size=9,
        pupil_samples=48,
        wavelength_nm=520.0,
        numerical_aperture=1.4,
        refractive_index=1.515,
        pixel_size_xy_um=0.1,
        z_step_um=0.3,
    )

    assert psf.shape == (15, 15)
    assert np.isclose(float(psf.sum()), 1.0, rtol=1e-5)
    assert psf[7, 7] == psf.max()


def test_born_wolf_psf_outputs_generates_one_kernel_per_auto_channel():
    reference = np.zeros((2, 3, 24, 24), dtype=np.float32)

    psfs = born_wolf_psf_outputs(
        reference,
        spatial_mode="3D ZYX",
        xy_size=15,
        z_size=5,
        pupil_samples=48,
        axis_names=("c", "z", "y", "x"),
        axis_types=("channel", "space", "space", "space"),
        axis_scales=(1.0, 0.3, 0.1, 0.1),
        axis_units=(None, "micrometer", "micrometer", "micrometer"),
        channel_emission_wavelengths=(520.0, 620.0),
        channel_emission_wavelength_units=("nanometer", "nanometer"),
        numerical_aperture=1.2,
        refractive_index=1.33,
        pixel_size_xy_um=0.1,
        z_step_um=0.3,
    )

    assert len(psfs) == 2
    assert psfs[0].shape == (5, 15, 15)
    assert psfs[1].shape == (5, 15, 15)
    assert np.isclose(float(psfs[0].sum()), 1.0, rtol=1e-5)
    assert np.isclose(float(psfs[1].sum()), 1.0, rtol=1e-5)
    assert not np.allclose(psfs[0], psfs[1])


def test_born_wolf_psf_pipeline_uses_carried_image_metadata():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("born_wolf_psf")
    pipeline.connect("input", node.id)
    pipeline.set_param(node.id, "xy_size", 15)
    pipeline.set_param(node.id, "z_size", 5)
    pipeline.set_param(node.id, "pupil_samples", 48)

    data = np.zeros((3, 16, 16), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", unit="micrometer", scale=0.4),
            AxisMetadata("y", "space", unit="micrometer", scale=0.08),
            AxisMetadata("x", "space", unit="micrometer", scale=0.08),
        ),
        channels=(
            ChannelMetadata(
                name="Far red",
                emission_wavelength=650.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.2,
            refractive_index=1.33,
        ),
    )
    assert state is not None

    pipeline.run(
        data,
        source_payloads={
            "input": SourcePayload(data, name="metadata test", image_state=state)
        },
    )

    psf = pipeline.outputs[node.id]
    psf_state = pipeline.output_states[node.id]

    assert psf.shape == (5, 15, 15)
    assert np.isclose(float(psf.sum()), 1.0, rtol=1e-5)
    assert psf_state is not None
    assert psf_state.axis_order == "ZYX"
    assert [axis.scale for axis in psf_state.axes] == [0.4, 0.08, 0.08]
    assert psf_state.channels[0].name == "Far red"


def test_born_wolf_psf_pipeline_outputs_channel_specific_psfs():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("born_wolf_psf")
    pipeline.connect("input", node.id)
    pipeline.set_param(node.id, "xy_size", 15)
    pipeline.set_param(node.id, "z_size", 5)
    pipeline.set_param(node.id, "pupil_samples", 48)

    data = np.zeros((2, 3, 16, 16), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", unit="micrometer", scale=0.4),
            AxisMetadata("y", "space", unit="micrometer", scale=0.08),
            AxisMetadata("x", "space", unit="micrometer", scale=0.08),
        ),
        channels=(
            ChannelMetadata(
                name="green",
                emission_wavelength=520.0,
                emission_wavelength_unit="nanometer",
            ),
            ChannelMetadata(
                name="red",
                emission_wavelength=620.0,
                emission_wavelength_unit="nanometer",
            ),
        ),
        acquisition=AcquisitionMetadata(
            objective_na=1.2,
            refractive_index=1.33,
        ),
    )
    assert state is not None

    pipeline.run(
        data,
        source_payloads={
            "input": SourcePayload(data, name="metadata test", image_state=state)
        },
    )

    ports = pipeline.output_ports(node.id)

    assert [port.label for port in ports] == ["green PSF", "red PSF"]
    assert len(pipeline.node_outputs[node.id]) == 2
    assert pipeline.node_outputs[node.id][0].shape == (5, 15, 15)
    assert pipeline.node_outputs[node.id][1].shape == (5, 15, 15)
    assert not np.allclose(
        pipeline.node_outputs[node.id][0],
        pipeline.node_outputs[node.id][1],
    )
    assert pipeline.node_output_states[node.id][0].axis_order == "ZYX"
    assert pipeline.node_output_states[node.id][0].channels[0].name == "green"
    assert pipeline.node_output_states[node.id][1].channels[0].name == "red"
    assert "wavelength 620" in pipeline.node_output_states[node.id][1].history[-1]


def test_born_wolf_psf_auto_requires_resolved_metadata():
    reference = np.zeros((4, 24, 24), dtype=np.uint16)

    with pytest.raises(ValueError, match="unresolved parameter"):
        born_wolf_psf(
            reference,
            spatial_mode="3D ZYX",
            xy_size=17,
            z_size=7,
            pupil_samples=64,
            axis_names=("z", "y", "x"),
            axis_types=("space", "space", "space"),
            axis_scales=(0.3, 0.1, 0.1),
            axis_units=("micrometer", "micrometer", "micrometer"),
        )


def test_prepare_validate_psf_clips_centers_forces_odd_and_normalizes():
    psf = np.zeros((4, 4), dtype=np.float32)
    psf[0, 1] = 5.0
    psf[3, 3] = -3.0
    psf[1, 1] = np.nan

    prepared = prepare_validate_psf(psf)

    assert prepared.shape == (5, 5)
    assert prepared.dtype == np.float32
    assert prepared[2, 2] == prepared.max()
    assert prepared.min() >= 0
    assert np.isclose(float(prepared.sum()), 1.0)


def test_prepare_validate_psf_rejects_invalid_psfs():
    with pytest.raises(ValueError, match="empty|invalid"):
        prepare_validate_psf(np.full((3, 3), np.nan, dtype=np.float32))

    with pytest.raises(ValueError, match="without time or channel axes"):
        prepare_validate_psf(np.zeros((2, 3, 3, 1), dtype=np.float32))


def test_richardson_lucy_deconvolution_restores_2d_fixture():
    truth = np.zeros((17, 17), dtype=np.float32)
    truth[8, 8] = 1.0
    psf = _compact_psf_2d()
    blurred = signal.convolve(truth, psf, mode="same").astype(np.float32)

    restored = richardson_lucy_deconvolution(
        [blurred, psf],
        spatial_mode="2D YX",
        iterations=8,
    )

    assert restored.shape == blurred.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0
    assert restored[8, 8] > blurred[8, 8]


def test_richardson_lucy_deconvolution_restores_3d_fixture():
    truth = np.zeros((5, 13, 13), dtype=np.float32)
    truth[2, 6, 6] = 1.0
    psf = _compact_psf_3d()
    blurred = signal.convolve(truth, psf, mode="same").astype(np.float32)

    restored = richardson_lucy_deconvolution(
        [blurred, psf],
        spatial_mode="3D ZYX",
        iterations=6,
    )

    assert restored.shape == blurred.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0
    assert restored[2, 6, 6] > blurred[2, 6, 6]


def test_richardson_lucy_tv_zero_regularization_matches_rl_update():
    image = np.zeros((13, 13), dtype=np.float32)
    image[6, 6] = 1.0
    image = signal.convolve(image, _compact_psf_2d(), mode="same").astype(np.float32)

    rl = richardson_lucy_deconvolution(
        [image, _compact_psf_2d()],
        spatial_mode="2D YX",
        iterations=5,
        preserve_input_scale=False,
    )
    tv_zero = richardson_lucy_tv_deconvolution(
        [image, _compact_psf_2d()],
        spatial_mode="2D YX",
        iterations=5,
        tv_regularization=0.0,
        preserve_input_scale=False,
    )

    np.testing.assert_allclose(tv_zero, rl, rtol=2e-5, atol=2e-6)


def test_richardson_lucy_tv_reduces_noise_variation_versus_ordinary_rl():
    yy, xx = np.mgrid[:32, :32]
    truth = (
        ((yy - 11) ** 2 + (xx - 11) ** 2 <= 4**2).astype(np.float32)
        + 0.7 * ((yy - 21) ** 2 + (xx - 22) ** 2 <= 3**2)
    )
    psf = _compact_psf_2d()
    blurred = signal.convolve(truth, psf, mode="same").astype(np.float32)
    rng = np.random.default_rng(12)
    noisy = np.clip(
        blurred + rng.normal(0.0, 0.03, size=blurred.shape).astype(np.float32),
        0.0,
        None,
    )

    rl = richardson_lucy_deconvolution(
        [noisy, psf],
        spatial_mode="2D YX",
        iterations=20,
    )
    tv = richardson_lucy_tv_deconvolution(
        [noisy, psf],
        spatial_mode="2D YX",
        iterations=20,
        tv_regularization=0.02,
        denominator_floor=0.2,
    )

    assert tv.shape == noisy.shape
    assert tv.dtype == np.float32
    assert np.all(np.isfinite(tv))
    assert tv.min() >= 0
    assert _total_variation(tv) < _total_variation(rl)


def test_richardson_lucy_2d_mode_processes_leading_axes_independently():
    stack = np.zeros((2, 15, 15), dtype=np.float32)
    stack[0, 7, 7] = 1.0
    stack[1, 4, 9] = 0.5
    psf = _compact_psf_2d()
    blurred = signal.convolve(stack, psf[None, :, :], mode="same").astype(np.float32)

    restored_stack = richardson_lucy_deconvolution(
        [blurred, psf],
        spatial_mode="2D YX",
        iterations=6,
    )
    restored_first = richardson_lucy_deconvolution(
        [blurred[0], psf],
        spatial_mode="2D YX",
        iterations=6,
    )
    restored_second = richardson_lucy_deconvolution(
        [blurred[1], psf],
        spatial_mode="2D YX",
        iterations=6,
    )

    np.testing.assert_allclose(restored_stack[0], restored_first)
    np.testing.assert_allclose(restored_stack[1], restored_second)


def test_richardson_lucy_rejects_mismatched_psf_dimensionality():
    image = np.zeros((5, 13, 13), dtype=np.float32)

    with pytest.raises(ValueError, match="dimensionality"):
        richardson_lucy_deconvolution(
            [image, _compact_psf_2d()],
            spatial_mode="3D ZYX",
        )


def test_richardson_lucy_tv_reports_progress_and_cancels_between_iterations():
    image = np.zeros((13, 13), dtype=np.float32)
    image[6, 6] = 1.0
    updates = []
    state = {"cancel": False}

    def report(update):
        updates.append((update.current, update.total, update.message))
        if update.current >= 1:
            state["cancel"] = True

    progress = ProgressContext(
        cancelled=lambda: state["cancel"],
        reporter=report,
    )

    with pytest.raises(OperationCancelled):
        richardson_lucy_tv_deconvolution(
            [image, _compact_psf_2d()],
            spatial_mode="2D YX",
            iterations=4,
            progress=progress,
        )

    assert updates[0] == (0, 4, "Richardson-Lucy TV deconvolution")
    assert updates[1] == (1, 4, "Richardson-Lucy TV deconvolution")


def test_deconvolution_pipeline_metadata_follows_image_input():
    image = np.zeros((2, 13, 13), dtype=np.float32)
    image[:, 6, 6] = 1.0
    psf = _compact_psf_2d()
    image_state = image_state_from_array(
        image,
        axes=(
            AxisMetadata("t", "time", unit="second", scale=2.0),
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
        source_name="image source",
        channels=(ChannelMetadata(name="GFP", color=0x00FF00),),
    )
    psf_state = image_state_from_array(
        psf,
        axes=(
            AxisMetadata("y", "space", unit="micrometer", scale=0.2),
            AxisMetadata("x", "space", unit="micrometer", scale=0.2),
        ),
        source_name="psf source",
        channels=(ChannelMetadata(name="PSF channel", color=0xFF00FF),),
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    decon = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(decon.id, "spatial_mode", "2D YX")
    pipeline.set_param(decon.id, "iterations", 3)

    assert [port.name for port in pipeline.input_ports(decon.id)] == ["image", "psf"]
    assert [port.label for port in pipeline.input_ports(decon.id)] == ["Image", "PSF"]

    pipeline.connect("input", decon.id, target_port=0)
    pipeline.connect(psf_source.id, decon.id, target_port=1)

    outputs = pipeline.run(
        image,
        source_payloads={
            "input": SourcePayload(image, name="image source", image_state=image_state),
            psf_source.id: SourcePayload(psf, name="psf source", image_state=psf_state),
        },
    )
    state = pipeline.output_states[decon.id]

    assert pipeline.is_manual_node(decon.id)
    assert outputs[decon.id].dtype == np.float32
    assert state.axis_order == "TYX"
    assert state.source_name == "image source"
    assert state.channels[0].name == "GFP"
    assert [axis.scale for axis in state.axes] == [2.0, 0.2, 0.2]
    assert "Richardson-Lucy Deconvolution" in state.history[-1]


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


def _different_range_filter_stack(dtype=np.float32):
    rng = np.random.default_rng(3)
    first = 100.0 + 50.0 * rng.random((12, 13))
    return np.stack((first, first + 1_000.0)).astype(dtype)


@pytest.mark.parametrize("x_size", (3, 4))
@pytest.mark.parametrize(
    ("operation", "module", "function_name", "kwargs"),
    (
        pytest.param(
            bilateral_filter,
            operations.restoration,
            "denoise_bilateral",
            {"diameter": 3, "sigma_color": 5.0, "sigma_space": 1.0},
            id="bilateral",
        ),
        pytest.param(
            unsharp_mask_filter,
            operations.filters,
            "unsharp_mask",
            {"radius": 1.0, "amount": 0.5},
            id="unsharp",
        ),
        pytest.param(
            non_local_means_filter,
            operations.restoration,
            "denoise_nl_means",
            {"patch_size": 3, "patch_distance": 1, "h": 0.05},
            id="non-local-means",
        ),
    ),
)
def test_denoising_filters_treat_rgb_sized_x_as_scalar_by_default(
    monkeypatch,
    operation,
    module,
    function_name,
    kwargs,
    x_size,
):
    calls = []

    def record_filter(image, **filter_kwargs):
        calls.append((tuple(image.shape), filter_kwargs.get("channel_axis")))
        return np.asarray(image).copy()

    monkeypatch.setattr(module, function_name, record_filter)
    data = np.arange(2 * 5 * x_size, dtype=np.float32).reshape(2, 5, x_size)

    result = operation(data, **kwargs)

    assert calls == [((5, x_size), None), ((5, x_size), None)]
    np.testing.assert_allclose(result, data, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "module", "function_name", "kwargs"),
    (
        pytest.param(
            bilateral_filter,
            operations.restoration,
            "denoise_bilateral",
            {"diameter": 3, "sigma_color": 5.0, "sigma_space": 1.0},
            id="bilateral",
        ),
        pytest.param(
            unsharp_mask_filter,
            operations.filters,
            "unsharp_mask",
            {"radius": 1.0, "amount": 0.5},
            id="unsharp",
        ),
        pytest.param(
            non_local_means_filter,
            operations.restoration,
            "denoise_nl_means",
            {"patch_size": 3, "patch_distance": 1, "h": 0.05},
            id="non-local-means",
        ),
    ),
)
def test_denoising_filters_use_declared_yxc_channel_axis_only(
    monkeypatch,
    operation,
    module,
    function_name,
    kwargs,
):
    calls = []

    def record_filter(image, **filter_kwargs):
        calls.append((tuple(image.shape), filter_kwargs.get("channel_axis")))
        return np.asarray(image).copy()

    monkeypatch.setattr(module, function_name, record_filter)
    data = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)

    scalar_result = operation(data, **kwargs)
    scalar_calls = list(calls)
    calls.clear()
    color_result = operation(data, channel_axis=2, **kwargs)

    assert scalar_calls == [((4, 3), None)] * 5
    assert calls == [((5, 4, 3), -1)]
    np.testing.assert_allclose(scalar_result, data, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(color_result, data, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "module", "function_name", "kwargs"),
    (
        pytest.param(
            bilateral_filter,
            operations.restoration,
            "denoise_bilateral",
            {"diameter": 3, "sigma_color": 5.0, "sigma_space": 1.0},
            id="bilateral",
        ),
        pytest.param(
            unsharp_mask_filter,
            operations.filters,
            "unsharp_mask",
            {"radius": 1.0, "amount": 0.5},
            id="unsharp",
        ),
        pytest.param(
            non_local_means_filter,
            operations.restoration,
            "denoise_nl_means",
            {"patch_size": 3, "patch_distance": 1, "h": 0.05},
            id="non-local-means",
        ),
    ),
)
def test_denoising_filters_restore_nontrailing_channel_axis(
    monkeypatch,
    operation,
    module,
    function_name,
    kwargs,
):
    calls = []

    def record_filter(image, **filter_kwargs):
        calls.append((tuple(image.shape), filter_kwargs.get("channel_axis")))
        return np.asarray(image).copy()

    monkeypatch.setattr(module, function_name, record_filter)
    data = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    original = data.copy()

    result = operation(data, channel_axis=0, **kwargs)

    assert calls == [((5, 4, 3), -1)]
    assert result.shape == data.shape
    np.testing.assert_allclose(result, data, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "operation",
    (bilateral_filter, unsharp_mask_filter, non_local_means_filter),
)
@pytest.mark.parametrize("channel_axis", (3, -4))
def test_denoising_filters_reject_out_of_range_channel_axis(
    operation,
    channel_axis,
):
    data = np.zeros((2, 5, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="channel_axis .* is out of range"):
        operation(data, channel_axis=channel_axis)


@pytest.mark.parametrize(
    "operation",
    (bilateral_filter, unsharp_mask_filter, non_local_means_filter),
)
@pytest.mark.parametrize("channel_axis", (True, 1.5, "2"))
def test_denoising_filters_reject_noninteger_channel_axis(
    operation,
    channel_axis,
):
    data = np.zeros((2, 5, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="channel_axis must be an integer or None"):
        operation(data, channel_axis=channel_axis)


@pytest.mark.parametrize(
    "operation",
    (bilateral_filter, unsharp_mask_filter, non_local_means_filter),
)
def test_denoising_filters_require_two_spatial_axes_for_channel_data(operation):
    data = np.zeros((5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="requires at least two spatial dimensions"):
        operation(data, channel_axis=1)


def _edge_threshold_channel_cases(*, include_automatic=False):
    cases = [
        pytest.param(sobel_filter, {}, id="sobel"),
        pytest.param(laplace_filter, {"kernel_size": 3}, id="laplace"),
        pytest.param(
            canny_edges,
            {"sigma": 1.0, "low_quantile": 0.05, "high_quantile": 0.2},
            id="canny",
        ),
        pytest.param(
            hysteresis_threshold,
            {
                "low_threshold": 0.25,
                "high_threshold": 0.7,
                "spatial_mode": "2D YX",
            },
            id="hysteresis",
        ),
        pytest.param(otsu_threshold, {}, id="otsu"),
        pytest.param(triangle_threshold, {}, id="triangle"),
        pytest.param(li_threshold, {}, id="li"),
        pytest.param(yen_threshold, {}, id="yen"),
        pytest.param(isodata_threshold, {}, id="isodata"),
        pytest.param(minimum_threshold, {}, id="minimum"),
        pytest.param(binary_threshold, {"threshold": 0.5}, id="binary"),
        pytest.param(
            adaptive_mean_threshold,
            {"block_size": 3, "c": 0.0},
            id="adaptive-mean",
        ),
        pytest.param(
            adaptive_gaussian_threshold,
            {"block_size": 3, "c": 0.0},
            id="adaptive-gaussian",
        ),
        pytest.param(
            sauvola_threshold,
            {"window_size": 3, "k": 0.2},
            id="sauvola",
        ),
        pytest.param(
            niblack_threshold,
            {"window_size": 3, "k": 0.2},
            id="niblack",
        ),
    ]
    if include_automatic:
        cases.append(
            pytest.param(
                operations.automatic_threshold_value,
                {"operation_id": "otsu_threshold"},
                id="automatic-threshold-value",
            )
        )
    return cases


def _edge_threshold_test_image(*, x_size=12):
    image = np.zeros((15, x_size), dtype=np.float32)
    image[:, x_size // 2 :] = 1.0
    image[5:10] += 0.25
    return image


@pytest.mark.parametrize("x_size", (3, 4))
@pytest.mark.parametrize(
    ("operation", "kwargs"),
    _edge_threshold_channel_cases(),
)
def test_edge_threshold_operations_treat_rgb_sized_x_as_scalar_by_default(
    operation,
    kwargs,
    x_size,
):
    plane = _edge_threshold_test_image(x_size=x_size)
    data = np.stack((plane, plane * 0.8 + 0.05)).astype(np.float32)
    original = data.copy()

    result = operation(data, **kwargs)

    assert result.shape == data.shape
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    _edge_threshold_channel_cases(),
)
def test_edge_threshold_operations_reduce_only_declared_rgb_axes(
    operation,
    kwargs,
):
    base = _edge_threshold_test_image()
    yxc = np.stack((base, base * 0.8 + 0.05, base * 0.6 + 0.1), axis=-1)
    yxc = yxc.astype(np.float32)
    cyx = np.moveaxis(yxc, -1, 0).copy()
    original_yxc = yxc.copy()
    original_cyx = cyx.copy()
    weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    luma = np.sum(yxc * weights, axis=-1, dtype=np.float32)

    expected = operation(luma, **kwargs)
    yxc_result = operation(yxc, channel_axis=2, **kwargs)
    cyx_result = operation(cyx, channel_axis=0, **kwargs)

    assert yxc_result.shape == base.shape
    assert cyx_result.shape == base.shape
    if np.issubdtype(np.asarray(expected).dtype, np.floating):
        np.testing.assert_allclose(yxc_result, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cyx_result, expected, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(yxc_result, expected)
        np.testing.assert_array_equal(cyx_result, expected)
    np.testing.assert_array_equal(yxc, original_yxc)
    np.testing.assert_array_equal(cyx, original_cyx)


def test_declared_rgb_reduction_preserves_remaining_axis_order():
    base = _edge_threshold_test_image()
    first = np.stack((base, base * 0.8, base * 0.6), axis=-1)
    second = np.stack((base * 0.4, base * 0.2, base), axis=-1)
    zyxc = np.stack((first, second)).astype(np.float32)
    zcyx = np.moveaxis(zyxc, -1, 1)

    trailing = binary_threshold(zyxc, threshold=0.4, channel_axis=3)
    nontrailing = binary_threshold(zcyx, threshold=0.4, channel_axis=1)

    assert trailing.shape == (2, *base.shape)
    np.testing.assert_array_equal(nontrailing, trailing)


def test_declared_rgba_ignores_alpha_channel():
    base = _edge_threshold_test_image()
    rgb = np.stack((base, base * 0.8, base * 0.6), axis=-1).astype(np.float32)
    low_alpha = np.concatenate(
        (rgb, np.zeros((*base.shape, 1), dtype=np.float32)),
        axis=-1,
    )
    high_alpha = low_alpha.copy()
    high_alpha[..., 3] = 10_000.0

    low_result = binary_threshold(low_alpha, threshold=0.4, channel_axis=2)
    high_result = binary_threshold(high_alpha, threshold=0.4, channel_axis=2)

    np.testing.assert_array_equal(high_result, low_result)


def test_declared_boolean_rgb_is_supported_as_real_valued_luma():
    rgb = np.zeros((5, 6, 3), dtype=bool)
    rgb[..., 0] = True
    rgb[2:, :, 1] = True
    expected_luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]

    result = binary_threshold(rgb, threshold=0.5, channel_axis=2)

    np.testing.assert_array_equal(result, expected_luma > 0.5)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    _edge_threshold_channel_cases(include_automatic=True),
)
@pytest.mark.parametrize("channel_axis", (3, -4))
def test_edge_threshold_operations_reject_out_of_range_channel_axes(
    operation,
    kwargs,
    channel_axis,
):
    data = np.zeros((2, 5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="channel_axis .* is out of range"):
        operation(data, channel_axis=channel_axis, **kwargs)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    _edge_threshold_channel_cases(include_automatic=True),
)
@pytest.mark.parametrize("channel_axis", (True, 1.5, "2"))
def test_edge_threshold_operations_reject_noninteger_channel_axes(
    operation,
    kwargs,
    channel_axis,
):
    data = np.zeros((2, 5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="channel_axis must be an integer or None"):
        operation(data, channel_axis=channel_axis, **kwargs)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    _edge_threshold_channel_cases(include_automatic=True),
)
def test_edge_threshold_operations_require_two_spatial_axes_for_channels(
    operation,
    kwargs,
):
    data = np.zeros((5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="requires at least two spatial dimensions"):
        operation(data, channel_axis=1, **kwargs)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    _edge_threshold_channel_cases(include_automatic=True),
)
def test_edge_threshold_operations_reject_empty_input(operation, kwargs):
    data = np.empty((0, 5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="requires non-empty image data"):
        operation(data, **kwargs)


@pytest.mark.parametrize("channel_count", (1, 2, 5))
def test_luma_reduction_rejects_non_rgb_channel_counts(channel_count):
    data = np.zeros((5, 6, channel_count), dtype=np.float32)

    with pytest.raises(ValueError, match="exactly 3 RGB or 4 RGBA"):
        binary_threshold(data, channel_axis=2)


@pytest.mark.parametrize(
    "data",
    (
        np.zeros((5, 6, 3), dtype=np.complex64),
        np.full((5, 6, 3), "red", dtype="U3"),
        np.full((5, 6, 3), object(), dtype=object),
    ),
)
def test_luma_reduction_rejects_non_real_image_data(data):
    with pytest.raises(ValueError, match="requires real-valued boolean, integer"):
        binary_threshold(data, channel_axis=2)


@pytest.mark.parametrize("x_size", (3, 4))
def test_automatic_threshold_value_treats_rgb_sized_x_as_scalar_by_default(x_size):
    plane = _edge_threshold_test_image(x_size=x_size)
    data = np.stack((plane, plane * 0.8 + 0.05)).astype(np.float32)
    original = data.copy()

    expected = operations._otsu_value(data)
    result = operations.automatic_threshold_value(data, "otsu_threshold")

    assert result == expected
    np.testing.assert_array_equal(data, original)


def test_automatic_threshold_value_reduces_declared_yxc_and_cyx_to_luma():
    base = _edge_threshold_test_image()
    yxc = np.stack((base, base * 0.8 + 0.05, base * 0.6 + 0.1), axis=-1)
    yxc = yxc.astype(np.float32)
    cyx = np.moveaxis(yxc, -1, 0).copy()
    weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
    luma = np.sum(yxc * weights, axis=-1, dtype=np.float32)
    expected = operations.automatic_threshold_value(luma, "otsu_threshold")

    yxc_result = operations.automatic_threshold_value(
        yxc,
        "otsu_threshold",
        channel_axis=2,
    )
    cyx_result = operations.automatic_threshold_value(
        cyx,
        "otsu_threshold",
        channel_axis=0,
    )

    assert yxc_result == expected
    assert cyx_result == expected


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize(
    ("operation", "kwargs"),
    (
        pytest.param(
            bilateral_filter,
            {"diameter": 3, "sigma_color": 10.0, "sigma_space": 1.0},
            id="bilateral",
        ),
        pytest.param(
            unsharp_mask_filter,
            {"radius": 1.0, "amount": 0.5},
            id="unsharp",
        ),
        pytest.param(
            non_local_means_filter,
            {
                "patch_size": 3,
                "patch_distance": 2,
                "h": 0.05,
                "fast_mode": True,
            },
            id="non-local-means",
        ),
    ),
)
def test_denoising_filters_preserve_float_scale_across_planes(
    operation,
    kwargs,
    dtype,
):
    data = _different_range_filter_stack(dtype)
    original = data.copy()

    result = operation(data, **kwargs)

    assert result.dtype == dtype
    assert result[0].min() > 50.0
    assert result[1].min() > 1_050.0
    np.testing.assert_allclose(result[1] - result[0], 1_000.0, atol=0.2)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    (
        pytest.param(
            bilateral_filter,
            {"diameter": 3, "sigma_color": 5.0, "sigma_space": 1.0},
            id="bilateral",
        ),
        pytest.param(
            unsharp_mask_filter,
            {"radius": 1.0, "amount": 1.0},
            id="unsharp",
        ),
        pytest.param(
            non_local_means_filter,
            {"patch_size": 3, "patch_distance": 2, "h": 0.05},
            id="non-local-means",
        ),
    ),
)
@pytest.mark.parametrize("value", (-10.0, 100.0))
def test_denoising_filters_preserve_constant_float_outside_unit_range(
    operation,
    kwargs,
    value,
):
    data = np.full((2, 8, 8), value, dtype=np.float32)

    result = operation(data, **kwargs)

    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize(
    ("operation_id", "operation", "params"),
    (
        pytest.param(
            "bilateral_filter",
            bilateral_filter,
            {"diameter": 3, "sigma_color": 10.0, "sigma_space": 1.0},
            id="bilateral",
        ),
        pytest.param(
            "unsharp_mask",
            unsharp_mask_filter,
            {"radius": 1.0, "amount": 0.5},
            id="unsharp",
        ),
        pytest.param(
            "non_local_means_filter",
            non_local_means_filter,
            {
                "patch_size": 3,
                "patch_distance": 2,
                "h": 0.05,
                "fast_mode": True,
            },
            id="non-local-means",
        ),
    ),
)
def test_pipeline_denoising_filters_preserve_float_scale(
    operation_id,
    operation,
    params,
):
    data = _different_range_filter_stack()
    original = data.copy()
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    pipeline.connect("input", node.id)
    for name, value in params.items():
        pipeline.set_param(node.id, name, value)

    result = pipeline.run(data)[node.id]

    np.testing.assert_allclose(result, operation(data, **params), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(result[1] - result[0], 1_000.0, atol=0.2)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    ("operation", "kwargs", "message"),
    (
        (bilateral_filter, {"diameter": 4}, "diameter must be odd"),
        (
            bilateral_filter,
            {"sigma_color": np.nan},
            "sigma_color must be a finite number",
        ),
        (
            bilateral_filter,
            {"sigma_color": 0.0},
            "sigma_color must be greater than 0",
        ),
        (
            bilateral_filter,
            {"sigma_space": np.inf},
            "sigma_space must be a finite number",
        ),
        (
            bilateral_filter,
            {"sigma_space": 0.0},
            "sigma_space must be greater than 0",
        ),
        (unsharp_mask_filter, {"radius": np.nan}, "radius must be a finite number"),
        (unsharp_mask_filter, {"radius": -1.0}, "radius must be at least 0"),
        (unsharp_mask_filter, {"amount": np.inf}, "amount must be a finite number"),
        (unsharp_mask_filter, {"amount": -1.0}, "amount must be at least 0"),
        (non_local_means_filter, {"patch_size": 4}, "patch_size must be odd"),
        (
            non_local_means_filter,
            {"patch_distance": 0},
            "patch_distance must be at least 1",
        ),
        (non_local_means_filter, {"h": np.nan}, "h must be a finite number"),
        (non_local_means_filter, {"h": -0.1}, "h must be at least 0"),
    ),
)
def test_denoising_filters_reject_invalid_scale_parameters(
    operation,
    kwargs,
    message,
):
    data = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        operation(data, **kwargs)


@pytest.mark.parametrize(
    "operation",
    (bilateral_filter, unsharp_mask_filter, non_local_means_filter),
)
def test_denoising_filters_reject_nonfinite_image_intensities(operation):
    data = np.zeros((8, 8), dtype=np.float32)
    data[3, 4] = np.nan

    with pytest.raises(ValueError, match="requires finite image intensities"):
        operation(data)


@pytest.mark.parametrize(
    ("low_sigma", "high_sigma", "message"),
    (
        (np.nan, 3.0, "finite numbers"),
        (1.0, np.inf, "finite numbers"),
        (-1.0, 3.0, "non-negative"),
        (1.0, -3.0, "non-negative"),
        (1.0, 1.0, "greater than low sigma"),
        (3.0, 1.0, "greater than low sigma"),
    ),
)
def test_difference_of_gaussians_rejects_invalid_sigma_pairs(
    low_sigma,
    high_sigma,
    message,
):
    data = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        difference_of_gaussians_filter(
            data,
            low_sigma=low_sigma,
            high_sigma=high_sigma,
        )


def test_rolling_ball_background_and_subtraction_preserve_dtype():
    yy, xx = np.mgrid[:31, :31]
    smooth_background = 30 + yy.astype(np.float32) * 1.2 + xx.astype(np.float32) * 0.5
    image = smooth_background.copy()
    image[15, 15] = 220
    image = np.clip(image, 0, 255).astype(np.uint8)

    background = rolling_ball_background(image, radius=7, disable_smoothing=True)
    corrected = subtract_background(image, radius=7, disable_smoothing=True)

    assert background.shape == image.shape
    assert corrected.shape == image.shape
    assert background.dtype == image.dtype
    assert corrected.dtype == image.dtype
    assert corrected[15, 15] > corrected[0, 0] + 100
    assert corrected[0, 0] < 5


def test_subtract_background_light_background_makes_dark_objects_positive():
    image = np.full((25, 25), 200, dtype=np.uint8)
    image[12, 12] = 20

    corrected = subtract_background(
        image,
        radius=5,
        light_background=True,
        disable_smoothing=True,
    )

    assert corrected.dtype == image.dtype
    assert corrected[12, 12] > 150
    assert corrected[0, 0] < 5


def test_rolling_ball_background_defaults_to_slice_wise_processing():
    data = np.zeros((3, 21, 21), dtype=np.float32)
    data[1, 10, 10] = 10.0

    corrected = subtract_background(
        data,
        radius=5,
        disable_smoothing=True,
        spatial_mode="2D YX",
    )

    assert corrected.shape == data.shape
    assert corrected.dtype == data.dtype
    assert np.allclose(corrected[0], 0)
    assert np.allclose(corrected[2], 0)
    assert corrected[1, 10, 10] > 5


def test_rolling_ball_background_processes_rgb_stack_channels_independently():
    data = np.zeros((2, 21, 21, 3), dtype=np.uint8)
    data[0, 10, 10, 0] = 200
    data[1, 10, 10, 1] = 180

    corrected = subtract_background(
        data,
        radius=5,
        disable_smoothing=True,
        spatial_mode="2D YX",
    )

    assert corrected.shape == data.shape
    assert corrected.dtype == data.dtype
    assert corrected[0, 10, 10, 0] > 100
    assert corrected[0, :, :, 1].max() == 0
    assert corrected[1, 10, 10, 1] > 100
    assert corrected[1, :, :, 0].max() == 0


def test_rolling_ball_background_reports_progress_and_cancels():
    data = np.zeros((3, 15, 15), dtype=np.uint8)
    data[:, 7, 7] = 200
    updates = []
    state = {"cancel": False}

    def report(update):
        updates.append((update.current, update.total, update.message))
        if update.current >= 1:
            state["cancel"] = True

    progress = ProgressContext(
        cancelled=lambda: state["cancel"],
        reporter=report,
    )

    try:
        subtract_background(
            data,
            radius=3,
            disable_smoothing=True,
            spatial_mode="2D YX",
            progress=progress,
        )
    except OperationCancelled:
        pass
    else:
        raise AssertionError("Expected cooperative cancellation.")

    assert updates[0] == (0, 3, "Rolling-ball background")
    assert updates[1] == (1, 3, "Rolling-ball background")


def test_subtract_background_pipeline_records_metadata_history():
    data = np.zeros((2, 16, 16), dtype=np.float32)
    data[:, 6:10, 6:10] = 20.0
    pipeline = PrototypePipeline()
    node = pipeline.add_node("subtract_background")
    pipeline.set_param(node.id, "radius", 5)
    pipeline.connect("input", node.id)

    outputs = pipeline.run(data, input_metadata={"axes": "ZYX"})

    assert outputs[node.id].shape == data.shape
    assert pipeline.output_states[node.id].history[-1] == (
        "Subtract Background: radius 5 px, 2D YX"
    )


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


@pytest.mark.parametrize(
    ("low_quantile", "high_quantile", "message"),
    (
        (np.nan, 0.2, "finite numbers"),
        (0.1, np.inf, "finite numbers"),
        (-0.1, 0.2, "at least 0"),
        (0.1, 1.1, "at most 1"),
        (0.8, 0.2, "must not exceed"),
    ),
)
def test_canny_rejects_invalid_quantile_pairs(
    low_quantile,
    high_quantile,
    message,
):
    data = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        canny_edges(
            data,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
        )


@pytest.mark.parametrize(
    ("low_threshold", "high_threshold", "message"),
    (
        (np.nan, 0.8, "finite numbers"),
        (0.4, np.inf, "finite numbers"),
        (0.8, 0.4, "must not exceed"),
    ),
)
def test_hysteresis_rejects_invalid_threshold_pairs(
    low_threshold,
    high_threshold,
    message,
):
    data = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        hysteresis_threshold(
            data,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )


def test_hysteresis_accepts_finite_native_intensity_thresholds():
    data = np.array([[0, 128, 255]], dtype=np.uint8)

    result = hysteresis_threshold(
        data,
        low_threshold=64,
        high_threshold=192,
        spatial_mode="2D YX",
    )

    np.testing.assert_array_equal(result, np.array([[False, True, True]]))


@pytest.mark.parametrize(
    ("operation_id", "params", "message"),
    (
        (
            "canny_edges",
            {"low_quantile": 0.8, "high_quantile": 0.2},
            "Canny low threshold .* must not exceed",
        ),
        (
            "hysteresis_threshold",
            {
                "low_threshold": 0.8,
                "high_threshold": 0.4,
                "spatial_mode": "2D YX",
            },
            "Hysteresis low threshold .* must not exceed",
        ),
        (
            "difference_of_gaussians",
            {"low_sigma": 3.0, "high_sigma": 1.0},
            "high sigma must be greater than low sigma",
        ),
    ),
)
def test_pipeline_surfaces_invalid_ordered_operation_parameters(
    operation_id,
    params,
    message,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    pipeline.connect("input", node.id)
    for name, value in params.items():
        pipeline.set_param(node.id, name, value)

    with pytest.raises(ValueError, match=message):
        pipeline.run(np.zeros((8, 8), dtype=np.float32))


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


def test_orthogonal_projection_keeps_xy_native_when_z_spacing_is_finer():
    data = np.zeros((75, 96, 128), dtype=np.uint8)

    projected = orthogonal_projection(
        data,
        method="Maximum",
        axis_names=("z", "y", "x"),
        axis_types=("space", "space", "space"),
        axis_scales=(1.0 / 6.25, 1.0, 1.0),
        axis_units=("micrometer", "micrometer", "micrometer"),
    )

    assert projected.shape == (108, 140)


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
    pipeline.run(data, input_metadata={"axes": "ZYX"})

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
        1.5,
        0.5,
    ]


def test_rescale_axes_accepts_explicit_output_sizes():
    data = np.zeros((4, 10, 20), dtype=np.uint16)
    state = image_state_from_array(data)

    scaled = rescale_axes(
        data,
        resize_mode="Output size",
        x_size=30,
        y_size=15,
        z_size=7,
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
            "resize_mode": "Output size",
            "x_size": 30,
            "y_size": 15,
            "z_size": 7,
            "interpolation": "Nearest neighbor",
        },
    )

    assert scaled.shape == (7, 15, 30)
    assert scaled.dtype == data.dtype
    assert [axis.scale for axis in scaled_state.axes] == [
        4 / 7,
        10 / 15,
        20 / 30,
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


def test_rescaled_z_orthogonal_projection_does_not_upscale_xy_panel():
    data = np.zeros((12, 96, 128), dtype=np.uint8)
    pipeline = PrototypePipeline()
    rescale = pipeline.add_node("rescale_axes")
    projection = pipeline.add_node("orthogonal_projection")
    pipeline.connect("input", rescale.id)
    pipeline.connect(rescale.id, projection.id)
    pipeline.set_param(rescale.id, "x_scale", 1.0)
    pipeline.set_param(rescale.id, "y_scale", 1.0)
    pipeline.set_param(rescale.id, "z_scale", 6.25)
    pipeline.set_param(rescale.id, "lock_xy", True)

    pipeline.run(data, input_metadata={"axes": "ZYX"})
    projected_state = pipeline.output_states[projection.id]

    assert pipeline.outputs[rescale.id].shape == (75, 96, 128)
    assert pipeline.outputs[projection.id].shape == (108, 140)
    assert projected_state is not None
    assert [axis.scale for axis in projected_state.axes] == [1.0, 1.0]


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


def test_extract_channel_uses_semantic_channel_axis():
    data = np.zeros((2, 3, 5, 6), dtype=np.uint16)
    data[:, 2] = 42

    channel = extract_channel(
        data,
        channel=2,
        axis_names=("z", "c", "y", "x"),
        axis_types=("space", "channel", "space", "space"),
    )

    assert channel.shape == (2, 5, 6)
    assert np.all(channel == 42)


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


@pytest.mark.parametrize(
    ("weights", "message"),
    (
        ("1", "requires exactly 2 weight"),
        ("1,2,3", "requires exactly 2 weight"),
        ("1,bad", "weight 2 must be a finite number"),
        ("1,nan", "weight 2 must be a finite number"),
        ("1,inf", "weight 2 must be a finite number"),
        ("1,,2", "weight 2 is empty"),
    ),
)
def test_calculate_weighted_image_rejects_invalid_weights(weights, message):
    first = np.ones((2, 3), dtype=np.uint16)
    second = np.ones((2, 3), dtype=np.uint16)

    with pytest.raises(ValueError, match=message):
        calculate_weighted_image(
            [first, second],
            input_count=2,
            weights=weights,
        )


def test_pipeline_surfaces_invalid_image_calculator_weights():
    data = np.ones((2, 3), dtype=np.uint16)
    pipeline = PrototypePipeline()
    calculator = pipeline.add_node("calculate_weighted_image")
    pipeline.connect("input", calculator.id, target_port=0)
    pipeline.connect("input", calculator.id, target_port=1)
    pipeline.set_param(calculator.id, "weights", "1,bad")

    with pytest.raises(ValueError, match="weight 2 must be a finite number"):
        pipeline.run(data)


def test_intensity_rescale_normalize_and_clip():
    data = np.arange(6, dtype=np.uint16).reshape(2, 3)

    rescaled = rescale_intensity(data, out_min=0, out_max=255)
    normalized = normalize_image(data, method="min-max")
    clipped = clip_intensity(
        data,
        cutoff_mode="Values",
        minimum=2,
        maximum=4,
    )

    assert rescaled.dtype == data.dtype
    assert rescaled.min() == 0
    assert rescaled.max() == 255
    value_rescaled = rescale_intensity(
        data,
        cutoff_mode="Values",
        in_low_value=2,
        in_high_value=4,
        out_min=0,
        out_max=10,
    )
    np.testing.assert_array_equal(
        value_rescaled,
        np.array([[0, 0, 0], [5, 10, 10]], dtype=np.uint16),
    )
    percentile_rescaled = rescale_intensity(
        data,
        cutoff_mode="Percentiles",
        in_low_percentile=20,
        in_high_percentile=80,
        in_low_value=0,
        in_high_value=1,
        out_min=0,
        out_max=10,
    )
    np.testing.assert_array_equal(
        percentile_rescaled,
        np.array([[0, 0, 3], [7, 10, 10]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        clip_intensity(
            data,
            cutoff_mode="Data range",
            minimum=2,
            maximum=4,
        ),
        data,
    )
    assert rescale_intensity(data.astype(np.float32)).dtype == np.float32
    assert normalized.dtype == np.float32
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalize_image(data.astype(np.float64)).dtype == np.float64
    assert clipped.dtype == data.dtype
    np.testing.assert_array_equal(clipped, np.array([[2, 2, 2], [3, 4, 4]]))


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (np.int64, np.iinfo(np.int64).min),
        (np.int64, 2**60),
        (np.int64, np.iinfo(np.int64).max - 3),
        (np.uint64, 2**63 + 123),
        (np.uint64, np.iinfo(np.uint64).max - 3),
    ],
)
def test_wide_integer_rescale_and_clip_preserve_adjacent_native_levels(
    dtype,
    base,
):
    data = np.asarray([base + offset for offset in range(4)], dtype=dtype)

    percentile_result = rescale_intensity(
        data,
        cutoff_mode="Percentiles",
        in_low_percentile=0,
        in_high_percentile=100,
        out_min=0,
        out_max=3,
    )
    value_result = rescale_intensity(
        data,
        cutoff_mode="Values",
        in_low_value=base + 1,
        in_high_value=base + 2,
        out_min=0,
        out_max=2,
    )
    clipped = clip_intensity(
        data,
        cutoff_mode="Values",
        minimum=base + 1,
        maximum=base + 2,
    )

    np.testing.assert_array_equal(
        percentile_result,
        np.asarray([0, 1, 2, 3], dtype=dtype),
    )
    np.testing.assert_array_equal(
        value_result,
        np.asarray([0, 0, 2, 2], dtype=dtype),
    )
    np.testing.assert_array_equal(
        clipped,
        np.asarray([base + 1, base + 1, base + 2, base + 2], dtype=dtype),
    )


def test_wide_integer_percentiles_keep_fractional_cutoffs_exact():
    base = 2**60
    data = np.asarray([base + offset for offset in range(4)], dtype=np.int64)

    low, high = operations.exact_integer_percentiles(data, (25, 75))
    result = rescale_intensity(
        data,
        cutoff_mode="Percentiles",
        in_low_percentile=25,
        in_high_percentile=75,
        out_min=0,
        out_max=6,
    )

    assert low == Fraction(4 * base + 3, 4)
    assert high == Fraction(4 * base + 9, 4)
    np.testing.assert_array_equal(result, np.array([0, 1, 5, 6]))


def test_integer_rescale_rejects_unrepresentable_input_or_output_spans():
    data = np.asarray([0, 2**53 + 1], dtype=np.int64)

    with pytest.raises(ValueError, match=r"cutoff span exceeds 2\^53"):
        rescale_intensity(data, out_min=0, out_max=1)
    with pytest.raises(ValueError, match=r"output span exceeds 2\^53"):
        rescale_intensity(
            np.asarray([0, 1], dtype=np.int64),
            out_min=0,
            out_max=2**53 + 1,
        )


def test_integer_clip_rejects_fractional_or_rounded_wide_bounds():
    data = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError, match="bounds must be whole numbers"):
        clip_intensity(
            data,
            cutoff_mode="Values",
            minimum=0.5,
            maximum=2,
        )

    base = 2**60
    wide = np.asarray([base, base + 1], dtype=np.int64)
    with pytest.raises(
        ValueError,
        match=r"saved as a floating-point value above 2\^53",
    ):
        clip_intensity(
            wide,
            cutoff_mode="Values",
            minimum=float(base),
            maximum=float(base + 1),
        )


@pytest.mark.parametrize(
    ("operation", "kwargs", "message"),
    [
        (
            rescale_intensity,
            {"cutoff_mode": "estimated"},
            "must be 'Percentiles' or 'Values'",
        ),
        (
            rescale_intensity,
            {
                "cutoff_mode": "Values",
                "in_low_value": 0.0,
                "in_high_value": np.nan,
            },
            "High input value must be a finite number",
        ),
        (
            rescale_intensity,
            {
                "cutoff_mode": "Percentiles",
                "in_low_percentile": 90.0,
                "in_high_percentile": 10.0,
            },
            "low percentile must not exceed",
        ),
        (
            clip_intensity,
            {"cutoff_mode": "automatic"},
            "must be 'Data range' or 'Values'",
        ),
        (
            clip_intensity,
            {"cutoff_mode": "Values", "minimum": 0.0, "maximum": np.inf},
            "Clip maximum must be a finite number",
        ),
    ],
)
def test_intensity_cutoff_modes_reject_invalid_or_ambiguous_values(
    operation,
    kwargs,
    message,
):
    data = np.arange(6, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        operation(data, **kwargs)


def test_boolean_intensity_passthrough_still_validates_active_cutoffs():
    mask = np.array([[False, True], [True, False]])

    with pytest.raises(ValueError, match="low input value must not exceed"):
        rescale_intensity(
            mask,
            cutoff_mode="Values",
            in_low_value=2,
            in_high_value=1,
        )
    with pytest.raises(ValueError, match="Clip minimum must not exceed"):
        clip_intensity(
            mask,
            cutoff_mode="Values",
            minimum=2,
            maximum=1,
        )

    np.testing.assert_array_equal(rescale_intensity(mask), mask)
    np.testing.assert_array_equal(clip_intensity(mask), mask)


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

    channels = split_channels(
        data,
        preview_channel=2,
        axis_names=("c", "y", "x"),
        axis_types=("channel", "space", "space"),
    )

    assert len(channels) == 3
    assert int(channels[0].max()) == 10
    assert int(channels[1].max()) == 20
    assert int(channels[2].max()) == 30


def test_split_channels_returns_true_channel_count():
    data = np.zeros((2, 8, 8), dtype=np.uint8)
    data[0] = 10
    data[1] = 20

    channels = split_channels(
        data,
        axis_names=("c", "y", "x"),
        axis_types=("channel", "space", "space"),
    )

    assert len(channels) == 2
    assert int(channels[0].max()) == 10
    assert int(channels[1].max()) == 20


def test_split_channels_requires_channel_axis_for_grayscale_input():
    data = np.full((5, 5), 7, dtype=np.uint8)

    with pytest.raises(ValueError, match="needs a channel axis"):
        split_channels(data)


def test_split_axis_can_split_grayscale_axis():
    data = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5)

    slices = split_axis(data, axis="axis:0")

    assert len(slices) == 3
    np.testing.assert_array_equal(slices[0], data[0])
    np.testing.assert_array_equal(slices[1], data[1])
    np.testing.assert_array_equal(slices[2], data[2])


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


def test_global_threshold_scope_rejects_unknown_mode():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)

    with pytest.raises(ValueError, match="must be 'Stack histogram' or 'Slice"):
        otsu_threshold(data, threshold_scope="banana")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (np.full(17, 3.25, dtype=np.float32), 3.25),
        (np.full(17, 42, dtype=np.uint16), 42.0),
    ],
)
def test_otsu_value_handles_constant_data(data, expected):
    assert operations._otsu_value(data) == expected


@pytest.mark.parametrize(
    "threshold_func",
    [
        otsu_threshold,
        triangle_threshold,
        li_threshold,
        yen_threshold,
        isodata_threshold,
        minimum_threshold,
    ],
)
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            np.array([], dtype=np.float32),
            "requires non-empty image data",
        ),
        (
            np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
            "at least one finite input value",
        ),
    ],
)
def test_automatic_thresholds_reject_inputs_without_finite_values(
    threshold_func,
    data,
    message,
):
    with pytest.raises(ValueError, match=message):
        threshold_func(data)


@pytest.mark.parametrize(
    ("value_func", "reference_func"),
    [
        (operations._otsu_value, operations.filters.threshold_otsu),
        (operations._triangle_value, operations.filters.threshold_triangle),
        (operations._yen_value, operations.filters.threshold_yen),
        (operations._isodata_value, operations.filters.threshold_isodata),
        (operations._minimum_value, operations.filters.threshold_minimum),
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        np.concatenate(
            [
                np.linspace(-4.0, -1.0, 613, dtype=np.float32),
                np.linspace(2.0, 8.0, 1_009, dtype=np.float32),
                np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
            ]
        ),
        np.concatenate(
            [
                np.full(1_000, 10, dtype=np.uint16),
                np.full(1_000, 200, dtype=np.uint16),
            ]
        ).reshape(40, 50)[:, ::2],
        np.concatenate(
            [
                np.full(1_000, 10, dtype=np.uint8),
                np.full(1_000, 200, dtype=np.uint8),
            ]
        ),
        np.concatenate(
            [
                np.full(1_000, -1_000, dtype=np.int16),
                np.full(1_000, 1_000, dtype=np.int16),
            ]
        ),
    ],
)
def test_histogram_thresholds_match_full_array_implementations(
    value_func,
    reference_func,
    data,
):
    if np.issubdtype(data.dtype, np.integer):
        values = data
    else:
        # Keep the input floating dtype: scikit-image deliberately uses
        # dtype-native histogram edges and centres, which is part of the
        # threshold algorithm's numerical contract.
        values = data[np.isfinite(data)]
    try:
        expected = float(reference_func(values))
    except Exception as exc:
        with pytest.raises(type(exc)):
            value_func(data)
    else:
        assert value_func(data) == expected


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_integer_histogram_uses_every_native_intensity_level(dtype):
    data = np.array([3, 3, 5, 8, 8, 8], dtype=dtype)

    summary = operations._finite_histogram(data, histogram_bins=2)
    expected_counts, expected_centers = exposure.histogram(data)

    np.testing.assert_array_equal(summary.counts, expected_counts)
    np.testing.assert_array_equal(summary.centers, expected_centers)
    assert summary.stats.count == data.size


def test_boolean_histogram_has_exact_false_and_true_levels():
    data = np.array([False, True, True, False, True], dtype=bool)

    summary = operations._finite_histogram(data)

    np.testing.assert_array_equal(summary.counts, [2, 3])
    np.testing.assert_array_equal(summary.centers, [0, 1])
    assert summary.stats.count == data.size


def test_float_histogram_counts_every_finite_value_with_requested_bins():
    data = np.array([0.0, 0.25, 0.5, 1.0, np.nan, np.inf, -np.inf])

    summary = operations._finite_histogram(data, histogram_bins=17)

    assert summary.counts is not None
    assert summary.counts.size == 17
    assert int(summary.counts.sum()) == 4
    assert summary.stats.count == 4


@pytest.mark.parametrize(
    ("value_func", "reference_func"),
    [
        (operations._otsu_value, operations.filters.threshold_otsu),
        (operations._triangle_value, operations.filters.threshold_triangle),
        (operations._yen_value, operations.filters.threshold_yen),
        (operations._isodata_value, operations.filters.threshold_isodata),
        (operations._minimum_value, operations.filters.threshold_minimum),
    ],
)
def test_float_thresholds_honor_requested_histogram_bins(value_func, reference_func):
    data = np.concatenate(
        [
            np.linspace(-4.0, -1.0, 613, dtype=np.float32),
            np.linspace(2.0, 8.0, 1_009, dtype=np.float32),
        ]
    ).astype(np.float64)
    try:
        expected = float(reference_func(data, nbins=17))
    except Exception as exc:
        with pytest.raises(type(exc)):
            value_func(data, histogram_bins=17)
    else:
        assert value_func(data, histogram_bins=17) == expected


def test_wide_integer_histogram_requires_explicit_dtype_conversion():
    data = np.array([0, 100_000], dtype=np.int32)

    with pytest.raises(ValueError, match="100,001 levels"):
        otsu_threshold(data)


@pytest.mark.parametrize(
    ("dtype", "base"),
    [
        (np.int64, np.iinfo(np.int64).min),
        (np.int64, 2**60),
        (np.int64, np.iinfo(np.int64).max - 220),
        (np.uint64, 2**63 + 123),
        (np.uint64, np.iinfo(np.uint64).max - 220),
    ],
)
@pytest.mark.parametrize(
    ("operation_id", "threshold_func"),
    [
        ("otsu_threshold", otsu_threshold),
        ("triangle_threshold", triangle_threshold),
        ("li_threshold", li_threshold),
        ("yen_threshold", yen_threshold),
        ("isodata_threshold", isodata_threshold),
        ("minimum_threshold", minimum_threshold),
    ],
)
def test_large_magnitude_integer_thresholds_preserve_native_level_distinctions(
    dtype,
    base,
    operation_id,
    threshold_func,
):
    rng = np.random.default_rng(2026)
    relative = np.rint(
        np.concatenate(
            [
                rng.normal(30, 6, 3_000),
                rng.normal(180, 9, 3_000),
            ]
        )
    )
    relative = np.clip(relative, 0, 220).astype(np.uint16)
    shifted = np.fromiter(
        (int(base) + int(value) for value in relative),
        dtype=dtype,
        count=relative.size,
    )

    expected = threshold_func(relative)
    result = threshold_func(shifted)
    threshold = operations.automatic_threshold_value(shifted, operation_id)

    np.testing.assert_array_equal(result, expected)
    assert isinstance(threshold, int)
    assert int(shifted.min()) <= threshold <= int(shifted.max())


@pytest.mark.parametrize(
    "base",
    [
        2**51,
        2**52,
        2**53,
        -(2**52),
        -(2**53),
    ],
)
def test_li_integer_cutoff_survives_float_boundary_rounding(base):
    rng = np.random.default_rng(2026)
    relative = np.rint(
        np.concatenate(
            [
                rng.normal(30, 6, 3_000),
                rng.normal(180, 9, 3_000),
            ]
        )
    )
    relative = np.clip(relative, 0, 220).astype(np.uint16)
    relative_threshold = operations.automatic_threshold_value(
        relative,
        "li_threshold",
    )
    assert isinstance(relative_threshold, float)
    shifted = np.fromiter(
        (int(base) + int(value) for value in relative),
        dtype=np.int64,
        count=relative.size,
    )

    threshold = operations.automatic_threshold_value(shifted, "li_threshold")

    assert threshold == int(base) + int(np.floor(relative_threshold))
    assert isinstance(threshold, int)
    np.testing.assert_array_equal(li_threshold(shifted), li_threshold(relative))


def test_li_rejects_integer_span_that_float64_cannot_represent_exactly():
    data = np.array([0, 2**53 + 1], dtype=np.uint64)

    with pytest.raises(ValueError, match="exceeds the exact float64 range"):
        li_threshold(data)


@pytest.mark.parametrize(
    "threshold_func",
    [
        otsu_threshold,
        triangle_threshold,
        li_threshold,
        yen_threshold,
        isodata_threshold,
    ],
)
def test_global_thresholds_mark_all_nonfinite_pixels_as_background(threshold_func):
    data = np.array([0.0, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)

    mask = threshold_func(data)

    assert not mask[2:].any()


def test_minimum_threshold_marks_nonfinite_pixels_as_background():
    rng = np.random.default_rng(11)
    data = np.concatenate(
        [
            rng.normal(0.0, 0.1, 2_000),
            rng.normal(1.0, 0.1, 2_000),
            np.array([np.nan, np.inf, -np.inf]),
        ]
    ).astype(np.float32)

    mask = minimum_threshold(data)

    assert not mask[-3:].any()


@pytest.mark.parametrize(
    "threshold_func",
    [
        otsu_threshold,
        triangle_threshold,
        li_threshold,
        yen_threshold,
        isodata_threshold,
        minimum_threshold,
    ],
)
def test_automatic_thresholds_preserve_existing_boolean_masks(threshold_func):
    mask = np.array([[False, True, False], [True, True, False]])

    result = threshold_func(mask)

    np.testing.assert_array_equal(result, mask)
    assert result is not mask


def test_automatic_boolean_threshold_marker_represents_passthrough_boundary():
    mask = np.array([False, True, True])

    assert operations.automatic_threshold_value(mask, "triangle_threshold") == 0.5


def test_minimum_threshold_does_not_silently_fall_back_to_image_mean():
    data = np.arange(16, dtype=np.float32)

    with pytest.raises(RuntimeError, match="Unable to find two maxima"):
        operations._minimum_value(data)


def test_minimum_threshold_smoothing_is_cooperatively_cancellable():
    data = np.random.default_rng(7).integers(
        0,
        65_536,
        size=10_000,
        dtype=np.uint16,
    )
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 3

    progress = ProgressContext(cancelled=cancelled)

    with pytest.raises(OperationCancelled):
        operations._minimum_value(
            data,
            max_iterations=1_000,
            progress=progress,
        )


@pytest.mark.parametrize(
    "value_func",
    [
        operations._otsu_value,
        operations._triangle_value,
        operations._yen_value,
        operations._isodata_value,
        operations._minimum_value,
    ],
)
def test_histogram_thresholds_read_large_arrays_in_bounded_chunks(
    monkeypatch,
    value_func,
):
    data = np.linspace(-5.0, 10.0, 101, dtype=np.float32)
    data[::19] = np.nan
    expected = value_func(data)
    inspected_sizes = []
    original_isfinite = operations.np.isfinite

    def recording_isfinite(values):
        inspected_sizes.append(np.asarray(values).size)
        return original_isfinite(values)

    with monkeypatch.context() as context:
        context.setattr(operations, "_THRESHOLD_CHUNK_SIZE", 11)
        context.setattr(operations.np, "isfinite", recording_isfinite)
        threshold = value_func(data)

    assert threshold == expected
    assert len(inspected_sizes) > 2
    assert max(inspected_sizes) <= 11


def test_native_integer_histogram_is_built_in_bounded_chunks(monkeypatch):
    data = np.arange(101, dtype=np.uint16)
    inspected_sizes = []
    original_bincount = operations.np.bincount

    def recording_bincount(values, *args, **kwargs):
        inspected_sizes.append(np.asarray(values).size)
        return original_bincount(values, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(operations, "_THRESHOLD_CHUNK_SIZE", 11)
        context.setattr(operations.np, "bincount", recording_bincount)
        summary = operations._finite_histogram(data)

    assert summary.counts is not None
    assert int(summary.counts.sum()) == data.size
    assert inspected_sizes
    assert max(inspected_sizes) <= 11


def test_li_threshold_keeps_its_raw_iterative_path(monkeypatch):
    data = np.concatenate(
        [
            np.linspace(0.0, 1.0, 101, dtype=np.float32),
            np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
        ]
    )
    finite = data[np.isfinite(data)].astype(np.float64, copy=False)
    expected = operations.filters.threshold_li(finite)
    monkeypatch.setattr(
        operations,
        "_finite_histogram",
        lambda _data: pytest.fail("Li must not use the binned histogram path"),
    )

    threshold = operations._li_value(data)

    assert threshold == expected


def test_scalar_grayscale_and_boolean_masks_do_not_make_input_copies():
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    mask = image > 4

    grayscale = operations._to_grayscale(image)
    converted_mask = operations._to_bool_mask(mask)

    assert grayscale is image
    assert grayscale.dtype == np.uint16
    assert converted_mask is mask


def test_rgb_grayscale_conversion_does_not_downcast_float64_data():
    rgb = np.array(
        [[[10_000_000_000.25, 10_000_000_001.5, 10_000_000_002.75]]],
        dtype=np.float64,
    )

    grayscale = operations._to_grayscale(rgb)
    expected = np.sum(
        rgb[..., :3] * np.array([0.299, 0.587, 0.114], dtype=np.float64),
        axis=-1,
    )

    assert grayscale.dtype == np.float64
    np.testing.assert_array_equal(grayscale, expected)


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
