from __future__ import annotations

import numpy as np

from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline


def _starter_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    return pipeline


def test_export_produces_valid_python():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    # Must compile as a module without syntax errors.
    compile(code, "<exported>", "exec")

    assert "def run_pipeline(" in code
    assert "def batch_process(" in code
    assert "gaussian_blur(" in code
    assert "otsu_threshold(" in code
    assert "sigma=1.2" in code
    assert "from napari_vipp.core.io import read_image, write_image" in code
    assert "skimage" not in code


def test_exported_run_pipeline_executes():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    run_pipeline = namespace["run_pipeline"]
    image = np.random.rand(4, 8, 8).astype(np.float32)
    results = run_pipeline(image)

    assert "threshold" in results
    assert results["threshold"].shape == image.shape
    assert results["threshold"].dtype == bool
    assert namespace["OUTPUT_NODES"] == ("threshold",)


def test_exported_rescale_axes_preserves_output_size_mode():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_axes")
    pipeline.connect("input", node.id)
    pipeline.set_param(node.id, "resize_mode", "Output size")
    pipeline.set_param(node.id, "x_size", 12)
    pipeline.set_param(node.id, "y_size", 8)
    pipeline.set_param(node.id, "z_size", 5)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](np.zeros((3, 4, 6), dtype=np.uint8))

    assert "resize_mode='Output size'" in code
    assert "x_size=12" in code
    assert results[node.id].shape == (5, 8, 12)


def test_export_handles_multi_input_nodes():
    pipeline = _starter_pipeline()
    add = pipeline.add_node("add_images")
    pipeline.connect("gaussian", add.id, target_port=1)
    pipeline.connect("input", add.id, target_port=0)

    code = export_pipeline_to_python(pipeline)
    compile(code, "<exported>", "exec")
    # Multi-input call should pass a list of upstream variables.
    assert "add_images([" in code
    assert "add_images([v_input, v_gaussian]" in code


def test_export_includes_subtract_background_node():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("subtract_background")
    pipeline.set_param(node.id, "radius", 7)
    pipeline.connect("input", node.id)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.zeros((21, 21), dtype=np.uint8)
    image[10, 10] = 200
    results = namespace["run_pipeline"](image)

    assert "subtract_background(" in code
    assert "radius=7" in code
    assert results[node.id].shape == image.shape
    assert results[node.id].dtype == image.dtype


def test_exported_touching_object_separation_pipeline_executes():
    pipeline = PrototypePipeline()
    distance = pipeline.add_node("euclidean_distance_transform")
    markers = pipeline.add_node("h_maxima_markers")
    watershed = pipeline.add_node("marker_controlled_watershed")
    pipeline.connect("input", distance.id)
    pipeline.connect(distance.id, markers.id)
    pipeline.connect(distance.id, watershed.id, target_port=0)
    pipeline.connect(markers.id, watershed.id, target_port=1)
    pipeline.connect("input", watershed.id, target_port=2)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    yy, xx = np.mgrid[:48, :64]
    image = (((yy - 24) ** 2 + (xx - 22) ** 2 <= 13**2) | (
        (yy - 24) ** 2 + (xx - 42) ** 2 <= 13**2
    ))
    results = namespace["run_pipeline"](image)

    assert "euclidean_distance_transform(" in code
    assert "h_maxima_markers(" in code
    assert "marker_controlled_watershed(" in code
    assert results[watershed.id].dtype == np.int32
    assert int(results[watershed.id].max()) == 2


def test_exported_intensity_measurement_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects_intensity")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect("input", measurements.id)

    image = np.zeros((7, 7), dtype=np.float32)
    image[1:3, 1:4] = 10
    image[4:6, 4:6] = 20
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)
    table = results[measurements.id]
    records = table.records()

    assert "measure_objects_with_intensity(" in code
    assert table.row_count == 2
    assert records[0]["intensity_mean"] == 10.0
    assert records[1]["intensity_mean"] == 20.0


def test_exported_label_volume_pipeline_executes():
    pipeline = _starter_pipeline()
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

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 10
    image[1, 7, 7] = 10
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)

    assert "label_connected_components(" in code
    assert "resolved_spatial_ndim=3" in code
    assert set(np.unique(results[relabeled.id])) == {0, 1}


def test_exported_label_property_filter_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    filtered = pipeline.add_node("filter_labels_by_property")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(filtered.id, "property_column", "area_pixels")
    pipeline.set_param(filtered.id, "min_value", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect(labels.id, filtered.id, target_port=0)
    pipeline.connect(measurements.id, filtered.id, target_port=1)

    image = np.zeros((8, 8), dtype=np.float32)
    image[1:4, 1:4] = 10
    image[6, 6] = 10
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)

    assert "filter_labels_by_property(" in code
    assert "resolved_spatial_ndim=2" in code
    assert set(np.unique(results[filtered.id])) == {0, 1}


def test_exported_clear_border_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    cleared = pipeline.add_node("clear_border_objects")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, cleared.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 0:3, 0:3] = 10
    image[1, 4:7, 4:7] = 10
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)

    assert "clear_border_objects(" in code
    assert "resolved_spatial_ndim=3" in code
    assert set(np.unique(results[cleared.id])) == {0, 2}


def test_exported_fill_holes_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    filled = pipeline.add_node("fill_holes")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(filled.id, "spatial_mode", "3D ZYX volume")
    pipeline.set_param(filled.id, "max_hole_size", 1)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, filled.id)

    mask = np.ones((3, 7, 7), dtype=bool)
    mask[1, 3, 3] = False
    mask[0, 1, 1] = False
    pipeline.run(mask.astype(np.float32), input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](mask.astype(np.float32))

    assert "fill_holes(" in code
    assert "max_hole_size=1" in code
    assert "resolved_spatial_ndim=3" in code
    assert results[filled.id][1, 3, 3]
    assert not results[filled.id][0, 1, 1]


def test_exported_remove_small_objects_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    filtered = pipeline.add_node("remove_small_objects")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(filtered.id, "min_size", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, filtered.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 1
    image[1, 7, 7] = 1
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)

    assert "remove_small_objects(" in code
    assert "resolved_spatial_ndim=3" in code
    assert results[filtered.id][:, 1:4, 1:4].all()
    assert not results[filtered.id][1, 7, 7]


def test_exported_measure_objects_pipeline_executes_and_saves_table(tmp_path):
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(measurements.id, "include_axis_descriptors", True)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 1
    image[1, 7, 7] = 1
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)
    table = results[measurements.id]
    output_path = tmp_path / "measurements.ome.tif"

    namespace["save_image"](table, output_path)

    assert "measure_objects(" in code
    assert "include_axis_descriptors=True" in code
    assert "from napari_vipp.core.tables import" in code
    assert table.row_count == 2
    assert table.columns[:2] == ("label_id", "volume_voxels")
    assert "major_axis_length_voxels" in table.columns
    csv_path = tmp_path / "measurements.ome.csv"
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").startswith(
        "label_id,volume_voxels"
    )


def test_exported_merged_table_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    morphology = pipeline.add_node("measure_objects")
    intensity = pipeline.add_node("measure_objects_intensity")
    merged = pipeline.add_node("merge_tables")
    annotated = pipeline.add_node("add_metadata_columns")
    selected = pipeline.add_node("select_table_columns")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(annotated.id, "metadata_columns", "condition=demo")
    pipeline.set_param(selected.id, "columns", "label_id,intensity_mean,condition")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, morphology.id)
    pipeline.connect(labels.id, intensity.id, target_port=0)
    pipeline.connect("input", intensity.id, target_port=1)
    pipeline.connect(morphology.id, merged.id, target_port=0)
    pipeline.connect(intensity.id, merged.id, target_port=1)
    pipeline.connect(merged.id, annotated.id)
    pipeline.connect(annotated.id, selected.id)

    image = np.zeros((7, 7), dtype=np.float32)
    image[1:3, 1:4] = 10
    image[4:6, 4:6] = 20
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)
    table = results[selected.id]

    assert "merge_tables(" in code
    assert "add_metadata_columns(" in code
    assert "select_table_columns(" in code
    assert table.row_count == 2
    assert table.columns == ("label_id", "intensity_mean", "condition")
    assert table.records()[0]["condition"] == "demo"


def test_exported_skeleton_analysis_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    skeleton = pipeline.add_node("skeletonize")
    measurements = pipeline.add_node("analyze_skeleton")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, skeleton.id)
    pipeline.connect(skeleton.id, measurements.id)

    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)
    table = results[measurements.id]
    record = table.records()[0]

    assert "skeletonize_mask(" in code
    assert "analyze_skeleton(" in code
    assert table.row_count == 1
    assert record["endpoint_voxel_count"] == 4
    assert record["branch_count"] == 4
    assert record["graph_node_count"] == 5
    assert record["graph_edge_count"] == 4
    assert record["voxel_graph_edge_count"] == 8
