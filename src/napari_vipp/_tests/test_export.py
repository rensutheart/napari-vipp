from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.export import (
    export_batch_runner_to_python,
    export_pipeline_to_python,
)
from napari_vipp.core.io import ImageDataset, ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import (
    AmbiguousAxisError,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.operations import (
    COMPOSITE_RGB_PERCENTILE_1_99,
    COMPOSITE_RGB_PRESERVE_VALUES,
)
from napari_vipp.core.pipeline import (
    GraphConnection,
    PrototypePipeline,
    SourcePayload,
)


def _starter_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    return pipeline


def _assert_embedded_operation(code: str, operation_id: str) -> None:
    assert f'"operation_id":"{operation_id}"' in code


def test_exported_batch_runner_uses_sibling_defaults_and_prints_summary(
    monkeypatch,
    tmp_path,
    capsys,
):
    script_path = tmp_path / "vipp_batch_runner.py"
    manifest_path = tmp_path / "vipp_batch_manifest.json"
    calls: list[tuple[str, str]] = []
    result = SimpleNamespace(
        summary={
            "completed": 3,
            "partial": 0,
            "skipped": 2,
            "failed": 0,
        },
        saved_paths=[tmp_path / "first.tif", tmp_path / "second.tif"],
        manifest_path=manifest_path,
        has_failures=False,
    )

    def fake_run_batch_from_files(workflow, config):
        calls.append((workflow, config))
        return result

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch_from_files",
        fake_run_batch_from_files,
    )
    code = export_batch_runner_to_python()
    compiled = compile(code, "<exported-batch-runner>", "exec")
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(script_path),
    }
    exec(compiled, namespace)

    assert namespace["main"]([]) == 0
    assert calls == [
        (
            None,
            str(tmp_path / "vipp_batch_config.json"),
        )
    ]
    assert capsys.readouterr().out == (
        "3 completed, 0 partial, 2 skipped, 0 failed; "
        f"2 outputs saved; manifest: {manifest_path}\n"
    )


def test_exported_batch_runner_passes_cli_overrides_and_reports_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    workflow_path = tmp_path / "custom-workflow.json"
    config_path = tmp_path / "custom-config.json"
    calls: list[tuple[str, str]] = []
    result = SimpleNamespace(
        summary={
            "completed": 1,
            "partial": 0,
            "skipped": 0,
            "failed": 1,
        },
        saved_paths=[tmp_path / "successful-output.tif"],
        manifest_path=tmp_path / "manifest.json",
        has_failures=True,
    )

    def fake_run_batch_from_files(workflow, config):
        calls.append((workflow, config))
        return result

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch_from_files",
        fake_run_batch_from_files,
    )
    code = export_batch_runner_to_python()
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_runner.py"),
    }
    exec(compile(code, "<exported-batch-runner>", "exec"), namespace)

    assert (
        namespace["main"](
            [
                "--workflow",
                str(workflow_path),
                "--config",
                str(config_path),
            ]
        )
        == 1
    )
    assert calls == [(str(workflow_path), str(config_path))]
    assert capsys.readouterr().out == (
        "1 completed, 0 partial, 0 skipped, 1 failed; "
        f"1 outputs saved; manifest: {result.manifest_path}\n"
    )


def test_exported_batch_runner_reports_preflight_exception(
    monkeypatch,
    tmp_path,
    capsys,
):
    def failing_run(_workflow, _config):
        raise ValueError("workflow/config mismatch")

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch_from_files",
        failing_run,
    )
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_pipeline.py"),
    }
    exec(
        compile(
            export_batch_runner_to_python(),
            "<exported-batch-runner>",
            "exec",
        ),
        namespace,
    )

    assert namespace["main"]([]) == 2
    assert "workflow/config mismatch" in capsys.readouterr().err


def test_export_produces_valid_python():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    # Must compile as a module without syntax errors.
    compile(code, "<exported>", "exec")

    assert "def run_pipeline(" in code
    assert "def batch_process(" in code
    _assert_embedded_operation(code, "gaussian_blur")
    _assert_embedded_operation(code, "otsu_threshold")
    assert '"sigma":1.2' in code
    assert "pipeline_from_workflow(json.loads(_WORKFLOW_JSON))" in code
    assert "ImageDataset, read_image, write_image" in code
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


def test_export_preserves_scalar_channel_contract_with_pipeline_parity():
    pipeline = PrototypePipeline()
    filtered = pipeline.add_node("bilateral_filter")
    pipeline.connect("input", filtered.id)
    image = np.random.default_rng(12).random((5, 7, 3), dtype=np.float32)

    native = pipeline.run(image, input_metadata={"axes": "ZYX"})[filtered.id]
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    exported = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )[filtered.id]

    assert '"channel_axis":-1' in code
    np.testing.assert_allclose(exported, native, rtol=0.0, atol=0.0)


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
    results = namespace["run_pipeline"](
        np.zeros((3, 4, 6), dtype=np.uint8),
        input_metadata={"axes": "ZYX"},
    )

    assert '"resize_mode":"Output size"' in code
    assert '"x_size":12' in code
    assert results[node.id].shape == (5, 8, 12)


def test_export_handles_multi_input_nodes():
    pipeline = _starter_pipeline()
    add = pipeline.add_node("add_images")
    pipeline.connect("gaussian", add.id, target_port=1)
    pipeline.connect("input", add.id, target_port=0)

    code = export_pipeline_to_python(pipeline)
    compile(code, "<exported>", "exec")
    _assert_embedded_operation(code, "add_images")
    assert '"source":"input"' in code
    assert '"source":"gaussian"' in code


def test_export_keeps_incomplete_multi_input_node_uncomputed():
    pipeline = PrototypePipeline()
    add = pipeline.add_node("add_images")
    pipeline.connect("input", add.id, target_port=0)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    results = namespace["run_pipeline"](np.ones((3, 4), dtype=np.float32))
    assert results[add.id] is None
    _assert_embedded_operation(code, "add_images")


@pytest.mark.parametrize(
    "name",
    [
        "bad-name",
        "class",
        "",
        "OUTPUT_NODES",
        "Path",
        "batch_process",
        "load_image",
        "read_image",
        "save_image",
    ],
)
def test_export_rejects_invalid_function_name(name):
    with pytest.raises(ValueError, match="function name"):
        export_pipeline_to_python(PrototypePipeline(), function_name=name)


def test_export_rejects_function_name_that_shadows_used_operation():
    with pytest.raises(ValueError, match="function name"):
        export_pipeline_to_python(
            _starter_pipeline(),
            function_name="gaussian_blur",
        )


def test_source_only_export_compiles_without_empty_operation_import():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    assert "from napari_vipp.core.operations import" not in code
    np.testing.assert_array_equal(namespace["run_pipeline"](image)["input"], image)


def test_custom_export_function_name_is_used_by_generated_harness():
    pipeline = _starter_pipeline()

    code = export_pipeline_to_python(pipeline, function_name="analyze_image")
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    assert "def analyze_image(" in code
    assert "results = analyze_image(load_image(source_path))" in code
    assert namespace["analyze_image"](image)["threshold"].dtype == bool


def test_export_uses_unique_variables_for_colliding_node_identifiers():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    first = pipeline.add_node("linear_scale_offset")
    second = pipeline.add_node("linear_scale_offset")
    combined = pipeline.add_node("add_images")
    pipeline.set_param(first.id, "alpha", 2.0)
    pipeline.set_param(first.id, "beta", 0.0)
    pipeline.set_param(second.id, "alpha", 3.0)
    pipeline.set_param(second.id, "beta", 0.0)
    renamed_ids = {first.id: "branch-a", second.id: "branch a"}
    renamed_nodes = [
        replace(node, id=renamed_ids.get(node.id, node.id))
        for node in pipeline.nodes.values()
    ]
    pipeline.restore_graph(
        renamed_nodes,
        [
            GraphConnection("input", "branch-a"),
            GraphConnection("input", "branch a"),
            GraphConnection("branch-a", combined.id, target_port=0),
            GraphConnection("branch a", combined.id, target_port=1),
        ],
    )

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    np.testing.assert_array_equal(namespace["run_pipeline"](image)[combined.id], 5.0)


def test_export_includes_richardson_lucy_deconvolution_call():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    decon = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(decon.id, "spatial_mode", "2D YX")
    pipeline.set_param(decon.id, "iterations", 2)
    pipeline.set_param(decon.id, "resolved_spatial_ndim", 2)
    pipeline.connect("input", decon.id, target_port=0)
    pipeline.connect(psf_source.id, decon.id, target_port=1)
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    psf = np.zeros((3, 3), dtype=np.float32)
    psf[1, 1] = 1.0

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image, psf)

    _assert_embedded_operation(code, "richardson_lucy_deconvolution")
    assert '"resolved_spatial_ndim":2' in code
    assert results[decon.id].dtype == np.float32
    assert results[decon.id].shape == image.shape


def test_export_compiles_named_tunnel_connections_as_normal_inputs():
    pipeline = _starter_pipeline()
    median = pipeline.add_node("median_filter")
    pipeline.add_output_tunnel("Raw", "input", 0)
    result = pipeline.connect_to_tunnel("Raw", median.id)
    assert result.success

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.random.rand(4, 8, 8).astype(np.float32)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "median_filter")
    assert results[median.id].shape == image.shape


def test_export_prefers_explicit_batch_output_nodes():
    pipeline = _starter_pipeline()
    marker = pipeline.add_node("batch_output")
    pipeline.set_param(marker.id, "tag", "blurred")
    pipeline.connect("gaussian", marker.id)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        np.random.rand(4, 8, 8).astype(np.float32)
    )

    _assert_embedded_operation(code, "batch_output")
    assert namespace["OUTPUT_NODES"] == (marker.id,)
    assert results[marker.id].shape == (4, 8, 8)
    assert "threshold" in results


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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )

    _assert_embedded_operation(code, "subtract_background")
    assert '"radius":7' in code
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

    _assert_embedded_operation(code, "euclidean_distance_transform")
    _assert_embedded_operation(code, "h_maxima_markers")
    _assert_embedded_operation(code, "marker_controlled_watershed")
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    table = results[measurements.id]
    records = table.records()

    _assert_embedded_operation(code, "measure_objects_intensity")
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "label_connected_components")
    assert '"resolved_spatial_ndim":3' in code
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )

    _assert_embedded_operation(code, "filter_labels_by_property")
    assert '"resolved_spatial_ndim":2' in code
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "clear_border_objects")
    assert '"resolved_spatial_ndim":3' in code
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
    results = namespace["run_pipeline"](
        mask.astype(np.float32),
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "fill_holes")
    assert '"max_hole_size":1' in code
    assert '"resolved_spatial_ndim":3' in code
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "remove_small_objects")
    assert '"resolved_spatial_ndim":3' in code
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )
    table = results[measurements.id]
    output_path = tmp_path / "measurements.ome.tif"

    namespace["save_image"](table, output_path)

    _assert_embedded_operation(code, "measure_objects")
    assert '"include_axis_descriptors":true' in code
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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    table = results[selected.id]

    _assert_embedded_operation(code, "merge_tables")
    _assert_embedded_operation(code, "add_metadata_columns")
    _assert_embedded_operation(code, "select_table_columns")
    assert table.row_count == 2
    assert table.columns == ("label_id", "intensity_mean", "condition")
    assert table.records()[0]["condition"] == "demo"


def test_exported_summary_table_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    annotated = pipeline.add_node("add_metadata_columns")
    summarized = pipeline.add_node("summarize_measurements")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(annotated.id, "metadata_columns", "condition=demo")
    pipeline.set_param(summarized.id, "group_by", "condition")
    pipeline.set_param(summarized.id, "value_columns", "area_pixels")
    pipeline.set_param(summarized.id, "statistics", "mean,min,max")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect(measurements.id, annotated.id)
    pipeline.connect(annotated.id, summarized.id)

    image = np.zeros((2, 12, 12), dtype=np.float32)
    image[0, 1:4, 1:5] = 10
    image[0, 7:10, 7:11] = 10
    image[1, 2:7, 2:6] = 10
    pipeline.run(image, input_metadata={"axes": "TYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "TYX"},
    )
    table = results[summarized.id]

    _assert_embedded_operation(code, "add_metadata_columns")
    _assert_embedded_operation(code, "summarize_measurements")
    assert table.row_count == 1
    assert table.records()[0]["condition"] == "demo"
    assert table.records()[0]["row_count"] == 3
    assert table.records()[0]["area_pixels_mean"] == 14.666666666666666


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
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    table = results[measurements.id]
    record = table.records()[0]

    _assert_embedded_operation(code, "skeletonize")
    _assert_embedded_operation(code, "analyze_skeleton")
    assert table.row_count == 1
    assert record["endpoint_voxel_count"] == 4
    assert record["branch_count"] == 4
    assert record["graph_node_count"] == 5
    assert record["graph_edge_count"] == 4
    assert record["voxel_graph_edge_count"] == 8


def test_exported_extract_channel_uses_shared_explicit_axis_semantics():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    extracted = pipeline.add_node("extract_channel")
    pipeline.set_param(extracted.id, "channel", 1)
    pipeline.connect("input", extracted.id)
    image = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    image[:, 0] = 10
    image[:, 1] = 42
    image[:, 2] = 90

    native = pipeline.run(
        image,
        input_metadata={"axes": "ZCYX"},
    )[extracted.id]
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZCYX"},
    )
    state = image_state_from_array(image, layer_metadata={"axes": "ZCYX"})
    series = ImageSeriesInfo(0, "0", "image", image.shape, "uint16", "ZCYX")
    dataset = ImageDataset(
        image,
        state,
        SourceInspection("memory://image", "test", (series,)),
        series,
    )
    dataset_results = namespace["run_pipeline"](dataset)

    np.testing.assert_array_equal(results[extracted.id], native)
    np.testing.assert_array_equal(dataset_results[extracted.id], native)
    assert results[extracted.id].shape == (2, 4, 5)
    assert results.image_states[extracted.id].axis_order == "ZYX"
    assert results.image_states[extracted.id].axes_explicit
    with pytest.raises(AmbiguousAxisError, match="explicit channel axis"):
        namespace["run_pipeline"](image)


@pytest.mark.parametrize(
    "intensity_mapping",
    [COMPOSITE_RGB_PRESERVE_VALUES, COMPOSITE_RGB_PERCENTILE_1_99],
)
def test_exported_composite_matches_native_values_colors_and_provenance(
    intensity_mapping,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    composite = pipeline.add_node("composite_to_rgb")
    pipeline.set_param(composite.id, "intensity_mapping", intensity_mapping)
    pipeline.connect("input", composite.id)
    image = np.arange(2 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 2, 4, 5)
    image[:, 1] *= 3
    state = image_state_from_array(
        image,
        layer_metadata={"axes": "ZCYX"},
        channels=(
            ChannelMetadata(name="yellow", color=0xFFFF00),
            ChannelMetadata(name="cyan", color=0x00FFFF),
        ),
    )
    payload = SourcePayload(image, image_state=state)

    native = pipeline.run(
        image,
        source_payloads={"input": payload},
    )[composite.id]
    native_state = pipeline.output_states[composite.id]
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )

    exported = namespace["run_pipeline"](
        image,
        source_image_states={"input": state},
    )

    np.testing.assert_array_equal(exported[composite.id], native)
    assert exported.image_states[composite.id].to_dict() == native_state.to_dict()
    assert exported.image_states[composite.id].axis_order == "Z,Y,X,rgb"


def test_exported_mask_uses_per_source_semantics_for_broadcasting():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(masked.id, "outside_value", -5)
    pipeline.connect(mask_source.id, threshold.id)
    pipeline.connect("input", masked.id, target_port=0)
    pipeline.connect(threshold.id, masked.id, target_port=1)
    image = np.arange(2 * 2 * 3 * 4, dtype=np.int16).reshape(2, 2, 3, 4)
    mask = np.zeros((2, 3, 4), dtype=np.float32)
    mask[0, :, 0] = 1
    mask[1, :, -1] = 1

    native = pipeline.run(
        image,
        input_metadata={"axes": "TZYX"},
        source_payloads={
            mask_source.id: SourcePayload(mask, {"axes": "TYX"}),
        },
    )[masked.id]
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    results = namespace["run_pipeline"](
        image,
        mask,
        input_metadata={"axes": "TZYX"},
        source_metadata={mask_source.id: {"axes": "TYX"}},
    )

    np.testing.assert_array_equal(results[masked.id], native)
    assert results.image_states[masked.id].axis_order == "TZYX"
    with pytest.raises(ValueError, match="explicit axis semantics"):
        namespace["run_pipeline"](image, mask)


def test_exported_multi_source_call_rejects_a_missing_binding():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second = pipeline.add_node("input")
    added = pipeline.add_node("add_images")
    pipeline.connect("input", added.id, target_port=0)
    pipeline.connect(second.id, added.id, target_port=1)
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    with pytest.raises(ValueError, match=rf"Source node {second.id!r} has no input"):
        namespace["run_pipeline"](np.ones((3, 4), dtype=np.float32))


def test_exported_source_payload_mapping_can_supply_every_source():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second = pipeline.add_node("input")
    added = pipeline.add_node("add_images")
    pipeline.connect("input", added.id, target_port=0)
    pipeline.connect(second.id, added.id, target_port=1)
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    first_data = np.ones((3, 4), dtype=np.float32)
    second_data = np.full((3, 4), 2, dtype=np.float32)

    results = namespace["run_pipeline"](
        source_payloads={
            "input": SourcePayload(first_data, {"axes": "YX"}),
            second.id: SourcePayload(second_data, {"axes": "YX"}),
        }
    )

    np.testing.assert_array_equal(results[added.id], 3)


def test_exported_sources_reject_unknown_or_duplicate_bindings():
    pipeline = PrototypePipeline()
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown exported source nodes"):
        namespace["run_pipeline"](
            source_payloads={"typo": SourcePayload(image)},
        )
    with pytest.raises(ValueError, match="supplied both positionally"):
        namespace["run_pipeline"](
            image,
            source_payloads={"input": SourcePayload(image)},
        )


def test_exported_workflow_refuses_an_unvalidated_runtime_version():
    pipeline = PrototypePipeline()
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    namespace["VIPP_VERSION"] = "different-runtime"

    with pytest.raises(RuntimeError, match="active runtime is different-runtime"):
        namespace["run_pipeline"](np.ones((3, 4), dtype=np.float32))


def test_exported_workflow_snapshot_is_revalidated_and_fresh_per_run():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    encoded = namespace["_WORKFLOW_JSON"]
    first = namespace["_new_pipeline"]()
    first.set_param("gaussian", "sigma", 9.0)

    second = namespace["_new_pipeline"]()

    assert namespace["_WORKFLOW_JSON"] == encoded
    assert second.nodes["gaussian"].params["sigma"] == 1.2


def test_exported_load_helper_keeps_the_complete_dataset():
    code = export_pipeline_to_python(_starter_pipeline())
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    dataset = object()
    namespace["read_image"] = lambda _path: dataset

    assert namespace["load_image"]("source.ome.tif") is dataset


def test_exported_save_helper_passes_carried_output_state(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    calibrated = pipeline.add_node("set_pixel_size")
    pipeline.set_param(calibrated.id, "x_size", 0.2)
    pipeline.set_param(calibrated.id, "y_size", 0.3)
    pipeline.set_param(calibrated.id, "unit", "micrometer")
    pipeline.connect("input", calibrated.id)
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        np.ones((3, 4), dtype=np.float32),
        input_metadata={"axes": "YX"},
    )
    captured: dict[str, object] = {}

    def fake_write_image(data, path, **kwargs):
        captured.update(data=data, path=path, **kwargs)

    namespace["write_image"] = fake_write_image
    output_path = tmp_path / "calibrated.ome.tif"
    namespace["save_image"](
        results[calibrated.id],
        output_path,
        image_state=results.image_states[calibrated.id],
    )

    assert captured["image_state"] is results.image_states[calibrated.id]
    assert captured["image_state"].axes[-1].scale == 0.2
    assert captured["image_state"].axes[-2].scale == 0.3
    assert "image_state=results.image_states.get" in code
