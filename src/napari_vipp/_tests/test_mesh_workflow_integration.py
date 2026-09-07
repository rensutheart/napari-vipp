"""Mesh objects remain truthful across nodes, the viewer and publication."""

import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.meshes import (
    MeshData,
    MeshObject,
    labels_to_3d_mesh,
    mask_to_3d_mesh,
    save_mesh_output,
)
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.ui.mesh_display import mesh_surface_display


def _labels():
    data = np.zeros((12, 13, 14), np.uint16)
    data[1:4, 1:5, 1:5] = 3
    data[6:11, 6:12, 6:13] = 17
    return data


def _metadata(data):
    state = image_state_from_array(
        data,
        axes=tuple(
            AxisMetadata(a, "space", "um", s)
            for a, s in zip("zyx", [2, 1, 1], strict=True)
        ),
    )
    return {"vipp_image_state": state.to_dict()}


def test_shared_vertex_objects_get_distinct_detached_display_colors():
    template = mask_to_3d_mesh(_labels() > 0)
    mesh = MeshData(
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 1, 2], [0, 3, 1]]),
        template.state,
        np.array([3, 17]),
        (MeshObject(3, "Red", (1, 0, 0, 1)), MeshObject(17, "Blue", (0, 0, 1, 1))),
    )
    (vertices, faces, values), colors = mesh_surface_display(mesh)
    assert len(vertices) == 6  # Shared scientific vertices split only for display.
    np.testing.assert_array_equal(values[faces], [[3] * 3, [17] * 3])
    np.testing.assert_array_equal(colors[faces[0]], [[1, 0, 0, 1]] * 3)
    np.testing.assert_array_equal(colors[faces[1]], [[0, 0, 1, 1]] * 3)
    assert not np.shares_memory(vertices, mesh.vertices)
    assert not np.shares_memory(faces, mesh.faces)


def test_mesh_compute_contracts_and_category_order():
    from napari_vipp.core.compute_contracts import ValueKind
    from napari_vipp.core.compute_specs import compute_specs_for
    from napari_vipp.core.pipeline import PALETTE_NODE_LIBRARY

    categories = list(dict.fromkeys(spec.category for spec in PALETTE_NODE_LIBRARY))
    assert categories.index("3D Meshes") == categories.index("Morphology") + 1
    for operation in (
        "color_mesh_objects",
        "combine_meshes",
        "split_mesh_objects",
        "filter_mesh_objects",
        "smooth_mesh",
        "simplify_mesh",
    ):
        spec = compute_specs_for(operation)[0]
        assert all(port.value_kind == ValueKind.MESH for port in spec.input_ports)
        assert all(port.value_kind == ValueKind.MESH for port in spec.output_ports)


def test_surface_replacement_handles_new_geometry_and_colors(qtbot):
    from napari.components import ViewerModel

    from napari_vipp._widget import VippWidget

    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    data = _labels()
    one = labels_to_3d_mesh(np.where(data == 3, data, 0))
    two = labels_to_3d_mesh(data)
    two = replace(
        two,
        state=replace(
            two.state,
            spatial_axes=(
                AxisMetadata("z", "space", "nm", 2000, 10000),
                AxisMetadata("y", "space", "um", 0.5, 20),
                AxisMetadata("x", "space", "um", 0.25, -3),
            ),
        ),
    )
    name = widget._inspect_layer_name
    metadata = {"napari_vipp_kind": "inspect"}
    widget._set_or_add_mesh_layer(name, one, metadata, "inspect")
    layer = viewer.layers[name]
    widget._set_or_add_mesh_layer(name, two, metadata, "inspect")
    assert viewer.layers[name] is layer
    surface, colors = mesh_surface_display(two)
    np.testing.assert_array_equal(layer.data[0], surface[0])
    np.testing.assert_allclose(layer.vertex_colors, colors)
    np.testing.assert_allclose(layer.scale, [2, 0.5, 0.25])
    np.testing.assert_allclose(layer.translate, [10, 20, -3])
    assert {obj["object_id"] for obj in layer.metadata["vipp_mesh_objects"]} == {3, 17}
    widget._set_or_add_mesh_layer(name, one, metadata, "inspect")
    assert len(layer.vertex_colors) == len(mesh_surface_display(one)[0][0])


def test_mesh_inspector_dynamic_controls_and_safe_bounds(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("color_mesh_objects")
    assert not widget._node_preview_enabled(node.id)
    assert "color_map" not in widget._parameter_widgets
    widget._on_param_changed("color_by", "mesh_volume_physical")
    assert "color_map" in widget._parameter_widgets
    node = widget.add_node_from_palette("filter_mesh_objects")
    widget._parameter_widgets["minimum"].value_box.setValue(50)
    assert widget._parameter_widgets["maximum"].value_box.minimum() >= 50
    widget._on_param_changed("property_name", "triangle_count")
    assert widget._parameter_widgets["minimum"].value_box.decimals() == 0
    measure = widget.add_node_from_palette("measure_3d_mesh_morphology")
    assert widget.pipeline.connect(node.id, measure.id).success
    widget._render_parameters(measure.id)
    assert "minimum_voxel_count" not in widget._parameter_widgets
    assert "spatial_mode" not in widget._parameter_widgets


def test_smooth_mesh_strength_accepts_fractional_slider_and_entry_values(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("smooth_mesh")
    control = widget._parameter_widgets["strength"]
    assert control.value_box.decimals() == 2
    assert control.value_box.singleStep() == pytest.approx(0.01)
    assert control.value() == node.params["strength"] == 0.5
    assert "each iteration" in control.toolTip()
    assert "Shape and volume can change" in control.toolTip()

    # The slider has 101 positions, including fractional strengths and endpoints.
    assert (control.slider.minimum(), control.slider.maximum()) == (0, 100)
    for position in (0, 1, 25, 50, 99, 100):
        control.slider.setValue(position)
        assert control.value() == pytest.approx(position / 100)
        assert node.params["strength"] == pytest.approx(position / 100)

    control.value_box.setValue(0.37)
    control.value_box.stepUp()
    assert node.params["strength"] == pytest.approx(0.38)
    # Reselecting/rebuilding the inspector must not round the stored value.
    widget._render_parameters(node.id)
    assert widget._parameter_widgets["strength"].value() == pytest.approx(0.38)


@pytest.mark.parametrize("format", ["obj", "3mf"])
def test_object_writer_and_generated_python_preserve_format(tmp_path, format):
    from napari_vipp.core.export import export_pipeline_to_python

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    mesh_node = pipeline.add_node("mask_to_3d_mesh")
    pipeline.set_param(mesh_node.id, "object_mode", "Connected objects")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, mesh_node.id).success
    namespace = {"__name__": "exported_mesh_objects"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    data = _labels()
    results = namespace["run_pipeline"](data, input_metadata=_metadata(data))
    mesh = results[mesh_node.id]
    assert mesh.object_count == 2
    path = namespace["save_image"](
        mesh,
        tmp_path / f"objects.{format}",
        image_state=mesh.state,
        provenance=results,
        output_node_id=mesh_node.id,
    )
    assert path.suffix == f".{format}"
    assert len(list(tmp_path.glob("*.json"))) == 1
    if format == "3mf":
        assert zipfile.is_zipfile(path)
    else:
        text = path.read_text()
        assert "o VIPP_object_1" in text and "o VIPP_object_2" in text
        metadata = json.loads(
            next(
                line.removeprefix("# VIPP metadata: ")
                for line in text.splitlines()
                if line.startswith("# VIPP metadata: ")
            )
        )
        assert len(metadata["objects"]) == 2
    with pytest.raises(FileExistsError):
        save_mesh_output(mesh, path, overwrite=False)


def test_batch_3mf_plan_execution_and_keep_existing(tmp_path):
    import tifffile

    from napari_vipp.core.batch import ExistingFilePolicy, run_batch
    from napari_vipp.core.batch_setup import build_collection_batch_config
    from napari_vipp.core.workflow import serialize_workflow

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    threshold.params["threshold"] = 0.5
    mesh = pipeline.add_node("mask_to_3d_mesh")
    mesh.params["object_mode"] = "Connected objects"
    output = pipeline.add_node("batch_output")
    output.params["format"] = "3mf"
    for source, target in (
        ("input", threshold.id),
        (threshold.id, mesh.id),
        (mesh.id, output.id),
    ):
        assert pipeline.connect(source, target).success
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tifffile.imwrite(
        inputs / "mask.ome.tif",
        _labels(),
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": 2,
            "PhysicalSizeZUnit": "µm",
            "PhysicalSizeY": 1,
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeX": 1,
            "PhysicalSizeXUnit": "µm",
        },
    )
    workflow = serialize_workflow(pipeline)
    config = build_collection_batch_config(
        workflow,
        input_dir=inputs,
        output_dir=tmp_path / "outputs",
        pattern="*.ome.tif",
        image_format="ome-tiff",
        save_python_script=False,
    )
    result = run_batch(workflow, config)
    assert not result.has_failures, result.summary
    assert len(result.saved_paths) == 1
    path = result.saved_paths[0]
    assert path.suffix == ".3mf" and zipfile.is_zipfile(path)
    before = path.read_bytes()
    again = run_batch(
        workflow, replace(config, existing_file_policy=ExistingFilePolicy.SKIP)
    )
    assert again.summary["skipped"] == 1
    assert path.read_bytes() == before
