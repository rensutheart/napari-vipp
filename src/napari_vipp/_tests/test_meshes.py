"""Scientific and publication contracts for the first surface-mesh node."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.meshes import MeshData, mask_to_3d_mesh, save_mesh_output
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline


def _mask():
    data = np.zeros((6, 7, 8), dtype=bool)
    data[1:5, 1:6, 1:7] = True
    return data


def _pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    threshold.params["threshold"] = 0.5
    mesh = pipeline.add_node("mask_to_3d_mesh")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, mesh.id).success
    return pipeline, mesh.id


def _signed_volume(vertices, faces):
    triangles = vertices[faces]
    return (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )


def _obj(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    vertices = np.array(
        [
            [float(x) for x in line.split()[1:]]
            for line in lines
            if line.startswith("v ")
        ]
    )
    faces = np.array(
        [
            [int(x) - 1 for x in line.split()[1:]]
            for line in lines
            if line.startswith("f ")
        ]
    )
    metadata = json.loads(
        next(
            line.removeprefix("# VIPP metadata: ")
            for line in lines
            if line.startswith("# VIPP metadata: ")
        )
    )
    return vertices, faces, metadata


def test_full_resolution_closed_mesh_is_outward_and_does_not_mutate():
    source = _mask()
    before = source.copy()
    source.setflags(write=False)
    mesh = mask_to_3d_mesh(source)
    assert isinstance(mesh, MeshData)
    assert not mesh.vertices.flags.writeable and not mesh.faces.flags.writeable
    assert deepcopy(mesh) is mesh
    assert (
        mesh.nbytes
        == mesh.vertices.nbytes + mesh.faces.nbytes + mesh.face_object_ids.nbytes
    )
    np.testing.assert_array_equal(source, before)
    np.testing.assert_allclose(mesh.vertices.min(axis=0), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(mesh.vertices.max(axis=0), [4.5, 5.5, 6.5])
    edges = np.sort(
        np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
        ),
        axis=1,
    )
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    assert _signed_volume(mesh.vertices, mesh.faces) > 0


@pytest.mark.parametrize("axes", ["ZYX", "XYZ", "YZX"])
def test_axis_permutation_and_calibrated_obj(tmp_path, axes):
    canonical = _mask()
    order = tuple("ZYX".index(a) for a in axes)
    data = canonical.transpose(order)
    by_name = {
        "Z": AxisMetadata("z", "space", "um", 2, 20),
        "Y": AxisMetadata("y", "space", "nm", 500, 3000),
        "X": AxisMetadata("x", "space", "um", 0.25, -5),
    }
    state = image_state_from_array(
        data, axes=tuple(by_name[a] for a in axes), source_name="calibrated acquisition"
    )
    mesh = mask_to_3d_mesh(data, image_state=state)
    np.testing.assert_array_equal(mesh.vertices, mask_to_3d_mesh(canonical).vertices)
    path = save_mesh_output(mesh, tmp_path / "surface.obj")
    vertices, faces, metadata = _obj(path)
    np.testing.assert_allclose(
        vertices, (mesh.vertices * [2, 0.5, 0.25] + [20, 3, -5])[:, ::-1]
    )
    np.testing.assert_array_equal(faces, mesh.faces[:, ::-1])
    assert _signed_volume(vertices, faces) > 0
    assert metadata["export_unit"] == "um"
    assert metadata["source_name"] == "calibrated acquisition"
    assert metadata["step_size"] == 1


@pytest.mark.parametrize(
    "boundary", ["Close at image border", "Leave open at image border"]
)
def test_empty_mask_returns_empty_mesh_but_cannot_be_exported(tmp_path, boundary):
    mesh = mask_to_3d_mesh(np.zeros((3, 4, 5), bool), boundary)
    assert mesh.vertices.shape == mesh.faces.shape == (0, 3)
    with pytest.raises(ValueError, match="no triangles"):
        save_mesh_output(mesh, tmp_path / "empty.obj")
    assert not list(tmp_path.iterdir())


def test_border_policy_is_explicit():
    mask = np.ones((3, 4, 5), bool)
    closed = mask_to_3d_mesh(mask)
    opened = mask_to_3d_mesh(mask, "Leave open at image border")
    np.testing.assert_allclose(closed.vertices.min(axis=0), [-0.5] * 3)
    np.testing.assert_allclose(closed.vertices.max(axis=0), [2.5, 3.5, 4.5])
    assert len(closed.faces) > 0 and len(opened.faces) == 0
    half = mask.copy()
    half[:, :, 3:] = False
    opened = mask_to_3d_mesh(half, "Leave open at image border")
    assert np.all(opened.vertices[:, 2] == 2.5)


@pytest.mark.parametrize("shape", [(4, 5), (1, 4, 5), (2, 3, 4, 5)])
def test_rejects_non_volume_input(shape):
    with pytest.raises(ValueError, match="one 3D volume"):
        mask_to_3d_mesh(np.zeros(shape, bool))


@pytest.mark.parametrize("axes", ["CYX", "TYX", "QYX"])
def test_rejects_non_spatial_or_ambiguous_axes(axes):
    data = _mask()
    state = image_state_from_array(data, layer_metadata={"axes": axes})
    with pytest.raises(ValueError, match="explicitly declared"):
        mask_to_3d_mesh(data, image_state=state)


def test_rejects_implicit_axes_and_non_binary_data():
    with pytest.raises(ValueError, match="explicitly declared"):
        mask_to_3d_mesh(_mask(), image_state=image_state_from_array(_mask()))
    with pytest.raises(TypeError, match="binary mask"):
        mask_to_3d_mesh(_mask().astype(np.uint8))


def test_atomic_obj_failure_cancellation_and_no_overwrite(tmp_path, monkeypatch):
    import napari_vipp.core.meshes as module

    mesh = mask_to_3d_mesh(_mask())
    path = tmp_path / "surface.obj"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        save_mesh_output(mesh, path, overwrite=False)
    original = module.atomic_replace

    def fail(*_args):
        raise OSError("publication failed")

    monkeypatch.setattr(module, "atomic_replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        save_mesh_output(mesh, path)
    assert path.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(module, "atomic_replace", original)

    class Cancel:
        def check_cancelled(self):
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        save_mesh_output(mesh, path, progress=Cancel())
    assert path.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [path]
    path.unlink()
    assert save_mesh_output(mesh, path, overwrite=False) == path


def test_incompatible_units_do_not_write(tmp_path):
    mesh = mask_to_3d_mesh(_mask())
    axes = list(mesh.state.spatial_axes)
    axes[0] = replace(axes[0], unit="um")
    mesh = replace(mesh, state=replace(mesh.state, spatial_axes=tuple(axes)))
    with pytest.raises(ValueError, match="compatible units"):
        save_mesh_output(mesh, tmp_path / "surface.obj")
    assert not list(tmp_path.iterdir())


def test_pipeline_ports_metadata_and_workflow_roundtrip(tmp_path):
    from napari_vipp.core.batch_setup import pipeline_from_workflow
    from napari_vipp.core.metadata import format_compact_metadata, metadata_table_rows
    from napari_vipp.core.workflow import save_workflow

    pipeline, node_id = _pipeline()
    assert pipeline.operation_spec("mask_to_3d_mesh").execution_policy == "manual"
    image = pipeline.add_node("gaussian_blur")
    assert not pipeline.connect(node_id, image.id).success
    pipeline.remove_node(image.id)
    pipeline.run(_mask(), input_metadata={"axes": "ZYX"})
    mesh = pipeline.outputs[node_id]
    assert isinstance(mesh, MeshData)
    assert pipeline.output_states[node_id] is mesh.state
    assert "MESH:" in format_compact_metadata(mesh.state)
    assert any(row.label == "Vertices" for row in metadata_table_rows(mesh.state))
    path = tmp_path / "mesh.json"
    save_workflow(path, pipeline)
    restored = pipeline_from_workflow(json.loads(path.read_text()))
    restored.run(_mask(), input_metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(restored.outputs[node_id].faces, mesh.faces)


def test_napari_surface_inspect_pin_save_and_image_transition(qtbot, tmp_path):
    from napari.components import ViewerModel
    from napari.layers import Image, Surface

    from napari_vipp._widget import VippWidget

    viewer = ViewerModel()
    axes = tuple(
        AxisMetadata(n, "space", "um", s, t)
        for n, s, t in zip("zyx", [2, 0.5, 0.25], [10, 20, -3], strict=True)
    )
    state = image_state_from_array(_mask(), axes=axes)
    carried = {"vipp_image_state": state.to_dict()}
    viewer.add_image(_mask().astype(float), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_a, **_kw: False
    widget.pipeline, node_id = _pipeline()
    widget.pipeline.run(_mask(), input_metadata=carried)
    widget._build_graph_from_pipeline()
    widget.graph_view.select_node(node_id)
    widget.inspect_node(node_id)
    layers = widget._generated_layers_for_name(widget._inspect_layer_name)
    assert len(layers) == 1 and isinstance(layers[0], Surface)
    assert viewer.dims.ndisplay == 3
    np.testing.assert_allclose(layers[0].scale, [2, 0.5, 0.25])
    np.testing.assert_allclose(layers[0].translate, [10, 20, -3])
    assert not np.shares_memory(
        layers[0].data[0], widget.pipeline.outputs[node_id].vertices
    )
    assert "vipp_mesh_state" in layers[0].metadata
    assert "vipp_image_state" not in layers[0].metadata
    assert not widget._node_preview_enabled(node_id)
    assert "vertices" in widget._metadata_summary_text
    assert widget._node_can_pin(node_id)
    assert widget._inspector_profile_for_node(node_id).distribution_kind == "none"
    assert widget._save_node_output(node_id, str(tmp_path / "mesh.obj"))
    widget.pin_node(node_id)
    assert any(
        isinstance(layer, Surface)
        and layer.metadata.get("napari_vipp_kind") == "pinned"
        for layer in viewer.layers
    )
    widget.inspect_node("input")
    assert isinstance(
        widget._generated_layers_for_name(widget._inspect_layer_name)[0], Image
    )
    widget.inspect_node(node_id)
    assert isinstance(
        widget._generated_layers_for_name(widget._inspect_layer_name)[0], Surface
    )
    viewer.dims.ndisplay = 2
    widget._refresh_inspection_layer_if_active()
    assert viewer.dims.ndisplay == 2  # Refresh does not override the user's view.
    panel = widget.inspector_panel
    panel.setParent(None)
    qtbot.addWidget(panel)
    panel.setStyleSheet("font-family: Segoe UI; font-size: 12pt;")
    panel.resize(480, 900)
    panel.show()
    qtbot.wait(20)
    assert panel.grab().save(str(tmp_path / "mesh-inspector.png"))
    widget._parameter_widgets["boundary"].combo.setCurrentText(
        "Leave open at image border"
    )
    assert widget._save_node_output(node_id, str(tmp_path / "stale.obj")) is None
    assert not (tmp_path / "stale.obj").exists()
    widget.pipeline.run(np.zeros_like(_mask()), input_metadata=carried)
    widget._refresh_inspection_layer_if_active()
    assert not widget._generated_layers_for_name(widget._inspect_layer_name)
    assert not widget._node_can_pin(node_id)
    widget.inspect_node(node_id)
    assert "Empty mesh" in widget.status_label.text()


@pytest.mark.parametrize("invalid_format", ["stl", "tiff"])
def test_non_obj_export_is_rejected(tmp_path, invalid_format):
    with pytest.raises(ValueError, match="OBJ"):
        save_mesh_output(
            mask_to_3d_mesh(_mask()),
            tmp_path / f"mesh.{invalid_format}",
            format=invalid_format,
        )
    assert not list(tmp_path.iterdir())


def test_generated_python_executes_and_publishes_obj_with_provenance(tmp_path):
    from napari_vipp.core.export import export_pipeline_to_python

    pipeline, node_id = _pipeline()
    namespace = {"__name__": "exported_mesh_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](_mask(), input_metadata={"axes": "ZYX"})
    mesh = results[node_id]
    assert isinstance(mesh, MeshData)
    path = namespace["save_image"](
        mesh,
        tmp_path / "mesh.obj",
        image_state=results.image_states[node_id],
        provenance=results,
        output_node_id=node_id,
    )
    assert path.exists()
    vertices, faces, _ = _obj(path)
    assert _signed_volume(vertices, faces) > 0
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize("explicit_output", [False, True])
def test_batch_mesh_plan_execution_and_skip(tmp_path, explicit_output):
    import tifffile

    from napari_vipp.core.batch import ExistingFilePolicy, run_batch
    from napari_vipp.core.batch_setup import build_collection_batch_config
    from napari_vipp.core.workflow import serialize_workflow

    pipeline, mesh_id = _pipeline()
    if explicit_output:
        output = pipeline.add_node("batch_output")
        assert pipeline.connect(mesh_id, output.id).success
        pipeline.set_param(output.id, "format", "obj")
    workflow = serialize_workflow(pipeline)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tifffile.imwrite(
        inputs / "mask.ome.tif",
        _mask().astype(np.uint8),
        metadata={"axes": "ZYX"},
        photometric="minisblack",
    )
    config = build_collection_batch_config(
        workflow,
        input_dir=inputs,
        output_dir=tmp_path / "outputs",
        pattern="*.ome.tif",
        image_format="ome-tiff",
        save_python_script=False,
    )
    assert len(config.outputs) == 1 and config.outputs[0].kind == "mesh"
    result = run_batch(workflow, config)
    assert not result.has_failures, result.summary
    assert len(result.saved_paths) == 1 and result.saved_paths[0].suffix == ".obj"
    vertices, faces, _ = _obj(result.saved_paths[0])
    assert _signed_volume(vertices, faces) > 0
    before = result.saved_paths[0].read_bytes()
    again = run_batch(
        workflow, replace(config, existing_file_policy=ExistingFilePolicy.SKIP)
    )
    assert again.summary["skipped"] == 1
    assert result.saved_paths[0].read_bytes() == before
