"""Save Image accepts meshes and output format menus stay input-specific."""

import zipfile
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_mesh_input_measurements import _cube
from napari_vipp.core.meshes import MeshData
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import save_output
from napari_vipp.core.pipeline import PrototypePipeline


def _mesh():
    mesh = _cube()
    return replace(
        mesh,
        state=replace(
            mesh.state,
            spatial_axes=tuple(
                replace(axis, unit="um") for axis in mesh.state.spatial_axes
            ),
        ),
        objects=(replace(mesh.objects[0], color=(0.2, 0.8, 0.4, 1)),),
    )


def _image_metadata(data):
    state = image_state_from_array(
        data, axes=tuple(AxisMetadata(a, "space", "um", 1) for a in "zyx")
    )
    return {"vipp_image_state": state.to_dict()}


@pytest.mark.parametrize("format", ["obj", "3mf"])
@pytest.mark.parametrize("auto", [False, True])
def test_save_node_writes_mesh_and_preserves_identity(tmp_path, format, auto):
    mesh = _mesh()
    path = tmp_path / f"surface.{format}"
    result = save_output(
        mesh, enabled="on", path=str(path), format="auto" if auto else format
    )
    assert result is mesh
    assert not mesh.vertices.flags.writeable
    assert path.is_file()
    if format == "3mf":
        with zipfile.ZipFile(path) as archive:
            model = archive.read("3D/3dmodel.model").decode()
        assert 'unit="micron"' in model
        assert "basematerials" in model
    else:
        assert "o VIPP_object_1" in path.read_text()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_output(mesh, enabled="on", path=str(path))
    assert path.read_bytes() == original
    assert save_output(mesh, enabled="on", path=str(path), overwrite="yes") is mesh


def test_disabled_writer_does_not_write_or_convert_mesh(tmp_path):
    mesh = _mesh()
    assert save_output(mesh, path=str(tmp_path / "not-created.obj")) is mesh
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("format,suffix", [("tiff", ".tif"), ("obj", ".npy")])
def test_mesh_writer_rejects_incompatible_format_or_path(tmp_path, format, suffix):
    path = tmp_path / f"wrong{suffix}"
    with pytest.raises(ValueError, match="matching OBJ"):
        save_output(_mesh(), enabled="on", path=str(path), format=format)
    assert not path.exists()


def _writer_pipeline(path, format):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    mesh = pipeline.add_node("mask_to_3d_mesh")
    writer = pipeline.add_node("save_output")
    measure = pipeline.add_node("measure_3d_mesh_morphology")
    writer.params.update(enabled="on", path=str(path), format=format)
    for source, target in (
        ("input", threshold.id),
        (threshold.id, mesh.id),
        (mesh.id, writer.id),
        (writer.id, measure.id),
    ):
        assert pipeline.connect(source, target).success
    return pipeline, mesh.id, writer.id, measure.id


@pytest.mark.parametrize("mode", ["cpu", "auto"])
@pytest.mark.parametrize("format", ["obj", "3mf"])
def test_mesh_writer_runs_in_pipeline_with_downstream_measurement(
    tmp_path, mode, format
):
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.workflow import serialize_workflow

    path = tmp_path / f"pipeline.{format}"
    pipeline, mesh_id, writer_id, measure_id = _writer_pipeline(path, format)
    data = np.zeros((6, 7, 8), np.float32)
    data[1:5, 1:6, 1:7] = 1
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata=_image_metadata(data),
            input_name="mesh",
            source_payloads={},
            compute_request=ComputeRequest(mode=mode),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    assert path.is_file()
    result_pipeline = result.pipeline
    mesh = result_pipeline.outputs[mesh_id]
    assert isinstance(mesh, MeshData)
    assert result_pipeline.outputs[writer_id] is mesh
    assert result_pipeline.output_states[writer_id] is mesh.state
    assert result_pipeline.outputs[measure_id].row_count == 1


@pytest.mark.parametrize("format", ["obj", "3mf", "auto"])
def test_generated_python_can_execute_mesh_save_node(tmp_path, format):
    from napari_vipp.core.export import export_pipeline_to_python

    suffix = "3mf" if format == "auto" else format
    path = tmp_path / f"exported.{suffix}"
    pipeline, _mesh_id, writer_id, _measure_id = _writer_pipeline(path, format)
    namespace = {"__name__": "exported_mesh_writer"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    data = np.zeros((6, 7, 8), np.float32)
    data[1:5, 1:6, 1:7] = 1
    results = namespace["run_pipeline"](data, input_metadata=_image_metadata(data))
    assert isinstance(results[writer_id], MeshData)
    assert path.exists()
    saved = namespace["save_image"](
        results[writer_id], tmp_path / "second-copy", output_node_id=writer_id
    )
    assert saved.suffix == f".{suffix}"
    assert saved.is_file()


@pytest.mark.parametrize("format,filename", [("obj", "bad.obj"), ("auto", "old.3mf")])
def test_image_writer_rejects_mesh_format_and_old_mesh_path(tmp_path, format, filename):
    path = tmp_path / filename
    path.write_bytes(b"previous mesh")
    with pytest.raises(ValueError, match="input is an image, not a mesh"):
        save_output(
            np.zeros((3, 4), np.uint8),
            enabled="on",
            path=str(path),
            format=format,
            overwrite="yes",
        )
    assert path.read_bytes() == b"previous mesh"


def test_save_node_rejects_table_connections_and_preserves_mesh_port_type():
    from napari_vipp._graph import _types_compatible

    pipeline, _mesh_id, writer_id, _measure_id = _writer_pipeline("", "auto")
    assert pipeline.output_ports(writer_id)[0].output_type == "mesh"
    image_node = pipeline.add_node("gaussian_blur")
    assert not pipeline.connect(writer_id, image_node.id).success
    table = pipeline.add_node("measure_objects")
    assert not pipeline.connect(table.id, writer_id).success
    assert _types_compatible("mesh", "array_or_mesh")
    assert not _types_compatible("table", "array_or_mesh")


@pytest.mark.parametrize("operation", ["save_output", "batch_output"])
def test_format_menu_follows_connections_without_changing_saved_choices(
    qtbot, operation
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), np.float32)))
    qtbot.addWidget(widget)
    mesh = widget.add_node_from_palette("mask_to_3d_mesh")
    writer = widget.add_node_from_palette(operation)
    default = "auto" if operation == "save_output" else "batch default"

    def menu():
        combo = widget._parameter_widgets["format"].combo
        return combo, tuple(combo.itemData(i) for i in range(combo.count()))

    assert menu()[1] == (default,)
    assert widget.pipeline.connect(mesh.id, writer.id).success
    widget._render_parameters(writer.id)
    combo, choices = menu()
    assert choices == (default, "obj", "3mf")  # Even before mesh calculation.
    combo.setCurrentIndex(combo.findData("3mf"))
    assert writer.params["format"] == "3mf"
    assert widget.pipeline.disconnect(mesh.id, writer.id)
    assert widget.pipeline.connect("input", writer.id).success
    widget._refresh_selected_parameter_controls()
    combo, choices = menu()
    assert "tiff" in choices and not {"obj", "3mf", "csv", "tsv"}.intersection(choices)
    assert writer.params["format"] == "3mf"
    assert combo.currentIndex() == -1
    assert "saved: 3mf" in combo.placeholderText()
    # Selecting the default after an incompatible choice really emits an edit.
    combo.setCurrentIndex(combo.findData(default))
    assert writer.params["format"] == default
    widget.pipeline.disconnect("input", writer.id)
    widget.pipeline.connect(mesh.id, writer.id)
    widget._refresh_selected_parameter_controls()
    assert menu()[1] == (default, "obj", "3mf")
    assert writer.params["format"] == default


def test_batch_table_format_menu_is_table_only(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), np.float32)))
    qtbot.addWidget(widget)
    table = widget.add_node_from_palette("measure_objects")
    writer = widget.add_node_from_palette("batch_output")
    assert widget.pipeline.connect(table.id, writer.id).success
    widget._render_parameters(writer.id)
    combo = widget._parameter_widgets["format"].combo
    assert tuple(combo.itemData(i) for i in range(combo.count())) == (
        "batch default",
        "csv",
        "tsv",
    )
