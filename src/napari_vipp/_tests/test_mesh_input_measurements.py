"""Direct mesh measurements preserve geometry and image measurement contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.meshes import MeshData, MeshState, mask_to_3d_mesh
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import measure_3d_mesh_morphology
from napari_vipp.core.pipeline import PrototypePipeline


def _row(table):
    return dict(zip(table.columns, table.rows[0], strict=True))


def _cube():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    state = MeshState(
        8, 12, tuple(AxisMetadata(n, "space") for n in "zyx"), (2, 2, 2), "test"
    )
    return MeshData(vertices, faces, state)


def _pipeline():
    p = PrototypePipeline()
    p.reset_empty_graph()
    threshold = p.add_node("binary_threshold")
    threshold.params["threshold"] = 0.5
    mesh = p.add_node("mask_to_3d_mesh")
    measurements = p.add_node("measure_3d_mesh_morphology")
    assert p.connect("input", threshold.id).success
    assert p.connect(threshold.id, mesh.id).success
    assert p.connect(mesh.id, measurements.id).success
    return p, mesh.id, measurements.id


def _mask():
    mask = np.zeros((12, 13, 14), bool)
    mask[1:11, 1:12, 1:13] = True
    mask[4:8, 4:8, 4:8] = False  # A cavity must subtract, not add volume.
    return mask


def test_analytical_cube_geometry_and_immutable_buffers():
    mesh = _cube()
    before = mesh.vertices.copy(), mesh.faces.copy()
    row = _row(measure_3d_mesh_morphology(mesh, minimum_voxel_count=100000))
    assert row["mesh_status"] == "ok" and row["watertight"]
    assert row["mesh_volume_physical"] == pytest.approx(1)
    assert row["mesh_surface_area_physical"] == pytest.approx(6)
    assert row["solidity_3d"] == pytest.approx(1)
    assert row["vertex_count"] == 8 and row["triangle_count"] == 12
    assert "label_id" not in row and "voxel_count" not in row
    np.testing.assert_array_equal(mesh.vertices, before[0])
    np.testing.assert_array_equal(mesh.faces, before[1])
    assert not mesh.vertices.flags.writeable and not mesh.faces.flags.writeable


def test_calibration_mixed_compatible_units_and_large_translation():
    mesh = _cube()
    axes = tuple(
        AxisMetadata(n, "space", u, s)
        for n, u, s in zip("zyx", ("nm", "um", "um"), (2000, 3, 4), strict=True)
    )
    mesh = replace(
        mesh, vertices=mesh.vertices + 1e7, state=replace(mesh.state, spatial_axes=axes)
    )
    table = measure_3d_mesh_morphology(mesh)
    row = _row(table)
    assert row["mesh_volume_physical"] == pytest.approx(24)
    assert row["mesh_surface_area_physical"] == pytest.approx(52)
    assert row["physical_unit"] == "um"
    assert dict(table.column_units)["mesh_volume_physical"] == "um^3"
    assert [row[f"mesh_extent_{n}_physical"] for n in "zyx"] == pytest.approx([2, 3, 4])


@pytest.mark.parametrize("damage", ["open", "winding", "duplicate"])
def test_invalid_surface_reports_area_but_no_volume(damage):
    mesh = _cube()
    faces = mesh.faces.copy()
    if damage == "open":
        faces = faces[:-1]
    elif damage == "winding":
        faces[0] = faces[0, ::-1]
    else:
        faces = np.concatenate([faces, faces[:1]])
    row = _row(
        measure_3d_mesh_morphology(replace(mesh, faces=faces, face_object_ids=None))
    )
    assert row["mesh_status"] == "open_or_invalid_surface"
    assert not row["watertight"] and row["mesh_error"]
    assert row["mesh_surface_area_physical"] > 0
    for name in (
        "mesh_volume_physical",
        "surface_area_to_volume",
        "sphericity",
        "solidity_3d",
    ):
        assert np.isnan(row[name])


def test_empty_and_optional_hull_and_incompatible_units():
    empty = mask_to_3d_mesh(np.zeros((3, 3, 3), bool))
    row = _row(measure_3d_mesh_morphology(empty, include_convex_hull_metrics=False))
    assert row["mesh_status"] == "empty_mesh"
    assert not row["watertight"] and np.isnan(row["mesh_volume_physical"])
    assert not any("hull" in name for name in row)
    mesh = _cube()
    axes = list(mesh.state.spatial_axes)
    axes[0] = replace(axes[0], unit="um")
    with pytest.raises(ValueError, match="compatible units"):
        measure_3d_mesh_morphology(
            replace(mesh, state=replace(mesh.state, spatial_axes=tuple(axes)))
        )


def test_existing_mesh_matches_label_path_including_cavity(monkeypatch):
    from napari_vipp.core import operations

    mask = _mask()
    state = image_state_from_array(mask, layer_metadata={"axes": "ZYX"})
    mesh = mask_to_3d_mesh(mask, image_state=state)
    expected = _row(
        measure_3d_mesh_morphology(mask.astype(np.int32), spatial_mode="3D ZYX")
    )

    def forbidden(*_a, **_kw):
        pytest.fail("An existing mesh must not call marching cubes again")

    monkeypatch.setattr(operations.measure, "marching_cubes", forbidden)
    actual = _row(measure_3d_mesh_morphology(mesh))
    for name in expected:
        if name in actual and isinstance(expected[name], float):
            assert actual[name] == pytest.approx(
                expected[name], rel=2e-6, nan_ok=True
            ), name


def test_binary_mask_path_and_pipeline_workflow_restore(tmp_path):
    from napari_vipp.core.batch_setup import pipeline_from_workflow
    from napari_vipp.core.workflow import save_workflow

    mask = _mask()
    actual = measure_3d_mesh_morphology(mask, spatial_mode="3D ZYX")
    expected = measure_3d_mesh_morphology(mask.astype(np.int32), spatial_mode="3D ZYX")
    assert actual == expected
    p, mesh_id, result_id = _pipeline()
    p.run(mask, input_metadata={"axes": "ZYX"})
    assert "no remeshing" in p.output_states[result_id].history[-1]
    path = tmp_path / "mesh.json"
    save_workflow(path, p)
    restored = pipeline_from_workflow(json.loads(path.read_text()))
    restored.run(mask, input_metadata={"axes": "ZYX"})
    assert _row(restored.outputs[result_id])["mesh_volume_physical"] == pytest.approx(
        _row(p.outputs[result_id])["mesh_volume_physical"]
    )
    image = p.add_node("gaussian_blur")
    assert not p.connect(mesh_id, image.id).success


def test_generated_python_runs_shared_mesh_measurements():
    from napari_vipp.core.export import export_pipeline_to_python

    p, _mesh_id, result_id = _pipeline()
    namespace = {"__name__": "mesh_measurements_export"}
    exec(compile(export_pipeline_to_python(p), "<export>", "exec"), namespace)
    results = namespace["run_pipeline"](_mask(), input_metadata={"axes": "ZYX"})
    assert _row(results[result_id])["mesh_status"] == "ok"


@pytest.mark.parametrize("mode", ["cpu", "auto", "prefer_gpu"])
@pytest.mark.parametrize("cached_mesh", [False, True])
@pytest.mark.parametrize("input_kind", ["mesh", "mask"])
def test_detached_execution_keeps_existing_mesh_on_cpu(mode, cached_mesh, input_kind):
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.workflow import serialize_workflow

    p, _mesh_id, result_id = _pipeline()
    if input_kind == "mask":
        assert p.connect("binary_threshold_1", result_id).success
    data = _mask()
    cached = {}
    if cached_mesh:
        first = execute_pipeline_request(
            PipelineRunRequest(
                run_id=0,
                workflow=serialize_workflow(p),
                input_data=data,
                input_metadata={"axes": "ZYX"},
                input_name="mesh",
                source_payloads={},
                compute_request=ComputeRequest(mode=mode),
                manual_node_ids=frozenset(p.nodes) - {result_id},
            ),
            raise_errors=True,
        )
        p = first.pipeline
        assert p.outputs[result_id] is None
        cached = dict(
            dirty_node_ids=frozenset({result_id}),
            completed_node_ids=frozenset(p.completed_node_ids),
            cached_outputs=dict(p.outputs),
            cached_output_states=dict(p.output_states),
            cached_node_outputs=dict(p.node_outputs),
            cached_node_output_states=dict(p.node_output_states),
            cached_execution_states=dict(p.node_execution_states),
            cached_execution_messages=dict(p.node_execution_messages),
            cached_compute_provenance={
                **p.node_cache_lineage,
                **p.node_compute_provenance,
            },
        )
    request = PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(p),
        input_data=data,
        input_metadata={"axes": "ZYX"},
        input_name="mesh",
        source_payloads={},
        compute_request=ComputeRequest(mode=mode),
        manual_node_ids=frozenset({result_id}) if cached_mesh else frozenset(p.nodes),
        **cached,
    )
    started = []
    result = execute_pipeline_request(
        request, raise_errors=True, node_started_callback=started.append
    )
    assert not result.error and not result.cancelled
    if cached_mesh:
        assert started == [result_id]
        assert result.pipeline.outputs[_mesh_id] is p.outputs[_mesh_id]
    assert result.pipeline.outputs[result_id] is not None, (
        result.pipeline.node_execution_states,
        result.pipeline.node_execution_messages,
        result.execution_report.warnings,
    )
    assert _row(result.pipeline.outputs[result_id])["mesh_status"] == "ok"
    decision = next(
        d for d in result.execution_report.actual_decisions if d.node_id == result_id
    )
    assert decision.runtime_id == "cpu-numpy"


@pytest.mark.parametrize("input_kind", ["mesh", "object_array"])
def test_workload_mesh_type_is_derived_from_value_not_authored_parameters(input_kind):
    from napari_vipp.core.execution import _workload_parameters
    from napari_vipp.core.node_execution import PreparedNodeCall

    p, _mesh_id, result_id = _pipeline()
    p.nodes[result_id].params["_vipp_input_kind"] = "mesh"
    value = _cube() if input_kind == "mesh" else np.array(["not a mesh"], object)
    call = PreparedNodeCall(
        node_id=result_id,
        operation_id="measure_3d_mesh_morphology",
        cpu_function=measure_3d_mesh_morphology,
        inputs=(value,),
        kwargs={"_vipp_input_kind": "mesh"},
    )
    parameters = dict(_workload_parameters(p, result_id, call))
    assert parameters.get("_vipp_input_kind") == (
        "mesh" if input_kind == "mesh" else None
    )
    assert "_vipp_input_kind" not in dict(_workload_parameters(p, result_id, None))


def test_batch_mesh_measurements_publish_csv(tmp_path):
    import tifffile

    from napari_vipp.core.batch import run_batch
    from napari_vipp.core.batch_setup import build_collection_batch_config
    from napari_vipp.core.workflow import serialize_workflow

    p, _mesh_id, _result_id = _pipeline()
    workflow = serialize_workflow(p)
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
        save_python_script=False,
    )
    result = run_batch(workflow, config)
    assert not result.has_failures, result.summary
    assert len(result.saved_paths) == 1 and result.saved_paths[0].suffix == ".csv"
    assert "mesh_id" in result.saved_paths[0].read_text()


def test_mesh_measurement_cancellation():
    class Cancel:
        def check_cancelled(self):
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        measure_3d_mesh_morphology(_cube(), progress=Cancel())


def test_new_graph_cards_hide_only_non_thumbnail_outputs(qtbot):
    from qtpy.QtCore import QPointF

    from napari_vipp._graph import PipelineGraphView, _types_compatible
    from napari_vipp.core.pipeline import NODE_LIBRARY

    p = PrototypePipeline()
    p.reset_empty_graph()
    view = PipelineGraphView()
    qtbot.addWidget(view)
    for spec in NODE_LIBRARY:
        if spec.id == "input":
            continue
        node = p.add_node(spec.id)
        view.add_node(node, QPointF())
        kinds = [port.output_type for port in p.output_ports(node.id)]
        if kinds and all(kind in {"mesh", "table"} for kind in kinds):
            assert view._cards[node.id].preview.isHidden(), spec.id
    mesh = p.add_node("mask_to_3d_mesh")
    measure = p.add_node("measure_3d_mesh_morphology")
    assert _types_compatible(mesh.output_type, measure.input_type)
    assert _types_compatible("mask", measure.input_type)
    assert not _types_compatible("image", measure.input_type)


def test_mesh_inspector_hides_image_only_settings_and_placeholder(qtbot):
    from napari.components import ViewerModel

    from napari_vipp._widget import VippWidget

    viewer = ViewerModel()
    viewer.add_image(_mask().astype(float), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    p, _mesh_id, result_id = _pipeline()
    widget.pipeline = p
    widget._build_graph_from_pipeline()
    assert widget.graph_view._cards[result_id].preview.isHidden()
    p.run(_mask(), input_metadata={"axes": "ZYX"})
    widget._build_graph_from_pipeline()
    widget._update_thumbnails()
    widget.graph_view.select_node(result_id)
    widget.inspect_node(result_id)
    assert widget.graph_view._cards[result_id].preview.isHidden()
    assert "Existing mesh" in widget._operation_help_note(result_id)
    specs = {
        s.name: s for s in p.operation_spec("measure_3d_mesh_morphology").parameters
    }
    assert widget._parameter_spec_hidden(result_id, specs["spatial_mode"])
    assert widget._parameter_spec_hidden(result_id, specs["minimum_voxel_count"])
    assert not widget._parameter_spec_hidden(
        result_id, specs["include_convex_hull_metrics"]
    )
