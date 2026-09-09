"""The captured interactive example retains its settings and runs independently."""

import hashlib
import json
from pathlib import Path

import numpy as np

from napari_vipp._sample_data import _mesh_morphology_sample
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow, serialize_workflow
from napari_vipp.ui.examples import (
    EXAMPLE_WORKFLOWS,
    _example_workflow_by_id,
    _example_workflow_path,
)


def _example():
    spec = _example_workflow_by_id("mesh-objects")
    assert spec is not None and spec.category == "3D Meshes"
    path = _example_workflow_path(spec)
    document = load_workflow(path)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    return spec, path, json.loads(path.read_text()), pipeline


def test_mesh_example_preserves_authored_parameters_and_packaged_copy():
    spec, packaged, document, pipeline = _example()
    repository = Path(__file__).resolve().parents[3] / "examples" / spec.filename
    assert json.loads(repository.read_text()) == json.loads(packaged.read_text())
    # Freeze the captured document, including layout and inspector profiles,
    # independently of its filename and the example chooser's display name.
    assert hashlib.sha256(packaged.read_bytes()).hexdigest() == (
        "8412b1e6b87324b63ad9b984349a77d5371fbe786de4a601d5df0229d61ee180"
    )
    nodes = pipeline.nodes
    assert nodes["input"].params["source_mode"] == "sample"
    assert nodes["input"].params["sample_name"] == spec.samples[0]
    assert nodes["input"].params["file_path"] == ""
    assert nodes["mask_to_3d_mesh_1"].params["object_mode"] == "Single object"
    assert nodes["color_mesh_objects_1"].params == {
        "color_by": "triangle_count",
        "color_map": "turbo",
        "_vipp_auto_recalculate": True,
    }
    assert nodes["smooth_mesh_1"].params == {
        "iterations": 2,
        "strength": 1.0,
        "preserve_boundary": True,
        "_vipp_auto_recalculate": True,
    }
    assert nodes["simplify_mesh_1"].params == {
        "target_percent": 10.0,
        "aggressiveness": 4.0,
        "preserve_boundary": True,
        "_vipp_auto_recalculate": True,
    }
    for node_id, keep in (
        ("filter_mesh_objects_1", "In range"),
        ("filter_mesh_objects_2", "Outside range"),
    ):
        assert nodes[node_id].params == {
            "property_name": "mesh_volume_physical",
            "minimum": 10,
            "maximum": 1e12,
            "keep": keep,
        }
    assert nodes["batch_output_1"].params["format"] == "3mf"
    assert not any(n.operation_id == "save_output" for n in nodes.values())
    inspector = document["metadata"]["vipp"]["inspector"]
    assert inspector["selected_node_id"] == "smooth_mesh_1"
    assert inspector["display_profiles"]
    restored = {n["id"]: n["params"] for n in serialize_workflow(pipeline)["nodes"]}
    for node in document["nodes"]:
        for key, value in node["params"].items():
            assert restored[node["id"]][key] == value


def test_mesh_example_runs_with_five_calibrated_objects_without_writing(
    tmp_path, monkeypatch
):
    _spec, _path, document, pipeline = _example()
    data, kwargs, _kind = _mesh_morphology_sample()
    original = data.copy()
    data.setflags(write=False)
    monkeypatch.chdir(tmp_path)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=document,
            input_data=data,
            input_metadata=kwargs["metadata"],
            input_name=kwargs["name"],
            source_payloads={},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    outputs = result.pipeline.outputs
    assert outputs["mask_to_3d_mesh_1"].object_count == 1
    assert outputs["split_mesh_objects_1"].object_count == 5
    assert outputs["filter_mesh_objects_1"].object_count == 4
    assert outputs["filter_mesh_objects_2"].object_count == 1
    combined = outputs["combine_meshes_1"]
    refined = outputs["simplify_mesh_1"]
    assert combined.object_count == refined.object_count == 5
    assert refined.objects == combined.objects
    assert len(refined.faces) < len(combined.faces)
    assert [axis.scale for axis in refined.state.spatial_axes] == [2, 0.5, 0.5]
    assert all(axis.unit == "micrometer" for axis in refined.state.spatial_axes)
    table = outputs["measure_3d_mesh_morphology_1"]
    assert table.row_count == 5
    assert {row["mesh_id"] for row in table.records()} == {
        obj.object_id for obj in refined.objects
    }
    assert all(row["mesh_status"] == "ok" for row in table.records())
    assert outputs["batch_output_1"] is refined
    np.testing.assert_array_equal(data, original)
    assert not list(tmp_path.iterdir())


def test_only_one_mesh_example_is_registered_under_original_name():
    examples = [spec for spec in EXAMPLE_WORKFLOWS if spec.category == "3D Meshes"]
    assert [(spec.id, spec.title, spec.filename) for spec in examples] == [
        (
            "mesh-objects",
            "Mesh Objects, Colours & Refinement",
            "synthetic-mesh-objects.json",
        )
    ]
    assert _example_workflow_by_id("mesh-refinement-tuned") is None
    assert not (
        _example_workflow_path(examples[0]).parent
        / "synthetic-mesh-refinement-tuned.json"
    ).exists()


def test_mesh_example_can_be_selected_and_loaded_in_widget(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget
    from napari_vipp.ui.dialogs import ExampleWorkflowDialog

    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    dialog.filter_edit.setText("Mesh Objects, Colours")
    dialog.select_example("mesh-objects")
    assert dialog.selected_example().title == "Mesh Objects, Colours & Refinement"
    assert dialog.open_button.isEnabled()
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    path = widget.load_example_workflow("mesh-objects")
    assert path.name == "synthetic-mesh-objects.json"
    assert widget.pipeline.nodes["smooth_mesh_1"].params["strength"] == 1
    assert widget.pipeline.nodes["simplify_mesh_1"].params["aggressiveness"] == 4
    # The example launcher intentionally starts at Image Source; the saved
    # selected node and display profiles remain in the workflow document.
    assert widget._selected_node_id == "input"
