"""The authored segmentation/logic/mesh example remains portable and annotated."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path


def _example():
    spec = _example_workflow_by_id("separate-overlapping-objects")
    assert spec is not None
    path = _example_workflow_path(spec)
    graph = load_workflow(path)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        graph["nodes"], graph["connections"], graph["output_tunnels"]
    )
    return spec, path, json.loads(path.read_text(encoding="utf-8")), pipeline


def test_overlap_example_preserves_authored_science_and_display_profiles():
    spec, path, document, pipeline = _example()
    repository = Path(__file__).resolve().parents[3] / "examples" / spec.filename
    assert path.read_bytes() == repository.read_bytes()
    assert spec.title == "Separate Overlapping Objects"
    assert spec.category == "Segmentation & Labels"
    assert len(pipeline.nodes) == 27 and len(pipeline.connections) == 29
    notes = document.pop("notes")
    assert len(notes) == 6
    for step, note in enumerate(notes, 1):
        assert note["text"].startswith(f"{step}. ")
        assert note["position"][1] < min(
            pos[1] for pos in document["positions"].values()
        )
    # Protect authored settings, graph and display profiles without freezing
    # node coordinates: intentional layout/annotation improvements are allowed.
    document.pop("positions")
    assert (
        hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == "a64a83b57b853457ac04bb487097942db3fd5615a8ad7842cced5ee66893e233"
    )
    assert pipeline.nodes["input"].params["source_mode"] == "sample"
    assert pipeline.nodes["input"].params["sample_name"] == spec.samples[0]
    for node in pipeline.nodes.values():
        if node.operation_id == "save_output":
            assert node.params["enabled"] == "off"
            assert node.params["path"] == ""


def test_overlap_example_reconstructs_masks_and_retains_two_meshes(
    tmp_path, monkeypatch
):
    spec, _path, document, pipeline = _example()
    data, kwargs, _kind = next(
        item for item in make_sample_data() if item[1]["name"] == spec.samples[0]
    )
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
    np.testing.assert_array_equal(
        outputs["logical_xor_1"],
        np.logical_xor(outputs["otsu_threshold_2"], outputs["otsu_threshold_3"]),
    )
    np.testing.assert_array_equal(
        outputs["logical_or_1"],
        np.logical_or(outputs["dilate_1"], outputs["erode_1"]),
    )
    masks = [outputs["erode_2"], outputs["otsu_threshold_3"]]
    assert not np.array_equal(*masks)
    for index, mask in enumerate(masks, 1):
        assert mask.dtype == bool and mask.shape == data.shape
        assert mask.any() and ndimage.label(mask)[1] == 1
        np.testing.assert_array_equal(outputs[f"save_output_{3 - index}"], mask)
        mesh = outputs[f"mask_to_3d_mesh_{index}"]
        assert mesh.object_count == 1 and len(mesh.faces) > 0
        assert [axis.scale for axis in mesh.state.spatial_axes] == [0.45] * 3
        assert all(axis.unit == "micrometer" for axis in mesh.state.spatial_axes)
        table = outputs[f"measure_3d_mesh_morphology_{index}"]
        assert table.row_count == 1
        assert next(iter(table.records()))["mesh_status"] == "ok"
    combined = outputs["combine_meshes_1"]
    coloured = outputs["color_mesh_objects_3"]
    assert combined.object_count == coloured.object_count == 2
    assert len({obj.object_id for obj in coloured.objects}) == 2
    assert len({obj.color for obj in coloured.objects}) == 2
    assert len(combined.faces) == sum(
        len(outputs[f"mask_to_3d_mesh_{index}"].faces) for index in (1, 2)
    )
    np.testing.assert_array_equal(coloured.vertices, combined.vertices)
    np.testing.assert_array_equal(coloured.faces, combined.faces)
    np.testing.assert_array_equal(data, original)
    assert not list(tmp_path.iterdir())


def test_overlap_example_is_selectable_with_readable_notes(qtbot):
    from napari_vipp._graph import GraphNoteItem
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget
    from napari_vipp.ui.dialogs import ExampleWorkflowDialog

    dialog = ExampleWorkflowDialog()
    qtbot.addWidget(dialog)
    dialog.filter_edit.setText("overlapping")
    dialog.select_example("separate-overlapping-objects")
    assert dialog.selected_example().title == "Separate Overlapping Objects"
    assert dialog.open_button.isEnabled()
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    path = widget.load_example_workflow("separate-overlapping-objects")
    assert path.name == "synthetic-separate-overlapping-objects.json"
    assert len(widget._graph_notes) == 6
    _, _, document, _ = _example()
    # Check rendered note bounds at ordinary and larger system-font sizes.
    # Explanations stay above the graph at ordinary and larger system fonts.
    for point_size in (9, 14):
        rectangles = []
        for note in document["notes"]:
            item = GraphNoteItem(note["id"], note["text"], note["width"])
            font = item.font()
            font.setPointSizeF(point_size)
            item.setFont(font)
            item.setPos(*note["position"])
            rectangle = item.sceneBoundingRect()
            assert rectangle.bottom() < min(
                pos[1] for pos in document["positions"].values()
            )
            assert all(not rectangle.intersects(other) for other in rectangles)
            rectangles.append(rectangle)
