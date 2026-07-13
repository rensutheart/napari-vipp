from __future__ import annotations

from pathlib import Path

import pytest

from napari_vipp.core.batch import BATCH_WORKFLOW_FILENAME
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.ui.batch_controller import CollectionBatchController


def _explicit_batch_pipeline() -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    output = pipeline.add_node("batch_output")
    output.params.update(tag="result", format="npy")
    assert pipeline.connect("input", output.id).success
    return pipeline, output.id


def test_controller_previews_one_workflow_snapshot_and_current_pipeline(
    tmp_path,
):
    pipeline, output_id = _explicit_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    snapshots: list[dict] = []
    controller = CollectionBatchController(
        workflow_document_provider=lambda: snapshots.append(workflow) or workflow,
        pipeline_provider=lambda: pipeline,
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "field_a.npy").write_bytes(b"a")
    (input_dir / "field_b.npy").write_bytes(b"b")

    preview = controller.preview(
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
        preview_limit=1,
    )

    assert snapshots == [workflow]
    assert preview.total_items == 2
    assert len(preview) == 1
    assert preview.explicit_outputs
    assert preview.collision_count == 0
    assert preview[0].output_statuses == ("new",)
    assert controller.source_rows() == [
        {
            "node_id": "input",
            "title": pipeline.nodes["input"].title,
            "binding_mode": "collection",
        }
    ]
    assert preview[0].outputs[0].name.endswith("__result.npy")
    assert output_id in pipeline.nodes

    preview[0].outputs[0].parent.mkdir(parents=True)
    preview[0].outputs[0].write_bytes(b"existing")
    collision = controller.preview(
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
        preview_limit=1,
    )

    assert collision.collision_count == 1
    assert collision[0].output_statuses == ("exists; collision",)


def test_controller_saves_companion_and_rejects_a_different_workflow(
    tmp_path,
):
    pipeline, output_id = _explicit_batch_pipeline()
    current_workflow = [serialize_workflow(pipeline)]
    controller = CollectionBatchController(
        workflow_document_provider=lambda: current_workflow[0],
        pipeline_provider=lambda: pipeline,
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    config_path = tmp_path / "batch.json"

    saved_config, saved_workflow = controller.save_config(
        config_path,
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
    )

    assert saved_config == config_path
    assert saved_workflow == tmp_path / BATCH_WORKFLOW_FILENAME
    assert controller.load_config(saved_config).workflow_sha256

    pipeline.nodes[output_id].params["tag"] = "changed-scientific-output"
    current_workflow[0] = serialize_workflow(pipeline)

    with pytest.raises(ValueError, match="different workflow"):
        controller.load_config(saved_config)


def test_controller_rejects_reserved_companion_filename(tmp_path):
    pipeline, _output_id = _explicit_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
    )

    with pytest.raises(ValueError, match="reserved"):
        controller.save_config(
            Path(tmp_path / BATCH_WORKFLOW_FILENAME),
            input_dir=tmp_path,
            output_dir=tmp_path / "outputs",
        )
