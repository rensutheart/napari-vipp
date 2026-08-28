from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_vipp.core.batch import (
    BATCH_WORKFLOW_FILENAME,
    BatchNodeExecutionOverride,
    BatchScientificPreflightError,
    build_batch_plan,
)
from napari_vipp.core.compute import ComputeMode, ComputeRequest
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


def _axis_sensitive_batch_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    background = pipeline.add_node("subtract_background")
    pipeline.set_param(background.id, "radius", 1.0)
    pipeline.set_param(background.id, "spatial_mode", "3D ZYX")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, output.id).success
    return pipeline


def _bypassable_batch_pipeline() -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    crop = pipeline.add_node("crop_stack")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect("input", crop.id).success
    assert pipeline.connect(crop.id, output.id).success
    return pipeline, crop.id


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
    assert len(preview.items) == 2
    assert [item.index for item in preview.items] == [1, 2]
    assert preview.rows[0].batch_id == preview.items[0].batch_id
    assert [item.source_paths["input"].name for item in preview.items] == [
        "field_a.npy",
        "field_b.npy",
    ]
    assert preview.config.sources[0].node_id == "input"
    assert preview.config.resolve_path(preview.config.sources[0].input_dir) == input_dir
    assert preview.config.resolve_path(preview.config.output_dir) == (
        tmp_path / "outputs"
    )
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


def test_attached_preview_normalization_preserves_whole_batch_node_behavior(
    tmp_path,
):
    pipeline, crop_id = _bypassable_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    config = controller.build_config(
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
        node_execution_overrides=(BatchNodeExecutionOverride(crop_id, "bypass"),),
    )

    prepared = controller.prepare_attached_config_preview(config)

    assert prepared.config.node_execution_overrides == (
        BatchNodeExecutionOverride(crop_id, "bypass"),
    )


def test_controller_labels_each_series_from_a_collection_container(tmp_path):
    pipeline, _output_id = _explicit_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.savez(
        input_dir / "plate.ims-export.npz",
        field_a=np.ones((4, 5), dtype=np.uint8),
        field_b=np.full((4, 5), 2, dtype=np.uint8),
    )

    preview = controller.preview(
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
        pattern="*.npz",
        image_format="npy",
    )

    assert preview.total_items == 2
    assert [row.source_labels["input"] for row in preview.rows] == [
        "plate.ims-export.npz › field_a",
        "plate.ims-export.npz › field_b",
    ]


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
    source_path = input_dir / "field.npy"
    np.save(source_path, np.ones((3, 4), dtype=np.uint8))
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
    loaded = controller.load_config(saved_config)
    assert loaded.workflow_sha256
    assert [item.selector.key for item in loaded.sources[0].source_items] == ["field"]
    np.save(source_path, np.full((3, 4), 2, dtype=np.uint8))
    with pytest.raises(ValueError, match="no longer matches.*SourceItem"):
        build_batch_plan(loaded)

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


def test_controller_preserves_full_machine_execution_request(tmp_path):
    pipeline, _output_id = _explicit_batch_pipeline()
    # Workflow serialization intentionally keeps only portable authored intent;
    # the controller must separately capture machine-local batch execution caps.
    request = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        accelerator_memory_cap_bytes=4_000_000_000,
        accelerator_safety_reserve_bytes=500_000_000,
        allow_experimental=True,
    )
    workflow = serialize_workflow(pipeline, compute_request=request)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
        compute_request_provider=lambda: request,
    )

    config = controller.build_config(
        input_dir=tmp_path,
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
        image_format="npy",
    )

    assert config.compute_request == request
    assert config.compute_request.runtime_id == "cuda-cupy"
    assert config.compute_request.device_id == "cuda:0"
    assert config.compute_request.accelerator_memory_cap_bytes == 4_000_000_000
    assert config.compute_request.accelerator_safety_reserve_bytes == 500_000_000
    assert config.compute_request.allow_experimental is True


def test_controller_preview_requires_reviewed_axes_for_generic_tiff_stack(tmp_path):
    pipeline = _axis_sensitive_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    controller = CollectionBatchController(
        workflow_document_provider=lambda: workflow,
        pipeline_provider=lambda: pipeline,
    )
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    source_path = input_dir / "ordinary-stack.tif"
    tifffile.imwrite(
        source_path,
        np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9),
        photometric="minisblack",
    )

    values = {
        "input_dir": input_dir,
        "output_dir": tmp_path / "outputs",
        "pattern": "*.tif",
        "image_format": "npy",
    }
    with pytest.raises(
        BatchScientificPreflightError,
        match=r"raw QYX, effective QYX",
    ):
        controller.preview(**values)

    assert not values["output_dir"].exists()
    preview = controller.preview(
        **values,
        source_bindings=[
            {
                "node_id": "input",
                "title": "Image Source",
                "input_dir": input_dir,
                "pattern": "*.tif",
                "axis_declaration": "QYX -> ZYX",
            }
        ],
    )

    assert preview.total_items == 1
    declaration = preview.config.sources[0].axis_declaration
    assert declaration is not None
    assert declaration.display_text == "QYX -> ZYX"
    assert not values["output_dir"].exists()
