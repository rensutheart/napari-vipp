from __future__ import annotations

from pathlib import Path

import pytest

from napari_vipp.core.batch import ExistingFilePolicy, scientific_workflow_hash
from napari_vipp.core.batch_setup import (
    batch_output_node_ids,
    batch_saved_node_ids,
    batch_source_rows,
    build_collection_batch_config,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


def _explicit_batch_pipeline() -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    output = pipeline.add_node("batch_output")
    output.params.update(
        tag="segmentation labels",
        format="npy",
        subfolder="labels",
        filename_template="{batch_id}__{tag}",
        overwrite="yes",
    )
    assert pipeline.connect("input", output.id).success
    return pipeline, output.id


def test_build_collection_batch_config_maps_explicit_sources_and_outputs(tmp_path):
    pipeline, output_id = _explicit_batch_pipeline()
    second = pipeline.add_node("input")
    second.title = "Reference source"
    second.params["binding_mode"] = "collection"
    workflow = serialize_workflow(pipeline)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    config = build_collection_batch_config(
        workflow,
        input_dir="",
        output_dir=tmp_path / "outputs",
        pattern="*.tif",
        image_format="npy",
        save_python_script=False,
        source_bindings=[
            {
                "node_id": "input",
                "title": "Primary source",
                "input_dir": first_dir,
                "pattern": "*.npy",
            },
            {
                "node_id": second.id,
                "title": "Reference source",
                "input_dir": second_dir,
                "pattern": "*.ome.tif",
            },
        ],
        existing_file_policy=ExistingFilePolicy.SKIP.value,
        continue_on_error=False,
    )

    assert [source.node_id for source in config.sources] == ["input", second.id]
    assert [source.input_dir for source in config.sources] == [
        first_dir.resolve(),
        second_dir.resolve(),
    ]
    assert [source.pattern for source in config.sources] == ["*.npy", "*.ome.tif"]
    assert len(config.outputs) == 1
    output = config.outputs[0]
    assert output.node_id == output_id
    assert output.tag == "segmentation_labels"
    assert output.kind == "image"
    assert output.format == "npy"
    assert output.subfolder == "labels"
    assert output.filename_template == "{batch_id}__{tag}"
    assert output.overwrite == "yes"
    assert config.existing_file_policy is ExistingFilePolicy.SKIP
    assert not config.save_python_script
    assert not config.continue_on_error
    assert config.workflow_sha256 == scientific_workflow_hash(workflow)


def test_build_collection_batch_config_uses_collection_source_and_terminal_fallback(
    tmp_path,
):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    workflow = serialize_workflow(pipeline)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    config = build_collection_batch_config(
        workflow,
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
        pattern="*.npy",
    )

    assert [source.node_id for source in config.sources] == ["input"]
    assert config.sources[0].input_dir == input_dir.resolve()
    assert config.sources[0].pattern == "*.npy"
    assert [output.node_id for output in config.outputs] == ["threshold"]
    assert config.outputs[0].tag == "Otsu_Threshold-threshold"
    assert batch_output_node_ids(pipeline) == []
    assert batch_saved_node_ids(pipeline) == ["threshold"]


def test_batch_source_rows_follow_topological_source_order():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    second = pipeline.add_node("input")
    second.title = "Second source"

    assert batch_source_rows(pipeline) == [
        {
            "node_id": "input",
            "title": pipeline.nodes["input"].title,
            "binding_mode": "collection",
        },
        {
            "node_id": second.id,
            "title": "Second source",
            "binding_mode": "single item",
        },
    ]


def test_default_collection_bindings_preserve_topological_source_order(tmp_path):
    pipeline, _output_id = _explicit_batch_pipeline()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    second = pipeline.add_node("input")
    second.params["binding_mode"] = "collection"
    workflow = serialize_workflow(pipeline)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    config = build_collection_batch_config(
        workflow,
        input_dir=input_dir,
        output_dir=tmp_path / "outputs",
    )

    assert [source.node_id for source in config.sources] == ["input", second.id]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"output_dir": ""}, "(?i)output folder cannot be blank"),
        ({"source_bindings": []}, "(?i)at least one batch source"),
        ({"existing_file_policy": "invented"}, "(?i)unsupported existing-file"),
    ],
)
def test_build_collection_batch_config_rejects_invalid_form_values(
    tmp_path,
    changes,
    message,
):
    pipeline, _output_id = _explicit_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    values = {
        "input_dir": input_dir,
        "output_dir": tmp_path / "outputs",
        "source_bindings": None,
        "existing_file_policy": ExistingFilePolicy.ERROR.value,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        build_collection_batch_config(workflow, **values)


def test_source_binding_paths_are_resolved(tmp_path, monkeypatch):
    pipeline, _output_id = _explicit_batch_pipeline()
    workflow = serialize_workflow(pipeline)
    source_dir = tmp_path / "relative-source"
    source_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    config = build_collection_batch_config(
        workflow,
        input_dir="unused",
        output_dir="relative-output",
        source_bindings=[
            {
                "node_id": "input",
                "input_dir": Path("relative-source"),
                "pattern": "*.tif",
            }
        ],
    )

    assert config.sources[0].input_dir == source_dir.resolve()
    assert config.output_dir == (tmp_path / "relative-output").resolve()
