"""Persistence, relocation, and Save As preserve scientific input identity."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    scientific_workflow_hash,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_items import (
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceContainerMember,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
    SourceRevisionProof,
)
from napari_vipp.core.workflow import (
    deserialize_workflow,
    load_workflow,
    save_workflow,
    save_workflow_document,
    serialize_workflow,
)
from napari_vipp.core.workflow_paths import (
    relative_local_path,
    resolve_local_path,
    transform_workflow_paths,
)


def _item(path: Path) -> SourceItem:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return SourceItem(
        container=SourceContainerBundle(
            uri=str(path),
            format="npy",
            revision=SourceRevisionProof("file", digest, 1, len(content)),
            members=(SourceContainerMember(path.name, digest, len(content)),),
        ),
        selector=SourceItemSelector("array", "image"),
        reader=SourceReaderDescriptor("numpy", "numpy", "test"),
        capabilities=SourceCapabilities(),
        resolved=ResolvedSourceItemIdentity(
            "array", "Cells", "image", (2, 2), "uint16", ("Y", "X")
        ),
    )


def _project(root: Path) -> tuple[PrototypePipeline, Path, SourceItem]:
    data = root / "data"
    data.mkdir(parents=True)
    image = data / "cells.npy"
    image.write_bytes(b"exact source bytes")
    table = data / "measurements.json"
    table.write_bytes(b"exact measurement bytes")
    item = _item(image)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.set_param("input", "file_path", str(image))
    pipeline.nodes["input"].params["_vipp_source_item"] = item.to_dict()
    source = pipeline.add_node("table_source")
    pipeline.set_param(source.id, "dataset_path", str(table))
    pipeline.set_param(
        source.id, "dataset_sha256", hashlib.sha256(table.read_bytes()).hexdigest()
    )
    output = pipeline.add_node("save_output")
    pipeline.set_param(output.id, "path", str(root / "results" / "mask.tif"))
    workflow = root / "workflows" / "analysis.json"
    workflow.parent.mkdir()
    return pipeline, workflow, item


def _pipeline_from_loaded(loaded: dict) -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        loaded["nodes"], loaded["connections"], loaded["output_tunnels"]
    )
    return pipeline


def test_copy_project_and_open_from_unrelated_cwd_preserves_source_evidence(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    pipeline, workflow, item = _project(original)
    save_workflow(workflow, pipeline)
    raw = json.loads(workflow.read_text())
    saved = raw["nodes"][0]["params"]
    assert saved["file_path"] == "../data/cells.npy"
    assert saved["_vipp_source_item"]["container"]["uri"] == "../data/cells.npy"
    assert pipeline.nodes["input"].params["file_path"] == item.container.uri
    moved = tmp_path / "conference computer" / "VALIDATION"
    shutil.copytree(original, moved)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    loaded = load_workflow(moved / "workflows" / "analysis.json")
    params = loaded["nodes"][0].params
    moved_item = SourceItem.from_dict(params["_vipp_source_item"])
    expected = moved / "data" / "cells.npy"
    assert params["file_path"] == str(expected)
    assert moved_item == replace(
        item, container=replace(item.container, uri=str(expected))
    )
    assert hashlib.sha256(expected.read_bytes()).hexdigest() == (
        moved_item.container.revision.sha256
    )
    table = next(
        node for node in loaded["nodes"] if node.operation_id == "table_source"
    )
    assert table.params["dataset_path"] == str(moved / "data" / "measurements.json")
    assert (
        hashlib.sha256(Path(table.params["dataset_path"]).read_bytes()).hexdigest()
        == (table.params["dataset_sha256"])
    )
    output = next(
        node for node in loaded["nodes"] if node.operation_id == "save_output"
    )
    assert output.params["path"] == str(moved / "results" / "mask.tif")
    assert output.params["enabled"] == "off"


@pytest.mark.parametrize("raw_document", [False, True])
@pytest.mark.parametrize("relative_paths", [False, True])
def test_save_as_rebases_without_retargeting_or_mutating_sources(
    tmp_path, raw_document, relative_paths
):
    pipeline, original, item = _project(tmp_path / "project")
    save_workflow(original, pipeline)
    target = original.parent / "nested" / "saved-as.json"
    target.parent.mkdir()
    if raw_document:
        document = json.loads(original.read_text())
        before = deepcopy(document)
        save_workflow_document(
            target, document, source_path=original, relative_paths=relative_paths
        )
        assert document == before
    else:
        loaded = _pipeline_from_loaded(load_workflow(original))
        before = serialize_workflow(loaded)
        save_workflow(target, loaded, relative_paths=relative_paths)
        assert serialize_workflow(loaded) == before
    saved = json.loads(target.read_text())
    assert saved["nodes"][0]["params"]["file_path"] == (
        "../../data/cells.npy" if relative_paths else item.container.uri
    )
    reopened = load_workflow(target)["nodes"][0]
    assert reopened.params["file_path"] == item.container.uri
    assert SourceItem.from_dict(reopened.params["_vipp_source_item"]) == item


@pytest.mark.parametrize(
    ("value", "base", "expected"),
    [
        ("C:/project/data/cells.tif", "C:/project/workflows", "../data/cells.tif"),
        ("D:/data/cells.tif", "C:/project/workflows", "D:\\data\\cells.tif"),
        ("//server/share/data/a.tif", "//server/share/workflows", "../data/a.tif"),
        (
            "//server/other/data/a.tif",
            "//server/share/workflows",
            "\\\\server\\other\\data\\a.tif",
        ),
        ("/data/a.tif", "C:/project", "/data/a.tif"),
        ("C:/data/a.tif", "/project", "C:\\data\\a.tif"),
        ("/project/data/a.tif", "/project/workflows", "../data/a.tif"),
        ("", "/project", ""),
    ],
)
def test_relative_binding_handles_foreign_drives_and_shares(value, base, expected):
    assert relative_local_path(value, base) == expected


@pytest.mark.parametrize(
    "value",
    [
        "s3://bucket/a.zarr",
        "https://host/a.tif?x=1",
        "file:///C:/data/a.tif",
        "layer:current",
    ],
)
def test_uri_bindings_remain_opaque(value):
    assert resolve_local_path(value, "C:/project") == value
    assert relative_local_path(value, "C:/project") == value


def test_ambiguous_drive_relative_path_is_not_guessed():
    with pytest.raises(ValueError, match="Drive-relative.*ambiguous"):
        resolve_local_path("D:cells.tif", "C:/project")


def test_transform_changes_only_known_operational_fields():
    document = {
        "nodes": [
            {"operation_id": "input", "params": {"file_path": "data/a.tif"}},
            {
                "operation_id": "table_source",
                "params": {"dataset_path": "tables/a.json", "dataset_sha256": "a" * 64},
            },
            {"operation_id": "something_else", "params": {"path": "historical/a.tif"}},
        ],
        "metadata": {"provenance": {"path": "historical/a.tif"}},
        "batch_config": {"workflow": {"file": "separate.json"}},
    }
    before = deepcopy(document)
    result = transform_workflow_paths(document, "/project")
    assert document == before
    assert result["nodes"][0]["params"]["file_path"] == "/project/data/a.tif"
    assert result["nodes"][1]["params"]["dataset_path"] == "/project/tables/a.json"
    assert result["nodes"][1]["params"]["dataset_sha256"] == "a" * 64
    assert result["nodes"][2] == document["nodes"][2]
    assert result["metadata"] == document["metadata"]
    assert result["batch_config"] == document["batch_config"]


def test_attached_batch_hash_is_checked_before_rebase_and_updated_after(tmp_path):
    pipeline, path, item = _project(tmp_path / "project")
    output = pipeline.add_node("batch_output")
    pipeline.connect("input", output.id)
    document = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=path,
        workflow_sha256=scientific_workflow_hash(document),
        output_dir=path.parent.parent / "batch results",
        sources=(
            BatchSourceConfig(
                "input",
                "Cells",
                Path(item.container.uri).parent,
                "*.npy",
                source_items=(item,),
            ),
        ),
        outputs=(
            BatchOutputConfig(
                output.id, "Output", "result", "image", "npy", "", "{stem}"
            ),
        ),
    )
    document["batch_config"] = config.to_dict()
    save_workflow_document(path, document)
    raw = json.loads(path.read_text())
    assert raw["batch_config"]["workflow"] == {
        "file": path.name,
        "sha256": scientific_workflow_hash(raw),
    }
    loaded = load_workflow(path)
    runtime = serialize_workflow(_pipeline_from_loaded(loaded))
    assert loaded["batch_config"]["workflow"]["sha256"] == scientific_workflow_hash(
        runtime
    )
    assert loaded["batch_config"]["sources"][0]["source_items"][0] == item.to_dict()
    saved_as = path.parent / "nested" / "renamed.json"
    saved_as.parent.mkdir()
    save_workflow_document(saved_as, raw, source_path=path)
    saved_as_raw = json.loads(saved_as.read_text())
    assert saved_as_raw["batch_config"]["workflow"] == {
        "file": "renamed.json",
        "sha256": scientific_workflow_hash(saved_as_raw),
    }
    assert saved_as_raw["batch_config"]["output_dir"] == "../../batch results"
    assert saved_as_raw["batch_config"]["sources"][0]["input_dir"] == "../../data"
    assert (
        saved_as_raw["batch_config"]["sources"][0]["source_items"][0]["container"][
            "uri"
        ]
        == "../../data/cells.npy"
    )
    moved = tmp_path / "relocated"
    shutil.copytree(path.parent.parent, moved)
    moved_path = moved / "workflows" / "nested" / "renamed.json"
    moved_workflow = load_workflow(moved_path)
    moved_config = moved_workflow["batch_config"]
    assert moved_config["workflow"]["file"] == str(moved_path)
    assert moved_config["output_dir"] == str(moved / "batch results")
    assert moved_config["sources"][0]["input_dir"] == str(moved / "data")
    assert moved_config["sources"][0]["source_items"][0]["container"] == {
        **item.container.to_dict(),
        "uri": str(moved / "data" / "cells.npy"),
    }
    raw["nodes"][0]["params"]["file_path"] = "../different.npy"
    path.write_text(json.dumps(raw))
    invalid = load_workflow(path)
    assert "hash does not match" in invalid["batch_config_error"]
    assert invalid["batch_config"] == raw["batch_config"]
    with pytest.raises(ValueError, match="hash does not match"):
        save_workflow_document(path.with_name("bad.json"), raw, source_path=path)


@pytest.mark.parametrize("relative_paths", [False, True])
def test_save_live_relative_bindings_keeps_runtime_base_not_destination(
    tmp_path, monkeypatch, relative_paths
):
    project = tmp_path / "authoring-project"
    pipeline, _original, item = _project(project)
    relative = transform_workflow_paths(
        serialize_workflow(pipeline), project, relative=True
    )
    live = _pipeline_from_loaded(deserialize_workflow(relative))
    before = serialize_workflow(live)
    monkeypatch.chdir(project)
    target = project / "saved-elsewhere" / "nested" / "analysis.json"
    target.parent.mkdir(parents=True)
    save_workflow(target, live, relative_paths=relative_paths)
    restored = load_workflow(target)
    nodes = {node.operation_id: node for node in restored["nodes"]}
    assert Path(nodes["input"].params["file_path"]) == project / "data" / "cells.npy"
    assert nodes["input"].params["_vipp_source_item"] == item.to_dict()
    assert Path(nodes["table_source"].params["dataset_path"]) == (
        project / "data" / "measurements.json"
    )
    assert (
        nodes["table_source"].params["dataset_sha256"]
        == (pipeline.nodes[nodes["table_source"].id].params["dataset_sha256"])
    )
    assert Path(nodes["save_output"].params["path"]) == (
        project / "results" / "mask.tif"
    )
    assert serialize_workflow(live) == before


def test_open_relative_then_save_absolute_can_move_only_workflow(tmp_path, monkeypatch):
    project = tmp_path / "project"
    pipeline, original, item = _project(project)
    save_workflow(original, pipeline)
    loaded = _pipeline_from_loaded(load_workflow(original))
    before = serialize_workflow(loaded)
    save_workflow(original, loaded, relative_paths=False)
    raw = json.loads(original.read_text())
    nodes = {node["operation_id"]: node for node in raw["nodes"]}
    assert nodes["input"]["params"]["file_path"] == item.container.uri
    assert nodes["input"]["params"]["_vipp_source_item"] == item.to_dict()
    assert Path(nodes["table_source"]["params"]["dataset_path"]) == (
        project / "data" / "measurements.json"
    )
    assert Path(nodes["save_output"]["params"]["path"]) == (
        project / "results" / "mask.tif"
    )
    assert serialize_workflow(loaded) == before
    moved = tmp_path / "workflow-only" / "moved.json"
    moved.parent.mkdir()
    shutil.move(original, moved)
    unrelated = tmp_path / "unrelated-working-folder"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    reopened = _pipeline_from_loaded(load_workflow(moved))
    assert serialize_workflow(reopened) == before
    assert Path(reopened.nodes["input"].params["file_path"]).read_bytes() == (
        b"exact source bytes"
    )


def test_raw_relative_to_absolute_save_uses_original_base_and_round_trips(
    tmp_path, monkeypatch
):
    pipeline, original, item = _project(tmp_path / "project")
    save_workflow(original, pipeline)
    document = json.loads(original.read_text())
    before = deepcopy(document)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute = elsewhere / "absolute.json"
    save_workflow_document(
        absolute, document, source_path=original, relative_paths=False
    )
    assert document == before
    raw_absolute = json.loads(absolute.read_text())
    assert raw_absolute["nodes"][0]["params"]["file_path"] == item.container.uri
    assert raw_absolute["nodes"][0]["params"]["_vipp_source_item"] == item.to_dict()
    relative_again = original.parent / "relative-again.json"
    save_workflow_document(
        relative_again, raw_absolute, source_path=absolute, relative_paths=True
    )
    assert json.loads(relative_again.read_text()) == before
    assert (
        _pipeline_from_loaded(load_workflow(relative_again))
        .nodes["input"]
        .params["_vipp_source_item"]
        == item.to_dict()
    )


def test_absolute_save_retains_foreign_drives_shares_and_opaque_uris(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.set_param("input", "file_path", "D:/fixed-data/cells.tif")
    table = pipeline.add_node("table_source")
    pipeline.set_param(table.id, "dataset_path", "//server/share/measurements.json")
    pipeline.set_param(table.id, "dataset_sha256", "a" * 64)
    output = pipeline.add_node("save_output")
    pipeline.set_param(output.id, "path", "C:/results/mask.tif")
    target = tmp_path / "absolute.json"
    save_workflow(target, pipeline, relative_paths=False)
    params = {
        node["operation_id"]: node["params"]
        for node in json.loads(target.read_text())["nodes"]
    }
    assert params["input"]["file_path"] == "D:\\fixed-data\\cells.tif"
    assert params["table_source"]["dataset_path"] == (
        "\\\\server\\share\\measurements.json"
    )
    assert params["save_output"]["path"] == "C:\\results\\mask.tif"
    uri = "s3://bucket/cells.zarr?revision=exact"
    pipeline.set_param("input", "file_path", uri)
    save_workflow(target, pipeline, relative_paths=False)
    assert json.loads(target.read_text())["nodes"][0]["params"]["file_path"] == uri


def test_absolute_attached_batch_save_resolves_old_bindings_and_keeps_evidence(
    tmp_path,
):
    pipeline, original, item = _project(tmp_path / "project")
    output = pipeline.add_node("batch_output")
    pipeline.connect("input", output.id)
    document = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=original,
        workflow_sha256=scientific_workflow_hash(document),
        output_dir=original.parent.parent / "batch results",
        sources=(
            BatchSourceConfig(
                "input",
                "Cells",
                Path(item.container.uri).parent,
                "*.npy",
                source_items=(item,),
            ),
        ),
        outputs=(
            BatchOutputConfig(
                output.id, "Output", "result", "image", "npy", "", "{stem}"
            ),
        ),
    )
    document["batch_config"] = config.to_dict()
    save_workflow_document(original, document)
    raw_relative = json.loads(original.read_text())
    before = deepcopy(raw_relative)
    target = tmp_path / "workflow-only" / "absolute.json"
    target.parent.mkdir()
    save_workflow_document(
        target, raw_relative, source_path=original, relative_paths=False
    )
    assert raw_relative == before
    saved = json.loads(target.read_text())
    batch = saved["batch_config"]
    assert batch["workflow"] == {
        "file": str(target),
        "sha256": scientific_workflow_hash(saved),
    }
    assert batch["output_dir"] == str(config.output_dir)
    assert batch["sources"][0]["input_dir"] == str(config.sources[0].input_dir)
    assert batch["sources"][0]["source_items"][0] == item.to_dict()
    assert batch["outputs"] == raw_relative["batch_config"]["outputs"]
    moved = target.with_name("moved.json")
    target.rename(moved)
    loaded = load_workflow(moved)
    assert "batch_config_error" not in loaded
    assert loaded["batch_config"]["workflow"]["file"] == str(moved)
    assert loaded["batch_config"]["sources"][0]["source_items"][0] == item.to_dict()
    loaded_pipeline = _pipeline_from_loaded(loaded)
    assert loaded["batch_config"]["workflow"]["sha256"] == (
        scientific_workflow_hash(serialize_workflow(loaded_pipeline))
    )
    invalid = deepcopy(raw_relative)
    invalid["nodes"][0]["params"]["file_path"] = "../different.npy"
    old_bytes = moved.read_bytes()
    with pytest.raises(ValueError, match="hash does not match"):
        save_workflow_document(
            moved, invalid, source_path=original, relative_paths=False
        )
    assert moved.read_bytes() == old_bytes


@pytest.mark.parametrize("relative_paths", [None, 0, 1, "absolute", "relative"])
@pytest.mark.parametrize("raw_document", [False, True])
def test_save_rejects_invalid_path_policy_before_overwrite(
    tmp_path, relative_paths, raw_document
):
    pipeline = PrototypePipeline()
    target = tmp_path / "existing.json"
    target.write_bytes(b"original destination bytes")
    with pytest.raises(ValueError, match="relative_paths must be a boolean"):
        if raw_document:
            save_workflow_document(
                target, serialize_workflow(pipeline), relative_paths=relative_paths
            )
        else:
            save_workflow(target, pipeline, relative_paths=relative_paths)
    assert target.read_bytes() == b"original destination bytes"
