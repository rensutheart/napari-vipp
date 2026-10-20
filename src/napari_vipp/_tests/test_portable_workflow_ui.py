"""Save As keeps data bindings while a copied project changes its location."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QMessageBox

from napari_vipp._tests.test_batch import _batch_config, _batch_workflow
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import (
    BatchConfig,
    build_batch_plan,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.measurement_collection import (
    CollectionItem,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
    save_measurement_collection,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.source_item_persistence import SOURCE_ITEM_PARAMETER
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import (
    load_workflow,
    save_workflow,
    save_workflow_document,
    serialize_workflow,
)


def _save_as(widget, monkeypatch, target):
    monkeypatch.setattr(
        "napari_vipp._widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        "napari_vipp._widget.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    assert widget._save_workflow_dialog(force_choose_path=True)
    assert widget._workflow_tabs.current.path == target.resolve()


def test_gui_save_as_rebases_image_and_explicit_batch_then_copied_batch_runs(
    qtbot, monkeypatch, tmp_path,
):
    project = tmp_path / "original project"
    inputs = project / "data"
    inputs.mkdir(parents=True)
    data = np.arange(30, dtype=np.uint16).reshape(5, 6)
    image = inputs / "field.npy"
    np.save(image, data)
    workflow, output_ids = _batch_workflow()
    source = next(node for node in workflow["nodes"] if node["id"] == "input")
    source["params"].update(source_mode="file path", file_path=str(image))
    config = _batch_config(workflow, inputs, project / "outputs", output_ids)
    source_item = build_batch_plan(config).items[0].source_items["input"]
    source["params"][SOURCE_ITEM_PARAMETER] = source_item.to_dict()
    config = replace(
        config,
        workflow_sha256=scientific_workflow_hash(workflow),
        sources=(replace(config.sources[0], source_items=(source_item,)),),
    )
    workflow["batch_config"] = config.to_dict()
    first = project / "workflows" / "analysis.json"
    save_workflow_document(first, workflow)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    # This test exercises the real open/save UI and batch restoration. Pixel
    # execution is checked below by the actual headless batch engine.
    monkeypatch.setattr(widget, "run_pipeline", lambda *_args, **_kwargs: None)
    widget.load_workflow_file(first)
    qtbot.waitUntil(lambda: not widget._batch_workspace_preview_workers, timeout=10_000)
    assert widget._active_collection_batch_dialog is not None
    assert Path(widget.pipeline.nodes["input"].params["file_path"]) == image
    original_bytes = first.read_bytes()
    second = project / "conference" / "nested" / "renamed.json"
    _save_as(widget, monkeypatch, second)
    assert first.read_bytes() == original_bytes
    document = json.loads(second.read_text(encoding="utf-8"))
    saved_source = next(node for node in document["nodes"] if node["id"] == "input")
    assert saved_source["params"]["file_path"] == "../../data/field.npy"
    assert saved_source["params"][SOURCE_ITEM_PARAMETER]["container"]["uri"] == (
        "../../data/field.npy"
    )
    assert document["batch_config"]["workflow"]["file"] == second.name
    assert document["batch_config"]["workflow"]["sha256"] == (
        scientific_workflow_hash(document)
    )

    copied = tmp_path / "conference copy"
    shutil.copytree(project, copied)
    image.unlink()  # The original location cannot accidentally satisfy the run.
    monkeypatch.chdir(tmp_path)
    copied_workflow = copied / second.relative_to(project)
    restored = load_workflow(copied_workflow)
    graph = PrototypePipeline()
    graph.restore_graph(restored["nodes"], restored["connections"])
    runtime_document = serialize_workflow(
        graph, compute_request=restored["compute_request"],
    )
    runtime_config = BatchConfig.from_dict(
        restored["batch_config"], base_dir=copied_workflow.parent,
    )
    assert runtime_config.resolve_path(runtime_config.workflow_file) == copied_workflow
    result = run_batch(runtime_document, runtime_config, workflow_path=copied_workflow)
    assert result.summary["completed"] == 1
    assert not result.has_failures
    for output in result.saved_paths:
        assert output.is_relative_to(copied)
        np.testing.assert_array_equal(np.load(output), data)


def test_gui_save_as_rebases_verified_table_then_moved_results_execute(
    qtbot, monkeypatch, tmp_path,
):
    project = tmp_path / "results project"
    project.mkdir()
    table = TableData(("label_id", "area"), ((1, 3.5), (2, 6.5)))
    collection = collect_measurements(MeasurementPreview(
        MeasurementOutput("output", "Objects", "objects"),
        (CollectionItem("item-a", 1, "image-a", "ready", row_count=2,
                        result_sha256="c" * 64, table=table),),
        "run-a", "a" * 64, "b" * 64,
    ))
    dataset = project / "measurements.vipp-results.json"
    save_measurement_collection(collection, dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    graph = PrototypePipeline()
    graph.restore_graph([], [])
    source = graph.add_node("table_source")
    graph.set_param(source.id, "dataset_path", str(dataset))
    graph.set_param(source.id, "dataset_sha256", digest)
    first = project / "first.json"
    save_workflow(first, graph)
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *_args, **_kwargs: None)
    widget.load_workflow_file(first)
    second = project / "new location" / "results.json"
    _save_as(widget, monkeypatch, second)
    raw = json.loads(second.read_text(encoding="utf-8"))
    assert raw["nodes"][0]["params"]["dataset_path"] == (
        "../measurements.vipp-results.json"
    )
    assert raw["nodes"][0]["params"]["dataset_sha256"] == digest
    copied = tmp_path / "copied results"
    shutil.copytree(project, copied)
    dataset.unlink()
    monkeypatch.chdir(tmp_path)
    loaded = load_workflow(copied / second.relative_to(project))
    moved = PrototypePipeline()
    moved.restore_graph(loaded["nodes"], loaded["connections"])
    outputs = moved.run(None)
    assert outputs[source.id] == collection.table


def test_gui_opens_graph_but_does_not_accept_stale_batch_hash_after_rebasing(
    qtbot, monkeypatch, tmp_path,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    image = inputs / "field.npy"
    np.save(image, np.ones((3, 4), dtype=np.uint16))
    document, output_ids = _batch_workflow()
    source = next(node for node in document["nodes"] if node["id"] == "input")
    source["params"].update(source_mode="file path", file_path=str(image))
    runtime_hash = scientific_workflow_hash(document)
    config = _batch_config(document, inputs, tmp_path / "outputs", output_ids)
    document["batch_config"] = config.to_dict()
    path = save_workflow_document(tmp_path / "workflow.json", document)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert scientific_workflow_hash(saved) != runtime_hash
    # Deliberately wrong for the saved relative graph, but equal to the graph
    # after loading. Validation must happen before path resolution.
    saved["batch_config"]["workflow"]["sha256"] = runtime_hash
    path.write_text(json.dumps(saved), encoding="utf-8")
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *_args, **_kwargs: None)
    widget.load_workflow_file(path)
    assert Path(widget.pipeline.nodes["input"].params["file_path"]) == image
    assert widget._active_collection_batch_dialog is None
    assert "could not be restored" in widget._last_workflow_load_detail
    assert "hash" in widget._last_workflow_load_detail.lower()
