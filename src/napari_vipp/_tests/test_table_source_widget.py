from __future__ import annotations

from dataclasses import replace

import pytest
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.measurement_collection import (
    CollectionItem,
    MeasurementOutput,
    MeasurementPreview,
    collect_measurements,
    save_measurement_collection,
)
from napari_vipp.core.operations import merge_tables, summarize_measurements
from napari_vipp.core.table_source import load_table_source
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import load_workflow, save_workflow


def _dataset(tmp_path):
    table = TableData(
        ("label_id", "volume"),
        ((1, 2.5), (2, 4.0)),
        name="Objects",
        table_kind="object measurements",
        column_units=(("volume", "µm³"),),
    )
    first = CollectionItem(
        "item-1",
        1,
        "image-a",
        "ready",
        row_count=2,
        source_path="original-a.tif",
        result_path="original-a.csv",
        result_sha256="a" * 64,
        table=table,
    )
    second = replace(first, key="item-2", index=2, batch_id="image-b")
    preview = MeasurementPreview(
        MeasurementOutput("output", "Objects", "objects"),
        (first, second),
        "run-1",
        "b" * 64,
        "c" * 64,
    )
    collection = collect_measurements(preview)
    path = tmp_path / "collected.vipp-results.json"
    save_measurement_collection(collection, path)
    return path, collection


def _widget(qtbot, monkeypatch):
    widget = VippWidget(ViewerModel())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    return widget


def test_open_collection_keeps_original_tab_and_never_saves_over_dataset(
    qtbot, monkeypatch, tmp_path
):
    path, collection = _dataset(tmp_path)
    original = path.read_bytes()
    widget = _widget(qtbot, monkeypatch)
    first = widget._workflow_tabs.current
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda **kw: calls.append(True))
    widget._open_workflow_path(path)
    qtbot.waitUntil(lambda: len(widget._workflow_tabs) == 2, timeout=10000)
    assert first in tuple(widget._workflow_tabs)
    assert widget._workflow_tabs.current.path is None
    node = next(iter(widget.pipeline.nodes.values()))
    assert node.operation_id == "table_source"
    assert len(node.params["dataset_sha256"]) == 64
    payloads, layers = widget._source_payloads_for_pipeline()
    assert payloads[node.id].data == collection.table
    assert layers == []
    assert calls
    workflow = tmp_path / "results-workflow.json"
    save_workflow(workflow, widget.pipeline)
    assert path.read_bytes() == original
    saved = workflow.read_text(encoding="utf-8")
    assert "dataset_sha256" in saved
    assert '"rows"' not in saved
    assert load_workflow(workflow)["nodes"][0].params == node.params


def test_table_source_calculates_in_widget_without_images(qtbot, monkeypatch, tmp_path):
    path, collection = _dataset(tmp_path)
    widget = _widget(qtbot, monkeypatch)
    widget._open_workflow_path(path)
    qtbot.waitUntil(lambda: len(widget._workflow_tabs) == 2, timeout=10000)
    node_id = next(iter(widget.pipeline.nodes))
    qtbot.waitUntil(
        lambda: (
            isinstance(widget.pipeline.outputs.get(node_id), TableData)
            or "error" in widget.status_label.text().lower()
        ),
        timeout=30000,
    )
    assert widget.pipeline.outputs[node_id] == collection.table, (
        widget.status_label.text()
    )
    qtbot.waitUntil(lambda: widget._active_pipeline_run_id is None, timeout=10000)


def test_invalid_collection_does_not_create_tab(qtbot, monkeypatch, tmp_path):
    path = tmp_path / "bad.vipp-results.json"
    path.write_text('{"not": "a dataset"}', encoding="utf-8")
    widget = _widget(qtbot, monkeypatch)
    first = widget._workflow_tabs.current
    widget._open_workflow_path(path)
    qtbot.waitUntil(lambda: not widget._table_sources._workers, timeout=10000)
    assert len(widget._workflow_tabs) == 1
    assert widget._workflow_tabs.current is first
    assert "Could not open measurement dataset" in widget.status_label.text()


def test_collected_item_identity_prevents_cross_image_joins(tmp_path):
    _, collection = _dataset(tmp_path)
    table = collection.table
    # Same local label IDs in two images must produce four matches, not eight.
    merged = merge_tables((table, table))
    assert merged.row_count == 4
    summary = summarize_measurements(table, group_by="")
    assert not any(column.startswith("_vipp_row_") for column in summary.columns)


def test_unbound_saved_reference_requires_explicit_file_choice(
    qtbot, monkeypatch, tmp_path
):
    path, _ = _dataset(tmp_path)
    widget = _widget(qtbot, monkeypatch)
    node = widget.pipeline.add_node("table_source")
    node.params.update(dataset_path=str(path), dataset_sha256="")
    assert not widget._table_sources.pending()
    assert not widget._table_sources._workers
    with pytest.raises(ValueError, match="no recorded dataset fingerprint"):
        widget._table_sources.resolve(node)
    assert node.params["dataset_sha256"] == ""


def test_dataset_load_does_not_publish_after_node_reference_changed(
    qtbot, monkeypatch, tmp_path
):
    from napari_vipp.ui.table_sources import _Load

    path, _ = _dataset(tmp_path)
    payload = load_table_source(path)
    widget = _widget(qtbot, monkeypatch)
    node = widget.pipeline.add_node("table_source")
    node.params.update(
        dataset_path=str(path), dataset_sha256=payload.revision_token.file_sha256
    )
    key = widget._table_sources._key(node)
    context = (widget._workflow_tabs.current.session_id, node.id, key)
    worker = _Load(context, str(path), key[1], context)
    widget._table_sources._workers[context] = worker
    node.params["dataset_path"] = str(tmp_path / "another.vipp-results.json")
    widget._table_sources._finished((worker, payload, ""))
    assert key not in widget._table_sources._cache
    assert node.params["dataset_path"].endswith("another.vipp-results.json")


def test_table_source_cache_normalizes_forward_slash_paths(
    qtbot, monkeypatch, tmp_path
):
    from napari_vipp.ui.table_sources import _Load

    path, _ = _dataset(tmp_path)
    payload = load_table_source(path)
    widget = _widget(qtbot, monkeypatch)
    monkeypatch.setattr(widget, "run_pipeline", lambda **kw: None)
    node = widget.pipeline.add_node("table_source")
    node.params.update(
        dataset_path=path.as_posix(), dataset_sha256=payload.revision_token.file_sha256
    )
    key = widget._table_sources._key(node)
    context = (widget._workflow_tabs.current.session_id, node.id, key)
    worker = _Load(context, key[0], key[1], context)
    widget._table_sources._workers[context] = worker
    widget._table_sources._finished((worker, payload, ""))
    widget._table_sources._prune()
    assert widget._table_sources.resolve(node) is payload


def test_reopening_saved_results_workflow_checks_dataset_hash(
    qtbot, monkeypatch, tmp_path
):
    from napari_vipp.core.pipeline import PrototypePipeline

    path, _ = _dataset(tmp_path)
    payload = load_table_source(path)
    pipeline = PrototypePipeline()
    pipeline.restore_graph([], [])
    node = pipeline.add_node("table_source")
    node.params.update(
        dataset_path=str(path), dataset_sha256=payload.revision_token.file_sha256
    )
    workflow = tmp_path / "results-workflow.json"
    save_workflow(workflow, pipeline)
    # Different encoding is not the bound revision, even with identical table values.
    path.write_bytes(path.read_bytes() + b"\n")
    widget = _widget(qtbot, monkeypatch)
    widget.load_workflow_file(workflow)
    qtbot.waitUntil(lambda: not widget._table_sources._workers, timeout=10000)
    assert "hash does not match" in widget.status_label.text()
    assert widget.pipeline.outputs[node.id] is None
    assert widget.pipeline.nodes[node.id].params["dataset_sha256"] == (
        payload.revision_token.file_sha256
    )
