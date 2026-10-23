"""The Settings path preference changes saved bindings, not data or execution."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from napari_vipp import _widget as widget_module
from napari_vipp._tests.test_widget import (
    _configure_workflow_attached_batch,
    _Viewer,
)
from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import BatchConfig, scientific_workflow_hash
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow, save_workflow, serialize_workflow
from napari_vipp.ui import workflow_save_settings
from napari_vipp.ui.workflow_save_settings import (
    DEFAULT_WORKFLOW_PATH_POLICY,
    WORKFLOW_PATH_POLICY_CHOICES,
    WORKFLOW_PATH_POLICY_SETTING,
    WorkflowPathPolicy,
    WorkflowSavePolicy,
    load_workflow_path_policy,
)


@pytest.fixture
def path_widget_factory(qtbot, monkeypatch):
    """Exercise real document IO without allowing load-triggered pixel work."""
    widgets = []

    def make():
        widget = VippWidget(
            _Viewer(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
        )
        qtbot.addWidget(widget)
        runs = []
        monkeypatch.setattr(widget, "run_pipeline", lambda: runs.append(True))
        widgets.append(widget)
        return widget, runs

    yield make
    for widget in widgets:
        widget._discard_incomplete_startup_on_close = True


def _event_turn(qtbot):
    finished = []
    QTimer.singleShot(0, lambda: finished.append(True))
    qtbot.waitUntil(lambda: bool(finished), timeout=3000)


def _select_path_policy(widget, policy):
    index = widget.workflow_path_policy_combo.findData(policy.value)
    assert index >= 0
    widget.workflow_path_policy_combo.setCurrentIndex(index)
    assert widget._workflow_path_policy is policy
    assert load_workflow_path_policy() is policy


def _params(document, node_id):
    return next(node["params"] for node in document["nodes"] if node["id"] == node_id)


def _loaded_params(document, node_id):
    return next(node.params for node in document["nodes"] if node.id == node_id)


def _relative_recipe(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    data_dir = project / "data"
    data_dir.mkdir()
    source = data_dir / "source.npy"
    np.save(source, np.arange(24, dtype=np.uint16).reshape(4, 6))
    output = project / "results" / "processed.npy"
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params.update(
        source_mode="file path", file_path=str(source)
    )
    writer = pipeline.add_node("save_output")
    writer.params.update(enabled="on", path=str(output), format="npy")
    assert pipeline.connect("input", writer.id).success
    path = save_workflow(project / "analysis.json", pipeline)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert not Path(_params(document, "input")["file_path"]).is_absolute()
    assert not Path(_params(document, writer.id)["path"]).is_absolute()
    return path, source, output, writer.id


def test_settings_offers_relative_default_and_an_explicit_absolute_choice(
    path_widget_factory,
):
    widget, _runs = path_widget_factory()

    assert DEFAULT_WORKFLOW_PATH_POLICY is WorkflowPathPolicy.RELATIVE
    assert WORKFLOW_PATH_POLICY_CHOICES == (
        ("relative", "Relative to workflow file"),
        ("absolute", "Absolute file locations"),
    )
    assert widget._workflow_path_policy is WorkflowPathPolicy.RELATIVE
    assert widget.workflow_path_policy_combo.currentData() == "relative"
    widget._populate_settings_toolbar_menu()
    submenu = next(
        action.menu()
        for action in widget.settings_menu.actions()
        if action.text() == "Workflow paths"
    )
    assert submenu is not None
    assert [
        widget.workflow_path_policy_combo.itemText(index)
        for index in range(widget.workflow_path_policy_combo.count())
    ] == [label for _value, label in WORKFLOW_PATH_POLICY_CHOICES]
    assert [action.text() for action in submenu.actions()] == [
        label for _value, label in WORKFLOW_PATH_POLICY_CHOICES
    ]
    assert [action.isChecked() for action in submenu.actions()] == [True, False]
    assert all(action.isCheckable() for action in submenu.actions())

    submenu.actions()[1].trigger()

    assert widget.workflow_path_policy_combo.currentData() == "absolute"
    assert widget._workflow_path_policy is WorkflowPathPolicy.ABSOLUTE
    assert load_workflow_path_policy() is WorkflowPathPolicy.ABSOLUTE
    widget._populate_settings_toolbar_menu()
    reopened = next(
        action.menu()
        for action in widget.settings_menu.actions()
        if action.text() == "Workflow paths"
    )
    assert [action.isChecked() for action in reopened.actions()] == [False, True]


@pytest.mark.parametrize("invalid", [None, "", "unrecognized", 123])
def test_invalid_saved_path_setting_falls_back_without_overriding_it(
    path_widget_factory, invalid
):
    settings = workflow_save_settings._settings()
    settings.setValue(WORKFLOW_PATH_POLICY_SETTING, invalid)

    widget, _runs = path_widget_factory()

    assert widget._workflow_path_policy is WorkflowPathPolicy.RELATIVE
    assert widget.workflow_path_policy_combo.currentData() == "relative"
    assert settings.value(WORKFLOW_PATH_POLICY_SETTING) == invalid


def test_path_preference_persists_for_the_next_widget(path_widget_factory):
    first, _runs = path_widget_factory()
    _select_path_policy(first, WorkflowPathPolicy.ABSOLUTE)

    second, _runs = path_widget_factory()

    assert second._workflow_path_policy is WorkflowPathPolicy.ABSOLUTE
    assert second.workflow_path_policy_combo.currentData() == "absolute"


def test_changing_preference_does_not_write_recalculate_or_rebind_live_graph(
    path_widget_factory, qtbot, monkeypatch, tmp_path
):
    path, source, output, _writer_id = _relative_recipe(tmp_path)
    original_workflow = path.read_bytes()
    original_source = source.read_bytes()
    widget, runs = path_widget_factory()
    widget.load_workflow_file(path)
    _event_turn(qtbot)
    runs.clear()
    graph = serialize_workflow(widget.pipeline)
    session = widget._workflow_tabs.current
    assert not session.dirty

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("Changing a path preference must not save a workflow")

    with monkeypatch.context() as context:
        context.setattr(widget_module, "save_workflow", unexpected_write)
        context.setattr(widget_module, "save_workflow_document", unexpected_write)
        _select_path_policy(widget, WorkflowPathPolicy.ABSOLUTE)
        _event_turn(qtbot)

    assert runs == []
    assert serialize_workflow(widget.pipeline) == graph
    assert not session.dirty
    assert path.read_bytes() == original_workflow
    assert source.read_bytes() == original_source
    assert not output.exists()
    assert not widget.pipeline.completed_node_ids


def test_open_relative_switch_absolute_save_and_move_only_workflow(
    path_widget_factory, qtbot, monkeypatch, tmp_path
):
    path, source, output, writer_id = _relative_recipe(tmp_path)
    source_before = source.read_bytes()
    widget, runs = path_widget_factory()
    widget.load_workflow_file(path)
    _event_turn(qtbot)
    runs.clear()
    _select_path_policy(widget, WorkflowPathPolicy.ABSOLUTE)
    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: pytest.fail("Save must reuse the current path"),
    )

    assert widget._save_workflow_dialog()

    document = json.loads(path.read_text(encoding="utf-8"))
    assert Path(_params(document, "input")["file_path"]) == source.resolve()
    assert Path(_params(document, writer_id)["path"]) == output.resolve()
    moved = tmp_path / "workflow-only" / "moved.json"
    moved.parent.mkdir()
    shutil.move(str(path), moved)
    loaded = load_workflow(moved)
    assert Path(_loaded_params(loaded, "input")["file_path"]) == source.resolve()
    assert Path(_loaded_params(loaded, writer_id)["path"]) == output.resolve()
    assert source.read_bytes() == source_before
    assert not output.exists()
    assert runs == []


@pytest.mark.parametrize("path_policy", tuple(WorkflowPathPolicy))
@pytest.mark.parametrize("save_mode", ["save-as", "timestamped"])
def test_save_as_and_timestamp_versions_honor_path_preference(
    path_widget_factory, qtbot, monkeypatch, tmp_path, path_policy, save_mode
):
    path, source, output, writer_id = _relative_recipe(tmp_path)
    original = path.read_bytes()
    widget, runs = path_widget_factory()
    widget.load_workflow_file(path)
    _event_turn(qtbot)
    runs.clear()
    _select_path_policy(widget, path_policy)
    chosen = tmp_path / "separate-workflows" / "copy.json"
    chosen.parent.mkdir()
    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(chosen), ""),
    )
    if save_mode == "timestamped":
        index = widget.workflow_save_policy_combo.findData(
            WorkflowSavePolicy.TIMESTAMPED.value
        )
        widget.workflow_save_policy_combo.setCurrentIndex(index)
        monkeypatch.setattr(
            widget_module,
            "_workflow_timestamp_now",
            lambda: datetime(2026, 10, 23, 9, 10, 11, 123456),
        )

    assert widget._save_workflow_dialog(force_choose_path=True)

    saved = widget._workflow_tabs.current.path
    assert saved.parent == chosen.parent.resolve()
    if save_mode == "timestamped":
        assert saved.name == "copy__vipp-2026-10-23_09-10-11-123456.json"
    else:
        assert saved == chosen.resolve()
    raw = json.loads(saved.read_text(encoding="utf-8"))
    is_absolute = path_policy is WorkflowPathPolicy.ABSOLUTE
    assert Path(_params(raw, "input")["file_path"]).is_absolute() is is_absolute
    assert Path(_params(raw, writer_id)["path"]).is_absolute() is is_absolute
    loaded = load_workflow(saved)
    assert Path(_loaded_params(loaded, "input")["file_path"]) == source.resolve()
    assert Path(_loaded_params(loaded, writer_id)["path"]) == output.resolve()
    assert path.read_bytes() == original
    assert not output.exists()
    assert runs == []


def test_select_relative_again_rebases_against_the_new_workflow_folder(
    path_widget_factory, qtbot, monkeypatch, tmp_path
):
    path, source, output, writer_id = _relative_recipe(tmp_path)
    widget, runs = path_widget_factory()
    widget.load_workflow_file(path)
    _event_turn(qtbot)
    runs.clear()
    _select_path_policy(widget, WorkflowPathPolicy.ABSOLUTE)
    assert widget._save_workflow_dialog()
    _select_path_policy(widget, WorkflowPathPolicy.RELATIVE)
    chosen = tmp_path / "other" / "portable.json"
    chosen.parent.mkdir()
    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(chosen), ""),
    )

    assert widget._save_workflow_dialog(force_choose_path=True)

    raw = json.loads(chosen.read_text(encoding="utf-8"))
    relative_input = _params(raw, "input")["file_path"]
    relative_output = _params(raw, writer_id)["path"]
    assert not Path(relative_input).is_absolute()
    assert not Path(relative_output).is_absolute()
    assert (chosen.parent / relative_input).resolve() == source.resolve()
    assert (chosen.parent / relative_output).resolve() == output.resolve()
    assert Path(widget.pipeline.nodes["input"].params["file_path"]) == source.resolve()
    assert Path(widget.pipeline.nodes[writer_id].params["path"]) == output.resolve()
    assert runs == []
    assert not output.exists()


def test_absolute_save_keeps_attached_batch_locations_when_only_json_moves(
    path_widget_factory, monkeypatch, tmp_path
):
    widget, runs = path_widget_factory()
    _dialog, input_dir, output_dir = _configure_workflow_attached_batch(
        widget, tmp_path
    )
    _select_path_policy(widget, WorkflowPathPolicy.ABSOLUTE)
    target = tmp_path / "workflow-with-batch.json"
    monkeypatch.setattr(
        widget_module.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(target), ""),
    )
    monkeypatch.setattr(
        widget_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )

    assert widget._save_workflow_dialog()

    raw = json.loads(target.read_text(encoding="utf-8"))
    config = BatchConfig.from_dict(raw["batch_config"], base_dir=target.parent)
    assert config.workflow_file == target.resolve()
    assert config.workflow_sha256 == scientific_workflow_hash(raw)
    assert config.output_dir == output_dir.resolve()
    assert config.sources[0].input_dir == input_dir.resolve()
    moved = tmp_path / "workflow-only" / "batch.json"
    moved.parent.mkdir()
    shutil.move(str(target), moved)
    loaded = load_workflow(moved)
    assert not loaded.get("batch_config_error")
    restored = BatchConfig.from_dict(loaded["batch_config"], base_dir=moved.parent)
    assert restored.workflow_file == moved.resolve()
    assert restored.output_dir == output_dir.resolve()
    assert restored.sources[0].input_dir == input_dir.resolve()
    assert (input_dir / "field.npy").exists()
    assert not output_dir.exists()
    assert not widget.pipeline.completed_node_ids
    assert runs == []
