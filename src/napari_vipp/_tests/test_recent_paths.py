from __future__ import annotations

import os
from pathlib import Path

import pytest
from qtpy.QtCore import QSettings

from napari_vipp.ui import recent_paths


def test_recent_directory_persists_and_ignores_missing_paths(tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()

    recent_paths.remember_directory(recent_paths.INPUT_DIRECTORY, selected)

    assert recent_paths.recent_directory(recent_paths.INPUT_DIRECTORY) == str(
        selected.resolve()
    )
    selected.rmdir()
    assert recent_paths.recent_directory(recent_paths.INPUT_DIRECTORY) == ""


def test_initial_file_path_preserves_generic_fallback(tmp_path):
    assert (
        recent_paths.initial_file_path(
            recent_paths.WORKFLOW_DIRECTORY,
            "vipp_workflow.json",
        )
        == "vipp_workflow.json"
    )

    selected = tmp_path / "workflows"
    selected.mkdir()
    recent_paths.remember_directory(recent_paths.WORKFLOW_DIRECTORY, selected)

    assert recent_paths.initial_file_path(
        recent_paths.WORKFLOW_DIRECTORY,
        "vipp_workflow.json",
    ) == str(selected / "vipp_workflow.json")


def test_recent_workflows_are_absolute_newest_first_and_deduplicated(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    recent_paths.remember_workflow("first.json")
    recent_paths.remember_workflow(second)
    recent_paths.remember_workflow(tmp_path / "sub" / ".." / first.name)

    assert recent_paths.recent_workflows() == [str(first), str(second)]


def test_recent_workflows_bound_history_without_probing_missing_files(
    tmp_path, monkeypatch,
):
    paths = [tmp_path / f"workflow-{index}.json" for index in range(12)]
    for path in paths:
        recent_paths.remember_workflow(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", lambda *_a, **_k: pytest.fail("No disk probes"))
        assert recent_paths.recent_workflows() == [
            str(path) for path in reversed(paths[-recent_paths.MAX_RECENT_WORKFLOWS:])
        ]


def test_recent_workflows_ignore_malformed_settings(tmp_path):
    path = str(tmp_path / "workflow.json")
    settings = recent_paths._settings()
    for invalid in (None, 42, {"path": path}):
        settings.setValue(recent_paths.RECENT_WORKFLOWS, invalid)
        assert recent_paths.recent_workflows() == []
    settings.setValue(
        recent_paths.RECENT_WORKFLOWS,
        [None, 42, "", "relative.json", "bad\0path", path, path],
    )
    assert recent_paths.recent_workflows() == [path]
    settings.setValue(recent_paths.RECENT_WORKFLOWS, path)
    assert recent_paths.recent_workflows() == [path]


def test_recent_workflows_use_platform_path_case_rules(tmp_path):
    path = str(tmp_path / "workflow.json")
    recent_paths.remember_workflow(path)
    recent_paths.remember_workflow(path.upper())
    expected = [path.upper()] if os.name == "nt" else [path.upper(), path]
    assert recent_paths.recent_workflows() == expected


def test_recent_workflows_persist_across_settings_instances_and_clear_only_history(
    tmp_path, monkeypatch,
):
    ini = tmp_path / "history.ini"
    monkeypatch.setattr(
        recent_paths, "_settings", lambda: QSettings(str(ini), QSettings.IniFormat)
    )
    workflow = tmp_path / "kept.json"
    workflow.write_text("{}", encoding="utf-8")
    recent_paths.remember_directory(recent_paths.WORKFLOW_DIRECTORY, tmp_path)
    recent_paths.remember_workflow(workflow)
    assert recent_paths.recent_workflows() == [str(workflow)]

    recent_paths.clear_recent_workflows()

    assert recent_paths.recent_workflows() == []
    assert workflow.read_text(encoding="utf-8") == "{}"
    assert recent_paths.recent_directory(recent_paths.WORKFLOW_DIRECTORY) == str(
        tmp_path.resolve()
    )
