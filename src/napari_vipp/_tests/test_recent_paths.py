from __future__ import annotations

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
