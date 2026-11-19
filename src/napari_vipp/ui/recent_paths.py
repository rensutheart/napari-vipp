"""Persistent dialog directories and recently opened/saved workflow paths."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtCore import QSettings

INPUT_DIRECTORY = "dialogs/input-directory"
WORKFLOW_DIRECTORY = "dialogs/workflow-directory"
RECENT_WORKFLOWS = "workflows/recent-files"
MAX_RECENT_WORKFLOWS = 10


def _settings() -> QSettings:
    """Return plugin-owned settings independent of the napari host identity."""
    return QSettings("napari-vipp", "napari-vipp")


def recent_directory(key: str) -> str:
    """Return a persisted existing directory, or an empty fallback."""
    value = str(_settings().value(str(key), "") or "").strip()
    if not value:
        return ""
    directory = Path(value).expanduser()
    return str(directory) if directory.is_dir() else ""


def initial_file_path(key: str, fallback_name: str) -> str:
    """Build a file-dialog start path from a recent directory and filename."""
    directory = recent_directory(key)
    return str(Path(directory) / fallback_name) if directory else fallback_name


def initial_directory(key: str, current: str = "") -> str:
    """Prefer an explicit current value, otherwise use the recent directory."""
    current = str(current).strip()
    return current or recent_directory(key)


def remember_directory(key: str, directory: str | Path) -> None:
    """Persist an existing directory selected by the user."""
    path = Path(directory).expanduser()
    if not path.is_dir():
        return
    settings = _settings()
    settings.setValue(str(key), str(path.resolve()))
    settings.sync()


def remember_file_directory(key: str, filename: str | Path) -> None:
    """Persist the parent directory of a selected file."""
    remember_directory(key, Path(filename).expanduser().parent)


def _workflow_paths(settings: QSettings) -> list[str]:
    """Read bounded history without probing disks or offline network shares."""
    settings.sync()
    raw = settings.value(RECENT_WORKFLOWS, [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value or "\0" in value:
            continue
        if not os.path.isabs(value):
            continue
        path = os.path.normpath(value)
        key = os.path.normcase(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
        if len(paths) == MAX_RECENT_WORKFLOWS:
            break
    return paths


def recent_workflows() -> list[str]:
    """Return full workflow paths, newest first, retaining missing/offline files."""
    return _workflow_paths(_settings())


def remember_workflow(filename: str | Path) -> None:
    """Remember a successfully opened or saved workflow, not an attempted path."""
    path = os.path.abspath(os.path.expanduser(os.fspath(filename)))
    settings = _settings()
    previous = _workflow_paths(settings)
    key = os.path.normcase(path)
    paths = [path, *(item for item in previous if os.path.normcase(item) != key)]
    settings.setValue(RECENT_WORKFLOWS, paths[:MAX_RECENT_WORKFLOWS])
    settings.sync()


def clear_recent_workflows() -> None:
    """Forget only the history; never delete workflows or reset dialog folders."""
    settings = _settings()
    settings.setValue(RECENT_WORKFLOWS, [])
    settings.sync()


__all__ = [
    "INPUT_DIRECTORY",
    "MAX_RECENT_WORKFLOWS",
    "RECENT_WORKFLOWS",
    "WORKFLOW_DIRECTORY",
    "clear_recent_workflows",
    "initial_directory",
    "initial_file_path",
    "recent_directory",
    "recent_workflows",
    "remember_directory",
    "remember_file_directory",
    "remember_workflow",
]
