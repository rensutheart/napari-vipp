"""Persistent last-used directories for VIPP file and folder dialogs."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSettings

INPUT_DIRECTORY = "dialogs/input-directory"
WORKFLOW_DIRECTORY = "dialogs/workflow-directory"


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


__all__ = [
    "INPUT_DIRECTORY",
    "WORKFLOW_DIRECTORY",
    "initial_directory",
    "initial_file_path",
    "recent_directory",
    "remember_directory",
    "remember_file_directory",
]
