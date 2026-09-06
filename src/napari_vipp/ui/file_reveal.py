"""Reveal checked local files without interpreting paths as shell commands."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RevealResult:
    """Whether a request launched, and requested exact file selection."""

    success: bool
    selected: bool
    message: str


def file_reveal_label(platform_name: str | None = None) -> str:
    """Describe the actual platform action, including the Linux fallback."""

    platform_name = platform_name or sys.platform
    if platform_name == "win32":
        return "Find in File Explorer"
    if platform_name == "darwin":
        return "Find in Finder"
    return "Open containing folder"


def _launch(arguments: list[str]) -> None:
    subprocess.Popen(
        arguments,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def reveal_file(path: str | Path, *, platform_name: str | None = None) -> RevealResult:
    """Select one existing file, or explicitly open its parent on Linux.

    The existence check is repeated at activation, since a displayed result can
    outlive its output. Opening the containing folder is not reported as exact
    selection. The original absolute path is retained, including symlink names.
    """

    target = Path(path).expanduser().absolute()
    if not target.is_file():
        return RevealResult(False, False, f"File is missing or unavailable: {target}")
    platform_name = platform_name or sys.platform
    if platform_name == "win32":
        arguments = ["explorer.exe", "/select,", str(target)]
        selected = True
    elif platform_name == "darwin":
        arguments = ["open", "-R", str(target)]
        selected = True
    else:
        arguments = ["xdg-open", str(target.parent)]
        selected = False
    try:
        _launch(arguments)
    except OSError as exc:
        return RevealResult(False, False, f"Could not open the file manager: {exc}")
    message = (
        f"Requested selection of {target.name}."
        if selected
        else f"Opened containing folder; select {target.name} in the file manager."
    )
    return RevealResult(True, selected, message)


def open_folder(path: str | Path, *, platform_name: str | None = None) -> RevealResult:
    """Open one existing output directory without creating it."""

    target = Path(path).expanduser().absolute()
    if not target.is_dir():
        return RevealResult(False, False, f"Folder is missing or unavailable: {target}")
    platform_name = platform_name or sys.platform
    executable = (
        "explorer.exe"
        if platform_name == "win32"
        else "open"
        if platform_name == "darwin"
        else "xdg-open"
    )
    try:
        _launch([executable, str(target)])
    except OSError as exc:
        return RevealResult(False, False, f"Could not open the file manager: {exc}")
    return RevealResult(True, False, f"Opened output folder: {target}")


__all__ = ["RevealResult", "file_reveal_label", "open_folder", "reveal_file"]
