"""Read-only discovery of a usable system Python for the Windows setup GUI."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from napari_vipp.installer.discovery import InterpreterProbe, default_services
from napari_vipp.installer.models import ComputeTrack

# Python 3.12.10 is the final 3.12 release with an official Windows installer.
# It satisfies both the CPU and CUDA 13 tracks, so the novice setup route links
# to this exact release rather than a generic page that may offer Python 3.14.
PYTHON_DOWNLOAD_URL = "https://www.python.org/downloads/release/python-31210/"
_CPU_MINORS = frozenset({(3, 12), (3, 13)})
_CUDA_MINORS = frozenset({(3, 12)})
_LAUNCHER_PATH = re.compile(
    r'(?:^|\s)"?([A-Za-z]:[\\/].*?python(?:w)?\.exe)"?\s*$',
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PythonCandidate:
    """One independently probed base interpreter."""

    executable: Path
    version: tuple[int, int, int]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "executable", Path(self.executable))

    @property
    def version_text(self) -> str:
        return ".".join(str(part) for part in self.version)


def discover_python_candidates(
    *,
    environ: Mapping[str, str] | None = None,
    probe: Callable[[Path], InterpreterProbe] | None = None,
    candidate_paths: Iterable[tuple[str, Path]] | None = None,
    frozen: bool | None = None,
) -> tuple[PythonCandidate, ...]:
    """Return supported 64-bit CPython installations without changing them.

    A frozen setup executable is never treated as Python.  This distinction is
    essential because ``sys.executable`` points to the PyInstaller setup EXE in
    a published build, not to an interpreter capable of creating a venv.
    """

    environment = dict(os.environ if environ is None else environ)
    selected_probe = probe or default_services().interpreter_probe
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    sources = (
        tuple(candidate_paths)
        if candidate_paths is not None
        else _system_candidate_paths(environment, include_current=not is_frozen)
    )
    frozen_executable = _normal_path(sys.executable) if is_frozen else ""
    seen: set[str] = set()
    candidates: list[PythonCandidate] = []
    for source, path in sources:
        identity = _normal_path(path)
        if not identity or identity in seen or identity == frozen_executable:
            continue
        seen.add(identity)
        try:
            result = selected_probe(Path(path))
        except Exception:
            continue
        if (
            result.implementation.casefold() != "cpython"
            or result.pointer_bits != 64
            or result.version[:2] not in _CPU_MINORS
        ):
            continue
        candidates.append(
            PythonCandidate(
                executable=result.executable,
                version=result.version,
                source=source,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                tuple(-part for part in item.version),
                str(item.executable).casefold(),
                item.source.casefold(),
            ),
        )
    )


def choose_python(
    candidates: Iterable[PythonCandidate],
    track: ComputeTrack,
) -> PythonCandidate | None:
    """Choose the newest interpreter allowed for one released compute track."""

    accepted = _CUDA_MINORS if track is ComputeTrack.CUDA13 else _CPU_MINORS
    matches = [item for item in candidates if item.version[:2] in accepted]
    return min(
        matches,
        key=lambda item: (
            tuple(-part for part in item.version),
            str(item.executable).casefold(),
        ),
        default=None,
    )


def _system_candidate_paths(
    environ: Mapping[str, str],
    *,
    include_current: bool,
) -> tuple[tuple[str, Path], ...]:
    paths: list[tuple[str, Path]] = []
    requested = environ.get("VIPP_INSTALLER_PYTHON", "").strip()
    if requested:
        paths.append(("installer configuration", Path(requested)))
    paths.extend(_registry_python_paths())
    paths.extend(_python_launcher_paths(environ))
    for command in ("python3.13", "python3.12", "python"):
        resolved = shutil.which(command, path=environ.get("PATH"))
        if resolved:
            paths.append(("PATH", Path(resolved)))
    if include_current:
        paths.append(("current Python", Path(sys.executable)))
    return tuple(paths)


def _python_launcher_paths(
    environ: Mapping[str, str],
) -> tuple[tuple[str, Path], ...]:
    launcher = shutil.which("py", path=environ.get("PATH"))
    if not launcher:
        return ()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            (launcher, "-0p"),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            shell=False,
            env=dict(environ),
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    found: list[tuple[str, Path]] = []
    for line in completed.stdout.splitlines():
        match = _LAUNCHER_PATH.search(line.strip().strip('"'))
        if match:
            found.append(("Python launcher", Path(match.group(1).strip('"'))))
    return tuple(found)


def _registry_python_paths() -> tuple[tuple[str, Path], ...]:
    if sys.platform != "win32":
        return ()
    try:
        import winreg
    except ImportError:  # pragma: no cover - only possible on non-Windows Python
        return ()

    roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
    views = tuple(
        dict.fromkeys(
            (
                getattr(winreg, "KEY_WOW64_64KEY", 0),
                getattr(winreg, "KEY_WOW64_32KEY", 0),
            )
        )
    )
    found: list[tuple[str, Path]] = []
    for root in roots:
        for view in views:
            access = winreg.KEY_READ | view
            try:
                with winreg.OpenKey(root, r"Software\Python", 0, access) as python:
                    companies = _registry_subkeys(winreg, python)
            except OSError:
                continue
            for company in companies:
                company_key = rf"Software\Python\{company}"
                try:
                    with winreg.OpenKey(root, company_key, 0, access) as opened:
                        tags = _registry_subkeys(winreg, opened)
                except OSError:
                    continue
                for tag in tags:
                    install_key = rf"{company_key}\{tag}\InstallPath"
                    try:
                        with winreg.OpenKey(root, install_key, 0, access) as opened:
                            executable = _registry_value(
                                winreg,
                                opened,
                                "ExecutablePath",
                            )
                            install_root = _registry_value(winreg, opened, "")
                    except OSError:
                        continue
                    candidate = (
                        Path(executable)
                        if executable
                        else Path(install_root) / "python.exe"
                        if install_root
                        else None
                    )
                    if candidate is not None:
                        found.append((f"Windows registry ({company} {tag})", candidate))
    return tuple(found)


def _registry_subkeys(winreg, key) -> tuple[str, ...]:
    values: list[str] = []
    index = 0
    while True:
        try:
            values.append(winreg.EnumKey(key, index))
        except OSError:
            return tuple(values)
        index += 1


def _registry_value(winreg, key, name: str) -> str:
    try:
        value, _kind = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(value).strip() if value is not None else ""


def _normal_path(path: str | Path) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError):
        return ""


__all__ = [
    "PYTHON_DOWNLOAD_URL",
    "PythonCandidate",
    "choose_python",
    "discover_python_candidates",
]
