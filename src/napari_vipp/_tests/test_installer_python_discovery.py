from __future__ import annotations

import sys
from pathlib import Path

from napari_vipp.installer.discovery import InterpreterProbe
from napari_vipp.installer.models import ComputeTrack
from napari_vipp.installer.python_discovery import (
    choose_python,
    discover_python_candidates,
)


def _probe(path: Path) -> InterpreterProbe:
    text = path.name.casefold()
    version = (
        (3, 13, 2)
        if "313" in text
        else (3, 12, 10)
        if "312" in text
        else (3, 11, 9)
    )
    return InterpreterProbe(
        executable=path,
        base_executable=path,
        implementation="pypy" if "pypy" in text else "cpython",
        version=version,
        pointer_bits=32 if "32bit" in text else 64,
    )


def test_candidates_are_probed_deduplicated_and_filtered(tmp_path):
    python313 = tmp_path / "python313.exe"
    python312 = tmp_path / "python312.exe"
    paths = (
        ("registry", python312),
        ("launcher", python313),
        ("duplicate", python312),
        ("old", tmp_path / "python311.exe"),
        ("wrong implementation", tmp_path / "pypy312.exe"),
        ("wrong architecture", tmp_path / "python312-32bit.exe"),
    )

    candidates = discover_python_candidates(
        candidate_paths=paths,
        probe=_probe,
        frozen=False,
    )

    assert [item.executable for item in candidates] == [python313, python312]
    assert [item.source for item in candidates] == ["launcher", "registry"]


def test_track_selection_uses_313_for_cpu_and_312_for_cuda(tmp_path):
    candidates = discover_python_candidates(
        candidate_paths=(
            ("registry", tmp_path / "python312.exe"),
            ("launcher", tmp_path / "python313.exe"),
        ),
        probe=_probe,
        frozen=False,
    )

    assert choose_python(candidates, ComputeTrack.CPU).version[:2] == (3, 13)
    assert choose_python(candidates, ComputeTrack.CUDA13).version[:2] == (3, 12)


def test_frozen_setup_executable_is_never_returned_as_python(monkeypatch):
    setup_exe = Path(sys.executable)
    monkeypatch.setattr(sys, "executable", str(setup_exe))
    calls = []

    candidates = discover_python_candidates(
        candidate_paths=(("current process", setup_exe),),
        probe=lambda path: calls.append(path) or _probe(path),
        frozen=True,
    )

    assert candidates == ()
    assert calls == []


def test_probe_failures_are_skipped_without_stopping_discovery(tmp_path):
    broken = tmp_path / "broken-python.exe"
    working = tmp_path / "python312.exe"

    def probe(path):
        if path == broken:
            raise OSError("cannot start")
        return _probe(path)

    candidates = discover_python_candidates(
        candidate_paths=(("registry", broken), ("launcher", working)),
        probe=probe,
        frozen=False,
    )

    assert [item.executable for item in candidates] == [working]
