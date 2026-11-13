"""Actual owned Windows children in tiny package-free disposable prefixes.

Process enumeration is injected to this test's known process set, not the host's
unrelated user sessions. Runtime maintenance/permits/venv broker and file moves
are real. No pip installation, scientific package, UI or Windows entry is used.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from napari_vipp import _runtime_lease as rt
from napari_vipp._tests.test_installer_incremental import _distribution
from napari_vipp._tests.test_installer_inplace_live import (
    _ENTRIES,
    _NEW_VERSION,
    _OLD_VERSION,
    _assert_protocol_released,
    _live_source,
    _ordinary_startup,
    _sha,
)
from napari_vipp.installer import legacy_inplace as legacy
from napari_vipp.installer.engine import InstallStatus
from napari_vipp.installer.ownership import (
    inspect_ownership,
    ownership_path,
    write_ownership_record,
)

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="Real Windows runtime coordination required."
)


def _legacy_source(tmp_path, monkeypatch, *, fail=False):
    source = _live_source(tmp_path, fail_acceptance=fail)
    state = source.environment / rt.DIRECTORY
    # Only the known, just-created disposable fixture's empty protocol state.
    for name in (rt.CAPABILITY, "gate.lock"):
        (state / name).unlink()
    for name in ("leases", "permits", "pending"):
        (state / name).rmdir()
    state.rmdir()
    (source.site / "napari_vipp/_runtime_lease.py").unlink()
    legacy_app = _distribution(
        source.site,
        "napari-vipp",
        _OLD_VERSION,
        requirements=("dependency>=1",),
        extras=("app",),
        entries=_ENTRIES,
        files={
            "napari_vipp/__init__.py": (
                b"from importlib.metadata import version\n"
                b"__version__ = version('napari-vipp')\n"
            ),
            "napari_vipp/app.py": b"VALUE = 1\n\ndef main():\n    return 0\n",
            "../../Scripts/vipp-app.exe": b"disposable nonexecuted launcher",
        },
    )
    source.old = replace(
        source.old,
        packages=tuple(
            legacy_app if p.normalized_name == "napari-vipp" else p
            for p in source.old.packages
        ),
    )
    write_ownership_record(source.root, source.old)
    monkeypatch.setattr(legacy, "_windows_processes", lambda deadline: ())
    # Exact acceptance import command, not an unrestricted Python-code allowlist.
    code = (
        "import sys; from pathlib import Path; import napari_vipp; "
        f"assert napari_vipp.__version__ == {_NEW_VERSION!r}; "
        f"assert Path(sys.prefix) == Path({str(source.environment)!r}); "
        "assert Path(napari_vipp.__file__).parent == "
        f"Path({str(source.site / 'napari_vipp')!r}); "
        "assert napari_vipp._runtime_use_lease is not None; "
        "print('Owned changed-package import accepted')"
    )
    source.runner.allowed.add((str(source.runner.python), "-I", "-B", "-c", code))
    return source


def _preserved(source):
    selected = [
        source.environment / "Scripts/python.exe",
        source.environment / "Scripts/vipp-app.exe",
        source.environment / "pyvenv.cfg",
        source.environment / ".vipp-install-candidate.json",
        *sorted((source.site / "dependency").rglob("*")),
        *sorted((source.site / "dependency-1.2.dist-info").rglob("*")),
    ]
    return {
        str(p.relative_to(source.environment)): (
            p.stat().st_ino,
            p.stat().st_mtime_ns,
            p.stat().st_size,
            _sha(p.read_bytes()),
        )
        for p in selected
        if p.is_file()
    }


def test_first_legacy_upgrade_without_copy_and_early_startup_race_blocked(
    tmp_path, monkeypatch
):
    source = _legacy_source(tmp_path, monkeypatch)
    before, global_env = _preserved(source), dict(os.environ)
    prepared = source.engine.prepare(source.plan, resolve=False)
    assert prepared.reuse_plan.transaction_kind == "package_delta", (
        prepared.strategy_reason
    )
    assert source.runner.calls == []
    original_run = source.runner.run
    refused = []
    marker = tmp_path / "must-not-execute.txt"

    def with_race(argv, **kwargs):
        if kwargs.get("before_resume") is not None and not refused:
            race_code = (
                f"from pathlib import Path; Path({str(marker)!r})"
                ".write_text('unsafe start')"
            )
            race = (str(source.runner.python), "-I", "-B", "-c", race_code)
            source.runner.allowed.add(race)
            refused.append(original_run(race, env=kwargs.get("env")))
        return original_run(argv, **kwargs)

    monkeypatch.setattr(source.runner, "run", with_race)
    result = source.engine.apply(
        prepared, source.engine.authorize(prepared, confirmed=True)
    )
    assert result.status is InstallStatus.SUCCEEDED, result.technical_error + str(
        result.rollback.errors
    )
    assert result.environment_root == source.environment
    assert (
        inspect_ownership(source.root).record.installation_id
        == source.old.installation_id
    )
    assert _preserved(source) == before
    assert len(refused) == 1 and refused[0].returncode != 0
    assert not marker.exists()
    assert b"SystemExit" in refused[0].stderr.encode()
    assert (source.site / legacy.GUARD_NAME).read_bytes() == legacy.GUARD_BYTES
    _assert_protocol_released(source)
    _ordinary_startup(source, _NEW_VERSION)
    assert _preserved(source) == before
    assert os.environ == global_env


def test_failed_acceptance_restores_real_legacy_without_enrolling_it(
    tmp_path, monkeypatch
):
    source = _legacy_source(tmp_path, monkeypatch, fail=True)
    before = _preserved(source)
    owner = ownership_path(source.root).read_bytes()
    app = {
        p.relative_to(source.site).as_posix(): p.read_bytes()
        for folder in (
            source.site / "napari_vipp",
            source.site / f"napari_vipp-{_OLD_VERSION}.dist-info",
        )
        for p in folder.rglob("*")
        if p.is_file()
    }
    prepared = source.engine.prepare(source.plan)
    assert prepared.reuse_plan.transaction_kind == "package_delta"
    result = source.engine.apply(
        prepared, source.engine.authorize(prepared, confirmed=True)
    )
    assert result.status is InstallStatus.FAILED
    assert result.rollback.completed, result.rollback.errors
    assert ownership_path(source.root).read_bytes() == owner
    assert _preserved(source) == before
    assert not (source.environment / rt.DIRECTORY).exists()
    assert not (source.site / legacy.GUARD_NAME).exists()
    for relative, content in app.items():
        assert (source.site / relative).read_bytes() == content
    code = (
        "import napari_vipp, napari_vipp.app, dependency; "
        f"assert napari_vipp.__version__ == {_OLD_VERSION!r}; "
        "assert napari_vipp.app.VALUE == 1 and dependency.VALUE == 1; "
        "print('Restored legacy startup accepted')"
    )
    call = (str(source.runner.python), "-I", "-B", "-c", code)
    source.runner.allowed.add(call)
    assert source.runner.run(call).returncode == 0
