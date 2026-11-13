"""App-only engine integration uses only package-free disposable environments.

These Windows tests execute the real early runtime hook, maintenance interlock,
suspended-child permit, app exchange, and control commit/rollback.  They do not
install scientific packages, run pip, touch Windows registration, or launch UI.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import threading
import uuid
import zipfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from napari_vipp import _runtime_lease as protocol
from napari_vipp._tests.test_installer_engine import _plan
from napari_vipp._tests.test_installer_incremental import _distribution
from napari_vipp.installer.discovery import default_services
from napari_vipp.installer.engine import (
    CancellationToken,
    CommandResult,
    InstallStatus,
    ManagedInstallerEngine,
    ProgressStage,
    SubprocessCommandRunner,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    PlannedAction,
    PythonSnapshot,
    ReleaseSpec,
    UpdateStrategy,
    installation_request_fingerprint,
)
from napari_vipp.installer.ownership import (
    OwnershipRecord,
    inspect_ownership,
    managed_environments_root,
    ownership_path,
    write_ownership_record,
)
from napari_vipp.installer.planner import create_install_plan
from napari_vipp.installer.windows_backend import _vipp_launch_environment

pytestmark = pytest.mark.skipif(
    os.name != "nt"
    or sys.version_info[:2] not in {(3, 12), (3, 13)}
    or struct.calcsize("P") != 8,
    reason="The actual app-only protocol requires 64-bit Windows CPython 3.12/3.13.",
)

_OLD_VERSION = "1.0a1"
_NEW_VERSION = "1.0a2"
_ENTRIES = "[gui_scripts]\nvipp-app = napari_vipp.app:main\n"
_NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _wheel(path: Path, protocol_files: dict[str, bytes]) -> ReleaseSpec:
    """Write a small exact wheel; no wheel builder, pip, or source imports."""
    info = f"napari_vipp-{_NEW_VERSION}.dist-info"
    content = {
        **protocol_files,
        "napari_vipp/app.py": b"VALUE = 2\n\ndef main():\n    return 0\n",
        f"{info}/METADATA": (
            "Metadata-Version: 2.3\nName: napari-vipp\n"
            f"Version: {_NEW_VERSION}\nRequires-Python: >=3.12,<3.14\n"
            "Provides-Extra: app\nRequires-Dist: dependency>=1\n\n"
        ).encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        f"{info}/entry_points.txt": _ENTRIES.encode(),
    }
    rows = []
    for name, data in content.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        rows.append((name, "sha256=" + encoded.rstrip(b"=").decode(), len(data)))
    rows.append((f"{info}/RECORD", "", ""))
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    content[f"{info}/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            archive.writestr(name, data)
    return ReleaseSpec(
        "napari-vipp",
        _NEW_VERSION,
        # This release fixture contains only a tiny app and one tiny dependency.
        # Exercise the real capacity check, without demanding production GiB.
        managed_cpu_min_free_bytes=1024**2,
        wheel_path=path,
        wheel_sha256=_sha(path.read_bytes()),
    )


def _probe_code(version: str, value: int, *, fail: bool = False) -> str:
    return (
        "import importlib.metadata as m, json, sys; "
        "import napari_vipp, napari_vipp.app, dependency; "
        f"assert m.version('napari-vipp') == {version!r}; "
        f"assert napari_vipp.__version__ == {version!r}; "
        f"assert napari_vipp.app.VALUE == {value}; "
        "assert dependency.VALUE == 1; "
        "assert napari_vipp._runtime_use_lease is not None; "
        "assert not any(n in sys.modules for n in ('numpy', 'qtpy', 'PySide6')); "
        "print(json.dumps({'version': m.version('napari-vipp'), "
        "'prefix': sys.prefix, 'package': napari_vipp.__file__, "
        "'app_value': napari_vipp.app.VALUE, 'dependency_value': dependency.VALUE, "
        "'lease_registered': napari_vipp._runtime_use_lease is not None}), "
        "flush=True); " + ("sys.exit(23)" if fail else "sys.exit(0)")
    )


class _OnlyDisposablePython:
    """Real owned children, with an exact argv allowlist and a bounded deadline."""

    def __init__(self, python: Path, codes: tuple[str, ...]) -> None:
        self.python = python
        self.allowed = {(str(python), "-I", "-B", "-c", code) for code in codes}
        self.calls: list[tuple[str, ...]] = []
        self.results: list[CommandResult] = []
        self.permitted_calls = 0
        self._runner = SubprocessCommandRunner()

    def run(self, argv, *, cancellation=None, env=None, cwd=None, before_resume=None):
        call = tuple(str(value) for value in argv)
        self.calls.append(call)
        # A fallback resolver, ensurepip, venv, dependency copy, shell, or app
        # launcher is a test failure, not a fake successful command.
        assert call in self.allowed, f"Unexpected execution in app-only test: {call!r}"
        assert Path(call[0]) == self.python
        deadline = CancellationToken()
        timer = threading.Timer(25, deadline.cancel)
        timer.daemon = True

        def cancelled():
            check = getattr(cancellation, "is_set", None)
            return deadline.is_cancelled() or bool(callable(check) and check())

        timer.start()
        try:
            if before_resume is not None:
                self.permitted_calls += 1
            result = self._runner.run(
                call,
                cancellation=cancelled,
                env=env,
                cwd=cwd,
                before_resume=before_resume,
            )
            self.results.append(result)
            return result
        finally:
            timer.cancel()
            timer.join()


@dataclass
class _LiveSource:
    root: Path
    environment: Path
    site: Path
    old: OwnershipRecord
    engine: ManagedInstallerEngine
    plan: object
    runner: _OnlyDisposablePython
    startup_code: str


def _live_source(tmp_path: Path, *, fail_acceptance: bool = False) -> _LiveSource:
    root = tmp_path / "cpu"
    environment = managed_environments_root(root) / "old-accepted"
    # The environment exists only within this test's temporary directory. No
    # package bootstrap or inherited system site-packages is requested.
    base = Path(sys._base_executable).resolve(strict=True)
    # Production creates from the selected BASE interpreter, not the test
    # runner's venv. That also gives truthful pyvenv.cfg executable/home facts.
    creation = subprocess.run(
        (str(base), "-I", "-B", "-m", "venv", "--without-pip", str(environment)),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        creationflags=0x08000000,
    )
    assert creation.returncode == 0, creation.stderr
    site = environment / "Lib" / "site-packages"
    source = Path(protocol.__file__)
    protocol_files = {
        "napari_vipp/__init__.py": source.with_name("__init__.py").read_bytes(),
        "napari_vipp/_runtime_lease.py": source.read_bytes(),
    }
    old_app = _distribution(
        site,
        "napari-vipp",
        _OLD_VERSION,
        requirements=("dependency>=1",),
        extras=("app",),
        files={
            **protocol_files,
            "napari_vipp/app.py": b"VALUE = 1\n\ndef main():\n    return 0\n",
            # This exact owned launcher is never executed; only venv Python is.
            "../../Scripts/vipp-app.exe": b"disposable nonexecuted launcher",
        },
        entries=_ENTRIES,
    )
    dependency = _distribution(site, "dependency", "1.2")
    marker = environment / ".vipp-install-candidate.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "napari-vipp-install-candidate",
                "schema_version": 1,
                "run_id": str(uuid.uuid4()),
                "resolution_id": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    old = OwnershipRecord(
        installation_id=str(uuid.uuid4()),
        managed_root=root,
        environment_root=environment,
        distribution="napari-vipp",
        version=_OLD_VERSION,
        track=ComputeTrack.CPU,
        base_python=base,
        resolved_plan_id="a" * 64,
        packages=(old_app, dependency),
        created_at=_NOW.isoformat(),
        updated_at=_NOW.isoformat(),
        environment_marker_sha256=_sha(marker.read_bytes()),
    )
    # The production opt-in rejects modifying an already current environment.
    assert protocol.install_capability(environment, root, old.installation_id)
    write_ownership_record(root, old)
    assert protocol.assess_capability(environment, old.installation_id).eligible

    release = _wheel(
        tmp_path / f"napari_vipp-{_NEW_VERSION}-py3-none-any.whl", protocol_files
    )
    base_plan = _plan(root, release)
    request = replace(
        base_plan.request, python=base, update_strategy=UpdateStrategy.AUTO
    )
    snapshot = replace(
        base_plan.discovery,
        request_fingerprint=installation_request_fingerprint(request),
        python=PythonSnapshot(
            requested_executable=base,
            executable=base,
            base_executable=base,
            probe_succeeded=True,
            implementation="cpython",
            version=tuple(sys.version_info[:3]),
            pointer_bits=64,
        ),
    )
    plan = create_install_plan(request, discovery=snapshot, release=release)
    acceptance = _probe_code(_NEW_VERSION, 2, fail=fail_acceptance)
    startup = _probe_code(
        _OLD_VERSION if fail_acceptance else _NEW_VERSION, 1 if fail_acceptance else 2
    )
    python = environment / "Scripts" / "python.exe"
    owned_import = (
        "import sys; from pathlib import Path; import napari_vipp; "
        f"assert napari_vipp.__version__ == {_NEW_VERSION!r}; "
        f"assert Path(sys.prefix) == Path({str(environment)!r}); "
        "assert Path(napari_vipp.__file__).parent == "
        f"Path({str(site / 'napari_vipp')!r}); "
        "assert napari_vipp._runtime_use_lease is not None; "
        "print('Owned in-place VIPP import accepted')"
    )
    runner = _OnlyDisposablePython(python, (acceptance, startup, owned_import))
    plan = replace(
        plan,
        acceptance=(
            PlannedAction(
                "app-only-live-probe",
                "Import the actual staged VIPP and unchanged dependency",
                (str(python), "-I", "-B", "-c", acceptance),
            ),
        ),
    )
    folders = {name: tmp_path / name for name in ("documents", "desktop", "programs")}
    for path in folders.values():
        path.mkdir()
    engine = ManagedInstallerEngine(
        runner=runner,
        state_root=tmp_path / "installer-state",
        now=lambda: _NOW,
        identifier=lambda: str(uuid.uuid4()),
        require_embedded_wheel=True,
        # No LocalAppData, setup EXE, registry, or shortcut destination is used.
        known_folder_probe=folders.get,
    )
    return _LiveSource(root, environment, site, old, engine, plan, runner, startup)


def _unchanged_files(source: _LiveSource) -> dict[str, tuple[int, int, int, int, str]]:
    """Bind file identity as well as bytes, not just logical dependency versions."""
    selected = [
        source.environment / "Scripts" / "vipp-app.exe",
        source.environment / "Scripts" / "python.exe",
        source.environment / "pyvenv.cfg",
        source.environment / ".vipp-install-candidate.json",
        source.environment / protocol.DIRECTORY / protocol.CAPABILITY,
        *sorted((source.site / "dependency").rglob("*")),
        *sorted((source.site / "dependency-1.2.dist-info").rglob("*")),
    ]
    return {
        str(path.relative_to(source.environment)): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.stat().st_size,
            _sha(path.read_bytes()),
        )
        for path in selected
        if path.is_file()
    }


def _assert_protocol_released(source: _LiveSource) -> None:
    state = source.environment / protocol.DIRECTORY
    assert not (state / protocol.MAINTENANCE).exists()
    assert list((state / "leases").iterdir()) == []
    assert list((state / "permits").iterdir()) == []
    assert protocol.assess_capability(
        source.environment, source.old.installation_id
    ).eligible


def _ordinary_startup(source: _LiveSource, expected_version: str) -> None:
    python = source.runner.python
    environment = _vipp_launch_environment(python, default_services(), os.environ)
    result = source.runner.run(
        (str(python), "-I", "-B", "-c", source.startup_code),
        env=environment,
        cwd=source.root.parent / "documents",
    )
    assert result.returncode == 0, result.stderr
    probe = json.loads(result.stdout)
    assert probe["version"] == expected_version
    assert Path(probe["prefix"]) == source.environment
    assert Path(probe["package"]) == source.site / "napari_vipp" / "__init__.py"
    assert probe["dependency_value"] == 1
    assert probe["lease_registered"] is True
    _assert_protocol_released(source)


def test_actual_engine_inplace_updates_only_app_and_keeps_prefix(tmp_path):
    source = _live_source(tmp_path)
    preserved = _unchanged_files(source)
    parent_environment = dict(os.environ)
    prepared = source.engine.prepare(source.plan)
    assert prepared.applicable
    assert prepared.selected_strategy is UpdateStrategy.IN_PLACE, (
        prepared.strategy_reason
    )
    assert source.runner.calls == []  # No resolver, ensurepip, or installer ran.
    assert _unchanged_files(source) == preserved
    assert inspect_ownership(source.root).record == source.old

    result = source.engine.apply(
        prepared, source.engine.authorize(prepared, confirmed=True)
    )
    assert result.status is InstallStatus.SUCCEEDED, result.technical_error
    assert not result.rollback.attempted
    assert result.environment_root == source.environment
    current = inspect_ownership(source.root).record
    assert current is not None
    assert current.version == _NEW_VERSION
    assert current.installation_id == source.old.installation_id
    assert current.environment_root == source.old.environment_root
    assert current.base_python == source.old.base_python
    assert current.environment_marker_sha256 == source.old.environment_marker_sha256
    assert current.shortcuts == source.old.shortcuts == ()
    assert current.retired_environments == source.old.retired_environments == ()
    assert [p for p in current.packages if p.normalized_name != "napari-vipp"] == [
        p for p in source.old.packages if p.normalized_name != "napari-vipp"
    ]
    assert list(managed_environments_root(source.root).iterdir()) == [
        source.environment
    ]
    assert _unchanged_files(source) == preserved
    assert source.runner.permitted_calls == 2
    assert len(source.runner.calls) == 2
    assert source.runner.results[0].returncode == 0
    assert json.loads(source.runner.results[0].stdout)["app_value"] == 2
    _assert_protocol_released(source)
    _ordinary_startup(source, _NEW_VERSION)
    assert _unchanged_files(source) == preserved
    assert os.environ == parent_environment


def test_actual_engine_failed_acceptance_restores_app_and_normal_startup(tmp_path):
    source = _live_source(tmp_path, fail_acceptance=True)
    preserved = _unchanged_files(source)
    old_ownership = ownership_path(source.root).read_bytes()
    old_app = {
        p.relative_to(source.site).as_posix(): p.read_bytes()
        for directory in ("napari_vipp", f"napari_vipp-{_OLD_VERSION}.dist-info")
        for p in (source.site / directory).rglob("*")
        if p.is_file()
    }
    prepared = source.engine.prepare(source.plan)
    assert prepared.selected_strategy is UpdateStrategy.IN_PLACE, (
        prepared.strategy_reason
    )
    assert source.runner.calls == []
    result = source.engine.apply(
        prepared, source.engine.authorize(prepared, confirmed=True)
    )
    assert result.status is InstallStatus.FAILED
    assert result.rollback.attempted and result.rollback.completed, (
        result.rollback.errors
    )
    assert source.runner.permitted_calls == 1
    assert len(source.runner.calls) == 1
    assert source.runner.results[0].returncode == 23
    # The negative check really imported the new app under maintenance first.
    assert json.loads(source.runner.results[0].stdout)["app_value"] == 2
    assert ownership_path(source.root).read_bytes() == old_ownership
    assert inspect_ownership(source.root).record == source.old
    assert all(
        (source.site / name).read_bytes() == data for name, data in old_app.items()
    )
    assert not (source.site / f"napari_vipp-{_NEW_VERSION}.dist-info").exists()
    assert list(managed_environments_root(source.root).iterdir()) == [
        source.environment
    ]
    assert _unchanged_files(source) == preserved
    _assert_protocol_released(source)
    _ordinary_startup(source, _OLD_VERSION)
    assert _unchanged_files(source) == preserved


def test_actual_engine_cancel_after_app_backup_restores_and_releases_maintenance(
    tmp_path,
):
    source = _live_source(tmp_path)
    preserved = _unchanged_files(source)
    old_ownership = ownership_path(source.root).read_bytes()
    source.startup_code = _probe_code(_OLD_VERSION, 1)
    source.runner.allowed.add(
        (str(source.runner.python), "-I", "-B", "-c", source.startup_code)
    )
    prepared = source.engine.prepare(source.plan)
    assert prepared.selected_strategy is UpdateStrategy.IN_PLACE
    cancellation = CancellationToken()

    def observe(event):
        if event.stage is ProgressStage.INSTALLING and event.completed == 0:
            cancellation.cancel()

    result = source.engine.apply(
        prepared,
        source.engine.authorize(prepared, confirmed=True),
        progress=observe,
        cancellation=cancellation,
    )
    assert result.status is InstallStatus.CANCELLED
    assert result.rollback.completed, result.rollback.errors
    assert ownership_path(source.root).read_bytes() == old_ownership
    assert _unchanged_files(source) == preserved
    assert source.runner.calls == []
    _assert_protocol_released(source)
    _ordinary_startup(source, _OLD_VERSION)
    assert _unchanged_files(source) == preserved


def test_actual_owner_publication_then_writer_cleanup_error_stays_committed(
    tmp_path, monkeypatch
):
    """A late writer error cannot roll back an already authoritative new app."""
    from napari_vipp.installer import inplace_execution

    source = _live_source(tmp_path)
    preserved = _unchanged_files(source)
    old_ownership = ownership_path(source.root).read_bytes()
    prepared = source.engine.prepare(source.plan)
    assert prepared.selected_strategy is UpdateStrategy.IN_PLACE, (
        prepared.strategy_reason
    )
    assert source.runner.calls == []
    real_write = inplace_execution.write_ownership_record
    published = []

    def publish_then_cleanup_error(root, record):
        path = real_write(root, record)
        assert inspect_ownership(root).record == record
        published.append((path, record))
        raise OSError("Simulated writer cleanup failure after ownership publication")

    monkeypatch.setattr(
        inplace_execution, "write_ownership_record", publish_then_cleanup_error
    )
    result = source.engine.apply(
        prepared, source.engine.authorize(prepared, confirmed=True)
    )
    assert len(published) == 1
    assert published[0][0] == ownership_path(source.root)
    assert published[0][1].version == _NEW_VERSION
    assert result.status is InstallStatus.SUCCEEDED, result.technical_error
    assert not result.rollback.attempted
    assert result.rollback.completed, result.rollback.errors
    assert result.environment_root == source.environment
    current = inspect_ownership(source.root).record
    assert current == published[0][1]
    assert current is not None and current.version == _NEW_VERSION
    assert current.installation_id == source.old.installation_id
    assert current.environment_root == source.old.environment_root
    assert current.base_python == source.old.base_python
    assert ownership_path(source.root).read_bytes() != old_ownership
    assert (source.site / "napari_vipp/app.py").read_bytes() == (
        b"VALUE = 2\n\ndef main():\n    return 0\n"
    )
    assert not (source.site / f"napari_vipp-{_OLD_VERSION}.dist-info").exists()
    assert (source.site / f"napari_vipp-{_NEW_VERSION}.dist-info").is_dir()
    assert [p for p in current.packages if p.normalized_name != "napari-vipp"] == [
        p for p in source.old.packages if p.normalized_name != "napari-vipp"
    ]
    assert list(managed_environments_root(source.root).iterdir()) == [
        source.environment
    ]
    assert _unchanged_files(source) == preserved
    assert source.runner.permitted_calls == 2
    assert len(source.runner.calls) == 2
    assert all(result.returncode == 0 for result in source.runner.results)
    _assert_protocol_released(source)
    _ordinary_startup(source, _NEW_VERSION)
    assert _unchanged_files(source) == preserved
