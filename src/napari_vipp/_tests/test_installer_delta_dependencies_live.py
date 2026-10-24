"""Real tiny-prefix changed-dependency transactions, never a host installation.

The fixture's selected process set is injected empty; no user process inventory
is queried. Only exact allowlisted acceptance/startup children execute. The pip
report is controlled metadata, and verified cache entries are seeded from local
fixture bytes, so no pip, network, scientific package, installer or GUI runs.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import struct
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import replace

import pytest

from napari_vipp import _runtime_lease as rt
from napari_vipp._tests.test_installer_incremental import _distribution
from napari_vipp._tests.test_installer_inplace_live import (
    _NEW_VERSION,
    _OLD_VERSION,
    _assert_protocol_released,
    _sha,
)
from napari_vipp._tests.test_installer_package_delta_live import _legacy_source
from napari_vipp.installer import engine as engine_module
from napari_vipp.installer import incremental, legacy_inplace
from napari_vipp.installer.discovery import default_services
from napari_vipp.installer.engine import (
    AuthorizationError,
    CommandResult,
    InstallStatus,
    StalePreparedTransaction,
)
from napari_vipp.installer.models import PlannedAction, UpdateStrategy
from napari_vipp.installer.ownership import (
    inspect_ownership,
    managed_environments_root,
    ownership_path,
    write_ownership_record,
)
from napari_vipp.installer.package_cache import PackageArtifact, cache_package
from napari_vipp.installer.windows_backend import _vipp_launch_environment

pytestmark = pytest.mark.skipif(
    os.name != "nt"
    or sys.version_info[:2] not in {(3, 12), (3, 13)}
    or struct.calcsize("P") != 8,
    reason="Actual 64-bit Windows CPython 3.12/3.13 children are required.",
)


def _write_wheel(path, files):
    """Author tiny fixture ZIP/RECORD bytes, without a package build or pip."""
    infos = [name for name in files if name.endswith(".dist-info/METADATA")]
    assert len(infos) == 1
    record_name = infos[0].rsplit("/", 1)[0] + "/RECORD"
    payload = {name: data for name, data in files.items() if name != record_name}
    rows = [
        (
            name,
            "sha256="
            + base64.urlsafe_b64encode(bytes.fromhex(_sha(data))).decode().rstrip("="),
            str(len(data)),
        )
        for name, data in payload.items()
    ]
    rows.append((record_name, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    payload[record_name] = output.getvalue().encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    return path.read_bytes()


def _probe(source, *, new, fail=False):
    version, app_value, dependency_value = (
        (_NEW_VERSION, 2, 2) if new else (_OLD_VERSION, 1, 1)
    )
    dependency_version = "2.0" if new else "1.2"
    code = (
        "import importlib.metadata as m,json,sys;from pathlib import Path;"
        "import napari_vipp,napari_vipp.app,dependency;"
        f"assert napari_vipp.__version__=={version!r};"
        f"assert m.version('dependency')=={dependency_version!r};"
        f"assert napari_vipp.app.VALUE=={app_value};"
        f"assert dependency.VALUE=={dependency_value};"
        f"assert Path(sys.prefix)==Path({str(source.environment)!r});"
        "assert Path(dependency.__file__)=="
        f"Path({str(source.site / 'dependency/__init__.py')!r});"
        "assert not any(n in sys.modules for n in ('numpy','qtpy','PySide6'));"
    )
    if new:
        code += "assert napari_vipp._runtime_use_lease is not None;"
    code += (
        "print(json.dumps({'version':napari_vipp.__version__,"
        "'app_value':napari_vipp.app.VALUE,'dependency_value':dependency.VALUE,"
        "'dependency_version':m.version('dependency'),'prefix':sys.prefix}),flush=True);"
        + ("sys.exit(23)" if fail else "sys.exit(0)")
    )
    return code


def _stamps(source):
    paths = [
        source.environment / "Scripts/python.exe",
        source.environment / "Scripts/vipp-app.exe",
        source.environment / "pyvenv.cfg",
        source.environment / ".vipp-install-candidate.json",
        *sorted((source.site / "sentinel").rglob("*")),
        *sorted((source.site / "sentinel-1.0.dist-info").rglob("*")),
    ]
    return {
        path.relative_to(source.environment).as_posix(): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha(path.read_bytes()),
        )
        for path in paths
        if path.is_file()
    }


def _dependency_bytes(source):
    return {
        path.relative_to(source.site).as_posix(): path.read_bytes()
        for folder in (
            source.site / "dependency",
            source.site / "dependency-1.2.dist-info",
        )
        for path in folder.rglob("*")
        if path.is_file()
    }


def _assert_no_environment_clone(source):
    """Retained small transaction controls are evidence, not a cloned prefix."""
    for path in managed_environments_root(source.root).iterdir():
        if path == source.environment:
            continue
        assert path.name.startswith(".vipp-package-delta-")
        assert not (path / "pyvenv.cfg").exists()
        assert not (path / "Scripts/python.exe").exists()
        assert not (path / "payload").exists()


def _required_delta_source(tmp_path, monkeypatch, *, fail=False):
    source = _legacy_source(tmp_path, monkeypatch)
    sentinel = _distribution(
        source.site,
        "sentinel",
        "1.0",
        files={
            "sentinel/__init__.py": b"VALUE = 713\n",
            "sentinel/data.bin": b"untouched",
        },
    )
    source.old = replace(source.old, packages=(*source.old.packages, sentinel))
    write_ownership_record(source.root, source.old)
    with zipfile.ZipFile(source.plan.release.wheel_path) as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}
    metadata = f"napari_vipp-{_NEW_VERSION}.dist-info/METADATA"
    assert payload[metadata].count(b"dependency>=1") == 1
    payload[metadata] = payload[metadata].replace(b"dependency>=1", b"dependency>=2")
    app_path = tmp_path / "incoming" / source.plan.release.wheel_path.name
    app_bytes = _write_wheel(app_path, payload)
    release = replace(
        source.plan.release, wheel_path=app_path, wheel_sha256=_sha(app_bytes)
    )
    code = _probe(source, new=True, fail=fail)
    source.runner.allowed.add((str(source.runner.python), "-I", "-B", "-c", code))
    source.plan = replace(
        source.plan,
        release=release,
        acceptance=(
            PlannedAction(
                "required-dependency-live",
                "Import the actual updated dependency",
                (str(source.runner.python), "-I", "-B", "-c", code),
            ),
        ),
    )
    info = "dependency-2.0.dist-info"
    dependency_path = tmp_path / "fixture-cache-input/dependency-2.0-py3-none-any.whl"
    dependency_bytes = _write_wheel(
        dependency_path,
        {
            "dependency/__init__.py": b"VALUE = 2\n",
            f"{info}/METADATA": (
                b"Metadata-Version: 2.3\nName: dependency\nVersion: 2.0\n"
                b"Requires-Python: >=3.12\n\n"
            ),
            f"{info}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        },
    )
    artifact = PackageArtifact(
        "dependency",
        "2.0",
        "https://files.pythonhosted.org/packages/unit-fixture/dependency-2.0-py3-none-any.whl",
        _sha(dependency_bytes),
    )
    fetched = []

    @contextmanager
    def fixture_bytes(url):
        assert url == artifact.source_url
        fetched.append(url)
        yield io.BytesIO(dependency_bytes)

    state = source.engine._state_root
    cached = cache_package(state / "cache/wheels", artifact, fetcher=fixture_bytes)
    assert len(fetched) == 1
    cache_stamp = (
        cached.stat().st_ino,
        cached.stat().st_size,
        _sha(cached.read_bytes()),
    )
    expected_resolver = engine_module._resolution_argv(
        source.plan,
        index_url=source.engine._index_url,
        cache_directory=state / "cache/pip",
        installed_python=source.runner.python,
    )
    report = {
        "install": [
            {
                "metadata": {"name": "napari-vipp", "version": _NEW_VERSION},
                "requested": True,
                "download_info": {
                    "url": app_path.as_uri(),
                    "archive_info": {"hashes": {"sha256": release.wheel_sha256}},
                },
            },
            {
                "metadata": {"name": "dependency", "version": "2.0"},
                "requested": False,
                "download_info": {
                    "url": artifact.source_url,
                    "archive_info": {"hashes": {"sha256": artifact.sha256}},
                },
            },
        ]
    }
    report_calls, forbidden_downloads = [], []
    original = source.runner.run

    def controlled_report(argv, **kwargs):
        call = tuple(str(value) for value in argv)
        if call == expected_resolver:
            assert "--dry-run" in call and "--report" in call
            assert "--ignore-installed" not in call and "--target" not in call
            assert call[0] == str(source.environment / "Scripts/python.exe")
            assert call[call.index("--upgrade-strategy") + 1] == "only-if-needed"
            assert inspect_ownership(source.root).record == source.old
            assert "PYTHONPATH" not in kwargs["env"]
            report_calls.append(call)
            # This report is controlled metadata, never an executed pip command.
            return CommandResult(0, json.dumps(report), "")
        assert "pip" not in call and "venv" not in call and "ensurepip" not in call
        return original(call, **kwargs)

    monkeypatch.setattr(source.runner, "run", controlled_report)
    checked = source.engine._run_checked

    def no_download(argv, **kwargs):
        if "download" in argv:
            forbidden_downloads.append(tuple(argv))
            raise AssertionError("A verified cache hit must not download")
        return checked(argv, **kwargs)

    monkeypatch.setattr(source.engine, "_run_checked", no_download)

    def no_copy(*args, **kwargs):
        raise AssertionError(
            "Changed-package update cannot copy unchanged dependencies"
        )

    monkeypatch.setattr(incremental, "copy_reused_packages", no_copy)
    return source, cached, cache_stamp, report_calls, forbidden_downloads


@pytest.mark.parametrize(
    "fail", [False, True], ids=["accepts-changed-dependency", "restores-old-dependency"]
)
def test_actual_required_dependency_delta_uses_cache_and_never_clones(
    tmp_path, monkeypatch, fail
):
    source, cached, cache_stamp, reports, downloads = _required_delta_source(
        tmp_path, monkeypatch, fail=fail
    )
    preserved, dependency, owner, parent_env = (
        _stamps(source),
        _dependency_bytes(source),
        ownership_path(source.root).read_bytes(),
        dict(os.environ),
    )
    prefix_identity = (
        source.environment.stat().st_dev,
        source.environment.stat().st_ino,
    )
    prepared = source.engine.prepare(source.plan, resolve=False)
    assert prepared.selected_strategy is UpdateStrategy.IN_PLACE
    assert prepared.reuse_plan.transaction_kind == "package_delta"
    assert prepared.deferred_resolution and not prepared.resolution_complete
    assert reports == [] and source.runner.calls == []
    result = source.engine.apply(
        prepared, source.engine.authorize(prepared, confirmed=True)
    )
    assert len(reports) == 1 and downloads == []
    assert _stamps(source) == preserved
    assert (
        source.environment.stat().st_dev,
        source.environment.stat().st_ino,
    ) == prefix_identity
    assert (
        cached.stat().st_ino,
        cached.stat().st_size,
        _sha(cached.read_bytes()),
    ) == cache_stamp
    _assert_no_environment_clone(source)
    assert os.environ == parent_env
    assert (
        source.runner.results
        and json.loads(source.runner.results[0].stdout)["dependency_value"] == 2
    )
    assert json.loads(source.runner.results[0].stdout)["dependency_version"] == "2.0"
    if fail:
        assert result.status is InstallStatus.FAILED
        assert result.rollback.attempted and result.rollback.completed, (
            result.rollback.errors
        )
        assert source.runner.results[0].returncode == 23
        assert ownership_path(source.root).read_bytes() == owner
        assert inspect_ownership(source.root).record == source.old
        assert _dependency_bytes(source) == dependency
        assert not (source.site / "dependency-2.0.dist-info").exists()
        assert not (source.site / legacy_inplace.GUARD_NAME).exists()
        assert not (source.environment / rt.DIRECTORY).exists()
    else:
        assert result.status is InstallStatus.SUCCEEDED, result.technical_error + str(
            result.rollback.errors
        )
        accepted = inspect_ownership(source.root).record
        assert (
            accepted is not None
            and accepted.installation_id == source.old.installation_id
        )
        assert accepted.environment_root == source.environment
        assert {p.normalized_name: p.version for p in accepted.packages}[
            "dependency"
        ] == "2.0"
        assert not (source.site / "dependency-1.2.dist-info").exists()
        assert (source.site / "dependency-2.0.dist-info").is_dir()
        assert source.runner.permitted_calls == 2
        _assert_protocol_released(source)
    startup = _probe(source, new=not fail)
    source.runner.allowed.add((str(source.runner.python), "-I", "-B", "-c", startup))
    env = _vipp_launch_environment(source.runner.python, default_services(), os.environ)
    ordinary = source.runner.run(
        (str(source.runner.python), "-I", "-B", "-c", startup),
        env=env,
        cwd=tmp_path / "documents",
    )
    assert ordinary.returncode == 0, ordinary.stderr
    assert json.loads(ordinary.stdout)["dependency_value"] == (1 if fail else 2)
    assert _stamps(source) == preserved
    if not fail:
        _assert_protocol_released(source)


def test_stale_deferred_inventory_aborts_before_resolver_or_download(
    tmp_path, monkeypatch
):
    source, cached, cache_stamp, reports, downloads = _required_delta_source(
        tmp_path, monkeypatch
    )
    prepared = source.engine.prepare(source.plan, resolve=False)
    assert (
        prepared.deferred_resolution
        and prepared.reuse_plan.transaction_kind == "package_delta"
    )
    authorization = source.engine.authorize(prepared, confirmed=True)
    metadata = source.site / "sentinel-1.0.dist-info/METADATA"
    assert metadata.read_bytes().count(b"\n\n") == 1
    metadata.write_bytes(
        metadata.read_bytes().replace(b"\n\n", b"\nSummary: changed after review\n\n")
    )
    changed_metadata = metadata.read_bytes()
    owner = ownership_path(source.root).read_bytes()
    dependency = _dependency_bytes(source)
    with pytest.raises(StalePreparedTransaction, match="changed"):
        source.engine.apply(prepared, authorization)
    assert reports == [] and downloads == [] and source.runner.calls == []
    assert metadata.read_bytes() == changed_metadata
    assert ownership_path(source.root).read_bytes() == owner
    assert _dependency_bytes(source) == dependency
    assert not list((source.engine._state_root / "transactions").glob("*.json"))
    assert not (source.engine._state_root / "runs").exists()
    assert not (source.environment / rt.DIRECTORY).exists()
    assert list(managed_environments_root(source.root).iterdir()) == [
        source.environment
    ]
    assert (
        cached.stat().st_ino,
        cached.stat().st_size,
        _sha(cached.read_bytes()),
    ) == cache_stamp
    assert (
        source.engine._authorizations == {}
    )  # consumed once; no replacement consent issued
    with pytest.raises(AuthorizationError):
        source.engine.apply(prepared, authorization)
    assert reports == [] and downloads == [] and source.runner.calls == []
