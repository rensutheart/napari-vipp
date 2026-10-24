from __future__ import annotations

from dataclasses import replace

import pytest

from napari_vipp._tests.test_installer_engine import _plan
from napari_vipp._tests.test_installer_incremental import _wheel
from napari_vipp._tests.test_installer_incremental import source as source_fixture
from napari_vipp.installer.engine import (
    ResolutionError,
    ResolvedPackage,
    _merge_installed_resolution,
    _resolution_argv,
)
from napari_vipp.installer.incremental import assess_incremental, assess_installed

source = source_fixture


def test_installed_review_allows_required_dependency_change_without_copy(
    source, tmp_path
):
    record, _, python, _ = source
    incoming = _wheel(
        tmp_path / "new.whl", requirements=("dependency>=2", "new-dep>=1")
    )
    assert not assess_incremental(record, incoming, python).eligible
    reviewed = assess_installed(record, incoming, python)
    assert reviewed.eligible, reviewed.reason
    assert reviewed.transaction_kind == "package_delta"
    assert {p.name: p.version for p in reviewed.packages} == {
        "napari-vipp": "1.0a1",
        "dependency": "1.2",
    }
    final = _merge_installed_resolution(
        reviewed,
        (
            ResolvedPackage(
                "napari-vipp",
                incoming.version,
                incoming.wheel_path.as_uri(),
                incoming.wheel_sha256,
                True,
            ),
            ResolvedPackage(
                "dependency", "2.0", "https://files.pythonhosted.org/d.whl", "c" * 64
            ),
            ResolvedPackage(
                "new-dep", "1.0", "https://files.pythonhosted.org/n.whl", "d" * 64
            ),
        ),
    )
    assert {p.name: p.version for p in final} == {
        "napari-vipp": incoming.version,
        "dependency": "2.0",
        "new-dep": "1.0",
    }


def test_compatible_dependency_stays_at_installed_version_and_exact_wheel_identity(
    source,
):
    record, release, python, _ = source
    reviewed = assess_installed(record, release, python)
    incoming = ResolvedPackage(
        "napari-vipp",
        release.version,
        release.wheel_path.as_uri(),
        release.wheel_sha256,
        True,
    )
    final = _merge_installed_resolution(reviewed, (incoming,))
    retained = next(p for p in final if p.name == "dependency")
    owned = next(p for p in record.packages if p.name == "dependency")
    assert retained.version == "1.2"
    assert retained.sha256 == owned.sha256
    assert retained.source_url == "reuse://installed/dependency"


def test_inventory_review_binds_metadata_even_when_constraint_changes(source):
    record, release, python, site = source
    before = assess_installed(record, release, python)
    path = site / "dependency-1.2.dist-info/RECORD"
    path.write_bytes(path.read_bytes() + b"\n")
    after = assess_installed(record, release, python)
    assert not after.eligible or after.fingerprint != before.fingerprint


def test_installed_resolver_sees_current_packages_and_does_not_ignore_them(tmp_path):
    plan = _plan(tmp_path / "managed", _wheel(tmp_path / "new.whl"))
    python = tmp_path / "owned/Scripts/python.exe"
    argv = _resolution_argv(
        plan, index_url="https://pypi.org/simple", installed_python=python
    )
    assert argv[:5] == (str(python), "-I", "-B", "-m", "pip")
    assert "--ignore-installed" not in argv
    assert "--target" not in argv
    assert "--upgrade" not in argv
    assert argv[argv.index("--upgrade-strategy") + 1] == "only-if-needed"
    assert "--dry-run" in argv and "--only-binary=:all:" in argv
    fresh = _resolution_argv(plan, index_url="https://pypi.org/simple")
    assert "--ignore-installed" in fresh


def test_delta_resolution_requires_authenticated_inventory_and_one_app(source):
    record, release, python, _ = source
    reviewed = assess_installed(record, release, python)
    with pytest.raises(ResolutionError):
        _merge_installed_resolution(reviewed, ())
    app = ResolvedPackage(
        "napari-vipp",
        release.version,
        release.wheel_path.as_uri(),
        release.wheel_sha256,
        True,
    )
    with pytest.raises(ResolutionError):
        _merge_installed_resolution(reviewed, (app, app))
    with pytest.raises(ResolutionError):
        _merge_installed_resolution(replace(reviewed, eligible=False), (app,))
