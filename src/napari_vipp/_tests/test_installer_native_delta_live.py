"""Real private DLL exchange and rollback in disposable Windows prefixes.

Only the allowlisted fixture children load these two read-only System32 DLL
copies. There is no host installation, network, pip, GUI or GPU operation. The
known process inventory is injected empty by the existing isolated fixture;
Windows DLL loading, maintenance permits and changed-file exchanges are real.
"""

from __future__ import annotations

import json
import os
import stat
import struct
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp._tests import test_installer_delta_dependencies_live as dependency_live
from napari_vipp._tests.test_installer_incremental import _distribution
from napari_vipp._tests.test_installer_inplace_live import _sha
from napari_vipp.installer.ownership import write_ownership_record

pytestmark = pytest.mark.skipif(
    os.name != "nt"
    or sys.version_info[:2] not in {(3, 12), (3, 13)}
    or struct.calcsize("P") != 8,
    reason="Actual 64-bit Windows CPython 3.12/3.13 DLL loading is required.",
)

_PRIVATE_DLL = "vipp_delta_native_fixture.dll"
_NATIVE_WHEEL = "dependency-2.0-py3-none-win_amd64.whl"


def _system_dll_bytes(name):
    """Read exactly one fixed OS fixture, never write or load its shared path."""
    assert name in {"version.dll", "winmm.dll"}
    system = Path(os.environ["SystemRoot"]) / "System32" / name
    assert system.is_absolute()
    for path in (system, *system.parents):
        metadata = path.lstat()
        assert not stat.S_ISLNK(metadata.st_mode)
        assert not getattr(metadata, "st_file_attributes", 0) & 0x400
    assert system.is_file() and 0 < system.stat().st_size <= 8 * 1024**2
    data = system.read_bytes()
    assert data[:2] == b"MZ"
    pe = struct.unpack_from("<I", data, 60)[0]
    assert data[pe : pe + 4] == b"PE\0\0"
    assert struct.unpack_from("<H", data, pe + 4)[0] == 0x8664
    return data


def _native_dependency(value):
    return (
        "import ctypes\n"
        "from pathlib import Path\n"
        f"VALUE = {value}\n"
        f"NATIVE_PATH = Path(__file__).with_name({_PRIVATE_DLL!r})\n"
        "NATIVE_HANDLE = ctypes.WinDLL(str(NATIVE_PATH))\n"
    ).encode()


def _native_probe(original, old_bytes, new_bytes):
    def probe(source, *, new, fail=False):
        code = original(source, new=new, fail=fail)
        native = source.site / "dependency" / _PRIVATE_DLL
        digest = _sha(new_bytes if new else old_bytes)
        anchor = "print(json.dumps("
        assert code.count(anchor) == 1
        assertions = (
            "import hashlib;"
            "assert dependency.NATIVE_HANDLE._handle > 0;"
            f"assert dependency.NATIVE_PATH == Path({str(native)!r});"
            f"assert Path(dependency.NATIVE_HANDLE._name) == Path({str(native)!r});"
            "assert hashlib.sha256(dependency.NATIVE_PATH.read_bytes()).hexdigest()"
            f" == {digest!r};"
        )
        # Keep every existing app/version/prefix/runtime and scientific-import
        # assertion. Extend its evidence, without replacing its exit-23 branch.
        code = code.replace(anchor, assertions + anchor)
        original_fields = "'prefix':sys.prefix}"
        assert code.count(original_fields) == 1
        return code.replace(
            original_fields,
            "'prefix':sys.prefix,"
            "'native_handle':dependency.NATIVE_HANDLE._handle,"
            "'native_path':str(dependency.NATIVE_PATH),"
            "'native_sha256':hashlib.sha256("
            "dependency.NATIVE_PATH.read_bytes()).hexdigest()}",
        )

    return probe


@pytest.mark.parametrize(
    "fail", [False, True], ids=["accepts-private-dll", "restores-private-dll"]
)
def test_actual_native_dependency_delta_loads_and_restores_private_dll(
    tmp_path, monkeypatch, fail
):
    old_bytes, new_bytes = (
        _system_dll_bytes("version.dll"),
        _system_dll_bytes("winmm.dll"),
    )
    assert old_bytes != new_bytes and _sha(old_bytes) != _sha(new_bytes)
    original_write = dependency_live._write_wheel
    original_artifact = dependency_live.PackageArtifact
    original_source = dependency_live._required_delta_source
    original_probe = dependency_live._probe
    written_native, native_artifacts, fixtures = [], [], []

    def write_native_wheel(path, files):
        if "dependency-2.0.dist-info/METADATA" in files:
            files = dict(files)
            assert files["dependency/__init__.py"] == b"VALUE = 2\n"
            files["dependency/__init__.py"] = _native_dependency(2)
            files[f"dependency/{_PRIVATE_DLL}"] = new_bytes
            files["dependency-2.0.dist-info/WHEEL"] = (
                b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-win_amd64\n"
            )
            path = path.with_name(_NATIVE_WHEEL)
            written_native.append(path)
        # The original writer builds a fresh strict RECORD after this injection,
        # before the fixture computes its resolved/cache wheel SHA-256.
        return original_write(path, files)

    def native_artifact(name, version, url, digest):
        assert (name, version) == ("dependency", "2.0")
        assert url.endswith("/dependency-2.0-py3-none-any.whl")
        url = url.rsplit("/", 1)[0] + "/" + _NATIVE_WHEEL
        artifact = original_artifact(name, version, url, digest)
        native_artifacts.append(artifact)
        return artifact

    def native_source(path, patch, *, fail=False):
        fixture = original_source(path, patch, fail=fail)
        source, cached, _, _, _ = fixture
        previous = next(
            p for p in source.old.packages if p.normalized_name == "dependency"
        )
        old = _distribution(
            source.site,
            "dependency",
            "1.2",
            tags="py3-none-win_amd64",
            files={
                "dependency/__init__.py": _native_dependency(1),
                f"dependency/{_PRIVATE_DLL}": old_bytes,
            },
        )
        assert old == previous  # same owned package identity, new strict RECORD
        source.old = replace(
            source.old,
            packages=tuple(
                old if p.normalized_name == "dependency" else p
                for p in source.old.packages
            ),
        )
        write_ownership_record(source.root, source.old)
        native = source.site / "dependency" / _PRIVATE_DLL
        assert native.read_bytes() == old_bytes
        with zipfile.ZipFile(cached) as archive:
            assert archive.read(f"dependency/{_PRIVATE_DLL}") == new_bytes
            assert b"Tag: py3-none-win_amd64" in archive.read(
                "dependency-2.0.dist-info/WHEEL"
            )
        assert cached.name == _NATIVE_WHEEL
        fixtures.append((source, native, cached))
        return fixture

    monkeypatch.setattr(dependency_live, "_write_wheel", write_native_wheel)
    monkeypatch.setattr(dependency_live, "PackageArtifact", native_artifact)
    monkeypatch.setattr(
        dependency_live, "_probe", _native_probe(original_probe, old_bytes, new_bytes)
    )
    monkeypatch.setattr(dependency_live, "_required_delta_source", native_source)

    # Reuse the complete proven acceptance/rollback/cache/no-clone contract.
    # It additionally verifies prefix identity, unchanged sentinel inodes/hashes,
    # exact old RECORD restoration on failure, one report/no download, normal
    # startup, scoped child permits, released protocol and unchanged global env.
    dependency_live.test_actual_required_dependency_delta_uses_cache_and_never_clones(
        tmp_path, monkeypatch, fail
    )

    assert len(written_native) == len(native_artifacts) == len(fixtures) == 1
    source, native, cached = fixtures[0]
    assert cached.read_bytes() == written_native[0].read_bytes()
    assert _sha(cached.read_bytes()) == native_artifacts[0].sha256
    assert native.read_bytes() == (old_bytes if fail else new_bytes)
    acceptance = source.runner.results[0]
    accepted = json.loads(acceptance.stdout)
    assert acceptance.returncode == (23 if fail else 0)
    assert accepted["native_handle"] > 0
    assert accepted["native_path"] == str(native)
    assert accepted["native_sha256"] == _sha(new_bytes)
    startup = source.runner.results[-1]
    started = json.loads(startup.stdout)
    assert startup.returncode == 0  # bounded runner communicate/reap completed
    assert started["native_handle"] > 0
    assert started["native_path"] == str(native)
    assert started["native_sha256"] == _sha(old_bytes if fail else new_bytes)
