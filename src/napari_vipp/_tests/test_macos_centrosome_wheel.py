"""The sole relocation exception must remain tied to reviewed native wheels."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import package_macos_installer as packager

ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = ROOT / "packaging/macos/centrosome"
SPEC = importlib.util.spec_from_file_location(
    "centrosome_wheel_guard", RECIPE_DIR / "verify_wheel.py"
)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def _load(library="/usr/lib/libSystem.B.dylib", command=0xC):
    name = library.encode() + b"\0"
    size = (24 + len(name) + 7) // 8 * 8
    return struct.pack("<6I", command, size, 24, 0, 0, 0) + name.ljust(size - 24, b"\0")


def _bundle(commands=None, *, cpu=0x01000007):
    commands = [_load()] if commands is None else commands
    payload = b"".join(commands)
    return (
        struct.pack("<8I", 0xFEEDFACF, cpu, 0, 8, len(commands), len(payload), 0, 0)
        + payload
    )


def _wheel(tmp_path, monkeypatch, *, missing=False, extra=None):
    filename, _, cpu = guard.WHEELS["osx-64"]
    wheel = tmp_path / filename
    members = sorted(guard.EXTENSIONS)
    if missing:
        members.pop()
    with ZipFile(wheel, "w") as archive:
        for name in members:
            archive.writestr(name, _bundle(cpu=cpu))
        if extra:
            archive.writestr(extra, _bundle(cpu=cpu))
    monkeypatch.setitem(
        guard.WHEELS,
        "osx-64",
        (filename, hashlib.sha256(wheel.read_bytes()).hexdigest(), cpu),
    )
    return wheel


def test_recipe_pins_match_guard_and_stages_validator(tmp_path):
    staged = tmp_path / "isolated-input/centrosome"
    packager._stage_centrosome_recipe(ROOT, staged)
    recipe = (staged / "recipe.yaml").read_text(encoding="utf-8")
    assert (staged / "verify_wheel.py").read_bytes() == (
        RECIPE_DIR / "verify_wheel.py"
    ).read_bytes()
    assert "number: 1" in recipe and "string: py312_vipp_1" in recipe
    assert "binary_relocation: false" in recipe
    assert (
        '"${RECIPE_DIR}/verify_wheel.py" --target "${target_platform}" '
        '--installed "${SP_DIR}"'
        in recipe
    )
    for target, (filename, digest, _cpu) in guard.WHEELS.items():
        section = recipe.split(f'if: target_platform == "{target}"', 1)[1]
        section = section.split("\n  - if:", 1)[0].split("\nbuild:", 1)[0]
        assert f"file_name: {filename}" in section
        assert f"sha256: {digest}" in section
    app_recipe = (ROOT / "packaging/macos/recipe/recipe.yaml.in").read_text()
    assert "binary_relocation: false" not in app_recipe


@pytest.mark.parametrize("cpu", [0x01000007, 0x0100000C])
def test_reviewed_system_only_dependencies_need_no_relocation(cpu):
    data = _bundle([_load(), _load("/usr/lib/libc++.1.dylib")], cpu=cpu)
    assert set(guard.verify_macho(data, cpu=cpu)) == guard.SYSTEM_LIBRARIES
    assert data == _bundle([_load(), _load("/usr/lib/libc++.1.dylib")], cpu=cpu)


@pytest.mark.parametrize(
    "library",
    [
        "@rpath/libprivate.dylib",
        "@loader_path/libprivate.dylib",
        "libprivate.dylib",
        "/temporary/build/prefix/lib/libprivate.dylib",
        "/usr/lib/unreviewed.dylib",
    ],
)
def test_unreviewed_native_loads_fail_closed(library):
    with pytest.raises(ValueError, match="Unreviewed native dependency"):
        guard.verify_macho(_bundle([_load(), _load(library)]), cpu=0x01000007)


@pytest.mark.parametrize("command", [0x8000001C, 0x80000018, 0x8000001F, 0x27, 0xFFFF])
def test_rpath_and_unreviewed_load_commands_fail_closed(command):
    with pytest.raises(ValueError, match="Unreviewed Mach-O load command"):
        guard.verify_macho(_bundle([_load(), _load(command=command)]), cpu=0x01000007)


@pytest.mark.parametrize(
    "data",
    [
        b"short",
        _bundle(cpu=0x0100000C),
        _bundle()[:-1],
        _bundle([struct.pack("<2I", 0xC, 0)]),
        _bundle([struct.pack("<2I", 0xC, 8)]),
        _bundle([struct.pack("<6I", 0xC, 24, 4, 0, 0, 0)]),
        _bundle([struct.pack("<6I", 0xC, 32, 24, 0, 0, 0) + b"no-null!"]),
    ],
)
def test_invalid_or_wrong_architecture_bundles_fail_closed(data):
    with pytest.raises(ValueError):
        guard.verify_macho(data, cpu=0x01000007)


def test_all_six_extensions_checked_and_installed_bytes_preserved(
    tmp_path, monkeypatch
):
    wheel = _wheel(tmp_path, monkeypatch)
    installed = tmp_path / "site-packages"
    for name in guard.EXTENSIONS:
        path = installed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_bundle())
    before = wheel.read_bytes()
    report = guard.verify_wheel(wheel, "osx-64", installed)
    assert set(report["extensions"]) == guard.EXTENSIONS
    assert wheel.read_bytes() == before
    first = installed / sorted(guard.EXTENSIONS)[0]
    first.write_bytes(_bundle() + b"changed")
    with pytest.raises(ValueError, match="differs from upstream"):
        guard.verify_wheel(wheel, "osx-64", installed)


@pytest.mark.parametrize("extra", ["centrosome/private.dylib", "hidden/no-extension"])
def test_unreviewed_native_members_cannot_skip_validation(tmp_path, monkeypatch, extra):
    wheel = _wheel(tmp_path, monkeypatch, extra=extra)
    with pytest.raises(ValueError, match="Unreviewed native wheel member"):
        guard.verify_wheel(wheel, "osx-64")


def test_missing_extension_is_rejected(tmp_path, monkeypatch):
    wheel = _wheel(tmp_path, monkeypatch, missing=True)
    with pytest.raises(ValueError, match="all six"):
        guard.verify_wheel(wheel, "osx-64")


def test_upstream_wheel_hash_is_mandatory(tmp_path, monkeypatch):
    wheel = _wheel(tmp_path, monkeypatch)
    wheel.write_bytes(wheel.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="hash-pinned"):
        guard.verify_wheel(wheel, "osx-64")


@pytest.mark.parametrize(
    "filename", ["macos-installer.yml", "unsigned-installers-release.yml"]
)
def test_native_workflows_run_relocation_guard_regressions(filename):
    workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
    assert "src/napari_vipp/_tests/test_macos_centrosome_wheel.py" in workflow
