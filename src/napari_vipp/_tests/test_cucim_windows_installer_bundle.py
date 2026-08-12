from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGER_PATH = REPO_ROOT / "scripts" / "package_cucim_windows_installer.py"
SOURCE_MANIFEST_PATH = REPO_ROOT / "MANIFEST.in"
SOURCE_COMMIT = "1" * 40


def _load_packager():
    name = "_test_napari_vipp_cucim_windows_installer_packager"
    spec = importlib.util.spec_from_file_location(name, PACKAGER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packager():
    return _load_packager()


def test_bundle_is_deterministic_complete_and_contains_no_wheel(packager, tmp_path):
    first = packager.create_bundle_plan(
        repository_root=REPO_ROOT,
        output_path=tmp_path / "first.zip",
        source_commit=SOURCE_COMMIT,
    )
    second = packager.create_bundle_plan(
        repository_root=REPO_ROOT,
        output_path=tmp_path / "second.zip",
        source_commit=SOURCE_COMMIT,
    )

    first_hash = packager.write_bundle(first)
    second_hash = packager.write_bundle(second)

    assert first.output_path.read_bytes() == second.output_path.read_bytes()
    assert (
        first_hash
        == second_hash
        == hashlib.sha256(first.output_path.read_bytes()).hexdigest()
    )

    expected = {
        "Install VIPP cuCIM.cmd",
        "LICENSE",
        "NOTICE",
        "README.md",
        "bundle-manifest.json",
        "scripts/build_cucim_windows.ps1",
        "scripts/install_cucim_windows.cmd",
        "scripts/install_cucim_windows.ps1",
        "scripts/install_cucim_windows.py",
        "scripts/setup_gpu_dev.py",
    }
    with zipfile.ZipFile(first.output_path) as archive:
        assert set(archive.namelist()) == expected
        assert not any(name.lower().endswith(".whl") for name in archive.namelist())
        for info in archive.infolist():
            assert info.date_time == packager.FIXED_ZIP_TIMESTAMP
            assert info.compress_type == zipfile.ZIP_STORED
            assert stat.S_IFMT(info.external_attr >> 16) == stat.S_IFREG
        manifest = json.loads(archive.read("bundle-manifest.json"))
        assert manifest["vipp_version"] == "0.13.0a4"
        assert manifest["source_commit"] == SOURCE_COMMIT
        assert manifest["entrypoint"] == "Install VIPP cuCIM.cmd"
        assert manifest["contains_prebuilt_cucim_wheel"] is False
        assert set(manifest["files"]) == expected - {"bundle-manifest.json"}
        for name, record in manifest["files"].items():
            contents = archive.read(name)
            assert record == {
                "sha256": hashlib.sha256(contents).hexdigest(),
                "size_bytes": len(contents),
            }


def test_bundle_refuses_overwrite(packager, tmp_path):
    plan = packager.create_bundle_plan(
        repository_root=REPO_ROOT,
        output_path=tmp_path / "installer.zip",
        source_commit=SOURCE_COMMIT,
    )
    packager.write_bundle(plan)
    original = plan.output_path.read_bytes()

    with pytest.raises(packager.BundleError, match="Refusing to overwrite"):
        packager.write_bundle(plan)

    assert plan.output_path.read_bytes() == original


def test_bundle_plan_only_prints_hash_manifest_without_writing(
    packager,
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        packager,
        "_validated_clean_source_commit",
        lambda _root: SOURCE_COMMIT,
    )
    output = tmp_path / "planned.zip"
    exit_code = packager.main(["--output", str(output), "--plan-only"])

    assert exit_code == 0
    assert not output.exists()
    document = json.loads(capsys.readouterr().out)
    assert document["plan_only"] is True
    assert document["output"] == str(output.resolve())
    assert document["manifest"]["vipp_version"] == "0.13.0a4"
    assert document["manifest"]["source_commit"] == SOURCE_COMMIT
    assert all(
        len(record["sha256"]) == 64 for record in document["manifest"]["files"].values()
    )


def test_bundle_version_must_match_existing_environment_contract(
    packager,
    tmp_path,
):
    root = tmp_path / "repository"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "napari-vipp"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    (scripts / "setup_gpu_dev.py").write_text(
        'VIPP_RELEASE_VERSION = "0.13.0a4"\n',
        encoding="utf-8",
    )

    with pytest.raises(packager.BundleError, match="contract differ"):
        packager.create_bundle_plan(
            repository_root=root,
            output_path=tmp_path / "invalid.zip",
            source_commit=SOURCE_COMMIT,
        )


def test_bundle_rejects_invalid_source_revision(packager, tmp_path):
    with pytest.raises(packager.BundleError, match="source commit is invalid"):
        packager.create_bundle_plan(
            repository_root=REPO_ROOT,
            output_path=tmp_path / "invalid-source.zip",
            source_commit="dirty",
        )


def test_source_distribution_includes_double_click_installer_payload():
    manifest = SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
    scripts_rule = next(
        line
        for line in manifest.splitlines()
        if line.startswith("recursive-include scripts")
    )
    assert "*.cmd" in scripts_rule
    assert "*.md" in scripts_rule


@pytest.mark.skipif(sys.platform != "win32", reason="Windows cmd smoke")
def test_extracted_root_click_target_delegates_from_path_with_spaces(
    packager,
    tmp_path,
):
    plan = packager.create_bundle_plan(
        repository_root=REPO_ROOT,
        output_path=tmp_path / "installer.zip",
        source_commit=SOURCE_COMMIT,
    )
    packager.write_bundle(plan)
    extracted = tmp_path / "Extracted VIPP Installer"
    with zipfile.ZipFile(plan.output_path) as archive:
        archive.extractall(extracted)

    nested = extracted / "scripts" / "install_cucim_windows.cmd"
    nested.write_text(
        "@echo off\n"
        '> "%~dp0..\\invoked.txt" echo %*\n'
        "exit /b 7\n",
        encoding="ascii",
    )
    root_entry = extracted / "Install VIPP cuCIM.cmd"
    completed = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "call",
            str(root_entry),
            "--marker",
            "value with spaces",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 7
    marker = (extracted / "invoked.txt").read_text(encoding="utf-8").strip()
    assert marker == '--marker "value with spaces"'
