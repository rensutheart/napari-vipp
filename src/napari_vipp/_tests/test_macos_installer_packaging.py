from __future__ import annotations

import json
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts import package_macos_installer as packager

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_VERSION = tomllib.loads(
    (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]


def _wheel(path: Path, *, version: str = PROJECT_VERSION) -> Path:
    metadata = f"Metadata-Version: 2.4\nName: napari-vipp\nVersion: {version}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"napari_vipp-{version}.dist-info/METADATA", metadata)
        archive.writestr(
            f"napari_vipp-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    return path


def _release_state(
    *, version: str = PROJECT_VERSION, dirty: bool = False, tagged: bool = True
) -> packager.SourceState:
    expected_tag = f"v{version}"
    return packager.SourceState(
        version=version,
        commit="1" * 40,
        commit_count=445,
        expected_tag=expected_tag,
        exact_tags=(expected_tag,) if tagged else (),
        dirty=dirty,
    )


def _unsigned_finalize_fixture(tmp_path: Path):
    state = _release_state()
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    staging = staging_dir / (
        f"VIPP-{state.version}-macOS-arm64-SIGNING-STAGING.pkg"
    )
    staging.write_bytes(b"reviewed unsigned package")
    evidence = {}
    for key, filename in packager._CONSTRUCTOR_EVIDENCE.items():
        path = staging_dir / filename
        path.write_text(f"reviewed {key}\n", encoding="utf-8")
        evidence[key] = packager._file_record(path)
    manifest = staging_dir / (
        f"VIPP-{state.version}-macOS-arm64-SIGNING-STAGING-build.json"
    )
    document = {
        "schema": packager.BUILD_SCHEMA,
        "schema_version": packager.SCHEMA_VERSION,
        "status": "built",
        "development": False,
        "release_ready": False,
        "release_channel": "unsigned-alpha-staging",
        "unsigned_release_filename": (
            f"VIPP-{state.version}-macOS-arm64-UNSIGNED.pkg"
        ),
        "source": state.as_dict(),
        "architecture": "arm64",
        "target_platform": "osx-arm64",
        "minimum_macos": "13",
        "artifact": packager._file_record(staging),
        "wheel": {
            "filename": f"napari_vipp-{state.version}-py3-none-any.whl",
            "version": state.version,
            "sha256": "2" * 64,
            "contents_sha256": "3" * 64,
        },
        "local_conda_packages": [],
        "constructor_evidence": evidence,
    }
    packager._write_json(manifest, document)
    return state, staging, manifest, document


def _mock_unsigned_finalize_host(monkeypatch, state):
    monkeypatch.setattr(packager, "_require_macos", lambda: None)
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)
    monkeypatch.setattr(
        packager, "_macos_architecture", lambda: ("arm64", "osx-arm64")
    )
    monkeypatch.setattr(packager, "_verify_pkg_archive", lambda _path: None)
    monkeypatch.setattr(
        packager,
        "_inspect_unsigned_signature",
        lambda _path, *, status, label: {
            "status": status,
            "pkgutil_exit_code": 1,
            "pkgutil_output": f"{label}: Status: no signature",
        },
    )


def test_plan_binds_exact_wheel_without_requiring_builder_tools(
    tmp_path, monkeypatch
):
    wheel = _wheel(
        tmp_path / f"napari_vipp-{PROJECT_VERSION}-py3-none-any.whl"
    )
    monkeypatch.setattr(packager, "inspect_source", lambda _root: _release_state())
    monkeypatch.setattr(packager, "_macos_architecture", lambda: ("arm64", "osx-arm64"))

    plan = packager.build_installer(
        repository_root=REPO_ROOT,
        wheel_path=wheel,
        output_directory=tmp_path / "output",
        development=True,
        plan_only=True,
    )

    assert plan["status"] == "planned"
    assert plan["development"] is True
    assert plan["release_ready"] is False
    assert plan["target_platform"] == "osx-arm64"
    assert plan["minimum_macos"] == "13"
    assert plan["wheel"]["sha256"] == packager._sha256(wheel)
    assert str(plan["output_installer"]).endswith(
        f"VIPP-{PROJECT_VERSION}-macOS-arm64-DEVELOPMENT.pkg"
    )
    assert not (tmp_path / "output").exists()


def test_release_staging_requires_clean_exact_tag(tmp_path, monkeypatch):
    wheel = _wheel(
        tmp_path / f"napari_vipp-{PROJECT_VERSION}-py3-none-any.whl"
    )
    monkeypatch.setattr(
        packager, "inspect_source", lambda _root: _release_state(tagged=False)
    )

    with pytest.raises(
        packager.MacOSInstallerPackagingError,
        match="clean checkout at the exact",
    ):
        packager.build_installer(
            repository_root=REPO_ROOT,
            wheel_path=wheel,
            output_directory=tmp_path / "output",
            development=False,
            plan_only=True,
        )


def test_clean_exact_alpha_plan_reserves_staging_and_unsigned_names(
    tmp_path, monkeypatch
):
    state = _release_state()
    wheel = _wheel(
        tmp_path / f"napari_vipp-{state.version}-py3-none-any.whl"
    )
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)
    monkeypatch.setattr(
        packager, "_macos_architecture", lambda: ("arm64", "osx-arm64")
    )

    plan = packager.build_installer(
        repository_root=REPO_ROOT,
        wheel_path=wheel,
        output_directory=tmp_path / "output",
        development=False,
        plan_only=True,
    )

    assert plan["development"] is False
    assert plan["release_channel"] == "unsigned-alpha-staging"
    assert plan["release_ready"] is False
    assert str(plan["output_installer"]).endswith(
        f"VIPP-{state.version}-macOS-arm64-SIGNING-STAGING.pkg"
    )
    assert plan["unsigned_release_filename"] == (
        f"VIPP-{state.version}-macOS-arm64-UNSIGNED.pkg"
    )


@pytest.mark.parametrize("version", ["0.14.0b1", "0.14.0rc1", "0.14.0"])
def test_unsigned_release_lane_rejects_non_alpha_versions(
    tmp_path, monkeypatch, version
):
    state = _release_state(version=version)
    wheel = _wheel(
        tmp_path / f"napari_vipp-{version}-py3-none-any.whl", version=version
    )
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)

    with pytest.raises(
        packager.MacOSInstallerPackagingError, match="limited to X.Y.ZaN"
    ):
        packager.build_installer(
            repository_root=REPO_ROOT,
            wheel_path=wheel,
            output_directory=tmp_path / "output",
            development=False,
            plan_only=True,
        )


def test_plan_rejects_wheel_from_another_version(tmp_path, monkeypatch):
    wheel = _wheel(
        tmp_path / "napari_vipp-0.13.0-py3-none-any.whl", version="0.13.0"
    )
    monkeypatch.setattr(packager, "inspect_source", lambda _root: _release_state())

    with pytest.raises(
        packager.MacOSInstallerPackagingError,
        match="does not match project version",
    ):
        packager.build_installer(
            repository_root=REPO_ROOT,
            wheel_path=wheel,
            output_directory=tmp_path / "output",
            development=True,
            plan_only=True,
        )


def test_menu_template_renders_valid_numeric_apple_versions(tmp_path):
    source = packager.SourceState(
        version="0.14.0a2",
        commit="1" * 40,
        commit_count=445,
        expected_tag="v0.14.0a2",
        exact_tags=("v0.14.0a2",),
        dirty=False,
    )
    output = tmp_path / "vipp-menu.json"

    packager._render_menu_metadata(
        REPO_ROOT / "packaging/macos/vipp-menu.json.in", output, source
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["$schema"].endswith("/menuinst/menuinst-1-1-3.schema.json")
    item = document["menu_items"][0]
    assert item["name"] == "VIPP"
    assert item["command"][-2:] == ["--profile", "cpu"]
    assert item["platforms"]["osx"]["CFBundleVersion"] == "445"
    assert (
        item["platforms"]["osx"]["info_plist_extra"][
            "CFBundleShortVersionString"
        ]
        == "0.14.0"
    )
    assert "{{ MENU_ITEM_LOCATION }}" in item["platforms"]["osx"]["command"][0]


def test_macos_recipe_keeps_application_and_menu_packages_separate():
    recipe = (REPO_ROOT / "packaging/macos/recipe/recipe.yaml.in").read_text(
        encoding="utf-8"
    )

    assert "name: napari-vipp" in recipe
    assert "name: vipp-menu" in recipe
    assert "pin_subpackage('napari-vipp', exact=True)" in recipe
    assert "Menu/vipp-menu.json" in recipe
    assert "Menu/vipp.icns" in recipe
    assert "--no-index --no-deps --no-build-isolation" in recipe
    assert "vipp-compute-doctor = napari_vipp.core.compute_diagnostics:main" in recipe
    assert "__VIPP_WHEEL_SHA256__" in recipe
    assert "direct_url.json" in recipe
    assert "menuinst >=2.5,<3" in recipe
    assert '"${SRC_DIR}/' in recipe
    assert '"${PREFIX}/Menu' in recipe
    assert '"${{ SRC_DIR }}/' not in recipe
    assert '"${{ PREFIX }}/' not in recipe


def test_constructor_template_is_current_user_cpu_only_development_config():
    construct = (REPO_ROOT / "packaging/macos/construct.yaml.in").read_text(
        encoding="utf-8"
    )

    assert "pyside6=6.9.3" in construct
    assert "conda>=23.11" in construct
    assert "menu_packages:\n  - vipp-menu" in construct
    assert 'virtual_specs:\n  - "__osx>=13"' in construct
    assert "enable_currentUserHome: true" in construct
    assert "enable_localSystem: false" in construct
    assert "initialize_conda: false" in construct
    assert "register_envs: false" in construct
    assert "algorithm: sha256" in construct
    assert "signing_identity_name" not in construct
    assert "notarization_identity_name" not in construct


def test_builder_environment_pins_wheel_build_toolchain():
    environment = (
        REPO_ROOT / "packaging/macos/builder-environment.yml"
    ).read_text(encoding="utf-8")

    assert "python-build=1.5.0" in environment
    assert "setuptools=82.0.1" in environment
    assert "wheel=0.47.0" in environment
    assert packager.BUILDER_VERSION_PINS["setuptools"] == "82.0.1"
    assert packager.BUILDER_VERSION_PINS["wheel"] == "0.47.0"


def test_constructor_documents_render_separate_development_and_unsigned_alpha_text(
    tmp_path
):
    development = tmp_path / "development"
    release = tmp_path / "release"
    development.mkdir()
    release.mkdir()

    packager._stage_constructor_documents(
        REPO_ROOT, development, development=True
    )
    packager._stage_constructor_documents(REPO_ROOT, release, development=False)

    development_text = (development / "welcome.txt").read_text(encoding="utf-8")
    release_text = (release / "welcome.txt").read_text(encoding="utf-8")
    release_conclusion = (release / "conclusion.txt").read_text(encoding="utf-8")
    assert "DEVELOPMENT BUILD" in development_text
    assert "not a public release" in development_text
    assert "EXPLICITLY UNSIGNED ALPHA" in release_text
    assert "-UNSIGNED.pkg" in release_text
    assert "Never disable Gatekeeper" in release_text
    assert "Open Anyway" in release_conclusion
    assert "__VIPP_" not in development_text + release_text + release_conclusion


def test_development_signature_requires_exact_unsigned_status(
    tmp_path, monkeypatch
):
    installer = tmp_path / "VIPP-DEVELOPMENT.pkg"
    installer.touch()

    monkeypatch.setattr(
        packager.subprocess,
        "run",
        lambda *args, **kwargs: packager.subprocess.CompletedProcess(
            args[0], 1, stdout="Package: VIPP\nStatus: no signature", stderr=""
        ),
    )
    result = packager._inspect_development_signature(installer)
    assert result["status"] == "unsigned-development"

    monkeypatch.setattr(
        packager.subprocess,
        "run",
        lambda *args, **kwargs: packager.subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="pkgutil failed"
        ),
    )
    with pytest.raises(
        packager.MacOSInstallerPackagingError,
        match="Could not verify",
    ):
        packager._inspect_development_signature(installer)


def test_finalize_unsigned_creates_only_arch_qualified_release_assets(
    tmp_path, monkeypatch
):
    state, staging, manifest, _document = _unsigned_finalize_fixture(tmp_path)
    _mock_unsigned_finalize_host(monkeypatch, state)
    output = tmp_path / "release"

    result = packager.finalize_unsigned_installer(
        repository_root=REPO_ROOT,
        unsigned_staging_installer=staging,
        build_manifest_path=manifest,
        output_directory=output,
    )

    base = f"VIPP-{state.version}-macOS-arm64-UNSIGNED"
    expected = {
        f"{base}.pkg",
        f"{base}-release.json",
        f"{base}-constructor-info.json",
        f"{base}-licenses.json",
        f"{base}-lockfile.txt",
        f"{base}-package-list.txt",
        f"SHA256SUMS-macOS-arm64-{state.version}.txt",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert staging.read_bytes() == b"reviewed unsigned package"
    assert result["release_channel"] == "explicitly-unsigned-alpha"
    assert result["release_ready"] is True
    assert result["artifact"]["filename"] == f"{base}.pkg"
    assert result["signature"]["status"] == "explicitly-unsigned-alpha"
    warning = result["user_warning"]
    assert warning["signed"] is False
    assert warning["notarized"] is False
    assert warning["never_disable_gatekeeper"] is True

    checksum = output / f"SHA256SUMS-macOS-arm64-{state.version}.txt"
    lines = checksum.read_text(encoding="ascii").splitlines()
    assert len(lines) == 1
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert digest == packager._sha256(output / filename)


def test_finalize_unsigned_rejects_tampered_staging_package(tmp_path, monkeypatch):
    state, staging, manifest, _document = _unsigned_finalize_fixture(tmp_path)
    _mock_unsigned_finalize_host(monkeypatch, state)
    staging.write_bytes(b"tampered")

    with pytest.raises(
        packager.MacOSInstallerPackagingError, match="reviewed build record"
    ):
        packager.finalize_unsigned_installer(
            repository_root=REPO_ROOT,
            unsigned_staging_installer=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
        )


def test_finalize_unsigned_rejects_development_manifest(tmp_path, monkeypatch):
    state, staging, manifest, document = _unsigned_finalize_fixture(tmp_path)
    _mock_unsigned_finalize_host(monkeypatch, state)
    document["development"] = True
    packager._write_json(manifest, document)

    with pytest.raises(
        packager.MacOSInstallerPackagingError, match="DEVELOPMENT build"
    ):
        packager.finalize_unsigned_installer(
            repository_root=REPO_ROOT,
            unsigned_staging_installer=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "building", "status"),
        ("release_ready", True, "release-ready"),
        ("target_platform", "osx-64", "target platform"),
        ("minimum_macos", "12", "minimum macOS"),
        ("unsigned_release_filename", "VIPP.pkg", "unsigned filename"),
    ],
)
def test_finalize_unsigned_rejects_inconsistent_staging_metadata(
    tmp_path, monkeypatch, field, value, message
):
    state, staging, manifest, document = _unsigned_finalize_fixture(tmp_path)
    _mock_unsigned_finalize_host(monkeypatch, state)
    document[field] = value
    packager._write_json(manifest, document)

    with pytest.raises(packager.MacOSInstallerPackagingError, match=message):
        packager.finalize_unsigned_installer(
            repository_root=REPO_ROOT,
            unsigned_staging_installer=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
        )


def test_finalize_unsigned_rejects_changed_constructor_evidence(tmp_path, monkeypatch):
    state, staging, manifest, _document = _unsigned_finalize_fixture(tmp_path)
    _mock_unsigned_finalize_host(monkeypatch, state)
    (manifest.parent / "licenses.json").write_text("changed\n", encoding="utf-8")

    with pytest.raises(
        packager.MacOSInstallerPackagingError, match="constructor licenses differs"
    ):
        packager.finalize_unsigned_installer(
            repository_root=REPO_ROOT,
            unsigned_staging_installer=staging,
            build_manifest_path=manifest,
            output_directory=tmp_path / "release",
        )


def test_release_workflow_matches_macos_finalizer_contract():
    workflow = (
        REPO_ROOT / ".github/workflows/unsigned-installers-release.yml"
    ).read_text(encoding="utf-8")

    assert "*-SIGNING-STAGING.pkg" in workflow
    assert '--unsigned-staging-installer "$staging"' in workflow
    assert "VIPP-$version-macOS-${{ matrix.architecture }}-UNSIGNED.pkg" in workflow
    assert "SHA256SUMS-macOS-${{ matrix.architecture }}-$version.txt" in workflow


@pytest.mark.skipif(packager.sys.platform != "darwin", reason="macOS tools required")
def test_vipp_svg_renders_as_nonempty_icns(tmp_path):
    output = tmp_path / "vipp.icns"

    packager._render_macos_icon(
        REPO_ROOT / "src/napari_vipp/assets/branding/vipp-mark.svg",
        output,
        tmp_path / "icon-work",
    )

    assert output.read_bytes()[:4] == b"icns"
    assert output.stat().st_size > 10_000
