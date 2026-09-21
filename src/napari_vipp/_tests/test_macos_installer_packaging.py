from __future__ import annotations

import json
import re
import tomllib
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

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
    staging = staging_dir / (f"VIPP-{state.version}-macOS-arm64-SIGNING-STAGING.pkg")
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
        "unsigned_release_filename": (f"VIPP-{state.version}-macOS-arm64-UNSIGNED.pkg"),
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
    monkeypatch.setattr(packager, "_macos_architecture", lambda: ("arm64", "osx-arm64"))
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


def test_plan_binds_exact_wheel_without_requiring_builder_tools(tmp_path, monkeypatch):
    wheel = _wheel(tmp_path / f"napari_vipp-{PROJECT_VERSION}-py3-none-any.whl")
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
    wheel = _wheel(tmp_path / f"napari_vipp-{PROJECT_VERSION}-py3-none-any.whl")
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
    wheel = _wheel(tmp_path / f"napari_vipp-{state.version}-py3-none-any.whl")
    monkeypatch.setattr(packager, "inspect_source", lambda _root: state)
    monkeypatch.setattr(packager, "_macos_architecture", lambda: ("arm64", "osx-arm64"))

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
    wheel = _wheel(tmp_path / "napari_vipp-0.13.0-py3-none-any.whl", version="0.13.0")
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
    assert document["menu_name"] == "VIPP"
    assert len(document["menu_items"]) == 1
    item = document["menu_items"][0]
    assert item["name"] == "VIPP"
    expected_arguments = ["-m", "napari_vipp", "--desktop", "--profile", "auto"]
    assert item["command"] == ["{{ PYTHON }}", *expected_arguments]
    assert item["platforms"]["osx"]["command"] == [
        "{{ MENU_ITEM_LOCATION }}/Contents/Resources/python",
        *expected_arguments,
    ]
    assert item["platforms"]["osx"]["CFBundleName"] == "VIPP"
    assert item["platforms"]["osx"]["CFBundleDisplayName"] == "VIPP"
    assert item["icon"] == "{{ MENU_DIR }}/vipp.{{ ICON_EXT }}"
    assert item["platforms"]["osx"]["CFBundleVersion"] == "445"
    assert (
        item["platforms"]["osx"]["info_plist_extra"]["CFBundleShortVersionString"]
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


def test_macos_recipe_has_only_one_gui_entry_point():
    recipe = (REPO_ROOT / "packaging/macos/recipe/recipe.yaml.in").read_text(
        encoding="utf-8"
    )
    entry_points = recipe.split("entry_points:\n", 1)[1].split("      script:", 1)[0]
    actual = {
        name.strip(): target.strip()
        for line in entry_points.splitlines()
        for name, target in [line.strip().removeprefix("- ").split(" = ", 1)]
    }
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["gui-scripts"] == {"vipp-app": "napari_vipp.launcher:main_auto"}
    assert actual == {**project["scripts"], **project["gui-scripts"]}


def test_macos_recipe_psygnal_constraint_matches_embedded_wheel_metadata():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    wheel_requirement = next(
        requirement
        for dependency in project["dependencies"]
        if (requirement := Requirement(dependency)).name == "psygnal"
    )
    recipe = (REPO_ROOT / "packaging/macos/recipe/recipe.yaml.in").read_text(
        encoding="utf-8"
    )
    application = recipe.split("      name: napari-vipp\n", 1)[1]
    runtime = application.split("      run:\n", 1)[1].split("    tests:\n", 1)[0]
    psygnal_specs = [
        line.strip().removeprefix("- ")
        for line in runtime.splitlines()
        if line.strip().startswith("- psygnal ")
    ]

    # The embedded wheel is installed --no-deps. Conda must honor the same
    # compatibility bound; otherwise installation succeeds but pip check fails.
    assert len(psygnal_specs) == 1
    conda_requirement = Requirement(psygnal_specs[0])
    assert conda_requirement.specifier == wheel_requirement.specifier
    for version in ("0.14.0", "0.15.1"):
        assert conda_requirement.specifier.contains(version)
    for version in ("0.13.0", "0.16.0", "0.16.1"):
        assert not conda_requirement.specifier.contains(version)


def test_macos_recipe_supplies_every_direct_wheel_dependency():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    recipe = (REPO_ROOT / "packaging/macos/recipe/recipe.yaml.in").read_text(
        encoding="utf-8"
    )
    application = recipe.split("      name: napari-vipp\n", 1)[1]
    runtime = application.split("      run:\n", 1)[1].split("    tests:\n", 1)[0]
    actual = {
        Requirement(line.strip().removeprefix("- ")).name
        for line in runtime.splitlines()
        if line.strip().startswith("- ")
    }
    aliases = {"dask": "dask-core", "matplotlib": "matplotlib-base"}
    required = {
        aliases.get(requirement.name, requirement.name)
        for dependency in project["dependencies"]
        for requirement in [Requirement(dependency)]
    }
    assert required <= actual, (
        f"Missing --no-deps runtime packages: {required - actual}"
    )
    requirement = next(
        Requirement(value)
        for value in project["dependencies"]
        if Requirement(value).name == "centrosome"
    )
    assert str(requirement.specifier) == "==1.3.4"
    assert "- centrosome ==1.3.4" in runtime


def test_centrosome_recipe_is_pinned_native_cpython312_with_upstream_bounds():
    recipe = (REPO_ROOT / "packaging/macos/centrosome/recipe.yaml.in").read_text(
        encoding="utf-8"
    )
    assert 'version: "1.3.4"' in recipe
    assert "\n  noarch:" not in recipe
    assert "- python >=3.12,<3.13" in recipe
    assert "- python_abi 3.12.* *_cp312" in recipe
    assert "--no-index --no-deps --no-build-isolation" in recipe
    assert "centrosome-1.3.4.dist-info/licenses/LICENSE" in recipe
    assert "license_file: LICENSE" in recipe
    # These are the two reviewed release wheels, not a runtime PyPI lookup.
    expected = {
        "osx-arm64": (
            "macosx_11_0_arm64",
            "ee9190b8514e329972cfaa601a3ed66ec1a4f1968fed310dedb70d8a9236d667",
        ),
        "osx-64": (
            "macosx_10_13_x86_64",
            "b43e878fd0916b8a40b10294ee366a809a4c81e73eae9bec15684e6291368b0b",
        ),
    }
    for target, (tag, digest) in expected.items():
        section = recipe.split(f'if: target_platform == "{target}"', 1)[1]
        section = section.split("\n  - if:", 1)[0].split("\nbuild:", 1)[0]
        assert f"centrosome-1.3.4-cp312-cp312-{tag}.whl" in section
        assert f"sha256: {digest}" in section
        assert "https://files.pythonhosted.org/packages/" in section
    assert len(re.findall(r"\n      sha256: [a-f0-9]{64}\n", recipe)) == 2
    for dependency in (
        "deprecation",
        "numpy >=1.18.2",
        "pillow >=7.1.0,<12",
        "scikit-image >=0.17.2,<1",
        "scipy >=1.4.1,<2,!=1.11.0",
    ):
        assert f"    - {dependency}\n" in recipe
    for module in (
        "_propagate",
        "_cpmorphology2",
        "_convex_hull",
        "_filter",
        "_lapjv",
        "_fastemd",
    ):
        assert f"- centrosome.{module}" in recipe


@pytest.mark.parametrize("target", ["osx-arm64", "osx-64"])
def test_native_centrosome_is_built_and_tested_before_vipp(
    tmp_path, monkeypatch, target
):
    calls = []
    monkeypatch.setattr(
        packager, "_run", lambda command, **kwargs: calls.append(command)
    )
    channel = tmp_path / "channel"
    native_recipe = tmp_path / "centrosome"
    vipp_recipe = tmp_path / "recipe"
    packager._build_local_conda_packages(
        recipe_dir=vipp_recipe,
        centrosome_recipe_dir=native_recipe,
        channel_dir=channel,
        target_platform=target,
        rattler_build=Path("rattler-build"),
    )
    assert len(calls) == 3
    native, index, application = calls
    assert native[native.index("--recipe") + 1] == str(native_recipe / "recipe.yaml")
    assert native[native.index("--target-platform") + 1] == target
    assert native[native.index("--test") + 1] == "native"
    assert "conda_index" in index
    assert application[application.index("--recipe") + 1] == str(
        vipp_recipe / "recipe.yaml"
    )
    assert application[application.index("--channel") + 1] == channel.resolve().as_uri()


def _local_packages(tmp_path, target, *, native_subdir=None):
    packages = [
        tmp_path / "noarch" / f"napari-vipp-{PROJECT_VERSION}-0.conda",
        tmp_path / "noarch" / f"vipp-menu-{PROJECT_VERSION}-0.conda",
        tmp_path / (native_subdir or target) / "centrosome-1.3.4-py312_vipp_0.conda",
    ]
    for path in packages:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))
    return packages


@pytest.mark.parametrize("target", ["osx-arm64", "osx-64"])
def test_local_channel_evidence_includes_native_centrosome(
    tmp_path, monkeypatch, target
):
    packages = _local_packages(tmp_path, target)
    calls = []
    monkeypatch.setattr(
        packager, "_run", lambda command, **kwargs: calls.append(command)
    )
    records = packager._index_local_channel(
        channel_dir=tmp_path, target_platform=target
    )
    assert len(calls) == 1
    assert len(records) == 3
    assert {record["filename"] for record in records} == {
        path.name for path in packages
    }
    native = next(
        record for record in records if record["filename"].startswith("centrosome-")
    )
    assert native["subdir"] == target
    assert native["sha256"] == packager._sha256(packages[-1])


@pytest.mark.parametrize(
    "bad_package", ["missing", "noarch", "wrong_arch", "duplicate", "extra"]
)
def test_channel_rejects_missing_or_mispackaged_native_dependency(
    tmp_path, monkeypatch, bad_package
):
    target = "osx-arm64"
    subdir = {"noarch": "noarch", "wrong_arch": "osx-64"}.get(bad_package)
    packages = _local_packages(tmp_path, target, native_subdir=subdir)
    if bad_package == "missing":
        packages[-1].unlink()
    elif bad_package == "duplicate":
        packages[-1].with_name("centrosome-1.3.4-other_0.conda").touch()
    elif bad_package == "extra":
        packages[-1].with_name("unexpected-1-0.conda").touch()
    monkeypatch.setattr(
        packager,
        "_run",
        lambda *args, **kwargs: pytest.fail("must reject before indexing"),
    )
    with pytest.raises(packager.MacOSInstallerPackagingError):
        packager._index_local_channel(channel_dir=tmp_path, target_platform=target)


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


@pytest.mark.parametrize("architecture", ["arm64", "x86_64"])
def test_constructor_yaml_runtime_constraint_matches_conda_metadata(
    tmp_path, architecture
):
    output = tmp_path / "construct.yaml"
    packager._render_template(
        REPO_ROOT / "packaging/macos/construct.yaml.in",
        output,
        {
            "__VIPP_VERSION__": PROJECT_VERSION,
            "__VIPP_INSTALLER_FILENAME__": (
                f"VIPP-{PROJECT_VERSION}-macOS-{architecture}-DEVELOPMENT.pkg"
            ),
            "__VIPP_LOCAL_CHANNEL_URI__": (tmp_path / "channel").as_uri(),
        },
    )
    specs = output.read_text(encoding="utf-8").split("specs:\n", 1)[1]
    specs = specs.split("\nvirtual_specs:", 1)[0]
    yaml_specs = [
        line.removeprefix("  - ")
        for line in specs.splitlines()
        if line.startswith("  - ruamel.yaml")
    ]
    # conda 26.7.2 declares this range in its upstream Python METADATA.
    # The prior constructor solve admitted 0.19.1, then installed pip check failed.
    assert yaml_specs == ["ruamel.yaml>=0.11.14,<0.19"]
    requirement = Requirement(yaml_specs[0])
    for version in ("0.11.14", "0.18.17"):
        assert requirement.specifier.contains(version)
    for version in ("0.11.13", "0.19.0", "0.19.1"):
        assert not requirement.specifier.contains(version)


@pytest.mark.parametrize(
    "workflow", ["macos-installer.yml", "unsigned-installers-release.yml"]
)
def test_native_installer_workflows_keep_strict_installed_dependency_check(workflow):
    text = (REPO_ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
    check_lines = [line.strip() for line in text.splitlines() if "-m pip check" in line]
    assert check_lines == ['"$prefix/bin/python" -m pip check']
    centrosome_check = (
        '"$prefix/bin/python" scripts/smoke_centrosome_install.py --require-installed'
    )
    assert centrosome_check in text
    assert text.index(check_lines[0]) < text.index(centrosome_check)
    assert text.index(check_lines[0]) < text.index(
        '"$prefix/bin/python" scripts/smoke_mesh_install.py --require-installed'
    )


@pytest.mark.parametrize(
    "workflow", ["macos-installer.yml", "unsigned-installers-release.yml"]
)
def test_native_installer_workflows_check_rendered_desktop_launcher(tmp_path, workflow):
    menu = tmp_path / "vipp-menu.json"
    packager._render_menu_metadata(
        REPO_ROOT / "packaging/macos/vipp-menu.json.in", menu, _release_state()
    )
    item = json.loads(menu.read_text(encoding="utf-8"))["menu_items"][0]
    command = item["platforms"]["osx"]["command"]
    assert command[1:] == ["-m", "napari_vipp", "--desktop", "--profile", "auto"]
    expected_command = " ".join(command).replace("{{ MENU_ITEM_LOCATION }}", "$app")
    text = (REPO_ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
    # Keep the exact installed-script guard aligned with production menu metadata.
    # A pre-desktop command caused both architectures to exit before GUI launch.
    lines = [line.strip() for line in text.splitlines()]
    command_checks = [
        line for line in lines if "grep" in line and " -m napari_vipp " in line
    ]
    assert command_checks == [f'grep -Fq "{expected_command}" \\']
    check_index = lines.index(command_checks[0])
    assert lines[check_index + 1] == '"$launcher_script"'
    # Matching the script is not a replacement for launching the installed app.
    assert 'QT_API=pyqt6 "$launcher" \\' in lines[check_index + 2 :]
    assert 'test "$shortcut_ready" -eq 1' in lines
    child_profile = command[command.index("--profile") + 1]
    child_checks = [line for line in lines if "pgrep -f 'napari_vipp.app" in line]
    assert child_checks == [
        f"child_pid=\"$(pgrep -f 'napari_vipp.app.*--profile {child_profile}' "
        '| head -n 1)"'
    ]
    assert lines.count('kill -0 "$child_pid"') == 2
    assert 'QT_API=pyside6 "$prefix/bin/python" -m napari_vipp.app \\' in lines
    assert "--profile cpu --smoke-exit-after-ready \\" in lines
    assert 'grep -Fq "VIPP: VIPP is ready" "$RUNNER_TEMP/vipp-native.log"' in lines


def test_builder_environment_pins_wheel_build_toolchain():
    environment = (REPO_ROOT / "packaging/macos/builder-environment.yml").read_text(
        encoding="utf-8"
    )

    assert "python-build=1.5.0" in environment
    assert "setuptools=82.0.1" in environment
    assert "wheel=0.47.0" in environment
    assert packager.BUILDER_VERSION_PINS["setuptools"] == "82.0.1"
    assert packager.BUILDER_VERSION_PINS["wheel"] == "0.47.0"


def test_constructor_documents_render_separate_development_and_unsigned_alpha_text(
    tmp_path,
):
    development = tmp_path / "development"
    release = tmp_path / "release"
    development.mkdir()
    release.mkdir()

    packager._stage_constructor_documents(REPO_ROOT, development, development=True)
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
    for text in (development_text, release_text, release_conclusion):
        assert "~/Applications/VIPP.app" in text
        assert "Auto" in text
        assert "inside VIPP" in text
        assert "CPU-safe" not in text
        assert "VIPP Automatic" not in text
        assert "VIPP CPU" not in text
        assert "VIPP GPU" not in text
    assert "does not include NVIDIA CUDA" in release_text


def test_development_signature_requires_exact_unsigned_status(tmp_path, monkeypatch):
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
