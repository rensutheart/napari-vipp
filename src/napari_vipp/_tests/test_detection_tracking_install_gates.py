"""New scientific nodes must be exercised from clean/native installed packages."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SMOKE_SCRIPTS = ("smoke_detection_install.py", "smoke_tracking_install.py")


def _workflow(filename):
    return yaml.safe_load(
        (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("script", SMOKE_SCRIPTS)
@pytest.mark.parametrize(
    ("filename", "job", "step_name", "interpreter"),
    (
        (
            "ci.yml",
            "clean-distribution-install",
            "Verify installed package, resources, and headless entry points",
            "python",
        ),
        (
            "macos-installer.yml",
            "development-build",
            "Install and launch on the disposable runner",
            '"$prefix/bin/python"',
        ),
        (
            "unsigned-installers-release.yml",
            "macos",
            "Inspect, install, and launch the final macOS release",
            '"$prefix/bin/python"',
        ),
    ),
)
def test_installed_native_runtime_runs_detection_and_tracking_known_answers(
    filename, job, step_name, interpreter, script
):
    step = next(
        step for step in _workflow(filename)["jobs"][job]["steps"]
        if step["name"] == step_name
    )
    commands = step["run"]
    command = f"{interpreter} scripts/{script} --require-installed"
    assert commands.count(command) == 1
    # Native scientific checks follow consistency validation, using the installed
    # application's Python rather than the macOS builder/source environment.
    assert commands.index(f"{interpreter} -m pip check") < commands.index(command)


@pytest.mark.parametrize("script", SMOKE_SCRIPTS)
def test_macos_smoke_script_changes_trigger_native_package_validation(script):
    workflow = _workflow("macos-installer.yml")
    # PyYAML's YAML 1.1 loader treats the GitHub Actions `on` key as Boolean.
    triggers = workflow.get("on", workflow.get(True))
    assert f"scripts/{script}" in triggers["pull_request"]["paths"]


@pytest.mark.parametrize(
    ("filename", "job"),
    (
        ("windows-installer.yml", "development-build"),
        ("unsigned-installers-release.yml", "windows"),
    ),
)
def test_frozen_windows_setup_does_not_claim_installed_scientific_execution(
    filename, job
):
    commands = "\n".join(
        step.get("run", "") for step in _workflow(filename)["jobs"][job]["steps"]
    )
    # The EXE is the setup program, not an installed scientific Python runtime.
    # Windows scientific smoke belongs to clean-distribution-install above.
    for script in SMOKE_SCRIPTS:
        assert f"scripts/{script}" not in commands


@pytest.mark.parametrize(
    ("filename", "job"),
    (
        ("macos-installer.yml", "development-build"),
        ("unsigned-installers-release.yml", "macos"),
    ),
)
def test_native_macos_known_answers_keep_both_architectures(filename, job):
    matrix = _workflow(filename)["jobs"][job]["strategy"]["matrix"]["include"]
    assert {item["architecture"] for item in matrix} == {"arm64", "x86_64"}
