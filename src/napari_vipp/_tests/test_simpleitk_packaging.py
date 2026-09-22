"""The CPU acceleration dependency is shipped and recorded, without eager imports."""

import runpy
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]


def test_simpleitk_exact_pin_matches_offline_macos_recipe():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    requirements = {
        requirement.name.lower(): requirement
        for requirement in map(Requirement, project["dependencies"])
    }
    assert str(requirements["simpleitk"].specifier) == "==2.5.6"
    recipe = (ROOT / "packaging/macos/recipe/recipe.yaml.in").read_text(
        encoding="utf-8"
    )
    assert "- simpleitk ==2.5.6" in recipe
    assert "- SimpleITK" in recipe
    assert "itk" not in requirements  # Do not add the separate Python ITK package.


def test_simpleitk_versions_are_carried_without_importing_native_module(monkeypatch):
    import napari_vipp.core.batch as batch
    import napari_vipp.core.compute as compute
    import napari_vipp.core.reproducibility as reproducibility

    # A forbidden import must not prevent software inventory from being recorded.
    monkeypatch.setitem(sys.modules, "SimpleITK", None)
    monkeypatch.setattr(compute.importlib.metadata, "version", lambda _name: "2.5.6")
    monkeypatch.setattr(batch, "package_version", lambda _name: "2.5.6")
    monkeypatch.setattr(reproducibility, "version", lambda _name: "2.5.6")
    assert dict(compute._installed_scientific_stack_versions())["simpleitk"] == "2.5.6"
    assert batch._runtime_versions()["packages"]["simpleitk"] == "2.5.6"
    assert reproducibility._environment()["packages"]["simpleitk"] == "2.5.6"


def test_simpleitk_install_smoke_runs_native_code():
    smoke = runpy.run_path(str(ROOT / "scripts/smoke_simpleitk_install.py"))[
        "smoke_simpleitk_install"
    ]
    result = smoke()
    assert result["status"] == "passed"
    assert result["versions"]["simpleitk"] == "2.5.6"
    assert result["itk_version"]
    assert len(result["checks"]) == 2


def test_simpleitk_installed_gate_rejects_source_checkout():
    import napari_vipp

    if ROOT not in Path(napari_vipp.__file__).resolve().parents:
        pytest.skip("This guard tests a source-checkout invocation.")
    smoke = runpy.run_path(str(ROOT / "scripts/smoke_simpleitk_install.py"))[
        "smoke_simpleitk_install"
    ]
    with pytest.raises(RuntimeError, match="source checkout"):
        smoke(require_installed=True)


@pytest.mark.parametrize(
    ("filename", "job"),
    (
        ("ci.yml", "clean-distribution-install"),
        ("macos-installer.yml", "development-build"),
        ("unsigned-installers-release.yml", "macos"),
    ),
)
def test_installed_simpleitk_is_exercised_before_releasing(filename, job):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
    )
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"][job]["steps"])
    assert "scripts/smoke_simpleitk_install.py --require-installed" in commands
