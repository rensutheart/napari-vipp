"""The installed-package gate exercises native simplification and mesh export."""

import runpy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_mesh_install_smoke_runs_the_packaged_example():
    smoke = runpy.run_path(str(ROOT / "scripts/smoke_mesh_install.py"))[
        "smoke_mesh_install"
    ]
    report = smoke()
    assert report["status"] == "passed"
    assert report["objects"] == report["measurement_rows"] == 5
    assert report["triangles_before"] > report["triangles_after"] > 0
    assert report["exports"] == ["obj", "3mf"]


def test_installed_gate_rejects_source_checkout():
    import napari_vipp

    if ROOT not in Path(napari_vipp.__file__).resolve().parents:
        pytest.skip("This guard tests a source-checkout invocation.")
    smoke = runpy.run_path(str(ROOT / "scripts/smoke_mesh_install.py"))[
        "smoke_mesh_install"
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
def test_clean_and_native_installers_exercise_installed_mesh_stack(filename, job):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
    )
    scripts = "\n".join(step.get("run", "") for step in workflow["jobs"][job]["steps"])
    assert "scripts/smoke_mesh_install.py --require-installed" in scripts
    assert "-m napari_vipp.core.reader_support all-native" in scripts
