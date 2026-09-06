"""All public installers must embed the already-qualified Python wheel."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("platform", ("windows", "macos"))
def test_release_installers_reuse_exact_main_ci_distributions(platform):
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (root / ".github/workflows/unsigned-installers-release.yml").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["permissions"]["actions"] == "read"
    steps = workflow["jobs"][platform]["steps"]
    download = next(
        step
        for step in steps
        if step["name"] == "Download the wheel qualified by exact-main CI"
    )
    script = download["run"]
    assert "git rev-parse HEAD" in script
    assert "actions/workflows/ci.yml/runs?head_sha=" in script
    assert "status=success" in script
    assert 'select(.event == "push")' in script
    assert "gh run download" in script
    assert "python-distributions-" in script
    assert "-py3-none-any.whl" in script
    assert download["env"]["GH_TOKEN"] == "${{ github.token }}"
    # Rebuilding independently per platform would break byte identity between
    # the standalone wheel, the embedded wheels, and the PyPI publication.
    assert not any("python -m build" in step.get("run", "") for step in steps)
