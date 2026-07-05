from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_all_example_workflows_are_documented():
    examples_dir = REPO_ROOT / "examples"
    examples_readme = (examples_dir / "README.md").read_text(encoding="utf-8")
    workflow_names = sorted(path.name for path in examples_dir.glob("*.json"))

    assert workflow_names
    for workflow_name in workflow_names:
        assert workflow_name in examples_readme


def test_measurement_workflow_guide_links_reference_examples():
    guide = (REPO_ROOT / "docs" / "measurement-workflows.md").read_text(
        encoding="utf-8",
    )
    for workflow_name in (
        "red-channel-object-intensity-measurements.json",
        "red-channel-merged-measurement-table.json",
        "synthetic-measurement-summary.json",
        "synthetic-derived-object-morphology.json",
        "synthetic-3d-mesh-morphology.json",
        "synthetic-skeleton-qc.json",
        "synthetic-advanced-skeleton-network.json",
        "synthetic-colocalization-racc.json",
    ):
        assert workflow_name in guide


def test_analytical_phantom_validation_report_is_current():
    script_path = REPO_ROOT / "scripts" / "validate_calibrated_morphology_phantoms.py"
    spec = importlib.util.spec_from_file_location(
        "validate_calibrated_morphology_phantoms",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    checks = module.run_validation()
    failed = [check for check in checks if check.status != "PASS"]

    assert checks
    assert not failed
    assert (
        REPO_ROOT / "docs" / "analytical-phantom-validation.md"
    ).read_text(encoding="utf-8") == module.render_markdown(checks)
