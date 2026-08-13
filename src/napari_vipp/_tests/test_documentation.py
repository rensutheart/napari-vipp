from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[3]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
ABSOLUTE_HOME_PATHS = (
    re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+"),
    re.compile(r"/(?:Users|home)/[^/\s`\"']+"),
)
RELEASE_TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".md", ".svg", ".txt", ".yaml", ".yml"}
)


def test_local_markdown_links_resolve():
    markdown_files = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "docs").glob("*.md")),
    ]
    markdown_files.append(REPO_ROOT / "examples" / "README.md")
    missing: list[str] = []

    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw_target = match.group(1).strip().split(maxsplit=1)[0]
            target = unquote(raw_target.strip("<>"))
            if not target or target.startswith(
                ("#", "http://", "https://", "mailto:")
            ):
                continue
            relative_path = target.split("#", 1)[0]
            if relative_path and not (document.parent / relative_path).exists():
                missing.append(f"{document.relative_to(REPO_ROOT)} -> {target}")

    assert not missing, "Missing local Markdown targets:\n" + "\n".join(missing)


def test_documentation_index_has_no_orphaned_pages():
    index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    missing = [
        path.name
        for path in sorted((REPO_ROOT / "docs").glob("*.md"))
        if path.name != "README.md" and f"({path.name})" not in index
    ]

    assert not missing, f"Pages missing from docs/README.md: {missing}"


def test_release_bound_documentation_contains_no_absolute_home_paths():
    documents = [
        REPO_ROOT / "README.md",
        *sorted(
            path
            for path in (REPO_ROOT / "docs").rglob("*")
            if path.is_file() and path.suffix.lower() in RELEASE_TEXT_SUFFIXES
        ),
    ]
    leaked: list[str] = []

    for document in documents:
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if any(pattern.search(line) for pattern in ABSOLUTE_HOME_PATHS):
                leaked.append(f"{document.relative_to(REPO_ROOT)}:{line_number}")

    assert not leaked, "Absolute home paths in release documentation:\n" + "\n".join(
        leaked
    )


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


def test_windows_installer_quick_start_is_primary_and_truthful():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = (REPO_ROOT / "docs" / "quick-start.md").read_text(
        encoding="utf-8"
    )
    packaging_readme = (
        REPO_ROOT / "packaging" / "windows" / "README.md"
    ).read_text(encoding="utf-8")
    documentation_index = (REPO_ROOT / "docs" / "README.md").read_text(
        encoding="utf-8"
    )
    planner_guide = (
        REPO_ROOT / "docs" / "windows-installation-planner.md"
    ).read_text(encoding="utf-8")

    assert readme.index("### Windows Installer (Recommended Path)") < readme.index(
        'python -m pip install "napari[pyqt6]>=0.6"'
    )
    assert quick_start.index("## Windows: The Recommended Route") < quick_start.index(
        "## Manual Alpha Installation (Advanced And Non-Windows)"
    )
    assert (
        "https://github.com/rensutheart/napari-vipp/releases/download/"
        "v0.13.0a6/VIPP-Setup-0.13.0a6-Windows-x86_64-UNSIGNED.exe"
    ) in quick_start
    assert "SHA256SUMS-Windows-0.13.0a6.txt" in quick_start
    assert "**Unknown publisher**" in quick_start
    assert "**More info**, confirm" in quick_start
    assert "**Run anyway**" in quick_start
    assert "do not disable security" in quick_start
    assert "manual installation below" in quick_start
    assert "Windows Settings > Apps > Installed apps" in quick_start
    assert "Managed CPU and CUDA installations can coexist" in quick_start
    assert "expand **Advanced details**" in quick_start
    assert "use **Computer use**" in quick_start
    assert "For a CPU installation, open **VIPP**" in quick_start
    assert "rerun that version's VIPP setup `.exe`" in quick_start
    assert "A supported 64-bit Python is a separate prerequisite" in readme
    assert "separately installed supported 64-bit Python" in packaging_readme
    for document in (readme, quick_start):
        normalized = " ".join(document.split())
        assert "15 GiB" in normalized
        assert "disk storage, not GPU memory (VRAM)" in normalized
        assert "standard GPU installation works without" in normalized
        assert "cuCIM" in normalized
    assert "exact managed location" in packaging_readme
    assert "CPU or CUDA 13" in packaging_readme
    assert "installation-drive storage" in packaging_readme
    assert "DEVELOPMENT BUILD — local testing only" in packaging_readme
    assert "embedded channel" in packaging_readme
    assert "5 GiB" in quick_start
    assert "Windows temporary files and VIPP installer records" in quick_start
    assert "Setup identifies the exact location" in " ".join(quick_start.split())
    assert documentation_index.index("(quick-start.md)") < documentation_index.index(
        "(user-guide.md)"
    )
    assert "ready only for dependency resolution, never" in planner_guide
    assert "1 GiB free for CPU setup or 5 GiB free for CUDA setup" in " ".join(
        planner_guide.split()
    )
    assert "exact checked location, requirement, and available space" in " ".join(
        planner_guide.split()
    )
    assert "ready_for_apply: false" in planner_guide


def test_product_tagline_is_consistent_across_primary_surfaces():
    tagline = "Visual image processing made approachable"
    supporting_promise = "visual workflows for reproducible bioimage analysis"
    surfaces = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "docs" / "quick-start.md",
        REPO_ROOT / "docs" / "assets" / "branding" / "README.md",
        REPO_ROOT / "pyproject.toml",
    )

    for surface in surfaces:
        assert tagline in surface.read_text(encoding="utf-8")

    for surface in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "docs" / "assets" / "branding" / "README.md",
    ):
        normalized = " ".join(
            surface.read_text(encoding="utf-8").lower().split()
        )
        assert supporting_promise in normalized


def test_main_readme_is_concise_and_routes_gpu_details():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    gpu_guide = (REPO_ROOT / "docs" / "gpu-guide.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 300
    assert "## GPU Acceleration (Optional)" in readme
    assert "(docs/gpu-guide.md)" in readme
    assert "### GPU Execution And Development Environment" not in readme
    assert "23.57x" not in readme
    for required_section in (
        "## Compute Modes",
        "## Current Windows CUDA Qualification",
        "## Accelerated Operation Families",
        "## Optional cuCIM Add-on",
        "## Cross-device Reproducibility",
    ):
        assert required_section in gpu_guide


def test_safe_gpu_dtype_repair_is_explained_consistently():
    documents = (
        REPO_ROOT / "docs" / "planning.md",
        REPO_ROOT / "docs" / "user-guide.md",
        REPO_ROOT / "docs" / "gpu-guide.md",
        REPO_ROOT / "docs" / "durable-gpu-execution.md",
    )

    for document in documents:
        text = document.read_text(encoding="utf-8")
        assert "**Add conversion**" in text, document.name
        assert "`uint8`" in text, document.name
        assert "`uint16`" in text, document.name
        assert "`float32`" in text, document.name
        assert "Preserve" in text, document.name
        assert "GPU eligible" in text or "GPU eligibility" in text, document.name
        assert "guarantee" in text, document.name

    roadmap = documents[0].read_text(encoding="utf-8")
    assert "never silently insert casts" in roadmap


def test_installer_plan_prioritizes_nontechnical_managed_users():
    desktop_plan = (
        REPO_ROOT / "docs" / "desktop-startup-and-installer-plan.md"
    ).read_text(encoding="utf-8")

    assert "primary design persona is a physiologist" in desktop_plan
    assert "one clear confirmation: **Install VIPP**" in desktop_plan
    assert "under **Advanced details**" in desktop_plan


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
