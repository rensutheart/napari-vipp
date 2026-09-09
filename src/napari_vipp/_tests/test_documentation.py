from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
ABSOLUTE_HOME_PATHS = (
    re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+"),
    re.compile(r"/(?:Users|home)/[^/\s`\"']+"),
)
RELEASE_TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".md", ".svg", ".txt", ".yaml", ".yml"}
)
RELEASE_NOTE_BLOCK_START = re.compile(
    r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\||```|~~~|(?:-{3,}|_{3,}|\*{3,})\s*$)"
)


def _public_release_metadata(citation: str) -> tuple[str, str]:
    version_match = re.search(r'^version: "([^"\n]+)"$', citation, re.MULTILINE)
    release_date_match = re.search(
        r'^date-released: "(\d{4}-\d{2}-\d{2})"$', citation, re.MULTILINE
    )
    assert version_match is not None
    assert release_date_match is not None
    return version_match.group(1), release_date_match.group(1)


def _assert_release_version_contract(
    *, version: str, citation: str, changelog: str, release_notes: str, readme: str
) -> None:
    public_version, release_date = _public_release_metadata(citation)
    headers = re.findall(r"^## (.+)$", changelog, re.MULTILINE)
    if headers and headers[0] == "Unreleased":
        headers.pop(0)
    candidate_notices = re.findall(
        r"^\*\*Unreleased candidate\.\*\* (\S+) remains the public alpha\.",
        release_notes,
        re.MULTILINE,
    )
    assert headers
    if candidate_notices:
        assert candidate_notices == [public_version]
        assert version != public_version
        assert headers[0] == f"{version} - Unreleased"
        assert f"{public_version} - {release_date}" in headers[1:]
    else:
        assert version == public_version
        assert headers[0] == f"{version} - {release_date}"
    assert release_notes.startswith(f"# VIPP {version}\n")
    assert "release candidate" not in release_notes.casefold()
    assert f"releases/tag/v{public_version}" in readme
    assert f"napari-vipp=={public_version}" in readme


def test_release_version_contract_is_consistent() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    _assert_release_version_contract(
        version=project["project"]["version"],
        citation=(REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
        changelog=(REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        release_notes=(REPO_ROOT / "release-notes.md").read_text(encoding="utf-8"),
        readme=(REPO_ROOT / "README.md").read_text(encoding="utf-8"),
    )


def _release_contract_fixture(*, candidate: bool) -> dict[str, str]:
    public_version = "1.0.0a1"
    version = "1.0.0a2" if candidate else public_version
    published_header = f"## {public_version} - 2026-09-08\n"
    candidate_header = f"## {version} - Unreleased\n\n" if candidate else ""
    notice = (
        f"**Unreleased candidate.** {public_version} remains the public alpha.\n"
        if candidate
        else ""
    )
    return {
        "version": version,
        "citation": f'version: "{public_version}"\ndate-released: "2026-09-08"\n',
        "changelog": f"## Unreleased\n\n{candidate_header}{published_header}",
        "release_notes": f"# VIPP {version}\n\n{notice}",
        "readme": f"releases/tag/v{public_version}\nnapari-vipp=={public_version}\n",
    }


@pytest.mark.parametrize("candidate", [False, True], ids=["published", "candidate"])
def test_release_version_contract_accepts_explicit_states(candidate: bool) -> None:
    _assert_release_version_contract(**_release_contract_fixture(candidate=candidate))


@pytest.mark.parametrize(
    ("candidate", "field", "old", "new"),
    [
        (True, "release_notes", "**Unreleased candidate.**", "Candidate:"),
        (True, "release_notes", "1.0.0a1 remains", "1.0.0a2 remains"),
        (True, "release_notes", "# VIPP 1.0.0a2\n", "# VIPP 1.0.0a2 candidate\n"),
        (True, "changelog", "1.0.0a2 - Unreleased", "1.0.0a2 - 2026-09-09"),
        (True, "changelog", "1.0.0a1 - 2026-09-08", "1.0.0a1 - 2026-09-09"),
        (True, "readme", "releases/tag/v1.0.0a1", "releases/tag/v1.0.0a2"),
        (True, "readme", "napari-vipp==1.0.0a1", "napari-vipp==1.0.0a2"),
        (False, "version", "1.0.0a1", "1.0.0a2"),
        (False, "changelog", "1.0.0a1 - 2026-09-08", "1.0.0a1 - Unreleased"),
    ],
)
def test_release_version_contract_rejects_mixed_states(
    candidate: bool, field: str, old: str, new: str
) -> None:
    documents = _release_contract_fixture(candidate=candidate)
    assert old in documents[field]
    documents[field] = documents[field].replace(old, new)
    with pytest.raises(AssertionError):
        _assert_release_version_contract(**documents)


def test_release_notes_do_not_hard_wrap_prose() -> None:
    lines = (REPO_ROOT / "release-notes.md").read_text(encoding="utf-8").splitlines()
    continuation_lines: list[int] = []
    previous_line_has_content = False
    fence_marker: str | None = None

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        marker = stripped[:3]
        if fence_marker is not None:
            if marker == fence_marker:
                fence_marker = None
                previous_line_has_content = True
            continue
        if marker in {"```", "~~~"}:
            fence_marker = marker
            previous_line_has_content = True
            continue
        if not stripped:
            previous_line_has_content = False
            continue
        if previous_line_has_content and not RELEASE_NOTE_BLOCK_START.match(stripped):
            continuation_lines.append(line_number)
        previous_line_has_content = True

    assert not continuation_lines, (
        "release-notes.md must use one physical line per paragraph or list item; "
        f"hard-wrapped continuation lines: {continuation_lines}"
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
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative_path = target.split("#", 1)[0]
            if relative_path and not (document.parent / relative_path).exists():
                missing.append(f"{document.relative_to(REPO_ROOT)} -> {target}")

    assert not missing, "Missing local Markdown targets:\n" + "\n".join(missing)


def test_documentation_index_has_no_orphaned_pages():
    index = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    redirects = json.loads(
        (REPO_ROOT / "docs" / "public-guide-redirects.json").read_text()
    )
    missing = [
        path.name
        for path in sorted((REPO_ROOT / "docs").glob("*.md"))
        if path.name not in {"README.md", *redirects} and f"({path.name})" not in index
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


def test_public_guides_are_short_relocation_notes():
    redirects = json.loads(
        (REPO_ROOT / "docs" / "public-guide-redirects.json").read_text(encoding="utf-8")
    )
    assert len(redirects) == 9
    for name, entry in redirects.items():
        guide = (REPO_ROOT / "docs" / name).read_text(encoding="utf-8")
        assert len(guide.splitlines()) <= 24, name
        assert "not a second guide" in guide, name
        assert "```" not in guide, name
        assert entry["pages"], name
        for route in entry["pages"].values():
            assert ".." not in route and not route.startswith("/"), name
            assert (
                f"https://rensutheart.github.io/vipp-mkdocs/nightly/{route}/" in guide
            )


def test_windows_readme_and_packaging_keep_repository_facts():
    version, _ = _public_release_metadata(
        (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    packaging = (REPO_ROOT / "packaging/windows/README.md").read_text(encoding="utf-8")
    planner = (REPO_ROOT / "docs/windows-installation-planner.md").read_text(
        encoding="utf-8"
    )
    assert readme.index("| Windows 64-bit |") < readme.index(
        'python -m pip install "napari[pyqt6]>=0.6"'
    )
    assert f"VIPP-Setup-{version}-Windows-x86_64-UNSIGNED.exe" in readme
    assert "A supported 64-bit Python is a separate prerequisite" in readme
    assert "stable/getting-started/installation/" in readme
    assert "separately installed supported 64-bit Python" in packaging
    assert "exact managed location" in packaging
    assert "15 GiB of free disk space" in packaging
    assert "DEVELOPMENT BUILD — local testing only" in packaging
    assert "embedded channel" in packaging
    assert "ready only for dependency resolution, never" in planner
    assert "ready_for_apply: false" in planner
    assert "1 GiB free for CPU setup or 5 GiB free for CUDA setup" in " ".join(
        planner.split()
    )


def test_macos_readme_keeps_architecture_specific_packages():
    version, _ = _public_release_metadata(
        (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for architecture in ("arm64", "x86_64"):
        assert f"VIPP-{version}-macOS-{architecture}-UNSIGNED.pkg" in readme
    assert readme.index("| macOS Apple Silicon |") < readme.index(
        'python -m pip install "napari[pyqt6]>=0.6"'
    )


def test_product_tagline_is_consistent_across_primary_surfaces():
    tagline = "Visual image processing made approachable"
    supporting_promise = "visual workflows for reproducible bioimage analysis"
    surfaces = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "README.md",
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
        normalized = " ".join(surface.read_text(encoding="utf-8").lower().split())
        assert supporting_promise in normalized


def test_main_readme_routes_users_to_the_single_public_manual():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    redirects = json.loads(
        (REPO_ROOT / "docs/public-guide-redirects.json").read_text(encoding="utf-8")
    )
    assert len(readme.splitlines()) <= 300
    assert "## GPU Acceleration (Optional)" in readme
    assert "stable/how-to/choose-compute/" in readme
    assert "stable/reference/example-workflows/" in readme
    assert "(docs/planning.md)" in readme
    assert "(docs/architecture.md)" in readme
    for filename in redirects:
        assert f"(docs/{filename}" not in readme
    assert "### GPU Execution And Development Environment" not in readme
    assert "23.57x" not in readme


def test_safe_gpu_dtype_repair_is_explained_consistently():
    documents = (
        REPO_ROOT / "docs" / "planning.md",
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


def test_portable_gpu_segmentation_bridge_is_explained_consistently():
    documents = (REPO_ROOT / "docs" / "durable-gpu-execution.md",)

    for document in documents:
        text = " ".join(document.read_text(encoding="utf-8").split())
        assert "cupy-extract-channel-view-v1" in text, document.name
        assert "cupy-binary-threshold-f32-exact-v1" in text, document.name
        assert "cupyx-remove-small-objects-bool-v1" in text, document.name
        assert "cupyx-fill-holes-all-v1" in text, document.name
        assert "integer labels" in text.lower(), document.name
        assert "positive" in text and "hole" in text.lower(), document.name
        assert "allocation-sharing view" in text, document.name
        assert "one upload and one download" in text or (
            "one host-to-device and one device-to-host" in text
        ), document.name
        assert "retained terminal" in text, document.name
        assert "Prefer GPU" in text, document.name
        assert "fallback" in text, document.name


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
    assert (REPO_ROOT / "docs" / "analytical-phantom-validation.md").read_text(
        encoding="utf-8"
    ) == module.render_markdown(checks)
