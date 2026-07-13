from __future__ import annotations

from collections import Counter
from pathlib import Path

from napari_vipp._widget import EXAMPLE_WORKFLOWS

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGED_EXAMPLES_DIR = PACKAGE_ROOT / "examples"
REPOSITORY_EXAMPLES_DIR = REPO_ROOT / "examples"


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _workflow_filenames(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.json")}


def test_example_registry_ids_and_filenames_are_unique():
    example_ids = [spec.id for spec in EXAMPLE_WORKFLOWS]
    filenames = [spec.filename for spec in EXAMPLE_WORKFLOWS]

    assert not _duplicates(example_ids)
    assert not _duplicates(filenames)


def test_example_workflow_filename_sets_match_registry():
    registry_filenames = {spec.filename for spec in EXAMPLE_WORKFLOWS}
    packaged_filenames = _workflow_filenames(PACKAGED_EXAMPLES_DIR)
    repository_filenames = _workflow_filenames(REPOSITORY_EXAMPLES_DIR)

    assert packaged_filenames == registry_filenames
    assert repository_filenames == registry_filenames


def test_packaged_and_repository_example_workflows_are_byte_identical():
    for spec in EXAMPLE_WORKFLOWS:
        packaged = PACKAGED_EXAMPLES_DIR / spec.filename
        repository = REPOSITORY_EXAMPLES_DIR / spec.filename

        assert packaged.read_bytes() == repository.read_bytes(), spec.filename
