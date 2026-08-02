from __future__ import annotations

import ast
import importlib
from pathlib import Path

from napari_vipp.core import operations

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_operations_public_exports_are_the_extracted_callables() -> None:
    module = importlib.import_module("napari_vipp.core.sigma_filter")

    assert operations.sigma_filter is module.sigma_filter
    assert operations.sigma_filter_footprint is module.sigma_filter_footprint
    assert module.__all__ == ["sigma_filter", "sigma_filter_footprint"]


def test_sigma_cpu_and_gpu_modules_do_not_depend_on_operations_privates() -> None:
    module_paths = (
        PROJECT_ROOT / "src" / "napari_vipp" / "core" / "sigma_filter.py",
        PROJECT_ROOT / "src" / "napari_vipp" / "core" / "gpu" / "cupy_sigma.py",
    )

    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        operations_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "napari_vipp.core.operations"
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "napari_vipp.core.operations" for alias in node.names
                )
            )
        ]
        assert operations_imports == [], path


def test_operations_boundary_reexports_only_public_sigma_names() -> None:
    path = PROJECT_ROOT / "src" / "napari_vipp" / "core" / "operations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "napari_vipp.core.sigma_filter"
    ]

    aliases = [alias for node in imports for alias in node.names]
    assert [alias.name for alias in aliases] == [
        "sigma_filter",
        "sigma_filter_footprint",
    ]
    assert all(alias.asname == alias.name for alias in aliases)
    assert all(not alias.name.startswith("_") for alias in aliases)
