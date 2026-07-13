from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PACKAGE_ROOT / "core"
UI_ROOT = PACKAGE_ROOT / "ui"

FORBIDDEN_IMPORT_ROOTS = {
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "napari",
    "qtpy",
}
FORBIDDEN_UI_MODULES = {
    "napari_vipp._graph",
    "napari_vipp._theme",
    "napari_vipp._widget",
    "napari_vipp.ui",
}


def _package_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(relative.parts[:-1])


def _import_candidates(
    node: ast.Import | ast.ImportFrom,
    package: str,
) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]

    if node.level:
        base = resolve_name(f"{'.' * node.level}{node.module or ''}", package)
    else:
        base = node.module or ""
    candidates = [base] if base else []
    candidates.extend(
        f"{base}.{alias.name}" if base else alias.name
        for alias in node.names
        if alias.name != "*"
    )
    return candidates


def _is_forbidden(module: str) -> bool:
    if module.partition(".")[0] in FORBIDDEN_IMPORT_ROOTS:
        return True
    return any(
        module == ui_module or module.startswith(f"{ui_module}.")
        for ui_module in FORBIDDEN_UI_MODULES
    )


def test_core_does_not_import_ui_frameworks_or_ui_modules():
    violations: list[str] = []

    for path in sorted(CORE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        package = _package_name(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            forbidden = next(
                (
                    candidate
                    for candidate in _import_candidates(node, package)
                    if _is_forbidden(candidate)
                ),
                None,
            )
            if forbidden is not None:
                relative = path.relative_to(PACKAGE_ROOT.parent)
                violations.append(f"{relative}:{node.lineno} -> {forbidden}")

    assert not violations, (
        "The headless core must not import napari, Qt, or VIPP UI modules:\n"
        + "\n".join(violations)
    )


def test_ui_components_do_not_import_the_widget_composition_root():
    """Keep reusable UI components independent of the npe2 widget facade."""
    violations: list[str] = []

    for path in sorted(UI_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        package = _package_name(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            imported_widget = next(
                (
                    candidate
                    for candidate in _import_candidates(node, package)
                    if candidate == "napari_vipp._widget"
                    or candidate.startswith("napari_vipp._widget.")
                ),
                None,
            )
            if imported_widget is not None:
                relative = path.relative_to(PACKAGE_ROOT.parent)
                violations.append(
                    f"{relative}:{node.lineno} -> {imported_widget}"
                )

    assert not violations, (
        "Reusable UI modules must not import the widget composition root:\n"
        + "\n".join(violations)
    )
