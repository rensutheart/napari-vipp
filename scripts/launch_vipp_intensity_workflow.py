"""Launch napari with any bundled or external VIPP workflow preloaded."""

from __future__ import annotations

import argparse
from pathlib import Path

import napari

from napari_vipp._widget import (
    EXAMPLE_WORKFLOWS,
    VippWidget,
    _example_workflow_path,
)

_LEGACY_ALIASES = {
    "intensity": "object-intensity",
    "merged": "merged-measurements",
    "morphology": "derived-morphology",
    "mesh": "mesh-morphology",
    "object-coloc": "object-colocalization",
    "deconvolution": "deconvolution-2d",
}


def main(argv: list[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    if args.list:
        print(_example_listing())
        return

    workflow, selected_node = _workflow_args(
        [value for value in (args.workflow, args.node) if value is not None]
    )
    viewer = napari.Viewer()
    widget = VippWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        area="bottom",
        name="VIPP Workflow",
    )
    widget.load_workflow_file(workflow)
    if selected_node:
        widget.graph_view.select_node(selected_node)
    napari.run()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a bundled example id or a workflow JSON file in VIPP.",
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        default="object-intensity",
        help="Bundled example id or path to workflow JSON.",
    )
    parser.add_argument(
        "node",
        nargs="?",
        help="Optional node id to select after loading.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List bundled example ids and exit.",
    )
    return parser


def _workflow_args(args: list[str]) -> tuple[Path, str | None]:
    """Resolve a bundled example id or explicit workflow path."""
    first = args[0] if args else "object-intensity"
    selected_node = args[1] if len(args) > 1 else None
    requested = _LEGACY_ALIASES.get(first, first)
    by_id = {spec.id: spec for spec in EXAMPLE_WORKFLOWS}
    by_filename = {spec.filename: spec for spec in EXAMPLE_WORKFLOWS}
    spec = by_id.get(requested) or by_filename.get(requested)
    if spec is not None:
        return _example_workflow_path(spec), selected_node

    candidate = Path(first).expanduser()
    if candidate.is_file():
        return candidate, selected_node
    if candidate.suffix.lower() == ".json":
        raise FileNotFoundError(f"Workflow file does not exist: {candidate}")
    raise ValueError(
        f"Unknown example workflow {first!r}. Use --list to show valid ids."
    )


def _example_listing() -> str:
    lines = ["Bundled VIPP example workflows:"]
    for spec in EXAMPLE_WORKFLOWS:
        lines.append(f"  {spec.id:<24} {spec.title} [{spec.filename}]")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
