"""Launch napari with a VIPP example workflow preloaded."""

from __future__ import annotations

import sys
from pathlib import Path

import napari

from napari_vipp._widget import VippWidget


def main() -> None:
    viewer = napari.Viewer()
    widget = VippWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        area="bottom",
        name="VIPP Workflow",
    )
    workflow, selected_node = _workflow_args(sys.argv[1:])
    widget.load_workflow_file(workflow)
    widget.graph_view.select_node(selected_node)
    napari.run()


def _workflow_args(args: list[str]) -> tuple[Path, str]:
    examples = Path(__file__).resolve().parents[1] / "examples"
    workflows = {
        "intensity": (
            examples / "red-channel-object-intensity-measurements.json",
            "measure_objects_intensity_1",
        ),
        "merged": (
            examples / "red-channel-merged-measurement-table.json",
            "add_metadata_columns_1",
        ),
        "morphology": (
            examples / "synthetic-derived-object-morphology.json",
            "measure_objects_1",
        ),
        "mesh": (
            examples / "synthetic-3d-mesh-morphology.json",
            "measure_3d_mesh_morphology_1",
        ),
        "object-coloc": (
            examples / "synthetic-object-colocalization-association.json",
            "object_colocalization_metrics_1",
        ),
        "deconvolution": (
            examples / "synthetic-deconvolution-rl-tv.json",
            "richardson_lucy_tv_deconvolution_1",
        ),
        "deconvolution-3d": (
            examples / "synthetic-3d-deconvolution-rl-tv.json",
            "richardson_lucy_tv_deconvolution_1",
        ),
    }
    if not args:
        return workflows["intensity"]
    first = args[0]
    if first in workflows:
        workflow, default_node = workflows[first]
        return workflow, args[1] if len(args) > 1 else default_node
    candidate = Path(first).expanduser()
    if candidate.exists() or first.endswith(".json"):
        default_node = args[1] if len(args) > 1 else "add_metadata_columns_1"
        return candidate, default_node
    workflow, _default_node = workflows["intensity"]
    return workflow, first


if __name__ == "__main__":
    main()
