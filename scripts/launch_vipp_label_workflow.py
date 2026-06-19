"""Launch napari with the Otsu red-channel labeling workflow preloaded."""

from __future__ import annotations

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
    workflow = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "otsu-red-channel-labels.json"
    )
    widget.load_workflow_file(workflow)
    widget.graph_view.select_node("filter_labels_by_volume_1")
    napari.run()


if __name__ == "__main__":
    main()
