"""Launch napari with the VIPP widget and synthetic sample data."""

from __future__ import annotations

import napari

from napari_vipp._widget import VippWidget


def main() -> None:
    viewer = napari.Viewer()
    widget = VippWidget(viewer)
    widget.pipeline.nodes["input"].params.update(
        {
            "source_mode": "sample",
            "sample_name": "VIPP synthetic volume",
        }
    )
    widget.run_pipeline()
    widget.graph_view.select_node("input")
    viewer.window.add_dock_widget(
        widget,
        area="bottom",
        name="VIPP Workflow",
    )
    napari.run()


if __name__ == "__main__":
    main()
