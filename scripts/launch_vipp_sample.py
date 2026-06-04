"""Launch napari with the VIPP widget and synthetic sample data."""

from __future__ import annotations

import napari

from napari_vipp._sample_data import make_sample_data
from napari_vipp._widget import VippWidget


def main() -> None:
    viewer = napari.Viewer()
    for data, metadata, layer_type in make_sample_data():
        add = getattr(viewer, f"add_{layer_type}")
        add(data, **metadata)
    viewer.window.add_dock_widget(
        VippWidget(viewer),
        area="bottom",
        name="VIPP Workflow",
    )
    napari.run()


if __name__ == "__main__":
    main()
