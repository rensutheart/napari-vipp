"""Open one acquired statistics-paper workflow in the installed VIPP build."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        type=Path,
        nargs="?",
        default=Path(
            "D:/VIPP-paper-reproductions/validation/statistics-paper/"
            "vipp/idr0139/1093711385/O02_F001/workflow.json"
        ),
    )
    parser.add_argument("--calculate", action="store_true")
    args = parser.parse_args()
    if not args.workflow.is_file():
        parser.error(f"Workflow does not exist: {args.workflow}")

    import napari

    from napari_vipp._widget import VippWidget

    viewer = napari.Viewer(title="VIPP — statistics-paper reproduction")
    widget = VippWidget(viewer)
    viewer.window.add_dock_widget(widget, area="bottom", name="VIPP Workflow")
    widget.load_workflow_file(args.workflow, prefer_image_source=True)
    if args.calculate:
        widget._calculate_all_nodes()
    napari.run()


if __name__ == "__main__":
    main()
