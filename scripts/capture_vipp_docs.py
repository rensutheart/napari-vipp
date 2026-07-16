"""Capture reproducible dark-theme screenshots for the VIPP user manual.

The screenshots use only bundled synthetic samples and example workflows.  Run
this from a development checkout after installing ``napari-vipp[dev]``::

    python scripts/capture_vipp_docs.py --output-dir path/to/docs/assets/screenshots

Private widget attributes are used deliberately here: this is a release asset
generator, not part of the public plugin API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import napari
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from napari_vipp._widget import VippWidget


@dataclass(frozen=True)
class CaptureSpec:
    filename: str
    example_id: str
    selected_node: str
    zoom: int
    capture_mode: Literal["context", "workflow", "viewer", "graph"] = "workflow"
    show_entire_graph: bool = False
    show_library: bool = True
    calculate_manual: bool = False
    pinned_node: str | None = None
    ndisplay: int = 2
    scenario: Literal[
        "standard",
        "manual-frontier",
        "isolated-tuning",
        "psf-preflight",
    ] = "standard"
    asset_group: Literal["public", "app-user-guide"] = "public"


CAPTURES = (
    CaptureSpec(
        "first-workflow-overview.png",
        "label-cleanup",
        "filter_labels_by_volume_1",
        43,
        capture_mode="context",
        show_entire_graph=True,
        show_library=False,
    ),
    CaptureSpec(
        "inspect-intermediate-result.png",
        "object-intensity",
        "otsu_threshold_1",
        78,
        capture_mode="context",
    ),
    CaptureSpec(
        "colocalization-parallel-branches.png",
        "racc-colocalization",
        "colocalization_metrics_1",
        62,
        show_library=False,
        calculate_manual=True,
        pinned_node="colocalized_voxels_1",
    ),
    CaptureSpec(
        "deconvolution-alternative-branches.png",
        "deconvolution-2d",
        "richardson_lucy_tv_deconvolution_1",
        66,
        show_library=False,
        calculate_manual=True,
        scenario="psf-preflight",
    ),
    CaptureSpec(
        "mesh-measurement-table.png",
        "mesh-morphology",
        "select_table_columns_1",
        68,
        show_library=False,
        calculate_manual=True,
        pinned_node="label_connected_components_1",
    ),
    CaptureSpec(
        "mesh-3d-result.png",
        "mesh-morphology",
        "label_connected_components_1",
        68,
        capture_mode="viewer",
        ndisplay=3,
    ),
    CaptureSpec(
        "manual-execution-frontier.png",
        "deconvolution-2d",
        "richardson_lucy_tv_deconvolution_1",
        54,
        show_entire_graph=True,
        show_library=False,
        scenario="manual-frontier",
    ),
    CaptureSpec(
        "isolated-node-tuning.png",
        "label-cleanup",
        "gaussian",
        58,
        show_library=False,
        scenario="isolated-tuning",
    ),
    CaptureSpec(
        "vipp-3d-deconvolution-workspace.png",
        "deconvolution-3d",
        "richardson_lucy_tv_deconvolution_1",
        54,
        capture_mode="context",
        show_entire_graph=True,
        calculate_manual=False,
        asset_group="app-user-guide",
    ),
    CaptureSpec(
        "vipp-3d-deconvolution-graph.png",
        "deconvolution-3d",
        "richardson_lucy_tv_deconvolution_1",
        80,
        capture_mode="graph",
        show_entire_graph=True,
        show_library=False,
        calculate_manual=False,
        asset_group="app-user-guide",
    ),
)


def _settle(milliseconds: int = 900) -> None:
    """Give Qt and the vispy canvas time to finish painting."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("Qt application was not created by napari.")
    app.processEvents()
    QTest.qWait(milliseconds)
    app.processEvents()


def _capture(spec: CaptureSpec, output_dir: Path) -> Path:
    viewer = napari.Viewer(title="VIPP documentation capture")
    viewer.theme = "dark"
    viewer.window.resize(1800, 1040)
    widget = VippWidget(viewer)
    dock = viewer.window.add_dock_widget(
        widget,
        area="bottom",
        name="VIPP Workflow",
    )
    qt_window = viewer.window._qt_window
    qt_window.resizeDocks([dock], [570], Qt.Vertical)
    widget.load_example_workflow(spec.example_id)
    widget._set_left_panel_visible(spec.show_library)
    widget._set_right_panel_visible(True)
    widget.splitter.setSizes([250 if spec.show_library else 0, 1080, 390])
    widget.graph_view.set_zoom_percent(spec.zoom)
    widget.graph_view.select_node(spec.selected_node)
    if spec.scenario == "manual-frontier":
        rescale = widget.add_node_from_palette("rescale_intensity")
        threshold = widget.add_node_from_palette("otsu_threshold")
        widget._connect_nodes(
            "richardson_lucy_tv_deconvolution_1",
            rescale.id,
        )
        widget._connect_nodes(rescale.id, threshold.id)
        widget._auto_structure_graph()
        widget.graph_view.select_node(spec.selected_node)
    elif spec.scenario == "isolated-tuning":
        widget.run_pipeline(force_sync=True)
        widget.graph_view.select_node(spec.selected_node)
        if not widget._start_isolated_tuning(spec.selected_node):
            raise RuntimeError("Could not start the isolated-tuning capture.")
        widget._on_param_changed("sigma", 0.6)
        widget._debounce_timer.stop()
        widget.run_pipeline(force_sync=True)
    if spec.calculate_manual:
        widget.run_pipeline(
            force_sync=True,
            manual_node_ids=set(widget.pipeline.manual_node_ids()),
        )
    if spec.pinned_node:
        widget.pin_node(spec.pinned_node)
        widget.graph_view.select_node(spec.selected_node)
    # Keep public assets deterministic and avoid recording machine-specific RAM.
    widget.cache_status_label.setText("Cache ready")
    widget.cache_status_label.setToolTip(
        "Machine-specific memory values are hidden in documentation captures."
    )
    proxy = widget.graph_view._proxies.get(spec.selected_node)
    if spec.show_entire_graph:
        widget.graph_view.centerOn(widget.graph_view.scene.itemsBoundingRect().center())
    elif proxy is not None:
        widget.graph_view.centerOn(proxy)
    if spec.scenario == "psf-preflight":
        notice = widget._parameter_widgets.get("operation_notice")
        if notice is not None:
            widget.inspector_panel.ensureWidgetVisible(notice, 0, 16)
    _settle()

    target = output_dir / spec.filename
    if spec.capture_mode == "workflow":
        dock.setFloating(True)
        dock.resize(1700, 900)
        dock.show()
        widget.splitter.setSizes([260 if spec.show_library else 0, 1020, 420])
        if spec.show_entire_graph:
            widget.graph_view.centerOn(
                widget.graph_view.scene.itemsBoundingRect().center()
            )
        elif proxy is not None:
            widget.graph_view.centerOn(proxy)
        if spec.scenario == "psf-preflight":
            notice = widget._parameter_widgets.get("operation_notice")
            if notice is not None:
                widget.inspector_panel.ensureWidgetVisible(notice, 0, 16)
        _settle()
        dock.grab().save(str(target))
    elif spec.capture_mode == "graph":
        dock.setFloating(True)
        dock.resize(1700, 900)
        dock.show()
        widget.splitter.setSizes([0, 1280, 0])
        if spec.show_entire_graph:
            widget.graph_view.centerOn(
                widget.graph_view.scene.itemsBoundingRect().center()
            )
        elif proxy is not None:
            widget.graph_view.centerOn(proxy)
        _settle()
        widget.graph_view.viewport().grab().save(str(target))
    else:
        if spec.capture_mode == "viewer":
            dock.hide()
            viewer.dims.ndisplay = spec.ndisplay
            viewer.reset_view()
            if spec.ndisplay == 3:
                viewer.camera.angles = (25.0, -35.0, 115.0)
            _settle()
        viewer.window.screenshot(
            path=target,
            flash=False,
            canvas_only=False,
        )
    viewer.close()
    _settle(150)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/screenshots"),
        help="Directory that will receive the PNG files.",
    )
    parser.add_argument(
        "--only",
        choices=[spec.filename for spec in CAPTURES],
        action="append",
        help="Capture only the named file; repeat to select several.",
    )
    parser.add_argument(
        "--group",
        choices=("public", "app-user-guide"),
        default="public",
        help="Named screenshot set to capture when --only is not supplied.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.only or ())
    captures = [
        spec
        for spec in CAPTURES
        if (spec.filename in selected if selected else spec.asset_group == args.group)
    ]
    for spec in captures:
        target = _capture(spec, output_dir)
        print(target)


if __name__ == "__main__":
    main()
