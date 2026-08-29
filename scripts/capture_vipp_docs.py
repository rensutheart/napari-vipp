"""Capture reproducible dark-theme screenshots for the VIPP user manual.

The screenshots use only bundled synthetic samples and example workflows.  Run
this from a development checkout after installing ``napari-vipp[dev]``::

    python scripts/capture_vipp_docs.py --output-dir path/to/docs/assets/screenshots

Private widget attributes are used deliberately here: this is a release asset
generator, not part of the public plugin API.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import napari
from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QPixmap
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp._widget import VippWidget
from napari_vipp.core.pipeline import ParameterSpec
from napari_vipp.core.source_items import (
    ResolvedSourceItemIdentity,
    SourceCapabilities,
    SourceContainerBundle,
    SourceContainerMember,
    SourceItem,
    SourceItemSelector,
    SourceReaderDescriptor,
    SourceRevisionProof,
)
from napari_vipp.ui.batch import CollectionBatchActions, CollectionBatchDialog
from napari_vipp.ui.batch_overrides import (
    BatchOverrideParameterSpec,
    BatchOverrideSourceItem,
)
from napari_vipp.ui.controls import (
    ImageSourceControl,
    ImageSourceResolutionPresentation,
)
from napari_vipp.ui.dialogs import ExampleWorkflowDialog


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
    capture_width: int = 1700
    capture_height: int = 900


@dataclass(frozen=True)
class UiCaptureSpec:
    """Deterministic capture of a focused application UI surface."""

    filename: str
    subdirectory: Literal["sources", "workflows", "."]
    capture_kind: Literal[
        "image-source-resolution",
        "batch-workspace-overrides",
        "example-workflow-chooser",
    ]
    asset_group: Literal["source-batch", "app-user-guide"] = "source-batch"


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
        pinned_node="otsu_threshold_1",
    ),
    CaptureSpec(
        "colocalization-parallel-branches.png",
        "racc-colocalization",
        "colocalization_metrics_1",
        55,
        show_library=False,
        show_entire_graph=True,
        calculate_manual=True,
        pinned_node="colocalized_voxels_1",
        capture_height=1120,
    ),
    CaptureSpec(
        "deconvolution-alternative-branches.png",
        "deconvolution-2d",
        "richardson_lucy_tv_deconvolution_1",
        66,
        show_library=False,
        show_entire_graph=True,
        calculate_manual=True,
        scenario="psf-preflight",
    ),
    CaptureSpec(
        "mesh-measurement-table.png",
        "mesh-morphology",
        "select_table_columns_1",
        68,
        show_library=False,
        show_entire_graph=True,
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
        72,
        capture_mode="workflow",
        show_entire_graph=True,
        calculate_manual=True,
        asset_group="app-user-guide",
        capture_width=1800,
        capture_height=1000,
    ),
    CaptureSpec(
        "vipp-3d-deconvolution-graph.png",
        "deconvolution-3d",
        "richardson_lucy_tv_deconvolution_1",
        80,
        capture_mode="graph",
        show_entire_graph=True,
        show_library=False,
        calculate_manual=True,
        asset_group="app-user-guide",
        capture_width=2300,
        capture_height=1080,
    ),
)


UI_CAPTURES = (
    UiCaptureSpec(
        "image-source-multiscale-resolution.png",
        "sources",
        "image-source-resolution",
    ),
    UiCaptureSpec(
        "batch-workspace-overrides.png",
        "workflows",
        "batch-workspace-overrides",
    ),
    UiCaptureSpec(
        "vipp-example-chooser.png",
        ".",
        "example-workflow-chooser",
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
        dock.resize(spec.capture_width, spec.capture_height)
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
        dock.resize(spec.capture_width, spec.capture_height)
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


def _dark_capture_viewer() -> napari.Viewer:
    """Create a hidden viewer so focused widgets use napari's real dark theme."""

    viewer = napari.Viewer(title="VIPP focused documentation capture")
    viewer.theme = "dark"
    viewer.window._qt_window.hide()
    _settle(100)
    return viewer


def _save_widget(widget: QWidget, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    pixmap = widget.grab()
    if pixmap.isNull() or not pixmap.save(str(target), "PNG"):
        raise RuntimeError(f"Could not save documentation screenshot {target}.")


def _capture_image_source_resolution(target: Path) -> None:
    """Capture the real multiscale selector with deterministic synthetic facts."""

    viewer = _dark_capture_viewer()
    window = QWidget()
    window.setWindowTitle("VIPP documentation capture — Image Source")
    window.setObjectName("ImageSourceDocumentationCapture")
    window.setStyleSheet(viewer.window._qt_window.styleSheet())
    window.setFixedWidth(1240)
    window.move(80, 50)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(24, 20, 24, 20)
    layout.setSpacing(12)
    heading = QLabel(
        "<span style='font-size: 22px; font-weight: 650;'>Image Source</span>"
    )
    heading.setTextFormat(Qt.RichText)
    layout.addWidget(heading)
    note = QLabel(
        "Choose the napari display resolution here. Scientific processing and "
        "export remain fixed to analysis level 0."
    )
    note.setWordWrap(True)
    note.setStyleSheet("color: #94a3b8;")
    layout.addWidget(note)

    panel = QFrame()
    panel.setFrameShape(QFrame.StyledPanel)
    panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    panel.setStyleSheet(
        "QFrame { border: 1px solid #334155; border-radius: 5px; }"
        "QLabel, QWidget { border: none; }"
    )
    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(18, 16, 18, 16)
    control = ImageSourceControl(
        {
            "source_mode": "file path",
            "file_path": r"C:\VIPP Examples\multiscale-volume.ome.zarr",
            "series_index": 0,
            "binding_mode": "single item",
            # The deterministic OME-Zarr already reports ZYX. Trust it rather
            # than implying that a QYX-to-ZYX override is necessary.
            "axis_declaration": "",
        },
        layer_names=[],
        sample_names=[],
        series_options=[(0, "image")],
        source_summary="Selected image · OME-Zarr 0.5",
    )
    control.viewer_display_combo.setMinimumWidth(700)
    control.set_resolution_presentation(
        ImageSourceResolutionPresentation(
            analysis_axes="ZYX",
            analysis_shape=(12, 128, 160),
            level_shapes=((12, 128, 160), (12, 64, 80), (12, 32, 40)),
            preview_state="ready",
            preview_level=2,
            preview_shape=(12, 32, 40),
            viewer_choice="preview:auto",
            can_select_preview=True,
            can_retry=False,
        )
    )
    panel_layout.addWidget(control)
    layout.addWidget(panel)

    window.adjustSize()
    window.show()
    _settle(500)
    control.viewer_display_combo.showPopup()
    _settle(350)

    # QComboBox renders its menu as a separate native popup. Compose grabs of
    # the real window and real popup in logical Qt coordinates; this avoids
    # desktop bleed and DPI-dependent clipping without imitating either widget.
    base = window.grab()
    popup_window = control.viewer_display_combo.view().window()
    popup = popup_window.grab()
    pixel_ratio = float(base.devicePixelRatio())
    popup_ratio = float(popup.devicePixelRatio())
    popup_global = popup_window.mapToGlobal(popup_window.rect().topLeft())
    popup_position = window.mapFromGlobal(popup_global)
    panel_bottom = panel.mapTo(window, panel.rect().bottomLeft()).y()
    popup_bottom = popup_position.y() + (popup.height() / popup_ratio)
    capture_height = max(int(max(panel_bottom, popup_bottom) + 12), 1)
    pixmap = QPixmap(
        round(window.width() * pixel_ratio),
        round(capture_height * pixel_ratio),
    )
    pixmap.setDevicePixelRatio(pixel_ratio)
    pixmap.fill(window.palette().window().color())
    painter = QPainter(pixmap)
    painter.drawPixmap(0, 0, base)
    painter.drawPixmap(popup_position, popup)
    painter.end()
    target.parent.mkdir(parents=True, exist_ok=True)
    if pixmap.isNull() or not pixmap.save(str(target), "PNG"):
        raise RuntimeError(f"Could not save documentation screenshot {target}.")

    control.viewer_display_combo.hidePopup()
    window.close()
    viewer.close()
    _settle(150)


def _capture_sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _batch_source_item(seed: str, label: str) -> SourceItem:
    """Build stable SourceItem evidence without reading a private local path."""

    size = 3_932_160
    return SourceItem(
        SourceContainerBundle(
            uri=f"C:/VIPP Examples/source-aware/{seed}.ome.zarr",
            format="ome-zarr-0.5",
            revision=SourceRevisionProof(
                "directory",
                _capture_sha(seed),
                1,
                size,
            ),
            members=(
                SourceContainerMember(
                    "image/0",
                    _capture_sha(f"{seed}-member"),
                    size,
                ),
            ),
        ),
        SourceItemSelector(
            "image",
            "image",
            source_axes=("Z", "Y", "X"),
            effective_axes=("Z", "Y", "X"),
        ),
        SourceReaderDescriptor("ome-zarr", "ome-zarr", "0.5"),
        SourceCapabilities(
            pixel_lazy_inspection=True,
            lazy_data=True,
            level_enumeration=True,
            decoded_size_estimate=True,
        ),
        ResolvedSourceItemIdentity(
            key="image",
            name=label,
            kind="image",
            shape=(12, 128, 160),
            dtype="uint16",
            axes=("Z", "Y", "X"),
            raw_axes=("Z", "Y", "X"),
            estimated_decoded_bytes=size,
        ),
    )


def _batch_parameter(
    node_id: str,
    node_label: str,
    operation_id: str,
    name: str,
    label: str,
    kind: str,
    workflow_value: int | float,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
    decimals: int = 0,
) -> BatchOverrideParameterSpec:
    return BatchOverrideParameterSpec(
        node_id=node_id,
        node_label=node_label,
        operation_id=operation_id,
        parameter=ParameterSpec(
            name,
            label,
            kind,
            workflow_value,
            minimum,
            maximum,
            step,
            decimals,
        ),
        workflow_value=workflow_value,
    )


def _capture_batch_workspace(target: Path) -> None:
    """Capture completed metadata discovery and one typed sample override."""

    viewer = _dark_capture_viewer()
    actions = CollectionBatchActions(
        preview_batch=lambda _values, _limit: None,
        choose_demo=lambda _parent: None,
        source_rows=lambda: [],
        load_config=lambda _path: None,
        save_config=lambda _path, _values: (),
        preview_item=lambda _index: True,
    )
    dialog = CollectionBatchDialog(
        source_nodes=[
            {
                "node_id": "input",
                "title": "Image Source",
                "binding_mode": "collection",
                "axis_declaration": "",
            }
        ],
        actions=actions,
    )
    dialog.setStyleSheet(viewer.window._qt_window.styleSheet())
    dialog.resize(1740, 980)
    dialog.move(60, 30)

    source_row = dialog._source_rows[0]
    source_row["folder"].setText(r"C:\VIPP Examples\source-aware")
    source_row["pattern"].setText("*")
    source_row["axis_declaration"].setText("")
    dialog._set_output_path(
        r"C:\VIPP Examples\source-aware\output",
        suggested=False,
    )

    dim = _batch_source_item("0001_dim", "image")
    bright = _batch_source_item("0002_bright", "image")
    sources = [
        BatchOverrideSourceItem("input", "0001_dim · image", dim),
        BatchOverrideSourceItem("input", "0002_bright · image", bright),
    ]
    parameters = [
        _batch_parameter(
            "gaussian_blur_1",
            "Gaussian Blur",
            "gaussian_blur",
            "sigma",
            "Sigma",
            "float",
            1.0,
            0.0,
            100.0,
            0.1,
            2,
        ),
        _batch_parameter(
            "binary_threshold_1",
            "Binary Threshold",
            "binary_threshold",
            "threshold",
            "Threshold",
            "int",
            5000,
            0,
            65535,
            1,
        ),
        _batch_parameter(
            "remove_small_objects_1",
            "Remove Small Objects",
            "remove_small_objects",
            "minimum_size",
            "Minimum object size (pixels/voxels)",
            "int",
            80,
            0,
            1_000_000,
            1,
        ),
        _batch_parameter(
            "remove_outliers_1",
            "Remove Outliers (Binary)",
            "remove_outliers_binary",
            "radius",
            "Neighborhood radius (pixels/voxels)",
            "float",
            1.5,
            0.5,
            100.0,
            0.5,
            1,
        ),
    ]
    if not dialog.configure_parameter_overrides(sources, parameters, overrides=()):
        raise RuntimeError(dialog.parameter_override_editor.error_message)
    dialog.parameter_override_editor.editor_for(
        "input",
        bright,
        "binary_threshold_1",
        "threshold",
    ).setText("13000")
    table = dialog.parameter_override_editor.table
    table.setColumnWidth(0, 320)
    table.setColumnWidth(1, 230)
    table.setColumnWidth(2, 230)
    table.setColumnWidth(3, 360)
    table.setColumnWidth(4, 360)
    table.setMinimumHeight(170)
    table.setMaximumHeight(220)

    # This 2/2 state is completed metadata-only discovery, not a simulated
    # scientific run. Detailed run bars remain lower in the real workspace.
    dialog.show_workspace_activity(
        "Ready · 2 source items resolved",
        state="ready",
        current=2,
        total=2,
        progress_text="2 / 2",
        tooltip=(
            "Metadata-only discovery resolved two stable source items. No "
            "representative image has been calculated."
        ),
    )
    dialog.preview_status.setText(
        "Ready. Preview is optional; Run batch checks the plan again before saving."
    )
    dialog.content_scroll.verticalScrollBar().setValue(0)
    dialog.show()
    _settle(800)
    _save_widget(dialog, target)
    dialog.close()
    viewer.close()
    _settle(150)


def _capture_example_workflow_chooser(target: Path) -> None:
    """Capture the real chooser populated from the authoritative registry."""

    viewer = _dark_capture_viewer()
    dialog = ExampleWorkflowDialog()
    dialog.setStyleSheet(viewer.window._qt_window.styleSheet())
    dialog.resize(1180, 640)
    dialog.select_example("deconvolution-3d")
    dialog.show()
    _settle(600)
    _save_widget(dialog, target)
    dialog.close()
    viewer.close()
    _settle(150)


def _capture_ui(spec: UiCaptureSpec, output_dir: Path) -> Path:
    target = output_dir / spec.subdirectory / spec.filename
    if spec.capture_kind == "image-source-resolution":
        _capture_image_source_resolution(target)
    elif spec.capture_kind == "batch-workspace-overrides":
        _capture_batch_workspace(target)
    elif spec.capture_kind == "example-workflow-chooser":
        _capture_example_workflow_chooser(target)
    else:  # pragma: no cover - frozen Literal/dataclass route guard
        raise ValueError(f"Unsupported focused UI capture {spec.capture_kind!r}.")
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
        choices=[spec.filename for spec in (*CAPTURES, *UI_CAPTURES)],
        action="append",
        help="Capture only the named file; repeat to select several.",
    )
    parser.add_argument(
        "--group",
        choices=("public", "app-user-guide", "source-batch"),
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
        for spec in (*CAPTURES, *UI_CAPTURES)
        if (spec.filename in selected if selected else spec.asset_group == args.group)
    ]
    for spec in captures:
        target = (
            _capture(spec, output_dir)
            if isinstance(spec, CaptureSpec)
            else _capture_ui(spec, output_dir)
        )
        print(target)


if __name__ == "__main__":
    main()
