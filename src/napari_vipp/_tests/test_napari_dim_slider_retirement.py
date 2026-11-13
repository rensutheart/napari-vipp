"""Retired napari slice rows must not remain visible while awaiting deletion."""

from types import SimpleNamespace

import numpy as np
from napari._qt.widgets.qt_dims import QtDims
from napari._qt.widgets.qt_dims_slider import QtDimSliderWidget
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

from napari_vipp._widget import VippWidget
from napari_vipp.ui.napari_compat import install_dim_slider_retirement_safety


def test_switch_to_2d_hides_retired_native_slice_rows_immediately(qtbot):
    viewer = ViewerModel()
    dim_panel = QtDims(viewer.dims)
    qtbot.addWidget(dim_panel)
    dim_panel.resize(800, 120)
    dim_panel.show()

    # Supply only the stock dimension panel, with no OpenGL or native window.
    proxy = SimpleNamespace(
        dims=viewer.dims,
        window=SimpleNamespace(qt_viewer=SimpleNamespace(dims=dim_panel)),
    )
    # The widget calls the compatibility hook on its viewer, so attach the
    # native presentation seam while keeping a real ViewerModel underneath.
    for name in ("layers", "add_image", "add_labels", "cursor", "scene"):
        setattr(proxy, name, getattr(viewer, name))
    widget = VippWidget(proxy, defer_initial_run=True)
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    widget.load_example_workflow("label-cleanup")
    widget.run_pipeline(force_sync=True)

    viewer.dims.ndisplay = 2
    assert viewer.dims.axis_labels == tuple("CZYX")
    assert [
        row.axis_label.text()
        for row in dim_panel.slider_widgets
        if row.isVisible()
    ] == ["C", "Z"]
    original = QtDims._remove_slider_widget
    widget.load_example_workflow("review-channels-2d")
    assert viewer.dims.ndim == 2
    assert viewer.dims.axis_labels == ("Y", "X")
    assert tuple(viewer.dims.nsteps) == (80, 104)
    assert QtDims._remove_slider_widget is original
    # Include widgets not tracked by QtDims anymore. They can still be
    # rendered and intercept input until Qt processes their DeferredDelete.
    assert not [
        row.axis_label.text()
        for row in dim_panel.findChildren(QtDimSliderWidget)
        if row.isVisible()
    ]


def test_retiring_focused_slice_editor_does_not_commit_and_guard_is_local(qtbot):
    viewer = ViewerModel()
    viewer.add_image(np.zeros((4, 6, 12, 16)), rgb=False)
    panel = QtDims(viewer.dims)
    unrelated_panel = QtDims(ViewerModel().dims)
    qtbot.addWidget(panel)
    qtbot.addWidget(unrelated_panel)
    panel.resize(800, 120)
    panel.show()
    editor = panel.slider_widgets[0].curslice_label
    editor.setFocus()
    QApplication.processEvents()
    assert editor.hasFocus()

    other = unrelated_panel._remove_slider_widget
    proxy = SimpleNamespace(
        dims=viewer.dims,
        window=SimpleNamespace(qt_viewer=SimpleNamespace(dims=panel)),
    )
    assert install_dim_slider_retirement_safety(proxy)
    wrapped = panel._remove_slider_widget
    assert install_dim_slider_retirement_safety(proxy)
    assert panel._remove_slider_widget is wrapped
    assert unrelated_panel._remove_slider_widget == other

    commits = []
    editor.editingFinished.connect(lambda: commits.append(True))
    editor.setText("3")
    viewer.layers.clear()
    assert commits == []
    assert tuple(viewer.dims.current_step) == (0, 0)
