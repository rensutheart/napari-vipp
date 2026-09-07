"""Mesh-filter diagnostics describe full, immutable input object measurements."""

from dataclasses import replace

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtCore import QPoint, Qt

from napari_vipp._tests.test_mesh_objects import _cube
from napari_vipp._widget import VippWidget
from napari_vipp.core.mesh_objects import combine_meshes
from napari_vipp.core.pipeline import EXECUTION_READY, NODE_LIBRARY_BY_ID
from napari_vipp.ui.inspector import (
    HISTOGRAMS_SECTION,
    LABEL_DISTRIBUTION_SECTION,
    METADATA_SECTION,
    inspector_profile,
)


def _three_cubes(*, unit="nm", scale=1.0, open_final=False):
    cubes = []
    for object_id, side in enumerate((1, 2, 3), start=11):
        cube = _cube(object_id, unit=unit, scale=scale)
        cube = replace(cube, vertices=cube.vertices * side + (object_id - 11) * 5)
        if open_final and side == 3:
            cube = replace(
                cube, faces=cube.faces[:-1], face_object_ids=cube.face_object_ids[:-1]
            )
        cubes.append(cube)
    return combine_meshes(cubes), cubes[0]


def _publish_mesh(widget, node_id, mesh):
    pipeline = widget.pipeline
    pipeline.outputs[node_id] = mesh
    pipeline.output_states[node_id] = mesh.state
    pipeline.node_outputs[node_id] = [mesh]
    pipeline.node_output_states[node_id] = [mesh.state]
    pipeline.completed_node_ids.add(node_id)
    pipeline.node_execution_states[node_id] = EXECUTION_READY


def _mesh_inspector(qtbot, *, unit="nm", scale=1.0, open_final=False):
    viewer = ViewerModel()
    viewer.add_image(np.zeros((4, 4, 4), np.uint8), metadata={"axes": "ZYX"})
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *_args, **_kwargs: None
    source = widget.add_node_from_palette("mask_to_3d_mesh")
    filtered = widget.add_node_from_palette("filter_mesh_objects")
    widget._connect_nodes(source.id, filtered.id)
    mesh, first = _three_cubes(unit=unit, scale=scale, open_final=open_final)
    _publish_mesh(widget, source.id, mesh)
    # Deliberately different from the input: the histogram must not describe it.
    _publish_mesh(widget, filtered.id, first)
    widget._select_node(filtered.id)
    qtbot.waitUntil(
        lambda: widget._mesh_measurement_diagnostics.cached(mesh) is not None,
        timeout=5000,
    )
    qtbot.waitUntil(
        lambda: widget.label_volume_plot._hover_counts.size > 0, timeout=5000
    )
    return widget, filtered, mesh


def test_only_mesh_filter_prioritizes_its_input_measurement_distribution():
    profile = inspector_profile(NODE_LIBRARY_BY_ID["filter_mesh_objects"])
    assert profile.distribution_kind == "mesh_filter"
    assert LABEL_DISTRIBUTION_SECTION in profile.primary_sections
    assert METADATA_SECTION in profile.primary_sections
    assert profile.primary_sections.index(LABEL_DISTRIBUTION_SECTION) < (
        profile.primary_sections.index(METADATA_SECTION)
    )
    assert HISTOGRAMS_SECTION not in profile.primary_sections
    for operation_id in ("mask_to_3d_mesh", "smooth_mesh", "color_mesh_objects"):
        other = inspector_profile(NODE_LIBRARY_BY_ID[operation_id])
        assert other.distribution_kind == "none"
        assert LABEL_DISTRIBUTION_SECTION not in other.primary_sections


def test_histogram_uses_full_input_with_bounded_bins_and_no_image_placeholder(qtbot):
    widget, node, mesh = _mesh_inspector(qtbot)
    plot = widget.label_volume_plot
    table = widget._mesh_measurement_diagnostics.cached(mesh)

    assert table.row_count == mesh.object_count == 3
    assert widget.pipeline.outputs[node.id].object_count == 1
    assert plot._hover_counts.sum() == 3
    assert 1 <= plot._hover_counts.shape[1] <= 64
    assert len(plot._bin_edges) == plot._hover_counts.shape[1] + 1
    assert plot._x_range[0] <= 1 and plot._x_range[1] >= 27
    assert plot._x_range[1] < 100  # Not the enormous authored maximum.
    assert "Objects:" in plot._bin_tooltip(0)
    assert "[" in plot._bin_tooltip(0)
    assert not widget.label_volume_group.isHidden()
    assert widget.histograms_section.isHidden()
    assert widget.histogram_group.isHidden()
    assert widget.graph_view._cards[node.id].preview.isHidden()
    assert not widget._node_preview_enabled(node.id)
    assert plot._draggable_markers == {"min", "max"}


def test_bound_controls_markers_and_metric_changes_reuse_input_measurements(
    qtbot, monkeypatch
):
    from napari_vipp.ui import mesh_diagnostics

    widget, node, mesh = _mesh_inspector(qtbot)
    table = widget._mesh_measurement_diagnostics.cached(mesh)
    before_vertices = mesh.vertices.copy()
    before_faces = mesh.faces.copy()

    def unexpected_measurement(*_args, **_kwargs):
        pytest.fail("Display and range edits must reuse cached mesh measurements")

    monkeypatch.setattr(
        mesh_diagnostics, "measure_mesh_geometry", unexpected_measurement
    )
    widget._parameter_widgets["minimum"].value_box.setValue(8)
    widget._parameter_widgets["maximum"].value_box.setValue(27)
    widget._debounce_timer.stop()
    assert widget.label_volume_plot.marker_values() == {"min": 8.0, "max": 27.0}
    assert widget._parameter_widgets["maximum"].value_box.minimum() >= 8
    assert widget._parameter_widgets["minimum"].value_box.maximum() <= 27
    assert "3 input objects · 2 match · 0 unavailable" in (
        widget.label_volume_summary.text()
    )

    widget._parameter_widgets["maximum"].value_box.setValue(8)
    assert "3 input objects · 1 match · 0 unavailable" in (
        widget.label_volume_summary.text()
    )
    widget._on_param_changed("keep", "Outside range")
    assert "3 input objects · 2 match · 0 unavailable" in (
        widget.label_volume_summary.text()
    )
    widget._parameter_widgets["maximum"].value_box.setValue(27)

    widget._on_label_volume_marker_changed("min", 100)
    widget._debounce_timer.stop()
    assert node.params["minimum"] == node.params["maximum"] == 27
    assert widget._parameter_widgets["minimum"].value() == 27
    widget._on_label_volume_marker_changed("max", 1)
    widget._debounce_timer.stop()
    assert node.params["maximum"] == 27

    widget.label_volume_log_checkbox.setChecked(False)
    assert widget.label_volume_plot._x_scale == "linear"
    widget._on_param_changed("property_name", "triangle_count")
    widget._debounce_timer.stop()
    assert widget._parameter_widgets["minimum"].value_box.decimals() == 0
    assert "nm" not in widget.label_volume_plot._x_axis_label
    assert widget.label_volume_plot._hover_counts.sum() == 3
    control = widget._parameter_widgets["minimum"]
    control.slider.setValue(control.slider.minimum())
    widget._debounce_timer.stop()
    assert control.value() == node.params["minimum"]
    assert widget.label_volume_plot.marker_values()["min"] == control.value()
    assert widget._mesh_measurement_diagnostics.cached(mesh) is table
    np.testing.assert_array_equal(mesh.vertices, before_vertices)
    np.testing.assert_array_equal(mesh.faces, before_faces)
    assert not mesh.vertices.flags.writeable and not mesh.faces.flags.writeable


@pytest.mark.parametrize("scale", (1.0, 0.1))
def test_dragging_mesh_histogram_limit_updates_the_linked_numeric_control(qtbot, scale):
    widget, node, mesh = _mesh_inspector(qtbot, unit="mm", scale=scale)
    volume_scale = scale**3
    low, high, target = (value * volume_scale for value in (8, 27, 16))
    widget._parameter_widgets["minimum"].value_box.setValue(low)
    widget._parameter_widgets["maximum"].value_box.setValue(high)
    widget.label_volume_log_checkbox.setChecked(False)
    widget._debounce_timer.stop()
    table = widget._mesh_measurement_diagnostics.cached(mesh)
    plot = widget.label_volume_plot
    # Present the real connected plot independently of the scrollable inspector.
    plot.setParent(None)
    qtbot.addWidget(plot)
    plot.resize(600, 180)
    plot.show()
    qtbot.waitExposed(plot)
    rect = plot._plot_rect()
    start = QPoint(
        round(rect.left() + plot._x_fraction(low) * rect.width()), rect.center().y()
    )
    end = QPoint(
        round(rect.left() + plot._x_fraction(target) * rect.width()), rect.center().y()
    )
    with qtbot.waitSignal(plot.markerChanged):
        qtbot.mousePress(plot, Qt.LeftButton, pos=start)
        qtbot.mouseMove(plot, end)
        qtbot.mouseRelease(plot, Qt.LeftButton, pos=end)
    widget._debounce_timer.stop()

    assert node.params["minimum"] == pytest.approx(target, abs=0.1 * volume_scale)
    assert widget._parameter_widgets["minimum"].value() == node.params["minimum"]
    assert plot.marker_values()["min"] == node.params["minimum"]
    assert node.params["minimum"] <= node.params["maximum"]
    assert widget._mesh_measurement_diagnostics.cached(mesh) is table
    if scale < 1:
        assert 0 < node.params["minimum"] < 1
        assert widget._parameter_widgets["minimum"].value_box.decimals() >= 6


@pytest.mark.parametrize("unit", ("nm", "mm", None))
def test_mesh_histogram_labels_use_measurement_units(qtbot, unit):
    widget, _node, mesh = _mesh_inspector(qtbot, unit=unit)
    table = widget._mesh_measurement_diagnostics.cached(mesh)
    expected = dict(table.column_units)["mesh_volume_physical"]
    label = widget.label_volume_plot._x_axis_label
    # Permit plain-text or typographic powers, but never invent voxel counts.
    assert expected.replace("^3", "³") in label.replace("^3", "³")
    if unit is not None:
        assert "voxel" not in label.casefold()
        assert "voxels" not in widget.label_volume_summary.text().casefold()


def test_unavailable_mesh_volumes_are_excluded_from_histogram(qtbot):
    widget, _node, mesh = _mesh_inspector(qtbot, open_final=True)
    table = widget._mesh_measurement_diagnostics.cached(mesh)
    column = table.columns.index("mesh_volume_physical")
    values = np.array([row[column] for row in table.rows], dtype=float)
    assert table.row_count == 3 and np.isfinite(values).sum() == 2
    assert widget.label_volume_plot._hover_counts.sum() == 2
    widget._parameter_widgets["minimum"].value_box.setValue(1)
    widget._parameter_widgets["maximum"].value_box.setValue(8)
    assert "3 input objects · 2 match · 1 unavailable" in (
        widget.label_volume_summary.text()
    )
    widget._on_param_changed("keep", "Outside range")
    widget._debounce_timer.stop()
    assert "3 input objects · 0 match · 1 unavailable" in (
        widget.label_volume_summary.text()
    )
    assert widget.label_volume_plot._hover_counts.sum() == 2
