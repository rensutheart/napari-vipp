from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QBoxLayout,
    QFormLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QWidget,
)

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
    _select,
    _widget,
)
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import (
    INSPECTOR_COLOCALIZATION_DIAGNOSTICS_BREAKPOINT,
    INSPECTOR_DENSE_DIAGNOSTICS_BREAKPOINT,
    INSPECTOR_HEADER_ACTION_HORIZONTAL_PADDING,
    INSPECTOR_HEADER_STACK_BREAKPOINT,
    INSPECTOR_STACKED_FORM_BREAKPOINT,
    VippWidget,
    _InspectorNoteLabel,
)
from napari_vipp.core.pipeline import NODE_EXECUTION_BYPASS
from napari_vipp.ui.controls import BoolControl, ImageSourceControl
from napari_vipp.ui.palette_roles import blend_colors, theme_colors


def _assert_histogram_range(plot, expected) -> None:
    assert plot._x_range == pytest.approx(tuple(float(value) for value in expected))


@pytest.mark.parametrize(
    "operation_id",
    (
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    ),
)
def test_deconvolution_compares_observed_image_with_output_not_psf(
    qtbot,
    operation_id,
):
    """The PSF is scientific context, not the before-image distribution."""

    observed = np.linspace(100.0, 199.0, 100, dtype=np.float32).reshape(10, 10)
    psf = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    restored = np.linspace(300.0, 399.0, 100, dtype=np.float32).reshape(10, 10)
    widget = _widget(qtbot, observed, axes="YX")
    psf_source = widget.add_node_from_palette("gaussian_blur")
    deconvolution = widget.add_node_from_palette(operation_id)

    _publish_array_output(widget, "input", observed, axes="YX")
    _publish_array_output(widget, psf_source.id, psf, axes="YX")
    _publish_array_output(widget, deconvolution.id, restored, axes="YX")
    # Connect the auxiliary port first so this also guards against choosing an
    # input by connection insertion order rather than its declared role.
    widget._connect_nodes(psf_source.id, deconvolution.id, target_port=1)
    widget._connect_nodes("input", deconvolution.id, target_port=0)

    _select(widget, deconvolution.id)

    assert not widget.rescale_input_histogram_group.isHidden()
    assert widget.rescale_input_histogram_group.title() == "Image Input Histogram"
    _assert_histogram_range(widget.rescale_input_histogram_plot, (100.0, 199.0))
    assert widget.rescale_input_histogram_plot._x_range != pytest.approx((0.0, 1.0))
    assert not widget.histogram_group.isHidden()
    _assert_histogram_range(widget.histogram_plot, (300.0, 399.0))


def test_mask_image_compares_primary_image_with_output_not_mask(qtbot):
    """A Boolean mask must never replace the intensity image's input plot."""

    image = np.linspace(20.0, 119.0, 100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 3:9] = True
    masked = image.copy()
    masked[~mask] = -5.0
    widget = _widget(qtbot, image, axes="YX")
    mask_source = widget.add_node_from_palette("binary_threshold")
    operation = widget.add_node_from_palette("mask_image")

    _publish_array_output(widget, "input", image, axes="YX")
    _publish_array_output(widget, mask_source.id, mask, axes="YX")
    _publish_array_output(widget, operation.id, masked, axes="YX")
    widget._connect_nodes(mask_source.id, operation.id, target_port=1)
    widget._connect_nodes("input", operation.id, target_port=0)

    _select(widget, operation.id)

    assert not widget.rescale_input_histogram_group.isHidden()
    _assert_histogram_range(widget.rescale_input_histogram_plot, (20.0, 119.0))
    assert widget.rescale_input_histogram_plot._x_range != pytest.approx((0.0, 1.0))
    assert not widget.histogram_group.isHidden()
    _assert_histogram_range(
        widget.histogram_plot,
        (float(masked.min()), float(masked.max())),
    )


@pytest.mark.parametrize(
    "operation_id",
    (
        "assign_channel_colors",
        "reorder_axes",
        "set_microscope_metadata",
        "set_pixel_size",
    ),
)
def test_value_preserving_nodes_do_not_show_duplicate_before_after_distributions(
    qtbot,
    operation_id,
):
    """Exact value-preserving metadata/layout edits need one histogram only."""

    data = np.linspace(10.0, 49.0, 40, dtype=np.float32).reshape(5, 8)
    widget = _widget(qtbot, data, axes="YX")
    operation = widget.add_node_from_palette(operation_id)
    _publish_array_output(widget, "input", data, axes="YX")
    _publish_array_output(widget, operation.id, data, axes="YX")
    widget._connect_nodes("input", operation.id)

    _select(widget, operation.id)

    assert widget.rescale_input_histogram_group.isHidden()
    assert widget.histogram_group.isHidden()
    assert widget.histograms_section.isHidden()
    assert np.asarray(widget.rescale_input_histogram_plot._counts).size == 0
    assert np.asarray(widget.histogram_plot._counts).size == 0


@pytest.mark.parametrize(
    ("source_operation", "dtype", "expected_kind", "save_label"),
    (
        ("binary_threshold", bool, "mask", "Save mask…"),
        ("label_connected_components", np.int32, "labels", "Save labels…"),
    ),
)
def test_select_axis_slice_honors_effective_mask_or_label_type_end_to_end(
    qtbot,
    source_operation,
    dtype,
    expected_kind,
    save_label,
):
    source_data = np.zeros((3, 8, 9), dtype=dtype)
    source_data[:, 2:7, 3:8] = 1
    sliced = source_data[1]
    widget = _widget(qtbot, source_data, axes="ZYX")
    typed_source = widget.add_node_from_palette(source_operation)
    select_slice = widget.add_node_from_palette("select_axis_slice")

    _publish_array_output(widget, typed_source.id, source_data, axes="ZYX")
    widget._connect_nodes(typed_source.id, select_slice.id)
    _publish_array_output(widget, select_slice.id, sliced, axes="YX")

    _select(widget, select_slice.id)

    profile = widget._inspector_profile_for_node(select_slice.id)
    assert widget._node_output_type(select_slice.id) == expected_kind
    assert profile.output_action_kind == expected_kind
    assert profile.supports_pin
    assert widget.save_button.text() == save_label
    assert widget.save_button.isEnabled()
    assert widget.pin_button.isEnabled()
    if expected_kind == "mask":
        assert not widget.mask_summary_section.isHidden()
        assert widget.label_volume_group.isHidden()
    else:
        assert not widget.label_volume_group.isHidden()
        assert widget.mask_summary_section.isHidden()

    widget.pin_node(select_slice.id)

    pinned = widget._active_pinned_layer()
    assert pinned is not None
    assert pinned.layer_type == "labels"
    assert pinned.metadata["data_kind"] == expected_kind


def test_display_actions_are_disabled_until_selected_node_has_actual_data(qtbot):
    widget = _widget(qtbot, np.arange(64, dtype=np.uint16).reshape(8, 8), axes="YX")
    operation = widget.add_node_from_palette("gaussian_blur")
    widget._connect_nodes("input", operation.id)

    _select(widget, operation.id)

    assert widget._node_display_payload(operation.id)[0] is None
    assert not widget._node_can_pin(operation.id)
    assert not widget.pin_button.isEnabled()
    assert not widget.save_button.isEnabled()

    output = np.arange(64, dtype=np.uint16).reshape(8, 8)
    _publish_array_output(widget, operation.id, output, axes="YX")
    widget._sync_inspector_presentation()

    assert widget._node_can_pin(operation.id)
    assert widget.pin_button.isEnabled()
    assert widget.save_button.isEnabled()


def test_parameter_forms_stack_narrow_and_restore_wide_without_rebuilding(qtbot):
    data = np.arange(100, dtype=np.uint16).reshape(10, 10)
    widget = _widget(qtbot, data, axes="YX")
    mask_source = widget.add_node_from_palette("binary_threshold")
    operation = widget.add_node_from_palette("mask_image")
    widget._connect_nodes("input", operation.id, target_port=0)
    widget._connect_nodes(mask_source.id, operation.id, target_port=1)
    _select(widget, operation.id)
    outside_value = widget._parameter_widgets["outside_value"]
    outside_label = widget.parameter_form.labelForField(outside_value)

    narrow_width = INSPECTOR_STACKED_FORM_BREAKPOINT - 1
    widget.inspector_content.resize(narrow_width, 700)
    widget.inspector_viewport.resize(narrow_width, 700)
    widget._sync_inspector_responsive_layout()

    assert widget.parameter_form.rowWrapPolicy() == QFormLayout.WrapAllRows
    # Connection rows keep their icon + role + source pairing at every width;
    # their text wraps within the row instead of splitting into label/value
    # blocks like editable parameter forms.
    assert widget.connected_inputs_form.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert all(
        row.role_label.wordWrap() and row.source_label.wordWrap()
        for row in widget.connected_inputs_panel.rows
    )
    assert widget._parameter_widgets["outside_value"] is outside_value
    assert outside_label.wordWrap()
    assert outside_label.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred

    undo_count = len(widget._undo_stack)
    outside_value.value_box.setValue(7.0)
    widget._debounce_timer.stop()
    assert widget.pipeline.nodes[operation.id].params["outside_value"] == 7.0
    assert len(widget._undo_stack) == undo_count + 1

    wide_width = INSPECTOR_STACKED_FORM_BREAKPOINT + 200
    widget.inspector_content.resize(wide_width, 700)
    widget.inspector_viewport.resize(wide_width, 700)
    widget._sync_inspector_responsive_layout()

    assert widget.parameter_form.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert widget.connected_inputs_form.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert widget._parameter_widgets["outside_value"] is outside_value
    assert not outside_label.wordWrap()
    assert outside_label.sizePolicy().horizontalPolicy() == QSizePolicy.Preferred


def test_connected_input_card_tracks_variadic_and_colored_channel_ports(qtbot):
    widget = _widget(qtbot, np.zeros((12, 12), dtype=np.uint16), axes="YX")
    combine = widget.add_node_from_palette("combine_channels")
    _select(widget, combine.id)

    widget._on_combine_channels_input_count_changed(4)

    assert widget.connected_inputs_form.rowCount() == 4
    assert [row.role_label.text() for row in widget.connected_inputs_panel.rows] == [
        "Channel 1: Red",
        "Channel 2: Green",
        "Channel 3: Blue",
        "Channel 4: Magenta",
    ]

    widget._on_channel_color_changed(0, "Yellow")

    assert (
        widget.connected_inputs_panel.rows[0].role_label.text()
        == "Channel 1: Yellow"
    )


def test_connected_input_card_preserves_dynamic_source_output_label(qtbot):
    widget = _widget(qtbot)
    split = widget.add_node_from_palette("split_channels")
    add = widget.add_node_from_palette("add_images")
    widget._connect_nodes(split.id, add.id, target_port=0, source_port=2)

    _select(widget, add.id)

    assert "Split Channels · Ch 3" == (
        widget.connected_inputs_panel.rows[0].source_label.text()
    )


def test_connected_input_card_tracks_tunnel_connect_reroute_clear_and_remove(qtbot):
    widget = _widget(qtbot, np.zeros((12, 12), dtype=np.uint16), axes="YX")
    combine = widget.add_node_from_palette("combine_channels")
    blur = widget.add_node_from_palette("gaussian_blur")
    widget.pipeline.add_output_tunnel("Raw", "input", 0)
    widget._sync_port_tunnels()
    _select(widget, combine.id)

    assert widget.connected_inputs_panel.rows[0].binding.source_title is None

    widget._connect_input_to_tunnel("Raw", combine.id, 0)
    assert (
        widget.connected_inputs_panel.rows[0].source_label.text()
        == "Image Source · out"
    )

    assert widget._reroute_output_tunnel("Raw", blur.id, 0)
    assert (
        widget.connected_inputs_panel.rows[0].source_label.text()
        == "Gaussian Blur · out"
    )

    widget._clear_input_tunnel(combine.id, 0)
    assert widget.connected_inputs_panel.rows[0].binding.source_title is None

    widget._connect_input_to_tunnel("Raw", combine.id, 0)
    widget._remove_output_tunnel("Raw")
    assert widget.connected_inputs_panel.rows[0].binding.source_title is None


def test_connected_input_card_clears_when_upstream_node_is_deleted(qtbot):
    widget = _widget(qtbot, np.zeros((12, 12), dtype=np.uint16), axes="YX")
    blur = widget.add_node_from_palette("gaussian_blur")
    combine = widget.add_node_from_palette("combine_channels")
    widget._connect_nodes("input", blur.id)
    widget._connect_nodes(blur.id, combine.id, target_port=0)
    _select(widget, combine.id)

    assert (
        widget.connected_inputs_panel.rows[0].source_label.text()
        == "Gaussian Blur · out"
    )

    widget._delete_node(blur.id)

    assert widget.connected_inputs_panel.rows[0].binding.source_title is None
    assert widget.connected_inputs_panel.rows[0].source_label.text().startswith(
        "Not connected"
    )


def test_connected_input_card_clears_when_dynamic_source_ports_shrink(qtbot):
    widget = _widget(qtbot)
    split = widget.add_node_from_palette("split_channels")
    combine = widget.add_node_from_palette("combine_channels")
    widget._connect_nodes("input", split.id)
    widget._connect_nodes(split.id, combine.id, target_port=0, source_port=2)
    _select(widget, combine.id)

    assert (
        widget.connected_inputs_panel.rows[0].source_label.text()
        == "Split Channels · Ch 3"
    )

    one_channel = np.zeros((1, 10, 12), dtype=np.uint16)
    _publish_array_output(widget, "input", one_channel, axes="CYX")
    widget._refresh_dynamic_output_ports()

    assert widget.pipeline._input_connections(combine.id) == []
    assert widget.connected_inputs_panel.rows[0].binding.source_title is None


def test_narrow_measurement_booleans_keep_wrapped_labels_beside_checkboxes(
    qtbot,
):
    data = np.zeros((12, 12), dtype=np.float32)
    widget = _widget(qtbot, data, axes="YX")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_objects_intensity")
    widget._connect_nodes("input", labels.id)
    widget._connect_nodes(labels.id, measurements.id, target_port=0)
    widget._connect_nodes("input", measurements.id, target_port=1)
    _select(widget, measurements.id)

    boolean_names = (
        "include_shape_descriptors",
        "include_axis_descriptors",
        "include_2d_boundary_descriptors",
        "include_derived_shape_ratios",
        "include_2d_shape_moments",
    )
    controls = {
        name: widget._parameter_widgets[name] for name in boolean_names
    }
    assert all(isinstance(control, BoolControl) for control in controls.values())
    outer_labels = {
        name: widget.parameter_form.labelForField(control)
        for name, control in controls.items()
    }
    assert all(isinstance(label, QLabel) for label in outer_labels.values())

    changed = controls["include_axis_descriptors"]
    changed.checkbox.setChecked(True)
    widget._debounce_timer.stop()
    assert changed.value() is True
    assert widget.pipeline.nodes[measurements.id].params[
        "include_axis_descriptors"
    ] is True

    narrow_width = INSPECTOR_STACKED_FORM_BREAKPOINT - 1
    widget.inspector_content.resize(narrow_width, 700)
    widget.inspector_viewport.resize(narrow_width, 700)
    widget._sync_inspector_responsive_layout()

    for name, control in controls.items():
        assert widget._parameter_widgets[name] is control
        assert control.compact_label_mode
        assert control.inline_label.text() == outer_labels[name].text()
        assert control.inline_label.wordWrap()
        assert not control.inline_label.isHidden()
        assert outer_labels[name].isHidden()
        row_layout = control.layout()
        assert isinstance(row_layout, QBoxLayout)
        assert row_layout.direction() == QBoxLayout.LeftToRight
        assert row_layout.indexOf(control.inline_label) >= 0
        assert row_layout.indexOf(control.checkbox) >= 0

    long_label = controls["include_axis_descriptors"].inline_label
    assert long_label.heightForWidth(90) > long_label.fontMetrics().lineSpacing()
    compact_control = controls["include_axis_descriptors"]
    assert compact_control.hasHeightForWidth()
    assert (
        compact_control.heightForWidth(120)
        > compact_control.checkbox.sizeHint().height()
    )
    assert changed.value() is True

    wide_width = INSPECTOR_STACKED_FORM_BREAKPOINT + 200
    widget.inspector_content.resize(wide_width, 700)
    widget.inspector_viewport.resize(wide_width, 700)
    widget._sync_inspector_responsive_layout()

    for name, control in controls.items():
        assert widget._parameter_widgets[name] is control
        assert not control.compact_label_mode
        assert control.inline_label.isHidden()
        assert not outer_labels[name].isHidden()
        assert outer_labels[name].text() == control.spec.label
    assert changed.value() is True
    assert widget.pipeline.nodes[measurements.id].params[
        "include_axis_descriptors"
    ] is True


def test_mesh_morphology_guidance_is_concise_and_parameter_specific(qtbot):
    labels_data = np.zeros((4, 8, 9), dtype=np.int32)
    labels_data[1:3, 2:6, 3:7] = 1
    widget = _widget(qtbot, labels_data, axes="ZYX")
    labels = widget.add_node_from_palette("label_connected_components")
    measurements = widget.add_node_from_palette("measure_3d_mesh_morphology")
    _publish_array_output(widget, labels.id, labels_data, axes="ZYX")
    widget._connect_nodes(labels.id, measurements.id)

    _select(widget, measurements.id)

    notice = widget._parameter_widgets["operation_notice"]
    assert notice.text() == (
        "3D-only measurement: verify Z/Y/X axes and voxel spacing."
    )
    assert "minimum voxel count" not in notice.text().casefold()
    assert "mesh_status" in notice.toolTip()
    assert "mesh_error" in notice.toolTip()
    assert notice.accessibleDescription() == notice.toolTip()

    controls = {
        name: widget._parameter_widgets[name]
        for name in (
            "spatial_mode",
            "minimum_voxel_count",
            "include_convex_hull_metrics",
        )
    }
    for control in controls.values():
        label = widget.parameter_form.labelForField(control)
        assert control.toolTip()
        assert label.toolTip() == control.toolTip()
        assert all(
            child.toolTip() == control.toolTip()
            for child in control.findChildren(QWidget)
        )

    spatial_tooltip = controls["spatial_mode"].toolTip().casefold()
    minimum_tooltip = controls["minimum_voxel_count"].toolTip().casefold()
    hull_tooltip = controls["include_convex_hull_metrics"].toolTip().casefold()
    assert "3d-only" in spatial_tooltip
    assert "anisotropic" in spatial_tooltip
    assert "remain in the results table" in minimum_tooltip
    assert "nan" in minimum_tooltip
    assert "does not remove or relabel" in minimum_tooltip
    assert "smallest convex 3d polyhedron" in hull_tooltip
    assert "solidity" in hull_tooltip
    assert "mesh-to-hull surface-area ratio" in hull_tooltip
    assert "base mesh metrics remain available" in hull_tooltip


def test_explicit_parent_inspector_note_releases_narrow_wrapped_height(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    form = QFormLayout(parent)
    note = _InspectorNoteLabel(
        "Stack notice: this node processes each YX slice independently and "
        "does not use 3D neighborhoods. If another plane should be processed, "
        "use Reorder Axes first so the intended plane is YX.",
        parent,
    )
    form.addRow(note)

    parent.resize(780, 800)
    parent.show()
    qtbot.waitUntil(
        lambda: (
            note.width() > 700
            and note.height() > 0
            and note.minimumHeight() == note.maximumHeight() == note.height()
        )
    )
    wide_height = note.height()

    parent.resize(280, 800)
    qtbot.waitUntil(
        lambda: (
            note.width() < 300
            and note.height() > wide_height
            and note.minimumHeight() == note.maximumHeight() == note.height()
        )
    )
    narrow_height = note.height()

    parent.resize(780, 800)
    qtbot.waitUntil(
        lambda: (
            note.width() > 700
            and note.height() == wide_height
            and note.minimumHeight() == note.maximumHeight() == note.height()
        )
    )

    assert narrow_height > wide_height
    assert note.height() == wide_height


def test_narrow_crop_and_source_forms_drop_wide_minimums_but_keep_usable_controls(
    qtbot,
):
    data = np.zeros((12, 96, 128), dtype=np.uint16)
    widget = _widget(qtbot, data, axes="ZYX")
    crop = widget.add_node_from_palette("crop_stack")
    widget._connect_nodes("input", crop.id)
    _select(widget, crop.id)
    top_control = widget._parameter_widgets["top"]

    narrow_width = INSPECTOR_STACKED_FORM_BREAKPOINT - 1
    widget.inspector_content.resize(narrow_width, 700)
    widget.inspector_viewport.resize(narrow_width, 700)
    widget._sync_inspector_responsive_layout()

    assert widget.parameter_form_widget.minimumSizeHint().width() < 240
    assert top_control.slider.minimumWidth() >= 80
    assert top_control.value_box.minimumWidth() >= 70

    _select(widget, "input")
    source_control = widget._parameter_widgets["image_source"]
    assert isinstance(source_control, ImageSourceControl)
    widget.inspector_content.resize(narrow_width, 700)
    widget.inspector_viewport.resize(narrow_width, 700)
    widget._sync_inspector_responsive_layout()
    narrow_minimum = source_control.minimumSizeHint().width()

    assert source_control.form_layout.rowWrapPolicy() == QFormLayout.WrapAllRows
    assert narrow_minimum < INSPECTOR_STACKED_FORM_BREAKPOINT
    assert all(
        source_control.form_layout.labelForField(field)
        .sizePolicy()
        .horizontalPolicy()
        == QSizePolicy.Preferred
        for field in (
            source_control.mode_combo,
            source_control.axis_control,
            source_control.sample_row,
        )
    )

    wide_width = INSPECTOR_STACKED_FORM_BREAKPOINT + 200
    widget.inspector_content.resize(wide_width, 700)
    widget.inspector_viewport.resize(wide_width, 700)
    widget._sync_inspector_responsive_layout()

    assert source_control.form_layout.rowWrapPolicy() == QFormLayout.DontWrapRows
    assert widget._parameter_widgets["image_source"] is source_control
    assert source_control.minimumSizeHint().width() > narrow_minimum


def test_contextual_notes_name_values_and_follow_napari_qss_theme(qtbot):
    from napari._qt.qt_resources import get_stylesheet

    data = np.zeros((4, 8, 10), dtype=np.uint16)
    host = QMainWindow()
    host.setStyleSheet(
        get_stylesheet("dark", extra_variables={"font_size": "9pt"})
    )
    widget = VippWidget(
        _Viewer(data, metadata={"axes": "ZYX"}),
        parent=host,
        defer_initial_run=True,
    )
    widget.run_pipeline = lambda *args, **kwargs: None
    host.setCentralWidget(widget)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitUntil(lambda: widget.property("vippColorScheme") == "dark")

    threshold = widget.add_node_from_palette("otsu_threshold")
    widget._connect_nodes("input", threshold.id)

    _select(widget, threshold.id)

    spanning_notes = []
    for row in range(widget.parameter_form.rowCount()):
        item = widget.parameter_form.itemAt(row, QFormLayout.SpanningRole)
        if item is not None and isinstance(item.widget(), QLabel):
            spanning_notes.append(item.widget())
    visibility_note = next(
        note for note in spanning_notes if "Hidden settings preserved:" in note.text()
    )
    assert visibility_note.text() == (
        "Hidden settings preserved: Float histogram bins: 256; "
        "RGB/RGBA channel axis (-1 = scalar): -1."
    )
    assert "do not affect the current input or selected mode" in (
        visibility_note.toolTip()
    )
    assert "resolved input is not floating-point" in visibility_note.toolTip()
    assert "resolved input has no encoded RGB/RGBA axis" in (
        visibility_note.toolTip()
    )
    assert visibility_note.accessibleDescription() == visibility_note.toolTip()
    widget._add_operation_note(
        "2D operation — each YX slice is processed independently."
    )
    operation_note = widget._parameter_widgets["operation_notice"]

    def expected_note_colors() -> tuple[str, str]:
        colors = theme_colors(widget.parameter_form_widget.palette())
        return (
            blend_colors(colors.surface, colors.text, 0.82).name(),
            colors.warning.foreground.name(),
        )

    dark_secondary, dark_warning = expected_note_colors()
    qtbot.waitUntil(
        lambda: dark_secondary in visibility_note.styleSheet()
        and dark_warning in operation_note.styleSheet()
    )
    dark_styles = (
        visibility_note.styleSheet(),
        operation_note.styleSheet(),
    )

    host.setStyleSheet(
        get_stylesheet("light", extra_variables={"font_size": "9pt"})
    )
    qtbot.waitUntil(lambda: widget.property("vippColorScheme") == "light")
    light_secondary, light_warning = expected_note_colors()
    qtbot.waitUntil(
        lambda: light_secondary in visibility_note.styleSheet()
        and light_warning in operation_note.styleSheet()
    )
    assert visibility_note.styleSheet() != dark_styles[0]
    assert operation_note.styleSheet() != dark_styles[1]

    host.setStyleSheet(
        get_stylesheet("dark", extra_variables={"font_size": "9pt"})
    )
    qtbot.waitUntil(lambda: widget.property("vippColorScheme") == "dark")
    restored_secondary, restored_warning = expected_note_colors()
    qtbot.waitUntil(
        lambda: restored_secondary in visibility_note.styleSheet()
        and restored_warning in operation_note.styleSheet()
    )
    assert (restored_secondary, restored_warning) == (
        dark_secondary,
        dark_warning,
    )


def test_sole_hidden_channel_axis_uses_parameters_summary_guidance(qtbot):
    data = np.zeros((4, 8, 10), dtype=np.float32)
    widget = _widget(qtbot, data, axes="ZYX")
    unsharp = widget.add_node_from_palette("unsharp_mask")
    widget._connect_nodes("input", unsharp.id)

    _select(widget, unsharp.id)

    spanning_text = []
    for row in range(widget.parameter_form.rowCount()):
        item = widget.parameter_form.itemAt(row, QFormLayout.SpanningRole)
        if item is not None and isinstance(item.widget(), QLabel):
            spanning_text.append(item.widget().text())
    assert not any("Hidden setting" in text for text in spanning_text)

    guidance = widget.parameter_group.summary_label.toolTip()
    assert (
        "Hidden setting preserved: Channel axis (-1 = none): -1." in guidance
    )
    assert "do not affect the current input or selected mode" in guidance
    assert "resolved input is explicitly scalar" in guidance
    assert (
        widget.parameter_group.summary_label.accessibleDescription()
        == guidance
    )


def test_remove_small_label_objects_uses_input_size_distribution_and_marker(qtbot):
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:3, 1:3] = 1  # four pixels; removed by the authored cutoff
    labels[5:9, 5:9] = 2  # sixteen pixels; retained
    filtered = labels.copy()
    filtered[filtered == 1] = 0
    widget = _widget(qtbot, labels, axes="YX")
    label_source = widget.add_node_from_palette("label_connected_components")
    remove = widget.add_node_from_palette("remove_small_objects")
    widget._connect_nodes(label_source.id, remove.id)
    _publish_array_output(widget, label_source.id, labels, axes="YX")
    _publish_array_output(widget, remove.id, filtered, axes="YX")
    widget.pipeline.set_param(remove.id, "min_size", 6)

    _select(widget, remove.id)

    assert widget._node_output_type(remove.id) == "labels"
    assert widget.label_volume_group.title() == "Input Object Size Distribution"
    assert "2 objects" in widget.label_volume_summary.text()
    assert widget.label_volume_log_checkbox.text() == "Log size axis"
    assert widget.label_volume_plot.marker_values()["min"] == pytest.approx(6.0)
    assert "minimum marker" in widget.label_volume_interaction_hint.text()

    widget._on_label_volume_marker_changed("min", 10.0)

    assert widget.pipeline.nodes[remove.id].params["min_size"] == 10


def test_remove_small_mask_objects_uses_connected_components_and_connectivity(qtbot):
    mask = np.zeros((12, 12), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal: separate with face, joined with full connectivity
    mask[6:8, 7:10] = True
    filtered = mask.copy()
    filtered[1, 1] = False
    filtered[2, 2] = False
    widget = _widget(qtbot, mask, axes="YX")
    mask_source = widget.add_node_from_palette("binary_threshold")
    remove = widget.add_node_from_palette("remove_small_objects")
    widget._connect_nodes(mask_source.id, remove.id)
    _publish_array_output(widget, mask_source.id, mask, axes="YX")
    _publish_array_output(widget, remove.id, filtered, axes="YX")
    widget.pipeline.set_param(remove.id, "min_size", 2)

    _select(widget, remove.id)

    assert widget._node_output_type(remove.id) == "mask"
    assert widget.label_volume_group.title() == "Input Object Size Distribution"
    assert "3 objects" in widget.label_volume_summary.text()
    assert widget.label_volume_plot.marker_values()["min"] == pytest.approx(2.0)
    assert not widget.mask_summary_section.isHidden()

    widget._on_param_changed("connectivity", "Full connectivity")
    widget._debounce_timer.stop()

    assert "2 objects" in widget.label_volume_summary.text()
    assert widget._current_label_volume_key[-1] == "full connectivity"


def test_remove_small_object_diagnostic_refreshes_for_every_size_semantic_edit(
    qtbot,
    monkeypatch,
):
    data = np.zeros((3, 12, 12), dtype=bool)
    data[:, 2:8, 3:9] = True
    widget = _widget(qtbot, data, axes="ZYX")
    remove = widget.add_node_from_palette("remove_small_objects")
    widget._connect_nodes("threshold", remove.id)
    _select(widget, remove.id)
    refreshes = []
    monkeypatch.setattr(
        widget,
        "_update_label_volume_histogram",
        lambda: refreshes.append(dict(remove.params)),
    )

    widget._on_param_changed("min_size", 11)
    widget._on_param_changed("spatial_mode", "2D YX")
    widget._on_param_changed("connectivity", "Full connectivity")

    assert len(refreshes) == 3
    assert refreshes[-1]["connectivity"] == "Full connectivity"


@pytest.mark.parametrize(
    ("operation_id", "changed_names"),
    (
        ("remove_small_objects", {"min_size", "connectivity"}),
        ("filter_labels_by_property", {"min_value", "keep_mode"}),
    ),
)
def test_bulk_parameter_reconciliation_refreshes_node_specific_distribution(
    qtbot,
    monkeypatch,
    operation_id,
    changed_names,
):
    widget = _widget(qtbot)
    node = widget.add_node_from_palette(operation_id)
    _select(widget, node.id)
    refreshes = []
    monkeypatch.setattr(
        widget,
        "_update_label_volume_histogram",
        lambda: refreshes.append(node.id),
    )

    widget._reconcile_bulk_parameter_change(node.id, changed_names)

    assert refreshes == [node.id]


def test_unmaterialized_bypass_uses_concrete_resolved_output_port_type(qtbot):
    widget = _widget(qtbot, np.arange(64, dtype=np.uint16).reshape(8, 8), axes="YX")
    threshold = widget.pipeline.nodes["threshold"]
    assert threshold.output_type == "mask"

    widget.pipeline.set_node_execution_mode("threshold", NODE_EXECUTION_BYPASS)
    resolved = widget.pipeline.output_ports("threshold")[0].output_type

    assert resolved == "image"
    assert widget._node_output_type_for_payload("threshold", None, 0) == "image"


def test_born_wolf_psf_prioritizes_output_and_connected_metadata_context(qtbot):
    source = np.linspace(1.0, 20.0, 100, dtype=np.float32).reshape(10, 10)
    psf = np.zeros((9, 9), dtype=np.float32)
    psf[4, 4] = 1.0
    widget = _widget(qtbot, source, axes="YX")
    operation = widget.add_node_from_palette("born_wolf_psf")
    widget._connect_nodes("input", operation.id)
    _publish_array_output(widget, "input", source, axes="YX")
    _publish_array_output(widget, operation.id, psf, axes="YX")

    _select(widget, operation.id)

    assert not widget.connected_inputs_panel.isHidden()
    assert widget.rescale_input_histogram_group.isHidden()
    assert not widget.histogram_group.isHidden()
    _assert_histogram_range(widget.histogram_plot, (0.0, 1.0))


def test_colocalization_diagnostics_stack_only_in_a_narrow_inspector(qtbot):
    widget = _widget(qtbot)

    narrow_width = INSPECTOR_COLOCALIZATION_DIAGNOSTICS_BREAKPOINT - 1
    widget.inspector_content.resize(narrow_width, 800)
    widget.inspector_viewport.resize(narrow_width, 800)
    widget._sync_inspector_responsive_layout()
    assert (
        widget.colocalization_scatter_controls_layout.direction()
        == QBoxLayout.TopToBottom
    )
    assert (
        widget.colocalization_channel_histograms_layout.direction()
        == QBoxLayout.TopToBottom
    )
    assert widget.histogram_panels_layout.direction() == QBoxLayout.LeftToRight

    wide_width = INSPECTOR_COLOCALIZATION_DIAGNOSTICS_BREAKPOINT + 100
    widget.inspector_content.resize(wide_width, 800)
    widget.inspector_viewport.resize(wide_width, 800)
    widget._sync_inspector_responsive_layout()
    assert (
        widget.colocalization_scatter_controls_layout.direction()
        == QBoxLayout.LeftToRight
    )
    assert (
        widget.colocalization_channel_histograms_layout.direction()
        == QBoxLayout.LeftToRight
    )
    assert widget.histogram_panels_layout.direction() == QBoxLayout.LeftToRight


def test_paired_histograms_stack_at_their_smaller_breakpoint(qtbot):
    widget = _widget(qtbot)

    narrow_width = INSPECTOR_DENSE_DIAGNOSTICS_BREAKPOINT - 1
    widget.inspector_content.resize(narrow_width, 800)
    widget.inspector_viewport.resize(narrow_width, 800)
    widget._sync_inspector_responsive_layout()
    assert widget.histogram_panels_layout.direction() == QBoxLayout.TopToBottom

    wide_width = INSPECTOR_DENSE_DIAGNOSTICS_BREAKPOINT + 20
    widget.inspector_content.resize(wide_width, 800)
    widget.inspector_viewport.resize(wide_width, 800)
    widget._sync_inspector_responsive_layout()
    assert widget.histogram_panels_layout.direction() == QBoxLayout.LeftToRight


def test_inspector_header_stacks_actions_only_when_the_visible_buttons_do_not_fit(
    qtbot,
):
    widget = _widget(qtbot)
    labels = widget.add_node_from_palette("label_connected_components")
    _publish_array_output(
        widget,
        labels.id,
        np.ones((8, 8), dtype=np.int32),
        axes="YX",
    )
    _select(widget, labels.id)

    assert (
        widget.inspector_panel.horizontalScrollBarPolicy()
        == Qt.ScrollBarAlwaysOff
    )

    action_buttons = tuple(
        button
        for button in (
            widget.header_calculate_button,
            widget.pin_button,
            widget.save_button,
        )
        if not button.isHidden()
    )
    assert len(action_buttons) == 2
    required_width = sum(
        max(button.sizeHint().width(), button.minimumSizeHint().width())
        for button in action_buttons
    ) + widget.inspector_action_layout.spacing()

    narrow_width = required_width + INSPECTOR_HEADER_ACTION_HORIZONTAL_PADDING - 1
    widget.inspector_content.resize(narrow_width, 800)
    widget._sync_inspector_responsive_layout()

    assert widget.inspector_context_layout.direction() == QBoxLayout.TopToBottom
    assert widget.inspector_action_layout.direction() == QBoxLayout.TopToBottom
    assert widget.inspector_context_layout.contentsMargins().left() == 0

    fitting_width = required_width + INSPECTOR_HEADER_ACTION_HORIZONTAL_PADDING
    widget.inspector_content.resize(fitting_width, 800)
    widget._sync_inspector_responsive_layout()

    assert widget.inspector_context_layout.direction() == QBoxLayout.TopToBottom
    assert widget.inspector_action_layout.direction() == QBoxLayout.LeftToRight

    widget.inspector_content.resize(
        INSPECTOR_HEADER_STACK_BREAKPOINT + 100,
        800,
    )
    widget._sync_inspector_responsive_layout()

    assert widget.inspector_context_layout.direction() == QBoxLayout.LeftToRight
    assert widget.inspector_action_layout.direction() == QBoxLayout.LeftToRight
    assert widget.inspector_context_layout.contentsMargins().left() == 28


def test_image_colocalization_keeps_output_histogram_after_primary_scatter(qtbot):
    channel_1 = np.linspace(0.0, 100.0, 100, dtype=np.float32).reshape(10, 10)
    channel_2 = np.flip(channel_1, axis=1).copy()
    output = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    widget = _widget(qtbot, channel_1, axes="YX")
    operation = widget.add_node_from_palette("racc_index")
    widget._connect_nodes("input", operation.id, target_port=0)
    widget._connect_nodes("gaussian", operation.id, target_port=1)
    _publish_array_output(widget, "input", channel_1, axes="YX")
    _publish_array_output(widget, "gaussian", channel_2, axes="YX")
    _publish_array_output(widget, operation.id, output, axes="YX")

    _select(widget, operation.id)

    assert not widget.colocalization_scatter_group.isHidden()
    assert not widget.colocalization_input_histograms_panel.isHidden()
    assert not widget.histogram_group.isHidden()
    assert widget.histogram_group.title() == "Output Histogram"
    _assert_histogram_range(widget.histogram_plot, (0.0, 1.0))
