"""Filter inspector counts belong to current full-resolution calculated outputs."""

import numpy as np
import pytest

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
    _select,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_STALE, NODE_EXECUTION_BYPASS
from napari_vipp.ui.inspector import (
    FILTER_RESULT_SECTION,
    LABEL_DISTRIBUTION_SECTION,
    OBJECT_FILTER_OPERATION_IDS,
    PARAMETERS_SECTION,
)


def _ready_filter(qtbot, operation, source=None, *, axes="YX"):
    if source is None:
        source = np.zeros((8, 8), dtype=np.uint16)
        source[1:3, 1:3] = 4
        source[4:6, 4:6] = 900
    result = np.where(source == 4, source, 0)
    source.setflags(write=False)
    result.setflags(write=False)
    widget = _widget(qtbot, source, axes=axes)
    upstream = widget.add_node_from_palette(
        "binary_threshold" if source.dtype == bool else "label_connected_components"
    )
    widget._filter_test_source_id = upstream.id
    node = widget.add_node_from_palette(operation)
    widget._connect_nodes(upstream.id, node.id, target_port=0)
    _publish_array_output(widget, "input", source, axes=axes)
    _publish_array_output(widget, upstream.id, source, axes=axes)
    _publish_array_output(widget, node.id, result, axes=axes)
    # Match accepted run publication; this fixture deliberately suppresses
    # the real debounce runner while supplying its completed cache outputs.
    widget._pending_dirty_node_ids.clear()
    _select(widget, node.id)
    return widget, node, source, result


@pytest.mark.parametrize("operation", sorted(OBJECT_FILTER_OPERATION_IDS))
def test_filter_counts_follow_real_label_ids_and_live_result(qtbot, operation):
    widget, node, source, result = _ready_filter(qtbot, operation)
    panel = widget.object_filter_feedback

    def ready():
        assert panel.counts_label.text() == "2 input objects · 1 kept · 1 removed"

    qtbot.waitUntil(ready)
    assert not panel.isHidden()
    assert "Full 2D image" in panel.scope_label.text()
    profile = widget._inspector_profile_for_node(node.id)
    expected_order = (
        PARAMETERS_SECTION, LABEL_DISTRIBUTION_SECTION, FILTER_RESULT_SECTION
    )
    assert profile.primary_sections[:3] == expected_order
    layout_indices = [
        widget._inspector_layout.indexOf(widget._inspector_sections[name])
        for name in expected_order
    ]
    assert all(index >= 0 for index in layout_indices)
    assert layout_indices == sorted(layout_indices)
    assert not source.flags.writeable and not result.flags.writeable

    # Stale cached pixels deliberately remain available for inspection, but
    # cannot describe what the newly selected parameters actually retained.
    widget.pipeline.node_execution_states[node.id] = EXECUTION_STALE
    widget._sync_execution_ui()
    assert "Awaiting calculation" in panel.counts_label.text()
    assert not panel.scope_label.text()
    assert widget.pipeline.outputs[node.id] is result

    _publish_array_output(widget, node.id, source, axes="YX")
    widget._sync_execution_ui()
    qtbot.waitUntil(
        lambda: "2 input objects · 2 kept · 0 removed" == panel.counts_label.text()
    )


def test_ready_cached_sibling_counts_survive_latest_run_completion_set(qtbot):
    widget, node, _source, result = _ready_filter(qtbot, "filter_labels_by_volume")
    panel = widget.object_filter_feedback
    # A later run can calculate another branch while retaining this ready
    # filter's cached result without including it in that run's completed set.
    widget.pipeline.completed_node_ids.discard(node.id)
    widget._sync_execution_ui()
    qtbot.waitUntil(
        lambda: panel.counts_label.text() == "2 input objects · 1 kept · 1 removed"
    )
    assert node.id not in widget.pipeline.completed_node_ids
    assert widget.pipeline.outputs[node.id] is result

    # Cached sibling results still must not be described as current after an edit.
    widget._pending_dirty_node_ids.add(node.id)
    widget._sync_execution_ui()
    assert "Awaiting calculation" in panel.counts_label.text()


def test_filter_counts_count_same_id_in_separate_frames(qtbot):
    source = np.zeros((3, 8, 8), dtype=np.uint16)
    source[:, 1:3, 1:3] = 4
    source[:, 4:6, 4:6] = 900
    widget, _node, _source, _result = _ready_filter(
        qtbot,
        "filter_labels_by_volume",
        source,
        axes="TYX",
    )
    panel = widget.object_filter_feedback
    qtbot.waitUntil(
        lambda: "6 input objects · 3 kept · 3 removed" == panel.counts_label.text()
    )
    assert "3 independent 2D images" in panel.scope_label.text()


def test_property_counts_use_semantic_axes_not_interleaved_channels(qtbot):
    source = np.zeros((4, 2, 8, 8), dtype=np.uint16)
    source[:, :, 1:3, 1:3] = 4
    source[:, :, 4:6, 4:6] = 900
    widget, _node, _source, _result = _ready_filter(
        qtbot,
        "filter_labels_by_property",
        source,
        axes="ZCYX",
    )
    panel = widget.object_filter_feedback
    qtbot.waitUntil(
        lambda: "4 input objects · 2 kept · 2 removed" == panel.counts_label.text()
    )
    assert "2 independent 3D volumes" in panel.scope_label.text()


def test_bypass_missing_data_and_selection_do_not_claim_filtering(qtbot):
    widget, node, source, _result = _ready_filter(qtbot, "remove_small_objects")
    panel = widget.object_filter_feedback
    widget.pipeline.restore_node_execution_mode(node.id, NODE_EXECUTION_BYPASS)
    widget._update_object_filter_feedback()
    assert panel.counts_label.text() == "Bypassed · no filtering was applied."
    _select(widget, "input")
    assert panel.isHidden()
    # Late diagnostics cannot put the previous node's counts on Image Source.
    assert panel.diagnostics.wait(5000)
    qtbot.wait(20)
    assert panel.isHidden()
    assert np.count_nonzero(source) == 8


def test_missing_upstream_cache_does_not_invent_zero_objects(qtbot):
    widget, node, _source, _result = _ready_filter(qtbot, "filter_labels_by_volume")
    widget.pipeline.outputs.pop(widget._filter_test_source_id)
    widget.pipeline.node_outputs.pop(widget._filter_test_source_id, None)
    widget._update_object_filter_feedback()
    assert "not cached" in widget.object_filter_feedback.counts_label.text()
    assert "0 input" not in widget.object_filter_feedback.counts_label.text()


def test_boolean_border_counts_use_full_connectivity(qtbot):
    from napari_vipp.core.operations import clear_border_objects

    source = np.zeros((8, 8), dtype=bool)
    source[0, 0] = source[1, 1] = True  # One diagonal border-touching component.
    source[4:6, 4:6] = True
    result = clear_border_objects(source, resolved_spatial_ndim=2)
    widget, node, _source, _result = _ready_filter(
        qtbot,
        "clear_border_objects",
        source,
    )
    _publish_array_output(widget, node.id, result, axes="YX")
    widget._update_object_filter_feedback()
    panel = widget.object_filter_feedback
    qtbot.waitUntil(
        lambda: "2 input objects · 1 kept · 1 removed" == panel.counts_label.text()
    )
    assert "full connectivity" in panel.counts_label.toolTip()


def test_parameter_edit_replaces_confirmed_counts_until_recalculated(qtbot):
    widget, _node, _source, _result = _ready_filter(qtbot, "filter_labels_by_volume")
    panel = widget.object_filter_feedback
    qtbot.waitUntil(lambda: "1 kept" in panel.counts_label.text())
    widget._on_param_changed("min_volume", 100)
    widget._debounce_timer.stop()
    assert "Awaiting calculation" in panel.counts_label.text()
    assert "1 kept" not in panel.counts_label.text()


def test_real_pipeline_publication_populates_counts(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    data = np.zeros((32, 32), dtype=np.float32)
    data[4:8, 4:8] = 10
    data[20:28, 20:28] = 10
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    qtbot.addWidget(widget)
    labels = widget.add_node_from_palette("label_connected_components")
    filtered = widget.add_node_from_palette("filter_labels_by_volume")
    widget.pipeline.set_param(filtered.id, "min_volume", 25)
    widget._connect_nodes("threshold", labels.id)
    widget._connect_nodes(labels.id, filtered.id)
    _select(widget, filtered.id)
    panel = widget.object_filter_feedback
    qtbot.waitUntil(lambda: "input objects" in panel.counts_label.text())
    before = widget.pipeline.input_data_by_port_for_node(filtered.id)[0]
    after = widget.pipeline.outputs[filtered.id]
    count_in = len(np.unique(before[before > 0]))
    count_out = len(np.unique(after[after > 0]))
    assert count_in > 0
    assert panel.counts_label.text() == (
        f"{count_in} input objects · {count_out} kept · {count_in - count_out} removed"
    )
