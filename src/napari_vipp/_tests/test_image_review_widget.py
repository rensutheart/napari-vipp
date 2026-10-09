"""Review sinks preserve scientific execution and remain workflow-owned."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDialog, QPushButton

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
    _widget,
)
from napari_vipp._tests.test_widget import _QueuedThreadPool, _Viewer
from napari_vipp._widget import CACHE_MODE_LOW_MEMORY, VippWidget
from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.execution import execute_pipeline_request
from napari_vipp.core.pipeline import EXECUTION_READY, EXECUTION_STALE
from napari_vipp.core.workflow import serialize_workflow


class _ReviewWindow(QDialog):
    closed = Signal()

    def __init__(self, *, inputs, settings, on_settings_changed, parent):
        super().__init__(parent)
        self.inputs = inputs
        self.settings = deepcopy(settings)
        self.commit = on_settings_changed
        self.current = True
        self.message = ""
        self.updates = []

    def update_inputs(self, inputs):
        if len(inputs) != len(self.inputs) or any(
            old.kind != new.kind or old.grid != new.grid
            for old, new in zip(self.inputs, inputs, strict=True)
        ):
            raise ValueError("Input type/grid/layout changed. Reopen the review.")
        self.inputs = inputs
        self.updates.append(inputs)

    def set_pending(self, message):
        self.set_current(False, message)

    def set_current(self, current, message=""):
        self.current = current
        self.message = message

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


@pytest.fixture
def review_widget(qtbot, monkeypatch):
    from napari_vipp.ui import image_review

    monkeypatch.setattr(image_review, "ImageReviewWindow", _ReviewWindow)
    data = np.arange(120, dtype=np.float32).reshape(10, 12)
    widget = _widget(qtbot, data, axes="YX")
    _publish_array_output(widget, "input", data, axes="YX")
    before = scientific_workflow_hash(serialize_workflow(widget.pipeline))
    node = widget.pipeline.add_node("review_images")
    widget.graph_view.add_node(node, widget.graph_view.suggest_append_position("input"))
    widget.pipeline.connect("input", node.id, target_port=0)
    widget._pending_dirty_node_ids.clear()
    widget._debounce_timer.stop()
    return widget, node, data, before


def test_node_card_and_inspector_open_same_outputless_window(review_widget):
    widget, node, data, before = review_widget
    assert not widget.pipeline.output_ports(node.id)
    assert not widget.graph_view._proxies[node.id].output_ports
    card = widget.graph_view._cards[node.id]
    assert card.result_button.text() == "Open review…"
    assert not card.result_button.isHidden()
    widget._select_node(node.id)
    assert widget.save_button.isHidden()
    assert widget.pin_button.isHidden()
    assert widget.execution_group.isHidden()
    assert "Display only" in widget.selected_category_label.text()
    card.result_button.click()
    key = widget._image_reviews._context(node.id)
    window = widget._image_reviews.windows[key]
    assert len(window.inputs) == 1
    assert np.shares_memory(window.inputs[0].data, data)
    assert not window.inputs[0].data.flags.writeable
    buttons = widget.parameter_form_widget.findChildren(QPushButton)
    next(button for button in buttons if button.text() == "Open review…").click()
    assert widget._image_reviews.windows[key] is window
    assert scientific_workflow_hash(serialize_workflow(widget.pipeline)) == before


@pytest.mark.parametrize("mode", ("Off", "Slice", "MIP"))
def test_thumbnail_refresh_keeps_review_card_outputless(
    review_widget,
    monkeypatch,
    mode,
):
    widget, node, data, before = review_widget
    widget.preview_mode_combo.setCurrentText(mode)
    widget._build_graph_from_pipeline()
    widget._select_node(node.id)
    widget._update_thumbnails()
    assert not widget._node_preview_enabled(node.id)
    assert widget.thumbnail_checkbox.isHidden()
    output_ids = {key: id(value) for key, value in widget.pipeline.outputs.items()}
    states = dict(widget.pipeline.node_execution_states)

    def unexpected_preview(*args, **kwargs):
        pytest.fail("Outputless review must not render or scan thumbnail pixels")

    # Limit the trap to this direct refresh: unrelated source statistics may
    # finish later in Qt's event loop and still legitimately render images.
    with monkeypatch.context() as patch:
        patch.setattr("napari_vipp._widget.make_preview", unexpected_preview)
        patch.setattr(
            widget,
            "_thumbnail_contrast_limit_request",
            unexpected_preview,
        )
        widget._update_node_thumbnail(
            node.id,
            None,
            None,
            0,
            queue_stack_contrast=True,
        )
    card = widget.graph_view._cards[node.id]
    assert card.preview.isHidden()
    assert card.preview.text() == ""
    assert not widget.graph_view.node_has_thumbnail(node.id)
    assert card.metadata_label.isHidden()
    assert not card.result_button.isHidden()
    assert widget.pipeline.node_execution_states == states
    assert {
        key: id(value) for key, value in widget.pipeline.outputs.items()
    } == output_ids
    assert widget.pipeline.outputs["input"] is data
    assert scientific_workflow_hash(serialize_workflow(widget.pipeline)) == before


def test_display_recipe_persists_without_dirtying_or_replacing_science(review_widget):
    widget, node, data, before = review_widget
    window = widget._image_reviews.open_review(node.id)
    session = widget._workflow_tabs.current
    session.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    output_ids = {key: id(value) for key, value in widget.pipeline.outputs.items()}
    states = dict(widget.pipeline.node_execution_states)
    values = deepcopy(window.settings)
    values["a"]["contrast_limits"] = [12, 90]
    values["a"]["opacity"] = 0.7
    values["right"] = "a"
    values["orientation"] = "xy"
    values["show_axes"] = False
    window.commit(values)
    assert session.dirty
    assert not widget._pending_dirty_node_ids
    assert not widget._debounce_timer.isActive()
    assert node.params == {}
    assert widget.pipeline.node_execution_states == states
    assert {
        key: id(value) for key, value in widget.pipeline.outputs.items()
    } == output_ids
    assert np.array_equal(data, np.arange(120, dtype=np.float32).reshape(10, 12))
    document = serialize_workflow(widget.pipeline, metadata=widget._workflow_metadata())
    assert document["metadata"]["vipp"]["inspector"]["image_reviews"][node.id] == values
    assert scientific_workflow_hash(document) == before


def test_stale_review_is_disabled_and_closed_window_recreates(review_widget):
    widget, node, _data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    widget.pipeline.node_execution_states["input"] = EXECUTION_STALE
    widget._image_reviews.refresh()
    assert not window.current
    assert "stale" in window.message
    assert widget._image_reviews.open_review(node.id) is None
    widget._image_reviews.close()
    assert not widget._image_reviews.windows


def test_changed_cache_live_updates_same_window(review_widget):
    widget, node, data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    replacement = data.copy() + 1
    _publish_array_output(widget, "input", replacement, axes="YX")
    widget._image_reviews.refresh()
    assert window.current
    assert np.shares_memory(window.inputs[0].data, replacement)
    assert len(window.updates) == 1
    fresh = widget._image_reviews.open_review(node.id)
    assert fresh is window
    widget._image_reviews.refresh()
    assert len(window.updates) == 1
    fresh.close()
    assert not widget._image_reviews.windows
    assert widget._image_reviews.open_review(node.id) is not fresh


@pytest.mark.parametrize(
    "pending",
    ("ancestor", "manual", "inflight", "full", "run", "source", "queued", "load"),
)
def test_live_review_waits_for_current_complete_inputs(review_widget, pending):
    widget, node, data, _before = review_widget
    widget.pipeline.disconnect("input", node.id)
    widget.pipeline.connect("gaussian", node.id)
    _publish_array_output(widget, "gaussian", data, axes="YX")
    window = widget._image_reviews.open_review(node.id)
    original = window.inputs
    _publish_array_output(widget, "gaussian", data.copy() + 1, axes="YX")
    if pending == "ancestor":
        widget._pending_dirty_node_ids.add("input")
    elif pending == "manual":
        widget._pending_manual_node_ids.add("input")
    elif pending == "inflight":
        widget._inflight_dirty_node_ids = {"input"}
    elif pending == "full":
        widget._inflight_full_graph = True
    elif pending == "run":
        widget._active_pipeline_run_id = 123
    elif pending == "source":
        widget._active_source_load_id = 123
    elif pending == "queued":
        widget._pipeline_run_pending = True
    else:
        widget._source_load_pending = True
    widget._image_reviews.refresh()
    assert not window.current
    assert window.inputs is original
    assert not window.updates
    widget._pending_dirty_node_ids.clear()
    widget._pending_manual_node_ids.clear()
    widget._inflight_dirty_node_ids = None
    widget._inflight_full_graph = False
    widget._active_pipeline_run_id = None
    widget._active_source_load_id = None
    widget._pipeline_run_pending = False
    widget._source_load_pending = False
    widget._image_reviews.refresh()
    assert window.current
    assert len(window.updates) == 1


def test_ready_but_invalidated_cache_does_not_refresh(review_widget):
    widget, node, data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    original = window.inputs
    _publish_array_output(widget, "input", data.copy(), axes="YX")
    widget.pipeline.completed_node_ids.clear()
    assert widget.pipeline.node_execution_states["input"] == EXECUTION_READY
    widget._image_reviews.refresh()
    assert not window.current
    assert window.inputs is original
    assert not window.updates


def test_review_reads_accepted_cache_not_background_preview(review_widget, monkeypatch):
    widget, node, data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    transient = data.copy() + 100
    override = SimpleNamespace(
        output=transient,
        output_state=widget.pipeline.output_states["input"],
        node_outputs=[transient],
        node_output_states=widget.pipeline.node_output_states["input"],
    )
    monkeypatch.setattr(widget, "_background_node_result_override", lambda _: override)
    assert widget._node_output_payload_for_port("input", 0)[0] is transient
    widget._image_reviews.refresh()
    assert window.current
    assert not window.updates
    assert np.shares_memory(window.inputs[0].data, data)


@pytest.mark.parametrize("change", ("grid", "presence", "allocation", "render"))
def test_failed_live_replacement_preserves_pair_and_token(
    review_widget, monkeypatch, change
):
    widget, node, data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    original = window.inputs
    context = widget._image_reviews._context(node.id)
    token = widget._image_reviews._tokens[context]
    replacement = np.ones((20, 30)) if change == "grid" else data.copy() + 1
    _publish_array_output(widget, "input", replacement, axes="YX")
    if change == "presence":
        widget.pipeline.connect("gaussian", node.id, target_port=1)
        _publish_array_output(widget, "gaussian", data, axes="YX")
    elif change in {"allocation", "render"}:

        def denied(_inputs):
            if change == "render":
                raise RuntimeError("Native review refresh failed")
            raise MemoryError("Review resource budget exceeded")

        monkeypatch.setattr(window, "update_inputs", denied)
    widget._image_reviews.refresh()
    assert not window.current
    assert window.inputs is original
    assert widget._image_reviews._tokens[context] == token
    assert not window.updates


def test_explicit_open_recreates_unrecoverable_window_even_with_same_token(
    review_widget,
):
    widget, node, _data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    window.update_failed = True
    fresh = widget._image_reviews.open_review(node.id)
    assert fresh is not None and fresh is not window
    assert fresh.current


def test_visible_review_retains_only_direct_inputs_in_low_memory(review_widget):
    widget, node, data, _before = review_widget
    threshold = widget.pipeline.add_node("binary_threshold")
    widget.pipeline.connect("gaussian", threshold.id)
    widget.pipeline.disconnect("input", node.id)
    widget.pipeline.connect(threshold.id, node.id)
    _publish_array_output(widget, threshold.id, data > 50, axes="YX")
    window = widget._image_reviews.open_review(node.id)
    widget._selected_node_id = ""
    widget._active_pinned_node_id = None
    retained = widget._cache_retention_node_ids(CACHE_MODE_LOW_MEMORY)
    assert node.id in retained
    assert threshold.id in retained
    assert "gaussian" not in retained
    assert "input" not in retained
    window.hide()
    assert threshold.id not in widget._cache_retention_node_ids(CACHE_MODE_LOW_MEMORY)
    window.show()
    assert threshold.id in widget._cache_retention_node_ids(CACHE_MODE_LOW_MEMORY)
    window.close()
    assert threshold.id not in widget._cache_retention_node_ids(CACHE_MODE_LOW_MEMORY)


@pytest.mark.parametrize(
    "terminal", ("accepted", "cancelled", "superseded", "source_changed", "failed")
)
def test_smoothing_recalculation_updates_review_only_after_terminal_acceptance(
    qtbot, monkeypatch, terminal
):
    from napari_vipp.ui import image_review

    monkeypatch.setattr(image_review, "ImageReviewWindow", _ReviewWindow)
    data = np.zeros((16, 16), dtype=np.float32)
    data[4:12, 4:12] = 1.0
    widget = VippWidget(_Viewer(data, metadata={"axes": "YX"}))
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    threshold = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("gaussian", threshold.id)
    widget.pipeline.set_param(threshold.id, "threshold", 0.7)
    node = widget.add_node_from_palette("review_images")
    widget._connect_nodes("gaussian", node.id, target_port=0)
    widget._connect_nodes(threshold.id, node.id, target_port=1)
    widget.run_pipeline(force_sync=True)
    window = widget._image_reviews.open_review(node.id)
    assert window is not None
    original = window.inputs
    recipe = deepcopy(window.settings)
    widget.cache_mode_combo.setCurrentText(CACHE_MODE_LOW_MEMORY)
    widget.graph_view.select_node("gaussian")
    pool = _QueuedThreadPool()
    widget._pipeline_thread_pool = pool
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: True
    widget._parameter_widgets["sigma"].value_box.setValue(3.0)
    qtbot.waitUntil(lambda: len(pool.workers) == 1, timeout=3_000)
    assert not window.current
    assert window.inputs is original
    if terminal != "accepted":
        request = pool.workers[0].request
        result = execute_pipeline_request(request)
        assert result.error == ""
        assert not result.cancelled
        # Exercise the real terminal acceptance guard, without starting a
        # follow-up worker in this bounded rejected-result test.
        widget.run_pipeline = lambda *_args, **_kwargs: None
        if terminal == "cancelled":
            result = replace(result, cancelled=True)
        elif terminal == "superseded":
            widget.pipeline.set_param("gaussian", "sigma", 4.0)
            widget._mark_pipeline_dirty("gaussian")
        elif terminal == "source_changed":
            result = replace(result, source_revisions=("obsolete-source-token",))
            monkeypatch.setattr(
                widget._live_source_adapter, "tokens_are_current", lambda _: False
            )
        else:
            result = replace(result, pipeline=None, error="synthetic calculation error")
        widget._on_background_pipeline_finished(result)
        widget._image_reviews.refresh()
        assert window.inputs is original
        assert not window.current
        assert not window.updates
        return
    observed = []

    def check_intermediate(result):
        observed.append(result.node_id)
        assert window.inputs is original
        assert not window.updates

    # Worker signals connect the method at dispatch time, so also observe the
    # existing signal after its queued callback has delivered.
    pool.workers[0].signals.node_finished.connect(check_intermediate)
    pool.workers[0].run()
    qtbot.waitUntil(
        lambda: widget._active_pipeline_run_id is None and window.current,
        timeout=5_000,
    )
    assert "gaussian" in observed
    assert threshold.id in observed
    key = widget._image_reviews._context(node.id)
    assert widget._image_reviews.windows[key] is window
    assert len(window.updates) == 1
    assert window.settings == recipe
    for item, source in zip(window.inputs, ("gaussian", threshold.id), strict=True):
        assert np.shares_memory(item.data, widget.pipeline.outputs[source])
        assert not item.data.flags.writeable
    assert not np.array_equal(original[0].data, window.inputs[0].data)
    assert not np.array_equal(original[1].data, window.inputs[1].data)


def test_session_retirement_hides_then_disposes_only_its_own_window(review_widget):
    widget, node, _data, _before = review_widget
    window = widget._image_reviews.open_review(node.id)
    session = widget._workflow_tabs.current
    widget._image_reviews.hide_session(session.session_id)
    assert window.isHidden()
    widget._image_reviews.close_session("unrelated-session")
    assert window in widget._image_reviews.windows.values()
    widget._image_reviews.close_session(session.session_id)
    assert not widget._image_reviews.windows


def test_same_node_ids_in_different_tabs_keep_their_own_recipes(review_widget):
    widget, node, data, _before = review_widget
    first_session = widget._workflow_tabs.current
    first_window = widget._image_reviews.open_review(node.id)
    first_settings = deepcopy(first_window.settings)
    first_settings["a"]["opacity"] = 0.2
    first_window.commit(first_settings)
    widget._new_workflow()
    _publish_array_output(widget, "input", np.ones((20, 30)), axes="YX")
    second = widget.pipeline.add_node("review_images")
    widget.graph_view.add_node(
        second, widget.graph_view.suggest_append_position("input")
    )
    assert node.id == second.id
    widget.pipeline.connect("input", second.id)
    widget._pending_dirty_node_ids.clear()
    second_window = widget._image_reviews.open_review(second.id)
    assert first_window.isHidden()
    second_settings = deepcopy(second_window.settings)
    second_settings["a"]["opacity"] = 0.8
    second_window.commit(second_settings)
    first_window.commit(first_settings)
    assert widget._image_review_settings[second.id]["a"]["opacity"] == 0.8
    assert widget._activate_workflow_tab(0)
    assert widget._workflow_tabs.current is first_session
    assert widget._image_review_settings[node.id]["a"]["opacity"] == 0.2
    reopened = widget._image_reviews.open_review(node.id)
    assert reopened.inputs[0].data.shape == data.shape
    assert widget.pipeline.outputs["input"] is data
