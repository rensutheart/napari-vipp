"""Registration next steps preserve graph intent and remain a manual boundary."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QPointF

from napari_vipp.core.pipeline import EXECUTION_READY, ConnectionResult


@pytest.fixture
def next_step_widget(qtbot, monkeypatch):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((16, 18), np.float32)), defer_initial_run=True)
    qtbot.addWidget(widget)
    estimator = widget.add_node_from_palette("estimate_registration")
    widget.pipeline.set_param(estimator.id, "mode", "Time series")
    widget._sync_node_input_ports(estimator.id)
    widget._debounce_timer.stop()
    # Graph-authoring tests must not start scientific workers or depend on an
    # unrelated source calculation. Any attempt to start one is observable.
    calls = []
    monkeypatch.setattr(
        widget, "run_pipeline", lambda *args, **kwargs: calls.append(kwargs)
    )
    widget._history.clear()
    return widget, estimator, calls


def _connect(widget, source, target, source_port=0, target_port=0, tunnel_name=""):
    result = widget.pipeline.connect(
        source,
        target,
        source_port=source_port,
        target_port=target_port,
        tunnel_name=tunnel_name,
    )
    assert result.success, result.message
    widget._apply_connection_result_to_graph(result)
    widget._sync_node_output_ports(target)


def _prepare(widget, estimator, *, source_port=0, tunnel=False):
    # Apply Transform has two fixed array outputs, allowing an uncalculated
    # nonzero port to exercise exact wiring independently of image metadata.
    source = (
        widget.add_node_from_palette("apply_transform")
        if source_port
        else widget.pipeline.nodes["input"]
    )
    tunnel_name = "Moving image" if tunnel else ""
    if tunnel:
        widget.pipeline.add_output_tunnel(tunnel_name, source.id, source_port)
    _connect(
        widget,
        source.id,
        estimator.id,
        source_port=source_port,
        tunnel_name=tunnel_name,
    )
    widget.graph_view.select_node(estimator.id)
    widget._debounce_timer.stop()
    widget._history.clear()
    widget._registration_next_step.refresh()
    return source


def test_card_explains_missing_input_and_is_neutral(next_step_widget):
    widget, estimator, _ = next_step_widget
    controller = widget._registration_next_step
    controller.refresh()
    assert controller.section.title() == "Next step"
    assert controller.section.isExpanded()
    assert controller.main_button.text() == "Add Apply Transform"
    assert not controller.main_button.isEnabled()
    assert "Connect the moving image" in controller.explanation.text()
    assert "calculates the movement" in controller.description.text()
    assert "warning" not in controller.section.styleSheet().casefold()
    before = widget._current_history_snapshot()
    assert controller.add_apply_transform() is None
    assert widget._current_history_snapshot() == before
    assert estimator.id in widget.pipeline.nodes


@pytest.mark.parametrize("source_port,tunnel", [(0, False), (1, False), (1, True)])
def test_add_connects_exact_ports_preserves_branches_and_is_one_undo(
    next_step_widget,
    source_port,
    tunnel,
):
    widget, estimator, calls = next_step_widget
    source = _prepare(widget, estimator, source_port=source_port, tunnel=tunnel)
    sibling = widget.add_node_from_palette("gaussian_blur")
    _connect(widget, source.id, sibling.id, source_port=source_port)
    widget.graph_view.select_node(estimator.id)
    widget._history.clear()
    before = widget._current_history_snapshot()
    old_connections = tuple(widget.pipeline.connections)
    old_positions = widget.graph_view.node_positions()

    added = widget._registration_next_step.add_apply_transform()

    assert added is not None
    assert added.operation_id == "apply_transform"
    assert added.params["interpolation"] == "Automatic"
    assert widget._selected_node_id == added.id
    assert all(
        connection in widget.pipeline.connections for connection in old_connections
    )
    inputs = {
        connection.target_port: connection
        for connection in widget.pipeline.connections
        if connection.target_id == added.id
    }
    assert (inputs[0].source_id, inputs[0].source_port) == (source.id, source_port)
    assert inputs[0].tunnel_name == ("Moving image" if tunnel else "")
    assert (inputs[1].source_id, inputs[1].source_port) == (estimator.id, 0)
    assert all(
        widget.graph_view.node_positions()[key] == value
        for key, value in old_positions.items()
    )
    assert len(widget._history.undo_stack) == 1
    assert widget.pipeline.node_execution_states.get(added.id) != EXECUTION_READY
    assert widget.pipeline.outputs.get(added.id) is None
    assert not calls

    after = widget._current_history_snapshot()
    widget.undo()
    assert widget._current_history_snapshot().workflow == before.workflow
    assert widget._selected_node_id == estimator.id
    assert added.id not in widget.pipeline.nodes
    widget.redo()
    assert widget._current_history_snapshot().workflow == after.workflow
    assert added.id in widget.pipeline.nodes
    assert widget.pipeline.outputs.get(added.id) is None


def test_add_avoids_occupied_space_without_moving_existing_nodes(next_step_widget):
    widget, estimator, _ = next_step_widget
    _prepare(widget, estimator)
    position = widget.graph_view.suggest_append_position(estimator.id)
    obstacle = widget._add_node_at("gaussian_blur", position)
    second = widget._add_node_at("gaussian_blur", position + QPointF(0, 160))
    widget.graph_view.select_node(estimator.id)
    positions = widget.graph_view.node_positions()
    added = widget._registration_next_step.add_apply_transform()
    assert added is not None
    new_rect = widget.graph_view.node_scene_rect(added.id)
    assert new_rect.left() > widget.graph_view.node_scene_rect(estimator.id).right()
    for node_id in positions:
        assert widget.graph_view.node_positions()[node_id] == positions[node_id]
        assert not new_rect.intersects(widget.graph_view.node_scene_rect(node_id))
    assert new_rect.top() > widget.graph_view.node_scene_rect(second.id).bottom()
    assert obstacle.id in widget.pipeline.nodes


def test_existing_apply_is_shown_and_another_branch_can_be_added(next_step_widget):
    widget, estimator, calls = next_step_widget
    _prepare(widget, estimator)
    controller = widget._registration_next_step
    first = controller.add_apply_transform()
    widget.graph_view.select_node(estimator.id)
    controller.refresh()
    assert controller.matching_apply_nodes() == (first.id,)
    assert controller.main_button.text() == "Show Apply Transform"
    assert not controller.add_another_button.isHidden()
    before_nodes = set(widget.pipeline.nodes)
    before_history = len(widget._history.undo_stack)
    controller.main_button.click()
    assert widget._selected_node_id == first.id
    assert set(widget.pipeline.nodes) == before_nodes
    assert len(widget._history.undo_stack) == before_history

    widget.graph_view.select_node(estimator.id)
    second = controller.add_apply_transform()
    assert second.id != first.id
    widget.graph_view.select_node(estimator.id)
    controller.refresh()
    assert controller.matching_apply_nodes() == (first.id, second.id)
    assert len(controller.main_button.menu().actions()) == 2
    controller.main_button.menu().actions()[1].trigger()
    assert widget._selected_node_id == second.id
    assert not calls


@pytest.mark.parametrize(
    "mismatch", ["source", "source_port", "transform", "disconnected"]
)
def test_show_does_not_match_unrelated_apply_branches(next_step_widget, mismatch):
    widget, estimator, _ = next_step_widget
    source = _prepare(widget, estimator, source_port=1)
    other_estimator = widget.add_node_from_palette("estimate_registration")
    other = widget.add_node_from_palette("apply_transform")
    _connect(
        widget,
        "input" if mismatch == "source" else source.id,
        other.id,
        source_port=0 if mismatch in {"source", "source_port"} else 1,
    )
    if mismatch != "disconnected":
        _connect(
            widget,
            other_estimator.id if mismatch == "transform" else estimator.id,
            other.id,
            target_port=1,
        )
    widget.graph_view.select_node(estimator.id)
    controller = widget._registration_next_step
    controller.refresh()
    assert controller.matching_apply_nodes() == ()
    assert controller.main_button.text() == "Add Apply Transform"
    controller.show_apply_transform(other.id)
    assert widget._selected_node_id == estimator.id


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_active_pipeline_run_id", 777),
        ("_active_source_load_id", 777),
        ("_pipeline_run_pending", True),
        ("_source_load_pending", True),
        ("_isolated_tuning_node_id", "input"),
        (
            "_compute_runtime_quarantined_reason",
            "Restart VIPP before editing this workflow.",
        ),
    ],
)
def test_busy_authoring_is_disabled_and_rechecked_on_click(
    next_step_widget,
    monkeypatch,
    attribute,
    value,
):
    widget, estimator, calls = next_step_widget
    _prepare(widget, estimator)
    controller = widget._registration_next_step
    before = widget._current_history_snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(widget, attribute, value)
        controller.refresh()
        assert not controller.main_button.isEnabled()
        assert controller.explanation.text()
        assert controller.add_apply_transform() is None
    controller.refresh()
    assert controller.main_button.isEnabled()
    assert widget._current_history_snapshot() == before
    assert not calls


def test_pending_parameter_edit_blocks_add(next_step_widget):
    widget, estimator, calls = next_step_widget
    _prepare(widget, estimator)
    controller = widget._registration_next_step
    widget._debounce_timer.start(60_000)
    try:
        controller.refresh()
        assert not controller.main_button.isEnabled()
        assert "pending parameter edit" in controller.explanation.text()
        assert controller.add_apply_transform() is None
    finally:
        widget._debounce_timer.stop()
    assert not calls


def test_second_connection_failure_rolls_back_whole_addition(
    next_step_widget, monkeypatch
):
    widget, estimator, _ = next_step_widget
    _prepare(widget, estimator)
    before = widget._current_history_snapshot()
    original_connect = widget.pipeline.connect

    def fail_transform(source_id, target_id, **kwargs):
        if source_id == estimator.id:
            return ConnectionResult(False, "Test connection failure")
        return original_connect(source_id, target_id, **kwargs)

    monkeypatch.setattr(widget.pipeline, "connect", fail_transform)
    assert widget._registration_next_step.add_apply_transform() is None
    assert widget._current_history_snapshot().workflow == before.workflow
    assert widget._selected_node_id == estimator.id
    assert not widget._history.undo_stack
    assert "workflow was restored" in widget.status_label.text()


def test_stale_action_after_selection_change_does_not_edit(next_step_widget):
    widget, estimator, _ = next_step_widget
    _prepare(widget, estimator)
    controller = widget._registration_next_step
    widget.graph_view.select_node("input")
    before = widget._current_history_snapshot()
    assert controller.add_apply_transform() is None
    assert widget._current_history_snapshot() == before
