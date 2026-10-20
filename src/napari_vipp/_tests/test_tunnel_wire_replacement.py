from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtWidgets import QMenu

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.pipeline import ConnectionResult, GraphConnection, OutputTunnel


def _widget(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *_args, **_kwargs: None)
    widget._debounce_timer.stop()
    widget._history.clear()
    return widget


def _shared_tunnel_widget(qtbot, monkeypatch):
    widget = _widget(qtbot, monkeypatch)
    pipeline = widget.pipeline
    pipeline.reset_empty_graph()
    source = pipeline.add_node("split_channels")
    target = pipeline.add_node("add_images")
    sibling = pipeline.add_node("median_filter")
    direct = pipeline.add_node("gaussian_blur")
    other = pipeline.add_node("median_filter")
    assert pipeline.connect("input", target.id, 0).success
    pipeline.add_output_tunnel("Channel 3", source.id, 2)
    pipeline.add_output_tunnel("Channel 2", source.id, 1)
    assert pipeline.connect_to_tunnel("Channel 3", target.id, 1).success
    assert pipeline.connect_to_tunnel("Channel 3", sibling.id, 0).success
    assert pipeline.connect(source.id, direct.id, 0, 2).success
    assert pipeline.connect_to_tunnel("Channel 2", other.id, 0).success
    widget.graph_view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        positions={
            node_id: QPointF(index // 3 * 700, index % 3 * 350)
            for index, node_id in enumerate(pipeline.nodes)
        },
        output_tunnels=pipeline.output_tunnel_list(),
    )
    widget._selected_node_id = "input"
    widget._sync_all_output_ports()
    widget._history.clear()
    return widget, source, target, sibling, direct, other


def _graph_state(widget):
    return (
        tuple(widget.pipeline.connections),
        widget.pipeline.output_tunnel_list(),
        tuple(
            (item.source_id, item.target_id, item.target_port, item.source_port)
            for item in widget.graph_view._connections
        ),
        len(widget._undo_stack),
        len(widget._redo_stack),
    )


def _choose_menu_action(monkeypatch, label, before_choice=None):
    class ChoosingMenu(QMenu):
        def exec(self, _point):
            actions = [action for action in self.actions() if action.text() == label]
            assert len(actions) == 1 and actions[0].isEnabled()
            if before_choice is not None:
                before_choice()
            return actions[0]

    monkeypatch.setattr("napari_vipp._widget.QMenu", ChoosingMenu)


def test_input_tunnel_replacement_preserves_siblings_ports_and_source_declaration(
    qtbot, monkeypatch
):
    widget, source, target, sibling, direct, other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    before = _graph_state(widget)

    assert widget._replace_input_tunnel_with_wire(target.id, 1)

    assert GraphConnection(source.id, target.id, 1, 2) in widget.pipeline.connections
    assert GraphConnection(source.id, sibling.id, 0, 2, "Channel 3") in (
        widget.pipeline.connections
    )
    assert GraphConnection(source.id, direct.id, 0, 2) in widget.pipeline.connections
    assert GraphConnection(source.id, other.id, 0, 1, "Channel 2") in (
        widget.pipeline.connections
    )
    assert GraphConnection("input", target.id, 0, 0) in widget.pipeline.connections
    assert widget.pipeline.output_tunnel("Channel 3") == OutputTunnel(
        "Channel 3", source.id, 2
    )
    assert not widget.graph_view._proxies[target.id].input_port_at(1)._tunnel_label
    assert widget.graph_view._proxies[sibling.id].input_port_at(0)._tunnel_label == (
        "Channel 3"
    )
    assert len(widget._undo_stack) == 1
    after = tuple(widget.pipeline.connections)

    widget.undo()
    assert tuple(widget.pipeline.connections) == before[0]
    assert widget.pipeline.output_tunnel_list() == before[1]
    assert widget.graph_view._proxies[target.id].input_port_at(1)._tunnel_label == (
        "Channel 3"
    )
    widget.redo()
    assert tuple(widget.pipeline.connections) == after
    assert not widget.graph_view._proxies[target.id].input_port_at(1)._tunnel_label


def test_output_badge_request_replaces_every_subscription_in_one_undo(
    qtbot, monkeypatch
):
    widget, source, target, sibling, direct, other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    before = _graph_state(widget)

    widget.graph_view.tunnel_wires_requested.emit("Channel 3", source.id, 2)

    assert widget.pipeline.output_tunnel("Channel 3") is None
    assert widget.pipeline.output_tunnel("Channel 2") == OutputTunnel(
        "Channel 2", source.id, 1
    )
    for target_id, target_port in ((target.id, 1), (sibling.id, 0), (direct.id, 0)):
        assert GraphConnection(source.id, target_id, target_port, 2) in (
            widget.pipeline.connections
        )
        assert any(
            item.source_id == source.id
            and item.target_id == target_id
            and item.target_port == target_port
            and item.source_port == 2
            for item in widget.graph_view._connections
        )
    assert GraphConnection(source.id, other.id, 0, 1, "Channel 2") in (
        widget.pipeline.connections
    )
    assert not widget.graph_view._proxies[source.id].output_port_at(2)._tunnel_label
    assert len(widget._undo_stack) == 1
    after = tuple(widget.pipeline.connections)

    widget.undo()
    assert tuple(widget.pipeline.connections) == before[0]
    assert widget.pipeline.output_tunnel_list() == before[1]
    widget.redo()
    assert tuple(widget.pipeline.connections) == after
    assert widget.pipeline.output_tunnel("Channel 3") is None


@pytest.mark.parametrize("kind", ("input", "output"))
def test_port_context_menus_expose_direct_wire_replacement(qtbot, monkeypatch, kind):
    widget, source, target, sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    label = "Replace with wire" if kind == "input" else "Replace with wires"
    _choose_menu_action(monkeypatch, label)
    node_id, port = (target.id, 1) if kind == "input" else (source.id, 2)

    widget._show_port_context_menu(kind, node_id, port, QPoint())

    assert GraphConnection(source.id, target.id, 1, 2) in widget.pipeline.connections
    sibling_connection = widget.pipeline.tunnel_connection_for_input(sibling.id, 0)
    assert (sibling_connection is not None) == (kind == "input")
    assert len(widget._undo_stack) == 1


def test_right_click_output_badge_directly_restores_all_wires(qtbot, monkeypatch):
    widget, source, target, sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    view = widget.graph_view
    badge = view._proxies[source.id].output_port_at(2)._tunnel_badge
    selected = []

    def choose(menu, _point):
        actions = [
            action for action in menu.actions() if action.text() == "Replace with wires"
        ]
        assert len(actions) == 1 and actions[0].isEnabled()
        selected.append(actions[0].text())
        return actions[0]

    monkeypatch.setattr("napari_vipp._graph._exec_menu", choose)
    point = view.mapFromScene(badge.sceneBoundingRect().center())

    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=point)

    assert selected == ["Replace with wires"]
    assert widget.pipeline.output_tunnel("Channel 3") is None
    assert GraphConnection(source.id, target.id, 1, 2) in widget.pipeline.connections
    assert GraphConnection(source.id, sibling.id, 0, 2) in widget.pipeline.connections
    assert len(widget._undo_stack) == 1


def test_right_click_input_badge_restores_only_its_wire(qtbot, monkeypatch):
    widget, source, target, sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    _choose_menu_action(monkeypatch, "Replace with wire")
    view = widget.graph_view
    badge = view._proxies[target.id].input_port_at(1)._tunnel_badge
    point = view.mapFromScene(badge.sceneBoundingRect().center())

    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=point)

    assert GraphConnection(source.id, target.id, 1, 2) in widget.pipeline.connections
    assert GraphConnection(source.id, sibling.id, 0, 2, "Channel 3") in (
        widget.pipeline.connections
    )
    assert widget.pipeline.output_tunnel("Channel 3") is not None
    assert len(widget._undo_stack) == 1


def test_replacing_last_input_subscription_retains_unused_output_tunnel(
    qtbot, monkeypatch
):
    widget = _widget(qtbot, monkeypatch)
    widget.pipeline.add_output_tunnel("Raw", "input")
    result = widget.pipeline.connect_to_tunnel("Raw", "gaussian", 0)
    widget._apply_connection_result_to_graph(result)

    assert widget._replace_input_tunnel_with_wire("gaussian", 0)

    assert widget.pipeline.output_tunnel("Raw") == OutputTunnel("Raw", "input", 0)
    assert not any(connection.tunnel_name for connection in widget.pipeline.connections)
    assert widget.graph_view._proxies["input"].output_port_at(0)._tunnel_label == "Raw"


def test_replacing_unused_output_tunnel_removes_only_declaration(qtbot, monkeypatch):
    widget = _widget(qtbot, monkeypatch)
    widget.pipeline.add_output_tunnel("Unused", "threshold")
    widget._sync_port_tunnels()
    before = tuple(widget.pipeline.connections)

    assert widget._replace_output_tunnel_with_wires("Unused", "threshold", 0)

    assert tuple(widget.pipeline.connections) == before
    assert widget.pipeline.output_tunnel("Unused") is None
    assert len(widget._undo_stack) == 1
    widget.undo()
    assert widget.pipeline.output_tunnel("Unused") == OutputTunnel(
        "Unused", "threshold", 0
    )


def test_input_menu_rejects_subscription_changed_while_open(qtbot, monkeypatch):
    widget, source, target, _sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )

    def replace_subscription():
        assert widget.pipeline.connect_to_tunnel("Channel 2", target.id, 1).success

    _choose_menu_action(monkeypatch, "Replace with wire", replace_subscription)
    widget._show_input_tunnel_menu(target.id, 1, QPoint())

    assert GraphConnection(source.id, target.id, 1, 1, "Channel 2") in (
        widget.pipeline.connections
    )
    assert not widget._undo_stack
    assert widget.status_label.text() == "That input tunnel is no longer available."


def test_output_menu_rejects_subscriptions_changed_while_open(qtbot, monkeypatch):
    widget, source, _target, sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )

    def replace_subscription():
        assert widget.pipeline.connect_to_tunnel("Channel 2", sibling.id, 0).success

    _choose_menu_action(monkeypatch, "Replace with wires", replace_subscription)
    widget._show_output_tunnel_menu(source.id, 2, QPoint())

    assert widget.pipeline.output_tunnel("Channel 3") is not None
    assert GraphConnection(source.id, sibling.id, 0, 1, "Channel 2") in (
        widget.pipeline.connections
    )
    assert not widget._undo_stack
    assert "changed while its menu was open" in widget.status_label.text()


def test_output_badge_request_rejects_stale_source_port(qtbot, monkeypatch):
    widget, source, _target, _sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    before = _graph_state(widget)

    widget.graph_view.tunnel_wires_requested.emit("Channel 3", source.id, 1)

    assert _graph_state(widget) == before
    assert widget.status_label.text() == "That output tunnel is no longer available."


def test_output_conversion_validation_failure_restores_all_subscriptions(
    qtbot, monkeypatch
):
    widget, source, _target, _sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    before = _graph_state(widget)
    original_connect = widget.pipeline.connect
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return ConnectionResult(False, "The second subscription is unavailable.")
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "connect", fail_second)

    assert not widget._replace_output_tunnel_with_wires("Channel 3", source.id, 2)

    assert calls == 2
    assert _graph_state(widget) == before
    assert widget.status_label.text() == "The second subscription is unavailable."


def test_failed_visible_wire_update_rolls_back_tunnels_without_history(
    qtbot, monkeypatch
):
    widget, source, _target, _sibling, _direct, _other = _shared_tunnel_widget(
        qtbot, monkeypatch
    )
    before = _graph_state(widget)
    original_add = widget.graph_view.add_connection
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Could not display the wire.")
        return original_add(*args, **kwargs)

    monkeypatch.setattr(widget.graph_view, "add_connection", fail_once)

    assert not widget._replace_output_tunnel_with_wires("Channel 3", source.id, 2)

    assert _graph_state(widget) == before
    assert widget.graph_view._proxies[source.id].output_port_at(2)._tunnel_label == (
        "Channel 3"
    )
    assert "rolled back" in widget.status_label.text()


@pytest.mark.parametrize("kind", ("input", "output"))
def test_tunnel_wire_action_undo_and_redo_preserve_cached_results(
    qtbot, monkeypatch, kind
):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.float32).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    widget.run_pipeline(force_sync=True)
    widget._debounce_timer.stop()
    widget.pipeline.add_output_tunnel("Raw", "input")
    result = widget.pipeline.connect_to_tunnel("Raw", "gaussian", 0)
    widget._apply_connection_result_to_graph(result)
    widget._history.clear()
    before_outputs = dict(widget.pipeline.outputs)
    before_states = dict(widget.pipeline.node_execution_states)
    before_nodes = dict(widget.pipeline.nodes)
    before_pending = set(widget._pending_dirty_node_ids)
    assert before_outputs["gaussian"] is not None

    def no_calculation(*_args, **_kwargs):
        pytest.fail("Changing tunnel presentation must retain cached results.")

    monkeypatch.setattr(widget, "run_pipeline", no_calculation)
    monkeypatch.setattr(widget, "_invalidate_pipeline_cache", no_calculation)
    if kind == "input":
        assert widget._replace_input_tunnel_with_wire("gaussian", 0)
    else:
        assert widget._replace_output_tunnel_with_wires("Raw", "input", 0)

    def assert_runtime_retained():
        assert widget.pipeline.outputs.keys() == before_outputs.keys()
        assert all(
            widget.pipeline.outputs[node_id] is output
            for node_id, output in before_outputs.items()
        )
        assert all(
            widget.pipeline.nodes[node_id] is node
            for node_id, node in before_nodes.items()
        )
        assert widget.pipeline.node_execution_states == before_states
        assert widget._pending_dirty_node_ids == before_pending

    assert_runtime_retained()
    widget.undo()
    assert_runtime_retained()
    assert widget.pipeline.tunnel_connection_for_input("gaussian", 0) is not None
    widget.redo()
    assert_runtime_retained()
    assert GraphConnection("input", "gaussian", 0, 0) in widget.pipeline.connections
