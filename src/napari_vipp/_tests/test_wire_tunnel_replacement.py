from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QPointF

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


def _graph_state(widget):
    return (
        tuple(widget.pipeline.connections),
        widget.pipeline.output_tunnel_list(),
        tuple(widget.graph_view._connections),
        len(widget._undo_stack),
        len(widget._redo_stack),
    )


def test_wire_tunnel_request_replaces_only_clicked_wire_as_one_undo(qtbot, monkeypatch):
    widget = _widget(qtbot, monkeypatch)
    before = tuple(widget.pipeline.connections)
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")

    widget.graph_view.connection_tunnel_requested.emit(("input", "gaussian", 0, 0))

    replacement = GraphConnection("input", "gaussian", 0, 0, "Raw")
    assert replacement in widget.pipeline.connections
    assert widget.pipeline.output_tunnel("Raw") == OutputTunnel("Raw", "input", 0)
    assert GraphConnection("gaussian", "threshold") in widget.pipeline.connections
    assert len(widget._undo_stack) == 1
    assert not any(
        item.source_id == "input" and item.target_id == "gaussian"
        for item in widget.graph_view._connections
    )
    assert widget.graph_view._proxies["input"].output_port_at(0)._tunnel_label == "Raw"
    assert (
        widget.graph_view._proxies["gaussian"].input_port_at(0)._tunnel_label == "Raw"
    )

    widget.undo()

    assert tuple(widget.pipeline.connections) == before
    assert widget.pipeline.output_tunnel_list() == ()
    assert any(
        item.source_id == "input" and item.target_id == "gaussian"
        for item in widget.graph_view._connections
    )

    widget.redo()

    assert replacement in widget.pipeline.connections
    assert widget.pipeline.output_tunnel("Raw") == OutputTunnel("Raw", "input", 0)


def test_wire_tunnel_replacement_preserves_nonzero_ports_and_sibling_wire(
    qtbot, monkeypatch
):
    widget = _widget(qtbot, monkeypatch)
    widget.pipeline.reset_empty_graph()
    source = widget.pipeline.add_node("split_channels")
    target = widget.pipeline.add_node("add_images")
    sibling = widget.pipeline.add_node("median_filter")
    assert widget.pipeline.connect("input", target.id, target_port=0).success
    assert widget.pipeline.connect(source.id, target.id, 1, 2).success
    assert widget.pipeline.connect(source.id, sibling.id, 0, 2).success
    widget.graph_view.build_graph(
        widget.pipeline.nodes.values(),
        widget.pipeline.connections,
        positions={
            "input": QPointF(0, -300),
            source.id: QPointF(0, 0),
            target.id: QPointF(600, 0),
            sibling.id: QPointF(600, 300),
        },
    )
    widget._selected_node_id = "input"
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Channel 3")

    assert widget._replace_connection_with_tunnel((source.id, target.id, 1, 2))

    assert widget.pipeline.output_tunnel("Channel 3") == OutputTunnel(
        "Channel 3", source.id, 2
    )
    assert GraphConnection(source.id, target.id, 1, 2, "Channel 3") in (
        widget.pipeline.connections
    )
    assert GraphConnection(source.id, sibling.id, 0, 2) in widget.pipeline.connections
    assert GraphConnection("input", target.id, 0, 0) in widget.pipeline.connections
    assert widget.graph_view._proxies[source.id].output_port_at(2)._tunnel_label == (
        "Channel 3"
    )
    assert widget.graph_view._proxies[target.id].input_port_at(1)._tunnel_label == (
        "Channel 3"
    )
    assert any(
        item.source_id == source.id
        and item.target_id == sibling.id
        and item.source_port == 2
        for item in widget.graph_view._connections
    )


def test_wire_tunnel_replacement_reuses_existing_source_tunnel_without_prompt(
    qtbot, monkeypatch
):
    widget = _widget(qtbot, monkeypatch)
    widget.pipeline.add_output_tunnel("Raw", "input")
    widget._sync_port_tunnels()

    def no_prompt(*_args):
        pytest.fail("An existing source tunnel should be reused without a dialog.")

    monkeypatch.setattr(widget, "_prompt_tunnel_name", no_prompt)

    assert widget._replace_connection_with_tunnel(("input", "gaussian", 0, 0))

    assert widget.pipeline.output_tunnel_list() == (OutputTunnel("Raw", "input", 0),)
    assert GraphConnection("input", "gaussian", 0, 0, "Raw") in (
        widget.pipeline.connections
    )
    assert len(widget._undo_stack) == 1

    widget.undo()

    assert widget.pipeline.output_tunnel("Raw") == OutputTunnel("Raw", "input", 0)
    assert GraphConnection("input", "gaussian") in widget.pipeline.connections


@pytest.mark.parametrize("name", ("", "Already used"))
def test_canceled_or_invalid_wire_tunnel_name_leaves_graph_and_history_unchanged(
    qtbot, monkeypatch, name
):
    widget = _widget(qtbot, monkeypatch)
    widget.pipeline.add_output_tunnel("Already used", "threshold")
    widget._sync_port_tunnels()
    before = _graph_state(widget)
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: name)

    assert not widget._replace_connection_with_tunnel(("input", "gaussian", 0, 0))

    assert _graph_state(widget) == before


def test_wire_tunnel_validation_failure_removes_temporary_tunnel(qtbot, monkeypatch):
    widget = _widget(qtbot, monkeypatch)
    before = _graph_state(widget)
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")
    monkeypatch.setattr(
        widget.pipeline,
        "connect_to_tunnel",
        lambda *_args: ConnectionResult(False, "Connection validation failed."),
    )

    assert not widget._replace_connection_with_tunnel(("input", "gaussian", 0, 0))

    assert _graph_state(widget) == before
    assert widget.status_label.text() == "Connection validation failed."


def test_wire_tunnel_replacement_preserves_cached_outputs_without_recalculation(
    qtbot, monkeypatch
):
    widget = VippWidget(_Viewer(np.arange(64, dtype=np.float32).reshape(8, 8)))
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    widget.run_pipeline(force_sync=True)
    widget._debounce_timer.stop()
    before_outputs = dict(widget.pipeline.outputs)
    before_states = dict(widget.pipeline.node_execution_states)
    assert before_outputs["gaussian"] is not None
    monkeypatch.setattr(widget, "_prompt_tunnel_name", lambda *_args: "Raw")

    def no_calculation(*_args, **_kwargs):
        pytest.fail("Converting the same endpoints to a tunnel must retain outputs.")

    monkeypatch.setattr(widget, "run_pipeline", no_calculation)

    assert widget._replace_connection_with_tunnel(("input", "gaussian", 0, 0))

    assert widget.pipeline.outputs.keys() == before_outputs.keys()
    assert all(
        widget.pipeline.outputs[node_id] is output
        for node_id, output in before_outputs.items()
    )
    assert widget.pipeline.node_execution_states == before_states


def test_wire_changed_during_naming_dialog_is_not_replaced(qtbot, monkeypatch):
    widget = _widget(qtbot, monkeypatch)

    def change_wire(*_args):
        assert widget.pipeline.disconnect("input", "gaussian", 0)
        assert widget.pipeline.connect("input", "threshold", 0).success
        return "Raw"

    monkeypatch.setattr(widget, "_prompt_tunnel_name", change_wire)

    assert not widget._replace_connection_with_tunnel(("input", "gaussian", 0, 0))

    assert widget.pipeline.output_tunnel_list() == ()
    assert GraphConnection("input", "threshold") in widget.pipeline.connections
    assert len(widget._undo_stack) == 0
    assert widget.status_label.text() == "That wire is no longer available."
