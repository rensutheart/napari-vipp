from __future__ import annotations

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.snapshots import WorkflowSnapshot


def test_widget_history_uses_isolated_typed_workflow_snapshots(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    nested = {"channels": [{"scale": [0.4, 0.2, 0.2]}]}
    widget.pipeline.set_param("gaussian", "_vipp_nested_state", nested)

    snapshot = widget._current_history_snapshot()

    assert isinstance(snapshot.workflow, WorkflowSnapshot)
    nested["channels"][0]["scale"][0] = 99.0
    widget.pipeline.nodes["gaussian"].params["_vipp_nested_state"]["channels"].clear()

    widget._restore_history_snapshot(snapshot)

    assert widget.pipeline.nodes["gaussian"].params["_vipp_nested_state"] == {
        "channels": [{"scale": [0.4, 0.2, 0.2]}]
    }


def test_widget_history_snapshot_can_restore_an_empty_transient_graph(qtbot):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.pipeline.restore_graph((), ())
    empty = widget._current_history_snapshot(positions={})
    widget.pipeline.reset_starter_graph()

    widget._restore_history_snapshot(empty)

    assert widget.pipeline.nodes == {}
    assert widget.graph_view._cards == {}
