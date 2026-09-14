"""Display failures from delayed node selection stay in VIPP, not Qt."""

import pytest
from psygnal import Signal
from qtpy.QtCore import Qt, QTimer

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.ui.status import MessageSeverity


class _Transform:
    changed = Signal()


@pytest.mark.parametrize("refresh", ["inspector", "viewer"])
def test_delayed_selection_display_error_is_dismissible_and_preserves_result(
    qtbot, monkeypatch, refresh
):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget._selected_node_id = "gaussian"
    cached = widget.pipeline.outputs["gaussian"]
    execution = dict(widget.pipeline.node_execution_states)
    inspected = []
    original_inspect = widget._inspect_selected_node

    def fail_display():
        inspected.append(True)
        widget._selected_viewer_dims_refresh_pending = True
        raise ReferenceError("weakly-referenced object no longer exists")

    transform = _Transform()
    transform.changed.connect(fail_display)
    # Match the reported psygnal EmitLoopError wrapping ReferenceError, not
    # just a bare exception: the entire cause chain belongs in Details.
    monkeypatch.setattr(widget, "_inspect_selected_node", transform.changed.emit)
    finish = getattr(widget, f"_finish_selected_{refresh}_refresh")
    generation = getattr(widget, f"_selected_{refresh}_refresh_generation")
    QTimer.singleShot(
        0, lambda: finish(generation, "gaussian", select_layer=True)
    )
    qtbot.waitUntil(lambda: bool(inspected))

    assert widget.status_label.severity is MessageSeverity.WARNING
    assert "viewer could not fully update" in widget.status_label.text()
    assert "ReferenceError" not in widget.status_label.text()
    assert "weakly-referenced" in widget.status_label.toolTip()
    assert "EmitLoopError" in widget.status_label.toolTip()
    assert not widget._selected_viewer_refresh_in_progress
    assert not widget._selected_viewer_dims_refresh_pending
    assert not widget._selection_diagnostics_initializing
    assert widget.pipeline.outputs["gaussian"] is cached
    assert widget.pipeline.node_execution_states == execution
    assert not widget.status_actions.isHidden()

    qtbot.mouseClick(widget.status_actions.dismiss_button, Qt.LeftButton)
    assert not widget.status_label.text()
    assert widget.pipeline.outputs["gaussian"] is cached
    assert widget.pipeline.node_execution_states == execution

    # A later selection can retry normally: no stuck refresh ownership flag.
    monkeypatch.setattr(widget, "_inspect_selected_node", original_inspect)
    finish(generation, "gaussian", select_layer=True)
    assert not widget._selected_viewer_refresh_in_progress
    assert widget.pipeline.outputs["gaussian"] is cached


def test_stale_selection_does_not_publish_or_report_a_display_error(qtbot, monkeypatch):
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(widget)
    widget._selected_node_id = "gaussian"

    def must_not_inspect():
        pytest.fail("A stale selection attempted viewer publication")

    monkeypatch.setattr(widget, "_inspect_selected_node", must_not_inspect)
    before = widget.status_label.text()
    for refresh in ("inspector", "viewer"):
        finish = getattr(widget, f"_finish_selected_{refresh}_refresh")
        generation = getattr(widget, f"_selected_{refresh}_refresh_generation")
        finish(generation - 1, "gaussian", select_layer=True)
    assert widget.status_label.text() == before
