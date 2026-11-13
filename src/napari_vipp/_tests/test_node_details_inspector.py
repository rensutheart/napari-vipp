"""Open the real inspector action without running or editing a node."""

import json
from copy import deepcopy

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_ui_inspector_widget_integration import _select, _widget
from napari_vipp.ui.node_details import NodeDetailsDialog


@pytest.mark.parametrize(
    ("operation_id", "title", "setting", "key"),
    [
        ("input", "Image Source", "Source", "source_mode"),
        ("measure_objects", "Measure Objects", "Spatial processing", "spatial_mode"),
        ("born_wolf_psf", "Born-Wolf PSF", "Auto from metadata", "auto_parameters"),
    ],
)
def test_inspector_opens_details_without_changing_configuration(
    qtbot, monkeypatch, operation_id, title, setting, key
):
    widget = _widget(qtbot)
    node = (
        widget.pipeline.nodes["input"]
        if operation_id == "input"
        else widget.add_node_from_palette(operation_id)
    )
    _select(widget, node.id)
    widget._debounce_timer.stop()
    editor = widget.node_name_editor
    initial = deepcopy(node.params)
    current_controls = dict(widget._parameter_widgets)
    output_ids = {key: id(value) for key, value in widget.pipeline.outputs.items()}
    execution_states = dict(widget.pipeline.node_execution_states)
    history_depth = len(widget._history.undo_stack)
    monkeypatch.setattr(
        widget, "run_pipeline", lambda *a, **kw: pytest.fail("Unexpected execution")
    )

    assert editor.operation_label.text() == f"Operation: {title}"
    assert not any(
        setting + ":" in label.text() for label in editor.findChildren(QLabel)
    )
    widget._open_node_details("another-node")
    assert widget.findChild(NodeDetailsDialog) is None
    qtbot.mouseClick(editor.details_button, Qt.LeftButton)
    dialog = widget.findChild(NodeDetailsDialog)
    assert dialog is not None and dialog.isVisible()
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Settings"
    assert f"Operation\n{title}" in dialog.settings.toPlainText()
    assert setting in dialog.settings.toPlainText().splitlines()
    config = json.loads(dialog.json_view.toPlainText())
    assert config["operation_id"] == operation_id
    assert config["parameters"][key] == initial[key]
    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: widget.findChild(NodeDetailsDialog) is None)

    assert node.params == initial
    assert widget.pipeline.node_execution_states == execution_states
    assert {
        key: id(value) for key, value in widget.pipeline.outputs.items()
    } == output_ids
    assert widget._parameter_widgets == current_controls
    assert len(widget._history.undo_stack) == history_depth
    assert not widget._debounce_timer.isActive()
