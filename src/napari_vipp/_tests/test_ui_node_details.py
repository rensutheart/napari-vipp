"""Node details is a bounded, read-only view of authored configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QTextCursor, QTextTable
from qtpy.QtWidgets import QWidget

from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, OperationSpec, ParameterSpec
from napari_vipp.ui import node_details
from napari_vipp.ui.node_details import NodeDetailsDialog


def _settings_rows(dialog):
    (table,) = [
        frame
        for frame in dialog.settings.document().rootFrame().childFrames()
        if isinstance(frame, QTextTable)
    ]
    assert table.columns() == 2

    def text_at(row, column):
        cell = table.cellAt(row, column)
        cursor = cell.firstCursorPosition()
        cursor.setPosition(cell.lastCursorPosition().position(), QTextCursor.KeepAnchor)
        return cursor.selectedText()

    return {text_at(row, 0): text_at(row, 1) for row in range(table.rows())}


@pytest.fixture
def operation():
    def never_execute(*_args, **_kwargs):
        pytest.fail("Opening node details must not execute the operation")

    return OperationSpec(
        "inspect_only",
        "Inspect only",
        "Tests",
        None,
        "array",
        (
            ParameterSpec(
                "method",
                "Processing method",
                "choice",
                "auto",
                0,
                0,
                1,
                choices=("auto", "manual"),
                choice_labels=("Automatic", "Manual"),
            ),
            ParameterSpec("size", "Kernel size", "int", 3, 0, 99, 1),
            ParameterSpec("keep", "Keep result", "bool", False, 0, 1, 1),
            ParameterSpec("_internal_default", "Internal", "text", "hidden", 0, 0, 1),
        ),
        never_execute,
    )


def test_details_uses_labels_and_exact_json_without_mutating_parameters(
    qtbot, operation
):
    # Qt's plain text may normalize some Unicode spaces/separators. The
    # displayed/copyable JSON escapes them so JSON roundtripping is exact.
    literal = "<img src='file:///do-not-read.png'> µ\xa0\u2028\u2029"
    extra = {"axis": ["z", "y", "x"], "comment": literal, "limit": None}
    parameters = {
        "method": "manual",
        "size": 0,
        "extra_values": extra,
        "_internal_value": object(),
    }
    dialog = NodeDetailsDialog(
        node_name="My node <b>literal</b>", operation=operation, parameters=parameters
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog.tabs.currentIndex() == 0
    assert dialog.settings.isReadOnly() and dialog.json_view.isReadOnly()
    assert dialog.copy_button.isHidden()
    assert not dialog.close_button.autoDefault()
    settings = _settings_rows(dialog)
    assert settings["Node name"] == "My node <b>literal</b>"
    assert settings["Processing method"] == "Manual"
    assert settings["Kernel size"] == "0"
    assert settings["Keep result"] == "Off"
    assert settings["Extra values"] == "3 items (see JSON)"
    assert "Internal" not in settings
    preview = dialog.json_view.toPlainText()
    assert json.loads(preview) == {
        "operation_id": "inspect_only",
        "parameters": {
            "method": "manual",
            "size": 0,
            "keep": False,
            "extra_values": extra,
        },
    }
    assert '\n  "parameters": {' in preview
    assert parameters["extra_values"] is extra
    assert "keep" not in parameters
    # A later change does not rewrite an already-open snapshot.
    parameters["size"] = 4
    extra["axis"].append("c")
    assert _settings_rows(dialog) == settings
    assert dialog.json_view.toPlainText() == preview


def test_source_settings_use_the_registry_without_opening_the_path(qtbot):
    operation = NODE_LIBRARY_BY_ID["input"]
    path = "Z:/not-a-real-source/<sample>.tif"
    dialog = NodeDetailsDialog(
        node_name=operation.title,
        operation=operation,
        parameters={"source_mode": "file path", "file_path": path},
    )
    qtbot.addWidget(dialog)
    settings = _settings_rows(dialog)
    assert settings["Operation"] == "Image Source"
    assert settings["Source"] == "file path"
    assert settings["File path"] == path
    assert settings["Channel colours"] == "Empty"
    assert json.loads(dialog.json_view.toPlainText())["parameters"]["file_path"] == path


@pytest.mark.parametrize(
    ("auto", "wavelength", "display"),
    [(True, 0.0, "Not set manually"), (True, 520.0, "520.0"), (False, 0.0, "0.0")],
)
def test_psf_zero_hint_is_specific_to_auto_metadata_fields(
    qtbot, auto, wavelength, display
):
    parameters = {
        "auto_parameters": auto,
        "wavelength_nm": wavelength,
        "numerical_aperture": 1.4,
        "ordinary_zero": 0,
    }
    dialog = NodeDetailsDialog(
        node_name="",
        operation=NODE_LIBRARY_BY_ID["born_wolf_psf"],
        parameters=parameters,
    )
    qtbot.addWidget(dialog)
    settings = _settings_rows(dialog)
    assert settings["Emission wavelength nm"] == display
    assert settings["Numerical aperture"] == "1.4"
    assert settings["XY pixel size um"] == ("Not set manually" if auto else "0.0")
    assert settings["Ordinary zero"] == "0"
    configured = json.loads(dialog.json_view.toPlainText())["parameters"]
    assert {key: configured[key] for key in parameters} == parameters
    assert configured["pixel_size_xy_um"] == 0.0


def test_copy_requires_explicit_action_and_close_is_a_dismissal(
    qtbot, operation, monkeypatch
):
    copied = []
    clipboard = SimpleNamespace(setText=copied.append)
    monkeypatch.setattr(
        node_details, "QApplication", SimpleNamespace(clipboard=lambda: clipboard)
    )
    dialog = NodeDetailsDialog(node_name="", operation=operation, parameters={})
    qtbot.addWidget(dialog)
    dialog.show()
    assert copied == []
    dialog.tabs.setCurrentWidget(dialog.json_view)
    assert dialog.copy_button.isVisible() and dialog.copy_button.isEnabled()
    assert not dialog.copy_button.autoDefault()
    assert copied == []
    qtbot.mouseClick(dialog.copy_button, Qt.LeftButton)
    assert copied == [dialog.json_view.toPlainText()]
    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)
    assert not dialog.isVisible()


@pytest.mark.parametrize(
    "unsupported",
    [object(), float("nan"), {1: "numeric object key"}, ("tuple",), {"cycle": []}],
)
def test_unsupported_json_is_not_fabricated(qtbot, operation, unsupported):
    if isinstance(unsupported, dict) and "cycle" in unsupported:
        unsupported["cycle"].append(unsupported)
    dialog = NodeDetailsDialog(
        node_name="", operation=operation, parameters={"extra": unsupported}
    )
    qtbot.addWidget(dialog)
    assert _settings_rows(dialog)["Processing method"] == "Automatic"
    assert "cannot be represented faithfully" in dialog.json_view.toPlainText()
    assert not dialog.copy_button.isEnabled()


def test_large_configuration_is_bounded_and_explicit(qtbot, operation, monkeypatch):
    monkeypatch.setattr(node_details, "_MAX_JSON_LENGTH", 1500)
    parent = QWidget()
    qtbot.addWidget(parent)
    font = QFont(parent.font())
    font.setPointSizeF(16)
    parent.setFont(font)
    dialog = NodeDetailsDialog(
        node_name="Large input",
        operation=operation,
        parameters={"long": "a" * 5000, "nested": ["b" * 5000]},
        parent=parent,
    )
    qtbot.addWidget(dialog)
    dialog.resize(340, 380)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.width() == 340)
    assert dialog.rect().contains(dialog.tabs.geometry())
    assert _settings_rows(dialog)["Long"].startswith("a" * 1000 + "…")
    qtbot.waitUntil(lambda: dialog.settings.horizontalScrollBar().maximum() == 0)
    assert "too large to display as JSON" in dialog.json_view.toPlainText()
    assert not dialog.copy_button.isEnabled()
