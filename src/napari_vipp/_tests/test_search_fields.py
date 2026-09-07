from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QAction, QColor, QKeySequence, QPalette
from qtpy.QtWidgets import QShortcut, QToolButton, QWidget

from napari_vipp.ui.search_fields import (
    SearchLineEdit,
    workflow_find_shortcut_available,
)


def test_search_magnifier_uses_full_text_colour_and_tracks_theme(qtbot):
    field = SearchLineEdit()
    qtbot.addWidget(field)
    field.setPlaceholderText("Find in workflow")
    field.resize(320, 36)
    field.show()
    for text, base in (("#ffffff", "#414953"), ("#172434", "#ffffff")):
        palette = QPalette(field.palette())
        palette.setColor(QPalette.Text, QColor(text))
        palette.setColor(QPalette.Base, QColor(base))
        palette.setColor(QPalette.ButtonText, QColor("#777777"))
        field.setPalette(palette)
        assert field.search_action.isEnabled()
        icon = field.search_action.icon().pixmap(20, 20).toImage()
        opaque = [
            icon.pixelColor(x, y)
            for x in range(icon.width())
            for y in range(icon.height())
            if icon.pixelColor(x, y).alpha() > 240
        ]
        assert opaque
        assert all(colour.name() == text for colour in opaque)

    field.setText("otsu")
    qtbot.wait(20)  # Qt animates the trailing clear-button reveal.
    button = next(
        button
        for button in field.findChildren(QToolButton)
        if button.defaultAction() is field.search_action
    )
    assert button.isEnabled() and button.isVisible()
    qtbot.mouseClick(button, Qt.LeftButton)
    assert field.hasFocus()
    assert field.text() == "otsu"
    parent = QWidget()
    qtbot.addWidget(parent)
    field.setParent(parent)
    field.show()
    assert field.search_action.isEnabled()
    assert not field.search_action.icon().isNull()


@pytest.mark.parametrize(
    "owner", ("settings", "viewer", "layer", "class", "action", "shortcut", "chord")
)
def test_find_reserves_existing_host_shortcuts(qtbot, monkeypatch, owner):
    host = QWidget()
    qtbot.addWidget(host)
    settings = SimpleNamespace(shortcuts=SimpleNamespace(shortcuts={}))
    monkeypatch.setattr("napari.settings.get_settings", lambda: settings)
    viewer = SimpleNamespace(keymap={}, layers=[])
    assert workflow_find_shortcut_available(viewer, host)

    if owner == "settings":
        settings.shortcuts.shortcuts["napari:test"] = ["Control-F"]
    elif owner == "viewer":
        viewer.keymap["Control-F"] = lambda: None
    elif owner == "layer":
        viewer.layers.append(SimpleNamespace(keymap={"Control-F": lambda: None}))
    elif owner == "class":

        class Provider:
            class_keymap = {"Control-F": lambda: None}

        viewer.layers.append(Provider())
    elif owner in {"action", "chord"}:
        # An action can be owned elsewhere but attached to this host window.
        action = QAction("Existing find")
        action.setShortcut(
            QKeySequence("Ctrl+F, Ctrl+G" if owner == "chord" else "Ctrl+F")
        )
        host.addAction(action)
    else:
        shortcut = QShortcut(QKeySequence.Find, host)
        assert shortcut.parent() is host

    assert not workflow_find_shortcut_available(viewer, host)


def test_find_does_not_steal_keys_when_settings_cannot_be_read(qtbot, monkeypatch):
    host = QWidget()
    qtbot.addWidget(host)

    def unavailable():
        raise RuntimeError("Settings unavailable")

    monkeypatch.setattr("napari.settings.get_settings", unavailable)
    assert not workflow_find_shortcut_available(SimpleNamespace(), host)


def test_find_handles_real_napari_viewer_and_layer_keymaps(qtbot):
    import numpy as np
    from napari.components import ViewerModel

    viewer = ViewerModel()
    layer = viewer.add_image(np.zeros((4, 4)))
    host = QWidget()
    qtbot.addWidget(host)
    assert workflow_find_shortcut_available(viewer, host)
    layer.bind_key("Control-F", lambda _layer: None)
    assert not workflow_find_shortcut_available(viewer, host)
    layer.bind_key("Control-F", None, overwrite=True)
    assert workflow_find_shortcut_available(viewer, host)


def test_find_reserves_secondary_qt_shortcut_bindings(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    shortcut = QShortcut(QKeySequence("Ctrl+G"), host)
    if not hasattr(shortcut, "setKeys"):
        pytest.skip("Multiple QShortcut keys require Qt 6")
    shortcut.setKeys([QKeySequence("Ctrl+G"), QKeySequence.Find])
    assert not workflow_find_shortcut_available(SimpleNamespace(), host)
