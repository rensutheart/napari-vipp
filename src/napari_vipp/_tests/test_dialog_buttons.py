"""One OS policy for both custom footers and native-role button boxes."""

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout

from napari_vipp.ui.dialog_buttons import DialogButtonBox, add_dialog_buttons


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_custom_footer_order_preserves_roles(qtbot, platform):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    row = QHBoxLayout(dialog)
    row.addStretch(1)
    stop = QPushButton("Stop safely")
    action = QPushButton("Run")
    dismiss = QPushButton("Close")
    action.setDefault(True)
    clicked = []
    stop.clicked.connect(lambda: clicked.append("stop"))
    dismiss.clicked.connect(dialog.reject)
    ordered = add_dialog_buttons(
        row, actions=(action,), dismiss=dismiss, stop_actions=(stop,), platform=platform
    )
    dialog.show()
    qtbot.waitUntil(lambda: dismiss.width() > 0)
    assert action.isDefault()
    assert not dismiss.autoDefault()
    assert not stop.autoDefault()
    assert not stop.isDefault()
    assert stop.x() < min(action.x(), dismiss.x())
    assert (action.x() < dismiss.x()) == (platform == "win32")
    assert [b.text() for b in ordered] == (
        ["Stop safely", "Run", "Close"]
        if platform == "win32"
        else ["Stop safely", "Close", "Run"]
    )
    with qtbot.waitSignal(dialog.rejected):
        qtbot.keyClick(dialog, Qt.Key_Escape)
    assert clicked == []  # dismissing must not emit a work-cancellation click


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_native_box_matches_custom_policy(qtbot, platform):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    box = DialogButtonBox(
        DialogButtonBox.Save | DialogButtonBox.Cancel, platform=platform
    )
    layout.addWidget(box)
    save = box.button(DialogButtonBox.Save)
    cancel = box.button(DialogButtonBox.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    save.setDefault(True)
    cancel.setAutoDefault(False)
    dialog.show()
    qtbot.waitUntil(lambda: save.width() > 0)
    assert (save.x() < cancel.x()) == (platform == "win32")
    assert save.isDefault()
    with qtbot.waitSignal(dialog.accepted):
        qtbot.keyClick(dialog, Qt.Key_Return)


def test_duplicate_roles_are_rejected(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QHBoxLayout(dialog)
    button = QPushButton("Close")
    with pytest.raises(ValueError, match="only one role"):
        add_dialog_buttons(layout, actions=(button,), dismiss=button)


def test_footer_does_not_promote_an_action_when_return_edits_a_value(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    editor = QLineEdit("12.5")
    layout.addWidget(editor)
    row = QHBoxLayout()
    action, close = QPushButton("Export"), QPushButton("Close")
    called = []
    action.clicked.connect(lambda: called.append("export"))
    close.clicked.connect(dialog.reject)
    add_dialog_buttons(row, actions=(action,), dismiss=close)
    layout.addLayout(row)
    dialog.show()
    editor.setFocus()
    qtbot.keyClick(editor, Qt.Key_Return)
    assert called == []
    assert dialog.isVisible()
    assert not action.autoDefault() and not action.isDefault()
    assert not close.autoDefault() and not close.isDefault()
    assert action.property("vippActionRole") == "action"
    assert close.property("vippActionRole") == "dismiss"


def test_explicit_action_default_survives_nonautomatic_footer_policy(qtbot):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    row = QHBoxLayout(dialog)
    action, close = QPushButton("Open example"), QPushButton("Cancel")
    action.setDefault(True)
    action.clicked.connect(dialog.accept)
    add_dialog_buttons(row, actions=(action,), dismiss=close)
    dialog.show()
    assert action.isDefault() and not action.autoDefault()
    with qtbot.waitSignal(dialog.accepted):
        qtbot.keyClick(dialog, Qt.Key_Return)
