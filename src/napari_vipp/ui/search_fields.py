"""Visible in-field search affordances and non-conflicting Find shortcuts."""

from __future__ import annotations

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QAction, QKeySequence, QPalette
from qtpy.QtWidgets import QLineEdit, QShortcut, QWidget

from napari_vipp.ui.iconography import interface_icon


class SearchLineEdit(QLineEdit):
    """A search field with a full-contrast, theme-aware leading magnifier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.search_action = self.addAction(
            interface_icon("search", self.palette(), 20),
            QLineEdit.LeadingPosition,
        )
        self.search_action.setText("Focus search")
        self.search_action.setToolTip("Focus search")
        self.search_action.triggered.connect(
            lambda: self.setFocus(Qt.ShortcutFocusReason)
        )
        self.setClearButtonEnabled(True)
        self._refresh_search_icon()

    def _refresh_search_icon(self) -> None:
        palette = QPalette(self.palette())
        # Placeholder text is deliberately muted; the search affordance is not.
        palette.setColor(QPalette.ButtonText, palette.color(QPalette.Text))
        self.search_action.setIcon(interface_icon("search", palette, 20))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in {QEvent.PaletteChange, QEvent.StyleChange} and hasattr(
            self, "search_action"
        ):
            self._refresh_search_icon()


def workflow_find_shortcut_available(viewer, host: QWidget) -> bool:
    """Reserve Find for napari/custom bindings before offering workflow search.

    Called when the key is pressed, so newly configured shortcuts also win.
    Nothing is registered in or removed from napari's shortcut settings.
    """
    from napari.settings import get_settings
    from napari.utils.interactions import Shortcut

    find = QKeySequence(QKeySequence.Find)

    def conflicts(sequence: QKeySequence) -> bool:
        return not sequence.isEmpty() and find.matches(sequence) != QKeySequence.NoMatch

    try:
        for bindings in get_settings().shortcuts.shortcuts.values():
            if any(conflicts(QKeySequence(Shortcut(key).qt)) for key in bindings):
                return False

        # User/plugin keymaps can exist outside napari's settings registry.
        # Reserve layer bindings even when that layer is not currently active.
        for provider in (viewer, *getattr(viewer, "layers", ())):
            keymaps = [getattr(provider, "keymap", {})]
            keymaps.extend(
                cls.__dict__.get("class_keymap", {}) for cls in type(provider).__mro__
            )
            for keymap in keymaps:
                for key, callback in keymap.items():
                    if callback is None:
                        continue
                    if key is Ellipsis or conflicts(QKeySequence(Shortcut(key).qt)):
                        return False

        # Menu actions and Qt shortcuts may be owned by napari or other plugins.
        actions = set(host.findChildren(QAction))
        for widget in (host, *host.findChildren(QWidget)):
            actions.update(widget.actions())
        if any(
            conflicts(sequence) for action in actions for sequence in action.shortcuts()
        ):
            return False
        for shortcut in host.findChildren(QShortcut):
            keys = shortcut.keys() if hasattr(shortcut, "keys") else [shortcut.key()]
            if any(conflicts(key) for key in keys):
                return False
    except Exception:
        # If ownership cannot be established, never steal a host shortcut.
        return False
    return True
