"""Consistent paired bulk-selection actions for lists, tables and column trees."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QRect, QSize, Qt, QTimer
from qtpy.QtWidgets import QLabel, QLayout, QSizePolicy, QVBoxLayout, QWidget

from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class _SelectionButtonLayout(QLayout):
    """Keep the full pair together, wrapping instead of clipping narrow panels."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(6)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < self.count() else None

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), measure=True)

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def minimumSize(self):  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def sizeHint(self):  # noqa: N802
        hints = [item.sizeHint() for item in self._items]
        return QSize(
            sum(hint.width() for hint in hints)
            + max(0, len(hints) - 1) * self.spacing(),
            max((hint.height() for hint in hints), default=0),
        )

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, measure=False)

    def _arrange(self, rect: QRect, *, measure: bool) -> int:
        x, y, row_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint().expandedTo(item.minimumSize())
            if row_height and x + hint.width() > rect.x() + rect.width():
                x = rect.x()
                y += row_height + self.spacing()
                row_height = 0
            if not measure:
                item.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x += hint.width() + self.spacing()
            row_height = max(row_height, hint.height())
        return y + row_height - rect.y()


class BulkSelectionControls(QWidget):
    """One selection pair; callers retain scope, callbacks and enabled state."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        scope: str = "",
        select_tooltip: str = "",
        deselect_tooltip: str = "",
    ):
        super().__init__(parent)
        self.setObjectName("BulkSelectionControls")
        self.setAccessibleName("Bulk selection")
        self.button_row = QWidget(self)
        self.button_layout = _SelectionButtonLayout(self.button_row)
        self.select_all_button = ToolbarCommandButton("Select all", self.button_row)
        self.deselect_all_button = ToolbarCommandButton(
            "Deselect all", self.button_row
        )
        for button, tooltip in (
            (self.select_all_button, select_tooltip),
            (self.deselect_all_button, deselect_tooltip),
        ):
            button.setIconSize(QSize(18, 18))
            button.setAutoDefault(False)
            button.setDefault(False)
            button.setToolTip(tooltip or scope)
            button.setAccessibleDescription(scope)
            button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.button_layout.addWidget(button)
            button.installEventFilter(self)
        self.scope_label = QLabel(scope, self)
        self.scope_label.setObjectName("BulkSelectionScope")
        self.scope_label.setMinimumWidth(0)
        self.scope_label.setWordWrap(True)
        scope_policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        scope_policy.setHeightForWidth(True)
        self.scope_label.setSizePolicy(scope_policy)
        self.scope_label.setVisible(bool(scope))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.button_row)
        layout.addWidget(self.scope_label)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        row_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        row_policy.setHeightForWidth(True)
        self.button_row.setSizePolicy(row_policy)
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setSingleShot(True)
        self._geometry_timer.timeout.connect(self._sync_height)
        self.installEventFilter(self)
        self._icon_timer = QTimer(self)
        self._icon_timer.setSingleShot(True)
        self._icon_timer.timeout.connect(self._refresh_icons)
        self._refresh_icons()

    def set_scope(self, text: str) -> None:
        """Update a live filter summary without changing selection behavior."""
        self.scope_label.setText(text)
        self.scope_label.setVisible(bool(text))
        for button in (self.select_all_button, self.deselect_all_button):
            button.setAccessibleDescription(text)
        self._geometry_timer.start(0)

    def _sync_height(self) -> None:
        # Some inspector sections cap their children at preferred heights.
        # Reserve the actual width-dependent height so both wrapped buttons
        # and the complete scope survive resizing and live font changes.
        if not self.isVisible():
            return
        height = self.layout().totalHeightForWidth(max(1, self.width()))
        if height >= 0 and (
            self.minimumHeight() != height or self.maximumHeight() != height
        ):
            self.setFixedHeight(height)

    def _refresh_icons(self) -> None:
        for button, kind in (
            (self.select_all_button, "select_all"),
            (self.deselect_all_button, "deselect"),
        ):
            button.setIcon(toolbar_icon(kind, button.palette()))

    def eventFilter(self, watched, event):  # noqa: N802
        timer = getattr(self, "_icon_timer", None)
        if timer is not None and event.type() in {
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
            QEvent.Polish,
        }:
            timer.start(0)
        geometry_timer = getattr(self, "_geometry_timer", None)
        if geometry_timer is not None and event.type() in {
            QEvent.FontChange,
            QEvent.ApplicationFontChange,
            QEvent.StyleChange,
            QEvent.Polish,
            QEvent.Show,
            QEvent.Resize,
            QEvent.LayoutRequest,
        }:
            geometry_timer.start(0)
        return super().eventFilter(watched, event)
