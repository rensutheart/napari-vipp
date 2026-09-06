"""Table links activate on their visible text, not the rest of the row."""

from qtpy.QtCore import QRect, QSize, Qt, Signal
from qtpy.QtGui import QFontMetrics
from qtpy.QtWidgets import (
    QApplication,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
)


class BatchResultLinkTable(QTableWidget):
    """Retain normal row selection while treating underlined text as links."""

    linkActivated = Signal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hit testing needs the same style option as native item painting.
        # PyQt cannot call protected initStyleOption on Qt's C++-created default
        # delegate. A Python-owned standard delegate preserves the rendering
        # contract and makes that protected helper available on every binding.
        self._link_delegate = QStyledItemDelegate(self)
        self.setItemDelegate(self._link_delegate)
        self._pressed_link = None
        self.setMouseTracking(True)
        self.setAccessibleDescription(
            "Click blank row space to select. Click underlined text, or press "
            "Enter on a focused link, to open it."
        )

    def link_rect(self, index):
        """Return the painted, elided text bounds in viewport coordinates."""
        if not index.isValid():
            return QRect()
        item = self.item(index.row(), index.column())
        if item is None or not item.font().underline():
            return QRect()
        option = QStyleOptionViewItem()
        option.initFrom(self)
        option.widget = self
        option.font = self.font()
        option.rect = self.visualRect(index)
        self.itemDelegate().initStyleOption(option, index)
        text_rect = self.style().subElementRect(
            QStyle.SE_ItemViewItemText, option, self
        )
        metrics = QFontMetrics(option.font)
        text = metrics.elidedText(option.text, self.textElideMode(), text_rect.width())
        bounds = QStyle.alignedRect(
            option.direction,
            option.displayAlignment,
            QSize(metrics.horizontalAdvance(text), metrics.height()),
            text_rect,
        )
        return bounds.intersected(text_rect).intersected(self.viewport().rect())

    def mousePressEvent(self, event):  # noqa: N802
        position = event.position().toPoint()
        index = self.indexAt(position)
        self._pressed_link = None
        if event.button() == Qt.LeftButton and self.link_rect(index).contains(position):
            self._pressed_link = (index.row(), index.column(), index.data(), position)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        pressed = self._pressed_link
        self._pressed_link = None
        super().mouseReleaseEvent(event)
        if pressed is None or event.button() != Qt.LeftButton:
            return
        position = event.position().toPoint()
        index = self.indexAt(position)
        row, column, text, start = pressed
        if (
            index.isValid()
            and (index.row(), index.column(), index.data()) == (row, column, text)
            and (position - start).manhattanLength() <= QApplication.startDragDistance()
            and self.link_rect(index).contains(position)
        ):
            self.linkActivated.emit(row, column)

    def mouseMoveEvent(self, event):  # noqa: N802
        position = event.position().toPoint()
        if self.link_rect(self.indexAt(position)).contains(position):
            self.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.viewport().unsetCursor()
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            index = self.currentIndex()
            if not self.link_rect(index).isEmpty():
                self.linkActivated.emit(index.row(), index.column())
                event.accept()
                return
        super().keyPressEvent(event)
