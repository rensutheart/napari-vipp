"""Small, theme-aware widgets used by the paged batch override editor."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QRect, Qt, QTimer
from qtpy.QtGui import QColor, QFont, QFontMetrics, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QTableWidget,
)

from napari_vipp.ui.batch_table_style import batch_table_colors
from napari_vipp.ui.palette_roles import custom_paint_colors


class BatchOverrideParameterHeader(QHeaderView):
    """Keep node identity prominent without bolding units and workflow values.

    The model retains plain, multiline header text for accessibility, copying
    and tooltips. Only its presentation is split into independently styled lines.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)

    NODE_IDENTITY_ROLE = Qt.UserRole + 501

    def section_lines(self, logical_index) -> tuple[str, ...]:
        text = str(self.model().headerData(logical_index, Qt.Horizontal) or "")
        lines = text.splitlines()
        identity = self.model().headerData(
            logical_index, Qt.Horizontal, self.NODE_IDENTITY_ROLE
        )
        if identity and lines:
            label, node_id = identity
            lines[0:1] = [label, *([f"[{node_id}]"] if node_id else [])]
        return tuple(lines)

    def line_fonts(self) -> tuple[QFont, QFont]:
        normal = QFont(self.font())
        normal.setWeight(QFont.Normal)
        title = QFont(normal)
        title.setWeight(QFont.Bold)
        return title, normal

    def sizeHint(self):  # noqa: N802
        size = super().sizeHint()
        if self.count() > 1:
            title, normal = self.line_fonts()
            line_count = max(
                len(self.section_lines(column)) for column in range(1, self.count())
            )
            # QHeaderView caches its native section metrics. Resolve the mixed
            # fonts here as well so runtime font/theme changes resize the viewport.
            size.setHeight(
                QFontMetrics(title).height()
                + max(0, line_count - 1) * QFontMetrics(normal).height()
                + 16
            )
        return size

    def sectionSizeFromContents(self, logical_index):  # noqa: N802
        size = super().sectionSizeFromContents(logical_index)
        if logical_index:
            title, normal = self.line_fonts()
            size.setHeight(
                QFontMetrics(title).height()
                + max(0, len(self.section_lines(logical_index)) - 1)
                * QFontMetrics(normal).height()
                + 16
            )
        return size

    def paintSection(self, painter, rect, logical_index):  # noqa: N802
        if not logical_index:
            super().paintSection(painter, rect, logical_index)
            return
        lines = self.section_lines(logical_index)
        colors = batch_table_colors(self.palette())
        title_font, normal_font = self.line_fonts()
        title_height = QFontMetrics(title_font).height()
        line_height = QFontMetrics(normal_font).height()
        content_height = title_height + max(0, len(lines) - 1) * line_height
        y = rect.top() + max(4, (rect.height() - content_height) // 2)
        painter.save()
        painter.setClipRect(rect)
        painter.fillRect(rect, QColor(colors.header))
        painter.setPen(QColor(colors.border))
        painter.drawLine(rect.topRight(), rect.bottomRight())
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        for line_index, line in enumerate(lines):
            font = title_font if line_index == 0 else normal_font
            height = title_height if line_index == 0 else line_height
            painter.setFont(font)
            painter.setPen(
                QColor(
                    colors.disabled_text
                    if line.startswith("[") or line.startswith("Workflow:")
                    else colors.text
                )
            )
            # Long custom titles remain recoverable from the full header tooltip.
            # Middle elision also preserves a duplicate node's identifying suffix.
            visible = QFontMetrics(font).elidedText(
                line, Qt.ElideMiddle, max(0, rect.width() - 16)
            )
            painter.drawText(
                QRect(rect.left() + 8, y, max(0, rect.width() - 16), height),
                Qt.AlignCenter,
                visible,
            )
            y += height
        painter.restore()


class _BatchOverrideIdentityHeader(QHeaderView):
    """Give the frozen view the same viewport margin as the multiline header."""

    def __init__(self, source_header, parent):
        self._source_header = source_header
        super().__init__(Qt.Horizontal, parent)

    def sizeHint(self):  # noqa: N802
        size = super().sizeHint()
        size.setHeight(self._source_header.height())
        return size


class BatchOverrideTable(QTableWidget):
    """Keep sample names and checkboxes visible while parameter columns scroll."""

    def __init__(self, parent=None):
        super().__init__(0, 1, parent)
        self.setHorizontalHeader(BatchOverrideParameterHeader(self))
        self.verticalHeader().hide()
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.frozen_identity = QTableView(self)
        self.frozen_identity.setObjectName("batchOverrideFrozenIdentity")
        self.frozen_identity.setHorizontalHeader(
            _BatchOverrideIdentityHeader(self.horizontalHeader(), self.frozen_identity)
        )
        self.frozen_identity.setModel(self.model())
        self.frozen_identity.setSelectionModel(self.selectionModel())
        self.frozen_identity.setFocusPolicy(Qt.NoFocus)
        self.frozen_identity.setFrameShape(QFrame.NoFrame)
        self.frozen_identity.verticalHeader().hide()
        self.frozen_identity.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen_identity.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen_identity.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.frozen_identity.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.frozen_identity.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.frozen_identity.setSelectionMode(QAbstractItemView.SingleSelection)
        self.frozen_identity.setAlternatingRowColors(True)
        self.frozen_identity.horizontalHeader().setSectionsMovable(False)
        self._frozen_sync_timer = QTimer(self)
        self._frozen_sync_timer.setSingleShot(True)
        self._frozen_sync_timer.setInterval(0)
        self._frozen_sync_timer.timeout.connect(self.sync_frozen_identity)
        self.horizontalHeader().geometriesChanged.connect(self._queue_identity_sync)
        self.horizontalHeader().sectionResized.connect(self._section_resized)
        self.verticalHeader().sectionResized.connect(
            lambda row, _old, size: self.frozen_identity.setRowHeight(row, size)
        )
        self.verticalScrollBar().valueChanged.connect(
            self.frozen_identity.verticalScrollBar().setValue
        )
        self.frozen_identity.verticalScrollBar().valueChanged.connect(
            self.verticalScrollBar().setValue
        )

    def sync_frozen_identity(self):
        for column in range(1, self.columnCount()):
            if not self.frozen_identity.isColumnHidden(column):
                self.frozen_identity.setColumnHidden(column, True)
        if self.frozen_identity.columnWidth(0) != self.columnWidth(0):
            self.frozen_identity.setColumnWidth(0, self.columnWidth(0))
        header = self.frozen_identity.horizontalHeader()
        height = self.horizontalHeader().height()
        if header.minimumHeight() != height or header.maximumHeight() != height:
            header.setFixedHeight(height)
        # A fixed header height alone does not change QTableView's top viewport
        # margin: Qt asks sizeHint() again when computing the internal geometry.
        self.frozen_identity.updateGeometries()
        for row in range(self.rowCount()):
            if self.frozen_identity.rowHeight(row) != self.rowHeight(row):
                self.frozen_identity.setRowHeight(row, self.rowHeight(row))
        self._position_identity()

    def event(self, event):
        result = super().event(event)
        if event.type() in (
            QEvent.Show,
            QEvent.StyleChange,
            QEvent.FontChange,
            QEvent.LayoutRequest,
            QEvent.PolishRequest,
        ):
            self._queue_identity_sync()
        return result

    def _queue_identity_sync(self):
        if hasattr(self, "_frozen_sync_timer"):
            # Hidden tabs are configured before Qt knows the multiline header's
            # final styled height. Reconcile after the show/theme layout settles.
            self._frozen_sync_timer.start()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "frozen_identity"):
            self._position_identity()
            self._queue_identity_sync()

    def _section_resized(self, column, _old, size):
        if column == 0:
            self.frozen_identity.setColumnWidth(0, size)
            self._position_identity()

    def _position_identity(self):
        # QSS can supply a frame even for NoFrame. Align the two *viewports*,
        # accounting for that extra inset rather than stacking two borders.
        frozen_frame = self.frozen_identity.frameWidth()
        self.frozen_identity.setGeometry(
            self.viewport().x() - frozen_frame,
            self.horizontalHeader().y() - frozen_frame,
            self.columnWidth(0) + 2 * frozen_frame,
            self.viewport().height()
            + self.horizontalHeader().height()
            + 2 * frozen_frame,
        )
        self.frozen_identity.raise_()


class BatchOverrideDelegate(QStyledItemDelegate):
    """Render inheritance without allocating an editor for every matrix cell."""

    def __init__(self, owner):
        super().__init__(owner.table)
        self.owner = owner

    def paint(self, painter, option, index):
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        if index.column():
            styled.displayAlignment = Qt.AlignCenter
        if index.column() and not str(index.data(Qt.EditRole) or "").strip():
            binding = self.owner._visible_parameters[index.column() - 1]
            styled.text = "inherit " + self.owner._format_value(
                binding.workflow_value, binding
            )
            styled.palette.setColor(
                QPalette.Text, custom_paint_colors(styled.palette).muted_text
            )
        widget = styled.widget
        from qtpy.QtWidgets import QApplication, QStyle

        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, styled, painter, widget)

    def createEditor(self, parent, option, index):  # noqa: N802
        if not index.column():
            return None
        binding = self.owner._visible_parameters[index.column() - 1]
        source_key = self.owner._page_keys[index.row()]
        key = (source_key, binding.node_id, binding.parameter.name)
        editor = self.owner._make_editor(binding)
        editor.setParent(parent)
        editor.setProperty("override_key", key)
        self.owner._editors[key] = editor
        editor.textChanged.connect(
            lambda text, key=key: self.owner._cell_changed(key, text)
        )
        editor.editingFinished.connect(self.owner._filter_refresh_timer.start)
        return editor

    def setEditorData(self, editor, index):  # noqa: N802
        editor.blockSignals(True)
        editor.setText(str(index.data(Qt.EditRole) or ""))
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):  # noqa: N802
        # Raw text is persisted on every edit; the model is only the current page.
        model.setData(index, editor.text(), Qt.EditRole)

    def destroyEditor(self, editor, index):  # noqa: N802
        key = tuple(editor.property("override_key"))
        if self.owner._editors.get(key) is editor:
            self.owner._editors.pop(key, None)
        super().destroyEditor(editor, index)
