"""Categorized operation palette used by the VIPP workflow editor."""

from __future__ import annotations

from qtpy.QtCore import QMimeData, QSize, Qt, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

from napari_vipp._graph import OPERATION_MIME
from napari_vipp._theme import category_color, category_tint
from napari_vipp.core.pipeline import OperationSpec
from napari_vipp.ui.search import _fuzzy_match, _normalize_search_text


class NodePalette(QTreeWidget):
    """Small categorized operation palette for adding nodes."""

    operation_requested = Signal(str)

    def __init__(
        self,
        groups: dict[str, dict[str, list[OperationSpec]]],
        parent=None,
    ):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._category_items: list[QTreeWidgetItem] = []
        self._subcategory_items: list[QTreeWidgetItem] = []
        self._operation_items: list[QTreeWidgetItem] = []
        for category, subgroups in groups.items():
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsDragEnabled)
            self._style_category_item(category_item, category)
            self.addTopLevelItem(category_item)
            self._category_items.append(category_item)
            for subgroup, specs in subgroups.items():
                parent_item = category_item
                if subgroup:
                    subgroup_item = QTreeWidgetItem([subgroup])
                    subgroup_item.setFlags(
                        subgroup_item.flags() & ~Qt.ItemIsDragEnabled
                    )
                    subgroup_item.setData(
                        0,
                        Qt.UserRole + 1,
                        f"{category} {subgroup}",
                    )
                    self._style_subcategory_item(subgroup_item, category)
                    category_item.addChild(subgroup_item)
                    self._subcategory_items.append(subgroup_item)
                    parent_item = subgroup_item
                for spec in specs:
                    item = QTreeWidgetItem([spec.title])
                    item.setData(0, Qt.UserRole, spec.id)
                    item.setData(
                        0,
                        Qt.UserRole + 1,
                        f"{category} {subgroup} {spec.title} {spec.id}",
                    )
                    self._style_operation_item(item, category)
                    parent_item.addChild(item)
                    self._operation_items.append(item)
        self._no_results_item = QTreeWidgetItem(["No matching nodes"])
        self._no_results_item.setFlags(Qt.NoItemFlags)
        self._no_results_item.setHidden(True)
        self.addTopLevelItem(self._no_results_item)
        self._scroll_spacer = QTreeWidgetItem([""])
        self._scroll_spacer.setFlags(Qt.NoItemFlags)
        self._scroll_spacer.setSizeHint(0, QSize(1, 36))
        self.addTopLevelItem(self._scroll_spacer)
        self.expandAll()

    def _style_category_item(self, item: QTreeWidgetItem, category: str) -> None:
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QBrush(QColor(category_color(category))))
        item.setBackground(0, QBrush(QColor(category_tint(category))))

    def _style_subcategory_item(self, item: QTreeWidgetItem, category: str) -> None:
        font = item.font(0)
        font.setItalic(True)
        item.setFont(0, font)
        item.setForeground(0, QBrush(QColor(category_color(category))))

    def _style_operation_item(self, item: QTreeWidgetItem, category: str) -> None:
        item.setForeground(0, QBrush(QColor(category_color(category))))

    def mimeData(self, items):  # noqa: N802
        mime = QMimeData()
        if not items:
            return mime
        operation_id = items[0].data(0, Qt.UserRole)
        if operation_id:
            mime.setData(OPERATION_MIME, str(operation_id).encode())
        return mime

    def _on_item_double_clicked(self, item, _column) -> None:
        operation_id = item.data(0, Qt.UserRole)
        if operation_id:
            self.operation_requested.emit(str(operation_id))

    def set_filter_text(self, text: str) -> None:
        query = _normalize_search_text(text)
        visible_count = 0
        for category_item in self._category_items:
            category_visible, category_count = self._apply_filter_to_children(
                category_item,
                query,
            )
            visible_count += category_count
            category_item.setHidden(not category_visible)
            if category_visible:
                category_item.setExpanded(True)
        self._no_results_item.setHidden(not query or visible_count > 0)

    def _apply_filter_to_children(
        self,
        parent: QTreeWidgetItem,
        query: str,
    ) -> tuple[bool, int]:
        parent_visible = False
        visible_count = 0
        for index in range(parent.childCount()):
            item = parent.child(index)
            if item.data(0, Qt.UserRole):
                haystack = _normalize_search_text(item.data(0, Qt.UserRole + 1))
                visible = not query or _fuzzy_match(query, haystack)
                item.setHidden(not visible)
                parent_visible = parent_visible or visible
                visible_count += int(visible)
            else:
                child_visible, child_count = self._apply_filter_to_children(
                    item,
                    query,
                )
                item.setHidden(not child_visible)
                if child_visible:
                    item.setExpanded(True)
                parent_visible = parent_visible or child_visible
                visible_count += child_count
        return parent_visible, visible_count


__all__ = ["NodePalette"]
