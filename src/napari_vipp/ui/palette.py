"""Responsive, categorized node library used by the workflow editor."""

from __future__ import annotations

import inspect
from collections.abc import Iterable

from qtpy.QtCore import QEvent, QMimeData, QPoint, QPointF, QSize, Qt, Signal
from qtpy.QtGui import (
    QBrush,
    QGuiApplication,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp._graph import OPERATION_MIME
from napari_vipp.core.pipeline import OperationSpec
from napari_vipp.ui.iconography import (
    category_icon,
    interface_icon,
    operation_icon,
    palette_branch_color,
    palette_category_colors,
)
from napari_vipp.ui.panel_toggle import SidePanelToggleButton
from napari_vipp.ui.search import _fuzzy_match, _normalize_search_text

OPERATION_ROLE = Qt.UserRole
SEARCH_TEXT_ROLE = Qt.UserRole + 1
CATEGORY_ROLE = Qt.UserRole + 2


def _operation_count(subgroups: dict[str, list[OperationSpec]]) -> int:
    return sum(len(specs) for specs in subgroups.values())


def _exec_menu(menu: QMenu, position: QPoint):
    """Execute a context menu across supported Qt bindings."""

    if hasattr(menu, "exec"):
        return menu.exec(position)
    return menu.exec_(position)


def _operation_description(spec: OperationSpec) -> str:
    """Return a concise plain-text explanation for a node-library row."""

    if spec.id == "input":
        return "Load an image, image collection, or supported microscopy dataset."
    documentation = inspect.getdoc(spec.function) if spec.function else ""
    summary = documentation.splitlines()[0].strip() if documentation else ""
    if summary:
        return summary.replace("``", "").replace("`", "")
    return (
        f"Process {spec.input_type or 'workflow'} data and produce "
        f"{spec.output_type} output."
    )


class NodePalette(QTreeWidget):
    """Categorized node tree with drag, search, and keyboard activation."""

    operation_requested = Signal(str)

    def __init__(
        self,
        groups: dict[str, dict[str, list[OperationSpec]]],
        parent=None,
    ):
        super().__init__(parent)
        self._groups = groups
        self._scope_category: str | None = None
        self._filter_active = False
        self._expansion_snapshot: list[tuple[QTreeWidgetItem, bool]] = []
        self._category_items: list[QTreeWidgetItem] = []
        self._subcategory_items: list[QTreeWidgetItem] = []
        self._operation_items: list[QTreeWidgetItem] = []
        self._category_by_item: dict[int, str] = {}
        self._spec_by_item: dict[int, OperationSpec] = {}

        self.setColumnCount(2)
        self.setHeaderHidden(True)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.setRootIsDecorated(True)
        self.setIndentation(16)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumWidth(0)
        self.setAccessibleName("Available workflow nodes")
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)
        self._populate()

    def _populate(self) -> None:
        for category, subgroups in self._groups.items():
            count = _operation_count(subgroups)
            category_item = QTreeWidgetItem([category, str(count)])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsDragEnabled)
            category_item.setData(0, CATEGORY_ROLE, category)
            category_item.setData(0, SEARCH_TEXT_ROLE, category)
            category_item.setData(
                0,
                Qt.AccessibleTextRole,
                f"{category}, {count} nodes",
            )
            category_item.setData(
                0,
                Qt.AccessibleDescriptionRole,
                f"Node category containing {count} workflow nodes",
            )
            category_item.setToolTip(0, f"{category} — {count} nodes")
            self.addTopLevelItem(category_item)
            self._category_items.append(category_item)
            self._category_by_item[id(category_item)] = category
            for subgroup, specs in subgroups.items():
                parent_item = category_item
                if subgroup:
                    subgroup_item = QTreeWidgetItem([subgroup, str(len(specs))])
                    subgroup_item.setFlags(
                        subgroup_item.flags() & ~Qt.ItemIsDragEnabled
                    )
                    subgroup_item.setData(0, CATEGORY_ROLE, category)
                    subgroup_item.setData(
                        0,
                        Qt.AccessibleTextRole,
                        f"{subgroup}, {len(specs)} nodes",
                    )
                    subgroup_item.setData(
                        0,
                        SEARCH_TEXT_ROLE,
                        f"{category} {subgroup}",
                    )
                    category_item.addChild(subgroup_item)
                    self._subcategory_items.append(subgroup_item)
                    self._category_by_item[id(subgroup_item)] = category
                    parent_item = subgroup_item
                for spec in specs:
                    description = _operation_description(spec)
                    item = QTreeWidgetItem([spec.title, ""])
                    item.setData(0, OPERATION_ROLE, spec.id)
                    item.setData(0, CATEGORY_ROLE, category)
                    item.setData(
                        0,
                        SEARCH_TEXT_ROLE,
                        f"{category} {subgroup} {spec.title} {spec.id}",
                    )
                    item.setData(0, Qt.AccessibleDescriptionRole, description)
                    item.setToolTip(0, description)
                    parent_item.addChild(item)
                    self._operation_items.append(item)
                    self._category_by_item[id(item)] = category
                    self._spec_by_item[id(item)] = spec

        self._no_results_item = QTreeWidgetItem(["No matching nodes", ""])
        self._no_results_item.setFlags(Qt.NoItemFlags)
        self._no_results_item.setHidden(True)
        self.addTopLevelItem(self._no_results_item)
        self._scroll_spacer = QTreeWidgetItem(["", ""])
        self._scroll_spacer.setFlags(Qt.NoItemFlags)
        self._scroll_spacer.setSizeHint(0, QSize(1, 36))
        self.addTopLevelItem(self._scroll_spacer)
        self._restyle_items()
        self.expandAll()

    def _restyle_items(self) -> None:
        palette = self.palette()
        for item in self._category_items:
            category = self._category_by_item[id(item)]
            text, tint, _accent = palette_category_colors(category, palette)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setIcon(0, category_icon(category, palette))
            for column in (0, 1):
                item.setForeground(column, QBrush(text))
                item.setBackground(column, QBrush(tint))
            count_font = item.font(1)
            count_font.setBold(False)
            item.setFont(1, count_font)
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        for item in self._subcategory_items:
            category = self._category_by_item[id(item)]
            text, _tint, _accent = palette_category_colors(category, palette)
            font = item.font(0)
            font.setItalic(True)
            item.setFont(0, font)
            item.setForeground(0, QBrush(text))
            item.setForeground(1, QBrush(text))
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        for item in self._operation_items:
            category = self._category_by_item[id(item)]
            text, _tint, _accent = palette_category_colors(category, palette)
            item.setForeground(0, QBrush(text))
            item.setIcon(0, operation_icon(self._spec_by_item[id(item)], palette))

    def drawBranches(self, painter, rect, index) -> None:  # noqa: N802
        """Paint high-contrast, category-colored disclosure chevrons."""

        if not index.isValid() or not self.model().hasChildren(index):
            return
        category = str(index.data(CATEGORY_ROLE) or "")
        if self.selectionModel().isSelected(index):
            color = self.palette().color(QPalette.HighlightedText)
        else:
            color = palette_branch_color(category, self.palette())

        slot_width = max(12.0, min(float(self.indentation()), float(rect.width())))
        if self.layoutDirection() == Qt.RightToLeft:
            center_x = float(rect.left()) + slot_width / 2.0
        else:
            center_x = float(rect.right()) - slot_width / 2.0 + 1.0
        center_y = float(rect.center().y())
        path = QPainterPath()
        if self.isExpanded(index):
            path.moveTo(QPointF(center_x - 4.0, center_y - 2.2))
            path.lineTo(QPointF(center_x, center_y + 2.2))
            path.lineTo(QPointF(center_x + 4.0, center_y - 2.2))
        else:
            direction = -1.0 if self.layoutDirection() == Qt.RightToLeft else 1.0
            path.moveTo(QPointF(center_x - 2.2 * direction, center_y - 4.0))
            path.lineTo(QPointF(center_x + 2.2 * direction, center_y))
            path.lineTo(QPointF(center_x - 2.2 * direction, center_y + 4.0))

        pen = QPen(color, 1.9)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.restore()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._restyle_items()

    def mimeData(self, items):  # noqa: N802
        mime = QMimeData()
        if not items:
            return mime
        operation_id = items[0].data(0, OPERATION_ROLE)
        if operation_id:
            mime.setData(OPERATION_MIME, str(operation_id).encode())
        return mime

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            item = self.currentItem()
            operation_id = item.data(0, OPERATION_ROLE) if item else None
            if operation_id:
                self.operation_requested.emit(str(operation_id))
                event.accept()
                return
        super().keyPressEvent(event)

    def _on_item_double_clicked(self, item, _column) -> None:
        operation_id = item.data(0, OPERATION_ROLE)
        if operation_id:
            self.operation_requested.emit(str(operation_id))

    def _context_menu_for_item(
        self,
        item: QTreeWidgetItem | None,
    ) -> QMenu:
        menu = QMenu(self)
        operation_id = item.data(0, OPERATION_ROLE) if item else None
        if operation_id:
            add_action = menu.addAction("Add to workflow")
            add_action.triggered.connect(
                lambda _checked=False, value=str(operation_id): (
                    self.operation_requested.emit(value)
                )
            )
            menu.addSeparator()
        expand_action = menu.addAction("Expand all")
        expand_action.triggered.connect(self.expandAll)
        collapse_action = menu.addAction("Collapse all")
        collapse_action.triggered.connect(self.collapseAll)
        return menu

    def _open_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        if item is not None and item.data(0, OPERATION_ROLE):
            self.setCurrentItem(item)
        menu = self._context_menu_for_item(item)
        try:
            _exec_menu(menu, self.viewport().mapToGlobal(position))
        finally:
            menu.deleteLater()

    def set_scope_category(self, category: str | None) -> None:
        """Restrict the tree to one category without changing its contents."""

        self._scope_category = str(category) if category else None
        self.set_filter_text("")

    def set_filter_text(self, text: str) -> None:
        query = _normalize_search_text(text)
        if query and not self._filter_active:
            self._expansion_snapshot = [
                (item, item.isExpanded())
                for item in (*self._category_items, *self._subcategory_items)
            ]
            self._filter_active = True

        visible_count = 0
        for category_item in self._category_items:
            category = str(category_item.data(0, CATEGORY_ROLE))
            in_scope = self._scope_category is None or category == self._scope_category
            if in_scope:
                category_visible, category_count = self._apply_filter_to_children(
                    category_item,
                    query,
                )
            else:
                category_visible, category_count = False, 0
            visible_count += category_count
            category_item.setHidden(not category_visible)
            if query and category_visible:
                category_item.setExpanded(True)

        self._no_results_item.setHidden(not query or visible_count > 0)
        if not query and self._filter_active:
            for item, expanded in self._expansion_snapshot:
                item.setExpanded(expanded)
            self._expansion_snapshot.clear()
            self._filter_active = False

    def _apply_filter_to_children(
        self,
        parent: QTreeWidgetItem,
        query: str,
    ) -> tuple[bool, int]:
        parent_visible = False
        visible_count = 0
        for index in range(parent.childCount()):
            item = parent.child(index)
            if item.data(0, OPERATION_ROLE):
                haystack = _normalize_search_text(str(item.data(0, SEARCH_TEXT_ROLE)))
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
                if query and child_visible:
                    item.setExpanded(True)
                parent_visible = parent_visible or child_visible
                visible_count += child_count
        return parent_visible, visible_count


class _NodePalettePopup(QFrame):
    operation_requested = Signal(str)
    closed = Signal()

    def __init__(
        self,
        groups: dict[str, dict[str, list[OperationSpec]]],
        parent: QWidget,
    ) -> None:
        super().__init__(parent, Qt.Popup)
        self.setObjectName("vippNodePalettePopup")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(120, 160)
        self.resize(340, 520)
        self._search_edit: QLineEdit | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.title_label = QLabel("Nodes")
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        header.addWidget(self.title_label, 1)
        self.close_button = QToolButton()
        self.close_button.setAutoRaise(True)
        self.close_button.setToolTip("Close node library")
        self.close_button.setAccessibleName("Close node library")
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        layout.addLayout(header)
        self.search_host = QWidget()
        self.search_layout = QVBoxLayout(self.search_host)
        self.search_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.search_host)
        self.palette = NodePalette(groups)
        self.palette.operation_requested.connect(self._request_operation)
        layout.addWidget(self.palette, 1)
        self._refresh_icons()

    def _refresh_icons(self) -> None:
        self.close_button.setIcon(interface_icon("close", self.close_button.palette()))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._refresh_icons()

    def set_category(self, category: str) -> None:
        self.detach_search()
        self.title_label.setText(category)
        self.palette.set_scope_category(category)
        self.palette.expandAll()

    def set_global_search(self, search_edit: QLineEdit) -> None:
        self.title_label.setText("Search all nodes")
        self.palette.set_scope_category(None)
        self.attach_search(search_edit)
        self.palette.set_filter_text(search_edit.text())

    def attach_search(self, search_edit: QLineEdit) -> None:
        if self._search_edit is search_edit:
            return
        self.detach_search()
        self._search_edit = search_edit
        self.search_layout.addWidget(search_edit)
        self.search_host.show()

    def detach_search(self) -> QLineEdit | None:
        search_edit = self._search_edit
        if search_edit is not None:
            self.search_layout.removeWidget(search_edit)
            search_edit.setParent(None)
        self._search_edit = None
        self.search_host.hide()
        return search_edit

    def show_next_to(self, anchor: QWidget) -> None:
        self.adjustSize()
        width = max(300, min(420, self.sizeHint().width()))
        height = max(360, min(620, self.sizeHint().height()))
        origin = anchor.mapToGlobal(QPoint(anchor.width() + 6, 0))
        screen = anchor.screen() if hasattr(anchor, "screen") else None
        if screen is None:
            screen = QGuiApplication.screenAt(origin)
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, max(120, available.width() - 16))
            height = min(height, max(160, available.height() - 16))
            self.resize(width, height)
            origin.setX(
                max(
                    available.left(),
                    min(origin.x(), available.right() - self.width() + 1),
                )
            )
            origin.setY(
                max(
                    available.top(),
                    min(origin.y(), available.bottom() - self.height() + 1),
                )
            )
        else:
            self.resize(width, height)
        self.move(origin)
        self.show()
        self.raise_()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self.closed.emit()

    def _request_operation(self, operation_id: str) -> None:
        self.operation_requested.emit(operation_id)
        self.close()


class _NodeCategoryRail(QWidget):
    category_requested = Signal(str, object)
    search_requested = Signal(object)
    expand_requested = Signal()

    def __init__(
        self,
        groups: dict[str, dict[str, list[OperationSpec]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._groups = groups
        self._category_buttons: list[QToolButton] = []
        self.setAccessibleName("Compact node library")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(3)

        self.expand_button = SidePanelToggleButton("left")
        self.expand_button.set_expanded(False)
        self.expand_button.setAccessibleName("Expand node library")
        self.expand_button.setToolTip("Show category names and the full node tree")
        self.expand_button.clicked.connect(
            lambda _checked=False: self.expand_requested.emit()
        )
        outer.addWidget(self.expand_button, 0, Qt.AlignHCenter)

        self.search_button = self._control_button(
            "Search all nodes",
            "Search all available workflow nodes",
        )
        self.search_button.clicked.connect(
            lambda _checked=False: self.search_requested.emit(self.search_button)
        )
        outer.addWidget(self.search_button)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        category_host = QWidget()
        category_layout = QVBoxLayout(category_host)
        category_layout.setContentsMargins(0, 3, 0, 3)
        category_layout.setSpacing(3)
        for category, subgroups in groups.items():
            count = _operation_count(subgroups)
            button = self._control_button(
                f"{category}, {count} nodes",
                f"{category} — {count} nodes",
            )
            button.setProperty("vipp_category", category)
            button.clicked.connect(
                lambda _checked=False, name=category, anchor=button: (
                    self.category_requested.emit(name, anchor)
                )
            )
            category_layout.addWidget(button)
            self._category_buttons.append(button)
        category_layout.addStretch(1)
        scroll.setWidget(category_host)
        outer.addWidget(scroll, 1)
        self._refresh_icons()

    @staticmethod
    def _control_button(accessible_name: str, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setAutoRaise(True)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setIconSize(QSize(20, 20))
        button.setFixedHeight(36)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setAccessibleName(accessible_name)
        button.setToolTip(tooltip)
        return button

    def _refresh_icons(self) -> None:
        palette = self.palette()
        self.search_button.setIcon(interface_icon("search", palette, 20))
        for button in self._category_buttons:
            category = str(button.property("vipp_category"))
            button.setIcon(category_icon(category, palette, 20))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._refresh_icons()

    @property
    def category_buttons(self) -> Iterable[QToolButton]:
        return tuple(self._category_buttons)


class NodeLibraryPanel(QWidget):
    """Responsive node library with an inline tree and compact icon rail."""

    operation_requested = Signal(str)
    compact_changed = Signal(bool, bool)
    compact_requested = Signal(bool)

    COMPACT_WIDTH = 48
    EXPANDED_MINIMUM_WIDTH = 240

    def __init__(
        self,
        groups: dict[str, dict[str, list[OperationSpec]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._groups = groups
        self._compact = False
        self._last_popup_anchor: QWidget | None = None
        self.setObjectName("vippNodeLibrary")
        self.setMinimumWidth(self.COMPACT_WIDTH)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.expanded_page = QWidget()
        expanded_layout = QVBoxLayout(self.expanded_page)
        expanded_layout.setContentsMargins(4, 4, 4, 4)
        expanded_layout.setSpacing(4)
        header = QHBoxLayout()
        self.title_label = QLabel("Nodes")
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        header.addWidget(self.title_label, 1)
        self.collapse_button = SidePanelToggleButton("left")
        self.collapse_button.set_expanded(True)
        self.collapse_button.setToolTip("Collapse node library")
        self.collapse_button.setAccessibleName("Collapse node library")
        self.collapse_button.clicked.connect(
            lambda _checked=False: self.compact_requested.emit(True)
        )
        header.addWidget(self.collapse_button)
        expanded_layout.addLayout(header)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find a node to add…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setAccessibleName("Search workflow nodes")
        self.search_host = QWidget()
        self.search_layout = QVBoxLayout(self.search_host)
        self.search_layout.setContentsMargins(0, 0, 0, 0)
        self.search_layout.addWidget(self.search_edit)
        expanded_layout.addWidget(self.search_host)
        self.palette = NodePalette(groups)
        expanded_layout.addWidget(self.palette, 1)
        self.stack.addWidget(self.expanded_page)

        self.compact_rail = _NodeCategoryRail(groups)
        self.stack.addWidget(self.compact_rail)
        self.popup = _NodePalettePopup(groups, self)

        self.search_edit.textChanged.connect(self._filter_all_palettes)
        self.palette.operation_requested.connect(self.operation_requested.emit)
        self.popup.operation_requested.connect(self.operation_requested.emit)
        self.popup.closed.connect(self._restore_search_from_popup)
        self.compact_rail.search_requested.connect(self.open_global_search)
        self.compact_rail.category_requested.connect(self.open_category)
        self.compact_rail.expand_requested.connect(
            lambda: self.compact_requested.emit(False)
        )
        self._sync_popup_palette()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self._sync_popup_palette()

    def _sync_popup_palette(self) -> None:
        """Keep the top-level popup aligned with its docked panel theme."""

        popup = getattr(self, "popup", None)
        if popup is not None:
            # ``self.palette`` is the public NodePalette compatibility alias.
            # Call QWidget's descriptor directly for the actual Qt palette.
            popup.setPalette(QWidget.palette(self))

    @property
    def is_compact(self) -> bool:
        return self._compact

    def set_compact(
        self,
        compact: bool,
        *,
        user_initiated: bool = False,
    ) -> None:
        compact = bool(compact)
        if compact == self._compact:
            if user_initiated:
                self.compact_changed.emit(compact, True)
            return
        if self.popup.isVisible():
            self.popup.close()
        focus = QApplication.focusWidget()
        focus_was_in_hidden_page = bool(
            focus is not None
            and (
                focus is self.search_edit
                or self.expanded_page.isAncestorOf(focus)
                or self.compact_rail.isAncestorOf(focus)
            )
        )
        self._compact = compact
        self.stack.setCurrentWidget(
            self.compact_rail if compact else self.expanded_page
        )
        if focus_was_in_hidden_page:
            if compact:
                self.compact_rail.search_button.setFocus(Qt.OtherFocusReason)
            else:
                self.search_edit.setFocus(Qt.OtherFocusReason)
        self.compact_changed.emit(compact, bool(user_initiated))

    def open_global_search(self, anchor: QWidget) -> None:
        self._last_popup_anchor = anchor
        self.popup.set_global_search(self.search_edit)
        self.popup.show_next_to(anchor)
        self.search_edit.setFocus(Qt.PopupFocusReason)
        self.search_edit.selectAll()

    def open_category(self, category: str, anchor: QWidget) -> None:
        self._restore_search_from_popup()
        self._last_popup_anchor = anchor
        self.popup.set_category(category)
        self.popup.show_next_to(anchor)
        self.popup.palette.setFocus(Qt.PopupFocusReason)

    def dismiss_popup(self) -> None:
        if self.popup.isVisible():
            self.popup.close()
        else:
            self._restore_search_from_popup()

    def _filter_all_palettes(self, text: str) -> None:
        self.palette.set_filter_text(text)
        if self.popup._search_edit is self.search_edit:  # noqa: SLF001
            self.popup.palette.set_filter_text(text)

    def _restore_search_from_popup(self) -> None:
        search_edit = self.popup.detach_search()
        if search_edit is self.search_edit:
            self.search_layout.addWidget(self.search_edit)
            self.search_host.show()
        if self._last_popup_anchor is not None and self.isVisible():
            self._last_popup_anchor.setFocus(Qt.PopupFocusReason)
        self._last_popup_anchor = None

    def hideEvent(self, event) -> None:  # noqa: N802
        self.dismiss_popup()
        super().hideEvent(event)


__all__ = ["NodeLibraryPanel", "NodePalette"]
