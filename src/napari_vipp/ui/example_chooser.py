"""Guided, presentation-only browser for bundled workflow examples."""

from __future__ import annotations

from dataclasses import fields
from html import escape

from qtpy.QtCore import QEvent, QRect, QSize, Qt, QUrl
from qtpy.QtGui import QDesktopServices, QFontMetrics, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.dialog_buttons import DialogButtonBox as QDialogButtonBox
from napari_vipp.ui.examples import EXAMPLE_WORKFLOWS, ExampleWorkflowSpec
from napari_vipp.ui.palette_roles import blend_colors, theme_colors
from napari_vipp.ui.search import _normalize_search_text
from napari_vipp.ui.toolbar_controls import toolbar_icon

_CATEGORY_ORDER = (
    "Segmentation & Labels",
    "Measurements & Tables",
    "Colocalization & Association",
    "Restoration & PSF",
    "Skeletons & Networks",
    "3D Meshes",
    "Batch & Reproducibility",
)
_DEVELOPER_CATEGORY = "Developer & testing workflows"


class _ExampleTitleDelegate(QStyledItemDelegate):
    """Allow long workflow titles to wrap instead of becoming ellipses."""

    def sizeHint(self, option, index):  # noqa: N802
        tree = self.parent()
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        depth = 2 if index.parent().isValid() else 1
        width = max(80, tree.viewport().width() - depth * tree.indentation() - 20)
        bounds = QFontMetrics(styled.font).boundingRect(
            QRect(0, 0, width, 10000), Qt.TextWordWrap, styled.text
        )
        return QSize(width, max(30, bounds.height() + 14))


class ExampleWorkflowDialog(QDialog):
    """Browse guidance without loading images or changing the active workflow."""

    def __init__(self, parent=None, examples=EXAMPLE_WORKFLOWS):
        super().__init__(parent)
        self._examples: tuple[ExampleWorkflowSpec, ...] = tuple(examples)
        self._selected_example_id = ""
        self.setWindowTitle("Open VIPP Example")
        self.setMinimumSize(600, 480)
        self.resize(1040, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setObjectName("exampleBrowserSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(7)
        self.sidebar = QWidget()
        self.sidebar.setObjectName("exampleSidebar")
        self.sidebar.setAttribute(Qt.WA_StyledBackground, True)
        self.sidebar.setMinimumWidth(220)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(4, 6, 6, 4)
        side_layout.setSpacing(6)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Search examples, inputs or tasks")
        self.filter_edit.setAccessibleName("Search example workflows")
        self.filter_edit.setClearButtonEnabled(True)
        self.search_action = self.filter_edit.addAction(
            toolbar_icon("search", self.palette()), QLineEdit.LeadingPosition
        )
        self.filter_edit.setMinimumHeight(self.filter_edit.fontMetrics().height() + 16)
        self.filter_edit.installEventFilter(self)
        side_layout.addWidget(self.filter_edit)
        self.count_label = self._label()
        side_layout.addWidget(self.count_label)
        self.tree = QTreeWidget()
        self.tree.setObjectName("exampleTree")
        self.tree.viewport().setObjectName("exampleTreeViewport")
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        self.tree.setAccessibleName("Example workflows by category")
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setWordWrap(True)
        self.tree.setTextElideMode(Qt.ElideNone)
        self.tree.setUniformRowHeights(False)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.tree.setIndentation(14)
        self.tree.setItemDelegate(_ExampleTitleDelegate(self.tree))
        self.tree.viewport().installEventFilter(self)
        self.tree.installEventFilter(self)
        side_layout.addWidget(self.tree, 1)
        self.splitter.addWidget(self.sidebar)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QFrame.NoFrame)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll.setMinimumWidth(260)
        self.detail_panel = QWidget()
        self.detail_panel.setObjectName("exampleDetailPanel")
        self.detail_panel.setAttribute(Qt.WA_StyledBackground, True)
        detail = QVBoxLayout(self.detail_panel)
        detail.setContentsMargins(12, 8, 12, 10)
        detail.setSpacing(12)
        self.empty_label = self._label("Choose an example to see what it does.")
        detail.addWidget(self.empty_label)
        self.category_label = self._label()
        detail.addWidget(self.category_label)
        self.title_label = self._label()
        title_font = self.font()
        self._title_size = max(16, title_font.pointSizeF() * 1.6)
        title_font.setPointSizeF(self._title_size)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        detail.addWidget(self.title_label)
        self.details_label = self._label()
        detail.addWidget(self.details_label)
        self.data_panel, self.data_label = self._section("Input data", detail)
        self.method_panel, method_layout = self._panel(detail)
        self.method_name_label = self._label(bold=True)
        self.method_description_label = self._label()
        self.method_meaning_label = self._label()
        self.paper_link = self._label()
        self.paper_link.setTextFormat(Qt.RichText)
        self.paper_link.setOpenExternalLinks(False)
        self.paper_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.paper_link.linkActivated.connect(self._open_reference_link)
        self.paper_journal_label = self._label()
        for widget in (
            self.method_name_label,
            self.method_description_label,
            self.method_meaning_label,
            self.paper_link,
            self.paper_journal_label,
        ):
            method_layout.addWidget(widget)
        self.explore_panel, self.explore_label = self._section(
            "What to explore", detail, bullets=True
        )
        self.results_panel, self.results_label = self._section(
            "What you'll get", detail, bullets=True
        )
        self.try_panel, self.try_label = self._section("Try this", detail)
        self.try_panel.setObjectName("exampleTryPanel")
        self.try_panel.layout().setContentsMargins(10, 8, 10, 8)
        self.caution_label = self._label()
        detail.addWidget(self.caution_label)
        detail.addStretch(1)
        self.details_scroll.setWidget(self.detail_panel)
        self.splitter.addWidget(self.details_scroll)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([330, 660])
        layout.addWidget(self.splitter, 1)

        footer = QHBoxLayout()
        self.footer_hint = self._label()
        footer.addWidget(self.footer_hint, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        self.buttons.setLayoutDirection(Qt.LeftToRight)
        self.open_button = self.buttons.addButton(
            "Open example", QDialogButtonBox.AcceptRole
        )
        self.open_button.setDefault(True)
        footer.addWidget(self.buttons)
        layout.addLayout(footer)
        self.filter_edit.textChanged.connect(self._populate_tree)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.open_button.clicked.connect(self._accept_if_selected)
        self.buttons.rejected.connect(self.reject)
        self._apply_theme()
        self._populate_tree()
        self.filter_edit.setFocus()

    @staticmethod
    def _label(text="", *, bold=False):
        label = QLabel(text)
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        label.setIndent(0)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if bold:
            font = label.font()
            font.setBold(True)
            label.setFont(font)
        return label

    @staticmethod
    def _panel(layout):
        panel = QWidget()
        panel.setProperty("exampleSection", True)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        layout.addWidget(panel)
        return panel, panel_layout

    def _section(self, title, layout, *, bullets=False):
        panel, panel_layout = self._panel(layout)
        panel_layout.addWidget(self._label(title, bold=True))
        label = self._label()
        if bullets:
            label.setTextFormat(Qt.RichText)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            label.setAccessibleName(title)
        panel_layout.addWidget(label)
        return panel, label

    @staticmethod
    def _bullet_html(items):
        # Real list items give wrapped lines a hanging indent. Escape all copy:
        # guidance is text, never markup, links or embedded image instructions.
        if not items:
            return ""
        body = "".join(
            f'<li style="margin-bottom: {6 if i < len(items) - 1 else 0}px;">'
            f"{escape(item)}</li>"
            for i, item in enumerate(items)
        )
        return (
            '<ul style="-qt-list-indent: 0; margin-top: 0px; '
            'margin-bottom: 0px; margin-left: 18px;">'
            f"{body}</ul>"
        )

    def _apply_theme(self):
        colors = theme_colors(self.palette())
        # Base and Window can be identical under napari's global stylesheet.
        # Give both panes explicit fills, including their child viewports.
        sidebar_surface = colors.surface.name()
        detail_surface = blend_colors(colors.surface, colors.text, 0.055).name()
        self.sidebar.setStyleSheet(
            f"QWidget#exampleSidebar {{ background: {sidebar_surface}; }}"
            "QWidget#exampleSidebar QLabel { background: transparent; padding: 0px; }"
        )
        self.splitter.setStyleSheet(
            "QSplitter#exampleBrowserSplitter::handle:horizontal {"
            f" background: {colors.border.name()};"
            f" border-left: 3px solid {sidebar_surface};"
            f" border-right: 3px solid {detail_surface}; }}"
            "QSplitter#exampleBrowserSplitter::handle:vertical {"
            f" background: {colors.border.name()};"
            f" border-top: 3px solid {sidebar_surface};"
            f" border-bottom: 3px solid {detail_surface}; }}"
        )
        search_palette = QPalette(self.filter_edit.palette())
        search_palette.setColor(QPalette.PlaceholderText, colors.muted_text)
        self.filter_edit.setPalette(search_palette)
        self.search_action.setIcon(toolbar_icon("search", self.palette()))
        self.filter_edit.setStyleSheet("QLineEdit { min-height: 28px; padding: 4px; }")
        self.tree.setStyleSheet(
            f"QTreeWidget#exampleTree {{ background: {sidebar_surface};"
            f" color: {colors.text.name()}; border: 0px; }}"
            "QTreeWidget#exampleTree::item:selected {"
            f" background: {colors.info.surface.name()};"
            f" color: {colors.text.name()}; }}"
        )
        self.tree.viewport().setStyleSheet(
            f"QWidget#exampleTreeViewport {{ background: {sidebar_surface}; }}"
        )
        self.details_scroll.viewport().setStyleSheet(f"background: {detail_surface};")
        self.detail_panel.setStyleSheet(
            f"QWidget#exampleDetailPanel {{ background: {detail_surface}; }}"
            'QWidget[exampleSection="true"] { background: transparent; }'
            f"QWidget#exampleDetailPanel QLabel {{ color: {colors.text.name()};"
            " background: transparent; padding: 0px; }"
            f"QWidget#exampleTryPanel {{ background: {colors.info.surface.name()};"
            f" border-left: 3px solid {colors.info.accent.name()}; }}"
        )
        for label in (
            self.count_label,
            self.footer_hint,
            self.caution_label,
            self.paper_journal_label,
        ):
            label.setStyleSheet(f"color: {colors.muted_text.name()};")
        self.category_label.setStyleSheet(f"color: {colors.info.foreground.name()};")
        # Napari's inherited QLabel font rule overrides setFont alone.
        self.title_label.setStyleSheet(
            f"font-size: {self._title_size}pt; font-weight: bold;"
        )
        self.open_button.setStyleSheet(
            f"QPushButton {{ border: 1px solid {colors.info.accent.name()};"
            " padding: 7px 14px; font-weight: bold; }"
        )
        # Inherited pane styles can reset the placeholder role during polish.
        self.filter_edit.ensurePolished()
        self.filter_edit.setPalette(search_palette)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange) and hasattr(
            self, "open_button"
        ):
            self._apply_theme()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "splitter"):
            orientation = Qt.Vertical if self.width() < 780 else Qt.Horizontal
            if self.splitter.orientation() != orientation:
                self.splitter.setOrientation(orientation)
                self.splitter.setSizes(
                    [200, 450] if orientation == Qt.Vertical else [330, 660]
                )

    def eventFilter(self, obj, event):  # noqa: N802
        if (
            event.type() == QEvent.KeyPress
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and obj in (self.filter_edit, self.tree)
        ):
            self._accept_if_selected()
            return True
        if (
            event.type() == QEvent.Resize
            and hasattr(self, "tree")
            and obj is self.tree.viewport()
        ):
            self.tree.scheduleDelayedItemsLayout()
        return super().eventFilter(obj, event)

    @staticmethod
    def _category(spec):
        return (
            spec.guidance.chooser_category if spec.guidance else ""
        ) or spec.category

    def selected_example(self):
        return self._example_by_id(self._selected_example_id)

    def _example_by_id(self, example_id):
        return next((spec for spec in self._examples if spec.id == example_id), None)

    def select_example(self, example_id):
        for index in range(self.tree.topLevelItemCount()):
            category = self.tree.topLevelItem(index)
            for child_index in range(category.childCount()):
                child = category.child(child_index)
                if child.data(0, Qt.UserRole) == str(example_id):
                    category.setExpanded(True)
                    self.tree.setCurrentItem(child)
                    self.tree.scrollToItem(child)
                    return

    def _populate_tree(self):
        selected_id = self._selected_example_id
        query = _normalize_search_text(self.filter_edit.text())
        categories = {}
        for spec in self._examples:
            if not query or self._matches_query(spec, query):
                categories.setdefault(self._category(spec), []).append(spec)
        order = [name for name in _CATEGORY_ORDER if name in categories]
        order += [
            name
            for name in categories
            if name not in order and name != _DEVELOPER_CATEGORY
        ]
        if _DEVELOPER_CATEGORY in categories:
            order.append(_DEVELOPER_CATEGORY)
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            visible_ids = []
            for name in order:
                specs = categories[name]
                category = QTreeWidgetItem([f"{name} · {len(specs)}"])
                category.setFlags(category.flags() & ~Qt.ItemIsSelectable)
                font = category.font(0)
                font.setBold(True)
                category.setFont(0, font)
                self.tree.addTopLevelItem(category)
                for spec in specs:
                    item = QTreeWidgetItem([spec.title])
                    item.setData(0, Qt.UserRole, spec.id)
                    item.setToolTip(
                        0, spec.guidance.purpose if spec.guidance else spec.description
                    )
                    category.addChild(item)
                    visible_ids.append(spec.id)
                category.setExpanded(bool(query))
            target = (
                selected_id
                if selected_id in visible_ids
                else next(iter(visible_ids), "")
            )
            self.select_example(target)
        finally:
            self.tree.blockSignals(False)
        count = len(visible_ids)
        self.count_label.setText(
            f"{count} of {len(self._examples)} examples"
            if query
            else f"{count} examples"
        )
        self._on_selection_changed()

    @staticmethod
    def _matches_query(spec, query):
        parts = [
            spec.id,
            spec.category,
            spec.title,
            spec.filename,
            *spec.samples,
            spec.description,
        ]
        if spec.guidance:
            for field in fields(spec.guidance):
                value = getattr(spec.guidance, field.name)
                parts.extend((value,) if isinstance(value, str) else value)
        text = _normalize_search_text(" ".join(parts))
        return all(word in text for word in query.split())

    def _on_selection_changed(self):
        item = self.tree.currentItem()
        spec = self._example_by_id(item.data(0, Qt.UserRole) if item else "")
        self._selected_example_id = spec.id if spec else ""
        self.open_button.setEnabled(spec is not None)
        self.empty_label.setVisible(spec is None)
        self.empty_label.setText(
            "No examples match your search. Try another word or clear the search."
            if self.filter_edit.text().strip()
            else "Choose an example to see what it does."
        )
        for widget in (
            self.category_label,
            self.title_label,
            self.details_label,
            self.data_panel,
            self.method_panel,
            self.explore_panel,
            self.results_panel,
            self.try_panel,
            self.caution_label,
        ):
            widget.setVisible(spec is not None)
        self.open_button.setText(
            "Open batch demo..."
            if spec and spec.generated_batch_demo
            else "Open example"
        )
        self.footer_hint.setText(
            "Choose a folder for the demo working copy"
            if spec and spec.generated_batch_demo
            else "Opens in a new workflow tab"
            if spec
            else "Choose an example to continue"
        )
        if not spec:
            return
        guidance = spec.guidance
        self.category_label.setText(self._category(spec))
        self.title_label.setText(spec.title)
        self.details_label.setText(guidance.purpose if guidance else spec.description)
        self.data_label.setText(guidance.data if guidance else ", ".join(spec.samples))
        for name, panel, label in (
            ("explore", self.explore_panel, self.explore_label),
            ("results", self.results_panel, self.results_label),
        ):
            items = getattr(guidance, name, ())
            label.setText(self._bullet_html(items))
            panel.setVisible(bool(items))
        self.try_label.setText(getattr(guidance, "try_this", ""))
        self.try_panel.setVisible(bool(self.try_label.text()))
        self.caution_label.setText(getattr(guidance, "caution", ""))
        self.caution_label.setVisible(bool(self.caution_label.text()))
        self.method_panel.setVisible(bool(guidance and guidance.method_name))
        if guidance and guidance.method_name:
            self.method_name_label.setText(guidance.method_name)
            self.method_description_label.setText(guidance.method_description)
            self.method_meaning_label.setText(guidance.method_meaning)
            url = QUrl(guidance.paper_url)
            valid = url.isValid() and url.scheme() == "https" and bool(url.host())
            self.paper_link.setText(
                f'<a href="{escape(guidance.paper_url, quote=True)}">'
                f"Read the paper: {escape(guidance.paper_authors)}</a>"
                if valid
                else escape(guidance.paper_authors)
            )
            self.paper_link.setToolTip("Open the original paper in your web browser")
            self.paper_journal_label.setText(guidance.paper_journal)
        self.details_scroll.verticalScrollBar().setValue(0)

    @staticmethod
    def _open_reference_link(link):
        url = QUrl(link)
        if url.isValid() and url.scheme() == "https" and url.host():
            QDesktopServices.openUrl(url)

    def _on_item_double_clicked(self, item, _column):
        if item.data(0, Qt.UserRole):
            self.tree.setCurrentItem(item)
            self._accept_if_selected()

    def _accept_if_selected(self):
        if self.selected_example() is not None:
            self.accept()
