"""Integrated, node-bound views of existing measurement/summary/plot outputs.

This module does not calculate statistics or mutate a graph. All authored
changes are signalled to the workspace controller; the standard node controls,
table workers and figure exporter remain the owners of their respective UIs.
"""

from __future__ import annotations

from html import escape

from qtpy.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QSize,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from qtpy.QtGui import QFont, QFontMetrics, QIcon, QPainter, QPalette, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyleOptionButton,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.result_plots import (
    PlotRecipe,
    is_summary_table,
    measurement_label,
)
from napari_vipp.core.statistics import StatisticsRecipe
from napari_vipp.core.tables import TableData
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.iconography import interface_icon, palette_branch_color
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.plot_notice_state import (
    PlotNoticeState,
    plot_notes_are_caution,
    visible_plot_warnings,
)
from napari_vipp.ui.result_plots import (
    PlotExportDialog,
    PlotRecipeControls,
    ResultPlotCanvas,
    _PlotWarningLabel,
    _result_summary,
)
from napari_vipp.ui.result_table_dialog import ResultTableModel, ResultTablePanel
from napari_vipp.ui.statistics import (
    StatisticsPanel,
    statistics_preview_columns,
    statistics_preview_header,
)
from napari_vipp.ui.toolbar_controls import toolbar_icon


def _plain_tooltip(text: str) -> str:
    """Prevent Qt rich-text detection from interpreting authored labels.

    Ordinary tooltips retain their existing plain strings. If text contains
    markup delimiters, use explicitly escaped HTML and preserve line breaks.
    """
    if "<" not in text:
        return text
    return "<qt>" + escape(text).replace("\n", "<br>") + "</qt>"


def _label(text="", *, bold=False):
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


def _combo(name):
    combo = QComboBox()
    combo.setAccessibleName(name)
    combo.setMinimumWidth(0)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    return combo


class _SummaryTableModel(ResultTableModel):
    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if (
            role == Qt.DisplayRole
            and orientation == Qt.Horizontal
            and 0 <= section < self.table.column_count
        ):
            return statistics_preview_header(self.table, self.table.columns[section])
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            if self.raw_value(index.row(), index.column()) is None:
                return "—"
        return super().data(index, role)


class _WorkspaceTabBar(QTabBar):
    """Wide navigation tabs with the icon and label centered together."""

    def paintEvent(self, event):  # noqa: N802
        painter = QStylePainter(self)
        for index in range(self.count()):
            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.CE_TabBarTabShape, option)
            font = QFont(self.font())
            font.setBold(index == self.currentIndex())
            width = min(
                option.rect.width(),
                QFontMetrics(font).horizontalAdvance(option.text)
                + option.iconSize.width()
                + 26,
            )
            option.rect.setLeft(option.rect.center().x() - width // 2)
            option.rect.setWidth(width)
            painter.drawControl(QStyle.CE_TabBarTabLabel, option)


class _ConnectionArrow(QWidget):
    """A prominent, font-independent arrow aligned with the selector fields."""

    def __init__(self, target, parent=None):
        super().__init__(parent)
        self._target = target
        self.setFixedWidth(30)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setAccessibleName(f"Feeds into {target.accessibleName()}")
        self.setToolTip("Follow the connected data from left to right.")

    def arrowCenter(self):  # noqa: N802 - Qt geometry convention
        target_center = self._target.mapToGlobal(self._target.rect().center())
        return QPoint(self.width() // 2, self.mapFromGlobal(target_center).y())

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(theme_colors(self.palette()).info.accent, 3.0)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        center = self.arrowCenter()
        x, y = center.x(), center.y()
        painter.drawLine(QPointF(x - 10, y), QPointF(x + 9, y))
        painter.drawPolyline(
            [QPointF(x + 3, y - 6), QPointF(x + 9, y), QPointF(x + 3, y + 6)]
        )


class _InputDetailsButton(QPushButton):
    """An accessible disclosure header with its outline chevron at the right."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setAutoDefault(False)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    def chevronRect(self):  # noqa: N802 - Qt geometry convention
        size = self.iconSize()
        return QRect(
            self.width() - size.width() - 8,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )

    def paintEvent(self, event):  # noqa: N802
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        # Keep native hover/focus/keyboard feedback, but do not let the platform
        # put the icon beside the text or flip the text for a right-side glyph.
        option.text = ""
        option.icon = QIcon()
        painter.drawControl(QStyle.CE_PushButton, option)
        glyph = self.chevronRect()
        text_rect = self.rect().adjusted(8, 0, -8, 0)
        text_rect.setRight(glyph.left() - 8)
        painter.setPen(self.palette().color(QPalette.ButtonText))
        painter.drawText(
            text_rect,
            Qt.AlignLeft | Qt.AlignVCenter,
            self.fontMetrics().elidedText(
                self.text(), Qt.ElideRight, text_rect.width()
            ),
        )
        self.icon().paint(
            painter,
            glyph,
            Qt.AlignCenter,
            QIcon.Normal if self.isEnabled() else QIcon.Disabled,
        )


class _WorkspacePlotNoteText(_PlotWarningLabel):
    """Wrapped note text; its enclosing card owns the semantic background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMargin(0)

    def refresh_theme(self):
        if self._styling:
            return
        self._styling = True
        try:
            owner = self.parentWidget()
            colors = theme_colors(owner.palette() if owner else self.palette())
            style = (
                "QLabel { background: transparent; border: 0; padding: 0; "
                f"color: {colors.text.name()}; }}"
            )
            if self.styleSheet() != style:
                self.setStyleSheet(style)
        finally:
            self._styling = False
        self._fit_height()


class ResultsWorkspaceDialog(QDialog):
    """Reusable editors around real graph nodes, never a second analysis engine."""

    workflow_selected = Signal(str)
    data_selected = Signal(object)
    summary_selected = Signal(str)
    plot_selected = Signal(str)
    plot_scope_selected = Signal(str)
    add_summary_requested = Signal()
    add_plot_requested = Signal(str)
    summary_params_changed = Signal(dict)
    summary_upgrade_requested = Signal()
    plot_params_changed = Signal(dict)
    plot_source_changed = Signal(str)
    recalculate_requested = Signal(str)
    show_node_requested = Signal(str)
    export_completed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VippResultsWorkspace")
        self.setWindowTitle("Results Workspace — VIPP")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_WindowPropagation, True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(760, 520)
        self.resize(1200, 800)
        self._updating = False
        self._available = True
        self._workflow_selection_enabled = True
        self._has_workflow_choices = False
        self._has_data_choices = False
        self._workflow_activation_timer = QTimer(self)
        self._workflow_activation_timer.setSingleShot(True)
        self._data_node_id = ""
        self._data_title = "Input measurements"
        self._data_table = None
        self._data_current = False
        self._summary_current = False
        self._plot_current = False
        self._summary_bound = False
        self._plot_bound = False
        self._plot_unavailable = False
        self._summary_editable = True
        self._plot_editable = True
        self._plot_result = None
        self._plot_setup_message = ""
        self._plot_protected_paths = ()
        self._plot_context = ""
        self._plot_notice_state = PlotNoticeState()
        self._plot_notice_warnings = ()
        self._plot_notes_transient = False
        self._busy = False
        self._busy_text = "Updating results…"
        self._theme_refresh_in_progress = True
        self._theme_key = None
        self._control_surfaces = []
        self._right_scrolls = []
        self._context_labels = []
        self._show_node_buttons = []
        self.theme_timer = QTimer(self)
        self.theme_timer.setSingleShot(True)
        self.theme_timer.timeout.connect(self.refresh_theme)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        title = _label("Results Workspace", bold=True)
        font = title.font()
        font.setPointSize(font.pointSize() + 3)
        title.setFont(font)
        heading_row = QHBoxLayout()
        heading_row.addWidget(title, 1)
        self.source_label = _label("Connect a measurement table to get started.")
        self.source_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        heading_row.addWidget(self.source_label, 1)
        layout.addLayout(heading_row)
        self._build_connection_bar(layout)
        # The selectors already show the selected path. Retain its full text for
        # assistive technology and tooltips, without a second visible breadcrumb.
        self.connection_label = _label()
        self.connection_label.setParent(self)
        self.connection_label.setAccessibleName("Selected data connections")
        self.connection_label.hide()
        self.availability_label = _label()
        self.availability_label.hide()
        layout.addWidget(self.availability_label)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("ResultsWorkspaceTabs")
        self.tabs.setTabBar(_WorkspaceTabBar())
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setIconSize(QSize(18, 18))
        self.tabs.setAccessibleName("Results workspace views")
        layout.addWidget(self.tabs, 1)
        self._build_data()
        self._build_summary()
        self._build_plot()
        self.data_panel.export_guard = lambda: (
            self._available and self._data_current and not self._busy
        )
        self.summary_panel.export_guard = lambda: (
            self._available and self._summary_current and not self._busy
        )
        self.plotted_panel.export_guard = lambda: (
            self._available and self._plot_current and not self._busy
        )

        # The reserved slot never changes the table/canvas geometry on busy/idle.
        self.busy_slot = QWidget(self)
        busy_layout = QHBoxLayout(self.busy_slot)
        busy_layout.setContentsMargins(0, 0, 0, 0)
        self.busy_label = _label("Updating results…")
        self.busy_label.setWordWrap(False)
        self.progress = QProgressBar(self.busy_slot)
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(140, 10)
        for child in (self.busy_label, self.progress):
            policy = child.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            child.setSizePolicy(policy)
            child.hide()
        busy_layout.addWidget(self.busy_label, 1)
        busy_layout.addWidget(self.progress)
        layout.addWidget(self.busy_slot)

        self.footer = QHBoxLayout()
        self.recalculate_button = QPushButton("Calculate", self)
        self.recalculate_button.clicked.connect(
            lambda: self.recalculate_requested.emit(self.current_node_id())
        )
        self.footer.addWidget(self.recalculate_button)
        self.footer.addStretch(1)
        self.export_button = QPushButton("Export table…", self)
        self.export_button.clicked.connect(self.request_export)
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        add_dialog_buttons(
            self.footer, actions=(self.export_button,), dismiss=self.close_button
        )
        layout.addLayout(self.footer)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.tabs.currentChanged.connect(self._refresh_actions)
        self.set_workflows((), "")
        self.set_choices()
        self.set_data(None)
        self.set_summary()
        self.set_plot()
        self._theme_refresh_in_progress = False
        self.refresh_theme()

    def _build_connection_bar(self, layout):
        self.connection_bar = QFrame(self)
        self.connection_bar.setObjectName("WorkspaceConnections")
        self.connection_bar.setAccessibleName(
            "Workflow, data, statistics and plot connections"
        )
        row = QHBoxLayout(self.connection_bar)
        row.setContentsMargins(10, 8, 10, 10)
        row.setSpacing(10)
        self.workflow_selector = _combo("Workflow")
        self.data_selector = _combo("Data source")
        self.summary_selector = _combo("Statistics node")
        self.plot_selector = _combo("Plot")
        self.workflow_selector.setToolTip(
            "Choose an open workflow, then select one of its table sources. "
            "Switching workflows does not change node connections or data."
        )
        self.data_selector.setToolTip(
            "Browse a different table branch in this workflow. This changes "
            "the workspace view only; it does not change node connections or data."
        )
        self.summary_selector.setToolTip(
            "Choose a Statistics node connected to this data source, or select "
            "None — use input data to plot the input measurements directly. "
            "This selection applies to every tab and does not change connections."
        )
        self.plot_selector.setToolTip(
            "Choose a plot connected to the selected Statistics node, or directly "
            "to the data source when Statistics node is None — use input data."
        )
        self.add_summary_button = QPushButton("+")
        self.add_summary_button.setAccessibleName("Add summary")
        self.add_summary_button.setToolTip(
            "Add a Statistics node connected to this data source. "
            "Its settings and output are also available in the workflow."
        )
        self.add_plot_button = QPushButton("+")
        self.add_plot_button.setAccessibleName("Add plot")
        self.add_plot_button.setToolTip(
            "Add a Plot Results node connected to the selected Statistics node, "
            "or to the data source when Statistics node is None — use input data."
        )
        self.connection_groups = []
        self.connection_labels = []
        self.connection_arrows = []
        for index, (caption, combo, button) in enumerate(
            (
                ("Workflow", self.workflow_selector, None),
                ("Data source", self.data_selector, None),
                ("Statistics node", self.summary_selector, self.add_summary_button),
                ("Plot", self.plot_selector, self.add_plot_button),
            )
        ):
            if index:
                arrow = _ConnectionArrow(combo, self.connection_bar)
                self.connection_arrows.append(arrow)
                row.addWidget(arrow)
            group = QWidget(self.connection_bar)
            group.setMinimumWidth(0)
            group.setMaximumWidth(340 if button is None else 378)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            label = _label(caption, bold=True)
            label.setWordWrap(False)
            self.connection_labels.append(label)
            group_layout.addWidget(label)
            if combo is self.data_selector:
                self.data_source_label = label
            fields = QHBoxLayout()
            fields.setSpacing(5)
            combo.setMaximumWidth(340)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            fields.addWidget(combo, 1)
            if button is not None:
                button.setFixedWidth(28)
                fields.addWidget(button)
            group_layout.addLayout(fields)
            self.connection_groups.append(group)
            row.addWidget(group, 1)
        row.addStretch(0)
        layout.addWidget(self.connection_bar)
        self.workflow_selector.currentIndexChanged.connect(self._workflow_selected)
        self.workflow_selector.activated.connect(self._workflow_activated)
        self.data_selector.currentIndexChanged.connect(self._data_selected)
        self.summary_selector.currentIndexChanged.connect(self._summary_selected)
        self.plot_selector.currentIndexChanged.connect(self._plot_selected)
        self.add_summary_button.clicked.connect(self.add_summary_requested.emit)
        self.add_plot_button.clicked.connect(
            lambda: self.add_plot_requested.emit(
                str(self.summary_selector.currentData() or self._data_node_id)
            )
        )

    def _page(self, name, *, fit_plot=False):
        page = QWidget(self)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        context = QFrame(page)
        context.setObjectName("WorkspaceEditingContext")
        context_layout = QHBoxLayout(context)
        context_layout.setContentsMargins(10, 10, 10, 10)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        heading = _label(bold=True)
        detail = _label()
        text_layout.addWidget(heading)
        text_layout.addWidget(detail)
        context_layout.addLayout(text_layout, 1)
        show = QPushButton("Show node", context)
        show.setToolTip(
            "Return to this node in the workflow. Reopen Results Workspace "
            "from the inspector to continue here."
        )
        show.clicked.connect(
            lambda: self.show_node_requested.emit(self.current_node_id())
        )
        context_layout.addWidget(show)
        self._context_labels.append((heading, detail))
        self._show_node_buttons.append(show)
        page_layout.addWidget(context)
        splitter = QSplitter(Qt.Horizontal, page)
        page_layout.addWidget(splitter, 1)
        splitter.setChildrenCollapsible(False)
        controls = QWidget(splitter)
        controls.setObjectName("WorkspaceControls")
        self._control_surfaces.append(controls)
        left = QVBoxLayout(controls)
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(9)
        scroll = QScrollArea(splitter)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(240)
        right_widget = QWidget()
        right_widget.setMinimumWidth(0)
        right_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        right = QVBoxLayout(right_widget)
        right.setSizeConstraint(
            QLayout.SetNoConstraint if fit_plot else QLayout.SetMinimumSize
        )
        right.setContentsMargins(10, 10, 0, 0)
        right.setSpacing(8)
        splitter.addWidget(scroll)
        if fit_plot:
            # A figure belongs to the viewport, not a document scroll area.
            # Matplotlib's preferred figure size must not extend the page.
            self.plot_area = right_widget
            self.plot_area.setMinimumSize(0, 0)
            self.plot_area.installEventFilter(self)
            splitter.addWidget(right_widget)
        else:
            right_scroll = QScrollArea(splitter)
            right_scroll.setWidgetResizable(True)
            right_scroll.setFrameShape(QFrame.NoFrame)
            right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            right_scroll.setWidget(right_widget)
            self._right_scrolls.append(right_scroll)
            splitter.addWidget(right_scroll)
        splitter.setSizes([310, 850])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.tabs.addTab(page, name)
        return left, right

    @property
    def show_node_button(self):
        return self._show_node_buttons[self.tabs.currentIndex()]

    def _table_panel(self, layout):
        panel = ResultTablePanel(self)
        panel.setMinimumHeight(180)
        panel.layout().setContentsMargins(0, 0, 0, 0)
        panel.export_button.hide()
        panel.exportCompleted.connect(self.export_completed.emit)
        panel.recalculationRequested.connect(
            lambda: self.recalculate_requested.emit(self.current_node_id())
        )
        layout.addWidget(panel, 1)
        return panel

    def _build_data(self):
        left, right = self._page("Data")
        left.addWidget(_label("Input measurements", bold=True))
        self.data_description = _label()
        left.addWidget(self.data_description)
        left.addWidget(_label("Find rows in this view"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search all fields…")
        self.search.setToolTip(
            "Search only changes the visible rows. Summaries, plots and exported "
            "tables still use the complete connected data."
        )
        left.addWidget(self.search)
        left.addWidget(_label("Visible columns"))
        column_actions = QHBoxLayout()
        column_actions.setSpacing(6)
        self.select_all_columns_button = QPushButton("Select all")
        self.select_no_columns_button = QPushButton("Select none")
        for button, visible in (
            (self.select_all_columns_button, True),
            (self.select_no_columns_button, False),
        ):
            button.setAccessibleName(
                "Select all columns" if visible else "Deselect all columns"
            )
            button.setToolTip(
                ("Show" if visible else "Hide")
                + " every column in this view. Calculations and exports are unchanged."
            )
            button.clicked.connect(
                lambda _checked=False, show=visible: self._set_all_columns_visible(show)
            )
            column_actions.addWidget(button)
        column_actions.addStretch(1)
        left.addLayout(column_actions)
        self.column_list = QListWidget()
        self.column_list.setAccessibleName("Columns visible in the data view")
        left.addWidget(self.column_list, 1)
        left.addWidget(
            _label(
                "Search, column visibility and sorting change this view only. "
                "Use upstream table nodes to select or combine data for analysis."
            )
        )
        self.data_view_note = _label()
        right.addWidget(self.data_view_note)
        self.data_panel = self._table_panel(right)
        self.search_proxy = QSortFilterProxyModel(self.data_panel)
        self.search_proxy.setSourceModel(self.data_panel.model)
        self.search_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.search_proxy.setFilterKeyColumn(-1)
        self.data_panel.table_view.setModel(self.search_proxy)
        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(180)
        self.search.textChanged.connect(lambda: self.search_timer.start())
        self.search_timer.timeout.connect(self._search_changed)
        self.column_list.itemChanged.connect(self._column_changed)

    def _build_summary(self):
        left, right = self._page("Summary")
        self.summary_empty = _label(
            "No statistics node is selected. Choose one in the connection bar "
            "above, or use + beside Statistics node to add a summary."
        )
        left.addWidget(self.summary_empty)
        self.statistics_panel = StatisticsPanel(self)
        left.addWidget(self.statistics_panel)
        left.addStretch(1)
        self.statistics_panel.layout().removeWidget(self.statistics_panel.result_group)
        right.addWidget(self.statistics_panel.result_group)
        self.summary_evidence = QCheckBox("Show all fields")
        self.summary_evidence.setToolTip(
            "Show every field in the summary table. Export always includes this "
            "complete table, even when the compact view is shown."
        )
        table_heading = QHBoxLayout()
        table_heading.addWidget(_label("Results table", bold=True), 1)
        table_heading.addWidget(self.summary_evidence)
        right.addLayout(table_heading)
        self.summary_panel = self._table_panel(right)
        old = self.summary_panel.model
        self.summary_panel.model = _SummaryTableModel(parent=self.summary_panel)
        self.summary_panel.table_view.setModel(self.summary_panel.model)
        old.deleteLater()
        self.summary_evidence.toggled.connect(self._summary_columns)
        self.statistics_panel.params_changed.connect(self._summary_edited)
        self.statistics_panel.upgrade_requested.connect(
            self.summary_upgrade_requested.emit
        )

    def _build_plot(self):
        left, right = self._page("Plots", fit_plot=True)
        # Compatibility-only state for callers that still supply plot_scopes.
        # The visible Statistics selector now owns this choice for every tab.
        self.plot_scope_label = _label("Plots for")
        self.plot_scope_label.setParent(self)
        self.plot_scope_label.hide()
        self.plot_scope = _combo("Plots for")
        self.plot_scope.setParent(self)
        self.plot_scope.hide()
        self.plot_connection_button = QPushButton("Change plot input…")
        self.plot_connection_button.setCheckable(True)
        self.plot_connection_button.setToolTip(
            "Advanced: change this plot's connection in the workflow. "
            "Use the connection bar above to browse existing nodes instead."
        )
        self.plot_connection_editor = QWidget()
        editor_layout = QVBoxLayout(self.plot_connection_editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(6)
        editor_layout.addWidget(
            _label("Changing this input reconnects the plot node in the workflow.")
        )
        self.plot_source_label = _label("Change plot input")
        editor_layout.addWidget(self.plot_source_label)
        self.plot_source = _combo("Change plot input")
        self.plot_source.setToolTip(
            "This changes the selected Plot Results node's workflow connection. "
            "Original measurements and summary rows are different data. "
            "Changing the source preserves your saved column selections."
        )
        editor_layout.addWidget(self.plot_source)
        self.plot_connection_editor.hide()
        self.plot_connection_button.toggled.connect(
            self.plot_connection_editor.setVisible
        )
        self.plot_empty = _label(
            "Add a plot to explore the original measurements or a summary. "
            "Each plot keeps its own saved settings."
        )
        left.addWidget(self.plot_empty)
        self.plot_controls = PlotRecipeControls(self)
        left.addWidget(self.plot_controls)
        left.addWidget(self.plot_connection_button)
        left.addWidget(self.plot_connection_editor)
        left.addStretch(1)
        self.plot_info = QScrollArea()
        self.plot_info.setWidgetResizable(True)
        self.plot_info.setFrameShape(QFrame.NoFrame)
        self.plot_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_info.setAccessibleName("Plot input and notices")
        info_content = QWidget()
        self.plot_info_layout = QVBoxLayout(info_content)
        self.plot_info_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_info_layout.setSpacing(6)
        self.plot_info_layout.setAlignment(Qt.AlignTop)
        self.plot_info.setWidget(info_content)
        right.addWidget(self.plot_info)
        self.plot_fit_timer = QTimer(self)
        self.plot_fit_timer.setSingleShot(True)
        self.plot_fit_timer.timeout.connect(self._fit_plot_information)
        info_content.installEventFilter(self)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        self.plot_source_card = QFrame()
        self.plot_source_card.setObjectName("WorkspacePlotSource")
        source_layout = QVBoxLayout(self.plot_source_card)
        source_layout.setContentsMargins(4, 4, 4, 4)
        source_layout.setSpacing(4)
        self.plot_input_toggle = _InputDetailsButton("Input: No input selected")
        self.plot_input_toggle.setObjectName("WorkspacePlotInputToggle")
        self.plot_input_toggle.setAccessibleName("Show input details")
        self.plot_input_toggle.toggled.connect(self._toggle_plot_input)
        source_layout.addWidget(self.plot_input_toggle)
        self.plot_source_note = _label()
        self.plot_source_note.setContentsMargins(8, 0, 8, 5)
        source_layout.addWidget(self.plot_source_note)
        self.plot_source_note.hide()
        input_row.addWidget(self.plot_source_card, 1)
        self.plot_notes_reopen_button = QPushButton("Analysis notes")
        self.plot_notes_reopen_button.setAccessibleName("Show analysis notes")
        self.plot_notes_reopen_button.setToolTip(
            "Show the analysis notes you dismissed for this plot. "
            "Dismissing notes does not change results or exported figures."
        )
        self.plot_notes_reopen_button.clicked.connect(self._reopen_plot_notes)
        self.plot_notes_reopen_button.hide()
        input_row.addWidget(self.plot_notes_reopen_button, 0, Qt.AlignTop)
        self.plot_info_layout.addLayout(input_row)
        self.plot_status = _label()
        self.plot_info_layout.addWidget(self.plot_status)
        self.plot_notes_frame = QFrame(info_content)
        self.plot_notes_frame.setObjectName("WorkspacePlotNotes")
        notes_layout = QVBoxLayout(self.plot_notes_frame)
        notes_layout.setContentsMargins(10, 7, 10, 8)
        notes_layout.setSpacing(4)
        notes_header = QHBoxLayout()
        notes_header.addWidget(_label("Analysis notes", bold=True), 1)
        self.plot_notes_dismiss_button = QPushButton("Dismiss")
        self.plot_notes_dismiss_button.setAccessibleName("Dismiss analysis notes")
        self.plot_notes_dismiss_button.setToolTip(
            "Hide these notes for this plot. Use Analysis notes to show them again. "
            "New data or changed analysis settings will show relevant notes again."
        )
        self.plot_notes_dismiss_button.clicked.connect(self._dismiss_plot_notes)
        notes_header.addWidget(self.plot_notes_dismiss_button)
        notes_layout.addLayout(notes_header)
        self.plot_warning = _WorkspacePlotNoteText(self.plot_notes_frame)
        self.plot_warning.layout_changed.connect(lambda: self.plot_fit_timer.start(0))
        notes_layout.addWidget(self.plot_warning)
        self.plot_info_layout.addWidget(self.plot_notes_frame)
        self.plot_notes_frame.hide()
        self.plot_canvas = ResultPlotCanvas(self, fit_viewport=True)
        right.addWidget(self.plot_canvas, 1)
        self.plot_setup_card = QFrame()
        self.plot_setup_card.setObjectName("WorkspacePlotSetup")
        setup_layout = QVBoxLayout(self.plot_setup_card)
        setup_layout.setContentsMargins(16, 14, 16, 14)
        setup_layout.setSpacing(8)
        self.plot_setup_heading = _label("Choose what to plot", bold=True)
        setup_layout.addWidget(self.plot_setup_heading)
        self.plot_setup_note = _label()
        setup_layout.addWidget(self.plot_setup_note)
        self.choose_measurement_button = QPushButton("Choose measurement")
        self.choose_measurement_button.clicked.connect(self._focus_measurement)
        setup_layout.addWidget(self.choose_measurement_button, 0, Qt.AlignLeft)
        right.addWidget(self.plot_setup_card)
        self.plot_setup_space = QWidget()
        right.addWidget(self.plot_setup_space, 1)
        self.plot_setup_card.hide()
        self.plot_setup_space.hide()
        self.point_label = _label("Click a point to identify its source measurement.")
        self.point_label.setWordWrap(False)
        self.point_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.plot_canvas.point_selected.connect(self._set_point_description)
        self.plotted_data_button = QPushButton("View plotted data")
        self.plotted_data_button.setCheckable(True)
        plot_actions = QHBoxLayout()
        plot_actions.addWidget(self.point_label, 1)
        plot_actions.addWidget(self.plotted_data_button)
        right.addLayout(plot_actions)
        self.plotted_panel = self._table_panel(right)
        self.plotted_panel.setMinimumHeight(0)
        self.plotted_panel.setMaximumHeight(180)
        # This auxiliary view shares the chart's viewport. Table headers already
        # identify the fields; repeated table help must not squeeze out the plot.
        self.plotted_panel.summary_label.hide()
        self.plotted_panel.sort_hint.hide()
        self.plotted_panel.export_note.hide()
        self.plotted_panel.table_view.setToolTip(
            "Plotted values. Click a column heading to sort this view. "
            "Sorting does not change the plot or its source measurements."
        )
        self.plotted_panel.hide()
        self.plotted_data_button.toggled.connect(self._toggle_plotted_data)
        self.plot_scope.currentIndexChanged.connect(self._plot_scope_selected)
        self.plot_source.currentIndexChanged.connect(self._source_selected)
        self.plot_controls.params_changed.connect(self._plot_edited)

    def _toggle_plot_input(self, expanded):
        self.plot_source_note.setVisible(bool(expanded))
        self.plot_input_toggle.setAccessibleName(
            "Hide input details" if expanded else "Show input details"
        )
        colors = theme_colors(self.palette())
        icon_palette = QPalette(self.palette())
        icon_palette.setColor(
            QPalette.ButtonText, palette_branch_color("Image Data", self.palette())
        )
        icon_palette.setColor(QPalette.Button, colors.alternate_surface)
        self.plot_input_toggle.setIcon(
            interface_icon(
                "chevron-down" if expanded else "chevron-right", icon_palette, 18
            )
        )
        self.plot_input_toggle.setIconSize(QSize(18, 18))
        self.plot_fit_timer.start(0)

    def _dismiss_plot_notes(self):
        self._plot_notice_state.dismiss()
        self._sync_plot_notes()

    def _reopen_plot_notes(self):
        self._plot_notice_state.reopen()
        self._sync_plot_notes()

    def _sync_plot_notes(self):
        if not hasattr(self, "plot_notes_frame"):
            return
        current = (
            self._plot_current
            and self._plot_bound
            and self._available
            and not self._busy
        )
        has_notes = bool(self._plot_notice_warnings)
        dismissed = self._plot_notice_state.dismissed
        policy = self.plot_notes_frame.sizePolicy()
        policy.setRetainSizeWhenHidden(
            has_notes
            and not dismissed
            and (self._plot_notes_transient or self._busy)
            and self._plot_bound
        )
        self.plot_notes_frame.setSizePolicy(policy)
        self.plot_notes_frame.setVisible(current and has_notes and not dismissed)
        self.plot_warning.setVisible(has_notes)
        self.plot_notes_dismiss_button.setEnabled(current and has_notes)
        self.plot_notes_reopen_button.setText(
            f"Analysis notes ({len(self._plot_notice_warnings)})"
        )
        self.plot_notes_reopen_button.setVisible(current and has_notes and dismissed)
        self.plot_notes_reopen_button.setEnabled(current and has_notes)
        self._refresh_plot_note_theme()
        self.plot_fit_timer.start(0)

    def _refresh_plot_note_theme(self):
        colors = theme_colors(self.palette())
        caution = plot_notes_are_caution(self._plot_notice_warnings)
        self.plot_notes_frame.setProperty("caution", caution)
        surface = colors.warning.surface if caution else colors.alternate_surface
        accent = colors.warning.accent if caution else colors.border
        style = (
            "QFrame#WorkspacePlotNotes {"
            f"background: {surface.name()};"
            f"border-left: 3px solid {accent.name()}; }}"
            "QFrame#WorkspacePlotNotes QLabel { background: transparent; }"
        )
        if self.plot_notes_frame.styleSheet() != style:
            self.plot_notes_frame.setStyleSheet(style)
        self.plot_warning.refresh_theme()

    def _toggle_plotted_data(self, visible):
        # Make room before showing the table, avoiding an intermediate zero-height
        # chart and a failed Matplotlib layout during the next paint.
        self._fit_plot_information()
        self.plotted_panel.setVisible(visible)

    def _set_point_description(self, text):
        self.point_label.setToolTip(text)
        self.point_label.setAccessibleDescription(text)
        self.point_label.setText(
            self.point_label.fontMetrics().elidedText(
                text, Qt.ElideRight, max(40, self.point_label.width())
            )
        )

    def _fit_plot_information(self):
        """Bound explanatory text so even small windows retain a visible chart."""
        width = max(1, self.plot_info.viewport().width())
        content_height = self.plot_info_layout.totalHeightForWidth(width)
        if content_height < 0:
            content_height = self.plot_info_layout.sizeHint().height()
        inspecting_data = self.plotted_data_button.isChecked()
        available = self.plot_area.height()
        budget = (
            max(32, min(round(available * 0.22), available - 360))
            if inspecting_data
            else max(48, round(available * 0.3))
        )
        height = min(content_height + 2, budget)
        if self.plot_info.height() != height:
            self.plot_info.setFixedHeight(height)
        table_height = max(96, min(180, round(available * 0.25)))
        if self.plotted_panel.maximumHeight() != table_height:
            self.plotted_panel.setMaximumHeight(table_height)
        if self.point_label.toolTip():
            self._set_point_description(self.point_label.toolTip())

    def _focus_measurement(self):
        """Bring the required field into view even in a small window."""
        control = self.plot_controls.controls["y_column"]
        if control.currentData() not in (None, "", "auto"):
            control = self.plot_controls.controls["x_column"]
        parent = control.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            parent.ensureWidgetVisible(control)
        control.setFocus(Qt.OtherFocusReason)
        control.showPopup()

    @staticmethod
    def _choices(combo, choices, selected, empty, *, tooltips=None):
        expected = list(choices)
        if not expected:
            expected = [("", empty)]
        if selected and selected not in [item[0] for item in expected]:
            expected.append((selected, f"Unavailable: {selected}"))
        old = [(combo.itemData(i), combo.itemText(i)) for i in range(combo.count())]
        if old != expected:
            combo.clear()
            for value, label in expected:
                combo.addItem(label, value)
        # An alias can stay unchanged while its calculation settings change.
        # Refresh descriptions even when the visible choices did not change.
        for index, (value, label) in enumerate(expected):
            description = tooltips.get(value, label) if tooltips else label
            combo.setItemData(index, description, Qt.AccessibleDescriptionRole)
            combo.setItemData(index, _plain_tooltip(description), Qt.ToolTipRole)
        # Qt's QVariant comparison does not reliably find Python tuple user data
        # (table outputs use a node/port pair). Compare the Python values instead.
        selected_index = next(
            (
                index
                for index in range(combo.count())
                if combo.itemData(index) == selected
            ),
            -1,
        )
        combo.setCurrentIndex(selected_index if selected else 0)
        ResultsWorkspaceDialog._choice_tooltip(combo)

    @staticmethod
    def _choice_tooltip(combo):
        if not hasattr(combo, "_workspace_choice_help"):
            combo._workspace_choice_help = combo.toolTip()
        description = (
            combo.currentData(Qt.AccessibleDescriptionRole) or combo.currentText()
        )
        combo.setToolTip(
            _plain_tooltip(
                f"Selected: {description}\n\n{combo._workspace_choice_help}".strip()
            )
        )

    def set_choices(
        self,
        *,
        summaries=(),
        plots=(),
        summary_id="",
        plot_id="",
        plot_sources=(),
        plot_source_id="",
        data_sources=(),
        data_source=None,
        plot_scopes=(),
        plot_scope_id="",
        choice_tooltips=None,
    ):
        self._updating = True
        try:
            self._has_data_choices = any(choice[0] for choice in data_sources)
            self._summary_bound = bool(summary_id) and summary_id in {
                choice[0] for choice in summaries
            }
            self._plot_bound = bool(plot_id) and plot_id in {
                choice[0] for choice in plots
            }
            self._plot_unavailable = bool(plot_id) and not self._plot_bound
            self._choices(
                self.data_selector,
                data_sources,
                data_source,
                "No table selected",
                tooltips=choice_tooltips,
            )
            self._choices(
                self.plot_scope,
                plot_scopes,
                plot_scope_id,
                "No input table",
                tooltips=choice_tooltips,
            )
            self._choices(
                self.summary_selector,
                [("", "None — use input data"), *summaries],
                summary_id,
                "None — use input data",
                tooltips=choice_tooltips,
            )
            self._choices(
                self.plot_selector,
                plots,
                plot_id,
                "No plot yet",
                tooltips=choice_tooltips,
            )
            self._choices(
                self.plot_source,
                plot_sources,
                plot_source_id,
                "No input table",
                tooltips=choice_tooltips,
            )
            self.statistics_panel.setVisible(bool(summary_id))
            self.statistics_panel.result_group.setVisible(bool(summary_id))
            self.summary_empty.setVisible(not summary_id)
            self.plot_controls.setVisible(self._plot_bound)
            self.plot_empty.setVisible(not self._plot_bound)
            self.plot_connection_button.setVisible(self._plot_bound)
            if not self._plot_bound:
                self.plot_connection_button.setChecked(False)
        finally:
            self._updating = False
        self._sync_plot_context()
        self._refresh_actions()

    def set_workflows(self, workflows, workflow_id, *, enabled=True):
        """List open sessions without emitting a user navigation request."""
        updating = self._updating
        self._updating = True
        try:
            self._has_workflow_choices = any(choice[0] for choice in workflows)
            self._workflow_selection_enabled = bool(enabled)
            self._choices(
                self.workflow_selector, workflows, workflow_id, "No open workflows"
            )
        finally:
            self._updating = updating
        self._refresh_actions()

    def set_relationship(self, text, *, plot_context="", has_plot=True):
        """Describe graph connections without inferring or changing them."""
        self.connection_label.setText(text)
        self.connection_label.setToolTip(_plain_tooltip(text))
        self.connection_bar.setToolTip(_plain_tooltip(text))
        self.connection_bar.setAccessibleDescription(text)
        self._plot_context = plot_context
        self._sync_plot_context()

    def _sync_plot_context(self):
        if not hasattr(self, "plot_setup_card"):
            return
        empty = not self._plot_bound
        context = self._plot_context or (
            self.summary_selector.currentText()
            if self.summary_selector.currentData()
            else self._data_title
        )
        message = (
            f"No plots are connected to {context}. "
            "Use + beside Plot to add one, or choose another Statistics node above."
            if context and context != "No input table"
            else "Choose a data source, then add a plot for its measurements "
            "or for one of its summaries."
        )
        if self._plot_unavailable:
            message = (
                "The selected plot was removed or is no longer connected "
                + (f"to {context}. " if context else "to this table. ")
                + "Choose another connected plot in the connection bar above, "
                "or use + beside Plot to add one."
            )
        self.plot_empty.setText(
            "Select a connected plot or add another."
            if self._plot_unavailable
            else "Add a plot connected to this table."
        )
        self.plot_info.setVisible(not empty)
        self.plot_canvas.setVisible(not empty and not self._plot_setup_message)
        self.plot_setup_card.setVisible(empty or bool(self._plot_setup_message))
        self.plot_setup_space.setVisible(empty or bool(self._plot_setup_message))
        self.plot_setup_heading.setText(
            "Plot unavailable"
            if self._plot_unavailable
            else "No connected plots"
            if empty
            else "Choose what to plot"
        )
        self.plot_setup_note.setText(message if empty else self._plot_setup_message)
        self.choose_measurement_button.setVisible(not empty)
        self.plotted_data_button.setVisible(not empty)
        if empty:
            self._plot_current = False
            self._plot_result = None
            self.plot_canvas.set_result(None)
            self.plot_canvas.set_busy(False)
            self.plot_warning.clear()
            self.plot_warning.hide()
            self.plot_status.clear()
            self.plot_input_toggle.setText("Input: No plot selected")
            self.plot_input_toggle.setAccessibleDescription("No plot selected")
            self.plot_source_note.clear()
            self.point_label.hide()
            self.plotted_data_button.setChecked(False)
            self.plotted_panel.hide()

    def _workflow_selected(self, *_args):
        if not self._updating:
            # Qt emits activated immediately after currentIndexChanged for a
            # changed popup choice. Keep that pair to one navigation request.
            self._workflow_activation_timer.start(0)
            self._choice_tooltip(self.workflow_selector)
            workflow_id = self.workflow_selector.currentData()
            if workflow_id:
                self.workflow_selected.emit(str(workflow_id))

    def _workflow_activated(self, *_args):
        if self._workflow_activation_timer.isActive():
            self._workflow_activation_timer.stop()
        else:
            # The graph may have switched to another tab outside this window.
            # Choosing the displayed workflow again must return to its session.
            self._workflow_selected()

    def _data_selected(self, *_args):
        if not self._updating:
            self._choice_tooltip(self.data_selector)
            source = self.data_selector.currentData()
            if source:
                self.data_selected.emit(source)

    def _plot_scope_selected(self, *_args):
        if not self._updating:
            self.plot_scope_selected.emit(str(self.plot_scope.currentData() or ""))

    def _summary_selected(self, *_args):
        if not self._updating:
            self._choice_tooltip(self.summary_selector)
            self._summary_current = False
            self._refresh_actions()
            self.summary_selected.emit(str(self.summary_selector.currentData() or ""))

    def _plot_selected(self, *_args):
        if not self._updating:
            self._choice_tooltip(self.plot_selector)
            self._plot_current = False
            self._refresh_actions()
            self.plot_selected.emit(str(self.plot_selector.currentData() or ""))

    def _source_selected(self, *_args):
        if not self._updating and self.plot_selector.currentData():
            self._choice_tooltip(self.plot_source)
            self._plot_current = False
            self._refresh_actions()
            self.plot_source_changed.emit(str(self.plot_source.currentData() or ""))

    def _summary_edited(self, params):
        self._summary_current = False
        self._refresh_actions()
        self.summary_params_changed.emit(params)

    def _plot_edited(self, params):
        self._plot_current = False
        self._refresh_actions()
        self.plot_params_changed.emit(params)

    def set_data(self, table, *, title="", node_id="", stale=False, message=""):
        changed = table is not self._data_table
        self._data_table = table
        self._data_node_id = node_id
        self._data_title = title or "Input measurements"
        self._data_current = table is not None and not stale
        self.source_label.setText(
            f"{table.row_count:,} rows · {table.column_count:,} fields"
            if table is not None
            else "No current table is available."
        )
        self.data_description.setText(
            f"{table.row_count:,} rows · {table.column_count:,} fields"
            if table is not None
            else "Calculate the connected measurement table."
        )
        self._set_table(self.data_panel, table, title or "Measurements", node_id)
        self.data_panel.set_result_status(
            message or ("Input is out of date." if stale else "")
        )
        if changed:
            checked = {
                self.column_list.item(i).data(Qt.UserRole)
                for i in range(self.column_list.count())
                if self.column_list.item(i).checkState() == Qt.Checked
            }
            previous = {
                self.column_list.item(i).data(Qt.UserRole)
                for i in range(self.column_list.count())
            }
            self.column_list.blockSignals(True)
            self.column_list.clear()
            if table is not None:
                for column, name in enumerate(table.columns):
                    item = QListWidgetItem(
                        measurement_label(table, name), self.column_list
                    )
                    item.setData(Qt.UserRole, name)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    visible = name in checked or name not in previous
                    item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
                    self.data_panel.table_view.setColumnHidden(column, not visible)
            self.column_list.blockSignals(False)
            self._search_changed()
        self._sync_column_actions()
        self._refresh_actions()

    @staticmethod
    def _set_table(panel, table, title, node_id):
        if table is None:
            if panel.table is not None:
                panel.set_table(
                    TableData((), ()),
                    title=title,
                    default_export_name="results.csv",
                    context_key=(node_id, 0),
                )
            panel.summary_label.setText("No current table is available.")
            return
        panel.set_table(
            table,
            title=title,
            default_export_name="results.csv",
            context_key=(node_id, 0),
        )

    def _search_changed(self):
        self.search_proxy.setFilterFixedString(self.search.text())
        shown = self.search_proxy.rowCount()
        total = self._data_table.row_count if self._data_table is not None else 0
        note = (
            f"Showing {shown:,} of {total:,} rows. "
            "Analysis and export use all input rows."
        )
        if self.column_list.count() and not any(
            self.column_list.item(i).checkState() == Qt.Checked
            for i in range(self.column_list.count())
        ):
            note = (
                "No columns visible. Choose columns on the left or Select all.\n" + note
            )
        self.data_view_note.setText(note)

    def _column_changed(self, item):
        if self._data_table is None:
            return
        name = item.data(Qt.UserRole)
        self.data_panel.table_view.setColumnHidden(
            self._data_table.columns.index(name), item.checkState() != Qt.Checked
        )
        self._sync_column_actions()
        self._search_changed()

    def _set_all_columns_visible(self, visible):
        if self._data_table is None:
            return
        view = self.data_panel.table_view
        blocked = self.column_list.blockSignals(True)
        updating = view.updatesEnabled()
        view.setUpdatesEnabled(False)
        try:
            for column in range(self.column_list.count()):
                self.column_list.item(column).setCheckState(
                    Qt.Checked if visible else Qt.Unchecked
                )
                view.setColumnHidden(column, not visible)
        finally:
            self.column_list.blockSignals(blocked)
            view.setUpdatesEnabled(updating)
        self._sync_column_actions()
        self._search_changed()

    def _sync_column_actions(self):
        count = self.column_list.count() if self._data_table is not None else 0
        checked = sum(
            self.column_list.item(i).checkState() == Qt.Checked for i in range(count)
        )
        self.select_all_columns_button.setEnabled(checked < count)
        self.select_no_columns_button.setEnabled(checked > 0)

    def set_summary(
        self,
        *,
        table=None,
        params=None,
        result=None,
        stale=True,
        failed=False,
        message="",
        busy=False,
        editable=True,
    ):
        self._summary_editable = bool(editable)
        self.statistics_panel.set_state(
            table=table,
            params=params or StatisticsRecipe().to_params(),
            result=result,
            stale=stale or busy,
            failed=failed,
            message=message,
        )
        self.statistics_panel.setEnabled(self._available and self._summary_bound)
        self._summary_current = result is not None and not (stale or failed or busy)
        self._set_table(
            self.summary_panel,
            result,
            "Statistics",
            str(self.summary_selector.currentData() or ""),
        )
        self.summary_panel.set_result_status(
            message
            or (
                "Previous result; calculate again before exporting."
                if stale and result is not None
                else ""
            )
        )
        self._summary_columns()
        self._refresh_actions()

    def _summary_columns(self, *_args):
        table = self.summary_panel.table
        if table is None:
            return
        preview = statistics_preview_columns(table)
        visible = set(range(table.column_count) if preview is None else preview)
        for index in range(table.column_count):
            self.summary_panel.table_view.setColumnHidden(
                index, not self.summary_evidence.isChecked() and index not in visible
            )
        shown = (
            table.column_count if self.summary_evidence.isChecked() else len(visible)
        )
        self.summary_panel.summary_label.setText(
            f"{table.row_count:,} rows · "
            f"{shown:,} of {table.column_count:,} fields shown"
        )

    def set_plot(
        self,
        *,
        table=None,
        params=None,
        result=None,
        stale=True,
        failed=False,
        message="",
        busy=False,
        source_kind="original",
        protected_paths=(),
        editable=True,
        setup_message="",
    ):
        self._plot_editable = bool(editable)
        self._plot_setup_message = setup_message
        self._plot_result = result
        self._plot_protected_paths = tuple(protected_paths)
        self._plot_current = result is not None and not (
            stale or failed or busy or setup_message
        )
        self._plot_notes_transient = (stale or busy) and not (failed or setup_message)
        self.plot_controls.set_state(table, params or PlotRecipe().to_params())
        self.plot_controls.setEnabled(self._available and self._plot_bound)
        if failed and not setup_message:
            self.plot_canvas.set_error(message)
        else:
            self.plot_canvas.set_result(None if setup_message else result)
        self.plot_canvas.set_busy(busy)
        self.plot_canvas.setVisible(not setup_message)
        self.plot_setup_card.setVisible(bool(setup_message))
        self.plot_setup_space.setVisible(bool(setup_message))
        self.plot_setup_note.setText(setup_message)
        missing_y = self.plot_controls.controls["y_column"].currentData() in (
            None,
            "",
            "auto",
        )
        self.choose_measurement_button.setText(
            "Choose measurement" if missing_y else "Choose X measurement"
        )
        self.point_label.setVisible(bool(result) and not failed and not setup_message)
        source = self.plot_source.currentText() or "No input selected"
        rows = table.row_count if table is not None else 0
        is_summary = source_kind == "summary" or is_summary_table(table)
        self.plot_input_toggle.setText(f"Input: {source} · {rows:,} rows")
        self.plot_input_toggle.setAccessibleDescription(self.plot_input_toggle.text())
        source_note = (
            f"{rows:,} summary rows. Each point is one summary row, "
            "not an original object."
            if is_summary
            else f"{rows:,} original measurement rows. "
            "No Statistics node is used for this plot."
        )
        if result is not None:
            source_note += (
                f"\n{result.counts.plotted_points:,} plotted values from "
                f"{result.counts.eligible_rows:,} eligible input rows."
            )
        self.plot_source_note.setText(source_note)
        self.plot_input_toggle.setToolTip(f"Input: {source}\n{source_note}")
        status = (
            ""
            if failed or setup_message
            else message
            or (
                "Updating plot…"
                if busy
                else "Plot is out of date; calculate again before exporting."
                if stale and result is not None
                else _result_summary(result)
                if result is not None
                else "Choose a plot node or add a plot."
            )
        )
        self.plot_status.setText(status)
        # Counts are already shown in the input card; reserve this line for
        # actionable states, while keeping the full summary accessible.
        self.plot_status.setVisible(
            bool(status) and (not self._plot_current or bool(message))
        )
        self.plot_source_card.setToolTip(status)
        if self._plot_current and self._plot_bound:
            self._plot_notice_warnings = visible_plot_warnings(result)
            # PlotData freezes its own table copy, even for appearance-only
            # redraws. The connected immutable table is the stable revision.
            notice_table = table if table is not None else result.source_table
            self._plot_notice_state.update(
                str(self.plot_selector.currentData() or ""),
                notice_table,
                result.recipe.to_params(),
                self._plot_notice_warnings,
            )
            self.plot_warning.setText("\n".join(self._plot_notice_warnings))
        self.plotted_data_button.setEnabled(self._plot_current)
        if not self._plot_current:
            self.plotted_data_button.setChecked(False)
        self._set_table(
            self.plotted_panel,
            result.plotted_table if self._plot_current else None,
            "Plotted data",
            str(self.plot_selector.currentData() or ""),
        )
        self._sync_plot_context()
        self._refresh_actions()
        self.plot_fit_timer.start(0)

    def set_available(self, available, message=""):
        self._available = bool(available)
        self.availability_label.setText(message)
        self.availability_label.setVisible(bool(message))
        self._refresh_actions()

    def set_busy(self, busy, message=""):
        self._busy = bool(busy)
        self._busy_text = message or "Updating results…"
        self._sync_busy_text()
        self.busy_label.setVisible(self._busy)
        self.progress.setVisible(self._busy)
        self._refresh_actions()

    def _sync_busy_text(self):
        if not hasattr(self, "busy_label"):
            return
        self.busy_label.setToolTip(self._busy_text)
        self.busy_label.setText(
            self.busy_label.fontMetrics().elidedText(
                self._busy_text, Qt.ElideRight, max(40, self.busy_label.width())
            )
        )

    def refresh_theme(self):
        """Keep sidebar/table/canvas colors in step with live napari themes."""
        if getattr(self, "_theme_refresh_in_progress", True):
            return
        self._theme_refresh_in_progress = True
        try:
            colors = theme_colors(self.palette())
            key = tuple(
                color.rgba()
                for color in (
                    colors.surface,
                    colors.alternate_surface,
                    colors.text,
                    colors.border,
                )
            )
            for surface in self._control_surfaces:
                surface.setStyleSheet(
                    "QWidget#WorkspaceControls {"
                    f"background: {colors.alternate_surface.name()};"
                    "}"
                )
            self.connection_bar.setStyleSheet(
                "QFrame#WorkspaceConnections {"
                f"background: {colors.alternate_surface.name()};"
                f"border: 1px solid {colors.info.accent.name()};"
                "border-radius: 0; }"
                "QFrame#WorkspaceConnections > QWidget, "
                "QFrame#WorkspaceConnections QLabel { background: transparent; }"
            )
            for arrow in self.connection_arrows:
                arrow.update()
            self.tabs.setStyleSheet(
                "QTabWidget#ResultsWorkspaceTabs::pane { border: 0; }"
                "QTabBar::tab { padding: 11px 12px; border: 0; border-radius: 0;"
                f"border-bottom: 2px solid {colors.border.name()};"
                f"color: {colors.text.name()}; background: {colors.surface.name()}; }}"
                "QTabBar::tab:selected {"
                f"border-bottom-color: {colors.info.accent.name()};"
                f"background: {colors.alternate_surface.name()}; font-weight: bold; }}"
                "QFrame#WorkspaceEditingContext {"
                f"border-bottom: 1px solid {colors.border.name()}; }}"
            )
            for index, kind in enumerate(("table", "statistics", "histogram")):
                self.tabs.setTabIcon(index, interface_icon(kind, self.palette()))
            for button in self._show_node_buttons:
                button.setIcon(toolbar_icon("workflow", self.palette()))
            self.export_button.setIcon(interface_icon("save", self.palette()))
            self.recalculate_button.setIcon(toolbar_icon("refresh", self.palette()))
            self.plot_source_card.setStyleSheet(
                "QFrame#WorkspacePlotSource {"
                f"background: {colors.alternate_surface.name()};"
                f"border-left: 3px solid {colors.info.accent.name()}; }}"
                "QFrame#WorkspacePlotSource QLabel { background: transparent; }"
            )
            self.plot_input_toggle.setStyleSheet(
                "QPushButton#WorkspacePlotInputToggle { text-align: left;"
                "background: transparent; border: 1px solid transparent;"
                "padding: 5px 8px;"
                f"color: {colors.text.name()}; }}"
                "QPushButton#WorkspacePlotInputToggle:hover {"
                f"background: {colors.raised_surface.name()}; }}"
                "QPushButton#WorkspacePlotInputToggle:focus {"
                f"border-color: {colors.info.accent.name()}; }}"
            )
            self._toggle_plot_input(self.plot_input_toggle.isChecked())
            self._refresh_plot_note_theme()
            self.plot_setup_card.setStyleSheet(
                "QFrame#WorkspacePlotSetup {"
                f"background: {colors.info.surface.name()};"
                f"border-left: 3px solid {colors.info.accent.name()}; }}"
                "QFrame#WorkspacePlotSetup QLabel {"
                f"color: {colors.info.foreground.name()}; background: transparent; }}"
            )
            self.source_label.setStyleSheet(f"color: {colors.muted_text.name()};")
            for _heading, detail in self._context_labels:
                detail.setStyleSheet(f"color: {colors.muted_text.name()};")
            for panel in (self.data_panel, self.summary_panel, self.plotted_panel):
                panel.refresh_theme(self.palette())
            self.plot_canvas.setPalette(self.palette())
            if key != self._theme_key and self.plot_canvas.result is not None:
                # Canvas rendering depends on palette, but the recipe/result
                # stay the same. Never recalculate the scientific node here.
                result = self.plot_canvas.result
                self.plot_canvas.set_result(None)
                self.plot_canvas.set_result(result)
            self._theme_key = key
        finally:
            self._theme_refresh_in_progress = False

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.ApplicationPaletteChange,
            QEvent.PaletteChange,
            QEvent.StyleChange,
        } and not getattr(self, "_theme_refresh_in_progress", True):
            self.theme_timer.start(0)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._sync_busy_text()

    def eventFilter(self, watched, event):  # noqa: N802
        if hasattr(self, "plot_fit_timer") and (
            watched is self.plot_area
            and event.type() in (QEvent.Resize, QEvent.Show, QEvent.LayoutRequest)
            or watched is self.plot_info.widget()
            and event.type() == QEvent.LayoutRequest
        ):
            self.plot_fit_timer.start(0)
        return super().eventFilter(watched, event)

    def current_node_id(self):
        tab = self.tabs.currentIndex()
        if (tab == 1 and not self._summary_bound) or (
            tab == 2 and not self._plot_bound
        ):
            return ""
        return str(
            (
                self._data_node_id,
                self.summary_selector.currentData(),
                self.plot_selector.currentData(),
            )[self.tabs.currentIndex()]
            or ""
        )

    def show_tab(self, name):
        names = {"data": 0, "summary": 1, "plots": 2, "plot": 2}
        self.tabs.setCurrentIndex(names.get(str(name).lower(), 0))

    def _refresh_actions(self, *_args):
        if not hasattr(self, "export_button"):
            return
        self._sync_plot_notes()
        tab = self.tabs.currentIndex()
        current = (self._data_current, self._summary_current, self._plot_current)[tab]
        self.export_button.setText("Export figure…" if tab == 2 else "Export table…")
        self.export_button.setEnabled(current and self._available and not self._busy)
        self.show_node_button.setEnabled(
            bool(self.current_node_id()) and self._available
        )
        self.recalculate_button.setEnabled(
            bool(self.current_node_id())
            and self._available
            and not self._busy
            and not (tab == 2 and self._plot_setup_message)
        )
        heading, detail = self._context_labels[tab]
        name = (
            self._data_title
            if tab == 0
            else self.summary_selector.currentText()
            if tab == 1
            else self.plot_selector.currentText()
        )
        bound = (bool(self._data_node_id), self._summary_bound, self._plot_bound)[tab]
        heading.setText(
            f"{'Viewing' if tab == 0 else 'Editing'}: {name}"
            if bound
            else "Input measurements"
            if tab == 0
            else "Add a summary"
            if tab == 1
            else "Add a plot"
        )
        detail.setText(
            "Browse the connected table. Analysis uses all input rows."
            if tab == 0
            else "Changes update this node in the workflow."
            if bound
            else "Select the data and settings for a new workflow node."
        )
        self.add_summary_button.setEnabled(
            bool(self._data_node_id) and self._available and not self._busy
        )
        self.add_plot_button.setEnabled(
            bool(self._data_node_id)
            and (not self.summary_selector.currentData() or self._summary_bound)
            and self._available
            and not self._busy
        )
        self.workflow_selector.setEnabled(
            self._workflow_selection_enabled and self._has_workflow_choices
        )
        # A removed source must not trap the user in an unavailable workspace.
        # The controller supplies valid choices for the active workflow only.
        self.data_selector.setEnabled(self._has_data_choices)
        for widget in (
            self.summary_selector,
            self.plot_scope,
            self.plot_selector,
            self.plot_source,
        ):
            widget.setEnabled(self._available)
        self.statistics_panel.setEnabled(
            self._available and self._summary_bound and self._summary_editable
        )
        self.plot_controls.setEnabled(
            self._available and self._plot_bound and self._plot_editable
        )
        self.choose_measurement_button.setEnabled(
            self._available
            and self._plot_bound
            and self._plot_editable
            and not self._busy
        )
        self.plot_source.setEnabled(
            self._available
            and self._plot_bound
            and self._plot_editable
            and not self._busy
        )
        self.plot_connection_button.setEnabled(
            self._available
            and self._plot_bound
            and self._plot_editable
            and not self._busy
        )

    def request_export(self):
        if not self.export_button.isEnabled():
            return
        tab = self.tabs.currentIndex()
        if tab < 2:
            (self.data_panel, self.summary_panel)[tab].request_export()
            return
        result = self._plot_result
        dialog = PlotExportDialog(
            result,
            self,
            protected_paths=self._plot_protected_paths,
            is_current=lambda: (
                self._available
                and self._plot_current
                and self._plot_result is result
                and not self._busy
            ),
        )
        if dialog.exec() == QDialog.Accepted and dialog.exported is not None:
            self.export_completed.emit(str(dialog.exported.paths[0]))

    def closeEvent(self, event):  # noqa: N802
        self.search_timer.stop()
        self.theme_timer.stop()
        for panel in (self.data_panel, self.summary_panel, self.plotted_panel):
            panel._cancel_active_sort()
        super().closeEvent(event)
