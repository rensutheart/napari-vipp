"""Tabbed, task-oriented presentation for the retained batch dialog.

The dialog and its controller continue to own validation and execution. This
module arranges those controls and keeps browsing state separate from a runnable
plan, so an old table can never silently authorize a new run.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from time import monotonic

from qtpy.QtCore import QPoint, QSignalBlocker, QSize, Qt, QTimer, QUrl
from qtpy.QtGui import (
    QAction,
    QActionGroup,
    QFont,
    QFontMetrics,
    QPalette,
    QTextDocument,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyleOption,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.batch import (
    BatchScientificPreflightError,
    batch_item_file_policy_key,
)
from napari_vipp.ui.batch_check_progress import BatchCheckProgressPresentation
from napari_vipp.ui.batch_output_policy import (
    BatchExistingFilesControls,
    batch_work_counts,
    checked_output_message,
    item_file_choice,
    item_file_choice_label,
    output_action,
    output_counts_text,
    planned_item_status,
    with_existing_file_policy,
    with_item_file_policy,
    with_output_policy_config,
)
from napari_vipp.ui.batch_override_presentation import BatchOverridePresentation
from napari_vipp.ui.batch_setup import BatchSetupPresentation
from napari_vipp.ui.batch_table_style import apply_batch_table_style
from napari_vipp.ui.palette_roles import custom_paint_colors, theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class BatchActivityLabel(QLabel):
    """Keep full status text accessible while eliding a narrow footer visually."""

    def paintEvent(self, event) -> None:  # noqa: N802
        visible = self.fontMetrics().elidedText(
            self.text(), Qt.ElideRight, max(self.contentsRect().width(), 0)
        )
        if visible == self.text():
            super().paintEvent(event)
            return
        painter = QStylePainter(self)
        option = QStyleOption()
        option.initFrom(self)
        painter.drawPrimitive(QStyle.PE_Widget, option)
        painter.drawItemText(
            self.contentsRect(),
            self.alignment() | Qt.TextSingleLine,
            self.palette(),
            self.isEnabled(),
            visible,
            QPalette.WindowText,
        )


class BatchWorkflowTabBar(QTabBar):
    """Keep each icon next to its label inside an expanded navigation tab."""

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QStylePainter(self)
        for index in range(self.count()):
            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.CE_TabBarTabShape, option)
            # Native expanded tabs can pin the icon to the far left while
            # centering only the text. Center their combined label instead.
            font = QFont(self.font())
            font.setBold(index == self.currentIndex())
            text_width = (
                QFontMetrics(font).size(Qt.TextShowMnemonic, option.text).width()
            )
            width = (
                text_width
                + (option.iconSize.width() + 14 if not option.icon.isNull() else 0)
                + 12
            )
            width = min(width, option.rect.width())
            option.rect.setLeft(option.rect.center().x() - width // 2)
            option.rect.setWidth(width)
            painter.drawControl(QStyle.CE_TabBarTabLabel, option)


class BatchWorkflowWorkspace(
    BatchSetupPresentation, BatchCheckProgressPresentation, BatchOverridePresentation
):
    """UI mixin; all processing remains in CollectionBatchDialog/controller."""

    _ITEM_PAGE_SIZE = 50

    def _build_workspace(self, output_form) -> None:
        self._workspace_ready = False
        self._checking_plan = False
        self._display_plan = None
        self._item_page = 0
        self._checked_items: set[int] = set()
        self._current_item = 0
        self._item_run_states: dict[int, str] = {}
        self._rendering_items = False
        self._syncing_selection = False
        self._run_started_at: float | None = None
        self._last_elapsed = 0.0
        self._init_check_progress()

        # Reuse the real config actions. Examples live only in the overflow menu.
        toolbar = self.config_row.layout()
        toolbar.removeWidget(self.demo_config_button)
        self.demo_config_button.hide()
        toolbar.removeWidget(self.batch_activity_strip)
        self.config_name_label = QLabel("Unsaved configuration")
        self.config_name_label.setMinimumWidth(0)
        self.config_name_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        toolbar.addWidget(self.config_name_label, 1)
        self.compute_summary_label = QLabel("")
        self.compute_summary_label.setMinimumWidth(0)
        self.compute_summary_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.compute_icon_label = QLabel()
        self.compute_icon_label.setFixedSize(18, 18)
        toolbar.addWidget(self.compute_icon_label)
        toolbar.addWidget(self.compute_summary_label)
        self.more_button = ToolbarCommandButton()
        self.more_button.setObjectName("BatchMoreButton")
        self.more_button.setFixedSize(30, 30)
        self.more_button.setAccessibleName("More batch options")
        self.more_button.setToolTip("More batch options")
        self.more_menu = QMenu(self.more_button)
        self.demo_action = self.more_menu.addAction("Load demo configuration…")
        self.demo_action.triggered.connect(self.demo_config_button.click)
        self.more_button.setMenu(self.more_menu)
        toolbar.addWidget(self.more_button)
        self.more_menu.aboutToShow.connect(
            lambda: self.demo_action.setEnabled(self.demo_config_button.isEnabled())
        )
        for button in (
            self.load_config_button,
            self.save_config_button,
            self.more_button,
        ):
            button.setIconSize(QSize(18, 18))
        toolbar.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.setTabBar(BatchWorkflowTabBar())
        self.tabs.setObjectName("BatchWorkflowTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setIconSize(QSize(18, 18))
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.currentChanged.connect(self._sync_workspace)

        self._build_setup_page(output_form)

        self.items_page = QWidget()
        items = QVBoxLayout(self.items_page)
        items.setContentsMargins(10, 10, 10, 10)
        self.items_heading = QLabel("Review batch items")
        self.items_heading.setWordWrap(True)
        self.items_heading.setObjectName("BatchPageHeading")
        items_heading_row = QHBoxLayout()
        items_heading_row.addWidget(self.items_heading, 1)
        self.check_count_label = QLabel()
        self.check_count_label.setWordWrap(True)
        self.check_count_label.setAccessibleName("Batch check progress")
        self.check_count_label.hide()
        items_heading_row.addWidget(self.check_count_label)
        items.addLayout(items_heading_row)
        self.review_banner = QFrame()
        self.review_banner.setObjectName("BatchReviewBannerFrame")
        review_banner_layout = QHBoxLayout(self.review_banner)
        review_banner_layout.setContentsMargins(12, 8, 10, 8)
        review_banner_layout.setSpacing(12)
        review_banner_layout.addWidget(self.preview_status, 1)
        self.review_check_button = ToolbarCommandButton("Check batch again")
        self.review_check_button.setObjectName("BatchPrimaryAction")
        self.review_check_button.clicked.connect(self._check_batch)
        self.review_check_button.hide()
        review_banner_layout.addWidget(self.review_check_button, 0, Qt.AlignVCenter)
        items.addWidget(self.review_banner)
        self.existing_files_controls = BatchExistingFilesControls()
        self.existing_files_controls.policyChanged.connect(
            self._choose_existing_file_policy
        )
        self.existing_files_controls.resetItemsRequested.connect(
            self._reset_item_file_policies
        )
        items.addWidget(self.existing_files_controls)
        self.preview_status.setObjectName("BatchReviewBanner")
        self.items_command_row = QWidget()
        self.items_commands = QGridLayout(self.items_command_row)
        self.items_commands.setContentsMargins(0, 0, 0, 0)
        self.items_commands.setHorizontalSpacing(0)
        self.items_commands.setVerticalSpacing(8)
        self.recheck_item_button = ToolbarCommandButton("Recheck selected")
        self.recheck_item_button.setToolTip(
            "Recheck selected source revisions and output presence. "
            "Running always verifies the entire batch again."
        )
        self.recheck_item_button.clicked.connect(self._recheck_selected_items)
        self.load_overrides_button = ToolbarCommandButton("Load overrides")
        self.load_overrides_button.setToolTip(
            "Open per-sample overrides for the checked items. "
            "This does not load a separate configuration file."
        )
        self.load_overrides_button.clicked.connect(self._open_selected_overrides)
        self.preview_button.setText("Recheck all")
        self.preview_button.setToolTip(
            "Recheck the complete collection and output plan without "
            "calculating a representative image."
        )
        self.preview_button.clicked.disconnect(self._preview_batch)
        self.preview_button.clicked.connect(self._check_batch)
        self.item_filter = QComboBox()
        self.item_filter.addItems(["All items", "Needs attention", "Ready"])
        self.item_filter.setMinimumWidth(120)
        self.item_filter.setMaximumWidth(180)
        self.item_filter.setToolTip("Filter the reviewed batch items")
        self.item_search = QLineEdit()
        self.item_search.setPlaceholderText("Find item or source path")
        self.item_search.setClearButtonEnabled(True)
        self.item_search.setMinimumWidth(140)
        self.item_search_icon = self.item_search.addAction(
            toolbar_icon("search", self.palette()), QLineEdit.LeadingPosition
        )
        self.item_search.textChanged.connect(self._filter_items)
        self.item_filter.currentIndexChanged.connect(self._filter_items)
        self._item_command_widgets = (
            self.recheck_item_button,
            self.preview_item_button,
            self.load_overrides_button,
            self.preview_button,
            self.item_filter,
            self.item_search,
        )
        self.item_selection_commands = QWidget()
        self.item_selection_commands.setAccessibleName("Selected-item actions")
        selected_actions = QHBoxLayout(self.item_selection_commands)
        self.item_collection_commands = QWidget()
        self.item_collection_commands.setAccessibleName(
            "Whole-batch actions and filters"
        )
        collection_actions = QHBoxLayout(self.item_collection_commands)
        for group_layout, widgets in (
            (selected_actions, self._item_command_widgets[:3]),
            (collection_actions, self._item_command_widgets[3:]),
        ):
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(6)
            for widget in widgets:
                group_layout.addWidget(widget)
        self.item_selection_commands.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Fixed
        )
        self.item_collection_commands.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        collection_actions.setStretch(2, 1)
        self.preview_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.items_commands.addWidget(self.item_selection_commands, 0, 0)
        self.items_commands.addWidget(self.item_collection_commands, 0, 2)
        items.addWidget(self.items_command_row)
        self.preview_table.setMaximumHeight(16777215)
        self.preview_table.setMinimumHeight(160)
        self.preview_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.preview_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.preview_table.customContextMenuRequested.connect(self._item_context_menu)
        self.preview_table.itemChanged.connect(self._item_check_changed)
        self.preview_table.itemSelectionChanged.connect(self._show_item_details)
        self.preview_table.verticalHeader().hide()
        self.preview_table.setAlternatingRowColors(True)
        self.item_details = QTextBrowser()
        self.item_details.setObjectName("BatchItemDetails")
        self.item_details.setOpenLinks(False)
        self.item_details.setMinimumWidth(0)
        self.item_details.setMinimumHeight(100)
        self.item_details.anchorClicked.connect(self._item_detail_link)
        self.items_splitter = QSplitter(Qt.Horizontal)
        self.items_splitter.addWidget(self.preview_table)
        self.items_splitter.addWidget(self.item_details)
        self.items_splitter.setStretchFactor(0, 3)
        self.items_splitter.setStretchFactor(1, 2)
        self.items_splitter.setChildrenCollapsible(False)
        items.addWidget(self.items_splitter, 1)
        self.item_pager = QWidget()
        pager = QHBoxLayout(self.item_pager)
        pager.setContentsMargins(0, 0, 0, 0)
        self.item_range_label = QLabel("No items checked yet")
        pager.addWidget(self.item_range_label, 1)
        self.items_previous_button = QPushButton("Previous")
        self.items_next_button = QPushButton("Next")
        self.items_previous_button.clicked.connect(lambda: self._page_items(-1))
        self.items_next_button.clicked.connect(lambda: self._page_items(1))
        pager.addWidget(self.items_previous_button)
        pager.addWidget(self.items_next_button)
        items.addWidget(self.item_pager)
        items.addWidget(self.graph_preview_status)
        self.tabs.addTab(self.items_page, "Items && outputs")

        self._build_overrides_page()

        from napari_vipp.ui.batch_results import BatchResultsPanel

        self.run_page = QWidget()
        run = QVBoxLayout(self.run_page)
        run.setContentsMargins(10, 10, 10, 10)
        self.run_recap_label = QLabel("")
        self.run_recap_label.setWordWrap(True)
        self.run_recap_label.setObjectName("BatchRunRecap")
        run.addWidget(self.run_recap_label)
        self.results_panel = BatchResultsPanel()
        self.results_panel.policyChanged.connect(self._choose_existing_file_policy)
        self.results_panel.existing_files_controls.resetItemsRequested.connect(
            self._reset_item_file_policies
        )
        self.results_panel.itemRequested.connect(self._review_result_item)
        self.results_panel.checkBatchRequested.connect(self._check_batch)
        run.addWidget(self.results_panel, 1)
        # Preserve public progress controls for the host and older integrations.
        # The panel's copies are the one visible source of detailed progress.
        self.run_group.setParent(self.run_page)
        self.run_group.hide()
        self.run_tab = QWidget()
        run_tab_layout = QVBoxLayout(self.run_tab)
        run_tab_layout.setContentsMargins(10, 0, 10, 0)
        self.results_panel.layout().removeWidget(self.results_panel.artifact_toolbar)
        run_tab_layout.addWidget(self.results_panel.artifact_toolbar)
        self.run_scroll = self._workspace_scroll(self.run_page)
        # The scroll contents already have the tab's horizontal page gutters.
        run.setContentsMargins(0, 10, 0, 10)
        run_tab_layout.addWidget(self.run_scroll, 1)
        self.tabs.addTab(self.run_tab, "Run && results")

        self.footer = QFrame()
        self.footer.setObjectName("BatchWorkflowFooter")
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(8, 6, 8, 6)
        footer_layout.setSpacing(6)
        self.batch_activity_strip.setMinimumHeight(30)
        self.batch_activity_status.setMinimumWidth(0)
        self.source_detection_progress.setFixedWidth(112)
        progress_policy = self.source_detection_progress.sizePolicy()
        progress_policy.setRetainSizeWhenHidden(False)
        self.source_detection_progress.setSizePolicy(progress_policy)
        self.footer_elapsed_label = QLabel("")
        self.footer_elapsed_label.setMinimumWidth(78)
        self.footer_elapsed_label.hide()
        self.batch_activity_strip.layout().addWidget(self.footer_elapsed_label)
        self.batch_activity_strip.layout().setContentsMargins(0, 0, 0, 0)
        footer_layout.addWidget(self.batch_activity_strip, 1, Qt.AlignVCenter)
        self.next_button = ToolbarCommandButton("Check batch")
        self.next_button.setObjectName("BatchPrimaryAction")
        self.next_button.clicked.connect(self._next_batch_step)
        self.footer_overrides_button = ToolbarCommandButton("Load overrides")
        self.footer_overrides_button.clicked.connect(self._open_selected_overrides)
        # Keep the optional detour immediately left of the next step on every
        # platform, independent of native dialog-button role ordering.
        self.footer_action_row = QWidget()
        self.footer_action_row.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        action_layout = QHBoxLayout(self.footer_action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        action_layout.addStretch(1)
        self._footer_buttons = (
            self.footer_overrides_button,
            self.next_button,
            self.run_button,
            self.cancel_run_button,
            self.close_button,
        )
        for button in self._footer_buttons:
            self.button_box.removeButton(button)
            action_layout.addWidget(button)
        self.close_button.clicked.connect(self.reject)
        self.button_box.hide()
        footer_layout.addWidget(self.footer_action_row, 0, Qt.AlignVCenter)
        self.run_button.setObjectName("BatchPrimaryAction")
        self.cancel_run_button.setObjectName("BatchStopAction")
        self.cancel_run_button.setText("Stop safely")
        self._batch_command_icons = (
            (self.load_config_button, "open"),
            (self.save_config_button, "save"),
            (self.more_button, "more"),
            (self.recheck_item_button, "refresh"),
            (self.preview_item_button, "focus"),
            (self.load_overrides_button, "batch"),
            (self.preview_button, "recheck_all"),
            (self.review_check_button, "checklist"),
            (self.footer_overrides_button, "batch"),
            (self.run_button, "calculate"),
            (self.cancel_run_button, "stop"),
        )
        for button, _kind in self._batch_command_icons:
            button.setIconSize(QSize(18, 18))
        self.next_button.setIconSize(QSize(18, 18))
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.config_row)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.footer)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.parameter_override_editor.sourceSelected.connect(
            self._override_source_selected
        )
        self.parameter_override_editor.selectionChanged.connect(
            self._override_selection_changed
        )
        self._workspace_ready = True
        self._responsive_workspace()
        self._render_items()

    @staticmethod
    def _workspace_scroll(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        scroll.setMinimumSize(0, 0)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return scroll

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if getattr(self, "_workspace_ready", False):
            self._responsive_workspace()

    def _responsive_workspace(self) -> None:
        self._responsive_setup_context()
        narrow = self.width() < 960
        if getattr(self, "_last_workspace_narrow", None) != narrow:
            self._last_workspace_narrow = narrow
            self.setup_columns.removeWidget(self.destination_group)
            self.setup_columns.addWidget(
                self.destination_group,
                1 if narrow else 0,
                0 if narrow else 1,
                Qt.AlignTop,
            )
            self.setup_columns.setColumnStretch(1, 0 if narrow else 1)
            self.items_splitter.setOrientation(Qt.Vertical if narrow else Qt.Horizontal)
            self.items_splitter.setSizes([420, 200] if narrow else [600, 320])
        self._layout_item_commands()
        self._layout_compute_summary()
        self.config_name_label.setVisible(self.width() >= 620)
        self.source_detection_progress.setFixedWidth(88 if self.width() < 680 else 112)
        self._update_elapsed()
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _layout_item_commands(self) -> None:
        # Keep the three selected-item actions together. Recheck all belongs
        # with collection filters, with a deliberate gap rather than one long
        # undifferentiated button row. Wrap the whole group when it won't fit.
        selection = self.item_selection_commands
        collection = self.item_collection_commands
        available = max(0, self.width() - 40)
        required = (
            selection.sizeHint().width() + collection.minimumSizeHint().width() + 24
        )
        wrapped = required > available
        if getattr(self, "_item_commands_wrapped", None) != wrapped:
            self._item_commands_wrapped = wrapped
            self.items_commands.removeWidget(collection)
            self.items_commands.addWidget(
                collection,
                1 if wrapped else 0,
                0 if wrapped else 2,
                1,
                3 if wrapped else 1,
            )
        self.items_commands.setColumnMinimumWidth(1, 0 if wrapped else 24)
        self.items_commands.setColumnStretch(1, 0 if wrapped else 1)
        self.items_commands.setColumnStretch(2, 1 if wrapped else 3)

    def _toggle_node_behavior(self, checked: bool) -> None:
        self.node_execution_group.setVisible(checked)
        if checked:
            # Let the page recompute its scroll range before revealing the
            # first editable row; a large node list must not jump to its middle.
            QTimer.singleShot(0, self._reveal_node_behavior)

    def _reveal_node_behavior(self) -> None:
        if not self.node_behavior_toggle.isChecked():
            return
        first = next(iter(self._node_execution_combos.values()), None)
        if first is not None:
            self.overrides_scroll.ensureWidgetVisible(first, 0, 24)

    def _apply_workspace_theme(self) -> None:
        if not getattr(self, "_workspace_ready", False):
            return
        colors = custom_paint_colors(self.palette())
        tones = theme_colors(self.palette())
        apply_batch_table_style(
            self.preview_table,
            self.palette(),
            font=self.font(),
            horizontal_padding=12,
        )
        self._style_item_statuses()
        accent = tones.info.accent.name()
        self.tabs.setStyleSheet(
            "QTabWidget#BatchWorkflowTabs::pane { border: 0; }"
            "QTabBar::tab { padding: 9px 10px; border: 0; "
            "border-radius: 0; "
            f"border-bottom: 2px solid {colors.border.name()}; "
            f"color: {colors.text.name()}; background: {colors.surface.name()}; }}"
            f"QTabBar::tab:selected {{ border-bottom-color: {accent}; "
            f"background: {colors.alternate_surface.name()}; font-weight: bold; }}"
        )
        self.footer.setStyleSheet(
            "QFrame#BatchWorkflowFooter { "
            f"background: {colors.alternate_surface.name()};"
            f"border-top: 1px solid {colors.border.name()}; }}"
            "QPushButton#BatchPrimaryAction { font-weight: bold; "
            f"border: 1px solid {accent}; padding: 6px 12px; }}"
            "QPushButton#BatchStopAction { "
            f"color: {tones.error.foreground.name()}; "
            f"border: 1px solid {tones.error.accent.name()}; padding: 6px 10px; }}"
        )
        for heading in self.findChildren(QLabel, "BatchPageHeading"):
            heading.setStyleSheet(f"font-weight: bold; color: {colors.text.name()};")
        for label in (self.config_name_label, self.footer_elapsed_label):
            label.setStyleSheet(f"color: {colors.muted_text.name()};")
        self.item_details.setStyleSheet(
            f"QTextBrowser {{ background: {colors.alternate_surface.name()}; "
            f"color: {colors.text.name()}; border: 1px solid {colors.border.name()}; "
            "padding: 8px; }"
        )
        self.item_details.document().setDefaultStyleSheet(
            f"a {{ color: {tones.info.foreground.name()}; }}"
        )
        self._show_item_details()
        self.config_row.setStyleSheet(
            "QPushButton { padding: 2px 5px; }"
            "QPushButton#BatchMoreButton::menu-indicator { image: none; width: 0; }"
        )
        self.items_command_row.setStyleSheet("QPushButton { padding: 5px 8px; }")
        for button, kind in self._batch_command_icons:
            button.setIcon(toolbar_icon(kind, self.palette()))
        for index, kind in enumerate(("setup", "checklist", "batch", "activity")):
            self.tabs.setTabIcon(index, toolbar_icon(kind, self.palette()))
        self.compute_icon_label.setPixmap(
            toolbar_icon("compute", self.palette()).pixmap(18, 18)
        )
        self.item_search_icon.setIcon(toolbar_icon("search", self.palette()))
        self._update_next_button_icon()
        self._style_review_banner()
        self._apply_setup_theme()
        self._apply_override_presentation_theme()
        self._refresh_check_cells()
        # Qt does not propagate palettes/fonts through the styled tab boundary.
        # Polish first: the initial QSS polish would otherwise erase these roles
        # on a still-hidden page, leaving its tables in the previous theme.
        for panel in (self.parameter_override_editor, self.results_panel):
            panel.ensurePolished()
            if panel.palette() != self.palette():
                panel.setPalette(self.palette())
            if panel.font() != self.font():
                panel.setFont(self.font())
        self._layout_item_commands()
        self._reserve_footer_height()

    def _update_next_button_icon(self) -> None:
        kind = (
            "activity"
            if self.next_button.text() == "View run report"
            else "checklist"
            if self._preview_result is None
            else "next"
        )
        self.next_button.setIcon(toolbar_icon(kind, self.palette()))

    def _style_review_banner(self) -> None:
        tones = theme_colors(self.palette())
        state = self._batch_activity_state
        tone = (
            tones.info
            if self._checking_plan
            else tones.error
            if state == "error"
            else tones.warning
            if state == "warning"
            or (self._display_plan is not None and self._preview_result is None)
            else tones.success
            if self._preview_result is not None
            else tones.info
        )
        self.review_banner.setStyleSheet(
            "QFrame#BatchReviewBannerFrame {"
            f"background: {tone.surface.name()};"
            f"border-left: 3px solid {tone.accent.name()}; }}"
        )
        self.preview_status.setStyleSheet(
            "QLabel#BatchReviewBanner { border: none; padding: 0;"
            f"background: {tone.surface.name()}; color: {tone.foreground.name()};"
            "}"
        )

    def _reserve_footer_height(self) -> None:
        """Reserve the tallest action even when a shorter action is visible."""
        buttons = self._footer_buttons
        for button in buttons:
            button.ensurePolished()
        self.footer_action_row.setFixedHeight(
            max((button.sizeHint().height() for button in buttons), default=30)
        )
        self.footer.setFixedHeight(self.footer.layout().sizeHint().height())

    def _sync_workspace(self, *_args) -> None:
        if not getattr(self, "_workspace_ready", False):
            return
        plan = self._preview_result
        display = self._display_plan
        busy = self._run_in_progress or self._checking_plan
        valid = plan is not None and bool(plan.items)
        parameter_error = self.parameter_override_editor.error_message
        can_check = (
            self._actions is not None
            and not busy
            and not getattr(self, "_run_preparing", False)
            and not parameter_error
            and not self._representative_pending
        )
        section = self.tabs.currentIndex()
        report_available = (
            self.results_panel.has_run_report and not valid and section != 0
        )
        if section == 1 and getattr(self, "_item_reveal_pending", False):
            self._item_reveal_pending = False
            self._review_result_item(self._current_item)
        name = self._loaded_config_path
        self.config_name_label.setText(name.name if name else "Unsaved configuration")
        self.config_name_label.setToolTip(
            str(name) if name else "Not saved to a config"
        )
        self._refresh_compute_summary()
        inventory = self._check_rows
        item_count = (
            len(inventory)
            if inventory is not None
            else len(display.items)
            if display is not None
            else 0
        )
        self._sync_setup_summary()
        self._sync_override_presentation()
        self.tabs.setTabText(
            1,
            "Items && outputs"
            + (
                f" · {item_count:,}"
                + (
                    " files"
                    if inventory is not None and not self._check_is_sample_list
                    else ""
                )
                if inventory is not None or display is not None
                else ""
            ),
        )
        try:
            override_count = sum(
                len(entry.values) for entry in self.parameter_overrides()
            )
        except ValueError:
            override_count = 0
        self.tabs.setTabText(
            2, "Overrides" + (f" · {override_count:,}" if override_count else "")
        )
        self.override_empty_label.setVisible(
            not self.parameter_override_editor.configured
        )
        self.next_button.setVisible(
            not self._run_in_progress
            and not getattr(self, "_run_preparing", False)
            and not (section == 3 and valid)
        )
        self.next_button.setText(
            "Checking…"
            if self._checking_plan
            else "View run report"
            if report_available
            else "Check batch"
            if not valid
            else "Review items"
            if section == 0
            else "Continue to run"
        )
        self._update_next_button_icon()
        self._style_review_banner()
        self.next_button.setEnabled(not busy if report_available else bool(can_check))
        self.next_button.setToolTip(
            "Review the finished batch in Run & results. "
            "This does not check or run the batch again."
            if report_available
            else parameter_error
            or (
                "Check current inputs, settings and output destinations for a new run. "
                "This replaces the on-screen report, but does not process images or "
                "change output files."
                if section == 0 and self.results_panel.has_run_report
                else ""
            )
        )
        self.results_panel.set_check_action_state(
            needed=not valid
            and not self._run_in_progress
            and not getattr(self, "_run_preparing", False),
            enabled=bool(can_check),
            checking=self._checking_plan,
            reason=parameter_error,
        )
        # Both warning banners offer the same guarded, metadata-only check.
        results_check = self.results_panel.check_batch_button
        self.review_check_button.setVisible(not results_check.isHidden())
        self.review_check_button.setEnabled(results_check.isEnabled())
        self.review_check_button.setText(results_check.text())
        self.review_check_button.setToolTip(results_check.toolTip())
        self.run_button.setVisible(section == 3 and valid and not busy)
        work, kept = batch_work_counts(plan) if valid else (0, 0)
        self.run_button.setText(
            f"Run {work:,} items"
            if work
            else "Keep existing files"
            if valid
            else "Run batch"
        )
        self.run_button.setToolTip(
            parameter_error
            or (
                "Wait for the active sample preview to finish."
                if self._representative_pending
                else ""
            )
            or (
                f"{work:,} items to process; {kept:,} items keep all existing outputs. "
                "Inputs and output paths are checked again before writing. "
                "Ask before overwrite prompts at this step."
                if valid
                else "Check batch first."
            )
        )
        self.run_button.setEnabled(
            valid
            and not busy
            and not parameter_error
            and not self._representative_pending
            and self._actions is not None
        )
        self.footer_overrides_button.setVisible(section == 1 and valid and not busy)
        for controls in (
            self.existing_files_controls,
            self.results_panel.existing_files_controls,
        ):
            controls.set_plan(
                display,
                item_choices=self._item_file_policies,
                enabled=valid
                and not busy
                and not getattr(self, "_run_preparing", False),
                reset_enabled=not busy and not getattr(self, "_run_preparing", False),
            )
        self.footer_overrides_button.setEnabled(
            self.parameter_override_editor.configured
        )
        self.load_overrides_button.setEnabled(
            not busy and inventory is None and self.parameter_override_editor.configured
        )
        has_current = display is not None and bool(display.items)
        has_inventory = inventory is not None and bool(inventory)
        self.recheck_item_button.setEnabled(
            valid
            and not busy
            and self._actions is not None
            and self._actions.check_items is not None
        )
        self.preview_button.setEnabled(
            self._actions is not None and not busy and not parameter_error
        )
        self.preview_item_button.setEnabled(
            valid
            and not busy
            and not self._representative_pending
            and self._actions is not None
            and self._actions.preview_item is not None
            and bool(self.preview_table.selectionModel().selectedRows())
        )
        self.close_button.setText("Hide window" if self._run_in_progress else "Close")
        self.tabs.setTabEnabled(
            1,
            has_current
            or has_inventory
            or bool(self._item_file_policies)
            or self._checking_plan
            or self._batch_activity_state == "error",
        )
        self.tabs.setTabEnabled(
            2,
            inventory is None
            and (has_current or self.parameter_override_editor.configured),
        )
        self.tabs.setTabEnabled(
            3,
            has_current
            or bool(self._item_file_policies)
            or self._run_started_at is not None
            or getattr(self, "_run_preparing", False),
        )
        self.run_group.hide()
        self._update_run_recap()
        self._reserve_footer_height()

    def _layout_compute_summary(self) -> None:
        """Expose compute information once, in the toolbar or its overflow."""
        show_in_toolbar = self.width() >= 680
        self.compute_summary_label.setVisible(show_in_toolbar)
        self.compute_icon_label.setVisible(show_in_toolbar)
        if hasattr(self, "compute_info_action"):
            self.compute_info_separator.setVisible(not show_in_toolbar)
            self.compute_info_action.setVisible(not show_in_toolbar)

    def _refresh_compute_summary(self) -> None:
        label, detail = (
            "Compute · main toolbar",
            "Uses the main toolbar's compute settings.",
        )
        if self._actions is not None and self._actions.compute_summary is not None:
            label, detail = self._actions.compute_summary()
        self.compute_summary_label.setText(label)
        self.compute_summary_label.setToolTip(detail)
        self.compute_icon_label.setToolTip(detail)
        if not hasattr(self, "compute_info_action"):
            self.compute_info_separator = self.more_menu.addSeparator()
            self.compute_info_action = QAction("Compute details…", self.more_menu)
            self.more_menu.addAction(self.compute_info_action)
            self.compute_info_action.triggered.connect(
                lambda: self.show_workspace_activity(
                    self.compute_summary_label.toolTip(), state="info"
                )
            )
        self.compute_info_action.setToolTip(detail)
        self._layout_compute_summary()

    def _update_run_recap(self) -> None:
        plan = self._preview_result
        # The actionable banner already explains a stale plan. Avoid a second
        # instruction above it, especially while showing a previous run's results.
        self.run_recap_label.setVisible(plan is not None)
        if plan is None:
            return
        outputs = tuple(output for item in plan.items for output in item.outputs)
        self.run_recap_label.setText(
            f"{len(plan.items):,} items · {output_counts_text(outputs, plan.config)}\n"
            f"{self.existing_policy_combo.currentText()} · "
            + (
                "Continue after failures"
                if self.continue_checkbox.isChecked()
                else "Stop after an item failure"
            )
            + "\nRun checks once for file changes since the last Check."
        )

    def _next_batch_step(self) -> None:
        if self._checking_plan or self._run_in_progress:
            return
        if self._preview_result is None or not self._preview_result.items:
            if self.results_panel.has_run_report and self.tabs.currentIndex() != 0:
                self._show_run_report()
            else:
                self._check_batch()
        else:
            self.tabs.setCurrentIndex(1 if self.tabs.currentIndex() == 0 else 3)

    def _show_run_report(self) -> None:
        if not self.results_panel.has_run_report:
            return
        self.tabs.setCurrentIndex(3)
        self.results_panel.run_report.show()
        # Wait for the tab/layout change before positioning the report heading.
        QTimer.singleShot(0, self._scroll_to_run_report)

    def _scroll_to_run_report(self) -> None:
        if self.results_panel.has_run_report:
            top = self.results_panel.run_report.mapTo(self.run_page, QPoint(0, 0)).y()
            self.run_scroll.verticalScrollBar().setValue(max(0, top - 10))

    def _check_batch(self) -> bool:
        if (
            self._actions is None
            or self._run_in_progress
            or self._checking_plan
            or getattr(self, "_run_preparing", False)
            or self._representative_pending
        ):
            return False
        if self.parameter_override_editor.error_message:
            self._sync_workspace()
            return False
        self._checking_plan = True
        self._clear_check_progress()
        self._preview_result = None
        self.show_workspace_activity(
            "Checking source files and output destinations…",
            state="working",
            indeterminate=True,
            progress_text="Checking",
        )
        self._sync_workspace()
        try:
            if self._actions.check_batch is not None:
                accepted = self._actions.check_batch(
                    self.values(), self._ITEM_PAGE_SIZE
                )
                if accepted is False:
                    self._checking_plan = False
                    self._sync_workspace()
                    return False
                return True
            # Embedding/test contexts can supply the existing synchronous planner.
            for attempt in range(2):
                try:
                    result = self._actions.preview_batch(
                        self.values(), self._ITEM_PAGE_SIZE
                    )
                    break
                except BatchScientificPreflightError as exc:
                    if attempt == 0 and self.apply_axis_suggestion(exc):
                        continue
                    raise
            self.apply_preview_result(result, preview_representative=False)
            return True
        except Exception as exc:
            self._checking_plan = False
            self._show_preview_failure(str(exc))
            self._sync_workspace()
            return False

    def _recheck_selected_items(self) -> None:
        if self._preview_result is None or self._run_in_progress or self._checking_plan:
            return
        if self._actions is None or self._actions.check_items is None:
            return
        positions = tuple(sorted(self._checked_items or {self._current_item}))
        self._checking_plan = True
        self.show_workspace_activity(
            f"Rechecking {len(positions):,} selected item(s)…",
            state="working",
            indeterminate=True,
            progress_text="Checking",
        )
        self._sync_workspace()
        try:
            if self._actions.check_items(positions) is False:
                self._checking_plan = False
                self._sync_workspace()
        except Exception as exc:
            self.show_item_recheck_result(positions, str(exc), unchanged=False)

    def show_item_recheck_result(
        self, indices, message: str, *, unchanged: bool
    ) -> None:
        self._checking_plan = False
        if not unchanged:
            self._invalidate_preview_plan()
        self.preview_status.setText(message)
        self.show_workspace_activity(
            f"{len(indices):,} selected item(s) rechecked"
            if unchanged
            else "Check batch again before running",
            state="ready" if unchanged else "warning",
            tooltip=message,
        )
        self._sync_workspace()

    def _choose_existing_file_policy(self, policy: str) -> None:
        self.existing_policy_combo.setCurrentIndex(
            self.existing_policy_combo.findData(policy)
        )

    def _existing_file_policy_changed(self, *_args) -> None:
        """A policy choice does not invalidate checked source/content identities."""
        plan = self._preview_result
        if (
            plan is None
            or self._run_in_progress
            or getattr(self, "_checking_plan", False)
            or getattr(self, "_run_preparing", False)
        ):
            self._invalidate_preview_plan()
            return
        result = with_existing_file_policy(
            plan,
            self.existing_policy_combo.currentData(),
        )
        self._apply_file_policy_result(
            result, "Batch default updated · item choices retained."
        )

    def _apply_file_policy_result(self, result, message) -> None:
        self._preview_result = self._display_plan = result
        self._item_file_policies = result.config.item_file_policies
        self._loaded_config_path = None
        self.preview_status.setText(checked_output_message(result))
        self.results_panel.update_output_policy(result)
        self._render_items()
        self.show_workspace_activity(
            message + " Source checks retained.",
            state="warning" if result.collision_count else "ready",
        )
        self._sync_workspace()

    def _set_item_file_policy(self, position, policy) -> None:
        if (
            self._preview_result is None
            or self._run_in_progress
            or self._checking_plan
            or getattr(self, "_run_preparing", False)
        ):
            return
        try:
            result = with_item_file_policy(self._preview_result, position, policy)
        except ValueError as exc:
            self.show_workspace_activity(str(exc), state="warning")
            return
        self._apply_file_policy_result(
            result, f"File choice updated for item {position + 1}."
        )

    def _reset_item_file_policies(self) -> None:
        if (
            self._run_in_progress
            or self._checking_plan
            or getattr(self, "_run_preparing", False)
        ):
            return
        from dataclasses import replace

        self._item_file_policies = ()
        if self._preview_result is None:
            self._invalidate_preview_plan()
        else:
            result = with_output_policy_config(
                self._preview_result,
                replace(
                    self._preview_result.config,
                    item_file_policies=(),
                ),
            )
            self._apply_file_policy_result(
                result, "All items now use the batch default."
            )

    def _workspace_plan_applied(self, result) -> None:
        checking = self._checking_plan
        self._clear_check_progress()
        self._checking_plan = False
        prior = self._display_plan
        checked_keys = {
            self._plan_item_key(prior, index)
            for index in self._checked_items
            if prior is not None and 0 <= index < len(prior.items)
        } - {None}
        current_key = self._plan_item_key(prior, self._current_item)
        self._display_plan = result
        self._item_file_policies = result.config.item_file_policies
        self._run_started_at = None
        self._last_elapsed = 0.0
        self._elapsed_timer.stop()
        self._update_elapsed()
        self._item_page = 0
        self._item_run_states.clear()
        self._checked_items = {
            index
            for index in range(len(result.items))
            if self._plan_item_key(result, index) in checked_keys
        }
        if result.items and not self._checked_items:
            self._checked_items.add(0)
        self._current_item = next(
            (
                index
                for index in range(len(result.items))
                if current_key is not None
                and self._plan_item_key(result, index) == current_key
            ),
            0,
        )
        self._sync_override_checked_items()
        self.results_panel.set_plan(result)
        self._render_items()
        if checking:
            self.tabs.setCurrentIndex(1)
        self._sync_workspace()

    @staticmethod
    def _plan_item_key(plan, position: int) -> str | None:
        """Never remap selected samples solely by a row number or filename."""
        if plan is None or not 0 <= position < len(plan.items):
            return None
        item = plan.items[position]
        if item.parameter_override_source_item_key:
            return item.parameter_override_source_item_key
        if not plan.config.sources:
            return None
        source_id = plan.config.sources[0].node_id
        source = item.source_items.get(source_id)
        if source is None:
            return None
        from napari_vipp.core.batch_parameters import batch_source_item_override_key

        return batch_source_item_override_key(source_id, source)

    def _workspace_plan_invalidated(self) -> None:
        if not getattr(self, "_workspace_ready", False):
            return
        self._checking_plan = False
        self._clear_check_progress()
        self.preview_status.setText(
            "Settings changed. Check batch again before previewing or running. "
            "The table is the previous plan."
        )
        self._render_items()
        self.results_panel.invalidate_plan()
        self._sync_workspace()

    def _matching_item_positions(self) -> list[int]:
        plan = self._display_plan
        if plan is None:
            return []
        query = self.item_search.text().strip().casefold()
        mode = self.item_filter.currentIndex()
        positions = []
        for i, item in enumerate(plan.items):
            row = self._item_row(i)
            text = " ".join(
                [item.batch_id, *(str(path) for path in item.source_paths.values())]
            )
            issues = (
                any(
                    any(
                        word in str(status).lower()
                        for word in ("collision", "overlaps", "duplicate", "error")
                    )
                    for status in row.output_statuses
                )
                or self._preview_result is None
            )
            if query and query not in text.casefold():
                continue
            if (mode == 1 and not issues) or (mode == 2 and issues):
                continue
            positions.append(i)
        return positions

    def _item_row(self, position):
        from napari_vipp.ui.batch import BatchPreviewRow

        plan = self._display_plan
        item = plan.items[position]
        rows = getattr(self, "_display_rows_by_index", {})
        if item.index in rows:
            return rows[item.index]
        return BatchPreviewRow(
            batch_index=item.index,
            batch_id=item.batch_id,
            sources=item.source_paths,
            outputs=[out.path for out in item.outputs],
            output_statuses=tuple(out.status_text for out in item.outputs),
            source_labels={key: item.source_label(key) for key in item.source_paths},
        )

    def _render_items(self) -> None:
        if not getattr(self, "_workspace_ready", False):
            return
        if self._check_rows is not None:
            self._render_check_items()
            return
        self._display_rows_by_index = (
            {row.batch_index: row for row in self._display_plan.rows}
            if self._display_plan is not None
            else {}
        )
        positions = self._matching_item_positions()
        page_size = self._ITEM_PAGE_SIZE
        self._item_page = min(
            self._item_page, max((len(positions) - 1) // page_size, 0)
        )
        visible = positions[
            self._item_page * page_size : (self._item_page + 1) * page_size
        ]
        self._rendering_items = True
        blocker = QSignalBlocker(self.preview_table)
        try:
            self.preview_table.setColumnCount(6)
            self.preview_table.setHorizontalHeaderLabels(
                ["#", "Batch item", "Sources", "Outputs", "Checks", "Run result"]
            )
            self.preview_table.setRowCount(len(visible))
            self._preview_table_rows.clear()
            for row_index, position in enumerate(visible):
                row = self._item_row(position)
                self._preview_table_rows[row.batch_index] = row_index
                first = QTableWidgetItem(str(row.batch_index))
                first.setData(Qt.UserRole, position)
                first.setFlags(first.flags() | Qt.ItemIsUserCheckable)
                first.setCheckState(
                    Qt.Checked if position in self._checked_items else Qt.Unchecked
                )
                self.preview_table.setItem(row_index, 0, first)
                item = self._display_plan.items[position]
                config = self._display_plan.config
                disposition = planned_item_status(item.outputs, config)
                checks = (
                    "Blocked"
                    if disposition == "Blocked"
                    else "Review"
                    if disposition == "Needs decision"
                    else "Ready"
                )
                if self._preview_result is None:
                    checks = "Needs recheck"
                values = [
                    row.batch_id,
                    f"{len(row.sources)} paired"
                    if len(row.sources) > 1
                    else "1 source",
                    output_counts_text(item.outputs, config, compact=True),
                    checks,
                    self._item_run_states.get(row.batch_index, disposition),
                ]
                for col, value in enumerate(values, 1):
                    cell = QTableWidgetItem(value)
                    cell.setToolTip(
                        "\n".join(str(path) for path in row.outputs)
                        if col == 3
                        else value
                    )
                    self.preview_table.setItem(row_index, col, cell)
                    if col == 5 and item_file_choice(config, item) is not None:
                        font = QFont(self.preview_table.font())
                        font.setBold(True)
                        cell.setFont(font)
                        cell.setToolTip(item_file_choice_label(config, item))
                if position == self._current_item:
                    self.preview_table.selectRow(row_index)
            header = self.preview_table.horizontalHeader()
            for col in (0, 2, 3, 4, 5):
                header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            self.preview_table.resizeRowsToContents()
            self._style_item_statuses()
        finally:
            del blocker
            self._rendering_items = False
        if visible:
            start = self._item_page * page_size + 1
            count_text = f"{start:,}–{start + len(visible) - 1:,} of {len(positions):,}"
        else:
            count_text = (
                "No matching items" if self._display_plan else "No items checked yet"
            )
        hidden_checked = len(self._checked_items.difference(positions))
        count_text += f" · {len(self._checked_items):,} selected"
        if hidden_checked:
            count_text += f" ({hidden_checked:,} hidden by filters)"
        self.item_range_label.setText(count_text)
        multiple = len(positions) > page_size
        self.items_previous_button.setVisible(multiple)
        self.items_next_button.setVisible(multiple)
        self.items_previous_button.setEnabled(self._item_page > 0)
        self.items_next_button.setEnabled(
            (self._item_page + 1) * page_size < len(positions)
        )
        self._show_item_details()

    def _style_item_statuses(self) -> None:
        tones = theme_colors(self.palette())
        blocker = QSignalBlocker(self.preview_table)
        for row in range(self.preview_table.rowCount()):
            for column in (4, 5):
                cell = self.preview_table.item(row, column)
                if cell is None:
                    continue
                text = cell.text().casefold()
                if text in {"ready", "completed", "saved"}:
                    color = tones.success.foreground
                elif any(
                    word in text for word in ("error", "failed", "collision", "blocked")
                ):
                    color = tones.error.foreground
                elif column == 4 or text in {
                    "partial",
                    "cancelled",
                    "skipped",
                    "needs decision",
                    "will overwrite",
                }:
                    color = tones.warning.foreground
                elif text in {
                    "running",
                    "processing",
                    "keep existing",
                    "create missing",
                }:
                    color = tones.info.foreground
                else:
                    color = tones.text
                cell.setForeground(color)
        del blocker

    def _filter_items(self, *_args) -> None:
        self._item_page = 0
        self._render_items()
        self._sync_workspace()

    def _page_items(self, step: int) -> None:
        self._item_page = max(self._item_page + step, 0)
        self._render_items()
        self._sync_workspace()

    def _item_check_changed(self, item) -> None:
        if self._rendering_items or self._check_rows is not None or item.column() != 0:
            return
        position = int(item.data(Qt.UserRole))
        if item.checkState() == Qt.Checked:
            self._checked_items.add(position)
        else:
            self._checked_items.discard(position)
        self._sync_override_checked_items()
        self._render_items()

    def _show_item_details(self) -> None:
        if self._check_rows is not None:
            self._show_check_item_details()
            return
        if self._rendering_items or self._display_plan is None:
            return
        selected = self.preview_table.selectionModel().selectedRows()
        if selected:
            first = self.preview_table.item(selected[0].row(), 0)
            self._current_item = int(first.data(Qt.UserRole))
        if not 0 <= self._current_item < len(self._display_plan.items):
            self.item_details.clear()
            return
        row = self._item_row(self._current_item)
        source_titles = {
            str(source["node_id"]): str(source["title"]) for source in self._source_rows
        }
        colors = custom_paint_colors(self.palette())
        muted = colors.muted_text.name()
        link_color = theme_colors(self.palette()).info.foreground
        icon_palette = QPalette(self.palette())
        icon_palette.setColor(QPalette.ButtonText, link_color)
        self.item_details.document().addResource(
            QTextDocument.ImageResource,
            QUrl("vipp:reveal-source"),
            toolbar_icon("open", icon_palette).pixmap(20, 20).toImage(),
        )
        parts = [
            f'<p style="color: {muted}; margin-bottom: 4px;">Selected item</p>',
            '<p style="margin-top: 0; margin-bottom: 6px;">'
            f"<b>{escape(row.batch_id)}</b></p>",
            f'<p style="color: {muted};">Item {self._current_item + 1:,} '
            f"of {len(self._display_plan.items):,}</p>",
        ]
        for i, (node_id, path) in enumerate(row.sources.items()):
            title = source_titles.get(node_id, node_id)
            parts.append(
                f'<p style="margin-bottom: 8px;"><b>{escape(title)}</b>'
                f"<br>{escape(str(path))}</p>"
            )
            parts.append('<p style="margin-top: 8px; margin-bottom: 18px;">')
            if Path(path).exists():
                parts.append(
                    f'<a href="source:{i}" style="text-decoration: none; '
                    f'color: {link_color.name()};">'
                    '<img src="vipp:reveal-source" width="20" height="20" '
                    'style="vertical-align: middle;">&nbsp;&nbsp;'
                    f"{escape(self._file_reveal_label())}</a>"
                )
            else:
                parts.append("File is not currently available")
            parts.append("</p>")
        item = self._display_plan.items[self._current_item]
        bindings = {
            source.node_id: (
                source.axis_declaration.display_text
                if source.axis_declaration is not None
                else "Use file labels"
            )
            for source in self._display_plan.config.sources
        }
        parts.append(
            "<p><b>Image axes</b><br>"
            + "<br>".join(
                escape(
                    f"{source_titles.get(key, key)}: "
                    f"{bindings.get(key) or 'Use file labels'}"
                )
                for key in row.sources
            )
            + "</p>"
        )
        parts.append(
            "<p><b>Parameter overrides</b><br>"
            + (
                "<br>".join(
                    escape(f"{value.node_id} / {value.parameter}: {value.value}")
                    for value in item.parameter_overrides
                )
                or "Workflow values"
            )
            + "</p>"
        )
        parts.append(
            f'<hr style="color: {colors.border.name()};">'
            f"<p><b>{len(row.outputs)} output"
            f"{'s' if len(row.outputs) != 1 else ''}</b><br>"
            f"{escape(output_counts_text(item.outputs, self._display_plan.config))}</p>"
            '<ul style="margin-top: 0; margin-bottom: 0;">'
        )
        if item_file_choice(self._display_plan.config, item) is not None:
            parts.insert(
                len(parts) - 1,
                "<p><b>Existing files</b><br>"
                + escape(item_file_choice_label(self._display_plan.config, item))
                + "</p>",
            )
        show_filenames = getattr(self, "_show_output_filenames", False)
        titles = [output.node_title for output in item.outputs]
        for output in item.outputs:
            title = output.node_title or output.node_id
            if titles.count(output.node_title) > 1:
                title += f" ({output.node_id})"
            format_label = {
                "ome-tiff": "OME-TIFF",
                "ome-zarr": "OME-Zarr",
            }.get(output.format, output.format.upper())
            detail = f"{output.kind.capitalize()} · {format_label}"
            detail += (
                " · "
                + {
                    "create": "To create",
                    "keep": "Keep existing",
                    "overwrite": "Will overwrite",
                    "ask": "Needs decision when you run",
                    "blocked": "Blocked destination",
                }[output_action(output, self._display_plan.config)]
            )
            parts.append(
                '<li style="margin-top: 0; margin-bottom: 6px;">'
                f"<b>{escape(title)}</b><br>"
                f'<span style="color: {muted};">{escape(detail)}</span>'
            )
            if show_filenames:
                parts.append(f"<br>{escape(str(output.path))}")
            parts.append("</li>")
        parts.append("</ul>")
        parts.append(
            '<p style="margin-top: 8px; margin-bottom: 0;">'
            '<a href="output-names:toggle">'
            + ("Hide file paths" if show_filenames else "Show file paths")
            + "</a></p>"
        )
        self.item_details.setHtml("".join(parts))

    @staticmethod
    def _file_reveal_label() -> str:
        from napari_vipp.ui.file_reveal import file_reveal_label

        return file_reveal_label()

    def _reveal_source(self, source_index: int) -> None:
        from napari_vipp.ui.file_reveal import reveal_file

        if self._display_plan is None:
            return
        paths = tuple(self._item_row(self._current_item).sources.values())
        if not 0 <= source_index < len(paths):
            return
        try:
            result = reveal_file(paths[source_index])
            if not result.success or not result.selected:
                self.show_workspace_activity(
                    result.message, state="info" if result.success else "warning"
                )
        except (OSError, ValueError) as exc:
            self.show_workspace_activity(str(exc), state="warning")

    def _item_detail_link(self, url) -> None:
        if url.scheme() == "output-names":
            self._show_output_filenames = not getattr(
                self, "_show_output_filenames", False
            )
            self._show_item_details()
        elif url.scheme() == "source":
            try:
                self._reveal_source(int(url.path()))
            except ValueError:
                return

    def _item_context_menu(self, point) -> None:
        if self._check_rows is not None:
            return
        row_index = self.preview_table.rowAt(point.y())
        if row_index < 0:
            return
        self.preview_table.selectRow(row_index)
        self._show_item_details()
        menu = QMenu(self.preview_table)
        recheck = menu.addAction("Recheck this item")
        recheck.setIcon(self.recheck_item_button.icon())
        recheck.setEnabled(self.recheck_item_button.isEnabled())
        recheck.triggered.connect(self._recheck_current_item)
        preview = menu.addAction("Preview in graph")
        preview.setIcon(self.preview_item_button.icon())
        preview.setEnabled(self.preview_item_button.isEnabled())
        preview.triggered.connect(self._preview_selected_item)
        override = menu.addAction("Load overrides")
        override.setIcon(self.load_overrides_button.icon())
        override.setEnabled(self.load_overrides_button.isEnabled())
        override.triggered.connect(
            lambda: self._open_selected_overrides(current_only=True)
        )
        self._add_item_file_policy_actions(menu, self._current_item)
        menu.addSeparator()
        sources = tuple(self._item_row(self._current_item).sources.items())
        if len(sources) == 1:
            reveal = menu.addAction(self._file_reveal_label())
            reveal.setIcon(toolbar_icon("open", self.palette()))
            reveal.setEnabled(Path(sources[0][1]).exists())
            reveal.triggered.connect(lambda: self._reveal_source(0))
        else:
            reveal_menu = menu.addMenu(self._file_reveal_label())
            reveal_menu.setIcon(toolbar_icon("open", self.palette()))
            for i, (name, path) in enumerate(sources):
                action = reveal_menu.addAction(f"{name}: {Path(path).name}")
                action.setEnabled(Path(path).exists())
                action.triggered.connect(
                    lambda _checked=False, index=i: self._reveal_source(index)
                )
        menu.exec(self.preview_table.viewport().mapToGlobal(point))

    def _add_item_file_policy_actions(self, menu, position) -> None:
        """Target only the clicked row, never the independent checkbox selection."""
        # Native Windows menus render addSection() as an unlabeled separator.
        # Keep both the scope and current choice explicit with napari's theme.
        menu.addSeparator()
        heading = menu.addAction("Existing outputs · this item")
        heading.setEnabled(False)
        heading_font = QFont(menu.font())
        heading_font.setBold(True)
        heading.setFont(heading_font)
        menu.setToolTipsVisible(True)
        item = self._display_plan.items[position]
        config = self._display_plan.config
        choice = item_file_choice(config, item)
        policy = choice.policy.value if choice else None
        protected_nodes = {
            output.node_id for output in config.outputs if output.overwrite == "no"
        }
        protected = any(
            output.duplicate
            or output.input_collision
            or (output.exists and output.node_id in protected_nodes)
            for output in item.outputs
        )
        enabled = (
            self._preview_result is not None
            and not self._run_in_progress
            and not self._checking_plan
            and not getattr(self, "_run_preparing", False)
            and batch_item_file_policy_key(config, item) is not None
        )
        group = QActionGroup(menu)
        for value, label, icon in (
            ("skip", "Keep existing outputs", "open"),
            ("overwrite", "Rerun and overwrite outputs", "refresh"),
            (None, "Use batch default", "undo"),
        ):
            action = menu.addAction(label + (" (current)" if value == policy else ""))
            action.setIcon(toolbar_icon(icon, self.palette()))
            action.setData(("item_file_policy", value))
            action.setCheckable(True)
            group.addAction(action)
            action.setChecked(value == policy)
            action.setEnabled(enabled and not (value == "overwrite" and protected))
            action.setToolTip(
                "An output path is protected or conflicts with another destination "
                "or input. Resolve that conflict before choosing overwrite."
                if value == "overwrite" and protected
                else "Only this item changes. Missing outputs are still created. "
                "No files change until Run. Protected output paths remain protected."
            )
            action.triggered.connect(
                lambda _checked=False, selected=value, index=position: (
                    self._set_item_file_policy(index, selected)
                )
            )

    def _recheck_current_item(self) -> None:
        previous = self._checked_items
        self._checked_items = {self._current_item}
        try:
            self._recheck_selected_items()
        finally:
            self._checked_items = previous

    def _open_selected_overrides(self, _checked=False, *, current_only=False) -> None:
        editor = self.parameter_override_editor
        if not editor.configured or self._run_in_progress or self._checking_plan:
            return
        positions = (
            {self._current_item}
            if current_only
            else (self._checked_items or {self._current_item})
        )
        self._syncing_selection = True
        try:
            keys = [
                self._plan_item_key(self._display_plan, position)
                for position in sorted(positions)
            ]
            if any(key not in editor._source_keys for key in keys):
                self.show_workspace_activity(
                    "Check batch again to match these samples to their overrides.",
                    state="warning",
                )
                return
            editor.set_selected_source_keys(keys)
            if keys:
                editor.select_source(keys[0])
        finally:
            self._syncing_selection = False
        self._checked_items = set(positions)
        self.tabs.setCurrentIndex(2)

    def _override_source_selected(self, source_key: str, position: int) -> None:
        if self._syncing_selection:
            return
        if self._display_plan is not None:
            for position in range(len(self._display_plan.items)):
                if self._plan_item_key(self._display_plan, position) == source_key:
                    self._current_item = position
                    self._item_reveal_pending = True
                    break

    def _override_selection_changed(self, keys) -> None:
        if self._syncing_selection:
            return
        if self._display_plan is None:
            return
        key_set = set(keys)
        self._checked_items = {
            i
            for i in range(len(self._display_plan.items))
            if self._plan_item_key(self._display_plan, i) in key_set
        }
        self._render_items()

    def _sync_override_checked_items(self) -> None:
        """Share only exact reviewed identities across both sample tables."""
        editor = self.parameter_override_editor
        if not editor.configured or self._display_plan is None:
            return
        available = set(editor._source_keys)
        keys = [
            key
            for index in sorted(self._checked_items)
            if (key := self._plan_item_key(self._display_plan, index)) in available
        ]
        if set(keys) == set(editor.selected_source_keys()):
            return
        self._syncing_selection = True
        try:
            editor.set_selected_source_keys(keys)
        finally:
            self._syncing_selection = False

    def _review_result_item(self, position: int) -> None:
        self._current_item = position
        with QSignalBlocker(self.item_search):
            self.item_search.clear()
        with QSignalBlocker(self.item_filter):
            self.item_filter.setCurrentIndex(0)
        self._item_page = max(position, 0) // self._ITEM_PAGE_SIZE
        self._render_items()
        self.tabs.setCurrentIndex(1)

    def _update_elapsed(self) -> None:
        elapsed = self._last_elapsed
        if self._run_started_at is not None and self._run_in_progress:
            elapsed = monotonic() - self._run_started_at
        seconds = max(int(elapsed), 0)
        self.footer_elapsed_label.setText(
            f"{seconds // 60:02d}:{seconds % 60:02d} elapsed"
            if self._run_started_at is not None
            else ""
        )
        # Detailed timing is always available in Run & results. At compact
        # widths prioritize status, progress, and the Stop/Close controls.
        self.footer_elapsed_label.setVisible(
            self._run_started_at is not None and self.width() >= 760
        )

    def _workspace_run_started(self, total: int) -> None:
        self._run_started_at = monotonic()
        self._last_elapsed = 0.0
        self._checking_plan = False
        self._item_run_states = (
            {item.index: "Pending" for item in self._display_plan.items}
            if self._display_plan is not None
            else {}
        )
        self.results_panel.begin_run(total)
        self._elapsed_timer.start()
        self._update_elapsed()
        self.tabs.setCurrentIndex(3)
        self.cancel_run_button.setText("Stop safely")
        self._sync_workspace()

    def _workspace_run_finished(self) -> None:
        # A finished run is evidence to review, not authorization to run again.
        # Keep the displayed items/configuration, but require an explicit new check.
        if self.results_panel.has_run_report:
            self._preview_result = None
            self.run_button.setEnabled(False)
        if self._run_started_at is not None:
            self._last_elapsed = monotonic() - self._run_started_at
        self._elapsed_timer.stop()
        self._update_elapsed()
        self._sync_workspace()
        self._show_run_report()
