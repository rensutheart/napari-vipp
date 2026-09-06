"""Source-aware Setup presentation without changing batch planning semantics."""

from __future__ import annotations

from qtpy.QtCore import QPointF, QSize, Qt
from qtpy.QtGui import QPainter, QPalette, QPen
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.palette_roles import (
    custom_paint_colors,
    palette_is_dark,
    theme_colors,
)
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class BatchDisclosureButton(ToolbarCommandButton):
    """A labelled disclosure with room for the shared-style chevron."""

    def sizeHint(self):  # noqa: N802
        return super().sizeHint() + QSize(18, 0)

    def _disclosure_label_option(self):
        option = self._toolbar_style_option()
        option.rect.adjust(0, 0, -18, 0)
        return option

    def paintEvent(self, event):  # noqa: N802
        del event
        painter = QStylePainter(self)
        painter.drawControl(QStyle.CE_PushButtonBevel, self._toolbar_style_option())
        painter.drawControl(QStyle.CE_PushButtonLabel, self._disclosure_label_option())
        painter.setRenderHint(QPainter.Antialiasing)
        group = QPalette.Active if self.isEnabled() else QPalette.Disabled
        pen = QPen(self.palette().color(group, QPalette.ButtonText), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        x, y = self.width() - 10, self.height() / 2
        points = (
            ((x - 3, y - 1.5), (x, y + 1.5), (x + 3, y - 1.5))
            if self.isChecked()
            else ((x - 1.5, y - 3), (x + 1.5, y), (x - 1.5, y + 3))
        )
        painter.drawLine(QPointF(*points[0]), QPointF(*points[1]))
        painter.drawLine(QPointF(*points[1]), QPointF(*points[2]))


class BatchSetupPresentation:
    """Use live workflow identity and checked-plan evidence in Setup cards."""

    def _build_setup_page(self, output_form) -> None:
        self.setup_page = QWidget()
        setup = QVBoxLayout(self.setup_page)
        setup.setContentsMargins(10, 10, 10, 10)
        setup.setSpacing(12)
        heading_row = QHBoxLayout()
        heading = QLabel("Sources and destination")
        heading.setObjectName("BatchPageHeading")
        heading_row.addWidget(heading, 1)
        self.setup_state_label = QLabel("Not checked")
        self.setup_state_label.setAccessibleName("Setup check status")
        heading_row.addWidget(self.setup_state_label)
        setup.addLayout(heading_row)

        self.workflow_context = QFrame()
        self.workflow_context.setObjectName("BatchWorkflowContext")
        self.workflow_context_grid = QGridLayout(self.workflow_context)
        self.workflow_context_grid.setContentsMargins(0, 4, 0, 10)
        self.workflow_context_grid.setHorizontalSpacing(20)
        self.workflow_identity = QWidget()
        identity = QHBoxLayout(self.workflow_identity)
        identity.setContentsMargins(0, 0, 0, 0)
        identity.setSpacing(7)
        self.workflow_icon_label = QLabel()
        self.workflow_icon_label.setFixedSize(18, 18)
        identity.addWidget(self.workflow_icon_label)
        self.workflow_caption = QLabel("Workflow")
        identity.addWidget(self.workflow_caption)
        self.workflow_summary_label = QLabel("Current workflow")
        self.workflow_summary_label.setWordWrap(True)
        self.workflow_summary_label.setMinimumWidth(0)
        self.workflow_summary_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        identity.addWidget(self.workflow_summary_label, 1)
        self.sources_stat = QWidget()
        self.outputs_stat = QWidget()
        self.sources_caption, self.sources_summary_label = self._setup_stat(
            self.sources_stat, "Sources per item"
        )
        self.outputs_caption, self.outputs_summary_label = self._setup_stat(
            self.outputs_stat, "Outputs per item"
        )
        setup.addWidget(self.workflow_context)
        setup.addWidget(self.demo_guide_label)
        setup.addWidget(self.demo_path_row)

        self.setup_columns = QGridLayout()
        self.setup_columns.setHorizontalSpacing(12)
        self.setup_columns.setVerticalSpacing(12)
        self.setup_columns.setColumnStretch(0, 1)
        self.setup_columns.setColumnStretch(1, 1)
        self.source_card = QGroupBox()
        self.source_card.setObjectName("BatchSourcesCard")
        source_card_layout = QVBoxLayout(self.source_card)
        source_card_layout.setContentsMargins(10, 10, 10, 10)
        source_card_layout.setSpacing(9)
        self.source_heading, self.source_section_icon, self.source_section_title = (
            self._setup_heading("Sources & pairing")
        )
        self.source_count_label = QLabel("")
        self.source_heading.layout().addWidget(self.source_count_label)
        source_card_layout.addWidget(self.source_heading)
        self.source_group.setTitle("")
        self.source_group.setObjectName("BatchSourcesBody")
        self.source_layout.setContentsMargins(0, 0, 0, 0)
        self.source_layout.setSpacing(9)
        source_card_layout.addWidget(self.source_group)
        self.pairing_banner = QFrame()
        self.pairing_banner.setObjectName("BatchPairingBanner")
        pairing = QHBoxLayout(self.pairing_banner)
        pairing.setContentsMargins(8, 7, 8, 7)
        self.pairing_icon = QLabel()
        self.pairing_icon.setFixedSize(18, 18)
        pairing.addWidget(self.pairing_icon, 0, Qt.AlignTop)
        self.pairing_label = QLabel()
        self.pairing_label.setWordWrap(True)
        self.pairing_label.setMinimumWidth(0)
        self.pairing_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        pairing.addWidget(self.pairing_label, 1)
        source_card_layout.addWidget(self.pairing_banner)
        self.setup_columns.addWidget(self.source_card, 0, 0, Qt.AlignTop)

        self.destination_group = QGroupBox()
        self.destination_group.setObjectName("BatchDestinationCard")
        self.destination_group.setLayout(output_form)
        output_form.setContentsMargins(10, 10, 10, 10)
        output_form.setVerticalSpacing(7)
        output_form.setRowWrapPolicy(output_form.WrapAllRows)
        (
            self.destination_heading,
            self.destination_section_icon,
            self.destination_section_title,
        ) = self._setup_heading("Destination & run policy")
        output_form.insertRow(0, self.destination_heading)
        self.artifacts_toggle = BatchDisclosureButton("Files saved with the results")
        self.artifacts_toggle.setCheckable(True)
        self.artifacts_toggle.setIconSize(QSize(18, 18))
        self.artifacts_toggle.setAccessibleName("Files saved with the results")
        self.artifacts_toggle.setToolTip(
            "Show workflow and Python script output options"
        )
        self.artifacts_content = QWidget()
        artifacts = QVBoxLayout(self.artifacts_content)
        artifacts.setContentsMargins(0, 4, 0, 0)
        artifacts.addWidget(self.workflow_checkbox)
        artifacts.addWidget(self.script_checkbox)
        self.artifacts_help = QLabel(
            "Workflow JSON is required. The optional Python script lets you "
            "rerun this batch."
        )
        self.artifacts_help.setWordWrap(True)
        self.artifacts_help.setMinimumWidth(0)
        artifacts.addWidget(self.artifacts_help)
        self.artifacts_content.hide()
        self.artifacts_toggle.toggled.connect(self.artifacts_content.setVisible)
        output_form.addRow(self.artifacts_toggle)
        output_form.addRow(self.artifacts_content)
        self.setup_columns.addWidget(self.destination_group, 0, 1, Qt.AlignTop)
        setup.addLayout(self.setup_columns)
        self.help_label.setText(
            "Check batch reads source information and plans outputs. "
            "It does not calculate images or save results."
        )
        setup.addWidget(self.help_label)
        setup.addStretch(1)
        self.content_widget = self.setup_page
        self.content_scroll = self._workspace_scroll(self.setup_page)
        self.content_scroll.setObjectName("BatchWorkspaceScroll")
        self.tabs.addTab(self.content_scroll, "Setup")
        self._responsive_setup_context()

    @staticmethod
    def _setup_heading(text):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(7)
        icon = QLabel()
        icon.setFixedSize(18, 18)
        layout.addWidget(icon)
        title = QLabel(text)
        title.setWordWrap(True)
        layout.addWidget(title, 1)
        return row, icon, title

    @staticmethod
    def _setup_stat(parent, caption):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(caption)
        value = QLabel()
        value.setWordWrap(True)
        value.setMinimumWidth(0)
        layout.addWidget(label)
        layout.addWidget(value)
        return label, value

    def _responsive_setup_context(self) -> None:
        narrow = self.width() < 680
        grid = self.workflow_context_grid
        for widget in (self.workflow_identity, self.sources_stat, self.outputs_stat):
            grid.removeWidget(widget)
        grid.addWidget(self.workflow_identity, 0, 0, 1, 2 if narrow else 1)
        grid.addWidget(self.sources_stat, 1 if narrow else 0, 0 if narrow else 1)
        grid.addWidget(self.outputs_stat, 1 if narrow else 0, 1 if narrow else 2)
        grid.setColumnStretch(0, 1 if narrow else 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0 if narrow else 1)

    def _sync_setup_summary(self) -> None:
        name, detail = (
            "Current workflow",
            "Uses the current workflow in the main VIPP window.",
        )
        if self._actions is not None and self._actions.workflow_summary is not None:
            name, detail = self._actions.workflow_summary()
        self.workflow_summary_label.setText(name)
        self.workflow_summary_label.setToolTip(detail)
        self.workflow_summary_label.setAccessibleDescription(detail)
        source_count = len(self._source_rows)
        self.sources_summary_label.setText(
            f"{source_count:,} source" + ("s" if source_count != 1 else "")
        )
        plan = self._preview_result
        checked = plan is not None and bool(plan.items) and not self._checking_plan
        issues = checked and bool(plan.collision_count)
        state = (
            "Checking…"
            if self._checking_plan
            else "Needs attention"
            if issues or self._batch_activity_state in {"error", "warning"}
            else "Checked"
            if checked
            else "Not checked"
        )
        self.setup_state_label.setText(state)
        self.source_count_label.setText(f"{len(plan.items):,} items" if checked else "")
        if checked:
            output_text, source_counts = self._setup_checked_counts(plan)
            self.outputs_summary_label.setText(output_text)
            self.pairing_label.setText(
                f"{len(plan.items):,} batch items · "
                + (
                    "sources paired by sorted position."
                    if source_count > 1
                    else "one source item per batch item."
                )
            )
        else:
            self.outputs_summary_label.setText("Check batch to determine")
            self.pairing_label.setText(
                "Collections are paired by sorted position. Check batch to review "
                "the exact source files and destinations."
            )
        self.pairing_label.setToolTip(
            "Pairing uses sorted collection position, not matching filenames or "
            "biological identity. Multi-series files may contribute multiple "
            "source items."
        )
        for row in self._source_rows:
            label = row.get("count_label")
            if label is None:
                continue
            if not checked:
                label.setText("Checking…" if self._checking_plan else "Not checked")
                continue
            label.setText(source_counts.get(row["node_id"], "Not checked"))
        self._apply_setup_theme()

    def _setup_checked_counts(self, plan):
        # A checked result is immutable. Do not rescan thousands of source
        # entries whenever a node progress update refreshes the workspace.
        if getattr(self, "_setup_counts_plan", None) is plan:
            return self._setup_counts_value
        rows = {row.batch_index: row for row in plan.rows}
        counts = []
        source_paths = {}
        for item in plan.items:
            count = len(item.outputs)
            if not count and item.index in rows:
                count = len(rows[item.index].outputs)
            counts.append(count)
            for node_id, path in item.source_paths.items():
                source_paths.setdefault(node_id, []).append(path)
        lower, upper = min(counts), max(counts)
        outputs = str(lower) if lower == upper else f"{lower}–{upper}"
        output_text = f"{outputs} file" + ("s" if upper != 1 else "")
        source_counts = {}
        for node_id, paths in source_paths.items():
            files = len(set(paths))
            text = f"{files:,} file" + ("s" if files != 1 else "")
            if len(paths) != files:
                text += f" · {len(paths):,} source items"
            source_counts[node_id] = text
        self._setup_counts_plan = plan
        self._setup_counts_value = output_text, source_counts
        return self._setup_counts_value

    def _apply_setup_theme(self) -> None:
        colors = custom_paint_colors(self.palette())
        tones = theme_colors(self.palette())
        accent_palette = QPalette(self.palette())
        accent_palette.setColor(
            QPalette.ButtonText,
            tones.info.accent
            if palette_is_dark(self.palette())
            else tones.info.foreground,
        )
        for widget, kind in (
            (self.workflow_icon_label, "workflow"),
            (self.source_section_icon, "images"),
            (self.destination_section_icon, "destination"),
            (self.pairing_icon, "arrange"),
        ):
            widget.setPixmap(toolbar_icon(kind, accent_palette).pixmap(18, 18))
        self.output_button.setIcon(toolbar_icon("open", self.palette()))
        self.output_button.setIconSize(QSize(18, 18))
        self.artifacts_toggle.setIcon(toolbar_icon("archive", self.palette()))
        self.artifacts_toggle.setPalette(self.palette())
        self.artifacts_toggle.setStyleSheet(
            "text-align: left; padding: 6px 23px 6px 5px; border: 0; "
            f"border-top: 1px solid {colors.border.name()};"
        )
        self.workflow_context.setStyleSheet(
            "QFrame#BatchWorkflowContext { border: 0; "
            f"border-bottom: 1px solid {colors.border.name()}; }}"
        )
        card_style = (
            f"background: {colors.alternate_surface.name()}; "
            f"border: 1px solid {colors.border.name()}; border-radius: 4px;"
        )
        self.source_card.setStyleSheet(
            f"QGroupBox#BatchSourcesCard {{ {card_style} }} "
            "QGroupBox#BatchSourcesBody { border: 0; margin: 0; padding: 0; "
            "background: transparent; }"
        )
        self.destination_group.setStyleSheet(
            f"QGroupBox#BatchDestinationCard {{ {card_style} }}"
        )
        self.source_heading.setStyleSheet("background: transparent;")
        self.destination_heading.setStyleSheet("background: transparent;")
        for label in (
            self.workflow_caption,
            self.sources_caption,
            self.outputs_caption,
            self.source_count_label,
            self.artifacts_help,
        ):
            label.setStyleSheet(f"color: {colors.muted_text.name()};")
        for label in (
            self.workflow_summary_label,
            self.sources_summary_label,
            self.outputs_summary_label,
            self.source_section_title,
            self.destination_section_title,
        ):
            label.setStyleSheet(f"color: {colors.text.name()}; font-weight: bold;")
        state = self.setup_state_label.text()
        tone = (
            tones.success
            if state == "Checked"
            else tones.warning
            if state == "Needs attention"
            else tones.info
        )
        self.setup_state_label.setStyleSheet(
            f"color: {tone.foreground.name()}; background: {tone.surface.name()}; "
            "padding: 4px 7px; border-radius: 7px;"
        )
        self.pairing_banner.setStyleSheet(
            "QFrame#BatchPairingBanner { "
            f"background: {tones.info.surface.name()}; border: 0; border-radius: 0; }}"
        )
        self.pairing_label.setStyleSheet(f"color: {tones.info.foreground.name()};")
        for row in self._source_rows:
            if (label := row.get("count_label")) is not None:
                color = (
                    tones.success.foreground
                    if state == "Checked"
                    else colors.muted_text
                )
                label.setStyleSheet(f"color: {color.name()};")
