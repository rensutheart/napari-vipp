"""Override page sections and explicit whole-batch reset scope."""

from __future__ import annotations

from qtpy.QtCore import QSignalBlocker, QSize, Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.batch_setup import BatchDisclosureButton
from napari_vipp.ui.palette_roles import (
    blend_colors,
    custom_paint_colors,
    palette_is_dark,
    theme_colors,
)
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class BatchOverridePresentation:
    """Keep reset controls separate from sample selection and authored values."""

    def _build_overrides_page(self) -> None:
        self.overrides_page = QWidget()
        layout = QVBoxLayout(self.overrides_page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        heading = QHBoxLayout()
        title = QLabel("Adjust batch parameters")
        title.setObjectName("BatchPageHeading")
        heading.addWidget(title, 1)
        self.reset_all_overrides_button = ToolbarCommandButton("Reset all overrides…")
        self.reset_all_overrides_button.setIconSize(QSize(18, 18))
        self.reset_all_overrides_button.setToolTip(
            "Restore workflow values for every sample and every Run/Bypass choice, "
            "including values hidden by filters or column choices. "
            "The original workflow and sample selection stay unchanged."
        )
        self.reset_all_overrides_button.clicked.connect(self._reset_all_batch_overrides)
        heading.addWidget(self.reset_all_overrides_button)
        layout.addLayout(heading)
        self.override_empty_label = QLabel(
            "Check batch first to match editable parameters to exact samples. "
            "Only supported numeric parameters are exposed here."
        )
        self.override_empty_label.setWordWrap(True)
        layout.addWidget(self.override_empty_label)

        self.parameter_override_group.setTitle("")
        self.parameter_override_group.setObjectName("BatchParameterOverrideCard")
        self.parameter_override_group.setAccessibleName(
            "Per-sample parameter overrides"
        )
        section = QHBoxLayout()
        section.setSpacing(7)
        self.parameter_section_icon = QLabel()
        self.parameter_section_icon.setFixedSize(18, 18)
        self.parameter_section_title = QLabel("Per-sample parameter overrides")
        self.parameter_override_count_label = QLabel("0 overrides")
        section.addWidget(self.parameter_section_icon)
        section.addWidget(self.parameter_section_title, 1)
        section.addWidget(self.parameter_override_count_label)
        self.parameter_override_group.layout().insertLayout(0, section)
        # The matrix keeps useful working space; the page scrolls instead of
        # squeezing it when the node section is expanded below it.
        self.parameter_override_editor.table.setMinimumHeight(260)
        layout.addWidget(self.parameter_override_group, 1)

        self.node_behavior_card = QGroupBox()
        self.node_behavior_card.setObjectName("BatchNodeBehaviorCard")
        self.node_behavior_card.setAccessibleName("Run or bypass nodes")
        self.node_behavior_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        node_layout = QVBoxLayout(self.node_behavior_card)
        node_layout.setContentsMargins(
            self.parameter_override_group.layout().contentsMargins()
        )
        node_layout.setSpacing(10)
        node_heading = QHBoxLayout()
        self.node_behavior_toggle = BatchDisclosureButton("Run or bypass nodes")
        self.node_behavior_toggle.setObjectName("BatchNodeBehaviorDisclosure")
        self.node_behavior_toggle.setIconSize(QSize(18, 18))
        self.node_behavior_toggle.setCheckable(True)
        self.node_behavior_toggle.setFlat(True)
        self.node_behavior_toggle.setAccessibleName(
            "Run or bypass nodes for all samples"
        )
        self.node_behavior_toggle.toggled.connect(self._toggle_node_behavior)
        self.node_override_count_label = QLabel("All samples · 0 node overrides")
        node_heading.addWidget(self.node_behavior_toggle)
        node_heading.addStretch(1)
        node_heading.addWidget(self.node_override_count_label)
        self.node_behavior_toggle.setVisible(bool(self._node_execution_specs))
        self.node_override_count_label.setVisible(bool(self._node_execution_specs))
        node_layout.addLayout(node_heading)
        self.node_execution_group.setTitle("")
        self.node_execution_group.setObjectName("BatchNodeBehaviorContents")
        self.node_execution_group.setAccessibleName("Whole-batch node behavior")
        self.node_execution_group.layout().setContentsMargins(0, 0, 0, 0)
        self.node_execution_group.layout().setRowWrapPolicy(
            QFormLayout.WrapLongRows
        )
        self.node_execution_group.hide()
        node_layout.addWidget(self.node_execution_group)
        self.node_behavior_card.setVisible(bool(self._node_execution_specs))
        layout.addWidget(self.node_behavior_card)
        self.overrides_scroll = self._workspace_scroll(self.overrides_page)
        self.overrides_scroll.setObjectName("BatchOverridesScroll")
        self.overrides_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tabs.addTab(self.overrides_scroll, "Overrides")

    def _override_reset_counts(self) -> tuple[int, int]:
        editor = self.parameter_override_editor
        cells = editor.override_value_count()
        if self._pending_parameter_overrides:
            cells = max(
                cells,
                sum(len(entry.values) for entry in self._pending_parameter_overrides),
            )
        return cells, len(self.node_execution_overrides())

    def _sync_override_presentation(self) -> None:
        cells, nodes = self._override_reset_counts()
        self.parameter_override_count_label.setText(
            f"{cells:,} parameter override" + ("" if cells == 1 else "s")
        )
        self.node_override_count_label.setText(
            f"All samples · {nodes:,} node override" + ("" if nodes == 1 else "s")
        )
        self.reset_all_overrides_button.setEnabled(
            bool(cells or nodes)
            and not self._checking_plan
            and not self._run_in_progress
        )
        self._style_node_execution_choices()

    def _style_node_execution_choices(self) -> None:
        """Distinguish explicit batch intent, not calculation/result status."""
        palette = self.palette()
        tones = theme_colors(palette)
        for combo in self._node_execution_combos.values():
            mode = combo.currentData()
            tone = {"run": tones.active_mode, "bypass": tones.bypass}.get(mode)
            # Reserve the same border space in every state. Keep inherited
            # controls neutral, even when the workflow itself bypasses a node.
            style = (
                "QComboBox { border: 1px solid transparent; "
                "border-left: 3px solid transparent; border-radius: 2px; "
                "padding: 3px 10px 3px 8px; }"
            )
            if tone is not None:
                edge = "dotted" if mode == "bypass" else "solid"
                hover = blend_colors(tone.surface, tone.accent, 0.08)
                style += (
                    "QComboBox { "
                    f"background-color: {tone.surface.name()}; "
                    f"color: {tone.foreground.name()}; "
                    f"border-color: {tone.border.name()}; "
                    f"border-left: 3px {edge} {tone.accent.name()}; }}"
                    "QComboBox:hover { "
                    f"background-color: {hover.name()}; }}"
                )
            # An explicit choice remains recognizable while editing is locked
            # during a check/run; dim it without changing the layout or meaning.
            disabled_background = tone.surface if tone else tones.raised_surface
            disabled_text = tone.foreground if tone else tones.text
            disabled_text = blend_colors(disabled_background, disabled_text, 0.72)
            style += (
                "QComboBox:focus { "
                f"border-top-color: {tones.text.name()}; "
                f"border-right-color: {tones.text.name()}; "
                f"border-bottom-color: {tones.text.name()}; }}"
                "QComboBox:disabled { "
                f"background-color: {disabled_background.name()}; "
                f"color: {disabled_text.name()}; }}"
                # Do not tint all popup options as if they shared the selection.
                "QComboBox QAbstractItemView { "
                f"background-color: {tones.surface.name()}; "
                f"color: {tones.text.name()}; "
                f"selection-background-color: {tones.info.surface.name()}; "
                f"selection-color: {tones.info.foreground.name()}; }}"
            )
            # Workspace progress syncs frequently; repolish only actual changes.
            if combo.styleSheet() != style:
                combo.setStyleSheet(style)

    def _reset_all_batch_overrides(self) -> bool:
        cells, nodes = self._override_reset_counts()
        if self._checking_plan or self._run_in_progress or not (cells or nodes):
            return False
        answer = QMessageBox.warning(
            self,
            "Reset all batch overrides?",
            f"Reset {cells:,} per-sample parameter value(s) across all samples and "
            f"{nodes:,} Run/Bypass node choice(s)?\n\n"
            "This includes hidden rows and columns. Every value will use its "
            "workflow setting again. Your sample selection, source paths, output "
            "settings, and original workflow will not change.\n\n"
            "Check batch again before running.",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return False
        editor = self.parameter_override_editor
        blockers = [QSignalBlocker(editor)]
        blockers.extend(
            QSignalBlocker(combo) for combo in self._node_execution_combos.values()
        )
        had_pending = bool(self._pending_parameter_overrides)
        try:
            editor.reset_all_overrides(confirm=False)
            if had_pending and not editor.configured:
                editor.clear_contract()
            self._pending_parameter_overrides = ()
            for combo in self._node_execution_combos.values():
                combo.setCurrentIndex(0)
        finally:
            blockers.clear()
        self._loaded_config_path = None
        self._invalidate_preview_plan()
        self.parameterOverridesChanged.emit(())
        self.nodeExecutionOverridesChanged.emit(())
        self.show_workspace_activity(
            "All batch overrides reset · check batch again before running.",
            state="info",
        )
        self._sync_workspace()
        return True

    def _apply_override_presentation_theme(self) -> None:
        colors = custom_paint_colors(self.palette())
        tones = theme_colors(self.palette())
        accent = (
            tones.info.accent
            if palette_is_dark(self.palette())
            else tones.info.foreground
        )
        icon_palette = QPalette(self.palette())
        icon_palette.setColor(QPalette.ButtonText, accent)
        self.parameter_section_icon.setPixmap(
            toolbar_icon("batch", icon_palette).pixmap(18, 18)
        )
        self.node_behavior_toggle.setIcon(toolbar_icon("workflow", icon_palette))
        self.reset_all_overrides_button.setIcon(toolbar_icon("reset", self.palette()))
        self.parameter_section_title.setStyleSheet("font-weight: bold;")
        for label in (
            self.parameter_override_count_label,
            self.node_override_count_label,
        ):
            label.setStyleSheet(f"color: {colors.muted_text.name()};")
        self.parameter_override_group.setStyleSheet(
            "QGroupBox#BatchParameterOverrideCard { "
            f"border: 1px solid {colors.border.name()}; border-radius: 3px; "
            "margin-top: 0; padding-top: 0; }"
        )
        self.node_behavior_card.setStyleSheet(
            "QGroupBox#BatchNodeBehaviorCard { "
            f"border: 1px solid {colors.border.name()}; border-radius: 3px; "
            "margin-top: 0; padding-top: 0; }"
            "QGroupBox#BatchNodeBehaviorContents { "
            "border: 0; margin: 0; padding: 0; }"
        )
        self.node_behavior_toggle.setStyleSheet(
            "QPushButton#BatchNodeBehaviorDisclosure, "
            "QPushButton#BatchNodeBehaviorDisclosure:checked { "
            "background: transparent; border: 1px solid transparent; "
            "padding: 2px 0; text-align: left; font-weight: bold; "
            f"color: {colors.text.name()}; }}"
            "QPushButton#BatchNodeBehaviorDisclosure:hover { "
            f"background: {colors.alternate_surface.name()}; }}"
            "QPushButton#BatchNodeBehaviorDisclosure:focus { "
            f"border: 1px solid {accent.name()}; }}"
        )
        self._style_node_execution_choices()
