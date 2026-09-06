"""Guided, cancellable drafts for applying parameters to selected samples."""

from __future__ import annotations

from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtGui import QFont, QPalette
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.palette_roles import custom_paint_colors, theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


def _label(text, *, bold=False):
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    if bold:
        label.setStyleSheet("font-weight: bold;")
    return label


def _display_number(value):
    """Readable reference text only; never round editable or serialized values."""
    return str(value) if isinstance(value, int) else format(float(value), ".12g")


class BatchBulkEditDialog(QDialog):
    """Choose parameters, draft changes, then apply once to frozen targets."""

    def __init__(self, owner):
        self._ready = False
        super().__init__(owner)
        self.owner = owner
        self.targets = owner.selected_source_keys()
        self.bindings = {binding.key: binding for binding in owner._parameters}
        self.draft_fields = {}
        count = len(self.targets)
        noun = "sample" if count == 1 else "samples"
        self.setWindowTitle(f"Edit overrides for {count:,} selected {noun}")
        self.setMinimumSize(540, 480)
        available = self.screen().availableGeometry()
        self.resize(min(980, available.width() - 80), min(720, available.height() - 80))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        self.scope_label = _label(
            f"Apply the same changes to {count:,} selected {noun}."
        )
        scope = (
            "Only chosen parameters change. Other overrides and your workflow "
            "stay unchanged."
        )
        hidden = len(set(self.targets).difference(owner._matching_keys))
        other_pages = len(set(self.targets).difference(owner._page_keys)) - hidden
        if hidden:
            scope += f" {hidden:,} selected samples hidden by filters are included."
        if other_pages:
            scope += f" {other_pages:,} selected samples on other pages are included."
        self.scope_detail = _label(scope)
        layout.addWidget(self.scope_label)
        layout.addWidget(self.scope_detail)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        self.picker = QFrame()
        self.picker.setObjectName("BulkParameterPicker")
        picker = QVBoxLayout(self.picker)
        picker.setContentsMargins(12, 10, 12, 10)
        picker.setSpacing(8)
        picker.addWidget(_label("1 · Choose parameters", bold=True))
        self.search_label = _label("Find node or parameter")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search workflow parameters…")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Find node or parameter")
        self.search_label.setBuddy(self.search)
        picker.addWidget(self.search_label)
        picker.addWidget(self.search)
        self.parameter_tree, self.parameter_items = owner._parameter_tree(self)
        self.parameter_tree.setMinimumSize(0, 120)
        self.parameter_tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.parameter_tree.setUniformRowHeights(True)
        for key, item in self.parameter_items.items():
            binding = self.bindings[key]
            # The picker names parameters; values and inheritance belong in
            # the change cards, so selecting a parameter has an obvious next step.
            item.setText(0, binding.parameter.label)
            item.setToolTip(0, owner._parameter_title(binding))
        picker.addWidget(self.parameter_tree, 1)
        self.search_empty_label = _label("No matching parameters. Try another search.")
        self.search_empty_label.hide()
        picker.addWidget(self.search_empty_label)
        self.selection_summary = _label("No parameters chosen")
        picker.addWidget(self.selection_summary)
        self.splitter.addWidget(self.picker)

        self.changes = QFrame()
        self.changes.setObjectName("BulkParameterChanges")
        changes = QVBoxLayout(self.changes)
        changes.setContentsMargins(12, 10, 12, 10)
        changes.setSpacing(8)
        changes.addWidget(_label("2 · Define changes", bold=True))
        self.change_help = _label(
            "Set a value for every selected sample, or use the workflow value "
            "to remove their overrides for that parameter."
        )
        changes.addWidget(self.change_help)
        self.fields_scroll = QScrollArea()
        self.fields_scroll.setWidgetResizable(True)
        self.fields_scroll.setFrameShape(QFrame.NoFrame)
        self.fields_scroll.setMinimumSize(0, 160)
        self.fields = QWidget()
        self.fields_layout = QVBoxLayout(self.fields)
        self.fields_layout.setContentsMargins(0, 0, 0, 0)
        self.fields_layout.setSpacing(10)
        self.empty_state = _label(
            "Choose a parameter to begin\n\n"
            "Check a parameter in the list. Its value controls will appear here."
        )
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.fields_layout.addWidget(self.empty_state, 1)
        self.fields_layout.addStretch()
        self.fields_scroll.setWidget(self.fields)
        changes.addWidget(self.fields_scroll, 1)
        self.splitter.addWidget(self.changes)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, 1)

        self.feedback_label = _label("")
        layout.addWidget(self.feedback_label)
        footer = QHBoxLayout()
        footer.addStretch()
        self.cancel_button = ToolbarCommandButton("Cancel")
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button = ToolbarCommandButton(f"Apply to {count:,} {noun}")
        self.apply_button.setAutoDefault(False)
        self.apply_button.setObjectName("BulkApplyChanges")
        self.apply_button.clicked.connect(self._apply_draft)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.apply_button)
        layout.addLayout(footer)
        self.parameter_tree.itemChanged.connect(self._picks_changed)
        self.search.textChanged.connect(self._filter_parameters)
        self._ready = True
        self._apply_theme()
        self._resize_panels()
        self._validate_draft()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._ready:
            self._resize_panels()

    def _resize_panels(self):
        orientation = Qt.Vertical if self.width() < 780 else Qt.Horizontal
        if self.splitter.orientation() != orientation:
            self.splitter.setOrientation(orientation)
            self.splitter.setSizes([300, 400])

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if self._ready and event.type() in (
            QEvent.PaletteChange,
            QEvent.StyleChange,
            QEvent.FontChange,
        ):
            self._apply_theme()

    def _apply_theme(self):
        colors = custom_paint_colors(self.palette())
        tones = theme_colors(self.palette())
        card_style = (
            "QFrame#BulkParameterPicker, QFrame#BulkParameterChanges, "
            "QFrame#BulkParameterDraft {"
            f"border: 1px solid {colors.border.name()}; border-radius: 3px; }}"
        )
        for card in (self.picker, self.changes):
            card.setStyleSheet(card_style)
        for label in (self.scope_detail, self.change_help, self.selection_summary):
            label.setStyleSheet(f"color: {colors.muted_text.name()};")
        self.scope_label.setStyleSheet("font-weight: bold;")
        self.empty_state.setStyleSheet(
            f"color: {colors.muted_text.name()}; padding: 12px;"
        )
        self.parameter_tree.setStyleSheet(
            "QTreeView { border: 0; "
            f"background: {colors.surface.name()}; color: {colors.text.name()}; }}"
            "QTreeView::item { padding: 5px 4px; }"
            "QTreeView::indicator:unchecked {"
            f"border: 1px solid {colors.muted_text.name()}; border-radius: 2px; "
            f"background: {colors.surface.name()}; }}"
            "QTreeView::item:selected {"
            f"background: {tones.info.surface.name()}; "
            f"color: {tones.info.foreground.name()}; }}"
        )
        palette = QPalette(self.palette())
        palette.setColor(QPalette.ButtonText, tones.info.accent)
        for i in range(self.parameter_tree.topLevelItemCount()):
            group = self.parameter_tree.topLevelItem(i)
            font = QFont(self.font())
            font.setBold(True)
            group.setFont(0, font)
            group.setIcon(0, toolbar_icon("workflow", palette))
        self.apply_button.setIcon(toolbar_icon("checklist", self.palette()))
        self.apply_button.setIconSize(QSize(18, 18))
        self.apply_button.setStyleSheet(
            "QPushButton#BulkApplyChanges:enabled {"
            f"border: 1px solid {tones.info.accent.name()}; font-weight: bold; }}"
        )

    def _filter_parameters(self, query):
        self.owner._filter_parameter_tree(self.parameter_items, query)
        self.search_empty_label.setVisible(
            all(item.isHidden() for item in self.parameter_items.values())
        )
        self._validate_draft()

    def _draft_values(self):
        values = {}
        for key, (_row, mode, editor) in self.draft_fields.items():
            if mode.currentData() == "inherit":
                values[key] = None
            else:
                if not editor.text().strip():
                    raise ValueError(
                        "Enter a value for "
                        f"{self.owner._parameter_title(self.bindings[key])}, "
                        "or choose Use workflow value."
                    )
                values[key] = self.owner._parse_value(
                    editor.text().strip(), self.bindings[key]
                )
        return values

    def _validate_draft(self):
        error = ""
        try:
            self._draft_values()
        except (TypeError, ValueError) as exc:
            error = str(exc)
        count = len(self.draft_fields)
        hidden = sum(self.parameter_items[key].isHidden() for key in self.draft_fields)
        chosen = f"{count:,} parameter{'s' if count != 1 else ''} chosen"
        if hidden:
            chosen += f" · {hidden:,} hidden by search (still included)"
        self.selection_summary.setText(chosen if count else "No parameters chosen")
        self.empty_state.setVisible(not count)
        self.feedback_label.setText(
            error
            or (
                f"Ready to apply {count:,} parameter change(s) "
                f"to {len(self.targets):,} samples."
                if count
                else "Choose parameters in step 1, then enter their values in step 2."
            )
        )
        tones = theme_colors(self.palette())
        self.feedback_label.setStyleSheet(
            f"color: {(tones.error if error else tones.info).foreground.name()};"
        )
        self.apply_button.setEnabled(bool(self.targets and count) and not error)

    def _picks_changed(self, item, _column):
        key = item.data(0, Qt.UserRole)
        if key is None:
            return
        key = tuple(key)
        if item.checkState(0) != Qt.Checked:
            previous = self.draft_fields.pop(key, None)
            if previous:
                self.fields_layout.removeWidget(previous[0])
                previous[0].deleteLater()
            self._validate_draft()
            return
        if key in self.draft_fields:
            return
        binding = self.bindings[key]
        row = QFrame()
        row.setObjectName("BulkParameterDraft")
        form = QFormLayout(row)
        form.setContentsMargins(10, 10, 10, 10)
        form.setSpacing(8)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow(_label(self.owner._display_node_label(binding), bold=True))
        form.addRow(_label(binding.parameter.label))
        workflow = _display_number(binding.workflow_value)
        current = {
            self.owner._values.get((source, *key), "") for source in self.targets
        }
        inherited = sum(
            not self.owner._values.get((source, *key), "").strip()
            for source in self.targets
        )
        current_text = (
            "Mixed values across selected samples"
            if len(current) > 1
            else f"Current value: {next(iter(current), '') or workflow}"
        )
        reference = _label(
            f"{current_text}\nWorkflow value: {workflow} · "
            f"{inherited:,} samples using workflow"
        )
        reference.setToolTip(f"Exact workflow value: {binding.workflow_value!r}")
        form.addRow(reference)
        mode = QComboBox()
        mode.addItem("Set same value", "set")
        mode.addItem("Use workflow value", "inherit")
        mode.setAccessibleName(f"Action for {self.owner._parameter_title(binding)}")
        editor = self.owner._make_editor(binding)
        editor.setPlaceholderText("Enter a value")
        editor.setAccessibleName(self.owner._parameter_title(binding))
        editor.setToolTip(
            "Required for Set same value. A blank entry cannot be applied here. "
            "To remove existing overrides, choose Use workflow value instead."
        )
        form.addRow("Action", mode)
        form.addRow("New value", editor)
        row.inherit_notice = _label(
            f"Use workflow value: {workflow}. "
            "Existing overrides for this parameter will be removed."
        )
        form.addRow(row.inherit_notice)
        row.inherit_notice.hide()
        self.draft_fields[key] = (row, mode, editor)
        self.fields_layout.insertWidget(self.fields_layout.count() - 1, row)
        mode.currentIndexChanged.connect(
            lambda _index: self._mode_changed(mode, editor, workflow)
        )
        editor.textChanged.connect(self._validate_draft)
        self._validate_draft()

    def _mode_changed(self, mode, editor, workflow):
        setting = mode.currentData() == "set"
        editor.setEnabled(setting)
        editor.setVisible(setting)
        row = editor.parentWidget()
        row.layout().labelForField(editor).setVisible(setting)
        row.inherit_notice.setVisible(not setting)
        editor.setPlaceholderText(
            "Enter a value" if setting else f"Workflow: {workflow}"
        )
        self._validate_draft()

    def _apply_draft(self):
        try:
            self.owner.apply_selected_values(
                self._draft_values(), source_keys=self.targets
            )
        except (TypeError, ValueError) as exc:
            self.feedback_label.setText(str(exc))
            return
        self.accept()
