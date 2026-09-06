"""A compact human-readable view of a finished run's recorded evidence."""

from __future__ import annotations

from datetime import datetime

from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QFormLayout, QFrame, QHBoxLayout, QLabel, QVBoxLayout

from napari_vipp.ui.batch_setup import BatchDisclosureButton
from napari_vipp.ui.palette_roles import custom_paint_colors, theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


def _status(record) -> str:
    return str(getattr(record.status, "value", record.status)).lower()


def _finished_at(value: str) -> str:
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone()
            .strftime("%d %b %Y · %H:%M:%S %Z")
        )
    except (ValueError, TypeError):
        return "Not reported"


class BatchRunReport(QFrame):
    """No disk reads, validation, or execution: this is the returned run record."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BatchRunReport")
        self._tone = "success"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        heading = QHBoxLayout()
        self.icon = QLabel()
        heading.addWidget(self.icon)
        self.title_label = self._label("Run report")
        self.heading_labels = [self.title_label]
        self.title_label.setStyleSheet("font-weight: bold;")
        heading.addWidget(self.title_label, 1)
        self.elapsed_label = self._label("")
        heading.addWidget(self.elapsed_label)
        layout.addLayout(heading)
        self.outcome_label = self._label("")
        layout.addWidget(self.outcome_label)
        facts = QFormLayout()
        facts.setHorizontalSpacing(22)
        facts.setVerticalSpacing(6)
        facts.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        facts.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.fields = {}
        for name in ("Items", "Output files", "Finished", "Saved to"):
            key = self._label(name)
            key.setStyleSheet("font-weight: bold;")
            self.heading_labels.append(key)
            value = self._label("")
            self.fields[name] = value
            facts.addRow(key, value)
        layout.addLayout(facts)
        self.issues_heading = self._label("Failure reasons and run notes")
        self.heading_labels.append(self.issues_heading)
        layout.addWidget(self.issues_heading)
        self.issues_label = self._label("")
        layout.addWidget(self.issues_label)
        self.details_toggle = BatchDisclosureButton("Show all details")
        self.details_toggle.setCheckable(True)
        self.details_toggle.toggled.connect(self._show_issues)
        self._issues = ()
        layout.addWidget(self.details_toggle, 0, Qt.AlignLeft)
        self.evidence_label = self._label(
            "Recorded at run completion. Select an item below for its outputs "
            "and details."
        )
        layout.addWidget(self.evidence_label)
        technical = QHBoxLayout()
        self.manifest_button = ToolbarCommandButton("Find manifest JSON")
        technical.addWidget(self.manifest_button)
        self.technical_label = self._label(
            "Technical record of inputs, settings and results — "
            "for reproducibility and troubleshooting."
        )
        technical.addWidget(self.technical_label, 1)
        layout.addLayout(technical)
        self.hide()
        self._apply_theme()

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setTextFormat(Qt.PlainText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        return label

    def set_result(self, result, *, title: str, elapsed: str, tone: str, note: str):
        manifest = result.manifest
        records = tuple(manifest.items)
        outputs = [
            output for item in records for output in getattr(item, "outputs", ())
        ]
        saved = len(result.saved_paths)
        overwritten = sum(
            _status(output) == "completed"
            and getattr(output, "overwrote_existing", False)
            for output in outputs
        )
        kept = sum(
            _status(output) == "skipped"
            and bool(
                getattr(output, "existed_at_preflight", False)
                or getattr(output, "existing_identity", {})
            )
            for output in outputs
        )
        not_written = sum(_status(output) != "completed" for output in outputs) - kept
        failed_outputs = sum(_status(output) == "failed" for output in outputs)
        cancelled_outputs = sum(_status(output) == "cancelled" for output in outputs)
        self._tone = tone
        self.title_label.setText("Run report")
        self.elapsed_label.setText(elapsed)
        self.outcome_label.setText(
            "Batch finished · existing outputs kept. No samples needed processing."
            if records
            and kept == len(outputs)
            and kept
            and all(_status(item) == "skipped" for item in records)
            else title
        )
        summary = result.summary
        counts = [
            f"{summary[state]:,} {state}"
            for state in ("completed", "partial", "skipped", "cancelled", "failed")
            if summary.get(state)
        ]
        unfinished = sum(_status(item) in ("pending", "running") for item in records)
        if unfinished:
            counts.append(f"{unfinished:,} unfinished")
        self.fields["Items"].setText(
            f"{len(records):,} total · " + (" · ".join(counts) or "None processed")
        )
        file_counts = [f"{saved:,} saved this run"]
        if overwritten:
            file_counts[0] += f" ({overwritten:,} overwritten)"
        if kept:
            file_counts.append(f"{kept:,} existing kept")
        if failed_outputs:
            file_counts.append(f"{failed_outputs:,} failed")
        if cancelled_outputs:
            file_counts.append(f"{cancelled_outputs:,} cancelled")
        other_unwritten = not_written - failed_outputs - cancelled_outputs
        if other_unwritten:
            file_counts.append(f"{other_unwritten:,} not written")
        if any(not hasattr(item, "outputs") for item in records):
            file_counts.append("Output details not reported")
        self.fields["Output files"].setText(" · ".join(file_counts))
        self.fields["Finished"].setText(
            _finished_at(getattr(manifest, "finished_at", ""))
        )
        destination = str(getattr(manifest, "output_dir", "")) or "Not reported"
        self.fields["Saved to"].setText(destination)
        self.fields["Saved to"].setToolTip(destination)
        issues = []
        compute = getattr(manifest, "compute", {})
        if not compute.get("runtime_cleanup_succeeded", True):
            issues.append(
                "Runtime cleanup failed. Restart VIPP before calculating again "
                "or changing compute policy."
            )
        issues.extend(str(warning) for warning in compute.get("warnings", ()))
        for item in records:
            item_error = getattr(item, "error_message", "")
            # Group identical reasons for this item, retaining the affected node
            # titles. Put the reason first, so a long sample name cannot hide it.
            errors = {item_error: []} if item_error else {}
            for output in getattr(item, "outputs", ()):
                if _status(output) not in ("failed", "partial"):
                    continue
                reason = (
                    getattr(output, "error_message", "")
                    or item_error
                    or getattr(output, "error_type", "")
                    or "No failure reason was recorded."
                )
                name = getattr(output, "node_title", "") or getattr(
                    output, "node_id", "Output"
                )
                errors.setdefault(reason, []).append(name)
            if not errors and _status(item) in ("failed", "partial"):
                errors[
                    getattr(item, "error_type", "") or "No failure reason was recorded."
                ] = []
            for reason, node_names in errors.items():
                name = getattr(item, "batch_id", f"Item {item.index}")
                context = f"Item {item.index} · {name}"
                if node_names:
                    context += " · " + ", ".join(dict.fromkeys(node_names))
                issues.append(f"{reason}\n{context}")
        if note:
            issues.append(note)
        self._issues = tuple(dict.fromkeys(issues))
        self.details_toggle.setChecked(False)
        self._show_issues()
        self._apply_theme()
        self.show()

    def _show_issues(self, *_args):
        expanded = self.details_toggle.isChecked()
        shown = self._issues if expanded else self._issues[:3]
        if not expanded:
            shown = tuple(
                message[:237] + "…" if len(message) > 240 else message
                for message in shown
            )
        self.issues_heading.setVisible(bool(self._issues))
        self.issues_label.setVisible(bool(self._issues))
        self.issues_label.setText("\n\n".join(shown))
        self.issues_label.setToolTip("\n\n".join(self._issues))
        self.details_toggle.setVisible(
            len(self._issues) > 3 or any(len(message) > 240 for message in self._issues)
        )
        self.details_toggle.setText(
            "Show fewer details"
            if expanded
            else f"Show all details ({len(self._issues):,})"
        )

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.FontChange,
        ):
            self._apply_theme()

    def _apply_theme(self):
        if not hasattr(self, "technical_label") or getattr(
            self, "_applying_theme", False
        ):
            return
        self._applying_theme = True
        try:
            self._update_theme()
        finally:
            self._applying_theme = False

    def _update_theme(self):
        tone = getattr(theme_colors(self.palette()), self._tone)
        muted = custom_paint_colors(self.palette()).muted_text.name()
        parent = self.parentWidget()
        font = parent.font() if parent is not None else self.font()
        font_size = (
            f"{font.pointSizeF():g}pt"
            if font.pointSizeF() > 0
            else f"{font.pixelSize()}px"
        )
        font_style = f"font-size: {font_size};"
        # Set explicit label sizes: napari's inherited QSS otherwise shrinks
        # them independently of the tables and the user's chosen UI font.
        for label in self.findChildren(QLabel):
            label.setStyleSheet(font_style)
        for label in self.heading_labels:
            label.setStyleSheet(font_style + "font-weight: bold;")
        self.setStyleSheet(
            "QFrame#BatchRunReport {"
            f"border: 1px solid {tone.border.name()}; "
            f"border-left: 3px solid {tone.accent.name()};"
            "border-radius: 4px; }"
        )
        self.outcome_label.setStyleSheet(
            font_style + f"color: {tone.foreground.name()}; font-weight: bold;"
        )
        for label in (self.evidence_label, self.technical_label):
            label.setStyleSheet(font_style + f"color: {muted};")
        self.icon.setPixmap(toolbar_icon("activity", self.palette()).pixmap(18, 18))
        self.manifest_button.setIcon(toolbar_icon("open", self.palette()))
        self.details_toggle.setIcon(toolbar_icon("activity", self.palette()))
