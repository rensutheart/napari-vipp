"""Presentation of original-reference checks; never a source of run authority."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from html import escape

from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
)

from napari_vipp.core.reproduction import current_vipp_version, versions_match
from napari_vipp.ui.batch_reproduction_details import ReproductionCheckDetailsDialog
from napari_vipp.ui.palette_roles import theme_colors

_CHECK_LABELS = {
    "matched": "Verified",
    "changed": "Changed input",
    "missing": "Missing input",
    "extra": "Unexpected input",
    "ambiguous": "Ambiguous match",
    "unreadable": "Unreadable input",
    "selector-mismatch": "Different image selection",
}


@dataclass(frozen=True)
class _CompletedReproduction:
    """Display-only record; never usable as a runnable preflight check."""

    request: object
    text: str
    success: bool


class BatchReproductionPresentation:
    """Retain explicit mode and keep incomplete comparisons non-runnable."""

    def _build_reproduction_banner(self):
        self._completed_reproduction = None
        self.reproduction_banner = QFrame()
        self.reproduction_banner.setObjectName("BatchReproductionBanner")
        layout = QHBoxLayout(self.reproduction_banner)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(16)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        self.reproduction_mode_label = QLabel()
        self.reproduction_status_label = QLabel()
        for label in (self.reproduction_mode_label, self.reproduction_status_label):
            label.setTextFormat(Qt.PlainText)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            label.setWordWrap(True)
            text_layout.addWidget(label)
        layout.addLayout(text_layout, 1)
        action_layout = QVBoxLayout()
        action_layout.setSpacing(4)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.reproduction_details_button = QPushButton("Check details…")
        self.reproduction_details_button.clicked.connect(
            self._show_reproduction_details
        )
        self.reproduction_new_data_button = QPushButton("Use with new data…")
        self.reproduction_new_data_button.clicked.connect(
            self._use_reproduction_new_data
        )
        actions.addWidget(self.reproduction_details_button)
        actions.addWidget(self.reproduction_new_data_button)
        action_layout.addLayout(actions)
        self.reproduction_version_button = QPushButton(
            "Acknowledge version difference…"
        )
        self.reproduction_version_button.clicked.connect(
            self._acknowledge_reproduction_version
        )
        action_layout.addWidget(self.reproduction_version_button)
        layout.addLayout(action_layout)
        layout.setAlignment(action_layout, Qt.AlignRight | Qt.AlignVCenter)
        self.reproduction_banner.hide()
        self._reproduction_table_active = False

    def set_reproduction_request(self, request, *, package_context=False):
        """Install a host-selected mode without silently choosing a new one."""
        self._reproduction_request = request
        self._reproduction_context = bool(request is not None or package_context)
        self._reproduction_table_active = False
        self._invalidate_preview_plan()
        self._sync_reproduction_banner()

    def _reproduction_check(self):
        result = self._preview_result
        if (
            self._reproduction_request is None
            or result is None
            or result.config.reproduction != self._reproduction_request
        ):
            return None
        return result.reproduction

    def _record_completed_reproduction(self, result):
        """Retain the execution manifest's verdict, not the earlier UI check."""
        self._completed_reproduction = None
        request = self._reproduction_request
        record = getattr(result.manifest, "reproduction", None)
        if (
            request is None
            or request.mode != "reproduce"
            or not isinstance(record, Mapping)
            or record.get("reference_sha256") != request.reference.digest
        ):
            return
        matched = record.get("matched_count")
        mismatches = record.get("mismatch_count")
        rows = record.get("rows")
        verified = (
            record.get("status") == "verified"
            and record.get("can_run") is True
            and type(matched) is int
            and matched > 0
            and type(mismatches) is int
            and mismatches == 0
            and not record.get("problems")
            and isinstance(rows, (list, tuple))
            and len(rows) == matched
            and all(
                isinstance(row, Mapping) and row.get("status") == "matched"
                for row in rows
            )
        )
        summary = result.summary
        cancelled = result.cancelled or bool(summary.get("cancelled", 0))
        issues = bool(
            summary.get("failed", 0)
            or summary.get("partial", 0)
            or getattr(result, "has_failures", False)
        )
        skipped = bool(summary.get("skipped", 0))
        complete = (
            not cancelled
            and not issues
            and not skipped
            and bool(summary.get("completed", 0))
        )
        title = (
            "Run cancelled"
            if cancelled
            else "Run finished with issues"
            if issues
            else "Run finished with skipped items"
            if skipped
            else "Run complete"
            if complete
            else "Run finished"
        )
        text = title + (
            f" · Original inputs verified · {matched:,} matched"
            if verified
            else " · Original-input verification incomplete — review the run report"
        )
        self._completed_reproduction = _CompletedReproduction(
            request=request,
            text=text,
            success=bool(
                complete
                and verified
                and not record.get("version_override_used")
                and record.get("recorded_vipp_version")
                == request.reference.recorded_vipp_version
                and versions_match(
                    record.get("recorded_vipp_version", ""),
                    record.get("current_vipp_version", ""),
                )
            ),
        )

    def _reproduction_block_reason(self):
        request = self._reproduction_request
        if request is None:
            return ""
        if request.mode != "reproduce":
            return "Choose how to use this reproduction package before continuing."
        check = self._reproduction_check()
        if check is None:
            return (
                "Check the complete batch against the original inputs before running."
            )
        if not check.can_run:
            return (
                "Original-input checks need attention. Resolve every mismatch "
                "and Check batch again; selecting fewer rows cannot bypass this check."
            )
        return ""

    def _sync_reproduction_banner(self):
        if not hasattr(self, "reproduction_banner"):
            return
        request = self._reproduction_request
        self.reproduction_banner.setVisible(self._reproduction_context)
        if not self._reproduction_context:
            return
        busy = self._run_in_progress or self._checking_plan or self._run_preparing
        check = self._reproduction_check()
        completed = self._completed_reproduction
        if (
            busy
            or check is not None
            or completed is None
            or completed.request != request
        ):
            completed = None
        if request is None:
            title = "Use with new data · Original-input comparison is off"
            status = "Normal source, output and scientific safety checks still apply."
            different_version = False
            acknowledged = False
        else:
            expected = request.reference.recorded_vipp_version
            current = current_vipp_version()
            override = (
                request.version_override.to_dict() if request.version_override else {}
            )
            different_version = not versions_match(expected, current)
            acknowledged = different_version and override == {
                "recorded_vipp_version": expected,
                "current_vipp_version": current,
            }
            title = (
                "Reproduce original analysis"
                if request.mode == "reproduce"
                else "Reproduction package · Choose a mode before running"
            )
            title += (
                f" · Recorded VIPP {expected or 'not recorded'} · Current {current}"
            )
            if different_version:
                title += (
                    " · Version difference acknowledged (deviation)"
                    if acknowledged
                    else " · Version mismatch — acknowledgement required"
                )
            if completed is not None:
                status = completed.text
            elif check is None:
                status = (
                    "Checking the complete collection…"
                    if self._checking_plan
                    else "Not verified · Select the source folders and Check batch."
                )
            else:
                counts = Counter(row.status for row in check.rows)
                parts = [f"{check.matched_count:,} verified source checks"]
                parts.extend(
                    f"{count:,} {_CHECK_LABELS.get(state, state).lower()}"
                    for state, count in counts.items()
                    if state != "matched"
                )
                status = " · ".join(parts)
                status += (
                    " · Ready for reviewed run"
                    if check.can_run
                    else " · Run blocked — review Checks and resolve all differences"
                )
        self.reproduction_mode_label.setText(title)
        self.reproduction_status_label.setText(status)
        self.reproduction_status_label.setToolTip(status)
        if completed is not None:
            self.reproduction_status_label.setToolTip(
                status + ". This describes the finished run. "
                "Check batch again before starting another run."
            )
        self.reproduction_details_button.setVisible(check is not None)
        self.reproduction_details_button.setEnabled(not busy and check is not None)
        self.reproduction_new_data_button.setVisible(request is not None)
        self.reproduction_new_data_button.setEnabled(not busy)
        self.reproduction_version_button.setVisible(
            request is not None
            and request.mode == "reproduce"
            and different_version
            and not acknowledged
        )
        self.reproduction_version_button.setEnabled(not busy)
        tones = theme_colors(self.palette())
        tone = (
            tones.success
            if (completed is not None and completed.success)
            or (check is not None and check.can_run)
            else tones.warning
            if request is not None
            else tones.info
        )
        self.reproduction_banner.setStyleSheet(
            "QFrame#BatchReproductionBanner {"
            f"background: {tone.surface.name()};"
            f"border-left: 3px solid {tone.accent.name()}; }}"
        )
        self.reproduction_banner.setProperty(
            "status",
            "success"
            if tone is tones.success
            else "warning"
            if tone is tones.warning
            else "info",
        )
        for label in (self.reproduction_mode_label, self.reproduction_status_label):
            label.setStyleSheet(f"color: {tone.foreground.name()};")

    def _use_reproduction_new_data(self):
        if self._run_in_progress or self._checking_plan or self._run_preparing:
            return
        if (
            QMessageBox.question(
                self,
                "Use with new data?",
                "This stops comparison with the original run's inputs. The workflow "
                "and reviewed settings remain, but a new run will not be presented "
                "as reproducing that original input set. Normal safety checks still "
                "apply. Switch to new data?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        self.set_reproduction_request(None, package_context=True)
        self.reproductionChanged.emit(None)

    def _acknowledge_reproduction_version(self):
        request = self._reproduction_request
        if (
            request is None
            or request.mode != "reproduce"
            or self._run_in_progress
            or self._checking_plan
            or self._run_preparing
        ):
            return
        expected = request.reference.recorded_vipp_version
        current = current_vipp_version()
        if versions_match(expected, current):
            return
        if (
            QMessageBox.question(
                self,
                "Acknowledge a different VIPP version?",
                f"The original run used VIPP {expected}; this is VIPP {current}. "
                "Results may differ. This acknowledgement is retained as a deviation, "
                "not proof of equivalent results. Original-input checks remain "
                "required. "
                "Continue with this version?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
        request = replace(
            request,
            version_override={
                "recorded_vipp_version": expected,
                "current_vipp_version": current,
            },
        )
        self.set_reproduction_request(request)
        self.reproductionChanged.emit(request)

    def _show_reproduction_details(self):
        if self._run_in_progress or self._checking_plan or self._run_preparing:
            return
        check = self._reproduction_check()
        if check is None:
            return
        dialog = ReproductionCheckDetailsDialog(check, self)
        try:
            accepted = dialog.exec() == QDialog.Accepted
            review_target = dialog.review_target
        finally:
            dialog.deleteLater()
        if not accepted or self._reproduction_check() is not check:
            return
        if review_target == "setup":
            self.tabs.setCurrentIndex(0)
            return
        with QSignalBlocker(self.item_search), QSignalBlocker(self.item_filter):
            self.item_search.clear()
            self.item_filter.setCurrentIndex(1 if review_target == "attention" else 0)
        self._item_page = 0
        self._render_items()
        self.tabs.setCurrentIndex(1)

    def _render_reproduction_checks_table(self):
        """Show every failed comparison even when no runnable pairing exists."""
        check = self._reproduction_check()
        self._reproduction_table_active = check is not None and not check.can_run
        if not self._reproduction_table_active:
            return False
        query = self.item_search.text().strip().casefold()
        mode = self.item_filter.currentIndex()
        positions = [
            i
            for i, row in enumerate(check.rows)
            if (
                not query
                or query in f"{row.source_node_id} {row.path} {row.message}".casefold()
            )
            and (mode != 1 or row.status != "matched")
            and (mode != 2 or row.status == "matched")
        ]
        page_size = self._ITEM_PAGE_SIZE
        self._item_page = min(
            self._item_page, max((len(positions) - 1) // page_size, 0)
        )
        start = self._item_page * page_size
        visible = positions[start : start + page_size]
        self._rendering_items = True
        blocker = QSignalBlocker(self.preview_table)
        try:
            self.preview_table.setWordWrap(False)
            self.preview_table.setColumnCount(6)
            self.preview_table.setHorizontalHeaderLabels(
                [
                    "Original item",
                    "Source",
                    "File / selection",
                    "Details",
                    "Checks",
                    "Run result",
                ]
            )
            self.preview_table.setRowCount(len(visible))
            self._preview_table_rows.clear()
            tones = theme_colors(self.palette())
            for index, position in enumerate(visible):
                row = check.rows[position]
                values = (
                    str(row.item_index) if row.item_index is not None else "Unexpected",
                    row.source_node_id,
                    str(row.path or "Not found in selected collection"),
                    row.message,
                    _CHECK_LABELS.get(row.status, row.status),
                    "Not run",
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    cell.setData(Qt.UserRole, position)
                    cell.setToolTip(value)
                    if column == 4:
                        cell.setForeground(
                            tones.success.foreground
                            if row.status == "matched"
                            else tones.error.foreground
                        )
                    self.preview_table.setItem(index, column, cell)
            header = self.preview_table.horizontalHeader()
            for column in (0, 1, 4, 5):
                header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            for column in (2, 3):
                header.setSectionResizeMode(column, QHeaderView.Stretch)
            self.preview_table.resizeRowsToContents()
        finally:
            del blocker
            self._rendering_items = False
        self.item_range_label.setText(
            f"{start + 1 if visible else 0:,}–{start + len(visible):,} of "
            f"{len(positions):,} source checks · "
            f"{check.mismatch_count:,} mismatches total"
        )
        for button in (self.items_previous_button, self.items_next_button):
            button.setVisible(len(positions) > page_size)
        self.items_previous_button.setEnabled(self._item_page > 0)
        self.items_next_button.setEnabled(start + page_size < len(positions))
        self.item_details.setPlainText(
            "No subset can bypass original-input checks. Select a row to "
            "read its difference, correct the source settings, and Check batch again."
        )
        return True

    def _show_reproduction_item_details(self):
        if not self._reproduction_table_active or self._rendering_items:
            return False
        check = self._reproduction_check()
        selected = self.preview_table.selectionModel().selectedRows()
        if check is not None and selected:
            cell = self.preview_table.item(selected[0].row(), 0)
            position = cell.data(Qt.UserRole)
            if isinstance(position, int) and 0 <= position < len(check.rows):
                row = check.rows[position]
                self.item_details.setHtml(
                    f"<b>{escape(_CHECK_LABELS.get(row.status, row.status))}</b>"
                    f"<p>{escape(row.source_node_id)}<br>"
                    f"{escape(str(row.path or ''))}</p>"
                    f"<p>{escape(row.message)}</p><p>Correct the selected input "
                    "collection or image selection, then Check batch again.</p>"
                )
        return True
