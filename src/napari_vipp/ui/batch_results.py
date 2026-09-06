"""Read-only, evidence-backed run review and results for the batch workspace."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, QFileSystemWatcher, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.batch_output_policy import (
    BatchExistingFilesControls,
    checked_output_message,
    item_file_choice,
    item_file_choice_label,
    output_action,
    output_counts_text,
    planned_item_status,
)
from napari_vipp.ui.batch_progress import (
    BatchProgressLabel,
    batch_item_progress_text,
    operation_progress_text,
    preparation_stage,
)
from napari_vipp.ui.batch_result_links import BatchResultLinkTable
from napari_vipp.ui.batch_run_report import BatchRunReport
from napari_vipp.ui.batch_table_style import apply_batch_table_style
from napari_vipp.ui.file_reveal import file_reveal_label, open_folder, reveal_file
from napari_vipp.ui.palette_roles import custom_paint_colors, theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon

if TYPE_CHECKING:
    from napari_vipp.core.batch import BatchRunResult
    from napari_vipp.ui.batch import BatchPreviewResult


def _status(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def _duration(started: str, finished: str) -> float | None:
    if not started or not finished:
        return None
    try:
        seconds = (
            datetime.fromisoformat(finished.replace("Z", "+00:00"))
            - datetime.fromisoformat(started.replace("Z", "+00:00"))
        ).total_seconds()
    except (ValueError, TypeError):
        return None
    return seconds if seconds >= 0 else None


def _time_text(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, seconds = divmod(int(max(seconds, 0)), 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


@dataclass
class _ResultItem:
    index: int
    batch_id: str
    outputs: tuple = ()
    status: str = "not run"
    started: float | None = None
    elapsed: float | None = None
    record: object | None = None
    error: str = ""
    timing_source: str = "Timing not reported."
    paths: tuple[Path, ...] = field(default_factory=tuple)
    planned_status: str = "Not run"
    planned_outputs: str = ""
    file_choice: str = ""
    node_current: int = 0
    node_total: int = 0

    @property
    def display_status(self):
        return (
            self.planned_status
            if self.status == "not run" and self.record is None
            else self.status.title()
        )


class BatchResultsPanel(QWidget):
    """Review a plan and retain actual item/output outcomes after execution.

    ``itemRequested`` emits a zero-based full-plan position, including items
    beyond the currently visible page. Merely selecting a result never runs a
    graph preview. All file actions are read-only and remain available during a
    run; their paths are rechecked on activation.
    """

    itemRequested = Signal(int)
    checkBatchRequested = Signal()
    policyChanged = Signal(str)
    PAGE_SIZE = 100

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: list[_ResultItem] = []
        self._index_positions: dict[int, int] = {}
        self._page = 0
        self._running = False
        self._preparation_started = None
        self._run_started: float | None = None
        self._output_dir: Path | None = None
        self._report_path: Path | None = None
        self._tone = "info"
        self._has_result = False
        self._active_index: int | None = None
        self._selected_paths: list[Path] = []
        self._plan_config = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = self._label("Check batch to review the run plan.")
        self.summary_label.setObjectName("batchResultsSummary")
        self.summary_banner = QFrame()
        self.summary_banner.setObjectName("BatchResultsSummaryBanner")
        summary = QHBoxLayout(self.summary_banner)
        summary.setContentsMargins(10, 8, 10, 8)
        summary.setSpacing(12)
        summary.addWidget(self.summary_label, 1)
        self.check_batch_button = ToolbarCommandButton("Check batch again")
        self.check_batch_button.setObjectName("BatchPrimaryAction")
        self.check_batch_button.clicked.connect(self.checkBatchRequested.emit)
        self.check_batch_button.hide()
        summary.addWidget(self.check_batch_button, 0, Qt.AlignVCenter)
        layout.addWidget(self.summary_banner)
        self.existing_files_controls = BatchExistingFilesControls()
        self.existing_files_controls.policyChanged.connect(self.policyChanged.emit)
        layout.addWidget(self.existing_files_controls)
        self.review_group = QGroupBox("Review and run")
        review = QFormLayout(self.review_group)
        review.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        review.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.review_labels = {}
        for title in (
            "Save results to",
            "Batch",
            "Existing files",
            "On an item error",
            "Parameter overrides",
            "Node behavior",
            "Compute request",
        ):
            label = self._label("—")
            self.review_labels[title] = label
            review.addRow(title, label)
        layout.addWidget(self.review_group)

        self.run_report = BatchRunReport()
        self.run_report.manifest_button.clicked.connect(self._reveal_manifest)
        layout.addWidget(self.run_report)

        self.elapsed_label = self._label("Elapsed —")
        # The compact clock uses its natural single-line size, independently of
        # the two-line reservation for the changing status messages below it.
        self.elapsed_label.setWordWrap(False)
        self.elapsed_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.run_progress_label = BatchProgressLabel("No batch run is active.")
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat("Not run")
        self.operation_progress_label = BatchProgressLabel(
            "No node operation is active."
        )
        self.operation_progress_bar = QProgressBar()
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.progress_group = QWidget()
        progress = QVBoxLayout(self.progress_group)
        progress.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self.elapsed_label,
            self.run_progress_label,
            self.run_progress_bar,
            self.operation_progress_label,
            self.operation_progress_bar,
        ):
            progress.addWidget(widget)
        self.progress_group.hide()
        layout.addWidget(self.progress_group)

        self.items_heading = self._label("Batch items")
        self.items_heading.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.items_hint = self._label(
            "Select a row for outputs · click its name to review the item."
        )
        items_heading_row = QHBoxLayout()
        items_heading_row.addWidget(self.items_heading)
        items_heading_row.addWidget(self.items_hint, 1)
        layout.addLayout(items_heading_row)
        self.items_table = self._table(["Batch item", "Result", "Output files", "Time"])
        self.items_table.setMinimumHeight(140)
        self.items_table.setMaximumHeight(310)
        self.items_table.itemSelectionChanged.connect(self._show_selected_files)
        self.items_table.linkActivated.connect(self._item_clicked)
        layout.addWidget(self.items_table)
        page_row = QHBoxLayout()
        self.previous_page_button = ToolbarCommandButton("Previous")
        self.next_page_button = ToolbarCommandButton("Next")
        self.page_label = self._label("No items")
        self.previous_page_button.clicked.connect(lambda: self._change_page(-1))
        self.next_page_button.clicked.connect(lambda: self._change_page(1))
        page_row.addWidget(self.page_label, 1)
        page_row.addWidget(self.previous_page_button)
        page_row.addWidget(self.next_page_button)
        layout.addLayout(page_row)

        # Keep both tables on the same page gutters. The selected-item header
        # provides grouping without indenting the output table in a group box.
        self.files_group = QWidget()
        files = QVBoxLayout(self.files_group)
        files.setContentsMargins(0, 0, 0, 0)
        files.setSpacing(8)
        self.selected_item_header = QFrame()
        self.selected_item_header.setObjectName("BatchSelectedOutputsHeader")
        context = QVBoxLayout(self.selected_item_header)
        context.setContentsMargins(12, 8, 12, 8)
        context.setSpacing(5)
        caption = QHBoxLayout()
        caption.setSpacing(8)
        self.selected_item_icon = QLabel()
        self.selected_item_icon.setFixedSize(18, 18)
        self.selected_item_heading = self._label("Outputs for selected item")
        self.selected_item_heading.setWordWrap(False)
        self.selected_item_heading.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        self.selected_item_meta = self._label("")
        self.selected_item_meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        caption.addWidget(self.selected_item_icon)
        caption.addWidget(self.selected_item_heading)
        caption.addWidget(self.selected_item_meta, 1)
        context.addLayout(caption)
        self.selected_item_label = self._label("Select an item to inspect its files.")
        context.addWidget(self.selected_item_label)
        files.addWidget(self.selected_item_header)
        self.item_error_label = self._label("")
        self.item_error_label.hide()
        self.output_table = self._table(["Output file", "Status"])
        self.output_table.setMinimumHeight(95)
        self.output_table.setMaximumHeight(205)
        self.output_table.linkActivated.connect(self._output_clicked)
        self.output_table.itemSelectionChanged.connect(self._sync_reveal_button)
        files.addWidget(self.item_error_label)
        files.addWidget(self.output_table)
        file_actions = QHBoxLayout()
        self.review_item_button = ToolbarCommandButton(
            "Review item in Items && outputs"
        )
        self.review_item_button.clicked.connect(self._request_selected_item)
        self.reveal_button = ToolbarCommandButton(file_reveal_label())
        self.reveal_button.clicked.connect(self._reveal_selected_file)
        file_actions.addWidget(self.review_item_button)
        file_actions.addStretch(1)
        file_actions.addWidget(self.reveal_button)
        files.addLayout(file_actions)
        layout.addWidget(self.files_group)

        self.artifact_toolbar = QWidget()
        self.artifact_toolbar.setObjectName("BatchResultActions")
        artifact_layout = QVBoxLayout(self.artifact_toolbar)
        artifact_layout.setContentsMargins(0, 4, 0, 4)
        artifact_actions = QHBoxLayout()
        artifact_layout.addLayout(artifact_actions)
        self.output_folder_button = ToolbarCommandButton("Output folder")
        self.output_folder_button.clicked.connect(self._open_output_folder)
        self.refresh_files_button = ToolbarCommandButton("Refresh file status")
        self.refresh_files_button.setToolTip(
            "Update which output files, output folder, and manifest JSON exist "
            "on disk. "
            "This normally updates automatically; use this if a network drive "
            "has not reported a change.\n"
            "This does not validate the batch, prepare a new run, or change recorded "
            "results. Use Check batch in Setup to validate inputs and settings."
        )
        self.refresh_files_button.clicked.connect(self.refresh_files)
        for button in (
            self.output_folder_button,
            self.refresh_files_button,
        ):
            artifact_actions.addWidget(button)
        artifact_actions.addStretch(1)
        layout.insertWidget(0, self.artifact_toolbar)
        self._command_icons = (
            (self.previous_page_button, "previous"),
            (self.next_page_button, "next"),
            (self.review_item_button, "batch"),
            (self.reveal_button, "open"),
            (self.output_folder_button, "open"),
            (self.refresh_files_button, "refresh"),
            (self.check_batch_button, "checklist"),
        )
        self.file_action_label = self._label("")
        self.file_action_label.hide()
        artifact_layout.addWidget(self.file_action_label)
        layout.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_elapsed)
        self._file_watcher = QFileSystemWatcher(self)
        self._file_refresh_timer = QTimer(self)
        self._file_refresh_timer.setSingleShot(True)
        self._file_refresh_timer.setInterval(200)
        self._file_refresh_timer.timeout.connect(self.refresh_files)
        self._file_watcher.directoryChanged.connect(self._queue_file_refresh)
        self._file_watcher.fileChanged.connect(self._queue_file_refresh)
        self._render_page()
        self._apply_theme()

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return label

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        table = BatchResultLinkTable(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        return table

    def set_plan(self, preview: BatchPreviewResult | None) -> None:
        """Replace a plan when explicitly checked; clear the previous run view."""

        self._timer.stop()
        self._running = False
        self._run_started = None
        self._preparation_started = None
        self._has_result = False
        self.run_report.hide()
        self.summary_banner.show()
        self._report_path = None
        self._page = 0
        self._items = []
        self._output_dir = None
        self._active_index = None
        self.file_action_label.hide()
        self.review_group.show()
        self.progress_group.hide()
        self.elapsed_label.setText("Elapsed —")
        if preview is not None:
            config = preview.config
            self._plan_config = config
            self.existing_files_controls.set_plan(preview)
            self._output_dir = config.resolve_path(config.output_dir)
            preview_rows = {row.batch_index: row for row in preview.rows}
            for item in preview.items:
                paths = tuple(Path(output.path) for output in item.outputs)
                if not paths and item.index in preview_rows:
                    paths = tuple(
                        Path(path) for path in preview_rows[item.index].outputs
                    )
                self._items.append(
                    _ResultItem(
                        item.index,
                        item.batch_id,
                        outputs=tuple(item.outputs),
                        paths=paths,
                        planned_status=planned_item_status(item.outputs, config),
                        planned_outputs=output_counts_text(
                            item.outputs,
                            config,
                            compact=True,
                        ),
                        file_choice=(
                            item_file_choice_label(config, item)
                            if item_file_choice(config, item)
                            else ""
                        ),
                    )
                )
            outputs = tuple(output for item in preview.items for output in item.outputs)
            count = sum(len(entry.values) for entry in config.parameter_overrides)
            details = {
                "Save results to": str(self._output_dir),
                "Batch": (
                    f"{preview.total_items:,} items · "
                    f"{output_counts_text(outputs, config)}"
                ),
                "Existing files": {
                    "error": "Ask before overwrite",
                    "skip": "Skip existing files",
                    "overwrite": "Overwrite without asking",
                }.get(
                    _status(config.existing_file_policy),
                    str(config.existing_file_policy),
                ),
                "On an item error": (
                    "Continue with the next item"
                    if config.continue_on_error
                    else "Stop the batch"
                ),
                "Parameter overrides": (
                    f"{count:,} changes across "
                    f"{len(config.parameter_overrides):,} sources"
                ),
                "Node behavior": (
                    f"{len(config.node_execution_overrides):,} batch changes"
                    if config.node_execution_overrides
                    else "Use workflow"
                ),
                "Compute request": {
                    "cpu": "CPU",
                    "auto": "Auto",
                    "prefer_gpu": "Prefer GPU",
                    "custom": "Custom",
                }.get(
                    _status(config.compute_request.mode),
                    str(config.compute_request.mode),
                ),
            }
            for title, label in self.review_labels.items():
                label.setText(details[title])
                label.setToolTip(details[title])
            self._set_summary(
                checked_output_message(preview)
                + " Run performs one final validation to detect files changed "
                "since Check.",
                "warning" if preview.collision_count else "info",
            )
        else:
            self._plan_config = None
            self.existing_files_controls.hide()
            for label in self.review_labels.values():
                label.setText("—")
            self._set_summary("Check batch to review the run plan.", "info")
        self._index_positions = {
            item.index: position for position, item in enumerate(self._items)
        }
        self._render_page()

    def update_output_policy(self, preview) -> None:
        """Refresh decisions without moving the selected item or table page."""
        selected = self._selected_position()
        self.set_plan(preview)
        if selected is not None:
            self.select_item(selected)

    def invalidate_plan(
        self,
        message: str = (
            "Settings changed. Check inputs and output destinations "
            "before running again."
        ),
    ) -> None:
        """Flag stale settings while preserving any historical run evidence."""

        self._set_summary(message, "warning")
        self.summary_banner.show()

    def set_check_action_state(
        self, *, needed: bool, enabled: bool, checking: bool = False, reason: str = ""
    ) -> None:
        """The owning workspace decides when validating a new plan is safe."""
        self.check_batch_button.setVisible(needed)
        self.check_batch_button.setEnabled(enabled and not checking)
        self.check_batch_button.setText(
            "Checking…" if checking else "Check batch again"
        )
        self.check_batch_button.setToolTip(
            reason
            or "Validate source files, overrides, and output destinations, then review "
            "the updated list in Items & outputs. No images are calculated or saved. "
            "A successful check replaces this run view; saved reports remain on disk."
        )

    def begin_preparation(self, total: int) -> None:
        """Show startup work without clearing historical results or starting items."""
        if self._preparation_started is None:
            self._preparation_started = time.monotonic()
        self.review_group.hide()
        self.run_report.hide()
        self.summary_banner.show()
        self.progress_group.show()
        self.operation_progress_label.show()
        self.operation_progress_bar.show()
        self.run_progress_label.setText(
            (
                f"Preparing run · {total:,} planned items. "
                if total
                else "Preparing run. "
            )
            + "No items have started yet."
        )
        self.run_progress_bar.setRange(0, max(1, total))
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat(f"0 / {total}" if total else "No items started")
        self.operation_progress_label.setText(
            "Final validation · checking for source and destination changes "
            "since Check…"
        )
        self.operation_progress_bar.setRange(0, 0)
        self._set_summary("Preparing run · verifying the reviewed plan.", "info")
        self._update_elapsed()
        self._timer.start()

    def update_preparation(self, event) -> None:
        title, detail = preparation_stage(event)
        self.operation_progress_label.setText(detail)
        self.operation_progress_label.setToolTip(event.message or detail)
        self._set_summary(f"Preparing run · {title.lower()}.", "info")
        if event.byte_total:
            self.operation_progress_bar.setRange(0, 100)
            self.operation_progress_bar.setValue(
                min(100, int(100 * event.byte_current / event.byte_total))
            )
            self.operation_progress_bar.setFormat("Reading file · %p%")
        elif event.total and event.phase == "pipelines":
            self.operation_progress_bar.setRange(0, event.total)
            self.operation_progress_bar.setValue(event.current)
            self.operation_progress_bar.setFormat("%v / %m item workflows")
        else:
            self.operation_progress_bar.setRange(0, 0)

    def end_preparation(self) -> None:
        self._preparation_started = None
        if not self._running:
            self._timer.stop()
            self.progress_group.hide()
            self.review_group.show()
            self._set_summary("Run not started · review the plan below.", "info")

    def cancel_before_first_item(self) -> None:
        """Return to the checked plan after cancellation in the run worker."""
        self._stop_clock()
        self.end_preparation()
        for item in self._items:
            item.status = "not run"
        self._set_summary("Preparation cancelled · no items started.", "info")
        self._render_page()

    def begin_run(self, total: int) -> None:
        self._preparation_started = None
        self._running = True
        self._has_result = False
        self.run_report.hide()
        self.summary_banner.show()
        self._report_path = None
        self._run_started = time.monotonic()
        self._active_index = None
        for item in self._items:
            item.status = "pending"
            item.record = None
            item.error = ""
            item.started = None
            item.elapsed = None
            item.timing_source = "Timing not reported."
            item.node_current = item.node_total = 0
        self.review_group.hide()
        self.progress_group.show()
        self.operation_progress_label.show()
        self.operation_progress_bar.show()
        self.run_progress_bar.setRange(0, max(1, int(total)))
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat(f"0 / {total}")
        self.run_progress_label.setText(
            f"Preparing run · {total:,} planned items. No items have started yet."
        )
        self.operation_progress_bar.setRange(0, 0)
        self.operation_progress_label.setText("Preparing execution and item workflows…")
        self._set_summary("Preparing run · setting up item workflows.", "info")
        self._update_elapsed()
        self._timer.start()
        self._render_page()

    def update_item(self, index: int, total: int, batch_id: str, status: str) -> None:
        """Accept the core's one-based item-status callback, without fake files."""

        normalized = _status(status)
        if self._active_index is None:
            self._set_summary("Running · completed outputs are retained.", "info")
        position = self._index_positions.get(index)
        if position is None:
            position = len(self._items)
            self._items.append(_ResultItem(index, batch_id))
            self._index_positions[index] = position
        item = self._items[position]
        if normalized == "running" and item.status != "running":
            item.node_current = item.node_total = 0
        item.status = normalized
        if normalized == "running":
            self._active_index = index
            if item.started is None:
                item.started = time.monotonic()
        elif item.started is not None:
            item.elapsed = max(0, time.monotonic() - item.started)
            item.timing_source = "Elapsed captured from this item's progress callbacks."
        done = sum(
            entry.status in {"completed", "partial", "failed", "skipped"}
            for entry in self._items
        )
        self.run_progress_bar.setRange(0, max(1, total))
        self.run_progress_bar.setValue(min(done, total))
        self.run_progress_bar.setFormat(f"{done} / {total}")
        self.run_progress_label.setText(
            batch_item_progress_text(
                index,
                total,
                batch_id,
                normalized,
                item.node_current,
                item.node_total,
            )
        )
        # Do not move the selection while someone is reviewing another item.
        if position // self.PAGE_SIZE == self._page:
            self._render_page()
        self._sync_artifact_buttons()
        self._update_elapsed()

    def update_operation_progress(
        self,
        item_index: int,
        item_total: int,
        batch_id: str,
        node_id: str,
        operation_id: str,
        current: int,
        total: int,
        message: str = "",
        *,
        node_title: str = "",
        node_current: int = 0,
        node_total: int = 0,
    ) -> None:
        position = self._index_positions.get(item_index)
        if position is not None and item_index == self._active_index:
            item = self._items[position]
            if item.status == "running" and 0 < node_current <= node_total:
                item.node_current = max(item.node_current, node_current)
                item.node_total = node_total
                self.run_progress_label.setText(
                    batch_item_progress_text(
                        item_index,
                        item_total,
                        batch_id,
                        item.status,
                        item.node_current,
                        item.node_total,
                    )
                )
        self.operation_progress_bar.setRange(0, max(total, 0))
        self.operation_progress_bar.setValue(max(current, 0))
        self.operation_progress_bar.setFormat("%v / %m" if total else "%p%")
        self.operation_progress_label.setText(
            operation_progress_text(operation_id, message, node_title=node_title)
        )
        self.operation_progress_label.setToolTip(
            f"Item {item_index} of {item_total} · {batch_id}\n"
            f"Node: {node_id} · operation: {operation_id}"
        )

    def set_stopping(self) -> None:
        self._set_summary(
            "Stopping safely · outputs already saved are retained.", "warning"
        )
        self.operation_progress_label.setText("Waiting for a safe checkpoint…")

    def finish_run(self, result: BatchRunResult, note: str = "") -> None:
        """Use the authoritative item/output records, not the preview plan."""

        captured = self._stop_clock()
        self._has_result = True
        manifest = result.manifest
        records = tuple(manifest.items)
        for record in records:
            position = self._index_positions.get(record.index)
            if position is None:
                position = len(self._items)
                self._index_positions[record.index] = position
                self._items.append(
                    _ResultItem(
                        record.index,
                        getattr(record, "batch_id", f"Item {record.index}"),
                    )
                )
            item = self._items[position]
            item.status = _status(record.status)
            item.record = record
            item.error = str(getattr(record, "error_message", ""))
            if hasattr(record, "outputs"):
                item.outputs = tuple(record.outputs)
                item.paths = tuple(Path(output.path) for output in record.outputs)
            elapsed = _duration(
                getattr(record, "started_at", ""),
                getattr(record, "finished_at", ""),
            )
            if elapsed is not None:
                item.elapsed = elapsed
                item.timing_source = "Elapsed from the run report's item timestamps."
        output_dir = getattr(manifest, "output_dir", "")
        if output_dir:
            self._output_dir = Path(output_dir)
        # Prefer the immutable run-specific report when the core supplied it;
        # the latest-manifest path can be replaced by a subsequent batch run.
        self._report_path = Path(
            getattr(result, "manifest_archive_path", None) or result.manifest_path
        )
        reported = _duration(
            getattr(manifest, "started_at", ""),
            getattr(manifest, "finished_at", ""),
        )
        elapsed = reported if reported is not None else captured
        self.elapsed_label.setText(f"Total elapsed {_time_text(elapsed)}")
        self.elapsed_label.setToolTip(
            "Elapsed from run report timestamps; includes processing and writing."
            if reported is not None
            else "Elapsed captured in this window; includes processing and writing."
            if captured is not None
            else "Elapsed timing was not reported."
        )
        summary = result.summary
        counts = (
            " · ".join(
                f"{int(summary.get(state, 0)):,} {state}"
                for state in ("completed", "partial", "skipped", "cancelled", "failed")
                if summary.get(state, 0)
            )
            or "No items processed"
        )
        total_outputs = sum(len(getattr(record, "outputs", ())) for record in records)
        saved_count = len(result.saved_paths)
        outcome = f"{counts} · {saved_count:,} outputs saved"
        if total_outputs:
            outcome = f"{counts} · {saved_count:,} of {total_outputs:,} outputs saved"
        issues = int(summary.get("partial", 0)) + int(summary.get("failed", 0))
        compute = getattr(manifest, "compute", {})
        cleanup_failed = not bool(compute.get("runtime_cleanup_succeeded", True))
        has_issues = bool(
            issues or cleanup_failed or getattr(result, "has_failures", False)
        )
        cancelled = int(summary.get("cancelled", 0))
        title = (
            "Stopped · saved outputs retained"
            if cancelled
            else "Finished with issues"
            if has_issues
            else "Batch finished"
        )
        details = [title, outcome]
        if cleanup_failed:
            details.append(
                "Runtime cleanup did not finish successfully. See the run report."
            )
        if note:
            details.append(note)
        self._set_summary(
            "\n".join(details), "warning" if has_issues or cancelled else "success"
        )
        self.review_group.hide()
        self.progress_group.show()
        self.operation_progress_label.hide()
        self.operation_progress_bar.hide()
        terminal = sum(
            int(summary.get(state, 0))
            for state in (
                "completed",
                "partial",
                "skipped",
                "failed",
            )
        )
        self.run_progress_bar.setRange(0, max(len(records), 1))
        self.run_progress_bar.setValue(min(terminal, len(records)))
        self.run_progress_bar.setFormat(f"{terminal} / {len(records)}")
        self.run_progress_label.setText(outcome)
        self.run_report.set_result(
            result,
            title=title,
            elapsed=self.elapsed_label.text(),
            tone="warning" if has_issues or cancelled else "success",
            note=note,
        )
        # Replace live progress with the finished report; retain the labels as
        # recorded evidence for integrations, without duplicating them on screen.
        self.progress_group.hide()
        self.summary_banner.hide()
        self._render_page()

    def show_error(self, message: str) -> None:
        self._stop_clock()
        self._set_summary(f"Run failed\n{message}", "error")
        self.review_group.hide()
        self.progress_group.show()
        self.operation_progress_label.hide()
        self.operation_progress_bar.hide()
        self.run_progress_bar.setFormat("Failed")
        self.run_progress_label.setText(
            "Run stopped before a final report was returned."
        )
        position = self._index_positions.get(self._active_index)
        if position is not None and self._items[position].status == "running":
            item = self._items[position]
            item.status = "failed"
            item.error = message
            if item.started is not None:
                item.elapsed = max(0, time.monotonic() - item.started)
                item.timing_source = "Elapsed captured in this window."
        self._render_page()

    def _stop_clock(self) -> float | None:
        self._update_elapsed()
        captured = (
            max(0, time.monotonic() - self._run_started)
            if self._run_started is not None
            else None
        )
        self._timer.stop()
        self._running = False
        return captured

    def _update_elapsed(self) -> None:
        if self._preparation_started is not None:
            elapsed = _time_text(time.monotonic() - self._preparation_started)
            self.elapsed_label.setText(f"Preparing · {elapsed} elapsed")
            self.elapsed_label.setToolTip(
                "Time spent preparing this run, before processing items."
            )
        elif self._running and self._run_started is not None:
            self.elapsed_label.setText(
                f"Elapsed {_time_text(time.monotonic() - self._run_started)}"
            )
            self.elapsed_label.setToolTip("Elapsed captured in this window.")

    def _set_summary(self, text: str, tone: str) -> None:
        self.summary_banner.show()
        changed_tone = tone != self._tone
        self.summary_label.setText(text)
        self._tone = tone
        if changed_tone:
            self._apply_theme()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
            QEvent.StyleChange,
            QEvent.FontChange,
        ):
            self._apply_theme()
            if hasattr(self, "items_table"):
                self._render_page()
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._queue_file_refresh()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._queue_file_refresh()

    def hideEvent(self, event):  # noqa: N802
        super().hideEvent(event)
        if hasattr(self, "_file_refresh_timer"):
            self._file_refresh_timer.stop()
            paths = self._file_watcher.directories() + self._file_watcher.files()
            if paths:
                self._file_watcher.removePaths(paths)

    def _queue_file_refresh(self, *_args):
        if hasattr(self, "_file_refresh_timer") and self.isVisible():
            self._file_refresh_timer.start()

    def _sync_file_watches(self):
        if not hasattr(self, "_file_watcher") or not self.isVisible():
            return
        # Only watch this item's directories and known run artifacts, never
        # enumerate or hash all outputs in a potentially very large batch.
        directories = {path.parent for path in self._selected_paths}
        if self._output_dir is not None:
            directories.update((self._output_dir, self._output_dir.parent))
        if self._report_path is not None:
            directories.add(self._report_path.parent)
        desired = {str(path) for path in directories if path.is_dir()}
        current = set(self._file_watcher.directories())
        if current - desired:
            self._file_watcher.removePaths(sorted(current - desired))
        if desired - current:
            self._file_watcher.addPaths(sorted(desired - current))

    def _apply_theme(self) -> None:
        if not hasattr(self, "summary_banner"):
            return
        colors = theme_colors(self.palette())
        if hasattr(self, "run_report"):
            self.run_report.ensurePolished()
            if self.run_report.palette() != self.palette():
                self.run_report.setPalette(self.palette())
            if self.run_report.font() != self.font():
                self.run_report.setFont(self.font())
        for name in ("items_table", "output_table"):
            table = getattr(self, name, None)
            if table is not None:
                apply_batch_table_style(
                    table, self.palette(), font=self.font(), horizontal_padding=12
                )
                table.verticalHeader().setDefaultSectionSize(
                    max(32, table.fontMetrics().height() + 14)
                )
        if hasattr(self, "selected_item_header"):
            surface = custom_paint_colors(self.palette())
            self.selected_item_header.setStyleSheet(
                "QFrame#BatchSelectedOutputsHeader {"
                f"background: {colors.info.surface.name()};"
                f"border-left: 3px solid {colors.info.accent.name()}; }}"
            )
            for label in (self.items_heading, self.selected_item_heading):
                label.setStyleSheet("font-weight: bold;")
            self.items_hint.setStyleSheet(f"color: {surface.muted_text.name()};")
            self.selected_item_label.setStyleSheet(
                f"color: {colors.info.foreground.name()};"
            )
            self.selected_item_meta.setStyleSheet(
                f"color: {colors.info.foreground.name()};"
            )
            self.selected_item_icon.setPixmap(
                toolbar_icon("open", self.palette()).pixmap(18, 18)
            )
        for button, kind in getattr(self, "_command_icons", ()):
            button.setIcon(toolbar_icon(kind, self.palette()))
            button.setIconSize(QSize(18, 18))
        tone = getattr(colors, self._tone)
        self.summary_banner.setStyleSheet(
            "QFrame#BatchResultsSummaryBanner {"
            f"background: {tone.surface.name()};"
            f"border: 1px solid {tone.border.name()}; border-radius: 5px;"
            "}"
        )
        self.summary_label.setStyleSheet(
            "QLabel#batchResultsSummary { border: none; padding: 0;"
            f"background: {tone.surface.name()}; color: {tone.foreground.name()};"
            "}"
        )
        if hasattr(self, "item_error_label"):
            self.item_error_label.setStyleSheet(
                f"color: {colors.error.foreground.name()};"
            )

    def _selected_position(self) -> int | None:
        row = self.items_table.currentRow()
        position = self._page * self.PAGE_SIZE + row
        return position if row >= 0 and position < len(self._items) else None

    def select_item(self, position: int) -> bool:
        if not 0 <= position < len(self._items):
            return False
        self._page = position // self.PAGE_SIZE
        self._render_page()
        self.items_table.selectRow(position % self.PAGE_SIZE)
        return True

    def _change_page(self, delta: int) -> None:
        last_page = max(0, (len(self._items) - 1) // self.PAGE_SIZE)
        self._page = min(max(self._page + delta, 0), last_page)
        self.items_table.clearSelection()
        self.items_table.setCurrentCell(-1, -1)
        self._render_page()

    def _render_page(self) -> None:
        selected_row = self.items_table.currentRow()
        start = self._page * self.PAGE_SIZE
        visible = self._items[start : start + self.PAGE_SIZE]
        colors = theme_colors(self.palette())
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(len(visible))
        for row, item in enumerate(visible):
            if item.record is not None and hasattr(item.record, "outputs"):
                saved = sum(
                    _status(output.status) == "completed" for output in item.outputs
                )
                outputs = f"{saved} / {len(item.outputs)} saved"
            else:
                outputs = item.planned_outputs or f"{len(item.paths)} planned"
            values = (
                item.batch_id,
                item.display_status,
                outputs,
                _time_text(item.elapsed),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.UserRole, start + row)
                cell.setToolTip(item.timing_source if column == 3 else value)
                if column == 0:
                    # A new item's font() defaults to the application font,
                    # not the table's resolved font under napari's stylesheet.
                    font = QFont(self.items_table.font())
                    font.setUnderline(True)
                    cell.setFont(font)
                    cell.setForeground(QBrush(colors.info.foreground))
                    cell.setToolTip("Review this item in Items & outputs")
                elif column == 1:
                    if item.file_choice and item.status == "not run":
                        font = QFont(self.items_table.font())
                        font.setBold(True)
                        cell.setFont(font)
                        cell.setToolTip(item.file_choice)
                    tone = {
                        "completed": colors.success,
                        "partial": colors.warning,
                        "cancelled": colors.warning,
                        "failed": colors.error,
                        "running": colors.info,
                        "keep existing": colors.info,
                        "create missing": colors.info,
                        "needs decision": colors.warning,
                        "will overwrite": colors.warning,
                        "blocked": colors.error,
                    }.get(item.display_status.casefold())
                    if tone is not None:
                        cell.setForeground(QBrush(tone.foreground))
                self.items_table.setItem(row, column, cell)
        self.items_table.blockSignals(False)
        self._fit_table_height(self.items_table, maximum=310)
        if visible:
            self.items_table.selectRow(min(max(selected_row, 0), len(visible) - 1))
        self.page_label.setText(
            f"Items {start + 1:,}–{start + len(visible):,} of {len(self._items):,}"
            if visible
            else "No items"
        )
        multiple_pages = len(self._items) > self.PAGE_SIZE
        self.previous_page_button.setVisible(multiple_pages)
        self.next_page_button.setVisible(multiple_pages)
        self.previous_page_button.setEnabled(self._page > 0)
        self.next_page_button.setEnabled(start + len(visible) < len(self._items))
        self._show_selected_files()
        self._sync_artifact_buttons()

    def _item_clicked(self, row: int, column: int) -> None:
        if column == 0:
            position = self._page * self.PAGE_SIZE + row
            if 0 <= position < len(self._items):
                self.itemRequested.emit(position)

    def _request_selected_item(self) -> None:
        position = self._selected_position()
        if position is not None:
            self.itemRequested.emit(position)

    def _show_selected_files(self) -> None:
        position = self._selected_position()
        self._selected_paths = []
        self.output_table.setRowCount(0)
        self.item_error_label.hide()
        self.review_item_button.setEnabled(position is not None)
        if position is None:
            self.selected_item_label.setText("Select an item to inspect its files.")
            self.selected_item_meta.clear()
            self._sync_reveal_button()
            self._sync_file_watches()
            return
        item = self._items[position]
        self.selected_item_label.setText(item.batch_id)
        self.selected_item_meta.setText(
            f"Item {position + 1:,} of {len(self._items):,} · "
            f"{len(item.paths):,} output{'s' if len(item.paths) != 1 else ''} · "
            f"{item.display_status}"
        )
        errors = [item.error] if item.error else []
        seen_errors = set(errors)
        for output in item.outputs if item.record is not None else ():
            if _status(getattr(output, "status", "")) not in ("failed", "partial"):
                continue
            reason = (
                getattr(output, "error_message", "")
                or item.error
                or getattr(output, "error_type", "")
                or "No failure reason was recorded."
            )
            if reason not in seen_errors:
                name = getattr(output, "node_title", "") or "Output"
                errors.append(f"{name}: {reason}")
                seen_errors.add(reason)
        if errors:
            self.item_error_label.setText("\n".join(errors))
            self.item_error_label.show()
        self._selected_paths = list(item.paths)
        self.output_table.setRowCount(len(item.paths))
        colors = theme_colors(self.palette())
        for row, path in enumerate(item.paths):
            exists = path.is_file()
            output = item.outputs[row] if row < len(item.outputs) else None
            status, detail = self._output_status(item, output, exists)
            name = QTableWidgetItem(path.name)
            name.setData(Qt.UserRole, str(path))
            action = (
                file_reveal_label() if exists else "File is missing or not yet created."
            )
            name.setToolTip(f"{path}\n{detail}\n{action}")
            if exists:
                font = QFont(self.output_table.font())
                font.setUnderline(True)
                name.setFont(font)
                name.setForeground(QBrush(colors.info.foreground))
            self.output_table.setItem(row, 0, name)
            status_item = QTableWidgetItem(status)
            status_item.setToolTip(detail)
            if status.startswith("Failed"):
                status_item.setForeground(QBrush(colors.error.foreground))
            elif status == "Saved":
                status_item.setForeground(QBrush(colors.success.foreground))
            elif status.startswith("Saved ·"):
                status_item.setForeground(QBrush(colors.warning.foreground))
            self.output_table.setItem(row, 1, status_item)
        if item.paths:
            self.output_table.selectRow(0)
        self._fit_table_height(self.output_table, maximum=205)
        self._sync_reveal_button()
        self._sync_file_watches()

    @staticmethod
    def _fit_table_height(table: QTableWidget, *, maximum: int) -> None:
        content_height = (
            table.horizontalHeader().height()
            + table.rowCount() * table.verticalHeader().defaultSectionSize()
            + 2 * table.frameWidth()
        )
        table.setMaximumHeight(max(table.minimumHeight(), min(maximum, content_height)))

    def _output_status(self, item, output, exists: bool) -> tuple[str, str]:
        if item.record is not None and output is not None and hasattr(output, "status"):
            status = _status(output.status)
            message = str(getattr(output, "error_message", ""))
            if status == "completed":
                text = "Saved" if exists else "Saved · file missing"
            elif status == "failed":
                text = (
                    "Failed · existing file remains"
                    if exists
                    else "Failed · not created"
                )
            elif status == "skipped":
                text = "Skipped · existing file" if exists else "Skipped · no file"
            elif exists:
                text = f"{status.title()} · existing file"
            else:
                text = f"Not created · {status}"
            return (
                text,
                message
                or f"Run report: {status}. "
                f"Current path: {'exists' if exists else 'missing'}.",
            )
        if self._running or item.status != "not run":
            return (
                "Existing · awaiting report"
                if exists
                else "Not created / not yet reported",
                "No final output record has been received. "
                "File presence alone does not prove this run saved it.",
            )
        if output is not None and exists == output.exists:
            text = {
                "keep": "Keep existing · skip",
                "overwrite": "Will overwrite",
                "ask": "Needs decision when you run",
                "blocked": "Blocked destination",
                "create": "To create",
            }[output_action(output, self._plan_config)]
        else:
            text = (
                "Existing · not saved by this run"
                if exists
                else "Missing · checked again before run"
            )
        return (
            text,
            "Planned action only; no files have been written by this run. "
            "Existing files are not proof of a previous successful calculation. "
            "Current disk contents are checked again before writing.",
        )

    def _sync_reveal_button(self) -> None:
        row = self.output_table.currentRow()
        path = (
            self._selected_paths[row] if 0 <= row < len(self._selected_paths) else None
        )
        exists = path is not None and path.is_file()
        self.reveal_button.setEnabled(exists)
        self.reveal_button.setToolTip(
            f"{file_reveal_label()}: {path}"
            if exists
            else "Select an existing file. Missing files cannot be revealed."
        )

    def _output_clicked(self, row: int, column: int) -> None:
        if column == 0 and 0 <= row < len(self._selected_paths):
            self._reveal_path(self._selected_paths[row])

    def _reveal_selected_file(self) -> None:
        row = self.output_table.currentRow()
        if 0 <= row < len(self._selected_paths):
            self._reveal_path(self._selected_paths[row])

    def _reveal_path(self, path: Path) -> None:
        result = reveal_file(path)
        self.file_action_label.setText(result.message)
        self.file_action_label.show()
        self.refresh_files()

    def _sync_artifact_buttons(self) -> None:
        folder_exists = self._output_dir is not None and self._output_dir.is_dir()
        report_exists = self._report_path is not None and self._report_path.is_file()
        self.output_folder_button.setEnabled(folder_exists)
        self.output_folder_button.setToolTip(
            str(self._output_dir)
            if folder_exists
            else "The output folder has not been created or is unavailable."
        )
        self.run_report.manifest_button.setEnabled(report_exists)
        self.run_report.manifest_button.setToolTip(
            f"{file_reveal_label()}: {self._report_path}"
            if report_exists
            else "The technical manifest JSON is unavailable on disk. "
            "The run summary above is still available."
        )

    def _open_output_folder(self) -> None:
        if self._output_dir is not None:
            result = open_folder(self._output_dir)
            self.file_action_label.setText(result.message)
            self.file_action_label.show()
            self._sync_artifact_buttons()

    @property
    def has_run_report(self) -> bool:
        return self._has_result

    def _reveal_manifest(self) -> None:
        if self._report_path is not None:
            self._reveal_path(self._report_path)

    def refresh_files(self) -> None:
        """Refresh only existence evidence; never alter the run's saved records."""
        current_row = self.output_table.currentRow()
        selected_path = (
            self._selected_paths[current_row]
            if 0 <= current_row < len(self._selected_paths)
            else None
        )
        scroll = self.output_table.verticalScrollBar().value()
        self._show_selected_files()
        if selected_path in self._selected_paths:
            self.output_table.selectRow(self._selected_paths.index(selected_path))
        self.output_table.verticalScrollBar().setValue(scroll)
        self._sync_artifact_buttons()


__all__ = ["BatchResultsPanel"]
