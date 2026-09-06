"""Progressive, non-authoritative inventory shown while batch checks run."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

from qtpy.QtCore import QRectF, QSignalBlocker, Qt, QTimer
from qtpy.QtGui import QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QHeaderView, QTableWidgetItem

from napari_vipp.ui.palette_roles import theme_colors


@dataclass(frozen=True, slots=True)
class _CheckRow:
    name: str
    sources: tuple[tuple[str, str, Path], ...]
    outputs: int | None = None


class BatchCheckProgressPresentation:
    """Keep early directory evidence separate from the runnable batch plan."""

    def _init_check_progress(self) -> None:
        self._check_rows: tuple[_CheckRow, ...] | None = None
        self._check_file_states: dict[tuple[str, Path], str] = {}
        self._check_active_file: tuple[str, Path] | None = None
        self._check_phase = ""
        self._check_is_sample_list = False
        self._check_current = 0
        self._check_total = 0
        self._check_visible_positions: list[int] = []
        self._check_spinner_angle = 0
        self._check_spinner_timer = QTimer(self)
        self._check_spinner_timer.setInterval(100)
        self._check_spinner_timer.timeout.connect(self._animate_check_spinner)

    def _clear_check_progress(self) -> None:
        self._check_rows = None
        # The reviewed plan retains its existing wrapped-name presentation.
        self.preview_table.setWordWrap(True)
        self._check_file_states.clear()
        self._check_active_file = None
        self._check_visible_positions.clear()
        self._check_spinner_timer.stop()
        self._check_phase = ""
        self.check_count_label.hide()
        self.items_heading.setText("Review batch items")
        self.item_filter.setEnabled(True)

    def _finish_check_progress_failure(self) -> None:
        if self._check_rows is None:
            return
        if self._check_active_file is not None:
            self._check_file_states[self._check_active_file] = "Needs attention"
        self._check_phase = "failed"
        self._check_active_file = None
        self._check_spinner_timer.stop()
        self.items_heading.setText("Review source files · check incomplete")
        self.check_count_label.setText(
            f"{self._check_current:,} of {self._check_total:,} files checked"
            " · needs attention"
        )

    def show_check_progress(self, event, *, reveal_items: bool = True) -> None:
        """Show immutable worker evidence without publishing a runnable plan.

        Directory inventory counts files, not samples: one microscope file may
        expand to several series. The exact sample list replaces it only after
        readers have inspected the metadata. Even then the final validated plan
        must arrive via ``apply_preview_result`` before actions become available.
        """
        self._checking_plan = True
        self._preview_result = None
        self._check_phase = event.phase
        self._check_current = event.current
        self._check_total = event.total
        if event.phase == "discovered":
            self._clear_check_progress()
            self._check_phase = event.phase
            self._check_is_sample_list = False
            self._check_rows = tuple(
                _CheckRow(Path(path).name, ((node_id, title, Path(path)),))
                for node_id, title, paths in event.source_paths
                for path in paths
            )
            self._item_page = 0
            self._current_item = 0
            self.item_filter.setCurrentIndex(0)
            self.item_filter.setEnabled(False)
            self.items_heading.setText("Source files found · checking batch items")
            self.preview_status.setText(
                "Files are listed below while VIPP reads image metadata and "
                "verifies exact file contents. One file may contain several "
                "samples; the final batch list appears after these checks."
            )
            self.graph_preview_status.setText(
                "No workflow images are calculated or outputs saved during Check. "
                "Preview, overrides, and Run unlock when all checks finish."
            )
            self._render_items()
            self._sync_workspace()
            if reveal_items:
                self.tabs.setCurrentIndex(1)
        elif event.phase in {"checking", "checked"} and event.path is not None:
            key = (event.source_node_id, Path(event.path))
            if event.phase == "checking":
                self._check_active_file = key
                self._check_file_states[key] = "Checking…"
                self._check_spinner_timer.start()
            else:
                self._check_file_states[key] = (
                    "Needs attention" if event.warning else "Checked"
                )
                self._check_active_file = None
                self._check_spinner_timer.stop()
        elif event.phase == "planning" and event.plan is not None:
            self._check_active_file = None
            self._check_spinner_timer.stop()
            self._check_is_sample_list = True
            source_titles = {
                str(row["node_id"]): str(row["title"]) for row in self._source_rows
            }
            self._check_rows = tuple(
                _CheckRow(
                    item.batch_id,
                    tuple(
                        (node_id, source_titles.get(node_id, node_id), Path(path))
                        for node_id, path in item.source_paths.items()
                    ),
                    len(item.outputs),
                )
                for item in event.plan.items
            )
            self._item_page = 0
            self._current_item = 0
            self.items_heading.setText("Review batch items · finalizing checks")
            source_summary = (
                "Source inspection finished with warnings"
                if "Needs attention" in self._check_file_states.values()
                else "Source files are checked"
            )
            self.preview_status.setText(
                f"Found {len(self._check_rows):,} batch items. {source_summary}; "
                "validating workflow image axes and parameters before "
                "making this plan available."
            )
            self._render_items()
            self._sync_workspace()
        elif event.phase == "contract" and event.path is not None:
            self._check_active_file = (event.source_node_id, Path(event.path))
            self._check_spinner_timer.start()
        elif event.phase == "complete":
            self._check_active_file = None
            self._check_spinner_timer.stop()

        if self._check_rows is not None:
            count = f"{event.current:,} of {event.total:,} files checked"
            if self._check_is_sample_list:
                count += " · finalizing checks"
            self.check_count_label.setText(count)
            self.check_count_label.setToolTip(event.message)
            self.check_count_label.show()
            self._refresh_check_cells()
            if event.phase in {"checked", "planning"}:
                self._show_check_item_details()
        message = event.message or "Checking image metadata and file contents…"
        progress = f"{event.current:,} / {event.total:,}"
        if event.phase in {"planning", "contract", "complete"}:
            label = "Finalizing batch checks…"
        else:
            label = f"Checking files · {event.current:,} of {event.total:,} checked"
        if event.byte_total:
            percent = min(100, int(100 * event.byte_current / event.byte_total))
            label += f" · reading file {percent}%"
        self.show_workspace_activity(
            label,
            state="working",
            current=event.current,
            total=event.total,
            indeterminate=not event.total,
            progress_text=progress,
            tooltip=message,
        )

    def _check_row_status(self, row: _CheckRow) -> str:
        keys = tuple((node_id, path) for node_id, _title, path in row.sources)
        if self._check_active_file in keys:
            return "Validating…" if self._check_phase == "contract" else "Checking…"
        statuses = tuple(self._check_file_states.get(key, "Waiting") for key in keys)
        if "Needs attention" in statuses:
            return "Needs attention"
        if self._check_is_sample_list:
            return "Sources checked"
        if statuses and all(status == "Checked" for status in statuses):
            return "Checked"
        return "Not checked" if self._check_phase == "failed" else "Waiting"

    def _render_check_items(self) -> None:
        query = self.item_search.text().strip().casefold()
        positions = [
            index
            for index, row in enumerate(self._check_rows)
            if not query
            or query
            in " ".join(
                (row.name, *(str(path) for _node, _title, path in row.sources))
            ).casefold()
        ]
        page_size = self._ITEM_PAGE_SIZE
        self._item_page = min(
            self._item_page, max((len(positions) - 1) // page_size, 0)
        )
        visible = positions[
            self._item_page * page_size : (self._item_page + 1) * page_size
        ]
        self._check_visible_positions = visible
        self._rendering_items = True
        blocker = QSignalBlocker(self.preview_table)
        try:
            # File inventory is a compact, single-line list. In particular,
            # never measure wrapped names before the stretched column has its
            # final width: those oversized row heights persist after layout.
            self.preview_table.setWordWrap(False)
            self.preview_table.setColumnCount(6)
            self.preview_table.setHorizontalHeaderLabels(
                [
                    "#",
                    "Batch item" if self._check_is_sample_list else "Source file",
                    "Sources",
                    "Outputs",
                    "Checks",
                    "Run result",
                ]
            )
            self.preview_table.setRowCount(len(visible))
            self._preview_table_rows.clear()
            for table_row, position in enumerate(visible):
                row = self._check_rows[position]
                sources = (
                    f"{len(row.sources)} paired"
                    if len(row.sources) > 1
                    else "1 source"
                )
                values = (
                    str(position + 1),
                    row.name,
                    sources,
                    "Pending" if row.outputs is None else f"{row.outputs} planned",
                    self._check_row_status(row),
                    "Not run",
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    cell.setData(Qt.UserRole, position)
                    cell.setToolTip(
                        "\n".join(str(path) for _node, _title, path in row.sources)
                        if column == 1
                        else "\n".join(title for _node, title, _path in row.sources)
                        if column == 2
                        else value
                    )
                    self.preview_table.setItem(table_row, column, cell)
                if position == self._current_item:
                    self.preview_table.selectRow(table_row)
            header = self.preview_table.horizontalHeader()
            for column in (0, 2, 3, 5):
                header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            # Fit the usual status and spinner, not the longest exceptional
            # message. Full status remains in the tooltip and details panel.
            # Keep this width stable while per-file progress changes.
            header.setSectionResizeMode(4, QHeaderView.Fixed)
            status_width = max(
                self.preview_table.fontMetrics().horizontalAdvance(text)
                for text in (
                    "Checking…",
                    "Validating…",
                )
            )
            self.preview_table.setColumnWidth(4, status_width + 18 + 28)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            self.preview_table.resizeRowsToContents()
        finally:
            del blocker
            self._rendering_items = False
        unit = "items" if self._check_is_sample_list else "files"
        if visible:
            start = self._item_page * page_size + 1
            self.item_range_label.setText(
                f"{start:,}–{start + len(visible) - 1:,} of {len(positions):,} {unit}"
            )
        else:
            self.item_range_label.setText(f"No matching {unit}")
        multiple = len(positions) > page_size
        self.items_previous_button.setVisible(multiple)
        self.items_next_button.setVisible(multiple)
        self.items_previous_button.setEnabled(self._item_page > 0)
        self.items_next_button.setEnabled(
            (self._item_page + 1) * page_size < len(positions)
        )
        self._refresh_check_cells()
        self._show_check_item_details()

    def _animate_check_spinner(self) -> None:
        self._check_spinner_angle = (self._check_spinner_angle + 30) % 360
        self._refresh_check_cells()

    def _refresh_check_cells(self) -> None:
        if self._check_rows is None:
            return
        tones = theme_colors(self.palette())
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(tones.info.foreground, 2, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(QRectF(3, 3, 12, 12), self._check_spinner_angle * 16, 250 * 16)
        painter.end()
        blocker = QSignalBlocker(self.preview_table)
        for table_row, position in enumerate(self._check_visible_positions):
            cell = self.preview_table.item(table_row, 4)
            if cell is None:
                continue
            status = self._check_row_status(self._check_rows[position])
            cell.setText(status)
            cell.setToolTip(status)
            active = status in {"Checking…", "Validating…"}
            cell.setIcon(QIcon(pixmap) if active else QIcon())
            color = (
                tones.info.foreground
                if active
                else tones.success.foreground
                if status in {"Checked", "Sources checked"}
                else tones.warning.foreground
                if status == "Needs attention"
                else tones.text
            )
            cell.setForeground(color)
        del blocker

    def _show_check_item_details(self) -> None:
        if self._rendering_items or self._check_rows is None:
            return
        selected = self.preview_table.selectionModel().selectedRows()
        if selected:
            first = self.preview_table.item(selected[0].row(), 0)
            self._current_item = int(first.data(Qt.UserRole))
        if not 0 <= self._current_item < len(self._check_rows):
            self.item_details.clear()
            return
        row = self._check_rows[self._current_item]
        parts = [f"<p><b>{escape(row.name)}</b></p>"]
        for node_id, title, path in row.sources:
            status = self._check_file_states.get((node_id, path), "Waiting")
            parts.append(
                f"<p><b>{escape(title)}</b><br>{escape(str(path))}<br>"
                f"{escape(status)}</p>"
            )
        parts.append(
            "<p>This inventory is not yet a runnable plan. Checks read image "
            "metadata, verify the exact contents of input files, and validate "
            "output destinations and workflow parameters.</p>"
            "<p>No processing results are calculated or saved.</p>"
        )
        self.item_details.setHtml("".join(parts))
