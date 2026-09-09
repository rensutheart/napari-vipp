"""Offline, review-before-export UI for detached reproducibility evidence."""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from qtpy.QtCore import (
    QByteArray,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)


def build_reproducibility_package(*args, **kwargs):
    """Resolve the Qt-free builder lazily; importing this dialog never builds."""
    from napari_vipp.core.reproducibility import build_reproducibility_package

    return build_reproducibility_package(*args, **kwargs)


def export_reproducibility_package(*args, **kwargs):
    from napari_vipp.core.reproducibility import export_reproducibility_package

    return export_reproducibility_package(*args, **kwargs)


@dataclass(frozen=True)
class _TaskResult:
    task_id: str
    generation: int
    kind: str
    value: Any = None
    error: str = ""
    cancelled: bool = False


class _TaskSignals(QObject):
    finished = Signal(object)


class _PackageWorker(QRunnable):
    """Own only evidence and cancellation state, never a widget or bound method."""

    def __init__(self, task_id, generation, kind, arguments):
        super().__init__()
        self.task_id = task_id
        self.generation = generation
        self.kind = kind
        self.arguments = arguments
        self.cancel_event = threading.Event()
        self.signals = _TaskSignals()

    def run(self):
        value, error = None, ""
        try:
            if not self.cancel_event.is_set():
                if self.kind == "prepare":
                    value = build_reproducibility_package(**self.arguments)
                else:
                    value = export_reproducibility_package(**self.arguments)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        cancelled = self.cancel_event.is_set()
        result = _TaskResult(
            self.task_id, self.generation, self.kind,
            None if cancelled else value, error, cancelled,
        )
        try:
            self.signals.finished.emit(result)
        except RuntimeError:
            # QApplication may already be tearing down. No widget is accessed.
            pass


class _PackageTasks(QObject):
    """Application-owned jobs survive dialog deletion without blocking its close."""

    finished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: dict[str, _PackageWorker] = {}

    def start(self, generation, kind, arguments):
        task_id = uuid4().hex
        worker = _PackageWorker(task_id, generation, kind, arguments)
        self._jobs[task_id] = worker
        worker.signals.finished.connect(self._finished)
        QThreadPool.globalInstance().start(worker)
        return task_id

    def cancel(self, task_id):
        worker = self._jobs.get(task_id)
        if worker is not None:
            worker.cancel_event.set()

    @Slot(object)
    def _finished(self, result):
        self._jobs.pop(result.task_id, None)
        self.finished.emit(result)


def _task_service():
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication is required for package review.")
    service = getattr(app, "_vipp_reproducibility_tasks", None)
    if service is None:
        service = _PackageTasks(app)
        app._vipp_reproducibility_tasks = service
    return service


class _OfflineReportBrowser(QTextBrowser):
    """A report must never fetch local files, remote images or linked resources."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reflow_timer = QTimer(self)
        self._reflow_timer.setSingleShot(True)
        self._reflow_timer.timeout.connect(self._reflow)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)

    def loadResource(self, resource_type, name):  # noqa: N802
        return QByteArray()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        # Force percentage-width tables to reflow after narrowing. Qt may keep
        # their earlier wider layout even though document.textWidth is updated.
        # This changes only layout, never the exact prepared report contents.
        self._reflow_timer.start(0)

    def _reflow(self):
        self.document().setTextWidth(max(1, self.viewport().width()))


_PRIVACY_NOTICE = (
    "Nothing is uploaded. Workflow notes are included; images, results and previews "
    "are shared separately. Source folders are hidden. "
    "Filenames and notes can still identify people or samples; "
    "anonymising filenames does not anonymise scientific labels or free text. "
    "Review the report and every included file before sharing."
)


class ReproducibilityDialog(QDialog):
    """Review one recipe snapshot or one archived batch manifest, never both.

    Preparation uses a detached opening snapshot, not a live graph. The exact
    prepared package is exported only after a fresh explicit review. Closing a
    preparation discards its eventual result without waiting for native work.
    Export cannot be cancelled after its atomic writer starts; close is held
    until that operation reports success or failure.
    """

    exported = Signal(str)

    def __init__(
        self, parent=None, *, workflow=None, manifest_path=None, title="VIPP analysis"
    ):
        super().__init__(parent)
        if (workflow is None) == (manifest_path is None):
            raise ValueError("Choose one workflow snapshot or archived batch manifest.")
        self._workflow = copy.deepcopy(workflow)
        self._manifest_path = None if manifest_path is None else Path(manifest_path)
        self._tasks = _task_service()
        self._tasks.finished.connect(self._task_finished)
        self._generation = 0
        self._active_task: str | None = None
        self._active_kind = ""
        self._closed = False
        self._prepared_package = None
        self._member_names: list[str] = []
        self.setWindowTitle("Reproducibility package")
        self.resize(900, 720)

        layout = QVBoxLayout(self)
        self.source_label = self._label(
            "Workflow recipe snapshot — describes the graph at opening, not a "
            "verified record of a past run."
            if workflow is not None
            else "Archived batch evidence — prepared from the saved manifest, "
            "not the currently edited graph."
        )
        layout.addWidget(self.source_label)
        self.privacy_label = self._label(_PRIVACY_NOTICE)
        layout.addWidget(self.privacy_label)

        fields = QFormLayout()
        fields.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.title_edit = QLineEdit(str(title))
        self.title_edit.setPlaceholderText("VIPP analysis")
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional context; included when shared.")
        self.notes_edit.setMaximumHeight(80)
        fields.addRow("Report title", self.title_edit)
        fields.addRow("Notes (optional)", self.notes_edit)
        layout.addLayout(fields)
        self.anonymise_checkbox = QCheckBox("Anonymise filenames")
        self.anonymise_checkbox.setToolTip(
            "Replace filenames in the prepared package. Folder paths are always "
            "hidden. Review scientific labels and free text separately."
        )
        layout.addWidget(self.anonymise_checkbox)

        self.tabs = QTabWidget()
        self.report_browser = _OfflineReportBrowser()
        self.tabs.addTab(self.report_browser, "Report")
        contents = QSplitter(Qt.Horizontal)
        self.contents_list = QListWidget()
        self.member_preview = QPlainTextEdit()
        self.member_preview.setReadOnly(True)
        contents.addWidget(self.contents_list)
        contents.addWidget(self.member_preview)
        contents.setStretchFactor(1, 1)
        self.tabs.addTab(contents, "Package contents")
        self.privacy_browser = QPlainTextEdit()
        self.privacy_browser.setReadOnly(True)
        self.tabs.addTab(self.privacy_browser, "Privacy & omissions")
        layout.addWidget(self.tabs, 1)

        self.review_checkbox = QCheckBox(
            "I reviewed the report and included files for sharing."
        )
        self.review_checkbox.setEnabled(False)
        layout.addWidget(self.review_checkbox)
        self.status_label = self._label("Prepare the report to review its contents.")
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare report")
        self.export_button = QPushButton("Export package…")
        self.export_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        buttons.addWidget(self.prepare_button)
        buttons.addStretch(1)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.title_edit.textChanged.connect(self._invalidate)
        self.notes_edit.textChanged.connect(self._invalidate)
        self.anonymise_checkbox.toggled.connect(self._invalidate)
        self.review_checkbox.toggled.connect(self._update_controls)
        self.contents_list.currentRowChanged.connect(self._show_member)
        self.report_browser.anchorClicked.connect(self._open_report_link)
        self.prepare_button.clicked.connect(self.prepare_report)
        self.export_button.clicked.connect(self.export_package)
        self.close_button.clicked.connect(self.reject)
        self._clear_preview()

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        return label

    @property
    def prepared_package(self):
        """The currently reviewed snapshot, or None after an input change."""
        return self._prepared_package

    def _clear_preview(self):
        self._prepared_package = None
        self._member_names = []
        self.report_browser.setPlainText("Prepare the report to review it here.")
        self.contents_list.clear()
        self.member_preview.clear()
        self.privacy_browser.setPlainText(_PRIVACY_NOTICE)
        self.review_checkbox.setChecked(False)

    def _invalidate(self, *_args):
        self._generation += 1
        self._clear_preview()
        if self._active_task is not None and self._active_kind == "prepare":
            self._tasks.cancel(self._active_task)
            self.status_label.setText(
                "Settings changed. Previous preparation is finishing; "
                "prepare the report again afterward."
            )
        else:
            self.status_label.setText("Settings changed. Prepare a new report.")
        self._update_controls()

    def _update_controls(self, *_args):
        busy = self._active_task is not None
        writing = busy and self._active_kind == "export"
        ready = self._prepared_package is not None and not busy and not self._closed
        self.prepare_button.setEnabled(not busy and not self._closed)
        self.prepare_button.setText(
            "Preparing…" if busy and not writing else "Prepare report"
        )
        self.review_checkbox.setEnabled(ready)
        self.export_button.setEnabled(ready and self.review_checkbox.isChecked())
        self.close_button.setEnabled(not writing)
        for field in (self.title_edit, self.notes_edit, self.anonymise_checkbox):
            field.setEnabled(not writing)

    @Slot()
    def prepare_report(self):
        if self._closed or self._active_task is not None:
            return
        if not self.title_edit.text().strip():
            self.status_label.setText("Enter a report title before preparing.")
            return
        self._generation += 1
        self._clear_preview()
        self._active_kind = "prepare"
        self._active_task = self._tasks.start(
            self._generation,
            "prepare",
            {
                "workflow": copy.deepcopy(self._workflow),
                "manifest_path": self._manifest_path,
                "title": self.title_edit.text(),
                "anonymise_filenames": self.anonymise_checkbox.isChecked(),
                "notes": self.notes_edit.toPlainText(),
            },
        )
        self.status_label.setText("Preparing report in the background…")
        self._update_controls()

    @Slot(object)
    def _task_finished(self, result):
        if self._closed or result.task_id != self._active_task:
            return
        self._active_task = None
        self._active_kind = ""
        if result.generation != self._generation or result.cancelled:
            self.status_label.setText("Settings changed. Prepare a new report.")
        elif result.error:
            self.status_label.setText(
                f"Could not {result.kind} package. {result.error}"
            )
        elif result.kind == "prepare":
            try:
                self._show_package(result.value)
            except Exception as exc:
                self._clear_preview()
                self.status_label.setText(f"Could not preview package. {exc}")
        else:
            self.status_label.setText(f"Package saved locally: {result.value}")
            self.exported.emit(str(result.value))
        self._update_controls()

    def _show_package(self, package):
        # The archive's actual report bytes are the review source of truth.
        report = package.members["report.html"].decode("utf-8")
        self.report_browser.setHtml(report)
        self._prepared_package = package
        self._member_names = sorted(package.members)
        for name in self._member_names:
            self.contents_list.addItem(
                f"{name}  ({len(package.members[name]):,} bytes)"
            )
        self.contents_list.setCurrentRow(0)
        data = package.report_data
        sections = [_PRIVACY_NOTICE]
        privacy = data.get("privacy") or {}
        if privacy.get("anonymise_filenames") is True:
            sections.append("Filenames have been replaced with anonymous names.")
        elif privacy.get("anonymise_filenames") is False:
            sections.append("Original filenames are included.")
        for key, heading in (
            ("omissions", "Deliberately left out"),
            ("changes", "Changes made for sharing"),
            ("limitations", "Limits to keep in mind"),
        ):
            if data.get(key):
                sections.append(
                    heading + "\n" + "\n".join(f"• {item}" for item in data[key])
                )
        for key, value in data.items():
            if "warning" in key.casefold() and value:
                sections.append(
                    "Warnings\n"
                    + (
                        value
                        if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False)
                    )
                )
        sections.append("Use Package contents to inspect every included file.")
        self.privacy_browser.setPlainText("\n\n".join(sections))
        self.status_label.setText(
            "Prepared locally. Review all tabs, then confirm the contents for sharing."
        )

    def _show_member(self, row):
        if self._prepared_package is None or not 0 <= row < len(self._member_names):
            self.member_preview.clear()
            return
        data = self._prepared_package.members[self._member_names[row]]
        try:
            self.member_preview.setPlainText(data.decode("utf-8"))
        except UnicodeDecodeError:
            self.member_preview.setPlainText(
                "This member is not UTF-8 text and cannot be previewed here."
            )

    @Slot(QUrl)
    def _open_report_link(self, url):
        """Only explicit clicks may open our generated official installation pages.

        Relative workflow links work in an extracted report. In this unsaved
        preview they must not launch a stale local file or replace the graph.
        All automatic resource loading stays disabled in the text browser.
        """
        if self._prepared_package is None or self._closed:
            return
        target = url.toString()
        if target == "workflow.json":
            self.status_label.setText(
                "workflow.json is included in the package. Export and unzip it, "
                "then use Open in VIPP to open that file."
            )
            return
        from napari_vipp.core.reproducibility_install import (
            INSTALLATION_GUIDE_URL,
            installation_guidance,
        )

        data = self._prepared_package.report_data
        guidance = installation_guidance(
            data.get("environment"),
            recorded=data.get("package_kind") == "recorded_batch_run",
        )
        if target not in {guidance["url"], INSTALLATION_GUIDE_URL}:
            self.status_label.setText("This link is not a VIPP installation page.")
            return
        if not QDesktopServices.openUrl(QUrl(target)):
            self.status_label.setText("Could not open the VIPP installation page.")

    @Slot()
    def export_package(self):
        if (
            self._closed or self._active_task is not None
            or self._prepared_package is None or not self.review_checkbox.isChecked()
        ):
            return
        reviewed_package = self._prepared_package
        reviewed_generation = self._generation
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Export reproducibility package", "vipp-reproducibility-package.zip",
            "Reproducibility package (*.zip)",
        )
        if not path:
            return
        # A nested file dialog can dispatch other application events. Recheck
        # review state afterward rather than authorizing a replacement snapshot.
        if (
            self._closed or self._active_task is not None
            or self._prepared_package is not reviewed_package
            or self._generation != reviewed_generation
            or not self.review_checkbox.isChecked()
        ):
            return
        target = Path(path)
        if target.exists():
            self.status_label.setText(
                "That file already exists. Choose a new filename; "
                "existing packages are not overwritten."
            )
            return
        self._active_kind = "export"
        self._active_task = self._tasks.start(
            self._generation, "export",
            {"package": self._prepared_package, "path": target, "overwrite": False},
        )
        self.status_label.setText(
            "Writing the reviewed package locally… "
            "Close is available when writing ends."
        )
        self._update_controls()

    def _finish_close(self):
        self._closed = True
        self._generation += 1
        if self._active_task is not None:
            self._tasks.cancel(self._active_task)
        self._clear_preview()

    def reject(self):
        if self._active_task is not None and self._active_kind == "export":
            return
        self._finish_close()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        if self._active_task is not None and self._active_kind == "export":
            event.ignore()
            return
        self._finish_close()
        super().closeEvent(event)


__all__ = ["ReproducibilityDialog"]
