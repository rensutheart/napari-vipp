"""Review saved batch measurement tables without reopening source images."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from qtpy.QtCore import (
    QAbstractTableModel,
    QEvent,
    QObject,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
)

from napari_vipp.ui.batch_table_style import apply_batch_table_style
from napari_vipp.ui.dialog_buttons import DialogButtonBox as QDialogButtonBox
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.palette_roles import theme_colors

_ANNOTATIONS = ("condition", "sample", "replicate")
_READY = frozenset({"ready", "empty"})
_STATUS_LABELS = {
    "ready": "Ready",
    "empty": "Ready · no rows",
    "failed": "Failed",
    "skipped": "Skipped",
    "partial": "Incomplete",
    "missing": "Missing file",
    "changed": "File changed",
    "unsupported": "Unsupported table",
    "unavailable": "No saved result",
}


def _label(text="", *, bold=False):
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


def _short_text(value, maximum=240):
    text = str(value)
    return text if len(text) <= maximum else text[:maximum - 1] + "…"


def _basename(path):
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class _Request:
    manifest: str
    output_id: str | None = None
    preview: Any = None
    annotations: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    included_ids: tuple[str, ...] = ()
    reviewed_exclusions: bool = False
    path: str = ""
    expected_sha256: str = ""
    export_format: str = "csv"
    include_summary: bool = True
    overwrite: bool = False
    open_workflow: bool = False
    destination_revision: tuple[int, int, int, int] | None = None
    export_destination_revisions: (
        tuple[tuple[int, int, int, int] | None, ...] | None
    ) = None


@dataclass(frozen=True)
class _Reply:
    task_id: str
    generation: int
    kind: str
    value: Any = None
    error: str = ""
    cancelled: bool = False
    open_workflow: bool = False


@dataclass(frozen=True)
class _ExportChoice:
    format: str = "csv"
    include_summary: bool = True


class _ExportOptionsDialog(QDialog):
    """Keep export format and zero-image accounting explicit before choosing a file."""

    def __init__(self, parent=None):
        self._options_ready = False
        super().__init__(parent)
        self.setWindowTitle("Export reviewed results")
        self.setMinimumWidth(460)
        self.resize(570, 340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(_label("Choose a format", bold=True))
        self.format_combo = QComboBox()
        self.format_combo.setAccessibleName("Export format")
        self.format_combo.addItem("CSV · comma-separated table", "csv")
        self.format_combo.addItem("TSV · tab-separated table", "tsv")
        self.format_combo.addItem("Excel workbook (.xlsx)", "xlsx")
        layout.addWidget(self.format_combo)
        self.description_label = _label()
        layout.addWidget(self.description_label)
        self.summary_check = QCheckBox("Include image-summary file (recommended)")
        self.summary_check.setChecked(True)
        layout.addWidget(self.summary_check)
        self.summary_help = _label(
            "The summary lists every image, including images with no measurements "
            "and excluded images. It keeps these separate from the measurement rows."
        )
        layout.addWidget(self.summary_help)
        self.format_note = _label()
        layout.addWidget(self.format_note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Save)
        self.buttons.button(QDialogButtonBox.Save).setText("Choose file…")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        self._format_changed()
        if parent is not None:
            font = getattr(parent, "_requested_font", None) or parent.font()
            self.setFont(font)
            family = font.family().replace("'", "\\'")
            size = (
                f"{font.pointSizeF():g}pt" if font.pointSizeF() > 0
                else f"{font.pixelSize()}px"
            )
            for kind in (QLabel, QComboBox, QCheckBox, QPushButton):
                for widget in self.findChildren(kind):
                    widget.setStyleSheet(
                        f"font-family: '{family}'; font-size: {size};"
                    )
        self._options_ready = True
        self.ensurePolished()
        self._fit_content_height()

    def _format_changed(self):
        workbook = self.format_combo.currentData() == "xlsx"
        self.summary_check.setVisible(not workbook)
        self.description_label.setText(
            "One workbook with Measurements, Image summary and About this collection "
            "sheets. Images with no measurements remain in the summary, not in "
            "the measurement rows."
            if workbook else
            "One row per measurement, ready for Excel or other analysis software. "
            "The optional companion summary uses the same file format. Without it, "
            "images with no measurements and excluded images are absent from the table."
        )
        self.format_note.setText(
            "Excel has its own numeric precision. Save a VIPP collection when exact "
            "VIPP values and types must be retained."
            if workbook else
            "CSV/TSV do not store VIPP types or units. Spreadsheet apps may "
            "reinterpret text or IDs; use Excel workbook to keep text cells literal."
        )
        if self._options_ready:
            self._fit_content_height()

    def _fit_content_height(self):
        self.layout().invalidate()
        height = self.layout().totalHeightForWidth(self.width())
        if height > 0:
            self.setMinimumHeight(height)
            self.resize(self.width(), height)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._options_ready and event.oldSize().width() != event.size().width():
            self._fit_content_height()

    def choice(self):
        return _ExportChoice(
            self.format_combo.currentData(),
            self.format_combo.currentData() == "xlsx" or self.summary_check.isChecked(),
        )


def _choose_export_options(parent):
    dialog = _ExportOptionsDialog(parent)
    return dialog.choice() if dialog.exec() == QDialog.Accepted else None


def _export_targets(path, choice):
    from napari_vipp.core.measurement_export import measurement_export_targets

    return measurement_export_targets(
        path, format=choice.format, include_image_summary=choice.include_summary
    )


def _export_revisions(targets):
    from napari_vipp.core.measurement_export import (
        measurement_export_destination_revisions,
    )

    return measurement_export_destination_revisions(targets)


def _inspect(request, cancellation, progress):
    from napari_vipp.core.measurement_collection import (
        available_measurement_outputs,
        inspect_collection,
    )

    outputs = available_measurement_outputs(request.manifest)
    if cancellation.is_set() or not outputs:
        return outputs, None, ()
    output_id = request.output_id or outputs[0].node_id
    preview = inspect_collection(
        request.manifest,
        output_id,
        cancellation=cancellation,
        progress=progress,
    )
    preserved = []
    for item in preview.items:
        if cancellation.is_set():
            return outputs, None, ()
        fields = tuple(
            (field, value) for field in _ANNOTATIONS
            if (value := _preserved_annotation(item, field, cancellation)) is not None
        )
        if fields:
            preserved.append((item.key, fields))
    return outputs, preview, tuple(preserved)


def _collect(request, cancellation):
    from napari_vipp.core.measurement_collection import collect_measurements

    if cancellation.is_set():
        return None
    return collect_measurements(
        request.preview,
        annotations={key: dict(values) for key, values in request.annotations},
        included_ids=request.included_ids,
        reviewed_exclusions=request.reviewed_exclusions,
        cancellation=cancellation,
    )


def _save(request, cancellation, progress):
    from napari_vipp.core.measurement_collection import save_measurement_collection

    progress(0, 0, "Validating and saving the reviewed table snapshot…")
    collection = _collect(request, cancellation)
    if cancellation.is_set():
        return None
    save_measurement_collection(
        collection, request.path, cancellation=cancellation,
        expected_destination_revision=request.destination_revision,
    )
    return request.path


def _export(request, cancellation, progress):
    from napari_vipp.core.measurement_export import export_measurement_collection

    progress(0, 0, "Preparing the reviewed measurement rows and image inventory…")
    collection = _collect(request, cancellation)
    if cancellation.is_set():
        return None
    return export_measurement_collection(
        collection, request.path,
        format=request.export_format,
        include_image_summary=request.include_summary,
        overwrite=request.overwrite,
        expected_destination_revisions=request.export_destination_revisions,
        cancellation=cancellation,
        progress=lambda fraction, message: progress(
            round(fraction * 1000), 1000, message
        ),
    )


def _review(request, cancellation, progress):
    from napari_vipp.core.measurement_collection import (
        MeasurementOutput,
        MeasurementPreview,
        load_measurement_collection,
    )

    progress(0, 0, "Reading the saved collection; original files are not reopened.")
    collection = load_measurement_collection(
        request.manifest,
        expected_sha256=request.expected_sha256,
        cancellation=cancellation,
    )
    output = MeasurementOutput(
        collection.provenance["output_node_id"], collection.table.name, ""
    )
    preview = MeasurementPreview(
        output, collection.items, collection.provenance["run_id"],
        collection.provenance["manifest_sha256"],
        collection.provenance["workflow_sha256"],
    )
    columns = collection.table.columns
    item_column = columns.index("_vipp_item_key")
    field_columns = {
        field: [index for index, name in enumerate(columns) if name.casefold() == field]
        for field in _ANNOTATIONS
    }
    values = {}
    for index, row in enumerate(collection.table.rows):
        if index % 1024 == 0 and cancellation.is_set():
            return (), None, (), ()
        fields = values.setdefault(row[item_column], {})
        for field, indices in field_columns.items():
            observed = fields.setdefault(field, set())
            if len(observed) < 2:
                observed.update(str(row[column]) for column in indices)
    preserved = []
    for item in collection.items:
        fields = {}
        for field, indices in field_columns.items():
            if not indices:
                continue
            if field in collection.provenance["annotation_columns"]:
                fields[field] = collection.annotations.get(item.key, {}).get(field, "")
                continue
            observed = values.get(item.key, {}).get(field, set())
            if len(observed) == 1:
                fields[field] = next(iter(observed))
            elif observed:
                fields[field] = "Multiple values (kept)"
            else:
                fields[field] = (
                    "Existing column (no rows)" if item.included else "Not included"
                )
        preserved.append((item.key, tuple(fields.items())))
    included = tuple(item.key for item in collection.items if item.included)
    return (output,), preview, tuple(preserved), included


class _CollectionWorker(QThread):
    """Immutable work requests; no widget or GUI-bound callback enters run()."""

    completed = Signal(object)
    progressed = Signal(object)

    def __init__(self, task_id, generation, kind, request):
        super().__init__()
        self.task_id = task_id
        self.generation = generation
        self.kind = kind
        self.request = request
        self.cancellation = threading.Event()

    def _progress(self, current, total, message):
        self.progressed.emit((
            self.task_id, self.generation, current, total, str(message)
        ))

    def run(self):
        value, error = None, ""
        try:
            if not self.cancellation.is_set():
                operation = {
                    "save": _save, "export": _export,
                    "review": _review, "inspect": _inspect,
                }[self.kind]
                value = operation(self.request, self.cancellation, self._progress)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        # A completed atomic save wins over cancellation arriving after commit.
        committed = self.kind in {"save", "export"} and value is not None and not error
        cancelled = self.cancellation.is_set() and not committed
        self.completed.emit(_Reply(
            self.task_id, self.generation, self.kind,
            None if cancelled else value, error, cancelled, self.request.open_workflow,
        ))


class _CollectionTasks(QObject):
    """Application-owned threads survive a closed/deleted review dialog."""

    completed = Signal(object)
    progressed = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.jobs = {}
        parent.aboutToQuit.connect(self._shutdown)

    def start(self, generation, kind, request):
        task_id = uuid4().hex
        worker = _CollectionWorker(task_id, generation, kind, request)
        self.jobs[task_id] = worker
        worker.completed.connect(self.completed)
        worker.progressed.connect(self.progressed)
        worker.finished.connect(self._finished)
        worker.start()
        return task_id

    def cancel(self, task_id):
        worker = self.jobs.get(task_id)
        if worker is not None:
            worker.cancellation.set()

    @Slot()
    def _finished(self):
        worker = self.sender()
        self.jobs.pop(worker.task_id, None)
        worker.deleteLater()

    @Slot()
    def _shutdown(self):
        # Only application teardown waits; closing the dialog never waits.
        for worker in tuple(self.jobs.values()):
            worker.cancellation.set()
        for worker in tuple(self.jobs.values()):
            worker.wait()


def _tasks():
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("A QApplication is required to collect measurements.")
    service = getattr(app, "_vipp_measurement_collection_tasks", None)
    if service is None:
        service = _CollectionTasks(app)
        app._vipp_measurement_collection_tasks = service
    return service


def _preserved_annotation(item, field, cancellation=None):
    """Display existing table annotations without replacing their values."""
    table = item.table
    if table is None:
        return None
    columns = [
        index for index, column in enumerate(table.columns)
        if column.casefold() == field
    ]
    if not columns:
        return None
    values = set()
    for row_number, row in enumerate(table.rows):
        if row_number % 1024 == 0 and cancellation is not None:
            if cancellation.is_set():
                return None
        values.update(str(row[index]) for index in columns)
        if len(values) > 1:
            return "Multiple values (kept)"
    if not values:
        return "Existing column (no rows)"
    if len(values) == 1:
        return next(iter(values))
    return "Multiple values (kept)"


class _CollectionModel(QAbstractTableModel):
    changed = Signal()
    HEADERS = (
        "Include", "Image item", "Result", "Rows", "Source",
        "Condition", "Sample", "Replicate",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = ()
        self.included = set()
        self.annotations = {}
        self.preserved = {}
        self.editable = True

    def set_preview(self, preview, preserved=(), included_ids=None):
        self.beginResetModel()
        self.items = preview.items if preview is not None else ()
        self.included = (
            {item.key for item in self.items if item.status in _READY}
            if included_ids is None else set(included_ids)
        )
        self.annotations = {}
        self.preserved = {key: dict(fields) for key, fields in preserved}
        self.endResetModel()
        self.changed.emit()

    def rowCount(self, parent=None):  # noqa: N802
        return 0 if parent is not None and parent.isValid() else len(self.items)

    def columnCount(self, parent=None):  # noqa: N802
        return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self.items[index.row()]
        column = index.column()
        if column == 0 and role == Qt.CheckStateRole:
            return Qt.Checked if item.key in self.included else Qt.Unchecked
        if role == Qt.ToolTipRole:
            if column >= 5:
                field = _ANNOTATIONS[column - 5]
                preserved = self.preserved.get(item.key, {}).get(field)
                if preserved is not None:
                    return (
                        f"{preserved}\nSaved metadata is preserved and cannot be "
                        "overwritten here."
                    )
                return f"Optional {field} for this image item."
            if column == 4:
                return str(item.source_path)
            return str(item.message)
        if role == Qt.TextAlignmentRole and column == 3:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        if column == 0:
            return ""
        if column == 1:
            return item.batch_id
        if column == 2:
            return _STATUS_LABELS.get(item.status, item.status.replace("_", " "))
        if column == 3:
            return "—" if item.row_count is None else str(item.row_count)
        if column == 4:
            return _basename(item.source_path) or "Not recorded"
        field = _ANNOTATIONS[column - 5]
        preserved = self.preserved.get(item.key, {}).get(field)
        return (
            preserved if preserved is not None
            else self.annotations.get(item.key, {}).get(field, "")
        )

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        item = self.items[index.row()]
        if not self.editable or item.status not in _READY:
            return flags
        if index.column() == 0:
            return flags | Qt.ItemIsUserCheckable
        if index.column() >= 5:
            field = _ANNOTATIONS[index.column() - 5]
            if field not in self.preserved.get(item.key, {}):
                return flags | Qt.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.EditRole):  # noqa: N802
        if not index.isValid():
            return False
        item = self.items[index.row()]
        if role == Qt.CheckStateRole and self.flags(index) & Qt.ItemIsUserCheckable:
            if value == Qt.Checked:
                self.included.add(item.key)
            else:
                self.included.discard(item.key)
        elif role == Qt.EditRole and self.flags(index) & Qt.ItemIsEditable:
            field = _ANNOTATIONS[index.column() - 5]
            value = str(value).strip()
            self.annotations.setdefault(item.key, {})[field] = value
        else:
            return False
        self.dataChanged.emit(index, index)
        self.changed.emit()
        return True

    def apply_annotations(self, rows, field, value):
        if not rows:
            return
        for row in rows:
            self.annotations.setdefault(self.items[row].key, {})[field] = value
        column = 5 + _ANNOTATIONS.index(field)
        self.dataChanged.emit(
            self.index(min(rows), column), self.index(max(rows), column)
        )
        self.changed.emit()


class MeasurementCollectionDialog(QDialog):
    """Inspect one saved table output per image and save a detached collection."""

    collectionSaved = Signal(str)

    def __init__(
        self, manifest: str | Path, parent=None, *,
        _review_only=False, _expected_sha256="",
    ):
        self._ready = False
        self._requested_font = None
        super().__init__(parent)
        self.manifest = str(manifest)
        self._review_only = _review_only
        self._expected_sha256 = _expected_sha256
        self.preview = None
        self._generation = 0
        self._task_id = None
        self._closed = False
        self._service = _tasks()
        self._lifetime = {"task_id": None}
        self._service.completed.connect(self._completed)
        self._service.progressed.connect(self._progressed)
        service, lifetime = self._service, self._lifetime
        self.destroyed.connect(lambda: service.cancel(lifetime["task_id"]))
        self.setWindowTitle("Collect measurement results")
        self.setMinimumSize(600, 540)
        self.resize(1080, 740)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        self.title_label = _label("Collect measurement results", bold=True)
        layout.addWidget(self.title_label)
        self.intro_label = _label(
            "Combine saved tables from one batch output. Images are not opened "
            "and the workflow is not recalculated."
        )
        layout.addWidget(self.intro_label)
        chooser = QHBoxLayout()
        chooser.setSpacing(8)
        self.output_label = QLabel("Table output")
        chooser.addWidget(self.output_label)
        self.output_combo = QComboBox()
        self.output_combo.setMinimumWidth(0)
        self.output_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.output_combo.setMinimumContentsLength(12)
        self.output_combo.setAccessibleName("Saved measurement output")
        self.output_combo.currentIndexChanged.connect(self._output_changed)
        chooser.addWidget(self.output_combo, 1)
        self.inspect_button = QPushButton("Check files")
        self.inspect_button.setAutoDefault(False)
        self.inspect_button.clicked.connect(self.inspect)
        chooser.addWidget(self.inspect_button)
        layout.addLayout(chooser)

        self.status_panel = QFrame()
        self.status_panel.setObjectName("MeasurementCollectionStatus")
        status = QVBoxLayout(self.status_panel)
        status.setContentsMargins(10, 8, 10, 8)
        status.setSpacing(5)
        self.status_label = _label("Reading the saved batch record…", bold=True)
        self.progress_label = _label("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(8)
        status.addWidget(self.status_label)
        status.addWidget(self.progress_label)
        status.addWidget(self.progress_bar)
        layout.addWidget(self.status_panel)

        self.model = _CollectionModel(self)
        self.model.changed.connect(self._review_changed)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setMinimumSize(0, 140)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        self.item_detail = _label(
            "Select an image for file-check details. Double-click a blank "
            "annotation cell to edit it."
        )
        layout.addWidget(self.item_detail)
        self.annotation_help = _label(
            "Optional annotations · existing metadata is kept. Select rows to "
            "assign the same value to several images."
        )
        layout.addWidget(self.annotation_help)
        bulk = QGridLayout()
        bulk.setContentsMargins(0, 0, 0, 0)
        bulk.setHorizontalSpacing(8)
        self.field_combo = QComboBox()
        for field in _ANNOTATIONS:
            self.field_combo.addItem(field.title(), field)
        self.field_combo.setAccessibleName("Annotation field")
        self.annotation_edit = QLineEdit()
        self.annotation_edit.setPlaceholderText("Value for selected images")
        self.annotation_edit.setAccessibleName("Annotation value")
        self.apply_button = QPushButton("Apply to selected")
        self.apply_button.setAutoDefault(False)
        self.apply_button.clicked.connect(self.apply_annotation)
        bulk.addWidget(self.field_combo, 0, 0)
        bulk.addWidget(self.annotation_edit, 0, 1)
        bulk.addWidget(self.apply_button, 0, 2)
        bulk.setColumnStretch(1, 1)
        layout.addLayout(bulk)

        self.review_label = _label("")
        layout.addWidget(self.review_label)
        self.exclusions_check = QCheckBox("I have reviewed the excluded images.")
        self.exclusions_check.toggled.connect(self._update_actions)
        layout.addWidget(self.exclusions_check)
        self.open_workflow_check = QCheckBox(
            "Open VIPP collection in a results workflow"
        )
        self.open_workflow_check.setChecked(False)
        self.open_workflow_check.setToolTip(
            "Only applies to Save VIPP collection. Exporting CSV, TSV or Excel "
            "does not open a workflow tab."
        )
        layout.addWidget(self.open_workflow_check)
        footer = QHBoxLayout()
        self.footer_hint = _label(
            "Export CSV, TSV or Excel, or save a VIPP snapshot."
        )
        layout.addWidget(self.footer_hint)
        self.cancel_work_button = QPushButton("Cancel check")
        self.cancel_work_button.setAutoDefault(False)
        self.cancel_work_button.clicked.connect(self.cancel_work)
        work_actions = QHBoxLayout()
        work_actions.addStretch(1)
        work_actions.addWidget(self.cancel_work_button)
        status.addLayout(work_actions)
        self.close_button = QPushButton("Close")
        self.close_button.setAutoDefault(False)
        self.close_button.clicked.connect(self.reject)
        footer.addStretch(1)
        self.save_button = QPushButton("Save VIPP collection…")
        self.save_button.setAutoDefault(False)
        self.save_button.clicked.connect(self.save_collection)
        self.save_button.setToolTip(
            "Save a .vipp-results.json snapshot with typed tables, units, annotations "
            "and the complete image inventory."
        )
        self.export_button = QPushButton("Export results…")
        self.export_button.setAutoDefault(False)
        self.export_button.setDefault(True)
        self.export_button.clicked.connect(self.export_results)
        add_dialog_buttons(
            footer,
            actions=(self.save_button, self.export_button),
            dismiss=self.close_button,
        )
        layout.addLayout(footer)
        if self._review_only:
            self.setWindowTitle("Review measurement collection")
            self.title_label.setText("Saved measurement collection")
            self.intro_label.setText(
                "Review the images, exclusions and annotations recorded in this "
                "collection. Original images and output files are not reopened."
            )
            self.footer_hint.setText("Read-only review of the saved collection.")
            for widget in (
                self.output_label, self.output_combo, self.inspect_button,
                self.annotation_help, self.field_combo, self.annotation_edit,
                self.apply_button, self.exclusions_check, self.save_button,
                self.export_button, self.open_workflow_check,
            ):
                widget.hide()
        self._ready = True
        self._apply_theme()
        self._update_actions()
        QTimer.singleShot(0, self.inspect)

    def _apply_theme(self):
        colors = theme_colors(self.palette())
        tone = getattr(colors, getattr(self, "_tone", "info"))
        self.status_panel.setStyleSheet(
            "QFrame#MeasurementCollectionStatus {"
            f"background: {tone.surface.name()};"
            f"border-left: 3px solid {tone.accent.name()};"
            "} QFrame#MeasurementCollectionStatus QLabel {"
            f"background: transparent; color: {tone.foreground.name()};"
            "border: none; padding: 0px; }"
        )
        font = self._requested_font or self.font()
        apply_batch_table_style(self.table, self.palette(), font=font)
        family = font.family().replace("'", "\\'")
        size = (
            f"{font.pointSizeF():g}pt" if font.pointSizeF() > 0
            else f"{font.pixelSize()}px"
        )
        # Napari's broad QWidget stylesheet otherwise overrides widget fonts,
        # leaving labels tiny while the table correctly follows the user font.
        font_rule = f"font-family: '{family}'; font-size: {size};"
        self.table.horizontalHeader().setStyleSheet(
            font_rule + "font-weight: bold;"
        )
        for widget_type in (QLabel, QPushButton, QComboBox, QLineEdit, QCheckBox):
            for widget in self.findChildren(widget_type):
                weight = (
                    "font-weight: bold;" if widget in (
                        self.title_label, self.status_label,
                    ) else "font-weight: normal;"
                )
                widget.setStyleSheet(font_rule + weight)
        metric = QFontMetrics(font)
        self.table.verticalHeader().setDefaultSectionSize(metric.height() + 16)
        header_height = max(
            self.table.horizontalHeader().sizeHint().height(), metric.height() + 8
        )
        scrollbar_height = max(self.table.horizontalScrollBar().sizeHint().height(), 14)
        self.table.setMinimumHeight(
            140 + header_height + scrollbar_height + 2 * self.table.frameWidth()
        )
        scale = metric.height() / 19
        for column, width in enumerate((66, 142, 154, 56, 158, 115, 115, 105)):
            self.table.setColumnWidth(column, round(width * scale))

    def setFont(self, font):  # noqa: N802
        # An explicit user font must remain authoritative even when a parent
        # QWidget QSS resets QDialog.font() during polish. Inherited theme fonts
        # still flow normally when no explicit dialog font has been requested.
        self._requested_font = QFont(font)
        super().setFont(font)
        if self._ready:
            self._apply_theme()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if self._ready and event.type() in (QEvent.PaletteChange, QEvent.FontChange):
            self._apply_theme()

    def _status(self, title, detail="", tone="info"):
        self._tone = tone
        self.status_label.setText(title)
        self.progress_label.setText(_short_text(detail, 400))
        self.progress_label.setToolTip(detail)
        self.progress_label.setVisible(bool(detail))
        self._apply_theme()

    def _start(self, kind, request):
        self._generation += 1
        self._task_id = self._service.start(self._generation, kind, request)
        self._lifetime["task_id"] = self._task_id
        self.cancel_work_button.setText(
            {"save": "Cancel save", "export": "Cancel export"}.get(kind, "Cancel check")
        )
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self._update_actions()

    def inspect(self):
        if self._closed or self._task_id is not None:
            return False
        if self._review_only:
            self._status("Opening the saved collection…")
            self._start("review", _Request(
                self.manifest, expected_sha256=self._expected_sha256
            ))
            return True
        defaults = {item.key for item in self.model.items if item.status in _READY}
        if (
            (self.model.annotations or self.model.included != defaults)
            and not self._confirm_discard()
        ):
            return False
        self.preview = None
        self.model.set_preview(None)
        self._status(
            "Checking saved tables…", "Only table files and the batch record are read."
        )
        self._start("inspect", _Request(
            self.manifest, self.output_combo.currentData()
        ))
        return True

    def _confirm_discard(self):
        return QMessageBox.question(
            self, "Check files again?",
            "Checking files again resets your image selection and the annotations "
            "you added in this dialog. Saved metadata is unchanged. Continue?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) == QMessageBox.Yes

    def _output_changed(self):
        if self._ready and self._task_id is None:
            if not self.inspect() and self.preview is not None:
                self.output_combo.blockSignals(True)
                self.output_combo.setCurrentIndex(
                    self.output_combo.findData(self.preview.output.node_id)
                )
                self.output_combo.blockSignals(False)

    @Slot(object)
    def _progressed(self, value):
        task_id, generation, current, total, message = value
        if self._closed or task_id != self._task_id or generation != self._generation:
            return
        self.progress_label.setText(_short_text(message))
        self.progress_label.setToolTip(message)
        self.progress_label.setVisible(bool(message))
        self.progress_bar.setRange(0, max(0, total))
        self.progress_bar.setValue(current)

    @Slot(object)
    def _completed(self, reply):
        if (
            self._closed or reply.task_id != self._task_id
            or reply.generation != self._generation
        ):
            return
        self._task_id = None
        self._lifetime["task_id"] = None
        self.progress_bar.hide()
        if reply.cancelled:
            self._status("Cancelled", "No new files were saved.", "info")
        elif reply.error:
            title = {
                "save": "Could not save the collection",
                "export": "Could not export the results",
                "review": "Could not open the saved collection",
                "inspect": "Could not check the tables",
            }[reply.kind]
            self._status(title, reply.error, "error")
        elif reply.kind == "save":
            self._status("Collection saved", str(reply.value), "success")
            if reply.open_workflow:
                self.collectionSaved.emit(str(reply.value))
        elif reply.kind == "export":
            names = ", ".join(_basename(path) for path in reply.value.paths)
            notes = "\n".join(reply.value.notes)
            self._status(
                "Results exported", f"Saved {names}.", "success",
            )
            self.progress_label.setToolTip(
                "\n".join(str(path) for path in reply.value.paths)
                + (f"\n{notes}" if notes else "")
            )
        else:
            if reply.kind == "review":
                outputs, self.preview, preserved, included = reply.value
            else:
                outputs, self.preview, preserved = reply.value
                included = None
            self.output_combo.blockSignals(True)
            self.output_combo.clear()
            for output in outputs:
                self.output_combo.addItem(
                    f"{output.title} · {output.tag}" if output.tag else output.title,
                    output.node_id,
                )
            if self.preview is not None:
                self.output_combo.setCurrentIndex(
                    self.output_combo.findData(self.preview.output.node_id)
                )
            self.output_combo.blockSignals(False)
            self.model.set_preview(self.preview, preserved, included)
            if self.preview is None:
                self._status(
                    "No saved measurement output",
                    "This batch record has no table output to collect.", "warning",
                )
            elif self._review_only:
                self._status(
                    "Saved collection",
                    "The inventory below includes both included and excluded images. "
                    "Select an image to review its recorded file-check details.",
                    "info",
                )
            else:
                issues = sum(item.status not in _READY for item in self.preview.items)
                self._status(
                    "Review the collection" if issues else "Saved tables checked",
                    f"{issues:,} images need attention. Select a row for details."
                    if issues else "Choose images and add optional annotations below.",
                    "warning" if issues else "success",
                )
        self._update_actions()

    def cancel_work(self):
        if self._task_id is not None:
            self._service.cancel(self._task_id)
            self.cancel_work_button.setEnabled(False)
            self.progress_label.setText("Cancelling safely…")
            self.progress_label.show()

    def _selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            item = self.model.items[rows[0].row()]
            table_name = (
                _short_text(_basename(item.result_path), 100) or "No saved table"
            )
            self.item_detail.setText(
                f"{_short_text(item.batch_id, 70)} · {_short_text(item.message, 180)}\n"
                f"Table: {table_name}"
            )
            self.item_detail.setToolTip(
                f"{item.batch_id}\n{item.message}\nTable: {item.result_path}"
            )
        else:
            self.item_detail.setToolTip("")
            self.item_detail.setText(
                "Select an image for recorded file-check details."
                if self._review_only else
                "Select an image for file-check details. Double-click a blank "
                "annotation cell to edit it."
            )
        self._update_actions()

    def apply_annotation(self):
        if self._task_id is not None:
            return
        field = self.field_combo.currentData()
        value = self.annotation_edit.text().strip()
        selected = sorted({
            index.row() for index in self.table.selectionModel().selectedRows()
        })
        targets, protected, conflicts = [], 0, 0
        for row in selected:
            item = self.model.items[row]
            if item.status not in _READY:
                continue
            if field in self.model.preserved.get(item.key, {}):
                protected += 1
                continue
            old = self.model.annotations.get(item.key, {}).get(field, "")
            conflicts += bool(old and old != value)
            targets.append(row)
        if conflicts and QMessageBox.question(
            self, "Replace added annotations?",
            f"{conflicts:,} selected images already have a different {field} "
            "added in this dialog. Replace those values? Saved metadata is kept.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        self.model.apply_annotations(targets, field, value)
        self.item_detail.setText(
            f"{field.title()} applied to {len(targets):,} images."
            + (f" Existing metadata kept for {protected:,} images."
               if protected else "")
        )

    def _review_changed(self):
        self.exclusions_check.setChecked(False)
        included = len(self.model.included)
        excluded = len(self.model.items) - included
        rows = sum(
            item.row_count or 0 for item in self.model.items
            if item.key in self.model.included
        )
        empty = sum(
            item.status == "empty" and item.key in self.model.included
            for item in self.model.items
        )
        self.review_label.setText(
            f"{included:,} images included · {excluded:,} excluded · {rows:,} rows"
            + (f" · {empty:,} successful {'image' if empty == 1 else 'images'} "
               "with no rows" if empty else "")
        )
        self.exclusions_check.setVisible(bool(excluded) and not self._review_only)
        self._update_actions()

    def _update_actions(self):
        busy = self._task_id is not None
        self.model.editable = not busy and not self._review_only
        self.output_combo.setEnabled(not busy and self.output_combo.count() > 0)
        self.inspect_button.setEnabled(not busy)
        self.table.setEnabled(not busy)
        self.field_combo.setEnabled(not busy)
        self.annotation_edit.setEnabled(not busy)
        self.apply_button.setEnabled(
            not busy and bool(self.table.selectionModel().selectedRows())
        )
        self.exclusions_check.setEnabled(not busy)
        self.open_workflow_check.setEnabled(not busy)
        excluded = len(self.model.items) - len(self.model.included)
        self.save_button.setEnabled(
            not busy and not self._review_only and self.preview is not None
            and bool(self.model.included)
            and (not excluded or self.exclusions_check.isChecked())
        )
        self.export_button.setEnabled(self.save_button.isEnabled())
        self.cancel_work_button.setVisible(busy)
        self.cancel_work_button.setEnabled(busy)

    def save_collection(self):
        if not self.save_button.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save measurement collection",
            str(Path(self.manifest).parent / "measurement-results.vipp-results.json"),
            "VIPP measurement collection (*.vipp-results.json)",
            options=QFileDialog.DontConfirmOverwrite,
        )
        if not path:
            return
        if not path.endswith(".vipp-results.json"):
            path += ".vipp-results.json"
        from napari_vipp.core.measurement_collection import (
            capture_measurement_collection_destination_revision,
        )

        try:
            revision = capture_measurement_collection_destination_revision(path)
        except (OSError, ValueError) as exc:
            self._status("Choose a valid collection filename", str(exc), "warning")
            return
        if revision is not None and QMessageBox.question(
            self, "Replace measurement collection?",
            f"A file already exists at {path}. Replace it with this collection?",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        request = self._reviewed_request(
            path, open_workflow=self.open_workflow_check.isChecked(),
            destination_revision=revision,
        )
        self._status("Saving the VIPP collection…")
        self._start("save", request)

    def export_results(self):
        if not self.export_button.isEnabled():
            return
        choice = _choose_export_options(self)
        if choice is None:
            return
        filters = {
            "csv": "CSV table (*.csv)",
            "tsv": "TSV table (*.tsv)",
            "xlsx": "Excel workbook (*.xlsx)",
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "Export reviewed measurement results",
            str(Path(self.manifest).parent / f"measurement-results.{choice.format}"),
            filters[choice.format], options=QFileDialog.DontConfirmOverwrite,
        )
        if not path:
            return
        try:
            targets = _export_targets(path, choice)
            revisions = _export_revisions(targets)
        except (OSError, ValueError) as exc:
            self._status("Choose a valid export filename", str(exc), "warning")
            return
        existing = [
            target for target, revision in zip(targets, revisions, strict=True)
            if revision is not None
        ]
        if existing and QMessageBox.question(
            self, "Replace exported results?",
            "This export will write:\n"
            + "\n".join(str(target) for target in targets)
            + "\n\nThese existing files will be replaced:\n"
            + "\n".join(str(target) for target in existing),
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        request = self._reviewed_request(
            str(targets[0]), export_format=choice.format,
            include_summary=choice.include_summary, overwrite=bool(existing),
            export_destination_revisions=revisions,
        )
        self._status("Exporting reviewed results…")
        self._start("export", request)

    def _reviewed_request(self, path, **options):
        return _Request(
            self.manifest,
            preview=self.preview,
            annotations=tuple(
                (key, tuple(sorted(values.items())))
                for key, values in sorted(self.model.annotations.items())
            ),
            included_ids=tuple(sorted(self.model.included)),
            reviewed_exclusions=self.exclusions_check.isChecked(),
            path=path,
            **options,
        )

    def _detach(self):
        self._closed = True
        self._generation += 1
        if self._task_id is not None:
            self._service.cancel(self._task_id)

    def reject(self):
        self._detach()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        self._detach()
        super().closeEvent(event)


class MeasurementCollectionReviewDialog(MeasurementCollectionDialog):
    """Read-only inventory from a saved snapshot, with its bound hash verified."""

    def __init__(self, path: str | Path, parent=None, *, expected_sha256=""):
        super().__init__(
            path, parent, _review_only=True, _expected_sha256=expected_sha256
        )
