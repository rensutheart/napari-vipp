"""Collection batch dialog and UI-facing batch value objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BatchConfig,
    ExistingFilePolicy,
)
from napari_vipp.core.batch_demo import SyntheticBatchDemo


@dataclass(frozen=True)
class BatchSourceBinding:
    node_id: str
    title: str
    input_dir: Path | None
    pattern: str


@dataclass(frozen=True)
class BatchPreviewRow:
    batch_index: int
    batch_id: str
    sources: dict[str, Path]
    outputs: list[Path]
    output_statuses: tuple[str, ...] = ()
    explicit_outputs: bool = True


@dataclass(frozen=True)
class BatchPreviewResult:
    rows: tuple[BatchPreviewRow, ...]
    total_items: int
    collision_count: int
    explicit_outputs: bool

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


BatchDialogValues = dict[str, object]
PreviewBatchAction = Callable[[BatchDialogValues, int], BatchPreviewResult]
ChooseBatchDemoAction = Callable[[QWidget], SyntheticBatchDemo | None]
BatchSourceRowsAction = Callable[[], list[dict[str, str]]]
LoadBatchConfigAction = Callable[[str | Path], BatchConfig]
SaveBatchConfigAction = Callable[
    [str | Path, BatchDialogValues],
    tuple[Path, ...],
]


@dataclass(frozen=True)
class CollectionBatchActions:
    """Application actions required by :class:`CollectionBatchDialog`."""

    preview_batch: PreviewBatchAction
    choose_demo: ChooseBatchDemoAction
    source_rows: BatchSourceRowsAction
    load_config: LoadBatchConfigAction
    save_config: SaveBatchConfigAction


class CollectionBatchDialog(QDialog):
    """Front door for running a workflow over one or more local collections."""

    def __init__(
        self,
        parent=None,
        source_nodes: list[dict] | None = None,
        *,
        actions: CollectionBatchActions | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Run collection batch")
        self.setMinimumWidth(880)
        self._actions = actions
        self._source_rows: list[dict[str, object]] = []
        self._loaded_config_path: Path | None = None
        self._demo: SyntheticBatchDemo | None = None

        source_nodes = source_nodes or [
            {
                "node_id": "input",
                "title": "Image Source",
                "binding_mode": "collection",
            }
        ]

        self.output_edit = QLineEdit()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ome-tiff", "imagej-tiff", "tiff", "npy"])
        self.existing_policy_combo = QComboBox()
        self.existing_policy_combo.addItem("Error", ExistingFilePolicy.ERROR.value)
        self.existing_policy_combo.addItem("Skip", ExistingFilePolicy.SKIP.value)
        self.existing_policy_combo.addItem(
            "Overwrite", ExistingFilePolicy.OVERWRITE.value
        )
        self.workflow_checkbox = QCheckBox("Save workflow JSON")
        self.workflow_checkbox.setChecked(True)
        self.workflow_checkbox.setEnabled(False)
        self.workflow_checkbox.setToolTip(
            "The workflow companion is required for reproducible batch configs."
        )
        self.script_checkbox = QCheckBox("Save batch runner Python script")
        self.script_checkbox.setChecked(True)
        self.continue_checkbox = QCheckBox("Continue after item failures")
        self.continue_checkbox.setChecked(True)
        self.preview_button = QPushButton("Preview batch")
        self.preview_button.clicked.connect(self._preview_batch)
        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        self.preview_status.setStyleSheet("color: #94a3b8;")
        self.preview_table = QTableWidget(0, 4)
        self.preview_table.setHorizontalHeaderLabels(
            ["#", "Batch item", "Outputs", "Preflight"]
        )
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setMinimumHeight(170)
        self.preview_table.setMaximumHeight(260)

        source_group = QGroupBox("Batch sources")
        self.source_layout = QVBoxLayout(source_group)
        self._set_source_nodes(source_nodes)

        output_button = QPushButton("Folder...")
        output_button.clicked.connect(self._browse_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(output_button)

        form = QFormLayout()
        form.addRow("Output folder", output_row)
        form.addRow("Default image format", self.format_combo)
        form.addRow("Existing files", self.existing_policy_combo)
        form.addRow("", self.workflow_checkbox)
        form.addRow("", self.script_checkbox)
        form.addRow("", self.continue_checkbox)

        self.load_config_button = QPushButton("Load config...")
        self.load_config_button.clicked.connect(self._load_config)
        self.save_config_button = QPushButton("Save config...")
        self.save_config_button.clicked.connect(self._save_config)
        self.demo_config_button = QPushButton("Open batch demo...")
        self.demo_config_button.setToolTip(
            "Open a ready-to-run deterministic batch workspace with paired "
            "inputs, explicit outputs, provenance, and ground-truth validation."
        )
        self.demo_config_button.clicked.connect(self._create_demo)
        actions_available = self._actions is not None
        self.preview_button.setEnabled(actions_available)
        self.load_config_button.setEnabled(actions_available)
        self.save_config_button.setEnabled(actions_available)
        self.demo_config_button.setEnabled(actions_available)
        config_row = QWidget()
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.load_config_button)
        config_layout.addWidget(self.save_config_button)
        config_layout.addWidget(self.demo_config_button)
        config_layout.addStretch(1)

        help_label = QLabel(
            "Bind each Image Source that should change per batch item to a "
            "folder and file pattern. VIPP zips bound sources by sorted file "
            "order, assigns each row a stable batch ID, and saves only Batch "
            "Output nodes when present. Without Batch Output nodes, terminal "
            "graph outputs are saved as a compatibility fallback."
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #94a3b8;")

        self.demo_guide_label = QLabel("")
        self.demo_guide_label.setWordWrap(True)
        self.demo_guide_label.setTextFormat(Qt.RichText)
        self.demo_guide_label.setStyleSheet(
            "color: #dbeafe; padding: 9px; background: #172554; "
            "border: 1px solid #3b82f6; border-radius: 4px;"
        )
        self.demo_guide_label.hide()

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.preview_status, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.run_button = buttons.button(QDialogButtonBox.Ok)
        self.run_button.setText("Run batch")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(config_row)
        layout.addWidget(self.demo_guide_label)
        layout.addWidget(source_group)
        layout.addLayout(form)
        layout.addWidget(help_label)
        layout.addWidget(preview_row)
        layout.addWidget(self.preview_table)
        layout.addWidget(buttons)

    def set_demo_context(self, demo: SyntheticBatchDemo) -> None:
        """Present a generated bundle as a ready-to-run example workspace."""
        self._demo = demo
        self.run_button.setText("Run demo batch")
        self.demo_guide_label.setText(
            "<b>Ready-to-run batch demo</b><br>"
            "Two collection sources are paired by sorted position. The graph "
            "automatically shows the first paired field while the batch preview "
            "plans all three items and three explicit outputs per item: an NPY "
            "image, TIFF labels, and a TSV table. Explore source bindings, "
            "preview planning, Error/Skip/Overwrite replay policies, failure "
            "continuation, and portable config/runner controls below. Click "
            "<b>Run demo batch</b> to execute the workflow and validate its "
            "scientific outputs, manifests, archive, and per-item provenance "
            "against exact ground truth.<br>"
            f"Working copy: <code>{escape(str(demo.root))}</code>"
        )
        self.demo_guide_label.show()

    def _set_source_nodes(self, source_nodes: list[dict]) -> None:
        self.preview_table.setRowCount(0)
        self.preview_status.clear()
        for row in self._source_rows:
            widget = row["widget"]
            self.source_layout.removeWidget(widget)
            widget.deleteLater()
        self._source_rows.clear()
        for index, source in enumerate(source_nodes):
            row = self._make_source_row(
                str(source.get("node_id", f"source_{index + 1}")),
                str(source.get("title", f"Image Source {index + 1}")),
                str(source.get("binding_mode", "")),
                index=index,
            )
            self.source_layout.addWidget(row)
        if self._source_rows:
            self.input_edit = self._source_rows[0]["folder"]
            self.pattern_edit = self._source_rows[0]["pattern"]
        else:
            self.input_edit = QLineEdit()
            self.pattern_edit = QLineEdit("*.tif;*.tiff;*.ome.tif;*.ome.tiff")

    def _make_source_row(
        self,
        node_id: str,
        title: str,
        binding_mode: str,
        *,
        index: int,
    ) -> QWidget:
        folder_edit = QLineEdit()
        pattern_edit = QLineEdit("*.tif;*.tiff;*.ome.tif;*.ome.tiff")
        browse_button = QPushButton("Folder...")
        browse_button.clicked.connect(
            lambda _checked=False, edit=folder_edit: self._browse_source_input(edit)
        )
        title_label = QLabel(
            f"{title} ({node_id})"
            + ("  - collection" if binding_mode == "collection" else "")
        )
        title_label.setStyleSheet("font-weight: 650;")

        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(QLabel("Folder"))
        folder_layout.addWidget(folder_edit, 1)
        folder_layout.addWidget(browse_button)

        pattern_row = QWidget()
        pattern_layout = QHBoxLayout(pattern_row)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.addWidget(QLabel("Pattern"))
        pattern_layout.addWidget(pattern_edit, 1)

        row = QFrame()
        row.setFrameShape(QFrame.StyledPanel)
        row.setStyleSheet(
            "QFrame { border: 1px solid #334155; border-radius: 4px; }"
            "QLabel { border: none; }"
            "QLineEdit { border: 1px solid #475569; }"
        )
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.addWidget(title_label)
        row_layout.addWidget(folder_row)
        row_layout.addWidget(pattern_row)
        self._source_rows.append(
            {
                "node_id": node_id,
                "title": title,
                "folder": folder_edit,
                "pattern": pattern_edit,
                "index": index,
                "widget": row,
                "title_label": title_label,
                "binding_mode": binding_mode,
            }
        )
        return row

    def values(self) -> dict[str, object]:
        bindings = []
        for row in self._source_rows:
            bindings.append(
                {
                    "node_id": row["node_id"],
                    "title": row["title"],
                    "input_dir": row["folder"].text(),
                    "pattern": row["pattern"].text(),
                }
            )
        return {
            "input_dir": self.input_edit.text(),
            "output_dir": self.output_edit.text(),
            "pattern": self.pattern_edit.text(),
            "source_bindings": bindings,
            "image_format": self.format_combo.currentText(),
            "existing_file_policy": str(self.existing_policy_combo.currentData()),
            "save_workflow_snapshot": self.workflow_checkbox.isChecked(),
            "save_python_script": self.script_checkbox.isChecked(),
            "continue_on_error": self.continue_checkbox.isChecked(),
        }

    def _preview_batch(self) -> bool:
        if self._actions is None:
            self.preview_status.setText("Preview is available from the VIPP widget.")
            return False
        try:
            rows = self._actions.preview_batch(self.values(), 25)
        except Exception as exc:
            self.preview_table.setRowCount(0)
            self.preview_status.setText(f"Preview failed: {exc}")
            return False
        self.preview_table.setRowCount(len(rows))
        for row_index, item in enumerate(rows):
            self.preview_table.setItem(
                row_index,
                0,
                QTableWidgetItem(str(item.batch_index)),
            )
            source_text = "\n".join(
                f"{node_id}: {path.name}" for node_id, path in item.sources.items()
            )
            self.preview_table.setItem(
                row_index,
                1,
                QTableWidgetItem(f"{item.batch_id}\n{source_text}"),
            )
            output_text = "\n".join(str(path) for path in item.outputs)
            self.preview_table.setItem(row_index, 2, QTableWidgetItem(output_text))
            status_text = "\n".join(item.output_statuses)
            self.preview_table.setItem(row_index, 3, QTableWidgetItem(status_text))
        self.preview_table.resizeColumnsToContents()
        self.preview_table.resizeRowsToContents()
        total_items = getattr(rows, "total_items", len(rows))
        collision_count = getattr(
            rows,
            "collision_count",
            sum(
                status
                in {
                    "exists; collision",
                    "duplicate planned destination",
                    "destination overlaps an input",
                }
                for row in rows
                for status in row.output_statuses
            ),
        )
        explicit_outputs = getattr(
            rows,
            "explicit_outputs",
            bool(not rows or rows[0].explicit_outputs),
        )
        messages = [
            f"Showing {len(rows)} of {total_items} planned batch item(s)."
        ]
        if collision_count:
            messages.append(f"{collision_count} collision(s) need attention.")
        if not explicit_outputs:
            messages.append(
                "Compatibility fallback: terminal graph outputs will be saved; "
                "add Batch Output nodes to make the selection explicit."
            )
        if self._demo is not None:
            planned_outputs = sum(len(row.outputs) for row in rows)
            messages.append(
                "Demo ready - click Run demo batch to process "
                f"{total_items} paired items, write {planned_outputs} outputs, "
                "and validate the results and provenance."
            )
        self.preview_status.setText(" ".join(messages))
        return True

    def _create_demo(self) -> None:
        if self._actions is None:
            self.preview_status.setText(
                "Opening the synthetic demo is available from the VIPP widget."
            )
            return
        try:
            demo = self._actions.choose_demo(self)
            if demo is None:
                return
            self._set_source_nodes(self._actions.source_rows())
            config = self._actions.load_config(demo.config_path)
            self._apply_config(config)
            self._loaded_config_path = demo.config_path
            self.set_demo_context(demo)
            if not self._preview_batch():
                return
        except Exception as exc:
            self.preview_status.setText(f"Could not open batch demo: {exc}")
            return

    def _load_config(self) -> None:
        if self._actions is None:
            self.preview_status.setText(
                "Loading a batch config is available from the VIPP widget."
            )
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load batch configuration",
            str(self._loaded_config_path or BATCH_CONFIG_FILENAME),
            "VIPP batch config (*.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            config = self._actions.load_config(path)
            self._apply_config(config)
        except Exception as exc:
            self.preview_status.setText(f"Could not load batch config: {exc}")
            return
        self._loaded_config_path = Path(path)
        self.preview_status.setText(f"Loaded {Path(path).name}.")

    def _save_config(self) -> None:
        if self._actions is None:
            self.preview_status.setText(
                "Saving a batch config is available from the VIPP widget."
            )
            return
        default_dir = Path(self.output_edit.text()).expanduser()
        default_path = (
            default_dir / BATCH_CONFIG_FILENAME
            if str(default_dir).strip()
            else Path(BATCH_CONFIG_FILENAME)
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save batch configuration",
            str(default_path),
            "VIPP batch config (*.json);;JSON files (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            saved = self._actions.save_config(path, self.values())
        except Exception as exc:
            self.preview_status.setText(f"Could not save batch config: {exc}")
            return
        self._loaded_config_path = Path(path)
        names = ", ".join(item.name for item in saved)
        self.preview_status.setText(f"Saved {names}.")

    def _apply_config(self, config: BatchConfig) -> None:
        rows = {str(row["node_id"]): row for row in self._source_rows}
        missing = [
            source.node_id for source in config.sources if source.node_id not in rows
        ]
        if missing:
            raise ValueError(
                "Config references source nodes not present in this workflow: "
                + ", ".join(missing)
                + "."
            )
        configured_ids = [source.node_id for source in config.sources]
        ordered_rows = [rows[node_id] for node_id in configured_ids]
        ordered_rows.extend(
            row
            for row in self._source_rows
            if str(row["node_id"]) not in configured_ids
        )
        self._source_rows = ordered_rows
        for row in self._source_rows:
            row["folder"].clear()
            self.source_layout.removeWidget(row["widget"])
            self.source_layout.addWidget(row["widget"])
        for source in config.sources:
            row = rows[source.node_id]
            row["title"] = source.title
            suffix = (
                "  - collection" if row["binding_mode"] == "collection" else ""
            )
            row["title_label"].setText(
                f"{source.title} ({source.node_id}){suffix}"
            )
            row["folder"].setText(str(config.resolve_path(source.input_dir)))
            row["pattern"].setText(source.pattern)
        if self._source_rows:
            self.input_edit = self._source_rows[0]["folder"]
            self.pattern_edit = self._source_rows[0]["pattern"]
        self.output_edit.setText(str(config.resolve_path(config.output_dir)))
        format_index = self.format_combo.findText(config.default_image_format)
        if format_index >= 0:
            self.format_combo.setCurrentIndex(format_index)
        policy_index = self.existing_policy_combo.findData(
            config.existing_file_policy.value
        )
        if policy_index >= 0:
            self.existing_policy_combo.setCurrentIndex(policy_index)
        self.workflow_checkbox.setChecked(True)
        self.script_checkbox.setChecked(config.save_python_script)
        self.continue_checkbox.setChecked(config.continue_on_error)

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch input folder",
            self.input_edit.text(),
        )
        if path:
            self.input_edit.setText(path)

    def _browse_source_input(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch input folder",
            edit.text(),
        )
        if path:
            edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch output folder",
            self.output_edit.text(),
        )
        if path:
            self.output_edit.setText(path)

__all__ = [
    "BatchDialogValues",
    "BatchPreviewResult",
    "BatchPreviewRow",
    "BatchSourceBinding",
    "CollectionBatchActions",
    "CollectionBatchDialog",
]
