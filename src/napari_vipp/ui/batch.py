"""Retained collection batch workspace and UI-facing batch value objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal
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
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BatchConfig,
    BatchItemPlan,
    BatchRunResult,
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
    items: tuple[BatchItemPlan, ...]
    config: BatchConfig

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


BatchDialogValues = dict[str, object]
PreviewBatchAction = Callable[[BatchDialogValues, int], BatchPreviewResult]
PreviewBatchItemAction = Callable[[int], bool | None]
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
    preview_item: PreviewBatchItemAction | None = None


class CollectionBatchDialog(QDialog):
    """Front door for running a workflow over one or more local collections."""

    runRequested = Signal(object)
    previewInvalidated = Signal()

    def __init__(
        self,
        parent=None,
        source_nodes: list[dict] | None = None,
        *,
        actions: CollectionBatchActions | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch workspace")
        self.setMinimumSize(520, 360)
        self._actions = actions
        self._source_rows: list[dict[str, object]] = []
        self._loaded_config_path: Path | None = None
        self._demo: SyntheticBatchDemo | None = None
        self._preview_result: BatchPreviewResult | None = None
        self._preview_table_rows: dict[int, int] = {}
        self._run_control_enabled_states: dict[QWidget, bool] | None = None
        self._run_in_progress = False
        self._run_control_restore_timer = QTimer(self)
        self._run_control_restore_timer.setSingleShot(True)
        self._run_control_restore_timer.setInterval(50)
        self._run_control_restore_timer.timeout.connect(
            self._restore_deferred_run_controls
        )

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
        self.preview_button.setToolTip(
            "Plan every batch item and destination, then calculate the first "
            "item as a graph representative without saving batch outputs."
        )
        self.preview_button.clicked.connect(self._preview_batch)
        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        self.preview_status.setMinimumWidth(0)
        self.preview_status.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.preview_status.setStyleSheet("color: #94a3b8;")
        self.preview_table = QTableWidget(0, 5)
        self.preview_table.setHorizontalHeaderLabels(
            ["#", "Batch item", "Outputs", "Preflight", "Run status"]
        )
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.preview_table.setMinimumHeight(170)
        self.preview_table.setMaximumHeight(260)
        preview_header = self.preview_table.horizontalHeader()
        preview_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        preview_header.setSectionResizeMode(1, QHeaderView.Stretch)
        preview_header.setSectionResizeMode(2, QHeaderView.Stretch)
        preview_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        preview_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.preview_table.itemSelectionChanged.connect(self._sync_preview_item_button)
        self.preview_table.itemDoubleClicked.connect(
            self._preview_table_item_double_clicked
        )

        self.preview_item_button = QPushButton("Preview selected in graph")
        self.preview_item_button.setToolTip(
            "Load one representative planned item into the graph. This does not "
            "execute or save the full batch."
        )
        self.preview_item_button.clicked.connect(self._preview_selected_item)
        self.preview_item_button.setEnabled(False)
        self.graph_preview_status = QLabel(
            "Select a planned item to inspect one representative calculation "
            "in the graph."
        )
        self.graph_preview_status.setWordWrap(True)
        self.graph_preview_status.setMinimumWidth(0)
        self.graph_preview_status.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.graph_preview_status.setStyleSheet("color: #94a3b8;")

        self.source_group = QGroupBox("Batch sources")
        self.source_layout = QVBoxLayout(self.source_group)
        self._set_source_nodes(source_nodes)

        self.output_button = QPushButton("Folder...")
        self.output_button.clicked.connect(self._browse_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(self.output_button)

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
            "order and assigns each row a stable batch ID. Preview batch plans "
            "the complete collection and calculates the first row only as a "
            "representative graph view. Preview selected in graph changes that "
            "single representative; Run batch processes the full plan and saves "
            "only Batch Output nodes when present. Without Batch Output nodes, "
            "terminal graph outputs are saved as a compatibility fallback."
        )
        help_label.setWordWrap(True)
        help_label.setMinimumWidth(0)
        help_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        help_label.setStyleSheet("color: #94a3b8;")

        self.demo_guide_label = QLabel("")
        self.demo_guide_label.setWordWrap(True)
        self.demo_guide_label.setMinimumWidth(0)
        self.demo_guide_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.demo_guide_label.setTextFormat(Qt.RichText)
        self.demo_guide_label.setStyleSheet(
            "color: #dbeafe; padding: 9px; background: #172554; "
            "border: 1px solid #3b82f6; border-radius: 4px;"
        )
        self.demo_guide_label.hide()
        self.demo_path_edit = QLineEdit()
        self.demo_path_edit.setReadOnly(True)
        self.demo_path_edit.setMinimumWidth(0)
        self.demo_path_edit.setToolTip(
            "The writable working copy created for this batch demo."
        )
        self.demo_path_row = QWidget()
        demo_path_layout = QHBoxLayout(self.demo_path_row)
        demo_path_layout.setContentsMargins(0, 0, 0, 0)
        demo_path_layout.addWidget(QLabel("Working copy"))
        demo_path_layout.addWidget(self.demo_path_edit, 1)
        self.demo_path_row.hide()

        preview_row = QWidget()
        preview_layout = QHBoxLayout(preview_row)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.preview_status, 1)

        graph_preview_row = QWidget()
        graph_preview_layout = QHBoxLayout(graph_preview_row)
        graph_preview_layout.setContentsMargins(0, 0, 0, 0)
        graph_preview_layout.addWidget(self.preview_item_button)
        graph_preview_layout.addWidget(self.graph_preview_status, 1)

        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat("Not run")
        self.run_progress_bar.setTextVisible(True)
        self.run_progress_label = QLabel("No batch run is active.")
        self.run_progress_label.setWordWrap(True)
        self.run_progress_label.setMinimumWidth(0)
        self.run_progress_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.run_result_label = QLabel("")
        self.run_result_label.setWordWrap(True)
        self.run_result_label.setMinimumWidth(0)
        self.run_result_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        self.run_result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.run_result_label.setStyleSheet("color: #cbd5e1;")
        self.run_group = QGroupBox("Batch run")
        run_layout = QVBoxLayout(self.run_group)
        run_layout.addWidget(self.run_progress_label)
        run_layout.addWidget(self.run_progress_bar)
        run_layout.addWidget(self.run_result_label)
        self.run_group.hide()

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Close)
        self.run_button = self.button_box.button(QDialogButtonBox.Ok)
        self.run_button.setText("Run batch")
        self.run_button.clicked.connect(self._request_run)
        self.close_button = self.button_box.button(QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.reject)

        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(config_row)
        content_layout.addWidget(self.demo_guide_label)
        content_layout.addWidget(self.demo_path_row)
        content_layout.addWidget(self.source_group)
        content_layout.addLayout(form)
        content_layout.addWidget(help_label)
        content_layout.addWidget(preview_row)
        content_layout.addWidget(self.preview_table)
        content_layout.addWidget(graph_preview_row)
        content_layout.addWidget(self.run_group)

        self.content_scroll = QScrollArea()
        self.content_scroll.setObjectName("BatchWorkspaceScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.NoFrame)
        self.content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_scroll.setMinimumHeight(0)
        self.content_scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.content_scroll.setWidget(self.content_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.content_scroll, 1)
        layout.addWidget(self.button_box)

        self.output_edit.textChanged.connect(self._invalidate_preview_plan)
        self.format_combo.currentIndexChanged.connect(self._invalidate_preview_plan)
        self.existing_policy_combo.currentIndexChanged.connect(
            self._invalidate_preview_plan
        )
        self.script_checkbox.toggled.connect(self._invalidate_preview_plan)
        self.continue_checkbox.toggled.connect(self._invalidate_preview_plan)
        self.preview_status.setText(
            "Configure the collections, then click Preview batch to plan the "
            "complete run and calculate one graph representative without saving "
            "batch outputs."
        )

        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(
                min(840, max(self.minimumWidth(), available.width() - 80)),
                min(720, max(self.minimumHeight(), available.height() - 80)),
            )
        else:
            self.resize(840, 720)

    def set_demo_context(self, demo: SyntheticBatchDemo) -> None:
        """Present a generated bundle as a ready-to-run example workspace."""
        self._demo = demo
        self.run_button.setText("Run demo batch")
        self.demo_guide_label.setText(
            "<b>Ready-to-run batch demo</b><br>"
            "Two collection sources are paired by sorted position. The graph "
            "shows one representative paired field at a time; it is not the "
            "batch result. Preview batch plans all three items and three explicit "
            "outputs per item: an NPY image, TIFF labels, and a TSV table. Select "
            "or double-click a planned row to inspect that representative field "
            "through the graph. Explore source bindings, "
            "preview planning, Error/Skip/Overwrite replay policies, failure "
            "continuation, and portable config/runner controls below. Click "
            "<b>Run demo batch</b> to execute the workflow and validate its "
            "scientific outputs, manifests, archive, and per-item provenance "
            "against exact ground truth."
        )
        self.demo_guide_label.show()
        self.demo_path_edit.setText(str(demo.root))
        self.demo_path_edit.setCursorPosition(0)
        self.demo_path_row.show()

    def clear_demo_context(self) -> None:
        """Remove demo-only promises after setup or workflow customization."""
        self._demo = None
        run_button = getattr(self, "run_button", None)
        if run_button is not None:
            run_button.setText("Run batch")
        guide = getattr(self, "demo_guide_label", None)
        if guide is not None:
            guide.clear()
            guide.hide()
        path_edit = getattr(self, "demo_path_edit", None)
        if path_edit is not None:
            path_edit.clear()
        path_row = getattr(self, "demo_path_row", None)
        if path_row is not None:
            path_row.hide()

    def _set_source_nodes(self, source_nodes: list[dict]) -> None:
        self._invalidate_preview_plan()
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
        folder_edit.textChanged.connect(self._invalidate_preview_plan)
        pattern_edit.textChanged.connect(self._invalidate_preview_plan)
        title_label = QLabel(
            f"{title} ({node_id})"
            + ("  - collection" if binding_mode == "collection" else "")
        )
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title_label.setToolTip(title_label.text())
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
                "browse_button": browse_button,
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

    def _request_run(self) -> None:
        """Request execution without accepting or hiding this workspace."""
        if self._run_in_progress:
            return
        self.runRequested.emit(self.values())

    def _invalidate_preview_plan(self, *_args) -> None:
        """Discard a plan as soon as any setting that produced it changes."""
        if self._run_in_progress:
            return
        self.clear_demo_context()
        self._loaded_config_path = None
        self._preview_result = None
        self._preview_table_rows.clear()
        self.preview_table.setRowCount(0)
        self.preview_item_button.setEnabled(False)
        self.preview_status.setText(
            "Batch settings changed; click Preview batch to refresh the full plan."
        )
        self.graph_preview_status.setText(
            "Preview the batch before selecting a representative graph item."
        )
        self.previewInvalidated.emit()

    def invalidate_for_workflow_change(self) -> None:
        """Invalidate runnable planning while retaining the last graph sample."""
        if self._run_in_progress:
            return
        self._invalidate_preview_plan()
        self.preview_status.setText(
            "The scientific workflow changed; click Preview batch to refresh "
            "all destinations before running."
        )
        self.graph_preview_status.setText(
            "Representative navigation still uses the previous source pairing "
            "and calculates it through the edited graph."
        )

    def invalidate_for_source_change(self, message: str) -> None:
        """Require explicit Refresh when a reviewed source revision changed."""
        if self._run_in_progress:
            return
        self._invalidate_preview_plan()
        self.preview_status.setText(
            "A representative source changed after it was reviewed. Press "
            f"Refresh, then Preview batch again before running. {str(message)}"
        )
        self.graph_preview_status.setText(
            "The graph still uses its pinned earlier source revision until "
            "Refresh is pressed."
        )

    def begin_representative_source_refresh(
        self,
        position: int,
        total: int,
        batch_id: str,
    ) -> None:
        """Invalidate preflight while a representative revision is reloaded."""
        if self._run_in_progress:
            return
        self._invalidate_preview_plan()
        self.preview_status.setText(
            "Source snapshots are being refreshed. Wait for the representative "
            "calculation, then Preview batch again to review the runnable plan."
        )
        self.graph_preview_status.setText(
            f"Refreshing representative item {int(position) + 1} of "
            f"{int(total)} ({str(batch_id)}) through the graph..."
        )
        self.set_representative_pending(True)

    def show_plan_refresh_required(self, message: str) -> None:
        """Explain why a newly refreshed plan must be reviewed before running."""
        self.preview_status.setText(str(message))

    def set_representative_pending(self, pending: bool) -> None:
        """Keep full execution unavailable until graph preview is trustworthy."""
        if self._run_in_progress:
            return
        self.run_button.setEnabled(not bool(pending))

    def mark_plan_historical_after_run(self) -> None:
        """Retain run evidence but require fresh preflight before replay."""
        self._preview_result = None
        self.preview_item_button.setEnabled(False)
        self.preview_status.setText(
            "Historical preflight: the column above records the completed "
            "run's plan. "
            "Click Preview batch to refresh current inputs and destinations "
            "before running again."
        )

    def _sync_preview_item_button(self) -> None:
        action_available = bool(
            self._actions is not None and self._actions.preview_item is not None
        )
        has_selection = bool(self.preview_table.selectionModel().selectedRows())
        self.preview_item_button.setEnabled(
            action_available
            and self._preview_result is not None
            and has_selection
            and not self._run_in_progress
        )

    def select_preview_item(self, position: int) -> bool:
        """Select a zero-based full-plan position without previewing it again."""
        try:
            position = int(position)
        except (TypeError, ValueError):
            return False
        if self._preview_result is None or not (
            0 <= position < len(self._preview_result.items)
        ):
            return False
        table_row = self._preview_table_rows.get(position + 1)
        if table_row is None:
            self.preview_table.clearSelection()
            self._set_graph_preview_status(position)
            return True
        self.preview_table.selectRow(table_row)
        item = self.preview_table.item(table_row, 0)
        if item is not None:
            self.preview_table.scrollToItem(item)
        self._set_graph_preview_status(position)
        return True

    def _preview_selected_item(self) -> bool:
        """Load the selected full-plan position into the representative graph."""
        if self._run_in_progress:
            self.graph_preview_status.setText(
                "Representative graph preview is disabled while the full batch "
                "is running."
            )
            return False
        if self._actions is None or self._actions.preview_item is None:
            self.graph_preview_status.setText(
                "Representative graph preview is unavailable in this context."
            )
            return False
        selected = self.preview_table.selectionModel().selectedRows()
        if not selected:
            self.graph_preview_status.setText(
                "Select a planned batch row to preview it in the graph."
            )
            return False
        table_row = selected[0].row()
        item = self.preview_table.item(table_row, 0)
        if item is None:
            return False
        position = item.data(Qt.UserRole)
        try:
            position = int(position)
        except (TypeError, ValueError):
            return False
        if self._preview_result is None or not (
            0 <= position < len(self._preview_result.items)
        ):
            self.graph_preview_status.setText(
                "This plan is no longer current; preview the batch again."
            )
            return False
        try:
            outcome = self._actions.preview_item(position)
        except Exception as exc:
            self.graph_preview_status.setText(f"Graph preview failed: {exc}")
            return False
        if outcome is False:
            return False
        if outcome is None:
            self._set_graph_preview_status(position)
        return True

    def set_graph_preview_loading(self, position: int) -> None:
        """Show that the selected representative has not completed yet."""
        if self._preview_result is None or not (
            0 <= int(position) < len(self._preview_result.items)
        ):
            return
        item = self._preview_result.items[int(position)]
        self.graph_preview_status.setText(
            f"Loading representative item {int(position) + 1} of "
            f"{self._preview_result.total_items} ({item.batch_id}) through the "
            "graph..."
        )

    def set_graph_preview_ready(self, position: int) -> None:
        """Confirm that the matching representative calculation completed."""
        self._set_graph_preview_status(int(position))

    def show_graph_preview_error(self, position: int, message: str) -> None:
        """Retain the selected plan while surfacing a preview calculation error."""
        self.graph_preview_status.setText(
            f"Representative item {int(position) + 1} could not be shown: "
            f"{str(message).strip()}"
        )

    def _set_graph_preview_status(self, position: int) -> None:
        if self._preview_result is None:
            return
        plan_item = self._preview_result.items[position]
        self.graph_preview_status.setText(
            f"Graph preview: item {position + 1} of "
            f"{self._preview_result.total_items} ({plan_item.batch_id}). "
            "This is one representative calculation; the full batch has not "
            "been run or saved."
        )

    def _preview_table_item_double_clicked(self, item: QTableWidgetItem) -> None:
        self.preview_table.selectRow(item.row())
        self._preview_selected_item()

    def _preview_batch(self) -> bool:
        if self._actions is None:
            self.preview_status.setText("Preview is available from the VIPP widget.")
            return False
        try:
            result = self._actions.preview_batch(self.values(), 25)
        except Exception as exc:
            self.clear_demo_context()
            self._preview_result = None
            self._preview_table_rows.clear()
            self.preview_table.setRowCount(0)
            self.preview_status.setText(f"Preview failed: {exc}")
            self.graph_preview_status.setText(
                "Representative graph preview requires a valid batch plan."
            )
            self.preview_item_button.setEnabled(False)
            self.previewInvalidated.emit()
            return False
        self._preview_result = result
        self._reset_run_display()
        self._preview_table_rows = {
            item.batch_index: row_index for row_index, item in enumerate(result.rows)
        }
        self.preview_table.setRowCount(len(result))
        for row_index, item in enumerate(result):
            index_item = QTableWidgetItem(str(item.batch_index))
            index_item.setData(Qt.UserRole, item.batch_index - 1)
            self.preview_table.setItem(row_index, 0, index_item)
            source_text = "\n".join(
                f"{node_id}: {path.name}" for node_id, path in item.sources.items()
            )
            source_item = QTableWidgetItem(f"{item.batch_id}\n{source_text}")
            source_item.setToolTip(
                "\n".join(
                    f"{node_id}: {path}"
                    for node_id, path in item.sources.items()
                )
            )
            self.preview_table.setItem(row_index, 1, source_item)
            output_paths = [Path(path) for path in item.outputs]
            output_labels: list[str] = []
            for path in output_paths:
                try:
                    output_labels.append(str(path.relative_to(result.config.output_dir)))
                except ValueError:
                    output_labels.append(path.name)
            output_item = QTableWidgetItem("\n".join(output_labels))
            output_item.setToolTip("\n".join(str(path) for path in output_paths))
            self.preview_table.setItem(row_index, 2, output_item)
            status_text = "\n".join(item.output_statuses)
            self.preview_table.setItem(row_index, 3, QTableWidgetItem(status_text))
            self.preview_table.setItem(row_index, 4, QTableWidgetItem("Not run"))
        self.preview_table.resizeRowsToContents()
        total_items = result.total_items
        collision_count = result.collision_count
        explicit_outputs = result.explicit_outputs
        messages = [
            f"Showing {len(result)} of {total_items} planned batch item(s). "
            "The first row is calculated only as a graph representative; the "
            "full batch has not run and no batch outputs have been saved."
        ]
        if collision_count:
            messages.append(f"{collision_count} collision(s) need attention.")
        if not explicit_outputs:
            messages.append(
                "Compatibility fallback: terminal graph outputs will be saved; "
                "add Batch Output nodes to make the selection explicit."
            )
        if self._demo is not None:
            planned_outputs = sum(len(row.outputs) for row in result)
            messages.append(
                "Demo ready - click Run demo batch to process "
                f"{total_items} paired items, write {planned_outputs} outputs, "
                "and validate the results and provenance."
            )
        self.preview_status.setText(" ".join(messages))
        if result.rows:
            self.select_preview_item(0)
            if self._actions.preview_item is not None:
                self._preview_selected_item()
            else:
                self.graph_preview_status.setText(
                    "The full plan is ready. Representative graph preview is "
                    "unavailable in this context."
                )
        else:
            self.graph_preview_status.setText(
                "The batch plan contains no representative items to preview."
            )
        self._sync_preview_item_button()
        return True

    def begin_run(self, total: int) -> None:
        """Enter retained, determinate item-level batch progress mode."""
        total = max(int(total), 0)
        if self._run_control_enabled_states is None:
            self._run_control_enabled_states = {
                control: control.isEnabled() for control in self._run_controls()
            }
        for control in self._run_controls():
            control.setEnabled(False)
        self._run_in_progress = True
        self.run_group.show()
        self.run_progress_bar.setRange(0, max(total, 1))
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat(f"0 / {total}")
        self.run_progress_label.setText(
            f"Starting full batch run with {total} planned item(s)..."
        )
        self.run_result_label.clear()
        for table_row in range(self.preview_table.rowCount()):
            self._set_table_run_status(table_row, "Pending")

    def _reset_run_display(self) -> None:
        """Clear an earlier result when a fresh batch plan becomes current."""
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat("Not run")
        self.run_progress_label.setText("No batch run is active.")
        self.run_result_label.clear()
        self.run_group.hide()

    def update_run_progress(
        self,
        index: int,
        total: int,
        batch_id: str,
        status: str,
    ) -> None:
        """Apply one existing core item-status callback to the retained view."""
        index = max(int(index), 1)
        total = max(int(total), 0)
        normalized_status = str(status).strip().lower() or "running"
        completed = index - 1 if normalized_status == "running" else index
        completed = max(min(completed, total), 0)
        self.run_group.show()
        self.run_progress_bar.setRange(0, max(total, 1))
        self.run_progress_bar.setValue(completed)
        self.run_progress_bar.setFormat(f"{completed} / {total}")
        self.run_progress_label.setText(
            f"Item {index} of {total}: {batch_id} ({normalized_status})."
        )
        table_row = self._preview_table_rows.get(index)
        if table_row is not None:
            self._set_table_run_status(
                table_row,
                normalized_status.replace("_", " ").title(),
            )
            self.preview_table.selectRow(table_row)
            self.preview_table.scrollToItem(self.preview_table.item(table_row, 0))
        else:
            self.preview_table.clearSelection()

    def finish_run(
        self,
        result: BatchRunResult,
        validation_text: str = "",
        *,
        defer_control_restore: bool = False,
    ) -> None:
        """Retain the final manifest summary and reconcile every visible row."""
        manifest_items = tuple(result.manifest.items)
        for item in manifest_items:
            table_row = self._preview_table_rows.get(int(item.index))
            if table_row is None:
                continue
            status = getattr(item.status, "value", item.status)
            self._set_table_run_status(
                table_row,
                str(status).replace("_", " ").title(),
            )
        total = len(manifest_items)
        self.run_group.show()
        self.run_progress_bar.setRange(0, max(total, 1))
        self.run_progress_bar.setValue(total)
        self.run_progress_bar.setFormat(f"{total} / {total}")
        summary = result.summary
        self.run_progress_label.setText(
            f"Batch finished: {summary['completed']} completed, "
            f"{summary['partial']} partial, {summary['skipped']} skipped, "
            f"{summary['failed']} failed."
        )
        details = [
            f"{len(result.saved_paths)} output(s) saved.",
            f"Manifest: {result.manifest_path}",
        ]
        if validation_text:
            details.append(str(validation_text))
        self.run_result_label.setText("\n".join(details))
        self._finish_run_interaction(defer_control_restore)

    def show_run_error(
        self,
        message: str,
        *,
        defer_control_restore: bool = False,
    ) -> None:
        """Retain a terminal execution error and restore setup controls."""
        self.run_group.show()
        self.run_progress_label.setText("Batch failed before it could finish.")
        self.run_result_label.setText(str(message))
        self.run_progress_bar.setFormat("Failed")
        for table_row in range(self.preview_table.rowCount()):
            item = self.preview_table.item(table_row, 4)
            if item is None:
                continue
            prior_status = item.text().strip().lower()
            if prior_status == "running":
                item.setText("Failed")
            elif prior_status == "pending":
                item.setText("Not run")
        self._finish_run_interaction(defer_control_restore)

    def _finish_run_interaction(self, defer_control_restore: bool) -> None:
        """Consume queued run clicks before restoring setup controls."""
        if defer_control_restore:
            self._run_control_restore_timer.start()
            return
        self._run_control_restore_timer.stop()
        self._restore_deferred_run_controls()

    def _restore_deferred_run_controls(self) -> None:
        self._run_in_progress = False
        self._restore_run_controls()
        self._sync_preview_item_button()

    def _run_controls(self) -> tuple[QWidget, ...]:
        controls: list[QWidget] = [
            self.source_group,
            self.output_edit,
            self.output_button,
            self.format_combo,
            self.existing_policy_combo,
            self.workflow_checkbox,
            self.script_checkbox,
            self.continue_checkbox,
            self.load_config_button,
            self.save_config_button,
            self.demo_config_button,
            self.preview_button,
            self.preview_item_button,
            self.run_button,
        ]
        return tuple(dict.fromkeys(controls))

    def _restore_run_controls(self) -> None:
        states = self._run_control_enabled_states
        self._run_control_enabled_states = None
        if states is None:
            return
        for control, enabled in states.items():
            control.setEnabled(enabled)

    def _set_table_run_status(self, table_row: int, status: str) -> None:
        item = self.preview_table.item(table_row, 4)
        if item is None:
            item = QTableWidgetItem()
            self.preview_table.setItem(table_row, 4, item)
        item.setText(str(status))

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
        self.clear_demo_context()
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
    "PreviewBatchItemAction",
]
