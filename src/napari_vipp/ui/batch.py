"""Retained collection batch workspace and UI-facing batch value objects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from qtpy.QtCore import QEvent, Qt, QTimer, Signal
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
    DEFAULT_BATCH_SOURCE_PATTERN,
    BatchConfig,
    BatchItemPlan,
    BatchRunResult,
    BatchScientificPreflightError,
    ExistingFilePolicy,
)
from napari_vipp.core.batch_demo import SyntheticBatchDemo
from napari_vipp.core.batch_parameters import BatchSourceParameterOverrides
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.ui import recent_paths
from napari_vipp.ui.axis_interpretation import AxisInterpretationControl
from napari_vipp.ui.batch_overrides import (
    BatchOverrideParameterSpec,
    BatchOverrideSourceItem,
    BatchParameterOverrideEditor,
)


@dataclass(frozen=True)
class BatchSourceBinding:
    node_id: str
    title: str
    input_dir: Path | None
    pattern: str
    axis_declaration: str = ""


@dataclass(frozen=True)
class BatchPreviewRow:
    batch_index: int
    batch_id: str
    sources: dict[str, Path]
    outputs: list[Path]
    output_statuses: tuple[str, ...] = ()
    explicit_outputs: bool = True
    source_labels: dict[str, str] = field(default_factory=dict)


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
    cancelRequested = Signal()
    previewInvalidated = Signal()
    parameterOverridesChanged = Signal(object)

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
        self._loaded_compute_request: ComputeRequest | None = None
        self._compute_toolbar_fingerprint_at_load = ""
        self._demo: SyntheticBatchDemo | None = None
        self._preview_result: BatchPreviewResult | None = None
        self._preview_table_rows: dict[int, int] = {}
        self._pending_parameter_overrides: tuple[
            BatchSourceParameterOverrides, ...
        ] = ()
        self._run_control_enabled_states: dict[QWidget, bool] | None = None
        self._run_in_progress = False
        self._representative_pending = False
        self._activity_run_total = 0
        self._activity_run_index = 0
        self._activity_run_completed = 0
        self._output_path_is_suggested = True
        self._setting_suggested_output = False
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
        self.output_edit.setMinimumWidth(0)
        self.output_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_edit.installEventFilter(self)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ome-tiff", "imagej-tiff", "tiff", "npy"])
        self.existing_policy_combo = QComboBox()
        self.existing_policy_combo.addItem(
            "Ask before overwrite (recommended)",
            ExistingFilePolicy.ERROR.value,
        )
        self.existing_policy_combo.addItem(
            "Skip existing",
            ExistingFilePolicy.SKIP.value,
        )
        self.existing_policy_combo.addItem(
            "Overwrite without asking",
            ExistingFilePolicy.OVERWRITE.value,
        )
        self.existing_policy_combo.setToolTip(
            "Ask before overwrite confirms the exact existing outputs for each "
            "interactive run. Skip preserves them. Overwrite without asking is "
            "intended for a deliberately persistent replacement policy."
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
            "Optionally inspect every planned batch item and destination, then "
            "calculate the first item as a graph representative without saving "
            "batch outputs. Run batch performs its own fresh preflight."
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

        # Keep one concise Batch-only status surface in the fixed toolbar.  The
        # main VIPP progress indicator remains the source of truth for live graph
        # calculation; the detailed bars farther down retain item/node evidence
        # for a full batch run.
        self.batch_activity_strip = QFrame()
        self.batch_activity_strip.setObjectName("BatchWorkspaceActivityStrip")
        self.batch_activity_strip.setFrameShape(QFrame.NoFrame)
        self.batch_activity_strip.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )
        activity_layout = QHBoxLayout(self.batch_activity_strip)
        activity_layout.setContentsMargins(8, 0, 0, 0)
        activity_layout.setSpacing(8)
        self.batch_activity_status = QLabel(
            "Not checked · Preview or run to inspect batch items."
        )
        self.batch_activity_status.setObjectName("BatchWorkspaceActivityStatus")
        self.batch_activity_status.setMinimumWidth(96)
        self.batch_activity_status.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Fixed,
        )
        self.batch_activity_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.batch_activity_status.setAccessibleName("Batch workspace status")
        activity_layout.addWidget(self.batch_activity_status, 1)
        self.source_detection_progress = QProgressBar()
        self.source_detection_progress.setObjectName("BatchWorkspaceActivityProgress")
        self.batch_activity_progress = self.source_detection_progress
        self.source_detection_progress.setFixedWidth(128)
        self.source_detection_progress.setRange(0, 1)
        self.source_detection_progress.setValue(0)
        self.source_detection_progress.setFormat("Not active")
        self.source_detection_progress.setTextVisible(True)
        self.source_detection_progress.setAccessibleName(
            "Batch workspace activity progress"
        )
        self.source_detection_progress.setAccessibleDescription(
            "Summarizes Batch workspace planning and item progress. Main VIPP "
            "reports representative graph calculation; detailed batch run bars "
            "appear lower in this window."
        )
        self._batch_activity_tooltip = (
            "This toolbar status summarizes Batch workspace activity. The main "
            "VIPP progress bar reports representative scientific graph "
            "calculation. During a full batch, detailed item and node progress "
            "is retained in the Batch run section below."
        )
        self.batch_activity_strip.setToolTip(self._batch_activity_tooltip)
        self.batch_activity_status.setToolTip(self._batch_activity_tooltip)
        self.source_detection_progress.setToolTip(self._batch_activity_tooltip)
        self.source_detection_progress.hide()
        activity_layout.addWidget(self.source_detection_progress, 0)
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

        self.parameter_override_group = QGroupBox("Per-sample parameters (optional)")
        parameter_override_layout = QVBoxLayout(self.parameter_override_group)
        self.parameter_override_editor = BatchParameterOverrideEditor()
        parameter_override_layout.addWidget(self.parameter_override_editor)
        self.parameter_override_group.hide()
        self.parameter_override_editor.overridesChanged.connect(
            self._parameter_overrides_changed
        )
        self.parameter_override_editor.validityChanged.connect(
            self._sync_parameter_override_validity
        )

        self.output_button = QPushButton("Folder...")
        self.output_button.clicked.connect(self._browse_output)
        output_row = QWidget()
        output_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(self.output_button)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Output folder", output_row)
        form.addRow("Default image format", self.format_combo)
        form.addRow("Existing files", self.existing_policy_combo)
        form.addRow("", self.workflow_checkbox)
        form.addRow("", self.script_checkbox)
        form.addRow("", self.continue_checkbox)

        self.load_config_button = QPushButton("Load...")
        self.load_config_button.setToolTip("Load a saved Batch workspace config.")
        self.load_config_button.clicked.connect(self._load_config)
        self.save_config_button = QPushButton("Save...")
        self.save_config_button.setToolTip("Save this Batch workspace config.")
        self.save_config_button.clicked.connect(self._save_config)
        self.demo_config_button = QPushButton("Demo...")
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
        self.config_row = QWidget()
        self.config_row.setObjectName("BatchWorkspaceToolbar")
        self.config_row.setAccessibleName("Batch workspace toolbar")
        self.config_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        config_layout = QHBoxLayout(self.config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.load_config_button)
        config_layout.addWidget(self.save_config_button)
        config_layout.addWidget(self.demo_config_button)
        config_layout.addWidget(self.batch_activity_strip, 1)

        help_label = QLabel(
            "Choose the source and output folders. Preview checks a sample "
            "before anything is saved."
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
        self.operation_progress_bar = QProgressBar()
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.operation_progress_bar.setFormat("Not run")
        self.operation_progress_bar.setTextVisible(True)
        self.operation_progress_label = QLabel("No node operation is active.")
        self.operation_progress_label.setWordWrap(True)
        self.operation_progress_label.setMinimumWidth(0)
        self.operation_progress_label.setSizePolicy(
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
        run_layout.addWidget(self.operation_progress_label)
        run_layout.addWidget(self.operation_progress_bar)
        run_layout.addWidget(self.run_result_label)
        self.run_group.hide()

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Close)
        self.run_button = self.button_box.button(QDialogButtonBox.Ok)
        self.run_button.setText("Run batch")
        self.run_button.clicked.connect(self._request_run)
        self.cancel_run_button = self.button_box.addButton(
            "Cancel run",
            QDialogButtonBox.ActionRole,
        )
        self.cancel_run_button.setToolTip(
            "Request cooperative cancellation. The active CPU/GPU operation "
            "stops at its next safe checkpoint; completed outputs and the final "
            "manifest are retained."
        )
        self.cancel_run_button.clicked.connect(self._request_cancel)
        self.cancel_run_button.hide()
        self.close_button = self.button_box.button(QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.reject)

        self.content_widget = QWidget()
        self.content_widget.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
        )
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self.demo_guide_label)
        content_layout.addWidget(self.demo_path_row)
        content_layout.addWidget(self.source_group)
        content_layout.addWidget(self.parameter_override_group)
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
        layout.addWidget(self.config_row)
        layout.addWidget(self.content_scroll, 1)
        layout.addWidget(self.button_box)

        self.output_edit.textChanged.connect(self._output_path_changed)
        self.format_combo.currentIndexChanged.connect(self._invalidate_preview_plan)
        self.existing_policy_combo.currentIndexChanged.connect(
            self._invalidate_preview_plan
        )
        self.script_checkbox.toggled.connect(self._invalidate_preview_plan)
        self.continue_checkbox.toggled.connect(self._invalidate_preview_plan)
        self.preview_status.setText(
            "Ready. Preview checks one sample; Run batch checks again before saving."
        )
        self.show_workspace_activity(
            "Not checked · Preview or run to inspect batch items.",
            state="info",
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

    def eventFilter(self, watched, event):
        if watched is self.output_edit:
            deliberate_focus = event.type() == QEvent.FocusIn and event.reason() in {
                Qt.MouseFocusReason,
                Qt.TabFocusReason,
                Qt.BacktabFocusReason,
                Qt.ShortcutFocusReason,
            }
            if event.type() == QEvent.MouseButtonPress or deliberate_focus:
                self._acknowledge_output_path()
        return super().eventFilter(watched, event)

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
                (
                    None
                    if source.get("axis_declaration") is None
                    else str(source.get("axis_declaration", ""))
                ),
                index=index,
            )
            self.source_layout.addWidget(row)
        if self._source_rows:
            self.input_edit = self._source_rows[0]["folder"]
            self.pattern_edit = self._source_rows[0]["pattern"]
        else:
            self.input_edit = QLineEdit()
            self.pattern_edit = QLineEdit(DEFAULT_BATCH_SOURCE_PATTERN)
        self._refresh_suggested_output_path()

    def _make_source_row(
        self,
        node_id: str,
        title: str,
        binding_mode: str,
        axis_declaration: str | None = None,
        *,
        index: int,
    ) -> QWidget:
        folder_edit = QLineEdit()
        pattern_edit = QLineEdit(DEFAULT_BATCH_SOURCE_PATTERN)
        pattern_edit.setToolTip(
            "* discovers all supported image files and top-level OME-Zarr "
            "stores in this folder. Use semicolon-separated globs only when "
            "you want to narrow the collection."
        )
        axis_declaration_edit = AxisInterpretationControl()
        if axis_declaration is not None:
            axis_declaration_edit.setText(axis_declaration)
        browse_button = QPushButton("Folder...")
        browse_button.clicked.connect(
            lambda _checked=False, edit=folder_edit: self._browse_source_input(edit)
        )
        folder_edit.textChanged.connect(
            lambda text, edit=folder_edit: self._source_folder_changed(edit, text)
        )
        pattern_edit.textChanged.connect(
            lambda _text, control=axis_declaration_edit: self._source_pattern_changed(
                control
            )
        )
        axis_declaration_edit.textChanged.connect(self._invalidate_preview_plan)
        title_label = QLabel(
            title + ("  - collection" if binding_mode == "collection" else "")
        )
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title_label.setToolTip(f"Workflow source: {title} ({node_id})")
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
        pattern_label = QLabel("Pattern")
        pattern_label.setToolTip(pattern_edit.toolTip())
        pattern_layout.addWidget(pattern_label)
        pattern_layout.addWidget(pattern_edit, 1)

        declaration_row = QWidget()
        declaration_layout = QHBoxLayout(declaration_row)
        declaration_layout.setContentsMargins(0, 0, 0, 0)
        declaration_label = QLabel("Image stack")
        declaration_label.setToolTip(
            "VIPP normally trusts the file and visibly suggests Z stack only "
            "when this workflow proves that it needs one."
        )
        declaration_layout.addWidget(declaration_label)
        declaration_layout.addWidget(axis_declaration_edit, 1)

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
        row_layout.addWidget(declaration_row)
        self._source_rows.append(
            {
                "node_id": node_id,
                "title": title,
                "folder": folder_edit,
                "pattern": pattern_edit,
                "axis_declaration": axis_declaration_edit,
                "browse_button": browse_button,
                "index": index,
                "widget": row,
                "title_label": title_label,
                "binding_mode": binding_mode,
            }
        )
        return row

    def _source_folder_changed(self, edit: QLineEdit, _text: str) -> None:
        matching_row = next(
            (row for row in self._source_rows if edit is row["folder"]),
            None,
        )
        if self._output_path_is_suggested and matching_row is not None:
            self._refresh_suggested_output_path()
        if matching_row is not None:
            matching_row["axis_declaration"].source_binding_changed()
        self._invalidate_preview_plan()

    def _source_pattern_changed(
        self,
        control: AxisInterpretationControl,
    ) -> None:
        control.source_binding_changed()
        self._invalidate_preview_plan()

    def _refresh_suggested_output_path(self) -> None:
        if not self._output_path_is_suggested:
            return
        source = next(
            (
                row["folder"].text().strip()
                for row in self._source_rows
                if row["folder"].text().strip()
            ),
            "",
        )
        suggested = str(Path(source).expanduser() / "output") if source else ""
        self._set_output_path(suggested, suggested=True)

    def _set_output_path(self, path: str, *, suggested: bool) -> None:
        self._output_path_is_suggested = suggested
        self._setting_suggested_output = suggested
        try:
            self.output_edit.setText(path)
        finally:
            self._setting_suggested_output = False
        self._refresh_output_path_style()

    def _output_path_changed(self, _text: str) -> None:
        if self._setting_suggested_output:
            return
        self._acknowledge_output_path()
        self._invalidate_preview_plan()

    def _acknowledge_output_path(self) -> None:
        if not self._output_path_is_suggested:
            return
        self._output_path_is_suggested = False
        self._refresh_output_path_style()

    def _refresh_output_path_style(self) -> None:
        is_visible_suggestion = self._output_path_is_suggested and bool(
            self.output_edit.text().strip()
        )
        self.output_edit.setProperty("suggestedDefault", is_visible_suggestion)
        if is_visible_suggestion:
            self.output_edit.setStyleSheet(
                "QLineEdit { color: #f59e0b; border: 1px solid #f59e0b; }"
            )
            description = (
                "Suggested from the first bound batch source. Review this folder; "
                "click or edit the field to acknowledge it."
            )
        else:
            self.output_edit.setStyleSheet("")
            description = "Folder where VIPP will save this batch's outputs."
        self.output_edit.setToolTip(description)
        self.output_edit.setAccessibleDescription(description)

    def values(self) -> dict[str, object]:
        bindings = []
        for row in self._source_rows:
            bindings.append(
                {
                    "node_id": row["node_id"],
                    "title": row["title"],
                    "input_dir": row["folder"].text(),
                    "pattern": row["pattern"].text(),
                    "axis_declaration": row["axis_declaration"].text(),
                }
            )
        values: dict[str, object] = {
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
        if self.parameter_override_editor.configured:
            overrides = self.parameter_override_editor.overrides()
            if overrides:
                values["parameter_overrides"] = overrides
        elif self._pending_parameter_overrides:
            # Preserve loaded values while the controller builds a fresh plan.
            # The disabled Run button prevents their use before that contract is
            # verified by ``configure_parameter_overrides``.
            values["parameter_overrides"] = self._pending_parameter_overrides
        return values

    def configure_parameter_overrides(
        self,
        sources: list[BatchOverrideSourceItem],
        parameters: list[BatchOverrideParameterSpec],
        *,
        overrides: tuple[BatchSourceParameterOverrides, ...] | None = None,
    ) -> bool:
        """Install the exact planned SourceItem/ParameterSpec editing contract.

        ``None`` restores overrides loaded with a batch config. Passing an
        explicit empty tuple starts with every cell inheriting the workflow.
        """

        existing = self._pending_parameter_overrides if overrides is None else overrides
        self.parameter_override_group.show()
        configured = self.parameter_override_editor.configure(
            sources,
            parameters,
            overrides=existing,
        )
        if configured:
            self._pending_parameter_overrides = ()
        self._sync_parameter_override_validity(configured)
        return configured

    def parameter_overrides(self) -> tuple[BatchSourceParameterOverrides, ...]:
        """Return current canonical overrides for a configured plan."""

        if self.parameter_override_editor.configured:
            return self.parameter_override_editor.overrides()
        return self._pending_parameter_overrides

    def clear_parameter_override_contract(self) -> None:
        """Remove the optional editor without changing ordinary dialog values."""

        self._pending_parameter_overrides = ()
        self.parameter_override_editor.clear_contract()
        self.parameter_override_group.hide()
        self._sync_parameter_override_validity(True)

    def _parameter_overrides_changed(self) -> None:
        self._invalidate_preview_plan()
        try:
            overrides: object = self.parameter_override_editor.overrides()
        except ValueError:
            overrides = None
        self.parameterOverridesChanged.emit(overrides)
        self._sync_parameter_override_validity()

    def _sync_parameter_override_validity(self, *_args) -> None:
        if not hasattr(self, "parameter_override_editor"):
            return
        error = self.parameter_override_editor.error_message
        if error:
            self.run_button.setEnabled(False)
            self.run_button.setToolTip(error)
        elif not self._run_in_progress:
            self.run_button.setEnabled(
                self._actions is not None and not self._representative_pending
            )
            self.run_button.setToolTip("")

    def _request_run(self) -> None:
        """Request execution without accepting or hiding this workspace."""
        if self._run_in_progress:
            return
        self.begin_run_preflight()
        self.runRequested.emit(self.values())

    def begin_run_preflight(self) -> None:
        """Show the immediate metadata/path checks owned by a Run request."""

        self._activity_run_total = 0
        self._activity_run_index = 0
        self._activity_run_completed = 0
        self.show_workspace_activity(
            "Checking current inputs, destinations, and parameters…",
            state="working",
            indeterminate=True,
            progress_text="Checking",
        )
        self.batch_activity_strip.repaint()

    def _invalidate_preview_plan(self, *_args) -> None:
        """Discard a plan as soon as any setting that produced it changes."""
        if self._run_in_progress:
            return
        self._hide_source_detection_progress()
        self.clear_demo_context()
        self._loaded_config_path = None
        self._preview_result = None
        self._preview_table_rows.clear()
        self.preview_table.setRowCount(0)
        self.preview_item_button.setEnabled(False)
        self.preview_button.setEnabled(self._actions is not None)
        self.preview_status.setText(
            "Settings changed. Preview again, or run when ready."
        )
        self.show_workspace_activity(
            "Not checked · batch settings changed.",
            state="info",
        )
        self.graph_preview_status.setText(
            "Preview the batch to inspect a sample in the graph."
        )
        run_button = getattr(self, "run_button", None)
        if run_button is not None:
            run_button.setEnabled(self._actions is not None)
        self._sync_parameter_override_validity()
        self.previewInvalidated.emit()

    def invalidate_for_workflow_change(self) -> None:
        """Invalidate runnable planning while retaining the last graph sample."""
        if self._run_in_progress:
            return
        self._invalidate_preview_plan()
        self.preview_status.setText(
            "The scientific workflow changed. Run batch will rebuild all "
            "destinations, or use Preview batch to inspect them first."
        )
        self.show_workspace_activity(
            "Not checked · the scientific workflow changed.",
            state="warning",
        )
        self.graph_preview_status.setText(
            "Representative navigation still uses the previous source pairing "
            "and calculates it through the edited graph."
        )

    def invalidate_for_source_change(
        self,
        message: str,
        *,
        before_run_started: bool = False,
    ) -> None:
        """Require explicit Refresh when a reviewed source revision changed."""
        restore_controls = False
        if self._run_in_progress:
            if not before_run_started:
                return
            # The caller completed final source verification before handing any
            # work to a batch worker.  Consume the transient run-button state so
            # the now-invalid reviewed plan can actually be discarded.
            self._run_in_progress = False
            self.cancel_run_button.hide()
            self._run_control_restore_timer.stop()
            restore_controls = True
        self._invalidate_preview_plan()
        self.preview_status.setText(
            "A representative source changed after it was reviewed. Press "
            f"Refresh and wait for recalculation before running. {str(message)}"
        )
        self.show_workspace_activity(
            "Needs attention · a reviewed source changed.",
            state="warning",
            tooltip=str(message),
        )
        self.graph_preview_status.setText(
            "The graph still uses its pinned earlier source revision until "
            "Refresh is pressed."
        )
        if restore_controls:
            self._restore_run_controls()

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
            "calculation to finish. Run batch will then build a fresh plan; "
            "Preview batch remains available for optional inspection."
        )
        self.graph_preview_status.setText(
            f"Refreshing representative item {int(position) + 1} of "
            f"{int(total)} ({str(batch_id)}) through the graph..."
        )
        self.show_workspace_activity(
            f"Refreshing item {int(position) + 1} of {int(total)} in main VIPP…",
            state="working",
            indeterminate=True,
            progress_text="Graph",
        )
        self.set_representative_pending(True)

    def show_plan_refresh_required(self, message: str) -> None:
        """Explain why a newly refreshed plan must be reviewed before running."""
        self.preview_status.setText(str(message))
        self.show_workspace_activity(
            "Needs attention · review the refreshed batch plan.",
            state="warning",
            tooltip=str(message),
        )

    def set_representative_pending(self, pending: bool) -> None:
        """Keep full execution unavailable until graph preview is trustworthy."""
        if self._run_in_progress:
            return
        self._representative_pending = bool(pending)
        self._sync_parameter_override_validity()

    def mark_plan_historical_after_run(self) -> None:
        """Retain run evidence but require fresh preflight before replay."""
        self._preview_result = None
        self.preview_item_button.setEnabled(False)
        self.preview_status.setText(
            "Historical preflight: the column above records the completed "
            "run's plan. Run batch will preflight current inputs and destinations "
            "again; Preview batch remains available for inspection."
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
        self.show_workspace_activity(
            f"Calculating item {position + 1} of "
            f"{self._preview_result.total_items} in main VIPP…",
            state="working",
            indeterminate=True,
            progress_text="Graph",
        )
        try:
            outcome = self._actions.preview_item(position)
        except Exception as exc:
            self.graph_preview_status.setText(f"Graph preview failed: {exc}")
            self.show_workspace_activity(
                f"Needs attention · representative item {position + 1} failed.",
                state="error",
                tooltip=str(exc),
            )
            return False
        if outcome is False:
            if self.batch_activity_status.text().startswith("Calculating item"):
                self.show_workspace_activity(
                    "Needs attention · representative calculation did not start.",
                    state="warning",
                )
            return False
        if outcome is None:
            self.set_graph_preview_ready(position)
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
        self.show_workspace_activity(
            f"Calculating item {int(position) + 1} of "
            f"{self._preview_result.total_items} in main VIPP…",
            state="working",
            indeterminate=True,
            progress_text="Graph",
        )

    def set_graph_preview_ready(self, position: int) -> None:
        """Confirm that the matching representative calculation completed."""
        self._set_graph_preview_status(int(position))
        if self._preview_result is not None:
            self.show_workspace_activity(
                f"Ready · item {int(position) + 1} of "
                f"{self._preview_result.total_items} shown in main VIPP.",
                state="ready",
            )

    def show_graph_preview_error(self, position: int, message: str) -> None:
        """Retain the selected plan while surfacing a preview calculation error."""
        self.graph_preview_status.setText(
            f"Representative item {int(position) + 1} could not be shown: "
            f"{str(message).strip()}"
        )
        self.show_workspace_activity(
            f"Needs attention · representative item {int(position) + 1} failed.",
            state="error",
            tooltip=str(message),
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
            self.show_workspace_activity(
                "Not checked · Preview is unavailable in this context.",
                state="warning",
            )
            return False
        self.show_workspace_activity(
            "Checking batch inputs, outputs, and parameters…",
            state="working",
            indeterminate=True,
            progress_text="Checking",
        )
        self.batch_activity_strip.repaint()
        result = None
        for attempt in range(2):
            try:
                result = self._actions.preview_batch(self.values(), 25)
                break
            except BatchScientificPreflightError as exc:
                if attempt == 0 and self.apply_axis_suggestion(exc):
                    continue
                self._show_preview_failure(
                    exc.user_message,
                    technical_detail=exc.technical_detail,
                )
                return False
            except Exception as exc:
                self._show_preview_failure(
                    f"Preview could not be prepared: {exc}",
                )
                return False
        if result is None:
            self._show_preview_failure(
                "Preview could not be prepared because no batch plan was returned."
            )
            return False
        self.apply_preview_result(result, preview_representative=True)
        return True

    def apply_axis_suggestion(self, error: BatchScientificPreflightError) -> bool:
        """Apply one exact UI suggestion and make the change visible."""
        suggestion = error.axis_suggestion
        if suggestion is None:
            return False
        row = next(
            (
                item
                for item in self._source_rows
                if item["node_id"] == suggestion.source_node_id
            ),
            None,
        )
        if row is None:
            return False
        control = row["axis_declaration"]
        if not control.apply_z_stack_suggestion(suggestion):
            return False
        self.content_scroll.ensureWidgetVisible(row["widget"])
        return True

    def _show_preview_failure(
        self,
        message: str,
        *,
        technical_detail: str = "",
    ) -> None:
        """Show one actionable issue while retaining detail in a tooltip."""
        self._hide_source_detection_progress()
        self.clear_demo_context()
        self._preview_result = None
        self._preview_table_rows.clear()
        self.preview_table.setRowCount(0)
        concise = str(message).strip() or "Preview could not be prepared."
        self.preview_status.setText(concise)
        self.preview_status.setToolTip(str(technical_detail).strip())
        self.show_workspace_activity(
            "Needs attention · batch preview could not be prepared.",
            state="error",
            tooltip=str(technical_detail).strip() or concise,
        )
        self.graph_preview_status.setText(
            "Change the highlighted setting, then preview again."
        )
        self.preview_item_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.previewInvalidated.emit()

    def show_workspace_activity(
        self,
        message: str,
        *,
        state: str = "info",
        indeterminate: bool = False,
        current: int | None = None,
        total: int | None = None,
        progress_text: str = "",
        tooltip: str = "",
    ) -> None:
        """Update the fixed Batch summary without replacing detailed evidence.

        This public hook is also used by the owning VIPP widget for short
        queueing phases that exist outside the dialog's synchronous actions.
        """

        normalized_state = str(state).strip().lower()
        color = {
            "working": "#bfdbfe",
            "ready": "#86efac",
            "success": "#86efac",
            "warning": "#fbbf24",
            "error": "#fca5a5",
            "info": "#cbd5e1",
        }.get(normalized_state, "#cbd5e1")
        self.batch_activity_status.setText(str(message).strip() or "Not checked")
        self.batch_activity_status.setStyleSheet(f"color: {color};")
        activity_tooltip = str(tooltip).strip() or self._batch_activity_tooltip
        self.batch_activity_status.setToolTip(activity_tooltip)
        self.source_detection_progress.setToolTip(activity_tooltip)
        if indeterminate:
            self.source_detection_progress.setRange(0, 0)
            self.source_detection_progress.setFormat(
                str(progress_text).strip() or "Working"
            )
            self.source_detection_progress.show()
            return
        if total is not None and int(total) > 0:
            bounded_total = max(int(total), 1)
            bounded_current = max(min(int(current or 0), bounded_total), 0)
            self.source_detection_progress.setRange(0, bounded_total)
            self.source_detection_progress.setValue(bounded_current)
            self.source_detection_progress.setFormat(
                str(progress_text).strip() or f"{bounded_current} / {bounded_total}"
            )
            self.source_detection_progress.show()
            return
        self._hide_source_detection_progress()

    def _show_source_detection_progress(self) -> None:
        """Show automatic source inspection in the fixed activity strip."""

        self.show_workspace_activity(
            "Detecting batch items and checking saved sources…",
            state="working",
            indeterminate=True,
            progress_text="Checking",
        )

    def _hide_source_detection_progress(self) -> None:
        """Stop the compact activity bar without clearing its summary text."""

        self.source_detection_progress.hide()
        self.source_detection_progress.setRange(0, 1)
        self.source_detection_progress.setValue(0)
        self.source_detection_progress.setFormat("Not active")

    def begin_saved_workspace_discovery(self, override_count: int = 0) -> None:
        """Present automatic metadata-only discovery for a saved workspace."""

        self._preview_result = None
        self._preview_table_rows.clear()
        self.preview_table.setRowCount(0)
        self.preview_item_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        if int(override_count) > 0:
            self.parameter_override_group.show()
            self.parameter_override_editor.mark_saved_overrides_verifying(
                override_count
            )
        else:
            self.parameter_override_group.hide()
        self._show_source_detection_progress()
        self.preview_status.setText(
            "Detecting samples and checking source revisions for this saved "
            "Batch workspace. No representative image is being calculated."
        )
        self.preview_status.setToolTip("")
        self.graph_preview_status.setText(
            "The graph remains in its saved single-image state while source "
            "identities are checked."
        )
        self.run_button.setEnabled(False)

    def show_saved_workspace_discovery_failure(
        self,
        message: str,
        *,
        technical_detail: str = "",
    ) -> None:
        """End discovery while explaining any source or override discrepancy."""

        self._hide_source_detection_progress()
        self._preview_result = None
        self._preview_table_rows.clear()
        self.preview_table.setRowCount(0)
        self.preview_item_button.setEnabled(False)
        self.preview_button.setEnabled(self._actions is not None)
        concise = str(message).strip() or "Batch samples could not be detected."
        if self._pending_parameter_overrides:
            self.parameter_override_group.show()
            self.parameter_override_editor.mark_saved_overrides_pending_review(
                len(self._pending_parameter_overrides),
                reason=concise,
            )
            detail = " The saved values were kept and were not reassigned."
        else:
            self.parameter_override_editor.clear_contract()
            self.parameter_override_group.hide()
            detail = ""
        self.preview_status.setText(
            f"{concise}{detail} Resolve the source setting, then use Preview "
            "batch to verify again."
        )
        self.preview_status.setToolTip(str(technical_detail).strip())
        self.show_workspace_activity(
            "Needs attention · batch samples could not be verified.",
            state="error",
            tooltip=str(technical_detail).strip() or concise,
        )
        self.graph_preview_status.setText(
            "No representative image was loaded from the unverified collection."
        )
        self.run_button.setEnabled(False)

    def show_saved_workspace_discovery_success(
        self,
        *,
        item_count: int,
        override_count: int,
    ) -> None:
        """Confirm automatic discovery while leaving pixel preview optional."""

        self._hide_source_detection_progress()
        self.preview_button.setEnabled(self._actions is not None)
        if int(override_count) > 0:
            entries = "entry" if int(override_count) == 1 else "entries"
            message = (
                f"Restored and verified {int(item_count)} current batch item(s); "
                f"{int(override_count)} saved per-sample override {entries} "
                "matched an exact source revision. Nothing was saved."
            )
        else:
            message = (
                f"Restored and detected {int(item_count)} current batch item(s); "
                "source revisions were checked automatically. Nothing was saved."
            )
        self.preview_status.setText(message)
        self.preview_status.setToolTip("")
        item_word = "item" if int(item_count) == 1 else "items"
        self.show_workspace_activity(
            f"Ready · {int(item_count)} batch {item_word}.",
            state="ready",
        )
        self.graph_preview_status.setText(
            "No representative image was calculated. Select a row and use "
            "Preview selected in graph only when you want to inspect pixels."
        )

    def cancel_saved_workspace_discovery(
        self,
        message: str = (
            "Batch settings changed before sample detection finished. Use "
            "Preview batch to detect the current samples."
        ),
    ) -> None:
        """End automatic discovery after a settings change or cancellation."""

        self._hide_source_detection_progress()
        self.preview_button.setEnabled(self._actions is not None)
        concise = str(message).strip() or (
            "Batch sample detection was cancelled. Use Preview batch to try again."
        )
        if self._pending_parameter_overrides:
            self.parameter_override_group.show()
            self.parameter_override_editor.mark_saved_overrides_pending_review(
                len(self._pending_parameter_overrides),
                reason=concise,
            )
            self.run_button.setEnabled(False)
        else:
            self.parameter_override_editor.clear_contract()
            self.parameter_override_group.hide()
            self._sync_parameter_override_validity(True)
        self.preview_status.setText(concise)
        self.preview_status.setToolTip("")
        self.show_workspace_activity(
            "Not checked · sample detection was cancelled.",
            state="warning",
            tooltip=concise,
        )
        self.graph_preview_status.setText(
            "No representative image was loaded from the cancelled collection check."
        )

    def begin_saved_workspace_verification(self, override_count: int = 0) -> None:
        """Compatibility alias for saved-workspace discovery."""

        self.begin_saved_workspace_discovery(override_count)

    def show_saved_workspace_verification_failure(
        self,
        message: str,
        *,
        technical_detail: str = "",
    ) -> None:
        """Compatibility alias for saved-workspace discovery failure."""

        self.show_saved_workspace_discovery_failure(
            message,
            technical_detail=technical_detail,
        )

    def show_saved_workspace_verification_success(
        self,
        *,
        item_count: int,
        override_count: int,
    ) -> None:
        """Compatibility alias for saved-workspace discovery success."""

        self.show_saved_workspace_discovery_success(
            item_count=item_count,
            override_count=override_count,
        )

    def begin_saved_override_verification(self, count: int) -> None:
        """Compatibility alias for saved-workspace source verification."""

        self.begin_saved_workspace_discovery(count)

    def show_saved_override_verification_failure(
        self,
        message: str,
        *,
        technical_detail: str = "",
    ) -> None:
        """Compatibility alias for saved-workspace verification failure."""

        self.show_saved_workspace_discovery_failure(
            message,
            technical_detail=technical_detail,
        )

    def show_saved_override_verification_success(
        self,
        *,
        item_count: int,
        override_count: int,
    ) -> None:
        """Compatibility alias for saved-workspace verification success."""

        self.show_saved_workspace_discovery_success(
            item_count=item_count,
            override_count=override_count,
        )

    def apply_preview_result(
        self,
        result: BatchPreviewResult,
        *,
        preview_representative: bool,
    ) -> None:
        """Display one validated plan, optionally calculating a graph sample."""
        self._hide_source_detection_progress()
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
                f"{node_id}: {item.source_labels.get(node_id, path.name)}"
                for node_id, path in item.sources.items()
            )
            source_item = QTableWidgetItem(f"{item.batch_id}\n{source_text}")
            source_item.setToolTip(
                "\n".join(
                    f"{node_id}: {path}"
                    + (
                        f"\n  {item.source_labels[node_id]}"
                        if node_id in item.source_labels
                        else ""
                    )
                    for node_id, path in item.sources.items()
                )
            )
            self.preview_table.setItem(row_index, 1, source_item)
            output_paths = [Path(path) for path in item.outputs]
            output_labels: list[str] = []
            for path in output_paths:
                try:
                    output_labels.append(
                        str(path.relative_to(result.config.output_dir))
                    )
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
        messages = [f"Ready: {total_items} batch item(s) checked. Nothing was saved."]
        if collision_count:
            messages.append(
                f"{collision_count} existing output collision(s) need attention."
            )
        if not explicit_outputs:
            messages.append(
                "VIPP will save the final graph results because no Batch Output "
                "node was added."
            )
        if self._demo is not None:
            planned_outputs = sum(len(row.outputs) for row in result)
            messages.append(
                f"Demo ready: {total_items} paired items will write "
                f"{planned_outputs} outputs."
            )
        self.preview_status.setText(" ".join(messages))
        self.preview_status.setToolTip("")
        item_word = "item" if int(total_items) == 1 else "items"
        if collision_count:
            collision_word = "collision" if collision_count == 1 else "collisions"
            self.show_workspace_activity(
                f"Needs attention · {total_items} batch {item_word}, "
                f"{collision_count} output {collision_word}.",
                state="warning",
            )
        elif total_items:
            self.show_workspace_activity(
                f"Ready · {total_items} batch {item_word}.",
                state="ready",
            )
        else:
            self.show_workspace_activity(
                "Needs attention · no matching batch items.",
                state="warning",
            )
        if result.rows:
            self.select_preview_item(0)
            if (
                preview_representative
                and self._actions is not None
                and self._actions.preview_item is not None
            ):
                self._preview_selected_item()
            elif preview_representative:
                self.graph_preview_status.setText(
                    "The full plan is ready. Representative graph preview is "
                    "unavailable in this context."
                )
            else:
                self.graph_preview_status.setText(
                    "Run preflight is ready. No representative was loaded into "
                    "the graph; use Preview selected in graph later if desired."
                )
        else:
            self.graph_preview_status.setText(
                "The batch plan contains no representative items to preview."
            )
        self._sync_preview_item_button()

    def begin_run(self, total: int) -> None:
        """Enter retained, determinate item-level batch progress mode."""
        self._hide_source_detection_progress()
        total = max(int(total), 0)
        self._activity_run_total = total
        self._activity_run_index = 0
        self._activity_run_completed = 0
        self.show_workspace_activity(
            f"Running batch · 0 of {total} complete.",
            state="working",
            current=0,
            total=total,
        )
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
        self.operation_progress_bar.setRange(0, 0)
        self.operation_progress_bar.setFormat("Starting")
        self.operation_progress_label.setText(
            "Preparing the first item and its execution plan..."
        )
        self.cancel_run_button.setText("Cancel run")
        self.cancel_run_button.setEnabled(True)
        self.cancel_run_button.show()
        self.run_result_label.clear()
        for table_row in range(self.preview_table.rowCount()):
            self._set_table_run_status(table_row, "Pending")

    def _reset_run_display(self) -> None:
        """Clear an earlier result when a fresh batch plan becomes current."""
        self._activity_run_total = 0
        self._activity_run_index = 0
        self._activity_run_completed = 0
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat("Not run")
        self.run_progress_label.setText("No batch run is active.")
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.operation_progress_bar.setFormat("Not run")
        self.operation_progress_label.setText("No node operation is active.")
        self.cancel_run_button.hide()
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
        self._activity_run_total = total
        self._activity_run_index = index
        self._activity_run_completed = completed
        self.show_workspace_activity(
            f"Running batch · item {index} of {total}.",
            state="working",
            current=completed,
            total=total,
        )
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
    ) -> None:
        """Show nested progress for the currently executing CPU/GPU node."""
        item_index = max(int(item_index), 1)
        item_total = max(int(item_total), 0)
        current = max(int(current), 0)
        total = max(int(total), 0)
        if total:
            current = min(current, total)
            self.operation_progress_bar.setRange(0, total)
            self.operation_progress_bar.setValue(current)
            self.operation_progress_bar.setFormat(f"{current} / {total}")
        else:
            self.operation_progress_bar.setRange(0, 0)
            self.operation_progress_bar.setFormat("Working")
        operation = str(operation_id).strip() or str(node_id).strip() or "operation"
        detail = str(message).strip()
        suffix = f" — {detail}" if detail else ""
        self.operation_progress_label.setText(
            f"Item {item_index} of {item_total}: {batch_id}; "
            f"{operation} ({node_id}){suffix}."
        )

    def _request_cancel(self) -> None:
        if not self._run_in_progress or not self.cancel_run_button.isEnabled():
            return
        self.cancel_run_button.setEnabled(False)
        self.cancel_run_button.setText("Cancelling...")
        self.operation_progress_label.setText(
            "Cancellation requested; waiting for the active operation's next "
            "safe checkpoint..."
        )
        self.show_workspace_activity(
            "Cancelling safely after the current operation…",
            state="warning",
            current=self._activity_run_completed,
            total=self._activity_run_total,
        )
        self.cancelRequested.emit()

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
        cancelled = bool(getattr(result, "cancelled", False))
        failed = int(summary.get("failed", 0))
        partial = int(summary.get("partial", 0))
        completed = int(summary.get("completed", 0))
        if cancelled:
            self.show_workspace_activity(
                f"Cancelled · {completed} of {total} completed.",
                state="warning",
                current=total,
                total=total,
            )
        elif failed or partial:
            issue_count = failed + partial
            issue_word = "issue" if issue_count == 1 else "issues"
            self.show_workspace_activity(
                f"Completed with {issue_count} {issue_word} · {total} items.",
                state="warning",
                current=total,
                total=total,
            )
        else:
            self.show_workspace_activity(
                f"Complete · {total} of {total} items.",
                state="success",
                current=total,
                total=total,
            )
        self.run_progress_label.setText(
            ("Batch cancelled: " if cancelled else "Batch finished: ")
            + f"{summary['completed']} completed, "
            + f"{summary['partial']} partial, {summary['skipped']} skipped, "
            + f"{summary['failed']} failed"
            + (
                f", {summary.get('cancelled', 0)} cancelled."
                if "cancelled" in summary
                else "."
            )
        )
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(1)
        self.operation_progress_bar.setFormat("Cancelled" if cancelled else "Complete")
        self.operation_progress_label.setText(
            "The batch stopped at a safe cancellation checkpoint."
            if cancelled
            else "All planned item execution has finished."
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
        if self._activity_run_total > 0:
            self.show_workspace_activity(
                f"Failed · item {max(self._activity_run_index, 1)} of "
                f"{self._activity_run_total}.",
                state="error",
                current=self._activity_run_completed,
                total=self._activity_run_total,
                tooltip=str(message),
            )
        else:
            self.show_workspace_activity(
                "Failed · the batch did not start.",
                state="error",
                tooltip=str(message),
            )
        self.run_progress_bar.setFormat("Failed")
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.operation_progress_bar.setFormat("Stopped")
        self.operation_progress_label.setText(
            "The active operation stopped before the batch could finish."
        )
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

    def show_preflight_error(
        self,
        message: str,
        *,
        technical_detail: str = "",
    ) -> None:
        """Show one deterministic setup issue without runtime-failure noise."""
        concise = str(message).strip() or "The batch needs one setting changed."
        self.show_workspace_activity(
            "Needs attention · batch preflight did not pass.",
            state="error",
            tooltip=str(technical_detail).strip() or concise,
        )
        self.run_group.show()
        self.run_progress_label.setText("Batch not started.")
        self.run_result_label.setText(concise)
        self.run_result_label.setToolTip(str(technical_detail).strip())
        self.run_progress_bar.setRange(0, 1)
        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setFormat("Needs attention")
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.operation_progress_bar.setFormat("")
        self.operation_progress_label.setText(
            "Change the highlighted setting, then preview or run again."
        )
        self._finish_run_interaction(False)
        self.preview_status.setText(concise)
        self.preview_status.setToolTip(str(technical_detail).strip())
        self.run_button.setEnabled(False)

    def show_overwrite_cancelled(self, collision_count: int) -> None:
        """Retain a reviewed collision plan after the user declines replacement."""

        self._hide_source_detection_progress()
        count = max(int(collision_count), 0)
        output_word = "output" if count == 1 else "outputs"
        self.show_workspace_activity(
            "Batch not started · existing outputs were left unchanged.",
            state="warning",
        )
        self.preview_status.setText(
            f"Batch not started. {count} existing {output_word} were left "
            "unchanged. Run again to reconsider, or choose another Existing "
            "files policy."
        )
        self.run_button.setEnabled(
            self._actions is not None and not self._representative_pending
        )
        self._sync_parameter_override_validity()

    def _finish_run_interaction(self, defer_control_restore: bool) -> None:
        """Consume queued run clicks before restoring setup controls."""
        self.cancel_run_button.hide()
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
            self.parameter_override_group,
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
        self.preview_status.setText(
            f"Loaded {Path(path).name}. Its saved compute settings will be "
            "used unless you change the compute toolbar."
        )

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
        self._hide_source_detection_progress()
        self._loaded_compute_request = config.compute_request
        self._pending_parameter_overrides = tuple(config.parameter_overrides)
        self.parameter_override_editor.clear_contract()
        if self._pending_parameter_overrides:
            self.parameter_override_group.show()
            self.parameter_override_editor.mark_saved_overrides_pending_review(
                len(self._pending_parameter_overrides)
            )
        else:
            self.parameter_override_group.hide()
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
        self._acknowledge_output_path()
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
            row["axis_declaration"].setText("")
            self.source_layout.removeWidget(row["widget"])
            self.source_layout.addWidget(row["widget"])
        for source in config.sources:
            row = rows[source.node_id]
            row["title"] = source.title
            suffix = "  - collection" if row["binding_mode"] == "collection" else ""
            row["title_label"].setText(f"{source.title}{suffix}")
            row["title_label"].setToolTip(
                f"Workflow source: {source.title} ({source.node_id})"
            )
            row["folder"].setText(str(config.resolve_path(source.input_dir)))
            row["pattern"].setText(source.pattern)
            row["axis_declaration"].setText(
                ""
                if source.axis_declaration is None
                else source.axis_declaration.display_text
            )
        if self._source_rows:
            self.input_edit = self._source_rows[0]["folder"]
            self.pattern_edit = self._source_rows[0]["pattern"]
        self._set_output_path(
            str(config.resolve_path(config.output_dir)),
            suggested=False,
        )
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
        self.show_workspace_activity(
            "Not checked · loaded batch settings are ready for detection.",
            state="info",
        )

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch input folder",
            recent_paths.initial_directory(
                recent_paths.INPUT_DIRECTORY,
                self.input_edit.text(),
            ),
        )
        if path:
            recent_paths.remember_directory(
                recent_paths.INPUT_DIRECTORY,
                path,
            )
            self.input_edit.setText(path)

    def _browse_source_input(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch input folder",
            recent_paths.initial_directory(
                recent_paths.INPUT_DIRECTORY,
                edit.text(),
            ),
        )
        if path:
            recent_paths.remember_directory(
                recent_paths.INPUT_DIRECTORY,
                path,
            )
            edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select batch output folder",
            self.output_edit.text(),
        )
        if path:
            self._set_output_path(path, suggested=False)


__all__ = [
    "AxisInterpretationControl",
    "BatchDialogValues",
    "BatchOverrideParameterSpec",
    "BatchOverrideSourceItem",
    "BatchParameterOverrideEditor",
    "BatchPreviewResult",
    "BatchPreviewRow",
    "BatchSourceBinding",
    "CollectionBatchActions",
    "CollectionBatchDialog",
    "PreviewBatchItemAction",
]
