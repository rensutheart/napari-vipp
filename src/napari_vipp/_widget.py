"""napari dock widget for the VIPP workflow prototype."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QSignalBlocker, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.pipeline import PROTOTYPE_NODES, PrototypePipeline
from napari_vipp.core.preview import make_preview, normalize_thumbnail

if TYPE_CHECKING:
    import napari


class VippWidget(QWidget):
    """Visual node workflow composer hosted inside napari."""

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.pipeline = PrototypePipeline()
        self._selected_node_id = "gaussian"
        self._inspect_layer_name = "VIPP Inspect"
        self._last_input_layer_name: str | None = None

        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(220)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Slice", "MIP", "Off"])
        self.follow_dims_checkbox = QCheckBox("Follow napari dims")
        self.follow_dims_checkbox.setChecked(True)

        self.build_button = QPushButton("Build starter pipeline")
        self.refresh_button = QPushButton("Refresh")
        self.status_label = QLabel(
            "Select an image layer and build the starter pipeline."
        )
        self.status_label.setWordWrap(True)

        self.graph_view = PipelineGraphView()
        self.graph_view.setMinimumHeight(520)

        self.selected_title = QLabel("Gaussian Blur")
        self.selected_title.setStyleSheet("font-weight: 650;")
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 12.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setValue(self.pipeline.sigma)

        self.inspect_button = QPushButton("Inspect selected")
        self.pin_button = QPushButton("Pin selected")

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.run_pipeline)

        self._build_layout()
        self._connect_signals()
        self._refresh_layer_choices()
        self.graph_view.build_demo_graph(PROTOTYPE_NODES)
        self.run_pipeline()

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Input"))
        toolbar.addWidget(self.layer_combo, 1)
        toolbar.addWidget(QLabel("Preview"))
        toolbar.addWidget(self.preview_mode_combo)
        toolbar.addWidget(self.follow_dims_checkbox)
        toolbar.addWidget(self.build_button)
        toolbar.addWidget(self.refresh_button)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.graph_view)
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        root.addWidget(self.status_label)

    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.selected_title)

        parameter_group = QGroupBox("Parameters")
        form = QFormLayout(parameter_group)
        form.addRow("Sigma", self.sigma_spin)
        layout.addWidget(parameter_group)

        actions = QHBoxLayout()
        actions.addWidget(self.inspect_button)
        actions.addWidget(self.pin_button)
        layout.addLayout(actions)
        layout.addStretch(1)
        return panel

    def _connect_signals(self) -> None:
        self.build_button.clicked.connect(self.run_pipeline)
        self.refresh_button.clicked.connect(self._refresh_and_run)
        self.layer_combo.currentTextChanged.connect(self.run_pipeline)
        self.preview_mode_combo.currentTextChanged.connect(self._update_thumbnails)
        self.follow_dims_checkbox.toggled.connect(self._update_thumbnails)
        self.sigma_spin.valueChanged.connect(self._on_sigma_changed)
        self.inspect_button.clicked.connect(
            lambda: self.inspect_node(self._selected_node_id)
        )
        self.pin_button.clicked.connect(lambda: self.pin_node(self._selected_node_id))

        self.graph_view.node_selected.connect(self._select_node)
        self.graph_view.inspect_requested.connect(self.inspect_node)
        self.graph_view.pin_requested.connect(self.pin_node)

        try:
            self.viewer.layers.events.inserted.connect(
                lambda _=None: self._refresh_layer_choices()
            )
            self.viewer.layers.events.removed.connect(
                lambda _=None: self._refresh_layer_choices()
            )
        except Exception:
            pass
        try:
            self.viewer.dims.events.current_step.connect(
                lambda _=None: self._update_thumbnails()
            )
        except Exception:
            pass

    def _refresh_and_run(self) -> None:
        self._refresh_layer_choices()
        self.run_pipeline()

    def _refresh_layer_choices(self) -> None:
        current = self.layer_combo.currentText()
        with QSignalBlocker(self.layer_combo):
            self.layer_combo.clear()
            for layer in self.viewer.layers:
                if self._is_vipp_generated_layer(layer):
                    continue
                if hasattr(layer, "data"):
                    self.layer_combo.addItem(layer.name)
            if current:
                index = self.layer_combo.findText(current)
                if index >= 0:
                    self.layer_combo.setCurrentIndex(index)

    def _on_sigma_changed(self, value: float) -> None:
        self.pipeline.sigma = float(value)
        self._debounce_timer.start()

    def _select_node(self, node_id: str) -> None:
        self._selected_node_id = node_id
        title = next((n.title for n in PROTOTYPE_NODES if n.id == node_id), node_id)
        self.selected_title.setText(title)
        self.sigma_spin.setEnabled(node_id == "gaussian")

    def run_pipeline(self) -> None:
        layer = self._selected_input_layer()
        if layer is None:
            self.pipeline.run(None)
            self._update_thumbnails()
            self.status_label.setText("No image layer selected.")
            return

        self._last_input_layer_name = layer.name
        try:
            self.pipeline.run(layer.data)
        except Exception as exc:
            self.status_label.setText(f"Pipeline error: {exc}")
            return

        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self.status_label.setText(
            f"Starter pipeline updated from '{layer.name}'. "
            "Click a thumbnail to inspect it in napari."
        )

    def _update_thumbnails(self) -> None:
        mode = self.preview_mode_combo.currentText()
        current_step = (
            self._current_step() if self.follow_dims_checkbox.isChecked() else None
        )
        for node_id, data in self.pipeline.outputs.items():
            preview = make_preview(data, mode=mode, current_step=current_step)
            thumbnail = normalize_thumbnail(preview)
            self.graph_view.set_thumbnail(node_id, thumbnail)

    def inspect_node(self, node_id: str) -> None:
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to inspect yet.")
            return
        title = self._node_title(node_id)
        self._set_or_add_image_layer(
            self._inspect_layer_name,
            data,
            metadata={"napari_vipp_kind": "inspect", "node_id": node_id},
        )
        self.status_label.setText(f"Inspecting '{title}' in napari.")

    def pin_node(self, node_id: str) -> None:
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to pin yet.")
            return
        title = self._node_title(node_id)
        metadata = {"napari_vipp_kind": "pinned", "node_id": node_id}
        layer = self._pinned_layer_for_node(node_id)
        if layer is None:
            self._add_image_or_labels(self._pinned_layer_name(title), data, metadata)
            self.status_label.setText(
                f"Pinned '{title}' as a persistent napari layer."
            )
            return

        layer.data = self._display_data(data)
        layer.metadata.update(metadata)
        layer.visible = True
        self.status_label.setText(f"Updated pinned layer for '{title}'.")

    def _refresh_inspection_layer_if_active(self) -> None:
        layer = self._layer_by_name(self._inspect_layer_name)
        if layer is None:
            return
        node_id = getattr(layer, "metadata", {}).get("node_id")
        if (
            node_id in self.pipeline.outputs
            and self.pipeline.outputs[node_id] is not None
        ):
            layer.data = self._display_data(self.pipeline.outputs[node_id])

    def _selected_input_layer(self):
        name = self.layer_combo.currentText()
        if not name:
            return None
        return self._layer_by_name(name)

    def _layer_by_name(self, name: str):
        try:
            return self.viewer.layers[name]
        except Exception:
            for layer in self.viewer.layers:
                if layer.name == name:
                    return layer
        return None

    def _set_or_add_image_layer(self, name: str, data, metadata: dict) -> None:
        layer = self._layer_by_name(name)
        display_data = self._display_data(data)
        if layer is None:
            self._add_image_or_labels(name, data, metadata=metadata)
            return
        layer.data = display_data
        layer.metadata.update(metadata)
        layer.visible = True

    def _add_image_or_labels(self, name: str, data, metadata: dict):
        display_data = self._display_data(data)
        if np.asarray(data).dtype == bool and hasattr(self.viewer, "add_labels"):
            return self.viewer.add_labels(display_data, name=name, metadata=metadata)
        return self.viewer.add_image(display_data, name=name, metadata=metadata)

    def _display_data(self, data):
        arr = np.asarray(data)
        if arr.dtype == bool:
            return arr.astype(np.uint8)
        return data

    def _unique_layer_name(self, base: str) -> str:
        if self._layer_by_name(base) is None:
            return base
        index = 2
        while self._layer_by_name(f"{base} {index}") is not None:
            index += 1
        return f"{base} {index}"

    def _pinned_layer_name(self, title: str) -> str:
        return f"VIPP Pinned: {title}"

    def _pinned_layer_for_node(self, node_id: str):
        for layer in self.viewer.layers:
            try:
                if (
                    layer.metadata.get("napari_vipp_kind") == "pinned"
                    and layer.metadata.get("node_id") == node_id
                ):
                    return layer
            except Exception:
                continue
        return None

    def _current_step(self):
        try:
            return tuple(self.viewer.dims.current_step)
        except Exception:
            return None

    def _node_title(self, node_id: str) -> str:
        return next((n.title for n in PROTOTYPE_NODES if n.id == node_id), node_id)

    def _is_vipp_generated_layer(self, layer) -> bool:
        try:
            return str(layer.metadata.get("napari_vipp_kind", "")) in {
                "inspect",
                "pinned",
            }
        except Exception:
            return False
