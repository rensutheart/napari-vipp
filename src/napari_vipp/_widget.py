"""napari dock widget for the VIPP workflow prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import QMimeData, QSignalBlocker, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp._graph import OPERATION_MIME, PipelineGraphView
from napari_vipp.core.pipeline import (
    OperationSpec,
    PrototypePipeline,
    grouped_palette_specs,
)
from napari_vipp.core.preview import make_preview, normalize_thumbnail

if TYPE_CHECKING:
    import napari


@dataclass(frozen=True)
class ParameterBounds:
    minimum: float | int
    maximum: float | int
    step: float | int
    decimals: int


class NodePalette(QTreeWidget):
    """Small categorized operation palette for adding nodes."""

    operation_requested = Signal(str)

    def __init__(self, groups: dict[str, list[OperationSpec]], parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        for category, specs in groups.items():
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsDragEnabled)
            self.addTopLevelItem(category_item)
            for spec in specs:
                item = QTreeWidgetItem([spec.title])
                item.setData(0, Qt.UserRole, spec.id)
                category_item.addChild(item)
        self.expandAll()

    def mimeData(self, items):  # noqa: N802
        mime = QMimeData()
        if not items:
            return mime
        operation_id = items[0].data(0, Qt.UserRole)
        if operation_id:
            mime.setData(OPERATION_MIME, str(operation_id).encode())
        return mime

    def _on_item_double_clicked(self, item, _column) -> None:
        operation_id = item.data(0, Qt.UserRole)
        if operation_id:
            self.operation_requested.emit(str(operation_id))


class ParameterControl(QWidget):
    """Slider with numeric entry for a single node parameter."""

    valueChanged = Signal(object)

    def __init__(self, spec, value, bounds: ParameterBounds, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._is_integer = spec.kind == "int"
        self._scale = self._scale_for(bounds)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(120)
        if self._is_integer:
            self.value_box = QSpinBox()
        else:
            self.value_box = QDoubleSpinBox()
            self.value_box.setDecimals(bounds.decimals)
        self.value_box.setMinimumWidth(74)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_box)

        self.set_bounds(bounds, value, emit=False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.value_box.valueChanged.connect(self._on_box_changed)

    def value(self):
        return self.value_box.value()

    def set_bounds(
        self,
        bounds: ParameterBounds,
        value=None,
        emit: bool = False,
    ) -> None:
        current = self.value() if value is None else value
        current = self._clamped_value(current, bounds)
        self._scale = self._scale_for(bounds)

        with QSignalBlocker(self.slider), QSignalBlocker(self.value_box):
            if self._is_integer:
                self.value_box.setRange(int(bounds.minimum), int(bounds.maximum))
                self.value_box.setSingleStep(max(int(bounds.step), 1))
                self.slider.setRange(int(bounds.minimum), int(bounds.maximum))
                self.slider.setSingleStep(max(int(bounds.step), 1))
                self.slider.setValue(int(current))
                self.value_box.setValue(int(current))
            else:
                self.value_box.setDecimals(bounds.decimals)
                self.value_box.setRange(float(bounds.minimum), float(bounds.maximum))
                self.value_box.setSingleStep(float(bounds.step))
                self.slider.setRange(
                    self._to_slider(bounds.minimum),
                    self._to_slider(bounds.maximum),
                )
                self.slider.setSingleStep(max(self._to_slider(bounds.step), 1))
                self.slider.setValue(self._to_slider(current))
                self.value_box.setValue(float(current))

        if emit:
            self.valueChanged.emit(self.value())

    def _on_slider_changed(self, value: int) -> None:
        mapped = int(value) if self._is_integer else value / self._scale
        with QSignalBlocker(self.value_box):
            self.value_box.setValue(mapped)
        self.valueChanged.emit(self.value())

    def _on_box_changed(self, value) -> None:
        with QSignalBlocker(self.slider):
            slider_value = int(value) if self._is_integer else self._to_slider(value)
            self.slider.setValue(slider_value)
        self.valueChanged.emit(self.value())

    def _scale_for(self, bounds: ParameterBounds) -> int:
        if self._is_integer:
            return 1
        if bounds.decimals > 0:
            return 10**bounds.decimals
        step = float(bounds.step)
        return max(int(round(1 / step)), 1) if step > 0 else 100

    def _to_slider(self, value) -> int:
        return int(round(float(value) * self._scale))

    def _clamped_value(self, value, bounds: ParameterBounds):
        if value is None:
            value = bounds.minimum
        return min(max(value, bounds.minimum), bounds.maximum)


class VippWidget(QWidget):
    """Visual node workflow composer hosted inside napari."""

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.pipeline = PrototypePipeline()
        self._selected_node_id = "gaussian"
        self._active_pinned_node_id: str | None = None
        self._inspect_layer_name = "VIPP Inspect"
        self._last_input_layer_name: str | None = None

        self.layer_combo = QComboBox()
        self.layer_combo.setMinimumWidth(220)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Slice", "MIP", "Off"])
        self.follow_dims_checkbox = QCheckBox("Follow napari dims")
        self.follow_dims_checkbox.setChecked(True)

        self.build_button = QPushButton("Reset graph")
        self.refresh_button = QPushButton("Refresh")
        self.status_label = QLabel(
            "Select an image layer and build the starter pipeline."
        )
        self.status_label.setWordWrap(True)

        self.palette = NodePalette(grouped_palette_specs())
        self.palette.setMinimumWidth(190)
        self.graph_view = PipelineGraphView()
        self.graph_view.setMinimumHeight(520)

        self.selected_title = QLabel("Gaussian Blur")
        self.selected_title.setStyleSheet("font-weight: 650;")
        self.parameter_group = QGroupBox("Parameters")
        self.parameter_form = QFormLayout(self.parameter_group)
        self._parameter_widgets: dict[str, QWidget] = {}

        self.inspect_button = QPushButton("Inspect selected")
        self.pin_button = QPushButton("Pin selected")

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self.run_pipeline)

        self._build_layout()
        self._connect_signals()
        self._refresh_layer_choices()
        self._build_graph_from_pipeline()
        self._select_node(self._selected_node_id)
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
        splitter.addWidget(self.palette)
        splitter.addWidget(self.graph_view)
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([210, 850, 260])
        root.addWidget(splitter, 1)
        root.addWidget(self.status_label)

    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.selected_title)
        layout.addWidget(self.parameter_group)

        actions = QHBoxLayout()
        actions.addWidget(self.inspect_button)
        actions.addWidget(self.pin_button)
        layout.addLayout(actions)
        layout.addStretch(1)
        return panel

    def _connect_signals(self) -> None:
        self.build_button.clicked.connect(self._reset_graph)
        self.refresh_button.clicked.connect(self._refresh_and_run)
        self.layer_combo.currentTextChanged.connect(self.run_pipeline)
        self.preview_mode_combo.currentTextChanged.connect(self._update_thumbnails)
        self.follow_dims_checkbox.toggled.connect(self._update_thumbnails)
        self.inspect_button.clicked.connect(
            lambda: self.inspect_node(self._selected_node_id)
        )
        self.pin_button.clicked.connect(lambda: self.pin_node(self._selected_node_id))

        self.palette.operation_requested.connect(self.add_node_from_palette)
        self.graph_view.node_create_requested.connect(self._add_node_at)
        self.graph_view.node_selected.connect(self._select_node)
        self.graph_view.inspect_requested.connect(self.inspect_node)
        self.graph_view.pin_requested.connect(self.pin_node)
        self.graph_view.connection_requested.connect(self._connect_nodes)
        self.graph_view.connection_removed.connect(self._disconnect_nodes)
        self.graph_view.status_message.connect(self.status_label.setText)

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

    def add_node_from_palette(self, operation_id: str):
        return self._add_node_at(operation_id, self.graph_view.suggest_node_position())

    def _add_node_at(self, operation_id: str, position) -> object:
        node = self.pipeline.add_node(operation_id)
        self.graph_view.add_node(node, position)
        self._sync_pin_ui()
        self.graph_view.select_node(node.id)
        self.run_pipeline()
        self.status_label.setText(f"Added '{node.title}'.")
        return node

    def _reset_graph(self) -> None:
        self.pipeline.reset_starter_graph()
        self._clear_active_pin(status=False)
        self._build_graph_from_pipeline()
        self._select_node("gaussian")
        self.run_pipeline()
        self.status_label.setText("Starter graph restored.")

    def _build_graph_from_pipeline(self) -> None:
        self.graph_view.build_graph(
            self.pipeline.nodes.values(),
            self.pipeline.connections,
        )
        self._sync_pin_ui()

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

    def _connect_nodes(self, source_id: str, target_id: str) -> None:
        result = self.pipeline.connect(source_id, target_id)
        if not result.success:
            self.status_label.setText(result.message)
            return
        for connection in result.removed:
            self.graph_view.remove_connection(
                connection.source_id,
                connection.target_id,
                notify=False,
            )
        self.graph_view.add_connection(source_id, target_id)
        self.run_pipeline()
        self.status_label.setText(result.message)

    def _disconnect_nodes(self, source_id: str, target_id: str) -> None:
        if self.pipeline.disconnect(source_id, target_id):
            self.run_pipeline()
            self.status_label.setText(f"Disconnected {source_id} -> {target_id}.")

    def _select_node(self, node_id: str) -> None:
        if node_id not in self.pipeline.nodes:
            return
        self._selected_node_id = node_id
        node = self.pipeline.nodes[node_id]
        self.selected_title.setText(node.title)
        self._render_parameters(node_id)
        self._sync_pin_ui()
        self._keep_active_pin_on_top()

    def _render_parameters(self, node_id: str) -> None:
        self._clear_parameter_form()
        specs = self.pipeline.node_parameter_specs(node_id)
        self.parameter_group.setHidden(not specs)
        if not specs:
            return

        node = self.pipeline.nodes[node_id]
        for spec in specs:
            bounds = self._parameter_bounds_for(node_id, spec)
            widget = ParameterControl(spec, node.params.get(spec.name), bounds)
            node.params[spec.name] = widget.value()
            widget.valueChanged.connect(
                lambda value, name=spec.name: self._on_param_changed(name, value)
            )
            self.parameter_form.addRow(spec.label, widget)
            self._parameter_widgets[spec.name] = widget

    def _refresh_selected_parameter_controls(self) -> bool:
        if self._selected_node_id not in self.pipeline.nodes:
            return False
        changed = False
        node = self.pipeline.nodes[self._selected_node_id]
        for spec in self.pipeline.node_parameter_specs(self._selected_node_id):
            widget = self._parameter_widgets.get(spec.name)
            if widget is None:
                continue
            previous = node.params.get(spec.name)
            widget.set_bounds(
                self._parameter_bounds_for(self._selected_node_id, spec),
                previous,
                emit=False,
            )
            current = widget.value()
            if current != previous:
                node.params[spec.name] = current
                changed = True
        return changed

    def _parameter_bounds_for(self, node_id: str, spec) -> ParameterBounds:
        if spec.name == "threshold":
            return self._threshold_bounds(node_id, spec)
        if spec.name == "axis":
            return self._axis_bounds(node_id, spec)
        return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)

    def _threshold_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        arr = np.asarray(data)
        if arr.dtype == bool:
            return ParameterBounds(0, 1, 1, 0)
        if np.issubdtype(arr.dtype, np.integer):
            if arr.dtype == np.uint8:
                return ParameterBounds(0, 255, 1, 0)
            finite = _finite_values(arr)
            if finite.size:
                return ParameterBounds(
                    int(finite.min()),
                    int(finite.max()),
                    1,
                    0,
                )
            return ParameterBounds(0, 255, 1, 0)

        finite = _finite_values(arr)
        if finite.size == 0:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        minimum = float(finite.min())
        maximum = float(finite.max())
        if 0.0 <= minimum and maximum <= 1.0:
            return ParameterBounds(0.0, 1.0, 0.01, 3)
        if minimum == maximum:
            minimum, maximum = _expanded_bounds(minimum)
        step = max((maximum - minimum) / 200.0, 1e-6)
        return ParameterBounds(minimum, maximum, step, 3)

    def _axis_bounds(self, node_id: str, spec) -> ParameterBounds:
        data = self.pipeline.input_data_for_node(node_id)
        if data is None:
            return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals)
        maximum = max(np.asarray(data).ndim - 1, 0)
        return ParameterBounds(0, maximum, 1, 0)

    def _clear_parameter_form(self) -> None:
        self._parameter_widgets.clear()
        while self.parameter_form.count():
            item = self.parameter_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_param_changed(self, name: str, value) -> None:
        self.pipeline.set_param(self._selected_node_id, name, value)
        self._debounce_timer.start()

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
        if self._refresh_selected_parameter_controls():
            self.pipeline.run(layer.data)

        self._update_thumbnails()
        self._refresh_inspection_layer_if_active()
        self._refresh_pinned_layer_if_active()
        self.status_label.setText(
            f"Graph updated from '{layer.name}'. "
            "Connect ports to build alternate paths."
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
        self._set_or_add_generated_layer(
            self._inspect_layer_name,
            data,
            metadata={"napari_vipp_kind": "inspect", "node_id": node_id},
            role="inspect",
        )
        self._keep_active_pin_on_top()
        self.status_label.setText(f"Inspecting '{title}' in napari.")

    def pin_node(self, node_id: str) -> None:
        if node_id == self._active_pinned_node_id:
            self._clear_active_pin(status=True)
            return
        data = self.pipeline.outputs.get(node_id)
        if data is None:
            self.status_label.setText("That node has no output to pin yet.")
            return
        self._set_active_pin_layer(node_id, data)
        self.status_label.setText(f"Pinned '{self._node_title(node_id)}'.")

    def _set_active_pin_layer(self, node_id: str, data) -> None:
        title = self._node_title(node_id)
        metadata = {
            "napari_vipp_kind": "pinned",
            "node_id": node_id,
            "data_kind": self._data_kind(data),
            "display_kind": self._display_kind(data, "pinned"),
        }
        layer = self._active_pinned_layer()
        if (
            layer is not None
            and layer.metadata.get("display_kind") != metadata["display_kind"]
        ):
            self._remove_layer(layer)
            layer = None

        if layer is None:
            layer = self._add_image_or_labels(
                self._pinned_layer_name(title),
                data,
                metadata,
            )
        else:
            layer.data = self._display_data(data)
            layer.metadata.update(metadata)
            layer.name = self._pinned_layer_name(title)
            layer.visible = True

        self._move_layer_to_top(layer)
        self._active_pinned_node_id = node_id
        self._sync_pin_ui()

    def _clear_active_pin(self, status: bool) -> None:
        title = (
            self._node_title(self._active_pinned_node_id)
            if self._active_pinned_node_id in self.pipeline.nodes
            else None
        )
        layer = self._active_pinned_layer()
        if layer is not None:
            self._remove_layer(layer)
        self._active_pinned_node_id = None
        self._sync_pin_ui()
        if status and title is not None:
            self.status_label.setText(f"Unpinned '{title}'.")

    def _refresh_inspection_layer_if_active(self) -> None:
        layer = self._layer_by_name(self._inspect_layer_name)
        if layer is None:
            return
        node_id = getattr(layer, "metadata", {}).get("node_id")
        if (
            node_id in self.pipeline.outputs
            and self.pipeline.outputs[node_id] is not None
        ):
            self._set_or_add_generated_layer(
                self._inspect_layer_name,
                self.pipeline.outputs[node_id],
                metadata={"napari_vipp_kind": "inspect", "node_id": node_id},
                role="inspect",
            )
            self._keep_active_pin_on_top()

    def _refresh_pinned_layer_if_active(self) -> None:
        if self._active_pinned_node_id is None:
            return
        data = self.pipeline.outputs.get(self._active_pinned_node_id)
        if data is None:
            self._clear_active_pin(status=False)
            return
        self._set_active_pin_layer(self._active_pinned_node_id, data)

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

    def _set_or_add_generated_layer(
        self,
        name: str,
        data,
        metadata: dict,
        role: str,
    ) -> None:
        metadata = {
            **metadata,
            "data_kind": self._data_kind(data),
            "display_kind": self._display_kind(data, role),
        }
        layer = self._layer_by_name(name)
        display_data = self._display_data(data)
        if layer is None:
            self._add_image_or_labels(name, data, metadata=metadata)
            return
        if layer.metadata.get("display_kind") != metadata["display_kind"]:
            self._remove_layer(layer)
            self._add_image_or_labels(name, data, metadata=metadata)
            return
        layer.data = display_data
        layer.metadata.update(metadata)
        layer.visible = True
        self._configure_generated_layer(layer, data, metadata)

    def _add_image_or_labels(self, name: str, data, metadata: dict):
        display_data = self._display_data(data)
        if metadata["display_kind"] == "labels" and hasattr(self.viewer, "add_labels"):
            return self.viewer.add_labels(display_data, name=name, metadata=metadata)
        kwargs = {"name": name, "metadata": metadata}
        if metadata["data_kind"] == "mask":
            kwargs.update(
                {
                    "blending": "opaque",
                    "colormap": "gray",
                    "contrast_limits": (0, 1),
                }
            )
        return self.viewer.add_image(display_data, **kwargs)

    def _configure_generated_layer(self, layer, data, metadata: dict) -> None:
        if metadata["display_kind"] != "image":
            return
        if metadata["data_kind"] == "mask":
            for attr, value in (
                ("blending", "opaque"),
                ("colormap", "gray"),
                ("contrast_limits", (0, 1)),
            ):
                try:
                    setattr(layer, attr, value)
                except Exception:
                    pass

    def _display_kind(self, data, role: str) -> str:
        if role == "pinned" and self._data_kind(data) == "mask":
            return "labels"
        return "image"

    def _data_kind(self, data) -> str:
        return "mask" if np.asarray(data).dtype == bool else "image"

    def _display_data(self, data):
        arr = np.asarray(data)
        if arr.dtype == bool:
            return arr.astype(np.uint8)
        return data

    def _active_pinned_layer(self):
        for layer in self.viewer.layers:
            try:
                if layer.metadata.get("napari_vipp_kind") == "pinned":
                    return layer
            except Exception:
                continue
        return None

    def _move_layer_to_top(self, layer) -> None:
        layers = self.viewer.layers
        try:
            index = list(layers).index(layer)
            layers.move(index, len(layers))
            return
        except Exception:
            pass
        try:
            layers.remove(layer)
            layers.append(layer)
        except Exception:
            pass

    def _keep_active_pin_on_top(self) -> None:
        layer = self._active_pinned_layer()
        if layer is not None:
            self._move_layer_to_top(layer)

    def _sync_pin_ui(self) -> None:
        self.graph_view.set_pinned_node(self._active_pinned_node_id)
        if self._selected_node_id == self._active_pinned_node_id:
            self.pin_button.setText("Unpin selected")
        else:
            self.pin_button.setText("Pin selected")

    def _remove_layer(self, layer) -> None:
        try:
            self.viewer.layers.remove(layer)
        except Exception:
            pass

    def _pinned_layer_name(self, title: str) -> str:
        return f"VIPP Pinned: {title}"

    def _current_step(self):
        try:
            return tuple(self.viewer.dims.current_step)
        except Exception:
            return None

    def _node_title(self, node_id: str) -> str:
        return self.pipeline.nodes[node_id].title

    def _is_vipp_generated_layer(self, layer) -> bool:
        try:
            return str(layer.metadata.get("napari_vipp_kind", "")) in {
                "inspect",
                "pinned",
            }
        except Exception:
            return False


def _finite_values(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        arr = (
            arr[..., 0].astype(np.float32) * 0.299
            + arr[..., 1].astype(np.float32) * 0.587
            + arr[..., 2].astype(np.float32) * 0.114
        )
    return arr[np.isfinite(arr)]


def _expanded_bounds(value: float) -> tuple[float, float]:
    padding = abs(value) * 0.1 or 1.0
    return value - padding, value + padding
