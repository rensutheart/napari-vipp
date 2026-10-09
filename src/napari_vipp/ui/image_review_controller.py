"""Workflow-bound image review; presentation never dispatches scientific work."""

from __future__ import annotations

from copy import deepcopy
from weakref import ref

from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QLabel, QPushButton

from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.review_images import (
    default_review_settings,
    prepare_review_input,
    validate_review_pair,
    validate_review_settings,
)
from napari_vipp.ui.status import MessageSeverity

REVIEW_UPDATING_MESSAGE = "Updating · Showing previous results"
REVIEW_STALE_MESSAGE = "Stale inputs · Showing previous results"


class ImageReviewController(QObject):
    """Keep each read-only window attached to its originating workflow/node."""

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self.windows = {}
        self._tokens = {}

    def load_settings(self, raw):
        if not isinstance(raw, dict):
            raise ValueError("Image reviews must be a mapping of node IDs to settings.")
        settings_by_node = {
            node_id: validate_review_settings(settings)
            for node_id, settings in raw.items()
            if node_id in self.widget.pipeline.nodes
            and self.widget.pipeline.nodes[node_id].operation_id == "review_images"
        }
        self.close_session(self.widget._current_workflow_session_id())
        self.widget._image_review_settings = settings_by_node
        self.refresh()

    def _context(self, node_id):
        return self.widget._current_workflow_session_id(), node_id

    def _inputs(self, node_id, *, prepare=True):
        widget = self.widget
        if (
            node_id not in widget.pipeline.nodes
            or widget.pipeline.nodes[node_id].operation_id != "review_images"
        ):
            raise ValueError("This image review node no longer exists.")
        bindings = {
            edge.target_port: edge
            for edge in widget.pipeline._input_connections(node_id)
        }
        if 0 not in bindings:
            raise ValueError("Connect Image A, then calculate its upstream image.")
        # Per-node background previews are not one accepted A+B snapshot. In
        # particular, a new smoothed A may precede its downstream mask B.
        if (
            widget._active_pipeline_run_id is not None
            or widget._active_source_load_id is not None
            or widget._pipeline_run_pending
            or widget._source_load_pending
        ):
            raise ValueError(REVIEW_UPDATING_MESSAGE)
        relevant = widget.pipeline.ancestors_inclusive(
            edge.source_id for edge in bindings.values()
        )
        dirty = (
            widget._pending_dirty_node_ids
            | (widget._inflight_dirty_node_ids or set())
            | widget._pending_manual_node_ids
        )
        if relevant & dirty or widget._inflight_full_graph:
            raise ValueError(REVIEW_STALE_MESSAGE)
        descriptors, token = [], []
        for port in (0, 1):
            edge = bindings.get(port)
            if edge is None:
                descriptors.append(None)
                token.append(None)
                continue
            data, state = widget._node_output_payload_for_port(
                edge.source_id, edge.source_port, accepted_only=True
            )
            if (
                widget.pipeline.node_execution_states.get(edge.source_id)
                != EXECUTION_READY
                or edge.source_id not in widget.pipeline.completed_node_ids
                or data is None
            ):
                raise ValueError(
                    f"Image {'A' if port == 0 else 'B'} is stale or unavailable. "
                    "Calculate its upstream node before opening the review."
                )
            if prepare:
                descriptors.append(
                    prepare_review_input(
                        data, state, widget._node_title(edge.source_id)
                    )
                )
            token.append(
                (
                    edge.source_id,
                    edge.source_port,
                    id(data),
                    state,
                    getattr(data, "shape", None),
                    str(getattr(data, "dtype", "")),
                    widget._node_title(edge.source_id),
                )
            )
        if prepare:
            validate_review_pair(*descriptors)
        return tuple(item for item in descriptors if item is not None), tuple(token)

    def render_parameters(self, node_id):
        widget = self.widget
        widget.parameter_group.show()
        note = QLabel(
            "Open a separate linked viewer for Image A and optional Image B. "
            "Contrast, colours, opacity and layout are display only; "
            "they never change image values or recalculate the workflow. "
            "Open reviews follow accepted upstream recalculations. "
            "Both images must already have matching axes and calibration."
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.PlainText)
        widget.parameter_form.addRow(note)
        button = QPushButton("Open review…")
        button.setAccessibleName("Open linked image review")
        button.clicked.connect(lambda _checked=False: self.open_review(node_id))
        widget.parameter_form.addRow(button)

    def open_review(self, node_id):
        widget = self.widget
        key = self._context(node_id)
        try:
            inputs, token = self._inputs(node_id)
            window = self.windows.get(key)
            if window is not None and getattr(window, "update_failed", False):
                self._close_key(key)
                window = None
            if window is not None and self._tokens.get(key) != token:
                try:
                    window.update_inputs(inputs)
                except ValueError:
                    # Explicit Open may establish a new type/grid/layout. A
                    # background refresh must never silently reset that view.
                    self._close_key(key)
                    window = None
                else:
                    self._tokens[key] = token
            if window is None:
                from napari_vipp.ui.image_review import ImageReviewWindow

                settings = widget._image_review_settings.get(
                    node_id, default_review_settings()
                )
                window = ImageReviewWindow(
                    inputs=inputs,
                    settings=settings,
                    on_settings_changed=lambda values, context=key: self._commit(
                        context, values
                    ),
                    parent=widget,
                )
                window.setWindowTitle(f"{widget._node_title(node_id)} — Image review")
                self.windows[key] = window
                self._tokens[key] = token
                window_ref = ref(window)
                window.closed.connect(
                    lambda context=key, owned=window_ref: self._forget_closed(
                        context, owned()
                    )
                )
            window.set_current(True)
            window.show()
            window.raise_()
            window.activateWindow()
            return window
        except (ValueError, TypeError, MemoryError, RuntimeError) as exc:
            window = self.windows.get(key)
            if window is not None:
                window.set_pending(str(exc))
            widget._set_status(
                f"Image review: {exc}",
                severity=MessageSeverity.WARNING,
                actionable=True,
            )
            return None

    def _commit(self, context, values):
        widget = self.widget
        if context != self._context(context[1]):
            return
        try:
            _inputs, token = self._inputs(context[1], prepare=False)
        except (ValueError, TypeError, MemoryError):
            self.refresh()
            return
        if self._tokens.get(context) != token:
            self.refresh()
            return
        settings = validate_review_settings(values)
        if widget._image_review_settings.get(context[1]) == settings:
            return
        widget._image_review_settings[context[1]] = deepcopy(settings)
        # Save-persistent UI metadata gets its own dirty token. Do not touch
        # node parameters, compute/cache identity, history or dirty frontiers.
        widget._sync_current_workflow_tab_state()

    def refresh(self):
        current = self.widget._current_workflow_session_id()
        for key, window in tuple(self.windows.items()):
            if key[0] != current:
                continue
            try:
                _inputs, token = self._inputs(key[1], prepare=False)
                if token != self._tokens.get(key):
                    inputs, token = self._inputs(key[1])
                    window.update_inputs(inputs)
                    self._tokens[key] = token
            except (ValueError, TypeError, MemoryError, RuntimeError) as exc:
                window.set_pending(str(exc))
            else:
                window.set_current(True)

    def visible_review_nodes(self):
        """Retain only visible current-session sinks and their direct inputs."""
        current = self.widget._current_workflow_session_id()
        return {
            key[1]
            for key, window in self.windows.items()
            if key[0] == current
            and key[1] in self.widget.pipeline.nodes
            and window.isVisible()
        }

    def hide_session(self, session_id):
        for key, window in self.windows.items():
            if key[0] == session_id:
                window.hide()

    def _close_key(self, key):
        window = self.windows.pop(key, None)
        self._tokens.pop(key, None)
        if window is not None:
            window.close()
            window.deleteLater()

    def _forget_closed(self, key, window):
        if window is None:
            return
        if self.windows.get(key) is window:
            self.windows.pop(key, None)
            self._tokens.pop(key, None)
        window.deleteLater()

    def close_session(self, session_id):
        for key in tuple(self.windows):
            if key[0] == session_id:
                self._close_key(key)

    def close(self):
        for key in tuple(self.windows):
            self._close_key(key)
