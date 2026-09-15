"""Asynchronous loading and file-backed workflow tabs for measurement datasets."""

from __future__ import annotations

import threading
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton

from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot, WorkflowSnapshot
from napari_vipp.ui.history import WorkflowHistorySnapshot
from napari_vipp.ui.workflow_tabs import WorkflowTabSession


class _Signals(QObject):
    finished = Signal(object)


class _Load(QRunnable):
    def __init__(self, key, path, digest, context):
        super().__init__()
        self.key, self.path, self.digest, self.context = key, path, digest, context
        self.signals = _Signals()
        self.cancelled = threading.Event()

    def run(self):
        payload, error = None, ""
        try:
            from napari_vipp.core.table_source import load_table_source

            if not self.cancelled.is_set():
                payload = load_table_source(
                    self.path, self.digest, cancellation=self.cancelled
                )
        except Exception as exc:
            error = str(exc)
        self.signals.finished.emit((self, payload, error))


class TableSourceController(QObject):
    """Keep I/O off the GUI thread, and never attach a late load to another tab."""

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self._cache = {}
        self._workers = {}
        self._adoption_requests = set()
        self._serial = 0

    def close(self):
        for worker in self._workers.values():
            worker.cancelled.set()
        self._cache.clear()
        self._adoption_requests.clear()

    @staticmethod
    def _identity(path):
        try:
            stat = path.stat()
            return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            return None

    def _key(self, node, session=None):
        raw = str(node.params.get("dataset_path", "") or "")
        if not raw:
            return None
        path = Path(raw).expanduser()
        if session is None:
            session = self.widget._workflow_tabs.current
        if not path.is_absolute() and session is not None and session.path:
            path = session.path.parent / path
        path = path.absolute()
        return (
            str(path),
            str(node.params.get("dataset_sha256", "")),
            self._identity(path),
        )

    def pending(self):
        """Start any missing source loads and return whether calculation must wait."""
        widget = self.widget
        session = widget._workflow_tabs.current
        self._prune()
        pending = False
        for node in widget.pipeline.nodes.values():
            if node.operation_id != "table_source":
                continue
            key = self._key(node)
            if key is None:
                continue
            context = (session.session_id, node.id, key)
            if not key[1] and context not in self._adoption_requests:
                self._cache[key] = (
                    None,
                    "This workflow has no recorded dataset "
                    "fingerprint. Use Choose measurement dataset "
                    "to confirm the file you want to use.",
                )
                continue
            if key in self._cache:
                payload, _error = self._cache[key]
                if payload is not None and not key[1]:
                    self._adopt(node, key, payload)
                continue
            pending = True
            if context in self._workers:
                continue
            widget._mark_pipeline_dirty(node.id)
            self._start(context, key[0], key[1], context)
        if pending:
            widget.status_label.setText("Loading and verifying measurement dataset…")
        return pending

    def _adopt(self, node, key, payload):
        node.params["dataset_sha256"] = payload.revision_token.file_sha256
        new_key = (key[0], payload.revision_token.file_sha256, key[2])
        self._cache[new_key] = (payload, "")
        self._cache.pop(key, None)
        session = self.widget._workflow_tabs.current
        self._adoption_requests.discard((session.session_id, node.id, key))
        self.widget._sync_current_workflow_tab_state()

    def _prune(self):
        sessions = {s.session_id: s for s in self.widget._workflow_tabs}
        references = {
            key[:2]
            for session in self.widget._workflow_tabs
            for node in session.pipeline.nodes.values()
            if node.operation_id == "table_source"
            and (key := self._key(node, session)) is not None
        }
        for key in tuple(self._cache):
            if key[:2] not in references or self._identity(Path(key[0])) != key[2]:
                self._cache.pop(key, None)
        for worker in self._workers.values():
            if worker.context is None:
                continue
            session_id, node_id, _key = worker.context
            session = sessions.get(session_id)
            node = None if session is None else session.pipeline.nodes.get(node_id)
            current = None if node is None else self._key(node, session)
            if (
                node is None
                or node.operation_id != "table_source"
                or (None if current is None else current[:2])
                != (worker.path, worker.digest)
            ):
                worker.cancelled.set()

    def resolve(self, node):
        from napari_vipp.core.pipeline import SourcePayload

        key = self._key(node)
        if key is None:
            raise ValueError("Choose a saved .vipp-results.json file in Table Source.")
        cached = self._cache.get(key)
        if cached is None:
            return SourcePayload(None, name=Path(key[0]).name)
        payload, error = cached
        if error:
            raise ValueError(f"Table Source: {error}")
        return payload

    def _start(self, key, path, digest, context):
        worker = _Load(key, path, digest, context)
        self._workers[key] = worker
        worker.signals.finished.connect(self._finished)
        QThreadPool.globalInstance().start(worker)

    def open_collection(self, path):
        """Validate before creating a new, unsaved results-workflow tab."""
        reason = self.widget._workflow_tab_switch_block_reason()
        if reason:
            self.widget.status_label.setText(
                f"Wait until {reason} before opening results."
            )
            return
        self._serial += 1
        path = Path(path).expanduser().absolute()
        self._start(("open", self._serial), str(path), "", None)
        self.widget.status_label.setText("Opening measurement dataset…")

    def _finished(self, result):
        worker, payload, error = result
        self._workers.pop(worker.key, None)
        widget = self.widget
        if worker.cancelled.is_set() or widget._closing:
            return
        if worker.context is None:
            if error:
                widget._set_status(
                    f"Could not open measurement dataset: {error}",
                    severity="error",
                    actionable=True,
                )
                return
            self._open_tab(worker.path, payload)
            return
        session_id, node_id, key = worker.context
        # A result from a closed tab or an edited node is not published.
        session = next(
            (s for s in widget._workflow_tabs if s.session_id == session_id), None
        )
        if session is None:
            return
        node = session.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "table_source":
            return
        current = self._key(node, session)
        if current is None or current[:2] != (worker.path, worker.digest):
            return
        if self._identity(Path(worker.path)) != key[2]:
            if widget._workflow_tab_is_active(session_id):
                widget.run_pipeline()
            return
        self._cache[key] = (payload, error)
        if widget._workflow_tab_is_active(session_id):
            if payload is not None and not worker.digest:
                self._adopt(node, key, payload)
            widget._sync_current_workflow_tab_state()
            widget._refresh_selected_parameter_controls()
            widget.run_pipeline()

    def _open_tab(self, path, payload):
        widget = self.widget
        reason = widget._workflow_tab_switch_block_reason()
        if reason:
            widget.status_label.setText(
                f"Dataset saved. Wait until {reason}, then open {Path(path).name}."
            )
            return
        pipeline = PrototypePipeline()
        pipeline.restore_graph([], [])
        node = pipeline.add_node("table_source")
        node.params.update(
            dataset_path=path, dataset_sha256=payload.revision_token.file_sha256
        )
        snapshot = WorkflowHistorySnapshot(
            workflow=WorkflowSnapshot(GraphSnapshot.from_pipeline(pipeline)),
            selected_node_id=node.id,
        )
        title = Path(path).name.removesuffix(".vipp-results.json")
        session = WorkflowTabSession(pipeline, snapshot, title=f"Results · {title}")
        widget._workflow_tabs.add(session, make_current=False)
        key = (path, payload.revision_token.file_sha256, self._identity(Path(path)))
        self._cache[key] = (payload, "")
        index = widget._workflow_tabs.index_of(session.session_id)
        if not widget._activate_workflow_tab(index):
            widget._workflow_tabs.close(index, discard_unsaved=True)
            widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)
            return
        # Never use the dataset path as the workflow's Save destination.
        widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)
        widget.run_pipeline()

    def render_parameters(self, node_id):
        widget = self.widget
        node = widget.pipeline.nodes[node_id]
        widget.parameter_group.setHidden(False)
        label = QLabel(
            str(node.params.get("dataset_path", "")) or "No dataset selected"
        )
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        widget.parameter_form.addRow("Dataset", label)
        button = QPushButton("Choose measurement dataset…")
        button.clicked.connect(lambda: self.choose_file(node_id))
        widget.parameter_form.addRow(button)
        review = QPushButton("Review collection…")
        review.setEnabled(bool(node.params.get("dataset_path")))
        review.clicked.connect(lambda: self.review_collection(node_id))
        widget.parameter_form.addRow(review)
        note = QLabel(
            "Loads a saved collection without reopening images. Connect this table "
            "to Add Metadata Columns, Select Table Columns or Summarize Measurements. "
            "The workflow saves a file reference; keep the dataset alongside it."
        )
        note.setWordWrap(True)
        note.setMinimumWidth(0)
        widget.parameter_form.addRow(note)
        digest = str(node.params.get("dataset_sha256", ""))
        if digest:
            recorded = QLabel(f"Recorded file fingerprint: {digest[:12]}…")
            recorded.setToolTip(digest)
            recorded.setWordWrap(True)
            widget.parameter_form.addRow(recorded)

    def choose_file(self, node_id):
        widget = self.widget
        node = widget.pipeline.nodes.get(node_id)
        if node is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            widget,
            "Choose measurement dataset",
            str(node.params.get("dataset_path", "")),
            "VIPP measurement dataset (*.vipp-results.json)",
        )
        if not path:
            return
        before = widget._current_history_snapshot()
        node.params.update(dataset_path=str(Path(path).absolute()), dataset_sha256="")
        key = self._key(node)
        self._cache.pop(key, None)
        session = widget._workflow_tabs.current
        self._adoption_requests.add((session.session_id, node.id, key))
        widget._mark_pipeline_dirty(node_id)
        widget._push_undo_if_changed(before)
        widget._refresh_selected_parameter_controls()
        widget.run_pipeline()

    def review_collection(self, node_id):
        from napari_vipp.ui.measurement_collection import (
            MeasurementCollectionReviewDialog,
        )

        node = self.widget.pipeline.nodes.get(node_id)
        if node is None or not node.params.get("dataset_path"):
            return
        dialog = MeasurementCollectionReviewDialog(
            node.params["dataset_path"],
            expected_sha256=node.params.get("dataset_sha256", ""),
            parent=self.widget,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()
