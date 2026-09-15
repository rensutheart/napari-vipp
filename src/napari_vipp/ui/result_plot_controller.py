"""Workflow-bound plotting panels; Qt ownership stays outside the core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, QTimer

from napari_vipp.core.pipeline import (
    EXECUTION_ERROR,
    EXECUTION_READY,
    MANUAL_RUN_SKIP,
)
from napari_vipp.core.result_plots import PlotRecipe, is_plot_data
from napari_vipp.core.tables import is_table_data


class ResultPlotController(QObject):
    """Keep each detached plot bound to its originating workflow and node."""

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self.panels = {}
        self._keys = {}
        self._thumbnails = {}
        self._closed = False
        self._debounce_connected = False

    def close(self):
        self._closed = True
        for panel in self.panels.values():
            panel.close_plot()
            panel.deleteLater()
        self.panels.clear()
        self._keys.clear()
        self._thumbnails.clear()

    def _context(self, node_id):
        return self.widget._workflow_tabs.current.session_id, node_id

    def render_parameters(self, node_id):
        from napari_vipp.ui.result_plots import PlotResultsPanel

        if not self._debounce_connected:
            # The widget creates its debounce timer after this controller. An
            # edit may finish without dispatching (missing inputs, for example),
            # so refresh after the timeout as well as at worker state changes.
            self.widget._debounce_timer.timeout.connect(self.refresh)
            self._debounce_connected = True
        key = self._context(node_id)
        panel = self.panels.get(key)
        if panel is None:
            panel = PlotResultsPanel(parent=self.widget)
            panel.setProperty("vippPersistentPlotPanel", True)
            panel.params_changed.connect(
                lambda values, context=key: self._commit(context, values)
            )
            panel.layout_changed.connect(
                lambda: QTimer.singleShot(0, self.widget._sync_parameter_form_height)
            )
            self.panels[key] = panel
        self.widget.parameter_group.show()
        self.widget.parameter_form.addRow(panel)
        panel.show()
        self._refresh_panel(key, panel, force=True)

    def _commit(self, context, values):
        widget = self.widget
        if context[0] != widget._workflow_tabs.current.session_id:
            self._refresh_panel(context, self.panels[context], force=True)
            return
        node_id = context[1]
        node = widget.pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "plot_results":
            return
        try:
            # Validate the complete edit before touching the graph or undo stack.
            PlotRecipe.from_params(values)
            names = {p.name for p in widget.pipeline.node_parameter_specs(node_id)}
            changed = {
                name: value
                for name, value in values.items()
                if name in names and node.params.get(name) != value
            }
            if not changed:
                return
            widget._record_parameter_undo(node_id, "plot_recipe")
            for name, value in changed.items():
                widget.pipeline.set_param(node_id, name, value)
            widget._mark_pipeline_dirty(node_id)
            widget._sync_current_workflow_tab_state()
            # The graph retains upstream measurements. Only the plot frontier
            # and its consumers are invalidated by a recipe edit.
            widget._debounce_timer.start()
            self._refresh_panel(context, self.panels[context], force=True)
        except (ValueError, TypeError) as exc:
            widget.status_label.setText(f"Plot settings: {exc}")
            self._refresh_panel(context, self.panels[context], force=True)

    def _busy_state(self, node_id, ancestors, execution):
        """Describe real scheduled work, never infer progress from stale alone."""
        widget = self.widget
        if widget._closing or widget._compute_runtime_quarantined_reason:
            return False, ""
        run_id = widget._active_pipeline_run_id
        if (
            run_id is not None
            and widget._pipeline_user_cancel_requested_run_id == run_id
        ):
            return False, ""

        pending = widget._pending_dirty_node_ids & set(widget.pipeline.nodes)
        queued = widget._debounce_timer.isActive() or widget._pipeline_run_pending
        if queued and pending & ancestors:
            # Use the same manual-frontier rules as execution: a queued edit
            # behind an unrequested manual measurement is not a queued plot.
            manual_ids = (
                widget.pipeline.auto_recalculate_node_ids()
                | widget._pending_manual_node_ids
            )
            isolated = widget._isolated_tuning_node_id
            targets = None
            if isolated is not None and isolated in pending:
                pending = {isolated}
                targets = {isolated}
                manual_ids = {isolated}
            plan = widget.pipeline.plan_execution(
                pending,
                manual_mode=MANUAL_RUN_SKIP,
                manual_node_ids=manual_ids,
                target_node_ids=targets,
            )
            if node_id in plan.runnable_node_ids:
                return True, "Plot update queued…"

        if execution == EXECUTION_ERROR or run_id is None:
            return False, ""
        cancel = widget._pipeline_cancel_events.get(run_id)
        if cancel is not None and cancel.is_set():
            return False, ""
        context = widget._pipeline_run_context.get(run_id, ())
        runnable = context[6] if len(context) > 6 else ()
        if node_id not in runnable:
            return False, ""
        accepted = widget._background_execution_state_overrides.get(node_id)
        if accepted is not None and accepted[:2] == (run_id, EXECUTION_READY):
            # This plot has finished, even if an unrelated branch is still busy.
            return False, ""
        if widget._active_pipeline_node_id == node_id:
            return True, "Updating plot…"
        return True, "Preparing plot measurements…"

    def _refresh_panel(self, context, panel, *, force=False):
        widget = self.widget
        current = widget._workflow_tabs.current
        if current is None or current.session_id != context[0]:
            panel.set_stale("Return to this plot's workflow tab to edit or export it.")
            self._keys.pop(context, None)
            return
        node = widget.pipeline.nodes.get(context[1])
        if node is None:
            panel.set_stale("This plotting node has been removed.")
            return
        data, _state, _port = widget._node_display_payload(node.id)
        result = data if is_plot_data(data) else None
        table = widget.pipeline.input_data_for_node(node.id)
        if not is_table_data(table):
            table = result.source_table if result is not None else None
        execution, message = widget._node_execution_ui_state(node.id)
        ancestors = widget.pipeline.ancestors_inclusive({node.id})
        dirty = bool(ancestors & widget._pending_dirty_node_ids)
        busy, busy_message = self._busy_state(node.id, ancestors, execution)
        stale = execution != EXECUTION_READY or dirty
        stamp = (
            id(result),
            id(table),
            json.dumps(node.params, sort_keys=True),
            stale,
            execution,
            message,
            busy,
            busy_message,
        )
        if force or self._keys.get(context) != stamp:
            self._keys[context] = stamp
            protected = [current.path] if current.path is not None else []
            for source_id in ancestors:
                source = widget.pipeline.nodes.get(source_id)
                if source is None:
                    continue
                raw = source.params.get(
                    "dataset_path"
                    if source.operation_id == "table_source"
                    else "file_path"
                )
                if raw:
                    path = Path(str(raw)).expanduser()
                    if not path.is_absolute() and current.path is not None:
                        path = current.path.parent / path
                    protected.append(path)
            panel.set_state(
                table=table,
                params=node.params,
                result=result,
                stale=stale,
                failed=execution == EXECUTION_ERROR,
                busy=busy,
                busy_message=busy_message,
                protected_paths=protected,
                message=message
                or (
                    "Settings or inputs changed; waiting for the updated plot."
                    if stale and result is not None
                    else ""
                ),
            )
            if node.id == widget._selected_node_id:
                QTimer.singleShot(0, widget._sync_parameter_form_height)
            if dirty and not busy and execution != EXECUTION_ERROR:
                # Generic parameter controls start their timer just after dirty
                # state is published. One queued refresh observes that start;
                # the unchanged stamp prevents repeated polling while idle.
                QTimer.singleShot(0, self.refresh)

    def refresh(self):
        if self._closed or not self.panels:
            return
        sessions = {s.session_id: s for s in self.widget._workflow_tabs}
        for context, panel in tuple(self.panels.items()):
            session = sessions.get(context[0])
            node = None if session is None else session.pipeline.nodes.get(context[1])
            if node is None or node.operation_id != "plot_results":
                panel.close_plot()
                panel.hide()
                panel.deleteLater()
                self.panels.pop(context, None)
                self._keys.pop(context, None)
            else:
                self._refresh_panel(context, panel)

    def open_plot(self, node_id):
        context = self._context(node_id)
        if context not in self.panels:
            self.widget._select_node(node_id)
        panel = self.panels.get(context)
        if panel is not None:
            self._refresh_panel(context, panel, force=True)
            panel.open_plot()

    def thumbnail(self, node_id, result):
        """A detached figure preview, never an image-shaped scientific result."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from napari_vipp.core.plot_rendering import build_plot_figure

        context = self._context(node_id)
        cached = self._thumbnails.get(context)
        if cached is not None and cached[0] is result:
            return cached[1]
        figure = build_plot_figure(
            result,
            size_inches=(4.8, 3.0),
            dpi=80,
            publication=True,
            display_only=True,
            compact=True,
        )
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        # Graph cards own the Qt conversion. Keep the same detached RGB-array
        # contract as every other graph thumbnail, including restored tabs.
        thumbnail = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        figure.clear()
        # Keep only the current workflow's current plot results, bounded by nodes.
        self._thumbnails = {
            key: value
            for key, value in self._thumbnails.items()
            if key[0] == context[0] and key[1] in self.widget.pipeline.nodes
        }
        self._thumbnails[context] = (result, thumbnail)
        return thumbnail
