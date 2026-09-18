"""Workflow-bound Statistics editing and accepted-result presentation."""

from __future__ import annotations

import json

from qtpy.QtCore import QObject, QTimer

from napari_vipp.core.pipeline import EXECUTION_ERROR, EXECUTION_READY
from napari_vipp.core.statistics import StatisticsRecipe
from napari_vipp.core.tables import is_table_data


class StatisticsController(QObject):
    """Retain panels by workflow; all calculations remain with the executor."""

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self.panels = {}
        self._keys = {}
        self._revisions = {}
        self._closed = False

    def _context(self, node_id):
        return self.widget._workflow_tabs.current.session_id, node_id

    def render_parameters(self, node_id):
        panel = self.panel_for(node_id)
        self.widget.parameter_group.show()
        self.widget.parameter_form.addRow(panel)
        panel.show()

    def panel_for(self, node_id):
        """Bind the shared editor without changing the inspector selection."""
        from napari_vipp.ui.statistics import StatisticsPanel

        context = self._context(node_id)
        panel = self.panels.get(context)
        if panel is None:
            panel = StatisticsPanel(self.widget)
            panel.setProperty("vippPersistentStatisticsPanel", True)
            panel.params_changed.connect(
                lambda values, key=context: self._commit(key, values)
            )
            panel.upgrade_requested.connect(lambda key=context: self._upgrade(key))
            panel.layout_changed.connect(
                lambda: QTimer.singleShot(0, self.widget._sync_parameter_form_height)
            )
            self.panels[context] = panel
            panel.hide()
        self._refresh_panel(context, panel, force=True)
        return panel

    def _revision(self, node):
        return (
            id(self.widget.pipeline),
            id(node),
            json.dumps(node.params, sort_keys=True),
            id(self.widget.pipeline.input_data_for_node(node.id)),
        )

    def _editable_node(self, context):
        current = self.widget._workflow_tabs.current
        if self._closed or current is None or current.session_id != context[0]:
            return None
        node = self.widget.pipeline.nodes.get(context[1])
        if node is None or node.operation_id != "summarize_measurements":
            return None
        if self.widget.pipeline.node_is_bypassed(node.id):
            return None
        if self._revisions.get(context) != self._revision(node):
            panel = self.panels.get(context)
            if panel is not None:
                self._refresh_panel(context, panel, force=True)
            return None
        return node

    def _commit(self, context, values, *, upgrading=False):
        node = self._editable_node(context)
        if node is None:
            return
        widget = self.widget
        try:
            StatisticsRecipe.from_params(values)
            # A generic edit cannot opt an old workflow into changed mathematics.
            old_version = node.params.get("summary_version", 1)
            if (
                values.get("summary_version", old_version) != old_version
                and not upgrading
            ):
                raise ValueError(
                    "Use the explicit upgrade button to change the summary version."
                )
            names = {
                spec.name for spec in widget.pipeline.node_parameter_specs(node.id)
            }
            changed = {
                name: value
                for name, value in values.items()
                if name in names and node.params.get(name) != value
            }
            if not changed:
                return
            # One completed recipe change, including an upgrade, is one undo step.
            widget._finish_parameter_history_group(preserve_isolated_tuning=True)
            widget._record_parameter_undo(node.id, "statistics_recipe")
            for name, value in changed.items():
                widget.pipeline.set_param(node.id, name, value)
            widget._history.finish_group()
            widget._mark_pipeline_dirty(node.id)
            widget._sync_current_workflow_tab_state()
            widget._debounce_timer.start()
            self._refresh_panel(context, self.panels[context], force=True)
        except (TypeError, ValueError) as exc:
            widget.status_label.setText(f"Statistics settings: {exc}")
            self._refresh_panel(context, self.panels[context], force=True)

    def _upgrade(self, context):
        node = self._editable_node(context)
        if node is None or node.params.get("summary_version", 1) != 1:
            return
        values = StatisticsRecipe().to_params()
        for key in ("group_by", "value_columns", "statistics"):
            if key in node.params:
                values[key] = node.params[key]
        self._commit(context, values, upgrading=True)

    def _refresh_panel(self, context, panel, *, force=False):
        widget = self.widget
        current = widget._workflow_tabs.current
        if current is None or current.session_id != context[0]:
            panel.set_unavailable(
                "Return to this Statistics node's workflow tab to edit it."
            )
            self._keys.pop(context, None)
            self._revisions.pop(context, None)
            return
        node = widget.pipeline.nodes.get(context[1])
        if node is None:
            panel.set_unavailable("This Statistics node has been removed.")
            return
        table = widget.pipeline.input_data_for_node(node.id)
        table = table if is_table_data(table) else None
        result, _state, _port = widget._node_display_payload(node.id)
        result = result if is_table_data(result) else None
        execution, message = widget._node_execution_ui_state(node.id)
        ancestors = widget.pipeline.ancestors_inclusive({node.id})
        stale = execution != EXECUTION_READY or bool(
            ancestors & widget._pending_dirty_node_ids
        )
        revision = self._revision(node)
        stamp = (revision, id(result), stale, execution, message)
        if force or self._keys.get(context) != stamp:
            self._keys[context] = stamp
            self._revisions[context] = revision
            panel.set_state(
                table=table,
                params=node.params,
                result=result,
                stale=stale,
                failed=execution == EXECUTION_ERROR,
                message=message,
            )
            if node.id == widget._selected_node_id:
                QTimer.singleShot(0, widget._sync_parameter_form_height)

    def refresh(self):
        if self._closed or not self.panels:
            return
        sessions = {
            session.session_id: session for session in self.widget._workflow_tabs
        }
        for context, panel in tuple(self.panels.items()):
            session = sessions.get(context[0])
            node = None if session is None else session.pipeline.nodes.get(context[1])
            if node is None or node.operation_id != "summarize_measurements":
                panel.hide()
                panel.deleteLater()
                self.panels.pop(context, None)
                self._keys.pop(context, None)
                self._revisions.pop(context, None)
            else:
                self._refresh_panel(context, panel)

    def close(self):
        self._closed = True
        for panel in self.panels.values():
            panel.hide()
            panel.deleteLater()
        self.panels.clear()
        self._keys.clear()
        self._revisions.clear()
