"""Bind the Results Workspace to ordinary, undoable workflow nodes.

Window selections are presentation state. Scientific recipes and plot sources
live only in node parameters and graph connections, shared by every editor.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from qtpy.QtCore import QObject, QPointF, Qt

from napari_vipp.core.pipeline import EXECUTION_ERROR, EXECUTION_READY
from napari_vipp.core.result_plots import is_summary_table
from napari_vipp.core.tables import is_table_data


@dataclass
class _Workspace:
    dialog: object
    root: tuple = ()
    summary_id: str = ""
    plot_id: str = ""
    plot_scope_id: str = ""
    context_initialized: bool = False
    stamps: dict = field(default_factory=dict)
    revisions: dict = field(default_factory=dict)


class ResultsWorkspaceController(QObject):
    """Navigate exact table outputs and their directly connected results."""

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget
        self.windows = {}
        self._refreshing = False
        self._closed = False

    def _input(self, node_id):
        connections = self.widget.pipeline._input_connections(node_id)
        connection = next((c for c in connections if c.target_port == 0), None)
        return (
            (connection.source_id, int(connection.source_port))
            if connection is not None
            else None
        )

    def _origin(self, node_id, source_port=None):
        pipeline = self.widget.pipeline
        node = pipeline.nodes.get(node_id)
        if node is None:
            return None
        summary_id, plot_id, tab = "", "", "data"
        if node.operation_id == "plot_results":
            plot_id, tab = node.id, "plots"
            source = self._input(node.id)
            if source is None:
                return None
            node_id, source_port = source
            node = pipeline.nodes[node_id]
        if node.operation_id == "summarize_measurements":
            summary_id = node.id
            tab = "plots" if plot_id else "summary"
            source = self._input(node.id)
            if source is None:
                return None
            node_id, source_port = source
        ports = pipeline.output_ports(node_id)
        table_ports = [i for i, port in enumerate(ports) if port.output_type == "table"]
        if source_port is None:
            _data, _state, display_port = self.widget._node_display_payload(node_id)
            source_port = (
                display_port
                if display_port in table_ports
                else next(iter(table_ports), None)
            )
        if source_port not in table_ports:
            return None
        return (node_id, int(source_port)), summary_id, plot_id, tab

    def can_open(self, node_id):
        return bool(node_id and self._origin(node_id) is not None)

    def open_node(self, node_id, source_port=None):
        from napari_vipp.ui.results_workspace import ResultsWorkspaceDialog

        if self._closed:
            return None
        origin = self._origin(node_id, source_port)
        if origin is None:
            self.widget.status_label.setText(
                "Select a table-producing node, or connect a table to "
                "Statistics / Plot Results first."
            )
            return None
        root, summary_id, plot_id, tab = origin
        key = (self.widget._workflow_tabs.current.session_id, *root)
        # A window can browse another data branch without changing the graph.
        # Keep its original key as an event-owner token, not as the active input.
        match = next(
            (
                (k, e)
                for k, e in self.windows.items()
                if k[0] == key[0] and e.root == root
            ),
            None,
        )
        entry = None
        if match is not None:
            key, entry = match
        if entry is None:
            if key in self.windows:
                key = (*key, object())
            dialog = ResultsWorkspaceDialog(self.widget)
            dialog.setAttribute(Qt.WA_DeleteOnClose, True)
            entry = self.windows[key] = _Workspace(dialog, root=root)
            dialog.finished.connect(
                lambda _result, k=key, owner=entry: self._forget(k, owner)
            )
            dialog.destroyed.connect(
                lambda _obj=None, k=key, owner=entry: self._forget(k, owner)
            )

            def dispatch(method, *args):
                # Reject late events from a closed editor even if another
                # workspace has since reopened this same workflow/table key.
                if self.windows.get(key) is entry:
                    return method(key, *args)
                return None

            dialog.summary_selected.connect(
                lambda value: dispatch(self._select, "summary", value)
            )
            dialog.data_selected.connect(
                lambda value: dispatch(self._select_data, value)
            )
            dialog.plot_scope_selected.connect(
                lambda value: dispatch(self._select_scope, value)
            )
            dialog.plot_selected.connect(
                lambda value: dispatch(self._select, "plot", value)
            )
            dialog.add_summary_requested.connect(
                lambda: dispatch(self._add, "summarize_measurements")
            )
            dialog.add_plot_requested.connect(
                lambda source: dispatch(self._add, "plot_results", source)
            )
            dialog.summary_params_changed.connect(
                lambda values: dispatch(self._commit, "summary", values)
            )
            dialog.summary_upgrade_requested.connect(lambda: dispatch(self._upgrade))
            dialog.plot_params_changed.connect(
                lambda values: dispatch(self._commit, "plot", values)
            )
            dialog.plot_source_changed.connect(
                lambda source: dispatch(self._rewire, source)
            )
            dialog.recalculate_requested.connect(
                lambda target: dispatch(self._calculate, target)
            )
            dialog.show_node_requested.connect(
                lambda target: dispatch(self._show_node, target)
            )
            dialog.export_completed.connect(
                lambda path: self.widget.status_label.setText(f"Exported {path}")
            )
            dialog.tabs.currentChanged.connect(
                lambda _index: dispatch(self._tab_changed)
            )
        if summary_id:
            self._set_scope(entry, summary_id)
        if plot_id:
            self._set_scope(entry, self._input(plot_id)[0])
            entry.plot_id = plot_id
        self.refresh()
        entry.dialog.show_tab(tab)
        entry.dialog.show()
        entry.dialog.raise_()
        entry.dialog.activateWindow()
        return entry.dialog

    def _forget(self, key, owner):
        # A close can be followed immediately by reopen before Qt delivers the
        # old window's deferred deletion. Never remove its replacement window.
        if self.windows.get(key) is owner:
            self.windows.pop(key, None)

    def _active(self, key):
        current = self.widget._workflow_tabs.current
        root = self._root(key)
        return (
            not self._closed
            and key in self.windows
            and current is not None
            and key[0] == current.session_id
            and root[0] in self.widget.pipeline.nodes
            and root[1] < len(self.widget.pipeline.output_ports(root[0]))
            and self.widget.pipeline.output_ports(root[0])[root[1]].output_type
            == "table"
        )

    def _root(self, key):
        entry = self.windows.get(key)
        return entry.root if entry is not None else key[1:3]

    def _data_sources(self):
        """Exact analysis inputs, including filtered/merged and multi-port tables.

        Never flatten a transform or merge to an earlier ancestor: that would
        misrepresent the rows that the attached summaries actually calculate.
        """
        roots = dict.fromkeys(
            origin[0]
            for node in self.widget.pipeline.nodes.values()
            for port, output in enumerate(self.widget.pipeline.output_ports(node.id))
            if output.output_type == "table"
            and (origin := self._origin(node.id, port)) is not None
        )
        return list(roots)

    def _choices(self, key):
        root = self._root(key)
        nodes = self.widget.pipeline.nodes.values()
        summaries = [
            n.id
            for n in nodes
            if n.operation_id == "summarize_measurements" and self._input(n.id) == root
        ]
        sources = {root, *((node_id, 0) for node_id in summaries)}
        plots = [
            n.id
            for n in nodes
            if n.operation_id == "plot_results" and self._input(n.id) in sources
        ]
        return summaries, plots

    def _scope_plots(self, key):
        entry = self.windows[key]
        root = self._root(key)
        source = (
            entry.plot_scope_id or root[0],
            root[1] if entry.plot_scope_id in ("", root[0]) else 0,
        )
        return [n for n in self._choices(key)[1] if self._input(n) == source]

    @staticmethod
    def _set_scope(entry, source_id):
        if entry.plot_scope_id != source_id:
            entry.plot_scope_id = source_id
            entry.plot_id = ""
        # The middle selector is the one authority for every tab. An empty
        # summary is an intentional direct-input path, never an auto default.
        entry.summary_id = "" if source_id == entry.root[0] else source_id
        entry.context_initialized = True

    def _select_data(self, key, root):
        current = self.widget._workflow_tabs.current
        if key not in self.windows or current is None or key[0] != current.session_id:
            return
        root = tuple(root) if root else ()
        if root not in self._data_sources() or root == self._root(key):
            return
        entry = self.windows[key]
        entry.root = root
        entry.summary_id = entry.plot_id = ""
        entry.plot_scope_id = ""
        entry.context_initialized = False
        entry.stamps.clear()
        entry.revisions.clear()
        self.refresh()

    def _select_scope(self, key, source_id):
        if not self._active(key):
            return
        if source_id not in {self._root(key)[0], *self._choices(key)[0]}:
            return
        entry = self.windows[key]
        self._set_scope(entry, source_id)
        self.refresh()

    def _tab_changed(self, key):
        # Tabs change the editor, not the data → summary → plot selection.
        self.refresh()

    def _revision(self, node_id):
        pipeline = self.widget.pipeline
        node = pipeline.nodes.get(node_id)
        if node is None:
            return None
        return (
            id(pipeline),
            id(node),
            json.dumps(node.params, sort_keys=True),
            self._input(node_id),
            id(pipeline.input_data_for_node(node_id)),
            pipeline.node_is_bypassed(node_id),
        )

    def _editable(self, key, kind):
        if not self._active(key):
            return None
        entry = self.windows[key]
        node_id = getattr(entry, f"{kind}_id")
        choices = self._choices(key)[0] if kind == "summary" else self._scope_plots(key)
        if not node_id or node_id not in choices:
            return None
        if entry.revisions.get(kind) != self._revision(node_id):
            self.refresh()
            return None
        if self.widget.pipeline.node_is_bypassed(node_id):
            return None
        return node_id

    def _select(self, key, kind, node_id):
        if not self._active(key):
            return
        choices = self._choices(key)[0] if kind == "summary" else self._scope_plots(key)
        if node_id in choices or not node_id:
            entry = self.windows[key]
            if kind == "summary":
                self._set_scope(entry, node_id or self._root(key)[0])
            else:
                entry.plot_id = node_id
            self.refresh()

    def _commit(self, key, kind, values):
        node_id = self._editable(key, kind)
        if node_id is None:
            return
        controller = (
            self.widget._statistics if kind == "summary" else self.widget._result_plots
        )
        controller.panel_for(node_id)
        controller._commit((key[0], node_id), values)
        self.refresh()

    def _upgrade(self, key):
        node_id = self._editable(key, "summary")
        if node_id is not None:
            self.widget._statistics.panel_for(node_id)
            self.widget._statistics._upgrade((key[0], node_id))
            self.refresh()

    def _source_port(self, key, source_id):
        if source_id == self._root(key)[0]:
            return self._root(key)[1]
        if source_id in self._choices(key)[0]:
            return 0
        raise ValueError(
            "Choose the connected measurements or one of their summary tables."
        )

    def _source_is_summary(self, source, *, source_payloads=None):
        """Classify the effective data, including a bypassed Statistics node."""
        pipeline = self.widget.pipeline
        if source is None:
            return False
        source_node = pipeline.nodes[source[0]]
        while pipeline.node_is_bypassed(source_node.id):
            source = self._input(source_node.id)
            if source is None:
                return False
            source_node = pipeline.nodes[source[0]]
        payload = (source_payloads or {}).get(source[0])
        table = payload.data if payload is not None else self._output(*source)
        # A re-enabled Statistics node may still retain a bypassed raw-table
        # cache until its next calculation. Its new output will be a summary.
        return source_node.operation_id == "summarize_measurements" or is_summary_table(
            table
        )

    def plot_setup_message(self, node_id, *, source_payloads=None):
        """Describe an unfinished choice, not a failed scientific calculation.

        Summary columns include means, counts and provenance. Choosing one on
        the author's behalf is unsafe. Only untouched automatic axis fields
        are a setup state: named missing columns and all other invalid requests
        still reach the normal calculation/validation path.
        """
        pipeline = self.widget.pipeline
        node = pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "plot_results":
            return ""
        if pipeline.node_is_bypassed(node_id):
            return ""
        if not self._source_is_summary(
            self._input(node_id), source_payloads=source_payloads
        ):
            return ""
        missing_y = node.params.get("y_column", "auto") in ("", "auto")
        missing_x = node.params.get("plot_type") == "Scatter" and node.params.get(
            "x_column", "auto"
        ) in ("", "auto")
        if missing_y and missing_x:
            return (
                "Choose the X and Y measurements on the left. Your summary has "
                "several kinds of values, so you choose which two to compare. "
                "The plot will update when both are selected."
            )
        if missing_y:
            return (
                "Choose a measurement on the left, such as a mean, median or "
                "count. The plot will update when you make your selection."
            )
        if missing_x:
            return (
                "Choose the X measurement on the left. The plot will update "
                "when both X and Y measurements are selected."
            )
        return ""

    def pending_plot_setup_node_ids(self, *, source_payloads=None):
        """Unconfigured plots remain drafts during interactive automatic runs.

        Derive this from graph parameters, even when the window is closed or
        after undo/reload. No executable recipe or separate draft is hidden in
        the editor. Headless and batch execution retain strict validation.
        """
        return {
            node.id
            for node in self.widget.pipeline.nodes.values()
            if self.plot_setup_message(node.id, source_payloads=source_payloads)
        }

    def _add(self, key, operation_id, source_id=""):
        if not self._active(key):
            return None
        widget = self.widget
        source_id = source_id or self._root(key)[0]
        try:
            source_port = self._source_port(key, source_id)
        except ValueError as exc:
            widget.status_label.setText(str(exc))
            self.refresh()
            return None
        widget._finish_parameter_history_group()
        before = widget._current_history_snapshot()
        try:
            # A branch, not terminal-node append: measurements can feed several
            # independent summaries and figures without stealing connections.
            node = widget.pipeline.add_node(operation_id)
            widget.graph_view.add_node(
                node, widget.graph_view.suggest_append_position(source_id)
            )
            widget._sync_node_input_ports(node.id)
            widget._sync_node_output_ports(node.id)
            self._space_new_node(node.id)
            result = widget.pipeline.connect(
                source_id, node.id, target_port=0, source_port=source_port
            )
            if not result.success:
                raise ValueError(result.message)
            widget._apply_connection_result_to_graph(result)
            widget._sync_input_node_subtitle(node.id)
            kind = "summary" if operation_id == "summarize_measurements" else "plot"
            entry = self.windows[key]
            if kind == "summary":
                self._set_scope(entry, node.id)
            else:
                self._set_scope(entry, source_id)
            setattr(entry, f"{kind}_id", node.id)
            widget._sync_pin_ui()
            widget._refresh_graph_search_matches(reset_index=True)
            widget._mark_pipeline_dirty(node.id)
            widget._push_undo_if_changed(before)
            if not self.plot_setup_message(node.id):
                widget._debounce_timer.start()
            self.refresh()
            self.windows[key].dialog.show_tab(
                "summary" if kind == "summary" else "plots"
            )
            widget.status_label.setText(
                "Plot added. Choose a measurement in Results Workspace to begin."
                if self.plot_setup_message(node.id)
                else f"Added '{node.title}' to the workflow."
            )
            return node
        except Exception as exc:
            widget._restore_history_snapshot(before)
            widget.status_label.setText(f"Cannot add results node: {exc}")
            self.refresh()
            return None

    def _space_new_node(self, node_id):
        """Keep additional results branches clear of existing authored nodes."""
        graph = self.widget.graph_view
        rect = graph.node_scene_rect(node_id)
        if rect is None:
            return
        original_y = rect.top()
        blockers = [
            other
            for other_id in self.widget.pipeline.nodes
            if other_id != node_id
            and (other := graph.node_scene_rect(other_id)) is not None
        ]
        # Each move clears at least one blocker; avoid moving the user's graph.
        for _ in range(len(blockers) + 1):
            hits = [
                other
                for other in blockers
                if rect.adjusted(-24, -24, 24, 24).intersects(other)
            ]
            if not hits:
                break
            rect.moveTop(max(other.bottom() for other in hits) + 48)
        graph.move_nodes_by({node_id}, QPointF(0, rect.top() - original_y))

    def _rewire(self, key, source_id):
        node_id = self._editable(key, "plot")
        if node_id is None:
            return
        widget = self.widget
        try:
            source_port = self._source_port(key, source_id)
        except ValueError:
            self.refresh()
            return
        if self._input(node_id) == (source_id, source_port):
            return
        widget._finish_parameter_history_group()
        before = widget._current_history_snapshot()
        try:
            result = widget.pipeline.connect(
                source_id, node_id, target_port=0, source_port=source_port
            )
            if not result.success:
                raise ValueError(result.message)
            widget._apply_connection_result_to_graph(result)
            if source_id != self._root(key)[0] or is_summary_table(
                self._output(source_id, source_port)
            ):
                # Explicitly selecting summary data also selects summary rows.
                # Keep measurement/group names: unavailable fields must fail,
                # never become another statistic without the author's choice.
                widget.pipeline.set_param(node_id, "point_unit", "Objects")
            widget._sync_node_output_ports(node_id)
            widget._mark_pipeline_dirty(node_id)
            widget._push_undo_if_changed(before)
            if not self.plot_setup_message(node_id):
                widget._debounce_timer.start()
            self.refresh()
        except Exception as exc:
            widget._restore_history_snapshot(before)
            widget.status_label.setText(f"Cannot change plot input: {exc}")
            self.refresh()

    def _calculate(self, key, node_id):
        if self._active(key) and node_id in {
            self._root(key)[0],
            *sum(self._choices(key), []),
        }:
            if self.plot_setup_message(node_id):
                self.refresh()
                return
            if self.widget.pipeline.is_manual_node(node_id):
                self.widget._calculate_node(node_id)
            else:
                self.widget._mark_pipeline_dirty(node_id)
                self.widget._debounce_timer.start()

    def _show_node(self, key, node_id):
        if self._active(key) and node_id in {
            self._root(key)[0],
            *sum(self._choices(key), []),
        }:
            # On Windows an owned tool window can remain above its parent even
            # after raise_(). Hide, rather than close, so the graph is visible
            # and opening Results Workspace restores the same editor choices.
            self.windows[key].dialog.hide()
            self.widget.graph_view.select_node(node_id)
            self.widget._select_node(node_id)
            self.widget.graph_view.focus_node(node_id)
            window = self.widget.window()
            if window.isMinimized():
                window.showNormal()
            window.raise_()
            window.activateWindow()

    def _output(self, node_id, port):
        result = self.widget._background_node_result_override(node_id)
        if result is not None:
            outputs = result.node_outputs
            value = (
                outputs[port]
                if outputs and port < len(outputs)
                else result.output
                if port == 0
                else None
            )
        else:
            value = self.widget.pipeline._resolved_output(node_id, port)
        return value if is_table_data(value) else None

    def _state(self, node_id):
        widget = self.widget
        execution, message = widget._node_execution_ui_state(node_id)
        ancestors = widget.pipeline.ancestors_inclusive({node_id})
        stale = execution != EXECUTION_READY or bool(
            ancestors & widget._pending_dirty_node_ids
        )
        busy, busy_message = widget._result_plots._busy_state(
            node_id, ancestors, execution
        )
        return stale, execution == EXECUTION_ERROR, message, busy, busy_message

    def refresh(self, *_args):
        if self._closed or self._refreshing or not self.windows:
            return
        self._refreshing = True
        try:
            sessions = {session.session_id for session in self.widget._workflow_tabs}
            for key, entry in tuple(self.windows.items()):
                if key[0] not in sessions:
                    entry.dialog.close()
                    self.windows.pop(key, None)
                    continue
                self._refresh(key, entry)
        finally:
            self._refreshing = False

    def _refresh(self, key, entry):
        dialog, widget = entry.dialog, self.widget
        if not self._active(key):
            dialog.set_available(
                False,
                "Return to the source workflow tab. If its source node was "
                "removed, undo that change or open another table.",
            )
            dialog.set_busy(False)
            entry.stamps.clear()
            entry.revisions.clear()
            return
        dialog.set_available(True)
        root = self._root(key)
        summaries, all_plots = self._choices(key)
        if not entry.context_initialized:
            self._set_scope(entry, next(iter(summaries), root[0]))
        # Follow an explicit input edit (including undo/redo), but never follow
        # an unrelated sibling merely because the selected node was removed.
        source = self._input(entry.plot_id) if entry.plot_id else None
        previous_source = entry.stamps.get("plot_connection")
        if entry.plot_id in all_plots and source != previous_source:
            plot_id = entry.plot_id
            self._set_scope(entry, source[0])
            entry.plot_id = plot_id
        plots = self._scope_plots(key)
        if not entry.plot_id:
            entry.plot_id = next(iter(plots), "")
        source = self._input(entry.plot_id) if entry.plot_id else None
        entry.stamps["plot_connection"] = source
        table = self._output(*root)
        root_kind = (
            "Input summary table"
            if is_summary_table(table)
            or widget.pipeline.nodes[root[0]].operation_id == "summarize_measurements"
            else "Original measurements"
        )
        data_sources = self._data_sources()
        # Number duplicate names across the workflow, not across a filtered
        # dropdown. Statistics 2 stays Statistics 2 when another branch is shown.
        titles = {n: widget._node_title(n) for n in widget.pipeline.nodes}
        duplicates = Counter(titles.values())
        numbered = Counter()
        reserved = set(titles.values())
        friendly_titles = {}
        for node_id, title in titles.items():
            if duplicates[title] > 1:
                numbered[title] += 1
                label = f"{title} {numbered[title]}"
                while label in reserved:
                    numbered[title] += 1
                    label = f"{title} {numbered[title]}"
                reserved.add(label)
                friendly_titles[node_id] = label
            else:
                friendly_titles[node_id] = title
        titles = friendly_titles

        def data_title(output):
            node_id, port = output
            label = titles[node_id]
            ports = widget.pipeline.output_ports(node_id)
            if len(ports) > 1:
                label += f" · {ports[port].label}"
            return label

        choices = (
            root,
            tuple(data_sources),
            tuple(summaries),
            tuple(plots),
            tuple((n, self._input(n)) for n in all_plots),
            entry.summary_id,
            entry.plot_id,
            entry.plot_scope_id,
            source,
            root_kind,
            tuple(titles.items()),
            tuple(
                (
                    n,
                    self._source_is_summary((n, 0)),
                    widget.pipeline.node_is_bypassed(n),
                )
                for n in summaries
            ),
        )
        if entry.stamps.get("choices") != choices:
            entry.stamps["choices"] = choices
            scopes = [
                (root[0], f"Input data · {data_title(root)}"),
                *((n, f"Summary · {titles[n]}") for n in summaries),
            ]
            plot_counts = Counter(self._input(n)[0] for n in all_plots)
            dialog.set_choices(
                data_sources=[(output, data_title(output)) for output in data_sources],
                data_source=root,
                plot_scopes=[
                    (n, f"{label} ({plot_counts[n]} plots)") for n, label in scopes
                ],
                plot_scope_id=entry.plot_scope_id,
                summaries=[(n, titles[n]) for n in summaries],
                plots=[(n, titles[n]) for n in plots],
                summary_id=entry.summary_id,
                plot_id=entry.plot_id,
                plot_sources=[
                    (root[0], f"{root_kind} · {data_title(root)}"),
                    *(
                        (
                            n,
                            (
                                "Summary table"
                                if self._source_is_summary((n, 0))
                                else "Original measurements"
                            )
                            + f" · {titles[n]}"
                            + (
                                " (bypassed)"
                                if widget.pipeline.node_is_bypassed(n)
                                else ""
                            ),
                        )
                        for n in summaries
                    ),
                ],
                plot_source_id=source[0] if source else entry.plot_scope_id,
            )
        scope_title = titles.get(entry.plot_scope_id, "Unavailable summary")
        plot_title = titles.get(entry.plot_id, "No connected plot")
        tab = dialog.tabs.currentIndex()
        if tab == 0:
            relationship = (
                "Summaries and plots below are linked to this exact table output. "
                "Changing Data source only changes this view."
            )
        elif tab == 1:
            summary_title = titles.get(entry.summary_id, "None — use input data")
            relationship = f"{data_title(root)} → {summary_title}"
        elif entry.plot_scope_id == root[0]:
            relationship = (
                f"{data_title(root)} → {plot_title} · Uses the input data directly, "
                "without a Statistics node."
            )
        else:
            relationship = f"{data_title(root)} → {scope_title} → {plot_title}"
        dialog.set_relationship(
            relationship,
            plot_context=scope_title,
            has_plot=entry.plot_id in plots,
        )
        state = self._state(root[0])
        data_stamp = (root, id(table), state, data_title(root))
        if entry.stamps.get("data") != data_stamp:
            entry.stamps["data"] = data_stamp
            dialog.set_data(
                table,
                title=data_title(root),
                node_id=root[0],
                stale=state[0],
                message=state[2],
            )
        busy_states = {0: state[3:]}
        for kind, node_id in (("summary", entry.summary_id), ("plot", entry.plot_id)):
            revision = self._revision(node_id)
            if node_id not in (summaries if kind == "summary" else plots):
                if entry.stamps.get(kind) != (None,):
                    getattr(dialog, f"set_{kind}")()
                    entry.stamps[kind] = (None,)
                busy_states[1 if kind == "summary" else 2] = (False, "")
                continue
            controller = (
                widget._statistics if kind == "summary" else widget._result_plots
            )
            panel = controller.panels.get((key[0], node_id))
            if panel is None:
                panel = controller.panel_for(node_id)
            else:
                controller._refresh_panel((key[0], node_id), panel)
            node_state = self._state(node_id)
            setup_message = self.plot_setup_message(node_id) if kind == "plot" else ""
            if setup_message:
                # An unfinished editor is not a failed/queued calculation.
                node_state = (True, False, "", False, "")
            busy_states[1 if kind == "summary" else 2] = node_state[3:]
            stamp = (
                node_id,
                revision,
                id(panel.table),
                id(panel.result),
                node_state,
                setup_message,
            )
            entry.revisions[kind] = revision
            if entry.stamps.get(kind) == stamp:
                continue
            entry.stamps[kind] = stamp
            values = dict(
                table=panel.table,
                params=panel.params,
                result=panel.result,
                stale=node_state[0] or widget.pipeline.node_is_bypassed(node_id),
                failed=node_state[1],
                message=(
                    "This node is bypassed. Turn off bypass in the workflow to "
                    "edit or calculate its settings."
                    if widget.pipeline.node_is_bypassed(node_id)
                    else node_state[2]
                ),
                busy=node_state[3],
                editable=not widget.pipeline.node_is_bypassed(node_id),
            )
            if kind == "plot":
                values.update(
                    source_kind="summary"
                    if self._source_is_summary(source)
                    else "original",
                    protected_paths=panel.protected_paths,
                    setup_message=setup_message,
                )
            getattr(dialog, f"set_{kind}")(**values)
        busy, message = busy_states.get(dialog.tabs.currentIndex(), (False, ""))
        dialog.set_busy(
            busy, message.replace("plot", "results").replace("Plot", "Results")
        )

    def close(self):
        self._closed = True
        for entry in tuple(self.windows.values()):
            entry.dialog.close()
        self.windows.clear()
