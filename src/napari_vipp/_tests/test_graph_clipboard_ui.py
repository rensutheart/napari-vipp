from __future__ import annotations

from qtpy.QtCore import QPoint, QPointF, Qt

from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.pipeline import PrototypePipeline


def _build_view(qtbot) -> tuple[PipelineGraphView, PrototypePipeline]:
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    return view, pipeline


def test_ctrl_and_shift_click_selection_semantics(qtbot):
    view, _pipeline = _build_view(qtbot)

    view._handle_node_press("input", Qt.NoModifier)
    view._handle_node_press("gaussian", Qt.ControlModifier)

    assert view.selected_node_ids() == ("input", "gaussian")
    assert view.primary_node_id() == "gaussian"

    view._handle_node_press("threshold", Qt.ShiftModifier)

    assert view.selected_node_ids() == ("input", "gaussian", "threshold")
    assert view.primary_node_id() == "threshold"

    view._handle_node_press("gaussian", Qt.ControlModifier)

    assert view.selected_node_ids() == ("input", "threshold")
    assert view.primary_node_id() == "threshold"

    view._handle_node_press("input", Qt.NoModifier)

    assert view.selected_node_ids() == ("input",)
    assert view.primary_node_id() == "input"


def test_viewport_clicks_apply_additive_node_selection(qtbot):
    view, _pipeline = _build_view(qtbot)
    input_pos = view.mapFromScene(
        view._proxies["input"].sceneBoundingRect().center()
    )
    gaussian_pos = view.mapFromScene(
        view._proxies["gaussian"].sceneBoundingRect().center()
    )
    threshold_pos = view.mapFromScene(
        view._proxies["threshold"].sceneBoundingRect().center()
    )

    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=input_pos)
    qtbot.mouseClick(
        view.viewport(),
        Qt.LeftButton,
        Qt.ControlModifier,
        pos=gaussian_pos,
    )
    qtbot.mouseClick(
        view.viewport(),
        Qt.LeftButton,
        Qt.ShiftModifier,
        pos=threshold_pos,
    )

    assert view.selected_node_ids() == ("input", "gaussian", "threshold")


def test_right_click_selected_node_preserves_group_and_unselected_replaces_it(
    qtbot,
    monkeypatch,
):
    view, _pipeline = _build_view(qtbot)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", lambda *_args: None)
    view.set_selected_nodes(("input", "gaussian"), primary_node_id="gaussian")

    selected_pos = view.mapFromScene(
        view._proxies["input"].sceneBoundingRect().center()
    )
    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=selected_pos)

    assert view.selected_node_ids() == ("input", "gaussian")
    assert view.primary_node_id() == "input"

    unselected_pos = view.mapFromScene(
        view._proxies["threshold"].sceneBoundingRect().center()
    )
    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=unselected_pos)

    assert view.selected_node_ids() == ("threshold",)


def test_blank_click_clears_node_selection(qtbot):
    view, _pipeline = _build_view(qtbot)
    view.set_selected_nodes(("input", "gaussian"), primary_node_id="gaussian")
    blank = QPoint(view.viewport().width() - 8, view.viewport().height() - 8)
    assert view.itemAt(blank) is None

    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=blank)

    assert view.selected_node_ids() == ()
    assert view.primary_node_id() is None


def test_multi_node_delete_and_copy_emit_one_bulk_request(qtbot):
    view, _pipeline = _build_view(qtbot)
    view.set_selected_nodes(("input", "gaussian"), primary_node_id="gaussian")
    copied = []
    deleted = []
    view.nodes_copy_requested.connect(copied.append)
    view.nodes_delete_requested.connect(deleted.append)

    qtbot.keyClick(view, Qt.Key_C, Qt.ControlModifier)
    qtbot.keyClick(view, Qt.Key_Delete)

    assert copied == [("input", "gaussian")]
    assert deleted == [("input", "gaussian")]


def test_ctrl_v_requests_paste_at_viewport_center(qtbot):
    view, _pipeline = _build_view(qtbot)
    view.set_clipboard_state(True)
    requested = []
    view.paste_requested.connect(requested.append)
    expected = view.viewport_center_scene_position()

    qtbot.keyClick(view, Qt.Key_V, Qt.ControlModifier)

    assert len(requested) == 1
    assert (requested[0] - expected).manhattanLength() < 0.01


def test_node_menu_enables_paste_values_only_for_exact_operation(
    qtbot,
    monkeypatch,
):
    view, _pipeline = _build_view(qtbot)
    enabled = []
    requested = []
    view.node_paste_values_requested.connect(requested.append)

    def inspect_menu(menu, _pos):
        action = next(
            action
            for action in menu.actions()
            if action.text().startswith("Paste values")
        )
        enabled.append(action.isEnabled())
        return action if action.isEnabled() else None

    monkeypatch.setattr("napari_vipp._graph._exec_menu", inspect_menu)
    view.set_clipboard_state(
        True,
        copied_single_operation_id=view._proxies["gaussian"].operation_id,
        copied_single_title="Gaussian Blur",
    )

    view._show_node_context_menu("threshold", QPoint())
    view._show_node_context_menu("gaussian", QPoint())

    assert enabled == [False, True]
    assert requested == ["gaussian"]


def test_canvas_menu_pastes_at_clicked_scene_position(qtbot, monkeypatch):
    view, _pipeline = _build_view(qtbot)
    view.set_clipboard_state(True)
    requested = []
    view.paste_requested.connect(requested.append)
    scene_pos = QPointF(127.5, -84.0)

    def choose_paste(menu, _pos):
        return next(
            action
            for action in menu.actions()
            if action.text() == "Paste nodes here"
        )

    monkeypatch.setattr("napari_vipp._graph._exec_menu", choose_paste)

    view._show_canvas_context_menu(scene_pos, QPoint())

    assert requested == [scene_pos]


def test_selected_nodes_move_together_from_their_own_start_positions(qtbot):
    view, _pipeline = _build_view(qtbot)
    view.set_selected_nodes(("input", "gaussian"), primary_node_id="gaussian")
    starts = view._selected_node_start_positions("gaussian")
    delta = QPointF(73.0, -29.0)

    view._move_selected_nodes_during_drag(starts, delta)
    view._finish_selected_node_drag(starts)

    for node_id, start in starts.items():
        assert (view._proxies[node_id].pos() - (start + delta)).manhattanLength() < 0.01


def test_dragging_one_selected_node_moves_the_whole_selection(qtbot):
    view, _pipeline = _build_view(qtbot)
    view.set_selected_nodes(("input", "gaussian"), primary_node_id="gaussian")
    starts = {
        node_id: QPointF(view._proxies[node_id].pos())
        for node_id in ("input", "gaussian")
    }
    start = view.mapFromScene(
        view._proxies["gaussian"].sceneBoundingRect().center()
    )
    end = start + QPoint(58, 31)
    moved = []
    view.nodes_moved.connect(lambda old, new: moved.append((old, new)))

    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    input_delta = view._proxies["input"].pos() - starts["input"]
    gaussian_delta = view._proxies["gaussian"].pos() - starts["gaussian"]
    assert (input_delta - gaussian_delta).manhattanLength() < 0.01
    assert input_delta.manhattanLength() > 1
    assert len(moved) == 1


def test_plain_click_on_group_member_reduces_selection_when_not_dragged(qtbot):
    view, _pipeline = _build_view(qtbot)
    view.set_selected_nodes(("input", "gaussian"), primary_node_id="gaussian")
    position = view.mapFromScene(
        view._proxies["input"].sceneBoundingRect().center()
    )

    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=position)

    assert view.selected_node_ids() == ("input",)


def test_source_tunnel_badge_accepts_new_and_loose_node_insertion(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    loose = pipeline.add_node("median_filter")
    pipeline.add_output_tunnel("Raw", "input", 0)
    assert pipeline.connect_to_tunnel("Raw", "gaussian", 0).success
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    source = view._tunnel_source_ports["Raw"]
    badge_pos = source._tunnel_badge.mapToScene(
        source._tunnel_badge.boundingRect().center()
    )
    spliced = []
    view.tunnel_node_splice_requested.connect(
        lambda node_id, name, old, new: spliced.append((node_id, name, old, new))
    )
    old_pos = QPointF(view._proxies[loose.id].pos())
    new_pos = old_pos + QPointF(30, 20)

    assert view._update_tunnel_insert_preview(badge_pos) == "Raw"
    assert view.release_existing_node_insert(
        loose.id,
        old_pos,
        new_pos,
        badge_pos,
    )

    assert [(item[0], item[1]) for item in spliced] == [(loose.id, "Raw")]


def test_palette_tunnel_preview_shows_incompatible_validator_result(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.add_output_tunnel("Raw", "input", 0)
    view = PipelineGraphView()
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    source = view._tunnel_source_ports["Raw"]
    badge_pos = source._tunnel_badge.mapToScene(
        source._tunnel_badge.boundingRect().center()
    )
    messages = []
    view.status_message.connect(messages.append)
    view.set_tunnel_insert_validator(
        lambda operation_id, name, _node_id: (
            "incompatible",
            f"{operation_id} cannot preserve {name}",
        )
    )

    assert view._update_tunnel_insert_preview(badge_pos, "input") == "Raw"
    assert view._highlighted_tunnel_insert_state == "incompatible"
    assert source._drop_state == "incompatible"
    assert messages[-1] == "input cannot preserve Raw"


def test_subscriber_tunnel_badge_is_not_an_insertion_target(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.add_output_tunnel("Raw", "input", 0)
    assert pipeline.connect_to_tunnel("Raw", "gaussian", 0).success
    view = PipelineGraphView()
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    subscriber = view._tunnel_subscriber_ports["Raw"][0]
    badge_pos = subscriber._tunnel_badge.mapToScene(
        subscriber._tunnel_badge.boundingRect().center()
    )

    assert view._update_tunnel_insert_preview(badge_pos) == ""


def test_source_tunnel_context_menu_requests_operation_picker(qtbot, monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.add_output_tunnel("Raw", "input", 0)
    view = PipelineGraphView()
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    source = view._tunnel_source_ports["Raw"]
    requested = []
    view.tunnel_insert_requested.connect(
        lambda name, pos: requested.append((name, pos))
    )

    def choose_insert(menu, _pos):
        return next(
            action
            for action in menu.actions()
            if action.text() == "Insert node before 'Raw'..."
        )

    monkeypatch.setattr("napari_vipp._graph._exec_menu", choose_insert)
    position = QPointF(12, 34)

    view._show_tunnel_source_context_menu("Raw", source, position, QPoint())

    assert requested == [("Raw", position)]


def test_source_tunnel_context_menu_keeps_existing_tunnel_options(qtbot, monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.add_output_tunnel("Raw", "input", 0)
    view = PipelineGraphView()
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    source = view._tunnel_source_ports["Raw"]
    requested = []
    view.port_context_requested.connect(
        lambda kind, node_id, port_index, pos: requested.append(
            (kind, node_id, port_index, pos)
        )
    )

    def choose_tunnel_options(menu, _pos):
        return next(
            action for action in menu.actions() if action.text() == "Tunnel options..."
        )

    monkeypatch.setattr("napari_vipp._graph._exec_menu", choose_tunnel_options)
    global_position = QPoint(25, 50)

    view._show_tunnel_source_context_menu(
        "Raw", source, QPointF(12, 34), global_position
    )

    assert requested == [
        ("output", source.node_id, source.port_index, global_position)
    ]
