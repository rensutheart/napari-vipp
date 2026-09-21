"""Node names are visible identities, detached from scientific graph titles."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, Qt

from napari_vipp._graph import ComputeBadgeKind, PipelineGraphView
from napari_vipp.core.pipeline import PrototypePipeline


@pytest.fixture
def named_graph(qtbot):
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    view.show()
    return view, pipeline


def test_alias_keeps_category_binding_compute_badge_and_scientific_title(named_graph):
    view, pipeline = named_graph
    node = pipeline.nodes["input"]
    original_title = node.title
    card = view._cards[node.id]
    category = card.category_label.text()
    view.set_node_subtitle(node.id, "Cell images", "Complete collection binding")
    view.set_node_compute_badge(node.id, ComputeBadgeKind.CPU, tooltip="Used CPU")

    view.set_node_presentation(
        node.id,
        name="YAP/TAZ cells",
        operation=original_title,
        summary="336 fields · 12 wells",
    )

    assert card.title_label._full_text == "YAP/TAZ cells"
    assert card.operation_label._full_text == original_title
    assert not card.operation_label.isHidden()
    assert card.category_label.text() == category
    assert card.subtitle_label._full_text == "Cell images"
    assert card.subtitle_label.toolTip() == "Complete collection binding"
    assert card.compute_badge.text() == "CPU"
    assert card.compute_badge.toolTip() == "Used CPU"
    assert node.title == original_title
    for detail in ("YAP/TAZ cells", original_title, "336 fields", node.id):
        assert detail in card.title_label.toolTip()
        assert detail in card.accessibleDescription()


def test_long_name_is_elided_without_expanding_card(named_graph, qtbot):
    view, pipeline = named_graph
    node = pipeline.nodes["gaussian"]
    card = view._cards[node.id]
    # Start with a short alias so the operation row is present in both states.
    view.set_node_presentation(node.id, name="Smooth", operation=node.title, summary="")
    qtbot.wait(10)
    width = card.width()
    height = card.height()
    name = "Nuclear intensity across independently annotated experimental wells " * 8

    view.set_node_presentation(node.id, name=name, operation=node.title, summary="")
    qtbot.wait(10)

    assert card.width() == width
    assert card.height() == height
    assert card.title_label.text() != name.strip()
    assert "…" in card.title_label.text()
    assert (
        card.title_label.fontMetrics().horizontalAdvance(card.title_label.text())
        <= card.title_label.contentsRect().width()
    )
    assert card.title_row.rect().contains(card.title_label.geometry())
    assert card.title_label.accessibleName() == name.strip()
    assert name.strip() in card.title_label.toolTip()


def test_name_and_summary_render_markup_as_literal_text(named_graph):
    view, _pipeline = named_graph
    card = view._cards["gaussian"]
    view.set_node_presentation(
        "gaussian",
        name="<b>Cells</b>",
        operation="Gaussian <image>",
        summary="Intensity < 100 & area > 2",
    )

    assert card.title_label.textFormat() == Qt.PlainText
    assert card.operation_label.textFormat() == Qt.PlainText
    assert card.title_label.accessibleName() == "<b>Cells</b>"
    assert "&lt;b&gt;Cells&lt;/b&gt;" in card.title_label.toolTip()
    assert "Intensity &lt; 100 &amp; area &gt; 2" in card.title_label.toolTip()
    assert "Intensity < 100 & area > 2" in card.accessibleDescription()


def test_unchanged_operation_name_has_no_extra_row_or_geometry_work(
    named_graph, monkeypatch
):
    view, pipeline = named_graph
    node = pipeline.nodes["gaussian"]
    card = view._cards[node.id]
    proxy = view._proxies[node.id]
    height = card.height()
    geometry_calls = []
    monkeypatch.setattr(proxy, "refresh_ports", lambda: geometry_calls.append("ports"))
    monkeypatch.setattr(
        view, "reroute_connections", lambda **kw: geometry_calls.append("routes")
    )

    view.set_node_presentation(
        node.id, name=node.title, operation=node.title, summary="Sigma: 2"
    )
    view.set_node_presentation(
        node.id, name=node.title, operation=node.title, summary="Sigma: 2"
    )

    assert card.operation_label.isHidden()
    assert card.height() == height
    assert geometry_calls == []
    assert "Sigma: 2" in card.title_label.toolTip()


def test_reset_to_operation_hides_secondary_label_and_restores_height(
    named_graph, qtbot
):
    view, pipeline = named_graph
    node = pipeline.nodes["gaussian"]
    card = view._cards[node.id]
    height = card.height()
    view.set_node_presentation(
        node.id, name="Smooth cells", operation=node.title, summary=""
    )
    qtbot.wait(10)
    assert not card.operation_label.isHidden()
    assert card.height() > height

    view.set_node_presentation(
        node.id, name=node.title, operation=node.title, summary=""
    )
    qtbot.wait(10)

    assert card.operation_label.isHidden()
    assert card.height() == height


@pytest.mark.parametrize("node_id", ["input", "gaussian"])
def test_context_menu_requests_rename_for_input_and_operation(
    named_graph, monkeypatch, node_id
):
    view, _pipeline = named_graph
    requested = []
    view.node_rename_requested.connect(requested.append)

    def choose_rename(menu, _position):
        return next(action for action in menu.actions() if action.text() == "Rename…")

    monkeypatch.setattr("napari_vipp._graph._exec_menu", choose_rename)
    view._show_node_context_menu(node_id, QPoint())

    assert requested == [node_id]


def test_renaming_reflows_connected_ports(named_graph, qtbot):
    view, pipeline = named_graph
    node = pipeline.nodes["gaussian"]
    proxy = view._proxies[node.id]
    original_paths = [connection.path() for connection in proxy.connections]

    view.set_node_presentation(
        node.id, name="Smooth cell image", operation=node.title, summary=""
    )
    qtbot.wait(10)

    assert any(
        connection.path() != original
        for connection, original in zip(proxy.connections, original_paths, strict=True)
    )
    for connection in proxy.connections:
        path = connection.path()
        start = path.elementAt(0)
        end = path.elementAt(path.elementCount() - 1)
        expected_start = connection.source.port_scene_pos(
            "output", connection.source_port
        )
        expected_end = connection.target.port_scene_pos("input", connection.target_port)
        assert (start.x, start.y) == (expected_start.x(), expected_start.y())
        assert (end.x, end.y) == (expected_end.x(), expected_end.y())
