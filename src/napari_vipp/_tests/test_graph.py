from __future__ import annotations

from qtpy.QtCore import QPoint, QPointF, Qt

from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.pipeline import PROTOTYPE_NODES


def test_graph_node_can_be_dragged_with_mouse(qtbot):
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.resize(980, 520)
    view.build_demo_graph(PROTOTYPE_NODES)
    view.show()
    qtbot.waitExposed(view)

    proxy = view._proxies["gaussian"]
    before = QPointF(proxy.pos())
    connection = view._connections[0]
    path_before = connection.path().elementAt(3).x

    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(90, 45)
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert proxy.pos() != before
    assert connection.path().elementAt(3).x != path_before


def test_thumbnail_click_still_requests_inspection(qtbot):
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.resize(980, 520)
    view.build_demo_graph(PROTOTYPE_NODES)
    view.show()
    qtbot.waitExposed(view)

    inspected = []
    view.inspect_requested.connect(inspected.append)

    card = view._cards["gaussian"]
    preview_center = QPointF(card.preview.geometry().center())
    start = view.mapFromScene(
        view._proxies["gaussian"].mapToScene(preview_center)
    )
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=start)

    assert inspected == ["gaussian"]
