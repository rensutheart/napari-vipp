from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF, QRectF

import napari_vipp._graph as graph
from napari_vipp._tests.test_wire_route_compactness import _assert_clear_forward_route


def _staircase_geometry(vertical_sign=1):
    start, end = QPointF(260, 101), QPointF(742, 424)
    endpoints = (QRectF(50, 25, 210, 150), QRectF(742, 349, 210, 150))
    # The middle column and early vertical leg are blocked. A horizontal run
    # at the source's height followed by a late vertical leg stays clear.
    obstacles = (QRectF(427, 184, 148, 164), QRectF(118, 367, 210, 107))
    if vertical_sign < 0:
        start, end = QPointF(start.x(), -start.y()), QPointF(end.x(), -end.y())
        endpoints = tuple(
            QRectF(rect.left(), -rect.bottom(), rect.width(), rect.height())
            for rect in endpoints
        )
        obstacles = tuple(
            QRectF(rect.left(), -rect.bottom(), rect.width(), rect.height())
            for rect in obstacles
        )
    return start, end, endpoints, obstacles


@pytest.mark.parametrize("vertical_sign", (1, -1), ids=("down", "up"))
def test_clear_late_turn_avoids_equal_length_staircase(monkeypatch, vertical_sign):
    def unexpected_search(*_args, **_kwargs):
        pytest.fail("The clear two-turn candidate should not start visibility search.")

    monkeypatch.setattr(graph, "_wire_visibility_detour", unexpected_search)
    start, end, endpoints, obstacles = _staircase_geometry(vertical_sign)
    assert graph._route_collision_penalty(
        graph._flatten_wire_path(graph._bezier_wire_path(start, end)), obstacles,
    ) > 0

    path = graph._wire_path(start, end, endpoint_rects=endpoints, obstacles=obstacles)

    _assert_clear_forward_route(path, start, end, endpoints, obstacles)
    # Rounded corners start 18 px before the 36 px arrival stub. The first
    # straight segment should reach that corner instead of dropping early.
    first_line = path.elementAt(1)
    assert first_line.y == start.y()
    assert first_line.x >= end.x() - 54
    assert path.length() <= abs(end.x() - start.x()) + abs(end.y() - start.y())


def test_late_turn_still_avoids_a_card_on_the_source_lane():
    start, end, endpoints, obstacles = _staircase_geometry()
    obstacles += (QRectF(480, 70, 80, 60),)

    path = graph._wire_path(start, end, endpoint_rects=endpoints, obstacles=obstacles)

    _assert_clear_forward_route(path, start, end, endpoints, obstacles)
    assert path.elementAt(1).x < 480


def test_late_turn_still_reserves_distinct_source_vertical_runs():
    start, end, endpoints, obstacles = _staircase_geometry()
    peer = (QPointF(706, 150), QPointF(706, 390))
    path = graph._wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles, peer_paths=(peer,),
    )

    _assert_clear_forward_route(path, start, end, endpoints, obstacles)
    assert not graph._wire_lane_conflict(graph._flatten_wire_path(path), (peer,))
