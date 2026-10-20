from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF, QRectF

import napari_vipp._graph as graph
from napari_vipp._tests.test_graph_wire_routing import (
    _assert_clear,
    _assert_endpoint_route,
    _assert_no_shared_runs,
)


def test_independent_wire_prefers_near_gap_over_distant_perimeter(monkeypatch):
    start, end = QPointF(0, 0), QPointF(900, 0)
    endpoints = (QRectF(-100, -50, 100, 100), QRectF(900, -50, 100, 100))
    obstacles = (
        QRectF(300, -60, 140, 120),
        # An unrelated distant row must not pull this wire to its perimeter.
        QRectF(1800, -600, 120, 120),
    )
    peer = (QPointF(-200, -93.5), QPointF(1100, -93.5))
    ordinary = graph._card_avoiding_wire_path(
        start, end, obstacles=obstacles, endpoint_rects=endpoints
    )
    assert graph._wire_lane_conflict(graph._flatten_wire_path(ordinary), (peer,))
    offsets = []
    original_candidates = graph._wire_middle_candidates

    def traced_candidates(first, last, blockers):
        offsets.append(first.y())
        return original_candidates(first, last, blockers)

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("The nearby alternate lane is already in the ordinary catalogue.")

    monkeypatch.setattr(graph, "_wire_visibility_detour", unexpected_search)
    monkeypatch.setattr(graph, "_wire_middle_candidates", traced_candidates)
    path = graph._wire_path(
        start, end, obstacles=obstacles, endpoint_rects=endpoints, peer_paths=(peer,)
    )

    _assert_endpoint_route(path, start, end, endpoints)
    _assert_clear(path, obstacles)
    _assert_no_shared_runs([path, graph._rounded_polyline_path(list(peer), radius=0)])
    assert not graph._wire_lane_conflict(graph._flatten_wire_path(path), (peer,))
    assert path.boundingRect().top() >= -120
    assert path.boundingRect().bottom() <= 120
    assert graph._polyline_length(graph._flatten_wire_path(path)) < 1200
    assert len(offsets) == 1


def test_independent_wire_checks_near_shift_before_outer_detour(monkeypatch):
    start, end = QPointF(0, 0), QPointF(900, 0)
    endpoints = (QRectF(-100, -50, 100, 100), QRectF(900, -50, 100, 100))
    obstacles = (
        QRectF(300, -300, 140, 360),
        # Moving the departure lane just 12 pixels clears this small card.
        QRectF(60, 80, 30, 50),
    )
    peer = (QPointF(200, 70), QPointF(800, 70))
    offsets = []
    original_candidates = graph._wire_middle_candidates

    def traced_candidates(first, last, blockers):
        offsets.append(first.y())
        return original_candidates(first, last, blockers)

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("A near shifted lane already clears both cards and the peer.")

    monkeypatch.setattr(graph, "_wire_visibility_detour", unexpected_search)
    monkeypatch.setattr(graph, "_wire_middle_candidates", traced_candidates)
    path = graph._wire_path(
        start, end, obstacles=obstacles, endpoint_rects=endpoints, peer_paths=(peer,)
    )

    _assert_endpoint_route(path, start, end, endpoints)
    _assert_clear(path, obstacles)
    _assert_no_shared_runs([path, graph._rounded_polyline_path(list(peer), radius=0)])
    assert not graph._wire_lane_conflict(graph._flatten_wire_path(path), (peer,))
    assert path.boundingRect().top() >= -120
    assert path.boundingRect().bottom() <= 120
    assert graph._polyline_length(graph._flatten_wire_path(path)) < 1200
    assert 2 <= len(offsets) <= 5


def test_card_required_outer_route_does_not_expand_peer_offset_search(monkeypatch):
    start, end = QPointF(0, 0), QPointF(900, 0)
    endpoints = (QRectF(-100, -50, 100, 100), QRectF(900, -50, 100, 100))
    obstacles = (QRectF(300, -300, 140, 600),)
    ordinary = graph._card_avoiding_wire_path(
        start, end, obstacles=obstacles, endpoint_rects=endpoints
    )
    peer_y = ordinary.boundingRect().top()
    peer = (QPointF(-200, peer_y), QPointF(1100, peer_y))
    offsets = []
    original_candidates = graph._wire_middle_candidates

    def traced_candidates(first, last, blockers):
        offsets.append(first.y())
        return original_candidates(first, last, blockers)

    monkeypatch.setattr(graph, "_wire_middle_candidates", traced_candidates)
    path = graph._wire_path(
        start, end, obstacles=obstacles, endpoint_rects=endpoints, peer_paths=(peer,)
    )

    _assert_endpoint_route(path, start, end, endpoints)
    _assert_clear(path, obstacles)
    _assert_no_shared_runs([path, graph._rounded_polyline_path(list(peer), radius=0)])
    assert not graph._wire_lane_conflict(graph._flatten_wire_path(path), (peer,))
    assert min(path.boundingRect().top(), -path.boundingRect().bottom()) <= -300
    assert graph._polyline_length(graph._flatten_wire_path(path)) <= (
        ordinary.length() + 160
    )
    assert len(offsets) == 1
