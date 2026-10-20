from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF, QRectF

import napari_vipp._graph as graph
from napari_vipp._tests.test_graph_wire_routing import (
    _assert_clear,
    _assert_endpoint_route,
)


def _backward_approach_geometry(*, vertical_sign=1, mirror=False):
    # Reduced from the backward Otsu -> Erosion connection: peer lanes require
    # an outer route, and the lower-left card makes its curved arrival useful.
    start, end = QPointF(1000, 0), QPointF(-423, 324)
    endpoints = (QRectF(780, -111.5, 220, 223), QRectF(-423, 212.5, 220, 223))
    obstacles = tuple(QRectF(*rect) for rect in (
        (537, 524.5, 268, 271), (1142, 719.5, 268, 271),
        (-654, -265.5, 268, 271), (21, 373.5, 268, 271),
        (-60, 1.5, 268, 271), (-932, 251.5, 356, 206),
    ))
    peer_polygons = (
        ((-600, 354.5), (-467, 354.5), (-467, 186.5), (-80, 186.5),
         (-80, 292.5), (2566, 292.5), (2566, 133.5), (2602, 133.5)),
        ((640, 0), (780, 0)),
        ((1286, 1122), (1322, 1122), (1322, 1000.75), (384, 1000.75),
         (384, 0), (420, 0)),
        ((-752, -342), (-716, -342), (-716, 137), (-36, 137)),
        ((2022, 4.5), (2130, 4.5), (2130, 280.5), (-159, 280.5),
         (-159, 174.5), (-1016, 174.5), (-1016, 317.5), (-908, 317.5)),
    )
    peers = tuple(graph._flatten_wire_path(graph._rounded_polyline_path(
        [QPointF(*point) for point in polygon],
    )) for polygon in peer_polygons) + (graph._flatten_wire_path(
        graph._bezier_wire_path(QPointF(-203, 324), QPointF(45, 509)),
    ),)

    def point(value):
        return QPointF(-value.x() if mirror else value.x(),
                       vertical_sign * value.y())

    def rect(value):
        return QRectF(-value.right() if mirror else value.left(),
                      -value.bottom() if vertical_sign < 0 else value.top(),
                      value.width(), value.height())

    if mirror:
        start, end = end, start
        endpoints = endpoints[::-1]
    return (
        point(start), point(end), tuple(rect(value) for value in endpoints),
        tuple(rect(value) for value in obstacles),
        tuple(tuple(point(value) for value in peer) for peer in peers),
    )


def _assert_clear_route(path, start, end, endpoints, obstacles, peers):
    _assert_endpoint_route(path, start, end, endpoints)
    _assert_clear(path, obstacles)
    assert not graph._wire_lane_conflict(graph._flatten_wire_path(path), peers)


@pytest.mark.parametrize("vertical_sign", (1, -1), ids=("above", "below"))
def test_clear_departure_is_straight_without_removing_useful_arrival(vertical_sign):
    start, end, endpoints, obstacles, peers = _backward_approach_geometry(
        vertical_sign=vertical_sign,
    )

    path = graph._wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles, peer_paths=peers,
    )

    _assert_clear_route(path, start, end, endpoints, obstacles, peers)
    first = path.elementAt(1)
    assert first.isLineTo()
    assert first.y == start.y()
    assert first.x >= start.x() + 200
    # At the other end, a direct horizontal arrival would cut through the
    # lower-left card. Its curve must still approach from above/below that card.
    arrival = path.elementAt(path.elementCount() - 3)
    assert arrival.isCurveTo()
    assert vertical_sign * (end.y() - arrival.y) > 50


def test_clear_arrival_is_straight_without_removing_useful_departure():
    start, end, endpoints, obstacles, peers = _backward_approach_geometry(mirror=True)

    path = graph._wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles, peer_paths=peers,
    )

    _assert_clear_route(path, start, end, endpoints, obstacles, peers)
    arrival = path.toReversed().elementAt(1)
    assert arrival.isLineTo()
    assert arrival.y == end.y()
    assert arrival.x <= end.x() - 200
    assert path.elementAt(1).isCurveTo()


def test_straight_departure_is_not_used_through_a_card():
    start, end, endpoints, obstacles, peers = _backward_approach_geometry()
    blocker = QRectF(1140, -10, 45, 20)
    obstacles += (blocker,)

    path = graph._wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles, peer_paths=peers,
    )

    _assert_clear_route(path, start, end, endpoints, obstacles, peers)
    first = path.elementAt(1)
    assert not first.isLineTo() or first.x < blocker.left()


def test_straight_departure_is_not_used_on_a_distinct_source_run():
    start, end, endpoints, obstacles, peers = _backward_approach_geometry()
    occupied = (QPointF(1120, 0), QPointF(1280, 0))
    peers += (occupied,)

    path = graph._wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles, peer_paths=peers,
    )

    _assert_clear_route(path, start, end, endpoints, obstacles, peers)
    first = path.elementAt(1)
    assert not first.isLineTo() or first.x < occupied[0].x()
