"""Exact collision clipping used by the wire candidate scoring."""

from math import sqrt

import numpy as np
import pytest
from qtpy.QtCore import QPointF, QRectF

from napari_vipp._graph import (
    _route_collision_penalty,
    _scalar_route_collision_penalty,
    _segment_rect_penalty,
    _WireObstacles,
)


@pytest.mark.parametrize(
    ("coordinates", "rectangle", "expected"),
    [
        (((0, 0), (100, 0)), (50, -100, 0.25, 200), 25.25),
        (
            ((0, 0), (100, 100)),
            (49.999, 49.999, 0.002, 0.002),
            25 + 0.002 * sqrt(2),
        ),
        (((0, 10), (100, 10)), (40, 10, 20, 20), 0),
        (((0, 0), (50, 50)), (50, 50, 20, 20), 0),
        (((40, 10), (60, 10)), (20, 0, 60, 20), 45),
        (((0, 0), (0, 100)), (-0.001, 50, 0.002, 0.1), 25.1),
        (((0, 0), (0, 0)), (-1, -1, 2, 2), 25),
        (((0, 0), (100, 0)), (50, 0, 0, 20), 0),
    ],
    ids=[
        "thin-card", "tiny-diagonal-crossing", "edge-tangent", "corner-tangent",
        "segment-inside", "vertical-thin-card", "stationary-inside", "empty-card",
    ],
)
def test_wire_collision_clips_exact_geometry(coordinates, rectangle, expected):
    points = tuple(QPointF(*coordinate) for coordinate in coordinates)
    # Four obstacles also exercise the vectorized path for two-point wires.
    obstacles = (QRectF(*rectangle),) + tuple(
        QRectF(1000 + index * 100, 1000, 10, 10) for index in range(3)
    )
    assert _route_collision_penalty(points, obstacles) == pytest.approx(
        expected, abs=1e-8
    )


def test_wire_collision_matches_scalar_clipping_for_irregular_routes():
    rng = np.random.default_rng(156)
    for _ in range(100):
        points = tuple(
            QPointF(*coordinate)
            for coordinate in rng.uniform(-100, 100, (rng.integers(2, 40), 2))
        )
        count = rng.integers(2, 30)
        obstacles = tuple(
            QRectF(*position, *size)
            for position, size in zip(
                rng.uniform(-100, 100, (count, 2)),
                rng.uniform(0.0001, 100, (count, 2)),
                strict=True,
            )
        )
        expected = sum(
            _segment_rect_penalty(start, end, rectangle)
            for start, end in zip(points, points[1:], strict=False)
            for rectangle in obstacles
        )
        assert _route_collision_penalty(points, obstacles) == pytest.approx(
            expected, rel=1e-12, abs=1e-8
        )


def test_wire_collision_keeps_distant_obstacles_clear():
    points = (QPointF(0, 0), QPointF(500, 0), QPointF(500, 500))
    obstacles = (
        QRectF(100, 100, 200, 200),
        QRectF(100, 700, 200, 200),
        QRectF(800, 100, 200, 200),
    )
    assert _route_collision_penalty(points, obstacles) == 0


def test_wire_obstacle_snapshot_owns_rectangles_and_readonly_bounds():
    rectangle = QRectF(50, -10, 10, 20)
    obstacles = _WireObstacles((rectangle,))
    points = (QPointF(0, 0), QPointF(100, 0))
    rectangle.translate(1000, 1000)
    assert obstacles[0] == QRectF(50, -10, 10, 20)
    assert _route_collision_penalty(points, obstacles) == 35
    assert not obstacles.bounds.flags.writeable
    with pytest.raises(ValueError):
        obstacles.bounds[0, 0] = 1000


def test_invalid_rectangle_collection_retains_scalar_collision_behavior():
    points = tuple(QPointF(index * 30, index * 10) for index in range(6))
    obstacles = (
        QRectF(60, 0, -30, 40),
        QRectF(50, 10, 25, 40),
        QRectF(30, 30, 30, 0),
        QRectF(150, 200, 40, 50),
    )
    assert _route_collision_penalty(points, obstacles) == pytest.approx(
        _scalar_route_collision_penalty(points, obstacles), rel=1e-12, abs=1e-8
    )
