from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF, QRectF

import napari_vipp._graph as graph


def _assert_clear_forward_route(path, start, end, endpoint_rects, obstacles):
    assert path.pointAtPercent(0) == start
    assert path.pointAtPercent(1) == end
    assert path.pointAtPercent(0.001).x() > start.x()
    assert path.pointAtPercent(0.999).x() < end.x()
    protected = tuple(rect.adjusted(0.75, 0.75, -0.75, -0.75)
                      for rect in endpoint_rects)
    blockers = protected + tuple(obstacles)
    assert graph._route_collision_penalty(
        graph._flatten_wire_path(path), blockers,
    ) == 0
    assert not any(rect.contains(path.pointAtPercent(index / 1000))
                   for rect in blockers for index in range(1001))


@pytest.mark.parametrize(
    "vertical_sign", (1, -1), ids=("below-then-above", "above-then-below"),
)
def test_clear_distant_perimeter_is_improved_by_local_multi_bend_route(vertical_sign):
    start, end = QPointF(0, 0), QPointF(1000, 0)
    source = QRectF(-220, -100, 220, 217)
    target = QRectF(1000, -100, 220, 217)
    obstacles = (QRectF(200, -600, 100, 620), QRectF(500, -20, 100, 620))
    if vertical_sign < 0:
        source = QRectF(
            source.left(), -source.bottom(), source.width(), source.height(),
        )
        target = QRectF(
            target.left(), -target.bottom(), target.width(), target.height(),
        )
        obstacles = tuple(
            QRectF(rect.left(), -rect.bottom(), rect.width(), rect.height())
            for rect in obstacles
        )
    endpoints = (source, target)
    protected = tuple(rect.adjusted(0.5, 0.5, -0.5, -0.5) for rect in endpoints)
    blockers = graph._WireObstacles(protected + obstacles)
    candidates = graph._wire_route_candidates(start, end, blockers, protected=protected)
    clear_candidates = [path for path, points, _score in candidates
                        if not graph._route_collision_penalty(points, blockers)]
    # A clear cheap candidate exists. The old collision-only fallback therefore
    # skipped the local zigzag and took the perimeter more than 600 px away.
    assert clear_candidates
    assert min(path.length() for path in clear_candidates) > 2000

    path = graph._card_avoiding_wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles,
    )

    _assert_clear_forward_route(path, start, end, endpoints, obstacles)
    assert path.length() < 1300
    assert path.boundingRect().top() >= -80
    assert path.boundingRect().bottom() <= 80


@pytest.mark.parametrize("distant_count", (0, 164))
def test_short_clear_detour_does_not_start_visibility_improvement(
    monkeypatch, distant_count,
):
    def unexpected_search(*_args, **_kwargs):
        pytest.fail("A nearby clear candidate should not pay for visibility search.")

    monkeypatch.setattr(graph, "_wire_visibility_detour", unexpected_search)
    start, end = QPointF(0, 0), QPointF(600, 0)
    endpoints = (QRectF(-220, -100, 220, 217), QRectF(600, -100, 220, 217))
    obstacles = (QRectF(240, -80, 120, 160),) + tuple(
        QRectF(2000 + 330 * (index % 12), -1800 + 330 * (index // 12), 220, 217)
        for index in range(distant_count)
    )
    assert graph._route_collision_penalty(
        graph._flatten_wire_path(graph._bezier_wire_path(start, end)), obstacles,
    ) > 0

    path = graph._card_avoiding_wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles,
    )

    _assert_clear_forward_route(path, start, end, endpoints, obstacles)
    assert path.length() < 1000
    assert path.boundingRect().top() >= -150
    assert path.boundingRect().bottom() <= 150


def _zigzag_search_geometry():
    start, end = QPointF(0, 0), QPointF(1000, 0)
    endpoints = (QRectF(-220, -100, 220, 217), QRectF(1000, -100, 220, 217))
    obstacles = (QRectF(200, -600, 100, 620), QRectF(500, -20, 100, 620))
    protected = tuple(rect.adjusted(0.5, 0.5, -0.5, -0.5) for rect in endpoints)
    return start, end, endpoints, obstacles, protected


def test_exhausted_improvement_budget_retains_the_clear_incumbent(monkeypatch):
    start, end, endpoints, obstacles, protected = _zigzag_search_geometry()
    blockers = graph._WireObstacles(protected + obstacles)
    incumbent, _points, _score = min(
        graph._wire_route_candidates(start, end, blockers, protected=protected),
        key=lambda item: item[2],
    )
    original = graph._wire_visibility_detour
    searches = []

    def exhausted_search(*args, **kwargs):
        searches.append(True)
        # A genuine capped search reaches no goal in one visit. It must not
        # replace the already clear wire with partial/blocked geometry.
        kwargs["max_visits"] = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "_wire_visibility_detour", exhausted_search)

    path = graph._card_avoiding_wire_path(
        start, end, endpoint_rects=endpoints, obstacles=obstacles,
    )

    assert searches
    assert path == incumbent
    _assert_clear_forward_route(path, start, end, endpoints, obstacles)


@pytest.mark.parametrize("score_limit, finds_route", ((1200, False), (1500, True)))
def test_visibility_improvement_score_limit_includes_bend_cost(
    score_limit, finds_route,
):
    start, end, endpoints, obstacles, protected = _zigzag_search_geometry()
    blockers = graph._WireObstacles(protected + obstacles)

    path = graph._wire_visibility_detour(
        start, end, blockers, protected, max_visits=1024, score_limit=score_limit,
    )

    if finds_route:
        assert path is not None
        assert path.length() < 1200
        _assert_clear_forward_route(path, start, end, endpoints, obstacles)
    else:
        # The local route is shorter than 1200 px, but its six turns put the
        # length-plus-bends score above this limit, so it is not accepted.
        assert path is None


def test_visibility_search_cancellation_interrupts_optional_improvement():
    start, end, _endpoints, obstacles, protected = _zigzag_search_geometry()
    blockers = graph._WireObstacles(protected + obstacles)
    checks = []

    def cancelled():
        checks.append(True)
        return True

    with pytest.raises(graph._WireRoutingCancelled):
        graph._wire_visibility_detour(
            start, end, blockers, protected, max_visits=1024,
            score_limit=2500, cancel_callback=cancelled,
        )
    assert len(checks) == 1
