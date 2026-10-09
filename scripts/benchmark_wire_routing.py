"""Measure routing on saved workflows without running scientific operations.

Example (with QT_QPA_PLATFORM=offscreen):
    python scripts/benchmark_wire_routing.py WORKFLOW.json --repeat 3 \
        --background --output .cache/wire-routing-benchmark.json \
        --profile-dir .cache/profiles

Full-pass timings exclude the optional cProfile pass. Background measurements
separate immediate request latency, GUI geometry snapshot time, and final-layout
completion; a 10 ms Qt timer samples event-loop responsiveness during completion.
Every settled background layout must match its synchronous rendered geometry.
"""

from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from check_example_layouts import build_example, configure_application
from qtpy.QtCore import QEvent, QEventLoop, QTimer

from napari_vipp import _graph as graph
from napari_vipp._tests._qt_fonts import load_offscreen_windows_fonts


def wait_for_background_routes(view, *, timeout_ms: int = 30_000) -> None:
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(2)
    poll.timeout.connect(lambda: loop.quit() if not view.wire_routing_pending else None)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    poll.start()
    timeout.start(timeout_ms)
    loop.exec_()
    poll.stop()
    timeout.stop()
    if view.wire_routing_pending:
        raise TimeoutError("Background wire routes did not settle within 30 seconds")


def benchmark_background(view, repeat: int) -> dict:
    expected = tuple(item.path() for item in view._connections)
    view.set_background_wire_routing(True)
    wait_for_background_routes(view)
    snapshot_ms = []
    request_ms = []
    completion_ms = []
    heartbeat_gaps = []
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    previous_heartbeat = perf_counter()

    def on_heartbeat():
        nonlocal previous_heartbeat
        now = perf_counter()
        heartbeat_gaps.append((now - previous_heartbeat) * 1000)
        previous_heartbeat = now

    heartbeat.timeout.connect(on_heartbeat)
    for _ in range(repeat):
        started = perf_counter()
        view._wire_route_snapshot(True)
        snapshot_ms.append((perf_counter() - started) * 1000)
        view._mark_graph_geometry_changed()
        started = perf_counter()
        previous_heartbeat = started
        heartbeat.start()
        view.reroute_connections()
        request_ms.append((perf_counter() - started) * 1000)
        wait_for_background_routes(view)
        completion_ms.append((perf_counter() - started) * 1000)
        heartbeat.stop()
        if view._connections and (
            view._wire_routing_applied_revision != view.route_revision
        ):
            raise AssertionError("Background routes settled without a current result")
        if tuple(item.path() for item in view._connections) != expected:
            raise AssertionError("Background routes differ from synchronous geometry")
    view.set_background_wire_routing(False)
    return {
        "snapshot_ms": snapshot_ms,
        "request_ms": request_ms,
        "completion_ms": completion_ms,
        "heartbeat_count": len(heartbeat_gaps),
        "median_heartbeat_gap_ms": median(heartbeat_gaps) if heartbeat_gaps else None,
        "maximum_heartbeat_gap_ms": max(heartbeat_gaps) if heartbeat_gaps else None,
        "geometry_matches_synchronous": True,
    }


def benchmark(
    path: Path, repeat: int, phase: str, profile_dir: Path | None, background: bool
) -> dict:
    construction_started = perf_counter()
    view = build_example(path, phase)
    construction_ms = (perf_counter() - construction_started) * 1000
    route_samples = []
    full_pass_ms = []
    astar_calls = 0
    depth = 0
    profiled = False
    original_route = graph._wire_path
    original_search = graph._wire_visibility_detour

    def trace_route(start, end, **kwargs):
        nonlocal depth
        depth += 1
        started = perf_counter()
        previous_searches = astar_calls
        try:
            return original_route(start, end, **kwargs)
        finally:
            depth -= 1
            if depth == 0:
                route_samples.append({
                    "elapsed_ms": (perf_counter() - started) * 1000,
                    "start": [start.x(), start.y()],
                    "end": [end.x(), end.y()],
                    "obstacles": len(kwargs.get("obstacles", ())),
                    "peers": len(kwargs.get("peer_paths", ())),
                    "searches": astar_calls - previous_searches,
                    "profiled": profiled,
                })

    def trace_search(*args, **kwargs):
        nonlocal astar_calls
        astar_calls += 1
        return original_search(*args, **kwargs)

    graph._wire_path = trace_route
    graph._wire_visibility_detour = trace_search
    profile = cProfile.Profile()
    try:
        for _ in range(repeat):
            view._mark_graph_geometry_changed()
            started = perf_counter()
            view.reroute_connections()
            full_pass_ms.append((perf_counter() - started) * 1000)
        started = perf_counter()
        view.reroute_connections()
        unchanged_ms = (perf_counter() - started) * 1000
        if profile_dir is not None:
            profile_dir.mkdir(parents=True, exist_ok=True)
            view._mark_graph_geometry_changed()
            profiled = True
            profile.runcall(view.reroute_connections)
            profile.dump_stats(str(profile_dir / f"{path.stem}-{phase}.prof"))
    finally:
        graph._wire_path = original_route
        graph._wire_visibility_detour = original_search
    identities = {
        (
            item.source.port_scene_pos("output", item.source_port).x(),
            item.source.port_scene_pos("output", item.source_port).y(),
            item.target.port_scene_pos("input", item.target_port).x(),
            item.target.port_scene_pos("input", item.target_port).y(),
        ): [item.source_id, item.source_port, item.target_id, item.target_port]
        for item in view._connections
    }
    for sample in route_samples:
        sample["edge"] = identities.get((*sample["start"], *sample["end"]))
    result = {
        "workflow": str(path.resolve()),
        "phase": phase,
        "nodes": len(view._proxies),
        "wires": len(view._connections),
        "construction_ms": construction_ms,
        "full_pass_ms": full_pass_ms,
        "median_full_pass_ms": median(full_pass_ms),
        "unchanged_ms": unchanged_ms,
        "search_calls": astar_calls,
        "slowest_edges": sorted(
            (sample for sample in route_samples if not sample["profiled"]),
            key=lambda sample: sample["elapsed_ms"], reverse=True
        )[:10],
    }
    if background:
        result["background"] = benchmark_background(view, repeat)
    view.close()
    view.deleteLater()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflows", type=Path, nargs="+")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--phase", choices=("initial", "ready"), default="ready")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--background", action="store_true",
        help="Compare request and completion latency with a 10 ms GUI heartbeat",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    app = configure_application()
    load_offscreen_windows_fonts(app)
    reports = []
    for path in args.workflows:
        print(f"Routing {path} ({args.phase})...", flush=True)
        result = benchmark(
            path, args.repeat, args.phase, args.profile_dir, args.background
        )
        reports.append(result)
        print(json.dumps(result, indent=2), flush=True)
        app.sendPostedEvents(None, QEvent.DeferredDelete)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
