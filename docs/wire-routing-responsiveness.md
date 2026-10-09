# Workflow wire routing and responsiveness

Unreleased implementation evidence recorded on 2026-10-09. This record covers
canvas presentation; scientific graph execution and saved workflow schema are
unchanged. Public controls and tasks belong in the companion vipp-mkdocs manual.

## Routing contract

Wires from the same exact source node and output port can share natural trunks.
Different source nodes or output ports reserve separate collinear runs, while
isolated perpendicular crossings remain valid. Routing uses a stable connection
order, protects endpoint cards, and checks other cards and visible tunnel badges.
A direct consumer can leave its own named output badge.

Interactive workflow editors use one background routing worker per view. The GUI
captures owned numeric endpoint, card and badge geometry once per pass. Workers
compute the same routes as synchronous graph renderers without accessing scene
items. Requests coalesce; each geometry edit, including each actual pointer move,
invalidates older results. Closing, rebuilding, disabling background routing, and
destroying a view cancel its outstanding work. Cancellation is checked between
route candidates, during fallback search, and before publication.

Attached wires follow moved endpoints immediately with a bounded preview that
checks their own cards. Final obstacle and peer routing is applied only when its
generation, geometry revision and connection keys still match the current view.
Transient previews can differ from the settled layout.

Candidate scoring clips flattened segments against all relevant rectangles.
Scoped obstacle collections own rectangle copies and immutable numeric bounds;
vectorized clipping avoids repeatedly constructing bounds for each segment/card
pair. Small or invalid geometry retains the scalar implementation as a fallback
and independent clipping oracle.

## Recorded performance

The [machine-readable summary](benchmarks/wire-routing-2026-10-09.json) retains
timing samples, input/source hashes and the measurement boundary. Source-imported
Windows checks use offscreen Qt and build saved workflows in their ready
presentation without executing scientific operations. Each full pass explicitly
invalidates route caches; optional cProfile passes are excluded from timings.

| Ready workflow | Nodes / visible wires | Full pass before clipping optimization | Final full pass | Background request | Final layout completion | Maximum 10 ms timer gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local refined workflow | 27 / 29 | 21.26 ms | 14.74 ms | 0.25–0.26 ms | 31.61–32.63 ms | 20.91 ms |
| Bundled exhaustive inspector | 164 / 124 | 601.16 ms | 422.01 ms | 1.08–1.23 ms | 386.05–401.10 ms | 12.78 ms |

Full-pass columns are medians of three baseline and two final repetitions. Final
background geometry exactly matched the synchronous rendered paths in both
workflows. The inspector's GUI snapshot took 3.95–4.26 ms. Initial profiling
identified 295,470 scalar segment/rectangle checks and 31,441 repeated scene-bound
queries in one inspector pass; these baseline passes needed no visibility search.

These modest repetitions establish a local regression baseline, not timing
guarantees or frame-rate measurements. Concurrent test activity caused an earlier
79.6 ms timer-gap outlier. The final recorded run's smaller gap does not establish
a worst-case latency bound. Actual drawing, native Linux/macOS, installed or
frozen builds, and arbitrary large/pathological workflows are not qualified by
this benchmark. The user's live editor was not used for these measurements.

## Reproduction and regression evidence

With development dependencies installed, run from the repository root:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:QT_QPA_PLATFORM = "offscreen"
python scripts/benchmark_wire_routing.py `
    examples/exhaustive-inspector-showcase.json `
    --repeat 3 --background --output .cache/wire-routing.json
```

Use `--profile-dir .cache/wire-profiles` for an additional profiled pass, or
`--phase initial` for presentation before ready-state thumbnails and details.
The local refined input is retained outside the tracked fixtures; its recorded
hash identifies that measurement without publishing machine-specific paths.

Focused regressions cover independent source/output separation, shared source
trunks, card/badge clearance, stable synchronous/background geometry, immediate
drag endpoints, stale result rejection, coalescing, cancellation and destruction.
Twelve collision tests verify thin rectangles, edge/corner tangency, irregular
routes against scalar clipping, owned bounds and invalid geometry fallback.
Fifteen background-routing cases passed independently. These records do not
claim completion of the full repository suite.
