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

The compact-route follow-up orders existing peer lane control polygons by
Manhattan length plus 42 pixels per turn before rendering and checking them.
This avoids accepting a distant perimeter merely because it was generated
before nearby gaps. Duplicate cleaned polygons are skipped. An excessive clear
peer detour adding more than 160 pixels beyond its card-only route compares at
most four further departure/arrival offsets with eight candidates each and
retains its lowest clear length-plus-bend score;
short clear candidates still return immediately. A clear forward route more than
80 pixels beyond its own
endpoint cards' vertical envelope can try a local 1,024-visit visibility search.
The candidate is retained unless that search finds a clear route with a lower
length-plus-bend score. Blocked routes keep the existing 12,000-visit fallback;
direct curves, short detours and backward loops avoid the extra improvement
search. Routes remain deterministic heuristics, not global shortest-path claims.

## Recorded performance

The timings and full-suite qualification below cover the earlier responsiveness
implementation. The compact-route follow-up is recorded separately at the end
of this document so its measurements are not attributed to that earlier source.

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
Fifteen background-routing cases passed independently. The final full repository
suite completed in one source-imported, offscreen Windows process: 13,459 passed,
31 skipped, and two expected failures for the existing CuPy integer Gaussian
parity gaps. There were no unexpected failures. Parallel attempts exhausted this
machine's system commit limit and were interrupted; only the completed
single-process run is counted here.

Ruff, plugin-manifest validation, and wheel/sdist builds passed. The companion
manual update is commit `d15bfb76ab407c694d807478a0cd3f98f62843c9` in vipp-mkdocs;
its content checks and strict build passed. The live development workflow was
opened with the user's saved positions and tunnel, both tunnel menu actions were
checked and undone, and calculated previews were left ready for testing.

## Compact-route follow-up

The staggered-card regression has an available clear cheap perimeter route, so
the previous collision-only fallback skipped a shorter local zigzag. With the
same endpoint/card geometry, the route now measures 1,119.41 pixels instead of
2,260.94 pixels and stays between Y=-40 and Y=40 instead of climbing to Y=-644.
Mirrored tests check both directions. Independent-output cases also verify that
an existing nearby lane is considered before a distant row's perimeter and that
a small departure shift can reach a local gap. Every replacement is checked for
endpoint direction, rendered card clearance and independent-source separation.

The additional peer refinement performs at most 32 candidate clearance checks.
Tests require short candidates and genuinely card-required perimeter routes to
avoid further offsets; exhausted improvement budgets retain the clear incumbent.
The visibility score limit includes bends, and cancellation remains active.

The [follow-up timing record](benchmarks/wire-routing-compactness-2026-10-09.json)
identifies the final graph source hash, input hashes and baseline `b27e012`.
Baseline and final modules were measured sequentially in the same source-imported
Windows/offscreen environment, with three ready-state repetitions per workflow.

| Ready workflow | Nodes / visible wires | Baseline full-pass median | Follow-up full-pass median | Follow-up background request | Follow-up completion | Maximum 10 ms timer gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local saved test workflow | 27 / 28 | 64.31 ms | 52.93 ms | 0.25 ms | 68.03–68.95 ms | 24.40 ms |
| Bundled exhaustive inspector | 164 / 124 | 417.65 ms | 471.98 ms | 1.06–1.12 ms | 412.01–499.78 ms | 84.39 ms |

The larger graph's full-pass median increased about 13% in this small sample;
the editor still requests work quickly and computes the final route in its
worker. Background and synchronous paths matched exactly. These measurements
show the bounded quality/CPU tradeoff, not a speedup or a worst-case UI latency
guarantee. The earlier full-suite result above predates this follow-up. Its
affected-domain checks cover graph routing/editing, background lifecycle,
tunnel conversion, history, example layouts and documentation: 442 passed with
no failures. Ruff, manifest validation and wheel/sdist builds passed. Twenty-four
presentation frames have no sampled settled card intersections; two during-drag
intersections occur in the existing endpoint-only transient preview mode.

The companion manual change is commit `13c0000f0f1239158efb96ae577718356dacc333`;
its content checks and strict build passed. No live user workflow was changed
to reproduce the compact-route regressions.
