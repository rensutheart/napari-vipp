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

## Unnecessary-staircase follow-up

The cheap orthogonal catalogue now includes the arrival-side vertical turn as
well as the departure-side turn. This lets a clear wire stay horizontal until
near its destination, without adding visibility searches or increasing search
budgets. Existing rendered-clearance and independent-output lane checks still
filter the candidates, and the existing turn penalty selects the simpler route.

In the saved test layout, Logical XOR to Remove Small Objects changes from four
rounded corners to two. Both control polygons travel 1,195 scene pixels. The
rounded rendered path grows by 13.53 pixels because two corner cuts disappear;
the simpler route still has the lower existing length-plus-turn score. This is a
heuristic improvement, not a guarantee of globally minimal routes.

The [staircase timing record](benchmarks/wire-routing-staircase-2026-10-09.json)
records the baseline `0b3922e`, frozen source/input hashes and all samples. Its
27-node saved fixture predates the user's latest Combine Meshes placement.
Only the reported wire changes in that fixture; all 124 rendered paths in the
164-node exhaustive inspector remain unchanged. Both workflows have zero sampled
rendered card/tunnel/note intersections and zero independent straight-segment
overlaps longer than one pixel.

Three sequential repetitions gave saved-workflow full-pass medians of 70.60 ms
before and 69.90 ms after. The exhaustive inspector's synchronous median grew
from 441.34 to 507.07 ms (+14.9%), while its background completion median grew
from 416.79 to 418.50 ms (+0.4%). Five further repetitions with reversed order
gave synchronous medians of 508.60 and 530.18 ms (+4.2%) and background medians
of 485.90 and 478.15 ms (-1.6%). Maximum 10 ms heartbeat gaps were about 82 ms
before and 79 ms after in that second comparison. These small, variable samples
show a bounded extra candidate cost; they establish neither a speedup nor a
worst-case responsiveness bound. Background geometry matches synchronous
geometry in every measured pass, and neither workflow invokes visibility search.

Four regressions cover clear upward/downward late turns, a blocked source lane
and an occupied independent-output lane. Both clear-turn cases fail against the
baseline implementation. The affected-domain suite passed all 446 checks;
the earlier full repository run predates this follow-up. Ruff, plugin-manifest
validation and wheel/sdist builds passed. The companion manual change is commit
`b34045e5a1de3aee16f03b48365b9109160c6b77`; its 18 content checks, 50 route checks
and strict build passed.

The user's latest layout was saved before restarting the development tester
(workflow SHA-256 `92fd5af4092dac7a2b63cb08aac5b4515ad00e05d1e6e258a10323e9719704f4`).
The live source hashes match the tested files. After Calculate all, the UI
reports complete calculations and a 6.1 MB cache; all 27 output identities are
distinct, the 29 computational connections remain present and routing is
settled. The reported wire has two rounded corners and a 660-pixel initial
horizontal segment. One widget is registered in this maximized floating tester.
The original acquired-data recovery session was left untouched.

## Independent port-approach follow-up

A shared departure/arrival offset can help one end clear a card while adding
an unnecessary S connector and another turn at the clear end. After selecting
a clear peer route, each end can now try one port-aligned approach to its first
vertical segment. The remaining path is copied exactly. A replacement must
respect port direction, continue the original vertical direction and pass the
full rendered card/peer checks. The existing 42-pixel turn preference accounts
for the two removed connector direction changes. Each end is checked
independently, so a needed opposite detour is retained.

In the latest saved test fixture, Otsu Threshold to Erosion keeps its left
detour around Save Image. Its right departure now runs horizontally from
(4,171, 445.5) to the rounded turn near X=4,411, then rises directly to the
existing top route at Y=166.5. The former shifted S connector joined at
(4,375, 361.5), followed by another short step before reaching that column.
Rendered length grows from 3,114.66 to 3,176.30 pixels because the removed
S connector cut across its corner; the two fewer direction changes improve
the length-plus-turn preference. This is a bounded visual-quality choice,
not a globally shortest-path claim.

The refinement tries at most two prefixes on the chosen clear path. It does
not regenerate the candidate catalogue, add offsets or start visibility
searches. Cancellation and the existing background revision checks remain
active, and computational connections and cached scientific data are unchanged.

The [port-approach comparison](benchmarks/wire-routing-port-approaches-2026-10-09.json)
records baseline `985754d`, final source/input hashes and three sequential
repetitions per workflow. Only Otsu Threshold to Erosion changes in the saved
27-node/28-wire fixture; its left approach's final four control elements are
byte-identical. The exhaustive inspector changes only Threshold to Logical AND
input 0. Both workflows retain zero sampled rendered card/tunnel/note
intersections and zero checked independent straight overlaps longer than one
pixel. Every settled background layout equals its synchronous Qt geometry.

The latest saved layout remains expensive: baseline/final full-pass medians
are 2,313.44/2,302.92 ms and background completion medians are
3,094.20/2,393.88 ms. Both versions perform four visibility searches per pass
in that fixture, including the reported wire and mesh connections. The
exhaustive inspector's full-pass medians are 659.31/487.96 ms and its background
medians are 644.94/524.02 ms, with no searches. These sequential three-sample
timings are variable; they do not establish a speedup or latency bound. The
small post-clear refinement leaves search counts unchanged and does not solve
the saved fixture's existing multi-second routing cost.

Five new full-route regressions cover clear upward/downward departures, a
mirrored clear arrival and retained approaches when a card or distinct-source
run blocks the straight alternative. The three clear-approach cases fail the
baseline. The affected-domain suite passes all 451 checks. Ruff, manifest
validation and wheel/sdist builds pass; both built packages contain the tested
routing source. The earlier complete repository run predates this follow-up.
The companion manual change is commit
`006690a8bfab93f3dc750267c101251e4773919c`; its 18 content contracts, 50 route
checks and strict build pass.

The latest saved layout was reopened in a maximized development tester using
the exact tested source hashes. The UI reports complete calculations and a
6.1 MB cache, with 27 distinct calculated outputs, the same 29 computational
connections, one registered widget and settled background routing. The target
wire's first segment is horizontal and its useful left approach remains curved.
Both ends were brought into view for the user's next test. The original
acquired-data recovery windows remain open and untouched.
