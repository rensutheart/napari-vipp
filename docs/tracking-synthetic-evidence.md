# Synthetic tracking examples and acceptance

Status: unreleased. These examples test exact constructed positions and the
declared linking policy, not accuracy on acquired biological images.

`make_tracking_sample_data()` independently authors image arrays and truth
without calling detection, measurement or tracking. A separate **VIPP synthetic
tracking samples** manifest command exposes the same two samples included in
the general sample catalogue. All historical samples and examples are unchanged.

## Two-dimensional moving spots

The seven-frame `TCYX` sample includes an independent noise-control channel and
a moving-spots channel. The packaged workflow explicitly selects the latter,
runs **Detect Spots per Frame**, then **Build Tracks** with an
eight-pixel-per-frame gate and one allowed missing frame. Four integer-centered Gaussian spots are
present in frames 0, 1, 2, 4, 5 and 6; frame 3 has none. The resulting population
is 24 observations in four six-observation tracks. No synthetic row fills the
missing frame.

Two isolated trajectories have independently known track identities and speeds
of six and four pixels per second at the declared half-second sampling. The
lower pair crosses during the missing frame. Multiple associations satisfy the
position gate, and the minimum-distance policy can swap their construction
identities. Tests require review flags for this pair and do not assert
biological or construction-identity recovery through the crossing. A review
flag is not probability and cannot guarantee detection of every wrong link.

## Three-dimensional object centroids

The six-frame `TZYX` sample contains separate 3-by-3-by-3 cubes with changing
source label values. The packaged graph visibly thresholds values greater than
zero and labels connected components independently in each full 3D frame.
These explicit steps discard supplied label numbers and produce new local IDs;
they leave cube geometry and centroids unchanged. This uses the existing graph
type contract and does not introduce a new label-source declaration.

**Measure Objects** then feeds **Build Tracks**, using a two-micrometer-per-frame
gate and one missing frame. Counts are 2, 2, 1, 3, 3 and 3: fourteen observations
and three tracks with six, five and three observations. A moves one Z and X
sample per frame, B moves one X sample and is absent at frame 2, and C appears
at frame 3. Z/Y/X spacing is 1.5/0.5/0.4 micrometers and time sampling is 2.5
seconds. Thus A's step length is `sqrt(1.5^2 + 0.4^2)` micrometers; B's ordinary
step length is 0.4 micrometers, and its gap spans two sampling intervals.

The new object's earlier voxel position changes A's measured label ID from
one to two, while A retains track ID one. A separate direct-adapter test checks
that measurements of the supplied labels retain their original changing IDs;
the packaged graph makes no such preservation claim.

## Result handoff and checks

Both examples show tracked observations and the separate track-summary port.
A **Select Table Columns** consumer with `auto = all` exposes the complete,
unchanged summary as a primary node result. This supports generated Python's
existing primary-output contract without adding aggregation or changing the
generic exporter. No example writes files or edits source pixels.

`test_tracking_examples.py` checks exact coordinates, populations, unambiguous
IDs, calibrated motion, gaps, crossing flags, immutable inputs, packaged/repo
parity and generated-Python equality for both output tables. The existing
example-layout audit checks all four initial/ready new-example layouts.

`scripts/smoke_tracking_install.py` provides standalone installed-wheel
acceptance. It imports no tests, pytest, widget or example UI; it loads packaged
workflow resources and runs the shared executor. The CLI accepts
`--expected-package-root`, `--require-installed` and `--output-json` and verifies
the actual imported package and distribution roots. It reports versions,
exact source/track evidence, counts, centers, analytical speeds and unchanged
raw buffers. Windows local checks do not qualify other operating systems,
large-series resources, optional GPU routes or biological accuracy.

The same standalone script additionally samples an asymmetric analytical
Gaussian intensity field independently into five translated `TYX` frames. It
checks native **Previous frame** translation estimation on both sides of anchor
two, with anisotropic spatial spacing and nonzero spatial/time origins. Expected
moving-to-anchor physical matrices are authored from the known integer motion,
not obtained from the estimator. Matrix agreement and aligned intensities within
valid coverage must meet absolute tolerance `1e-12`, with unchanged raw input.
This small exact-translation acceptance does not qualify subpixel or rigid-motion
accuracy; those remain separate registration tests.
