# Detect Spots per Frame: implementation contract

Status: unreleased tracking-slice development. This is an engineering record,
not a second user guide or a biological detection-validation claim.

The canonical node/operation ID is `detect_spots_per_frame`, with the
`detect_spots_per_frame` function in `core/time_detection.py`. It processes
an explicit scalar `TYX` or `TZYX` input using the unchanged `core/detection.py`
scalar functions. Local peaks mode
calls `find_peaks` on each spatial frame. Template match mode calls
`template_match` and then `find_peaks` with the paired validity mask, using one
fixed supplied scalar `YX` or `ZYX` template. The scientific definitions,
authoritative references, numerical conditioning, template-center offset,
plateau ordering, exact separation and cap-last policies remain those in the
[scalar detection contract](detection-implementation.md).

The naming change does not alter the detection algorithm, observation evidence
or scientific limits.

No time registration, channel selection, resizing, permutation, smoothing,
temporal filtering, interpolation, or tracking occurs. Source and template
must already be materialized NumPy inputs with matching explicit metadata;
lazy inputs are rejected before any implicit array materialization. Time
must be the leading named axis with type `time`, and all trailing canonical
spatial axes must have type `space`. Numeric and value-domain restrictions
remain the scalar detector's rules, including finite values and exact-range
wide integers. A stored template-response stack is not a source-intensity
input. Local peaks mode rejects an attached template rather than ignoring it.

## Table and population contract

The single `time_detections` table prepends `t_index` to the scalar detector's
columns. `detection_id` restarts at one in every frame. It is not a persistent
track identity. Coordinates retain the original input spatial index lattice;
even-sized templates retain half-index centers and exclusive template bounds.
When spatial length units are known, physical columns and native per-axis
units are those produced by the scalar detector.

`ObservationSeriesMetadata` records the spatial source shape, axes, scale,
origin and units; explicit frame count; time scale, origin and unit; named
source-index coordinate and ID columns; and whole-series source identity and
SHA-256 revision. The hash includes dtype, full shape and every C-order byte,
using bounded iteration for noncontiguous input. Source UUID takes precedence,
then URI/series identity, then a content-bound fallback. Unknown units remain
unknown; no physical units or acquisition times are invented.

`frame_populations` contains every zero-based frame in order, including empty
frames. Its eligible count is the exact post-separation, before-cap population;
retained count is the returned row count. `maximum_detections` applies per
frame, after all separation decisions, with an explicit truncation flag. A
capped result can be inspected/exported but is not a complete observation
population for linking. The single-image `DetectionMetadata` tag is not
misapplied to the combined table. Pipeline parameters/history retain the
authored detection settings and upstream fixed-template provenance.

## Resource and cancellation contract

One frame is exposed as a read-only view without modifying the original
buffer or its writeable flag. Template scores, validity and intermediate
states are released before the next frame starts; no full time-response stack
is assembled or returned. The input remains resident, and all returned rows
and small frame-population records accumulate until successful completion.

Conservative additional-allocation admission checks frame metadata initially
and retained row storage as each frame completes. Existing scalar guards
independently admit each frame's correlation and peak workspace against the
current host-memory snapshot. These are estimates, not measured large-volume
memory qualification. Exact whole-series hashing is an additional full read;
template matching also retains its existing per-frame revision work.

Scalar progress is mapped into a monotone whole-series range with frame
labels. Cancellation is checked while hashing, within scalar cooperative
boundaries, during row assembly, between frames and before returning. Native
FFT/filter/sort calls retain their existing non-interruptible boundaries. A
failure or cancellation returns no partial scientific table.

## Focused evidence and limitations

`_tests/test_time_detection.py` uses independently planted moving peaks and
fixed template patterns in 2D and 3D time series. It checks empty frames,
per-frame caps, exact centers/bounds/calibration, anisotropic physical
separation, source identities and full-series revisions, invalid axes and
numeric input, no input mutation, response lifetime with weak references,
resource refusals and cancellation. It also checks the CPU-manual node
declaration and unchanged scalar parameter defaults.

The new node is CPU-only. No GPU path, lazy/out-of-core series execution,
changing template bank, biological accuracy benchmark, representative
large-series performance qualification, or new claims about the scalar
algorithm are included. Workflow, batch/export, metadata persistence, linking
and review UI are qualified by their owning integration tests. The public
manual companion update belongs in `vipp-mkdocs`, not in this repository.
