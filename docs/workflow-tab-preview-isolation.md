# Workflow-tab preview isolation

## Failure and scope

Differently sized cached workflows shared napari's camera without retaining a
per-tab view. A real ViewerModel/offscreen Qt regression reproduced a small
ZYX workflow returning to the larger Crop workflow's framing instead of its
own tuned center and zoom. Generic napari `reset_view()` is insufficient:
retained native layers, including invisible layers, contribute to global bounds.

Crop retirement had a separate synchronous event hazard. Removing its owned
layers while the outgoing Crop remained selected could call the dims handler,
which was allowed to recreate that Crop ROI during removal. Ordinary real-event
removal cases passed before the fix; a bounded injected point notification tests
this callback ordering explicitly. The user's exact intermittent persistent
outline sequence was not independently reproduced without that injection.

## Presentation contract

- Capture each tab's camera center, zoom, angles and perspective, plus dimension
  order, display mode and world-coordinate slice points, before layer retirement.
- Keep this immutable snapshot in the tab's runtime cache only. It is not workflow
  schema, history, scientific parameters, cache identity or serialized output.
- Retire outgoing Crop/source-preview/Inspect/pin presentation and invalidate
  queued selection work before binding the incoming session. Crop retirement
  suppresses synchronous dims callbacks and direct Crop-surface recreation.
- Restore once, after the incoming selected surface exists. Queued callbacks
  require the current session, selection generation, selected node and exact
  target layer, even when different tabs reuse node IDs.
- Restore dimensions before camera fields, since napari display-mode callbacks
  can modify the camera. Adapt rank by right alignment and clamp nondisplayed
  points to the selected target's calibrated sample-center bounds.
- Fit a first-visited tab to its selected layer, not the global layer union. Use
  calibrated world bounds, affine pixel-edge expansion and scene-unit conversion.
  Three-dimensional fitting uses camera up/right projections of those bounds.
  It follows orthographic-style framing, not an exact perspective-frustum solver.
- Never adopt another session's Crop source/ROI. Ownership is metadata-based;
  user layers sharing the same display name are neither claimed nor deleted.
- Closing an active Crop tab uses the same retirement boundary. Ordinary
  same-node refreshes do not arm a tab view restore or reset a tuned camera.

Global napari dimension ranges can still include a retained, unrelated native
image. The fix preserves that user layer and frames/clamps the selected target;
it does not delete user data to force the global ranges to shrink.

## Verification boundary

Focused source-imported Windows checks use real napari ViewerModel and offscreen
Qt, alongside existing tab, source-preview, Crop, inspection-profile and camera
contracts. Synthetic modest-sized YX/ZYX inputs exercise differing extents without
allocating large research volumes. Assertions retain cached object identities,
read-only image bytes, user annotations, scientific settings and zero tab-switch
calculations. Evidence is under
`VIPP-local-tests/workflow-tab-preview-20261004` outside the checkout.

The completed focused checks cover 15 new tab/Crop integration cases, 20 view
helper cases, 48 existing tab/dims/selection/compatibility/display cases, 57
selected existing widget contracts and 26 documentation cases: **166 distinct
passing cases**, not a full-suite run. Repository Ruff, manifest and whitespace
checks passed. The companion manual passed 16 content contracts, 50 routes and
a strict build. Original failing camera evidence remains preserved.

This is not a fresh full suite, installed-wheel/frozen-GUI qualification or manual
acceptance of the open VIPP session. Native Linux/macOS, actual vispy 3D canvas,
very large-volume performance and minimum napari/dependency versions are not
qualified by these checks. The live session and its environment remain untouched.
