# VIPP 0.16.0a3

VIPP 0.16.0a3 adds template-based detection, per-frame spot detection, bounded tracking and previous-frame time-series registration, with clearer result review and isolated workflow previews.

**Alpha software.** Preserve original images, saved workflows and decisive outputs. Check representative data before scientific interpretation.

## Detect structures and follow observations

- Template Match compares complete, fixed-size and fixed-orientation 2D/3D templates using normalized correlation. Scores retain calibrated template-centre coordinates and a paired valid-score mask; resemblance scores are not detection probabilities.
- Find Peaks provides explicit thresholds, pixel or physical separation, deterministic plateau handling and exact pre-limit counts. Review detections on their source image and add a connected Find Peaks node from the suggested next step without calculating automatically.
- Detect Spots per Frame detects each scalar TYX image or TZYX volume independently, using local peaks or one fixed template. Empty frames and capped populations are recorded explicitly; its detection IDs are local to each frame, not persistent object identities.
- Build Tracks links complete observations by position within authored displacement and missing-frame limits. It reports observation and track-summary tables, calibrated distance/time and ambiguous links. Review trajectories and linked table rows read-only; dashed gaps show observed endpoints, not inferred motion.
- Bundled 2D/3D examples demonstrate known detection locations, noise controls, motion, empty frames, gaps and competing links. Object measurements from explicit whole time series can also supply tracking observations while preserving original frame-local labels.

## Register adjacent time points

- Estimate Registration now offers Previous frame alongside Fixed reference. It estimates original adjacent volumes toward the chosen anchor and composes their physical-coordinate transforms; Apply Transform resamples each original volume only once, sharing the transform across channels.
- Diagnostics distinguish local-pair quality from cumulative alignment to the anchor. Local failures stop the result. Cumulative displacement and overlap can be reported or checked against the authored limits; adjacent estimation errors may accumulate, so review coverage and representative alignment.

## Review and workspace improvements

- Histogram cards have Open histogram; measurement, statistics, detection and tracking cards have Open measurements. Eligible time-series cards also expose Review trajectories. These shortcuts open cached results without recalculating or changing the selected node.
- Detection, tracking and registration sliders use practical tuning ranges while numeric fields retain wider valid values. Inspector summaries wrap fully, and unchanged connected-input cards stay in place during live tuning.
- Authenticated upstream results remain current after an initial synchronous CPU run, preventing the first selected-node recalculation from leaving Build Tracks waiting on otherwise current detections. Genuine input, parameter or compute-policy changes still invalidate results.
- Each workflow tab retains its own camera, slice position and 2D/3D view. Newly viewed tabs fit the selected image instead of unrelated large layers; Crop outlines and delayed presentation callbacks remain isolated between workflows. These are display-only changes, not scientific recalculation.

## Scope and compatibility

Detection and tracking remain CPU-only. Position-based linking cannot establish biological identity at crossings; no appearance or morphology model, prediction, identity editing, division/fusion lineage, rotation/scale template bank or GPU detection/tracking is added. Registration remains global rather than deformable. Synthetic known answers do not establish acquired-microscopy accuracy or large-volume performance. Runtime dependency pins and existing filtering/deconvolution methods are unchanged.

New nodes and parameters require this version. Preserve copies before resaving workflows; reproduction version acknowledgements and verified-resume restrictions remain in force. macOS remains CPU-only. Desktop installers are unsigned alpha builds, and macOS packages are unnotarized.

See the [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a3/docs/release-qualification-baseline.md), [detection evidence](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a3/docs/detection-implementation.md), [tracking evidence](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a3/docs/tracking-implementation.md) and [workflow-preview contract](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a3/docs/workflow-tab-preview-isolation.md) for the exact scope and limitations of checks. Verify platform downloads using the release checksum files.
