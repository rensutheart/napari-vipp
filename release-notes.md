# VIPP 0.14.0a3

VIPP 0.14.0a3 makes volumetric cropping more responsive and memory-aware, adds safe node bypass across compatible workflows, improves ordinary workflow editing, and qualifies VIPP against napari 0.9 while retaining the declared napari 0.6 minimum.

This remains alpha software. Keep original data and workflows, verify the published checksum before running an unsigned installer, and review important outputs before using them for scientific conclusions or publication.

## Responsive volumetric cropping and bounded source reads

- Crop Stack now supports explicit Z-start and Z-end margins while preserving time, channel, calibration, origin, history, and backward-compatible zero-Z behavior.
- Crop edits show an immediate 2D or 3D ROI guide, commit one undoable scientific change after interaction, and expose a presentation-only outline-thickness control for low- and high-resolution images.
- For one strictly eligible direct local OME-Zarr `Image Source → Crop Stack` path, VIPP reads only the exact retained level-0 window while preserving the full logical source identity and the same Crop output, metadata, history, cache identity, provenance, batch behavior, and generated execution as the eager path.
- When a complete eligible source does not fit the safe RAM budget, Image Source can offer an explicit centred fitted Crop Stack as a reviewable starting point; VIPP never silently crops data or claims that the proposed region is scientifically appropriate.
- Crop selection and preview handling are hardened against napari layer-model re-entrancy failures.

## Safe node bypass and batch execution profiles

- Compatible fixed-single-output processing nodes can now be bypassed without deleting or rewiring them; the exact primary input, metadata, and device residency pass through while incompatible sources, writers, terminal nodes, tunnels, and dynamic or multi-output operations fail closed.
- Bypassed cards retain a presentation-only would-run thumbnail plus a badge, faded dotted treatment, and pass-through cue, while hypothetical preview pixels remain outside downstream analysis, caching, timing, export, batch output, and provenance.
- The Batch workspace adds explicit **Use workflow / Run / Bypass** profiles recorded separately from authored graph intent.
- Workflow schema 6 and batch configuration/manifest schema 5 preserve bypass intent and continue to read supported earlier versions.

## Workflow editing improvements

- `Ctrl+S` now saves from graph, inspector, or viewer focus without colliding with napari shortcuts; ordinary saves overwrite the current JSON by default, while Settings can require confirmation or create timestamped versions.
- Undo and Redo restore parameter edits in place, retain unaffected cards, thumbnails, viewer layers, and caches, and recalculate only the changed node and its descendants.
- Slider scrubbing creates one Undo step when the gesture completes rather than recording intermediate calculated values.
- Partial Image Source axis text remains an editor draft until a complete valid mapping is committed.

## napari 0.9 and Qt compatibility

- CI now qualifies the retained `napari==0.6.0` boundary, exact `napari==0.9.0`, and the latest supported napari, including plugin-manifest, import, application start/close, and focused real-viewer integration.
- VIPP remains binding-neutral through `qtpy`, with PyQt6 exercised on Windows/Linux and PySide6 on macOS.
- Generated Image and Labels layers carry VIPP axis names when supported, including correct displayed rank, RGB/RGBA component exclusion, in-place layer reuse, and protection against hidden previews replacing the selected scientific layer's labels.
- Camera and native-window access now use feature-detected public paths with bounded fallbacks for older supported napari versions.

## Installers and packages

- The explicitly unsigned Windows installer is rebuilt from the exact 0.14.0a3 wheel with the napari-0.9-compatible Qt toolchain.
- Separate explicitly unsigned, unnotarized, CPU-only macOS packages are provided for Apple Silicon and Intel, each with architecture-specific checksums and release evidence.
- The exact wheel and source archive are available from both the GitHub prerelease and PyPI.

## Qualification scope and remaining limits

The changed release domains are core/UI behavior, workflow schema and provenance, source I/O and memory planning, shared CPU/GPU execution coordination, dependency and Qt compatibility, installers, and documentation. GPU provider kernels and their admitted scientific regions are unchanged and retain their recorded qualification; exact a3 distributions, Windows setup, both macOS packages, checksums, manifests, versions, and public URLs are regenerated from the immutable release tag.

Exact source-window pushdown currently requires one sole direct, non-bypassed local OME-Zarr Crop Stack with explicit compatible axes. Branches, tunnels, unsupported readers, remote stores, and ambiguous or stale evidence use the ordinary full-read path only when memory preflight permits it. General lazy or chunked graph execution remains future work.

The Windows and macOS installers are convenience alpha artifacts and are not code-signed; the macOS packages are also not notarized and remain CPU-only.
