# Research And Publication Record

Last reviewed: 2026-06-21

This file is the persistent record for a future VIPP journal paper and public
online documentation. It separates design intent from claims that have been
demonstrated by code, tests, user studies, or benchmarks.

## Research Aim

VIPP investigates whether a napari-native visual graph can make multidimensional
bioimage workflows easier to construct, inspect, reproduce, and batch-execute
without hiding axis semantics or scientific metadata.

Primary workflow domains:

- nuclei and cell segmentation;
- puncta and spot analysis;
- mitochondrial object and network analysis;
- pixel-based and object-based colocalization;
- 2D images, true 3D fluorescence z-stacks, and time/channel dimensions.

Registration and deconvolution are deliberately later scope.

The mitochondrial measurement target is also exploratory and statistical: VIPP
should eventually extract selectable high-dimensional object, intensity,
surface, and network features, merge them into one per-object table, and support
downstream analyses such as PCA or treatment-group separation. The detailed
roadmap is tracked in [mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Design Principles To Evaluate

1. The graph is the visible record of processing decisions.
2. Masks, labels, images, tables, and future points have distinct port types.
3. Axis and physical-scale metadata travel with arrays and are transformed when
   operations change dimensionality.
4. Interactive tuning and batch execution use the same headless operations and
   I/O layer.
5. OME formats are primary scientific exchange formats; ImageJ TIFF remains an
   explicit interoperability mode.
6. Source metadata and workflow provenance are retained without being
   misrepresented as current output properties.

## Evidence Status

Implemented and covered by automated tests:

- editable typed graph execution, workflow persistence, and Python export;
- semantic image/mask/label outputs and multi-output channel routing;
- typed Mask Image with named image/mask ports and RGB/multichannel mask
  broadcasting;
- image/mask/label pinning as persistent napari preview layers while editing
  other nodes;
- first-class table outputs and basic label-object measurements;
- named typed input ports and intensity-aware per-object measurement tables;
- generic skeletonization and skeleton-network measurement tables with 2D/3D
  graph node, graph edge, isolate, and cycle metrics;
- metadata-aware 2D/3D label cleanup;
- OME-TIFF, ImageJ TIFF, conventional TIFF, OME-Zarr 0.4/0.5, and NumPy I/O,
  plus common raster import and 2D raster export;
- TIFF series and OME-Zarr image selection;
- OME-TIFF axes, physical scale, channel names, and VIPP provenance round trips;
- ImageJ hyperstack axes and calibration;
- lazy OME-Zarr image reads;
- OME-Zarr label groups and image-plus-label analysis package export.

Not yet evidence-backed and therefore not suitable as paper claims:

- usability improvements over existing tools;
- scalability to whole-slide or very large volumetric datasets;
- numerical equivalence to Fiji, CellProfiler, scikit-image, or other tools
  across a benchmark corpus;
- reproducible collection batch execution;
- complete OME metadata fidelity;
- HCS plate/well/field interoperability;
- biological validity of segmentation or measurement workflows;
- mitochondrial-specific normalized network measurements.

## Evaluation Plan

Maintain versioned benchmark workflows and datasets for:

- 2D nuclei segmentation;
- 3D nuclei segmentation with touching-object separation;
- puncta detection and per-cell assignment;
- mitochondrial morphology/network measurements;
- high-dimensional object feature extraction for PCA/treatment-group analysis;
- two-channel colocalization;
- TIFF, OME-TIFF, ImageJ TIFF, and OME-Zarr metadata round trips.

For each benchmark record:

- source dataset, license, checksum, and citation;
- expected axis order, scale, units, and channel identities;
- workflow JSON and software environment;
- numerical outputs and tolerance;
- runtime, peak memory, and hardware;
- comparison implementation and parameter mapping;
- known failure modes.

Usability evaluation should measure task completion, error rate, time, and
participant understanding of axes, labels, and provenance. Any study involving
participants must use the appropriate institutional ethics process.

## Reproducibility Artifacts

Target paper release artifacts:

- tagged source release and archived DOI;
- environment lock files for supported platforms;
- versioned workflow JSON files;
- benchmark data acquisition scripts or stable public dataset references;
- machine-readable result tables;
- generated figures and analysis scripts;
- user guide and API/developer documentation;
- CITATION.cff and preferred citation text;
- explicit limitations and data-format support matrix.

## Documentation Structure

Use these persistent roles:

- `README.md`: product overview and installation entry point;
- `docs/io-user-guide.md`: user-facing import/export behavior;
- `docs/planning.md`: current priorities and milestones;
- `docs/node-roadmap.md`: algorithm/node prioritization;
- `docs/architecture.md`: developer implementation reference;
- `docs/ome-io-plan.md`: accepted scientific I/O architecture and status;
- this file: research questions, evidence, benchmarks, and publication record.

Do not place unverified performance or usability claims in user documentation.
When a feature changes, update its user guide, architecture status, changelog,
tests, and evidence status together.
