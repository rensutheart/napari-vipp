# Research And Publication Record

Last reviewed: 2026-09-17

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

PSF-aware deconvolution now has an implemented foundation but still needs
reference and real-data validation before it can support publication claims.
Registration remains later scope.

The mitochondrial measurement target is also exploratory and statistical: VIPP
should eventually extract selectable high-dimensional object, intensity,
surface, and network features, merge them into one per-object table, and support
downstream analyses such as PCA or treatment-group separation. The detailed
roadmap is tracked in [mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Publication Intent And Citation Practice

An eventual VIPP paper remains an explicit project goal. Preserve the sources,
design rationale, limitations, and versioned evidence needed to write it as
development proceeds; a planned feature is not a demonstrated paper result.
Use primary methodological and software references, cite the original datasets
and their licenses, and identify the exact versions or commits actually used.
Credit conceptual inspiration separately from borrowed or adapted code. Verify
publication details and prefer a final published paper over an earlier preprint
when it supports the claim; keep attribution and license records for reuse.

A relevant conceptual reference is:

- Marcotti S, Gerontogianni L, Kelly G, Barry DJ (2026). **Practical statistics
  for bioimage analysis – a guide to experimental design and data
  interpretation.** *Journal of Cell Science* **139**(10), jcs264367.
  [doi:10.1242/jcs.264367](https://doi.org/10.1242/jcs.264367)
  ([published full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13286366/)).
- Companion code: FrancisCrickInstitute, **Enhancing-Reproducibility**, reviewed
  at commit [`a81d20c9393485e57395cd4a1be9e7ff83c7f11d`](https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility/tree/a81d20c9393485e57395cd4a1be9e7ff83c7f11d).

The paper motivates explicit experimental units, appropriate controls,
independent repetition, effect sizes, and uncertainty alongside computational
traceability. It does not validate VIPP, its statistical implementations, or any
VIPP workflow. Claims about VIPP still need release-specific tests, independent
reruns and, where relevant, assay validation. The case-specific sample sizes
are not universal acquisition thresholds or stopping rules.

Review source behavior before adapting the illustrative notebooks: the
[effect-size sampling code](https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility/blob/a81d20c9393485e57395cd4a1be9e7ff83c7f11d/notebooks/utility_functions.py#L335-L362)
uses repeated sampling without replacement and plots median/25th–75th percentile
bands (IQR, not confidence intervals); its
[dependencies](https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility/blob/a81d20c9393485e57395cd4a1be9e7ff83c7f11d/requirements.txt)
are unpinned. Cite the originating method, document adaptations, and validate
experimental-unit assumptions and interval semantics in any VIPP implementation.

The [tentative 0.17 plan](planning.md#planned-017-reproducibility-and-publication-support)
tracks possible support for these practices. This is future planning, not
implemented behavior or a release promise. Additional study-design metadata,
decision reasons, and publication notes must remain optional and must never
block processing, saving, collecting results, exporting, or ordinary workflow
use, and must not require an acknowledgement. Show missing reasons as
**User did not specify**; do not invent them or silently imply review. Preserve
the actual decisions and execution evidence even when a reason is absent.
This documentation principle does not relax existing scientific input checks
or supply missing information needed for a requested statistical calculation.

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
- calibrated extended object morphology columns for physical centroids,
  bounding boxes, equivalent diameters, selected object lengths/areas/volumes,
  and defensible 2D physical perimeter variants;
- analytical phantom validation for calibrated morphology using exact
  rectangle/cuboid phantoms plus tolerance-based sphere/ellipsoid phantoms:
  [analytical-phantom-validation.md](analytical-phantom-validation.md);
- two-channel colocalization metrics, Costes thresholding, RACC-like index
  outputs, ROI-restricted variants, object-restricted colocalization tables,
  label-overlap association, nearest-object distances, and event localization;
- named typed input ports and intensity-aware per-object measurement tables;
- generic skeletonization, skeleton-network measurement tables, skeleton
  keypoint masks, branch/component labels, and short-branch pruning with 2D/3D
  graph node, graph edge, isolate, and cycle metrics;
- metadata-aware 2D/3D label cleanup;
- OME-TIFF, ImageJ TIFF, conventional TIFF, OME-Zarr 0.4/0.5, and NumPy I/O,
  plus common raster import and 2D raster export;
- TIFF series and OME-Zarr image selection;
- OME-TIFF axes, physical scale, channel names, and VIPP provenance round trips;
- ImageJ hyperstack axes and calibration;
- lazy OME-Zarr image reads;
- OME-Zarr label groups and image-plus-label analysis package export.
- Born-Wolf PSF generation, measured-PSF preparation, and 2D/3D
  Richardson-Lucy/Richardson-Lucy-TV execution with deterministic examples.

Not yet evidence-backed and therefore not suitable as paper claims:

- usability improvements over existing tools;
- scalability to whole-slide or very large volumetric datasets;
- numerical equivalence to Fiji, CellProfiler, scikit-image, or other tools
  across a benchmark corpus;
- independent, publication-level validation of collection batch replay across
  machines and facilities (the implementation already has deterministic local
  execution evidence);
- numerical or biological validation of deconvolution outputs on a reference
  corpus;
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

- `docs/README.md`: audience-oriented documentation index;
- `README.md`: product overview and installation entry point;
- `docs/io-user-guide.md`: user-facing import/export behavior;
- `docs/planning.md`: current priorities and milestones;
- `docs/node-roadmap.md`: algorithm/node prioritization;
- `docs/architecture.md`: developer implementation reference;
- `docs/analytical-phantom-validation.md`: deterministic analytical phantom
  validation report for calibrated object and mesh morphology measurements;
- `docs/colocalization-method-notes.md`: publication-facing method definitions
  for implemented colocalization, RACC-like, ROI, object-restricted, and
  object-association calculations;
- `docs/ome-io-plan.md`: accepted scientific I/O architecture and status;
- this file: research questions, evidence, benchmarks, and publication record.

Do not place unverified performance or usability claims in user documentation.
When a feature changes, update its user guide, architecture status, changelog,
tests, and evidence status together.
