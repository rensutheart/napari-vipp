# napari-vipp technical figures

This directory contains six publication-oriented figures for `napari-vipp`.
The SVG files are the editable vector masters; the matching PNG files are
high-resolution previews for browsers, slide software, and quick review.

Terminology follows the user interface and project documentation. **VIPP** is
the Visual Image Processing Platform; **napari-vipp** is its napari-native
implementation and package.

## Suggested publication order

1. **System overview** — the shortest introduction to inputs, the napari/VIPP
   workspace, execution, and results.
2. **Interactive-to-reusable workflow** — the main user and methods narrative.
3. **Processing and analysis pathways** — the scientific capability map.
4. **Worked processing example (optional companion)** — source-backed visual
   evidence for one concrete segmentation path. It can instead be presented as
   Figure 3b or moved to supplementary material when space is limited.
5. **Batch publication and provenance** — deterministic collection execution,
   integrity gates, and durable records.
6. **Software architecture** — the implementation-oriented component view.

This order moves from a reader-facing system explanation to increasingly
technical detail. The figures can also stand alone; their captions do not
depend on earlier figures.

## Figure 1 — System overview

**Files:**
[`napari-vipp-system-overview.svg`](napari-vipp-system-overview.svg) and
[`napari-vipp-system-overview.png`](napari-vipp-system-overview.png)

**Communicates:** What napari-vipp is, how the workflow editor and napari viewer
work together, where computation occurs, and which interactive and durable
results leave the system.

**Appropriate use:** Main system figure in a journal paper; project overview on
a poster; documentation landing page; introductory presentation slide.

**Suggested caption:** **Figure 1. System overview of napari-vipp.** Scientific
inputs from napari layers, local files and stores, bundled samples, and optional
readers enter a typed workflow within the napari workspace. The workflow
exchanges full-resolution layers with the viewer, executes with authoritative
CPU kernels or eligible optional GPU segments, and yields interactive layers,
saved images and tables, reusable workflows or generated code, and
provenance-bearing run records.

## Figure 2 — Interactive-to-reusable workflow

**Files:**
[`napari-vipp-workflow.svg`](napari-vipp-workflow.svg) and
[`napari-vipp-workflow.png`](napari-vipp-workflow.png)

**Communicates:** The primary user journey from source selection and graph
authoring through iterative inspection and tuning to reviewed interactive,
headless, or batch reuse.

**Appropriate use:** Central conference-poster figure; journal workflow figure;
user documentation; methods presentation.

**Suggested caption:** **Figure 2. Interactive authoring and reusable execution
of a VIPP workflow.** A researcher selects a revisioned scientific source,
constructs a typed graph, calculates and inspects intermediate outputs, and
iteratively refines the workflow. The reviewed graph and portable compute
intent can be reopened in VIPP, executed through generated Python or a command
line, or paired with a reviewed batch configuration; these routes use the same
scientific execution service. Workflow persistence excludes runtime image and
table caches.

## Figure 3 — Representative processing and analysis pathways

**Files:**
[`napari-vipp-processing-pathways.svg`](napari-vipp-processing-pathways.svg) and
[`napari-vipp-processing-pathways.png`](napari-vipp-processing-pathways.png)

**Communicates:** How compatible typed scientific data roles can branch and
recombine to turn multidimensional intensity images into masks, labels,
measurements, visual quality-control outputs, and analysis-ready tables.

**Appropriate use:** Scientific methods paper; workflow documentation; image
analysis poster; domain-focused presentation.

**Suggested caption:** **Figure 3. Representative composable processing and
analysis pathways in VIPP.** Prepared multidimensional images can branch through
restoration, segmentation and object separation, measurement, colocalization,
skeleton and network analysis, and table composition. Images, masks, label
images, and tables remain distinct typed data roles while compatible branches
can be omitted, rearranged, or recombined. The diagram is a capability map, not
a mandatory fixed pipeline; semantic axes and physical calibration remain part
of the scientific contracts.

## Figure 4 — Worked processing example (optional companion)

**Files:**
[`napari-vipp-processing-example.svg`](napari-vipp-processing-example.svg) and
[`napari-vipp-processing-example.png`](napari-vipp-processing-example.png)

**Communicates:** A source-backed, single left-to-right execution of the bundled
Portable GPU Segmentation Bridge workflow on deterministic checked-in data,
including intermediate images, orthogonal 3D cavity evidence, and quantitative
cleanup results.

**Appropriate use:** Optional companion to Figure 3; worked-example panel in a
methods paper; reproducibility supplement; technical poster inset. Renumber it
as Figure 3b when the pathway map and worked example are presented together.

**Suggested caption:** **Figure 4. Worked example of three-dimensional
segmentation cleanup.** The third channel of a deterministic CZYX `uint16`
sample is converted to `float32`, Gaussian blurred, thresholded at the bundled
workflow's fixed sample-specific value, cleaned by three-dimensional
small-object removal and hole filling, and labeled into four connected
components. The operations use the complete ZYX volume; independently
display-normalized maximum-intensity projections and real orthogonal XY/XZ/YZ
index views are used only for communication. Regenerated evidence confirms
removal of a 19-voxel speck, setting 31 enclosed background voxels to foreground
across three Z planes, and four final components; actual CPU/GPU decisions
remain execution provenance rather than an assumption made from the requested
compute mode.

## Figure 5 — Batch publication and provenance

**Files:**
[`napari-vipp-batch-provenance.svg`](napari-vipp-batch-provenance.svg) and
[`napari-vipp-batch-provenance.png`](napari-vipp-batch-provenance.png)

**Communicates:** How one reviewed workflow is applied to local collections and
how source reverification, private staging, cleanup evidence, and manifests
guard durable result publication.

**Appropriate use:** Reproducibility or provenance section of a journal paper;
technical documentation; software poster inset; batch-processing presentation.

**Suggested caption:** **Figure 5. Durable collection execution and provenance
in VIPP.** Source collections are bound and paired deterministically, output
collisions are checked, and a representative scientific contract is preflighted
before publication begins. Each item captures source identities, executes a
fresh workflow while recording actual implementations and cleanup evidence,
stages available outputs privately, reverifies every source, and promotes valid
artifacts one at a time. Failed integrity gates withhold new publication and are
recorded explicitly; promotion is atomic per artifact, so a later failure can
produce a documented partial multi-output item rather than an unreported
all-or-nothing result.

## Figure 6 — Software architecture

**Files:**
[`napari-vipp-software-architecture.svg`](napari-vipp-software-architecture.svg)
and
[`napari-vipp-software-architecture.png`](napari-vipp-software-architecture.png)

**Communicates:** The main conceptual software boundaries and how interactive
napari use, generated code, local collection batch processing, normalized I/O,
the scientific core, compute implementations, and durable outputs connect.

**Appropriate use:** Software-focused journal paper; developer documentation;
technical poster inset; architecture presentation.

**Suggested caption:** **Figure 6. Conceptual architecture of napari-vipp.** The
interactive napari dock, generated Python or command-line use, and local
collection batch execution converge on a shared core that is independent of Qt
and napari. The application layer manages napari adapters, background work, and
stale-result rejection, while the core contains the typed workflow graph and
operation catalog, isolated execution, and persistence and batch-publication
services. Normalized source payloads feed the core; execution selects
authoritative CPU kernels or eligible CUDA segments, and accepted results
become napari layers or durable images, tables, workflows, manifests, and
provenance records.

## Shared visual language

The figures use the same semantic palette and pair colour with text, shape, or
line style so that meaning does not depend on colour alone:

- navy: the napari host or viewer;
- blue: scientific image data, sources, and normalized I/O;
- violet: workflow authoring, interaction, and application control;
- green: scientific operations, execution, and compute backends;
- rose: masks, label images, and segmentation pathways;
- amber: quantitative results, persistence, and reusable artifacts;
- red: failed integrity checks or withheld publication;
- neutral grey: structural containers and entry surfaces;
- dashed outlines or arrows: optional or conditional paths.

The shared style definitions live near the top of
[`scripts/build_figures.py`](scripts/build_figures.py):

- typography: `.title`, `.subtitle`, `.section`, `.card-title`, `.body`,
  `.small`, and `.arrow-label`;
- geometry: `.card`, `.mini-card`, `.band`, and `.outer-frame`;
- semantic fills: `.navy`, `.blue`, `.violet`, `.green`, `.rose`, `.amber`,
  `.red`, and their band variants;
- connectors: `.connector` and its semantic colour variants, with `.optional`
  and `.soft-dash` for conditional paths;
- supporting marks: `.divider`, `.port`, `.image-frame`, and `.table-grid`.

## Semantic groups in the SVG masters

The SVGs use ordinary vector elements organized into named top-level `<g>`
groups. Connectors are separated from cards so their stacking order can be
maintained while editing.

- **System overview:** `scientific-inputs`, `napari-workspace`,
  `workflow-execution`, `results-and-reuse`, and `scientific-context-band`.
- **Workflow:** `interactive-authoring-steps`, `reuse-routes`, and
  `workflow-definition-note`.
- **Processing pathways:** `prepared-image-hub` and separate preparation,
  segmentation, specialized-analysis, and table-composition lanes.
- **Worked example:** `worked-example-stages` and `example-evidence-notes`.
- **Batch and provenance:** `run-level-planning`, `per-item-loop`,
  `integrity-outcomes`, and `batch-footer-note`.
- **Architecture:** separate interactive, generated-code, and batch entry
  groups, plus `shared-core` and `supporting-systems`.

Each figure also has a `title-band` and a dedicated connector group. Preserve
these boundaries when moving components or changing the stacking order.

## Editing in Affinity Designer

1. Open an SVG master as a document rather than placing it as a single object.
   The named SVG groups should remain available as groups in the Layers panel.
2. Keep connector groups below component-card groups. Move an entire semantic
   group when changing layout, then adjust its connector paths and arrow labels.
3. Keep text editable and use the shared Arial/Liberation Sans font stack.
   Check for reflow if Affinity substitutes another font; convert text to curves
   only in a final submission copy when a publisher requires it.
4. Reuse the established semantic fills, strokes, corner radii, and dashed
   optional style. Do not assign a new colour to each peer component.
5. The worked-example raster panels are embedded in its SVG. Regenerate their
   source assets and rebuild the SVG instead of manually retouching scientific
   image content.
6. `scripts/build_figures.py` overwrites the six canonical SVG masters. Save
   exploratory Affinity edits under a different name, or reflect accepted
   changes in the generator before rebuilding.
7. Export preview PNGs at 2× each SVG's native viewBox dimensions, retaining the
   white background. Inspect both the SVG and PNG at the final publication width
   before delivery.

## Rebuilding the figures and worked-example evidence

Run the following from the repository root in a napari-vipp development
environment with its Python dependencies available:

```powershell
python docs/figures/scripts/generate_processing_example_assets.py
python docs/figures/scripts/build_figures.py
python docs/figures/scripts/qa_figures.py
```

The first command, implemented by
[`scripts/generate_processing_example_assets.py`](scripts/generate_processing_example_assets.py),
regenerates the deterministic worked-example PNG assets and
[`assets/portable-gpu-segmentation-bridge/evidence.json`](assets/portable-gpu-segmentation-bridge/evidence.json).
It uses the checked-in `VIPP synthetic GPU segmentation cleanup` sample and the
authoritative VIPP operations named by the bundled Portable GPU Segmentation
Bridge workflow. It fails if the expected voxel counts or component volumes
change. Cleanup and connected-component labeling use the complete ZYX volume;
the generated projections and orthogonal index views are display-only.

The second command uses [`scripts/build_figures.py`](scripts/build_figures.py)
to read that evidence, embed the worked-example images into its SVG, and
rewrite all six SVG masters. It does not render the matching PNG previews;
export those separately after reviewing the regenerated SVGs. Run the asset
generator before the figure builder whenever the sample or scientific
operations change. The final command performs read-only structural QA on all
six SVGs, including unique IDs, view boxes, marker definitions, semantic title
and description elements, forbidden editor-specific markup, and the embedded
worked-example images.

## Recommended lead figures

- **Journal paper:** Figure 1 is the strongest compact system figure; use
  Figure 6 when software boundaries are the main subject.
- **Conference poster:** Figure 2 works best as the central narrative, with
  Figure 3 or the optional Figure 4 providing the scientific example.
- **Reproducibility supplement:** Figures 4 and 5 provide the most concrete
  scientific and publication-integrity evidence.

## Scope and interpretation

The diagrams abstract the implementation into scientific concepts rather than
Python modules. They intentionally show optional GPU computation—not an AI or
learned-model subsystem—because the current system uses classical image
processing with authoritative CPU kernels and qualified CuPy/CuPyX
acceleration. Figure 3 presents representative composable pathways rather than
a fixed VIPP pipeline. Figure 4 is one deterministic, sample-specific worked
example and is not a general threshold recommendation. Figure 5 gives
provenance its strongest emphasis on generated and batch publication routes; it
does not imply that every informal interactive save produces the complete batch
artifact set.
