# Image Import And Export

Last reviewed: 2026-08-27

VIPP uses one headless I/O layer for interactive sources, quick saves, Save
Image nodes, and exported Python scripts. The explicit format choice matters:
OME-TIFF, ImageJ TIFF, and conventional TIFF are different formats with
different metadata and compatibility goals.

## Import

The Image Source node supports napari layers, bundled samples, and local files
or stores.

Supported file sources:

| Source | Current behavior |
| --- | --- |
| OME-TIFF | Reads image series, semantic axes, physical scale, channel names and selected acquisition metadata. |
| ImageJ TIFF | Reads hyperstack axes, Z spacing, frame interval, unit, and XY resolution where present. |
| TIFF | Reads independent TIFF series and basic axes. |
| OME-Zarr 0.4/0.5 | Discovers image and label groups plus declared levels/transforms. A lower local level can be used for presentation while level 0 remains the analysis image; label previews retain label semantics. |
| NPY/NPZ | Reads one NPY array or a selected NPZ member. |
| PNG/JPEG/BMP/GIF/WebP/TGA/PNM | Reads ordinary raster images through imageio/Pillow. RGB/RGBA files are treated as rendered color images; grayscale files are treated as intensity images. Animated raster files use a leading time axis. |

Microscope acquisition formats use optional reader packages so the base VIPP
install stays lighter and avoids forcing proprietary-format dependencies onto
every user. If a required reader is missing, VIPP shows an optional-reader
dialog with the required extra and a generic pip suggestion. Use the safer
environment-bound commands below with the Python interpreter from the
environment that launches VIPP, never a global/base Python. Keep the VIPP
release pinned, restart napari after installing a new reader, then reopen the
file. The commands below install the current published `0.14.0a2` release. A
source checkout should use its managed development environment. A future
installer should expose these as Add/Repair components rather than terminal
steps.

| Format family | Extensions | Install command |
| --- | --- | --- |
| Zeiss CZI | `.czi` | `python -m pip install "napari-vipp[czi]==0.14.0a2"` |
| Nikon ND2 | `.nd2` | `python -m pip install "napari-vipp[nd2]==0.14.0a2"` |
| Broad microscope reader set | `.czi`, `.nd2`, `.ims`, `.lif`, `.lof`, `.xlif`, `.oir`, `.oib`, `.oif`, `.vsi` | `python -m pip install "napari-vipp[microscope]==0.14.0a2"` |
| BioIO/Bio-Formats fallback | `.ims` and Leica/Olympus/Bio-Formats-backed sources | `python -m pip install "napari-vipp[bioformats]==0.14.0a2"` |

Use the format-specific extra when you know what you need. Use
`napari-vipp[microscope]` on a workstation intended to open mixed acquisition
formats.

The 0.14.0a1 qualified reader contract is intentionally narrower
than the list of advertised extensions:

| Reader route | 0.14.0a1 evidence | Important limit |
| --- | --- | --- |
| Nikon ND2 | Lazy inspection/data, decoded-size estimate, stable items, calibration, channels, and objective metadata | Optional `nd2` dependency |
| Leica LIF | Inspection/read calibration, channels, objective metadata | Eager pixels; `liffile` and Bio-Formats may expose different item topology, so recorded backend changes are refused for review |
| Zeiss CZI / LSM | Stable CZI scenes and LSM main/RGB-thumbnail items with metadata parity | Native pixels are eager; no CZI pyramid claim |
| Olympus OIR / OIB | Authoritative TCYX/CZYX contracts with calibration, channels, and objective/acquisition facts | Native pixels are eager |
| Olympus OIF / VSI | Companion trees are part of exact source revision; VSI primary/macro items and metadata are retained | VSI needs optional Java/codecs and its large decode remains cache-gated |
| Imaris IMS | Logical TCZYX shape, metadata parity, decoded-size reporting, actionable Java readiness errors | IMS pyramid enumeration and cheap lower-level reads are not claimed |

LOF/XLIF and broad Bio-Formats extensions remain capability-advertised, not
qualified scientific claims, until a licensed hash-frozen acceptance source is
added.

For multi-series TIFF, NPZ, microscope containers such as LIF/IMS, or
multi-image OME-Zarr, select the required item in `Series / image`. VIPP records
a SourceItem v1 stable selector, reader key/version, normalized axes and shape,
and exact container revision. A saved item resolves by stable key; changed
bytes, missing companions, ambiguous legacy indices, or an unexpected reader
topology fail visibly instead of selecting another image. Time, channel, and Z
remain axes inside that item. Use graph nodes such as Select Axis Slice to
subset them reproducibly.

Ordinary raster formats are also available as export targets only for 2D
intensity images and 2D RGB/RGBA images. Use OME-TIFF, ImageJ TIFF, TIFF,
OME-Zarr, or NPY for stacks, metadata-rich outputs, and exact numeric exchange.

Interactive file-path sources use a pinned scientific snapshot. VIPP verifies
the exact file or complete directory/companion bundle before and after
inspection and analysis loading, then materializes the selected level-0 series
into an owned read-only NumPy array and reuses it for that SourceItem revision.
An external change is therefore not silently mixed into a running graph. Press
`Refresh` to discard the pinned snapshot, inspect the new revision, and load it
explicitly.

For a local multiscale OME-Zarr image or label, VIPP may first display a sliced
lower level labelled `Preview level N - analysis remains full resolution`.
That result is presentation-only: it never replaces SourceItem analysis level
0, scientific cache data, or output provenance. Superseded preview/load work is
cooperatively cancelled and generation-checked before publication. A
single-level source reports that no lower-level preview exists. OME-Zarr,
microscope formats, and large files use the background queue with decoded-memory
preflight and truthful progress; a monolithic eager reader does not invent an
internal percentage.

`Binding: collection` marks an Image Source node as a per-item source for
`Batch workspace...`. The graph still represents one scientific item at a
time. After planning, VIPP swaps the complete set of paired collection paths
into those Image Source nodes as a transient representative and runs the same
graph once per batch item during full execution. The transient paths are not
written into the workflow or its scientific hash. A separate Batch Input node
is therefore unnecessary. If no Image Source node is marked as a collection,
the first Image Source node is used as the folder input for convenience.

## Export Choices

| Format | Use when |
| --- | --- |
| OME-Zarr | The image is large, chunked access matters, or it will later form part of an image-plus-label analysis package. Version 0.4 is the default writer. |
| OME-TIFF | A portable single processed image with OME-XML metadata is required. This is the default quick-save format. |
| ImageJ TIFF | Direct ImageJ/Fiji hyperstack behavior is the priority. Binary masks are written as `uint8` values `0` and `255`. |
| TIFF | Broad TIFF compatibility or preservation of 32-bit integer label IDs is required. |
| NPY | Exact array exchange is needed and scientific image metadata is not required. |
| PNG/JPEG/BMP/GIF/WebP/TGA/PNM | A 2D display image is needed. PNG can preserve 16-bit grayscale values and label IDs up to 65535; JPEG/WebP/BMP/GIF/TGA-style outputs are 8-bit display exports. JPEG cannot store alpha. |

ImageJ TIFF cannot safely represent 32-bit integer label IDs. Use conventional
TIFF, OME-TIFF, or Export OME Analysis Dataset for those labels.

## Collection Batch Runs

`Batch workspace...` configures and executes the current graph over local image
collections. The retained workspace shows one source row for each `Image
Source` node in the workflow. Bind a
source row to a folder and one or more glob patterns, separated by semicolons,
when that node should receive a different file for every batch item. The default
`*` includes recognized image files and directory stores such as `.ome.zarr`;
narrow the pattern only when the folder contains sources that should not join
the batch. A blank row is reproducible only when that `Image Source` already
uses a fixed local file path; napari-layer and bundled-sample sources must be
bound to a collection before saving or running a batch config.

The main toolbar places `Batch workspace...` between workflow loading and the
separate export actions. It is the single entry point for opening or returning
to the retained workspace; the representative strip only navigates samples and
reports batch progress.

After a fresh plan resolves stable SourceItems, `Per-sample parameters
(optional)` shows eligible authored numeric scientific controls as columns and
primary source items as rows. Enter only values that differ for an item; a blank
cell visibly inherits the saved workflow value. Source paths/selectors,
destinations, compute/cache controls, derived fields, expressions, and topology
cannot be overridden. Every value uses the node's normal integer/float contract,
and duplicate, stale, zero-match, multi-match, or invalid rows stop preflight.
Preview and execution apply overrides to a detached effective workflow, so the
live and saved base workflow remain unchanged. The effective values and hashes
are retained in config, checkpoint, manifest, and item provenance.

When a workflow includes its Batch workspace, reopening it automatically detects
the current samples in a background, metadata-only preflight and restores the
table without requiring `Preview batch`. A compact status and activity indicator
sit at the right of the fixed Batch toolbar, labelling that source-discovery work
separately from the main VIPP graph-calculation bar; the two jobs may run
concurrently. The status remains visible after fast operations as
`Ready - N batch items`, the adjacent indicator becomes indeterminate for
discovery and preflight, and it mirrors overall item progress during a full run.
The detailed per-item and per-operation bars remain in the Batch run section. No
representative pixels are calculated until explicitly requested.
If the workspace includes per-sample values, the same preflight also checks
their keys against the current exact source revisions.
Saving those values freezes the reviewed collection inventory, so changing an
inherited row is not silently overlooked. A changed or missing source keeps the
saved values quarantined, disables Run, and explains that no value was reassigned
by filename or collection order.

The easiest way to explore batching is `Open example...` -> `Deterministic
Batch & Provenance` -> `Open batch demo...`. Choose where to save the demo's
small working copy; VIPP then opens the batch workspace with its two-source
workflow and config loaded. The graph automatically displays the first paired
NumPy field through every connected node. Use `Previous`, `Next`, or the
representative slider to move through all three pairs; both source paths change
together. Selecting a table row and clicking `Preview selected in graph` (or
double-clicking the row) performs the same representative calculation. The
highlighted demo guide points to `Run demo batch` and describes
the nine planned NPY/TIFF/TSV outputs, saved config and runner, manifests,
archive, per-item provenance, and exact ground-truth validation. The same
action is available as `Demo...` inside this dialog.

Existing demo directories are never replaced. Loading is confirmed because it
replaces the current graph. After execution, the app validates the bundle
inputs, scientific outputs, config/workflow hashes, manifest records, archive,
and sidecars and shows the pass/fail result in the batch summary. The selected
working-copy location remains available for inspecting those artifacts.

When multiple source rows are bound, VIPP sorts the matched files for each row
and expands any multi-series container into its inspected image items. It then
pairs those image items by position. Each bound source must therefore resolve
to the same number of image items. A batch row shows both the container name
and series/scene name, and output names include the series identity so two
images from the same file cannot collide. The first bound source is the primary
source used for default naming. Each item gets a stable batch index (`0001`,
`0002`, ...) and a stable batch id.

Each collection source has an `Image stack` choice. A new unsaved source row
starts at `Automatic (recommended)`. If a representative reports exactly `QYX`
and then reaches a workflow step that explicitly requires `ZYX`, VIPP selects
`Stack planes are depth slices (Z stack)`, retries the check, and shows a notice
explaining that the concrete choice will be saved. This is a narrow
workflow-based suggestion, not evidence that every TIFF page is scientifically
a depth slice. Select `Use the file's labels unchanged` to opt out; VIPP respects
that decision rather than selecting Z again for that source.

The resulting `QYX -> ZYX` declaration is guarded: the raw side must match
exactly for every item. It changes semantic names in place and never transposes
pixels. It also does not discover a missing Z calibration.
`Something else (advanced)...` keeps uncommon mappings out of the normal
workflow and reveals its text field only when selected. `Reorder Axes` performs
the opposite kind of change: it transposes pixels with their complete axis
records but never renames Q as Z.

`Preview batch` is optional. Use it when you want to inspect item ids, bound
source files, planned output filenames, existing-path collision state, and one
representative graph calculation before execution. Planning itself does not
process or save the collection. The persistent strip above the graph says
`Representative only - this does not run or save the batch.`, shows `Item N of
M`, the batch ID, and every paired filename. Moving its slider calculates only
the selected representative
through the live graph. It does not create batch outputs. The workspace table
shows up to the first 25 plan rows, while the slider covers the complete plan.
`Run batch` always performs fresh planning and a representative scientific-axis
preflight, and does not require a representative preview. Planning
metadata-inspects every matched supported container to resolve and verify its
SourceItems without reading every image's complete pixels. It then applies
source declarations and checks the first item's graph contract before creating
the output directory, run artifacts, or a CPU/GPU device context. A
deterministic inventory or contract error stops the run even under `Continue
after item failures`; an unreadable item remains governed by that continuation
choice. The graph calculation remains representative-only, and each later item
is reverified when read. If there is no displayed plan, Run immediately executes
that fresh plan in the same click. If an already displayed plan changed
unexpectedly through files, destinations, collision states, or the scientific
graph, VIPP refreshes the table and stops so you can review it before clicking
Run again. Editing batch settings or the graph deliberately invalidates the old
runnable plan, but the slider stays available as an explicitly labelled view of
the previous source pairing and Run can build and execute a new plan.
Files opened as representatives are pinned to their verified revision. If one
is overwritten in place after review, Run stops and asks you to refresh while
the graph keeps showing the earlier verified bytes rather than silently mixing
revisions.

The workspace remains available during and after execution. Its determinate
overall progress bar reports collection items, while a second current-operation
bar identifies the active item, node/operation, and truthful synchronized
checkpoint. The `Run status` column tracks each displayed row, and the final
summary retains completed/partial/skipped/cancelled/failed counts, validation
text, and the manifest path. A monolithic library call or file writer may
finish its current call before the operation bar moves; VIPP does not invent
internal percentages. `Cancel run` sets the shared cooperative token and waits
for synchronization and cleanup before finalizing the cancelled item and
manifest. On smaller displays, the workspace body scrolls vertically while
`Run batch`, `Cancel run`, and `Close` remain fixed at the bottom. Reopen the
same workspace from the main toolbar's `Batch workspace...` button.
After a run, its preflight and row statuses remain visible as historical run
evidence. Run preflights current paths again before replay; use Preview first
only when you want to inspect them.

Add `Batch Output` nodes to mark the exact images, masks, labels, RGB outputs,
or tables that should be saved. Each `Batch Output` marker is pass-through
during normal graph execution and can define a tag, optional subfolder, filename
template, format override, and overwrite behavior. If the graph has no
`Batch Output` nodes, VIPP falls back to saving terminal graph outputs for every
matched item. Image-like fallback outputs use the dialog format; table fallback
outputs are saved as CSV. This fallback preserves older and ad-hoc workflows,
but the preview warns because terminal-node selection can change when the graph
is edited. A terminal node with multiple output ports is rejected because the
fallback cannot identify which port to save. Use explicit `Batch Output` nodes
for a saved, reviewable run.

Default explicit-output naming is:

```text
{source_stem}__{tag}
```

Supported filename-template fields are `{batch_id}`, `{batch_index}`,
`{source_name}`, `{source_stem}`, `{primary_source_stem}`, `{tag}`,
`{node_id}`, and `{node_title}`. VIPP appends the appropriate extension unless
the template already includes a known image or table extension.

Use `Save...` to write a versioned `vipp_batch_config.json`, and `Load...` to
restore it. The configuration records the source-node bindings,
folders and patterns, output folder, default image format, existing-file
policy, required workflow companion, optional runner choice, workflow hash, and
resolved declarations for the selected outputs. Config schema version 4 adds
canonical SourceItem records and typed per-sample parameter overrides to the
version-3 guarded source-axis declarations and full compute request, including
CPU/Auto/Prefer-GPU/Custom mode, fallback policy, per-node preferences,
runtime/device, memory cap/reserve, policy IDs, and experimental admission.
Version-1 configs migrate to explicit CPU because they had no accelerator
intent; version-2 configs retain their saved compute request; version-3 configs
retain their source declarations and acquire SourceItems when resolved. Older
versions contain no parameter overrides and become version 4 when reviewed and
saved. Loading against a different workflow reports the hash mismatch instead
of silently using stale output selections.

The friendly GUI choice is saved as the same explicit declaration used by
headless runs. Code that constructs a source binding directly can use
`AxisDeclaration("QYX", "ZYX")`; a JSON config stores the corresponding
`"source_axes": "QYX"` and `"effective_axes": "ZYX"`. Headless execution does not
invent that declaration when it is absent. Saving while the recommended choice
has not needed to change anything conservatively stores no declaration; loading
that config shows `Use the file's labels unchanged`, not automatic mode.
The same is true for historic or headless configs with no declaration. Once the
guarded Z-stack suggestion has been applied, saving records the concrete
`QYX -> ZYX` declaration and loading restores
`Stack planes are depth slices (Z stack)`.

In the GUI, the loaded config's compute request remains effective while the
toolbar compute request is unchanged from load time. Changing any toolbar
compute setting selects the complete current toolbar request for the next
preview, save, or run. In headless replay, the config request is the default and
an explicit function or CLI override applies only to that invocation. The
manifest keeps both configured and effective requests plus separate saved and
effective config hashes.

When a Batch workspace is active, `Save workflow...` asks whether to include
that validated batch config inside the workflow JSON. `Yes` creates one file;
loading it restores and opens the workspace, then starts background
metadata-only source discovery. That refreshes the current sample plan without
calculating a representative image or running the graph. `No` saves the
ordinary graph-only workflow, and `Cancel` saves nothing. The attached config
contains local input/output paths but not input pixels, so review those paths
before sharing or moving the file. Keep using standalone `Save...` when a
separate config and headless runner are needed.

`Continue after item failures` is enabled by default. Clear it only when a
pipeline exception or failed output should stop execution; intentional skips
alone do not stop the run. Any items not attempted after that point are
recorded as skipped.

The existing-file policy applies wherever a `Batch Output` node uses `batch
default`:

| Policy | Existing planned destination |
| --- | --- |
| `Ask before overwrite (recommended)` | In the Batch workspace, list the exact existing outputs and ask before replacing them for this run. Cancel preserves every file. Headless execution retains the underlying fail-closed `error` policy because it cannot ask. |
| `Skip existing` | Preserve the file and record the planned output as `skipped`. |
| `Overwrite without asking` | Replace the file and record the new write normally. |

An explicit `yes` or `no` overwrite value on a `Batch Output` node overrides
that default. Duplicate output destinations, outputs overlapping inputs, and
explicitly protected outputs are never made replaceable by the confirmation.
Preview the batch again after changing either policy.

A run started from the dialog writes the resolved configuration into the output
folder:

- `vipp_batch_config.json`: the resolved configuration used for that run;

Every dialog or headless execution writes:

- `vipp_batch_manifest.json`: the latest run metadata plus per-item and
  per-output status.

A headless replay uses the existing config and workflow files at their recorded
locations rather than copying them into the output folder.

The manifest identifies the workflow/config hashes, embeds the canonical config
and scientific graph, records VIPP and relevant runtime package versions, each
input and available source metadata, every planned output policy/path, and
errors. Manifest schema version 4 adds canonical SourceItem/source-revision
evidence plus requested/effective parameter overrides and effective workflow
hashes to the version-3 raw/effective axes, source declarations, compute
request/environment, exact implementation identities, fallback reasons,
structured OOM retry/memory records, warnings, and cleanup proof. The embedded
config retains intended declarations for sources skipped or failed before
reading. A run-id manifest preserves each finished run. During execution, a
run-id sidecar directory checkpoints each item and its outputs. There is a
small interruption window between promoting an output and updating its
sidecar, so the sidecars are a recovery trail rather than a transaction log.
After a process interruption, inspect that run-id sidecar directory for the
last checkpoints; the canonical latest/archive manifests are finalized only
when the runner exits normally.
Output records move through `pending` to `completed`, `skipped`, `cancelled`,
or `failed`; item records may additionally be `running` or `partial`. Each
published output record has
`provenance_status: produced` and an `execution_provenance_sha256` link to the
item's full execution document. The final summary counts completed, partial,
skipped, cancelled, and failed items separately.

The dialog always writes:

- `vipp_batch_workflow.json`: the workflow graph and node positions;

It can additionally write:

- `vipp_batch_pipeline.py`: a thin command-line runner that defaults to the
  workflow recorded by the config and delegates to the same headless batch core
  as the dialog. `--workflow` and `--config` select artifacts;
  `--compute-mode`, `--fallback-policy`, and repeatable
  `--node-preference NODE_ID=PREFERENCE` override compute intent for one run;
  and `--progress` prints both item and current-operation streams. One `Ctrl+C`
  requests cooperative cancellation and returns exit code 130 after cleanup.

This batch runner is intentionally different from `Export Python...`. The
export embeds a validated immutable workflow and executes it through the same
headless pipeline service as VIPP, while its command-line folder harness is a
primary-source convenience rather than the complete multi-source collection
configuration used by batch runs. The convenience loop hashes its local source
before reading, verifies the identity after materialization, and privately
stages and rollback-protects the complete requested output/sidecar set. It does
not provide multi-source pairing, collision planning, a final source recheck
immediately before publication, checkpoints, manifests, or automatic replay
guarantees. Use the saved batch runner for
production collection processing. See
[Durable GPU execution](durable-gpu-execution.md) for commands and the complete
fallback/provenance/cancellation contract.

Under visible fallback, a classified retryable runtime OOM is synchronized,
cleaned, and retried once for that complete segment on CPU; strict policy does
not retry. Both outcomes are structured in the item execution record. No output
is promoted when GPU cleanup is false or unknown. The next batch item retains
the saved request and is planned independently rather than inheriting the
previous item's CPU fallback.

Current batch execution remains local-file oriented. Time, channel, and Z stay
inside each paired source item; VIPP does not yet iterate selected semantic-axis
combinations. Plate/well/field discovery and HCS traversal are also deferred.

## Export OME Analysis Dataset

`Export OME dataset...` writes one reference image plus every available graph
label output into a single `.ome.zarr` store:

```text
/
  s0                      reference image level 0
  labels/
    nuclei/
      s0                  label image level 0
    cells/
      s0
```

Label outputs are written as OME-Zarr `image-label` groups, retain integer label
IDs, and include a source relationship back to the reference image. VIPP also
stores label-node identity and operation history in namespaced provenance.

Version 0.4 is the default export target; 0.5 is available in the export dialog.

## Metadata Policy

VIPP distinguishes current structural metadata, preserved acquisition facts,
and original source/provenance metadata. Writers emit metadata that remains
valid for the processed output. They do not restore obsolete source dimensions
or channels after cropping, projection, splitting, or other transformations.

The selected-node inspector shows the normalized metadata used by the graph.
Raw OME metadata is retained by the dataset reader for provenance but is not
presented as editable output metadata.

## Current Limitations

- Lower-level presentation preview is limited to local OME-Zarr 0.4/0.5. The
  graph still materializes the complete selected level-0 analysis array.
- Remote stores, IMS pyramid enumeration/cheap reads, and general
  operation-level lazy or chunked graph execution are not included.
- Native LIF, CZI, OIR, OIB, and LSM pixel reads remain eager even though their
  inspection metadata and capabilities are explicit.
- Per-sample overrides are numeric scalar parameters only; expressions,
  filename rules, CSV import/export, source selectors, and topology changes are
  deferred.
- Plate/well/field browsing, HCS traversal, and semantic-axis batch iteration
  remain planned work.
