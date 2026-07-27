# Image Import And Export

Last reviewed: 2026-07-08

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
| OME-Zarr 0.4/0.5 | Discovers image groups and label groups, reads multiscale levels lazily, and marks label groups as label images. Level 0 is the analysis image. |
| NPY/NPZ | Reads one NPY array or a selected NPZ member. |
| PNG/JPEG/BMP/GIF/WebP/TGA/PNM | Reads ordinary raster images through imageio/Pillow. RGB/RGBA files are treated as rendered color images; grayscale files are treated as intensity images. Animated raster files use a leading time axis. |

Microscope acquisition formats use optional reader packages so the base VIPP
install stays lighter and avoids forcing proprietary-format dependencies onto
every user. If a required reader is missing, VIPP shows an optional-reader
dialog with a copyable install command. Restart napari after installing a new
reader, then reopen the file.

| Format family | Extensions | Install command |
| --- | --- | --- |
| Zeiss CZI | `.czi` | `pip install "napari-vipp[czi]"` |
| Nikon ND2 | `.nd2` | `pip install "napari-vipp[nd2]"` |
| Broad microscope reader set | `.czi`, `.nd2`, `.lif`, `.lof`, `.xlif`, `.oir`, `.oib`, `.oif`, `.vsi` | `pip install "napari-vipp[microscope]"` |
| BioIO/Bio-Formats fallback | Leica/Olympus/Bio-Formats-backed sources | `pip install "napari-vipp[bioformats]"` |

Use the format-specific extra when you know what you need. Use
`napari-vipp[microscope]` on a workstation intended to open mixed acquisition
formats.

For multi-series TIFF or multi-image OME-Zarr, select the required item in
`Series / image`. Time, channel, and Z remain axes inside that item. Use graph
nodes such as Select Axis Slice to subset them reproducibly.

Ordinary raster formats are also available as export targets only for 2D
intensity images and 2D RGB/RGBA images. Use OME-TIFF, ImageJ TIFF, TIFF,
OME-Zarr, or NPY for stacks, metadata-rich outputs, and exact numeric exchange.

Interactive file-path sources use a pinned scientific snapshot. VIPP verifies
the exact file or directory-store contents before and after both inspection and
loading, fully materializes the selected series into an owned read-only NumPy
array, and reuses that array for the resolved path and series. An external file
or store change is therefore not silently mixed into a running graph. Press
`Refresh` to discard the pinned snapshot, inspect the new revision, and load it
explicitly. OME-Zarr, microscope formats, and large files are materialized on
the background queue; small files normally retain synchronous loading.

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
when that node should receive a different file for every batch item. A blank
row is reproducible only when that `Image Source` already uses a fixed local
file path; napari-layer and bundled-sample sources must be bound to a collection
before saving or running a batch config.

The main toolbar places `Batch workspace...` between workflow loading and the
separate export actions. It is the single entry point for opening or returning
to the retained workspace; the representative strip only navigates samples and
reports batch progress.

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
action is available as `Open batch demo...` inside this dialog.

Existing demo directories are never replaced. Loading is confirmed because it
replaces the current graph. After execution, the app validates the bundle
inputs, scientific outputs, config/workflow hashes, manifest records, archive,
and sidecars and shows the pass/fail result in the batch summary. The selected
working-copy location remains available for inspecting those artifacts.

When multiple source rows are bound, VIPP sorts the matched files for each row
and pairs them by position. Each bound source must match the same number of
files, so item 1 uses the first file from every bound source, item 2 uses the
second file from every bound source, and so on. The first bound source is the
primary source used for default naming. Each item gets a stable batch index
(`0001`, `0002`, ...) and a stable batch id such as `0001_field_a`.

`Preview batch` is optional. Use it when you want to inspect item ids, bound
source files, planned output filenames, existing-path collision state, and one
representative graph calculation before execution. Planning itself does not
process or save the collection. The persistent strip above the graph says
`Representative only - this does not run or save the batch.`, shows `Item N of
M`, the batch ID, and every paired filename. Moving its slider calculates only
the selected representative
through the live graph. It does not create batch outputs. The workspace table
shows up to the first 25 plan rows, while the slider covers the complete plan.
`Run batch` always performs a fresh, plan-only preflight and does not require a
representative preview. If there is no displayed plan, it immediately executes
that fresh plan in the same click. If an already displayed plan changed
unexpectedly through files, destinations, collision states, or the scientific
graph, VIPP refreshes the table and stops so you can review it before clicking
Run again. Editing batch settings or the graph deliberately invalidates the
old runnable plan, but the slider stays available as an explicitly labelled
view of the previous source pairing and Run can build and execute a new plan.
Files opened as representatives are pinned to their verified revision. If one
is overwritten in place after review, Run stops and asks you to refresh while
the graph keeps showing the earlier verified bytes rather than silently mixing
revisions.

The workspace remains available during and after execution. Its determinate
progress bar reports completed items, the `Run status` column tracks each
displayed row, and the final summary retains completed/partial/skipped/failed
counts, validation text, and the manifest path. Progress is item-level; a long
single item can therefore remain on `running` until that graph invocation
finishes. On smaller displays, the workspace body scrolls vertically while
`Run batch` and `Close` remain fixed at the bottom. Reopen the same workspace
from the main toolbar's `Batch workspace...` button.
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

Use `Save config...` to write a versioned `vipp_batch_config.json`, and `Load
config...` to restore it. The configuration records the source-node bindings,
folders and patterns, output folder, default image format, existing-file
policy, required workflow companion, optional runner choice, workflow hash, and
resolved declarations for the selected outputs. Loading it against a different
workflow reports the hash mismatch instead of silently using stale output
selections.

When a Batch workspace is active, `Save workflow...` asks whether to include
that validated batch config inside the workflow JSON. `Yes` creates one file;
loading it restores and opens the workspace without planning or calculating a
representative. `No` saves the ordinary graph-only workflow, and `Cancel` saves
nothing. The attached config contains local input/output paths but not input
pixels, so review those paths before sharing or moving the file. Keep using
standalone `Save config...` when a separate config and headless runner are
needed.

`Continue after item failures` is enabled by default. Clear it only when a
pipeline exception or failed output should stop execution; intentional skips
alone do not stop the run. Any items not attempted after that point are
recorded as skipped.

The existing-file policy applies wherever a `Batch Output` node uses `batch
default`:

| Policy | Existing planned destination |
| --- | --- |
| `Error` | Report a collision and require it to be resolved before execution. |
| `Skip` | Preserve the file and record the planned output as `skipped`. |
| `Overwrite` | Replace the file and record the new write normally. |

An explicit `yes` or `no` overwrite value on a `Batch Output` node overrides
that default. Preview the batch again after changing either policy.

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
errors. A run-id manifest preserves each finished run. During execution, a
run-id sidecar directory checkpoints each item and its outputs. There is a
small interruption window between promoting an output and updating its
sidecar, so the sidecars are a recovery trail rather than a transaction log.
After a process interruption, inspect that run-id sidecar directory for the
last checkpoints; the canonical latest/archive manifests are finalized only
when the runner exits normally.
Output records move through `pending` to `completed`, `skipped`, or `failed`.
Item records may also be `running` or `partial`; the final summary counts
completed, partial, skipped, and failed items separately.

The dialog always writes:

- `vipp_batch_workflow.json`: the workflow graph and node positions;

It can additionally write:

- `vipp_batch_pipeline.py`: a thin command-line runner that defaults to the
  workflow recorded by the config and delegates to the same headless batch core
  as the dialog. `--workflow` can override that recorded path.

This batch runner is intentionally different from `Export Python...`. The
export embeds a validated immutable workflow and executes it through the same
headless pipeline service as VIPP, while its command-line folder harness is a
primary-source convenience rather than the complete multi-source collection
configuration used by batch runs.

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

- OME-Zarr pyramid generation and preview-level selection are not exposed.
- The graph still materializes lazy arrays when an eager processing node or
  preview requires NumPy data.
- Plate/well/field browsing, HCS traversal, remote URIs, and semantic-axis
  batch iteration remain planned work.
