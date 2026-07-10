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

`Binding: collection` marks an Image Source node as the per-item source for
`Run batch...`. Interactively it still uses the selected file or series as the
representative item; in the batch dialog VIPP binds each matched folder item to
that source node and runs the same graph once per item. If no Image Source node
is marked as a collection, the first Image Source node is used as the folder
input for convenience.

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

`Run batch...` executes the current graph over local image collections. The
dialog shows one source row for each `Image Source` node in the workflow. Bind a
source row to a folder and one or more glob patterns, separated by semicolons,
when that node should receive a different file for every batch item. Leave a
source row blank when that source should keep its current fixed layer, file, or
sample data.

When multiple source rows are bound, VIPP sorts the matched files for each row
and pairs them by position. Each bound source must match the same number of
files, so item 1 uses the first file from every bound source, item 2 uses the
second file from every bound source, and so on. The first bound source is the
primary source used for default naming. Each item gets a stable batch index
(`0001`, `0002`, ...) and a stable batch id such as `0001_field_a`.

Use `Preview batch` before running to check the item ids, bound source files,
and planned output filenames. The preview does not execute the image-processing
graph; it only resolves the file pairing and output paths.

Add `Batch Output` nodes to mark the exact images, masks, labels, RGB outputs,
or tables that should be saved. Each `Batch Output` marker is pass-through
during normal graph execution and can define a tag, optional subfolder, filename
template, format override, and overwrite behavior. If the graph has no
`Batch Output` nodes, VIPP falls back to saving terminal graph outputs for every
matched item. Image-like fallback outputs use the dialog format; table fallback
outputs are saved as CSV.

Default explicit-output naming is:

```text
{source_stem}__{tag}
```

Supported filename-template fields are `{batch_id}`, `{batch_index}`,
`{source_name}`, `{source_stem}`, `{primary_source_stem}`, `{tag}`,
`{node_id}`, and `{node_title}`. VIPP appends the appropriate extension unless
the template already includes a known image or table extension.

The dialog can also write two reproducibility companions into the output
folder:

- `vipp_batch_workflow.json`: the workflow graph and node positions;
- `vipp_batch_pipeline.py`: the same headless Python export used by
  `Export Python...`.

Current batch execution is intentionally local-file oriented. It does not yet
provide stable plate/well/field identities, per-item provenance manifests, HCS
plate traversal, or iteration over semantic axes such as each timepoint or each
channel.

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
- Plate/well/field browsing, remote URIs, saved batch configuration,
  semantic-axis iteration, and per-item provenance remain planned work.
