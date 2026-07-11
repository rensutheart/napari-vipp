# First-Class OME Import And Export

Status: accepted architecture; phases 1-3 foundations implemented

Last reviewed: 2026-07-10

This document defines how VIPP should treat OME-TIFF, OME-Zarr, and conventional
TIFF/ImageJ files as first-class image sources and outputs.

The design goal is not merely to read pixel arrays. VIPP should preserve,
display, transform, and export metadata that remains valid after processing,
while retaining original acquisition metadata and workflow provenance without
misrepresenting them as current output properties.

## Recommended Product Contract

VIPP should expose these distinct formats:

| Format | Import | Export | Primary use |
| --- | --- | --- | --- |
| OME-Zarr | Yes | Yes | Large, chunked, multiscale datasets; image plus label analysis packages |
| OME-TIFF / BigTIFF | Yes | Yes | Portable OME image exchange with rich OME-XML metadata |
| ImageJ TIFF | Yes | Yes | Direct Fiji/ImageJ hyperstack compatibility |
| Conventional TIFF | Yes | Yes | Broad TIFF compatibility and integer label images |
| NumPy NPY/NPZ | Yes | Yes | Internal/debug array exchange with limited metadata |
| Common raster PNG/JPEG/BMP/GIF/WebP/TGA/PNM | Yes | 2D only | Convenience I/O for ordinary display-image sources |

OME-TIFF and ImageJ TIFF must remain separate export modes. They use different
metadata conventions. Fiji can open OME-TIFF through Bio-Formats, but a file
written as an ImageJ hyperstack is not equivalent to an OME-TIFF with full
OME-XML metadata.

## Implementation Status

Implemented:

- one headless I/O registry used by file sources, quick save, Save Image, and
  generated Python load/save helpers;
- content-based OME-TIFF, ImageJ TIFF, and conventional TIFF detection;
- adaptive TIFF series discovery and selection in Image Source;
- normalized axes, channel, acquisition, source, and provenance records;
- OME-TIFF import/export with axes, physical scale, channel metadata, source
  identity, and VIPP operation history;
- explicit ImageJ TIFF export with hyperstack ordering and calibration;
- conventional TIFF export that preserves 32-bit label IDs;
- common raster import/export through imageio/Pillow, with RGB/RGBA files
  marked as rendered color images, grayscale files marked as intensity images,
  and export limited to 2D intensity or 2D RGB/RGBA arrays;
- local OME-Zarr 0.4 and 0.5 image import/export;
- lazy Dask-backed OME-Zarr level access;
- OME-Zarr axes, scale, channel names, and namespaced VIPP provenance;
- OME-Zarr label-group import/export through image-plus-label analysis
  packages;
- a stored single-item/collection binding mode, with the selected item serving
  as the interactive representative.

Still incomplete:

- generated pyramids and preview-level selection;
- plate/well/field selectors;
- anonymous HTTP reads and other URI transports;
- operation-level lazy execution and memory-aware materialization;
- batch collection execution and multi-source pairing;
- full instrument/objective/detector reconstruction and per-plane metadata;
- metadata editing UI;
- metadata propagation through generated scripts remains less complete than
  interactive graph execution.

## Architecture

Add a headless I/O package rather than continuing to place format logic in the
Qt widget or generic processing operations:

```text
src/napari_vipp/core/io/
  __init__.py
  model.py          normalized dataset, series, and inspection records
  registry.py       format detection and reader/writer dispatch
  tiff.py           OME-TIFF, ImageJ TIFF, and conventional TIFF
  ome_zarr.py       OME-Zarr 0.4/0.5 image reader/writer
  raster.py         common PNG/JPEG/BMP/GIF/WebP/TGA/PNM import and 2D export
  numpy_io.py       NPY/NPZ support
```

The UI, graph source nodes, quick-save action, Save Image node, and generated
Python scripts should all call this same headless I/O layer.

### Dataset Result

Readers should return one normalized object rather than an array plus an
unstructured metadata dictionary:

```text
ImageDataset
  data                  NumPy or Dask-compatible array
  image_state           current normalized VIPP state
  acquisition           normalized acquisition metadata
  display               channel names, colors, windows, visibility
  source                 URI, format, series/image path, source UUID
  original_metadata      raw OME-XML or OME-Zarr attributes for provenance
  multiscale_levels      optional pyramid levels
  associated_labels      optional OME-Zarr label datasets
```

`SourcePayload` should carry an `ImageDataset` or a prebuilt `ImageState`, so
file metadata is parsed once by the reader rather than converted into an
ad-hoc napari metadata dictionary and parsed again by the pipeline.

### Normalized Metadata

Extend the current metadata model with structured records for:

- image/series name and description;
- channel name, color, fluorophore, excitation wavelength, emission wavelength,
  and display window;
- physical pixel/voxel sizes and units;
- time increment and unit;
- acquisition date;
- objective, instrument, detector, and acquisition settings when available;
- per-plane time, exposure, and stage position when available;
- source format, source URI, source image/series identifier, and source UUID;
- software creator and versions;
- VIPP workflow provenance and operation history.

Raw source metadata must also be retained separately. It must not be copied
blindly into the current output metadata after transformations.

## Metadata Validity Policy

Metadata should be divided into three classes:

1. **Current structural metadata**
   Axes, shape, dtype, scale, translation, units, channel count, and semantic
   image kind. VIPP transforms these fields as nodes crop, reduce, split,
   combine, convert, or label data.
2. **Preserved acquisition metadata**
   Channel names, wavelengths, acquisition date, instrument, objective, and
   detector information. These remain attached only when the operation leaves
   their meaning valid. For example, extracting one channel keeps that
   channel's metadata; combining unrelated sources must not pretend they came
   from one original acquisition.
3. **Original metadata and provenance**
   Raw OME-XML/OME-Zarr attributes, source identifiers, and workflow history.
   These remain available for audit but are identified as source metadata, not
   current output metadata.

Writers should emit the richest valid current metadata and attach source
metadata or the VIPP workflow as provenance. A writer must not restore original
dimension sizes, plane coordinates, or channel records that no longer match the
processed array.

## Import Design

### OME-TIFF Import

Use `tifffile.TiffFile` for pixel/series access and `ome-types` for typed
OME-XML parsing.

Import should support:

- `.ome.tif`, `.ome.tiff`, and OME BigTIFF variants;
- multiple image series/scenes;
- dimension order and pixel type;
- physical sizes, units, time increment, channel metadata, and image names;
- acquisition/instrument metadata when present;
- lazy or Zarr-backed access for large tiled/pyramidal TIFF where practical;
- explicit failure for malformed OME metadata, with an optional separate
  conventional-TIFF fallback selected by the user.

The Image Source node should inspect the selected file and add only the
selectors that its structure requires. For example:

```text
Source: experiment.ome.tif
Series: Position 3
```

or:

```text
Source: screen.ome.zarr
Plate: Plate 1
Well: B03
Field: 2
```

The selected item remains one complete image, potentially containing all of
its `T`, `C`, `Z`, `Y`, and `X` axes.

### OME-Zarr Import

Use `ome-zarr-py`/Zarr for OME-Zarr access and Dask-compatible arrays for
chunked data.

Import should support:

- OME-Zarr 0.4/Zarr v2 and OME-Zarr 0.5/Zarr v3;
- local directories and, by architecture, `fsspec`-compatible URIs;
- image-group selection within a store;
- multiscale level discovery;
- level 0 as the default analysis data;
- lower-resolution levels for previews without changing analysis resolution;
- axes, scale and translation transforms, units, channel display metadata, and
  image names;
- OME-Zarr label groups as first-class `labels` sources;
- plate/well/field hierarchy discovery.

The first implementation can expose ordinary image groups and labels before a
full plate browser, but the data model must retain plate/well/field identity.

### ImageJ And Conventional TIFF

Use `tifffile` directly. Parse ImageJ axes, spacing, unit, frame interval,
channel count, and display metadata where present. Conventional TIFF should
remain the fallback for TIFF files that are neither OME-TIFF nor ImageJ
hyperstacks.

Format detection should use file contents and TIFF metadata, not only filename
suffixes.

## Export Design

### OME-TIFF Export

Write a real OME-TIFF/BigTIFF with OME-XML. Include, when valid:

- dimension order, shape, dtype, and image name;
- physical sizes and units;
- time increment and unit;
- channel names, colors, fluorophore and wavelengths;
- acquisition date and preserved instrument/objective/detector metadata;
- per-plane timing, exposure, and position when still valid;
- `Creator` identifying VIPP and its version;
- structured annotations containing VIPP workflow/provenance identifiers.

BigTIFF should be selected automatically when the estimated output size
requires it, with a user override.

### ImageJ TIFF

Keep a dedicated ImageJ TIFF mode using ImageJ hyperstack metadata:

- reorder to ImageJ-compatible `TZCYXS`;
- include channels, slices, frames, spacing, time interval, unit, and display
  ranges where representable;
- convert binary masks to `uint8` `0`/`255`;
- warn or switch format when dtype, dimensions, or label IDs cannot be
  represented safely.

This mode prioritizes direct ImageJ/Fiji compatibility over complete OME
metadata.

### Conventional TIFF

Keep conventional TIFF for broad compatibility and for arrays that should not
be forced into ImageJ conventions. Preserve integer label IDs and write basic
resolution/unit tags where possible.

### OME-Zarr Export

Write:

- the highest-resolution array as level 0;
- optional generated multiscale levels;
- ordered axes with scale and translation transforms;
- channel names, colors, display windows, and image name;
- configurable chunks and compression;
- OME-Zarr label groups with `image-label` metadata;
- a namespaced `vipp` metadata block containing workflow provenance, software
  versions, source references, and operation history.

OME-Zarr should support two export surfaces:

1. **Save Image**
   Writes one image-like node output as a standalone OME-Zarr image.
2. **Export OME Analysis Dataset**
   A graph-aware or multi-input export that writes a reference intensity image
   plus one or more segmentation outputs under `labels/`. This is the preferred
   route for preserving the relationship between source images and derived
   labels.

## Save UI

Replace the current generic format choice with:

- Auto from extension;
- OME-Zarr;
- OME-TIFF;
- ImageJ TIFF;
- TIFF;
- NPY.

Show only parameters relevant to the selected format:

- compression;
- chunk shape for OME-Zarr;
- multiscale generation and scale factors;
- OME-Zarr version;
- BigTIFF auto/on/off;
- include workflow provenance;
- overwrite behavior.

The quick-save dialog should use the same writer registry. Suggested default:

- `.ome.zarr` for analysis packages or labels associated with an image;
- `.ome.tif` for a portable single processed image;
- `.tif` with ImageJ mode when explicitly selected for direct ImageJ use.

## Smart Image Source And Batch Binding

The Image Source node should be adaptive, but ordinary graph execution should
remain single-item:

```text
one source item -> one complete image dataset -> one pipeline execution
```

This avoids introducing a hidden list or collection type into every image
operation. Gaussian Blur, thresholding, labels, measurements, and saving should
continue to operate on one image dataset with explicit semantic axes.

### Adaptive Source Controls

After inspecting a source, the node may add:

- image/series or scene selector;
- plate selector;
- well selector;
- field/site selector;
- OME-Zarr image-group or label-group selector;
- advanced resolution-level selector;
- calibration summary plus a recommended path to `Set Pixel Size / Units` for
  sources with missing or incorrect pixel/voxel size metadata;
- source summary showing axes, shape, scale, channels, and estimated memory.

Controls that do not apply to a source should remain hidden.

Time, channel, and Z should normally remain axes inside the selected image, not
source selectors. Users should subset those axes with explicit graph nodes such
as `Select Axis Slice`. This keeps the operation visible and reproducible in the
workflow. A source-level subset can be added later as an optimization for data
too large to load, but it should be recorded explicitly in source parameters.

Multiscale resolution also needs special treatment:

- analysis defaults to the highest-resolution level;
- previews may transparently use a lower-resolution level;
- manually processing a lower-resolution level is an advanced, explicit source
  setting because it changes measurement scale and results.

### Interactive And Batch Modes

The source node should expose two binding modes:

1. **Single item**
   The user selects one series, well, field, image group, or label group and
   tunes the workflow interactively.
2. **Collection**
   The node stores a query such as all series, selected wells, all fields in a
   plate, or files matching a pattern. The interactive graph displays one
   representative item, while the batch runner executes the complete graph once
   per matched item.

Collection mode must not make the node output a list of images. Instead, the
batch engine binds each collection item to the source node and invokes the same
single-item graph repeatedly:

```text
for item in source_collection:
    outputs = run_pipeline(item)
    save outputs using item identity and provenance
```

Each batch item should carry a stable identity containing, where applicable:

- source URI;
- image/series index and name;
- plate, well, field/site, and acquisition identifiers;
- source UUID;
- output naming tokens.

This identity should be written to output provenance and made available to save
templates, for example:

```text
{plate}_{well}_{field}_{source_stem}__{node}.{extension}
```

For workflows with multiple Image Source nodes, batch configuration must define
how items are paired:

- one varying source plus fixed reference sources;
- pair by filename or metadata key;
- Cartesian product only when explicitly requested.

This pairing policy belongs to a batch configuration, not to image-port
connections.

## Large-Data Execution

OME-Zarr cannot be first-class only at the file-dialog level. Large datasets
must not be unconditionally converted with `np.asarray`.

Recommended direction:

- permit Dask-compatible arrays in `ImageDataset` and `SourcePayload`;
- let previews use an appropriate multiscale level;
- mark operations as chunk-safe, blockwise with overlap, global-reduction, or
  eager-only;
- compute only at explicit execution/export boundaries;
- initially allow eager-only nodes to show an estimated memory cost and require
  confirmation before materializing large data.

This can be implemented in stages, but the I/O interfaces should accept lazy
arrays from the beginning to avoid another redesign.

## Python Export

Generated scripts should import the same I/O registry and pass `ImageDataset`
metadata through execution. The current generated `load_image()` and
`save_image()` helpers should be removed once the shared I/O layer exists.

Batch export should preserve source metadata independently for every input file
and add the exported workflow identifier to every output.

## Dependencies

Recommended core dependencies for first-class OME support:

- `tifffile` for TIFF, ImageJ TIFF, OME-TIFF pixels, and TIFF series;
- `ome-types` for validated OME-XML parsing and generation;
- `ome-zarr` and `zarr` for OME-Zarr;
- `dask[array]` for chunked/lazy arrays;
- `fsspec` for local and remote stores.
- `imageio` and `pillow` for common raster image import and 2D export.

Bio-Formats is not required for OME-TIFF or OME-Zarr. It can remain a later,
optional route for proprietary microscope formats.

## Phased Implementation

### Phase 1: Metadata And Reader/Writer Foundation

Status: complete on 2026-06-15.

1. Add the I/O package, registry, `ImageDataset`, acquisition metadata, channel
   metadata, display metadata, and provenance types.
2. Update `SourcePayload` and pipeline execution to accept a prebuilt state.
3. Implement OME-TIFF/ImageJ/conventional TIFF reading with metadata.
4. Implement explicit OME-TIFF, ImageJ TIFF, conventional TIFF, and NPY writers.
5. Route the widget, Save Image node, quick save, and generated Python through
   the shared registry.

### Phase 2: OME-Zarr Images

Status: local 0.4/0.5 image read/write and lazy levels are implemented.
Pyramids, preview-level selection, and URI transports remain open.

1. Add OME-Zarr 0.4 and 0.5 reading.
2. Add local OME-Zarr image writing with axes, transforms, channels, chunks,
   compression, and optional pyramids.
3. Use lower-resolution data for previews while preserving level 0 analysis.
4. Add URI-ready storage interfaces.

### Phase 3: Labels And Analysis Packages

Status: image-plus-label OME-Zarr analysis dataset export and label-group
import are implemented. Label colors and label-property tables remain open.

1. Import OME-Zarr labels as `labels`.
2. Add Export OME Analysis Dataset for reference image plus label outputs.
3. Add label colors, source relationships, and future label-property tables.

### Phase 4: Scale And Advanced Containers

1. Add Dask-aware operation capability declarations and controlled
   materialization.
2. Add remote reads and optional remote writes.
3. Add plate/well/field browsing and export.
4. Add richer ROIs, points, tables, and acquisition metadata when those graph
   types exist.

## Accepted Decisions

Accepted on 2026-06-15:

1. **Default OME-Zarr export version**
   Read 0.4 and 0.5; write 0.4 by default and offer 0.5 explicitly. Revisit the
   default as viewer and analysis-tool support converges on 0.5/Zarr v3.
2. **Large-data scope**
   The I/O and source model is lazy-capable now. Operation-level lazy execution
   will be delivered incrementally.
3. **Multiple images and HCS stores**
   Use the adaptive Image Source model above. Interactive execution binds one
   selected item; collection mode defines future batch iteration.
4. **Labels export**
   Use a separate Export OME Analysis Dataset surface for source image plus
   labels. Do not silently treat a label array as an intensity image.
5. **Remote storage**
   Design paths as URIs. Add anonymous HTTP reads before credential UI or
   remote writes.
6. **Metadata editing**
   Preserve and inspect metadata first. Pixel/voxel size and unit correction is
   now handled by `Set Pixel Size / Units` as an explicit calibration-repair
   node. `Rescale Axes` now updates physical scale metadata when it changes the
   sampled pixel grid. More general metadata editing can come later. Future
   resampling or axis-scaling nodes must update scale, translation, and units
   automatically when they change geometry.

No decision is needed about retaining ImageJ TIFF: it should remain a supported,
explicit export target alongside OME-TIFF and OME-Zarr.

## Standards And Library References

- [OME-Zarr 0.5 specification](https://ngff.openmicroscopy.org/0.5/)
- [OME-Zarr Python reader/writer](https://ome-zarr.readthedocs.io/)
- [OME-TIFF specification](https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html)
- [tifffile](https://github.com/cgohlke/tifffile)
- [ome-types](https://ome-types.readthedocs.io/)
