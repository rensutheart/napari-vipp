# napari-vipp

napari-vipp is a napari-native visual image-processing pipeline builder for
bioimage analysis. Build a typed node graph, inspect intermediate images and
tables, tune parameters, save the workflow, and run the same operations without
hiding axis or physical-scale metadata.

> **Alpha software:** expect breaking workflow and parameter changes. Validate
> outputs on representative data before scientific interpretation or
> publication.

## Install And Open

VIPP requires Python 3.10 or newer. If napari is not already installed, install
it with a Qt backend at the same time:

```bash
python -m pip install "napari[pyqt6]" napari-vipp
napari
```

In napari, open:

```text
Plugins > VIPP Workflow (napari-vipp)
```

Use `Open example...` for a runnable workflow with synthetic data. A good first
choice is `Red-Channel Label Cleanup`; select nodes from left to right to review
their parameters, thumbnails, metadata, and outputs.

![VIPP example workflow chooser](docs/assets/user-guide/vipp-example-chooser.png)

## What It Supports

| Area | Current alpha capabilities |
| --- | --- |
| Graph authoring | Searchable node palette, typed ports, dynamic outputs, cycle prevention, undo/redo, graph notes, named tunnels, auto-layout, and saved positions. |
| Images and metadata | Semantic T/C/Z/Y/X axes, scale/units/origin, channel and acquisition metadata, source identity, and operation history. |
| Image processing | Intensity transforms, filters, background correction, thresholding, watershed, binary/label morphology, channels, axes, masks, and composites. |
| Measurements | Object and intensity tables, calibrated morphology, 3D mesh morphology, skeleton/network analysis, colocalization, object association, and table composition. |
| Restoration | Born-Wolf PSF generation, measured-PSF preparation, and manual/cached 2D or 3D Richardson-Lucy and RL-TV deconvolution. |
| Reuse and automation | Workflow JSON, generated headless Python, explicit batch outputs, local collection batch runs, dry-run previews, and workflow/script artifacts. |
| I/O | OME-TIFF, ImageJ TIFF, TIFF, local OME-Zarr 0.4/0.5, NPY/NPZ, common 2D raster formats, and optional microscope readers. |

Most graph operations are still eager. Large z-stacks and OME-Zarr datasets
therefore need deliberate cache, preview, and output choices; see the
[cache and memory guide](docs/cache-and-memory.md).

## Optional Microscope Readers

Install only the reader family you need, then restart napari:

| Format family | Install command |
| --- | --- |
| Nikon ND2 | `python -m pip install "napari-vipp[nd2]"` |
| Zeiss CZI | `python -m pip install "napari-vipp[czi]"` |
| Mixed microscope formats | `python -m pip install "napari-vipp[microscope]"` |
| BioIO/Bio-Formats fallback | `python -m pip install "napari-vipp[bioformats]"` |

These routes are an experimental foundation: axes and common metadata are
normalized where the source reader exposes them, but format-specific coverage
still needs validation against a broader corpus of real acquisition files.

## Workflow Basics

1. Add or select an `Image Source` for a napari layer, file, or bundled sample.
2. Add nodes from the palette and connect compatible output and input ports.
3. Select a node to tune parameters and inspect its output metadata.
4. Click `Calculate` for manual/cached nodes such as measurements and
   deconvolution.
5. Pin important image outputs into napari for full-resolution comparison.
6. Save the graph with `Save workflow...`.
7. Add `Batch Output` nodes before `Run batch...` when exact saved outputs
   matter.

Workflow JSON stores the graph and optional VIPP UI state, not cached pixels or
tables. `Export Python...` emits direct calls to the headless operation and I/O
functions; it does not reproduce interactive caches or full runtime metadata
propagation. See the [user guide](docs/user-guide.md) for details and caveats.

## Documentation

- [Documentation index](docs/README.md)
- [User guide](docs/user-guide.md)
- [Image import and export](docs/io-user-guide.md)
- [Example workflow index](examples/README.md)
- [Measurement workflows](docs/measurement-workflows.md)
- [Operator tips](docs/operator-tips.md)
- [Developer notes](docs/developer-notes.md)
- [Current planning and roadmap](docs/planning.md)

## Development

Create a local environment and install the development dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the required checks:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
```

Launch a development instance with `python scripts/launch_vipp_sample.py`. The
[architecture reference](docs/architecture.md) explains the graph, metadata,
execution, persistence, and UI boundaries.

## Roadmap

The current public alpha is `0.11.0a1`, which introduced the PSF/restoration
and optional microscope-reader foundations. The next planned milestone focuses
on saved batch configuration and per-item provenance, followed by scalable
OME-Zarr previews and broader scientific validation. See
[planning.md](docs/planning.md) for the maintained release order and evidence
gates.

## Citation, Acknowledgement, And License

If VIPP contributes to your work, acknowledge `napari-vipp` and link to the
[project repository](https://github.com/rensutheart/napari-vipp). Citation
metadata is available in [CITATION.cff](CITATION.cff); a DOI or manuscript
citation can be added when available.

napari-vipp is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full terms.
