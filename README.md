<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/branding/vipp-logo-dark.svg">
    <img src="docs/assets/branding/vipp-logo.svg" alt="VIPP" width="420">
  </picture>
</p>

# VIPP — Visual Image Processing Platform

**Visual workflows for reproducible bioimage analysis.**

[![CI](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml/badge.svg)](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![Python](https://img.shields.io/pypi/pyversions/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![License](https://img.shields.io/pypi/l/napari-vipp.svg)](LICENSE)

`napari-vipp` is the napari-native implementation of **VIPP, the Visual Image
Processing Platform**. Build typed node graphs, inspect intermediate images and
tables, tune parameters, save workflows, and repeat the same operations without
hiding axis or physical-scale metadata.

> **Alpha software:** expect breaking workflow and parameter changes. Validate
> outputs on representative data before scientific interpretation or
> publication.

VIPP's implemented safeguards include stable source revisions, physical-grid
checks, exact unsampled diagnostics, detached viewer layers, atomic artifacts,
and batch publication only after source reverification. See the
[scientific integrity boundaries](docs/architecture.md#scientific-integrity-boundaries)
and the contributor [scientific behavior requirements](CONTRIBUTING.md#scientific-behavior-requirements).

## Install And Open

VIPP requires Python 3.12 or newer. If napari is not already installed, install
it with a Qt backend at the same time:

```bash
python -m pip install "napari[pyqt6]"
python -m pip install --pre napari-vipp
vipp
```

The `--pre` flag is required while VIPP is published as an alpha release. It is
kept on the VIPP command so napari itself can continue to resolve to a stable
release.

In napari, open:

```text
Plugins > VIPP Workflow (napari-vipp)
```

Use `Open example...` for a runnable workflow with synthetic data. A good first
choice is `Red-Channel Label Cleanup`; select nodes from left to right to review
their parameters, thumbnails, metadata, and outputs. To explore collection
processing, open `Deterministic Batch & Provenance`; VIPP prepares a small
self-contained working copy and opens it already configured and previewed.

![VIPP example workflow chooser](docs/assets/user-guide/vipp-example-chooser.png)

## What It Supports

| Area | Current alpha capabilities |
| --- | --- |
| Graph authoring | Searchable node palette, typed ports, dynamic outputs, cycle prevention, undo/redo, graph notes, named tunnels, auto-layout, and saved positions. |
| Images and metadata | Semantic T/C/Z/Y/X axes, scale/units/origin, channel and acquisition metadata, source identity, and operation history. |
| Image processing | Intensity transforms, filters, background correction, thresholding, watershed, binary/label morphology, channels, axes, masks, and composites. |
| Measurements | Object and intensity tables, calibrated morphology, 3D mesh morphology, skeleton/network analysis, colocalization, object association, and table composition. |
| Restoration | Born-Wolf PSF generation, measured-PSF preparation, and manual/cached 2D or 3D Richardson-Lucy and RL-TV deconvolution. |
| Reuse and automation | Workflow JSON, generated headless Python, explicit batch outputs, reviewed collection plans, representative navigation, retained batch results, and workflow/config/manifest artifacts. |
| I/O | OME-TIFF, ImageJ TIFF, TIFF, local OME-Zarr 0.4/0.5, NPY/NPZ, common 2D raster formats, and optional microscope readers. |

Most graph operations are still eager. Large z-stacks and OME-Zarr datasets
therefore need deliberate cache, preview, and output choices; see the
[cache and memory guide](docs/cache-and-memory.md).

## Optional Microscope Readers

Install only the reader family you need, then restart napari:

| Format family | Install command |
| --- | --- |
| Nikon ND2 | `python -m pip install --pre "napari-vipp[nd2]"` |
| Zeiss CZI | `python -m pip install --pre "napari-vipp[czi]"` |
| Mixed microscope formats | `python -m pip install --pre "napari-vipp[microscope]"` |
| BioIO/Bio-Formats fallback | `python -m pip install --pre "napari-vipp[bioformats]"` |

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
7. Add `Batch Output` nodes before `Batch workspace...` when exact saved outputs
   matter.
8. Optionally click `Preview batch` to inspect the complete plan and use the
   representative slider or a preview-table row without running or saving the
   full batch. Preview is not required: `Run batch` performs its own preflight.
9. Run the collection from the retained workspace with one click, where
   item-level progress, final statuses, validation, and the
   `vipp_batch_manifest.json` path remain available for inspection.
10. To validate the complete batch path without your own files, choose
   `Open example...` -> `Deterministic Batch & Provenance` -> `Open batch
   demo...`. Choose where to save its small working copy, review the populated
   graph, move through all three paired fields with the representative slider,
   review the three-item/nine-output batch preview, then click `Run demo batch`. VIPP
   checks the finished outputs and provenance against exact ground truth
   automatically.

Workflow JSON stores the graph and optional VIPP UI state, not cached pixels or
tables. When Batch workspace is active, Save workflow can optionally attach its
versioned config so the same workspace reopens from that one JSON file; local
paths are included, but source pixels are not. `Export Python...` embeds a
validated immutable workflow and executes it
through the same headless pipeline engine as VIPP, including normalized
`ImageState` propagation. See the [user guide](docs/user-guide.md) for source
binding, runtime-version, and command-line details.

## Documentation

- [Published VIPP documentation](https://rensutheart.github.io/vipp-mkdocs/)
- [Categorized 0.12 release notes](CHANGELOG.md#0120a2---2026-07-16)
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

Launch a development instance from the repository with `./vipp`; it uses the
project's `.venv-macos` environment directly, so shell activation is not
required. The installed `vipp` command and `python -m napari_vipp` are also
supported. To open the synthetic sample with a pipeline run already completed, use
`python scripts/launch_vipp_sample.py`. The
[architecture reference](docs/architecture.md) explains the graph, metadata,
execution, persistence, and UI boundaries.

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request, use [SUPPORT.md](SUPPORT.md) for help and issue-reporting
guidance, and report suspected vulnerabilities privately through
[SECURITY.md](SECURITY.md). All project interactions follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## 0.12 Alpha Highlights

`0.12.0a2` is the current alpha. It builds on the 0.12 architecture and
reproducibility baseline with:

- isolated node tuning that keeps downstream propagation paused until the
  latest local result is applied or the session is cancelled;
- bright actionable and dark waiting graph states, an attention-colored
  `Calculate all`, and progressive node previews during longer runs;
- exact-pixel napari layer reuse and display-resolution thumbnail rendering
  that reduce UI stalls without changing scientific arrays;
- configurable port labels and responsive graph layout; and
- clearer PSF preflight, Nyquist, support, centering, and boundary-tail
  feedback for deconvolution workflows.

The 0.12 foundation also provides:

- workflow schema version 3 records explicit axis, channel, grid, and operation
  choices instead of restoring ambiguous scientific defaults;
- verified file and live-layer revisions, physical-grid checks, detached viewer
  layers, and atomic artifacts reject stale or silently repaired inputs;
- generated Python and collection batching now use the same validated headless
  executor as the interactive graph;
- the retained batch workspace adds reviewed plans, representative navigation,
  explicit outputs, per-item provenance, collision policies, progress, final
  statuses, manifests, and deterministic validation;
- exact diagnostics, background workers, and platform-specific memory reporting
  improve responsiveness without changing the population being measured;
- Richardson-Lucy TV controls now explain parameter effects and provide
  practical linear or geometric slider windows without limiting exact spinner
  entry; and
- the former monolithic widget has been decomposed into focused Qt-free core and
  UI service modules with dependency-direction tests.

Breaking alpha changes are intentional where preserving an older implicit
behavior would weaken scientific validity. See the categorized
[0.12 release notes](CHANGELOG.md#0120a2---2026-07-16), the
[upgrade and workflow contract](docs/user-guide.md#save-workflow-json), and
[planning.md](docs/planning.md) for later milestones. Semantic-axis collection
iteration, HCS traversal, scalable OME-Zarr previews, and broader scientific
validation remain future work.

## Citation, Acknowledgement, And License

If VIPP contributes to your work, acknowledge `napari-vipp` and link to the
[project repository](https://github.com/rensutheart/napari-vipp). Citation
metadata is available in [CITATION.cff](CITATION.cff); a DOI or manuscript
citation can be added when available.

napari-vipp is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full terms.
