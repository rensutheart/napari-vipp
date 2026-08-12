<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rensutheart/napari-vipp/main/docs/assets/branding/vipp-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/rensutheart/napari-vipp/main/docs/assets/branding/vipp-logo.svg" alt="VIPP" width="420">
  </picture>
</p>

# VIPP — Visual Image Processing Platform

**Visual image processing made approachable.**

VIPP's scientific purpose is to support **visual workflows for reproducible
bioimage analysis**: workflows that can be inspected, saved, rerun, and
reported with their important context intact.

[![CI](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml/badge.svg)](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![Python](https://img.shields.io/pypi/pyversions/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![License](https://img.shields.io/pypi/l/napari-vipp.svg)](LICENSE)

`napari-vipp` is the napari-native implementation of VIPP. Build typed visual
workflows, inspect intermediate images and tables, tune parameters, save and
batch-run graphs, and retain the metadata and provenance needed to understand
how results were produced.

> **Alpha software:** expect breaking workflow and parameter changes. Validate
> outputs on representative data before scientific interpretation or
> publication.

VIPP includes physical-grid checks, exact unsampled diagnostics, detached
viewer layers, atomic artifacts, source reverification, and publication only
after successful batch validation. See the
[scientific integrity boundaries](docs/architecture.md#scientific-integrity-boundaries)
and [scientific behavior requirements](CONTRIBUTING.md#scientific-behavior-requirements).

## Quick Start

### Windows Installer (Recommended Path)

The normal experience is one Windows `.exe`: download
[`VIPP-Setup-0.13.0a5-Windows-x86_64-UNSIGNED.exe`](https://github.com/rensutheart/napari-vipp/releases/download/v0.13.0a5/VIPP-Setup-0.13.0a5-Windows-x86_64-UNSIGNED.exe)
from the official `v0.13.0a5` release, double-click it, and keep the recommended
managed VIPP environment. Review CPU or qualified NVIDIA GPU setup, then launch
VIPP from the shortcuts it creates. A supported 64-bit Python is a separate prerequisite;
if it is missing, setup links to the official Python 3.12.10 installer and lets
the user check again afterward.

Before **Install** is enabled, setup shows the exact managed location, the
resolved CPU or NVIDIA CUDA 13 route, and whether shortcuts will be added to the
Start Menu only or to both the Start Menu and Desktop. GPU setup currently
needs at least 15 GiB free on the installation drive while it runs. This is disk
storage, not GPU memory (VRAM). It also needs at least 5 GiB free on each drive
used for Windows temporary files and VIPP installer records; CPU setup needs at
least 1 GiB there. Setup names the exact location if that check fails. The
standard GPU installation works without cuCIM; that optional add-on is installed
separately afterward.

> **Unsigned alpha:** this release is intentionally not Authenticode-signed,
> so Windows will show **Unknown publisher** and may show
> **Windows protected your PC**. Download only the explicitly named
> `-UNSIGNED.exe` from the official
> [`v0.13.0a5` GitHub release](https://github.com/rensutheart/napari-vipp/releases/tag/v0.13.0a5),
> verify its SHA-256 against the attached `SHA256SUMS` file, then select
> **More info > Run anyway**. Stop if the hash differs or antivirus identifies
> a threat; never disable Windows security. The [Quick Start](docs/quick-start.md)
> gives the exact verification steps and a manual fallback.

The [VIPP Quick Start](docs/quick-start.md) explains the installer-first flow,
current manual commands, CPU/GPU choices, prerequisites, the advanced
existing-napari route, update/repair/uninstall behavior, and a first workflow.
After a standard CUDA installation passes Compute Doctor, the separate
[`0.13.0a5` cuCIM add-on](https://github.com/rensutheart/napari-vipp/releases/download/v0.13.0a5/napari-vipp-cucim-installer-0.13.0a5-windows.zip)
can build the optional pinned provider locally. It contains no prebuilt cuCIM
wheel; verify the ZIP against the release checksum before extracting it.

### Manual Installation (Advanced And Non-Windows)

VIPP `0.13.0a5` supports CPython 3.12 and 3.13. Create and activate a dedicated
virtual environment, then install:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a5"
vipp
```

An exact alpha version does not need pip's `--pre` option. The optional public
GPU route currently requires native Windows, CPython 3.12, a compatible NVIDIA
driver/GPU, and the pinned CUDA 13 stack. Follow the
[Quick Start](docs/quick-start.md#nvidia-cuda-13-on-windows) rather than adding
GPU packages to a general napari environment by hand.

Inside an existing napari session, open:

```text
Plugins > VIPP Workflow (napari-vipp)
```

### First Workflow

Choose **Open example...** and start with **Red-Channel Label Cleanup**. Select
nodes from left to right to inspect parameters, previews, metadata, and
outputs. **Deterministic Batch & Provenance** is a self-contained introduction
to collection processing.

![VIPP example workflow chooser](https://raw.githubusercontent.com/rensutheart/napari-vipp/main/docs/assets/user-guide/vipp-example-chooser.png)

## What VIPP Supports

| Area | Current alpha capabilities |
| --- | --- |
| Graph authoring | Searchable node palette and canvas, typed ports, dynamic outputs, cycle prevention, undo/redo, graph notes, named tunnels, insert-on-wire, saved positions, and auto-layout. |
| Images and metadata | Semantic T/C/Z/Y/X axes, scale, units, origin, channel/acquisition metadata, source identity, and operation history. |
| Processing | Intensity transforms, filters, background correction, thresholding, watershed, binary/label morphology, channels, axes, masks, and composites. |
| Measurements | Object and intensity tables, calibrated/mesh morphology, skeleton/network analysis, colocalization, object association, and table composition. |
| Restoration | Born-Wolf PSF generation, measured-PSF preparation, Richardson-Lucy, and RL-TV deconvolution in 2D/3D. |
| Reuse and automation | Independent workflow tabs, workflow JSON, generated Python, explicit batch outputs, plan review, progress/cancellation, manifests, and provenance artifacts. |
| I/O | OME-TIFF, ImageJ TIFF, TIFF, local OME-Zarr 0.4/0.5, NPY/NPZ, common 2D raster formats, and optional microscope readers. |

Most graph operations are still eager. Large z-stacks and OME-Zarr datasets
need deliberate cache, preview, and output choices; see
[Cache and memory](docs/cache-and-memory.md).

<a id="gpu-execution-and-development-environment"></a>

## GPU Acceleration (Optional)

VIPP remains fully usable on CPU on Windows, Linux, and macOS. The current
public GPU route is native Windows, CPython 3.12, CUDA 13, and a compatible
NVIDIA GPU with compute capability 7.5 or newer. GPU model names are recorded
for reproducibility rather than used as an allowlist.

**Auto** is recommended and may correctly choose CPU. **Prefer GPU**
requests every scientifically eligible GPU implementation, while **Custom**
adds per-node choices and **Find fastest pipeline…**. Unsupported operations,
data, parameters, environments, or memory conditions visibly use CPU. cuCIM is
optional and separately installed.

See the [GPU Guide](docs/gpu-guide.md) for qualification, supported operation
families, setup, benchmarking, fallback, cuCIM, and cross-device
reproducibility. Durable batch/generated-Python behavior is documented in
[Durable GPU execution](docs/durable-gpu-execution.md); exact scientific and
benchmark evidence remains in the linked phase records rather than this
landing page.

## Workflow Basics

1. Add an **Image Source** for a napari layer, file, or bundled sample.
2. Add nodes from the palette and connect compatible ports.
3. Select a node to tune parameters and inspect output metadata.
4. Click **Calculate** for manual/cached analysis or deconvolution nodes.
5. Pin important image outputs into napari for full-resolution comparison.
6. Save the graph with **Save workflow...**.
7. Add explicit **Batch Output** nodes before running a collection when exact
   saved outputs matter.

The [User Guide](docs/user-guide.md) covers graph controls, workflow schema,
batch configuration, source-axis declarations, export, and scientific review.

## Optional Microscope Readers

Reader packages should be installed with the Python interpreter from the exact
environment that launches VIPP, never a global/base Python. Keep the VIPP
version pinned and restart napari afterward. For example:

```bash
python -m pip install "napari-vipp[nd2]==0.13.0a5"
```

Available extras include `nd2`, `czi`, `microscope`, and `bioformats`. See
[Image import and export](docs/io-user-guide.md) for formats, limitations, and
the matching commands.

## Windows Installer And Startup

VIPP `0.13.0a5` provides a novice-facing Windows setup window, automatic CPU/GPU
recommendation, exact dependency review, transactional install/update/repair,
owned shortcuts, independent CPU/GPU Apps & Features entries, ownership-safe
uninstall, and acceptance checks. It installs into a private managed location
and never overwrites an unrelated folder or manually managed napari environment.
It also provides branded Automatic/CPU/Prefer-GPU
launchers, lightweight in-napari loading host, read-only planning CLI, and the
separate optional cuCIM local-build add-on.

See the [Desktop startup and installer plan](docs/desktop-startup-and-installer-plan.md)
and [Windows installation planner](docs/windows-installation-planner.md).

## Documentation

- [Quick Start](docs/quick-start.md)
- [User Guide](docs/user-guide.md)
- [GPU Guide](docs/gpu-guide.md)
- [Image import and export](docs/io-user-guide.md)
- [Example workflow index](examples/README.md)
- [Measurement workflows](docs/measurement-workflows.md)
- [Durable GPU execution](docs/durable-gpu-execution.md)
- [Architecture](docs/architecture.md)
- [Planning and roadmap](docs/planning.md)
- [Published versioned manual](https://rensutheart.github.io/vipp-mkdocs/)

## Development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use
[SUPPORT.md](SUPPORT.md) for help, report vulnerabilities privately through
[SECURITY.md](SECURITY.md), and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Current Alpha

`0.13.0a5` adds the installer-first Windows experience, branded launchers, and
the optional verified local cuCIM add-on. It retains compatible NVIDIA CUDA 13
admission without a model-name allowlist and the exact software, scientific,
memory, and workload gates from `0.13.0a4`. The 0.13 series also adds portable
compute intent,
actual implementation provenance, unified interactive/batch/export execution,
workflow schema 4, batch schema 3, independent workflow tabs, expanded
measurements/restoration, and substantial I/O, graph, progress, cancellation,
and publication hardening.

See the categorized [0.13.0a5 release notes](CHANGELOG.md#0130a5---2026-08-12)
and [roadmap](docs/planning.md) for details and remaining milestones.

## Citation, Acknowledgement, And License

If VIPP contributes to your work, acknowledge `napari-vipp` and link to the
[project repository](https://github.com/rensutheart/napari-vipp). Citation
metadata is available in [CITATION.cff](CITATION.cff); a DOI or manuscript
citation can be added when available.

napari-vipp is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full terms.
