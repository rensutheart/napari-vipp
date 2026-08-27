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

VIPP `0.14.0a2` is published as an official GitHub prerelease with checksum
sidecars and as an exact-version package on PyPI. Use only those official
release surfaces; do not download a file from a guessed asset URL.

The normal experience is one Windows `.exe`: download
[`VIPP-Setup-0.14.0a2-Windows-x86_64-UNSIGNED.exe`](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a2/VIPP-Setup-0.14.0a2-Windows-x86_64-UNSIGNED.exe)
from the official `v0.14.0a2` release, double-click it, and keep the recommended
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
standard GPU installation includes every current GPU implementation.

Setup also shows separate rounded estimates before installation: CPU is
approximately 250 MiB to download, 1.5 GiB
installed, and 2.5 GiB peak working space; CUDA is approximately 1.5 GiB,
5 GiB, and 7 GiB respectively. These are orientation, not replacements for the
enforced disk minimums above and not a VRAM comparison. Setup reports its phase
and elapsed time, keeps indeterminate work visibly active, preserves the latest
concrete activity through quiet periods, and exposes its setup log. Byte
progress appears only when a trustworthy total exists.

For `0.14.0a2`, one-click managed installation uses only two fixed per-track
roots. Windows supplies the canonical Local App Data directory through
`SHGetKnownFolderPath(FOLDERID_LocalAppData)`; setup appends
`VIPP\environments\cpu` or `VIPP\environments\cuda13`. Custom managed roots are
not accepted. The CUDA path must contain ASCII characters only because the
pinned CuPy 14.1.1 runtime cannot reliably compile CUDA kernels from a Windows
environment path containing characters such as `Å` or `é`. Spaces are
supported. If canonical Local App Data contains a non-ASCII character, the
one-click CUDA route is unavailable and setup offers CPU instead. The fixed CPU
root remains supported with Unicode paths.

Expert-selected existing environments remain a separate, non-mutating route;
setup does not move, edit, or turn them into managed installations. If an older
installer-owned CUDA copy is already in an incompatible path, setup will not
update or repair it in place. After any separately recorded recovery from an
earlier interrupted transaction, the newly blocked selection performs no new
mutation of that copy. It can be removed through its ownership-bound Windows
Apps uninstaller, but this account cannot use one-click CUDA until its canonical
Local App Data path is ASCII-compatible.

> **Unsigned alpha:** this release is intentionally not Authenticode-signed,
> so Windows will show **Unknown publisher** and may show
> **Windows protected your PC**. Download only the explicitly named
> `-UNSIGNED.exe` from the official
> [`v0.14.0a2` GitHub release](https://github.com/rensutheart/napari-vipp/releases/tag/v0.14.0a2),
> verify its SHA-256 against the attached `SHA256SUMS` file, then select
> **More info > Run anyway**. Stop if the hash differs or antivirus identifies
> a threat; never disable Windows security. The [Quick Start](docs/quick-start.md)
> gives the exact verification steps and a manual fallback.

The [VIPP Quick Start](docs/quick-start.md) explains the installer-first flow,
current manual commands, CPU/GPU choices, prerequisites, the advanced
existing-napari route, update/repair/uninstall behavior, and a first workflow.

### macOS Installer (Recommended Path)

Download the offline package for your Mac from the official `v0.14.0a2`
release:

- [Apple Silicon (`arm64`) PKG](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a2/VIPP-0.14.0a2-macOS-arm64-UNSIGNED.pkg)
  ([SHA-256 file](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a2/SHA256SUMS-macOS-arm64-0.14.0a2.txt))
- [Intel (`x86_64`) PKG](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a2/VIPP-0.14.0a2-macOS-x86_64-UNSIGNED.pkg)
  ([SHA-256 file](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a2/SHA256SUMS-macOS-x86_64-0.14.0a2.txt))

The package installs a private CPU-only environment under `~/Library/vipp`
and creates `~/Applications/VIPP.app`; it does not need a separately installed
Python. Allow approximately 3 GB of free disk space. The alpha is explicitly
unsigned and not notarized. Verify the matching architecture-specific
`SHA256SUMS-macOS-*.txt` file before opening it. If macOS blocks the verified
package, open **System Settings > Privacy & Security**, confirm the VIPP
package name, choose **Open Anyway**, and approve the installer. Do not bypass
the warning if the checksum differs or the package came from another source.

The [macOS packaging guide](packaging/macos/README.md) documents the pinned
offline build, managed layout, architecture split, lifecycle checks, removal,
and future Developer ID/notarization path. A DMG is unnecessary because the PKG
is already double-clickable and performs the installation itself.

### Manual Installation (Advanced And Portable)

VIPP `0.14.0a2` supports CPython 3.12 and 3.13. Create and activate a dedicated
virtual environment, then install:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.14.0a2"
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
| Graph authoring | Searchable node palette and canvas, typed ports, multi-node selection and movement, copy/paste and exact-operation value transfer, dynamic outputs, cycle prevention, undo/redo, graph notes, named tunnels, insert-on-wire or before-tunnel editing, saved positions, and auto-layout. |
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
data, parameters, environments, or memory conditions visibly use CPU.

See the [GPU Guide](docs/gpu-guide.md) for qualification, supported operation
families, setup, benchmarking, fallback, and cross-device reproducibility.
Durable batch/generated-Python behavior is documented in
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
python -m pip install "napari-vipp[nd2]==0.14.0a2"
```

Available extras include `nd2`, `czi`, `microscope`, and `bioformats`. See
[Image import and export](docs/io-user-guide.md) for formats, limitations, and
the matching commands.

## Windows Installer And Startup

VIPP `0.14.0a2` provides a novice-facing Windows setup window, automatic CPU/GPU
recommendation, exact dependency review, transactional install/update/repair,
owned shortcuts, independent CPU/GPU Apps & Features entries, ownership-safe
uninstall, and acceptance checks. It installs into a private managed location
and never overwrites an unrelated folder or manually managed napari environment.
It also provides branded Automatic/CPU/Prefer-GPU
launchers, lightweight in-napari loading host, read-only planning CLI, and the
same reviewed CuPy/CuPyX implementation set in every CUDA installation.

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

`0.14.0a2` adds native Apple Silicon and Intel installers, allows detached VIPP
windows to be resized freely in both dimensions, and fixes PySide6
compatibility in dialogs, menus, rendering, and delayed Qt callbacks. The
SourceItem, reader, batch-override, OME-Zarr preview, workflow, and scientific
contracts introduced in `0.14.0a1` remain unchanged. CPU remains the portable
reference, and unsupported inputs, parameters, environments, or memory
conditions continue to use it visibly.

See the categorized [0.14.0a2 release notes](CHANGELOG.md#0140a2---2026-08-27)
and [roadmap](docs/planning.md) for details and remaining milestones.

## Citation, Acknowledgement, And License

If VIPP contributes to your work, acknowledge `napari-vipp` and link to the
[project repository](https://github.com/rensutheart/napari-vipp). Citation
metadata is available in [CITATION.cff](CITATION.cff); a DOI or manuscript
citation can be added when available.

napari-vipp is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full terms.
