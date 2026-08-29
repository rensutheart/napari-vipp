<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/rensutheart/napari-vipp/main/docs/assets/branding/vipp-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/rensutheart/napari-vipp/main/docs/assets/branding/vipp-logo.svg" alt="VIPP" width="420">
  </picture>
</p>

# VIPP — Visual Image Processing Platform

**Design inspectable bioimage workflows with visual feedback at every stage.**

Visual image processing made approachable through **visual workflows for
reproducible bioimage analysis**.

[![CI](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml/badge.svg)](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![Python](https://img.shields.io/pypi/pyversions/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![License](https://img.shields.io/pypi/l/napari-vipp.svg)](LICENSE)

VIPP is an open-source, napari-native platform for building bioimage analyses
as visual graphs. Connect image, mask, label and table nodes, inspect parameters
and intermediate results, compare processing routes on the same input, and then
move from representative samples to collection-wide execution.

Build and run workflows from the available nodes without writing Python. Save
them as versioned JSON, export them as Python, and retain the source, axes,
calibration, parameters, outputs and implementation provenance needed to
understand how results were produced.

> **Alpha software:** workflows and parameters may still change. Validate every
> analysis on representative data before scientific interpretation or
> publication.

![VIPP workspace with a calculated 3D deconvolution workflow](https://raw.githubusercontent.com/rensutheart/napari-vipp/main/docs/assets/user-guide/vipp-3d-deconvolution-workspace.png)

*A calculated 3D deconvolution workflow with the node catalogue, thumbnail
previews and selected-node inspector visible together.*

## Why VIPP

- **See each transformation.** Inspect images, histograms, tables, metadata and
  parameters while designing the workflow.
- **Iterate locally.** Recalculate one stage without repeatedly running every
  later step.
- **Compare approaches.** Apply different processing routes to the same input
  and inspect the consequences side by side.
- **Check before scaling.** Test representative samples before running the
  workflow across an image collection.
- **Keep the analysis record.** Save the graph, export Python, define batch
  outputs and preserve execution provenance.
- **Accelerate selected operations.** Dedicated GPU implementations are
  available for supported filtering, restoration, segmentation and measurement
  operations.

## Install

The current published release is
[`v0.14.0a2`](https://github.com/rensutheart/napari-vipp/releases/tag/v0.14.0a2).

| Platform | Recommended route |
| --- | --- |
| Windows 64-bit | Download `VIPP-Setup-0.14.0a2-Windows-x86_64-UNSIGNED.exe` from the release page. The setup application creates and manages a dedicated VIPP environment. A supported 64-bit Python is a separate prerequisite. |
| macOS Apple Silicon | Download `VIPP-0.14.0a2-macOS-arm64-UNSIGNED.pkg`. The package is self-contained and CPU-only. |
| macOS Intel | Download `VIPP-0.14.0a2-macOS-x86_64-UNSIGNED.pkg`. The package is self-contained and CPU-only. |
| Linux or an existing Python environment | Use the manual installation below. CPU execution is supported. |

The desktop installers are unsigned alpha builds. Download them only from the
official release, verify the matching SHA-256 file, and follow the
[Quick Start](docs/quick-start.md) for the exact platform instructions.

For a manual installation, use a dedicated CPython 3.12 or 3.13 environment:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.14.0a2"
vipp
```

Inside an existing napari session, open:

```text
Plugins > VIPP Workflow (napari-vipp)
```

## Try A Bundled Workflow

No external data is needed for the first run:

1. Launch VIPP and select **Open example...**.
2. Open **Red-Channel Label Cleanup**.
3. Select the nodes from left to right. Inspect their parameters, previews,
   histograms, metadata and outputs.
4. Change a parameter and compare the affected stages. Use **Calculate** where
   a computationally intensive node requires an explicit run.
5. Pin an output into napari for full-resolution inspection, then save the
   workflow as JSON.

Next, open **Deterministic Batch & Provenance** for a self-contained example of
collection processing and reproducibility artifacts. See the
[example workflow index](examples/README.md) for more starting points.

## Analysis Coverage

| Goal | Available building blocks |
| --- | --- |
| Prepare images | Intensity transforms, background correction, filtering, denoising, channel handling, axis operations, masks and volume regions of interest. |
| Segment structures | Global and local thresholds, edges, watershed, binary morphology, label cleanup and connected-component operations. |
| Quantify results | Object and intensity measurements, calibrated 3D mesh morphology, skeleton and network analysis, colocalisation, object association and table composition. |
| Restore images | Measured or generated point-spread functions, Richardson–Lucy and RL–TV deconvolution in 2D and 3D. |
| Reuse analyses | Workflow JSON, generated Python, explicit batch outputs, collection manifests and execution provenance. |

VIPP reads OME-TIFF, ImageJ TIFF, TIFF, local OME-Zarr 0.4/0.5,
NPY/NPZ and common 2D image formats. Optional readers add formats including ND2
and CZI. The [I/O guide](docs/io-user-guide.md) documents the complete matrix,
metadata behavior and limitations.

Most graph operations currently materialize their inputs in memory. Plan cache,
preview and output choices deliberately for large z-stacks and OME-Zarr data.
See [Cache and memory](docs/cache-and-memory.md).

## GPU Acceleration (Optional)

VIPP is fully usable on CPU on Windows, Linux and macOS. Selected
computationally intensive operations offer dedicated GPU acceleration on
qualified NVIDIA systems. VIPP records the implementation that ran, and a GPU
request that cannot be honored falls back visibly to CPU.

The current public GPU route uses native 64-bit Windows, CPython 3.12, CUDA 13
and an NVIDIA GPU with compute capability 7.5 or newer. Performance depends on
the operation, data and hardware. See the [GPU Guide](docs/gpu-guide.md) for
installation, supported operation families, qualification and benchmarking.

## Reproducibility And Scientific Traceability

VIPP keeps workflow structure, parameters and intermediate decisions visible
during interactive design. Saved and batch-run workflows can retain source
identity, semantic axes, physical calibration, selected outputs and the actual
CPU or GPU implementation used.

This record supports reproducibility, but it does not establish biological
validity automatically. Users remain responsible for checking assumptions,
parameter choices and results on suitable controls and representative data.
Read the [scientific integrity boundaries](docs/architecture.md#scientific-integrity-boundaries)
and [scientific behavior requirements](CONTRIBUTING.md#scientific-behavior-requirements).

## Documentation

- [Quick Start](docs/quick-start.md) — installers, manual setup and the first workflow
- [User Guide](docs/user-guide.md) — graph authoring, inspection, batch processing and export
- [Example workflows](examples/README.md) — bundled starting points by analysis task
- [Image import and export](docs/io-user-guide.md) — formats, metadata and collection inputs
- [GPU Guide](docs/gpu-guide.md) — acceleration, qualification and benchmarking
- [Measurement workflows](docs/measurement-workflows.md) — quantitative analysis patterns
- [Architecture](docs/architecture.md) — execution and scientific-integrity contracts
- [Published manual](https://rensutheart.github.io/vipp-mkdocs/) — versioned documentation site
- [Changelog](CHANGELOG.md) and [roadmap](docs/planning.md)

## Development And Support

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Ask usage
questions in [GitHub Discussions](https://github.com/rensutheart/napari-vipp/discussions),
use [SUPPORT.md](SUPPORT.md) for help, report vulnerabilities privately through
[SECURITY.md](SECURITY.md), and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation And License

If VIPP contributes to your work, acknowledge `napari-vipp` and link to this
repository. Citation metadata is available in [CITATION.cff](CITATION.cff). A
DOI or manuscript citation will be added when available.

napari-vipp is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full terms.
