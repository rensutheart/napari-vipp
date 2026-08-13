# VIPP 0.13.0a6

VIPP 0.13.0a6 is a substantial usability and reliability release. It makes everyday graph editing faster, gives users a much clearer answer about whether GPU acceleration is actually available, improves multi-series and microscope-file handling, and adds stronger installation and release checks.

VIPP remains alpha software. Keep the original data and workflow, test representative images, and review important outputs before using them for scientific conclusions or publication.

## Features added

### Easier graph editing and reuse

- Select several nodes and move, copy, paste, or delete them as a group.
- Copy a node or connected group and paste it on the canvas. VIPP keeps internal connections, applicable tunnels, notes, positions, and settings while assigning safe new identities.
- Right-click a matching node and use **Paste Values** to transfer its settings. VIPP only enables this for the same operation, validates the change first, and makes the whole transfer one undoable action.
- Add a processing step before an existing named tunnel without rebuilding the downstream graph. Use **Insert node before tunnel**, drop a compatible palette item onto the tunnel, or drag a loose node onto it.
- Open **Graph Editing Acceptance Check** from the example chooser for an annotated workflow that explains exactly what to test.

### GPU setup that explains itself

- Compute Doctor now reports three separate things in plain language: whether CUDA works, whether optional GPU libraries such as cuCIM are ready, and which of VIPP's 13 reviewed GPU regions this computer can actually use.
- The window presents one recommended next step and keeps technical detail collapsed unless it is needed.
- A privacy-redacted support report can be saved for troubleshooting without exposing local paths, credentials, node identities, or machine fingerprints.
- A new strict qualification runner checks every public GPU implementation for scientific agreement, difficult inputs, metadata, input safety, memory, cancellation, cleanup, fallback, provenance, and complete-workflow timing.

### Better support for real microscope collections

- Imaris `.ims` files can use the shared optional microscope/BioIO reader route.
- Multi-series containers become clearly named batch items. The selected series remains visible in browsing, output names, manifests, and provenance. The `.ims`, metadata, and multi-series work incorporates and hardens contributions from Tom Naber in pull requests [#8](https://github.com/rensutheart/napari-vipp/pull/8), [#10](https://github.com/rensutheart/napari-vipp/pull/10), and [#13](https://github.com/rensutheart/napari-vipp/pull/13).
- The new **Set Microscope Metadata** node records up to three emission wavelengths, objective numerical aperture, and immersion refractive index without changing image pixels. A value of zero leaves existing metadata unchanged.

### Stronger installation and packaging checks

- Clean wheel and source-package installations are now checked across Windows, Linux, and macOS on supported Python versions, instead of testing a package only inside a development environment.
- The separate Windows cuCIM bundle has a scheduled reproducibility check. A protected real-GPU canary is also defined for a trusted Windows GPU runner; a skipped hardware job is never presented as a successful GPU test.
- A new field-acceptance checklist records fresh CPU and CUDA installation, paths with spaces and non-English characters, interrupted-install rollback, repair, update, uninstall, and a novice's first workflow. Anything not actually tested stays marked **not run**.

## Bug fixes

- CPU and Prefer GPU now both ignore deliberately loose graph fragments during normal calculation. This fixes the Graph Editing Acceptance Check and prevents disconnected demonstration nodes from being treated as runnable workflow branches.
- Multi-series batch execution now reads the selected series rather than silently falling back to the container's first series, and generated names remain collision-safe.
- Parameter pasting is transactional: invalid settings, interface refresh failures, and dynamic-port conflicts leave the target node and workflow unchanged.

## Windows installation

For the shortest route, download `VIPP-Setup-0.13.0a6-Windows-x86_64-UNSIGNED.exe` and `SHA256SUMS-Windows-0.13.0a6.txt` from this GitHub release. Verify the SHA-256 value before opening the installer.

This alpha is intentionally not Authenticode-signed. Windows will show **Unknown publisher** and may show **Windows protected your PC**. After verifying the official checksum, select **More info > Run anyway**. Stop if the checksum differs or antivirus reports a threat; never disable Windows security. If organizational policy does not allow the unsigned installer, use the manual installation route in the Quick Start.

The managed installer can install CPU or compatible NVIDIA CUDA 13 environments, keep CPU and GPU installations side by side, create launch shortcuts, repair or update an owned installation, and remove it without touching unrelated Python or napari environments. The standard CUDA installation includes the reviewed CuPy/CuPyX route and works without optional cuCIM.

## Optional cuCIM

Users who want the cuCIM-backed operations can download `napari-vipp-cucim-installer-0.13.0a6-windows.zip` from this release after installing the normal CUDA edition. The bundle contains no cuCIM wheel: it builds the pinned source locally for that user, verifies the result, and installs the private wheel into the selected VIPP environment. Extract the ZIP before running `Install VIPP cuCIM.cmd`. If updating from 0.13.0a5, update VIPP first and rebuild cuCIM with the matching a6 bundle; do not reuse the a5 bundle or move its private wheel between environments.

## Manual installation or upgrade

VIPP supports CPython 3.12 and 3.13 for CPU use. In PowerShell, Command Prompt, or a terminal with the intended environment activated:

```text
python -m pip install --upgrade "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a6"
vipp
```

For the supported native-Windows CUDA 13 route, use CPython 3.12 and follow the versioned Windows CUDA guide instead of mixing CUDA packages manually. Preserve the old environment and open copies of important workflows first. This release continues to read workflow schema 3 and writes schema 4; its batch configuration and manifest schema remain version 3.

When upgrading an installer-managed 0.13.0a5 installation, run the a6 installer, select the same CPU or CUDA route and managed location, review the detected update, and let setup retain the old working copy until a6 passes its checks. You can instead keep a5 and a6 side by side by choosing a new managed location. Do not point setup at an unrelated manually managed napari environment.

## What we validated

The complete source suite passed locally with 4,572 tests passing, five environment-dependent tests skipped, and two documented expected failures. The strict RTX 5090 quick qualification covered all 13 public GPU implementations across all 10 required evidence areas (130 of 130 mappings), and Compute Doctor admitted all 13 reviewed GPU regions on that reference system. Clean wheel and source-package installation tests, package metadata checks, documentation tests, and build checks also passed.

Those results do not replace testing the exact published installer on another computer. If you are helping test this release, use the [Windows field-acceptance checklist](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a6/docs/windows-installer-field-acceptance.md) and report only what you actually exercised.

See the [Quick Start](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a6/docs/quick-start.md), [GPU Guide](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a6/docs/gpu-guide.md), [full changelog](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a6/CHANGELOG.md#0130a6---2026-08-13), and [roadmap](https://github.com/rensutheart/napari-vipp/blob/v0.13.0a6/docs/planning.md) for more detail.
