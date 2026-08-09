# Desktop Startup And Installer Plan

This document separates the startup experience implemented in the source tree
from the signed, one-download installers that remain future work. The product
goal is that a microscopy user can start VIPP without first learning napari,
Python virtual environments, or terminal activation.

## Source-Current Startup Foundation

The packaged launcher provides the same branded startup window on Windows,
Linux, and macOS. It reports real milestones from a separate VIPP application
process rather than advancing a cosmetic timer:

1. start Python;
2. load napari and its plugins;
3. create the viewer;
4. load VIPP's scientific modules;
5. build the workflow interface; and
6. prepare the initial workflow.

Elapsed time and a retained diagnostic log remain available when startup
fails. After five minutes, the user can keep waiting or hide the splash; neither
choice terminates VIPP. This matters on a first CUDA/cuCIM launch, when local
kernel compilation and cache creation can legitimately take several minutes.

Installed entry points are split by intent:

| Command | Intended shortcut label | Initial compute policy |
| --- | --- | --- |
| `vipp-app` | VIPP Automatic | Auto |
| `vipp-cpu` | VIPP CPU | CPU only |
| `vipp-prefer-gpu` | VIPP Prefer GPU | Prefer scientifically eligible GPU implementations |
| `vipp` | VIPP command-line launcher | Auto, or `--profile`/`--no-splash` |

The graphical commands share the same identity and milestones. Their badge and
accent differ so the session policy is visible without suggesting that a GPU
shortcut guarantees every node will run on a GPU. Scientific, environment,
memory, and fallback gates still apply.

When VIPP is opened from napari's Plugins menu, napari cannot display the
standalone process splash. The plugin therefore returns a lightweight branded
host immediately, imports the large scientific composition root in the
background, constructs Qt widgets on the GUI thread, and evaluates the initial
workflow exactly once. Import and construction failures are visible and
retryable in the panel.

## Installation Personas

The eventual installer should make four choices explicit without presenting a
large dependency matrix to ordinary users:

| Existing setup | CPU | GPU |
| --- | --- | --- |
| No napari environment | Create a managed VIPP environment and CPU shortcut | Create a managed VIPP CUDA environment and Auto/CPU/Prefer-GPU shortcuts |
| Existing napari environment | Validate Python/Qt, then install VIPP into the selected environment | Validate the exact supported GPU stack before changing the selected environment |

Creating a separate managed environment should be the recommended default.
Installing into an existing napari environment is an expert route because its
packages may conflict with VIPP's release constraints. The installer must show
the selected interpreter, environment directory, planned package changes, and
rollback boundary before applying them.

Platform scope should remain truthful:

- Windows: managed CPU and qualified NVIDIA CUDA routes;
- Linux: managed CPU first, then GPU only after native-Linux policy evidence is
  released;
- macOS: managed CPU while no Apple accelerator provider is admitted.

Python and the NVIDIA display driver may remain separate prerequisites in the
first installer generation. The bootstrapper should detect them, link to the
correct prerequisite when absent, and resume without requiring users to copy a
series of shell commands. The pip-provided CUDA component wheels remain inside
the managed environment; a separate system CUDA Toolkit is not required for
the standard VIPP CUDA route.

## Separate cuCIM Local-Build Installer

cuCIM remains a separate Windows download because VIPP does not redistribute a
prebuilt private cuCIM wheel. Release maintainers create the deterministic
bundle from a clean, committed source tree with:

```powershell
py -3.12 .\scripts\package_cucim_windows_installer.py
```

After extraction, the user double-clicks `Install VIPP cuCIM.cmd`. The bundle:

- asks for `python.exe` in an already installed released VIPP CUDA 13
  environment;
- runs the pinned local source build and independent reproduction/probes;
- passes the resulting wheel and strict build manifest to the existing
  released-environment setup verifier;
- runs CUDA, cuCIM, provenance, and `pip check` acceptance; and
- retains the local wheel, manifest, log, and atomic run journal so a completed
  build can resume installation without rebuilding.

The ZIP contains no cuCIM wheel. Its manifest records the VIPP version, exact
source commit, entry point, file sizes, and SHA-256 of every bundled helper.
There is no approval hash for a user to type or edit. See
[`scripts/README-cucim-windows-installer.md`](../scripts/README-cucim-windows-installer.md)
for prerequisites and support paths.

## Installer Delivery Stages

1. **Packaged startup layer:** ship branding, the graphical entry points, real
   startup milestones, plugin loading host, and wheel smoke tests.
2. **Windows bootstrapper:** add a signed per-user `.exe` that offers managed
   CPU or CUDA installation, existing-environment validation, install location,
   shortcuts, progress, logs, repair, and uninstall. Keep cuCIM as the separate
   locally building bundle.
3. **Linux desktop package:** reuse the same Python launcher and environment
   plan, creating `.desktop` entries and icons without assuming one desktop
   environment.
4. **macOS application/bootstrapper:** reuse the launcher and CPU environment
   plan with an app bundle and normal macOS signing/notarization.

The front ends may differ, but the environment planner and acceptance contract
should remain shared and testable without a GUI.

## Release Gates

Every installer or launcher release should verify:

- clean CPU installation and launch on Windows, Linux, and macOS;
- Windows CUDA installation, compute doctor, real eligible GPU execution,
  visible CPU decisions, cleanup, and fallback reporting;
- spaces and non-ASCII characters in user-selected paths;
- cancel, retry, repair, uninstall, insufficient disk, offline/network failure,
  and interrupted-download behavior;
- exact wheel/version/entry-point/branding contents from the immutable release
  tag; and
- no environment mutation during plan-only checks or before the user confirms
  the reviewed plan.
