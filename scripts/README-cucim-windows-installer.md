# VIPP Windows cuCIM installer coordinator

After extracting the standalone download, double-click
`Install VIPP cuCIM.cmd` in its top-level folder. It adds the pinned local
cuCIM build to an already installed VIPP CUDA 13 environment. The launcher
opens a standard Windows file selector for that environment's
`Scripts\python.exe`, then:

1. proves that the interpreter belongs to the exact released VIPP CUDA 13
   environment supported by this source revision;
2. runs `build_cucim_windows.ps1`, including its pinned source checkout, two
   independent builds, canonical payload comparison, licence checks, exact
   package inventory, `pip check`, and real-GPU probe;
3. retains the wheel and strict builder manifest in a unique per-run artifact
   directory;
4. delegates installation to
   `setup_gpu_dev.py --existing-environment`, which revalidates the manifest,
   archive and canonical payload, installs only the pinned runtime additions
   and local wheel, proves PEP 610 installed provenance, runs CUDA and cuCIM
   probes, runs `pip check`, and atomically writes the environment record.

There is no checksum or approval value for a user to copy or edit.

From a source checkout, the equivalent entry point is
`scripts\Install VIPP cuCIM.cmd`.

## Command-line use

```powershell
.\scripts\install_cucim_windows.ps1 `
  -TargetPython "C:\path\to\VIPP\Scripts\python.exe"
```

Preview the read-only plan without creating a log, building, or installing:

```powershell
.\scripts\install_cucim_windows.ps1 `
  -TargetPython "C:\path\to\VIPP\Scripts\python.exe" `
  -PlanOnly
```

If a build succeeded but installation did not, rerun without rebuilding:

```powershell
.\scripts\install_cucim_windows.ps1 `
  -TargetPython "C:\path\to\VIPP\Scripts\python.exe" `
  -ArtifactDirectory "C:\path\shown\in\the\failed\run\record"
```

By default, retained files are under
`%LOCALAPPDATA%\napari-vipp\cucim-installer`:

- `artifacts\RUN_ID` contains the wheel and builder manifest;
- `logs\RUN_ID.log` contains all progress and command output;
- `runs\RUN_ID.json` contains the plan, exact setup plan, status, and failure
  details when applicable;
- `builder-work` contains the reusable pinned source cache and isolated build
  environment.

The only expected machine-wide prerequisites are 64-bit CPython 3.12, Git for
Windows, PowerShell, and a compatible NVIDIA driver/GPU. The selected target
must already be an exact released VIPP CUDA 13 virtual environment.

The first local build is lengthy and network-, disk-, and GPU-intensive. Let it
finish without closing the console or suspending the machine. If cancellation
is unavoidable, the coordinator sends a bounded stop request to the complete
PowerShell process group and retains its log and run record. A wheel/manifest
pair from a build that completed successfully can resume only the installation
stage with `-ArtifactDirectory`; partial build output is never admitted.

## Current user-interface boundary

This is a console-progress installer with a graphical target-Python file
selector, not yet a signed native Windows wizard (`.msi`/`.exe`). The backend
is intentionally separated from that future packaging layer: a native wizard
can call the same plan and execution functions without changing any pinned
build, artifact-admission, environment-admission, or provenance rules.

## Creating the standalone download

Release maintainers can create a deterministic ZIP from a clean, committed
source tree that users extract and run without cloning the repository:

```powershell
py -3.12 .\scripts\package_cucim_windows_installer.py
```

The ZIP contains this guide, the project licence/notice, the double-click entry
point, coordinator, pinned builder, and released-environment setup helper in
their working relative layout. `bundle-manifest.json` records the VIPP version,
exact source commit, and SHA-256/size of every payload file. It deliberately
contains no prebuilt cuCIM wheel: each user's machine still performs the pinned
local build and retains its own manifest-verified artifact.
