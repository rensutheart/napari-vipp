# VIPP Quick Start

**Visual image processing made approachable.**

## Windows: The Recommended Route

The normal VIPP experience will be a signed Windows installer: download one
`.exe`, double-click it, review the proposed location and compute option, and
launch VIPP from the shortcuts it creates.

> **Current status:** the signed installer is the next delivery stage and is
> not yet published. VIPP `0.13.0a4` still uses the manual installation below.
> Until an installer is attached to an official
> [napari-vipp GitHub release](https://github.com/rensutheart/napari-vipp/releases),
> do not download similarly named installers from another location.

When the installer is released, this page and the repository README will link
directly to the deterministic release asset
`VIPP-Setup-<version>-Windows-x86_64.exe`. The ordinary flow will be:

1. Download the signed Windows VIPP installer from the official release.
2. Double-click it. No terminal activation should be required.
3. Keep the recommended private VIPP environment. The one-click installer does
   not modify an existing napari environment.
4. Accept the recommended compute route:
   - **CPU** works on every supported Windows computer; or
   - **NVIDIA GPU** is offered only when the driver, Python, CUDA 13, GPU
     architecture, package, memory, and scientific gates can be satisfied.
5. In **Reviewed settings**, confirm the exact installation location, the CPU
   or NVIDIA CUDA 13 route, and whether shortcuts will be added to the Start
   Menu only or to both the Start Menu and Desktop. Select **Install**, then
   wait for setup and its final checks to finish. If you change the
   computer-use choice, installation location, or desktop-shortcut choice,
   select **Check these settings** again. Setup will not enable **Install** for
   settings it has not checked.
6. Open **VIPP Automatic** from the created shortcut. CUDA installations also
   provide **VIPP CPU** and **VIPP Prefer GPU** shortcuts.

The installer does not silently replace an existing installation. An older
installer-owned copy is offered as **Update**, with the old working environment
kept until the replacement passes its checks. The same healthy version offers
**Open VIPP** and an optional **Repair**. A newer version is not downgraded.
Files, folders, shortcuts, or napari environments that the installer does not
own are never overwritten; setup asks for a separate managed location instead.
Managed CPU and CUDA installations can coexist. Remove either one later from
**Windows Settings > Apps > Installed apps**; each has its own ownership-bound
uninstaller and the other installation is left intact.

Automatic and Prefer GPU never guarantee that every node runs on a GPU. VIPP
uses CPU whenever an operation, datatype, parameter, environment, or memory
condition is outside the validated GPU region, and reports that decision.

Installing into an existing napari environment is an **Advanced manual** route.
The first one-click installer deliberately leaves those environments unchanged;
use the version-pinned instructions below only when that integration is needed.

The first installer release will treat Python and, for GPU use, a sufficiently
recent NVIDIA display driver as separate prerequisites. It will detect a
missing Python and link to the official
[Python 3.12.10 Windows release](https://www.python.org/downloads/release/python-31210/),
whose recommended 64-bit installer supports both VIPP CPU and CUDA setups. It
then allows the user to retry discovery without copying terminal commands. The
normal CUDA route installs its CUDA component packages inside the managed
environment; it does not require a system CUDA Toolkit, Visual Studio, CMake,
or `nvcc`. The first VIPP installation requires an internet connection while
the signed bootstrapper obtains packages from PyPI. Setup resolves and
hash-locks the concrete binary package set before the user confirms it. GPU
setup is a large download and can take several minutes. It currently needs at
least 15 GiB free on the installation drive while setup runs. This is disk
storage, not GPU memory (VRAM). Setup allows up to 120 seconds without receiving
network data before treating an attempt as stalled, then retries only a limited
number of times; 120 seconds is not a limit on the total download or
installation time. If a temporary network problem still stops setup, the
incomplete new copy is rolled back and any previous working VIPP remains active.
After the connection recovers, **Try again** rechecks the computer-use,
location, and shortcut choices currently shown and presents them for review
before **Install** can be selected again.

The standard GPU installation works without cuCIM. The optional, separate
cuCIM Windows installer performs its verified build locally after the standard
VIPP CUDA environment is working; cuCIM is not required to start VIPP or use
the other qualified CuPy/CuPyX GPU operations.

## Available Today: Manual Alpha Installation

Use these commands only until the signed installer is published, or when an
advanced installation needs terminal-level control.

### CPU On Windows, Linux, Or macOS

VIPP `0.13.0a4` supports CPython 3.12 and 3.13. Create and activate a dedicated
virtual environment first; do not install the application into a global/base
Python. Then run:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a4"
vipp
```

macOS is CPU-only in this alpha. CPU is also the authoritative fallback on
Windows and Linux.

### NVIDIA CUDA 13 On Windows

The current CUDA route requires native 64-bit Windows and CPython 3.12:

```powershell
py -3.12 -m venv ".venv-vipp-gpu-cu13"
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install "napari[pyqt6]>=0.6" "napari-vipp[gpu-cuda13]==0.13.0a4"
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13
& ".\.venv-vipp-gpu-cu13\Scripts\vipp.exe"
```

The GPU model is recorded for reproducibility rather than checked against a
model allowlist. The current public gate requires NVIDIA compute capability
7.5 or newer, CUDA runtime API 13.2, driver API 13.3 or newer, and the pinned
scientific/provider stack. On a mixed-GPU workstation, every currently visible
CUDA device must meet the architecture floor because the released runtime
probes all ordinals before choosing its default; the installer must identify
any failing ordinal. Unsupported work stays on CPU with an explanation.

For the optional local cuCIM build, follow the
[Windows CUDA and cuCIM guide](https://rensutheart.github.io/vipp-mkdocs/0.13.0a4/getting-started/windows-cuda/)
only after the standard compute doctor passes.

### Existing napari Environment (Advanced)

For an existing, isolated napari virtual environment, use that environment's
Python explicitly. A conservative CPU update is:

```powershell
$napariPython = "C:\Path\To\napari-env\Scripts\python.exe"
& $napariPython -m pip install "napari-vipp==0.13.0a4"
& $napariPython -m pip check
```

Do not use this route for a global Python, an environment that exposes system
site-packages, an editable VIPP checkout, or an environment with multiple Qt
bindings. The source-current planner additionally requires stable napari 0.6
or newer and PyQt6. Adding CUDA 13 to an existing environment is an expert
operation because Python, Qt, NumPy/SciPy/scikit-image, CuPy, and CUDA-package
constraints can conflict; a fresh managed CUDA environment remains the safer
manual route until the signed installer can review exact dependency changes.

## First Workflow

After VIPP opens:

1. Select **Open example...**.
2. Choose **Red-Channel Label Cleanup**.
3. Select nodes from left to right to inspect inputs, parameters, previews,
   metadata, and outputs.
4. Save the workflow when you want to reuse it.

Inside an existing napari session, open **Plugins > VIPP Workflow
(napari-vipp)**. Source-current builds show a branded loading panel while the
full workflow interface starts.

## Advanced And Implementation References

- [GPU guide](gpu-guide.md): qualification, compute modes, eligible operation
  families, CPU fallback, benchmarking, cuCIM, and reproducibility.
- [Windows installer and planning contract](windows-installation-planner.md):
  the implemented read-only planner, transactional managed installer, update
  and repair behavior, and release-safety boundary.
- [Desktop startup and installer plan](desktop-startup-and-installer-plan.md):
  delivery stages for Windows, Linux, and macOS.
- [User guide](user-guide.md): workflows, previews, saving, export, and batch
  processing.
