# VIPP Quick Start

**Visual image processing made approachable.**

## Windows: The Recommended Route

The normal VIPP experience is one Windows installer: download the explicitly
named unsigned `.exe`, verify it, review the proposed location and compute
option, and launch VIPP from the shortcuts it creates.

**[Download the VIPP 0.13.0a5 Windows installer (unsigned alpha)](https://github.com/rensutheart/napari-vipp/releases/download/v0.13.0a5/VIPP-Setup-0.13.0a5-Windows-x86_64-UNSIGNED.exe)**

Use only the file attached to the official
[`v0.13.0a5` GitHub release](https://github.com/rensutheart/napari-vipp/releases/tag/v0.13.0a5).
This alpha is intentionally not Authenticode-signed. **Unknown publisher** and
a **Windows protected your PC** warning are therefore expected. The same
release includes the SHA-256 checksum and release manifest.

1. Download both `VIPP-Setup-0.13.0a5-Windows-x86_64-UNSIGNED.exe` and
   `SHA256SUMS-Windows-0.13.0a5.txt` from the official release.
2. Open PowerShell in the download folder and run:

   ```powershell
   Get-FileHash -Algorithm SHA256 `
     .\VIPP-Setup-0.13.0a5-Windows-x86_64-UNSIGNED.exe
   ```

   The 64-character hash must exactly match the hash beside that filename in
   `SHA256SUMS-Windows-0.13.0a5.txt`. Stop and delete the installer if it does
   not match.
3. Double-click the installer. If Windows shows **Windows protected your PC**,
   select **More info**, confirm the app name ends in `-UNSIGNED.exe` and the
   publisher is **Unknown publisher**, then select **Run anyway**.
4. If Microsoft Defender or another antivirus identifies a threat, stop—do not
   allow it and do not disable security. Some work or school computers and
   Windows 11 systems with Smart App Control may block unsigned applications
   without offering **Run anyway**. Use the manual installation below on those
   computers.

After that one-time Windows warning, the ordinary setup flow is:

1. Keep the recommended private VIPP environment. The one-click installer does
   not modify an existing napari environment.
2. Leave **Automatic** selected for the simplest setup. To choose manually,
   expand **Advanced details**, then use **Computer use** to select **CPU** or
   **NVIDIA GPU**. CPU works on every supported Windows computer. Automatic
   chooses NVIDIA only when the driver, Python, CUDA 13, GPU architecture,
   package, memory, and scientific gates can be satisfied. The explicit NVIDIA
   choice stays visible, but setup will block installation and explain what is
   missing if those checks do not pass.
3. In **Reviewed settings**, confirm the exact installation location, the CPU
   or NVIDIA CUDA 13 route, and whether shortcuts will be added to the Start
   Menu only or to both the Start Menu and Desktop. Select **Install**, then
   wait for setup and its final checks to finish. If you change the
   computer-use choice, installation location, or desktop-shortcut choice,
   select **Check these settings** again. Setup will not enable **Install** for
   settings it has not checked.
4. For a CPU installation, open **VIPP** from the created shortcut. A CUDA
   installation instead provides **VIPP Automatic**, **VIPP CPU**, and
   **VIPP Prefer GPU** shortcuts; start with **VIPP Automatic**.

The installer does not silently replace an existing installation. An older
installer-owned copy is offered as **Update**, with the old working environment
kept until the replacement passes its checks. To repair the same healthy
version, rerun that version's VIPP setup `.exe`; its reviewed screen offers
**Open VIPP** and **Repair**. A newer version is not downgraded.
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

The installer treats Python and, for GPU use, a sufficiently
recent NVIDIA display driver as separate prerequisites. It will detect a
missing Python and link to the official
[Python 3.12.10 Windows release](https://www.python.org/downloads/release/python-31210/),
whose recommended 64-bit installer supports both VIPP CPU and CUDA setups. It
then allows the user to retry discovery without copying terminal commands. The
normal CUDA route installs its CUDA component packages inside the managed
environment; it does not require a system CUDA Toolkit, Visual Studio, CMake,
or `nvcc`. The first VIPP installation requires an internet connection while
the bootstrapper obtains packages from PyPI. Setup resolves and
hash-locks the concrete binary package set before the user confirms it. GPU
setup is a large download and can take several minutes. It currently needs at
least 15 GiB free on the installation drive while setup runs. This is disk
storage, not GPU memory (VRAM). GPU setup also needs at least 5 GiB free on each
drive used for Windows temporary files and VIPP installer records (normally the
Windows system drive); CPU setup needs at least 1 GiB there. Setup identifies
the exact location if this additional check fails. Setup allows up to 120
seconds without receiving network data before treating an attempt as stalled,
then retries only a limited number of times; 120 seconds is not a limit on the
total download or installation time. If a temporary network problem still
stops setup, the incomplete new copy is rolled back and any previous working
VIPP remains active.
After the connection recovers, **Try again** rechecks the computer-use,
location, and shortcut choices currently shown and presents them for review
before **Install** can be selected again.

The standard GPU installation works without cuCIM. The optional, separate
cuCIM Windows installer performs its verified build locally after the standard
VIPP CUDA environment is working; cuCIM is not required to start VIPP or use
the other qualified CuPy/CuPyX GPU operations.

## Manual Alpha Installation (Advanced And Non-Windows)

Use these commands when an advanced installation needs terminal-level control
or when installing on Linux or macOS.

### CPU On Windows, Linux, Or macOS

VIPP `0.13.0a5` supports CPython 3.12 and 3.13. Create and activate a dedicated
virtual environment first; do not install the application into a global/base
Python. Then run:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a5"
vipp
```

macOS is CPU-only in this alpha. CPU is also the authoritative fallback on
Windows and Linux.

### NVIDIA CUDA 13 On Windows

The current CUDA route requires native 64-bit Windows and CPython 3.12:

```powershell
py -3.12 -m venv ".venv-vipp-gpu-cu13"
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install "napari[pyqt6]>=0.6" "napari-vipp[gpu-cuda13]==0.13.0a5"
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
[Windows CUDA and cuCIM guide](https://rensutheart.github.io/vipp-mkdocs/0.13.0a5/getting-started/windows-cuda/)
only after the standard compute doctor passes.

### Existing napari Environment (Advanced)

For an existing, isolated napari virtual environment, use that environment's
Python explicitly. A conservative CPU update is:

```powershell
$napariPython = "C:\Path\To\napari-env\Scripts\python.exe"
& $napariPython -m pip install "napari-vipp==0.13.0a5"
& $napariPython -m pip check
```

Do not use this route for a global Python, an environment that exposes system
site-packages, an editable VIPP checkout, or an environment with multiple Qt
bindings. The planner additionally requires stable napari 0.6
or newer and PyQt6. Adding CUDA 13 to an existing environment is an expert
operation because Python, Qt, NumPy/SciPy/scikit-image, CuPy, and CUDA-package
constraints can conflict; a fresh managed CUDA environment remains the safer
manual route until the installer can review exact dependency changes.

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
