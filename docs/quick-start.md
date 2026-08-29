# VIPP Quick Start

**Visual image processing made approachable.**

## Windows: The Recommended Route

The normal VIPP experience is one Windows installer: download the explicitly
named unsigned `.exe`, verify it, review the proposed location and compute
option, and launch VIPP from the shortcuts it creates.

The official `0.14.0a3` prerelease and its checksum sidecars are public. Use
only that release surface; never download a guessed release asset.

**[Download the VIPP 0.14.0a3 Windows installer (unsigned alpha)](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a3/VIPP-Setup-0.14.0a3-Windows-x86_64-UNSIGNED.exe)**

Use only the file attached to the official
[`v0.14.0a3` GitHub release](https://github.com/rensutheart/napari-vipp/releases/tag/v0.14.0a3).
This alpha is intentionally not Authenticode-signed. **Unknown publisher** and
a **Windows protected your PC** warning are therefore expected. The same
release includes the SHA-256 checksum and release manifest.

1. Download both `VIPP-Setup-0.14.0a3-Windows-x86_64-UNSIGNED.exe` and
   `SHA256SUMS-Windows-0.14.0a3.txt` from the official release.
2. Open PowerShell in the download folder and run:

   ```powershell
   Get-FileHash -Algorithm SHA256 `
     .\VIPP-Setup-0.14.0a3-Windows-x86_64-UNSIGNED.exe
   ```

   The 64-character hash must exactly match the hash beside that filename in
   `SHA256SUMS-Windows-0.14.0a3.txt`. Stop and delete the installer if it does
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

1. Keep the fixed managed VIPP environment. The one-click installer does not
   modify an existing napari environment or accept a custom managed root.
2. Leave **Automatic** selected for the simplest setup. To choose manually,
   expand **Advanced details**, then use **Computer use** to select **CPU** or
   **NVIDIA GPU**. CPU works on every supported Windows computer. Automatic
   chooses NVIDIA only when the driver, Python, CUDA 13, GPU architecture,
   package, memory, and scientific gates can be satisfied. The explicit NVIDIA
   choice stays visible, but setup will block installation and explain what is
   missing if those checks do not pass.
3. In **Reviewed settings**, confirm the fixed installation location, the CPU
   or NVIDIA CUDA 13 route, and whether shortcuts will be added to the Start
   Menu only or to both the Start Menu and Desktop. Select **Install VIPP**,
   then wait for setup and its final checks to finish. If you change the
   computer-use choice or desktop-shortcut choice,
   select **Check these settings** again. Setup will not enable **Install VIPP**
   for settings it has not checked. Windows obtains canonical Local App Data with
   `SHGetKnownFolderPath(FOLDERID_LocalAppData)`; one-click setup uses only
   `VIPP\environments\cpu` or `VIPP\environments\cuda13` beneath it. Custom
   managed roots are not accepted. In `0.14.0a3`, the complete CUDA path must
   use ASCII characters only because CuPy 14.1.1 cannot reliably compile CUDA
   kernels from a Windows environment path containing characters such as `Å`
   or `é`. Spaces are supported. If canonical Local App Data contains a
   non-ASCII character, one-click CUDA is unavailable before environment
   creation or package download and setup offers CPU. The fixed CPU path
   remains Unicode-safe.
4. For a CPU installation, open **VIPP** from the created shortcut. A CUDA
   installation instead provides **VIPP Automatic**, **VIPP CPU**, and
   **VIPP Prefer GPU** shortcuts; start with **VIPP Automatic**.

The installer does not silently replace an existing installation. An older
installer-owned copy is offered as **Update**, with the old working environment
kept until the replacement passes its checks. To repair the same healthy
version, rerun that version's VIPP setup `.exe`; its reviewed screen offers
**Open VIPP** and **Repair**. A newer version is not downgraded.
Files, folders, shortcuts, or napari environments that the installer does not
own are never overwritten; setup blocks rather than choosing another managed
location.
Managed CPU and CUDA installations can coexist. Remove either one later from
**Windows Settings > Apps > Installed apps**; each has its own ownership-bound
uninstaller and the other installation is left intact.

An installer-owned CUDA copy already stored in a non-ASCII path is a special
case: `0.14.0a3` will not update or repair it in place. Graphical setup may
first complete and record recovery from an earlier interrupted transaction;
after that separate recovery, the newly blocked selection performs no new
mutation of the old copy, shortcuts, or ownership record. Do not move or rename
its virtual environment or start a second managed CUDA installation: the CUDA
track has one Windows Apps entry and shared shortcut names. Select **Open
Installed apps**, uninstall **VIPP (GPU)**, and let that ownership-bound removal
finish. If canonical Local App Data is non-ASCII, this account still cannot use
one-click CUDA; setup offers CPU. This is a deliberate release boundary, not
an in-place or fallback migration.

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

Setup makes the scale and activity of this work explicit. Its review shows
rounded estimates of 250 MiB download,
1.5 GiB installed, and 2.5 GiB peak working space for CPU, or 1.5 GiB, 5 GiB,
and 7 GiB respectively for CUDA. These estimates are separate from the
enforced 5/1 GiB CPU and 15/5 GiB CUDA installation/temp-drive minimums above;
none is a VRAM comparison. During setup, the current phase, elapsed time, quiet
heartbeat, latest concrete activity, and setup-log access remain visible.
Progress stays indeterminate unless the underlying dependency tool provides a
trustworthy byte total.

After the connection recovers, **Try again** rechecks the computer-use and
shortcut choices currently shown and presents them for review
before **Install VIPP** can be selected again.

The standard GPU installation includes every current reviewed CuPy/CuPyX
implementation. No separately built GPU provider is required.

## macOS: The Recommended Route

Choose the package that matches **Apple menu > About This Mac**:

- **Apple Silicon** (`Chip: Apple ...`):
  [download the arm64 PKG](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a3/VIPP-0.14.0a3-macOS-arm64-UNSIGNED.pkg)
  and its
  [SHA-256 file](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a3/SHA256SUMS-macOS-arm64-0.14.0a3.txt).
- **Intel** (`Processor: Intel ...`):
  [download the x86_64 PKG](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a3/VIPP-0.14.0a3-macOS-x86_64-UNSIGNED.pkg)
  and its
  [SHA-256 file](https://github.com/rensutheart/napari-vipp/releases/download/v0.14.0a3/SHA256SUMS-macOS-x86_64-0.14.0a3.txt).

Each package is offline, CPU-only, and current-user-only. It provides its own
managed Python environment at `~/Library/vipp`, creates
`~/Applications/VIPP.app`, and needs approximately 3 GB of free disk space.
It does not modify the system Python or shell startup files.

This alpha is explicitly unsigned and not notarized. Before opening it, verify
that its SHA-256 matches the line in the downloaded checksum file. The optional
Terminal check for Apple Silicon is:

```bash
shasum -a 256 VIPP-0.14.0a3-macOS-arm64-UNSIGNED.pkg
```

Use the `x86_64` filename on Intel. Stop if the value differs or the package did
not come from the official release. Double-click the verified PKG. If macOS
blocks it because the developer cannot be verified, choose **Done**, open
**System Settings > Privacy & Security**, find the blocked VIPP package under
**Security**, choose **Open Anyway**, confirm the exact package name, and
approve the Installer prompt. A managed work or school Mac may prohibit this
override; use the manual route below or ask its administrator instead of
weakening security settings.

After installation, open **VIPP** from `~/Applications`. First launch may take
longer while napari loads. To remove this alpha, move only
`~/Applications/VIPP.app` and `~/Library/vipp` to Trash. There is no graphical
updater or uninstaller yet; installing a later package replaces the managed
environment only when that later release explicitly documents a supported
update. Otherwise remove both installer-owned paths before reinstalling.

The [macOS packaging guide](../packaging/macos/README.md) records the exact
offline build, lifecycle checks, licence inventory, and future Developer ID and
notarization path. The PKG itself performs installation, so a DMG wrapper is not
required.

## Manual Alpha Installation (Advanced And Portable)

Use these commands for terminal-level control, Linux, or integration with an
existing napari environment.

### CPU On Windows, Linux, Or macOS

VIPP `0.14.0a3` supports CPython 3.12 and 3.13. Create and activate a dedicated
virtual environment first; do not install the application into a global/base
Python. Then run:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.14.0a3"
vipp
```

macOS is CPU-only in this alpha. CPU is also the authoritative fallback on
Windows and Linux.

### NVIDIA CUDA 13 On Windows

The current CUDA route requires native 64-bit Windows and CPython 3.12:

Run these commands from an ASCII-only working directory:

```powershell
py -3.12 -m venv ".venv-vipp-gpu-cu13"
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install "napari[pyqt6]>=0.6" "napari-vipp[gpu-cuda13]==0.14.0a3"
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13
& ".\.venv-vipp-gpu-cu13\Scripts\vipp.exe"
```

This advanced manual environment is separate from one-click management. The
installer does not move or edit it. Create a fresh manual environment rather
than moving or renaming one whose complete path is incompatible.

The GPU model is recorded for reproducibility rather than checked against a
model allowlist. The current public gate requires NVIDIA compute capability
7.5 or newer, CUDA runtime API 13.2, driver API 13.3 or newer, and the pinned
scientific/provider stack. On a mixed-GPU workstation, every currently visible
CUDA device must meet the architecture floor because the released runtime
probes all ordinals before choosing its default; the installer must identify
any failing ordinal. Unsupported work stays on CPU with an explanation.

For troubleshooting or the Advanced source-checkout route, follow the
[GPU guide](gpu-guide.md).

### Existing napari Environment (Advanced)

For an existing, isolated napari virtual environment, use that environment's
Python explicitly. A conservative CPU update is:

```powershell
$napariPython = "C:\Path\To\napari-env\Scripts\python.exe"
& $napariPython -m pip install "napari-vipp==0.14.0a3"
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
  families, CPU fallback, benchmarking, and reproducibility.
- [Windows installer and planning contract](windows-installation-planner.md):
  the implemented read-only planner, transactional managed installer, update
  and repair behavior, and release-safety boundary.
- [Desktop startup and installer plan](desktop-startup-and-installer-plan.md):
  delivery stages for Windows, Linux, and macOS.
- [User guide](user-guide.md): workflows, previews, saving, export, and batch
  processing.
