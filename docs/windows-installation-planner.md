# Windows Installer And Planning Contract

VIPP `0.13.0a7` contains both the read-only installation planner and a
transactional managed-environment executor behind a novice-facing Windows
setup window. The 0.13.0a7 release executable must be built from the immutable
`v0.13.0a7` tag with that tag's exact wheel, published only with an explicit
`-UNSIGNED` filename, and bound to its release manifest and SHA-256 checksum.
Windows therefore reports **Unknown publisher** for this alpha; the
[installer-first quick start](quick-start.md) requires checksum verification
before the user chooses **More info > Run anyway**.

This command is an implementation, diagnostics, and advanced-support surface.
The ordinary user path is the [installer-first quick start](quick-start.md).
The planner CLI on this page is for development, diagnostics, and advanced
support; users should download the exact release `.exe` rather
than copy planner or pip commands.

## Supported Plans

The planner covers four explicit routes:

| Environment | Compute track | Result |
| --- | --- | --- |
| Managed | CPU | A new VIPP app environment and a CPU launcher profile in each selected shortcut location |
| Managed | CUDA 13 | A new VIPP app/CUDA environment and Automatic, CPU, and Prefer-GPU profiles in each selected shortcut location |
| Existing napari venv | CPU | Plan the exact VIPP release and selected launcher profile without replacing napari or Qt |
| Existing napari venv | CUDA 13 | Plan the exact CUDA extra and selected launcher profiles after environment and hardware validation |

The existing-environment route is deliberately conservative. This first slice
accepts a normal 64-bit CPython virtual environment containing napari 0.6 or
newer and PyQt6. It rejects global Python, editable VIPP, mixed Qt bindings,
conflicting CuPy tracks, and environments that inherit system site-packages.
Managed installation remains the recommended route.

CPU installation supports CPython 3.12 and 3.13. CUDA installation requires
CPython 3.12 on native 64-bit Windows. CUDA discovery calls the installed
NVIDIA driver API directly; it does not import CuPy or restrict admission to a
GPU model name. The packaged public policy currently requires driver API
`13030` or newer and compute capability 7.5 or newer on every visible device.
Because the released runtime probes every visible CUDA ordinal before choosing
ordinal 0 as its default, every visible device must meet that architecture
floor; device ordinals are retained in the plan. The installer still runs the
complete VIPP compute doctor before it activates an installation.

For `0.13.0a7`, one-click managed setup calls
`SHGetKnownFolderPath(FOLDERID_LocalAppData)` and accepts only the exact track
roots `VIPP\environments\cpu` and `VIPP\environments\cuda13` beneath that
canonical directory. A custom managed root is a blocking request, not an
alternate installation route. The complete CUDA path must contain ASCII
characters because CuPy 14.1.1 cannot reliably compile NVRTC kernels when its
Windows environment include path contains non-ASCII characters. Spaces remain
supported. If canonical Local App Data is non-ASCII, CUDA one-click setup is
unavailable before dependency resolution, environment creation, or download,
and the UI offers CPU. The fixed CPU root remains Unicode-safe.

An expert-selected existing environment remains a separate, non-mutating
route. The planner can inspect it and report an incompatible CUDA path, but it
never moves, renames, edits, or converts that environment into a managed copy.

This means a mixed-generation workstation that also exposes a pre-Turing GPU
is blocked in this first installer slice even when another visible GPU
qualifies. Future per-device selection must be implemented in the runtime and
persisted by the launcher before the installer can safely relax that boundary.

## Managed Install, Update, And Repair

The setup program uses one fixed managed environment per track and requires only
one consequential confirmation. Before confirmation it discovers the computer,
resolves the exact binary packages, verifies their SHA-256 digests, and shows a
plain summary; package detail remains under **Advanced details**. Apply uses the
reviewed, hash-locked resolution rather than resolving a second time.
Changing an install-relevant selection after reviewâ€”the compute track,
selected existing environment, or desktop-shortcut choiceâ€”invalidates that
prepared transaction. The setup window disables Apply
and requires **Check these settings** again, so an earlier review or one-use
authorization cannot be applied to different settings.

Installation is staged in a new permanent versioned environment. VIPP,
`pip check`, the selected compute doctor, launchers, and shortcuts must pass before
the ownership record is switched. This gives each target one of these explicit
states:

| Existing target | Ordinary action |
| --- | --- |
| Missing or empty | **Install VIPP** |
| Older installer-owned VIPP | **Update VIPP**; retain the old active copy until acceptance |
| Same healthy version | **Open VIPP**, with **Repair** available |
| Same damaged version | **Repair VIPP** into a clean environment |
| Newer installer-owned version | Open it; never downgrade automatically |
| Non-empty unowned directory or unowned shortcut | Block; never overwrite or switch to a custom managed root |

An older installer-owned CUDA copy whose managed root contains non-ASCII
characters is not an ordinary in-place update or repair. After completing and
recording any recovery from a prior interrupted transaction, the newly blocked
selection performs no new mutation of its environment, ownership record, or
shortcuts and offers **Open Installed apps**. The user may uninstall **VIPP
(GPU)** through its ownership-bound remover, but setup does not offer another
managed root. If canonical Local App Data is non-ASCII, one-click CUDA remains
unavailable and the UI offers CPU. A second managed CUDA copy cannot safely
coexist because that track has one Apps & Features key and shared shortcut
names. Setup never relocates the old virtual environment and does not claim an
in-place or fallback migration.

An installation-specific ownership record is the only authority for replacing
shortcuts or retiring environments. Cancellation or failure before commit
removes only the marked candidate and preserves the previous active copy. A
cleanup problem is reported with the exact retained path rather than being
described as a successful rollback.

GPU dependency packages are large, so a healthy download may take several
minutes. Resolver and installer network operations use a 120-second idle socket
timeout and at most eight retries. The timeout measures a period with no network
data; it does not cap the total duration of a download or installation.
A transient network failure during Apply enters the same ownership-bound
rollback: the incomplete candidate is not activated, and any previous active
environment is preserved. **Try again** repeats read-only preparation for the
exact current selection and presents the resulting package review for a fresh
confirmation; it never reuses the failed transaction's resolution or
authorization.

Adding VIPP to a user-owned existing napari environment remains an Advanced,
non-mutating route in this installer slice. The setup program can validate it,
but it does not automatically modify that environment.

## Inspect A Plan

Run the source-current command from an installed VIPP environment. Spaces are
accepted as normal single arguments on every route. CPU paths may contain
Unicode characters; a managed or existing CUDA 13 environment root containing
non-ASCII characters is reported as a blocking preflight issue as described
above. Managed `--install-root` accepts only the exact canonical root; these
examples mirror the Windows known-folder result rather than offering a custom
location.

```powershell
$localAppData = [Environment]::GetFolderPath(
  [Environment+SpecialFolder]::LocalApplicationData
)
```

Managed CPU:

```powershell
vipp-install-plan plan `
  --mode managed `
  --track cpu `
  --base-python "C:\Path\To\Python313\python.exe" `
  --install-root (Join-Path $localAppData "VIPP\environments\cpu")
```

Managed CUDA 13:

```powershell
vipp-install-plan plan `
  --mode managed `
  --track cuda13 `
  --base-python "C:\Path\To\Python312\python.exe" `
  --install-root (Join-Path $localAppData "VIPP\environments\cuda13")
```

Existing napari virtual environment:

```powershell
vipp-install-plan plan `
  --mode existing `
  --track cpu `
  --environment-python "C:\Path\To\napari-env\Scripts\python.exe"
```

Use `--shortcuts none|desktop|start-menu|both` to review another shortcut
scope. `--shortcut-directory` can point at an explicit future destination.

The command writes one deterministic JSON document to standard output:

- exit `0`: preflight passed and ready only for dependency resolution, never
  ready for Apply, including plans with informational notices or warnings;
- exit `2`: blocked by one or more validation issues; and
- exit `3`: the discovery or packaged planner itself failed unexpectedly.

JSON is emitted as UTF-8 even when Windows redirects standard output through a
legacy console encoding, so Unicode installation paths remain lossless.

## Safety Boundary

Planning performs no pip install, venv creation, downloads, network access,
registry changes, shortcut creation, temporary-directory creation, or target
directory writes. It runs only an isolated standard-library identity probe in
the selected Python (`-I -S -B`) and, for CUDA plans, read-only NVIDIA driver
API discovery. Installed distributions in an existing venv are inspected from
metadata without importing napari, Qt, NumPy, or CuPy.

The selected interpreter, installation target, and shortcut parents must be
direct local paths. UNC paths, mapped remote volumes, and symbolic-link or
junction traversal are rejected before content is inspected. Existing venvs
must declare `include-system-site-packages = false` in `pyvenv.cfg` so inherited
package conflicts cannot escape the reviewed metadata inventory.
Reserved Windows device names, alternate-data-stream syntax, invalid/control
characters, and components ending in a period or space are also rejected
before path access.

For managed mode, discovery obtains canonical Local App Data from the Windows
known-folder API and derives the exact per-track root. A caller-supplied path
that differs from that value is rejected before resolution or mutation; setup
does not search for or create an alternate managed location.

Every planner document is schema `napari-vipp-install-plan`, version 2, and
explicitly records `plan_only: true` and `mutation_performed: false`. It contains:

- the selected interpreter and resolved environment target;
- a fingerprint binding discovery to the exact user request;
- stable validation issue codes and remediations;
- the exact top-level VIPP release requirement;
- proposed future action argument arrays, never shell command strings;
- intended shortcuts and acceptance commands;
- disk capacity and the rollback ownership boundary.

The free-space requirement is 5 GiB for a managed CPU environment, 15 GiB for a
managed CUDA environment, 2 GiB for an existing CPU environment, and 12 GiB
for an existing CUDA environment. These are conservative free-space gates for
downloads, extraction, installation, and rollback staging on local disks; they
are not claims about final environment size or GPU memory. A GPU with less than
15 GiB of VRAM can still qualify. VIPP evaluates available VRAM separately for
each operation and visibly uses CPU when a particular workload does not fit.

Before the guided setup begins dependency resolution, the transactional engine
also checks every volume used for Windows temporary files and VIPP installer
records. Each such volume needs at least 1 GiB free for CPU setup or 5 GiB free
for CUDA setup. This is a separate gate from the managed installation location;
if it fails, setup names the exact checked location, requirement, and available
space.

The standalone planner deliberately stops before network dependency resolution
or mutation. It records `ready_for_resolution: true` but always keeps
`ready_for_apply: false`,
`resolution_required: true`, and `execution_authorized: false`.

The setup application consumes that plan in a separate preparation phase. Its
executor performs non-mutating pip resolution, retains the exact artifact URLs
and SHA-256 values, and enables Apply only after a one-use confirmation bound to
that immutable resolution. The planner JSON therefore remains an honest
read-only artifact even though the complete setup application can continue to
an authorized transaction.

Before it reviews a new plan, graphical setup also completes any durable
rollback journal left by an earlier interrupted setup. That recovery may remove
only residue already authorized by the prior transaction; it is not a mutation
from the newly selected plan. A CUDA path blocker still prevents all new
dependency resolution, download, environment creation, shortcut, and registry
work.
