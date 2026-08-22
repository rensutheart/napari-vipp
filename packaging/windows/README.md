# Windows Setup Packaging

This directory contains the reviewed PyInstaller configuration for the single
novice-facing Windows VIPP setup program. CPU or NVIDIA GPU support is selected
inside that program; there are not separate CPU and GPU setup executables.
The setup program uses a separately installed supported 64-bit Python. If none
is available, its guided screen links to the official Python 3.12.10 installer
and lets the user check again without entering terminal commands.
Its approval screen names the exact managed location, resolved CPU or CUDA 13
route, and Start Menu/Desktop shortcut scope. The GPU space allowance is
described as installation-drive storage rather than GPU memory (VRAM). The UI
and user guide separately explain the 5 GiB GPU (1 GiB CPU) allowance on every
drive used for Windows temporary files and installer records, and setup names
the exact failing location.

The release build embeds the exact `napari-vipp` wheel built from the same tag.
The wheel SHA-256 is checked before the installer resolves dependencies and its
path, digest, source commit, and version are retained in build/release manifests.

Use [`scripts/package_windows_installer.py`](../../scripts/package_windows_installer.py)
from the repository root. `build --development` creates a local smoke artifact.
That executable identifies itself as **DEVELOPMENT BUILD — local testing only**
in its title, `--version` output, and Advanced details, and its embedded channel
marker cannot be finalized as a release.
An official build requires a clean exact `v<version>` tag and produces only a
`SIGNING-STAGING` executable. The real signing hook is
[`scripts/sign_windows_installer.ps1`](../../scripts/sign_windows_installer.ps1).
Only `finalize`, after verifying the approved Authenticode signer and timestamp,
may create `VIPP-Setup-<version>-Windows-x86_64.exe`.

When a release explicitly chooses to ship unsigned, `finalize-unsigned` may
instead create only
`VIPP-Setup-<version>-Windows-x86_64-UNSIGNED.exe`. It requires the same clean
exact tag, reproducible wheel, frozen-payload and checksum checks, rejects a
development build or changed staging file, confirms Windows reports
`NotSigned`, and writes the release manifest, notices, and SHA-256 sidecar. It
cannot create the filename reserved for a signed release.

The EXE embeds the VIPP licence, the exact CPython/Tcl-Tk/PyInstaller licence
texts from the pinned build runtime, the public compute policy, and a VIPP icon.
The combined third-party notices are also a persistent release sidecar and are
available from the installer's advanced details.

The standard CUDA environment contains every current reviewed GPU provider;
there is no separate companion provider asset.

The canonical release build commands and risk-based gate selection are in the
[release runbook](../../docs/release-runbook.md). Full clean-machine lifecycle
acceptance is a conditional release-candidate/production or installer-change
procedure, not a requirement for every iterative alpha.
