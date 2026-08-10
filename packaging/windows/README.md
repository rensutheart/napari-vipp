# Windows Setup Packaging

This directory contains the reviewed PyInstaller configuration for the single
novice-facing Windows VIPP setup program. CPU or NVIDIA GPU support is selected
inside that program; there are not separate CPU and GPU setup executables.

The release build embeds the exact `napari-vipp` wheel built from the same tag.
The wheel SHA-256 is checked before the installer resolves dependencies and its
path, digest, source commit, and version are retained in build/release manifests.

Use [`scripts/package_windows_installer.py`](../../scripts/package_windows_installer.py)
from the repository root. `build --development` creates a local smoke artifact.
An official build requires a clean exact `v<version>` tag and produces only a
`SIGNING-STAGING` executable. The real signing hook is
[`scripts/sign_windows_installer.ps1`](../../scripts/sign_windows_installer.ps1).
Only `finalize`, after verifying the approved Authenticode signer and timestamp,
may create `VIPP-Setup-<version>-Windows-x86_64.exe`.

The EXE embeds the VIPP licence, the exact CPython/Tcl-Tk/PyInstaller licence
texts from the pinned build runtime, the public compute policy, and a VIPP icon.
The combined third-party notices are also a persistent release sidecar and are
available from the installer's advanced details.

cuCIM is never part of the primary EXE. When selected, its deterministic
no-wheel local-build ZIP is a separate companion release asset. Finalization
binds that ZIP's version, source commit, and SHA-256 into the release manifest.

The complete operator sequence and clean-machine acceptance gates are in the
[release runbook](../../docs/release-runbook.md).
