# VIPP 0.13.0a5

VIPP 0.13.0a5 introduces the recommended Windows installer and branded
desktop launchers. The managed installer is designed for people who do not want
to manage Python environments or CUDA packages themselves: it checks the
computer, recommends CPU or a qualified NVIDIA CUDA 13 setup, shows the exact
location and shortcut choices, and performs final acceptance checks before the
new installation becomes active.

VIPP remains alpha software. Validate representative data and retain the exact
application, workflow, inputs, environment, implementation provenance, and
batch artifacts used for consequential analysis.

## Windows installation

Download `VIPP-Setup-0.13.0a5-Windows-x86_64-UNSIGNED.exe` only from this
release. This alpha is intentionally not Authenticode-signed. Windows will
therefore show **Unknown publisher** and may show **Windows protected your PC**.
Verify the installer against the attached `SHA256SUMS-Windows-0.13.0a5.txt`
before selecting **More info > Run anyway**. Do not continue if the hash differs
or antivirus identifies a threat, and never disable Windows security. A machine
whose policy does not offer **Run anyway** must use the manual installation.

The exact release manifest and third-party notices are attached beside the
installer. The explicit `-UNSIGNED` filename is intentional; no unsigned file
is published under the filename reserved for a future signed installer.

The managed setup provides:

- an Automatic recommendation with clear CPU and NVIDIA GPU choices;
- a private VIPP environment that does not overwrite unrelated folders or
  manually managed napari environments;
- transactional install, update, and repair, with the previous working version
  retained until its replacement passes acceptance;
- separate CPU and CUDA installations that may coexist and can be removed
  independently from Windows Installed apps;
- Start Menu shortcuts, an optional Desktop shortcut, and branded Automatic,
  CPU, and Prefer-GPU launch profiles where applicable; and
- bounded network retries, explicit disk-space checks, retained diagnostics,
  and ownership-safe rollback and uninstall.

CPU setup supports CPython 3.12 and 3.13. The managed CUDA route uses CPython
3.12 and requires native 64-bit Windows, a compatible NVIDIA display driver,
and a GPU meeting the released CUDA policy. The normal route installs CUDA
libraries inside the VIPP environment; it does not require a separate CUDA
Toolkit, Visual Studio, CMake, or `nvcc`.

Python remains a separate prerequisite in this alpha. If a supported Python is
missing, setup links to the official CPython 3.12.10 Windows installer and lets
the user check again afterward.

## Startup and napari integration

Installed GUI entry points provide branded Automatic, CPU-only, and Prefer-GPU
sessions with real startup progress and retained diagnostics. Opening VIPP from
an existing napari session now shows a lightweight branded loading host while
the full editor starts. Startup failures remain visible and retryable.

## GPU and optional cuCIM

VIPP retains the compatible-device CUDA 13 policy from 0.13.0a4: GPU model
names are recorded for provenance rather than used as an allowlist, while the
exact driver, compute capability, Python, scientific-stack, provider, memory,
workload, and scientific-parity gates still apply. Auto may correctly choose
CPU. Prefer GPU requests each eligible accelerated implementation, with visible
CPU fallback for work outside the reviewed region.

The standard CUDA installation works without cuCIM. This release includes a
separate no-wheel cuCIM Windows bundle that builds the pinned cuCIM source
locally and installs the resulting private wheel only after its manifest,
artifact, environment, and acceptance checks pass. No redistributable cuCIM
wheel is included.

## Manual installation

The pip route remains available for Linux, macOS, existing-napari integration,
and advanced use:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a5"
vipp
```

CPU supports CPython 3.12 and 3.13. CUDA is native-Windows/CPython-3.12-only in
this alpha. Follow the versioned Quick Start and Windows CUDA guide before
adding GPU packages to an environment manually.

## Validation

The exact 0.13.0a5 release candidate passed the cross-platform source, package,
and installer-smoke suites. The development installer also passed fresh CPU
and CUDA managed installation, real Auto and Prefer-GPU execution with CPU
parity and cleanup, optional cuCIM installation and execution, update/repair,
and independent CPU/CUDA removal on the Windows reference system. The release
file is produced only from the immutable tag, explicitly named `-UNSIGNED`,
bound to the exact wheel and source commit, hash-recorded, and rechecked before
publication.

See the repository [Quick Start](docs/quick-start.md), [GPU Guide](docs/gpu-guide.md),
and [changelog](CHANGELOG.md#0130a5---2026-08-12) for complete instructions and
technical detail.
