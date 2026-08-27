# Desktop Startup And Installer Plan

This document separates the startup and transactional installer implemented in
the source tree from future code-signing work. VIPP `0.14.0a1` publishes one
checksum-qualified, explicitly unsigned Windows setup executable; an
Authenticode-signed release asset is not yet available. The product goal is that
a microscopy user can start VIPP without first learning napari, Python virtual
environments, or terminal activation.

The primary design persona is a physiologist who may not know what Python,
napari, virtual environments, CUDA packages, dependency resolution, or a
rollback boundary mean. The recommended path must therefore make the safe
choice on the user's behalf, explain only the practical effect, and require
one clear confirmation: **Install VIPP**. Package lists, interpreter paths,
provenance, and rollback details remain available under **Advanced details**
and in the retained log, but understanding them is never a prerequisite for a
normal managed installation.

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
choice terminates VIPP. This matters on a first CUDA launch, when local
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

## Internal Installation Routes

The implementation supports four routes, but the ordinary installer must not
present them as four technical choices:

| Existing setup | CPU | GPU |
| --- | --- | --- |
| No napari environment | Create a managed VIPP environment and CPU shortcut | Create a managed VIPP CUDA environment and Auto/CPU/Prefer-GPU shortcuts |
| Existing napari environment | Leave it unchanged; link to the advanced version-pinned manual route | Leave it unchanged; CUDA integration remains an expert manual route |

Creating a separate managed environment should be the recommended default.
Installing into an existing napari environment is an expert route because its
packages may conflict with VIPP's release constraints. The installer must show
ordinary users a plain-language summary of what will be installed, where it
will appear, the required disk space, and which shortcuts will be created.
The selected interpreter, exact package changes, and rollback boundary belong
under **Advanced details** rather than in the mandatory decision path.

The normal managed flow should have only one consequential user decision:
**Install VIPP**. Read-only checking runs first, the appropriate managed route
is already recommended, and the default location and shortcut are already
selected. The per-track managed location is fixed; changing the compute route
or Desktop-shortcut choice is optional and advanced, not another required
approval. Any change to the computer-use route, existing environment, or
desktop-shortcut choice must invalidate the prior review and require **Check
these settings** again. **Install VIPP** must remain disabled until the exact
current choices have passed checking and package review.

Hardware and software discovery chooses the recommended CPU or qualified GPU
route automatically and describes it as, for example, **GPU acceleration is
available on this computer**. It must not ask the user to interpret package
names or approve each internal command. Existing-napari installation, manual
CPU/GPU selection, package inspection, and custom shortcut locations are
advanced choices.

For a managed CPU installation, the ordinary shortcut is simply **VIPP**, not
**VIPP CPU**. When qualified GPU installation is added, **VIPP** starts in Auto;
CPU-reference and Prefer-GPU launchers can remain in the Start menu or behind
an advanced shortcut option rather than cluttering the desktop.

Platform scope should remain truthful:

- Windows: managed CPU and qualified NVIDIA CUDA routes;
- Linux: managed CPU first, then GPU only after native-Linux policy evidence is
  released;
- macOS: managed CPU while no Apple accelerator provider is admitted.

Python and the NVIDIA display driver are separate prerequisites in the first
installer generation. The bootstrapper must detect them, link to the correct
supported prerequisite when absent, and resume without requiring users to copy
a series of shell commands. The pip-provided CUDA component wheels remain inside
the managed environment; a separate system CUDA Toolkit is not required for
the standard VIPP CUDA route.

## Documentation And Discovery Contract

The release installer must be the first installation path shown in the README,
documentation index, release notes, and versioned user manual. The ordinary
quick start should fit on one screen: download and verify the exact release
`.exe`, handle any documented Windows publisher warning, keep the managed-
environment recommendation, review the
compute route and location, install, then open the created VIPP shortcut.

Documentation must remain truthful while delivery is staged:

- before the installer artifact exists, label it **not yet published** and keep a
  clearly separated manual fallback for the current release;
- after publication, link directly to the exact release asset and publish its
  deterministic name, version, signing status, and SHA-256 beside the link. An
  unsigned alpha must contain `-UNSIGNED` in its filename, state that **Unknown
  publisher** is expected, and put checksum verification before **More info >
  Run anyway**;
- present managed installation first and move existing-napari installation,
  the headless planner, raw pip commands, and environment repair into
  **Advanced** or troubleshooting sections;
- explain CPU versus Automatic/Prefer-GPU behavior without suggesting that a
  GPU installation forces every workflow node onto a GPU.

The maintained [quick start](quick-start.md) is the source-tree version of that
user journey. Installer implementation and documentation changes belong in the
same reviewed change so the public path cannot lag behind executable behavior.

## Installer Delivery Stages

1. **Packaged startup layer:** ship branding, the graphical entry points, real
   startup milestones, plugin loading host, and wheel smoke tests.
2. **Headless Windows installation planner:** discover managed CPU/CUDA and
   selected existing-napari routes, validate Python/GPU/path/disk/shortcut
   preconditions, and emit a stable plan without performing mutations. This
   source-current slice is implemented; see the
   [Windows installation planner](windows-installation-planner.md).
3. **Windows bootstrapper:** the source-current transactional managed executor,
   novice setup window, exact wheel payload, shortcut ownership, update/repair,
   independent CPU/GPU Apps & Features entries, ownership-safe uninstall, and
   signed and explicitly-unsigned asset gates are implemented. Complete the
   clean-machine acceptance, immutable tagging, selected release finalization,
   and release publication before making the `.exe` the live Quick Start.
4. **Linux desktop package:** reuse the same Python launcher and environment
   plan, creating `.desktop` entries and icons without assuming one desktop
   environment.
5. **macOS application/bootstrapper:** reuse the launcher and CPU environment
   plan with an app bundle and normal macOS signing/notarization.

The front ends may differ, but the environment planner and acceptance contract
should remain shared and testable without a GUI.

The current implementation resolves and retains concrete dependency changes
before confirmation. The ordinary UI summarizes them in plain language and
exposes the full list only under **Advanced details**, then applies the exact
hash-locked plan after one explicit confirmation. It provides cancellation,
retained logs, an ownership record, acceptance checks, owned shortcuts,
bounded rollback, update/repair, and an ownership-safe Windows uninstall path.
Large GPU dependency downloads can legitimately take several minutes. Network
operations use bounded retries and a 120-second no-data timeout rather than a
120-second total-install timer. If a transient network failure still stops
Apply, the incomplete candidate is rolled back without replacing a previous
working copy. **Try again** reruns checking and resolution for the exact choices
currently shown, then requires a new review and confirmation instead of reusing
the failed transaction.
Remaining release work is clean-machine acceptance, selected signed or
explicitly-unsigned finalization, and publishing the verified same-tag assets.

## Release Gates

Every installer or launcher release should verify:

- the README, quick start, documentation index, release notes, and versioned
  manual all point first to the exact installer artifact, state its signing
  status, and provide checksum-first instructions, or explicitly say that it
  has not yet been published;
- clean CPU installation and launch on Windows, Linux, and macOS;
- Windows CUDA installation, compute doctor, real eligible GPU execution,
  visible CPU decisions, cleanup, and fallback reporting;
- spaces in managed CPU and CUDA paths, Unicode managed CPU paths, and the
  pre-download ASCII-only guidance for managed and existing CUDA paths,
  introduced in `0.13.0a7` and retained in `0.14.0a1`, including exact
  canonical Local-App-Data roots, rejection of
  custom managed roots, CPU availability when canonical Local App Data makes
  CUDA incompatible, a separate non-mutating existing-environment route, and
  separately reported prior-transaction recovery;
- cancel, retry, repair, uninstall, insufficient disk, offline/network failure,
  and interrupted-download behavior;
- changing any install-relevant choice after review disables **Install VIPP**
  until the exact current choices are checked again;
- a multi-minute GPU download remains visibly active, the 120-second network
  timeout applies only to an idle connection, retries are bounded, and a
  transient terminal failure rolls back before **Try again** repeats the current
  reviewed choices;
- exact wheel/version/entry-point/branding contents from the immutable release
  tag; and
- no environment mutation during plan-only checks or before the user confirms
  the reviewed plan;
- a first-time nontechnical user can install, open, cancel, retry, repair, and
  uninstall VIPP without a terminal or knowledge of Python, napari, CUDA, or
  package management; and
- keyboard and screen-reader navigation, readable high-DPI layouts, and plain-
  language error summaries work, while raw commands and exceptions remain in
  **Advanced details** and the support report.
