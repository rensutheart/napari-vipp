# VIPP 0.16.0a6

VIPP 0.16.0a6 is a performance and Windows setup reliability release. It makes startup visible sooner, moves reader-library preparation behind the open interface, and handles brief Windows file locks more reliably.

## Startup you can see and use

- The splash paints before the main application process starts. Its clock and minimize/hide controls remain responsive while Windows starts that process.
- On Windows, sharp prepared branding and ordinary vector text appear without waiting for system-font discovery. The version, current stage and elapsed time remain visible; the loading message uses Inter Medium throughout startup.
- OME-XML and OME-Zarr libraries begin warming in the background after the standalone interface or napari plugin panel is ready. Later file selection reuses those imports. A file selected immediately can still wait for its library, and an early TIFF load runs in a worker while XML support is warming.
- Startup logs retain elapsed, process CPU and per-stage timings to help diagnose slow launches.

## More reliable Windows setup

- Setup retries brief Windows sharing locks when publishing or restoring shortcuts, installation records and retained artifacts. Shortcuts are created privately before atomic publication.
- A persistent shortcut lock produces a clear explanation and retains the existing ownership, transaction and rollback protections. It does not silently overwrite foreign installations or bypass validation.

## Performance and compatibility limits

Startup still loads napari and scientific dependencies. Windows security scanning and cache state can dominate total launch time; this release does not change security policies or promise a fixed startup duration. The [startup investigation](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a6/docs/benchmarks/windows-startup-2026-10-21.md) distinguishes installed, instrumented, recovery-paused and controlled preview measurements; no verified reboot-cold or same-version cross-computer comparison is claimed.

Scientific algorithms, image interpretation, workflow/batch/manifest schema numbers, compute policy, GPU admission and runtime/build dependency pins are unchanged. Reader warming only prepares imports; it does not read your files or calculate a workflow. Closing immediately during an active native import may wait for that import to finish safely.

The [release plan](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a6/docs/release-0.16.0a6-plan.md) and [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a6/docs/release-qualification-baseline.md) separate changed-domain checks from carried-forward evidence. Desktop installers are explicitly unsigned alpha builds; macOS packages are separate CPU-only arm64/x86_64 artifacts and are also unnotarized. Verify their matching checksums. See the [versioned manual](https://rensutheart.github.io/vipp-mkdocs/0.16.0a6/).

[Full source comparison: 0.16.0a5 → 0.16.0a6](https://github.com/rensutheart/napari-vipp/compare/v0.16.0a5...v0.16.0a6)
