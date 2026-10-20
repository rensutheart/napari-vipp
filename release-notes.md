# VIPP 0.16.0a5

VIPP 0.16.0a5 focuses on keeping workflows safe, moving projects between folders, clearer graph editing and guided Windows GPU setup. The unfinished AI assistant, agent/MCP, webhook-bot and saved assistant-session work is not included.

**Alpha software.** Preserve original images, saved workflows, environments and decisive outputs. Check representative data before scientific interpretation.

## Keep and move your work

- Workflow autosave keeps bounded, separate local graph working copies. After an interrupted session, recover named or unnamed workflows as new unsaved tabs without replacing the original files or calculating automatically.
- Recovery preserves the workflow recipe and executable file bindings, including portable paths, but not source pixels, calculated images/tables, Undo history, in-flight work or the Batch workspace/configuration. Save Batch configuration separately. Autosave is not a source-data backup.
- Workflow, batch and generated-Python paths resolve from their saved document's folder where possible. Copy documents and data together, preserving the folder layout. Save As rebases paths without moving data; another drive/share stays absolute. Source revisions and dataset checksums are retained. Recheck batch destinations and Skip/Overwrite choices after moving a project.
- No-overwrite image saving stages the complete result privately and publishes only if its destination is still absent. Existing or racing output files/stores are preserved. Filesystems without a safe exclusive-publication primitive fail closed; there is no unsafe overwrite fallback. This protects an individual image write, not a transaction over every batch output and provenance file.

## Clearer Windows setup

- Compute setup and memory offers **Set up GPU…** or **Repair GPU setup…** for the supported Windows NVIDIA route. Review a separate managed CUDA installation of the exact current VIPP build, follow progress, and open the verified build in another window. Current workflows remain open and are not transferred automatically. Verification alone does not install packages.
- Matching status cards, a clear next action, collapsed advanced/technical details and a readable log make both verification and guided setup easier to follow. The manual command is labelled with its purpose, and an already-ready build focuses on **Open VIPP with GPU**, not irrelevant download estimates.
- Setup checks Apps & Features ownership before downloads, environment replacement and upgrade commit. Unsafe registration or interrupted-recovery conflicts offer **Get setup help** instead of an ineffective retry loop. Automatic recovery still requires exact recorded ownership evidence; matching a folder or installation ID is not permission to overwrite a different installation.

## Graph and preview improvements

- Keep registration inspector sections from overlapping when help text grows in a narrow panel.
- Wires from distinct output ports use separate lanes. Bounded background routing prefers clear local paths and avoids unnecessary staircases while retaining card clearance; complex routing is cancellable rather than blocking editing.
- Convert wires to named tunnels, a tunnel input back to a wire, or all subscribers back to wires as one undoable edit, preserving connections and calculated results.
- Reopen a minimized floating VIPP window from the original napari Plugins menu. Native Linux/macOS window-manager parity is not inferred from Windows checks.
- Results/workflow readiness checks no longer discard queued thumbnail work or leave a stale thumbnail busy indicator. Scientific calculations, measurements and statistics are unchanged.

## Compatibility and qualification boundaries

Workflow, batch-config and manifest schema numbers remain 6. The node catalogue, scientific algorithms, GPU kernels/admission regions, runtime/build dependency pins, supported Python routes and macOS installer engine are unchanged. GPU setup does not admit additional hardware or scientific operation regions.

The [release plan](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a5/docs/release-0.16.0a5-plan.md) and [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a5/docs/release-qualification-baseline.md) separate changed-domain gates from evidence that may carry forward. Consult the release record for exact commits, platform runs and artifact hashes; a local unit-test pass is not a full frozen Windows GUI lifecycle, broad filesystem qualification, large-volume performance or biological validation.

Desktop installers are explicitly unsigned alpha builds. macOS packages are separate CPU-only arm64/x86_64 artifacts and are also unnotarized. Verify downloads using the matching release checksum files. The [versioned manual](https://rensutheart.github.io/vipp-mkdocs/0.16.0a5/) documents recovery, portable paths and setup limitations.
