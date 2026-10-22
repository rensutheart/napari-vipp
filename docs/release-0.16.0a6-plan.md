# 0.16.0a6 performance release

This iterative alpha covers only the startup and Windows installer changes
after public `v0.16.0a5` (`57248e0d8a7cc7c81fb0533fa8b705d4ddb370f7`).
The user approved the final splash appearance and authorized publication on
2026-10-22. Follow the [release runbook](release-runbook.md) and the
[qualification declaration](release-qualification-baseline.md#0160a6-qualification-declaration).

## Included behavior

- Paint the splash before starting the application child; keep process creation
  and diagnostic-log work off the GUI thread. On Windows, prepared branding and
  ordinary vector lettering avoid initial system-font discovery. The clock,
  stage, version and minimize/hide controls remain available. The status line
  uses Inter Medium throughout startup; bundled vector assets include licences.
- Schedule serial OME-XML/OME-Zarr dependency imports after interface readiness.
  An early TIFF selection uses background loading while the XML import is
  pending. File operations retain their authoritative imports and errors.
- Record monotonic elapsed, CPU and per-stage startup timings. Keep development
  profiling and asset generation separate from application startup.
- Bound transient Windows sharing-lock retries when publishing shortcuts,
  records and retained artifacts. Create shortcuts privately and stage neutral
  temporary files before atomic publication; preserve ownership, hash,
  rollback and active-environment checks. Explain persistent shortcut locks.

## Evidence and limits

The [Windows investigation](benchmarks/windows-startup-2026-10-21.md) records
the slow-launch Defender trace, installed development-build repair and shortcut
launches, native first-paint/clock checks, four display-scale checks, exact
runtime wheel contents and normal application launch/shutdown. Installed timing
observations precede the final typography refinement. Recovery-paused launches
are distinguished from launches that calculate the normal welcome workflow;
there is no verified reboot-cold or same-version desktop comparison. Do not
advertise a fixed total startup time or elimination of security scanning.

Focused startup, warm-up, reader and installer regression checks cover the
changed boundaries. The existing supported-platform CI matrix must pass on the
candidate and exact final main. Its quality job independently builds and
compares the wheel and canonical source archive. Retain that exact-main pair
and verify both vector-lettering assets and font licences before packaging.
Build the exact-tag unsigned Windows installer and separate CPU-only unsigned,
unnotarized macOS arm64/x86_64 packages using the existing release workflow.
These checks do not establish a new full installer lifecycle or GPU baseline.

Scientific algorithms, source interpretation, schema versions, compute policy,
runtime/build pins and macOS installer implementation are unchanged. Carry
their behavioral qualification from a5. Closing immediately during an active
native import can wait for that import to finish safely. A file selected before
reader warming finishes can still wait for its library. No security policy,
antivirus exclusions or hardware settings are changed by this release.

Merge one application PR and one companion manual PR, require final-main CI,
tag immutably, retain verified Python/installer artifacts, publish identical
GitHub/PyPI Python bytes, and only then deploy numbered/default documentation.
The scheduled release follow-up records transient run IDs and hashes outside
the public task guides. No publication is claimed by this plan alone.
