# VIPP 0.14.0a2

> **PyPI distribution note:** The files published to PyPI as `0.14.0a2` were uploaded from the earlier pre-resize-fix build and PyPI does not permit replacing them. For the detached-window resizing fix, use the Windows installer or the wheel attached to this GitHub release.

VIPP 0.14.0a2 makes the normal desktop application and an offline installer available on macOS, refreshes the unsigned Windows installer, and fixes detached-window resizing. It is a focused UI compatibility and packaging alpha; the scientific contracts released in 0.14.0a1 are unchanged.

This remains alpha software. Keep original data and workflows, verify the published checksum before running an unsigned installer, and review important outputs before using them for scientific conclusions or publication.

## Easier macOS installation

- Separate Apple Silicon (`arm64`) and Intel (`x86_64`) PKGs install a private, CPU-only environment under `~/Library/vipp` and create `~/Applications/VIPP.app`.
- The installer is offline and includes its own Python, napari, Qt, and VIPP environment. Users do not need a terminal, a pre-existing Python installation, or a DMG wrapper.
- Each architecture has its own release manifest and SHA-256 checksum file. The package is explicitly unsigned and not notarized, so the quick start documents macOS's **Open Anyway** path.

## Cross-Qt compatibility fixes

- Detached VIPP windows can now be maximized or resized freely in both width and height. Reattaching them restores napari's original dock constraints.
- Qt dialog result handling and widget rendering now use APIs shared by PyQt6 and PySide6.
- Settings-menu wrappers are retained while their native actions are visible.
- Delayed dock and thumbnail callbacks now stop safely after their owning Qt object has been destroyed.
- Native Cocoa startup and clean shutdown are exercised by the macOS installer checks rather than relying on Qt's headless offscreen backend.

## Windows and manual installation

The familiar explicitly unsigned Windows setup executable is rebuilt from the exact 0.14.0a2 wheel and remains the recommended Windows route. The wheel attached to this GitHub release is the exact-version manual-installation route that includes the detached-window resizing fix.

## Qualification scope

The changed release domains are core/UI compatibility, detached dock sizing, macOS installer and packaging infrastructure, dependency/toolchain inputs for that installer, and documentation. Focused PyQt6/PySide6 dock checks and native Apple Silicon lifecycle tests cover the compatibility delta; the release workflow also builds and tests the Intel package. Unchanged SourceItem, workflow/schema, scientific GPU, and Windows transactional-installer behavior carry forward from their recorded 0.14.0a1 and 0.13.0a8 baselines. Exact 0.14.0a2 wheel, source archive, installer, checksum, version, and public-URL facts are regenerated from the release tag.
