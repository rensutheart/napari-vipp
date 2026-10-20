# Floating VIPP recovery on Windows

Unreleased qualification on 2026-10-09. This covers window presentation and
plugin instance reuse; it does not change scientific execution.

## Regression and implementation

Clearing a floating dock's native owner after it was shown did not explicitly
register a taskbar window. An already unowned dock also skipped repair.
The native helper now sets `WS_EX_APPWINDOW`, clears `WS_EX_TOOLWINDOW`, and
refreshes a visible window through hide/style/show without activation. Native
and Qt visibility, minimized/maximized state, normal geometry, parentage, and
dock membership survive the repair.

Standalone and development launchers formerly registered `VIPP Workflow`, while
napari's plugin command searched for `VIPP Workflow (napari-vipp)`. The launchers
now use that canonical key. Napari's menu action also toggles visibility of
minimized windows; a plugin-local menu observer restores the existing minimized
dock after that action completes. The menu-opening token is weakly held and
invalidates earlier captures, including when two viewers share one QAction.
Active Qt host ownership is checked before scheduling and applying recovery,
including when a second viewer has no VIPP instance or menu observer yet.
Closing, redocking, deletion, canceled menus, and lifecycle shutdown cancel
recovery. Normal visible-window toggling remains unchanged.

## Native checks

Run `scripts/qualify_floating_dock.py` with the Windows Qt platform to create
only task-owned probe windows. The [native record](floating-dock-windows-2026-10-09.json)
contains source hashes and runtime versions.

The seven cases cover hidden, normal, minimized, maximized, minimized from
maximized, and two already unowned tool windows. Each verifies owner/style repair
and unchanged Qt/native display state, flags, normal placement, and the native
restore-to-maximized flag. Each also redocks as a native child without the
independent taskbar style, refloats, and completes three minimize/maximize cycles.
Visible cases remain independently visible when their napari-like host minimizes.

These probes verify Windows registration styles and native state. They do not
enumerate Explorer taskbar buttons or qualify other operating systems.

## Automated and live application checks

Focused regressions cover pointer-safe Win32 bindings, failed repairs, plugin
command reuse, actual app-model menu actions, two-viewer capture invalidation,
startup wrappers, lifecycle shutdown, docking, and preserved cache identities.
The affected baseline passed 168 tests with one offscreen canvas skip. All 21
additional plugin recovery tests passed on the final source, including real
QModelMenu popup mouse selection and both shared-action races. Launcher,
branding, architecture, and documentation checks also pass. The full
scientific suite was not rerun for this window-only change.

A fresh native napari session loaded the saved 27-node, 29-connection wire test
workflow and completed calculation (6.1 MB cache). After minimizing its maximized
floating VIPP window, selecting **Plugins → VIPP Workflow (napari-vipp)** through
the actual menu restored it maximized. The native window, widget, pipeline, and
all 27 output object identities remained the same; the viewer still contained
one VIPP editor and one canonical dock registration. Native owner zero,
`WS_EX_APPWINDOW`, and absence of `WS_EX_TOOLWINDOW` were retained, with no repair
warning. Running source hashes matched the final working tree and packaged wheel
source. The test workflow was left open with its calculated previews.

Ruff, the npe2 manifest check, wheel/source-distribution builds, and independent
review passed. The companion manual passed its 18 content contracts, 50 route
checks, and strict MkDocs build.

The companion [vipp-mkdocs](https://github.com/rensutheart/vipp-mkdocs) manual
change is local commit `63c574065aaf2a0f57efd6cf8e500125310a4e37` on the
documentation branch; it has not been pushed or published.
