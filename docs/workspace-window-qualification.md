# Workspace window contract and Windows qualification

Unreleased after 0.16.0a3. `CollectionBatchDialog` and
`ResultsWorkspaceDialog` use `ui/workspace_window.py` for native window controls.

Their QObject parents remain unchanged. On native Windows, only the HWND owner
is removed. A recreated owner is accepted only when it belongs to a verified
top-level QWidget ancestor, including the napari host and its floating VIPP
dock. Qt can reuse the original host owner after that dock detaches. Unrelated
owners, invalid handles and native child windows remain rejected.

Ownership repair is deferred and coalesced after show, handle and state changes.
It does not change visibility, geometry or window state. Explicit reopening
clears only the minimized state, preserving a prior maximized state. Windows
handles native title-bar double-click; a second application toggle would undo
it. Only registered passive toolbar surfaces handle client-area double-click,
leaving buttons, selectors and editable/selectable content alone.

## Native acceptance, 2026-10-08

Passed on Windows 11 build 26200, Python 3.12.9, PyQt6 / Qt 6.10.2 and the native
`windows` platform plugin. The production VIPP widget and both workspace dialogs
ran in a QMainWindow/QDockWidget host on a private desktop without global input
or changes to the user's running application. The viewer was a test adapter;
this check did not qualify napari's OpenGL renderer.

The checks verified:

- Both workspaces remained natively visible with owner HWND 0 when the host was
  minimized, using Qt and native system commands, before and after VIPP detached.
- Hiding and reopening each workspace repaired Qt's recreated host ownership.
- Each workspace restored its normal or maximized state after minimizing,
  retaining its normal restore geometry.
- Native caption and passive-toolbar double-click each maximized or restored
  exactly once and remained settled after event processing.
- Pipeline, nodes, outputs, Batch configuration and Results table/selection
  retained their identities; no calculation or run callbacks occurred.
- Parent destruction removed all windows with ownership repairs still queued.

`test_workspace_windows.py` covers the portable controls, state/data retention,
queued cleanup, and accepted/rejected native-owner cases. Existing floating-dock
and Results/Batch lifecycle regressions remain in place.

Native Linux/macOS ownership and minimization are unqualified. Three physical
monitors and cross-monitor DPI transitions were not exercised in this check.
