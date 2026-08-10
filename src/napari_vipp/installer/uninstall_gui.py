"""Small novice-facing GUI route for the frozen VIPP uninstaller."""

from __future__ import annotations

import argparse
import ctypes
import os
import queue
import sys
import threading
import uuid
from collections.abc import Callable, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Protocol, cast

from napari_vipp.installer.uninstall import (
    DeferredSelfDelete,
    ManagedUninstaller,
    UninstallError,
    UninstallResult,
    UninstallStatus,
    WindowsRegistryBackend,
    read_windows_shortcut_target,
    schedule_deferred_self_delete,
)

_FOLDERID_DESKTOP = "B4BFCC3A-DB2C-424C-B029-7FE99A87C641"
_FOLDERID_PROGRAMS = "A77F5D77-2E2B-44C3-A6A2-ABA601054A51"


class UninstallExitCode(IntEnum):
    """Deterministic process result for the frozen uninstall route."""

    COMPLETED = 0
    CANCELLED = 10
    REFUSED = 20
    INCOMPLETE = 21
    UI_ERROR = 22
    SELF_DELETE_FAILED = 23


class MessageBoxes(Protocol):
    """Tk message-box surface used by the novice flow."""

    def askyesno(self, title: str, message: str, **options: object) -> bool: ...

    def showinfo(self, title: str, message: str, **options: object) -> object: ...

    def showerror(self, title: str, message: str, **options: object) -> object: ...


class RootWindow(Protocol):
    """Tiny subset of ``tkinter.Tk`` needed by the uninstall flow."""

    def withdraw(self) -> None: ...

    def destroy(self) -> None: ...


ProgressRunner = Callable[
    [RootWindow, Callable[[], UninstallResult]],
    UninstallResult,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VIPP Setup",
        description="Remove one explicitly selected managed VIPP installation.",
    )
    parser.add_argument("--uninstall", action="store_true", required=True)
    parser.add_argument(
        "--managed-root",
        required=True,
        type=_absolute_path,
        help=argparse.SUPPRESS,
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    uninstaller: ManagedUninstaller | None = None,
    root_factory: Callable[[], RootWindow] | None = None,
    messageboxes: MessageBoxes | None = None,
    shortcut_roots_provider: Callable[[], Sequence[Path]] | None = None,
    scheduler: Callable[[DeferredSelfDelete], None] = (schedule_deferred_self_delete),
    current_executable: str | Path | None = None,
    current_pid: int | None = None,
    progress_runner: ProgressRunner | None = None,
) -> int:
    """Show one plain confirmation and run the exact hash-bound uninstall."""

    args = build_parser().parse_args(argv)
    if (
        os.name != "nt"
        and uninstaller is None
        and root_factory is None
        and messageboxes is None
    ):
        print("The VIPP uninstaller currently supports Windows.", file=sys.stderr)
        return int(UninstallExitCode.UI_ERROR)

    if root_factory is None or messageboxes is None:
        try:
            import tkinter as tk
            from tkinter import messagebox
        except ImportError as exc:
            print(f"VIPP Setup needs the Windows Tk runtime: {exc}", file=sys.stderr)
            return int(UninstallExitCode.UI_ERROR)
        if root_factory is None:
            root_factory = tk.Tk
        if messageboxes is None:
            messageboxes = messagebox

    try:
        root = root_factory()
    except Exception as exc:
        print(f"VIPP Setup could not open its removal window: {exc}", file=sys.stderr)
        return int(UninstallExitCode.UI_ERROR)
    root.withdraw()
    _close_frozen_splash()

    selected_uninstaller = uninstaller or ManagedUninstaller(
        registry=WindowsRegistryBackend(),
        shortcut_target_reader=read_windows_shortcut_target,
        current_executable=(
            Path(current_executable)
            if current_executable is not None
            else Path(sys.executable)
        ),
    )
    roots_provider = shortcut_roots_provider or windows_shortcut_roots
    assert messageboxes is not None
    try:
        try:
            prepared = selected_uninstaller.prepare(
                args.managed_root,
                shortcut_roots=tuple(roots_provider()),
            )
        except (OSError, UninstallError, ValueError) as exc:
            messageboxes.showerror(
                "VIPP could not be removed safely",
                (
                    "VIPP did not remove anything because it could not verify "
                    "this installation.\n\n"
                    f"Details: {exc}"
                ),
                parent=root,
            )
            return int(UninstallExitCode.REFUSED)

        confirmed = messageboxes.askyesno(
            "Remove VIPP?",
            (
                f"Remove VIPP {prepared.record.version} from this Windows "
                "account?\n\n"
                "This removes the VIPP program and the shortcuts created by "
                "VIPP Setup. Your images, analysis files, Python installations, "
                "and other napari environments will not be removed."
            ),
            parent=root,
            icon="warning",
        )
        if not confirmed:
            return int(UninstallExitCode.CANCELLED)

        def remove_vipp() -> UninstallResult:
            return selected_uninstaller.apply(
                prepared,
                selected_uninstaller.authorize(prepared),
                current_executable=(
                    Path(current_executable)
                    if current_executable is not None
                    else Path(sys.executable)
                ),
                current_pid=current_pid or os.getpid(),
            )

        runner = progress_runner
        if runner is None:
            # Test doubles and non-Tk embedders retain the old direct route.
            # A real ``tkinter.Tk`` always exposes ``tk`` and uses the visible,
            # responsive progress window below.
            runner = (
                _run_removal_with_progress
                if hasattr(root, "tk")
                else _run_removal_directly
            )
        try:
            result = runner(root, remove_vipp)
        except (OSError, UninstallError, ValueError) as exc:
            messageboxes.showerror(
                "VIPP removal stopped",
                (
                    "VIPP stopped before it could safely finish. Choose Remove "
                    "again from Windows Settings.\n\n"
                    f"Details: {exc}"
                ),
                parent=root,
            )
            return int(UninstallExitCode.INCOMPLETE)

        if result.status is not UninstallStatus.COMPLETED:
            messageboxes.showerror(
                "VIPP cleanup is incomplete",
                format_incomplete_result(result),
                parent=root,
            )
            return int(UninstallExitCode.INCOMPLETE)

        # This modal result is shown and acknowledged before the running cached
        # setup executable is ever scheduled for deletion.
        messageboxes.showinfo(
            "VIPP was removed",
            (
                "VIPP was removed from this Windows account.\n\n"
                "Your images and analysis files were not changed."
            ),
            parent=root,
        )
        if result.deferred_self_delete is not None:
            try:
                scheduler(result.deferred_self_delete)
            except Exception as exc:
                cleanup_paths = [result.deferred_self_delete.target]
                if result.deferred_self_delete.journal_path is not None:
                    cleanup_paths.append(result.deferred_self_delete.journal_path)
                rendered_paths = "\n".join(str(path) for path in cleanup_paths)
                messageboxes.showerror(
                    "Cached setup files remain",
                    (
                        "VIPP was removed, but Windows could not schedule cleanup "
                        "of the exact cached files below. You may delete only these "
                        "paths manually after this window closes.\n\n"
                        f"{rendered_paths}\n\n{exc}"
                    ),
                    parent=root,
                )
                return int(UninstallExitCode.SELF_DELETE_FAILED)
        return int(UninstallExitCode.COMPLETED)
    finally:
        root.destroy()


def _run_removal_directly(
    _root: RootWindow,
    operation: Callable[[], UninstallResult],
) -> UninstallResult:
    """Compatibility route for non-Tk hosts and lightweight test doubles."""

    return operation()


def _run_removal_with_progress(
    root: RootWindow,
    operation: Callable[[], UninstallResult],
) -> UninstallResult:
    """Keep Tk responsive while the verified removal runs in one worker."""

    import tkinter as tk
    from tkinter import ttk

    window = tk.Toplevel(root)  # type: ignore[arg-type]
    window.title("Removing VIPP…")
    window.resizable(False, False)
    window.protocol("WM_DELETE_WINDOW", lambda: None)

    frame = ttk.Frame(window, padding=(28, 24))
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text="Removing VIPP…",
        font=("Segoe UI", 12, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            "VIPP is removing only the program files and shortcuts created "
            "by VIPP Setup. This can take a few minutes."
        ),
        wraplength=390,
        justify="left",
    ).pack(anchor="w", pady=(8, 18))
    progress = ttk.Progressbar(frame, mode="indeterminate", length=390)
    progress.pack(fill="x")
    progress.start(12)

    outcomes: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def work() -> None:
        try:
            outcomes.put((True, operation()))
        except BaseException as exc:
            # Passing the exception back prevents a dead worker from leaving a
            # permanent progress window. It is re-raised on the GUI thread.
            outcomes.put((False, exc))

    worker = threading.Thread(
        target=work,
        daemon=True,
        name="vipp-uninstall-worker",
    )
    finished: list[tuple[bool, object]] = []

    def finish_when_ready() -> None:
        try:
            finished.append(outcomes.get_nowait())
        except queue.Empty:
            window.after(50, finish_when_ready)
            return
        progress.stop()
        window.grab_release()
        window.destroy()

    window.grab_set()
    window.deiconify()
    window.lift()
    worker.start()
    window.after(50, finish_when_ready)
    window.wait_window()
    worker.join()

    if not finished:
        raise RuntimeError("The VIPP removal window closed unexpectedly.")
    succeeded, value = finished[0]
    if not succeeded:
        raise cast(BaseException, value)
    return cast(UninstallResult, value)


def format_incomplete_result(result: UninstallResult) -> str:
    """Render exact preserved paths and errors without hiding technical detail."""

    lines = [
        "VIPP cleanup is incomplete. No unverified item was deleted.",
        "",
        "Items still present:",
    ]
    if result.issues:
        for issue in result.issues:
            lines.extend(
                (
                    f"- {issue.path}",
                    f"  {issue.operation}: {issue.error}",
                )
            )
    else:
        for path in result.preserved_paths:
            lines.append(f"- {path}")
    lines.append("")
    if result.retry_via_apps:
        lines.append(
            "Close VIPP and try Remove again. If the same item remains, keep "
            "this list for support."
        )
    else:
        lines.append(
            "VIPP itself is removed and the Windows Remove entry is gone. After "
            "closing Setup, only the exact cached paths above may be deleted "
            "manually or kept for support."
        )
    return "\n".join(lines)


def windows_shortcut_roots() -> tuple[Path, Path]:
    """Resolve this account's Desktop and Start Menu Programs known folders."""

    if os.name != "nt":
        raise OSError("Windows known folders are unavailable on this system.")
    return (
        _known_folder_path(_FOLDERID_DESKTOP),
        _known_folder_path(_FOLDERID_PROGRAMS),
    )


def _known_folder_path(identifier: str) -> Path:
    class GUID(ctypes.Structure):
        _fields_ = (
            ("data1", ctypes.c_uint32),
            ("data2", ctypes.c_uint16),
            ("data3", ctypes.c_uint16),
            ("data4", ctypes.c_ubyte * 8),
        )

    guid = GUID.from_buffer_copy(uuid.UUID(identifier).bytes_le)
    shell32 = ctypes.windll.shell32
    ole32 = ctypes.windll.ole32
    shell32.SHGetKnownFolderPath.argtypes = (
        ctypes.POINTER(GUID),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = (ctypes.c_void_p,)
    value = ctypes.c_wchar_p()
    result = shell32.SHGetKnownFolderPath(
        ctypes.byref(guid),
        0,
        None,
        ctypes.byref(value),
    )
    if result != 0 or not value.value:
        raise OSError(f"Windows could not resolve known folder {identifier}.")
    try:
        return Path(value.value)
    finally:
        ole32.CoTaskMemFree(ctypes.cast(value, ctypes.c_void_p))


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(
            "--managed-root must be an explicit absolute path"
        )
    return path


def _close_frozen_splash() -> None:
    try:
        import pyi_splash

        pyi_splash.close()
    except (ImportError, RuntimeError):
        pass


__all__ = [
    "UninstallExitCode",
    "build_parser",
    "format_incomplete_result",
    "main",
    "windows_shortcut_roots",
]
