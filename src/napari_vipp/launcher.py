"""Cross-platform branded launcher for the full VIPP application process.

Importing this module is intentionally cheap.  Qt is imported only when a
splash is actually requested; napari and scientific libraries live solely in
the child process started as ``sys.executable -m napari_vipp.app``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from napari_vipp.startup import (
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    LaunchProfile,
    StartupChannel,
    StartupPhase,
    StartupStateMachine,
    StatusReader,
    create_startup_log_path,
)


def installed_version() -> str:
    """Return the distribution version without importing the full package."""
    try:
        return version("napari-vipp")
    except PackageNotFoundError:
        return "development"


def build_child_command(
    *,
    executable: str | os.PathLike[str],
    profile: LaunchProfile | str,
    channel: StartupChannel,
) -> list[str]:
    """Build a shell-free command that remains correct for paths with spaces."""
    parsed_profile = LaunchProfile.parse(profile)
    return [
        os.fspath(executable),
        "-m",
        "napari_vipp.app",
        "--profile",
        parsed_profile.value,
        "--startup-channel",
        os.fspath(channel.path),
        "--startup-token",
        channel.token,
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build the public VIPP launcher argument parser."""
    parser = argparse.ArgumentParser(
        prog="vipp",
        description="Start VIPP with immediate branded progress feedback.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in LaunchProfile),
        default=LaunchProfile.AUTO.value,
        help="Initial compute profile for this session (default: auto).",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Run VIPP directly in this process, retaining terminal output.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Seconds before offering to keep waiting or hide the splash.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"VIPP {installed_version()}",
    )
    return parser


class LauncherController:
    """Coordinate the child, authenticated status channel, timer, and Qt view."""

    def __init__(
        self,
        *,
        application: Any,
        splash: Any,
        timer: Any,
        profile: LaunchProfile,
        channel: StartupChannel,
        log_path: Path,
        timeout_seconds: float,
        executable: str | os.PathLike[str] | None = None,
        process_factory: Any = subprocess.Popen,
        schedule: Any = None,
    ) -> None:
        self.application = application
        self.splash = splash
        self.timer = timer
        self.profile = profile
        self.channel = channel
        self.log_path = Path(log_path)
        self.executable = sys.executable if executable is None else executable
        self.process_factory = process_factory
        self.schedule = schedule
        self.reader = StatusReader(channel.path, channel.token)
        self.machine = StartupStateMachine(timeout_seconds=timeout_seconds)
        self.process: subprocess.Popen[bytes] | None = None
        self._finishing = False
        self._cleaned = False

        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.splash.keep_waiting_requested.connect(self.keep_waiting)
        self.splash.hide_requested.connect(self.hide)
        self.splash.close_requested.connect(self.close)
        self.splash.open_log_requested.connect(self.open_log)
        self.application.aboutToQuit.connect(self._cleanup)

    def start(self) -> None:
        """Spawn the application child with no shell or inherited input."""
        command = build_child_command(
            executable=self.executable,
            profile=self.profile,
            channel=self.channel,
        )
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_stream:
                log_stream.write(
                    "VIPP startup diagnostic log\n"
                    f"Version: {installed_version()}\n"
                    f"Profile: {self.profile.value}\n"
                    f"Python: {os.fspath(self.executable)}\n\n"
                )
                log_stream.flush()
                self.process = self.process_factory(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    shell=False,
                )
        except BaseException as exc:
            self.machine.protocol_failed(exc)
            self._render()
            return
        self.timer.start()
        self._render()

    def poll(self) -> None:
        """Apply new child records and update elapsed/terminal state."""
        if self._finishing:
            return
        try:
            for event in self.reader.read_new():
                self.machine.accept(event)
        except BaseException as exc:
            self.machine.protocol_failed(exc)

        process = self.process
        returncode = None if process is None else process.poll()
        if returncode is not None:
            self.machine.process_exited(returncode)
        self.machine.check_timeout()
        self._render()

    def _render(self) -> None:
        snapshot = self.machine.snapshot
        self.splash.update_elapsed(self.machine.elapsed())
        self.splash.update_snapshot(snapshot)
        if snapshot.phase is StartupPhase.READY:
            self._finish_after_ready()
        elif snapshot.phase is StartupPhase.FAILED:
            self.timer.stop()
            if not self.splash.isVisible():
                self.splash.show()

    def _finish_after_ready(self) -> None:
        if self._finishing:
            return
        self._finishing = True
        self.timer.stop()
        # Leave the completed milestone visible briefly enough to register.
        if self.schedule is None:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(350, self._quit_launcher)
        else:
            self.schedule(350, self._quit_launcher)

    def keep_waiting(self) -> None:
        """Reset the non-destructive timeout window."""
        self.machine.keep_waiting()
        self._render()

    def hide(self) -> None:
        """Hide feedback while monitoring continues; never terminate VIPP."""
        self.machine.hide()
        self.splash.hide()

    def close(self) -> None:
        """Close a terminal error view without touching the child process."""
        self._quit_launcher()

    def open_log(self) -> None:
        """Open the per-launch diagnostic log with the platform file handler."""
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(os.fspath(self.log_path)))

    def _quit_launcher(self) -> None:
        self._cleanup()
        self.splash.permit_close()
        self.splash.close()
        self.application.quit()

    def _cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.channel.cleanup()


def run_splash(
    profile: LaunchProfile,
    *,
    timeout_seconds: float,
) -> int:
    """Create the lightweight Qt splash and monitor the full child app."""
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from napari_vipp.ui.startup_splash import StartupSplash

    existing_application = QApplication.instance()
    application = existing_application or QApplication(sys.argv[:1])
    if existing_application is None:
        application.setApplicationName("VIPP")
        application.setOrganizationName("VIPP")

    channel = StartupChannel.create()
    try:
        log_path = create_startup_log_path(profile)
        splash = StartupSplash(
            profile=profile,
            version=installed_version(),
            log_path=log_path,
        )
        timer = QTimer(application)
        controller = LauncherController(
            application=application,
            splash=splash,
            timer=timer,
            profile=profile,
            channel=channel,
            log_path=log_path,
            timeout_seconds=timeout_seconds,
        )
        # Keep a strong reference for the full event-loop lifetime.
        application._vipp_launcher_controller = controller
        splash.show()
        QTimer.singleShot(0, controller.start)
    except BaseException:
        channel.cleanup()
        raise
    if existing_application is not None:
        return 0
    return int(application.exec_())


def main(argv: list[str] | None = None) -> int:
    """Launch VIPP with the selected session profile."""
    args = build_parser().parse_args(argv)
    profile = LaunchProfile.parse(args.profile)
    if args.timeout <= 0:
        build_parser().error("--timeout must be positive")
    if args.no_splash:
        from napari_vipp.app import main as app_main

        return app_main(["--profile", profile.value])
    try:
        return run_splash(profile, timeout_seconds=args.timeout)
    except ImportError as exc:
        if sys.stderr is not None:
            print(
                "VIPP needs a working Qt installation to show its startup window: "
                f"{exc}",
                file=sys.stderr,
            )
        return 2
    except Exception as exc:
        if sys.stderr is not None:
            print(
                "VIPP could not create its startup window: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return 2


def main_auto() -> int:
    """GUI-entry-point facade for an Automatic shortcut."""
    return main([*sys.argv[1:], "--profile", LaunchProfile.AUTO.value])


def main_cpu() -> int:
    """GUI-entry-point facade for a CPU-safe-mode shortcut."""
    return main([*sys.argv[1:], "--profile", LaunchProfile.CPU.value])


def main_prefer_gpu() -> int:
    """GUI-entry-point facade for a Prefer-GPU shortcut."""
    return main([*sys.argv[1:], "--profile", LaunchProfile.PREFER_GPU.value])


__all__ = [
    "LauncherController",
    "build_child_command",
    "build_parser",
    "installed_version",
    "main",
    "main_auto",
    "main_cpu",
    "main_prefer_gpu",
    "run_splash",
]
