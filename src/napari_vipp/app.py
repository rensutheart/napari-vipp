"""The full VIPP application process started behind the lightweight splash."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Protocol

from napari_vipp.startup import STAGE_MESSAGES, LaunchProfile, StatusEmitter


class StartupReporter(Protocol):
    """Minimal reporter interface used while importing the application stack."""

    def progress(self, stage: str, message: str | None = None) -> None: ...

    def ready(self, message: str = "VIPP is ready") -> None: ...

    def failure(self, message: str, *, error: str = "") -> None: ...

    def close(self) -> None: ...


class _ConsoleReporter:
    """Small fallback for direct, terminal-based application launches."""

    def progress(self, stage: str, message: str | None = None) -> None:
        rendered = message or STAGE_MESSAGES.get(stage, stage.replace("_", " "))
        if sys.stderr is not None:
            print(f"VIPP: {rendered}", file=sys.stderr, flush=True)

    def ready(self, message: str = "VIPP is ready") -> None:
        if sys.stderr is not None:
            print(f"VIPP: {message}", file=sys.stderr, flush=True)

    def failure(self, message: str, *, error: str = "") -> None:
        if sys.stderr is not None:
            print(f"VIPP: {message}", file=sys.stderr, flush=True)
            if error:
                print(error, file=sys.stderr, flush=True)

    def close(self) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    """Build the child application's deliberately small argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m napari_vipp.app",
        description="Run the full VIPP napari application.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in LaunchProfile),
        default=LaunchProfile.AUTO.value,
        help="Initial compute profile for this VIPP session.",
    )
    parser.add_argument(
        "--startup-channel",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--startup-token", help=argparse.SUPPRESS)
    parser.add_argument(
        "--smoke-exit-after-ready",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _make_reporter(args: argparse.Namespace) -> StartupReporter:
    channel = args.startup_channel
    token = args.startup_token
    if (channel is None) != (token is None):
        raise ValueError(
            "--startup-channel and --startup-token must always be supplied together."
        )
    if channel is None:
        return _ConsoleReporter()
    return StatusEmitter(channel, token)


def _construct_vipp_widget(widget_type, viewer, profile: LaunchProfile):
    """Construct the facade through its staged, single-initial-run API."""
    return widget_type(
        viewer,
        defer_initial_run=True,
        initial_compute_mode=profile.value,
    )


def _configure_initial_workflow(widget) -> None:
    """Configure and execute the bundled synthetic workflow exactly once."""
    widget.pipeline.nodes["input"].params.update(
        {
            "source_mode": "sample",
            "sample_name": "VIPP synthetic volume",
        }
    )
    widget.run_initial_pipeline_once()
    widget.graph_view.select_node("input")


def run_application(
    profile: LaunchProfile,
    reporter: StartupReporter,
    *,
    smoke_exit_after_ready: bool = False,
) -> int:
    """Load napari, build VIPP, report readiness, and enter the Qt event loop."""
    reporter.progress("loading_napari")
    import napari

    reporter.progress("creating_viewer")
    viewer = napari.Viewer(title="VIPP")

    reporter.progress("loading_vipp")
    from napari_vipp._widget import VippWidget

    reporter.progress("building_interface")
    widget = _construct_vipp_widget(VippWidget, viewer, profile)
    viewer.window.add_dock_widget(
        widget,
        area="bottom",
        name="VIPP Workflow",
    )

    reporter.progress("preparing_workflow")
    _configure_initial_workflow(widget)

    # Force one event pass so "ready" means the actual viewer and dock widget
    # have had an opportunity to become visible, not merely that they exist.
    from qtpy.QtWidgets import QApplication

    viewer.show(block=False)
    QApplication.processEvents()
    reporter.ready("VIPP is ready")
    reporter.close()

    if smoke_exit_after_ready:
        # Installer CI uses a real Cocoa event loop and then asks it to stop
        # normally.  This catches native Qt teardown failures without adding a
        # visible user-facing option or killing the process mid-cleanup.
        from qtpy.QtCore import QTimer

        QTimer.singleShot(750, QApplication.instance().quit)

    napari.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the child application, reporting any pre-readiness exception."""
    args = build_parser().parse_args(argv)
    profile = LaunchProfile.parse(args.profile)
    try:
        reporter = _make_reporter(args)
    except Exception as exc:
        if sys.stderr is not None:
            print(f"VIPP startup channel error: {exc}", file=sys.stderr)
        return 2

    try:
        reporter.progress("starting_python")
        return run_application(
            profile,
            reporter,
            smoke_exit_after_ready=args.smoke_exit_after_ready,
        )
    except KeyboardInterrupt:
        try:
            reporter.failure("VIPP startup was interrupted.")
        except Exception:
            pass
        return 130
    except BaseException as exc:
        short_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        try:
            reporter.failure("VIPP could not finish starting.", error=short_error)
        except Exception:
            pass
        traceback.print_exc()
        return 1
    finally:
        try:
            reporter.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StartupReporter",
    "build_parser",
    "main",
    "run_application",
]
