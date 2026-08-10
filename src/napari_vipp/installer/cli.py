"""JSON-only command line for the non-mutating VIPP installation planner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from napari_vipp.installer.discovery import DiscoveryServices, discover_installation
from napari_vipp.installer.models import (
    ComputeTrack,
    HostSnapshot,
    InstallMode,
    InstallRequest,
    ReleaseSpec,
    ShortcutScope,
)
from napari_vipp.installer.planner import create_install_plan, current_release_spec


class CliUsageError(ValueError):
    """Invalid command-line intent that should remain machine-readable."""


class _PlanArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately plan-only installer parser."""

    parser = _PlanArgumentParser(
        prog="vipp-install-plan",
        description=(
            "Inspect a VIPP Windows installation plan without changing files, "
            "packages, shortcuts, or the registry."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan",
        help="Emit the complete plan as JSON and perform no mutations.",
    )
    plan.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in InstallMode),
        required=True,
        help="Create a managed environment or inspect a selected napari venv.",
    )
    plan.add_argument(
        "--track",
        choices=tuple(track.value for track in ComputeTrack),
        required=True,
        help="Use the portable CPU route or the qualified CUDA 13 route.",
    )
    python_group = plan.add_mutually_exclusive_group()
    python_group.add_argument(
        "--base-python",
        type=Path,
        help="64-bit CPython used to create a managed environment.",
    )
    python_group.add_argument(
        "--environment-python",
        type=Path,
        help="Scripts\\python.exe inside an existing napari virtual environment.",
    )
    plan.add_argument(
        "--install-root",
        type=Path,
        help=(
            "Managed environment directory. Defaults below %%LOCALAPPDATA%%\\VIPP."
        ),
    )
    plan.add_argument(
        "--shortcuts",
        choices=tuple(scope.value for scope in ShortcutScope),
        default=ShortcutScope.DESKTOP.value,
        help="Intended shortcut scope for the later executor (default: desktop).",
    )
    plan.add_argument(
        "--shortcut-directory",
        type=Path,
        help="Explicit directory used instead of the discovered Windows location.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    services: DiscoveryServices | None = None,
    release: ReleaseSpec | None = None,
    host: HostSnapshot | None = None,
) -> int:
    """Emit a stable JSON plan; never execute any planned action."""

    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        request = _request_from_args(args)
        selected_release = release or current_release_spec()
        discovery = discover_installation(
            request,
            services=services,
            sys_platform=host.sys_platform if host else None,
            platform_system=host.platform_system if host else None,
            machine=host.machine if host else None,
        )
        plan = create_install_plan(
            request,
            discovery=discovery,
            release=selected_release,
        )
    except CliUsageError as exc:
        _write_failure(exc, status="invalid_request")
        return 2
    except Exception as exc:
        _write_failure(exc, status="discovery_failed")
        return 3
    _write_utf8(plan.to_json())
    return 0 if plan.ready else 2


def _request_from_args(args: argparse.Namespace) -> InstallRequest:
    mode = InstallMode(args.mode)
    if mode is InstallMode.MANAGED:
        if args.environment_python is not None:
            raise CliUsageError(
                "Managed mode does not accept --environment-python; use --base-python."
            )
        selected_python = args.base_python or Path(sys.executable)
    else:
        if args.base_python is not None:
            raise CliUsageError(
                "Existing mode does not accept --base-python; use --environment-python."
            )
        if args.environment_python is None:
            raise CliUsageError(
                "Existing mode requires --environment-python pointing to the selected "
                "venv."
            )
        selected_python = args.environment_python
    return InstallRequest(
        mode=mode,
        track=ComputeTrack(args.track),
        python=selected_python,
        install_root=args.install_root,
        shortcut_scope=ShortcutScope(args.shortcuts),
        shortcut_directory=args.shortcut_directory,
    )


def _write_failure(exc: BaseException, *, status: str) -> None:
    document = {
        "schema": "napari-vipp-install-plan-error",
        "schema_version": 1,
        "plan_only": True,
        "mutation_performed": False,
        "ready": False,
        "status": status,
        "error": f"{type(exc).__name__}: {exc}",
    }
    _write_utf8(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_utf8(text: str) -> None:
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(text.encode("utf-8"))
        buffer.flush()
        return
    sys.stdout.write(text)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
