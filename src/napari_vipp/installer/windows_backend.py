"""Windows planner/executor adapter used by the novice setup window."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from napari_vipp.installer.discovery import (
    DiscoveryServices,
    default_services,
    discover_installation,
)
from napari_vipp.installer.frontend import (
    InstallationCancelled,
    InstallerSelection,
    InstallOutcome,
    PreparedInstall,
    ProgressUpdate,
    TargetKind,
    TrackChoice,
)
from napari_vipp.installer.models import (
    ComputeTrack,
    InstallMode,
    InstallPlan,
    InstallRequest,
    IssueSeverity,
    ReleaseSpec,
    ShortcutScope,
)
from napari_vipp.installer.ownership import OwnershipState, inspect_ownership
from napari_vipp.installer.payload import bundled_release_spec
from napari_vipp.installer.planner import create_install_plan
from napari_vipp.installer.python_discovery import (
    PYTHON_DOWNLOAD_URL,
    PythonCandidate,
    choose_python,
    discover_python_candidates,
)

_ERROR_FRIENDLY_TEXT = {
    "host_platform_unsupported": (
        "This setup program currently supports 64-bit Windows computers."
    ),
    "host_architecture_unsupported": (
        "VIPP needs a 64-bit Windows computer with an x86-64 processor."
    ),
    "python_probe_failed": (
        "Setup found Python, but it could not verify that installation."
    ),
    "python_version_unsupported": (
        "VIPP needs a supported 64-bit Python installation."
    ),
    "python_implementation_unsupported": "VIPP needs the standard CPython program.",
    "python_pointer_width_unsupported": "VIPP needs 64-bit Python.",
    "insufficient_disk_space": (
        "There is not enough free space to install VIPP safely."
    ),
    "nvidia_driver_probe_failed": (
        "Setup could not verify a compatible NVIDIA graphics driver."
    ),
    "nvidia_driver_version_unsupported": (
        "The NVIDIA graphics driver must be updated before GPU setup can continue."
    ),
    "nvidia_compute_capability_unsupported": (
        "The available NVIDIA graphics hardware is not supported by this release."
    ),
    "shortcut_collision_unowned": (
        "A shortcut with the same name already exists and setup will not replace it."
    ),
    "install_target_protected": (
        "The selected installation location is not safe for automatic setup."
    ),
    "install_target_parent_invalid": (
        "The selected installation location is not available."
    ),
}


class WindowsInstallerBackend:
    """Connect read-only planning to the transactional managed apply engine."""

    def __init__(
        self,
        *,
        preferred_python: Path | None = None,
        engine: object | None = None,
        release: ReleaseSpec | None = None,
        services: DiscoveryServices | None = None,
        environ: Mapping[str, str] | None = None,
        candidate_finder: Callable[..., tuple[PythonCandidate, ...]] = (
            discover_python_candidates
        ),
    ) -> None:
        self._preferred_python = (
            Path(preferred_python) if preferred_python is not None else None
        )
        self._engine = engine
        self._release = release
        self._services = services or default_services()
        self._environ = dict(os.environ if environ is None else environ)
        self._candidate_finder = candidate_finder

    def inspect(
        self,
        selection: InstallerSelection,
        *,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
        repair: bool = False,
    ) -> PreparedInstall:
        progress(ProgressUpdate("python", "Looking for a supported Python…", 1, 5))
        self._check_cancelled(cancellation)
        release = self._release or bundled_release_spec()
        if selection.existing_python is not None:
            return self._inspect_existing_route(selection, release)

        self._recover_interrupted(selection)
        owned = self._owned_installation(selection)
        candidates = self._python_candidates(owned)
        progress(
            ProgressUpdate(
                "python-found",
                _python_found_message(candidates),
                1,
                5,
            )
        )
        progress(ProgressUpdate("hardware", "Checking available hardware…", 2, 5))
        plan, fallback_note = self._recommended_plan(
            selection,
            release,
            candidates,
            owned=owned,
        )
        self._check_cancelled(cancellation)
        if plan is None:
            track = (
                owned.record.track
                if owned is not None
                else _explicit_track(selection.track) or ComputeTrack.CPU
            )
            target = (
                owned.record.managed_root
                if owned is not None
                else selection.install_root
                or _default_install_root(self._environ, track)
            )
            python_requirement = (
                "64-bit Python 3.12"
                if track is ComputeTrack.CUDA13
                else "64-bit Python 3.12 or 3.13"
            )
            return PreparedInstall(
                kind=TargetKind.BLOCKED,
                target=target,
                release_version=release.version,
                track=track,
                plain_summary=(
                    f"VIPP needs {python_requirement}. Install the official "
                    "64-bit Python 3.12.10 release, then return here and choose "
                    "Check again."
                ),
                technical_details=(
                    "No supported CPython installation was found through the "
                    "Windows registry, Python launcher, PATH, or installer override."
                ),
                reason="Python is required before VIPP can be installed.",
                help_url=PYTHON_DOWNLOAD_URL,
            )

        progress(
            ProgressUpdate(
                "hardware-checked",
                _hardware_checked_message(plan, fallback_note=fallback_note),
                2,
                5,
            )
        )
        foreign = _plan_is_foreign(plan)
        if foreign or not plan.ready:
            details = _join_details(fallback_note, plan.to_json())
            return PreparedInstall(
                kind=TargetKind.FOREIGN if foreign else TargetKind.BLOCKED,
                target=plan.discovery.filesystem.target,
                release_version=release.version,
                track=plan.request.track,
                plain_summary=(
                    "Setup found files in this folder that it did not create. "
                    "For safety, it will not replace or remove them."
                    if foreign
                    else _friendly_plan_error(plan)
                ),
                technical_details=details,
                required_free_bytes=plan.required_free_bytes,
                reason=_first_remediation(plan),
            )

        progress(
            ProgressUpdate(
                "resolution",
                (
                    "Reviewing exact packages from PyPI — the first check can "
                    "take several minutes."
                ),
                3,
                5,
            )
        )
        token = _engine_cancellation_token(cancellation)
        transaction = self._selected_engine().prepare(
            plan,
            progress=lambda event: progress(_progress_update(event)),
            cancellation=token,
            repair=repair,
        )
        self._check_cancelled(cancellation)
        prepared = _prepared_for_transaction(
            transaction,
            plan=plan,
            fallback_note=fallback_note,
        )
        progress(
            ProgressUpdate(
                "decision",
                _installation_decision_message(prepared.kind),
                5,
                5,
            )
        )
        return prepared

    def apply(
        self,
        prepared: PreparedInstall,
        *,
        confirmed: bool,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
    ) -> InstallOutcome:
        if prepared.kind not in {
            TargetKind.NEW,
            TargetKind.UPDATE,
            TargetKind.REPAIR,
        }:
            raise RuntimeError("This reviewed setup state cannot be applied.")
        transaction = prepared.payload
        if transaction is None:
            raise RuntimeError("The reviewed transaction is unavailable.")
        engine = self._selected_engine()
        authorization = engine.authorize(transaction, confirmed=confirmed)
        token = _engine_cancellation_token(cancellation)
        try:
            result = engine.apply(
                transaction,
                authorization,
                progress=lambda event: progress(_progress_update(event)),
                cancellation=token,
            )
        except Exception as exc:
            if cancellation.is_set() or _looks_cancelled(exc):
                raise InstallationCancelled(details=str(exc)) from exc
            raise
        # The ownership record is the executor's commit point. A cancellation
        # click arriving after that point cannot turn a completed installation
        # into a reported cancellation.
        if _result_cancelled(result):
            raise InstallationCancelled(details=_json_details(result))
        if not _result_succeeded(result):
            raise RuntimeError(_result_error(result))
        if result.launcher_path is None:
            raise RuntimeError("VIPP passed setup but its launcher was not recorded.")
        launcher = Path(result.launcher_path)
        result_message = str(getattr(result, "message", "")).strip()
        registration_warning = str(getattr(result, "registration_warning", "")).strip()
        visible_message = result_message or (
            "VIPP was installed successfully. You can open it now or use its "
            "shortcut later."
        )
        if (
            registration_warning
            and "run vipp setup again" not in visible_message.casefold()
        ):
            visible_message += (
                "\n\nVIPP is ready, but Windows could not finish its Repair and "
                "Remove entry. Run VIPP Setup again to finish that step."
            )
        return InstallOutcome(
            launcher=launcher,
            message=visible_message,
            technical_details=_json_details(result),
            payload=result,
        )

    def open_vipp(self, launcher: Path) -> None:
        launcher = Path(launcher)
        if not launcher.is_file():
            raise FileNotFoundError(f"The VIPP launcher is missing: {launcher}")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        working_directory = _stable_launch_directory(
            self._services,
            launcher=launcher,
        )
        subprocess.Popen(
            (str(launcher),),
            cwd=str(working_directory),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            shell=False,
            creationflags=creationflags,
        )

    def _selected_engine(self):
        if self._engine is None:
            from napari_vipp.installer.engine import ManagedInstallerEngine

            self._engine = ManagedInstallerEngine()
        return self._engine

    def _python_candidates(self, owned) -> tuple[PythonCandidate, ...]:
        paths = []
        if self._preferred_python is not None:
            paths.append(("setup option", self._preferred_python))
        if owned is not None:
            paths.append(("previous VIPP installation", owned.record.base_python))
        preferred = (
            self._candidate_finder(
                environ=self._environ,
                candidate_paths=tuple(paths),
            )
            if paths
            else ()
        )
        discovered = self._candidate_finder(environ=self._environ)
        by_path: dict[str, PythonCandidate] = {}
        for candidate in (*preferred, *discovered):
            key = os.path.normcase(os.path.abspath(candidate.executable))
            by_path.setdefault(key, candidate)
        return tuple(by_path.values())

    def _owned_installation(self, selection: InstallerSelection):
        roots: list[Path]
        if selection.install_root is not None:
            roots = [selection.install_root]
        else:
            explicit = _explicit_track(selection.track)
            tracks = (
                (explicit,)
                if explicit is not None
                else (ComputeTrack.CUDA13, ComputeTrack.CPU)
            )
            roots = [_default_install_root(self._environ, track) for track in tracks]
        for root in roots:
            inspection = inspect_ownership(root)
            if inspection.state is OwnershipState.VALID:
                return inspection
        return None

    def _recover_interrupted(self, selection: InstallerSelection) -> None:
        recover = getattr(self._selected_engine(), "recover_interrupted", None)
        if not callable(recover):
            return
        if selection.install_root is not None:
            roots = (selection.install_root,)
        else:
            explicit = _explicit_track(selection.track)
            tracks = (
                (explicit,)
                if explicit is not None
                else (ComputeTrack.CUDA13, ComputeTrack.CPU)
            )
            roots = tuple(
                _default_install_root(self._environ, track) for track in tracks
            )
        for root in roots:
            result = recover(root)
            if getattr(result, "completed", True) is False:
                errors = "; ".join(getattr(result, "errors", ()))
                raise RuntimeError(
                    "An interrupted VIPP setup needs manual cleanup before setup "
                    f"can continue. {errors}"
                )

    def _recommended_plan(
        self,
        selection: InstallerSelection,
        release: ReleaseSpec,
        candidates: tuple[PythonCandidate, ...],
        *,
        owned,
    ) -> tuple[InstallPlan | None, str]:
        if owned is not None:
            owned_track = owned.record.track
            explicit = _explicit_track(selection.track)
            if explicit is not None and explicit is not owned_track:
                return self._make_plan(
                    selection,
                    release,
                    candidates,
                    explicit,
                ), ""
            return (
                self._make_plan(
                    selection,
                    release,
                    candidates,
                    owned_track,
                ),
                "",
            )
        explicit = _explicit_track(selection.track)
        if explicit is not None:
            return (
                self._make_plan(selection, release, candidates, explicit),
                "",
            )
        cuda_plan = self._make_plan(
            selection,
            release,
            candidates,
            ComputeTrack.CUDA13,
        )
        if cuda_plan is not None and cuda_plan.ready:
            return cuda_plan, ""
        cpu_plan = self._make_plan(
            selection,
            release,
            candidates,
            ComputeTrack.CPU,
        )
        if cpu_plan is None:
            return None, ""
        cuda_reason = (
            _friendly_plan_error(cuda_plan)
            if cuda_plan is not None
            else "Python 3.12 was not available for the GPU route."
        )
        return cpu_plan, f"Automatic selection used CPU. {cuda_reason}"

    def _make_plan(
        self,
        selection: InstallerSelection,
        release: ReleaseSpec,
        candidates: tuple[PythonCandidate, ...],
        track: ComputeTrack,
    ) -> InstallPlan | None:
        candidate = choose_python(candidates, track)
        if candidate is None:
            return None
        request = InstallRequest(
            mode=InstallMode.MANAGED,
            track=track,
            python=candidate.executable,
            install_root=selection.install_root,
            shortcut_scope=(
                ShortcutScope.BOTH
                if selection.create_desktop_shortcut
                else ShortcutScope.START_MENU
            ),
        )
        discovery = discover_installation(
            request,
            services=self._services,
            environ=self._environ,
        )
        return create_install_plan(request, discovery=discovery, release=release)

    def _inspect_existing_route(
        self,
        selection: InstallerSelection,
        release: ReleaseSpec,
    ) -> PreparedInstall:
        assert selection.existing_python is not None
        root = selection.existing_python.parent.parent
        return PreparedInstall(
            kind=TargetKind.BLOCKED,
            target=root,
            release_version=release.version,
            track=_explicit_track(selection.track) or ComputeTrack.CPU,
            plain_summary=(
                "The one-click setup keeps existing napari environments unchanged. "
                "Use the recommended managed location, or follow the expert "
                "existing-environment instructions."
            ),
            technical_details=(
                "Transactional mutation of a user-owned napari environment is not "
                "enabled in this setup build."
            ),
            reason="Choose the recommended managed installation for automatic setup.",
        )

    @staticmethod
    def _check_cancelled(cancellation: threading.Event) -> None:
        if cancellation.is_set():
            raise InstallationCancelled()


def _prepared_for_transaction(
    transaction: object,
    *,
    plan: InstallPlan,
    fallback_note: str,
) -> PreparedInstall:
    inspection = getattr(transaction, "target_inspection", None)
    kind_value = getattr(getattr(inspection, "kind", None), "value", None)
    if kind_value is None:
        kind_value = getattr(getattr(transaction, "operation", None), "value", None)
    kind = _target_kind(kind_value)
    installed_version = getattr(inspection, "installed_version", None)
    if installed_version is None:
        installed_version = getattr(inspection, "current_version", None)
    launcher = getattr(inspection, "launcher_path", None)
    if launcher is None and kind in {TargetKind.CURRENT, TargetKind.NEWER}:
        launcher = _launcher_for_plan(plan)
    summary = _transaction_summary(
        kind,
        release_version=plan.release.version,
        installed_version=installed_version,
        track=plan.request.track,
    )
    transaction_details = _json_details(transaction)
    return PreparedInstall(
        kind=kind,
        target=plan.discovery.filesystem.target,
        release_version=plan.release.version,
        track=plan.request.track,
        plain_summary=summary,
        technical_details=_join_details(
            fallback_note,
            plan.to_json(),
            transaction_details,
        ),
        payload=(
            transaction
            if kind in {TargetKind.NEW, TargetKind.UPDATE, TargetKind.REPAIR}
            else None
        ),
        installed_version=installed_version,
        required_free_bytes=plan.required_free_bytes,
        launcher=Path(launcher) if launcher is not None else None,
        reason=str(getattr(inspection, "reason", "")),
    )


def _target_kind(value: object) -> TargetKind:
    normalized = str(value or "").casefold()
    aliases = {
        "new": TargetKind.NEW,
        "update": TargetKind.UPDATE,
        "current": TargetKind.CURRENT,
        "repair": TargetKind.REPAIR,
        "newer": TargetKind.NEWER,
        "foreign": TargetKind.FOREIGN,
    }
    return aliases.get(normalized, TargetKind.BLOCKED)


def _transaction_summary(
    kind: TargetKind,
    *,
    release_version: str,
    installed_version: str | None,
    track: ComputeTrack,
) -> str:
    compute = (
        "GPU acceleration will be available, with safe CPU fallback."
        if track is ComputeTrack.CUDA13
        else "This setup uses the reliable CPU route."
    )
    if kind is TargetKind.NEW:
        return (
            "VIPP will be installed in its own safe location for this Windows "
            "account. Any VIPP or napari environment installed elsewhere is "
            f"left unchanged, and VIPP shortcuts will be added. {compute}"
        )
    if kind is TargetKind.UPDATE:
        return (
            f"VIPP {installed_version or 'an older version'} will be updated to "
            f"{release_version}. The working version is kept until checks pass."
        )
    if kind is TargetKind.REPAIR:
        return (
            "Setup found a VIPP installation that needs repair. It will rebuild "
            "the program safely before replacing the damaged copy."
        )
    if kind is TargetKind.CURRENT:
        return f"VIPP {release_version} is installed and passed its setup checks."
    if kind is TargetKind.NEWER:
        return (
            f"VIPP {installed_version or 'a newer version'} is already installed. "
            f"This {release_version} setup will not downgrade it."
        )
    if kind is TargetKind.FOREIGN:
        return (
            "Setup found files it did not create and will not overwrite or remove "
            "them. Choose another location to continue."
        )
    return "Setup cannot safely use the selected location."


def _progress_update(event: object) -> ProgressUpdate:
    stage = getattr(event, "stage", "setup")
    stage_value = str(getattr(stage, "value", stage))
    indeterminate = stage_value in {"installing", "resolving"}
    return ProgressUpdate(
        stage=stage_value,
        message=str(getattr(event, "message", "Working…")),
        completed=(
            None if indeterminate else _optional_int(getattr(event, "completed", None))
        ),
        total=(None if indeterminate else _optional_int(getattr(event, "total", None))),
    )


def _engine_cancellation_token(event: threading.Event):
    from napari_vipp.installer.engine import CancellationToken

    return CancellationToken(event)


def _python_found_message(candidates: tuple[PythonCandidate, ...]) -> str:
    versions = tuple(dict.fromkeys(candidate.version_text for candidate in candidates))
    if not versions:
        return "No supported 64-bit Python was found."
    if len(versions) == 1:
        return f"Found 64-bit Python {versions[0]}."
    return "Found supported 64-bit Python versions " + ", ".join(versions) + "."


def _hardware_checked_message(
    plan: InstallPlan,
    *,
    fallback_note: str,
) -> str:
    if plan.request.track is ComputeTrack.CUDA13 and plan.ready:
        nvidia = plan.discovery.nvidia
        device = nvidia.devices[0] if nvidia is not None and nvidia.devices else None
        if device is not None:
            return f"{device.name} is eligible for the CUDA option."
        return "This computer is eligible for the CUDA option."
    if fallback_note:
        return "Hardware check finished. CPU setup is recommended for this computer."
    if plan.request.track is ComputeTrack.CPU:
        return "Hardware check finished. CPU setup is selected."
    return "Hardware check finished. The selected GPU option needs attention."


def _installation_decision_message(kind: TargetKind) -> str:
    return {
        TargetKind.NEW: "Checks finished. Setup recommends installing VIPP.",
        TargetKind.UPDATE: "Checks finished. Setup recommends updating VIPP.",
        TargetKind.REPAIR: "Checks finished. Setup recommends repairing VIPP.",
        TargetKind.CURRENT: "Checks finished. VIPP is ready to open.",
        TargetKind.NEWER: "Checks finished. The newer VIPP version will be kept.",
        TargetKind.FOREIGN: "Checks finished. Choose another installation folder.",
        TargetKind.BLOCKED: "Checks finished. Setup needs your attention.",
    }[kind]


def _result_succeeded(result: object) -> bool:
    status = getattr(getattr(result, "status", None), "value", "")
    return str(status).casefold() in {"success", "succeeded", "completed"}


def _stable_launch_directory(
    services: DiscoveryServices,
    *,
    launcher: Path,
) -> Path:
    probe = services.known_folder_probe
    try:
        documents = probe("documents") if probe is not None else None
    except Exception:
        documents = None
    environment = launcher.parent.parent
    for candidate in (documents, Path.home()):
        if candidate is not None:
            selected = Path(os.path.abspath(candidate))
            try:
                redirected = services.reparse_probe(selected)
            except Exception:
                redirected = True
            if (
                selected.is_dir()
                and not redirected
                and not _same_or_descendant(selected, environment)
            ):
                return selected
    raise RuntimeError(
        "Windows could not find a stable Documents or home folder for VIPP."
    )


def _same_or_descendant(path: Path, parent: Path) -> bool:
    try:
        return os.path.normcase(os.path.commonpath((path, parent))) == os.path.normcase(
            os.path.abspath(parent)
        )
    except ValueError:
        return False


def _result_cancelled(result: object) -> bool:
    status = getattr(getattr(result, "status", None), "value", "")
    return str(status).casefold() in {"cancelled", "canceled"}


def _result_error(result: object) -> str:
    message = str(
        getattr(result, "message", "VIPP setup did not complete successfully.")
    )
    technical = str(getattr(result, "technical_error", "")).strip()
    summary = f"{message} {technical}".strip()
    return _join_details(summary, _json_details(result))


def _looks_cancelled(exc: Exception) -> bool:
    return "cancel" in type(exc).__name__.casefold()


def _json_details(value: object) -> str:
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        try:
            return json.dumps(as_dict(), indent=2, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            pass
    to_json = getattr(value, "to_json", None)
    if callable(to_json):
        try:
            return str(to_json()).strip()
        except (TypeError, ValueError):
            pass
    return repr(value)


def _plan_is_foreign(plan: InstallPlan) -> bool:
    foreign_codes = {
        "managed_target_already_exists",
        "managed_target_ownership_mismatch",
    }
    return any(issue.code in foreign_codes for issue in plan.issues)


def _friendly_plan_error(plan: InstallPlan | None) -> str:
    if plan is None:
        return "Setup could not find the software required for this option."
    errors = [issue for issue in plan.issues if issue.severity is IssueSeverity.ERROR]
    if not errors:
        return "This setup option is ready."
    return _ERROR_FRIENDLY_TEXT.get(errors[0].code, errors[0].message)


def _first_remediation(plan: InstallPlan) -> str:
    return next(
        (
            issue.remediation
            for issue in plan.issues
            if issue.severity is IssueSeverity.ERROR and issue.remediation
        ),
        "Choose another option or check again after correcting the problem.",
    )


def _launcher_for_plan(plan: InstallPlan) -> Path | None:
    preferred = "auto" if plan.request.track is ComputeTrack.CUDA13 else "cpu"
    for shortcut in plan.shortcuts:
        if shortcut.profile == preferred:
            return shortcut.executable
    return plan.shortcuts[0].executable if plan.shortcuts else None


def _explicit_track(choice: TrackChoice) -> ComputeTrack | None:
    if choice is TrackChoice.CPU:
        return ComputeTrack.CPU
    if choice is TrackChoice.CUDA13:
        return ComputeTrack.CUDA13
    return None


def _default_install_root(
    environ: Mapping[str, str],
    track: ComputeTrack,
) -> Path:
    base = environ.get("LOCALAPPDATA")
    local = Path(base) if base else Path.home() / "AppData" / "Local"
    suffix = "cuda13" if track is ComputeTrack.CUDA13 else "cpu"
    return local / "VIPP" / "environments" / suffix


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _join_details(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip())


__all__ = ["WindowsInstallerBackend"]
